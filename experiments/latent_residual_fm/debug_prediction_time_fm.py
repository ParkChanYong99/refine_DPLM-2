#!/usr/bin/env python3
"""Sanity-check prediction-time DPLM2Bit -> latent residual FM refinement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import torch

from byprot.datamodules.pdb_dataset import utils as du
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.flow_matching import euler_sample
from experiments.latent_residual_fm.fm_model import LatentResidualFM


EXPECTED_OPTIMIZER_STEP = 100_000
GENERATION_MAX_ITER = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="data-bin/latent_residual_fm/afdb_l512_sharded",
    )
    parser.add_argument(
        "--split_csv",
        default=(
            "data-bin/latent_residual_fm/afdb_l512_sharded/"
            "splits/split_v1.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/latent_residual_fm/runs/"
            "fm_modeA_current_main/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_index", type=int, default=0)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


def load_validation_sample(
    dataset_dir: Path, split_csv: Path, val_index: int
) -> dict[str, Any]:
    require(val_index >= 0, "--val_index must be non-negative")
    required = {"sample_id", "shard_file", "index_in_shard", "length", "split"}
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        require(not missing, f"split CSV missing columns: {sorted(missing)}")
        validation_rows = [row for row in reader if row["split"] == "val"]

    require(validation_rows, "validation split is empty")
    require(
        val_index < len(validation_rows),
        f"--val_index {val_index} is outside {len(validation_rows)} validation rows",
    )
    row = validation_rows[val_index]
    index = int(row["index_in_shard"])
    length = int(row["length"])
    require(index >= 0 and length > 0, "invalid validation split row")

    shard_path = dataset_dir / row["shard_file"]
    shard = torch.load(str(shard_path), map_location="cpu", mmap=True)
    require(isinstance(shard, dict), f"invalid shard payload: {shard_path}")
    require("samples" in shard and "num_samples" in shard, f"invalid shard: {shard_path}")
    samples = shard["samples"]
    require(shard["num_samples"] == len(samples), f"bad shard count: {shard_path}")
    require(index < len(samples), f"index outside shard: {shard_path}")
    sample = samples[index]
    require(sample["sample_id"] == row["sample_id"], "split/shard sample ID mismatch")
    require(int(sample["length"]) == length, "split/shard sample length mismatch")
    require(tuple(sample["aatype"].shape) == (length,), "invalid sample aatype shape")
    require(tuple(sample["z_cont"].shape) == (length, EXPECTED_CODEBOOK_DIM), "invalid z_cont shape")
    require(tuple(sample["res_mask"].shape) == (length,), "invalid sample res_mask shape")
    return sample


def build_folding_input(
    model: torch.nn.Module, aatype: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the same folding batch as generate_dplm2.initialize_conditional_generation."""
    require(aatype.ndim == 1, "aatype must have shape [L]")
    length = int(aatype.shape[0])
    tokenizer = model.tokenizer
    aa_sequence = du.aatype_to_seq(aatype.long().cpu().tolist())
    require(len(aa_sequence) == length, "aatype-to-sequence length mismatch")

    aa_string = tokenizer.aa_cls_token + aa_sequence + tokenizer.aa_eos_token
    struct_string = (
        tokenizer.struct_cls_token
        + tokenizer.struct_mask_token * length
        + tokenizer.struct_eos_token
    )
    batch_struct = tokenizer.batch_encode_plus(
        [struct_string],
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    batch_aa = tokenizer.batch_encode_plus(
        [aa_string],
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )
    input_tokens = torch.cat(
        (batch_struct["input_ids"], batch_aa["input_ids"]), dim=1
    ).to(device)

    # This is the official folding convention: predict structure residues and
    # preserve the non-special amino-acid tokens as the partial condition.
    non_special = model.get_non_special_symbol_mask(input_tokens)
    type_ids = model.get_modality_type(input_tokens)
    input_tokens.masked_fill_(
        (type_ids == model.struct_type) & non_special,
        tokenizer._token_to_id[tokenizer.struct_mask_token],
    )
    partial_mask = type_ids == model.aa_type

    struct_tokens, aa_tokens = input_tokens.chunk(2, dim=1)
    expected_half = (1, length + 2)
    require(tuple(struct_tokens.shape) == expected_half, "structure half misaligned")
    require(tuple(aa_tokens.shape) == expected_half, "AA half misaligned")
    require(int(struct_tokens[0, 0]) == model.struct_bos_id, "structure BOS mismatch")
    require(int(struct_tokens[0, -1]) == model.struct_eos_id, "structure EOS mismatch")
    require(int(aa_tokens[0, 0]) == model.aa_bos_id, "AA BOS mismatch")
    require(int(aa_tokens[0, -1]) == model.aa_eos_id, "AA EOS mismatch")
    require(
        int((struct_tokens == model.struct_mask_id).sum()) == length,
        "folding structure residues are not all masked",
    )
    return input_tokens, partial_mask


def reconstruct_prediction_latent(
    model: torch.nn.Module, final_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    non_special = model.get_non_special_symbol_mask(final_tokens)
    rebuilt = model.prepare_for_struct_tokenizer(
        {"output_tokens": final_tokens.clone()}, non_special
    )
    return rebuilt["final_struct_feature"], rebuilt["res_mask"]


def extract_final_token_condition(
    model: torch.nn.Module,
    final_tokens: torch.Tensor,
    residue_length: int,
) -> tuple[torch.Tensor, ...]:
    # This deliberately performs a new forward on the public, post-generation
    # tokens.  Hidden states retained inside generate()/forward_decoder are not used.
    final_forward = model(final_tokens)
    all_hidden_states = final_forward.get("all_hidden_states")
    require(all_hidden_states is not None, "final-token forward returned no hidden states")
    require(
        len(all_hidden_states) == EXPECTED_NUM_LAYERS + 1,
        "hidden tuple must contain embedding plus 33 Transformer states",
    )
    transformer_states = tuple(all_hidden_states[1:])
    require(
        len(transformer_states) == EXPECTED_NUM_LAYERS,
        "expected 33 Transformer hidden states after excluding embedding state",
    )

    struct_tokens, aa_tokens = final_tokens.chunk(2, dim=1)
    require(struct_tokens.shape == aa_tokens.shape, "final modality halves differ")
    positions = struct_tokens.ne(model.struct_bos_id) & struct_tokens.ne(
        model.struct_eos_id
    )
    require(
        int(positions.sum()) == residue_length,
        "final structure-token residue alignment count mismatch",
    )

    aligned: list[torch.Tensor] = []
    for index, state in enumerate(transformer_states):
        require(state.ndim == 3, f"hidden state {index} is not rank 3")
        struct_state, _ = state.chunk(2, dim=1)
        layer = struct_state[positions].view(1, residue_length, state.shape[-1])
        require(
            tuple(layer.shape) == (1, residue_length, EXPECTED_HIDDEN_SIZE),
            f"hidden state {index} alignment produced {tuple(layer.shape)}",
        )
        require(bool(torch.isfinite(layer).all()), f"hidden state {index} is non-finite")
        aligned.append(layer)
    return tuple(aligned)


def load_fm(
    checkpoint_path: Path, device: torch.device
) -> tuple[LatentResidualFM, int, float, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    require("fm_model_state_dict" in checkpoint, "checkpoint lacks fm_model_state_dict")
    for key in ("optimizer_step", "best_val_loss", "best_step"):
        require(key in checkpoint, f"checkpoint lacks {key}")
    optimizer_step = int(checkpoint["optimizer_step"])
    best_val_loss = float(checkpoint["best_val_loss"])
    best_step = int(checkpoint["best_step"])
    require(
        optimizer_step == EXPECTED_OPTIMIZER_STEP,
        f"expected optimizer_step={EXPECTED_OPTIMIZER_STEP}, got {optimizer_step}",
    )
    require(best_step == EXPECTED_OPTIMIZER_STEP, f"unexpected best_step={best_step}")
    require(torch.isfinite(torch.tensor(best_val_loss)).item(), "best_val_loss is non-finite")

    fm = LatentResidualFM().to(device)
    fm.load_state_dict(checkpoint["fm_model_state_dict"], strict=True)
    fm.requires_grad_(False).eval()
    del checkpoint
    require(not fm.training, "FM is not in eval mode")
    return fm, optimizer_step, best_val_loss, best_step


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    require(bool(torch.isfinite(tensor).all()), f"{name} contains NaN/Inf")


def assert_atom37(name: str, decoded: dict[str, torch.Tensor], length: int) -> None:
    require("atom37_positions" in decoded, f"{name} decode lacks atom37_positions")
    atom37 = decoded["atom37_positions"]
    require(
        tuple(atom37.shape) == (1, length, 37, 3),
        f"{name} atom37 shape is {tuple(atom37.shape)}",
    )
    assert_finite(f"{name} atom37", atom37)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    require(prediction.shape == target.shape, "latent diagnostic shapes differ")
    require(mask.shape == prediction.shape[:2], "latent diagnostic mask misaligned")
    denominator = mask.float().sum() * prediction.shape[-1]
    require(bool(denominator > 0), "latent diagnostic mask is empty")
    squared = (prediction.float() - target.float()).square()
    return float((squared * mask[..., None]).sum() / denominator)


def main() -> int:
    args = parse_args()
    require(args.num_flow_steps > 0, "--num_flow_steps must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sample = load_validation_sample(
        Path(args.dataset_dir), Path(args.split_csv), args.val_index
    )
    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype = sample["aatype"].long()
    z_cont = sample["z_cont"].float().view(1, length, EXPECTED_CODEBOOK_DIM).to(device)
    diagnostic_mask = sample["res_mask"].float().view(1, length).to(device)

    checks: dict[str, bool] = {}
    model, struct_tokenizer = load_models(args.dplm_checkpoint, device, checks)
    dplm_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    require(dplm_trainable == 0, "DPLM trainable parameter count is not zero")
    fm, optimizer_step, best_val_loss, best_step = load_fm(
        Path(args.checkpoint), device
    )

    with torch.inference_mode():
        input_tokens, partial_mask = build_folding_input(model, aatype, device)
        generated = model.generate(
            input_tokens=input_tokens,
            max_iter=GENERATION_MAX_ITER,
            temperature=1.0,
            unmasking_strategy="deterministic",
            sampling_strategy="argmax",
            partial_masks=partial_mask,
        )
        require("output_tokens" in generated, "generation returned no output_tokens")
        final_tokens = generated["output_tokens"]
        require(
            tuple(final_tokens.shape) == (1, 2 * (length + 2)),
            f"generated token shape is {tuple(final_tokens.shape)}",
        )

        predicted_z_quant, res_mask = reconstruct_prediction_latent(model, final_tokens)
        require(
            tuple(predicted_z_quant.shape) == (1, length, EXPECTED_CODEBOOK_DIM),
            f"predicted z_quant shape is {tuple(predicted_z_quant.shape)}",
        )
        require(tuple(res_mask.shape) == (1, length), "generated res_mask misaligned")
        require(bool(res_mask.bool().all()), "generated sample has masked-out residues")
        hidden_states = extract_final_token_condition(model, final_tokens, length)

        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(args.seed)
        r0 = torch.randn(
            predicted_z_quant.shape,
            dtype=predicted_z_quant.dtype,
            device=device,
            generator=noise_generator,
        )
        r_hat = euler_sample(
            fm,
            predicted_z_quant,
            hidden_states,
            res_mask,
            num_steps=args.num_flow_steps,
            initial_noise=r0,
        )
        z_refined = predicted_z_quant + r_hat

        bit_only = struct_tokenizer.detokenize(
            predicted_z_quant, res_mask=res_mask
        )
        fm_refined = struct_tokenizer.detokenize(z_refined, res_mask=res_mask)

    tensors: Sequence[tuple[str, torch.Tensor]] = (
        ("generated output tokens", final_tokens),
        ("predicted z_quant", predicted_z_quant),
        ("generated res_mask", res_mask),
        ("initial Gaussian r0", r0),
        ("r_hat", r_hat),
        ("z_refined", z_refined),
        ("stored z_cont", z_cont),
        ("stored diagnostic res_mask", diagnostic_mask),
    )
    for name, tensor in tensors:
        assert_finite(name, tensor)
    assert_atom37("Bit-only", bit_only, length)
    assert_atom37("FM-refined", fm_refined, length)

    bit_mse = masked_mse(predicted_z_quant, z_cont, diagnostic_mask)
    refined_mse = masked_mse(z_refined, z_cont, diagnostic_mask)
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated = peak_reserved = 0.0

    print("\n[PREDICTION-TIME FM SANITY]")
    print(f"sample_id: {sample_id}")
    print(f"sequence length: {length}")
    print(f"generated structure token shape: {(1, length)}")
    print(f"generated combined output token shape: {tuple(final_tokens.shape)}")
    print(f"predicted_z_quant shape: {tuple(predicted_z_quant.shape)}")
    print(f"number of hidden states: {len(hidden_states)}")
    print(f"aligned hidden shape: {tuple(hidden_states[-1].shape)}")
    print(f"r_hat shape: {tuple(r_hat.shape)}")
    print(f"r_hat abs mean: {r_hat.float().abs().mean().item():.9g}")
    print(f"r_hat abs max: {r_hat.float().abs().max().item():.9g}")
    print(f"z_refined shape: {tuple(z_refined.shape)}")
    print(f"Bit-only atom37 shape: {tuple(bit_only['atom37_positions'].shape)}")
    print(f"FM-refined atom37 shape: {tuple(fm_refined['atom37_positions'].shape)}")
    print("all tensors finite: True")
    print(f"DPLM trainable params: {dplm_trainable}")
    print(f"FM checkpoint optimizer_step: {optimizer_step}")
    print(f"FM checkpoint best_val_loss: {best_val_loss:.16g}")
    print(f"FM checkpoint best_step: {best_step}")
    print(f"GPU peak allocated MiB: {peak_allocated:.2f}")
    print(f"GPU peak reserved MiB: {peak_reserved:.2f}")
    print("\n[PREDICTION-TIME LATENT DIAGNOSTIC; NOT A FOLDING METRIC]")
    print(f"MSE(predicted_z_quant, z_cont): {bit_mse:.9g}")
    print(f"MSE(z_refined, z_cont): {refined_mse:.9g}")
    print("\nPREDICTION_TIME_FM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

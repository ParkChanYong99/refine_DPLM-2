#!/usr/bin/env python3
"""Read-only runtime checks for DPLM-2.1 Bit latent-residual interfaces.

This script intentionally contains no FM network, source distribution, loss,
optimizer, dataset builder, or backward pass.  It verifies only the frozen base
model, the tokenizer's [B, L, 13] latent path, generation-time hidden capture,
and residue-wise tensor alignment.
"""

import argparse
import pickle
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset
from byprot.models.dplm2 import DPLM2Bit
from generate_dplm2 import initialize_conditional_generation


EXPECTED_HIDDEN_SIZE = 1280
EXPECTED_NUM_LAYERS = 33
EXPECTED_CODEBOOK_DIM = 13
EXPECTED_STRUCT_OFFSET = 36

SUMMARY_ITEMS = (
    "checkpoint loaded",
    "model frozen",
    "tokenizer frozen",
    "z_cont [B,L,13]",
    "z_quant [B,L,13]",
    "residual [B,L,13]",
    "rank-3 z_quant decode",
    "rank-3 z_cont decode",
    "hidden size matches runtime config",
    "33 transformer layers",
    "hidden tuple = N + 1",
    "embedding state excluded from baseline",
    "final-token extra forward hidden obtained",
    "final-token hidden aligned",
    "predicted z_quant aligned",
    "res_mask aligned",
    "generated final_struct_feature rank-3 latent",
    "final tokens -> z_quant reconstruction",
    "generated continuous latent decode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="airkingbd/dplm2_bit_650m")
    parser.add_argument(
        "--pkl_path",
        default="data-bin/cameo2022/preprocessed/7dz2_C.pkl",
    )
    parser.add_argument(
        "--fasta_path", default="data-bin/cameo2022/aatype.fasta"
    )
    # Two iterations are enough to exercise the repository's iterative decoder
    # while limiting memory and runtime on one RTX 2000 Ada.
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    """Raise an informative failure instead of silently accepting a mismatch."""
    if not condition:
        raise AssertionError(message)


def tensor_stats(tensor: torch.Tensor) -> str:
    values = tensor.detach().float()
    return (
        f"min={values.min().item():.6g} max={values.max().item():.6g} "
        f"mean={values.mean().item():.6g} std={values.std().item():.6g}"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "--device cuda requested but CUDA is unavailable")
    return device


def load_models(
    model_name: str, device: torch.device, checks: Dict[str, bool]
) -> Tuple[DPLM2Bit, torch.nn.Module]:
    """Load and freeze DPLM/structure tokenizer; all tensors remain inference-only."""
    model = DPLM2Bit.from_pretrained(model_name)
    checks["checkpoint loaded"] = True

    model.requires_grad_(False).eval().to(device)
    struct_tokenizer = model.struct_tokenizer
    struct_tokenizer.requires_grad_(False).eval()

    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer_trainable = sum(
        p.numel() for p in struct_tokenizer.parameters() if p.requires_grad
    )
    checks["model frozen"] = model_trainable == 0 and not model.training
    checks["tokenizer frozen"] = (
        tokenizer_trainable == 0 and not struct_tokenizer.training
    )

    config = model.net.config
    dtype = next(model.parameters()).dtype
    print("\n[MODEL CONFIG]")
    print(f"model class: {type(model).__name__}")
    print(f"checkpoint name: {model_name}")
    print(f"device: {next(model.parameters()).device}")
    print(f"dtype: {dtype}")
    print(f"model.net.config.hidden_size: {config.hidden_size}")
    print(f"model.net.config.num_hidden_layers: {config.num_hidden_layers}")
    print(
        "structure tokenizer codebook_embed_dim: "
        f"{struct_tokenizer.codebook_embed_dim}"
    )
    print(f"model.struct_vocab_offset: {model.struct_vocab_offset}")
    print(f"trainable base parameter count: {model_trainable}")
    print(f"trainable tokenizer parameter count: {tokenizer_trainable}")

    require(
        config.hidden_size == EXPECTED_HIDDEN_SIZE,
        f"exact checkpoint hidden_size must be {EXPECTED_HIDDEN_SIZE}, got {config.hidden_size}",
    )
    require(
        config.num_hidden_layers == EXPECTED_NUM_LAYERS,
        f"exact checkpoint num_hidden_layers must be {EXPECTED_NUM_LAYERS}, got {config.num_hidden_layers}",
    )
    require(
        struct_tokenizer.codebook_embed_dim == EXPECTED_CODEBOOK_DIM,
        f"structure codebook dimension must be {EXPECTED_CODEBOOK_DIM}, got "
        f"{struct_tokenizer.codebook_embed_dim}",
    )
    require(
        model.struct_vocab_offset == EXPECTED_STRUCT_OFFSET,
        f"structure vocabulary offset must be {EXPECTED_STRUCT_OFFSET}, got "
        f"{model.struct_vocab_offset}",
    )
    require(checks["model frozen"], "DPLM base model is not fully frozen/eval")
    require(
        checks["tokenizer frozen"], "structure tokenizer is not fully frozen/eval"
    )
    return model, struct_tokenizer


def load_coordinate_sample(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load one coordinate pickle into tokenizer fields with batch shape B=1."""
    with open(path, "rb") as handle:
        raw = dict(pickle.load(handle))
    # Use the repository-native coordinate feature path; no coordinate residual
    # or coordinate refinement logic is copied or evaluated here.
    sample = PdbDataset.process_chain(raw, random_crop=False)
    batch = {
        "all_atom_positions": sample["all_atom_positions"].unsqueeze(0).to(device),
        "res_mask": sample["res_mask"].unsqueeze(0).to(device),
        "seq_length": sample["seq_length"].reshape(1).to(device),
    }
    require(
        batch["all_atom_positions"].ndim == 4
        and batch["all_atom_positions"].shape[0] == 1
        and batch["all_atom_positions"].shape[-2:] == (37, 3),
        f"unexpected all_atom_positions shape {tuple(batch['all_atom_positions'].shape)}",
    )
    require(
        batch["res_mask"].shape == batch["all_atom_positions"].shape[:2],
        "coordinate residue mask is not aligned",
    )
    return batch


def check_tokenizer_latents(
    struct_tokenizer: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    checks: Dict[str, bool],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify coordinates -> z_cont/z_quant/raw IDs, all residue-shaped [B,L,*]."""
    z_cont, encoder_feats = struct_tokenizer.encode(
        batch["all_atom_positions"], batch["res_mask"], batch["seq_length"]
    )
    z_quant, _, (_, _, struct_ids) = struct_tokenizer.quantize(
        z_cont, mask=batch["res_mask"].bool()
    )
    r_target = z_cont - z_quant
    valid = batch["res_mask"].bool()

    checks["z_cont [B,L,13]"] = (
        z_cont.ndim == 3
        and z_cont.shape[-1] == struct_tokenizer.codebook_embed_dim
    )
    checks["z_quant [B,L,13]"] = (
        z_quant.ndim == 3 and z_quant.shape == z_cont.shape
    )
    checks["residual [B,L,13]"] = r_target.shape == z_cont.shape
    require(checks["z_cont [B,L,13]"], f"invalid z_cont shape {tuple(z_cont.shape)}")
    require(checks["z_quant [B,L,13]"], f"invalid z_quant shape {tuple(z_quant.shape)}")
    require(
        struct_ids.shape[:2] == z_cont.shape[:2],
        f"raw struct IDs {tuple(struct_ids.shape)} do not align with z_cont",
    )
    require(valid.any().item(), "coordinate sample has no valid residues")
    valid_ids = struct_ids[valid]
    require(
        valid_ids.min().item() >= 0
        and valid_ids.max().item() < struct_tokenizer.num_codebook,
        f"valid raw LFQ IDs outside [0,{struct_tokenizer.num_codebook - 1}]",
    )
    valid_quant = z_quant[valid]
    require(
        torch.allclose(valid_quant.abs(), torch.ones_like(valid_quant), atol=1e-6),
        "valid LFQ forward values are not within tolerance of -1/+1",
    )

    print("\n[TOKENIZER]")
    print(f"encoder_feats shape: {tuple(encoder_feats.shape)}")
    print(f"z_cont shape: {tuple(z_cont.shape)}")
    print(f"z_quant shape: {tuple(z_quant.shape)}")
    print(f"struct_ids shape: {tuple(struct_ids.shape)}")
    print(f"r_target shape: {tuple(r_target.shape)}")
    print(f"z_cont {tensor_stats(z_cont[valid])}")
    print(f"z_quant unique values on valid residues: {torch.unique(valid_quant).tolist()}")
    print(f"r_target {tensor_stats(r_target[valid])}")
    print(
        "raw struct_ids min/max on valid residues: "
        f"{valid_ids.min().item()}/{valid_ids.max().item()}"
    )
    return z_cont, z_quant, struct_ids, r_target


def valid_decoder_coordinates(
    decoder_out: Dict[str, torch.Tensor], res_mask: torch.Tensor
) -> bool:
    positions = decoder_out["atom37_positions"]
    atom_mask = decoder_out["atom37_mask"].bool()
    selected = atom_mask & res_mask.bool().unsqueeze(-1)
    return bool(selected.any().item() and torch.isfinite(positions[selected]).all().item())


def check_continuous_decoder(
    struct_tokenizer: torch.nn.Module,
    z_cont: torch.Tensor,
    z_quant: torch.Tensor,
    res_mask: torch.Tensor,
    checks: Dict[str, bool],
) -> None:
    """Verify rank-3 [B,L,13] inputs bypass ID lookup and decode to atom37."""
    expected = (*z_quant.shape[:2], 37, 3)
    quant_out = struct_tokenizer.detokenize(z_quant, res_mask=res_mask)
    quant_ok = (
        tuple(quant_out["atom37_positions"].shape) == expected
        and quant_out["atom37_mask"].shape[:2] == z_quant.shape[:2]
        and valid_decoder_coordinates(quant_out, res_mask)
    )
    checks["rank-3 z_quant decode"] = quant_ok
    require(quant_ok, "rank-3 z_quant continuous decode failed")
    quant_shape = tuple(quant_out["atom37_positions"].shape)
    del quant_out

    cont_out = struct_tokenizer.detokenize(z_cont, res_mask=res_mask)
    cont_ok = (
        tuple(cont_out["atom37_positions"].shape) == expected
        and cont_out["atom37_mask"].shape[:2] == z_cont.shape[:2]
        and valid_decoder_coordinates(cont_out, res_mask)
    )
    checks["rank-3 z_cont decode"] = cont_ok
    require(cont_ok, "rank-3 z_cont continuous decode failed")
    cont_shape = tuple(cont_out["atom37_positions"].shape)
    del cont_out

    print("\n[CONTINUOUS LATENT DECODER]")
    print(f"decode(z_quant) atom37 shape: {quant_shape}")
    print(f"decode(z_cont) atom37 shape: {cont_shape}")
    print("finite check: True")


def build_folding_input(
    fasta_path: str,
    pkl_path: str,
    model: DPLM2Bit,
    device: torch.device,
    checks: Dict[str, bool],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Reuse the official helper and select one B=1 folding input [1,2*S]."""
    helper_args = SimpleNamespace(task="folding", batch_size=1)
    batches, name_lists = initialize_conditional_generation(
        fasta_path, model.tokenizer, device, helper_args, model=model
    )
    require(len(batches) > 0, f"no valid sequences found in {fasta_path}")

    requested_id = Path(pkl_path).stem
    selected = None
    for batch, names in zip(batches, name_lists):
        if requested_id in names:
            selected = (batch, names[names.index(requested_id)])
            break
    if selected is None:
        selected = (batches[0], name_lists[0][0])
        print(
            f"[FOLDING INPUT] ID {requested_id!r} not matched; "
            f"using first valid helper sequence {selected[1]!r}"
        )
    batch, selected_name = selected
    input_ids = batch["input_tokens"]
    partial_mask = batch["partial_mask"]
    require(input_ids.shape[0] == 1, "folding debug batch size must be one")
    require(input_ids.shape[1] % 2 == 0, "combined modality length is not even")
    struct_ids, aa_ids = input_ids.chunk(2, dim=1)
    s = struct_ids.shape[1]

    modality_ok = struct_ids.shape == aa_ids.shape == (1, s)
    checks["structure/AA modality split"] = modality_ok
    require(modality_ok, "structure/AA halves do not have identical [B,S] shape")
    require(struct_ids[0, 0].item() == model.struct_bos_id, "structure BOS mismatch")
    require((struct_ids == model.struct_eos_id).sum().item() == 1, "structure EOS mismatch")
    require(aa_ids[0, 0].item() == model.aa_bos_id, "AA BOS mismatch")
    require((aa_ids == model.aa_eos_id).sum().item() == 1, "AA EOS mismatch")
    num_struct_masks = (struct_ids == model.struct_mask_id).sum().item()
    require(num_struct_masks == s - 2, "structure residues are not all mask tokens")

    print("\n[FOLDING INPUT]")
    print(f"selected sequence ID: {selected_name}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"half length S: {s}")
    print(f"structure half shape: {tuple(struct_ids.shape)}")
    print(f"AA half shape: {tuple(aa_ids.shape)}")
    print(f"structure BOS ID: {struct_ids[0, 0].item()}")
    eos_pos = torch.where(struct_ids[0] == model.struct_eos_id)[0].item()
    print(f"structure EOS ID: {struct_ids[0, eos_pos].item()}")
    print(
        f"AA BOS/EOS IDs: {aa_ids[0,0].item()}/"
        f"{model.aa_eos_id} (present=True)"
    )
    print(f"number of structure mask tokens: {num_struct_masks}")
    return input_ids, partial_mask, selected_name


def capture_generation_hidden_states(
    model: DPLM2Bit,
    input_ids: torch.Tensor,
    partial_mask: torch.Tensor,
    max_iter: int,
    checks: Dict[str, bool],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Capture only the last unmodified forward_decoder return, then restore it."""
    require(max_iter > 0, "--max_iter must be positive")
    capture: Dict[str, object] = {}
    original_forward_decoder = model.forward_decoder

    def wrapped_forward_decoder(*args, **kwargs):
        result = original_forward_decoder(*args, **kwargs)
        # Overwrite, never append: only the last iteration's 34 tensors survive.
        capture["output_tokens"] = result["output_tokens"].detach().clone()
        capture["all_hidden_states"] = result["all_hidden_states"]
        return result

    model.forward_decoder = wrapped_forward_decoder
    try:
        generated = model.generate(
            input_tokens=input_ids,
            max_iter=max_iter,
            temperature=1.0,
            unmasking_strategy="stochastic1.0",
            sampling_strategy="argmax",
            partial_masks=partial_mask,
        )
    finally:
        model.forward_decoder = original_forward_decoder

    captured_ok = "output_tokens" in capture and "all_hidden_states" in capture
    checks["prediction-time hidden captured"] = captured_ok
    require(captured_ok, "last forward_decoder hidden state was not captured")
    captured_tokens = capture["output_tokens"]
    final_tokens = generated["output_tokens"]
    token_match = torch.equal(captured_tokens, final_tokens)
    checks["captured output matches final generated output"] = token_match

    print("\n[PREDICTION-TIME HIDDEN CAPTURE]")
    print(f"last decoder output equals public final output: {token_match}")
    if not token_match:
        mismatches = (captured_tokens != final_tokens).sum().item()
        print(
            "diagnostic: forward_decoder sampled tokens were subsequently passed "
            "through generate()._reparam_decoding(); the public result differs at "
            f"{mismatches} positions, so the captured state cannot be claimed to "
            "align with every final returned token."
        )
    return generated, capture


def check_hidden_tuple(
    model: DPLM2Bit,
    all_hidden_states: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
    checks: Dict[str, bool],
) -> List[torch.Tensor]:
    """Validate [embedding/pre-layer, 33 transformer outputs], each [B,2*S,H]."""
    n = model.net.config.num_hidden_layers
    h = model.net.config.hidden_size
    expected_shape = (*input_ids.shape, h)
    tuple_ok = len(all_hidden_states) == n + 1
    shape_ok = all(tuple(x.shape) == expected_shape for x in all_hidden_states)
    checks["hidden tuple = N + 1"] = tuple_ok
    checks["hidden size matches runtime config"] = shape_ok
    checks["33 transformer layers"] = n == EXPECTED_NUM_LAYERS
    layer_states = list(all_hidden_states[1:])
    checks["embedding state excluded from baseline"] = len(layer_states) == n

    require(tuple_ok, f"hidden tuple length {len(all_hidden_states)} != N+1 ({n+1})")
    require(shape_ok, f"one or more hidden tensors differ from {expected_shape}")
    require(len(layer_states) == n, "baseline layer ensemble is not all_hidden_states[1:]")
    print("\n[HIDDEN STATES]")
    print(f"len(all_hidden_states): {len(all_hidden_states)}")
    print(f"num_hidden_layers: {n}")
    print(f"all hidden tensor shape: {expected_shape}")
    print("all_hidden_states[0]: embedding-derived/pre-layer state (excluded)")
    print(f"baseline all_hidden_states[1:]: {len(layer_states)} transformer states")
    return layer_states


def align_structure_hidden_states(
    model: DPLM2Bit,
    combined_tokens: torch.Tensor,
    layer_states: Sequence[torch.Tensor],
    checks: Dict[str, bool],
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Apply source-equivalent BOS/EOS selection to IDs and [B,2*S,H] states."""
    struct_tokens, _ = combined_tokens.chunk(2, dim=1)
    bsz, s = struct_tokens.shape
    residue_positions = struct_tokens.ne(model.struct_eos_id) & struct_tokens.ne(
        model.struct_bos_id
    )
    counts = residue_positions.sum(dim=1)
    require(
        torch.all(counts == s - 2).item(),
        f"prepare_for_struct_tokenizer requires S-2 selected positions, got {counts.tolist()}",
    )

    raw_ids = struct_tokens[residue_positions].view(bsz, s - 2)
    raw_ids = raw_ids - model.struct_vocab_offset
    res_mask = model.get_non_special_symbol_mask(combined_tokens).chunk(2, dim=1)[0]
    res_mask = res_mask[residue_positions].view(bsz, s - 2).int()
    raw_ids = raw_ids.masked_fill(~res_mask.bool(), 0)
    predicted_z_quant = model.struct_tokenizer.quantize.get_codebook_entry(raw_ids)

    aligned = []
    for state in layer_states:
        struct_state, _ = state.chunk(2, dim=1)
        aligned.append(
            struct_state[residue_positions].view(bsz, s - 2, state.shape[-1])
        )

    latent_dim = model.struct_tokenizer.codebook_embed_dim
    expected_z_shape = (bsz, s - 2, latent_dim)
    checks["BOS/EOS removed correctly"] = tuple(counts.tolist()) == (s - 2,) * bsz
    checks["predicted z_quant aligned"] = tuple(predicted_z_quant.shape) == expected_z_shape
    checks["hidden states aligned"] = (
        len(aligned) == model.net.config.num_hidden_layers
        and all(
            x.shape == (bsz, s - 2, model.net.config.hidden_size) for x in aligned
        )
    )
    checks["res_mask aligned"] = (
        res_mask.shape == predicted_z_quant.shape[:2]
        and all(x.shape[:2] == res_mask.shape for x in aligned)
    )
    require(checks["predicted z_quant aligned"], "predicted latent alignment failed")
    require(checks["hidden states aligned"], "transformer hidden alignment failed")
    require(checks["res_mask aligned"], "residue mask alignment failed")

    print("\n[STRUCTURE HIDDEN ALIGNMENT]")
    print(f"predicted raw LFQ IDs shape: {tuple(raw_ids.shape)}")
    print(f"predicted z_quant shape: {tuple(predicted_z_quant.shape)}")
    print(f"aligned transformer layers: {len(aligned)}")
    print(f"aligned hidden shape: {tuple(aligned[-1].shape)}")
    print(f"res_mask shape: {tuple(res_mask.shape)}")
    return predicted_z_quant, res_mask, aligned


def compare_post_generation_hidden(
    model: DPLM2Bit,
    final_tokens: torch.Tensor,
    prediction_layers: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
) -> None:
    """Diagnostic only: this post-generation state is not the canonical condition."""
    post = model(final_tokens)["all_hidden_states"][-1]
    post_struct, _ = post.chunk(2, dim=1)
    final_struct, _ = final_tokens.chunk(2, dim=1)
    positions = final_struct.ne(model.struct_eos_id) & final_struct.ne(
        model.struct_bos_id
    )
    bsz, s = final_struct.shape
    post_aligned = post_struct[positions].view(bsz, s - 2, post.shape[-1])
    valid = res_mask.bool().unsqueeze(-1).expand_as(post_aligned)
    mae = (prediction_layers[-1] - post_aligned).abs()[valid].float().mean()
    print("\n[POST-GENERATION HIDDEN DIAGNOSTIC]")
    print(f"prediction-time vs post-generation final-layer MAE: {mae.item():.6g}")
    print("post-generation hidden is diagnostic only, not the canonical condition")


def check_generated_latent(
    model: DPLM2Bit,
    generated: Dict[str, torch.Tensor],
    rebuilt_z_quant: torch.Tensor,
    rebuilt_res_mask: torch.Tensor,
    checks: Dict[str, bool],
) -> None:
    """Verify generated IDs -> [B,L,13] and rank-3 continuous decoder output."""
    final_feature = generated["final_struct_feature"]
    public_mask = generated["res_mask"]
    latent_ok = (
        final_feature.ndim == 3
        and final_feature.shape[-1] == model.struct_tokenizer.codebook_embed_dim
        and final_feature.shape[:2] == public_mask.shape
    )
    checks["generated z_quant [B,L,13]"] = latent_ok
    require(latent_ok, f"invalid final_struct_feature shape {tuple(final_feature.shape)}")
    require(
        public_mask.shape == rebuilt_res_mask.shape,
        "public and rebuilt generated residue masks differ in shape",
    )
    valid = public_mask.bool()
    max_diff = (
        (final_feature - rebuilt_z_quant).abs()[valid].max().item()
        if valid.any()
        else float("nan")
    )
    same = torch.allclose(final_feature[valid], rebuilt_z_quant[valid], atol=0, rtol=0)
    require(same, f"public final_struct_feature differs from rebuilt LFQ latent: {max_diff}")

    decoded = model.struct_tokenizer.detokenize(final_feature, res_mask=public_mask)
    expected_shape = (*final_feature.shape[:2], 37, 3)
    decode_ok = (
        tuple(decoded["atom37_positions"].shape) == expected_shape
        and valid_decoder_coordinates(decoded, public_mask)
    )
    checks["generated z_quant continuous decode"] = decode_ok
    require(decode_ok, "generated z_quant continuous decoder smoke test failed")

    print("\n[GENERATED LATENT]")
    print(f"final_struct_feature shape: {tuple(final_feature.shape)}")
    print(f"rebuilt latent allclose on valid residues: {same}")
    print(f"max absolute difference: {max_diff:.6g}")
    print(f"generated decode atom37 shape: {tuple(decoded['atom37_positions'].shape)}")
    print("generated decode valid coordinate finite: True")



def check_final_generation_interfaces(
    model: DPLM2Bit,
    generated: Dict[str, torch.Tensor],
    capture: Dict[str, object],
    checks: Dict[str, bool],
) -> None:
    """Validate canonical final-token [B,L,13]/[B,L,H] interfaces.

    The extra frozen forward is called inside main's inference_mode. Hook data
    from the final forward_decoder call is retained only for diagnostics.
    """
    required = {"output_tokens", "res_mask", "final_struct_feature"}
    require(required.issubset(generated), f"generate keys missing: {required - set(generated)}")
    final_tokens = generated["output_tokens"]
    public_mask = generated["res_mask"]
    final_feature = generated["final_struct_feature"]

    print("\n[GENERATION RETURN]")
    for key, value in generated.items():
        print(f"{key} shape = {tuple(value.shape)}")

    latent_ok = (
        final_feature.ndim == 3
        and final_feature.shape[-1] == model.struct_tokenizer.codebook_embed_dim
        and final_feature.shape[-1] == EXPECTED_CODEBOOK_DIM
        and final_feature.shape[:2] == public_mask.shape
    )
    checks["generated final_struct_feature rank-3 latent"] = latent_ok
    require(latent_ok, f"invalid final_struct_feature shape {tuple(final_feature.shape)}")

    # Reuse repository offset/BOS/EOS/padding/LFQ logic exactly.
    non_special = model.get_non_special_symbol_mask(final_tokens)
    rebuilt = model.prepare_for_struct_tokenizer(
        {"output_tokens": final_tokens.clone()}, non_special
    )
    rebuilt_z_quant = rebuilt["final_struct_feature"]
    rebuilt_mask = rebuilt["res_mask"]
    print("\n[FINAL TOKEN LATENT RECONSTRUCTION]")
    print(f"z_quant_from_final_tokens.shape: {tuple(rebuilt_z_quant.shape)}")
    print(f"final_struct_feature.shape: {tuple(final_feature.shape)}")
    print(f"res_mask.shape: {tuple(public_mask.shape)}")
    shapes_match = (
        rebuilt_z_quant.shape == final_feature.shape
        and rebuilt_mask.shape == public_mask.shape
    )
    require(shapes_match, "official rebuilt/public latent or mask shapes differ")
    valid = public_mask.bool()
    require(valid.any().item(), "generated residue mask has no valid positions")
    max_diff = (final_feature - rebuilt_z_quant).abs()[valid].max().item()
    same = torch.allclose(final_feature[valid], rebuilt_z_quant[valid], atol=0, rtol=0)
    checks["final tokens -> z_quant reconstruction"] = same
    print(f"allclose on valid residues: {same}")
    print(f"max absolute difference: {max_diff:.6g}")
    require(same, f"official generated/rebuilt latent mismatch: {max_diff}")

    decoded = model.struct_tokenizer.detokenize(final_feature, res_mask=public_mask)
    atom37 = decoded["atom37_positions"]
    decode_ok = (
        atom37.ndim == 4
        and atom37.shape[0] == final_feature.shape[0]
        and atom37.shape[1] == final_feature.shape[1]
        and tuple(atom37.shape[-2:]) == (37, 3)
        and valid_decoder_coordinates(decoded, public_mask)
    )
    checks["generated continuous latent decode"] = decode_ok
    require(decode_ok, "official generated continuous latent decode failed")
    print(f"generated decode atom37 shape: {tuple(atom37.shape)}")
    print("generated decode valid coordinate finite: True")
    del decoded

    # Canonical FM condition source: final post-update tokens receive one frozen
    # extra forward; embedding state [0] is excluded by check_hidden_tuple().
    final_forward = model(final_tokens)
    final_hidden = final_forward.get("all_hidden_states")
    checks["final-token extra forward hidden obtained"] = final_hidden is not None
    require(final_hidden is not None, "final-token forward returned no hidden tuple")
    final_layers = check_hidden_tuple(model, final_hidden, final_tokens, checks)

    struct_tokens, aa_tokens = final_tokens.chunk(2, dim=1)
    require(struct_tokens.shape == aa_tokens.shape, "final modality split failed")
    bsz, s = struct_tokens.shape
    positions = struct_tokens.ne(model.struct_eos_id) & struct_tokens.ne(model.struct_bos_id)
    counts = positions.sum(dim=1)
    require(torch.all(counts == s - 2).item(), f"BOS/EOS selection counts: {counts.tolist()}")
    aligned_layers = []
    for state in final_layers:
        struct_state, _ = state.chunk(2, dim=1)
        aligned_layers.append(
            struct_state[positions].view(bsz, s - 2, state.shape[-1])
        )
    checks["final-token hidden aligned"] = (
        len(aligned_layers) == model.net.config.num_hidden_layers
        and all(layer.shape == (bsz, s - 2, model.net.config.hidden_size) for layer in aligned_layers)
    )
    checks["predicted z_quant aligned"] = (
        rebuilt_z_quant.shape == (bsz, s - 2, model.struct_tokenizer.codebook_embed_dim)
    )
    checks["res_mask aligned"] = (
        rebuilt_mask.shape == rebuilt_z_quant.shape[:2]
        and all(layer.shape[:2] == rebuilt_mask.shape for layer in aligned_layers)
    )
    require(checks["final-token hidden aligned"], "final-token hidden alignment failed")
    require(checks["predicted z_quant aligned"], "predicted z_quant alignment failed")
    require(checks["res_mask aligned"], "res_mask alignment failed")
    print("\n[FINAL-TOKEN HIDDEN ALIGNMENT]")
    print(f"transformer layer count: {len(aligned_layers)}")
    print(f"aligned hidden shape: {tuple(aligned_layers[-1].shape)}")
    print(f"predicted z_quant shape: {tuple(rebuilt_z_quant.shape)}")
    print(f"res_mask shape: {tuple(rebuilt_mask.shape)}")

    captured_ok = "output_tokens" in capture and "all_hidden_states" in capture
    print(f"\n[DIAGNOSTIC] prediction-time hidden captured: {captured_ok}")
    if captured_ok:
        captured_tokens = capture["output_tokens"]
        token_match = torch.equal(captured_tokens, final_tokens)
        mismatch_count = (captured_tokens != final_tokens).sum().item()
        print(
            "[DIAGNOSTIC] last forward_decoder tokens vs final post-update tokens: "
            f"equal={token_match}, mismatch_count={mismatch_count}, "
            f"valid_structure_residue_count={int(public_mask.sum().item())}"
        )
        prediction_last = capture["all_hidden_states"][-1]
        prediction_struct, _ = prediction_last.chunk(2, dim=1)
        prediction_aligned = prediction_struct[positions].view(
            bsz, s - 2, prediction_last.shape[-1]
        )
        expanded_valid = rebuilt_mask.bool().unsqueeze(-1).expand_as(prediction_aligned)
        mae = (prediction_aligned - aligned_layers[-1]).abs()[expanded_valid]
        print(
            "[DIAGNOSTIC] prediction-time vs post-generation hidden MAE: "
            f"{mae.float().mean().item():.6g}"
        )
    else:
        print("[DIAGNOSTIC] last forward_decoder token comparison unavailable")
        print("[DIAGNOSTIC] prediction-time vs post-generation hidden MAE unavailable")


def print_summary(checks: Dict[str, bool]) -> None:
    print("\n" + "=" * 50)
    print("DPLM-2.1 LATENT RESIDUAL FM INTERFACE CHECK")
    print("=" * 50)
    groups = (3, 3, 2, 4, 4, 3)
    index = 0
    for group_size in groups:
        for name in SUMMARY_ITEMS[index : index + group_size]:
            print(f"[{'PASS' if checks.get(name, False) else 'FAIL'}] {name}")
        print()
        index += group_size
    print("[NOT CONFIRMED]")
    print("- ResDiff training이 Mode A인지 Mode B인지")
    print("- ResDiff condition이 structure hidden인지 AA hidden인지 둘 다인지")
    print("- exact AdaLN topology")
    print("- embedding state가 논문 ResDiff에 실제 포함되는지")
    print(
        "- baseline decision: Transformer 33 layer outputs = "
        "all_hidden_states[1:]"
    )


def main() -> int:
    args = parse_args()
    checks = {name: False for name in SUMMARY_ITEMS}
    seed_everything(args.seed)

    try:
        device = resolve_device(args.device)
        with torch.inference_mode():
            model, struct_tokenizer = load_models(args.model_name, device, checks)
            coordinate_batch = load_coordinate_sample(args.pkl_path, device)
            z_cont, z_quant, _, _ = check_tokenizer_latents(
                struct_tokenizer, coordinate_batch, checks
            )
            check_continuous_decoder(
                struct_tokenizer,
                z_cont,
                z_quant,
                coordinate_batch["res_mask"],
                checks,
            )
            del coordinate_batch, z_cont, z_quant

            input_ids, partial_mask, _ = build_folding_input(
                args.fasta_path, args.pkl_path, model, device, checks
            )
            generated, capture = capture_generation_hidden_states(
                model, input_ids, partial_mask, args.max_iter, checks
            )
            check_final_generation_interfaces(model, generated, capture, checks)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()

    print_summary(checks)
    return 0 if all(checks.get(name, False) for name in SUMMARY_ITEMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

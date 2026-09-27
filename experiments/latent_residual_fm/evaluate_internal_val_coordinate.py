#!/usr/bin/env python3
"""Continuous-latent reconstruction coordinate diagnostics on internal AFDB val."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from byprot.datamodules.pdb_dataset import residue_constants
from byprot.datamodules.pdb_dataset.utils import align_structures
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    GENERATION_MAX_ITER,
    assert_atom37,
    build_folding_input,
    extract_final_token_condition,
    load_fm,
    load_validation_sample,
    masked_mse,
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.flow_matching import euler_sample


CSV_COLUMNS = (
    "val_index",
    "sample_id",
    "length",
    "bit_latent_mse",
    "fm_latent_mse",
    "latent_relative_improvement_percent",
    "bit_ca_rmsd",
    "fm_ca_rmsd",
    "ca_absolute_improvement",
    "ca_relative_improvement_percent",
    "ca_improved",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "bb_absolute_improvement",
    "bb_relative_improvement_percent",
    "bb_improved",
)

CA_INDEX = residue_constants.atom_order["CA"]
BACKBONE_INDICES = tuple(
    residue_constants.atom_order[name] for name in ("N", "CA", "C", "O")
)


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
    parser.add_argument(
        "--output_csv",
        default=(
            "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
            "internal_val_coordinate_20.csv"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_val_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


def validate_decoder_output(
    name: str, decoded: dict[str, torch.Tensor], length: int
) -> None:
    assert_atom37(name, decoded, length)
    require("atom37_mask" in decoded, f"{name} decode lacks atom37_mask")
    atom_mask = decoded["atom37_mask"]
    require(
        tuple(atom_mask.shape) == (1, length, 37),
        f"{name} atom37 mask shape is {tuple(atom_mask.shape)}",
    )
    require(bool(torch.isfinite(atom_mask).all()), f"{name} atom37 mask is non-finite")


def common_residue_mask(
    decoded_outputs: Sequence[dict[str, torch.Tensor]],
    residue_mask: torch.Tensor,
    atom_indices: Sequence[int],
) -> torch.Tensor:
    common = residue_mask.bool().clone()
    indices = torch.tensor(atom_indices, dtype=torch.long, device=residue_mask.device)
    for decoded in decoded_outputs:
        common &= decoded["atom37_mask"][:, :, indices].bool().all(dim=-1)
    return common


def kabsch_atom_rmsd(
    moving: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    residue_mask: torch.Tensor,
    atom_indices: Sequence[int],
    metric_name: str,
) -> float:
    """Kabsch-align selected atoms and return sqrt(mean squared distance)."""
    moving_positions = moving["atom37_positions"][0].float()
    reference_positions = reference["atom37_positions"][0].float()
    moving_atom_mask = moving["atom37_mask"][0].bool()
    reference_atom_mask = reference["atom37_mask"][0].bool()
    require(
        residue_mask.ndim == 2 and residue_mask.shape[0] == 1,
        f"{metric_name} residue mask must have shape [1,L]",
    )
    require(
        moving_positions.shape == reference_positions.shape,
        f"{metric_name} coordinate shapes differ",
    )

    indices = torch.tensor(atom_indices, dtype=torch.long, device=moving_positions.device)
    valid_residue = residue_mask[0].bool()
    valid_residue = valid_residue & moving_atom_mask[:, indices].all(dim=-1)
    valid_residue = valid_residue & reference_atom_mask[:, indices].all(dim=-1)
    valid_count = int(valid_residue.sum())
    require(valid_count >= 3, f"{metric_name} requires at least 3 valid residues")

    moving_points = moving_positions[valid_residue][:, indices].reshape(-1, 3).double()
    reference_points = (
        reference_positions[valid_residue][:, indices].reshape(-1, 3).double()
    )
    require(
        bool(torch.isfinite(moving_points).all()),
        f"{metric_name} moving coordinates are non-finite",
    )
    require(
        bool(torch.isfinite(reference_points).all()),
        f"{metric_name} reference coordinates are non-finite",
    )

    # Repository-native align_structures implements Kabsch with
    # torch.linalg.svd and a determinant-based reflection correction.
    batch_indices = torch.zeros(
        moving_points.shape[0], dtype=torch.long, device=moving_points.device
    )
    aligned, centered_reference, _ = align_structures(
        moving_points, batch_indices, reference_points
    )
    squared_distance = (aligned - centered_reference).square().sum(dim=-1)
    rmsd = torch.sqrt(squared_distance.mean())
    require(bool(torch.isfinite(rmsd)), f"{metric_name} RMSD is non-finite")

    self_aligned, self_reference, _ = align_structures(
        reference_points, batch_indices, reference_points
    )
    self_rmsd = torch.sqrt(
        (self_aligned - self_reference).square().sum(dim=-1).mean()
    )
    require(
        bool(torch.isfinite(self_rmsd)),
        f"{metric_name} reference self-RMSD is non-finite",
    )
    require(
        float(self_rmsd) <= 1e-6,
        f"{metric_name} reference self-RMSD is not near zero: {float(self_rmsd):.9g}",
    )
    return float(rmsd)


def relative_improvement(baseline: float, proposed: float, name: str) -> float:
    require(math.isfinite(baseline) and math.isfinite(proposed), f"{name} is non-finite")
    require(baseline > 0.0, f"{name} baseline must be positive")
    return (baseline - proposed) / baseline * 100.0


def evaluate_one(
    *,
    model: torch.nn.Module,
    struct_tokenizer: torch.nn.Module,
    fm: torch.nn.Module,
    sample: dict[str, Any],
    val_index: int,
    base_seed: int,
    num_flow_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype = sample["aatype"].long()
    z_cont = sample["z_cont"].float().view(
        1, length, EXPECTED_CODEBOOK_DIM
    ).to(device)
    diagnostic_mask = sample["res_mask"].float().view(1, length).to(device)
    require(bool(torch.isfinite(z_cont).all()), "stored z_cont is non-finite")
    require(bool(torch.isfinite(diagnostic_mask).all()), "stored mask is non-finite")

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
    predicted_z_quant, generated_mask = reconstruct_prediction_latent(
        model, final_tokens
    )
    require(
        tuple(predicted_z_quant.shape) == (1, length, EXPECTED_CODEBOOK_DIM),
        f"predicted z_quant shape is {tuple(predicted_z_quant.shape)}",
    )
    require(tuple(generated_mask.shape) == (1, length), "generated mask is misaligned")
    require(bool(generated_mask.bool().all()), "generated sample has masked residues")

    # The reused helper obtains only final-token, structure-side Transformer
    # outputs 1..33. Generation-intermediate, AA-side, embedding, and GT-token
    # teacher-forced hidden states never enter the FM condition.
    hidden_states = extract_final_token_condition(model, final_tokens, length)
    generator = torch.Generator(device=device)
    generator.manual_seed(base_seed + val_index)
    r0 = torch.randn(
        predicted_z_quant.shape,
        dtype=predicted_z_quant.dtype,
        device=device,
        generator=generator,
    )
    r_hat = euler_sample(
        fm,
        predicted_z_quant,
        hidden_states,
        generated_mask,
        num_steps=num_flow_steps,
        initial_noise=r0,
    )
    z_refined = predicted_z_quant + r_hat
    require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")
    require(bool(torch.isfinite(z_refined).all()), "z_refined is non-finite")

    # The stored residue mask is applied identically to all three decoder calls
    # and all coordinate/latent diagnostics.
    oracle_decode = struct_tokenizer.detokenize(z_cont, res_mask=diagnostic_mask)
    bit_decode = struct_tokenizer.detokenize(
        predicted_z_quant, res_mask=diagnostic_mask
    )
    fm_decode = struct_tokenizer.detokenize(z_refined, res_mask=diagnostic_mask)
    validate_decoder_output("oracle continuous", oracle_decode, length)
    validate_decoder_output("Bit-only", bit_decode, length)
    validate_decoder_output("FM-refined", fm_decode, length)
    decoded_outputs = (oracle_decode, bit_decode, fm_decode)
    ca_metric_mask = common_residue_mask(
        decoded_outputs, diagnostic_mask, (CA_INDEX,)
    )
    bb_metric_mask = common_residue_mask(
        decoded_outputs, diagnostic_mask, BACKBONE_INDICES
    )

    bit_latent_mse = masked_mse(predicted_z_quant, z_cont, diagnostic_mask)
    fm_latent_mse = masked_mse(z_refined, z_cont, diagnostic_mask)
    latent_relative = relative_improvement(
        bit_latent_mse, fm_latent_mse, "latent MSE"
    )

    bit_ca_rmsd = kabsch_atom_rmsd(
        bit_decode, oracle_decode, ca_metric_mask, (CA_INDEX,), "Bit-only CA"
    )
    fm_ca_rmsd = kabsch_atom_rmsd(
        fm_decode, oracle_decode, ca_metric_mask, (CA_INDEX,), "FM-refined CA"
    )
    ca_absolute = bit_ca_rmsd - fm_ca_rmsd
    ca_relative = relative_improvement(bit_ca_rmsd, fm_ca_rmsd, "CA RMSD")

    bit_bb_rmsd = kabsch_atom_rmsd(
        bit_decode,
        oracle_decode,
        bb_metric_mask,
        BACKBONE_INDICES,
        "Bit-only backbone",
    )
    fm_bb_rmsd = kabsch_atom_rmsd(
        fm_decode,
        oracle_decode,
        bb_metric_mask,
        BACKBONE_INDICES,
        "FM-refined backbone",
    )
    bb_absolute = bit_bb_rmsd - fm_bb_rmsd
    bb_relative = relative_improvement(bit_bb_rmsd, fm_bb_rmsd, "backbone RMSD")

    result = {
        "val_index": val_index,
        "sample_id": sample_id,
        "length": length,
        "bit_latent_mse": bit_latent_mse,
        "fm_latent_mse": fm_latent_mse,
        "latent_relative_improvement_percent": latent_relative,
        "bit_ca_rmsd": bit_ca_rmsd,
        "fm_ca_rmsd": fm_ca_rmsd,
        "ca_absolute_improvement": ca_absolute,
        "ca_relative_improvement_percent": ca_relative,
        "ca_improved": fm_ca_rmsd < bit_ca_rmsd,
        "bit_bb_rmsd": bit_bb_rmsd,
        "fm_bb_rmsd": fm_bb_rmsd,
        "bb_absolute_improvement": bb_absolute,
        "bb_relative_improvement_percent": bb_relative,
        "bb_improved": fm_bb_rmsd < bit_bb_rmsd,
    }
    print(
        f"[sample {val_index}] id={sample_id} L={length} "
        f"CA(bit={bit_ca_rmsd:.6g}, fm={fm_ca_rmsd:.6g}, "
        f"improved={result['ca_improved']}) "
        f"BB(bit={bit_bb_rmsd:.6g}, fm={fm_bb_rmsd:.6g}, "
        f"improved={result['bb_improved']})"
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict[str, float]:
    require(values, "cannot summarize empty metrics")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def print_metric_aggregate(
    label: str,
    rows: list[dict[str, Any]],
    baseline_key: str,
    proposed_key: str,
    absolute_key: str,
    relative_key: str,
    improved_key: str,
) -> None:
    baseline = describe([float(row[baseline_key]) for row in rows])
    proposed = describe([float(row[proposed_key]) for row in rows])
    absolute = describe([float(row[absolute_key]) for row in rows])
    relative = describe([float(row[relative_key]) for row in rows])
    num_improved = sum(bool(row[improved_key]) for row in rows)
    best = max(rows, key=lambda row: float(row[absolute_key]))
    worst = min(rows, key=lambda row: float(row[absolute_key]))

    print(f"\n{label}:")
    print(f"num_improved: {num_improved}")
    print(f"improved_percent: {num_improved / len(rows) * 100.0:.6f}")
    print(f"Bit-only {label} RMSD mean: {baseline['mean']:.9g}")
    print(f"Bit-only {label} RMSD median: {baseline['median']:.9g}")
    print(f"Bit-only {label} RMSD min: {baseline['min']:.9g}")
    print(f"Bit-only {label} RMSD max: {baseline['max']:.9g}")
    print(f"FM-refined {label} RMSD mean: {proposed['mean']:.9g}")
    print(f"FM-refined {label} RMSD median: {proposed['median']:.9g}")
    print(f"FM-refined {label} RMSD min: {proposed['min']:.9g}")
    print(f"FM-refined {label} RMSD max: {proposed['max']:.9g}")
    print(f"{label} absolute improvement mean: {absolute['mean']:.9g}")
    print(f"{label} absolute improvement median: {absolute['median']:.9g}")
    print(f"{label} relative improvement percent mean: {relative['mean']:.9g}")
    print(f"{label} relative improvement percent median: {relative['median']:.9g}")
    print(
        f"best {label} improvement sample: val_index={best['val_index']} "
        f"sample_id={best['sample_id']} improvement={float(best[absolute_key]):.9g}"
    )
    print(
        f"worst {label} improvement sample: val_index={worst['val_index']} "
        f"sample_id={worst['sample_id']} improvement={float(worst[absolute_key]):.9g}"
    )


def print_aggregate(
    rows: list[dict[str, Any]], output_csv: Path, device: torch.device
) -> None:
    print(
        "\n[CONTINUOUS-LATENT RECONSTRUCTION COORDINATE DIAGNOSTIC; "
        "NOT A FOLDING BENCHMARK OR GROUND-TRUTH RMSD]"
    )
    print(f"num_samples: {len(rows)}")
    print_metric_aggregate(
        "CA",
        rows,
        "bit_ca_rmsd",
        "fm_ca_rmsd",
        "ca_absolute_improvement",
        "ca_relative_improvement_percent",
        "ca_improved",
    )
    print_metric_aggregate(
        "backbone",
        rows,
        "bit_bb_rmsd",
        "fm_bb_rmsd",
        "bb_absolute_improvement",
        "bb_relative_improvement_percent",
        "bb_improved",
    )

    bit_latent_mean = statistics.fmean(float(row["bit_latent_mse"]) for row in rows)
    fm_latent_mean = statistics.fmean(float(row["fm_latent_mse"]) for row in rows)
    latent_improved = sum(
        float(row["fm_latent_mse"]) < float(row["bit_latent_mse"]) for row in rows
    )
    print("\nlatent diagnostic:")
    print(f"Bit-only latent MSE mean: {bit_latent_mean:.9g}")
    print(f"FM latent MSE mean: {fm_latent_mean:.9g}")
    print(f"latent improved percent: {latent_improved / len(rows) * 100.0:.6f}")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated = peak_reserved = 0.0
    print(f"GPU peak allocated MiB: {peak_allocated:.2f}")
    print(f"GPU peak reserved MiB: {peak_reserved:.2f}")
    print(f"output CSV path: {output_csv}")


def main() -> int:
    args = parse_args()
    require(args.start_val_index >= 0, "--start_val_index must be non-negative")
    require(args.num_samples > 0, "--num_samples must be positive")
    require(args.num_flow_steps == 100, "this diagnostic requires 100 Euler steps")
    require(GENERATION_MAX_ITER == 100, "folding generation requires max_iter=100")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    checks: dict[str, bool] = {}
    model, struct_tokenizer = load_models(args.dplm_checkpoint, device, checks)
    require(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        == 0,
        "DPLM trainable parameter count is not zero",
    )
    fm, _, _, _ = load_fm(Path(args.checkpoint), device)

    rows: list[dict[str, Any]] = []
    dataset_dir = Path(args.dataset_dir)
    split_csv = Path(args.split_csv)
    with torch.inference_mode():
        for val_index in range(
            args.start_val_index, args.start_val_index + args.num_samples
        ):
            sample = load_validation_sample(dataset_dir, split_csv, val_index)
            rows.append(
                evaluate_one(
                    model=model,
                    struct_tokenizer=struct_tokenizer,
                    fm=fm,
                    sample=sample,
                    val_index=val_index,
                    base_seed=args.seed,
                    num_flow_steps=args.num_flow_steps,
                    device=device,
                )
            )
            del sample

    require(len(rows) == args.num_samples, "not all requested samples succeeded")
    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print_aggregate(rows, output_csv, device)
    print("\nINTERNAL_VAL_COORDINATE_EVAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

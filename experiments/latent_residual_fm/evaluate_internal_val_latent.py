#!/usr/bin/env python3
"""Evaluate prediction-time latent diagnostics on internal AFDB validation samples."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import torch

from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    GENERATION_MAX_ITER,
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
    "bit_mse",
    "fm_mse",
    "absolute_improvement",
    "relative_improvement_percent",
    "improved",
    "r_hat_abs_mean",
    "r_hat_abs_max",
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
    parser.add_argument("--output_csv", default=(
        "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
        "internal_val_latent_20.csv"
    ))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_val_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


def evaluate_one(
    *,
    model: torch.nn.Module,
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
    require(
        tuple(generated_mask.shape) == (1, length),
        "generated residue mask is not aligned",
    )
    require(bool(generated_mask.bool().all()), "generated sample has masked residues")

    # The reused helper performs the required extra frozen forward on the final
    # public output tokens, excludes all_hidden_states[0], and residue-aligns
    # only the structure-side states.  No generation-intermediate hidden is used.
    hidden_states = extract_final_token_condition(model, final_tokens, length)

    sample_seed = base_seed + val_index
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(sample_seed)
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
        generated_mask,
        num_steps=num_flow_steps,
        initial_noise=r0,
    )
    z_refined = predicted_z_quant + r_hat

    require(bool(torch.isfinite(predicted_z_quant).all()), "predicted z_quant is non-finite")
    require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")
    require(bool(torch.isfinite(z_refined).all()), "z_refined is non-finite")
    require(bool(torch.isfinite(z_cont).all()), "stored z_cont is non-finite")

    bit_mse = masked_mse(predicted_z_quant, z_cont, diagnostic_mask)
    fm_mse = masked_mse(z_refined, z_cont, diagnostic_mask)
    require(bit_mse > 0.0, "bit_mse must be positive for relative improvement")
    absolute_improvement = bit_mse - fm_mse
    relative_improvement = absolute_improvement / bit_mse * 100.0
    valid_r_hat = r_hat.float().abs()[generated_mask.bool()]
    require(valid_r_hat.numel() > 0, "r_hat has no valid residues")

    result = {
        "val_index": val_index,
        "sample_id": sample_id,
        "length": length,
        "bit_mse": bit_mse,
        "fm_mse": fm_mse,
        "absolute_improvement": absolute_improvement,
        "relative_improvement_percent": relative_improvement,
        "improved": fm_mse < bit_mse,
        "r_hat_abs_mean": float(valid_r_hat.mean()),
        "r_hat_abs_max": float(valid_r_hat.max()),
    }
    print(
        f"[sample {val_index}] id={sample_id} L={length} "
        f"bit_mse={bit_mse:.9g} fm_mse={fm_mse:.9g} "
        f"absolute_improvement={absolute_improvement:.9g} "
        f"improved={result['improved']} seed={sample_seed}"
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict[str, float]:
    require(values, "cannot summarize an empty metric list")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def print_aggregate(
    rows: list[dict[str, Any]], output_csv: Path, device: torch.device
) -> None:
    bit = describe([float(row["bit_mse"]) for row in rows])
    fm = describe([float(row["fm_mse"]) for row in rows])
    absolute = describe([float(row["absolute_improvement"]) for row in rows])
    relative = describe(
        [float(row["relative_improvement_percent"]) for row in rows]
    )
    num_improved = sum(bool(row["improved"]) for row in rows)
    improved_percent = num_improved / len(rows) * 100.0
    best = max(rows, key=lambda row: float(row["absolute_improvement"]))
    worst = min(rows, key=lambda row: float(row["absolute_improvement"]))

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated = peak_reserved = 0.0

    print("\n[PREDICTION-TIME LATENT DIAGNOSTIC; NOT A FOLDING BENCHMARK]")
    print(f"num_samples: {len(rows)}")
    print(f"num_improved: {num_improved}")
    print(f"improved_percent: {improved_percent:.6f}")
    print("\nBit-only MSE:")
    print(f"mean: {bit['mean']:.9g}")
    print(f"median: {bit['median']:.9g}")
    print(f"min: {bit['min']:.9g}")
    print(f"max: {bit['max']:.9g}")
    print("\nFM-refined MSE:")
    print(f"mean: {fm['mean']:.9g}")
    print(f"median: {fm['median']:.9g}")
    print(f"min: {fm['min']:.9g}")
    print(f"max: {fm['max']:.9g}")
    print("\nabsolute improvement:")
    print(f"mean: {absolute['mean']:.9g}")
    print(f"median: {absolute['median']:.9g}")
    print("\nrelative improvement percent:")
    print(f"mean: {relative['mean']:.9g}")
    print(f"median: {relative['median']:.9g}")
    print(
        "best improvement sample: "
        f"val_index={best['val_index']} sample_id={best['sample_id']} "
        f"absolute_improvement={float(best['absolute_improvement']):.9g}"
    )
    print(
        "worst improvement sample: "
        f"val_index={worst['val_index']} sample_id={worst['sample_id']} "
        f"absolute_improvement={float(worst['absolute_improvement']):.9g}"
    )
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
    model, _ = load_models(args.dplm_checkpoint, device, checks)
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
    print("\nINTERNAL_VAL_LATENT_EVAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

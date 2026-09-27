#!/usr/bin/env python3
"""Continuous-latent reconstruction alpha-sweep diagnostic on internal AFDB val."""

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
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.evaluate_internal_val_coordinate import (
    BACKBONE_INDICES,
    CA_INDEX,
    common_residue_mask,
    kabsch_atom_rmsd,
    relative_improvement,
    validate_decoder_output,
)
from experiments.latent_residual_fm.evaluate_internal_val_latent import describe
from experiments.latent_residual_fm.flow_matching import euler_sample


ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
CSV_COLUMNS = (
    "val_index",
    "sample_id",
    "length",
    "alpha",
    "ca_rmsd",
    "bb_rmsd",
    "bit_ca_rmsd",
    "bit_bb_rmsd",
    "ca_absolute_improvement_vs_alpha0",
    "bb_absolute_improvement_vs_alpha0",
    "ca_relative_improvement_percent_vs_alpha0",
    "bb_relative_improvement_percent_vs_alpha0",
    "ca_improved_vs_alpha0",
    "bb_improved_vs_alpha0",
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
            "internal_val_alpha_sweep_20.csv"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_val_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


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
) -> list[dict[str, Any]]:
    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype = sample["aatype"].long()
    z_cont = sample["z_cont"].float().view(
        1, length, EXPECTED_CODEBOOK_DIM
    ).to(device)
    diagnostic_mask = sample["res_mask"].float().view(1, length).to(device)
    require(bool(torch.isfinite(z_cont).all()), "stored z_cont is non-finite")
    require(bool(torch.isfinite(diagnostic_mask).all()), "stored mask is non-finite")

    # These expensive prediction-time operations occur exactly once per sample.
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

    # This is the single extra frozen forward on final public output_tokens.
    # The reused helper returns only structure-side all_hidden_states[1:34].
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
    require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")
    full_refined = predicted_z_quant + r_hat

    oracle_decode = struct_tokenizer.detokenize(z_cont, res_mask=diagnostic_mask)
    validate_decoder_output("oracle continuous", oracle_decode, length)

    alpha_latents: dict[float, torch.Tensor] = {}
    alpha_decodes: dict[float, dict[str, torch.Tensor]] = {}
    for alpha in ALPHAS:
        z_alpha = predicted_z_quant + alpha * r_hat
        require(bool(torch.isfinite(z_alpha).all()), f"alpha={alpha:.2f} latent is non-finite")
        alpha_latents[alpha] = z_alpha
        decoded = struct_tokenizer.detokenize(z_alpha, res_mask=diagnostic_mask)
        validate_decoder_output(f"alpha={alpha:.2f}", decoded, length)
        alpha_decodes[alpha] = decoded

    require(
        torch.allclose(alpha_latents[0.0], predicted_z_quant, atol=0.0, rtol=0.0),
        "alpha=0 latent is not exactly Bit-only predicted_z_quant",
    )
    require(
        torch.allclose(alpha_latents[1.0], full_refined, atol=0.0, rtol=0.0),
        "alpha=1 latent is not exactly predicted_z_quant + r_hat",
    )

    all_decodes = (oracle_decode,) + tuple(alpha_decodes[alpha] for alpha in ALPHAS)
    ca_metric_mask = common_residue_mask(
        all_decodes, diagnostic_mask, (CA_INDEX,)
    )
    bb_metric_mask = common_residue_mask(
        all_decodes, diagnostic_mask, BACKBONE_INDICES
    )

    # alpha=0 uses the exact same latent and decoder definition as the existing
    # Bit-only coordinate diagnostic and is the shared baseline for every alpha.
    bit_decode = alpha_decodes[0.0]
    bit_ca_rmsd = kabsch_atom_rmsd(
        bit_decode, oracle_decode, ca_metric_mask, (CA_INDEX,), "alpha=0 CA"
    )
    bit_bb_rmsd = kabsch_atom_rmsd(
        bit_decode,
        oracle_decode,
        bb_metric_mask,
        BACKBONE_INDICES,
        "alpha=0 backbone",
    )

    rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        if alpha == 0.0:
            ca_rmsd = bit_ca_rmsd
            bb_rmsd = bit_bb_rmsd
        else:
            ca_rmsd = kabsch_atom_rmsd(
                alpha_decodes[alpha],
                oracle_decode,
                ca_metric_mask,
                (CA_INDEX,),
                f"alpha={alpha:.2f} CA",
            )
            bb_rmsd = kabsch_atom_rmsd(
                alpha_decodes[alpha],
                oracle_decode,
                bb_metric_mask,
                BACKBONE_INDICES,
                f"alpha={alpha:.2f} backbone",
            )
        ca_absolute = bit_ca_rmsd - ca_rmsd
        bb_absolute = bit_bb_rmsd - bb_rmsd
        rows.append(
            {
                "val_index": val_index,
                "sample_id": sample_id,
                "length": length,
                "alpha": alpha,
                "ca_rmsd": ca_rmsd,
                "bb_rmsd": bb_rmsd,
                "bit_ca_rmsd": bit_ca_rmsd,
                "bit_bb_rmsd": bit_bb_rmsd,
                "ca_absolute_improvement_vs_alpha0": ca_absolute,
                "bb_absolute_improvement_vs_alpha0": bb_absolute,
                "ca_relative_improvement_percent_vs_alpha0": relative_improvement(
                    bit_ca_rmsd, ca_rmsd, "CA RMSD"
                ),
                "bb_relative_improvement_percent_vs_alpha0": relative_improvement(
                    bit_bb_rmsd, bb_rmsd, "backbone RMSD"
                ),
                "ca_improved_vs_alpha0": ca_rmsd < bit_ca_rmsd,
                "bb_improved_vs_alpha0": bb_rmsd < bit_bb_rmsd,
            }
        )

    require(len(rows) == len(ALPHAS), "sample did not produce all alpha rows")
    print(
        f"[sample {val_index}] id={sample_id} L={length} "
        f"seed={base_seed + val_index} alpha_rows={len(rows)}"
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_alpha(rows: list[dict[str, Any]]) -> dict[str, float]:
    ca = describe([float(row["ca_rmsd"]) for row in rows])
    bb = describe([float(row["bb_rmsd"]) for row in rows])
    ca_absolute = describe(
        [float(row["ca_absolute_improvement_vs_alpha0"]) for row in rows]
    )
    bb_absolute = describe(
        [float(row["bb_absolute_improvement_vs_alpha0"]) for row in rows]
    )
    ca_relative = describe(
        [float(row["ca_relative_improvement_percent_vs_alpha0"]) for row in rows]
    )
    bb_relative = describe(
        [float(row["bb_relative_improvement_percent_vs_alpha0"]) for row in rows]
    )
    ca_num_improved = sum(bool(row["ca_improved_vs_alpha0"]) for row in rows)
    bb_num_improved = sum(bool(row["bb_improved_vs_alpha0"]) for row in rows)
    return {
        "ca_mean": ca["mean"],
        "ca_median": ca["median"],
        "ca_min": ca["min"],
        "ca_max": ca["max"],
        "ca_num_improved": float(ca_num_improved),
        "ca_improved_percent": ca_num_improved / len(rows) * 100.0,
        "ca_absolute_mean": ca_absolute["mean"],
        "ca_absolute_median": ca_absolute["median"],
        "ca_relative_mean": ca_relative["mean"],
        "ca_relative_median": ca_relative["median"],
        "ca_worst_degradation": max(
            float(row["ca_rmsd"]) - float(row["bit_ca_rmsd"]) for row in rows
        ),
        "bb_mean": bb["mean"],
        "bb_median": bb["median"],
        "bb_min": bb["min"],
        "bb_max": bb["max"],
        "bb_num_improved": float(bb_num_improved),
        "bb_improved_percent": bb_num_improved / len(rows) * 100.0,
        "bb_absolute_mean": bb_absolute["mean"],
        "bb_absolute_median": bb_absolute["median"],
        "bb_relative_mean": bb_relative["mean"],
        "bb_relative_median": bb_relative["median"],
        "bb_worst_degradation": max(
            float(row["bb_rmsd"]) - float(row["bit_bb_rmsd"]) for row in rows
        ),
    }


def print_alpha_summary(alpha: float, summary: dict[str, float]) -> None:
    print(f"\nalpha: {alpha:.2f}")
    print("CA:")
    print(f"mean RMSD: {summary['ca_mean']:.9g}")
    print(f"median RMSD: {summary['ca_median']:.9g}")
    print(f"min: {summary['ca_min']:.9g}")
    print(f"max: {summary['ca_max']:.9g}")
    print(f"num improved vs alpha=0: {int(summary['ca_num_improved'])}")
    print(f"improved percent: {summary['ca_improved_percent']:.6f}")
    print(f"mean absolute improvement vs alpha=0: {summary['ca_absolute_mean']:.9g}")
    print(f"median absolute improvement vs alpha=0: {summary['ca_absolute_median']:.9g}")
    print(f"mean relative improvement percent: {summary['ca_relative_mean']:.9g}")
    print(f"median relative improvement percent: {summary['ca_relative_median']:.9g}")
    print(f"worst CA degradation vs alpha=0: {summary['ca_worst_degradation']:.9g}")
    print("Backbone:")
    print(f"mean RMSD: {summary['bb_mean']:.9g}")
    print(f"median RMSD: {summary['bb_median']:.9g}")
    print(f"min: {summary['bb_min']:.9g}")
    print(f"max: {summary['bb_max']:.9g}")
    print(f"num improved vs alpha=0: {int(summary['bb_num_improved'])}")
    print(f"improved percent: {summary['bb_improved_percent']:.6f}")
    print(f"mean absolute improvement vs alpha=0: {summary['bb_absolute_mean']:.9g}")
    print(f"median absolute improvement vs alpha=0: {summary['bb_absolute_median']:.9g}")
    print(f"mean relative improvement percent: {summary['bb_relative_mean']:.9g}")
    print(f"median relative improvement percent: {summary['bb_relative_median']:.9g}")
    print(f"worst BB degradation vs alpha=0: {summary['bb_worst_degradation']:.9g}")


def print_aggregate(
    rows: list[dict[str, Any]], output_csv: Path, device: torch.device
) -> None:
    summaries: dict[float, dict[str, float]] = {}
    print(
        "\n[CONTINUOUS-LATENT RECONSTRUCTION ALPHA-SWEEP DIAGNOSTIC; "
        "INTERNAL VALIDATION ABLATION, NOT A FINAL FOLDING BENCHMARK]"
    )
    for alpha in ALPHAS:
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        require(len(alpha_rows) * len(ALPHAS) == len(rows), f"alpha={alpha} row count mismatch")
        summaries[alpha] = summarize_alpha(alpha_rows)
        print_alpha_summary(alpha, summaries[alpha])

    lowest_mean_ca = min(ALPHAS, key=lambda alpha: summaries[alpha]["ca_mean"])
    lowest_median_ca = min(ALPHAS, key=lambda alpha: summaries[alpha]["ca_median"])
    highest_ca_improved = max(
        ALPHAS, key=lambda alpha: summaries[alpha]["ca_improved_percent"]
    )
    lowest_mean_bb = min(ALPHAS, key=lambda alpha: summaries[alpha]["bb_mean"])
    lowest_median_bb = min(ALPHAS, key=lambda alpha: summaries[alpha]["bb_median"])
    print("\n[SEPARATE INTERNAL-VALIDATION ALPHA CRITERIA]")
    print(f"lowest mean CA RMSD alpha: {lowest_mean_ca:.2f}")
    print(f"lowest median CA RMSD alpha: {lowest_median_ca:.2f}")
    print(f"highest CA improved-percent alpha: {highest_ca_improved:.2f}")
    print(f"lowest mean backbone RMSD alpha: {lowest_mean_bb:.2f}")
    print(f"lowest median backbone RMSD alpha: {lowest_median_bb:.2f}")
    print("These are separate internal-validation ablation criteria, not test-set tuning.")

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
            rows.extend(
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

    expected_rows = args.num_samples * len(ALPHAS)
    require(len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}")
    require(
        len({int(row["val_index"]) for row in rows}) == args.num_samples,
        "not all requested validation samples succeeded",
    )
    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print_aggregate(rows, output_csv, device)
    print("\nINTERNAL_VAL_ALPHA_SWEEP_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

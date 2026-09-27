#!/usr/bin/env python3
"""Random-100 continuous-latent reconstruction alpha-sweep diagnostic."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from experiments.latent_residual_fm.debug_interfaces import (
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    GENERATION_MAX_ITER,
    load_fm,
    load_validation_sample,
)
from experiments.latent_residual_fm.evaluate_internal_val_alpha_sweep import (
    ALPHAS,
    CSV_COLUMNS,
    evaluate_one,
    print_aggregate,
    summarize_alpha,
    write_csv,
)


EXPECTED_VALIDATION_SAMPLES = 2_048
EXPECTED_SELECTED_SAMPLES = 100


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
            "internal_val_alpha_sweep_100.csv"
        ),
    )
    parser.add_argument(
        "--indices_csv",
        default=(
            "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
            "internal_val_alpha_sweep_100_indices.csv"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--noise_base_seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


def count_validation_rows(split_csv: Path) -> int:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require("split" in (reader.fieldnames or ()), "split CSV lacks split column")
        return sum(row["split"] == "val" for row in reader)


def select_validation_indices(
    num_validation_samples: int, num_samples: int, selection_seed: int
) -> list[int]:
    require(
        num_validation_samples == EXPECTED_VALIDATION_SAMPLES,
        f"expected {EXPECTED_VALIDATION_SAMPLES} validation samples, "
        f"got {num_validation_samples}",
    )
    require(
        num_samples == EXPECTED_SELECTED_SAMPLES,
        f"this ablation requires exactly {EXPECTED_SELECTED_SAMPLES} samples",
    )
    selected = random.Random(selection_seed).sample(
        range(num_validation_samples), num_samples
    )
    repeated = random.Random(selection_seed).sample(
        range(num_validation_samples), num_samples
    )
    require(selected == repeated, "deterministic validation re-selection mismatch")
    require(len(selected) == num_samples, "selected sample count mismatch")
    require(len(set(selected)) == num_samples, "selected validation indices are not unique")
    require(
        all(0 <= index < num_validation_samples for index in selected),
        "selected validation index is out of range",
    )
    require(
        set(selected) != set(range(num_samples)),
        "selection must not be the first 100 validation samples",
    )
    return selected


def write_indices_csv(path: Path, selected_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("selection_order", "val_index"))
        writer.writeheader()
        for selection_order, val_index in enumerate(selected_indices):
            writer.writerow(
                {"selection_order": selection_order, "val_index": val_index}
            )


def validate_output_rows(rows: list[dict[str, Any]], selected_indices: list[int]) -> None:
    expected_rows = EXPECTED_SELECTED_SAMPLES * len(ALPHAS)
    require(len(rows) == expected_rows, f"expected {expected_rows} rows, got {len(rows)}")
    counts = Counter(int(row["val_index"]) for row in rows)
    require(set(counts) == set(selected_indices), "output validation index set mismatch")
    require(
        all(count == len(ALPHAS) for count in counts.values()),
        "each validation index must have exactly five alpha rows",
    )
    for val_index in selected_indices:
        sample_alphas = {
            float(row["alpha"]) for row in rows if int(row["val_index"]) == val_index
        }
        require(sample_alphas == set(ALPHAS), f"val_index={val_index} alpha set mismatch")
    for alpha in ALPHAS:
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        require(
            len(alpha_rows) == EXPECTED_SELECTED_SAMPLES,
            f"alpha={alpha:.2f} must have exactly 100 rows",
        )

    numeric_columns = tuple(
        column
        for column in CSV_COLUMNS
        if column not in {"sample_id", "ca_improved_vs_alpha0", "bb_improved_vs_alpha0"}
    )
    for row in rows:
        for column in numeric_columns:
            require(
                math.isfinite(float(row[column])),
                f"non-finite output value in {column}",
            )


def print_preregistered_selection_rule() -> None:
    print("[PRE-REGISTERED INTERNAL-VALIDATION ALPHA SELECTION RULE]")
    print("PRIMARY: lowest mean CA RMSD on this 100-sample internal validation ablation")
    print("SECONDARY DIAGNOSTICS:")
    print("- median CA RMSD")
    print("- CA improved percent")
    print("- worst CA degradation")
    print("- backbone metrics")
    print(
        "If primary and secondary diagnostics conflict, all criteria are reported "
        "without changing alpha based on anticipated test performance."
    )


def print_primary_candidate(rows: list[dict[str, Any]]) -> None:
    summaries = {
        alpha: summarize_alpha(
            [row for row in rows if float(row["alpha"]) == alpha]
        )
        for alpha in ALPHAS
    }
    primary_alpha = min(ALPHAS, key=lambda alpha: summaries[alpha]["ca_mean"])
    print("\n[PRIMARY INTERNAL-VALIDATION CANDIDATE]")
    print(f"lowest mean CA RMSD alpha: {primary_alpha:.2f}")
    print(
        "This candidate is fixed from the internal-validation ablation before "
        "examining any later test-set results."
    )


def main() -> int:
    args = parse_args()
    require(args.num_samples == EXPECTED_SELECTED_SAMPLES, "--num_samples must be 100")
    require(args.num_flow_steps == 100, "this diagnostic requires 100 Euler steps")
    require(GENERATION_MAX_ITER == 100, "folding generation requires max_iter=100")
    print_preregistered_selection_rule()

    split_csv = Path(args.split_csv)
    validation_total = count_validation_rows(split_csv)
    selected_indices = select_validation_indices(
        validation_total, args.num_samples, args.selection_seed
    )
    print(f"validation total: {validation_total}")
    print(f"selection seed: {args.selection_seed}")
    print(f"selected sample count: {len(selected_indices)}")
    print(f"selected first 10 indices: {selected_indices[:10]}")
    print(f"selected last 10 indices: {selected_indices[-10:]}")
    indices_csv = Path(args.indices_csv)
    write_indices_csv(indices_csv, selected_indices)
    print(f"selected indices CSV path: {indices_csv}")

    seed_everything(args.noise_base_seed)
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
    with torch.inference_mode():
        for selection_order, val_index in enumerate(selected_indices):
            sample = load_validation_sample(dataset_dir, split_csv, val_index)
            sample_rows = evaluate_one(
                model=model,
                struct_tokenizer=struct_tokenizer,
                fm=fm,
                sample=sample,
                val_index=val_index,
                base_seed=args.noise_base_seed,
                num_flow_steps=args.num_flow_steps,
                device=device,
            )
            require(
                len(sample_rows) == len(ALPHAS),
                f"selection_order={selection_order} did not complete all alphas",
            )
            rows.extend(sample_rows)
            del sample, sample_rows

    validate_output_rows(rows, selected_indices)
    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)
    print_aggregate(rows, output_csv, device)
    print_primary_candidate(rows)
    print("\nINTERNAL_VAL_ALPHA_SWEEP_100_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent held-out replication of the fixed 50-update AM candidate."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm import pilot_adjoint_lambda as pilot
from experiments.latent_residual_fm import replicate_adjoint_lambda_candidate as replication


OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_50step_replication")
PILOT_TRAIN_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_train_indices.csv"
)
PILOT_VALIDATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_validation_indices.csv"
)
PREVIOUS_REPLICATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_replication/"
    "replication_validation_indices.csv"
)
ALPHA_INDICES_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
    "internal_val_alpha_sweep_100_indices.csv"
)
ALPHA_RESULTS_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
    "internal_val_alpha_sweep_100.csv"
)
LAMBDA_REWARD = 1.0
TOTAL_UPDATES = 50
VALIDATION_PER_BIN = 16
EXPECTED_TRAIN_COUNT = 16
EXPECTED_VALIDATION_COUNT = 64
EXPECTED_ALPHA_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded"
    )
    parser.add_argument(
        "--split_csv",
        default="data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv",
    )
    parser.add_argument("--pilot_train_indices", default=str(PILOT_TRAIN_RELATIVE))
    parser.add_argument(
        "--pilot_validation_indices", default=str(PILOT_VALIDATION_RELATIVE)
    )
    parser.add_argument(
        "--previous_replication_indices",
        default=str(PREVIOUS_REPLICATION_RELATIVE),
    )
    parser.add_argument("--alpha_indices", default=str(ALPHA_INDICES_RELATIVE))
    parser.add_argument("--alpha_results", default=str(ALPHA_RESULTS_RELATIVE))
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/latent_residual_fm/runs/"
            "fm_modeA_current_main/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(OUTPUT_RELATIVE))
    return parser.parse_args()


def validation_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "val"]
    integration.require(rows, "internal validation split is empty")
    integration.require(
        len({row["sample_id"] for row in rows}) == len(rows),
        "duplicate IDs in internal validation split",
    )
    return rows


def alpha_selection_ids(
    rows: Sequence[dict[str, str]],
    indices_path: Path,
    results_path: Path,
) -> set[str]:
    with indices_path.open("r", newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    integration.require(
        len(index_rows) == EXPECTED_ALPHA_COUNT,
        f"expected {EXPECTED_ALPHA_COUNT} alpha-selection indices",
    )
    ordered = sorted(index_rows, key=lambda row: int(row["selection_order"]))
    integration.require(
        [int(row["selection_order"]) for row in ordered]
        == list(range(EXPECTED_ALPHA_COUNT)),
        "alpha selection order is incomplete",
    )
    indices = [int(row["val_index"]) for row in ordered]
    integration.require(len(set(indices)) == EXPECTED_ALPHA_COUNT, "duplicate alpha index")
    integration.require(
        all(0 <= index < len(rows) for index in indices), "alpha index out of range"
    )
    expected_pairs = {(index, rows[index]["sample_id"]) for index in indices}

    with results_path.open("r", newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    observed_pairs = {
        (int(row["val_index"]), row["sample_id"]) for row in result_rows
    }
    integration.require(
        expected_pairs == observed_pairs,
        "alpha-selection index/sample mapping differs from result artifact",
    )
    return {sample_id for _, sample_id in expected_pairs}


def select_new_validation(
    rows: Sequence[dict[str, str]], excluded_ids: set[str]
) -> tuple[pilot.Selection, ...]:
    selected: list[pilot.Selection] = []
    for lower, upper in pilot.LENGTH_BINS:
        candidates = sorted(
            (
                (split_index, row)
                for split_index, row in enumerate(rows)
                if lower <= int(row["length"]) <= upper
                and row["sample_id"] not in excluded_ids
            ),
            key=lambda item: (int(item[1]["length"]), item[1]["sample_id"]),
        )
        integration.require(
            len(candidates) >= VALIDATION_PER_BIN,
            f"validation bin {lower}-{upper} has fewer than 16 new samples",
        )
        for split_index, row in candidates[:VALIDATION_PER_BIN]:
            selected.append(
                pilot.Selection(
                    split="val",
                    split_index=split_index,
                    sample_id=row["sample_id"],
                    length=int(row["length"]),
                    length_bin=f"{lower}-{upper}",
                    shard_file=row["shard_file"],
                    index_in_shard=int(row["index_in_shard"]),
                )
            )
    integration.require(
        len(selected) == EXPECTED_VALIDATION_COUNT,
        f"selected {len(selected)} validation samples",
    )
    integration.require(
        len({item.sample_id for item in selected}) == EXPECTED_VALIDATION_COUNT,
        "duplicate new validation sample",
    )
    counts = Counter(item.length_bin for item in selected)
    integration.require(
        counts
        == {
            f"{lower}-{upper}": VALIDATION_PER_BIN
            for lower, upper in pilot.LENGTH_BINS
        },
        "new validation length distribution mismatch",
    )
    return tuple(selected)


def target_deltas(
    base: pilot.ValidationResult, step50: pilot.ValidationResult
) -> list[dict[str, float | int | str]]:
    integration.require(
        len(base.targets) == len(step50.targets) == EXPECTED_VALIDATION_COUNT,
        "target count mismatch",
    )
    rows: list[dict[str, float | int | str]] = []
    for base_target, am_target in zip(base.targets, step50.targets):
        integration.require(
            base_target.sample_id == am_target.sample_id
            and base_target.length == am_target.length,
            "base/AM target order mismatch",
        )
        delta = base_target.aligned_loss - am_target.aligned_loss
        status = (
            "tied"
            if abs(delta) <= pilot.TIE_ATOL
            else "improved" if delta > 0.0 else "worse"
        )
        rows.append(
            {
                "sample_id": base_target.sample_id,
                "length": base_target.length,
                "base_aligned_loss": base_target.aligned_loss,
                "step50_aligned_loss": am_target.aligned_loss,
                "delta_base_minus_step50": delta,
                "base_aligned_sqrt": base_target.aligned_sqrt,
                "step50_aligned_sqrt": am_target.aligned_sqrt,
                "base_latent_mse": base_target.latent_mse,
                "step50_latent_mse": am_target.latent_mse,
                "status": status,
            }
        )
    return rows


def write_per_target(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "sample_id",
        "length",
        "base_aligned_loss",
        "step50_aligned_loss",
        "delta_base_minus_step50",
        "base_aligned_sqrt",
        "step50_aligned_sqrt",
        "base_latent_mse",
        "step50_latent_mse",
        "status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ranked_text(label: str, rows: Sequence[dict[str, object]]) -> list[str]:
    output = [label]
    for row in rows:
        output.append(
            f"{row['sample_id']} L={row['length']} "
            f"delta={float(row['delta_base_minus_step50']):.9g} "
            f"base={float(row['base_aligned_loss']):.9g} "
            f"step50={float(row['step50_aligned_loss']):.9g}"
        )
    return output


def summary_text(
    base: pilot.ValidationResult,
    step50: pilot.ValidationResult,
    largest_improvements: Sequence[dict[str, object]],
    largest_degradations: Sequence[dict[str, object]],
    mean_without_largest: float,
    diagnostics: dict[str, float | int],
    primary_pass: bool,
    stronger_pass: bool,
    finite_pass: bool,
    frozen_pass: bool,
    peak_allocated: float,
    peak_reserved: float,
    runtime_seconds: float,
) -> str:
    lines = [
        "ADJOINT_50STEP_REPLICATION",
        "",
        "NON-PRODUCTION REPLICATION",
        f"train proteins: {EXPECTED_TRAIN_COUNT}",
        f"new validation proteins: {EXPECTED_VALIDATION_COUNT}",
        "overlap pilot validation: 0",
        "overlap previous replication: 0",
        "overlap alpha sweep: 0",
        "",
        f"base mean: {base.mean_loss:.9g}",
        f"base median: {base.median_loss:.9g}",
        f"base latent MSE: {base.mean_latent_mse:.9g}",
        "",
        f"step50 mean: {step50.mean_loss:.9g}",
        f"step50 median: {step50.median_loss:.9g}",
        f"delta: {step50.delta_vs_base:.9g}",
        f"relative mean improvement percent: "
        f"{100.0 * step50.delta_vs_base / base.mean_loss:.9g}",
        f"improved/worse/tied: {step50.improved}/{step50.worse}/{step50.tied}",
        f"step50 latent MSE: {step50.mean_latent_mse:.9g}",
        "",
        "outlier diagnostic",
        f"mean delta all 64: {step50.delta_vs_base:.9g}",
        f"mean delta excluding largest improvement: {mean_without_largest:.9g}",
    ]
    lines.extend(ranked_text("largest 5 improvements", largest_improvements))
    lines.extend(ranked_text("largest 5 degradations", largest_degradations))
    lines.extend(
        [
            "",
            "training diagnostics",
            f"AM loss mean/range: {diagnostics['am_loss_mean']:.9g} "
            f"[{diagnostics['am_loss_min']:.9g}, {diagnostics['am_loss_max']:.9g}]",
            f"terminal L mean/range: {diagnostics['terminal_loss_mean']:.9g} "
            f"[{diagnostics['terminal_loss_min']:.9g}, "
            f"{diagnostics['terminal_loss_max']:.9g}]",
            f"preclip grad mean/range: {diagnostics['preclip_grad_mean']:.9g} "
            f"[{diagnostics['preclip_grad_min']:.9g}, "
            f"{diagnostics['preclip_grad_max']:.9g}]",
            f"clipped updates/fraction: {diagnostics['clipped_updates']}/"
            f"{TOTAL_UPDATES} ({diagnostics['clip_fraction']:.9g})",
            f"parameter drift: {step50.parameter_drift:.9g}",
            f"relative parameter drift: {step50.relative_parameter_drift:.9g}",
            f"NaN/Inf free: {finite_pass}",
            f"frozen audit: {'PASS' if frozen_pass else 'FAIL'}",
            f"peak CUDA allocated/reserved MiB: "
            f"{peak_allocated:.2f}/{peak_reserved:.2f}",
            f"runtime seconds: {runtime_seconds:.3f}",
            "",
            f"primary criterion: {'PASS' if primary_pass else 'FAIL'}",
            f"stronger >=39/64: {'PASS' if stronger_pass else 'FAIL'}",
            "production hyperparameters: NOT SELECTED/FROZEN",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    integration.seed_everything(42)
    device = integration.resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    split_csv = Path(args.split_csv)
    train_path = Path(args.pilot_train_indices)
    pilot_validation_path = Path(args.pilot_validation_indices)
    previous_replication_path = Path(args.previous_replication_indices)
    alpha_indices_path = Path(args.alpha_indices)
    alpha_results_path = Path(args.alpha_results)
    readonly_paths = (
        checkpoint_path,
        train_path,
        pilot_validation_path,
        previous_replication_path,
        alpha_indices_path,
        alpha_results_path,
    )
    total_start = time.perf_counter()
    stage = "startup"
    current_step = 0
    current_sample = "none"
    metadata: dict = {
        "label": "NON_PRODUCTION_REPLICATION",
        "status": "RUNNING",
        "production_hyperparameters_selected": False,
    }
    print("ADJOINT_50STEP_REPLICATION", flush=True)
    try:
        integration.require(device.type == "cuda", "replication requires CUDA")
        integration.require(
            output_dir.resolve().is_relative_to(
                (REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()
            ),
            "output_dir must remain below the 50-step replication directory",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        digests_before = {str(path): pilot.file_digest(path) for path in readonly_paths}

        stage = "load exclusion populations"
        train = replication.read_selections(train_path, "train")
        pilot_validation = replication.read_selections(pilot_validation_path, "val")
        previous_replication = replication.read_selections(
            previous_replication_path, "val"
        )
        integration.require(len(train) == EXPECTED_TRAIN_COUNT, "train count mismatch")
        integration.require(len(pilot_validation) == 16, "pilot validation count mismatch")
        integration.require(
            len(previous_replication) == 32,
            "previous replication validation count mismatch",
        )
        rows = validation_rows(split_csv)
        alpha_ids = alpha_selection_ids(rows, alpha_indices_path, alpha_results_path)
        integration.require(len(alpha_ids) == EXPECTED_ALPHA_COUNT, "alpha ID count mismatch")

        train_ids = {item.sample_id for item in train}
        pilot_validation_ids = {item.sample_id for item in pilot_validation}
        previous_replication_ids = {item.sample_id for item in previous_replication}
        excluded = (
            train_ids | pilot_validation_ids | previous_replication_ids | alpha_ids
        )
        selected = select_new_validation(rows, excluded)
        selected_ids = {item.sample_id for item in selected}
        overlaps = {
            "train": len(selected_ids & train_ids),
            "pilot_validation": len(selected_ids & pilot_validation_ids),
            "previous_replication": len(selected_ids & previous_replication_ids),
            "alpha_sweep": len(selected_ids & alpha_ids),
        }
        integration.require(all(value == 0 for value in overlaps.values()), "overlap audit")
        pilot.write_indices(output_dir / "validation_indices_64.csv", selected)

        print(f"train proteins: {len(train)}", flush=True)
        print(f"new validation proteins: {len(selected)}", flush=True)
        print(f"overlap pilot validation: {overlaps['pilot_validation']}", flush=True)
        print(
            f"overlap previous replication: {overlaps['previous_replication']}",
            flush=True,
        )
        print(f"overlap alpha sweep: {overlaps['alpha_sweep']}", flush=True)
        print("new validation IDs:", flush=True)
        for item in selected:
            print(
                f"  {item.sample_id} length={item.length} bin={item.length_bin}",
                flush=True,
            )

        metadata.update(
            {
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": digests_before[str(checkpoint_path)],
                "lambda_reward": LAMBDA_REWARD,
                "learning_rate": pilot.LEARNING_RATE_DIAGNOSTIC,
                "optimizer": "AdamW",
                "weight_decay": pilot.WEIGHT_DECAY,
                "global_gradient_clip": pilot.GRADIENT_CLIP_NORM,
                "total_updates": TOTAL_UPDATES,
                "physical_batch": 1,
                "gradient_accumulation": 1,
                "am_timestep_policy": "paper_H2_20",
                "per_timestep_loss_clipping": False,
                "alpha_refine": pilot.ALPHA_REFINE,
                "checkpoint_selection_performed": False,
                "evaluated_steps": [0, 50],
                "excluded_alpha_selection_count": len(alpha_ids),
                "overlaps": overlaps,
                "train_samples": [asdict(item) for item in train],
                "validation_samples": [asdict(item) for item in selected],
                "readonly_input_sha256": digests_before,
            }
        )
        pilot.write_json(output_dir / "metadata.json", metadata)

        stage = "load frozen DPLM and cache conditions"
        extractor = integration.DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        pilot._TOKENIZER_HOLDER.clear()
        pilot._TOKENIZER_HOLDER.append(tokenizer)
        dplm_versions = pilot.model_versions(dplm)
        tokenizer_versions = pilot.model_versions(tokenizer)
        dataset_dir = Path(args.dataset_dir)
        train_bundles = tuple(
            pilot.load_bundle(dataset_dir, item, extractor, tokenizer, device)
            for item in train
        )
        validation_bundles = tuple(
            pilot.load_bundle(dataset_dir, item, extractor, tokenizer, device)
            for item in selected
        )

        stage = "base validation"
        reference_base, reference_step, _, _ = integration.load_fm(
            checkpoint_path, device
        )
        integration.require(reference_step == 100_000, "base checkpoint step mismatch")
        reference_base.requires_grad_(False).eval()
        base_result, _ = pilot.evaluate(
            reference_base,
            validation_bundles,
            None,
            None,
            0,
            None,
            device,
        )
        pilot.print_validation("base", base_result)
        del reference_base
        torch.cuda.empty_cache()

        stage = "fresh lambda=1 initialization"
        base, base_step, _, _ = integration.load_fm(checkpoint_path, device)
        fine, fine_step, _, _ = integration.load_fm(checkpoint_path, device)
        integration.require(base_step == fine_step == 100_000, "checkpoint step mismatch")
        base.requires_grad_(False).eval()
        fine.requires_grad_(True).eval()
        integration.require(
            pilot.model_pair_initialization(base, fine, train_bundles[0]),
            "fresh base/fine copies differ",
        )
        base_versions = pilot.model_versions(base)
        optimizer = torch.optim.AdamW(
            fine.parameters(),
            lr=pilot.LEARNING_RATE_DIAGNOSTIC,
            weight_decay=pilot.WEIGHT_DECAY,
        )
        subset_rng = random.Random(pilot.SUBSET_SEED)
        updates: list[pilot.UpdateMetric] = []
        for step in range(1, TOTAL_UPDATES + 1):
            current_step = step
            bundle = train_bundles[(step - 1) % len(train_bundles)]
            current_sample = bundle.selection.sample_id
            stage = f"AM update {step}"
            metric = pilot.update_once(
                step,
                bundle,
                LAMBDA_REWARD,
                base,
                fine,
                tokenizer,
                optimizer,
                subset_rng,
                device,
            )
            updates.append(metric)
            if step == 1 or step % 10 == 0:
                print(
                    f"update {step}: sample={metric.sample_id} "
                    f"AM={metric.am_loss:.7g} terminal={metric.terminal_loss:.7g} "
                    f"grad={metric.preclip_grad_norm:.7g}->{metric.postclip_grad_norm:.7g} "
                    f"clipped={metric.clipped} seconds={metric.seconds:.3f}",
                    flush=True,
                )

        stage = "predeclared step50 validation"
        step50, _ = pilot.evaluate(
            fine,
            validation_bundles,
            base_result,
            LAMBDA_REWARD,
            50,
            base,
            device,
        )
        pilot.print_validation("step50", step50)
        per_target = target_deltas(base_result, step50)
        descending = sorted(
            per_target,
            key=lambda row: float(row["delta_base_minus_step50"]),
            reverse=True,
        )
        largest_improvements = descending[:5]
        largest_degradations = list(reversed(descending[-5:]))
        largest_delta = float(largest_improvements[0]["delta_base_minus_step50"])
        mean_without_largest = (
            sum(float(row["delta_base_minus_step50"]) for row in per_target)
            - largest_delta
        ) / (len(per_target) - 1)
        write_per_target(output_dir / "per_target_results.csv", per_target)

        diagnostics = pilot.update_summary(updates)
        finite_pass = bool(
            base_result.finite
            and step50.finite
            and math.isfinite(mean_without_largest)
            and all(
                all(
                    math.isfinite(value)
                    for value in (
                        metric.am_loss,
                        metric.terminal_loss,
                        metric.terminal_adjoint_norm,
                        metric.preclip_grad_norm,
                        metric.postclip_grad_norm,
                    )
                )
                for metric in updates
            )
            and all(bool(torch.isfinite(parameter).all()) for parameter in fine.parameters())
        )
        primary_pass = bool(
            step50.mean_loss < base_result.mean_loss
            and step50.improved > step50.worse
            and step50.median_loss <= base_result.median_loss
            and finite_pass
        )
        stronger_pass = step50.improved >= 39
        digests_after = {str(path): pilot.file_digest(path) for path in readonly_paths}
        frozen_pass = bool(
            digests_before == digests_after
            and pilot.versions_unchanged(base, base_versions)
            and pilot.versions_unchanged(dplm, dplm_versions)
            and pilot.versions_unchanged(tokenizer, tokenizer_versions)
            and integration.no_parameter_grads(base)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
        )
        integration.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        runtime_seconds = time.perf_counter() - total_start
        overall = primary_pass and finite_pass and frozen_pass

        text = summary_text(
            base_result,
            step50,
            largest_improvements,
            largest_degradations,
            mean_without_largest,
            diagnostics,
            primary_pass,
            stronger_pass,
            finite_pass,
            frozen_pass,
            peak_allocated,
            peak_reserved,
            runtime_seconds,
        )
        (output_dir / "summary.txt").write_text(text, encoding="utf-8")
        metadata.update(
            {
                "status": "PASS" if overall else "FAIL",
                "base": replication.validation_payload(
                    base_result, base_result.mean_loss
                ),
                "step50": replication.validation_payload(
                    step50, base_result.mean_loss
                ),
                "mean_delta_all_64": step50.delta_vs_base,
                "largest_single_improvement": largest_improvements[0],
                "largest_single_degradation": largest_degradations[0],
                "mean_delta_excluding_largest_improvement": mean_without_largest,
                "largest_5_improvements": largest_improvements,
                "largest_5_degradations": largest_degradations,
                "training_diagnostics": diagnostics,
                "primary_criterion": primary_pass,
                "stronger_39_of_64_criterion": stronger_pass,
                "finite_audit": finite_pass,
                "frozen_audit": frozen_pass,
                "readonly_inputs_unchanged": digests_before == digests_after,
                "parameter_drift": step50.parameter_drift,
                "relative_parameter_drift": step50.relative_parameter_drift,
                "peak_cuda_allocated_mib": peak_allocated,
                "peak_cuda_reserved_mib": peak_reserved,
                "runtime_seconds": runtime_seconds,
                "production_hyperparameters_selected": False,
            }
        )
        pilot.write_json(output_dir / "metadata.json", metadata)

        print("\nADJOINT_50STEP_REPLICATION", flush=True)
        print(f"train proteins: {len(train)}", flush=True)
        print(f"new validation proteins: {len(selected)}", flush=True)
        print(f"overlap pilot validation: {overlaps['pilot_validation']}", flush=True)
        print(
            f"overlap previous replication: {overlaps['previous_replication']}",
            flush=True,
        )
        print(f"overlap alpha sweep: {overlaps['alpha_sweep']}", flush=True)
        print(f"base mean: {base_result.mean_loss:.9g}", flush=True)
        print(f"base median: {base_result.median_loss:.9g}", flush=True)
        print(f"step50 mean: {step50.mean_loss:.9g}", flush=True)
        print(f"step50 median: {step50.median_loss:.9g}", flush=True)
        print(f"delta: {step50.delta_vs_base:.9g}", flush=True)
        print(
            f"relative mean improvement: "
            f"{100.0 * step50.delta_vs_base / base_result.mean_loss:.9g}%",
            flush=True,
        )
        print(
            f"improved/worse/tied: {step50.improved}/{step50.worse}/{step50.tied}",
            flush=True,
        )
        print("outlier diagnostic:", flush=True)
        print(
            f"  largest improvement: {largest_improvements[0]['sample_id']} "
            f"delta={float(largest_improvements[0]['delta_base_minus_step50']):.9g}",
            flush=True,
        )
        print(
            f"  largest degradation: {largest_degradations[0]['sample_id']} "
            f"delta={float(largest_degradations[0]['delta_base_minus_step50']):.9g}",
            flush=True,
        )
        print(
            f"  mean delta excluding largest improvement: {mean_without_largest:.9g}",
            flush=True,
        )
        print(f"training clip fraction: {diagnostics['clip_fraction']:.9g}", flush=True)
        print(
            f"training relative drift: {step50.relative_parameter_drift:.9g}",
            flush=True,
        )
        print(f"primary criterion: {'PASS' if primary_pass else 'FAIL'}", flush=True)
        print(
            f"stronger >=39/64: {'PASS' if stronger_pass else 'FAIL'}",
            flush=True,
        )
        print(f"NaN/Inf audit: {'PASS' if finite_pass else 'FAIL'}", flush=True)
        print(f"frozen audit: {'PASS' if frozen_pass else 'FAIL'}", flush=True)
        print(
            f"peak CUDA MiB allocated/reserved: "
            f"{peak_allocated:.2f}/{peak_reserved:.2f}",
            flush=True,
        )
        print(f"runtime seconds: {runtime_seconds:.3f}", flush=True)
        print("production hyperparameters: NOT SELECTED/FROZEN", flush=True)
        marker = (
            "ADJOINT_50STEP_REPLICATION_PASS"
            if overall
            else "ADJOINT_50STEP_REPLICATION_FAIL"
        )
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if overall else 1
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        metadata.update(
            {
                "status": "FAIL",
                "failure_stage": stage,
                "failure_step": current_step,
                "failure_sample": current_sample,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "production_hyperparameters_selected": False,
            }
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            pilot.write_json(output_dir / "metadata.json", metadata)
        except Exception:
            pass
        print(f"failure point: {stage}", flush=True)
        print(f"failure step/sample: {current_step}/{current_sample}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("production hyperparameters: NOT SELECTED/FROZEN", flush=True)
        print("overall: ADJOINT_50STEP_REPLICATION_FAIL", flush=True)
        print("ADJOINT_50STEP_REPLICATION_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

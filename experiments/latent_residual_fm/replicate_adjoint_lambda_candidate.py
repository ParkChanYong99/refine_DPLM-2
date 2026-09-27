#!/usr/bin/env python3
"""Replicate the predeclared lambda=1, step=200 AM pilot candidate.

The training population and stochastic schedule match the lambda pilot.  The
only new population is a disjoint, deterministic set of 32 internal-validation
proteins.  Step 50 and 100 are diagnostics; only step 200 determines the
replication result.
"""

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


OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_lambda_replication")
PILOT_TRAIN_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_train_indices.csv"
)
PILOT_VALIDATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_validation_indices.csv"
)
LAMBDA_REWARD = 1.0
VALIDATION_PER_BIN = 8
EVALUATION_STEPS = (50, 100, 200)
EXPECTED_TRAIN_COUNT = 16
EXPECTED_VALIDATION_COUNT = 32


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


def read_selections(path: Path, expected_split: str) -> tuple[pilot.Selection, ...]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selections = tuple(
        pilot.Selection(
            split=row["split"],
            split_index=int(row["split_index"]),
            sample_id=row["sample_id"],
            length=int(row["length"]),
            length_bin=row["length_bin"],
            shard_file=row["shard_file"],
            index_in_shard=int(row["index_in_shard"]),
        )
        for row in rows
    )
    integration.require(
        all(item.split == expected_split for item in selections),
        f"unexpected split in {path}",
    )
    integration.require(
        len({item.sample_id for item in selections}) == len(selections),
        f"duplicate IDs in {path}",
    )
    return selections


def select_replication_validation(
    split_csv: Path,
    excluded_ids: set[str],
) -> tuple[pilot.Selection, ...]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        validation_rows = [
            row for row in csv.DictReader(handle) if row.get("split") == "val"
        ]
    integration.require(validation_rows, "internal validation split is empty")
    selected: list[pilot.Selection] = []
    for lower, upper in pilot.LENGTH_BINS:
        candidates = sorted(
            (
                (split_index, row)
                for split_index, row in enumerate(validation_rows)
                if lower <= int(row["length"]) <= upper
                and row["sample_id"] not in excluded_ids
            ),
            key=lambda item: (int(item[1]["length"]), item[1]["sample_id"]),
        )
        integration.require(
            len(candidates) >= VALIDATION_PER_BIN,
            f"validation bin {lower}-{upper} has fewer than eight new samples",
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
        f"selected {len(selected)} replication validation samples",
    )
    integration.require(
        len({item.sample_id for item in selected}) == EXPECTED_VALIDATION_COUNT,
        "duplicate replication validation sample",
    )
    return tuple(selected)


def validate_distribution(
    selections: Sequence[pilot.Selection], expected_per_bin: int
) -> bool:
    counts = Counter(item.length_bin for item in selections)
    expected = {
        f"{lower}-{upper}": expected_per_bin for lower, upper in pilot.LENGTH_BINS
    }
    return counts == expected


def write_results_csv(
    path: Path,
    base: pilot.ValidationResult,
    validations: Sequence[pilot.ValidationResult],
) -> None:
    fields = (
        "record",
        "step",
        "sample_id",
        "length",
        "aligned_loss",
        "aligned_sqrt",
        "latent_mse",
        "mean_aligned_loss",
        "median_aligned_loss",
        "mean_latent_mse",
        "delta_vs_base",
        "relative_mean_improvement_percent",
        "improved",
        "worse",
        "tied",
        "parameter_drift",
        "relative_parameter_drift",
        "finite",
        "predeclared_primary_step",
    )
    all_results = (base, *validations)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in all_results:
            relative = (
                0.0
                if result.step == 0
                else 100.0 * result.delta_vs_base / base.mean_loss
            )
            writer.writerow(
                {
                    "record": "aggregate",
                    "step": result.step,
                    "mean_aligned_loss": result.mean_loss,
                    "median_aligned_loss": result.median_loss,
                    "mean_latent_mse": result.mean_latent_mse,
                    "delta_vs_base": result.delta_vs_base,
                    "relative_mean_improvement_percent": relative,
                    "improved": result.improved,
                    "worse": result.worse,
                    "tied": result.tied,
                    "parameter_drift": result.parameter_drift,
                    "relative_parameter_drift": result.relative_parameter_drift,
                    "finite": result.finite,
                    "predeclared_primary_step": result.step == 200,
                }
            )
            for target in result.targets:
                writer.writerow(
                    {
                        "record": "target",
                        "step": result.step,
                        "sample_id": target.sample_id,
                        "length": target.length,
                        "aligned_loss": target.aligned_loss,
                        "aligned_sqrt": target.aligned_sqrt,
                        "latent_mse": target.latent_mse,
                        "finite": all(
                            math.isfinite(value)
                            for value in (
                                target.aligned_loss,
                                target.aligned_sqrt,
                                target.latent_mse,
                            )
                        ),
                        "predeclared_primary_step": result.step == 200,
                    }
                )


def validation_payload(result: pilot.ValidationResult, base_mean: float) -> dict:
    return {
        "step": result.step,
        "mean_aligned_loss": result.mean_loss,
        "median_aligned_loss": result.median_loss,
        "mean_latent_mse": result.mean_latent_mse,
        "delta_vs_base": result.delta_vs_base,
        "relative_mean_improvement_percent": (
            0.0 if result.step == 0 else 100.0 * result.delta_vs_base / base_mean
        ),
        "improved": result.improved,
        "worse": result.worse,
        "tied": result.tied,
        "parameter_drift": result.parameter_drift,
        "relative_parameter_drift": result.relative_parameter_drift,
        "finite": result.finite,
        "seconds": result.seconds,
    }


def summary_text(
    base: pilot.ValidationResult,
    validations: Sequence[pilot.ValidationResult],
    diagnostics: dict[str, float | int],
    primary_pass: bool,
    stronger_pass: bool,
    frozen_pass: bool,
    peak_allocated: float,
    peak_reserved: float,
    runtime_seconds: float,
) -> str:
    step200 = next(result for result in validations if result.step == 200)
    lines = [
        "ADJOINT_LAMBDA_REPLICATION",
        "",
        "NON-PRODUCTION REPLICATION",
        f"train samples: {EXPECTED_TRAIN_COUNT}",
        f"new validation samples: {EXPECTED_VALIDATION_COUNT}",
        "overlap with train: 0",
        "overlap with previous validation: 0",
        "",
        f"base mean: {base.mean_loss:.9g}",
        f"base median: {base.median_loss:.9g}",
        f"base latent MSE: {base.mean_latent_mse:.9g}",
    ]
    for result in validations:
        lines.extend(
            [
                "",
                f"step {result.step}",
                f"mean: {result.mean_loss:.9g}",
                f"median: {result.median_loss:.9g}",
                f"delta: {result.delta_vs_base:.9g}",
                f"relative improvement percent: "
                f"{100.0 * result.delta_vs_base / base.mean_loss:.9g}",
                f"improved/worse/tied: "
                f"{result.improved}/{result.worse}/{result.tied}",
                f"latent MSE: {result.mean_latent_mse:.9g}",
            ]
        )
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
            f"postclip grad mean/range: {diagnostics['postclip_grad_mean']:.9g} "
            f"[{diagnostics['postclip_grad_min']:.9g}, "
            f"{diagnostics['postclip_grad_max']:.9g}]",
            f"clipped updates/fraction: {diagnostics['clipped_updates']}/200 "
            f"({diagnostics['clip_fraction']:.9g})",
            f"parameter drift: {step200.parameter_drift:.9g}",
            f"relative parameter drift: {step200.relative_parameter_drift:.9g}",
            f"NaN/Inf free: {step200.finite}",
            f"frozen audit: {'PASS' if frozen_pass else 'FAIL'}",
            f"peak CUDA allocated/reserved MiB: "
            f"{peak_allocated:.2f}/{peak_reserved:.2f}",
            f"runtime seconds: {runtime_seconds:.3f}",
            "",
            f"replication primary criterion: {'PASS' if primary_pass else 'FAIL'}",
            f"stronger 20/32 improvement criterion: "
            f"{'PASS' if stronger_pass else 'FAIL'}",
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
    train_indices_path = Path(args.pilot_train_indices)
    pilot_validation_path = Path(args.pilot_validation_indices)
    total_start = time.perf_counter()
    stage = "startup"
    current_step = 0
    current_sample = "none"
    metadata: dict = {
        "label": "NON_PRODUCTION_REPLICATION",
        "status": "RUNNING",
        "production_hyperparameters_frozen": False,
    }
    print("ADJOINT_LAMBDA_REPLICATION", flush=True)
    try:
        integration.require(device.type == "cuda", "replication requires CUDA")
        integration.require(
            output_dir.resolve().is_relative_to(
                (REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()
            ),
            "output_dir must remain below the replication directory",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        checkpoint_digest_before = pilot.file_digest(checkpoint_path)
        train_indices_digest_before = pilot.file_digest(train_indices_path)
        pilot_validation_digest_before = pilot.file_digest(pilot_validation_path)

        stage = "load fixed pilot train indices"
        train_selections = read_selections(train_indices_path, "train")
        old_validation = read_selections(pilot_validation_path, "val")
        integration.require(
            len(train_selections) == EXPECTED_TRAIN_COUNT,
            "pilot train set must contain 16 samples",
        )
        integration.require(
            len(old_validation) == 16,
            "pilot validation set must contain 16 samples",
        )
        integration.require(validate_distribution(train_selections, 4), "bad train bins")
        integration.require(validate_distribution(old_validation, 4), "bad old validation bins")
        train_ids = {item.sample_id for item in train_selections}
        old_validation_ids = {item.sample_id for item in old_validation}

        stage = "select disjoint replication validation"
        replication_validation = select_replication_validation(
            Path(args.split_csv), train_ids | old_validation_ids
        )
        new_validation_ids = {item.sample_id for item in replication_validation}
        overlap_train = train_ids & new_validation_ids
        overlap_previous = old_validation_ids & new_validation_ids
        integration.require(not overlap_train, "replication validation overlaps train")
        integration.require(
            not overlap_previous,
            "replication validation overlaps lambda-pilot validation",
        )
        integration.require(
            validate_distribution(replication_validation, VALIDATION_PER_BIN),
            "bad replication validation bins",
        )
        pilot.write_indices(
            output_dir / "replication_validation_indices.csv", replication_validation
        )
        print(f"train samples: {len(train_selections)}", flush=True)
        print(f"new validation samples: {len(replication_validation)}", flush=True)
        print(f"overlap with train: {len(overlap_train)}", flush=True)
        print(f"overlap with previous validation: {len(overlap_previous)}", flush=True)
        print("new validation IDs:", flush=True)
        for item in replication_validation:
            print(
                f"  {item.sample_id} length={item.length} bin={item.length_bin}",
                flush=True,
            )

        metadata.update(
            {
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_sha256": checkpoint_digest_before,
                "pilot_train_indices": str(train_indices_path),
                "pilot_train_indices_sha256": train_indices_digest_before,
                "pilot_validation_indices": str(pilot_validation_path),
                "pilot_validation_indices_sha256": pilot_validation_digest_before,
                "lambda_reward": LAMBDA_REWARD,
                "learning_rate": pilot.LEARNING_RATE_DIAGNOSTIC,
                "optimizer": "AdamW",
                "weight_decay": pilot.WEIGHT_DECAY,
                "global_gradient_clip": pilot.GRADIENT_CLIP_NORM,
                "updates": pilot.NUM_UPDATES,
                "physical_batch": 1,
                "gradient_accumulation": 1,
                "am_timestep_policy": "paper_H2_20",
                "per_timestep_am_loss_clipping": False,
                "alpha_refine": pilot.ALPHA_REFINE,
                "trajectory_seed_namespace": pilot.TRAJECTORY_SEED_NAMESPACE,
                "inference_seed_namespace": pilot.INFERENCE_SEED_NAMESPACE,
                "train_samples": [asdict(item) for item in train_selections],
                "replication_validation_samples": [
                    asdict(item) for item in replication_validation
                ],
                "overlap_with_train": len(overlap_train),
                "overlap_with_previous_validation": len(overlap_previous),
                "predeclared_primary_step": 200,
                "checkpoint_selection_performed": False,
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
            for item in train_selections
        )
        validation_bundles = tuple(
            pilot.load_bundle(dataset_dir, item, extractor, tokenizer, device)
            for item in replication_validation
        )

        stage = "base replication validation"
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
        validations: list[pilot.ValidationResult] = []

        for step in range(1, pilot.NUM_UPDATES + 1):
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
            if step in EVALUATION_STEPS:
                stage = f"replication validation step {step}"
                validation, _ = pilot.evaluate(
                    fine,
                    validation_bundles,
                    base_result,
                    LAMBDA_REWARD,
                    step,
                    base,
                    device,
                )
                validations.append(validation)
                pilot.print_validation(f"step {step}", validation)

        integration.require(
            tuple(result.step for result in validations) == EVALUATION_STEPS,
            "validation grid incomplete",
        )
        step200 = validations[-1]
        diagnostics = pilot.update_summary(updates)
        all_finite = bool(
            base_result.finite
            and all(result.finite for result in validations)
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
            step200.mean_loss < base_result.mean_loss
            and step200.improved > step200.worse
            and all_finite
        )
        stronger_pass = step200.improved >= 20

        checkpoint_digest_after = pilot.file_digest(checkpoint_path)
        train_indices_digest_after = pilot.file_digest(train_indices_path)
        pilot_validation_digest_after = pilot.file_digest(pilot_validation_path)
        frozen_pass = bool(
            checkpoint_digest_before == checkpoint_digest_after
            and train_indices_digest_before == train_indices_digest_after
            and pilot_validation_digest_before == pilot_validation_digest_after
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
        overall = primary_pass and frozen_pass and all_finite

        write_results_csv(
            output_dir / "replication_results.csv", base_result, validations
        )
        text = summary_text(
            base_result,
            validations,
            diagnostics,
            primary_pass,
            stronger_pass,
            frozen_pass,
            peak_allocated,
            peak_reserved,
            runtime_seconds,
        )
        (output_dir / "replication_summary.txt").write_text(text, encoding="utf-8")
        metadata.update(
            {
                "status": "PASS" if overall else "FAIL",
                "base": validation_payload(base_result, base_result.mean_loss),
                "validations": [
                    validation_payload(result, base_result.mean_loss)
                    for result in validations
                ],
                "training_diagnostics": diagnostics,
                "primary_replication_delta": step200.delta_vs_base,
                "relative_mean_improvement_percent": (
                    100.0 * step200.delta_vs_base / base_result.mean_loss
                ),
                "replication_primary_criterion": primary_pass,
                "stronger_20_of_32_criterion": stronger_pass,
                "finite_audit": all_finite,
                "frozen_audit": frozen_pass,
                "source_checkpoint_unchanged": (
                    checkpoint_digest_before == checkpoint_digest_after
                ),
                "pilot_inputs_unchanged": bool(
                    train_indices_digest_before == train_indices_digest_after
                    and pilot_validation_digest_before == pilot_validation_digest_after
                ),
                "peak_cuda_allocated_mib": peak_allocated,
                "peak_cuda_reserved_mib": peak_reserved,
                "runtime_seconds": runtime_seconds,
                "production_hyperparameters_selected": False,
            }
        )
        pilot.write_json(output_dir / "metadata.json", metadata)

        print("\nADJOINT_LAMBDA_REPLICATION", flush=True)
        print(f"train samples: {len(train_selections)}", flush=True)
        print(f"new validation samples: {len(replication_validation)}", flush=True)
        print(f"overlap with previous validation: {len(overlap_previous)}", flush=True)
        print(f"base mean: {base_result.mean_loss:.9g}", flush=True)
        print(f"base median: {base_result.median_loss:.9g}", flush=True)
        for result in validations:
            print(f"step {result.step}:", flush=True)
            print(f"  mean: {result.mean_loss:.9g}", flush=True)
            print(f"  median: {result.median_loss:.9g}", flush=True)
            print(f"  delta: {result.delta_vs_base:.9g}", flush=True)
            print(
                f"  relative improvement: "
                f"{100.0 * result.delta_vs_base / base_result.mean_loss:.9g}%",
                flush=True,
            )
            print(
                f"  improved/worse/tied: "
                f"{result.improved}/{result.worse}/{result.tied}",
                flush=True,
            )
        print(f"training clip fraction: {diagnostics['clip_fraction']:.9g}", flush=True)
        print(
            f"training relative parameter drift: "
            f"{step200.relative_parameter_drift:.9g}",
            flush=True,
        )
        print(
            f"replication primary criterion: {'PASS' if primary_pass else 'FAIL'}",
            flush=True,
        )
        print(
            f"stronger 20/32 improvement criterion: "
            f"{'PASS' if stronger_pass else 'FAIL'}",
            flush=True,
        )
        print(f"NaN/Inf audit: {'PASS' if all_finite else 'FAIL'}", flush=True)
        print(f"frozen audit: {'PASS' if frozen_pass else 'FAIL'}", flush=True)
        print(
            f"peak CUDA MiB allocated/reserved: "
            f"{peak_allocated:.2f}/{peak_reserved:.2f}",
            flush=True,
        )
        print(f"runtime seconds: {runtime_seconds:.3f}", flush=True)
        print("production hyperparameters: NOT SELECTED/FROZEN", flush=True)
        marker = (
            "ADJOINT_LAMBDA_REPLICATION_PASS"
            if overall
            else "ADJOINT_LAMBDA_REPLICATION_FAIL"
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
        print("overall: ADJOINT_LAMBDA_REPLICATION_FAIL", flush=True)
        print("ADJOINT_LAMBDA_REPLICATION_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

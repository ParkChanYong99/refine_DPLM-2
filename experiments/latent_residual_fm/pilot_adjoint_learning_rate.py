#!/usr/bin/env python3
"""Pilot lower learning rates for REPEAT_16 Adjoint Matching on AM dev 256."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.latent_residual_fm import compare_adjoint_training_diversity as diversity
from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm import pilot_adjoint_lambda as pilot
from experiments.latent_residual_fm.flow_matching import euler_sample


OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_lr_pilot")
PROTOCOL_ROOT = Path("experiments/latent_residual_fm/runs/am_validation_protocol")
DEVELOPMENT_RELATIVE = PROTOCOL_ROOT / "am_development_256.csv"
LOCKED_RELATIVE = PROTOCOL_ROOT / "am_locked_confirmation_256.csv"
PROTOCOL_METADATA_RELATIVE = PROTOCOL_ROOT / "metadata.json"
PROTOCOL_FREEZE_RELATIVE = PROTOCOL_ROOT / "VALIDATION_PROTOCOL_FREEZE.txt"
PILOT_TRAIN_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_train_indices.csv"
)
HISTORICAL_ROOT = Path(
    "experiments/latent_residual_fm/runs/am_training_diversity"
)
HISTORICAL_RESULTS_RELATIVE = HISTORICAL_ROOT / "per_target_development_results.csv"
HISTORICAL_METADATA_RELATIVE = HISTORICAL_ROOT / "metadata.json"
HISTORICAL_SUMMARY_RELATIVE = HISTORICAL_ROOT / "diversity_summary.txt"
HISTORICAL_SOURCE_RELATIVE = Path(
    "experiments/latent_residual_fm/compare_adjoint_training_diversity.py"
)

LR_5E7 = "lr_5e_7"
LR_1E6 = "lr_1e_6"
LR_2E6 = "lr_2e_6_historical"
NEW_ARMS = ((LR_5E7, 5e-7), (LR_1E6, 1e-6))
LR_LABELS = (LR_5E7, LR_1E6, LR_2E6)
LR_DISPLAY = {
    LR_5E7: "5e-7",
    LR_1E6: "1e-6",
    LR_2E6: "2e-6 historical",
}
TOTAL_UPDATES = 200
EXPECTED_DEVELOPMENT = 256
EXPECTED_LOCKED = 256
EXPECTED_REPEAT = 16
LAMBDA_REWARD = 1.0
LENGTH_BINS = pilot.LENGTH_BINS


@dataclass
class LRTrainingResult:
    label: str
    learning_rate: float
    updates: list[pilot.UpdateMetric]
    parameter_drift: float
    relative_parameter_drift: float
    finite: bool
    frozen_ok: bool
    runtime_seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded"
    )
    parser.add_argument("--development_csv", default=str(DEVELOPMENT_RELATIVE))
    parser.add_argument("--locked_csv", default=str(LOCKED_RELATIVE))
    parser.add_argument(
        "--protocol_metadata", default=str(PROTOCOL_METADATA_RELATIVE)
    )
    parser.add_argument("--protocol_freeze", default=str(PROTOCOL_FREEZE_RELATIVE))
    parser.add_argument("--pilot_train", default=str(PILOT_TRAIN_RELATIVE))
    parser.add_argument(
        "--historical_results", default=str(HISTORICAL_RESULTS_RELATIVE)
    )
    parser.add_argument(
        "--historical_metadata", default=str(HISTORICAL_METADATA_RELATIVE)
    )
    parser.add_argument(
        "--historical_summary", default=str(HISTORICAL_SUMMARY_RELATIVE)
    )
    parser.add_argument(
        "--historical_source", default=str(HISTORICAL_SOURCE_RELATIVE)
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def validate_historical_reference(
    results_path: Path,
    metadata_path: Path,
    summary_path: Path,
    source_path: Path,
    development_ids: set[str],
    locked_ids: set[str],
    development_sha256: str,
    locked_sha256: str,
) -> tuple[dict[str, dict[str, str]], dict[str, object], list[str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["status"] == "PASS", "historical experiment did not PASS")
    require(metadata["finite_audit"] is True, "historical finite audit failed")
    require(metadata["frozen_audit"] is True, "historical frozen audit failed")
    require(
        metadata["readonly_inputs_unchanged"] is True,
        "historical read-only audit failed",
    )
    require(
        metadata["development_sha256"] == development_sha256,
        "historical development SHA differs",
    )
    require(
        metadata["locked_sha256"] == locked_sha256,
        "historical locked SHA differs",
    )
    require(
        metadata["locked_confirmation_evaluated"] is False
        and metadata["locked_dataset_items_loaded"] is False,
        "historical locked protection failed",
    )
    rows = read_csv(results_path)
    require(len(rows) == EXPECTED_DEVELOPMENT, "historical development N differs")
    by_id = {row["protein_id"]: row for row in rows}
    require(len(by_id) == EXPECTED_DEVELOPMENT, "historical duplicate IDs")
    require(set(by_id) == development_ids, "historical development IDs differ")
    require(not (set(by_id) & locked_ids), "historical row overlaps locked IDs")
    require(
        "ADJOINT_TRAINING_DIVERSITY_PASS"
        in summary_path.read_text(encoding="utf-8"),
        "historical PASS marker absent",
    )

    expected: dict[str, object] = {
        "trajectory_seed_policy": "10000 + optimizer_step",
        "timestep_subset_seed_policy": "20000 + optimizer_step",
        "lambda_reward": LAMBDA_REWARD,
        "learning_rate": 2e-6,
        "optimizer": "AdamW",
        "weight_decay": pilot.WEIGHT_DECAY,
        "global_gradient_clip": pilot.GRADIENT_CLIP_NORM,
        "total_optimizer_updates_per_arm": TOTAL_UPDATES,
        "physical_batch": 1,
        "gradient_accumulation": 1,
        "K": pilot.K,
        "h": pilot.H,
        "am_timestep_policy": "paper_H2_20",
        "per_timestep_am_loss_clipping": False,
        "alpha_refine": pilot.ALPHA_REFINE,
        "local_first_step": "DETERMINISTIC_WARM_START",
    }
    differences = [
        f"{key}: historical={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    source_text = source_path.read_text(encoding="utf-8")
    schedule_evidence = (
        "repeat[index % len(repeat)] for index in range(TOTAL_UPDATES)"
        in source_text
    )
    if not schedule_evidence:
        differences.append("historical source cyclic REPEAT_16 schedule not verified")
    require(
        metadata["repeat16_training"]["finite"] is True
        and metadata["repeat16_training"]["frozen_ok"] is True,
        "historical REPEAT_16 training audit failed",
    )
    return by_id, metadata, differences


def stage_repeat_bundles(
    selections: Sequence[pilot.Selection],
    dataset_dir: Path,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> dict[str, pilot.Bundle]:
    require(len(selections) == EXPECTED_REPEAT, "REPEAT_16 selection size differs")
    staged: dict[str, pilot.Bundle] = {}
    for ordinal, selection in enumerate(selections, start=1):
        bundle = pilot.load_bundle(
            dataset_dir, selection, extractor, tokenizer, device
        )
        staged[selection.sample_id] = diversity.move_bundle(bundle, "cpu")
        del bundle
        torch.cuda.empty_cache()
        print(
            f"TRAIN bundle staging {ordinal}/{len(selections)}: "
            f"{selection.sample_id}",
            flush=True,
        )
    require(len(staged) == EXPECTED_REPEAT, "REPEAT_16 staging incomplete")
    return staged


def train_lr_arm(
    label: str,
    learning_rate: float,
    schedule: Sequence[pilot.Selection],
    staged_bundles: dict[str, pilot.Bundle],
    tokenizer: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, LRTrainingResult]:
    require(len(schedule) == TOTAL_UPDATES, f"{label} schedule length differs")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    base, base_step, _, _ = integration.load_fm(checkpoint_path, device)
    fine, fine_step, _, _ = integration.load_fm(checkpoint_path, device)
    require(base_step == fine_step == 100_000, f"{label} checkpoint step differs")
    base.requires_grad_(False).eval()
    fine.requires_grad_(True).eval()
    probe = diversity.move_bundle(staged_bundles[schedule[0].sample_id], device)
    require(
        pilot.model_pair_initialization(base, fine, probe),
        f"{label} fresh initialization differs",
    )
    base_versions = pilot.model_versions(base)
    optimizer = torch.optim.AdamW(
        fine.parameters(), lr=learning_rate, weight_decay=pilot.WEIGHT_DECAY
    )
    del probe
    torch.cuda.empty_cache()

    updates: list[pilot.UpdateMetric] = []
    for step, selection in enumerate(schedule, start=1):
        bundle = diversity.move_bundle(
            staged_bundles[selection.sample_id], device
        )
        metric = diversity.update_once_diversity(
            step, bundle, base, fine, tokenizer, optimizer, device
        )
        updates.append(metric)
        if step == 1 or step % 10 == 0:
            print(
                f"{label} update {step}: sample={metric.sample_id} "
                f"AM={metric.am_loss:.7g} terminal={metric.terminal_loss:.7g} "
                f"grad={metric.preclip_grad_norm:.7g}->"
                f"{metric.postclip_grad_norm:.7g} clipped={metric.clipped}",
                flush=True,
            )
        del bundle
        torch.cuda.empty_cache()

    drift, relative_drift = pilot.parameter_drift(base, fine)
    finite = bool(
        all(
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
    frozen_ok = bool(
        pilot.versions_unchanged(base, base_versions)
        and integration.no_parameter_grads(base)
    )
    integration.synchronize(device)
    result = LRTrainingResult(
        label=label,
        learning_rate=learning_rate,
        updates=updates,
        parameter_drift=drift,
        relative_parameter_drift=relative_drift,
        finite=finite,
        frozen_ok=frozen_ok,
        runtime_seconds=time.perf_counter() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
    )
    del optimizer, base
    torch.cuda.empty_cache()
    return fine, result


def evaluate_new_arms(
    models: Sequence[tuple[str, torch.nn.Module]],
    selections: Sequence[pilot.Selection],
    historical_by_id: dict[str, dict[str, str]],
    dataset_dir: Path,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    integration.synchronize(device)
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for ordinal, selection in enumerate(selections, start=1):
            historical = historical_by_id[selection.sample_id]
            require(
                int(historical["length"]) == selection.length
                and historical["length_bin"] == selection.length_bin,
                f"historical identity differs: {selection.sample_id}",
            )
            bundle = pilot.load_bundle(
                dataset_dir, selection, extractor, tokenizer, device
            )
            generator = torch.Generator(device=device).manual_seed(
                pilot.stable_seed(
                    pilot.INFERENCE_SEED_NAMESPACE, selection.sample_id
                )
            )
            initial_noise = torch.randn(
                bundle.z_quant.shape,
                generator=generator,
                device=device,
                dtype=bundle.z_quant.dtype,
            ) * bundle.res_mask[..., None]
            row: dict[str, object] = {
                "protein_id": selection.sample_id,
                "length": selection.length,
                "length_bin": selection.length_bin,
                "base_aligned_loss": float(historical["base_aligned_loss"]),
                "base_latent_mse": float(historical["base_latent_mse"]),
                f"{LR_2E6}_aligned_loss": float(
                    historical["repeat16_aligned_loss"]
                ),
                f"{LR_2E6}_latent_mse": float(
                    historical["repeat16_latent_mse"]
                ),
                f"{LR_2E6}_delta_vs_base": float(
                    historical["repeat16_delta_vs_base"]
                ),
            }
            for label, model in models:
                residual = euler_sample(
                    model,
                    bundle.z_quant,
                    bundle.hidden_states,
                    bundle.res_mask,
                    100,
                    initial_noise=initial_noise.clone(),
                )
                decoded = tokenizer.detokenize(
                    bundle.z_quant + pilot.ALPHA_REFINE * residual,
                    res_mask=bundle.res_mask,
                )
                prediction, target, valid = integration.backbone_points(
                    decoded, bundle.oracle, bundle.res_mask
                )
                aligned = float(
                    integration.aligned_squared_rmsd(prediction, target, valid)
                )
                latent = pilot.masked_latent_mse(
                    residual,
                    bundle.z_cont - bundle.z_quant,
                    bundle.res_mask,
                )
                require(
                    math.isfinite(aligned) and math.isfinite(latent),
                    f"non-finite evaluation: {selection.sample_id}/{label}",
                )
                row[f"{label}_aligned_loss"] = aligned
                row[f"{label}_latent_mse"] = latent
                row[f"{label}_delta_vs_base"] = (
                    float(row["base_aligned_loss"]) - aligned
                )
                del residual, decoded, prediction, target, valid
            rows.append(row)
            if ordinal == 1 or ordinal % 16 == 0:
                print(
                    f"development evaluation {ordinal}/{len(selections)}: "
                    f"{selection.sample_id}",
                    flush=True,
                )
            del bundle, initial_noise
            if ordinal % 16 == 0:
                torch.cuda.empty_cache()
    integration.synchronize(device)
    return rows, time.perf_counter() - start


def compare_counts(deltas: Sequence[float]) -> tuple[int, int, int]:
    better = sum(delta > pilot.TIE_ATOL for delta in deltas)
    worse = sum(delta < -pilot.TIE_ATOL for delta in deltas)
    return better, worse, len(deltas) - better - worse


def model_summary(
    rows: Sequence[dict[str, object]], label: str
) -> dict[str, object]:
    losses = [float(row[f"{label}_aligned_loss"]) for row in rows]
    latent = [float(row[f"{label}_latent_mse"]) for row in rows]
    deltas = (
        [0.0] * len(rows)
        if label == "base"
        else [float(row[f"{label}_delta_vs_base"]) for row in rows]
    )
    improved, worse, tied = compare_counts(deltas)
    return {
        "n": len(rows),
        "mean_loss": statistics.fmean(losses),
        "median_loss": statistics.median(losses),
        "mean_latent_mse": statistics.fmean(latent),
        "mean_delta_vs_base": statistics.fmean(deltas),
        "median_delta_vs_base": statistics.median(deltas),
        "improved_vs_base": improved,
        "worse_vs_base": worse,
        "tied_vs_base": tied,
    }


def outlier_summary(
    rows: Sequence[dict[str, object]], label: str
) -> dict[str, object]:
    ranked = sorted(
        (
            {
                "protein_id": str(row["protein_id"]),
                "length": int(row["length"]),
                "delta": float(row[f"{label}_delta_vs_base"]),
            }
            for row in rows
        ),
        key=lambda entry: float(entry["delta"]),
        reverse=True,
    )
    deltas = [float(entry["delta"]) for entry in ranked]
    return {
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "largest_5_improvements": ranked[:5],
        "largest_5_degradations": list(reversed(ranked[-5:])),
        "mean_delta_excluding_largest_1_improvement": statistics.fmean(
            deltas[1:]
        ),
        "mean_delta_excluding_largest_5_improvements": statistics.fmean(
            deltas[5:]
        ),
    }


def length_bin_summaries(
    rows: Sequence[dict[str, object]]
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for lower, upper in LENGTH_BINS:
        label = f"{lower}-{upper}"
        subset = [row for row in rows if row["length_bin"] == label]
        require(len(subset) == 64, f"development bin {label} N differs")
        output[label] = {
            lr_label: model_summary(subset, lr_label)
            for lr_label in LR_LABELS
        }
    return output


def robust_criteria(
    base: dict[str, object],
    summary: dict[str, object],
    outliers: dict[str, object],
    finite: bool,
) -> dict[str, bool]:
    criteria = {
        "A_mean_below_base": summary["mean_loss"] < base["mean_loss"],
        "B_median_not_above_base": summary["median_loss"] <= base["median_loss"],
        "C_improved_gt_worse": (
            summary["improved_vs_base"] > summary["worse_vs_base"]
        ),
        "D_top5_removed_delta_nonnegative": (
            outliers["mean_delta_excluding_largest_5_improvements"] >= 0.0
        ),
        "E_finite": finite,
    }
    criteria["candidate"] = all(criteria.values())
    return criteria


def training_diagnostics(result: LRTrainingResult) -> dict[str, object]:
    diagnostics: dict[str, object] = dict(pilot.update_summary(result.updates))
    diagnostics.update(
        {
            "update_count": len(result.updates),
            "learning_rate": result.learning_rate,
            "parameter_drift": result.parameter_drift,
            "relative_parameter_drift": result.relative_parameter_drift,
            "finite": result.finite,
            "frozen_ok": result.frozen_ok,
            "runtime_seconds": result.runtime_seconds,
            "peak_allocated_mib": result.peak_allocated_mib,
            "peak_reserved_mib": result.peak_reserved_mib,
        }
    )
    return diagnostics


def write_per_target(
    path: Path, rows: Sequence[dict[str, object]]
) -> None:
    fields = (
        "protein_id",
        "length",
        "length_bin",
        "base_aligned_loss",
        f"{LR_5E7}_aligned_loss",
        f"{LR_1E6}_aligned_loss",
        f"{LR_2E6}_aligned_loss",
        f"{LR_5E7}_delta_vs_base",
        f"{LR_1E6}_delta_vs_base",
        f"{LR_2E6}_delta_vs_base",
        "base_latent_mse",
        f"{LR_5E7}_latent_mse",
        f"{LR_1E6}_latent_mse",
        f"{LR_2E6}_latent_mse",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_summary_csv(
    path: Path,
    base: dict[str, object],
    overall: dict[str, dict[str, object]],
    bins: dict[str, dict[str, dict[str, object]]],
    outliers: dict[str, dict[str, object]],
    criteria: dict[str, dict[str, bool]],
    relative_drifts: dict[str, float],
    clip_fractions: dict[str, float],
) -> None:
    fields = (
        "scope",
        "length_bin",
        "lr",
        "source",
        "n",
        "mean_loss",
        "median_loss",
        "mean_latent_mse",
        "mean_delta_vs_base",
        "median_delta_vs_base",
        "improved",
        "worse",
        "tied",
        "top1_removed_mean_delta",
        "top5_removed_mean_delta",
        "criterion_A",
        "criterion_B",
        "criterion_C",
        "criterion_D",
        "criterion_E",
        "robust_candidate",
        "relative_parameter_drift",
        "clip_fraction",
    )

    def row_for(
        scope: str,
        length_bin: str,
        label: str,
        summary: dict[str, object],
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "scope": scope,
            "length_bin": length_bin,
            "lr": "BASE" if label == "base" else LR_DISPLAY[label],
            "source": (
                "historical_read_only"
                if label in {"base", LR_2E6}
                else "new_step200"
            ),
            "n": summary["n"],
            "mean_loss": summary["mean_loss"],
            "median_loss": summary["median_loss"],
            "mean_latent_mse": summary["mean_latent_mse"],
            "mean_delta_vs_base": summary["mean_delta_vs_base"],
            "median_delta_vs_base": summary["median_delta_vs_base"],
            "improved": summary["improved_vs_base"],
            "worse": summary["worse_vs_base"],
            "tied": summary["tied_vs_base"],
        }
        if scope == "overall" and label != "base":
            row.update(
                {
                    "top1_removed_mean_delta": outliers[label][
                        "mean_delta_excluding_largest_1_improvement"
                    ],
                    "top5_removed_mean_delta": outliers[label][
                        "mean_delta_excluding_largest_5_improvements"
                    ],
                    "criterion_A": criteria[label]["A_mean_below_base"],
                    "criterion_B": criteria[label][
                        "B_median_not_above_base"
                    ],
                    "criterion_C": criteria[label]["C_improved_gt_worse"],
                    "criterion_D": criteria[label][
                        "D_top5_removed_delta_nonnegative"
                    ],
                    "criterion_E": criteria[label]["E_finite"],
                    "robust_candidate": criteria[label]["candidate"],
                    "relative_parameter_drift": relative_drifts[label],
                    "clip_fraction": clip_fractions[label],
                }
            )
        return row

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row_for("overall", "all", "base", base))
        for label in LR_LABELS:
            writer.writerow(row_for("overall", "all", label, overall[label]))
        for length_bin, records in bins.items():
            for label in LR_LABELS:
                writer.writerow(
                    row_for("length_bin", length_bin, label, records[label])
                )


def ranked_lines(
    title: str, entries: Sequence[dict[str, object]]
) -> list[str]:
    lines = [title]
    for entry in entries:
        lines.append(
            f"  {entry['protein_id']} L={entry['length']} "
            f"delta={float(entry['delta']):.9g}"
        )
    return lines


def make_summary_text(
    base: dict[str, object],
    overall: dict[str, dict[str, object]],
    bins: dict[str, dict[str, dict[str, object]]],
    outliers: dict[str, dict[str, object]],
    criteria: dict[str, dict[str, bool]],
    diagnostics: dict[str, dict[str, object]],
    historical_control: str,
    selected: str | None,
    staging_seconds: float,
    evaluation_seconds: float,
    total_seconds: float,
    global_peak_allocated: float,
    global_peak_reserved: float,
) -> str:
    lines = [
        "ADJOINT_LR_PILOT",
        "",
        "NON-PRODUCTION DEVELOPMENT EXPERIMENT",
        f"development N: {base['n']}",
        f"historical 2e-6 control: {historical_control}",
        f"BASE mean/median: {base['mean_loss']:.9g} / {base['median_loss']:.9g}",
    ]
    for label in LR_LABELS:
        summary = overall[label]
        outlier = outliers[label]
        lines.extend(
            [
                "",
                f"LR={LR_DISPLAY[label]}",
                f"mean/median: {summary['mean_loss']:.9g} / "
                f"{summary['median_loss']:.9g}",
                f"mean/median delta: {summary['mean_delta_vs_base']:.9g} / "
                f"{summary['median_delta_vs_base']:.9g}",
                f"improved/worse/tied: {summary['improved_vs_base']}/"
                f"{summary['worse_vs_base']}/{summary['tied_vs_base']}",
                f"latent MSE: {summary['mean_latent_mse']:.9g}",
                f"top1/top5-removed mean delta: "
                f"{outlier['mean_delta_excluding_largest_1_improvement']:.9g} / "
                f"{outlier['mean_delta_excluding_largest_5_improvements']:.9g}",
                "robust A/B/C/D/E: "
                + "/".join(
                    "PASS" if criteria[label][key] else "FAIL"
                    for key in (
                        "A_mean_below_base",
                        "B_median_not_above_base",
                        "C_improved_gt_worse",
                        "D_top5_removed_delta_nonnegative",
                        "E_finite",
                    )
                ),
                f"robust candidate: {criteria[label]['candidate']}",
            ]
        )
        if label in diagnostics:
            diagnostic = diagnostics[label]
            lines.extend(
                [
                    f"AM loss mean/range: {diagnostic['am_loss_mean']:.9g} "
                    f"[{diagnostic['am_loss_min']:.9g}, "
                    f"{diagnostic['am_loss_max']:.9g}]",
                    f"terminal loss mean/range: "
                    f"{diagnostic['terminal_loss_mean']:.9g} "
                    f"[{diagnostic['terminal_loss_min']:.9g}, "
                    f"{diagnostic['terminal_loss_max']:.9g}]",
                    f"preclip grad mean/range: "
                    f"{diagnostic['preclip_grad_mean']:.9g} "
                    f"[{diagnostic['preclip_grad_min']:.9g}, "
                    f"{diagnostic['preclip_grad_max']:.9g}]",
                    f"postclip grad mean/range: "
                    f"{diagnostic['postclip_grad_mean']:.9g} "
                    f"[{diagnostic['postclip_grad_min']:.9g}, "
                    f"{diagnostic['postclip_grad_max']:.9g}]",
                    f"clip count/fraction: {diagnostic['clipped_updates']}/200 "
                    f"({diagnostic['clip_fraction']:.9g})",
                    f"parameter/relative drift: "
                    f"{diagnostic['parameter_drift']:.9g} / "
                    f"{diagnostic['relative_parameter_drift']:.9g}",
                    f"training runtime seconds: "
                    f"{diagnostic['runtime_seconds']:.3f}",
                    f"peak CUDA allocated/reserved MiB: "
                    f"{diagnostic['peak_allocated_mib']:.2f}/"
                    f"{diagnostic['peak_reserved_mib']:.2f}",
                ]
            )

    lines.extend(["", "Length-bin results"])
    for length_bin, records in bins.items():
        lines.append(f"{length_bin} N=64")
        for label in LR_LABELS:
            record = records[label]
            lines.append(
                f"  {LR_DISPLAY[label]} mean/median/delta/counts: "
                f"{record['mean_loss']:.9g} / {record['median_loss']:.9g} / "
                f"{record['mean_delta_vs_base']:.9g} / "
                f"{record['improved_vs_base']}/{record['worse_vs_base']}/"
                f"{record['tied_vs_base']}"
            )

    lines.append("")
    for label in LR_LABELS:
        outlier = outliers[label]
        lines.extend(
            [
                f"Outlier diagnostics LR={LR_DISPLAY[label]}",
                f"  mean/median delta: {outlier['mean_delta']:.9g} / "
                f"{outlier['median_delta']:.9g}",
                f"  top1/top5-removed mean delta: "
                f"{outlier['mean_delta_excluding_largest_1_improvement']:.9g} / "
                f"{outlier['mean_delta_excluding_largest_5_improvements']:.9g}",
            ]
        )
        lines.extend(
            ranked_lines(
                "  largest 5 improvements",
                outlier["largest_5_improvements"],
            )
        )
        lines.extend(
            ranked_lines(
                "  largest 5 degradations",
                outlier["largest_5_degradations"],
            )
        )

    selection_marker = (
        f"SELECTED_LR = {LR_DISPLAY[selected]}"
        if selected is not None
        else "LR_PILOT_NO_ROBUST_CANDIDATE"
    )
    lines.extend(
        [
            "",
            selection_marker,
            f"TRAIN bundle staging seconds: {staging_seconds:.3f}",
            f"development evaluation seconds: {evaluation_seconds:.3f}",
            f"total runtime seconds: {total_seconds:.3f}",
            f"global peak CUDA allocated/reserved MiB: "
            f"{global_peak_allocated:.2f}/{global_peak_reserved:.2f}",
            "LOCKED_CONFIRMATION_EVALUATED = NO",
            "CAMEO_PDB_DATE_EVALUATED = NO",
            "production checkpoint selected: NO",
            "ADJOINT_LR_PILOT_PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    integration.seed_everything(42)
    device = integration.resolve_device(args.device)
    dataset_dir = Path(args.dataset_dir)
    development_path = Path(args.development_csv)
    locked_path = Path(args.locked_csv)
    protocol_metadata_path = Path(args.protocol_metadata)
    protocol_freeze_path = Path(args.protocol_freeze)
    pilot_train_path = Path(args.pilot_train)
    historical_results_path = Path(args.historical_results)
    historical_metadata_path = Path(args.historical_metadata)
    historical_summary_path = Path(args.historical_summary)
    historical_source_path = Path(args.historical_source)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    readonly_paths = (
        development_path,
        locked_path,
        protocol_metadata_path,
        protocol_freeze_path,
        pilot_train_path,
        historical_results_path,
        historical_metadata_path,
        historical_summary_path,
        historical_source_path,
        checkpoint_path,
    )
    total_start = time.perf_counter()
    stage = "startup"
    locked_evaluated = False
    cameo_evaluated = False
    metadata: dict[str, object] = {
        "label": "NON_PRODUCTION_DEVELOPMENT",
        "status": "RUNNING",
        "locked_confirmation_evaluated": False,
        "cameo_pdb_date_evaluated": False,
        "production_checkpoint_selected": False,
    }
    print("ADJOINT_LR_PILOT", flush=True)
    try:
        require(device.type == "cuda", "experiment requires CUDA")
        require(
            output_dir.resolve().is_relative_to(
                (REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()
            ),
            "output directory must remain below am_lr_pilot",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        hashes_before = {
            str(path): pilot.file_digest(path) for path in readonly_paths
        }

        stage = "validate frozen protocol"
        development, locked_ids, protocol_metadata = diversity.validate_protocol(
            development_path,
            locked_path,
            protocol_metadata_path,
            protocol_freeze_path,
        )
        development_ids = {selection.sample_id for selection in development}
        require(len(development) == EXPECTED_DEVELOPMENT, "development N differs")
        require(not (development_ids & locked_ids), "development/locked overlap")
        repeat = diversity.read_pilot_train(pilot_train_path)
        repeat_ids = {selection.sample_id for selection in repeat}
        require(not (repeat_ids & development_ids), "TRAIN/development overlap")
        require(not (repeat_ids & locked_ids), "TRAIN/locked overlap")
        schedule = tuple(
            repeat[index % len(repeat)] for index in range(TOTAL_UPDATES)
        )

        stage = "validate historical 2e-6 reference"
        historical_by_id, historical_metadata, historical_differences = (
            validate_historical_reference(
                historical_results_path,
                historical_metadata_path,
                historical_summary_path,
                historical_source_path,
                development_ids,
                locked_ids,
                hashes_before[str(development_path)],
                hashes_before[str(locked_path)],
            )
        )
        historical_control = (
            "EXACT_CONTROLLED_REFERENCE"
            if not historical_differences
            else "HISTORICAL_CONTROLLED_REFERENCE_WITH_DIFFERENCES"
        )
        metadata.update(
            {
                "base_checkpoint": str(checkpoint_path),
                "base_checkpoint_sha256": hashes_before[str(checkpoint_path)],
                "development_csv": str(development_path),
                "development_sha256": hashes_before[str(development_path)],
                "development_n": len(development),
                "locked_csv": str(locked_path),
                "locked_sha256": hashes_before[str(locked_path)],
                "locked_n_read_for_protection_audit": len(locked_ids),
                "historical_results": str(historical_results_path),
                "historical_results_sha256": hashes_before[
                    str(historical_results_path)
                ],
                "historical_metadata_sha256": hashes_before[
                    str(historical_metadata_path)
                ],
                "historical_control_classification": historical_control,
                "historical_control_differences": historical_differences,
                "new_learning_rates": [learning_rate for _, learning_rate in NEW_ARMS],
                "historical_learning_rate": 2e-6,
                "lambda_reward": LAMBDA_REWARD,
                "optimizer": "AdamW",
                "weight_decay": pilot.WEIGHT_DECAY,
                "global_gradient_clip": pilot.GRADIENT_CLIP_NORM,
                "total_optimizer_updates_per_new_arm": TOTAL_UPDATES,
                "physical_batch": 1,
                "gradient_accumulation": 1,
                "K": pilot.K,
                "h": pilot.H,
                "am_timestep_policy": "paper_H2_20",
                "per_timestep_am_loss_clipping": False,
                "alpha_refine": pilot.ALPHA_REFINE,
                "local_first_step": "DETERMINISTIC_WARM_START",
                "trajectory_seed_policy": "10000 + optimizer_step",
                "timestep_subset_seed_policy": "20000 + optimizer_step",
                "inference_seed_namespace": pilot.INFERENCE_SEED_NAMESPACE,
                "train_population": "REPEAT_16",
                "train_schedule": "deterministic cyclic order for 200 updates",
                "train_samples": [asdict(selection) for selection in repeat],
                "readonly_input_sha256": hashes_before,
            }
        )
        write_json(output_dir / "metadata.json", metadata)
        print(f"development N: {len(development)}", flush=True)
        print(
            f"historical 2e-6 control: {historical_control}", flush=True
        )
        print(
            "locked protection audit: IDs/hash read; no dataset item loaded",
            flush=True,
        )

        stage = "load frozen DPLM"
        extractor = integration.DPLMConditionExtractor(
            args.dplm_checkpoint, device=device
        )
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        pilot._TOKENIZER_HOLDER.clear()
        pilot._TOKENIZER_HOLDER.append(tokenizer)
        dplm_versions = pilot.model_versions(dplm)
        tokenizer_versions = pilot.model_versions(tokenizer)

        stage = "stage REPEAT_16 bundles on CPU"
        staging_start = time.perf_counter()
        staged_bundles = stage_repeat_bundles(
            repeat, dataset_dir, extractor, tokenizer, device
        )
        staging_seconds = time.perf_counter() - staging_start
        dplm.net.to("cpu")
        torch.cuda.empty_cache()

        models: dict[str, torch.nn.Module] = {}
        training_results: dict[str, LRTrainingResult] = {}
        for label, learning_rate in NEW_ARMS:
            stage = f"train {label}"
            model, result = train_lr_arm(
                label,
                learning_rate,
                schedule,
                staged_bundles,
                tokenizer,
                checkpoint_path,
                device,
            )
            models[label] = model
            training_results[label] = result
            if label != NEW_ARMS[-1][0]:
                model.to("cpu")
                torch.cuda.empty_cache()

        stage = "restore DPLM for development evaluation"
        del staged_bundles
        gc.collect()
        dplm.net.to(device)
        models[LR_5E7].to(device)
        torch.cuda.empty_cache()

        stage = "evaluate development 256 only"
        evaluation_rows, evaluation_seconds = evaluate_new_arms(
            ((LR_5E7, models[LR_5E7]), (LR_1E6, models[LR_1E6])),
            development,
            historical_by_id,
            dataset_dir,
            extractor,
            tokenizer,
            device,
        )
        require(
            len(evaluation_rows) == EXPECTED_DEVELOPMENT,
            "development evaluation incomplete",
        )
        require(
            {str(row["protein_id"]) for row in evaluation_rows}
            == development_ids,
            "development evaluation IDs differ",
        )
        require(
            not (
                {str(row["protein_id"]) for row in evaluation_rows}
                & locked_ids
            ),
            "locked ID appeared in evaluation",
        )

        base_summary = model_summary(evaluation_rows, "base")
        overall = {
            label: model_summary(evaluation_rows, label)
            for label in LR_LABELS
        }
        outliers = {
            label: outlier_summary(evaluation_rows, label)
            for label in LR_LABELS
        }
        bins = length_bin_summaries(evaluation_rows)
        new_diagnostics = {
            label: training_diagnostics(training_results[label])
            for label, _ in NEW_ARMS
        }
        historical_diagnostics = dict(
            historical_metadata["repeat16_training"]
        )
        historical_diagnostics["update_count"] = TOTAL_UPDATES
        diagnostics = {
            **new_diagnostics,
            LR_2E6: historical_diagnostics,
        }
        finite_by_label = {
            LR_5E7: training_results[LR_5E7].finite,
            LR_1E6: training_results[LR_1E6].finite,
            LR_2E6: bool(historical_diagnostics["finite"]),
        }
        criteria = {
            label: robust_criteria(
                base_summary,
                overall[label],
                outliers[label],
                finite_by_label[label],
            )
            for label in LR_LABELS
        }
        relative_drifts = {
            label: float(diagnostics[label]["relative_parameter_drift"])
            for label in LR_LABELS
        }
        clip_fractions = {
            label: float(diagnostics[label]["clip_fraction"])
            for label in LR_LABELS
        }
        candidates = [
            label for label in LR_LABELS if criteria[label]["candidate"]
        ]
        selected = (
            min(
                candidates,
                key=lambda label: (
                    float(overall[label]["median_loss"]),
                    -int(overall[label]["improved_vs_base"]),
                    float(overall[label]["mean_loss"]),
                    relative_drifts[label],
                ),
            )
            if candidates
            else None
        )

        all_finite = bool(
            all(finite_by_label.values())
            and all(
                math.isfinite(float(value))
                for row in evaluation_rows
                for key, value in row.items()
                if key not in {"protein_id", "length_bin"}
            )
        )
        frozen_ok = bool(
            all(result.frozen_ok for result in training_results.values())
            and pilot.versions_unchanged(dplm, dplm_versions)
            and pilot.versions_unchanged(tokenizer, tokenizer_versions)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
        )
        hashes_after = {
            str(path): pilot.file_digest(path) for path in readonly_paths
        }
        inputs_unchanged = hashes_before == hashes_after
        require(inputs_unchanged, "read-only input changed")
        require(not locked_evaluated, "locked confirmation was evaluated")
        require(not cameo_evaluated, "CAMEO/PDB-date was evaluated")

        write_per_target(
            output_dir / "per_target_development_results.csv",
            evaluation_rows,
        )
        write_summary_csv(
            output_dir / "lr_pilot_summary.csv",
            base_summary,
            overall,
            bins,
            outliers,
            criteria,
            relative_drifts,
            clip_fractions,
        )
        integration.synchronize(device)
        evaluation_peak_allocated = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        evaluation_peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 1024**2
        )
        global_peak_allocated = max(
            evaluation_peak_allocated,
            *(result.peak_allocated_mib for result in training_results.values()),
        )
        global_peak_reserved = max(
            evaluation_peak_reserved,
            *(result.peak_reserved_mib for result in training_results.values()),
        )
        total_seconds = time.perf_counter() - total_start
        summary_text = make_summary_text(
            base_summary,
            overall,
            bins,
            outliers,
            criteria,
            diagnostics,
            historical_control,
            selected,
            staging_seconds,
            evaluation_seconds,
            total_seconds,
            global_peak_allocated,
            global_peak_reserved,
        )
        (output_dir / "lr_pilot_summary.txt").write_text(
            summary_text, encoding="utf-8"
        )

        implementation_pass = all_finite and frozen_ok and inputs_unchanged
        selection_marker = (
            f"SELECTED_LR_{LR_DISPLAY[selected]}"
            if selected is not None
            else "LR_PILOT_NO_ROBUST_CANDIDATE"
        )
        metadata.update(
            {
                "status": "PASS" if implementation_pass else "FAIL",
                "base_development": base_summary,
                "lr_development": overall,
                "outlier_diagnostics": outliers,
                "length_bin_results": bins,
                "robust_criteria": criteria,
                "selected_lr": (
                    None if selected is None else LR_DISPLAY[selected]
                ),
                "selection_marker": selection_marker,
                "new_training_diagnostics": new_diagnostics,
                "historical_2e_6_training_diagnostics": historical_diagnostics,
                "training_bundle_staging_seconds": staging_seconds,
                "development_evaluation_seconds": evaluation_seconds,
                "total_runtime_seconds": total_seconds,
                "global_peak_cuda_allocated_mib": global_peak_allocated,
                "global_peak_cuda_reserved_mib": global_peak_reserved,
                "evaluation_peak_cuda_allocated_mib": evaluation_peak_allocated,
                "evaluation_peak_cuda_reserved_mib": evaluation_peak_reserved,
                "finite_audit": all_finite,
                "frozen_audit": frozen_ok,
                "readonly_inputs_unchanged": inputs_unchanged,
                "locked_confirmation_evaluated": locked_evaluated,
                "locked_dataset_items_loaded": False,
                "cameo_pdb_date_evaluated": cameo_evaluated,
                "production_checkpoint_selected": False,
                "checkpoint_artifacts_written": False,
            }
        )
        write_json(output_dir / "metadata.json", metadata)

        print("\nADJOINT_LR_PILOT", flush=True)
        print(f"development N: {len(evaluation_rows)}", flush=True)
        print(
            f"BASE mean/median: {base_summary['mean_loss']:.9g} / "
            f"{base_summary['median_loss']:.9g}",
            flush=True,
        )
        for label in LR_LABELS:
            summary = overall[label]
            outlier = outliers[label]
            print(f"LR={LR_DISPLAY[label]}:", flush=True)
            print(
                f"  mean/median: {summary['mean_loss']:.9g} / "
                f"{summary['median_loss']:.9g}",
                flush=True,
            )
            print(
                f"  delta: {summary['mean_delta_vs_base']:.9g}",
                flush=True,
            )
            print(
                f"  improved/worse/tied: {summary['improved_vs_base']}/"
                f"{summary['worse_vs_base']}/{summary['tied_vs_base']}",
                flush=True,
            )
            print(
                f"  top1/top5-removed delta: "
                f"{outlier['mean_delta_excluding_largest_1_improvement']:.9g} / "
                f"{outlier['mean_delta_excluding_largest_5_improvements']:.9g}",
                flush=True,
            )
            print(
                f"  clip fraction: {diagnostics[label]['clip_fraction']:.9g}",
                flush=True,
            )
            print(
                f"  relative drift: "
                f"{diagnostics[label]['relative_parameter_drift']:.9g}",
                flush=True,
            )
            print(
                "  robust A/B/C/D/E: "
                + "/".join(
                    "PASS" if criteria[label][key] else "FAIL"
                    for key in (
                        "A_mean_below_base",
                        "B_median_not_above_base",
                        "C_improved_gt_worse",
                        "D_top5_removed_delta_nonnegative",
                        "E_finite",
                    )
                ),
                flush=True,
            )
        print("Length-bin results: recorded", flush=True)
        print("Outlier diagnostics: recorded", flush=True)
        print(selection_marker, flush=True)
        print("LOCKED_CONFIRMATION_EVALUATED: NO", flush=True)
        print("CAMEO_PDB_DATE_EVALUATED: NO", flush=True)
        marker = (
            "ADJOINT_LR_PILOT_PASS"
            if implementation_pass
            else "ADJOINT_LR_PILOT_FAIL"
        )
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if implementation_pass else 1
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        metadata.update(
            {
                "status": "FAIL",
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "locked_confirmation_evaluated": locked_evaluated,
                "cameo_pdb_date_evaluated": cameo_evaluated,
                "production_checkpoint_selected": False,
            }
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "metadata.json", metadata)
        except Exception:
            pass
        print(f"failure stage: {stage}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("LOCKED_CONFIRMATION_EVALUATED: NO", flush=True)
        print("CAMEO_PDB_DATE_EVALUATED: NO", flush=True)
        print("overall: ADJOINT_LR_PILOT_FAIL", flush=True)
        print("ADJOINT_LR_PILOT_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

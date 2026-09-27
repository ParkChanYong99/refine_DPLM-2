#!/usr/bin/env python3
"""Compare repeated-16 versus unique-200 AM training diversity on AM dev 256."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
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

from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm import pilot_adjoint_lambda as pilot
from experiments.latent_residual_fm.flow_matching import euler_sample


OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_training_diversity")
PROTOCOL_ROOT = Path("experiments/latent_residual_fm/runs/am_validation_protocol")
DEVELOPMENT_RELATIVE = PROTOCOL_ROOT / "am_development_256.csv"
LOCKED_RELATIVE = PROTOCOL_ROOT / "am_locked_confirmation_256.csv"
PROTOCOL_METADATA_RELATIVE = PROTOCOL_ROOT / "metadata.json"
PROTOCOL_FREEZE_RELATIVE = PROTOCOL_ROOT / "VALIDATION_PROTOCOL_FREEZE.txt"
PILOT_TRAIN_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_train_indices.csv"
)
SPLIT_RELATIVE = Path(
    "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv"
)

ARM_REPEAT = "REPEAT_16"
ARM_UNIQUE = "UNIQUE_200"
LAMBDA_REWARD = 1.0
TOTAL_UPDATES = 200
UNIQUE_SELECTION_SEED = 5151
TRAJECTORY_SEED_BASE = 10_000
TIMESTEP_SEED_BASE = 20_000
UNIQUE_PER_BIN = 50
ADDITIONAL_PER_BIN = 46
EXPECTED_DEVELOPMENT = 256
EXPECTED_LOCKED = 256
EXPECTED_REPEAT = 16
EXPECTED_UNIQUE = 200
LENGTH_BINS = pilot.LENGTH_BINS


@dataclass
class ArmTrainingResult:
    name: str
    updates: list[pilot.UpdateMetric]
    parameter_drift: float
    relative_parameter_drift: float
    finite: bool
    frozen_ok: bool
    runtime_seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    schedule_ids: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded"
    )
    parser.add_argument("--split_csv", default=str(SPLIT_RELATIVE))
    parser.add_argument("--development_csv", default=str(DEVELOPMENT_RELATIVE))
    parser.add_argument("--locked_csv", default=str(LOCKED_RELATIVE))
    parser.add_argument(
        "--protocol_metadata", default=str(PROTOCOL_METADATA_RELATIVE)
    )
    parser.add_argument("--protocol_freeze", default=str(PROTOCOL_FREEZE_RELATIVE))
    parser.add_argument("--pilot_train", default=str(PILOT_TRAIN_RELATIVE))
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
    require(rows, f"empty CSV: {path}")
    return rows


def bin_label(length: int) -> str:
    for lower, upper in LENGTH_BINS:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    raise AssertionError(f"length outside bins: {length}")


def read_pilot_train(path: Path) -> tuple[pilot.Selection, ...]:
    rows = read_csv(path)
    require(len(rows) == EXPECTED_REPEAT, "pilot train must contain 16 samples")
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
    require(all(item.split == "train" for item in selections), "non-train pilot row")
    require(len({item.sample_id for item in selections}) == EXPECTED_REPEAT, "duplicate pilot train")
    counts = Counter(item.length_bin for item in selections)
    require(
        counts == {f"{a}-{b}": 4 for a, b in LENGTH_BINS},
        f"pilot train bins differ: {counts}",
    )
    return selections


def read_protocol_selection(
    path: Path, expected_n: int
) -> tuple[pilot.Selection, ...]:
    rows = read_csv(path)
    require(len(rows) == expected_n, f"{path} N={len(rows)}, expected {expected_n}")
    selections = tuple(
        pilot.Selection(
            split="val",
            split_index=int(row["validation_index"]),
            sample_id=row["protein_id"],
            length=int(row["length"]),
            length_bin=row["length_bin"],
            shard_file=row["shard_file"],
            index_in_shard=int(row["index_in_shard"]),
        )
        for row in rows
    )
    require(len({item.sample_id for item in selections}) == expected_n, f"duplicate IDs in {path}")
    return selections


def validate_protocol(
    development_path: Path,
    locked_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[tuple[pilot.Selection, ...], set[str], dict[str, object]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["validation_protocol_frozen"] is True, "validation protocol not frozen")
    require(metadata["status"] == "PASS", "validation protocol status is not PASS")
    development_hash = pilot.file_digest(development_path)
    locked_hash = pilot.file_digest(locked_path)
    require(development_hash == metadata["development_sha256"], "development SHA mismatch")
    require(locked_hash == metadata["locked_sha256"], "locked SHA mismatch")
    freeze_text = freeze_path.read_text(encoding="utf-8")
    require(development_hash in freeze_text and locked_hash in freeze_text, "freeze marker SHA mismatch")
    development = read_protocol_selection(development_path, EXPECTED_DEVELOPMENT)
    locked = read_protocol_selection(locked_path, EXPECTED_LOCKED)
    development_ids = {item.sample_id for item in development}
    locked_ids = {item.sample_id for item in locked}
    require(not (development_ids & locked_ids), "development/locked overlap")
    require(
        Counter(item.length_bin for item in development)
        == {f"{a}-{b}": 64 for a, b in LENGTH_BINS},
        "development bin mismatch",
    )
    return development, locked_ids, metadata


def train_rows(split_csv: Path) -> list[dict[str, str]]:
    rows = [row for row in read_csv(split_csv) if row.get("split") == "train"]
    require(rows, "empty train split")
    require(len({row["sample_id"] for row in rows}) == len(rows), "duplicate train ID")
    return rows


def choose_unique_200(
    rows: Sequence[dict[str, str]], repeat: Sequence[pilot.Selection]
) -> tuple[pilot.Selection, ...]:
    repeat_ids = {item.sample_id for item in repeat}
    by_id = {row["sample_id"]: (index, row) for index, row in enumerate(rows)}
    for item in repeat:
        require(item.sample_id in by_id, f"repeat sample absent from train split: {item.sample_id}")
        index, row = by_id[item.sample_id]
        require(index == item.split_index, f"repeat split index mismatch: {item.sample_id}")
        require(int(row["length"]) == item.length, f"repeat length mismatch: {item.sample_id}")

    rng = random.Random(UNIQUE_SELECTION_SEED)
    additions: list[pilot.Selection] = []
    for lower, upper in LENGTH_BINS:
        label = f"{lower}-{upper}"
        candidates = sorted(
            (
                (index, row)
                for index, row in enumerate(rows)
                if lower <= int(row["length"]) <= upper
                and row["sample_id"] not in repeat_ids
            ),
            key=lambda item: (item[1]["sample_id"], item[0]),
        )
        require(len(candidates) >= ADDITIONAL_PER_BIN, f"insufficient train bin {label}")
        for split_index, row in rng.sample(candidates, ADDITIONAL_PER_BIN):
            additions.append(
                pilot.Selection(
                    split="train",
                    split_index=split_index,
                    sample_id=row["sample_id"],
                    length=int(row["length"]),
                    length_bin=label,
                    shard_file=row["shard_file"],
                    index_in_shard=int(row["index_in_shard"]),
                )
            )
    rng.shuffle(additions)
    unique = tuple(repeat) + tuple(additions)
    require(len(unique) == EXPECTED_UNIQUE, "UNIQUE_200 size mismatch")
    require(len({item.sample_id for item in unique}) == EXPECTED_UNIQUE, "UNIQUE_200 duplicate")
    counts = Counter(item.length_bin for item in unique)
    require(
        counts == {f"{a}-{b}": UNIQUE_PER_BIN for a, b in LENGTH_BINS},
        f"UNIQUE_200 bins differ: {counts}",
    )
    require(tuple(item.sample_id for item in unique[:16]) == tuple(item.sample_id for item in repeat), "first 16 schedule mismatch")
    return unique


def write_unique_indices(path: Path, selections: Sequence[pilot.Selection]) -> None:
    fields = (
        "selection_order",
        "split",
        "split_index",
        "sample_id",
        "length",
        "length_bin",
        "shard_file",
        "index_in_shard",
        "is_original_repeat16",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(selections):
            row = asdict(item)
            row.update(
                {
                    "selection_order": index,
                    "is_original_repeat16": index < EXPECTED_REPEAT,
                }
            )
            writer.writerow({field: row[field] for field in fields})


def move_nested(value: object, device: torch.device | str) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_nested(item, device) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device) for item in value]
    return value


def move_bundle(bundle: pilot.Bundle, device: torch.device | str) -> pilot.Bundle:
    return pilot.Bundle(
        selection=bundle.selection,
        z_cont=bundle.z_cont.to(device),
        z_quant=bundle.z_quant.to(device),
        res_mask=bundle.res_mask.to(device),
        hidden_states=tuple(state.to(device) for state in bundle.hidden_states),
        oracle=move_nested(bundle.oracle, device),
    )


def stage_training_bundles(
    selections: Sequence[pilot.Selection],
    dataset_dir: Path,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> dict[str, pilot.Bundle]:
    staged: dict[str, pilot.Bundle] = {}
    for ordinal, selection in enumerate(selections, start=1):
        bundle = pilot.load_bundle(
            dataset_dir, selection, extractor, tokenizer, device
        )
        staged[selection.sample_id] = move_bundle(bundle, "cpu")
        del bundle
        torch.cuda.empty_cache()
        if ordinal == 1 or ordinal % 10 == 0:
            print(
                f"TRAIN bundle staging {ordinal}/{len(selections)}: "
                f"{selection.sample_id}",
                flush=True,
            )
    require(len(staged) == EXPECTED_UNIQUE, "TRAIN bundle staging incomplete")
    return staged


def update_once_diversity(
    step: int,
    bundle: pilot.Bundle,
    base: torch.nn.Module,
    fine: torch.nn.Module,
    tokenizer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> pilot.UpdateMetric:
    integration.synchronize(device)
    start = time.perf_counter()
    base_fn = integration.velocity_adapter(
        base, bundle.z_quant, bundle.hidden_states, bundle.res_mask
    )
    fine_fn = integration.velocity_adapter(
        fine, bundle.z_quant, bundle.hidden_states, bundle.res_mask
    )
    trajectory_seed = TRAJECTORY_SEED_BASE + step
    states = integration.build_trajectory(fine_fn, bundle.res_mask, trajectory_seed)
    integration.require(len(states) == pilot.K + 1, "trajectory grid incomplete")
    x_hat_1 = integration.make_x_hat_1(states, base_fn, bundle.res_mask)
    terminal_loss, a1 = pilot.terminal_adjoint(
        x_hat_1, bundle, tokenizer, LAMBDA_REWARD
    )
    adjoints = pilot.build_adjoints(states, a1, base_fn, bundle.res_mask)
    selected_timesteps = pilot.choose_timesteps(
        random.Random(TIMESTEP_SEED_BASE + step)
    )
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index in selected_timesteps:
        time_value = torch.tensor(
            index * pilot.H, device=device, dtype=bundle.z_quant.dtype
        )
        losses.append(
            pilot.adjoint_matching_loss(
                states[index],
                adjoints[index],
                time_value,
                fine_fn,
                base_fn,
                pilot.sigma_offset(time_value, pilot.H),
                bundle.res_mask,
            )
        )
    loss = torch.stack(losses).sum()
    integration.require(bool(torch.isfinite(loss)), "AM loss NaN/Inf")
    loss_value = float(loss.detach())
    loss.backward()
    preclip = float(
        torch.nn.utils.clip_grad_norm_(
            fine.parameters(), pilot.GRADIENT_CLIP_NORM, error_if_nonfinite=True
        )
    )
    postclip = pilot.gradient_norm(fine)
    integration.require(math.isfinite(preclip) and math.isfinite(postclip), "grad NaN/Inf")
    integration.require(integration.no_parameter_grads(base), "base received gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    integration.synchronize(device)
    metric = pilot.UpdateMetric(
        step=step,
        sample_id=bundle.selection.sample_id,
        am_loss=loss_value,
        terminal_loss=terminal_loss,
        terminal_adjoint_norm=float(torch.linalg.vector_norm(a1)),
        preclip_grad_norm=preclip,
        postclip_grad_norm=postclip,
        clipped=preclip > pilot.GRADIENT_CLIP_NORM,
        seconds=time.perf_counter() - start,
    )
    del base_fn, fine_fn, states, x_hat_1, a1, adjoints, losses, loss
    return metric


def train_arm(
    name: str,
    schedule: Sequence[pilot.Selection],
    staged_bundles: dict[str, pilot.Bundle],
    tokenizer: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ArmTrainingResult]:
    require(len(schedule) == TOTAL_UPDATES, f"{name} schedule length mismatch")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    base, base_step, _, _ = integration.load_fm(checkpoint_path, device)
    fine, fine_step, _, _ = integration.load_fm(checkpoint_path, device)
    integration.require(base_step == fine_step == 100_000, "checkpoint step mismatch")
    base.requires_grad_(False).eval()
    fine.requires_grad_(True).eval()

    probe = move_bundle(staged_bundles[schedule[0].sample_id], device)
    integration.require(
        pilot.model_pair_initialization(base, fine, probe),
        f"{name} fresh base/fine copies differ",
    )
    base_versions = pilot.model_versions(base)
    optimizer = torch.optim.AdamW(
        fine.parameters(),
        lr=pilot.LEARNING_RATE_DIAGNOSTIC,
        weight_decay=pilot.WEIGHT_DECAY,
    )
    del probe
    torch.cuda.empty_cache()

    updates: list[pilot.UpdateMetric] = []
    for step, item in enumerate(schedule, start=1):
        bundle = move_bundle(staged_bundles[item.sample_id], device)
        metric = update_once_diversity(
            step, bundle, base, fine, tokenizer, optimizer, device
        )
        updates.append(metric)
        if step == 1 or step % 10 == 0:
            print(
                f"{name} update {step}: sample={metric.sample_id} "
                f"AM={metric.am_loss:.7g} terminal={metric.terminal_loss:.7g} "
                f"grad={metric.preclip_grad_norm:.7g}->{metric.postclip_grad_norm:.7g} "
                f"clipped={metric.clipped}",
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
    result = ArmTrainingResult(
        name=name,
        updates=updates,
        parameter_drift=drift,
        relative_parameter_drift=relative_drift,
        finite=finite,
        frozen_ok=frozen_ok,
        runtime_seconds=time.perf_counter() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        schedule_ids=tuple(item.sample_id for item in schedule),
    )
    del optimizer, base
    torch.cuda.empty_cache()
    return fine, result


def evaluate_development(
    base: torch.nn.Module,
    repeat_model: torch.nn.Module,
    unique_model: torch.nn.Module,
    selections: Sequence[pilot.Selection],
    dataset_dir: Path,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    integration.synchronize(device)
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    models = (
        ("base", base),
        ("repeat16", repeat_model),
        ("unique200", unique_model),
    )
    with torch.no_grad():
        for ordinal, selection in enumerate(selections, start=1):
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
                del residual, decoded, prediction, target, valid
            row["repeat16_delta_vs_base"] = float(row["base_aligned_loss"]) - float(
                row["repeat16_aligned_loss"]
            )
            row["unique200_delta_vs_base"] = float(row["base_aligned_loss"]) - float(
                row["unique200_aligned_loss"]
            )
            row["unique200_delta_vs_repeat16"] = float(
                row["repeat16_aligned_loss"]
            ) - float(row["unique200_aligned_loss"])
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
    tied = len(deltas) - better - worse
    return better, worse, tied


def model_summary(
    rows: Sequence[dict[str, object]], label: str
) -> dict[str, object]:
    losses = [float(row[f"{label}_aligned_loss"]) for row in rows]
    latent = [float(row[f"{label}_latent_mse"]) for row in rows]
    if label == "base":
        deltas = [0.0] * len(rows)
    else:
        deltas = [
            float(row["base_aligned_loss"]) - float(row[f"{label}_aligned_loss"])
            for row in rows
        ]
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


def direct_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    deltas = [float(row["unique200_delta_vs_repeat16"]) for row in rows]
    better, worse, tied = compare_counts(deltas)
    repeat_losses = [float(row["repeat16_aligned_loss"]) for row in rows]
    unique_losses = [float(row["unique200_aligned_loss"]) for row in rows]
    return {
        "n": len(rows),
        "mean_difference_repeat_minus_unique": statistics.fmean(deltas),
        "median_difference_repeat_minus_unique": (
            statistics.median(repeat_losses) - statistics.median(unique_losses)
        ),
        "median_paired_delta_repeat_minus_unique": statistics.median(deltas),
        "unique_better": better,
        "repeat_better": worse,
        "tied": tied,
    }


def length_bin_summaries(
    rows: Sequence[dict[str, object]]
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for lower, upper in LENGTH_BINS:
        label = f"{lower}-{upper}"
        subset = [row for row in rows if row["length_bin"] == label]
        require(len(subset) == 64, f"development bin {label} N={len(subset)}")
        output[label] = {
            "base": model_summary(subset, "base"),
            "repeat16": model_summary(subset, "repeat16"),
            "unique200": model_summary(subset, "unique200"),
            "unique200_vs_repeat16": direct_summary(subset),
        }
    return output


def outlier_summary(
    rows: Sequence[dict[str, object]], candidate: str, reference: str
) -> dict[str, object]:
    scored = []
    for row in rows:
        delta = float(row[f"{reference}_aligned_loss"]) - float(
            row[f"{candidate}_aligned_loss"]
        )
        scored.append(
            {
                "protein_id": row["protein_id"],
                "length": row["length"],
                "length_bin": row["length_bin"],
                "delta": delta,
                "reference_loss": row[f"{reference}_aligned_loss"],
                "candidate_loss": row[f"{candidate}_aligned_loss"],
            }
        )
    ranked = sorted(scored, key=lambda row: float(row["delta"]), reverse=True)
    deltas = [float(row["delta"]) for row in ranked]
    return {
        "candidate": candidate,
        "reference": reference,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "largest_5_improvements": ranked[:5],
        "largest_5_degradations": list(reversed(ranked[-5:])),
        "mean_delta_excluding_largest_1_improvement": statistics.fmean(deltas[1:]),
        "mean_delta_excluding_largest_5_improvements": statistics.fmean(deltas[5:]),
    }


def write_per_target(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "protein_id",
        "length",
        "length_bin",
        "base_aligned_loss",
        "repeat16_aligned_loss",
        "unique200_aligned_loss",
        "repeat16_delta_vs_base",
        "unique200_delta_vs_base",
        "unique200_delta_vs_repeat16",
        "base_latent_mse",
        "repeat16_latent_mse",
        "unique200_latent_mse",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_summary_csv(
    path: Path,
    overall: dict[str, dict[str, object]],
    bins: dict[str, dict[str, object]],
) -> None:
    fields = (
        "scope",
        "length_bin",
        "record",
        "n",
        "mean_loss",
        "median_loss",
        "mean_latent_mse",
        "mean_delta_vs_base",
        "median_delta_vs_base",
        "improved",
        "worse",
        "tied",
        "mean_difference_repeat_minus_unique",
        "median_difference_repeat_minus_unique",
        "unique_better",
        "repeat_better",
    )

    def emit(writer: csv.DictWriter, scope: str, length_bin: str, records: dict[str, object]) -> None:
        for name in ("base", "repeat16", "unique200"):
            summary = records[name]
            writer.writerow(
                {
                    "scope": scope,
                    "length_bin": length_bin,
                    "record": name,
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
            )
        direct = records["unique200_vs_repeat16"]
        writer.writerow(
            {
                "scope": scope,
                "length_bin": length_bin,
                "record": "unique200_vs_repeat16",
                "n": direct["n"],
                "tied": direct["tied"],
                "mean_difference_repeat_minus_unique": direct[
                    "mean_difference_repeat_minus_unique"
                ],
                "median_difference_repeat_minus_unique": direct[
                    "median_difference_repeat_minus_unique"
                ],
                "unique_better": direct["unique_better"],
                "repeat_better": direct["repeat_better"],
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        emit(writer, "overall", "all", overall)
        for label, records in bins.items():
            emit(writer, "length_bin", label, records)


def diagnostic_payload(result: ArmTrainingResult) -> dict[str, object]:
    output = pilot.update_summary(result.updates)
    output.update(
        {
            "parameter_drift": result.parameter_drift,
            "relative_parameter_drift": result.relative_parameter_drift,
            "finite": result.finite,
            "frozen_ok": result.frozen_ok,
            "runtime_seconds": result.runtime_seconds,
            "peak_allocated_mib": result.peak_allocated_mib,
            "peak_reserved_mib": result.peak_reserved_mib,
        }
    )
    return output


def ranked_lines(title: str, entries: Sequence[dict[str, object]]) -> list[str]:
    lines = [title]
    for entry in entries:
        lines.append(
            f"  {entry['protein_id']} L={entry['length']} "
            f"delta={float(entry['delta']):.9g}"
        )
    return lines


def summary_text(
    overall: dict[str, dict[str, object]],
    bins: dict[str, dict[str, object]],
    repeat_training: dict[str, object],
    unique_training: dict[str, object],
    outliers: dict[str, dict[str, object]],
    diversity_supported: bool,
    broad_improvement: bool,
    evaluation_seconds: float,
    total_seconds: float,
    global_peak_allocated: float,
    global_peak_reserved: float,
) -> str:
    base = overall["base"]
    repeat = overall["repeat16"]
    unique = overall["unique200"]
    direct = overall["unique200_vs_repeat16"]
    lines = [
        "ADJOINT_TRAINING_DIVERSITY",
        "",
        "NON-PRODUCTION DEVELOPMENT EXPERIMENT",
        f"development N: {base['n']}",
        "",
        f"BASE mean/median: {base['mean_loss']:.9g} / {base['median_loss']:.9g}",
        f"BASE latent MSE: {base['mean_latent_mse']:.9g}",
    ]
    for label, metrics, training in (
        (ARM_REPEAT, repeat, repeat_training),
        (ARM_UNIQUE, unique, unique_training),
    ):
        lines.extend(
            [
                "",
                label,
                f"mean: {metrics['mean_loss']:.9g}",
                f"median: {metrics['median_loss']:.9g}",
                f"delta vs base: {metrics['mean_delta_vs_base']:.9g}",
                f"improved/worse/tied: {metrics['improved_vs_base']}/"
                f"{metrics['worse_vs_base']}/{metrics['tied_vs_base']}",
                f"latent MSE: {metrics['mean_latent_mse']:.9g}",
                f"AM loss mean/range: {training['am_loss_mean']:.9g} "
                f"[{training['am_loss_min']:.9g}, {training['am_loss_max']:.9g}]",
                f"terminal loss mean/range: {training['terminal_loss_mean']:.9g} "
                f"[{training['terminal_loss_min']:.9g}, "
                f"{training['terminal_loss_max']:.9g}]",
                f"preclip gradient mean/range: {training['preclip_grad_mean']:.9g} "
                f"[{training['preclip_grad_min']:.9g}, "
                f"{training['preclip_grad_max']:.9g}]",
                f"postclip gradient mean/range: {training['postclip_grad_mean']:.9g} "
                f"[{training['postclip_grad_min']:.9g}, "
                f"{training['postclip_grad_max']:.9g}]",
                f"clip count/fraction: {training['clipped_updates']}/200 "
                f"({training['clip_fraction']:.9g})",
                f"parameter drift: {training['parameter_drift']:.9g}",
                f"relative drift: {training['relative_parameter_drift']:.9g}",
                f"runtime seconds: {training['runtime_seconds']:.3f}",
                f"peak CUDA allocated/reserved MiB: "
                f"{training['peak_allocated_mib']:.2f}/"
                f"{training['peak_reserved_mib']:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "DIRECT UNIQUE_200 vs REPEAT_16",
            f"mean difference repeat-unique: "
            f"{direct['mean_difference_repeat_minus_unique']:.9g}",
            f"median difference repeat-unique: "
            f"{direct['median_difference_repeat_minus_unique']:.9g}",
            f"unique better/repeat better/tied: {direct['unique_better']}/"
            f"{direct['repeat_better']}/{direct['tied']}",
            "",
            "DIVERSITY_HYPOTHESIS: "
            + ("SUPPORTED" if diversity_supported else "NOT_SUPPORTED"),
            (
                "DIVERSITY_HYPOTHESIS_SUPPORTED"
                if diversity_supported
                else "DIVERSITY_HYPOTHESIS_NOT_SUPPORTED"
            ),
            "UNIQUE_200_BROAD_DEV_IMPROVEMENT: "
            + ("YES" if broad_improvement else "NO"),
            "",
            "Length-bin results",
        ]
    )
    for label, records in bins.items():
        r = records["repeat16"]
        u = records["unique200"]
        d = records["unique200_vs_repeat16"]
        lines.extend(
            [
                f"{label} N={r['n']}",
                f"  repeat mean/median/delta/counts: {r['mean_loss']:.9g} / "
                f"{r['median_loss']:.9g} / {r['mean_delta_vs_base']:.9g} / "
                f"{r['improved_vs_base']}/{r['worse_vs_base']}/{r['tied_vs_base']}",
                f"  unique mean/median/delta/counts: {u['mean_loss']:.9g} / "
                f"{u['median_loss']:.9g} / {u['mean_delta_vs_base']:.9g} / "
                f"{u['improved_vs_base']}/{u['worse_vs_base']}/{u['tied_vs_base']}",
                f"  direct mean difference/counts: "
                f"{d['mean_difference_repeat_minus_unique']:.9g} / "
                f"{d['unique_better']}/{d['repeat_better']}/{d['tied']}",
            ]
        )
    lines.append("")
    for name, diagnostic in outliers.items():
        lines.extend(
            [
                f"Outlier diagnostics {name}",
                f"  mean/median delta: {diagnostic['mean_delta']:.9g} / "
                f"{diagnostic['median_delta']:.9g}",
                f"  mean excluding top1/top5 improvements: "
                f"{diagnostic['mean_delta_excluding_largest_1_improvement']:.9g} / "
                f"{diagnostic['mean_delta_excluding_largest_5_improvements']:.9g}",
            ]
        )
        lines.extend(ranked_lines("  largest 5 improvements", diagnostic["largest_5_improvements"]))
        lines.extend(ranked_lines("  largest 5 degradations", diagnostic["largest_5_degradations"]))
    lines.extend(
        [
            "",
            f"development evaluation seconds: {evaluation_seconds:.3f}",
            f"total runtime seconds: {total_seconds:.3f}",
            f"global peak CUDA allocated/reserved MiB: "
            f"{global_peak_allocated:.2f}/{global_peak_reserved:.2f}",
            "LOCKED_CONFIRMATION_EVALUATED = NO",
            "CAMEO_PDB_DATE_EVALUATED = NO",
            "production hyperparameters/checkpoint selected: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    integration.seed_everything(42)
    device = integration.resolve_device(args.device)
    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    split_path = Path(args.split_csv)
    development_path = Path(args.development_csv)
    locked_path = Path(args.locked_csv)
    protocol_metadata_path = Path(args.protocol_metadata)
    protocol_freeze_path = Path(args.protocol_freeze)
    pilot_train_path = Path(args.pilot_train)
    checkpoint_path = Path(args.checkpoint)
    readonly_paths = (
        split_path,
        development_path,
        locked_path,
        protocol_metadata_path,
        protocol_freeze_path,
        pilot_train_path,
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
        "production_hyperparameters_selected": False,
    }
    print("ADJOINT_TRAINING_DIVERSITY", flush=True)
    try:
        require(device.type == "cuda", "experiment requires CUDA")
        require(
            output_dir.resolve().is_relative_to(
                (REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()
            ),
            "output directory must remain below am_training_diversity",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        hashes_before = {str(path): pilot.file_digest(path) for path in readonly_paths}

        stage = "validate frozen development and locked protocol"
        development, locked_ids, protocol_metadata = validate_protocol(
            development_path,
            locked_path,
            protocol_metadata_path,
            protocol_freeze_path,
        )
        development_ids = {item.sample_id for item in development}
        require(not (development_ids & locked_ids), "development/locked overlap")
        repeat = read_pilot_train(pilot_train_path)
        unique = choose_unique_200(train_rows(split_path), repeat)
        require(not ({item.sample_id for item in unique} & development_ids), "train/development overlap")
        require(not ({item.sample_id for item in unique} & locked_ids), "train/locked overlap")
        repeat_schedule = tuple(repeat[index % len(repeat)] for index in range(TOTAL_UPDATES))
        unique_schedule = unique
        write_unique_indices(output_dir / "unique200_train_indices.csv", unique)

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
                "unique_selection_seed": UNIQUE_SELECTION_SEED,
                "trajectory_seed_policy": "10000 + optimizer_step",
                "timestep_subset_seed_policy": "20000 + optimizer_step",
                "lambda_reward": LAMBDA_REWARD,
                "learning_rate": pilot.LEARNING_RATE_DIAGNOSTIC,
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
                "unique200_samples": [asdict(item) for item in unique],
                "repeat16_schedule_counts": Counter(item.sample_id for item in repeat_schedule),
                "readonly_input_sha256": hashes_before,
            }
        )
        write_json(output_dir / "metadata.json", metadata)
        print(f"development N: {len(development)}", flush=True)
        print("locked protection audit: IDs/hash read; no dataset item loaded", flush=True)
        print("UNIQUE_200 bins: 50/50/50/50; duplicates: 0", flush=True)

        stage = "load frozen DPLM"
        extractor = integration.DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        pilot._TOKENIZER_HOLDER.clear()
        pilot._TOKENIZER_HOLDER.append(tokenizer)
        dplm_versions = pilot.model_versions(dplm)
        tokenizer_versions = pilot.model_versions(tokenizer)

        stage = "stage TRAIN bundles on CPU"
        staging_start = time.perf_counter()
        training_bundles = stage_training_bundles(
            unique, dataset_dir, extractor, tokenizer, device
        )
        training_bundle_staging_seconds = time.perf_counter() - staging_start
        metadata["training_bundle_staging_seconds"] = training_bundle_staging_seconds
        metadata["training_bundles_staged_on_cpu"] = len(training_bundles)
        dplm.net.to("cpu")
        torch.cuda.empty_cache()

        stage = "train REPEAT_16"
        repeat_model, repeat_result = train_arm(
            ARM_REPEAT,
            repeat_schedule,
            training_bundles,
            tokenizer,
            checkpoint_path,
            device,
        )
        repeat_model.to("cpu")
        torch.cuda.empty_cache()
        stage = "train UNIQUE_200"
        unique_model, unique_result = train_arm(
            ARM_UNIQUE,
            unique_schedule,
            training_bundles,
            tokenizer,
            checkpoint_path,
            device,
        )

        stage = "restore DPLM for development evaluation"
        del training_bundles
        gc.collect()
        dplm.net.to(device)
        torch.cuda.empty_cache()
        stage = "load fresh base for development evaluation"
        repeat_model.to(device)
        base_model, base_step, _, _ = integration.load_fm(checkpoint_path, device)
        require(base_step == 100_000, "base evaluation checkpoint step mismatch")
        base_model.requires_grad_(False).eval()
        stage = "evaluate development 256 only"
        evaluation_rows, evaluation_seconds = evaluate_development(
            base_model,
            repeat_model,
            unique_model,
            development,
            dataset_dir,
            extractor,
            tokenizer,
            device,
        )
        require(len(evaluation_rows) == EXPECTED_DEVELOPMENT, "development evaluation incomplete")
        require(
            {str(row["protein_id"]) for row in evaluation_rows} == development_ids,
            "development evaluation IDs differ",
        )
        require(
            not ({str(row["protein_id"]) for row in evaluation_rows} & locked_ids),
            "locked ID appeared in evaluation",
        )

        overall = {
            "base": model_summary(evaluation_rows, "base"),
            "repeat16": model_summary(evaluation_rows, "repeat16"),
            "unique200": model_summary(evaluation_rows, "unique200"),
            "unique200_vs_repeat16": direct_summary(evaluation_rows),
        }
        bins = length_bin_summaries(evaluation_rows)
        outliers = {
            "repeat16_vs_base": outlier_summary(evaluation_rows, "repeat16", "base"),
            "unique200_vs_base": outlier_summary(evaluation_rows, "unique200", "base"),
            "unique200_vs_repeat16": outlier_summary(
                evaluation_rows, "unique200", "repeat16"
            ),
        }
        direct = overall["unique200_vs_repeat16"]
        diversity_supported = bool(
            overall["unique200"]["mean_loss"] < overall["repeat16"]["mean_loss"]
            and overall["unique200"]["median_loss"]
            <= overall["repeat16"]["median_loss"]
            and direct["unique_better"] > direct["repeat_better"]
        )
        unique_summary = overall["unique200"]
        base_summary = overall["base"]
        broad_improvement = bool(
            unique_summary["mean_loss"] < base_summary["mean_loss"]
            and unique_summary["median_loss"] <= base_summary["median_loss"]
            and unique_summary["improved_vs_base"] > unique_summary["worse_vs_base"]
        )
        all_finite = bool(
            repeat_result.finite
            and unique_result.finite
            and all(
                math.isfinite(float(value))
                for row in evaluation_rows
                for key, value in row.items()
                if key not in {"protein_id", "length_bin"}
            )
        )
        frozen_ok = bool(
            repeat_result.frozen_ok
            and unique_result.frozen_ok
            and pilot.versions_unchanged(dplm, dplm_versions)
            and pilot.versions_unchanged(tokenizer, tokenizer_versions)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
        )
        hashes_after = {str(path): pilot.file_digest(path) for path in readonly_paths}
        inputs_unchanged = hashes_before == hashes_after
        require(inputs_unchanged, "read-only input changed")
        require(not locked_evaluated, "locked confirmation was evaluated")
        require(not cameo_evaluated, "CAMEO/PDB-date was evaluated")

        write_per_target(
            output_dir / "per_target_development_results.csv", evaluation_rows
        )
        write_summary_csv(
            output_dir / "diversity_summary.csv", overall, bins
        )
        repeat_diagnostics = diagnostic_payload(repeat_result)
        unique_diagnostics = diagnostic_payload(unique_result)
        integration.synchronize(device)
        evaluation_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        evaluation_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        global_peak_allocated = max(
            repeat_result.peak_allocated_mib,
            unique_result.peak_allocated_mib,
            evaluation_peak_allocated,
        )
        global_peak_reserved = max(
            repeat_result.peak_reserved_mib,
            unique_result.peak_reserved_mib,
            evaluation_peak_reserved,
        )
        total_seconds = time.perf_counter() - total_start
        text = summary_text(
            overall,
            bins,
            repeat_diagnostics,
            unique_diagnostics,
            outliers,
            diversity_supported,
            broad_improvement,
            evaluation_seconds,
            total_seconds,
            global_peak_allocated,
            global_peak_reserved,
        )
        (output_dir / "diversity_summary.txt").write_text(text, encoding="utf-8")
        implementation_pass = all_finite and frozen_ok and inputs_unchanged
        metadata.update(
            {
                "status": "PASS" if implementation_pass else "FAIL",
                "base_development": overall["base"],
                "repeat16_development": overall["repeat16"],
                "unique200_development": overall["unique200"],
                "direct_unique200_vs_repeat16": direct,
                "diversity_hypothesis": (
                    "SUPPORTED" if diversity_supported else "NOT_SUPPORTED"
                ),
                "unique200_broad_dev_improvement": broad_improvement,
                "length_bin_results": bins,
                "outlier_diagnostics": outliers,
                "repeat16_training": repeat_diagnostics,
                "unique200_training": unique_diagnostics,
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
                "checkpoint_selection_performed": False,
                "production_hyperparameters_selected": False,
            }
        )
        write_json(output_dir / "metadata.json", metadata)

        print("\nADJOINT_TRAINING_DIVERSITY", flush=True)
        print(f"development N: {len(evaluation_rows)}", flush=True)
        print(
            f"BASE mean/median: {overall['base']['mean_loss']:.9g} / "
            f"{overall['base']['median_loss']:.9g}",
            flush=True,
        )
        for label, summary, diagnostics in (
            (ARM_REPEAT, overall["repeat16"], repeat_diagnostics),
            (ARM_UNIQUE, overall["unique200"], unique_diagnostics),
        ):
            print(f"ARM {label}:", flush=True)
            print(f"  mean: {summary['mean_loss']:.9g}", flush=True)
            print(f"  median: {summary['median_loss']:.9g}", flush=True)
            print(f"  delta vs base: {summary['mean_delta_vs_base']:.9g}", flush=True)
            print(
                f"  improved/worse/tied: {summary['improved_vs_base']}/"
                f"{summary['worse_vs_base']}/{summary['tied_vs_base']}",
                flush=True,
            )
            print(f"  clip fraction: {diagnostics['clip_fraction']:.9g}", flush=True)
            print(
                f"  AM loss mean/range: {diagnostics['am_loss_mean']:.9g} "
                f"[{diagnostics['am_loss_min']:.9g}, "
                f"{diagnostics['am_loss_max']:.9g}]",
                flush=True,
            )
            print(
                f"  terminal loss mean/range: "
                f"{diagnostics['terminal_loss_mean']:.9g} "
                f"[{diagnostics['terminal_loss_min']:.9g}, "
                f"{diagnostics['terminal_loss_max']:.9g}]",
                flush=True,
            )
            print(
                f"  pre/postclip gradient mean: "
                f"{diagnostics['preclip_grad_mean']:.9g} / "
                f"{diagnostics['postclip_grad_mean']:.9g}",
                flush=True,
            )
            print(
                f"  relative drift: {diagnostics['relative_parameter_drift']:.9g}",
                flush=True,
            )
        print("DIRECT UNIQUE_200 vs REPEAT_16:", flush=True)
        print(
            f"  mean difference repeat-unique: "
            f"{direct['mean_difference_repeat_minus_unique']:.9g}",
            flush=True,
        )
        print(
            f"  median difference repeat-unique: "
            f"{direct['median_difference_repeat_minus_unique']:.9g}",
            flush=True,
        )
        print(
            f"  unique better/repeat better/tied: {direct['unique_better']}/"
            f"{direct['repeat_better']}/{direct['tied']}",
            flush=True,
        )
        print(
            "DIVERSITY_HYPOTHESIS: "
            + ("SUPPORTED" if diversity_supported else "NOT_SUPPORTED"),
            flush=True,
        )
        print(
            (
                "DIVERSITY_HYPOTHESIS_SUPPORTED"
                if diversity_supported
                else "DIVERSITY_HYPOTHESIS_NOT_SUPPORTED"
            ),
            flush=True,
        )
        print(
            "UNIQUE_200_BROAD_DEV_IMPROVEMENT: "
            + ("YES" if broad_improvement else "NO"),
            flush=True,
        )
        print("Length-bin results:", flush=True)
        for label, records in bins.items():
            d = records["unique200_vs_repeat16"]
            print(
                f"  {label}: repeat_mean={records['repeat16']['mean_loss']:.9g} "
                f"unique_mean={records['unique200']['mean_loss']:.9g} "
                f"repeat-unique={d['mean_difference_repeat_minus_unique']:.9g} "
                f"counts={d['unique_better']}/{d['repeat_better']}/{d['tied']}",
                flush=True,
            )
        print("Outlier diagnostics: recorded", flush=True)
        print("LOCKED_CONFIRMATION_EVALUATED: NO", flush=True)
        print("CAMEO_PDB_DATE_EVALUATED: NO", flush=True)
        print(f"NaN/Inf: {'PASS' if all_finite else 'FAIL'}", flush=True)
        marker = (
            "ADJOINT_TRAINING_DIVERSITY_PASS"
            if implementation_pass
            else "ADJOINT_TRAINING_DIVERSITY_FAIL"
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
                "production_hyperparameters_selected": False,
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
        print("overall: ADJOINT_TRAINING_DIVERSITY_FAIL", flush=True)
        print("ADJOINT_TRAINING_DIVERSITY_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

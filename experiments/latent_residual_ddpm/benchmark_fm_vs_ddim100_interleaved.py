#!/usr/bin/env python3
"""Same-session interleaved residual latency benchmark for FM-100/DDIM-100."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from experiments.latent_residual_ddpm.benchmark_fm_vs_ddpm_latency import (
    BATCH_SIZE,
    DATASET_DIR,
    DPLM_CHECKPOINT,
    FM_CHECKPOINT,
    DDPM_CHECKPOINT,
    LENGTH_BINS,
    NUM_TARGETS,
    TARGETS_CSV,
    load_samples,
    make_noise,
    prepare_condition,
    read_targets,
    timed_cuda,
    validate_shared_inputs,
)
from experiments.latent_residual_ddpm.debug_inference_integration import (
    all_grads_none,
    load_ddpm,
)
from experiments.latent_residual_ddpm.evaluate_ddpm_100nfe import (
    DDIM_NFE,
    ddim_timesteps,
    sample_residual_ddim100,
)
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import load_fm
from experiments.latent_residual_fm.flow_matching import euler_sample


OUTPUT_ROOT = Path(
    "generation-results/dplm2_bit_650m_final/ddpm_100nfe/latency_interleaved"
)
RAW_CSV = OUTPUT_ROOT / "raw_repeats.csv"
PER_TARGET_CSV = OUTPUT_ROOT / "per_target.csv"
BY_BIN_CSV = OUTPUT_ROOT / "by_length_bin.csv"
SUMMARY_TXT = OUTPUT_ROOT / "summary.txt"
METADATA_JSON = OUTPUT_ROOT / "metadata.json"

FM_NFE = 100
REPEATS = 5
BASE_SEED = 50_000

RAW_COLUMNS = (
    "protein_id", "validation_index", "length", "length_bin", "repeat",
    "seed", "method_order", "fm_ms", "fm_gpu_ms", "ddim100_ms",
    "ddim100_gpu_ms", "fm_finite", "ddim100_finite",
)
PER_TARGET_COLUMNS = (
    "protein_id", "validation_index", "length", "length_bin", "seed",
    "method_order", "fm_median_ms", "ddim100_median_ms",
    "ddim100_over_fm_ratio", "ddim100_minus_fm_ms", "fm_ms_per_nfe",
    "ddim100_ms_per_nfe", "fm_residues_per_second",
    "ddim100_residues_per_second",
)
BIN_COLUMNS = (
    "length_bin", "num_targets", "fm_median_ms", "ddim100_median_ms",
    "ddim100_over_fm_median_latency_ratio",
    "median_target_wise_ddim100_over_fm_ratio",
    "median_paired_difference_ms", "fm_median_residues_per_second",
    "ddim100_median_residues_per_second",
)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_fm(
    fm: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden: Sequence[torch.Tensor],
    mask: torch.Tensor,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    return euler_sample(
        fm, z_quant, hidden, mask,
        num_steps=FM_NFE, initial_noise=initial_noise,
    )


def run_ddim(
    ddpm: torch.nn.Module,
    schedule: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden: Sequence[torch.Tensor],
    mask: torch.Tensor,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    output, metadata = sample_residual_ddim100(
        ddpm, schedule, z_quant, hidden, mask,
        initial_noise=initial_noise, return_metadata=True,
    )
    require(metadata["nfe"] == DDIM_NFE == 100, "DDIM NFE mismatch")
    require(metadata["timesteps"] == ddim_timesteps(), "DDIM grid mismatch")
    return output


def validate_output(
    output: torch.Tensor, mask: torch.Tensor, length: int, label: str,
) -> None:
    require(tuple(output.shape) == (BATCH_SIZE, length, EXPECTED_CODEBOOK_DIM),
            f"{label} output shape mismatch")
    require(bool(torch.isfinite(output).all()), f"{label} output non-finite")
    padding = ~mask.bool()
    if bool(padding.any()):
        require(torch.equal(output[padding], torch.zeros_like(output[padding])),
                f"{label} padding is nonzero")


def summarize_targets(
    targets: Sequence[Mapping[str, Any]], raw: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in targets:
        rows = [row for row in raw if row["protein_id"] == target["protein_id"]]
        require(len(rows) == REPEATS, "target repeat count mismatch")
        fm_ms = statistics.median(float(row["fm_ms"]) for row in rows)
        ddim_ms = statistics.median(float(row["ddim100_ms"]) for row in rows)
        length = int(target["length"])
        result.append({
            "protein_id": target["protein_id"],
            "validation_index": target["validation_index"],
            "length": length, "length_bin": target["length_bin"],
            "seed": BASE_SEED + int(target["validation_index"]),
            "method_order": rows[0]["method_order"],
            "fm_median_ms": fm_ms, "ddim100_median_ms": ddim_ms,
            "ddim100_over_fm_ratio": ddim_ms / fm_ms,
            "ddim100_minus_fm_ms": ddim_ms - fm_ms,
            "fm_ms_per_nfe": fm_ms / FM_NFE,
            "ddim100_ms_per_nfe": ddim_ms / DDIM_NFE,
            "fm_residues_per_second": length / (fm_ms / 1000.0),
            "ddim100_residues_per_second": length / (ddim_ms / 1000.0),
        })
    require(len(result) == NUM_TARGETS, "per-target count mismatch")
    return result


def summarize_bins(per_target: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for length_bin in LENGTH_BINS:
        rows = [row for row in per_target if row["length_bin"] == length_bin]
        require(len(rows) == 4, f"{length_bin} count mismatch")
        fm = [float(row["fm_median_ms"]) for row in rows]
        ddim = [float(row["ddim100_median_ms"]) for row in rows]
        result.append({
            "length_bin": length_bin, "num_targets": len(rows),
            "fm_median_ms": statistics.median(fm),
            "ddim100_median_ms": statistics.median(ddim),
            "ddim100_over_fm_median_latency_ratio": (
                statistics.median(ddim) / statistics.median(fm)
            ),
            "median_target_wise_ddim100_over_fm_ratio": statistics.median(
                float(row["ddim100_over_fm_ratio"]) for row in rows
            ),
            "median_paired_difference_ms": statistics.median(
                float(row["ddim100_minus_fm_ms"]) for row in rows
            ),
            "fm_median_residues_per_second": statistics.median(
                float(row["fm_residues_per_second"]) for row in rows
            ),
            "ddim100_median_residues_per_second": statistics.median(
                float(row["ddim100_residues_per_second"]) for row in rows
            ),
        })
    return result


def descriptive(values: Sequence[float]) -> dict[str, float]:
    require(len(values) == NUM_TARGETS and all(math.isfinite(x) for x in values),
            "aggregate values invalid")
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "std": statistics.pstdev(values), "min": min(values), "max": max(values),
    }


def summary_text(
    per_target: Sequence[Mapping[str, Any]],
    bins: Sequence[Mapping[str, Any]],
    gpu: str,
    dtype: torch.dtype,
    runtime_seconds: float,
) -> str:
    fm_values = [float(row["fm_median_ms"]) for row in per_target]
    ddim_values = [float(row["ddim100_median_ms"]) for row in per_target]
    differences = [float(row["ddim100_minus_fm_ms"]) for row in per_target]
    ratios = [float(row["ddim100_over_fm_ratio"]) for row in per_target]
    fm = descriptive(fm_values)
    ddim = descriptive(ddim_values)
    lines = [
        "FM_VS_DDIM100_INTERLEAVED_LATENCY",
        f"GPU: {gpu}", f"dtype: {dtype}", f"targets: {NUM_TARGETS}",
        f"repeats: {REPEATS}", f"batch size: {BATCH_SIZE}",
        f"FM NFE: {FM_NFE}", f"DDIM NFE: {DDIM_NFE}", "",
        "[FM-100]",
        f"mean ms: {fm['mean']:.9f}", f"median ms: {fm['median']:.9f}",
        f"std/min/max ms: {fm['std']:.9f} / {fm['min']:.9f} / {fm['max']:.9f}",
        f"median ms/NFE: {fm['median'] / FM_NFE:.9f}",
        f"median residues/s: {statistics.median(float(row['fm_residues_per_second']) for row in per_target):.9f}",
        "", "[Matched DDPM (DDIM-100)]",
        f"mean ms: {ddim['mean']:.9f}", f"median ms: {ddim['median']:.9f}",
        f"std/min/max ms: {ddim['std']:.9f} / {ddim['min']:.9f} / {ddim['max']:.9f}",
        f"median ms/NFE: {ddim['median'] / DDIM_NFE:.9f}",
        f"median residues/s: {statistics.median(float(row['ddim100_residues_per_second']) for row in per_target):.9f}",
        "",
        f"median target-wise DDIM/FM ratio: {statistics.median(ratios):.9f}",
        f"aggregate median ratio: {ddim['median'] / fm['median']:.9f}",
        f"median paired difference ms: {statistics.median(differences):.9f}",
        "", "[length bins]",
    ]
    for row in bins:
        lines.append(
            f"{row['length_bin']} N={row['num_targets']}: FM={float(row['fm_median_ms']):.6f} ms, "
            f"DDIM={float(row['ddim100_median_ms']):.6f} ms, "
            f"ratio={float(row['ddim100_over_fm_median_latency_ratio']):.6f}, "
            f"residues/s FM/DDIM={float(row['fm_median_residues_per_second']):.6f}/"
            f"{float(row['ddim100_median_residues_per_second']):.6f}"
        )
    lines.extend([
        "", "same-session: YES", "interleaved: YES", "NaN/Inf: PASS",
        f"runtime seconds: {runtime_seconds:.6f}",
        "overall: FM_VS_DDIM100_INTERLEAVED_LATENCY_PASS", "",
    ])
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    require(torch.cuda.is_available(), "CUDA is required")
    require(FM_NFE == DDIM_NFE == 100 and REPEATS == 5 and BATCH_SIZE == 1,
            "fixed timing protocol changed")
    device = resolve_device("cuda")
    seed_everything(BASE_SEED)
    targets = read_targets()
    require(TARGETS_CSV.is_file() and len(targets) == NUM_TARGETS == 16,
            "frozen latency target roster mismatch")
    samples = load_samples(targets)

    checks: dict[str, bool] = {}
    dplm, _tokenizer = load_models(DPLM_CHECKPOINT, device, checks)
    fm, _, _, _ = load_fm(FM_CHECKPOINT, device)
    ddpm, schedule, _ = load_ddpm(DDPM_CHECKPOINT, device)
    require(not dplm.training and not fm.training and not ddpm.training,
            "model eval mode mismatch")
    require(all(not parameter.requires_grad for module in (dplm, fm, ddpm)
                for parameter in module.parameters()), "trainable parameter found")
    require(next(fm.parameters()).dtype == next(ddpm.parameters()).dtype == torch.float32,
            "residual dtype is not torch.float32")
    require(sum(parameter.numel() for parameter in fm.parameters()) ==
            sum(parameter.numel() for parameter in ddpm.parameters()),
            "residual architecture parameter count mismatch")

    raw: list[dict[str, Any]] = []
    warmed_bins: set[str] = set()
    with torch.inference_mode():
        for target_index, target in enumerate(targets):
            protein_id = str(target["protein_id"])
            length = int(target["length"])
            length_bin = str(target["length_bin"])
            validation_index = int(target["validation_index"])
            _, z_quant, mask, hidden, _, _ = prepare_condition(
                dplm, samples[protein_id], length, device
            )
            validate_shared_inputs(z_quant, hidden, mask, length)
            seed = BASE_SEED + validation_index
            initial_noise = make_noise(z_quant, seed)
            fm_first = target_index % 2 == 0
            method_order = "FM->DDIM" if fm_first else "DDIM->FM"

            if length_bin not in warmed_bins:
                warm_methods = ("fm", "ddim") if fm_first else ("ddim", "fm")
                for method in warm_methods:
                    if method == "fm":
                        warm = run_fm(fm, z_quant, hidden, mask, initial_noise.clone())
                    else:
                        warm = run_ddim(
                            ddpm, schedule, z_quant, hidden, mask,
                            initial_noise.clone(),
                        )
                    validate_output(warm, mask, length, f"{method} warmup")
                torch.cuda.synchronize(device)
                warmed_bins.add(length_bin)

            fm_reference: torch.Tensor | None = None
            ddim_reference: torch.Tensor | None = None
            for repeat in range(REPEATS):
                timings: dict[str, tuple[torch.Tensor, float, float]] = {}
                methods = ("fm", "ddim") if fm_first else ("ddim", "fm")
                for method in methods:
                    if method == "fm":
                        timings[method] = timed_cuda(
                            device,
                            lambda: run_fm(
                                fm, z_quant, hidden, mask, initial_noise.clone()
                            ),
                        )
                    else:
                        timings[method] = timed_cuda(
                            device,
                            lambda: run_ddim(
                                ddpm, schedule, z_quant, hidden, mask,
                                initial_noise.clone(),
                            ),
                        )
                fm_output, fm_ms, fm_gpu_ms = timings["fm"]
                ddim_output, ddim_ms, ddim_gpu_ms = timings["ddim"]
                validate_output(fm_output, mask, length, "FM")
                validate_output(ddim_output, mask, length, "DDIM")
                if fm_reference is None:
                    fm_reference = fm_output.detach().clone()
                    ddim_reference = ddim_output.detach().clone()
                else:
                    require(torch.equal(fm_reference, fm_output),
                            "FM repeat output changed")
                    require(torch.equal(ddim_reference, ddim_output),
                            "DDIM repeat output changed")
                raw.append({
                    "protein_id": protein_id, "validation_index": validation_index,
                    "length": length, "length_bin": length_bin,
                    "repeat": repeat + 1, "seed": seed,
                    "method_order": method_order, "fm_ms": fm_ms,
                    "fm_gpu_ms": fm_gpu_ms, "ddim100_ms": ddim_ms,
                    "ddim100_gpu_ms": ddim_gpu_ms,
                    "fm_finite": True, "ddim100_finite": True,
                })
            atomic_write_csv(RAW_CSV, RAW_COLUMNS, raw)
            print(
                f"[{target_index + 1}/{NUM_TARGETS}] {protein_id} L={length} "
                f"order={method_order} completed", flush=True,
            )

    require(len(raw) == NUM_TARGETS * REPEATS == 80, "raw row count mismatch")
    require(warmed_bins == set(LENGTH_BINS), "length-bin warmup coverage mismatch")
    require(all_grads_none(dplm) and all_grads_none(fm) and all_grads_none(ddpm),
            "timing accumulated gradients")
    per_target = summarize_targets(targets, raw)
    bins = summarize_bins(per_target)
    runtime_seconds = time.perf_counter() - started
    summary = summary_text(
        per_target, bins, torch.cuda.get_device_name(device),
        next(fm.parameters()).dtype, runtime_seconds,
    )
    atomic_write_csv(PER_TARGET_CSV, PER_TARGET_COLUMNS, per_target)
    atomic_write_csv(BY_BIN_CSV, BIN_COLUMNS, bins)
    atomic_write_text(SUMMARY_TXT, summary)
    metadata = {
        "status": "FM_VS_DDIM100_INTERLEAVED_LATENCY_PASS",
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(next(fm.parameters()).dtype), "batch_size": BATCH_SIZE,
        "targets": NUM_TARGETS, "repeats": REPEATS,
        "fm_nfe": FM_NFE, "ddim100_nfe": DDIM_NFE,
        "same_process": True, "interleaved_by_target": True,
        "method_order": "even target FM->DDIM; odd target DDIM->FM",
        "warmup": "one untimed call per method on first target of each length bin",
        "timing": "CUDA synchronize + time.perf_counter; CUDA events diagnostic",
        "timed_scope": "residual sampler only",
        "excluded": [
            "model loading", "dataset disk IO", "DPLM Bit generation",
            "extra hidden-state forward", "decoder",
        ],
        "same_initial_noise_per_target_and_method": True,
        "target_seed": "50000 + validation_index",
        "target_csv": str(TARGETS_CSV),
        "target_csv_sha256": sha256_file(TARGETS_CSV),
        "dataset_root": str(DATASET_DIR),
        "fm_checkpoint": str(FM_CHECKPOINT),
        "fm_checkpoint_sha256": sha256_file(FM_CHECKPOINT),
        "ddpm_checkpoint": str(DDPM_CHECKPOINT),
        "ddpm_checkpoint_sha256": sha256_file(DDPM_CHECKPOINT),
        "quality_metrics_computed": False,
        "CAMEO_PDB_date_inference_performed": False,
        "training_performed": False, "alpha_tuning_performed": False,
        "finite_outputs": True, "no_gradients": True,
        "runtime_seconds": runtime_seconds,
    }
    atomic_write_text(METADATA_JSON, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("overall: FM_VS_DDIM100_INTERLEAVED_LATENCY_FAIL", flush=True)
        raise

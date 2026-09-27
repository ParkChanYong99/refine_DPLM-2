#!/usr/bin/env python3
"""Benchmark finalized FM versus Matched DDPM synchronized inference latency."""

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
from typing import Any, Callable, Mapping, Sequence

import torch

from experiments.latent_residual_ddpm.debug_inference_integration import (
    NUM_TIMESTEPS,
    all_grads_none,
    load_ddpm,
)
from experiments.latent_residual_ddpm.ddpm_sampling import sample_residual_ddpm
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
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
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.flow_matching import euler_sample


DATASET_DIR = Path("data-bin/latent_residual_fm/afdb_l512_sharded")
SELECTION_CSV = Path(
    "experiments/latent_residual_fm/runs/am_50step_replication/"
    "validation_indices_64.csv"
)
FM_CHECKPOINT = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt"
)
DDPM_CHECKPOINT = Path(
    "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main/checkpoint_best.pt"
)
DPLM_CHECKPOINT = "airkingbd/dplm2_bit_650m"
OUTPUT_ROOT = Path(
    "generation-results/dplm2_bit_650m_final/latency_benchmark"
)
TARGETS_CSV = OUTPUT_ROOT / "latency_targets.csv"
RAW_CSV = OUTPUT_ROOT / "latency_raw_repeats.csv"
PER_TARGET_CSV = OUTPUT_ROOT / "latency_per_target.csv"
BY_BIN_CSV = OUTPUT_ROOT / "latency_by_length_bin.csv"
SUMMARY_TXT = OUTPUT_ROOT / "latency_summary.txt"
METADATA_JSON = OUTPUT_ROOT / "metadata.json"

LENGTH_BINS = ("50-128", "129-256", "257-384", "385-512")
TARGETS_PER_BIN = 4
NUM_TARGETS = 16
REPEATS = 3
FM_NFE = 100
DDPM_NFE = 1000
ALPHA = 0.50
BASE_SEED = 50_000
BATCH_SIZE = 1

TARGET_COLUMNS = (
    "selection_order", "protein_id", "validation_index", "length", "length_bin",
    "shard_file", "index_in_shard",
)
RAW_COLUMNS = (
    "protein_id", "validation_index", "length", "length_bin", "repeat",
    "seed", "method_order", "T_bit_ms", "T_bit_gpu_ms",
    "T_extra_forward_ms", "T_extra_forward_gpu_ms",
    "T_fm_residual_ms", "T_fm_residual_gpu_ms",
    "T_ddpm_residual_ms", "T_ddpm_residual_gpu_ms",
    "T_fm_decode_ms", "T_fm_decode_gpu_ms",
    "T_ddpm_decode_ms", "T_ddpm_decode_gpu_ms",
    "fm_peak_allocated_mib", "fm_peak_reserved_mib",
    "ddpm_peak_allocated_mib", "ddpm_peak_reserved_mib",
    "fm_activation_peak_delta_mib", "ddpm_activation_peak_delta_mib",
)
PER_TARGET_COLUMNS = (
    "protein_id", "validation_index", "length", "length_bin",
    "T_bit_ms", "T_extra_forward_ms", "T_fm_residual_ms",
    "T_ddpm_residual_ms", "T_fm_decode_ms", "T_ddpm_decode_ms",
    "T_postbit_fm_ms", "T_postbit_ddpm_ms", "T_e2e_fm_ms", "T_e2e_ddpm_ms",
    "residual_speedup", "postbit_speedup", "end_to_end_speedup",
    "fm_ms_per_nfe", "ddpm_ms_per_nfe", "ms_per_nfe_ratio_ddpm_over_fm",
    "fm_residual_residues_per_sec", "ddpm_residual_residues_per_sec",
    "fm_end_to_end_residues_per_sec", "ddpm_end_to_end_residues_per_sec",
)
BIN_COLUMNS = (
    "length_bin", "num_targets", "fm_residual_median_ms",
    "ddpm_residual_median_ms", "residual_speedup_median_latency_ratio",
    "median_target_wise_residual_speedup", "fm_end_to_end_median_ms",
    "ddpm_end_to_end_median_ms", "end_to_end_speedup_median_latency_ratio",
    "median_target_wise_end_to_end_speedup",
    "fm_residual_median_residues_per_sec",
    "ddpm_residual_median_residues_per_sec",
    "fm_end_to_end_median_residues_per_sec",
    "ddpm_end_to_end_median_residues_per_sec",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
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


def read_targets() -> list[dict[str, Any]]:
    require(SELECTION_CSV.is_file(), f"selection CSV missing: {SELECTION_CSV}")
    with SELECTION_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "split_index", "sample_id", "length", "length_bin",
            "shard_file", "index_in_shard",
        }
        require(required.issubset(reader.fieldnames or ()), "selection CSV schema mismatch")
        rows = list(reader)
    selected: list[dict[str, Any]] = []
    for length_bin in LENGTH_BINS:
        candidates = [row for row in rows if row["length_bin"] == length_bin]
        require(
            len(candidates) >= TARGETS_PER_BIN,
            f"{length_bin} has fewer than {TARGETS_PER_BIN} candidates",
        )
        for row in candidates[:TARGETS_PER_BIN]:
            length = int(row["length"])
            validation_index = int(row["split_index"])
            require(row["split"] == "val", "selected row is not validation")
            selected.append(
                {
                    "selection_order": len(selected),
                    "protein_id": row["sample_id"],
                    "validation_index": validation_index,
                    "length": length,
                    "length_bin": length_bin,
                    "shard_file": row["shard_file"],
                    "index_in_shard": int(row["index_in_shard"]),
                }
            )
    require(len(selected) == NUM_TARGETS, "target count mismatch")
    require(
        len({row["protein_id"] for row in selected}) == NUM_TARGETS,
        "duplicate target ID",
    )
    require(
        all(sum(row["length_bin"] == b for row in selected) == TARGETS_PER_BIN
            for b in LENGTH_BINS),
        "length-bin allocation mismatch",
    )
    return selected


def load_samples(targets: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for target in targets:
        by_shard.setdefault(str(target["shard_file"]), []).append(target)
    for shard_name, shard_targets in by_shard.items():
        path = DATASET_DIR / shard_name
        require(path.is_file(), f"shard missing: {path}")
        shard = torch.load(str(path), map_location="cpu", mmap=True)
        require(isinstance(shard, dict) and "samples" in shard, f"bad shard: {path}")
        require(int(shard["num_samples"]) == len(shard["samples"]), "bad shard count")
        for target in shard_targets:
            sample = shard["samples"][int(target["index_in_shard"])]
            require(sample["sample_id"] == target["protein_id"], "sample ID mismatch")
            require(int(sample["length"]) == target["length"], "sample length mismatch")
            require(tuple(sample["aatype"].shape) == (target["length"],), "aatype shape")
            samples[str(target["protein_id"])] = sample
    require(len(samples) == NUM_TARGETS, "loaded sample count mismatch")
    return samples


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed_cuda(
    device: torch.device, function: Callable[[], Any]
) -> tuple[Any, float, float]:
    synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    result = function()
    end_event.record()
    synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    gpu_ms = float(start_event.elapsed_time(end_event))
    require(math.isfinite(wall_ms) and wall_ms > 0.0, "invalid wall latency")
    require(math.isfinite(gpu_ms) and gpu_ms > 0.0, "invalid GPU latency")
    return result, wall_ms, gpu_ms


def make_noise(
    reference: torch.Tensor, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def run_fm(
    fm: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    return euler_sample(
        fm, z_quant, hidden_states, res_mask,
        num_steps=FM_NFE, initial_noise=initial_noise,
    )


def run_ddpm(
    ddpm: torch.nn.Module,
    schedule: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    initial_noise: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    reverse_generator = torch.Generator(device=z_quant.device)
    reverse_generator.manual_seed(seed + 1)
    result, metadata = sample_residual_ddpm(
        model=ddpm,
        schedule=schedule,
        z_quant=z_quant,
        hidden_states=hidden_states,
        res_mask=res_mask,
        initial_noise=initial_noise,
        generator=reverse_generator,
        return_metadata=True,
    )
    require(metadata["nfe"] == DDPM_NFE, "Matched DDPM NFE mismatch")
    require(
        metadata["timesteps"] == tuple(range(DDPM_NFE, 0, -1)),
        "Matched DDPM timestep order mismatch",
    )
    return result


def decode(
    tokenizer: torch.nn.Module,
    z_quant: torch.Tensor,
    residual: torch.Tensor,
    res_mask: torch.Tensor,
    length: int,
    label: str,
) -> Mapping[str, torch.Tensor]:
    z_refined = z_quant + ALPHA * residual
    require(bool(torch.isfinite(z_refined).all()), f"{label} latent non-finite")
    decoded = tokenizer.detokenize(z_refined, res_mask=res_mask)
    assert_atom37(label, decoded, length)
    return decoded


def validate_shared_inputs(
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    length: int,
) -> None:
    require(tuple(z_quant.shape) == (BATCH_SIZE, length, EXPECTED_CODEBOOK_DIM), "z_quant")
    require(tuple(res_mask.shape) == (BATCH_SIZE, length), "res_mask")
    require(len(hidden_states) == EXPECTED_NUM_LAYERS == 33, "hidden-state count")
    require(
        all(tuple(state.shape) == (BATCH_SIZE, length, EXPECTED_HIDDEN_SIZE)
            for state in hidden_states),
        "hidden-state alignment",
    )
    tensors = (z_quant, res_mask, *hidden_states)
    require(all(bool(torch.isfinite(value).all()) for value in tensors), "non-finite condition")
    require(
        all(value.device == z_quant.device for value in tensors),
        "condition device mismatch",
    )


def prepare_condition(
    dplm: torch.nn.Module,
    sample: Mapping[str, Any],
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], float, float]:
    aatype = sample["aatype"].long()
    input_tokens, partial_mask = build_folding_input(dplm, aatype, device)
    generated, bit_ms, bit_gpu_ms = timed_cuda(
        device,
        lambda: dplm.generate(
            input_tokens=input_tokens,
            max_iter=GENERATION_MAX_ITER,
            temperature=1.0,
            unmasking_strategy="deterministic",
            sampling_strategy="argmax",
            partial_masks=partial_mask,
        ),
    )
    require("output_tokens" in generated, "generation returned no output_tokens")
    final_tokens = generated["output_tokens"]
    require(
        tuple(final_tokens.shape) == (BATCH_SIZE, 2 * (length + 2)),
        "final token shape mismatch",
    )
    z_quant, res_mask = reconstruct_prediction_latent(dplm, final_tokens)
    hidden_states, extra_ms, extra_gpu_ms = timed_cuda(
        device,
        lambda: extract_final_token_condition(dplm, final_tokens, length),
    )
    validate_shared_inputs(z_quant, hidden_states, res_mask, length)
    require(bit_gpu_ms > 0.0 and extra_gpu_ms > 0.0, "invalid component GPU time")
    return final_tokens, z_quant, res_mask, hidden_states, bit_ms, extra_ms


def warm_up_bin(
    *,
    bin_index: int,
    target: Mapping[str, Any],
    sample: Mapping[str, Any],
    dplm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    fm: torch.nn.Module,
    ddpm: torch.nn.Module,
    schedule: torch.nn.Module,
    device: torch.device,
) -> None:
    length = int(target["length"])
    print(f"[WARM-UP] bin={target['length_bin']} target={target['protein_id']}", flush=True)
    _, z_quant, res_mask, hidden_states, _, _ = prepare_condition(
        dplm, sample, length, device
    )
    seed = BASE_SEED + int(target["validation_index"])
    initial_noise = make_noise(z_quant, seed)
    order = ("fm", "ddpm") if bin_index % 2 == 0 else ("ddpm", "fm")
    for method in order:
        if method == "fm":
            residual = run_fm(fm, z_quant, hidden_states, res_mask, initial_noise)
        else:
            residual = run_ddpm(
                ddpm, schedule, z_quant, hidden_states, res_mask, initial_noise, seed
            )
        decoded = decode(tokenizer, z_quant, residual, res_mask, length, f"warmup-{method}")
        del residual, decoded
    synchronize(device)
    del z_quant, res_mask, hidden_states, initial_noise


def benchmark_target(
    *,
    bin_index: int,
    target: Mapping[str, Any],
    sample: Mapping[str, Any],
    dplm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    fm: torch.nn.Module,
    ddpm: torch.nn.Module,
    schedule: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = int(target["length"])
    validation_index = int(target["validation_index"])
    seed = BASE_SEED + validation_index
    order = ("fm", "ddpm") if bin_index % 2 == 0 else ("ddpm", "fm")
    for repeat in range(REPEATS):
        _, z_quant, res_mask, hidden_states, bit_ms, extra_ms = prepare_condition(
            dplm, sample, length, device
        )
        initial_noise = make_noise(z_quant, seed)
        values: dict[str, Any] = {
            "protein_id": target["protein_id"],
            "validation_index": validation_index,
            "length": length,
            "length_bin": target["length_bin"],
            "repeat": repeat,
            "seed": seed,
            "method_order": "->".join(order),
            "T_bit_ms": bit_ms,
            "T_extra_forward_ms": extra_ms,
        }
        # Record GPU-event measurements independently from primary synchronized wall time.
        # Bit/extra helpers already validate events; repeat them in explicit timed calls is
        # deliberately avoided because those components must execute once per repeat.
        values["T_bit_gpu_ms"] = ""
        values["T_extra_forward_gpu_ms"] = ""
        residuals: dict[str, torch.Tensor] = {}
        for method in order:
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            if method == "fm":
                residual, wall_ms, gpu_ms = timed_cuda(
                    device,
                    lambda: run_fm(
                        fm, z_quant, hidden_states, res_mask, initial_noise
                    ),
                )
                values["T_fm_residual_ms"] = wall_ms
                values["T_fm_residual_gpu_ms"] = gpu_ms
                residuals["fm"] = residual
            else:
                residual, wall_ms, gpu_ms = timed_cuda(
                    device,
                    lambda: run_ddpm(
                        ddpm, schedule, z_quant, hidden_states, res_mask,
                        initial_noise, seed,
                    ),
                )
                values["T_ddpm_residual_ms"] = wall_ms
                values["T_ddpm_residual_gpu_ms"] = gpu_ms
                residuals["ddpm"] = residual
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            values[f"{method}_peak_allocated_mib"] = peak_allocated / (1024 ** 2)
            values[f"{method}_peak_reserved_mib"] = peak_reserved / (1024 ** 2)
            values[f"{method}_activation_peak_delta_mib"] = (
                peak_allocated - baseline_allocated
            ) / (1024 ** 2)
        for method in order:
            decoded, wall_ms, gpu_ms = timed_cuda(
                device,
                lambda m=method: decode(
                    tokenizer, z_quant, residuals[m], res_mask, length,
                    f"{m}-refined",
                ),
            )
            values[f"T_{method}_decode_ms"] = wall_ms
            values[f"T_{method}_decode_gpu_ms"] = gpu_ms
            del decoded
        require(
            all(bool(torch.isfinite(value).all()) for value in residuals.values()),
            "residual NaN/Inf",
        )
        rows.append(values)
        del z_quant, res_mask, hidden_states, initial_noise, residuals
        print(
            f"[TIMED] {target['protein_id']} L={length} repeat={repeat + 1}/{REPEATS} "
            f"FM={values['T_fm_residual_ms']:.3f} ms "
            f"Matched-DDPM={values['T_ddpm_residual_ms']:.3f} ms",
            flush=True,
        )
    return rows


def median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    require(all(math.isfinite(value) and value > 0 for value in values), f"bad {key}")
    return statistics.median(values)


def summarize_targets(
    targets: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in targets:
        rows = [row for row in raw_rows if row["protein_id"] == target["protein_id"]]
        require(len(rows) == REPEATS, "repeat count mismatch")
        length = int(target["length"])
        item: dict[str, Any] = {
            "protein_id": target["protein_id"],
            "validation_index": target["validation_index"],
            "length": length,
            "length_bin": target["length_bin"],
        }
        for key in (
            "T_bit_ms", "T_extra_forward_ms", "T_fm_residual_ms",
            "T_ddpm_residual_ms", "T_fm_decode_ms", "T_ddpm_decode_ms",
        ):
            item[key] = median(rows, key)
        item["T_postbit_fm_ms"] = (
            item["T_extra_forward_ms"] + item["T_fm_residual_ms"]
            + item["T_fm_decode_ms"]
        )
        item["T_postbit_ddpm_ms"] = (
            item["T_extra_forward_ms"] + item["T_ddpm_residual_ms"]
            + item["T_ddpm_decode_ms"]
        )
        item["T_e2e_fm_ms"] = item["T_bit_ms"] + item["T_postbit_fm_ms"]
        item["T_e2e_ddpm_ms"] = item["T_bit_ms"] + item["T_postbit_ddpm_ms"]
        item["residual_speedup"] = item["T_ddpm_residual_ms"] / item["T_fm_residual_ms"]
        item["postbit_speedup"] = item["T_postbit_ddpm_ms"] / item["T_postbit_fm_ms"]
        item["end_to_end_speedup"] = item["T_e2e_ddpm_ms"] / item["T_e2e_fm_ms"]
        item["fm_ms_per_nfe"] = item["T_fm_residual_ms"] / FM_NFE
        item["ddpm_ms_per_nfe"] = item["T_ddpm_residual_ms"] / DDPM_NFE
        item["ms_per_nfe_ratio_ddpm_over_fm"] = (
            item["ddpm_ms_per_nfe"] / item["fm_ms_per_nfe"]
        )
        item["fm_residual_residues_per_sec"] = length * 1000 / item["T_fm_residual_ms"]
        item["ddpm_residual_residues_per_sec"] = (
            length * 1000 / item["T_ddpm_residual_ms"]
        )
        item["fm_end_to_end_residues_per_sec"] = length * 1000 / item["T_e2e_fm_ms"]
        item["ddpm_end_to_end_residues_per_sec"] = (
            length * 1000 / item["T_e2e_ddpm_ms"]
        )
        result.append(item)
    return result


def summarize_bins(per_target: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_bin in LENGTH_BINS:
        items = [row for row in per_target if row["length_bin"] == length_bin]
        require(len(items) == TARGETS_PER_BIN, "bin target count mismatch")
        fm_res = statistics.median(float(x["T_fm_residual_ms"]) for x in items)
        ddpm_res = statistics.median(float(x["T_ddpm_residual_ms"]) for x in items)
        fm_e2e = statistics.median(float(x["T_e2e_fm_ms"]) for x in items)
        ddpm_e2e = statistics.median(float(x["T_e2e_ddpm_ms"]) for x in items)
        rows.append(
            {
                "length_bin": length_bin,
                "num_targets": len(items),
                "fm_residual_median_ms": fm_res,
                "ddpm_residual_median_ms": ddpm_res,
                "residual_speedup_median_latency_ratio": ddpm_res / fm_res,
                "median_target_wise_residual_speedup": statistics.median(
                    float(x["residual_speedup"]) for x in items
                ),
                "fm_end_to_end_median_ms": fm_e2e,
                "ddpm_end_to_end_median_ms": ddpm_e2e,
                "end_to_end_speedup_median_latency_ratio": ddpm_e2e / fm_e2e,
                "median_target_wise_end_to_end_speedup": statistics.median(
                    float(x["end_to_end_speedup"]) for x in items
                ),
                "fm_residual_median_residues_per_sec": statistics.median(
                    float(x["fm_residual_residues_per_sec"]) for x in items
                ),
                "ddpm_residual_median_residues_per_sec": statistics.median(
                    float(x["ddpm_residual_residues_per_sec"]) for x in items
                ),
                "fm_end_to_end_median_residues_per_sec": statistics.median(
                    float(x["fm_end_to_end_residues_per_sec"]) for x in items
                ),
                "ddpm_end_to_end_median_residues_per_sec": statistics.median(
                    float(x["ddpm_end_to_end_residues_per_sec"]) for x in items
                ),
            }
        )
    return rows


def descriptive(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def summary_text(
    per_target: Sequence[Mapping[str, Any]],
    by_bin: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    total_seconds: float,
) -> str:
    fm_res = descriptive(per_target, "T_fm_residual_ms")
    ddpm_res = descriptive(per_target, "T_ddpm_residual_ms")
    fm_post = descriptive(per_target, "T_postbit_fm_ms")
    ddpm_post = descriptive(per_target, "T_postbit_ddpm_ms")
    fm_e2e = descriptive(per_target, "T_e2e_fm_ms")
    ddpm_e2e = descriptive(per_target, "T_e2e_ddpm_ms")
    bit = descriptive(per_target, "T_bit_ms")
    extra = descriptive(per_target, "T_extra_forward_ms")
    residual_target_speedup = statistics.median(
        float(row["residual_speedup"]) for row in per_target
    )
    residual_median_ratio = ddpm_res["median"] / fm_res["median"]
    post_target_speedup = statistics.median(
        float(row["postbit_speedup"]) for row in per_target
    )
    post_median_ratio = ddpm_post["median"] / fm_post["median"]
    e2e_target_speedup = statistics.median(
        float(row["end_to_end_speedup"]) for row in per_target
    )
    e2e_median_ratio = ddpm_e2e["median"] / fm_e2e["median"]
    fm_peak_allocated = max(float(row["fm_peak_allocated_mib"]) for row in raw_rows)
    fm_peak_reserved = max(float(row["fm_peak_reserved_mib"]) for row in raw_rows)
    ddpm_peak_allocated = max(float(row["ddpm_peak_allocated_mib"]) for row in raw_rows)
    ddpm_peak_reserved = max(float(row["ddpm_peak_reserved_mib"]) for row in raw_rows)

    lines = [
        "FM_VS_MATCHED_DDPM_LATENCY_BENCHMARK",
        "",
        f"GPU: {torch.cuda.get_device_name(device)}",
        f"dtype: {dtype}",
        f"batch size: {BATCH_SIZE}",
        f"targets: {NUM_TARGETS}",
        f"repeats: {REPEATS}",
        f"FM NFE: {FM_NFE}",
        f"Matched DDPM NFE: {DDPM_NFE}",
        "",
        "RESIDUAL-ONLY",
        f"FM mean ms: {fm_res['mean']:.6f}",
        f"FM median ms: {fm_res['median']:.6f}",
        f"FM std/min/max ms: {fm_res['std']:.6f} / {fm_res['min']:.6f} / {fm_res['max']:.6f}",
        f"Matched DDPM mean ms: {ddpm_res['mean']:.6f}",
        f"Matched DDPM median ms: {ddpm_res['median']:.6f}",
        f"Matched DDPM std/min/max ms: {ddpm_res['std']:.6f} / {ddpm_res['min']:.6f} / {ddpm_res['max']:.6f}",
        f"median target-wise speedup: {residual_target_speedup:.6f}",
        f"median-latency ratio: {residual_median_ratio:.6f}",
        f"FM ms/NFE: {fm_res['median'] / FM_NFE:.9f}",
        f"Matched DDPM ms/NFE: {ddpm_res['median'] / DDPM_NFE:.9f}",
        f"median ms/NFE ratio (Matched DDPM / FM): {(ddpm_res['median']/DDPM_NFE)/(fm_res['median']/FM_NFE):.6f}",
        "",
        "POST-BIT",
        f"FM mean/median ms: {fm_post['mean']:.6f} / {fm_post['median']:.6f}",
        f"Matched DDPM mean/median ms: {ddpm_post['mean']:.6f} / {ddpm_post['median']:.6f}",
        f"median target-wise speedup: {post_target_speedup:.6f}",
        f"median-latency ratio: {post_median_ratio:.6f}",
        "",
        "END-TO-END (derived synchronized sequential component sum)",
        f"shared Bit mean/median ms: {bit['mean']:.6f} / {bit['median']:.6f}",
        f"shared extra-forward mean/median ms: {extra['mean']:.6f} / {extra['median']:.6f}",
        f"FM mean/median ms: {fm_e2e['mean']:.6f} / {fm_e2e['median']:.6f}",
        f"Matched DDPM mean/median ms: {ddpm_e2e['mean']:.6f} / {ddpm_e2e['median']:.6f}",
        f"median target-wise speedup: {e2e_target_speedup:.6f}",
        f"median-latency ratio: {e2e_median_ratio:.6f}",
        "",
        "LENGTH-BIN",
    ]
    for row in by_bin:
        lines.append(
            f"{row['length_bin']} N={row['num_targets']}: "
            f"residual FM/Matched-DDPM={float(row['fm_residual_median_ms']):.3f}/"
            f"{float(row['ddpm_residual_median_ms']):.3f} ms "
            f"speedup={float(row['residual_speedup_median_latency_ratio']):.4f}; "
            f"e2e FM/Matched-DDPM={float(row['fm_end_to_end_median_ms']):.3f}/"
            f"{float(row['ddpm_end_to_end_median_ms']):.3f} ms "
            f"speedup={float(row['end_to_end_speedup_median_latency_ratio']):.4f}; "
            f"residual residues/s={float(row['fm_residual_median_residues_per_sec']):.3f}/"
            f"{float(row['ddpm_residual_median_residues_per_sec']):.3f}; "
            f"e2e residues/s={float(row['fm_end_to_end_median_residues_per_sec']):.3f}/"
            f"{float(row['ddpm_end_to_end_median_residues_per_sec']):.3f}"
        )
    lines.extend(
        [
            "",
            "PEAK CUDA (both residual models resident; method block reset separately)",
            f"FM allocated/reserved MiB: {fm_peak_allocated:.3f} / {fm_peak_reserved:.3f}",
            f"Matched DDPM allocated/reserved MiB: {ddpm_peak_allocated:.3f} / {ddpm_peak_reserved:.3f}",
            f"total benchmark runtime seconds: {total_seconds:.3f}",
            "",
            "SANITY",
            "same 16 targets/z_quant/condition/masks/batch/GPU/dtype: PASS",
            "DPLM frozen; FM/Matched DDPM eval; inference_mode/no gradients: PASS",
            "all sampled residuals and decoded atom37 tensors finite: PASS",
            "NFE reduction: 1000 -> 100 = 10x fewer residual-model evaluations",
            "measured speedups: reported above; no fixed latency speedup assumed",
            "",
            "overall: LATENCY_BENCHMARK_PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.perf_counter()
    require(torch.cuda.is_available(), "CUDA GPU is required")
    require(FM_NFE == 100 and DDPM_NFE == NUM_TIMESTEPS == 1000, "NFE changed")
    require(ALPHA == 0.50 and GENERATION_MAX_ITER == 100, "final protocol changed")
    device = resolve_device("cuda")
    seed_everything(BASE_SEED)
    targets = read_targets()
    samples = load_samples(targets)
    atomic_write_csv(TARGETS_CSV, TARGET_COLUMNS, targets)

    checks: dict[str, bool] = {}
    dplm, tokenizer = load_models(DPLM_CHECKPOINT, device, checks)
    fm, _, _, _ = load_fm(FM_CHECKPOINT, device)
    ddpm, schedule, ddpm_metadata = load_ddpm(DDPM_CHECKPOINT, device)
    require(not dplm.training and not tokenizer.training, "frozen model not eval")
    require(not fm.training and not ddpm.training and not schedule.training, "residual model not eval")
    require(
        sum(parameter.numel() for parameter in fm.parameters())
        == sum(parameter.numel() for parameter in ddpm.parameters()),
        "residual architectures differ in parameter count",
    )
    require(
        all(not parameter.requires_grad for module in (dplm, tokenizer, fm, ddpm)
            for parameter in module.parameters()),
        "trainable inference parameter found",
    )
    dtype = next(dplm.parameters()).dtype
    require(next(fm.parameters()).dtype == next(ddpm.parameters()).dtype, "FM/DDPM dtype differs")
    if torch.cuda.memory_allocated(device) > 0:
        torch.cuda.empty_cache()

    raw_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for bin_index, length_bin in enumerate(LENGTH_BINS):
            bin_targets = [row for row in targets if row["length_bin"] == length_bin]
            first = bin_targets[0]
            warm_up_bin(
                bin_index=bin_index,
                target=first,
                sample=samples[str(first["protein_id"])],
                dplm=dplm,
                tokenizer=tokenizer,
                fm=fm,
                ddpm=ddpm,
                schedule=schedule,
                device=device,
            )
            for target in bin_targets:
                raw_rows.extend(
                    benchmark_target(
                        bin_index=bin_index,
                        target=target,
                        sample=samples[str(target["protein_id"])],
                        dplm=dplm,
                        tokenizer=tokenizer,
                        fm=fm,
                        ddpm=ddpm,
                        schedule=schedule,
                        device=device,
                    )
                )
                atomic_write_csv(RAW_CSV, RAW_COLUMNS, raw_rows)
                print(
                    f"[PROGRESS] {len(raw_rows) // REPEATS}/{NUM_TARGETS} targets complete",
                    flush=True,
                )

    require(len(raw_rows) == NUM_TARGETS * REPEATS, "raw row count mismatch")
    require(all_grads_none(dplm) and all_grads_none(tokenizer), "frozen base gradients")
    require(all_grads_none(fm) and all_grads_none(ddpm), "residual model gradients")
    per_target = summarize_targets(targets, raw_rows)
    by_bin = summarize_bins(per_target)
    atomic_write_csv(PER_TARGET_CSV, PER_TARGET_COLUMNS, per_target)
    atomic_write_csv(BY_BIN_CSV, BIN_COLUMNS, by_bin)
    total_seconds = time.perf_counter() - started
    summary = summary_text(per_target, by_bin, raw_rows, device, dtype, total_seconds)
    atomic_write_text(SUMMARY_TXT, summary)
    metadata = {
        "status": "LATENCY_BENCHMARK_PASS",
        "gpu_name": torch.cuda.get_device_name(device),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "dplm_dtype": str(dtype),
        "residual_dtype": str(next(fm.parameters()).dtype),
        "batch_size": BATCH_SIZE,
        "dplm_checkpoint": DPLM_CHECKPOINT,
        "fm_checkpoint": str(FM_CHECKPOINT),
        "fm_checkpoint_sha256": sha256_file(FM_CHECKPOINT),
        "ddpm_checkpoint": str(DDPM_CHECKPOINT),
        "ddpm_checkpoint_sha256": sha256_file(DDPM_CHECKPOINT),
        "selection_csv": str(SELECTION_CSV),
        "selection_csv_sha256": sha256_file(SELECTION_CSV),
        "fm_nfe": FM_NFE,
        "matched_ddpm_nfe": DDPM_NFE,
        "matched_ddpm_checkpoint_metadata": ddpm_metadata,
        "alpha": ALPHA,
        "number_of_targets": NUM_TARGETS,
        "repeats": REPEATS,
        "warmup_policy": (
            "one untimed full FM and Matched DDPM refinement on first target of "
            "each length bin; order alternates by bin"
        ),
        "method_order_policy": "FM->Matched DDPM for bins 1/3; reverse for bins 2/4",
        "timing_method": (
            "CUDA synchronize + time.perf_counter primary; CUDA events recorded "
            "for residual/decode diagnostics"
        ),
        "end_to_end_definition": "derived sum of synchronized sequential component medians",
        "excluded_from_timing": [
            "Python startup", "checkpoint loading", "dataset disk IO",
            "model construction", "initial CUDA allocation",
        ],
        "total_benchmark_runtime_seconds": total_seconds,
        "sanity_checks": {
            "same_targets": True,
            "same_z_quant_condition_masks_within_repeat": True,
            "same_batch_gpu_dtype": True,
            "dplm_frozen": True,
            "fm_matched_ddpm_eval": True,
            "inference_mode_no_gradients": True,
            "finite_outputs": True,
        },
    }
    atomic_write_text(METADATA_JSON, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary, end="", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("overall: LATENCY_BENCHMARK_FAIL", flush=True)
        raise

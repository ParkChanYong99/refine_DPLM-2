#!/usr/bin/env python3
"""Compute-matched FM-100 versus Matched DDPM (DDIM-100) evaluation.

The trained Matched DDPM checkpoint and the finalized Local Bit prediction
manifests are frozen.  This supplementary evaluator adds only a deterministic
eta=0 DDIM sampler with exactly 100 epsilon-model evaluations.  It never
regenerates CAMEO/PDB-date Bit predictions and never tunes alpha.
"""

from __future__ import annotations

import argparse
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

from experiments.latent_residual_ddpm.benchmark_fm_vs_ddpm_latency import (
    ALPHA as LATENCY_ALPHA,
    BASE_SEED as LATENCY_BASE_SEED,
    DDPM_NFE as ANCESTRAL_NFE,
    LENGTH_BINS,
    NUM_TARGETS as LATENCY_NUM_TARGETS,
    PER_TARGET_CSV as NATIVE_LATENCY_CSV,
    REPEATS as LATENCY_REPEATS,
    TARGETS_CSV as LATENCY_TARGETS_CSV,
    load_samples as load_latency_samples,
    make_noise,
    prepare_condition,
    read_targets as read_latency_targets,
    timed_cuda,
    validate_shared_inputs,
)
from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.debug_inference_integration import (
    EXPECTED_BEST_STEP,
    EXPECTED_DDPM_PARAMETERS,
    EXPECTED_OPTIMIZER_STEP,
    NUM_TIMESTEPS,
    all_grads_none,
    load_ddpm,
    synchronize,
    timed_call,
)
from experiments.latent_residual_ddpm.evaluate_final_ddpm import (
    BASE_SEED,
    DDPM_CHECKPOINT,
    EXPECTED_COUNTS,
    FINAL_ALPHA,
    PAIRED_REFERENCE_CSVS,
    load_official_metadata,
    manifest_raw_indices,
    manifest_token_sha256,
    metric_triplet,
    target_roster,
)
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
    assert_atom37,
    build_folding_input,
    extract_final_token_condition,
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.generate_final_folding_predictions import (
    DPLM_CHECKPOINT,
    Target,
    finalize_decoder_output,
)


OUTPUT_ROOT = Path("generation-results/dplm2_bit_650m_final/ddpm_100nfe")
CONFIG_JSON = OUTPUT_ROOT / "ddim100_config.json"
TIMESTEPS_CSV = OUTPUT_ROOT / "ddim100_timesteps.csv"
CAMEO_CSV = OUTPUT_ROOT / "cameo_per_target.csv"
PDB_DATE_CSV = OUTPUT_ROOT / "pdb_date_per_target.csv"
AGGREGATE_CSV = OUTPUT_ROOT / "aggregate_summary.csv"
LATENCY_CSV = OUTPUT_ROOT / "latency_per_target.csv"
LATENCY_SUMMARY = OUTPUT_ROOT / "latency_summary.txt"
SUMMARY_TXT = OUTPUT_ROOT / "summary.txt"
METADATA_JSON = OUTPUT_ROOT / "metadata.json"

FM_CHECKPOINT = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt"
)
ANCESTRAL_RESULTS = {
    "cameo2022": Path(
        "generation-results/dplm2_bit_650m_final/ddpm_matched/cameo_per_target.csv"
    ),
    "PDB_date": Path(
        "generation-results/dplm2_bit_650m_final/ddpm_matched/pdb_date_per_target.csv"
    ),
}
DISPLAY_NAMES = {"cameo2022": "CAMEO", "PDB_date": "PDB-date"}
DDIM_NFE = 100
DDIM_ETA = 0.0
FM_NFE = 100
BATCH_SIZE = 1

TIMESTEP_COLUMNS = (
    "evaluation_index", "timestep", "previous_timestep", "tau",
    "alpha_bar_t", "alpha_bar_previous",
)
PER_TARGET_COLUMNS = (
    "benchmark", "target_id", "target_index", "length", "seed",
    "manifest_token_sha256", "bit_ca_rmsd", "fm_ca_rmsd",
    "ancestral_ca_rmsd", "ddim100_ca_rmsd", "bit_backbone_rmsd",
    "fm_backbone_rmsd", "ancestral_backbone_rmsd",
    "ddim100_backbone_rmsd", "bit_tm_score", "fm_tm_score",
    "ancestral_tm_score", "ddim100_tm_score", "fm_bb_advantage",
    "fm_tm_advantage", "ddim100_bb_advantage_over_ancestral",
    "ddim100_tm_advantage_over_ancestral", "ddim100_nfe", "alpha",
    "extra_forward_seconds", "ddim100_sampling_seconds",
    "decode_metric_seconds", "target_total_seconds",
)
AGGREGATE_COLUMNS = (
    "benchmark", "num_samples",
    "fm_mean_backbone_rmsd", "fm_median_backbone_rmsd",
    "ddim100_mean_backbone_rmsd", "ddim100_median_backbone_rmsd",
    "ancestral_mean_backbone_rmsd", "ancestral_median_backbone_rmsd",
    "fm_mean_tm_score", "fm_median_tm_score",
    "ddim100_mean_tm_score", "ddim100_median_tm_score",
    "ancestral_mean_tm_score", "ancestral_median_tm_score",
    "fm_vs_ddim100_mean_bb_advantage", "fm_vs_ddim100_median_bb_advantage",
    "fm_vs_ddim100_mean_tm_advantage", "fm_vs_ddim100_median_tm_advantage",
    "ddim100_vs_ancestral_mean_bb_advantage",
    "ddim100_vs_ancestral_median_bb_advantage",
    "ddim100_vs_ancestral_mean_tm_advantage",
    "ddim100_vs_ancestral_median_tm_advantage",
)
LATENCY_COLUMNS = (
    "protein_id", "validation_index", "length", "length_bin", "seed",
    "fm100_residual_ms", "ancestral1000_residual_ms",
    "ddim100_repeat_1_ms", "ddim100_repeat_2_ms", "ddim100_repeat_3_ms",
    "ddim100_residual_median_ms", "ddim100_over_fm100_latency_ratio",
    "ancestral1000_over_ddim100_latency_ratio", "fm100_ms_per_nfe",
    "ancestral1000_ms_per_nfe", "ddim100_ms_per_nfe",
    "fm100_residues_per_second", "ancestral1000_residues_per_second",
    "ddim100_residues_per_second",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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


def write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            require(json.load(handle) == payload, f"protocol mismatch: {path}")
    else:
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ddim_timesteps() -> tuple[int, ...]:
    ascending = torch.linspace(
        1, NUM_TIMESTEPS, DDIM_NFE, dtype=torch.float64
    ).round().to(torch.long)
    require(int(ascending[0]) == 1 and int(ascending[-1]) == NUM_TIMESTEPS,
            "DDIM endpoints changed")
    require(bool((ascending[1:] > ascending[:-1]).all()),
            "DDIM respacing produced duplicate/non-monotone timesteps")
    result = tuple(int(value) for value in ascending.flip(0).tolist())
    require(len(result) == DDIM_NFE and len(set(result)) == DDIM_NFE,
            "DDIM timestep count/uniqueness mismatch")
    require(result[0] == NUM_TIMESTEPS and result[-1] == 1 and
            all(a > b for a, b in zip(result, result[1:])),
            "DDIM reverse order mismatch")
    return result


@torch.inference_mode()
def sample_residual_ddim100(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    *,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Deterministic epsilon-prediction DDIM with eta=0 and 100 NFE."""
    require(schedule.num_timesteps == NUM_TIMESTEPS == 1000,
            "DDPM training horizon mismatch")
    require(z_quant.ndim == 3 and z_quant.shape[-1] == EXPECTED_CODEBOOK_DIM,
            "z_quant shape mismatch")
    require(tuple(res_mask.shape) == tuple(z_quant.shape[:2]), "res_mask shape mismatch")
    require(len(hidden_states) == EXPECTED_NUM_LAYERS == 33,
            "condition layer count mismatch")
    require(all(tuple(state.shape[:2]) == tuple(z_quant.shape[:2])
                for state in hidden_states), "condition alignment mismatch")
    mask = res_mask.to(device=z_quant.device, dtype=z_quant.dtype)
    mask_3d = mask[..., None]
    if initial_noise is None:
        x_t = torch.randn(
            z_quant.shape, dtype=z_quant.dtype, device=z_quant.device,
            generator=generator,
        )
    else:
        require(initial_noise.shape == z_quant.shape and
                initial_noise.device == z_quant.device and
                initial_noise.dtype == z_quant.dtype,
                "initial noise mismatch")
        x_t = initial_noise
    x_t = x_t * mask_3d
    require(bool(torch.isfinite(x_t).all()), "initial DDIM state non-finite")
    model.eval()
    evaluated: list[int] = []
    grid = ddim_timesteps()
    for index, timestep in enumerate(grid):
        previous = grid[index + 1] if index + 1 < len(grid) else 0
        tau = torch.full(
            (z_quant.shape[0],), float(timestep) / schedule.num_timesteps,
            dtype=z_quant.dtype, device=z_quant.device,
        )
        epsilon = model(
            x_t=x_t * mask_3d, t=tau, z_quant=z_quant,
            hidden_states=hidden_states, res_mask=mask,
        )
        require(epsilon.shape == x_t.shape and epsilon.dtype == x_t.dtype and
                epsilon.device == x_t.device, "epsilon prediction mismatch")
        epsilon = epsilon * mask_3d
        alpha_bar_t = schedule.alpha_bars[timestep - 1].to(
            device=x_t.device, dtype=x_t.dtype
        )
        alpha_bar_previous = (
            schedule.alpha_bars[previous - 1].to(device=x_t.device, dtype=x_t.dtype)
            if previous > 0 else torch.ones((), device=x_t.device, dtype=x_t.dtype)
        )
        x0_hat = (
            x_t - torch.sqrt(1.0 - alpha_bar_t) * epsilon
        ) / torch.sqrt(alpha_bar_t)
        # eta=0: no stochastic sigma term.  previous=0 uses alpha_bar_0=1.
        x_t = (
            torch.sqrt(alpha_bar_previous) * x0_hat
            + torch.sqrt(1.0 - alpha_bar_previous) * epsilon
        ) * mask_3d
        require(bool(torch.isfinite(x_t).all()),
                f"DDIM state non-finite after mathematical t={timestep}")
        evaluated.append(timestep)
    require(tuple(evaluated) == grid and len(evaluated) == DDIM_NFE,
            "DDIM NFE audit failed")
    result = x_t * mask_3d
    if return_metadata:
        return result, {
            "sampler": "Matched DDPM (DDIM-100)", "eta": DDIM_ETA,
            "nfe": len(evaluated), "timesteps": tuple(evaluated),
            "prediction_type": "epsilon",
        }
    return result


class _SyntheticEpsilonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[float] = []

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor], res_mask: torch.Tensor,
    ) -> torch.Tensor:
        del z_quant, hidden_states
        self.calls.append(float(t[0]))
        return (0.125 * torch.tanh(x_t)) * res_mask[..., None]


def synthetic_sanity() -> dict[str, Any]:
    schedule = DDPMSchedule().to(dtype=torch.float32)
    model = _SyntheticEpsilonModel().eval()
    z_quant = torch.zeros(1, 5, EXPECTED_CODEBOOK_DIM)
    hidden = tuple(torch.zeros(1, 5, EXPECTED_HIDDEN_SIZE)
                   for _ in range(EXPECTED_NUM_LAYERS))
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(913)
    initial = torch.randn(z_quant.shape, generator=generator)
    first, metadata = sample_residual_ddim100(
        model, schedule, z_quant, hidden, mask,
        initial_noise=initial.clone(), return_metadata=True,
    )
    first_calls = tuple(model.calls)
    model.calls.clear()
    second = sample_residual_ddim100(
        model, schedule, z_quant, hidden, mask, initial_noise=initial.clone()
    )
    require(metadata["nfe"] == DDIM_NFE and
            metadata["timesteps"] == ddim_timesteps(), "synthetic NFE audit")
    require(len(first_calls) == DDIM_NFE and len(model.calls) == DDIM_NFE,
            "synthetic model call count mismatch")
    require(torch.equal(first, second), "DDIM eta=0 is not deterministic")
    require(bool(torch.isfinite(first).all()), "synthetic result non-finite")
    require(torch.equal(first[:, 3:], torch.zeros_like(first[:, 3:])),
            "synthetic padding did not remain zero")
    require(not model.training and all_grads_none(model),
            "synthetic model eval/no-gradient audit failed")
    return {
        "nfe": metadata["nfe"], "deterministic": True,
        "finite": True, "padding_zero": True, "no_gradients": True,
    }


def timestep_rows(schedule: DDPMSchedule) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = ddim_timesteps()
    alpha_bars = schedule.alpha_bars.detach().cpu().double()
    for index, timestep in enumerate(grid):
        previous = grid[index + 1] if index + 1 < len(grid) else 0
        rows.append({
            "evaluation_index": index,
            "timestep": timestep,
            "previous_timestep": previous,
            "tau": timestep / NUM_TIMESTEPS,
            "alpha_bar_t": float(alpha_bars[timestep - 1]),
            "alpha_bar_previous": (
                float(alpha_bars[previous - 1]) if previous > 0 else 1.0
            ),
        })
    return rows


def read_indexed(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    require(path.is_file(), f"source artifact missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        target_id = row[id_column]
        require(target_id and target_id not in result, f"duplicate target: {target_id}")
        result[target_id] = row
    return result


def finite(row: Mapping[str, str], column: str) -> float:
    value = float(row[column])
    require(math.isfinite(value), f"non-finite source value: {column}")
    return value


def source_rows(
    dataset: str, eligible: Sequence[Target]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    fm = read_indexed(PAIRED_REFERENCE_CSVS[dataset], "sample_id")
    ancestral = read_indexed(ANCESTRAL_RESULTS[dataset], "target_id")
    expected = {target.sample_id for target in eligible}
    require(set(fm) == set(ancestral) == expected,
            f"{dataset} finalized FM/ancestral target set mismatch")
    for target in eligible:
        f = fm[target.sample_id]
        a = ancestral[target.sample_id]
        require(int(f["length"]) == int(a["length"]) == target.length,
                f"{target.sample_id} source length mismatch")
        for left, right in (
            ("bit_ca_rmsd", "bit_ca_rmsd"),
            ("bit_bb_rmsd", "bit_backbone_rmsd"),
            ("bit_bb_tmscore", "bit_tm_score"),
        ):
            require(math.isclose(finite(f, left), finite(a, right),
                                 rel_tol=1e-12, abs_tol=1e-12),
                    f"{target.sample_id} frozen Bit source mismatch")
    return fm, ancestral


def frozen_condition(
    target: Target, manifest_row: Mapping[str, str], dplm: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, float]:
    from byprot.datamodules.pdb_dataset import utils as du

    aatype = torch.tensor(du.seq_to_aatype(target.sequence), dtype=torch.long)
    input_tokens, partial_mask = build_folding_input(dplm, aatype, device)
    lexemes, raw_indices = manifest_raw_indices(manifest_row, target)
    offset = int(dplm.struct_vocab_offset)
    combined: list[int] = []
    for lexeme, raw_index in zip(lexemes, raw_indices):
        require(lexeme in dplm.tokenizer._token_to_id, "unknown frozen token lexeme")
        token_id = int(dplm.tokenizer._token_to_id[lexeme])
        require(token_id - offset == raw_index, "frozen token/index mismatch")
        combined.append(token_id)
    final_tokens = input_tokens.clone()
    final_tokens[:, 1:target.length + 1] = torch.tensor(
        combined, dtype=final_tokens.dtype, device=device
    ).view(1, target.length)
    require(torch.equal(final_tokens[partial_mask], input_tokens[partial_mask]),
            "frozen token replacement changed AA side")
    z_quant, res_mask = reconstruct_prediction_latent(dplm, final_tokens)
    hidden, seconds = timed_call(
        device, lambda: extract_final_token_condition(dplm, final_tokens, target.length)
    )
    hidden = tuple(hidden)
    validate_shared_inputs(z_quant, hidden, res_mask, target.length)
    return z_quant, res_mask, hidden, aatype.to(device), seconds


def run_ddim(
    ddpm: torch.nn.Module, schedule: DDPMSchedule, z_quant: torch.Tensor,
    hidden: Sequence[torch.Tensor], mask: torch.Tensor, seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled, seconds = timed_call(
        device,
        lambda: sample_residual_ddim100(
            ddpm, schedule, z_quant, hidden, mask,
            generator=generator, return_metadata=True,
        ),
    )
    residual, metadata = sampled
    require(metadata["nfe"] == DDIM_NFE and
            metadata["timesteps"] == ddim_timesteps(), "actual DDIM NFE audit")
    require(tuple(residual.shape) == tuple(z_quant.shape) and
            bool(torch.isfinite(residual).all()), "actual DDIM residual invalid")
    return residual, metadata, seconds


def decode_metric(
    tokenizer: torch.nn.Module, z_quant: torch.Tensor, residual: torch.Tensor,
    mask: torch.Tensor, aatype: torch.Tensor, target: Target, gt_metadata: Any,
) -> tuple[tuple[float, float, float], float]:
    started = time.perf_counter()
    z_refined = z_quant + FINAL_ALPHA * residual
    require(bool(torch.isfinite(z_refined).all()), "refined latent non-finite")
    decoded = tokenizer.detokenize(z_refined, res_mask=mask)
    output = finalize_decoder_output(
        "Matched DDPM (DDIM-100)", decoded,
        aatype.view(1, target.length), mask, target.sample_id, target.length,
    )
    assert_atom37("Matched DDPM (DDIM-100)", output, target.length)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ddim100-pdb-", dir=OUTPUT_ROOT) as name:
        tokenizer.output_to_pdb(dict(output), output_dir=name)
        prediction = Path(name) / f"{target.sample_id}.pdb"
        metrics = metric_triplet(prediction, target, gt_metadata)
    return metrics, time.perf_counter() - started


def quality_row(
    dataset: str, target: Target, manifest_row: Mapping[str, str],
    fm: Mapping[str, str], ancestral: Mapping[str, str],
    dplm: torch.nn.Module, tokenizer: torch.nn.Module, ddpm: torch.nn.Module,
    schedule: DDPMSchedule, gt_metadata: Any, device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    z_quant, mask, hidden, aatype, extra_seconds = frozen_condition(
        target, manifest_row, dplm, device
    )
    seed = BASE_SEED + target.fasta_index
    residual, metadata, sampling_seconds = run_ddim(
        ddpm, schedule, z_quant, hidden, mask, seed, device
    )
    metrics, decode_seconds = decode_metric(
        tokenizer, z_quant, residual, mask, aatype, target, gt_metadata
    )
    bit = (finite(fm, "bit_ca_rmsd"), finite(fm, "bit_bb_rmsd"),
           finite(fm, "bit_bb_tmscore"))
    fm_metrics = (finite(fm, "fm_ca_rmsd"), finite(fm, "fm_bb_rmsd"),
                  finite(fm, "fm_bb_tmscore"))
    ancestral_metrics = (
        finite(ancestral, "ddpm_ca_rmsd"),
        finite(ancestral, "ddpm_backbone_rmsd"),
        finite(ancestral, "ddpm_tm_score"),
    )
    require(metadata["nfe"] == DDIM_NFE, "quality NFE mismatch")
    return {
        "benchmark": DISPLAY_NAMES[dataset], "target_id": target.sample_id,
        "target_index": target.fasta_index, "length": target.length,
        "seed": seed, "manifest_token_sha256": manifest_token_sha256(manifest_row, target),
        "bit_ca_rmsd": bit[0], "fm_ca_rmsd": fm_metrics[0],
        "ancestral_ca_rmsd": ancestral_metrics[0], "ddim100_ca_rmsd": metrics[0],
        "bit_backbone_rmsd": bit[1], "fm_backbone_rmsd": fm_metrics[1],
        "ancestral_backbone_rmsd": ancestral_metrics[1],
        "ddim100_backbone_rmsd": metrics[1], "bit_tm_score": bit[2],
        "fm_tm_score": fm_metrics[2], "ancestral_tm_score": ancestral_metrics[2],
        "ddim100_tm_score": metrics[2],
        "fm_bb_advantage": metrics[1] - fm_metrics[1],
        "fm_tm_advantage": fm_metrics[2] - metrics[2],
        "ddim100_bb_advantage_over_ancestral": ancestral_metrics[1] - metrics[1],
        "ddim100_tm_advantage_over_ancestral": metrics[2] - ancestral_metrics[2],
        "ddim100_nfe": metadata["nfe"], "alpha": FINAL_ALPHA,
        "extra_forward_seconds": extra_seconds,
        "ddim100_sampling_seconds": sampling_seconds,
        "decode_metric_seconds": decode_seconds,
        "target_total_seconds": time.perf_counter() - started,
    }


def load_partial(
    path: Path, dataset: str, eligible: Sequence[Target]
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == PER_TARGET_COLUMNS,
                f"partial schema mismatch: {path}")
        rows = list(reader)
    by_index = {target.fasta_index: target for target in eligible}
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["target_index"])
        require(index in by_index and index not in result, "partial target mismatch")
        target = by_index[index]
        require(row["benchmark"] == DISPLAY_NAMES[dataset] and
                row["target_id"] == target.sample_id and
                int(row["length"]) == target.length and
                int(row["ddim100_nfe"]) == DDIM_NFE and
                float(row["alpha"]) == FINAL_ALPHA,
                "partial protocol mismatch")
        for column in PER_TARGET_COLUMNS:
            if column not in {"benchmark", "target_id", "manifest_token_sha256"}:
                require(math.isfinite(float(row[column])), f"partial non-finite: {column}")
        result[index] = dict(row)
    return result


def aggregate(dataset: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == EXPECTED_COUNTS[dataset], "aggregate N mismatch")
    result: dict[str, Any] = {
        "benchmark": DISPLAY_NAMES[dataset], "num_samples": len(rows)
    }
    for method in ("fm", "ddim100", "ancestral"):
        for metric in ("backbone_rmsd", "tm_score"):
            values = [float(row[f"{method}_{metric}"]) for row in rows]
            result[f"{method}_mean_{metric}"] = statistics.fmean(values)
            result[f"{method}_median_{metric}"] = statistics.median(values)
    for column, prefix in (
        ("fm_bb_advantage", "fm_vs_ddim100_bb_advantage"),
        ("fm_tm_advantage", "fm_vs_ddim100_tm_advantage"),
        ("ddim100_bb_advantage_over_ancestral", "ddim100_vs_ancestral_bb_advantage"),
        ("ddim100_tm_advantage_over_ancestral", "ddim100_vs_ancestral_tm_advantage"),
    ):
        values = [float(row[column]) for row in rows]
        result[f"{prefix.replace('_bb_advantage', '')}_mean_bb_advantage" if "_bb_" in prefix
               else f"{prefix.replace('_tm_advantage', '')}_mean_tm_advantage"] = statistics.fmean(values)
        result[f"{prefix.replace('_bb_advantage', '')}_median_bb_advantage" if "_bb_" in prefix
               else f"{prefix.replace('_tm_advantage', '')}_median_tm_advantage"] = statistics.median(values)
    require(set(result) == set(AGGREGATE_COLUMNS),
            "aggregate columns construction mismatch")
    return result


def actual_internal_sanity(
    dplm: torch.nn.Module, tokenizer: torch.nn.Module, ddpm: torch.nn.Module,
    schedule: DDPMSchedule, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = read_latency_targets()
    samples = load_latency_samples(targets)
    first = targets[0]
    sample = samples[str(first["protein_id"])]
    final_tokens, z_quant, mask, hidden, _, _ = prepare_condition(
        dplm, sample, int(first["length"]), device
    )
    del final_tokens
    seed = LATENCY_BASE_SEED + int(first["validation_index"])
    initial = make_noise(z_quant, seed)
    first_residual = sample_residual_ddim100(
        ddpm, schedule, z_quant, hidden, mask, initial_noise=initial.clone()
    )
    second_residual = sample_residual_ddim100(
        ddpm, schedule, z_quant, hidden, mask, initial_noise=initial.clone()
    )
    require(torch.equal(first_residual, second_residual),
            "actual internal DDIM determinism failed")
    require(tuple(first_residual.shape) == (1, int(first["length"]), EXPECTED_CODEBOOK_DIM)
            and bool(torch.isfinite(first_residual).all()),
            "actual internal residual shape/finite audit failed")
    decoded = tokenizer.detokenize(
        z_quant + FINAL_ALPHA * first_residual, res_mask=mask
    )
    assert_atom37("internal Matched DDPM (DDIM-100)", decoded, int(first["length"]))
    require(all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "actual internal sanity accumulated gradients")
    cache = {
        "target": first, "sample": sample, "z_quant": z_quant,
        "mask": mask, "hidden": hidden,
    }
    return {
        "target_id": first["protein_id"], "validation_index": first["validation_index"],
        "length": first["length"], "shape": list(first_residual.shape),
        "deterministic": True, "finite": True, "decoder_atom37": True,
        "nfe": DDIM_NFE,
    }, cache


def native_latency_rows() -> dict[str, dict[str, str]]:
    native = read_indexed(NATIVE_LATENCY_CSV, "protein_id")
    require(len(native) == LATENCY_NUM_TARGETS, "native latency target count mismatch")
    return native


def benchmark_latency(
    dplm: torch.nn.Module, ddpm: torch.nn.Module, schedule: DDPMSchedule,
    device: torch.device, first_cache: Mapping[str, Any],
) -> list[dict[str, Any]]:
    targets = read_latency_targets()
    samples = load_latency_samples(targets)
    native = native_latency_rows()
    require(LATENCY_REPEATS == 3 and LATENCY_ALPHA == FINAL_ALPHA == 0.50,
            "latency protocol changed")
    rows: list[dict[str, Any]] = []
    warmed_bins: set[str] = set()
    for ordinal, target in enumerate(targets, start=1):
        protein_id = str(target["protein_id"])
        length = int(target["length"])
        validation_index = int(target["validation_index"])
        length_bin = str(target["length_bin"])
        if protein_id == first_cache["target"]["protein_id"]:
            z_quant = first_cache["z_quant"]
            mask = first_cache["mask"]
            hidden = first_cache["hidden"]
            warmed_bins.add(length_bin)  # actual internal sanity was the untimed warm-up.
        else:
            _, z_quant, mask, hidden, _, _ = prepare_condition(
                dplm, samples[protein_id], length, device
            )
        validate_shared_inputs(z_quant, hidden, mask, length)
        seed = LATENCY_BASE_SEED + validation_index
        initial = make_noise(z_quant, seed)
        if length_bin not in warmed_bins:
            sample_residual_ddim100(
                ddpm, schedule, z_quant, hidden, mask,
                initial_noise=initial.clone(),
            )
            synchronize(device)
            warmed_bins.add(length_bin)
        repeat_ms: list[float] = []
        reference_output: torch.Tensor | None = None
        for _ in range(LATENCY_REPEATS):
            output, wall_ms, _ = timed_cuda(
                device,
                lambda: sample_residual_ddim100(
                    ddpm, schedule, z_quant, hidden, mask,
                    initial_noise=initial.clone(),
                ),
            )
            require(bool(torch.isfinite(output).all()), "latency DDIM non-finite")
            if reference_output is None:
                reference_output = output.detach().clone()
            else:
                require(torch.equal(reference_output, output),
                        "latency repeat determinism failed")
            repeat_ms.append(wall_ms)
        require(len(repeat_ms) == LATENCY_REPEATS, "latency repeat count mismatch")
        ddim_median = statistics.median(repeat_ms)
        source = native[protein_id]
        require(int(source["validation_index"]) == validation_index and
                int(source["length"]) == length and source["length_bin"] == length_bin,
                "native latency identity mismatch")
        fm_ms = finite(source, "T_fm_residual_ms")
        ancestral_ms = finite(source, "T_ddpm_residual_ms")
        row = {
            "protein_id": protein_id, "validation_index": validation_index,
            "length": length, "length_bin": length_bin, "seed": seed,
            "fm100_residual_ms": fm_ms,
            "ancestral1000_residual_ms": ancestral_ms,
            "ddim100_repeat_1_ms": repeat_ms[0],
            "ddim100_repeat_2_ms": repeat_ms[1],
            "ddim100_repeat_3_ms": repeat_ms[2],
            "ddim100_residual_median_ms": ddim_median,
            "ddim100_over_fm100_latency_ratio": ddim_median / fm_ms,
            "ancestral1000_over_ddim100_latency_ratio": ancestral_ms / ddim_median,
            "fm100_ms_per_nfe": fm_ms / FM_NFE,
            "ancestral1000_ms_per_nfe": ancestral_ms / ANCESTRAL_NFE,
            "ddim100_ms_per_nfe": ddim_median / DDIM_NFE,
            "fm100_residues_per_second": length / (fm_ms / 1000.0),
            "ancestral1000_residues_per_second": length / (ancestral_ms / 1000.0),
            "ddim100_residues_per_second": length / (ddim_median / 1000.0),
        }
        rows.append(row)
        atomic_write_csv(LATENCY_CSV, LATENCY_COLUMNS, rows)
        print(f"[latency {ordinal}/{len(targets)}] {protein_id} "
              f"FM={fm_ms:.3f} ms DDIM-100={ddim_median:.3f} ms "
              f"ancestral={ancestral_ms:.3f} ms", flush=True)
    require(len(rows) == LATENCY_NUM_TARGETS and warmed_bins == set(LENGTH_BINS),
            "latency coverage/warm-up mismatch")
    return rows


def latency_report(rows: Sequence[Mapping[str, Any]]) -> str:
    fm = [float(row["fm100_residual_ms"]) for row in rows]
    ancestral = [float(row["ancestral1000_residual_ms"]) for row in rows]
    ddim = [float(row["ddim100_residual_median_ms"]) for row in rows]
    ratios = [float(row["ddim100_over_fm100_latency_ratio"]) for row in rows]
    fm_throughput = [float(row["fm100_residues_per_second"]) for row in rows]
    ancestral_throughput = [
        float(row["ancestral1000_residues_per_second"]) for row in rows
    ]
    ddim_throughput = [float(row["ddim100_residues_per_second"]) for row in rows]
    lines = [
        "COMPUTE-MATCHED RESIDUAL LATENCY",
        f"targets: {len(rows)}; batch size: {BATCH_SIZE}; repeats: {LATENCY_REPEATS}",
        f"FM-100 median residual ms: {statistics.median(fm):.9f}",
        f"Matched DDPM (DDIM-100) median residual ms: {statistics.median(ddim):.9f}",
        f"Matched DDPM ancestral-1000 reference median residual ms: {statistics.median(ancestral):.9f}",
        f"DDIM-100 / FM-100 aggregate median latency ratio: "
        f"{statistics.median(ddim) / statistics.median(fm):.9f}",
        f"median target-wise DDIM-100 / FM-100 latency ratio: {statistics.median(ratios):.9f}",
        f"FM-100 median ms/NFE: {statistics.median(fm) / FM_NFE:.9f}",
        f"DDIM-100 median ms/NFE: {statistics.median(ddim) / DDIM_NFE:.9f}",
        f"ancestral-1000 reference median ms/NFE: "
        f"{statistics.median(ancestral) / ANCESTRAL_NFE:.9f}",
        f"FM-100 median residual residues/s: {statistics.median(fm_throughput):.9f}",
        f"DDIM-100 median residual residues/s: {statistics.median(ddim_throughput):.9f}",
        f"ancestral-1000 reference median residual residues/s: "
        f"{statistics.median(ancestral_throughput):.9f}",
    ]
    for length_bin in LENGTH_BINS:
        selected = [row for row in rows if row["length_bin"] == length_bin]
        require(len(selected) == 4, f"latency bin {length_bin} N mismatch")
        lines.append(
            f"{length_bin}: FM/DDIM/ancestral median ms = "
            f"{statistics.median(float(r['fm100_residual_ms']) for r in selected):.6f} / "
            f"{statistics.median(float(r['ddim100_residual_median_ms']) for r in selected):.6f} / "
            f"{statistics.median(float(r['ancestral1000_residual_ms']) for r in selected):.6f}"
        )
    return "\n".join(lines) + "\n"


def quality_report(aggregates: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "COMPUTE_MATCHED_100NFE",
        f"DDPM training T: {NUM_TIMESTEPS}",
        "DDPM accelerated sampler: DDIM eta=0",
        f"DDPM sampling NFE: {DDIM_NFE}",
        f"FM sampling NFE: {FM_NFE}",
        f"alpha: {FINAL_ALPHA:.2f}",
    ]
    for row in aggregates:
        lines.extend([
            f"\n{row['benchmark']}: N={row['num_samples']}",
            f"FM-100 BB RMSD: {float(row['fm_mean_backbone_rmsd']):.9g}",
            f"FM-100 TM: {float(row['fm_mean_tm_score']):.9g}",
            f"Matched DDPM (DDIM-100) BB RMSD: "
            f"{float(row['ddim100_mean_backbone_rmsd']):.9g}",
            f"Matched DDPM (DDIM-100) TM: {float(row['ddim100_mean_tm_score']):.9g}",
            f"Matched DDPM ancestral-1000 reference BB RMSD: "
            f"{float(row['ancestral_mean_backbone_rmsd']):.9g}",
            f"Matched DDPM ancestral-1000 reference TM: "
            f"{float(row['ancestral_mean_tm_score']):.9g}",
        ])
    lines.extend([
        "\nNFE audit: PASS", "determinism: PASS", "NaN/Inf: PASS",
        "overall: COMPUTE_MATCHED_100NFE_PASS",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    require(FINAL_ALPHA == LATENCY_ALPHA == 0.50, "frozen alpha changed")
    require(NUM_TIMESTEPS == ANCESTRAL_NFE == 1000 and
            DDIM_NFE == FM_NFE == 100, "NFE protocol mismatch")
    synthetic = synthetic_sanity()
    grid = ddim_timesteps()
    cpu_schedule = DDPMSchedule()
    config = {
        "method": "Matched DDPM (DDIM-100)", "training_T": NUM_TIMESTEPS,
        "sampling_nfe": DDIM_NFE, "eta": DDIM_ETA,
        "prediction_type": "epsilon", "alpha": FINAL_ALPHA,
        "respacing": "round(linspace(1,1000,100)), reversed",
        "first_timestep": grid[0], "last_timestep": grid[-1],
        "final_previous_timestep": 0,
        "final_previous_alpha_bar": 1.0,
    }
    write_or_validate_json(CONFIG_JSON, config)
    atomic_write_csv(TIMESTEPS_CSV, TIMESTEP_COLUMNS, timestep_rows(cpu_schedule))

    rosters: dict[str, list[Target]] = {}
    manifests: dict[str, dict[int, dict[str, str]]] = {}
    gt_metadata: dict[str, Any] = {}
    fm_sources: dict[str, dict[str, dict[str, str]]] = {}
    ancestral_sources: dict[str, dict[str, dict[str, str]]] = {}
    source_hashes: dict[str, str] = {}
    for dataset in EXPECTED_COUNTS:
        eligible, _, manifest, manifest_sha, _ = target_roster(dataset)
        rosters[dataset] = eligible
        manifests[dataset] = manifest
        gt_metadata[dataset] = load_official_metadata(dataset)
        fm, ancestral = source_rows(dataset, eligible)
        fm_sources[dataset] = fm
        ancestral_sources[dataset] = ancestral
        source_hashes[f"{dataset}_fm_paired"] = sha256_file(PAIRED_REFERENCE_CSVS[dataset])
        source_hashes[f"{dataset}_ancestral"] = sha256_file(ANCESTRAL_RESULTS[dataset])
        source_hashes[f"{dataset}_manifest"] = manifest_sha

    device = resolve_device(args.device)
    seed_everything(BASE_SEED)
    checks: dict[str, bool] = {}
    dplm, tokenizer = load_models(DPLM_CHECKPOINT, device, checks)
    ddpm, schedule, checkpoint_info = load_ddpm(DDPM_CHECKPOINT, device)
    require(int(checkpoint_info["optimizer_step"]) == EXPECTED_OPTIMIZER_STEP and
            int(checkpoint_info["best_step"]) == EXPECTED_BEST_STEP and
            int(checkpoint_info["parameter_count"]) == EXPECTED_DDPM_PARAMETERS,
            "frozen Matched DDPM checkpoint mismatch")
    require(not dplm.training and not tokenizer.training and not ddpm.training and
            all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "inference model state mismatch")

    internal, first_latency_cache = actual_internal_sanity(
        dplm, tokenizer, ddpm, schedule, device
    )
    aggregates: list[dict[str, Any]] = []
    for dataset in EXPECTED_COUNTS:
        output = CAMEO_CSV if dataset == "cameo2022" else PDB_DATE_CSV
        eligible = rosters[dataset]
        completed = load_partial(output, dataset, eligible)
        print(f"[{DISPLAY_NAMES[dataset]}] N={len(eligible)} resume={len(completed)}",
              flush=True)
        for ordinal, target in enumerate(eligible, start=1):
            if target.fasta_index in completed:
                continue
            row = quality_row(
                dataset, target, manifests[dataset][target.fasta_index],
                fm_sources[dataset][target.sample_id],
                ancestral_sources[dataset][target.sample_id],
                dplm, tokenizer, ddpm, schedule, gt_metadata[dataset], device,
            )
            completed[target.fasta_index] = row
            ordered = [completed[item.fasta_index] for item in eligible
                       if item.fasta_index in completed]
            atomic_write_csv(output, PER_TARGET_COLUMNS, ordered)
            print(f"[{DISPLAY_NAMES[dataset]} {ordinal}/{len(eligible)}] "
                  f"{target.sample_id} DDIM-100 BB={row['ddim100_backbone_rmsd']:.6g} "
                  f"TM={row['ddim100_tm_score']:.6g}", flush=True)
        require(len(completed) == EXPECTED_COUNTS[dataset], "quality evaluation incomplete")
        rows = [completed[target.fasta_index] for target in eligible]
        aggregates.append(aggregate(dataset, rows))
        atomic_write_csv(AGGREGATE_CSV, AGGREGATE_COLUMNS, aggregates)

    latency_rows = benchmark_latency(
        dplm, ddpm, schedule, device, first_latency_cache
    )
    latency_text = latency_report(latency_rows)
    atomic_write_text(LATENCY_SUMMARY, latency_text)
    report = quality_report(aggregates)
    atomic_write_text(SUMMARY_TXT, report)
    require(all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "evaluation accumulated gradients")
    metadata = {
        "status": "COMPUTE_MATCHED_100NFE_PASS",
        "description": "compute-matched supplementary evaluation",
        "sampler": "Matched DDPM (DDIM-100)", "eta": DDIM_ETA,
        "training_T": NUM_TIMESTEPS, "ddim_nfe": DDIM_NFE, "fm_nfe": FM_NFE,
        "alpha": FINAL_ALPHA, "timestep_sequence": list(grid),
        "synthetic_sanity": synthetic, "actual_internal_sanity": internal,
        "CAMEO_PDB_date_used_for_tuning": False,
        "new_CAMEO_PDB_date_DPLM_generation": False,
        "frozen_prediction_manifests_reused": True,
        "latency_native_FM_ancestral_artifact_reused": str(NATIVE_LATENCY_CSV),
        "latency_DDIM100_measured_in_this_run": True,
        "latency_batch_size": BATCH_SIZE, "latency_repeats": LATENCY_REPEATS,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(next(ddpm.parameters()).dtype),
        "pytorch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "fm_checkpoint": str(FM_CHECKPOINT),
        "fm_checkpoint_sha256": sha256_file(FM_CHECKPOINT),
        "ddpm_checkpoint": str(DDPM_CHECKPOINT),
        "ddpm_checkpoint_sha256": sha256_file(DDPM_CHECKPOINT),
        "source_artifact_sha256": source_hashes,
        "latency_targets_sha256": sha256_file(LATENCY_TARGETS_CSV),
        "native_latency_sha256": sha256_file(NATIVE_LATENCY_CSV),
        "total_runtime_seconds": time.perf_counter() - started,
        "sanity": {
            "exact_nfe": True, "deterministic": True, "finite": True,
            "padding_zero": True, "eval_mode": True, "no_gradients": True,
            "same_frozen_bit_targets": True,
        },
    }
    atomic_write_text(METADATA_JSON, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(report, end="")
    print(latency_text, end="")
    print(f"output root: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("overall: COMPUTE_MATCHED_100NFE_FAIL", flush=True)
        raise

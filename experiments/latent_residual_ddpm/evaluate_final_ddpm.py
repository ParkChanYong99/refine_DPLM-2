#!/usr/bin/env python3
"""Frozen-manifest Bit versus matched-DDPM forward-folding evaluation.

This script does not select alpha or tune any test-time setting.  It reuses the
exact frozen DPLM structure-token predictions from finalized Local Bit/FM
generation manifests. This avoids regeneration variability and ensures that
Local Bit, FM, and Matched DDPM share each target's base-model prediction.
The repository's official GT loader and metric implementation are unchanged.
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
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from byprot.datamodules.pdb_dataset import utils as du
from byprot.utils.protein import utils as evaluator_utils
from byprot.utils.protein.evaluator_dplm2 import EvalRunner, load_pdb_by_name
from experiments.latent_residual_ddpm.debug_inference_integration import (
    EXPECTED_BEST_STEP,
    EXPECTED_DDPM_PARAMETERS,
    EXPECTED_OPTIMIZER_STEP,
    NUM_TIMESTEPS,
    all_grads_none,
    load_ddpm,
    run_sampler,
    synchronize,
    timed_call,
)
from experiments.latent_residual_fm.compare_cameo_bit_vs_fm import (
    DEFAULT_BIT_EVAL_DIR as CAMEO_REFERENCE_EVAL,
    PAIRED_CSV as CAMEO_PAIRED_CSV,
    collect_top_samples,
    read_aggregate,
    validate_aggregate,
)
from experiments.latent_residual_fm.compare_pdb_date_bit_vs_fm import (
    DEFAULT_BIT_EVAL_DIR as PDB_DATE_REFERENCE_EVAL,
    PAIRED_CSV as PDB_DATE_PAIRED_CSV,
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
    build_folding_input,
    extract_final_token_condition,
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.generate_final_folding_predictions import (
    DATASET_FASTAS,
    DPLM_CHECKPOINT,
    DPLM_MAX_ITER,
    MAX_LENGTH,
    MIN_LENGTH,
    Target,
    finalize_decoder_output,
    output_layout,
    read_manifest,
    read_targets,
    valid_pdb,
)


FINAL_ALPHA = 0.50
BASE_SEED = 42
EXPECTED_COUNTS = {"cameo2022": 163, "PDB_date": 442}
DDPM_CHECKPOINT = Path(
    "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main/checkpoint_best.pt"
)
ALPHA_SELECTION_ARTIFACT = Path(
    "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main/"
    "internal_val_alpha_sweep_100_summary.csv"
)
OUTPUT_ROOT = Path("generation-results/dplm2_bit_650m_final/ddpm_matched")
METADATA_NAME = "protocol_metadata.json"
REFERENCE_EVAL_DIRS = {
    "cameo2022": CAMEO_REFERENCE_EVAL,
    "PDB_date": PDB_DATE_REFERENCE_EVAL,
}
PAIRED_REFERENCE_CSVS = {
    "cameo2022": CAMEO_PAIRED_CSV,
    "PDB_date": PDB_DATE_PAIRED_CSV,
}
GT_METADATA = {
    "cameo2022": (Path("data-bin/metadata/pdb_afdb_cameo.csv"), Path("data-bin")),
    "PDB_date": (Path("data-bin/metadata/pdb_date.csv"), Path("data-bin/PDB_date")),
}
PREDICTION_SOURCE = "finalized_generation_manifest"
PER_TARGET_COLUMNS = (
    "benchmark", "target_id", "target_index", "length", "seed", "manifest_token_sha256",
    "bit_ca_rmsd", "ddpm_ca_rmsd", "ca_improvement",
    "bit_backbone_rmsd", "ddpm_backbone_rmsd", "backbone_improvement",
    "bit_tm_score", "ddpm_tm_score", "tm_improvement", "ddpm_nfe", "alpha",
    "extra_forward_seconds", "ddpm_sampling_seconds",
    "decode_metric_seconds", "target_total_seconds",
)
SUMMARY_COLUMNS = (
    "benchmark", "num_samples",
    "bit_mean_ca_rmsd", "ddpm_mean_ca_rmsd",
    "bit_median_ca_rmsd", "ddpm_median_ca_rmsd",
    "bit_mean_backbone_rmsd", "ddpm_mean_backbone_rmsd",
    "bit_median_backbone_rmsd", "ddpm_median_backbone_rmsd",
    "bit_mean_tm_score", "ddpm_mean_tm_score",
    "bit_median_tm_score", "ddpm_median_tm_score",
    "mean_ca_improvement", "median_ca_improvement",
    "mean_backbone_improvement", "median_backbone_improvement",
    "mean_tm_improvement", "median_tm_improvement",
    "ca_improved", "ca_worsened", "ca_tied",
    "backbone_improved", "backbone_worsened", "backbone_tied",
    "tm_improved", "tm_worsened", "tm_tied",
    "extra_forward_seconds", "ddpm_sampling_seconds",
    "decode_metric_seconds", "target_total_seconds",
)
METRIC_SOURCE_PATHS = (
    Path("src/byprot/utils/protein/evaluator_dplm2.py"),
    Path("src/byprot/utils/protein/utils.py"),
    Path("experiments/latent_residual_fm/generate_final_folding_predictions.py"),
    Path("experiments/latent_residual_ddpm/evaluate_final_ddpm.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required artifact missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
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


def validate_frozen_alpha() -> str:
    require(FINAL_ALPHA == 0.50, "final DDPM alpha must remain 0.50")
    with ALPHA_SELECTION_ARTIFACT.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"alpha", "num_samples", "mean_ca_rmsd", "selected"}
        require(required.issubset(reader.fieldnames or ()), "alpha selection artifact schema mismatch")
        rows = list(reader)
    require(len(rows) == 5, "alpha selection artifact must contain five grid rows")
    require(
        [float(row["alpha"]) for row in rows] == [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha selection grid changed",
    )
    require(all(int(row["num_samples"]) == 100 for row in rows), "alpha selection N mismatch")
    selected = [row for row in rows if row["selected"].strip().lower() == "true"]
    require(len(selected) == 1 and float(selected[0]["alpha"]) == FINAL_ALPHA,
            "frozen internal-validation selected alpha is not 0.50")
    require(
        float(selected[0]["mean_ca_rmsd"])
        == min(float(row["mean_ca_rmsd"]) for row in rows),
        "selected alpha is not lowest mean CA RMSD",
    )
    return sha256_file(ALPHA_SELECTION_ARTIFACT)


def manifest_raw_indices(row: Mapping[str, str], target: Target) -> tuple[list[str], list[int]]:
    """Manifest lexemes are batch-decoded structure tokens, i.e. raw LFQ indices."""
    lexemes = row["struct_token_sequence"].split(",")
    require(len(lexemes) == target.length and all(lexeme.isdigit() for lexeme in lexemes),
            f"{target.sample_id} manifest structure-token count/format mismatch")
    raw_indices = [int(lexeme) for lexeme in lexemes]
    require(all(0 <= index < 8192 for index in raw_indices),
            f"{target.sample_id} manifest LFQ index out of range")
    return lexemes, raw_indices


def manifest_token_sha256(row: Mapping[str, str], target: Target) -> str:
    _, raw_indices = manifest_raw_indices(row, target)
    canonical = ",".join(str(index) for index in raw_indices).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def reference_metric_tuple(target: Target, reference: Mapping[str, Any]) -> tuple[float, float, float]:
    require(target.sample_id in reference, f"missing frozen Bit artifact: {target.sample_id}")
    prior = reference[target.sample_id]
    require(prior.sample_id == target.sample_id and prior.length == target.length,
            f"{target.sample_id} frozen Bit artifact identity/length mismatch")
    result = (float(prior.ca_rmsd_to_gt), float(prior.bb_rmsd_to_gt),
              float(prior.bb_tmscore_to_gt))
    require(all(math.isfinite(value) for value in result),
            f"{target.sample_id} frozen Bit metric is non-finite")
    return result


def target_roster(
    dataset: str,
) -> tuple[list[Target], dict[str, Any], dict[int, dict[str, str]], str, str]:
    targets = read_targets(DATASET_FASTAS[dataset])
    eligible = [target for target in targets if target.eligible]
    require(len(eligible) == EXPECTED_COUNTS[dataset],
            f"{dataset} eligible count changed: {len(eligible)}")
    require(MIN_LENGTH == 60 and MAX_LENGTH == 512, "FM length filter changed")
    fm_layout = output_layout(dataset)
    existing = read_manifest(fm_layout.manifest)
    require(list(existing) == list(range(len(targets))),
            f"{dataset} finalized FM manifest order/FASTA count differs")
    for target in targets:
        row = existing[target.fasta_index]
        require(int(row["fasta_index"]) == target.fasta_index and
                row["sample_id"] == target.sample_id and
                int(row["length"]) == target.length and
                row["aa_sequence"] == target.sequence,
                f"{dataset}/{target.sample_id} finalized manifest identity/AA mismatch")
        require(int(row["sample_seed"]) == BASE_SEED + target.fasta_index and
                int(row["base_seed"]) == BASE_SEED and
                int(row["dplm_max_iter"]) == DPLM_MAX_ITER == 100 and
                row["unmasking_strategy"] == "deterministic" and
                row["sampling_strategy"] == "argmax" and
                row["dplm_checkpoint"] == DPLM_CHECKPOINT,
                f"{dataset}/{target.sample_id} frozen generation provenance mismatch")
        if target.eligible:
            require(row["status"] == "success" and
                    row["generation_success"].lower() == "true",
                    f"finalized FM target was not successful: {target.sample_id}")
            manifest_raw_indices(row, target)
    reference = collect_top_samples(REFERENCE_EVAL_DIRS[dataset], f"{dataset} Local Bit")
    require(len(reference) == EXPECTED_COUNTS[dataset] and
            set(reference) == {target.sample_id for target in eligible},
            f"{dataset} eligible/reference Bit target ID set mismatch")
    aggregate = read_aggregate(REFERENCE_EVAL_DIRS[dataset], f"{dataset} Local Bit")
    validate_aggregate(f"{dataset} Local Bit", reference, aggregate)

    paired_path = PAIRED_REFERENCE_CSVS[dataset]
    require(paired_path.is_file(), f"finalized FM paired artifact missing: {paired_path}")
    with paired_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "length", "bit_ca_rmsd", "bit_bb_rmsd", "bit_bb_tmscore"}
        require(required.issubset(reader.fieldnames or ()),
                "finalized FM paired Bit columns missing")
        paired_rows = list(reader)
    paired = {row["sample_id"]: row for row in paired_rows}
    require(len(paired_rows) == len(paired) == len(eligible) and
            set(paired) == set(reference),
            f"{dataset} finalized FM paired Bit target IDs differ")
    reference_digest = hashlib.sha256()
    for target in eligible:
        bit = reference_metric_tuple(target, reference)
        artifact_path = reference[target.sample_id].path
        require(artifact_path.parent.name == target.sample_id and artifact_path.is_file(),
                f"{target.sample_id} finalized Bit top_sample.csv identity differs")
        with artifact_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 1 and "sample_path" in rows[0],
                f"{target.sample_id} finalized Bit sample path missing")
        sample_path = Path(rows[0]["sample_path"])
        require(sample_path.parent.name == target.sample_id and sample_path.is_file(),
                f"{target.sample_id} finalized Bit prediction artifact missing")
        paired_row = paired[target.sample_id]
        require(int(paired_row["length"]) == target.length and
                (float(paired_row["bit_ca_rmsd"]),
                 float(paired_row["bit_bb_rmsd"]),
                 float(paired_row["bit_bb_tmscore"])) == bit,
                f"{target.sample_id} frozen Bit metrics differ from finalized FM paired artifact")
        reference_digest.update(
            f"{target.fasta_index}:{target.sample_id}:{sha256_file(artifact_path)}\n".encode()
        )
    return eligible, reference, existing, sha256_file(fm_layout.manifest), reference_digest.hexdigest()

def load_official_metadata(dataset: str) -> Any:
    csv_path, data_dir = GT_METADATA[dataset]
    require(csv_path.is_file() and data_dir.is_dir(), "official GT metadata unavailable")
    metadata = EvalRunner.load_metadata(
        None, SimpleNamespace(csv_path=str(csv_path), data_dir=str(data_dir))
    )
    require(metadata is not None, f"{dataset} official GT metadata load failed")
    return metadata


def output_ddpm_path(dataset: str, target: Target) -> Path:
    return OUTPUT_ROOT / dataset / "matched_ddpm_alpha050/folding/pdb" / f"{target.sample_id}.pdb"


def save_ddpm_pdb_atomically(
    tokenizer: torch.nn.Module, output: Mapping[str, Any],
    path: Path, target: Target,
) -> None:
    """Use the finalized tokenizer PDB writer, committing only the DDPM branch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ddpm-pdb-", dir=path.parent) as temp_dir:
        tokenizer.output_to_pdb(dict(output), output_dir=temp_dir)
        temporary = Path(temp_dir) / f"{target.sample_id}.pdb"
        require(valid_pdb(temporary, target.length),
                f"invalid temporary DDPM PDB: {target.sample_id}")
        os.replace(temporary, path)
    require(valid_pdb(path, target.length), f"invalid DDPM PDB: {target.sample_id}")

def metric_triplet(path: Path, target: Target, gt_metadata: Any) -> tuple[float, float, float]:
    require(valid_pdb(path, target.length), f"invalid prediction PDB: {path}")
    # The official forward-folding evaluator resolves GT by exact pdb_name and
    # passes GT N/CA/C coordinates to process_folded_outputs.  Reuse both APIs.
    gt = load_pdb_by_name(target.sample_id, gt_metadata)
    positions = gt["all_atom_positions"]
    require(tuple(positions.shape) == (target.length, 37, 3),
            f"{target.sample_id} official GT coordinate shape mismatch")
    true_bb = positions[..., :3, :].reshape(-1, 3).cpu().numpy()
    metrics = evaluator_utils.process_folded_outputs(str(path), None, true_bb)
    require(len(metrics) == 1, f"{target.sample_id} expected one official metric row")
    row = metrics.iloc[0]
    result = (
        float(row["ca_rmsd_to_gt"]),
        float(row["bb_rmsd_to_gt"]),
        float(row["bb_tmscore_to_gt"]),
    )
    require(all(math.isfinite(value) for value in result),
            f"{target.sample_id} official metric is non-finite")
    return result


def check_bit_reference(
    dataset: str, target: Target, observed: tuple[float, float, float], reference: Any
) -> None:
    """Resume rows must contain the exact numbers read from the frozen Bit CSV."""
    expected = reference_metric_tuple(target, reference)
    for label, current, previous in zip(("CA RMSD", "backbone RMSD", "TM-score"),
                                        observed, expected):
        require(current == previous,
                f"{dataset}/{target.sample_id} frozen Bit artifact {label} changed: "
                f"row={current:.16g}, artifact={previous:.16g}")

def evaluate_target(
    dataset: str,
    target: Target,
    manifest_row: Mapping[str, str],
    dplm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    ddpm: torch.nn.Module,
    schedule: Any,
    gt_metadata: Any,
    reference: Any,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample_seed = BASE_SEED + target.fasta_index
    require(manifest_row["sample_id"] == target.sample_id and
            int(manifest_row["fasta_index"]) == target.fasta_index and
            manifest_row["aa_sequence"] == target.sequence,
            f"{target.sample_id} frozen manifest row changed")
    try:
        aatype = torch.tensor(du.seq_to_aatype(target.sequence), dtype=torch.long)
    except ValueError as exc:
        raise AssertionError(f"unsupported amino acid in {target.sample_id}") from exc
    require(tuple(aatype.shape) == (target.length,), "target aatype length changed")
    ddpm_path = output_ddpm_path(dataset, target)
    bit_metrics = reference_metric_tuple(target, reference)
    token_digest = manifest_token_sha256(manifest_row, target)

    with torch.inference_mode():
        # The finalized FM generator stored batch-decoded structure-half
        # vocabulary lexemes. They denote raw LFQ indices, not combined IDs.
        # Use the actual tokenizer mapping and DPLM struct_vocab_offset rule.
        input_tokens, partial_mask = build_folding_input(dplm, aatype, device)
        lexemes, raw_indices = manifest_raw_indices(manifest_row, target)
        offset = int(dplm.struct_vocab_offset)
        combined_ids: list[int] = []
        for position, (lexeme, raw_index) in enumerate(zip(lexemes, raw_indices)):
            require(lexeme in dplm.tokenizer._token_to_id,
                    f"{target.sample_id} unknown manifest lexeme at {position}: {lexeme}")
            token_id = int(dplm.tokenizer._token_to_id[lexeme])
            require(token_id - offset == raw_index,
                    f"{target.sample_id} manifest lexeme/combined-ID mismatch at {position}")
            combined_ids.append(token_id)
        final_tokens = input_tokens.clone()
        final_tokens[:, 1:target.length + 1] = torch.tensor(
            combined_ids, dtype=final_tokens.dtype, device=device
        ).view(1, target.length)
        require(tuple(final_tokens.shape) == (1, 2 * (target.length + 2)),
                "frozen final tokens misaligned")
        require(torch.equal(final_tokens[partial_mask], input_tokens[partial_mask]),
                "frozen manifest replacement changed GT AA-side tokens")
        structure_half, _ = final_tokens.chunk(2, dim=1)
        require(structure_half[0, 1:-1].long().tolist() == combined_ids,
                "final structure residues differ from frozen manifest")

        predicted_z_quant, residue_mask = reconstruct_prediction_latent(dplm, final_tokens)
        expected_shape = (1, target.length, EXPECTED_CODEBOOK_DIM)
        require(tuple(predicted_z_quant.shape) == expected_shape,
                "frozen predicted z_quant shape mismatch")
        require(tuple(residue_mask.shape) == (1, target.length) and
                bool(residue_mask.bool().all()), "frozen residue mask misaligned")
        require(bool(torch.isfinite(predicted_z_quant).all()) and
                bool(((predicted_z_quant == -1) | (predicted_z_quant == 1)).all()),
                "frozen LFQ z_quant must be finite ±1 bits")

        # One EXTRA frozen forward on the reconstructed final tokens.
        # extract_final_token_condition drops the embedding state and returns
        # only structure-side residue-aligned 33 Transformer outputs.
        hidden_states, extra_forward_seconds = timed_call(
            device, lambda: extract_final_token_condition(dplm, final_tokens, target.length)
        )
        require(len(hidden_states) == EXPECTED_NUM_LAYERS == 33 and
                all(tuple(state.shape) == (1, target.length, EXPECTED_HIDDEN_SIZE)
                    for state in hidden_states), "frozen final-token hidden alignment changed")

        r_hat, sampling_metadata, sampling_seconds, rng_unchanged = run_sampler(
            ddpm, schedule, predicted_z_quant, hidden_states, residue_mask,
            sample_seed, device,
        )
        require(rng_unchanged, "local DDPM Generator changed global RNG")
        require(sampling_metadata["nfe"] == NUM_TIMESTEPS == 1000 and
                sampling_metadata["timesteps"] == tuple(range(1000, 0, -1)),
                "full ancestral DDPM T/NFE/order changed")
        require(tuple(r_hat.shape) == expected_shape and bool(torch.isfinite(r_hat).all()),
                "DDPM residual invalid")
        z_refined = predicted_z_quant + FINAL_ALPHA * r_hat
        require(tuple(z_refined.shape) == expected_shape and bool(torch.isfinite(z_refined).all()),
                "DDPM alpha-0.50 latent invalid")

        decode_started = time.perf_counter()
        ddpm_decoded = tokenizer.detokenize(z_refined, res_mask=residue_mask)
        target_aatype = aatype.view(1, target.length).to(device)
        ddpm_output = finalize_decoder_output(
            "DDPM-refined", ddpm_decoded, target_aatype, residue_mask,
            target.sample_id, target.length,
        )
        save_ddpm_pdb_atomically(tokenizer, ddpm_output, ddpm_path, target)
        synchronize(device)
        ddpm_metrics = metric_triplet(ddpm_path, target, gt_metadata)
        decode_metric_seconds = time.perf_counter() - decode_started

    require(all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "inference accumulated parameter gradients")
    return {
        "benchmark": dataset,
        "target_id": target.sample_id,
        "target_index": target.fasta_index,
        "length": target.length,
        "seed": sample_seed,
        "manifest_token_sha256": token_digest,
        "bit_ca_rmsd": bit_metrics[0],
        "ddpm_ca_rmsd": ddpm_metrics[0],
        "ca_improvement": bit_metrics[0] - ddpm_metrics[0],
        "bit_backbone_rmsd": bit_metrics[1],
        "ddpm_backbone_rmsd": ddpm_metrics[1],
        "backbone_improvement": bit_metrics[1] - ddpm_metrics[1],
        "bit_tm_score": bit_metrics[2],
        "ddpm_tm_score": ddpm_metrics[2],
        "tm_improvement": ddpm_metrics[2] - bit_metrics[2],
        "ddpm_nfe": sampling_metadata["nfe"],
        "alpha": FINAL_ALPHA,
        "extra_forward_seconds": extra_forward_seconds,
        "ddpm_sampling_seconds": sampling_seconds,
        "decode_metric_seconds": decode_metric_seconds,
        "target_total_seconds": time.perf_counter() - started,
    }

def protocol_metadata(
    dataset: str, eligible: Sequence[Target], fm_manifest_sha256: str,
    checkpoint_sha256: str, alpha_sha256: str, reference_digest: str,
) -> dict[str, Any]:
    fasta = DATASET_FASTAS[dataset]
    reference_dir = REFERENCE_EVAL_DIRS[dataset]
    paired_csv = PAIRED_REFERENCE_CSVS[dataset]
    manifest_path = output_layout(dataset).manifest
    metadata_csv, metadata_data_dir = GT_METADATA[dataset]
    return {
        "benchmark": dataset,
        "expected_benchmark_counts": EXPECTED_COUNTS.copy(),
        "eligible_targets": [
            {"target_index": target.fasta_index, "target_id": target.sample_id,
             "length": target.length, "sequence": target.sequence}
            for target in eligible
        ],
        "eligible_count": EXPECTED_COUNTS[dataset],
        "fasta_path": str(fasta),
        "fasta_sha256": sha256_file(fasta),
        "prediction_source": PREDICTION_SOURCE,
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": fm_manifest_sha256,
        "fm_generation_manifest_path": str(manifest_path),
        "fm_generation_manifest_sha256": fm_manifest_sha256,
        "new_dplm_generation": False,
        "frozen_manifest_tokens_reused": True,
        "dplm_generation": "not performed; frozen manifest tokens reused",
        "fairness": (
            "The final matched DDPM evaluation reuses the exact frozen DPLM "
            "structure-token predictions from the finalized Local Bit/FM generation "
            "manifests. This avoids regeneration variability and ensures that "
            "Local Bit, FM, and Matched DDPM are conditioned on the same "
            "base-model prediction for each target."
        ),
        "bit_baseline_source": "finalized Local Bit top_sample.csv",
        "reference_bit_eval_dir": str(reference_dir),
        "reference_bit_aggregate_sha256": sha256_file(reference_dir / "forward_fold_metrics.csv"),
        "reference_bit_top_samples_sha256": reference_digest,
        "reference_bit_fm_paired_csv": str(paired_csv),
        "reference_bit_fm_paired_sha256": sha256_file(paired_csv),
        "gt_metadata_csv": str(metadata_csv),
        "gt_metadata_data_dir": str(metadata_data_dir),
        "gt_metadata_csv_sha256": sha256_file(metadata_csv),
        "ddpm_checkpoint": str(DDPM_CHECKPOINT),
        "ddpm_checkpoint_sha256": checkpoint_sha256,
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "best_step": EXPECTED_BEST_STEP,
        "best_val_loss": 0.150297817,
        "ddpm_parameters": EXPECTED_DDPM_PARAMETERS,
        "alpha": FINAL_ALPHA,
        "alpha_selection_artifact": str(ALPHA_SELECTION_ARTIFACT),
        "alpha_selection_artifact_sha256": alpha_sha256,
        "sampler": "full ancestral DDPM",
        "T": NUM_TIMESTEPS,
        "NFE": NUM_TIMESTEPS,
        "dplm_checkpoint": DPLM_CHECKPOINT,
        "generation_max_iter_provenance": DPLM_MAX_ITER,
        "unmasking_strategy_provenance": "deterministic",
        "sampling_strategy_provenance": "argmax",
        "extra_frozen_dplm_forward": True,
        "condition_transformer_layers": EXPECTED_NUM_LAYERS,
        "condition_structure_side_only": True,
        "seed_policy": "42 + fasta_index (zero-based FASTA record index)",
        "base_seed": BASE_SEED,
        "length_filter": [MIN_LENGTH, MAX_LENGTH],
        "metric_route": "official evaluator load_pdb_by_name + process_folded_outputs",
        "metric_source_sha256": {str(path): sha256_file(path) for path in METRIC_SOURCE_PATHS},
        "CAMEO_PDB_date_used_for_tuning": False,
    }

def load_or_create_metadata(path: Path, expected: dict[str, Any], result_path: Path) -> None:
    require(not result_path.exists() or path.is_file(),
            f"partial result exists without protocol metadata: {result_path}")
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored.get("prediction_source") != PREDICTION_SOURCE:
            raise RuntimeError(
                f"legacy/regeneration protocol metadata exists: {path}. "
                "Do not mix old partial outputs with frozen-manifest evaluation; "
                "archive or remove them manually after inspection."
            )
        require(stored == expected, f"resume protocol mismatch: {path}")
    else:
        atomic_write_text(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")

def validate_row(
    row: Mapping[str, Any], target: Target, dataset: str, reference: Any,
    manifest_row: Mapping[str, str],
) -> None:
    require(row["benchmark"] == dataset and row["target_id"] == target.sample_id,
            "partial result benchmark/target mismatch")
    require(int(row["target_index"]) == target.fasta_index,
            "partial result FASTA index mismatch")
    require(int(row["length"]) == target.length, "partial result length mismatch")
    require(int(row["seed"]) == BASE_SEED + target.fasta_index,
            "partial result target seed mismatch")
    require(row["manifest_token_sha256"] == manifest_token_sha256(manifest_row, target),
            "partial result frozen manifest token digest mismatch")
    require(float(row["alpha"]) == FINAL_ALPHA, "partial result alpha mismatch")
    require(int(row["ddpm_nfe"]) == NUM_TIMESTEPS, "partial result NFE mismatch")
    for column in PER_TARGET_COLUMNS:
        if column not in {"benchmark", "target_id", "manifest_token_sha256"}:
            require(math.isfinite(float(row[column])), f"partial {column} is non-finite")
    bit = (float(row["bit_ca_rmsd"]), float(row["bit_backbone_rmsd"]),
           float(row["bit_tm_score"]))
    check_bit_reference(dataset, target, bit, reference)
    for baseline, refined, change, sign in (
        ("bit_ca_rmsd", "ddpm_ca_rmsd", "ca_improvement", 1),
        ("bit_backbone_rmsd", "ddpm_backbone_rmsd", "backbone_improvement", 1),
        ("bit_tm_score", "ddpm_tm_score", "tm_improvement", -1),
    ):
        expected = (float(row[baseline]) - float(row[refined])) * sign
        require(math.isclose(float(row[change]), expected, rel_tol=1e-12, abs_tol=1e-12),
                f"partial paired {change} mismatch")


def load_partial(
    path: Path, dataset: str, eligible: Sequence[Target], reference: Any,
    manifest: Mapping[int, Mapping[str, str]],
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == PER_TARGET_COLUMNS,
                f"partial result columns differ: {path}")
        rows = list(reader)
    by_index = {target.fasta_index: target for target in eligible}
    completed: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["target_index"])
        require(index in by_index and index not in completed,
                "partial result has unknown or duplicate target index")
        target = by_index[index]
        validate_row(row, target, dataset, reference, manifest[index])
        ddpm_path = output_ddpm_path(dataset, target)
        require(valid_pdb(ddpm_path, target.length),
                f"partial DDPM PDB missing/corrupt: {target.sample_id}")
        completed[index] = row
    return completed

def summarize(dataset: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == EXPECTED_COUNTS[dataset], f"{dataset} incomplete summary")
    result: dict[str, Any] = {"benchmark": dataset, "num_samples": len(rows)}
    for name in ("ca_rmsd", "backbone_rmsd", "tm_score"):
        for prefix in ("bit", "ddpm"):
            values = [float(row[f"{prefix}_{name}"]) for row in rows]
            result[f"{prefix}_mean_{name}"] = statistics.fmean(values)
            result[f"{prefix}_median_{name}"] = statistics.median(values)
    for name, prefix in (("ca", "ca"), ("backbone", "backbone"), ("tm", "tm")):
        values = [float(row[f"{prefix}_improvement"]) for row in rows]
        result[f"mean_{name}_improvement"] = statistics.fmean(values)
        result[f"median_{name}_improvement"] = statistics.median(values)
        result[f"{name}_improved"] = sum(value > 0 for value in values)
        result[f"{name}_worsened"] = sum(value < 0 for value in values)
        result[f"{name}_tied"] = sum(value == 0 for value in values)
    for name in (
        "extra_forward_seconds", "ddpm_sampling_seconds",
        "decode_metric_seconds", "target_total_seconds",
    ):
        result[name] = sum(float(row[name]) for row in rows)
    return result


def main() -> int:
    args = parse_args()
    require(FINAL_ALPHA == 0.50 and BASE_SEED == 42 and DPLM_MAX_ITER == 100,
            "frozen final DDPM protocol changed")
    require(PREDICTION_SOURCE == "finalized_generation_manifest",
            "final prediction source changed")
    alpha_sha256 = validate_frozen_alpha()
    checkpoint_sha256 = sha256_file(DDPM_CHECKPOINT)
    device = resolve_device(args.device)
    seed_everything(BASE_SEED)

    rosters: dict[str, list[Target]] = {}
    references: dict[str, Any] = {}
    manifests: dict[str, dict[int, dict[str, str]]] = {}
    partials: dict[str, dict[int, dict[str, Any]]] = {}
    gt_metadata: dict[str, Any] = {}
    for dataset in EXPECTED_COUNTS:
        eligible, reference, manifest, fm_manifest_sha256, reference_digest = target_roster(dataset)
        rosters[dataset] = eligible
        references[dataset] = reference
        manifests[dataset] = manifest
        gt_metadata[dataset] = load_official_metadata(dataset)
        result_path = OUTPUT_ROOT / ("cameo_per_target.csv" if dataset == "cameo2022"
                                     else "pdb_date_per_target.csv")
        metadata = protocol_metadata(
            dataset, eligible, fm_manifest_sha256, checkpoint_sha256, alpha_sha256,
            reference_digest,
        )
        # Incompatible metadata from the failed regeneration run is never
        # overwritten or mixed with this frozen-manifest protocol.
        load_or_create_metadata(OUTPUT_ROOT / dataset / METADATA_NAME, metadata, result_path)
        partials[dataset] = load_partial(result_path, dataset, eligible, reference, manifest)

    checks: dict[str, bool] = {}
    dplm, tokenizer = load_models(DPLM_CHECKPOINT, device, checks)
    ddpm, schedule, checkpoint_info = load_ddpm(DDPM_CHECKPOINT, device)
    require(int(checkpoint_info["optimizer_step"]) == EXPECTED_OPTIMIZER_STEP,
            "DDPM optimizer step mismatch")
    require(int(checkpoint_info["best_step"]) == EXPECTED_BEST_STEP,
            "DDPM best step mismatch")
    require(int(checkpoint_info["parameter_count"]) == EXPECTED_DDPM_PARAMETERS,
            "DDPM parameter count mismatch")
    require(all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "parameters have gradients before final evaluation")

    summaries: list[dict[str, Any]] = []
    for dataset in EXPECTED_COUNTS:
        eligible = rosters[dataset]
        completed = partials[dataset]
        manifest = manifests[dataset]
        result_path = OUTPUT_ROOT / ("cameo_per_target.csv" if dataset == "cameo2022"
                                     else "pdb_date_per_target.csv")
        print(f"[{dataset}] frozen-manifest eligible={len(eligible)} "
              f"resume_completed={len(completed)}")
        for ordinal, target in enumerate(eligible, start=1):
            if target.fasta_index in completed:
                print(f"[{dataset} {ordinal}/{len(eligible)}] resume-skip {target.sample_id}")
                continue
            row = evaluate_target(
                dataset, target, manifest[target.fasta_index],
                dplm, tokenizer, ddpm, schedule,
                gt_metadata[dataset], references[dataset], device,
            )
            validate_row(row, target, dataset, references[dataset],
                         manifest[target.fasta_index])
            completed[target.fasta_index] = row
            ordered = [completed[item.fasta_index] for item in eligible
                       if item.fasta_index in completed]
            atomic_write_csv(result_path, PER_TARGET_COLUMNS, ordered)
            print(f"[{dataset} {ordinal}/{len(eligible)}] completed "
                  f"{target.sample_id} seed={row['seed']} NFE={row['ddpm_nfe']} "
                  f"manifest_token_sha256={row['manifest_token_sha256']}")

        require(len(completed) == EXPECTED_COUNTS[dataset], f"{dataset} incomplete")
        rows = [completed[target.fasta_index] for target in eligible]
        summary = summarize(dataset, rows)
        old = read_aggregate(REFERENCE_EVAL_DIRS[dataset], f"{dataset} Local Bit")
        require(math.isclose(summary["bit_mean_backbone_rmsd"], old.mean_bb_rmsd,
                             rel_tol=1e-6, abs_tol=1e-6),
                f"{dataset} frozen Bit mean backbone RMSD differs from final reference")
        require(math.isclose(summary["bit_mean_tm_score"], old.mean_bb_tmscore,
                             rel_tol=1e-6, abs_tol=1e-6),
                f"{dataset} frozen Bit mean TM-score differs from final reference")
        summaries.append(summary)
        atomic_write_csv(OUTPUT_ROOT / "summary.csv", SUMMARY_COLUMNS, summaries)
        print(f"[{dataset}] complete N={len(rows)} frozen Bit artifact consistency PASS")

    require({row["benchmark"]: row["num_samples"] for row in summaries} == EXPECTED_COUNTS,
            "final CAMEO/PDB-date counts mismatch")
    require(FINAL_ALPHA == 0.50 and schedule.num_timesteps == 1000,
            "frozen alpha or sampler horizon changed")
    require(all_grads_none(dplm) and all_grads_none(tokenizer) and all_grads_none(ddpm),
            "parameters have gradients after final evaluation")
    lines = [
        "LOCAL MATCHED RESIDUAL DDPM FINAL EVALUATION",
        "prediction source = finalized Local Bit/FM generation manifests",
        "new DPLM generation = False; frozen manifest tokens reused = True",
        "alpha = 0.50 (frozen internal-validation selection)",
        "sampler = full ancestral DDPM; T=1000; NFE=1000",
        "CAMEO/PDB-date used for tuning = False",
    ]
    for row in summaries:
        lines.extend([
            f"\n[{row['benchmark']}] N={row['num_samples']}",
            f"Bit mean/median CA RMSD: {row['bit_mean_ca_rmsd']:.9g} / {row['bit_median_ca_rmsd']:.9g}",
            f"DDPM mean/median CA RMSD: {row['ddpm_mean_ca_rmsd']:.9g} / {row['ddpm_median_ca_rmsd']:.9g}",
            f"Bit mean/median backbone RMSD: {row['bit_mean_backbone_rmsd']:.9g} / {row['bit_median_backbone_rmsd']:.9g}",
            f"DDPM mean/median backbone RMSD: {row['ddpm_mean_backbone_rmsd']:.9g} / {row['ddpm_median_backbone_rmsd']:.9g}",
            f"Bit mean/median TM-score: {row['bit_mean_tm_score']:.9g} / {row['bit_median_tm_score']:.9g}",
            f"DDPM mean/median TM-score: {row['ddpm_mean_tm_score']:.9g} / {row['ddpm_median_tm_score']:.9g}",
            f"mean/median CA improvement: {row['mean_ca_improvement']:.9g} / {row['median_ca_improvement']:.9g}",
            f"mean/median backbone improvement: {row['mean_backbone_improvement']:.9g} / {row['median_backbone_improvement']:.9g}",
            f"mean/median TM improvement: {row['mean_tm_improvement']:.9g} / {row['median_tm_improvement']:.9g}",
            f"CA improved/worsened/tied: {row['ca_improved']}/{row['ca_worsened']}/{row['ca_tied']}",
            f"backbone improved/worsened/tied: {row['backbone_improved']}/{row['backbone_worsened']}/{row['backbone_tied']}",
            f"TM improved/worsened/tied: {row['tm_improved']}/{row['tm_worsened']}/{row['tm_tied']}",
            f"extra DPLM forward seconds: {row['extra_forward_seconds']:.6f}",
            f"DDPM sampling seconds: {row['ddpm_sampling_seconds']:.6f}",
            f"decode/metric seconds: {row['decode_metric_seconds']:.6f}",
            f"target total seconds: {row['target_total_seconds']:.6f}",
            "frozen Bit baseline artifact consistency: PASS",
        ])
    report = "\n".join(lines) + "\n"
    atomic_write_text(OUTPUT_ROOT / "summary.txt", report)
    print(report)
    print(f"output root: {OUTPUT_ROOT}")
    print("DDPM_FINAL_EVALUATION_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

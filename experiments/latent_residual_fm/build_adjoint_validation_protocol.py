#!/usr/bin/env python3
"""Freeze disjoint AM development and locked-confirmation validation sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_validation_protocol")
SPLIT_RELATIVE = Path(
    "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv"
)
ALPHA_INDICES_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
    "internal_val_alpha_sweep_100_indices.csv"
)
PILOT_VALIDATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_validation_indices.csv"
)
REPLICATION_VALIDATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_replication/"
    "replication_validation_indices.csv"
)
STEP50_VALIDATION_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_50step_replication/"
    "validation_indices_64.csv"
)
PILOT_TRAIN_RELATIVE = Path(
    "experiments/latent_residual_fm/runs/am_lambda_pilot/pilot_train_indices.csv"
)
EXPECTED_SOURCE_VALIDATION = 2048
SET_SIZE = 256
PER_BIN = 64
DEVELOPMENT_SEED = 4242
LOCKED_SEED = 4343
LENGTH_BINS = ((50, 128), (129, 256), (257, 384), (385, 512))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_csv", default=str(SPLIT_RELATIVE))
    parser.add_argument("--alpha_indices", default=str(ALPHA_INDICES_RELATIVE))
    parser.add_argument(
        "--pilot_validation", default=str(PILOT_VALIDATION_RELATIVE)
    )
    parser.add_argument(
        "--replication_validation", default=str(REPLICATION_VALIDATION_RELATIVE)
    )
    parser.add_argument(
        "--step50_validation", default=str(STEP50_VALIDATION_RELATIVE)
    )
    parser.add_argument("--pilot_train", default=str(PILOT_TRAIN_RELATIVE))
    parser.add_argument("--output_dir", default=str(OUTPUT_RELATIVE))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def source_validation(path: Path) -> tuple[dict[str, object], ...]:
    rows = [row for row in read_csv(path) if row.get("split") == "val"]
    require(
        len(rows) == EXPECTED_SOURCE_VALIDATION,
        f"source validation N={len(rows)}, expected {EXPECTED_SOURCE_VALIDATION}",
    )
    resolved: list[dict[str, object]] = []
    for validation_index, row in enumerate(rows):
        length = int(row["length"])
        length_bin = bin_label(length)
        resolved.append(
            {
                "validation_index": validation_index,
                "sample_id": row["sample_id"],
                "length": length,
                "length_bin": length_bin,
                "shard_file": row["shard_file"],
                "index_in_shard": int(row["index_in_shard"]),
            }
        )
    require(
        len({str(row["sample_id"]) for row in resolved}) == len(resolved),
        "duplicate sample ID in source validation",
    )
    return tuple(resolved)


def bin_label(length: int) -> str:
    for lower, upper in LENGTH_BINS:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    raise AssertionError(f"validation length {length} is outside protocol bins")


def alpha_ids(
    path: Path, source: Sequence[dict[str, object]]
) -> set[str]:
    rows = read_csv(path)
    require(len(rows) == 100, f"alpha index N={len(rows)}, expected 100")
    ordered = sorted(rows, key=lambda row: int(row["selection_order"]))
    require(
        [int(row["selection_order"]) for row in ordered] == list(range(100)),
        "alpha selection_order is incomplete",
    )
    indexes = [int(row["val_index"]) for row in ordered]
    require(len(set(indexes)) == 100, "duplicate alpha validation index")
    require(
        all(0 <= index < len(source) for index in indexes),
        "alpha validation index out of range",
    )
    return {str(source[index]["sample_id"]) for index in indexes}


def indexed_validation_ids(
    path: Path,
    source: Sequence[dict[str, object]],
    expected_n: int,
) -> set[str]:
    rows = read_csv(path)
    require(len(rows) == expected_n, f"{path} N={len(rows)}, expected {expected_n}")
    identifiers: set[str] = set()
    for row in rows:
        validation_index = int(row["split_index"])
        require(
            0 <= validation_index < len(source),
            f"validation index out of range in {path}",
        )
        reference = source[validation_index]
        require(row["split"] == "val", f"non-validation row in {path}")
        require(
            row["sample_id"] == reference["sample_id"],
            f"sample/index mismatch in {path}: {row['sample_id']}",
        )
        require(
            int(row["length"]) == reference["length"],
            f"length mismatch in {path}: {row['sample_id']}",
        )
        identifiers.add(row["sample_id"])
    require(len(identifiers) == expected_n, f"duplicate sample ID in {path}")
    return identifiers


def train_ids(path: Path, expected_n: int) -> set[str]:
    rows = read_csv(path)
    require(len(rows) == expected_n, f"pilot train N={len(rows)}, expected {expected_n}")
    require(all(row["split"] == "train" for row in rows), "non-train pilot row")
    identifiers = {row["sample_id"] for row in rows}
    require(len(identifiers) == expected_n, "duplicate pilot train ID")
    return identifiers


def select_stratified(
    source: Sequence[dict[str, object]],
    excluded: set[str],
    seed: int,
) -> tuple[dict[str, object], ...]:
    rng = random.Random(seed)
    chosen: list[dict[str, object]] = []
    for lower, upper in LENGTH_BINS:
        label = f"{lower}-{upper}"
        candidates = sorted(
            (
                row
                for row in source
                if row["length_bin"] == label
                and str(row["sample_id"]) not in excluded
            ),
            key=lambda row: (str(row["sample_id"]), int(row["validation_index"])),
        )
        require(
            len(candidates) >= PER_BIN,
            f"bin {label} has only {len(candidates)} unused candidates",
        )
        sampled = rng.sample(candidates, PER_BIN)
        chosen.extend(
            sorted(sampled, key=lambda row: int(row["validation_index"]))
        )
    require(len(chosen) == SET_SIZE, f"selected N={len(chosen)}, expected {SET_SIZE}")
    require(
        len({str(row["sample_id"]) for row in chosen}) == SET_SIZE,
        "duplicate selected sample",
    )
    return tuple(chosen)


def distribution(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    lengths = [int(row["length"]) for row in rows]
    counts = Counter(str(row["length_bin"]) for row in rows)
    return {
        "total_n": len(rows),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": statistics.fmean(lengths),
        "length_median": statistics.median(lengths),
        "bin_counts": {
            f"{lower}-{upper}": counts[f"{lower}-{upper}"]
            for lower, upper in LENGTH_BINS
        },
    }


def write_selection(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "validation_index",
        "protein_id",
        "length",
        "length_bin",
        "shard_file",
        "index_in_shard",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "validation_index": row["validation_index"],
                    "protein_id": row["sample_id"],
                    "length": row["length"],
                    "length_bin": row["length_bin"],
                    "shard_file": row["shard_file"],
                    "index_in_shard": row["index_in_shard"],
                }
            )


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def overlap_audit(
    development: set[str],
    locked: set[str],
    alpha: set[str],
    pilot: set[str],
    replication: set[str],
    step50: set[str],
    train: set[str],
) -> dict[str, int]:
    return {
        "development_vs_locked": len(development & locked),
        "development_vs_alpha100": len(development & alpha),
        "development_vs_lambda_pilot16": len(development & pilot),
        "development_vs_replication32": len(development & replication),
        "development_vs_50step64": len(development & step50),
        "locked_vs_alpha100": len(locked & alpha),
        "locked_vs_lambda_pilot16": len(locked & pilot),
        "locked_vs_replication32": len(locked & replication),
        "locked_vs_50step64": len(locked & step50),
        "development_vs_pilot_train16": len(development & train),
        "locked_vs_pilot_train16": len(locked & train),
        "development_internal_duplicates": SET_SIZE - len(development),
        "locked_internal_duplicates": SET_SIZE - len(locked),
    }


def summary_line(name: str, summary: dict[str, object]) -> str:
    return (
        f"{name}: N={summary['total_n']} "
        f"range={summary['length_min']}-{summary['length_max']} "
        f"mean={float(summary['length_mean']):.6f} "
        f"median={float(summary['length_median']):.6f} "
        f"bins={summary['bin_counts']}"
    )


def readme_text(
    development_sha256: str,
    locked_sha256: str,
    creation_timestamp: str,
) -> str:
    return f"""# Adjoint Matching validation protocol

Frozen at: `{creation_timestamp}`

## AM DEVELOPMENT 256

- May be used repeatedly for AM hyperparameter and design comparisons.
- Contains 64 proteins from each frozen length bin.
- Seed: `{DEVELOPMENT_SEED}`.
- SHA256: `{development_sha256}`.

## AM LOCKED CONFIRMATION 256

- Model inference is prohibited before the complete AM protocol is frozen.
- It must not be used to choose lambda, learning rate, update count, clipping,
  or timestep policy.
- Evaluate it exactly once after the final AM checkpoint and protocol are frozen.
- Seed: `{LOCKED_SEED}`.
- SHA256: `{locked_sha256}`.

## Selection procedure

All previously used internal-validation IDs listed in `metadata.json` were
excluded. Within each length bin, remaining candidates were sorted by protein
ID and validation index. One `random.Random(seed)` instance sampled 64 members
per bin in bin order. The locked set additionally excluded the development set.

## External test policy

CAMEO and PDB-date remain untouched test benchmarks and must not be used for AM
development or hyperparameter selection.

This freezes only the validation protocol. It does not freeze the AM algorithm,
hyperparameters, training protocol, or checkpoint.
"""


def freeze_text(development_sha256: str, locked_sha256: str) -> str:
    return f"""ADJOINT MATCHING VALIDATION PROTOCOL FREEZE

AM development 256 SHA256: {development_sha256}
AM locked confirmation 256 SHA256: {locked_sha256}

The locked set must remain unopened to model inference until the complete AM
protocol is frozen. It may then be evaluated exactly once.

Existing CAMEO and PDB-date benchmarks remain test-only.

This marker freezes the validation protocol only. It is not an AM algorithm or
hyperparameter freeze and it is not SPEC_FREEZE.
"""


def main() -> int:
    args = parse_args()
    split_path = Path(args.split_csv)
    alpha_path = Path(args.alpha_indices)
    pilot_path = Path(args.pilot_validation)
    replication_path = Path(args.replication_validation)
    step50_path = Path(args.step50_validation)
    train_path = Path(args.pilot_train)
    output_dir = Path(args.output_dir)
    input_paths = (
        split_path,
        alpha_path,
        pilot_path,
        replication_path,
        step50_path,
        train_path,
    )
    print("ADJOINT_VALIDATION_PROTOCOL")
    try:
        require(
            output_dir.resolve().is_relative_to(
                (REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()
            ),
            "output directory must remain below am_validation_protocol",
        )
        input_hashes_before = {str(path): sha256_file(path) for path in input_paths}
        source = source_validation(split_path)
        alpha = alpha_ids(alpha_path, source)
        pilot_validation = indexed_validation_ids(pilot_path, source, 16)
        replication_validation = indexed_validation_ids(
            replication_path, source, 32
        )
        step50_validation = indexed_validation_ids(step50_path, source, 64)
        train = train_ids(train_path, 16)
        source_ids = {str(row["sample_id"]) for row in source}
        require(not (source_ids & train), "pilot train IDs occur in validation split")

        previously_used = (
            alpha | pilot_validation | replication_validation | step50_validation
        )
        development_rows = select_stratified(
            source, previously_used | train, DEVELOPMENT_SEED
        )
        development_ids = {str(row["sample_id"]) for row in development_rows}
        locked_rows = select_stratified(
            source, previously_used | train | development_ids, LOCKED_SEED
        )
        locked_ids = {str(row["sample_id"]) for row in locked_rows}
        audits = overlap_audit(
            development_ids,
            locked_ids,
            alpha,
            pilot_validation,
            replication_validation,
            step50_validation,
            train,
        )
        require(all(value == 0 for value in audits.values()), f"overlap audit: {audits}")

        source_summary = distribution(source)
        development_summary = distribution(development_rows)
        locked_summary = distribution(locked_rows)
        require(
            development_summary["bin_counts"]
            == {f"{a}-{b}": PER_BIN for a, b in LENGTH_BINS},
            "development bin audit failed",
        )
        require(
            locked_summary["bin_counts"]
            == {f"{a}-{b}": PER_BIN for a, b in LENGTH_BINS},
            "locked bin audit failed",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        development_path = output_dir / "am_development_256.csv"
        locked_path = output_dir / "am_locked_confirmation_256.csv"
        write_selection(development_path, development_rows)
        write_selection(locked_path, locked_rows)
        development_sha256 = sha256_file(development_path)
        locked_sha256 = sha256_file(locked_path)
        creation_timestamp = datetime.now(timezone.utc).isoformat()
        (output_dir / "README.md").write_text(
            readme_text(development_sha256, locked_sha256, creation_timestamp),
            encoding="utf-8",
        )
        (output_dir / "VALIDATION_PROTOCOL_FREEZE.txt").write_text(
            freeze_text(development_sha256, locked_sha256), encoding="utf-8"
        )

        input_hashes_after = {str(path): sha256_file(path) for path in input_paths}
        require(input_hashes_before == input_hashes_after, "input artifact changed")
        metadata: dict[str, object] = {
            "status": "PASS",
            "creation_timestamp_utc": creation_timestamp,
            "source_validation_split": str(split_path),
            "source_validation_n": len(source),
            "source_validation_summary": source_summary,
            "excluded_artifacts": {
                "alpha100": str(alpha_path),
                "lambda_pilot16": str(pilot_path),
                "lambda_replication32": str(replication_path),
                "50step_replication64": str(step50_path),
                "am_pilot_train16_overlap_assertion": str(train_path),
            },
            "excluded_sample_counts": {
                "alpha100": len(alpha),
                "lambda_pilot16": len(pilot_validation),
                "lambda_replication32": len(replication_validation),
                "50step_replication64": len(step50_validation),
                "unique_previously_used_validation": len(previously_used),
            },
            "development_seed": DEVELOPMENT_SEED,
            "locked_seed": LOCKED_SEED,
            "selection_algorithm": (
                "sort candidates by (protein_id, validation_index); use one "
                "random.Random(seed) across bins in declared bin order; sample 64/bin"
            ),
            "length_bins": [
                {"lower": lower, "upper": upper, "count_per_set": PER_BIN}
                for lower, upper in LENGTH_BINS
            ],
            "development": development_summary,
            "locked_confirmation": locked_summary,
            "development_sha256": development_sha256,
            "locked_sha256": locked_sha256,
            "overlap_audits": audits,
            "all_overlap_audits_zero": True,
            "locked_evaluation_performed": False,
            "model_or_checkpoint_loaded": False,
            "training_performed": False,
            "inference_performed": False,
            "performance_metrics_computed": False,
            "input_artifact_sha256": input_hashes_before,
            "input_artifacts_unchanged": True,
            "validation_protocol_frozen": True,
            "am_algorithm_or_hyperparameters_frozen": False,
        }
        write_json(output_dir / "metadata.json", metadata)

        print(f"source validation: {len(source)}")
        print("previously used excluded:")
        print(f"  alpha: {len(alpha)}")
        print(f"  lambda pilot: {len(pilot_validation)}")
        print(f"  lambda replication: {len(replication_validation)}")
        print(f"  50-step replication: {len(step50_validation)}")
        print(f"  unique excluded total: {len(previously_used)}")
        print(summary_line("source distribution", source_summary))
        print(summary_line("development", development_summary))
        print(f"development SHA256: {development_sha256}")
        print(summary_line("locked", locked_summary))
        print(f"locked SHA256: {locked_sha256}")
        print("overlap audits: ALL ZERO")
        for name, value in audits.items():
            print(f"  {name}: {value}")
        print("locked evaluation performed: NO")
        print("training/inference performed: NO/NO")
        print("overall: ADJOINT_VALIDATION_PROTOCOL_PASS")
        print("ADJOINT_VALIDATION_PROTOCOL_PASS")
        return 0
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        print("locked evaluation performed: NO")
        print("training/inference performed: NO/NO")
        print("overall: ADJOINT_VALIDATION_PROTOCOL_FAIL")
        print("ADJOINT_VALIDATION_PROTOCOL_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

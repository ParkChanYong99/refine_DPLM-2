#!/usr/bin/env python3
"""Create and validate a deterministic length-stratified AFDB latent split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EXPECTED_TOTAL = 176_733
DEFAULT_VAL_SIZE = 2_048
LENGTH_BINS = ((50, 128), (129, 256), (257, 384), (385, 512))
OUTPUT_COLUMNS = (
    "sample_id",
    "processed_path",
    "archive_member",
    "shard_file",
    "index_in_shard",
    "length",
    "split",
    "length_bin",
)
REQUIRED_MANIFEST_COLUMNS = set(OUTPUT_COLUMNS[:6]) | {"status"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def length_bin(length: int) -> str:
    for lower, upper in LENGTH_BINS:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    raise ValueError(f"length {length} is outside the configured bins")


def read_saved_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        for line_number, source in enumerate(reader, start=2):
            if source["status"] != "SAVED":
                continue
            try:
                length = int(source["length"])
                index = int(source["index_in_shard"])
            except ValueError as exc:
                raise ValueError(
                    f"manifest line {line_number} has a non-integer length or index"
                ) from exc
            rows.append(
                {
                    "sample_id": source["sample_id"],
                    "processed_path": source["processed_path"],
                    "archive_member": source["archive_member"],
                    "shard_file": source["shard_file"],
                    "index_in_shard": index,
                    "length": length,
                    "length_bin": length_bin(length),
                }
            )
    return rows


def duplicate_count(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def allocate_validation(bin_counts: np.ndarray, val_size: int) -> np.ndarray:
    if val_size <= 0 or val_size >= int(bin_counts.sum()):
        raise ValueError("val_size must be positive and smaller than the dataset")
    exact = bin_counts.astype(np.float64) * val_size / int(bin_counts.sum())
    allocation = np.floor(exact).astype(np.int64)
    remaining = val_size - int(allocation.sum())
    order = sorted(
        range(len(bin_counts)),
        key=lambda index: (-(exact[index] - allocation[index]), index),
    )
    for index in order[:remaining]:
        allocation[index] += 1
    if int(allocation.sum()) != val_size or bool(np.any(allocation > bin_counts)):
        raise ValueError("validation allocation is inconsistent")
    return allocation


def sample_validation_ids(
    rows: list[dict[str, Any]], val_size: int, seed: int
) -> tuple[set[str], np.ndarray]:
    labels = [f"{lower}-{upper}" for lower, upper in LENGTH_BINS]
    indexes_by_bin = [
        np.asarray(
            [index for index, row in enumerate(rows) if row["length_bin"] == label],
            dtype=np.int64,
        )
        for label in labels
    ]
    bin_counts = np.asarray([len(indexes) for indexes in indexes_by_bin], dtype=np.int64)
    allocation = allocate_validation(bin_counts, val_size)
    rng = np.random.default_rng(seed)
    selected_indexes: list[int] = []
    for indexes, count in zip(indexes_by_bin, allocation):
        chosen = rng.choice(indexes, size=int(count), replace=False)
        selected_indexes.extend(int(index) for index in chosen)
    selected_ids = {rows[index]["sample_id"] for index in selected_indexes}
    if len(selected_ids) != val_size:
        raise ValueError("sampled validation IDs are not unique")
    return selected_ids, allocation


def validate_shard_references(
    rows: list[dict[str, Any]], dataset_dir: Path
) -> tuple[int, int, int]:
    references: dict[str, list[int]] = {}
    invalid_indices = 0
    for row in rows:
        shard_file = str(row["shard_file"])
        references.setdefault(shard_file, []).append(int(row["index_in_shard"]))
        if int(row["index_in_shard"]) < 0:
            invalid_indices += 1

    missing_shards = 0
    for shard_file in sorted(references):
        shard_path = dataset_dir / shard_file
        if not shard_path.is_file():
            missing_shards += 1
            continue
        shard = torch.load(str(shard_path), map_location="cpu", mmap=True)
        if not isinstance(shard, dict) or "num_samples" not in shard:
            raise ValueError(f"shard has no valid num_samples field: {shard_path}")
        num_samples = int(shard["num_samples"])
        if num_samples < 0:
            raise ValueError(f"shard has negative num_samples: {shard_path}")
        invalid_indices += sum(index >= num_samples for index in references[shard_file])
        del shard
    return len(references), missing_shards, invalid_indices


def atomic_csv_save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def length_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    lengths = np.asarray([row["length"] for row in rows], dtype=np.int64)
    if not len(lengths):
        raise ValueError("cannot summarize an empty split")
    return {
        "count": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "split_v1.csv"
    metadata_path = output_dir / "split_v1_meta.json"

    rows = read_saved_rows(manifest_path)
    duplicate_ids = duplicate_count(row["sample_id"] for row in rows)
    duplicate_paths = duplicate_count(row["processed_path"] for row in rows)
    if duplicate_ids or duplicate_paths:
        raise ValueError(
            f"manifest duplicates: sample_id={duplicate_ids}, processed_path={duplicate_paths}"
        )
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} saved rows, found {len(rows)}")

    validation_ids, allocation = sample_validation_ids(rows, args.val_size, args.seed)
    repeated_ids, repeated_allocation = sample_validation_ids(rows, args.val_size, args.seed)
    reproducible = validation_ids == repeated_ids and np.array_equal(
        allocation, repeated_allocation
    )
    for row in rows:
        row["split"] = "val" if row["sample_id"] in validation_ids else "train"

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    all_ids = {row["sample_id"] for row in rows}
    train_ids = {row["sample_id"] for row in train_rows}
    val_ids = {row["sample_id"] for row in val_rows}
    overlap = len(train_ids & val_ids)
    union_matches = train_ids | val_ids == all_ids

    referenced_shards, missing_shards, invalid_indices = validate_shard_references(
        rows, dataset_dir
    )
    expected_train = EXPECTED_TOTAL - args.val_size
    pass_conditions = (
        len(rows) == EXPECTED_TOTAL
        and len(train_rows) == expected_train
        and len(val_rows) == args.val_size
        and overlap == 0
        and union_matches
        and duplicate_ids == 0
        and duplicate_paths == 0
        and missing_shards == 0
        and invalid_indices == 0
        and reproducible
    )
    if not pass_conditions:
        raise ValueError("split validation failed before output creation")

    labels = [f"{lower}-{upper}" for lower, upper in LENGTH_BINS]
    bin_audit: dict[str, dict[str, float | int]] = {}
    for label in labels:
        all_count = sum(row["length_bin"] == label for row in rows)
        train_count = sum(row["length_bin"] == label for row in train_rows)
        val_count = sum(row["length_bin"] == label for row in val_rows)
        bin_audit[label] = {
            "total_count": all_count,
            "train_count": train_count,
            "val_count": val_count,
            "all_percentage": 100.0 * all_count / len(rows),
            "val_percentage": 100.0 * val_count / len(val_rows),
        }
    max_percentage_difference = max(
        abs(float(audit["val_percentage"]) - float(audit["all_percentage"]))
        for audit in bin_audit.values()
    )
    all_stats = length_stats(rows)
    train_stats = length_stats(train_rows)
    val_stats = length_stats(val_rows)

    metadata = {
        "schema_version": 1,
        "source_dataset": str(dataset_dir),
        "source_manifest": str(manifest_path),
        "total_samples": len(rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "seed": args.seed,
        "stratification": "length",
        "length_bins": [list(bounds) for bounds in LENGTH_BINS],
        "purpose": "training checkpoint selection and early stopping",
        "important_note": "CAMEO and PDB-date are not used in this validation split.",
        "per_bin_counts": {
            label: {
                "total_count": audit["total_count"],
                "train_count": audit["train_count"],
                "val_count": audit["val_count"],
            }
            for label, audit in bin_audit.items()
        },
    }
    atomic_csv_save(rows, csv_path)
    atomic_json_save(metadata, metadata_path)

    print("=" * 60)
    print("AFDB LATENT TRAIN / VALIDATION SPLIT")
    print("=" * 60)
    print(f"total: {len(rows)}")
    print(f"train: {len(train_rows)}")
    print(f"validation: {len(val_rows)}")
    print(f"\nseed: {args.seed}")
    print(f"reproducible: {reproducible}")
    print("\nOVERLAP")
    print(f"train-val overlap: {overlap}")
    print(f"duplicate sample IDs: {duplicate_ids}")
    print(f"duplicate processed paths: {duplicate_paths}")
    print("\nLENGTH")
    for name, stats in (("all", all_stats), ("train", train_stats), ("val", val_stats)):
        print(
            f"{name} mean/median: {stats['mean']:.6f} / {stats['median']:.6f} "
            f"(count={stats['count']}, min={stats['min']}, max={stats['max']})"
        )
    print("\nLENGTH BINS")
    for label in labels:
        audit = bin_audit[label]
        print(f"\n{label}:")
        print(f"  all: {audit['total_count']} ({audit['all_percentage']:.6f}%)")
        print(f"  train: {audit['train_count']}")
        print(f"  val: {audit['val_count']} ({audit['val_percentage']:.6f}%)")
    print(f"\nmax distribution percentage difference: {max_percentage_difference:.6f}")
    print("\nSHARD REFERENCES")
    print(f"referenced shards: {referenced_shards}")
    print(f"missing shards: {missing_shards}")
    print(f"invalid indices: {invalid_indices}")
    print(f"\noutput CSV: {csv_path}")
    print(f"output metadata: {metadata_path}")
    print("\nFINAL STATUS:")
    print("SPLIT_PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

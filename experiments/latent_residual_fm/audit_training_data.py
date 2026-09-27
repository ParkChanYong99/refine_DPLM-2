#!/usr/bin/env python3
"""Read-only availability audit for PDB+SwissProt training coordinates.

The audit reads parquet metadata and, only for paths resolved by explicit rules,
optionally opens a few coordinate files.  It never downloads, copies, rewrites,
or caches dataset content.  CAMEO 2022 and PDB-date are evaluation-only and are
excluded from every training-availability statistic.
"""

import argparse
import os
import pickle
import random
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq


REQUIRED_COLUMNS = ("processed_path", "struct_seq", "aa_seq")
MAX_CLASSIFICATION_EXAMPLES = 5
MAX_MISSING_EXAMPLES = 10


@dataclass
class SplitStats:
    rows: int = 0
    valid_paths: int = 0
    absolute: int = 0
    relative: int = 0
    invalid: int = 0
    existing: int = 0
    missing: int = 0
    unresolved: int = 0

    @property
    def resolvable_candidates(self) -> int:
        return self.existing + self.missing


@dataclass
class AuditResult:
    shard_counts: Counter = field(default_factory=Counter)
    shard_metadata_rows: Counter = field(default_factory=Counter)
    split_stats: Dict[str, SplitStats] = field(
        default_factory=lambda: defaultdict(SplitStats)
    )
    examples: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    missing_examples: List[str] = field(default_factory=list)
    unresolved_examples: List[str] = field(default_factory=list)
    existing_files: List[Path] = field(default_factory=list)
    path_counts: Counter = field(default_factory=Counter)
    audited_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data-bin/pdb_swissprot")
    parser.add_argument(
        "--coordinate_root",
        default=None,
        help="Explicit root joined to relative processed_path values.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Audit at most this many rows globally; default audits all rows.",
    )
    parser.add_argument("--num_probe_files", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def discover_parquet_files(data_dir: Path) -> List[Path]:
    """Find actual parquet shards without assuming shard names or counts."""
    require(data_dir.is_dir(), f"data_dir does not exist: {data_dir}")
    files = sorted(path for path in data_dir.rglob("*.parquet") if path.is_file())
    require(files, f"no parquet files found below {data_dir}")
    return files


def infer_repository_split(path: Path, data_dir: Path) -> str:
    """Use existing directory labels only; never create a new data split."""
    relative_parts = {part.lower() for part in path.relative_to(data_dir).parts[:-1]}
    if "valid" in relative_parts or "validation" in relative_parts:
        return "valid"
    if "train" in relative_parts:
        return "train"
    return "unknown"


def inspect_parquet_files(
    files: Sequence[Path], data_dir: Path
) -> Tuple[Dict[Path, str], Counter, Counter]:
    """Validate schemas and report physical row counts without loading tables."""
    splits: Dict[Path, str] = {}
    shard_counts: Counter = Counter()
    metadata_rows: Counter = Counter()
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        require(not missing, f"{path} missing required columns: {missing}")
        split = infer_repository_split(path, data_dir)
        splits[path] = split
        shard_counts[split] += 1
        metadata_rows[split] += parquet.metadata.num_rows
        print("\n[PARQUET]")
        print(f"file path: {path}")
        print(f"repository split: {split}")
        print(f"number of rows: {parquet.metadata.num_rows}")
        print(f"column names: {columns}")
    return splits, shard_counts, metadata_rows


def iter_processed_paths(
    files: Sequence[Path],
    splits: Dict[Path, str],
    max_rows: Optional[int],
) -> Iterator[Tuple[str, object]]:
    """Stream only processed_path batches to avoid copying full parquet rows."""
    remaining = max_rows
    for path in files:
        if remaining is not None and remaining <= 0:
            return
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["processed_path"], use_threads=False
        ):
            values = batch.column(0).to_pylist()
            if remaining is not None:
                values = values[:remaining]
            for value in values:
                yield splits[path], value
            if remaining is not None:
                remaining -= len(values)
                if remaining <= 0:
                    return


def classify_processed_path(value: object) -> Tuple[str, Optional[str]]:
    """Return absolute, relative, or invalid without rewriting the value."""
    if not isinstance(value, str) or not value.strip():
        return "invalid", None
    raw = value.strip()
    return ("absolute" if os.path.isabs(raw) else "relative"), raw


def resolve_processed_path(
    kind: str, raw_path: Optional[str], coordinate_root: Optional[Path]
) -> Tuple[str, Optional[Path]]:
    """Apply only the two explicit resolution candidates authorized by CLI."""
    if kind == "invalid" or raw_path is None:
        return "invalid", None
    if kind == "absolute":
        candidate = Path(raw_path)
    elif coordinate_root is not None:
        candidate = coordinate_root / raw_path
    else:
        return "unresolved", None
    return ("existing" if candidate.is_file() else "missing"), candidate


def remember_example(container: List[str], value: str, limit: int) -> None:
    if len(container) < limit and value not in container:
        container.append(value)


def audit_resolution(
    files: Sequence[Path],
    splits: Dict[Path, str],
    coordinate_root: Optional[Path],
    max_rows: Optional[int],
) -> AuditResult:
    """Classify and resolve every audited path while retaining bounded examples."""
    result = AuditResult()
    for split, value in iter_processed_paths(files, splits, max_rows):
        result.audited_rows += 1
        stats = result.split_stats[split]
        stats.rows += 1
        kind, raw = classify_processed_path(value)
        setattr(stats, kind, getattr(stats, kind) + 1)
        if raw is not None:
            stats.valid_paths += 1
            result.path_counts[raw] += 1
            remember_example(
                result.examples[kind], raw, MAX_CLASSIFICATION_EXAMPLES
            )
        outcome, candidate = resolve_processed_path(kind, raw, coordinate_root)
        if outcome in ("existing", "missing", "unresolved"):
            setattr(stats, outcome, getattr(stats, outcome) + 1)
        if outcome == "existing" and candidate is not None:
            result.existing_files.append(candidate)
        elif outcome == "missing" and candidate is not None:
            remember_example(
                result.missing_examples, str(candidate), MAX_MISSING_EXAMPLES
            )
        elif outcome == "unresolved" and raw is not None:
            remember_example(
                result.unresolved_examples, raw, MAX_MISSING_EXAMPLES
            )
    return result


def add_stats(items: Iterable[SplitStats]) -> SplitStats:
    total = SplitStats()
    for item in items:
        for field_name in total.__dataclass_fields__:
            setattr(total, field_name, getattr(total, field_name) + getattr(item, field_name))
    return total


def print_path_audit(result: AuditResult, coordinate_root: Optional[Path]) -> None:
    print("\n[REPOSITORY RESOLUTION RULE]")
    print("source: src/byprot/datamodules/pdb_dataset/pdb_datamodule.py")
    print("function: PdbDataset._process_csv_row")
    print('behavior: csv_row["processed_path"].format(data_dir=dataset_cfg.data_dir), then du.read_pkl(path)')
    print("audit policy: no implicit prefix or placeholder rewrite is applied")
    print(f"explicit coordinate_root: {coordinate_root if coordinate_root else 'None'}")

    total = add_stats(result.split_stats.values())
    print("\n[PROCESSED PATH CLASSIFICATION]")
    print(f"absolute paths: {total.absolute}")
    print(f"relative paths: {total.relative}")
    print(f"empty/null/invalid paths: {total.invalid}")
    for kind in ("absolute", "relative"):
        print(f"{kind} examples (max 5): {result.examples[kind]}")

    print("\n[MISSING EXAMPLES]")
    if result.missing_examples:
        for index, value in enumerate(result.missing_examples, 1):
            print(f"{index}. {value}")
    else:
        print("none")
    print("\n[UNRESOLVED RELATIVE EXAMPLES]")
    if result.unresolved_examples:
        for index, value in enumerate(result.unresolved_examples, 1):
            print(f"{index}. {value}")
    else:
        print("none")

    for split in ("train", "valid", "unknown"):
        stats = result.split_stats.get(split)
        if stats is None or stats.rows == 0:
            continue
        denominator = stats.resolvable_candidates
        rate = f"{stats.existing / denominator:.6%}" if denominator else "N/A"
        print(f"\n[{split.upper()}]")
        print(f"rows: {stats.rows}")
        print(f"existing: {stats.existing}")
        print(f"missing: {stats.missing}")
        print(f"unresolved: {stats.unresolved}")
        print(f"invalid: {stats.invalid}")
        print(f"resolution rate: {rate}")


def array_description(value: object) -> Tuple[Tuple[int, ...], str]:
    array = np.asarray(value)
    return tuple(array.shape), str(array.dtype)


def probe_coordinate_file(path: Path, index: int, verbose: bool) -> Tuple[bool, Optional[dict]]:
    """Open one resolved coordinate pickle read-only and validate raw fields."""
    print(f"\n[PROBE {index}]")
    print(f"path: {path}")
    try:
        with path.open("rb") as handle:
            raw = dict(pickle.load(handle))
        expected = ("atom_positions", "atom_mask", "aatype", "bb_mask", "residue_index")
        missing = [key for key in expected if key not in raw]
        if missing:
            print(f"missing expected fields: {missing}")
            return False, raw
        length = len(raw["aatype"])
        print(f"sequence length: {length}")
        for key in expected:
            shape, dtype = array_description(raw[key])
            print(f"{key}: shape={shape}, dtype={dtype}")
        coordinates = np.asarray(raw["atom_positions"])
        mask = np.asarray(raw["atom_mask"]).astype(bool)
        finite = bool(np.isfinite(coordinates[mask]).all()) if mask.any() else False
        shapes_ok = (
            coordinates.shape == (length, 37, 3)
            and np.asarray(raw["atom_mask"]).shape == (length, 37)
            and all(np.asarray(raw[key]).shape == (length,) for key in ("aatype", "bb_mask", "residue_index"))
        )
        print(f"finite valid coordinates: {finite}")
        print(f"L <= 512: {length <= 512}")
        if verbose:
            print(f"all keys: {sorted(raw)}")
        return shapes_ok and finite, raw
    except Exception as exc:
        print(f"probe open FAILED: {type(exc).__name__}: {exc}")
        return False, None


def probe_repository_preprocessing(raw: dict, index: int) -> bool:
    """Call the repository's read-only PdbDataset.process_chain contract."""
    try:
        from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset

        processed = PdbDataset.process_chain(raw, random_crop=False)
        fields = (
            "all_atom_positions",
            "all_atom_mask",
            "res_mask",
            "seq_length",
            "aatype",
            "residue_index",
        )
        missing = [key for key in fields if key not in processed]
        if missing:
            print(f"[PROBE {index} PREPROCESSING] missing fields: {missing}")
            return False
        print(f"[PROBE {index} PREPROCESSING]")
        for key in fields:
            value = processed[key]
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        length = processed["all_atom_positions"].shape[0]
        return (
            tuple(processed["all_atom_positions"].shape) == (length, 37, 3)
            and tuple(processed["res_mask"].shape) == (length,)
            and tuple(processed["aatype"].shape) == (length,)
        )
    except Exception as exc:
        print(f"[PROBE {index} PREPROCESSING] FAILED: {type(exc).__name__}: {exc}")
        return False


def run_probes(
    existing_files: Sequence[Path], num_probe_files: int, seed: int, verbose: bool
) -> Tuple[int, int, str]:
    require(num_probe_files >= 0, "--num_probe_files must be non-negative")
    unique_files = sorted(set(existing_files))
    if not unique_files or num_probe_files == 0:
        print("\n[COORDINATE PROBE]")
        print("[NOT TESTED] no resolved existing coordinate files selected")
        return 0, 0, "NOT TESTED"
    count = min(num_probe_files, len(unique_files))
    selected = random.Random(seed).sample(unique_files, count)
    opened = 0
    failed = 0
    preprocess_results: List[bool] = []
    for index, path in enumerate(selected, 1):
        raw_ok, raw = probe_coordinate_file(path, index, verbose)
        if raw is None:
            failed += 1
            continue
        opened += 1
        if not raw_ok:
            failed += 1
        preprocess_results.append(probe_repository_preprocessing(raw, index))
    if not preprocess_results:
        compatibility = "NOT TESTED"
    elif all(preprocess_results):
        compatibility = "PASS"
    else:
        compatibility = "FAIL"
    return opened, failed, compatibility


def evaluation_coordinate_presence(workspace: Path) -> Tuple[bool, bool]:
    def contains_pickle(path: Path) -> bool:
        return path.is_dir() and any(path.rglob("*.pkl"))

    return (
        contains_pickle(workspace / "data-bin/cameo2022"),
        contains_pickle(workspace / "data-bin/PDB_date"),
    )


def determine_training_status(
    result: AuditResult,
    max_rows: Optional[int],
    opened: int,
    compatibility: str,
) -> Tuple[str, List[str]]:
    train = result.split_stats.get("train", SplitStats())
    complete_resolution = (
        train.rows > 0
        and train.existing == train.rows
        and train.missing == 0
        and train.unresolved == 0
        and train.invalid == 0
    )
    if complete_resolution and max_rows is None and opened > 0 and compatibility == "PASS":
        status = "READY"
    elif train.existing > 0:
        status = "PARTIALLY_AVAILABLE"
    else:
        status = "NOT_AVAILABLE"
    reasons = [
        f"Audited training rows: {train.rows}; resolved existing coordinate files: {train.existing}.",
        f"Training missing/unresolved/invalid: {train.missing}/{train.unresolved}/{train.invalid}.",
        f"Repository preprocessing probe: {compatibility} ({opened} files opened).",
    ]
    if max_rows is not None:
        reasons.append("This was a max_rows-limited audit, so it cannot establish full-corpus readiness.")
    return status, reasons


def summarize_results(
    result: AuditResult,
    shard_counts: Counter,
    metadata_rows: Counter,
    opened: int,
    failed: int,
    compatibility: str,
    cameo_present: bool,
    pdb_date_present: bool,
    status: str,
    reasons: Sequence[str],
) -> None:
    total = add_stats(result.split_stats.values())
    denominator = total.resolvable_candidates
    rate = f"{total.existing / denominator:.6%}" if denominator else "N/A"
    existing_total_rate = (
        f"{total.existing / total.rows:.6%}" if total.rows else "N/A"
    )
    duplicates = sum(count - 1 for count in result.path_counts.values() if count > 1)
    print("\n" + "=" * 50)
    print("DPLM-2.1 LATENT RESIDUAL FM TRAINING DATA AUDIT")
    print("=" * 50)
    print("\n[PARQUET]")
    print(f"train shards: {shard_counts['train']}")
    print(f"valid shards: {shard_counts['valid']}")
    print(f"train rows: {result.split_stats['train'].rows}")
    print(f"valid rows: {result.split_stats['valid'].rows}")
    print(f"total rows: {total.rows}")
    print(f"physical metadata rows (all shards): {sum(metadata_rows.values())}")
    print(f"duplicate processed_path rows: {duplicates}")
    print("\n[PROCESSED PATHS]")
    print(f"absolute: {total.absolute}")
    print(f"relative: {total.relative}")
    print(f"invalid: {total.invalid}")
    print("\n[RESOLUTION]")
    print(f"existing: {total.existing}")
    print(f"missing: {total.missing}")
    print(f"unresolved: {total.unresolved}")
    print(f"resolution rate (existing / resolvable candidates): {rate}")
    print(f"existing / total rows: {existing_total_rate}")
    print("\n[COORDINATE PROBE]")
    print(f"files opened: {opened}")
    print(f"files failed: {failed}")
    print(f"repository preprocessing compatible: {compatibility}")
    print("\n[EVALUATION DATA — EXCLUDED]")
    print(f"CAMEO 2022 coordinates present: {'YES' if cameo_present else 'NO'}")
    print(f"PDB_date coordinates present: {'YES' if pdb_date_present else 'NO'}")
    print("These files are excluded from training availability statistics.")
    print("\n[TRAINING DATA STATUS]")
    print(status)
    for reason in reasons[:5]:
        print(reason)


def main() -> int:
    args = parse_args()
    try:
        require(args.max_rows is None or args.max_rows >= 0, "--max_rows must be non-negative")
        data_dir = Path(args.data_dir)
        coordinate_root = Path(args.coordinate_root) if args.coordinate_root else None
        files = discover_parquet_files(data_dir)
        splits, shard_counts, metadata_rows = inspect_parquet_files(files, data_dir)
        result = audit_resolution(files, splits, coordinate_root, args.max_rows)
        result.shard_counts = shard_counts
        result.shard_metadata_rows = metadata_rows
        print_path_audit(result, coordinate_root)
        opened, failed, compatibility = run_probes(
            result.existing_files, args.num_probe_files, args.seed, args.verbose
        )
        cameo_present, pdb_date_present = evaluation_coordinate_presence(Path.cwd())
        status, reasons = determine_training_status(
            result, args.max_rows, opened, compatibility
        )
        summarize_results(
            result,
            shard_counts,
            metadata_rows,
            opened,
            failed,
            compatibility,
            cameo_present,
            pdb_date_present,
            status,
            reasons,
        )
        return 0
    except Exception as exc:
        print(f"[AUDIT FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

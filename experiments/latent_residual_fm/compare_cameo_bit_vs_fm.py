#!/usr/bin/env python3
"""Create a validated per-protein CAMEO comparison from two evaluator runs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_NUM_SAMPLES = 163
AGGREGATE_REL_TOL = 1e-9
AGGREGATE_ABS_TOL = 1e-9
DEFAULT_BIT_EVAL_DIR = Path(
    "generation-results/dplm2_bit_650m_final/cameo2022/bit_only/folding/"
    "forward_folding/run_2026-09-03_09-08-54/folding/eval"
)
DEFAULT_FM_EVAL_DIR = Path(
    "generation-results/dplm2_bit_650m_final/cameo2022/"
    "latent_fm_alpha050/folding/forward_folding/"
    "run_2026-09-05_12-47-22/folding/eval"
)
OUTPUT_DIR = Path(
    "generation-results/dplm2_bit_650m_final/cameo2022/comparison"
)
PAIRED_CSV = OUTPUT_DIR / "cameo2022_bit_vs_fm_paired.csv"
SUMMARY_CSV = OUTPUT_DIR / "cameo2022_bit_vs_fm_summary.csv"

PAPER_BIT_RMSD = 6.4028
PAPER_BIT_TMSCORE = 0.8380
PAPER_RESDIFF_RMSD = 6.1781
PAPER_RESDIFF_TMSCORE = 0.8428

TOP_SAMPLE_REQUIRED_COLUMNS = {
    "length",
    "sample_id",
    "ca_rmsd_to_gt",
    "bb_rmsd_to_gt",
    "bb_tmscore_to_gt",
}
AGGREGATE_REQUIRED_COLUMNS = {
    "Average bb_rmsd_to_gt",
    "Average bb_tmscore_to_gt",
    "Total samples",
}
PAIRED_COLUMNS = (
    "sample_id",
    "length",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "rmsd_improvement",
    "bit_bb_tmscore",
    "fm_bb_tmscore",
    "tmscore_improvement",
    "rmsd_improved",
    "tmscore_improved",
    "bit_ca_rmsd",
    "fm_ca_rmsd",
    "ca_rmsd_improvement",
)
SUMMARY_COLUMNS = (
    "num_samples",
    "bit_mean_bb_rmsd",
    "bit_median_bb_rmsd",
    "bit_mean_bb_tmscore",
    "bit_median_bb_tmscore",
    "fm_mean_bb_rmsd",
    "fm_median_bb_rmsd",
    "fm_mean_bb_tmscore",
    "fm_median_bb_tmscore",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_count",
    "rmsd_improved_percent",
    "rmsd_worsened_count",
    "rmsd_tie_count",
    "mean_tmscore_improvement",
    "median_tmscore_improvement",
    "tmscore_improved_count",
    "tmscore_improved_percent",
    "tmscore_worsened_count",
    "tmscore_tie_count",
    "relative_mean_rmsd_reduction_percent",
    "relative_mean_tmscore_increase_percent",
    "bit_aggregate_mean_bb_rmsd",
    "bit_aggregate_mean_bb_tmscore",
    "bit_aggregate_total_samples",
    "fm_aggregate_mean_bb_rmsd",
    "fm_aggregate_mean_bb_tmscore",
    "fm_aggregate_total_samples",
    "paper_reference_only_bit_rmsd",
    "paper_reference_only_bit_tmscore",
    "paper_reference_only_resdiff_rmsd",
    "paper_reference_only_resdiff_tmscore",
    "paper_reference_notice",
)


@dataclass(frozen=True)
class SampleMetrics:
    sample_id: str
    length: int
    ca_rmsd_to_gt: float
    bb_rmsd_to_gt: float
    bb_tmscore_to_gt: float
    path: Path


@dataclass(frozen=True)
class AggregateMetrics:
    mean_bb_rmsd: float
    mean_bb_tmscore: float
    total_samples: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit_eval_dir", type=Path, default=DEFAULT_BIT_EVAL_DIR)
    parser.add_argument("--fm_eval_dir", type=Path, default=DEFAULT_FM_EVAL_DIR)
    return parser.parse_args()


def finite_float(value: str, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"invalid {field} in {path}: {value!r}") from exc
    require(math.isfinite(result), f"non-finite {field} in {path}")
    return result


def read_single_csv_row(
    path: Path, required_columns: set[str]
) -> dict[str, str]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns.difference(reader.fieldnames or ())
        require(not missing, f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
    require(len(rows) == 1, f"expected exactly one data row in {path}, got {len(rows)}")
    return rows[0]


def collect_top_samples(eval_dir: Path, label: str) -> dict[str, SampleMetrics]:
    require(eval_dir.is_dir(), f"{label} evaluator directory does not exist: {eval_dir}")
    paths = sorted(eval_dir.rglob("top_sample.csv"))
    require(paths, f"no top_sample.csv files found under {eval_dir}")

    samples: dict[str, SampleMetrics] = {}
    duplicate_ids: list[str] = []
    for path in paths:
        sample_id = path.parent.name
        length_directory = path.parent.parent.name
        require(sample_id, f"empty sample directory name: {path}")
        require(
            length_directory.startswith("length_")
            and length_directory[len("length_") :].isdigit(),
            f"unexpected evaluator directory structure: {path}",
        )
        directory_length = int(length_directory[len("length_") :])
        row = read_single_csv_row(path, TOP_SAMPLE_REQUIRED_COLUMNS)
        row_length_value = finite_float(row["length"], "length", path)
        require(row_length_value.is_integer(), f"non-integral length in {path}")
        row_length = int(row_length_value)
        require(row_length == directory_length, f"length mismatch in {path}")
        require(60 <= row_length <= 512, f"evaluator length is out of range in {path}")

        # evaluator_dplm2.py uses the pdb_name for the sample directory.  Its
        # top_sample.csv sample_id is the batch-local integer (observed as 0),
        # so the exact metadata ID must be recovered from path.parent.name.
        metric = SampleMetrics(
            sample_id=sample_id,
            length=row_length,
            ca_rmsd_to_gt=finite_float(row["ca_rmsd_to_gt"], "ca_rmsd_to_gt", path),
            bb_rmsd_to_gt=finite_float(row["bb_rmsd_to_gt"], "bb_rmsd_to_gt", path),
            bb_tmscore_to_gt=finite_float(
                row["bb_tmscore_to_gt"], "bb_tmscore_to_gt", path
            ),
            path=path,
        )
        if sample_id in samples:
            duplicate_ids.append(sample_id)
        else:
            samples[sample_id] = metric

    print(f"{label} top_sample.csv count: {len(paths)}")
    print(f"{label} unique sample ID count: {len(samples)}")
    print(f"{label} duplicate IDs: {sorted(set(duplicate_ids))}")
    require(not duplicate_ids, f"{label} has duplicate sample IDs: {sorted(set(duplicate_ids))}")
    require(
        len(paths) == len(samples),
        f"{label} top_sample.csv count and unique ID count differ",
    )
    return samples


def read_aggregate(eval_dir: Path, label: str) -> AggregateMetrics:
    path = eval_dir / "forward_fold_metrics.csv"
    row = read_single_csv_row(path, AGGREGATE_REQUIRED_COLUMNS)
    total_value = finite_float(row["Total samples"], "Total samples", path)
    require(total_value.is_integer(), f"non-integral Total samples in {path}")
    aggregate = AggregateMetrics(
        mean_bb_rmsd=finite_float(
            row["Average bb_rmsd_to_gt"], "Average bb_rmsd_to_gt", path
        ),
        mean_bb_tmscore=finite_float(
            row["Average bb_tmscore_to_gt"],
            "Average bb_tmscore_to_gt",
            path,
        ),
        total_samples=int(total_value),
    )
    require(aggregate.total_samples > 0, f"{label} aggregate has no samples")
    return aggregate


def validate_aggregate(
    label: str,
    samples: Mapping[str, SampleMetrics],
    aggregate: AggregateMetrics,
) -> None:
    recomputed_rmsd = statistics.fmean(
        sample.bb_rmsd_to_gt for sample in samples.values()
    )
    recomputed_tm = statistics.fmean(
        sample.bb_tmscore_to_gt for sample in samples.values()
    )
    require(
        aggregate.total_samples == len(samples),
        f"{label} aggregate Total samples={aggregate.total_samples}, "
        f"but found {len(samples)} top_sample.csv files",
    )
    require(
        math.isclose(
            recomputed_rmsd,
            aggregate.mean_bb_rmsd,
            rel_tol=AGGREGATE_REL_TOL,
            abs_tol=AGGREGATE_ABS_TOL,
        ),
        f"{label} recomputed BB RMSD mean {recomputed_rmsd:.16g} does not match "
        f"aggregate {aggregate.mean_bb_rmsd:.16g}",
    )
    require(
        math.isclose(
            recomputed_tm,
            aggregate.mean_bb_tmscore,
            rel_tol=AGGREGATE_REL_TOL,
            abs_tol=AGGREGATE_ABS_TOL,
        ),
        f"{label} recomputed TM-score mean {recomputed_tm:.16g} does not match "
        f"aggregate {aggregate.mean_bb_tmscore:.16g}",
    )


def make_paired_rows(
    bit_samples: Mapping[str, SampleMetrics],
    fm_samples: Mapping[str, SampleMetrics],
) -> list[dict[str, Any]]:
    bit_ids = set(bit_samples)
    fm_ids = set(fm_samples)
    require(
        bit_ids == fm_ids,
        "Bit/FM sample ID sets differ: "
        f"Bit-only={sorted(bit_ids - fm_ids)}, FM-only={sorted(fm_ids - bit_ids)}",
    )
    require(
        len(bit_ids) == EXPECTED_NUM_SAMPLES,
        f"expected exactly {EXPECTED_NUM_SAMPLES} paired CAMEO samples, got {len(bit_ids)}",
    )

    paired: list[dict[str, Any]] = []
    for sample_id in sorted(bit_ids):
        bit = bit_samples[sample_id]
        fm = fm_samples[sample_id]
        require(bit.length == fm.length, f"Bit/FM length mismatch for {sample_id}")
        rmsd_improvement = bit.bb_rmsd_to_gt - fm.bb_rmsd_to_gt
        tmscore_improvement = fm.bb_tmscore_to_gt - bit.bb_tmscore_to_gt
        paired.append(
            {
                "sample_id": sample_id,
                "length": bit.length,
                "bit_bb_rmsd": bit.bb_rmsd_to_gt,
                "fm_bb_rmsd": fm.bb_rmsd_to_gt,
                "rmsd_improvement": rmsd_improvement,
                "bit_bb_tmscore": bit.bb_tmscore_to_gt,
                "fm_bb_tmscore": fm.bb_tmscore_to_gt,
                "tmscore_improvement": tmscore_improvement,
                "rmsd_improved": rmsd_improvement > 0.0,
                "tmscore_improved": tmscore_improvement > 0.0,
                "bit_ca_rmsd": bit.ca_rmsd_to_gt,
                "fm_ca_rmsd": fm.ca_rmsd_to_gt,
                "ca_rmsd_improvement": bit.ca_rmsd_to_gt - fm.ca_rmsd_to_gt,
            }
        )
    return paired


def count_directions(values: Sequence[float]) -> tuple[int, int, int]:
    improved = sum(value > 0.0 for value in values)
    worsened = sum(value < 0.0 for value in values)
    ties = sum(value == 0.0 for value in values)
    require(improved + worsened + ties == len(values), "direction counts do not sum")
    return improved, worsened, ties


def make_summary(
    paired: Sequence[Mapping[str, Any]],
    bit_aggregate: AggregateMetrics,
    fm_aggregate: AggregateMetrics,
) -> dict[str, Any]:
    bit_rmsd = [float(row["bit_bb_rmsd"]) for row in paired]
    fm_rmsd = [float(row["fm_bb_rmsd"]) for row in paired]
    bit_tm = [float(row["bit_bb_tmscore"]) for row in paired]
    fm_tm = [float(row["fm_bb_tmscore"]) for row in paired]
    rmsd_improvements = [float(row["rmsd_improvement"]) for row in paired]
    tm_improvements = [float(row["tmscore_improvement"]) for row in paired]
    rmsd_improved, rmsd_worsened, rmsd_ties = count_directions(rmsd_improvements)
    tm_improved, tm_worsened, tm_ties = count_directions(tm_improvements)

    mean_bit_rmsd = statistics.fmean(bit_rmsd)
    mean_fm_rmsd = statistics.fmean(fm_rmsd)
    mean_bit_tm = statistics.fmean(bit_tm)
    mean_fm_tm = statistics.fmean(fm_tm)
    require(mean_bit_rmsd != 0.0, "Bit mean RMSD is zero")
    require(mean_bit_tm != 0.0, "Bit mean TM-score is zero")
    num_samples = len(paired)
    return {
        "num_samples": num_samples,
        "bit_mean_bb_rmsd": mean_bit_rmsd,
        "bit_median_bb_rmsd": statistics.median(bit_rmsd),
        "bit_mean_bb_tmscore": mean_bit_tm,
        "bit_median_bb_tmscore": statistics.median(bit_tm),
        "fm_mean_bb_rmsd": mean_fm_rmsd,
        "fm_median_bb_rmsd": statistics.median(fm_rmsd),
        "fm_mean_bb_tmscore": mean_fm_tm,
        "fm_median_bb_tmscore": statistics.median(fm_tm),
        "mean_rmsd_improvement": statistics.fmean(rmsd_improvements),
        "median_rmsd_improvement": statistics.median(rmsd_improvements),
        "rmsd_improved_count": rmsd_improved,
        "rmsd_improved_percent": rmsd_improved / num_samples * 100.0,
        "rmsd_worsened_count": rmsd_worsened,
        "rmsd_tie_count": rmsd_ties,
        "mean_tmscore_improvement": statistics.fmean(tm_improvements),
        "median_tmscore_improvement": statistics.median(tm_improvements),
        "tmscore_improved_count": tm_improved,
        "tmscore_improved_percent": tm_improved / num_samples * 100.0,
        "tmscore_worsened_count": tm_worsened,
        "tmscore_tie_count": tm_ties,
        "relative_mean_rmsd_reduction_percent": (
            (mean_bit_rmsd - mean_fm_rmsd) / mean_bit_rmsd * 100.0
        ),
        "relative_mean_tmscore_increase_percent": (
            (mean_fm_tm - mean_bit_tm) / mean_bit_tm * 100.0
        ),
        "bit_aggregate_mean_bb_rmsd": bit_aggregate.mean_bb_rmsd,
        "bit_aggregate_mean_bb_tmscore": bit_aggregate.mean_bb_tmscore,
        "bit_aggregate_total_samples": bit_aggregate.total_samples,
        "fm_aggregate_mean_bb_rmsd": fm_aggregate.mean_bb_rmsd,
        "fm_aggregate_mean_bb_tmscore": fm_aggregate.mean_bb_tmscore,
        "fm_aggregate_total_samples": fm_aggregate.total_samples,
        "paper_reference_only_bit_rmsd": PAPER_BIT_RMSD,
        "paper_reference_only_bit_tmscore": PAPER_BIT_TMSCORE,
        "paper_reference_only_resdiff_rmsd": PAPER_RESDIFF_RMSD,
        "paper_reference_only_resdiff_tmscore": PAPER_RESDIFF_TMSCORE,
        "paper_reference_notice": "PAPER REFERENCE ONLY; NOT LOCAL PAIRED RESULT",
    }


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def describe_extremes(paired: Sequence[Mapping[str, Any]]) -> None:
    best_rmsd = max(paired, key=lambda row: float(row["rmsd_improvement"]))
    worst_rmsd = min(paired, key=lambda row: float(row["rmsd_improvement"]))
    best_tm = max(paired, key=lambda row: float(row["tmscore_improvement"]))
    worst_tm = min(paired, key=lambda row: float(row["tmscore_improvement"]))
    print(
        "best RMSD improvement: "
        f"{best_rmsd['sample_id']} improvement={float(best_rmsd['rmsd_improvement']):.9g} "
        f"Bit={float(best_rmsd['bit_bb_rmsd']):.9g} "
        f"FM={float(best_rmsd['fm_bb_rmsd']):.9g}"
    )
    print(
        "worst RMSD degradation: "
        f"{worst_rmsd['sample_id']} "
        f"degradation={float(worst_rmsd['fm_bb_rmsd']) - float(worst_rmsd['bit_bb_rmsd']):.9g} "
        f"Bit={float(worst_rmsd['bit_bb_rmsd']):.9g} "
        f"FM={float(worst_rmsd['fm_bb_rmsd']):.9g}"
    )
    print(
        "best TM improvement: "
        f"{best_tm['sample_id']} improvement={float(best_tm['tmscore_improvement']):.9g} "
        f"Bit={float(best_tm['bit_bb_tmscore']):.9g} "
        f"FM={float(best_tm['fm_bb_tmscore']):.9g}"
    )
    print(
        "worst TM degradation: "
        f"{worst_tm['sample_id']} "
        f"degradation={float(worst_tm['bit_bb_tmscore']) - float(worst_tm['fm_bb_tmscore']):.9g} "
        f"Bit={float(worst_tm['bit_bb_tmscore']):.9g} "
        f"FM={float(worst_tm['fm_bb_tmscore']):.9g}"
    )


def print_summary(
    summary: Mapping[str, Any], paired: Sequence[Mapping[str, Any]]
) -> None:
    print("\n[CAMEO PAIRED COMPARISON]\n")
    print(f"num paired samples: {summary['num_samples']}")
    print(f"Bit mean BB RMSD: {float(summary['bit_mean_bb_rmsd']):.9g}")
    print(f"FM mean BB RMSD: {float(summary['fm_mean_bb_rmsd']):.9g}")
    print(f"absolute RMSD improvement: {float(summary['mean_rmsd_improvement']):.9g}")
    print(
        "relative RMSD reduction: "
        f"{float(summary['relative_mean_rmsd_reduction_percent']):.9g}%"
    )
    print(f"Bit mean TM-score: {float(summary['bit_mean_bb_tmscore']):.9g}")
    print(f"FM mean TM-score: {float(summary['fm_mean_bb_tmscore']):.9g}")
    print(
        "absolute TM-score improvement: "
        f"{float(summary['mean_tmscore_improvement']):.9g}"
    )
    print(
        f"RMSD improved: {summary['rmsd_improved_count']} "
        f"({float(summary['rmsd_improved_percent']):.6f}%)"
    )
    print(f"RMSD worsened: {summary['rmsd_worsened_count']}")
    print(f"RMSD ties: {summary['rmsd_tie_count']}")
    print(
        f"TM-score improved: {summary['tmscore_improved_count']} "
        f"({float(summary['tmscore_improved_percent']):.6f}%)"
    )
    print(f"TM-score worsened: {summary['tmscore_worsened_count']}")
    print(f"TM-score ties: {summary['tmscore_tie_count']}")
    describe_extremes(paired)
    print(f"paired CSV: {PAIRED_CSV}")
    print(f"summary CSV: {SUMMARY_CSV}")
    print("\n[PAPER REFERENCE ONLY — NOT LOCAL PAIRED RESULT]")
    print(f"Paper Bit: RMSD {PAPER_BIT_RMSD:.4f}, TM-score {PAPER_BIT_TMSCORE:.4f}")
    print(
        "Paper Bit + RESDIFF: "
        f"RMSD {PAPER_RESDIFF_RMSD:.4f}, TM-score {PAPER_RESDIFF_TMSCORE:.4f}"
    )


def main() -> int:
    args = parse_args()
    bit_samples = collect_top_samples(args.bit_eval_dir, "Bit")
    fm_samples = collect_top_samples(args.fm_eval_dir, "FM")
    require(
        len(bit_samples) == EXPECTED_NUM_SAMPLES,
        f"expected {EXPECTED_NUM_SAMPLES} Bit samples, got {len(bit_samples)}",
    )
    require(
        len(fm_samples) == EXPECTED_NUM_SAMPLES,
        f"expected {EXPECTED_NUM_SAMPLES} FM samples, got {len(fm_samples)}",
    )

    bit_aggregate = read_aggregate(args.bit_eval_dir, "Bit")
    fm_aggregate = read_aggregate(args.fm_eval_dir, "FM")
    validate_aggregate("Bit", bit_samples, bit_aggregate)
    validate_aggregate("FM", fm_samples, fm_aggregate)
    paired = make_paired_rows(bit_samples, fm_samples)
    summary = make_summary(paired, bit_aggregate, fm_aggregate)

    atomic_write_csv(PAIRED_CSV, PAIRED_COLUMNS, paired)
    atomic_write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, [summary])
    print_summary(summary, paired)
    print("\nCAMEO_BIT_VS_FM_PAIRED_COMPARISON_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare publication-facing tables from already finalized result CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FINAL_ALPHA = 0.50
REFERENCE_ABS_TOL = 1e-7
PERCENT_REFERENCE_ABS_TOL = 1e-6
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12
OUTPUT_DIR = Path("generation-results/dplm2_bit_650m_final/final_results")

MAIN_PERFORMANCE_CSV = OUTPUT_DIR / "main_performance_table.csv"
PAIRED_STATISTICAL_CSV = OUTPUT_DIR / "paired_statistical_table.csv"
LOCAL_SUMMARY_CSV = OUTPUT_DIR / "local_bit_vs_fm_summary.csv"
BASELINE_QUARTILES_CSV = OUTPUT_DIR / "baseline_difficulty_quartiles.csv"
FIGURE_BASELINE_CSV = OUTPUT_DIR / "figure_baseline_difficulty.csv"
FIGURE_PERFORMANCE_CSV = OUTPUT_DIR / "figure_benchmark_performance.csv"
FINAL_SUMMARY_TXT = OUTPUT_DIR / "final_results_summary.txt"
FINAL_METADATA_TXT = OUTPUT_DIR / "final_results_metadata.txt"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    paired_summary_path: Path
    expected_n: int
    expected_bit_rmsd: float
    expected_fm_rmsd: float
    expected_bit_tm: float
    expected_fm_tm: float
    expected_rmsd_improved_count: int
    expected_rmsd_improved_percent: float
    expected_tm_improved_count: int
    expected_tm_improved_percent: float
    paper_bit_rmsd: float
    paper_bit_tm: float
    paper_resdiff_rmsd: float
    paper_resdiff_tm: float


DEFAULT_BENCHMARKS = (
    BenchmarkSpec(
        key="cameo2022",
        display_name="CAMEO",
        paired_summary_path=Path(
            "generation-results/dplm2_bit_650m_final/cameo2022/comparison/"
            "cameo2022_bit_vs_fm_summary.csv"
        ),
        expected_n=163,
        expected_bit_rmsd=6.28888164,
        expected_fm_rmsd=6.19972204,
        expected_bit_tm=0.841421163,
        expected_fm_tm=0.844954294,
        expected_rmsd_improved_count=91,
        expected_rmsd_improved_percent=55.828221,
        expected_tm_improved_count=98,
        expected_tm_improved_percent=60.122699,
        paper_bit_rmsd=6.4028,
        paper_bit_tm=0.8380,
        paper_resdiff_rmsd=6.1781,
        paper_resdiff_tm=0.8428,
    ),
    BenchmarkSpec(
        key="PDB_date",
        display_name="PDB-DATE",
        paired_summary_path=Path(
            "generation-results/dplm2_bit_650m_final/PDB_date/comparison/"
            "pdb_date_bit_vs_fm_summary.csv"
        ),
        expected_n=442,
        expected_bit_rmsd=3.11376375,
        expected_fm_rmsd=3.10744498,
        expected_bit_tm=0.91379618,
        expected_fm_tm=0.91385542,
        expected_rmsd_improved_count=216,
        expected_rmsd_improved_percent=48.868778,
        expected_tm_improved_count=211,
        expected_tm_improved_percent=47.737557,
        paper_bit_rmsd=3.2213,
        paper_bit_tm=0.9043,
        paper_resdiff_rmsd=3.0168,
        paper_resdiff_tm=0.9076,
    ),
)

SUMMARY_REQUIRED = {
    "num_samples",
    "bit_mean_bb_rmsd",
    "bit_mean_bb_tmscore",
    "fm_mean_bb_rmsd",
    "fm_mean_bb_tmscore",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_count",
    "rmsd_improved_percent",
    "mean_tmscore_improvement",
    "median_tmscore_improvement",
    "tmscore_improved_count",
    "tmscore_improved_percent",
    "relative_mean_rmsd_reduction_percent",
    "relative_mean_tmscore_increase_percent",
}
STATISTICS_REQUIRED = {
    "benchmark",
    "metric",
    "num_samples",
    "mean_improvement",
    "median_improvement",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
    "wilcoxon_raw_p",
    "wilcoxon_holm_p",
    "wilcoxon_holm_significant_0_05",
    "rank_biserial",
    "improved_count",
    "worsened_count",
    "tie_count",
    "improved_percent",
    "binomial_two_sided_p",
}
QUARTILE_REQUIRED = {
    "benchmark",
    "group_type",
    "quartile",
    "num_samples",
    "mean_bit_rmsd",
    "mean_fm_rmsd",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_percent",
    "mean_bit_tmscore",
    "mean_fm_tmscore",
    "mean_tmscore_improvement",
    "median_tmscore_improvement",
    "tmscore_improved_percent",
}

MAIN_PERFORMANCE_COLUMNS = (
    "benchmark",
    "method",
    "result_type",
    "num_samples",
    "bb_rmsd",
    "bb_tmscore",
    "rmsd_delta_vs_local_bit",
    "tmscore_delta_vs_local_bit",
    "rmsd_relative_change_percent",
)
PAIRED_STATISTICAL_COLUMNS = (
    "benchmark",
    "metric",
    "num_samples",
    "mean_improvement",
    "median_improvement",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
    "bootstrap_ci_excludes_zero",
    "wilcoxon_raw_p",
    "wilcoxon_holm_p",
    "holm_significant_0_05",
    "rank_biserial",
    "improved_count",
    "worsened_count",
    "tie_count",
    "improved_percent",
    "binomial_two_sided_p",
    "interpretation",
)
LOCAL_SUMMARY_COLUMNS = (
    "benchmark",
    "N",
    "bit_rmsd",
    "fm_rmsd",
    "rmsd_improvement",
    "rmsd_relative_reduction_percent",
    "bit_tmscore",
    "fm_tmscore",
    "tmscore_improvement",
    "rmsd_improved_count",
    "rmsd_improved_percent",
    "tm_improved_count",
    "tm_improved_percent",
    "rmsd_holm_p",
    "tm_holm_p",
    "rmsd_rank_biserial",
    "tm_rank_biserial",
)
BASELINE_QUARTILE_COLUMNS = (
    "benchmark",
    "quartile",
    "num_samples",
    "mean_bit_rmsd",
    "mean_fm_rmsd",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_percent",
    "mean_bit_tmscore",
    "mean_fm_tmscore",
    "mean_tmscore_improvement",
    "median_tmscore_improvement",
    "tmscore_improved_percent",
)
FIGURE_BASELINE_COLUMNS = (
    "benchmark",
    "quartile",
    "quartile_order",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_percent",
)
FIGURE_PERFORMANCE_COLUMNS = (
    "benchmark",
    "metric",
    "bit_value",
    "fm_value",
    "improvement",
    "relative_improvement_percent",
    "better_direction",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameo_summary", type=Path, default=DEFAULT_BENCHMARKS[0].paired_summary_path)
    parser.add_argument("--pdb_date_summary", type=Path, default=DEFAULT_BENCHMARKS[1].paired_summary_path)
    parser.add_argument(
        "--statistics_csv",
        type=Path,
        default=Path(
            "generation-results/dplm2_bit_650m_final/statistics/"
            "final_paired_statistics.csv"
        ),
    )
    parser.add_argument(
        "--quartiles_csv",
        type=Path,
        default=Path(
            "generation-results/dplm2_bit_650m_final/analysis/"
            "final_success_failure_quartiles.csv"
        ),
    )
    parser.add_argument(
        "--success_failure_summary",
        type=Path,
        default=Path(
            "generation-results/dplm2_bit_650m_final/analysis/"
            "final_success_failure_summary.txt"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"input result file does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        require(not missing, f"{path} missing columns: {sorted(missing)}")
        return list(reader)


def read_single_row(path: Path, required: set[str]) -> dict[str, str]:
    rows = read_csv(path, required)
    require(len(rows) == 1, f"expected one data row in {path}, got {len(rows)}")
    return rows[0]


def finite_float(value: str, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"invalid {field} in {path}: {value!r}") from exc
    require(math.isfinite(result), f"non-finite {field} in {path}")
    return result


def integer_value(value: str, field: str, path: Path) -> int:
    result = finite_float(value, field, path)
    require(result.is_integer(), f"non-integral {field} in {path}")
    return int(result)


def bool_value(value: str, field: str, path: Path) -> bool:
    normalized = value.strip().lower()
    require(normalized in {"true", "false"}, f"invalid {field} in {path}: {value!r}")
    return normalized == "true"


def load_and_validate_summaries(
    specs: Sequence[BenchmarkSpec], paths: Sequence[Path]
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    numeric_fields = SUMMARY_REQUIRED.difference(
        {"num_samples", "rmsd_improved_count", "tmscore_improved_count"}
    )
    for spec, path in zip(specs, paths):
        raw = read_single_row(path, SUMMARY_REQUIRED)
        parsed: dict[str, Any] = {
            field: finite_float(raw[field], field, path) for field in numeric_fields
        }
        for field in ("num_samples", "rmsd_improved_count", "tmscore_improved_count"):
            parsed[field] = integer_value(raw[field], field, path)
        require(parsed["num_samples"] == spec.expected_n, f"{spec.display_name} N mismatch")
        checks = (
            ("bit_mean_bb_rmsd", spec.expected_bit_rmsd, REFERENCE_ABS_TOL),
            ("fm_mean_bb_rmsd", spec.expected_fm_rmsd, REFERENCE_ABS_TOL),
            ("bit_mean_bb_tmscore", spec.expected_bit_tm, REFERENCE_ABS_TOL),
            ("fm_mean_bb_tmscore", spec.expected_fm_tm, REFERENCE_ABS_TOL),
            (
                "rmsd_improved_percent",
                spec.expected_rmsd_improved_percent,
                PERCENT_REFERENCE_ABS_TOL,
            ),
            (
                "tmscore_improved_percent",
                spec.expected_tm_improved_percent,
                PERCENT_REFERENCE_ABS_TOL,
            ),
        )
        for field, expected, tolerance in checks:
            require(
                math.isclose(parsed[field], expected, rel_tol=0.0, abs_tol=tolerance),
                f"{spec.display_name} {field} differs from reference check",
            )
        require(
            parsed["rmsd_improved_count"] == spec.expected_rmsd_improved_count,
            f"{spec.display_name} RMSD improved count mismatch",
        )
        require(
            parsed["tmscore_improved_count"] == spec.expected_tm_improved_count,
            f"{spec.display_name} TM improved count mismatch",
        )
        require(
            math.isclose(
                parsed["mean_rmsd_improvement"],
                parsed["bit_mean_bb_rmsd"] - parsed["fm_mean_bb_rmsd"],
                rel_tol=ALLCLOSE_RTOL,
                abs_tol=ALLCLOSE_ATOL,
            ),
            f"{spec.display_name} RMSD improvement definition mismatch",
        )
        require(
            math.isclose(
                parsed["mean_tmscore_improvement"],
                parsed["fm_mean_bb_tmscore"] - parsed["bit_mean_bb_tmscore"],
                rel_tol=ALLCLOSE_RTOL,
                abs_tol=ALLCLOSE_ATOL,
            ),
            f"{spec.display_name} TM improvement definition mismatch",
        )
        summaries[spec.key] = parsed
    return summaries


def sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def load_and_validate_statistics(
    path: Path, summaries: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    raw_rows = read_csv(path, STATISTICS_REQUIRED)
    require(len(raw_rows) == 4, f"statistical CSV must have exactly four rows: {path}")
    parsed_rows: dict[tuple[str, str], dict[str, Any]] = {}
    numeric_fields = STATISTICS_REQUIRED.difference(
        {"benchmark", "metric", "num_samples", "improved_count", "worsened_count", "tie_count", "wilcoxon_holm_significant_0_05"}
    )
    for raw in raw_rows:
        benchmark = raw["benchmark"]
        metric = raw["metric"]
        key = (benchmark, metric)
        require(benchmark in summaries, f"unexpected benchmark in statistics: {benchmark}")
        require(metric in {"rmsd", "tmscore"}, f"unexpected metric: {metric}")
        require(key not in parsed_rows, f"duplicate statistical row: {key}")
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "metric": metric,
            "num_samples": integer_value(raw["num_samples"], "num_samples", path),
            "improved_count": integer_value(raw["improved_count"], "improved_count", path),
            "worsened_count": integer_value(raw["worsened_count"], "worsened_count", path),
            "tie_count": integer_value(raw["tie_count"], "tie_count", path),
            "wilcoxon_holm_significant_0_05": bool_value(
                raw["wilcoxon_holm_significant_0_05"],
                "wilcoxon_holm_significant_0_05",
                path,
            ),
        }
        row.update({field: finite_float(raw[field], field, path) for field in numeric_fields})
        require(row["num_samples"] == summaries[benchmark]["num_samples"], f"{key} N mismatch")
        require(
            row["improved_count"] + row["worsened_count"] + row["tie_count"]
            == row["num_samples"],
            f"{key} direction counts do not sum to N",
        )
        summary_prefix = "rmsd" if metric == "rmsd" else "tmscore"
        require(
            row["improved_count"] == summaries[benchmark][f"{summary_prefix}_improved_count"],
            f"{key} improved count differs from paired summary",
        )
        require(
            math.isclose(
                row["improved_percent"],
                summaries[benchmark][f"{summary_prefix}_improved_percent"],
                rel_tol=0.0,
                abs_tol=ALLCLOSE_ATOL,
            ),
            f"{key} improved percent differs from paired summary",
        )
        ci_excludes_zero = (
            row["bootstrap_ci95_lower"] > 0.0
            or row["bootstrap_ci95_upper"] < 0.0
        )
        holm_significant = row["wilcoxon_holm_p"] < 0.05
        require(
            holm_significant == row["wilcoxon_holm_significant_0_05"],
            f"{key} Holm significance flag mismatch",
        )
        interpretation = (
            "paired rank shift supported"
            if holm_significant
            else "no statistically supported paired rank shift"
        )
        if not ci_excludes_zero:
            interpretation += "; mean CI includes zero"
        if sign(row["mean_improvement"]) != sign(row["median_improvement"]):
            interpretation += "; mean-median direction mismatch"
        row["bootstrap_ci_excludes_zero"] = ci_excludes_zero
        row["holm_significant_0_05"] = holm_significant
        row["interpretation"] = interpretation
        parsed_rows[key] = row
    require(
        set(parsed_rows)
        == {
            ("cameo2022", "rmsd"),
            ("cameo2022", "tmscore"),
            ("PDB_date", "rmsd"),
            ("PDB_date", "tmscore"),
        },
        "statistical CSV benchmark/metric rows are incomplete",
    )
    return parsed_rows


def load_baseline_quartiles(path: Path) -> list[dict[str, Any]]:
    raw_rows = read_csv(path, QUARTILE_REQUIRED)
    numeric_fields = QUARTILE_REQUIRED.difference(
        {"benchmark", "group_type", "quartile", "num_samples"}
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw["group_type"] != "baseline_rmsd":
            continue
        row: dict[str, Any] = {
            "benchmark": raw["benchmark"],
            "quartile": raw["quartile"],
            "num_samples": integer_value(raw["num_samples"], "num_samples", path),
        }
        row.update({field: finite_float(raw[field], field, path) for field in numeric_fields})
        rows.append(row)
    require(len(rows) == 8, "expected four baseline RMSD quartiles per benchmark")
    for benchmark, expected_n in (("cameo2022", 163), ("PDB_date", 442)):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        require(len(benchmark_rows) == 4, f"{benchmark} baseline quartile count mismatch")
        require(
            {row["quartile"] for row in benchmark_rows} == {"Q1", "Q2", "Q3", "Q4"},
            f"{benchmark} baseline quartile labels mismatch",
        )
        require(
            sum(int(row["num_samples"]) for row in benchmark_rows) == expected_n,
            f"{benchmark} baseline quartile sample counts do not sum to N",
        )
        ordered = sorted(benchmark_rows, key=lambda row: int(str(row["quartile"])[1:]))
        mean_bit_rmsd = [float(row["mean_bit_rmsd"]) for row in ordered]
        require(
            all(left <= right for left, right in zip(mean_bit_rmsd, mean_bit_rmsd[1:])),
            f"{benchmark} baseline quartiles are not ordered by Bit RMSD",
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["benchmark"] == "cameo2022" else 1,
            int(str(row["quartile"])[1:]),
        ),
    )


def validate_success_failure_summary(path: Path) -> None:
    require(path.is_file(), f"success/failure summary does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    for fragment in (
        "POST-HOC ANALYSIS ONLY",
        "NOT USED FOR MODEL OR HYPERPARAMETER SELECTION",
        "FINAL_SUCCESS_FAILURE_ANALYSIS_PASS",
    ):
        require(fragment in text, f"success/failure summary missing: {fragment}")


def make_main_performance_rows(
    specs: Sequence[BenchmarkSpec], summaries: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        local = summaries[spec.key]
        common = {
            "benchmark": spec.key,
            "rmsd_delta_vs_local_bit": "",
            "tmscore_delta_vs_local_bit": "",
            "rmsd_relative_change_percent": "",
        }
        rows.extend(
            [
                common
                | {
                    "method": "Paper Bit",
                    "result_type": "PAPER_REFERENCE",
                    "num_samples": "",
                    "bb_rmsd": spec.paper_bit_rmsd,
                    "bb_tmscore": spec.paper_bit_tm,
                },
                common
                | {
                    "method": "Paper Bit + RESDIFF",
                    "result_type": "PAPER_REFERENCE",
                    "num_samples": "",
                    "bb_rmsd": spec.paper_resdiff_rmsd,
                    "bb_tmscore": spec.paper_resdiff_tm,
                },
                common
                | {
                    "method": "Local Bit",
                    "result_type": "LOCAL_REPRODUCTION",
                    "num_samples": spec.expected_n,
                    "bb_rmsd": local["bit_mean_bb_rmsd"],
                    "bb_tmscore": local["bit_mean_bb_tmscore"],
                },
                {
                    "benchmark": spec.key,
                    "method": "Local Bit + Latent FM",
                    "result_type": "LOCAL_REPRODUCTION",
                    "num_samples": spec.expected_n,
                    "bb_rmsd": local["fm_mean_bb_rmsd"],
                    "bb_tmscore": local["fm_mean_bb_tmscore"],
                    "rmsd_delta_vs_local_bit": local["mean_rmsd_improvement"],
                    "tmscore_delta_vs_local_bit": local["mean_tmscore_improvement"],
                    "rmsd_relative_change_percent": local[
                        "relative_mean_rmsd_reduction_percent"
                    ],
                },
            ]
        )
    return rows


def validate_performance_result_types(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 8, "main performance table must have eight rows")
    for row in rows:
        is_paper = str(row["method"]).startswith("Paper ")
        require(
            (row["result_type"] == "PAPER_REFERENCE") == is_paper,
            f"paper/local result type mixed for {row['method']}",
        )
        if is_paper:
            require(row["num_samples"] == "", "paper N must remain unspecified")
            require(row["rmsd_delta_vs_local_bit"] == "", "paper/local RMSD delta mixed")
            require(row["tmscore_delta_vs_local_bit"] == "", "paper/local TM delta mixed")
        if row["method"] != "Local Bit + Latent FM":
            require(
                row["rmsd_relative_change_percent"] == "",
                "relative local delta is only defined for Local Bit + Latent FM",
            )


def make_statistical_rows(
    statistics: Mapping[tuple[str, str], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {column: statistics[key][column] for column in PAIRED_STATISTICAL_COLUMNS}
        for key in (
            ("cameo2022", "rmsd"),
            ("cameo2022", "tmscore"),
            ("PDB_date", "rmsd"),
            ("PDB_date", "tmscore"),
        )
    ]


def make_local_summary_rows(
    specs: Sequence[BenchmarkSpec],
    summaries: Mapping[str, Mapping[str, Any]],
    statistics: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        local = summaries[spec.key]
        rmsd_stat = statistics[(spec.key, "rmsd")]
        tm_stat = statistics[(spec.key, "tmscore")]
        rows.append(
            {
                "benchmark": spec.key,
                "N": spec.expected_n,
                "bit_rmsd": local["bit_mean_bb_rmsd"],
                "fm_rmsd": local["fm_mean_bb_rmsd"],
                "rmsd_improvement": local["mean_rmsd_improvement"],
                "rmsd_relative_reduction_percent": local[
                    "relative_mean_rmsd_reduction_percent"
                ],
                "bit_tmscore": local["bit_mean_bb_tmscore"],
                "fm_tmscore": local["fm_mean_bb_tmscore"],
                "tmscore_improvement": local["mean_tmscore_improvement"],
                "rmsd_improved_count": local["rmsd_improved_count"],
                "rmsd_improved_percent": local["rmsd_improved_percent"],
                "tm_improved_count": local["tmscore_improved_count"],
                "tm_improved_percent": local["tmscore_improved_percent"],
                "rmsd_holm_p": rmsd_stat["wilcoxon_holm_p"],
                "tm_holm_p": tm_stat["wilcoxon_holm_p"],
                "rmsd_rank_biserial": rmsd_stat["rank_biserial"],
                "tm_rank_biserial": tm_stat["rank_biserial"],
            }
        )
    return rows


def make_figure_baseline_rows(
    quartiles: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": row["benchmark"],
            "quartile": row["quartile"],
            "quartile_order": int(str(row["quartile"])[1:]),
            "mean_rmsd_improvement": row["mean_rmsd_improvement"],
            "median_rmsd_improvement": row["median_rmsd_improvement"],
            "rmsd_improved_percent": row["rmsd_improved_percent"],
        }
        for row in quartiles
    ]


def make_figure_performance_rows(
    specs: Sequence[BenchmarkSpec], summaries: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        local = summaries[spec.key]
        rows.extend(
            [
                {
                    "benchmark": spec.key,
                    "metric": "BB_RMSD",
                    "bit_value": local["bit_mean_bb_rmsd"],
                    "fm_value": local["fm_mean_bb_rmsd"],
                    "improvement": local["mean_rmsd_improvement"],
                    "relative_improvement_percent": local[
                        "relative_mean_rmsd_reduction_percent"
                    ],
                    "better_direction": "lower",
                },
                {
                    "benchmark": spec.key,
                    "metric": "TM_SCORE",
                    "bit_value": local["bit_mean_bb_tmscore"],
                    "fm_value": local["fm_mean_bb_tmscore"],
                    "improvement": local["mean_tmscore_improvement"],
                    "relative_improvement_percent": local[
                        "relative_mean_tmscore_increase_percent"
                    ],
                    "better_direction": "higher",
                },
            ]
        )
    return rows


def format_float(value: Any) -> str:
    return f"{float(value):.9g}"


def build_final_summary_text(
    specs: Sequence[BenchmarkSpec],
    summaries: Mapping[str, Mapping[str, Any]],
    statistics: Mapping[tuple[str, str], Mapping[str, Any]],
    quartiles: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["[FINAL RESULTS SUMMARY]", "", "[MAIN LOCAL RESULTS]", ""]
    for spec in specs:
        local = summaries[spec.key]
        lines.extend(
            [
                f"{spec.display_name}:",
                f"N: {spec.expected_n}",
                f"Bit RMSD: {format_float(local['bit_mean_bb_rmsd'])}",
                f"FM RMSD: {format_float(local['fm_mean_bb_rmsd'])}",
                f"RMSD improvement: {format_float(local['mean_rmsd_improvement'])}",
                f"Bit TM: {format_float(local['bit_mean_bb_tmscore'])}",
                f"FM TM: {format_float(local['fm_mean_bb_tmscore'])}",
                f"TM improvement: {format_float(local['mean_tmscore_improvement'])}",
                f"RMSD improved %: {format_float(local['rmsd_improved_percent'])}",
                f"TM improved %: {format_float(local['tmscore_improved_percent'])}",
                "",
            ]
        )

    lines.extend(["[PAIRED STATISTICS]", ""])
    for spec in specs:
        for metric, display_metric in (("rmsd", "RMSD"), ("tmscore", "TM")):
            row = statistics[(spec.key, metric)]
            lines.extend(
                [
                    f"{spec.display_name} {display_metric}:",
                    "bootstrap CI: "
                    f"[{format_float(row['bootstrap_ci95_lower'])}, "
                    f"{format_float(row['bootstrap_ci95_upper'])}]",
                    f"Holm p: {format_float(row['wilcoxon_holm_p'])}",
                    f"rank-biserial: {format_float(row['rank_biserial'])}",
                    f"interpretation: {row['interpretation']}",
                    "",
                ]
            )

    lines.extend(
        [
            "[PAPER REFERENCES]",
            "",
            "PAPER REFERENCE ONLY",
            "NOT LOCAL PAIRED RESULT",
            "",
        ]
    )
    for spec in specs:
        lines.extend(
            [
                f"{spec.display_name}:",
                f"Paper Bit: RMSD={spec.paper_bit_rmsd:.4f}, TM-score={spec.paper_bit_tm:.4f}",
                "Paper Bit + RESDIFF: "
                f"RMSD={spec.paper_resdiff_rmsd:.4f}, TM-score={spec.paper_resdiff_tm:.4f}",
                "",
            ]
        )

    lines.extend(["[POST-HOC DESCRIPTIVE PATTERN]", ""])
    for spec in specs:
        benchmark_rows = {
            row["quartile"]: row
            for row in quartiles
            if row["benchmark"] == spec.key
        }
        lines.extend(
            [
                f"{spec.display_name} baseline RMSD:",
                "Q1 mean RMSD improvement: "
                f"{format_float(benchmark_rows['Q1']['mean_rmsd_improvement'])}",
                "Q4 mean RMSD improvement: "
                f"{format_float(benchmark_rows['Q4']['mean_rmsd_improvement'])}",
                "",
            ]
        )
    lines.extend(
        [
            "POST-HOC ANALYSIS ONLY",
            "NOT USED FOR MODEL OR HYPERPARAMETER SELECTION",
        ]
    )
    return "\n".join(lines) + "\n"


def build_metadata_text() -> str:
    require(FINAL_ALPHA == 0.50, "final FM alpha must remain fixed at 0.50")
    return """[FINAL RESULTS METADATA]

FM checkpoint:
experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt

optimizer_step:
100000

FM alpha:
0.50

FM Euler steps:
100

DPLM:
airkingbd/dplm2_bit_650m

DPLM generation:
max_iter=100
unmasking_strategy=deterministic
sampling_strategy=argmax

CAMEO evaluation N:
163

PDB-date evaluation N:
442

alpha selection:
internal validation only
not selected using CAMEO/PDB-date

statistics:
paired bootstrap 10000
seed 42
two-sided Wilcoxon
Holm correction across 4 tests

post-hoc analysis:
descriptive/exploratory only

baseline difficulty quartiles:
Q1 = lowest Bit RMSD
Q4 = highest Bit RMSD
descriptive only; not used for model or hyperparameter selection
"""


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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    require(FINAL_ALPHA == 0.50, "final FM alpha must remain fixed at 0.50")
    specs = (
        BenchmarkSpec(
            **{
                **DEFAULT_BENCHMARKS[0].__dict__,
                "paired_summary_path": args.cameo_summary,
            }
        ),
        BenchmarkSpec(
            **{
                **DEFAULT_BENCHMARKS[1].__dict__,
                "paired_summary_path": args.pdb_date_summary,
            }
        ),
    )
    summaries = load_and_validate_summaries(
        specs, (args.cameo_summary, args.pdb_date_summary)
    )
    statistics = load_and_validate_statistics(args.statistics_csv, summaries)
    quartiles = load_baseline_quartiles(args.quartiles_csv)
    validate_success_failure_summary(args.success_failure_summary)

    main_performance = make_main_performance_rows(specs, summaries)
    validate_performance_result_types(main_performance)
    paired_statistics = make_statistical_rows(statistics)
    local_summary = make_local_summary_rows(specs, summaries, statistics)
    figure_baseline = make_figure_baseline_rows(quartiles)
    figure_performance = make_figure_performance_rows(specs, summaries)
    final_summary = build_final_summary_text(specs, summaries, statistics, quartiles)
    metadata = build_metadata_text()

    atomic_write_csv(MAIN_PERFORMANCE_CSV, MAIN_PERFORMANCE_COLUMNS, main_performance)
    atomic_write_csv(PAIRED_STATISTICAL_CSV, PAIRED_STATISTICAL_COLUMNS, paired_statistics)
    atomic_write_csv(LOCAL_SUMMARY_CSV, LOCAL_SUMMARY_COLUMNS, local_summary)
    atomic_write_csv(BASELINE_QUARTILES_CSV, BASELINE_QUARTILE_COLUMNS, quartiles)
    atomic_write_csv(FIGURE_BASELINE_CSV, FIGURE_BASELINE_COLUMNS, figure_baseline)
    atomic_write_csv(FIGURE_PERFORMANCE_CSV, FIGURE_PERFORMANCE_COLUMNS, figure_performance)
    atomic_write_text(FINAL_SUMMARY_TXT, final_summary)
    atomic_write_text(FINAL_METADATA_TXT, metadata)

    print("[FINAL RESULTS TABLE PREPARATION]")
    for spec in specs:
        local = summaries[spec.key]
        print(f"\n{spec.display_name}:")
        print(f"Bit RMSD: {format_float(local['bit_mean_bb_rmsd'])}")
        print(f"FM RMSD: {format_float(local['fm_mean_bb_rmsd'])}")
        print(f"Bit TM: {format_float(local['bit_mean_bb_tmscore'])}")
        print(f"FM TM: {format_float(local['fm_mean_bb_tmscore'])}")
    print("\ncreated files:")
    for path in (
        MAIN_PERFORMANCE_CSV,
        PAIRED_STATISTICAL_CSV,
        LOCAL_SUMMARY_CSV,
        BASELINE_QUARTILES_CSV,
        FIGURE_BASELINE_CSV,
        FIGURE_PERFORMANCE_CSV,
        FINAL_SUMMARY_TXT,
        FINAL_METADATA_TXT,
    ):
        print(path)
    print("\nPaper reference separated: YES")
    print("Test-set hyperparameter tuning performed: NO")
    print("\nFINAL_RESULTS_TABLES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

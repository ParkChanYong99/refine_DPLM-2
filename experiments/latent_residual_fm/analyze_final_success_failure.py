#!/usr/bin/env python3
"""Post-hoc descriptive success/failure analysis of final paired test results."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ANALYSIS_ROLE = "POST-HOC EXPLORATORY"
POST_HOC_NOTICE = (
    "POST-HOC ANALYSIS ONLY\n"
    "NOT USED FOR MODEL OR HYPERPARAMETER SELECTION"
)
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12
REFERENCE_ABS_TOL = 1e-7

OUTPUT_DIR = Path("generation-results/dplm2_bit_650m_final/analysis")
SAMPLES_CSV = OUTPUT_DIR / "final_success_failure_samples.csv"
CORRELATIONS_CSV = OUTPUT_DIR / "final_success_failure_correlations.csv"
QUARTILES_CSV = OUTPUT_DIR / "final_success_failure_quartiles.csv"
EXTREMES_CSV = OUTPUT_DIR / "final_success_failure_extremes.csv"
SUMMARY_TXT = OUTPUT_DIR / "final_success_failure_summary.txt"

PAIRED_REQUIRED_COLUMNS = {
    "sample_id",
    "length",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "rmsd_improvement",
    "bit_bb_tmscore",
    "fm_bb_tmscore",
    "tmscore_improvement",
}
MANIFEST_REQUIRED_COLUMNS = {
    "sample_id",
    "length",
    "generation_success",
}
RESIDUAL_COLUMNS = ("r_hat_abs_mean", "r_hat_abs_max")

SAMPLE_COLUMNS = (
    "benchmark",
    "sample_id",
    "length",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "rmsd_improvement",
    "bit_bb_tmscore",
    "fm_bb_tmscore",
    "tmscore_improvement",
    "r_hat_abs_mean",
    "r_hat_abs_max",
    "baseline_rmsd_quartile",
    "length_quartile",
)
CORRELATION_COLUMNS = (
    "benchmark",
    "analysis",
    "x_variable",
    "y_variable",
    "spearman_rho",
    "two_sided_p",
    "num_samples",
    "analysis_role",
)
QUARTILE_COLUMNS = (
    "benchmark",
    "group_type",
    "quartile",
    "num_samples",
    "mean_length",
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
EXTREME_COLUMNS = (
    "benchmark",
    "metric",
    "selection",
    "rank",
    "sample_id",
    "length",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "rmsd_improvement",
    "bit_bb_tmscore",
    "fm_bb_tmscore",
    "tmscore_improvement",
    "degradation",
    "r_hat_abs_mean",
    "r_hat_abs_max",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    paired_path: Path
    manifest_path: Path
    expected_samples: int
    reference_bit_mean_rmsd: float
    reference_fm_mean_rmsd: float
    reference_rmsd_improved: int
    reference_tm_improved: int


DEFAULT_SPECS = (
    BenchmarkSpec(
        key="cameo2022",
        display_name="CAMEO",
        paired_path=Path(
            "generation-results/dplm2_bit_650m_final/cameo2022/comparison/"
            "cameo2022_bit_vs_fm_paired.csv"
        ),
        manifest_path=Path(
            "generation-results/dplm2_bit_650m_final/cameo2022/"
            "generation_manifest.csv"
        ),
        expected_samples=163,
        reference_bit_mean_rmsd=6.28888164,
        reference_fm_mean_rmsd=6.19972204,
        reference_rmsd_improved=91,
        reference_tm_improved=98,
    ),
    BenchmarkSpec(
        key="PDB_date",
        display_name="PDB-DATE",
        paired_path=Path(
            "generation-results/dplm2_bit_650m_final/PDB_date/comparison/"
            "pdb_date_bit_vs_fm_paired.csv"
        ),
        manifest_path=Path(
            "generation-results/dplm2_bit_650m_final/PDB_date/"
            "generation_manifest.csv"
        ),
        expected_samples=442,
        reference_bit_mean_rmsd=3.11376375,
        reference_fm_mean_rmsd=3.10744498,
        reference_rmsd_improved=216,
        reference_tm_improved=211,
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameo_paired_csv", type=Path, default=DEFAULT_SPECS[0].paired_path)
    parser.add_argument("--cameo_manifest", type=Path, default=DEFAULT_SPECS[0].manifest_path)
    parser.add_argument("--pdb_date_paired_csv", type=Path, default=DEFAULT_SPECS[1].paired_path)
    parser.add_argument("--pdb_date_manifest", type=Path, default=DEFAULT_SPECS[1].manifest_path)
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], set[str]]:
    require(path.is_file(), f"CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = required.difference(fieldnames)
        require(not missing, f"{path} missing columns: {sorted(missing)}")
        return list(reader), fieldnames


def finite_float(value: str, field: str, path: Path, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"invalid {field} at {path}:{row_number}: {value!r}"
        ) from exc
    require(math.isfinite(result), f"non-finite {field} at {path}:{row_number}")
    return result


def integer_value(value: str, field: str, path: Path, row_number: int) -> int:
    result = finite_float(value, field, path, row_number)
    require(result.is_integer(), f"non-integral {field} at {path}:{row_number}")
    return int(result)


def true_value(value: str) -> bool:
    return value.strip().lower() == "true"


def unique_by_id(
    rows: Sequence[dict[str, str]], path: Path, label: str
) -> dict[str, tuple[int, dict[str, str]]]:
    indexed: dict[str, tuple[int, dict[str, str]]] = {}
    duplicates: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        require(sample_id, f"empty sample_id at {path}:{row_number}")
        if sample_id in indexed:
            duplicates.append(sample_id)
        else:
            indexed[sample_id] = (row_number, row)
    require(not duplicates, f"{label} duplicate sample IDs: {sorted(set(duplicates))}")
    return indexed


def load_joined_samples(spec: BenchmarkSpec) -> tuple[list[dict[str, Any]], bool]:
    paired_rows, _ = read_csv(spec.paired_path, PAIRED_REQUIRED_COLUMNS)
    manifest_rows, manifest_columns = read_csv(
        spec.manifest_path, MANIFEST_REQUIRED_COLUMNS
    )
    require(
        len(paired_rows) == spec.expected_samples,
        f"{spec.display_name} expected {spec.expected_samples} paired rows, "
        f"got {len(paired_rows)}",
    )
    paired_by_id = unique_by_id(paired_rows, spec.paired_path, "paired CSV")
    manifest_by_id = unique_by_id(manifest_rows, spec.manifest_path, "manifest")
    require(
        len(paired_by_id) == spec.expected_samples,
        f"{spec.display_name} unique paired sample count mismatch",
    )
    joined_ids = set(paired_by_id).intersection(manifest_by_id)
    require(
        joined_ids == set(paired_by_id),
        f"{spec.display_name} paired samples missing from manifest: "
        f"{sorted(set(paired_by_id) - joined_ids)}",
    )
    require(
        len(joined_ids) == spec.expected_samples,
        f"{spec.display_name} inner join produced {len(joined_ids)} samples",
    )

    residual_columns_present = all(
        column in manifest_columns for column in RESIDUAL_COLUMNS
    )
    require(
        residual_columns_present
        or not any(column in manifest_columns for column in RESIDUAL_COLUMNS),
        f"{spec.display_name} manifest has only one residual magnitude column",
    )

    joined: list[dict[str, Any]] = []
    for sample_id in sorted(joined_ids):
        paired_number, paired = paired_by_id[sample_id]
        manifest_number, manifest = manifest_by_id[sample_id]
        paired_length = integer_value(
            paired["length"], "length", spec.paired_path, paired_number
        )
        manifest_length = integer_value(
            manifest["length"], "length", spec.manifest_path, manifest_number
        )
        require(
            paired_length == manifest_length,
            f"{spec.display_name} length mismatch for {sample_id}: "
            f"paired={paired_length}, manifest={manifest_length}",
        )
        require(
            true_value(manifest["generation_success"]),
            f"{spec.display_name} manifest generation_success is not True for {sample_id}",
        )

        numeric_values = {
            column: finite_float(
                paired[column], column, spec.paired_path, paired_number
            )
            for column in (
                "bit_bb_rmsd",
                "fm_bb_rmsd",
                "rmsd_improvement",
                "bit_bb_tmscore",
                "fm_bb_tmscore",
                "tmscore_improvement",
            )
        }
        recomputed_rmsd = (
            numeric_values["bit_bb_rmsd"] - numeric_values["fm_bb_rmsd"]
        )
        recomputed_tm = (
            numeric_values["fm_bb_tmscore"] - numeric_values["bit_bb_tmscore"]
        )
        require(
            math.isclose(
                numeric_values["rmsd_improvement"],
                recomputed_rmsd,
                rel_tol=ALLCLOSE_RTOL,
                abs_tol=ALLCLOSE_ATOL,
            ),
            f"{spec.display_name} RMSD improvement mismatch for {sample_id}",
        )
        require(
            math.isclose(
                numeric_values["tmscore_improvement"],
                recomputed_tm,
                rel_tol=ALLCLOSE_RTOL,
                abs_tol=ALLCLOSE_ATOL,
            ),
            f"{spec.display_name} TM improvement mismatch for {sample_id}",
        )

        if residual_columns_present:
            r_hat_abs_mean: float | str = finite_float(
                manifest["r_hat_abs_mean"],
                "r_hat_abs_mean",
                spec.manifest_path,
                manifest_number,
            )
            r_hat_abs_max: float | str = finite_float(
                manifest["r_hat_abs_max"],
                "r_hat_abs_max",
                spec.manifest_path,
                manifest_number,
            )
        else:
            r_hat_abs_mean = ""
            r_hat_abs_max = ""

        joined.append(
            {
                "benchmark": spec.key,
                "sample_id": sample_id,
                "length": paired_length,
                "bit_bb_rmsd": numeric_values["bit_bb_rmsd"],
                "fm_bb_rmsd": numeric_values["fm_bb_rmsd"],
                "rmsd_improvement": recomputed_rmsd,
                "bit_bb_tmscore": numeric_values["bit_bb_tmscore"],
                "fm_bb_tmscore": numeric_values["fm_bb_tmscore"],
                "tmscore_improvement": recomputed_tm,
                "r_hat_abs_mean": r_hat_abs_mean,
                "r_hat_abs_max": r_hat_abs_max,
                "baseline_rmsd_quartile": "",
                "length_quartile": "",
            }
        )
    return joined, residual_columns_present


def values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    result = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    require(result.shape == (len(rows),), f"{key} array shape mismatch")
    require(bool(np.isfinite(result).all()), f"{key} contains NaN/Inf")
    return result


def validate_reference(spec: BenchmarkSpec, rows: Sequence[Mapping[str, Any]]) -> None:
    bit_mean = float(np.mean(values(rows, "bit_bb_rmsd")))
    fm_mean = float(np.mean(values(rows, "fm_bb_rmsd")))
    rmsd_improved = int(np.count_nonzero(values(rows, "rmsd_improvement") > 0.0))
    tm_improved = int(np.count_nonzero(values(rows, "tmscore_improvement") > 0.0))
    require(
        math.isclose(bit_mean, spec.reference_bit_mean_rmsd, abs_tol=REFERENCE_ABS_TOL),
        f"{spec.display_name} Bit mean RMSD differs from reference check",
    )
    require(
        math.isclose(fm_mean, spec.reference_fm_mean_rmsd, abs_tol=REFERENCE_ABS_TOL),
        f"{spec.display_name} FM mean RMSD differs from reference check",
    )
    require(
        rmsd_improved == spec.reference_rmsd_improved,
        f"{spec.display_name} RMSD improved count differs from reference check",
    )
    require(
        tm_improved == spec.reference_tm_improved,
        f"{spec.display_name} TM improved count differs from reference check",
    )


def correlation_row(
    benchmark: str,
    analysis: str,
    x_variable: str,
    y_variable: str,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    require(x.shape == y.shape and x.ndim == 1, "correlation arrays are misaligned")
    result = spearmanr(x, y, alternative="two-sided")
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    require(math.isfinite(rho), f"non-finite Spearman rho: {x_variable} vs {y_variable}")
    require(math.isfinite(p_value), f"non-finite Spearman p: {x_variable} vs {y_variable}")
    return {
        "benchmark": benchmark,
        "analysis": analysis,
        "x_variable": x_variable,
        "y_variable": y_variable,
        "spearman_rho": rho,
        "two_sided_p": p_value,
        "num_samples": x.size,
        "analysis_role": ANALYSIS_ROLE,
    }


def build_correlations(
    rows: Sequence[Mapping[str, Any]], residuals_available: bool
) -> list[dict[str, Any]]:
    benchmark = str(rows[0]["benchmark"])
    length = values(rows, "length")
    bit_rmsd = values(rows, "bit_bb_rmsd")
    bit_tm = values(rows, "bit_bb_tmscore")
    rmsd_improvement = values(rows, "rmsd_improvement")
    tm_improvement = values(rows, "tmscore_improvement")
    specifications = [
        ("length_effect", "length", "rmsd_improvement", length, rmsd_improvement),
        ("length_effect", "length", "tmscore_improvement", length, tm_improvement),
        (
            "baseline_difficulty",
            "bit_bb_rmsd",
            "rmsd_improvement",
            bit_rmsd,
            rmsd_improvement,
        ),
        (
            "baseline_difficulty",
            "bit_bb_tmscore",
            "tmscore_improvement",
            bit_tm,
            tm_improvement,
        ),
    ]
    if residuals_available:
        r_mean = values(rows, "r_hat_abs_mean")
        r_max = values(rows, "r_hat_abs_max")
        specifications.extend(
            [
                (
                    "residual_magnitude",
                    "r_hat_abs_mean",
                    "rmsd_improvement",
                    r_mean,
                    rmsd_improvement,
                ),
                (
                    "residual_magnitude",
                    "r_hat_abs_mean",
                    "tmscore_improvement",
                    r_mean,
                    tm_improvement,
                ),
                (
                    "residual_magnitude",
                    "r_hat_abs_max",
                    "rmsd_improvement",
                    r_max,
                    rmsd_improvement,
                ),
                (
                    "residual_magnitude",
                    "r_hat_abs_max",
                    "tmscore_improvement",
                    r_max,
                    tm_improvement,
                ),
                (
                    "residual_magnitude_absolute_change",
                    "r_hat_abs_mean",
                    "absolute_rmsd_change",
                    r_mean,
                    np.abs(rmsd_improvement),
                ),
            ]
        )
    return [
        correlation_row(benchmark, analysis, x_name, y_name, x, y)
        for analysis, x_name, y_name, x, y in specifications
    ]


def assign_empirical_quartiles(
    rows: list[dict[str, Any]], value_key: str, output_key: str
) -> list[list[dict[str, Any]]]:
    # Stable sorting plus array_split creates four near-equal descriptive groups;
    # Q1 has the lowest values and Q4 the highest. Sample ID order breaks ties.
    order = sorted(
        range(len(rows)),
        key=lambda index: (float(rows[index][value_key]), str(rows[index]["sample_id"])),
    )
    index_groups = np.array_split(np.asarray(order, dtype=np.int64), 4)
    groups: list[list[dict[str, Any]]] = []
    for quartile_number, indices in enumerate(index_groups, start=1):
        require(indices.size > 0, f"empty Q{quartile_number} for {value_key}")
        group = [rows[int(index)] for index in indices]
        for row in group:
            row[output_key] = f"Q{quartile_number}"
        groups.append(group)
    require(sum(len(group) for group in groups) == len(rows), "quartile coverage mismatch")
    return groups


def quartile_summary(
    benchmark: str,
    group_type: str,
    quartile: str,
    group: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rmsd_improvement = values(group, "rmsd_improvement")
    tm_improvement = values(group, "tmscore_improvement")
    return {
        "benchmark": benchmark,
        "group_type": group_type,
        "quartile": quartile,
        "num_samples": len(group),
        "mean_length": float(np.mean(values(group, "length"))),
        "mean_bit_rmsd": float(np.mean(values(group, "bit_bb_rmsd"))),
        "mean_fm_rmsd": float(np.mean(values(group, "fm_bb_rmsd"))),
        "mean_rmsd_improvement": float(np.mean(rmsd_improvement)),
        "median_rmsd_improvement": float(np.median(rmsd_improvement)),
        "rmsd_improved_percent": float(np.mean(rmsd_improvement > 0.0) * 100.0),
        "mean_bit_tmscore": float(np.mean(values(group, "bit_bb_tmscore"))),
        "mean_fm_tmscore": float(np.mean(values(group, "fm_bb_tmscore"))),
        "mean_tmscore_improvement": float(np.mean(tm_improvement)),
        "median_tmscore_improvement": float(np.median(tm_improvement)),
        "tmscore_improved_percent": float(np.mean(tm_improvement > 0.0) * 100.0),
    }


def build_quartiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    benchmark = str(rows[0]["benchmark"])
    baseline_groups = assign_empirical_quartiles(
        rows, "bit_bb_rmsd", "baseline_rmsd_quartile"
    )
    length_groups = assign_empirical_quartiles(rows, "length", "length_quartile")
    summaries: list[dict[str, Any]] = []
    for group_type, groups in (
        ("baseline_rmsd", baseline_groups),
        ("length", length_groups),
    ):
        for quartile_number, group in enumerate(groups, start=1):
            summaries.append(
                quartile_summary(
                    benchmark, group_type, f"Q{quartile_number}", group
                )
            )
    return summaries


def extreme_row(
    row: Mapping[str, Any], metric: str, selection: str, rank: int
) -> dict[str, Any]:
    improvement_key = "rmsd_improvement" if metric == "rmsd" else "tmscore_improvement"
    result = {
        column: row[column]
        for column in (
            "benchmark",
            "sample_id",
            "length",
            "bit_bb_rmsd",
            "fm_bb_rmsd",
            "rmsd_improvement",
            "bit_bb_tmscore",
            "fm_bb_tmscore",
            "tmscore_improvement",
            "r_hat_abs_mean",
            "r_hat_abs_max",
        )
    }
    result.update(
        {
            "metric": metric,
            "selection": selection,
            "rank": rank,
            "degradation": (
                -float(row[improvement_key]) if selection == "top_degradation" else ""
            ),
        }
    )
    return result


def build_extremes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for metric, improvement_key in (
        ("rmsd", "rmsd_improvement"),
        ("tmscore", "tmscore_improvement"),
    ):
        improvements = [row for row in rows if float(row[improvement_key]) > 0.0]
        degradations = [row for row in rows if float(row[improvement_key]) < 0.0]
        require(len(improvements) >= 10, f"fewer than 10 {metric} improvements")
        require(len(degradations) >= 10, f"fewer than 10 {metric} degradations")
        best = sorted(
            improvements,
            key=lambda row: (-float(row[improvement_key]), str(row["sample_id"])),
        )[:10]
        worst = sorted(
            degradations,
            key=lambda row: (float(row[improvement_key]), str(row["sample_id"])),
        )[:10]
        results.extend(
            extreme_row(row, metric, "top_improvement", rank)
            for rank, row in enumerate(best, start=1)
        )
        results.extend(
            extreme_row(row, metric, "top_degradation", rank)
            for rank, row in enumerate(worst, start=1)
        )
    return results


def benchmark_summary(
    rows: Sequence[Mapping[str, Any]], residuals_available: bool
) -> dict[str, Any]:
    rmsd_improvement = values(rows, "rmsd_improvement")
    tm_improvement = values(rows, "tmscore_improvement")
    summary = {
        "benchmark": rows[0]["benchmark"],
        "num_samples": len(rows),
        "mean_length": float(np.mean(values(rows, "length"))),
        "mean_bit_rmsd": float(np.mean(values(rows, "bit_bb_rmsd"))),
        "mean_fm_rmsd": float(np.mean(values(rows, "fm_bb_rmsd"))),
        "mean_rmsd_improvement": float(np.mean(rmsd_improvement)),
        "median_rmsd_improvement": float(np.median(rmsd_improvement)),
        "rmsd_improved_percent": float(np.mean(rmsd_improvement > 0.0) * 100.0),
        "mean_bit_tmscore": float(np.mean(values(rows, "bit_bb_tmscore"))),
        "mean_fm_tmscore": float(np.mean(values(rows, "fm_bb_tmscore"))),
        "mean_tmscore_improvement": float(np.mean(tm_improvement)),
        "median_tmscore_improvement": float(np.median(tm_improvement)),
        "tmscore_improved_percent": float(np.mean(tm_improvement > 0.0) * 100.0),
        "mean_r_hat_abs_mean": "",
        "median_r_hat_abs_mean": "",
    }
    if residuals_available:
        r_mean = values(rows, "r_hat_abs_mean")
        summary["mean_r_hat_abs_mean"] = float(np.mean(r_mean))
        summary["median_r_hat_abs_mean"] = float(np.median(r_mean))
    return summary


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


def fmt(value: Any) -> str:
    return "NA" if value == "" else f"{float(value):.9g}"


def find_correlation(
    correlations: Sequence[Mapping[str, Any]], benchmark: str, x: str, y: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in correlations
        if row["benchmark"] == benchmark
        and row["x_variable"] == x
        and row["y_variable"] == y
    ]
    require(len(matches) == 1, f"correlation lookup failed: {benchmark} {x} {y}")
    return matches[0]


def correlation_text(row: Mapping[str, Any]) -> str:
    return (
        f"rho={fmt(row['spearman_rho'])}, two-sided p={fmt(row['two_sided_p'])} "
        f"({ANALYSIS_ROLE})"
    )


def build_summary_text(
    specs: Sequence[BenchmarkSpec],
    samples_by_benchmark: Mapping[str, Sequence[Mapping[str, Any]]],
    correlations: Sequence[Mapping[str, Any]],
    quartiles: Sequence[Mapping[str, Any]],
    extremes: Sequence[Mapping[str, Any]],
    benchmark_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = ["[FINAL SUCCESS/FAILURE ANALYSIS]", "", POST_HOC_NOTICE, ""]
    lines.append(
        "All correlations and quartiles are descriptive/exploratory; correlation "
        "does not establish causation."
    )
    lines.append(
        "Residual-magnitude association does not by itself establish overshoot as a cause."
    )
    lines.append("")
    for spec in specs:
        benchmark = spec.key
        summary = benchmark_summaries[benchmark]
        lines.extend([f"[{spec.display_name}]", f"N: {summary['num_samples']}"])
        requested_correlations = (
            ("length", "rmsd_improvement", "length vs RMSD improvement"),
            ("length", "tmscore_improvement", "length vs TM improvement"),
            ("bit_bb_rmsd", "rmsd_improvement", "Bit RMSD vs RMSD improvement"),
            (
                "bit_bb_tmscore",
                "tmscore_improvement",
                "Bit TM-score vs TM improvement",
            ),
            ("r_hat_abs_mean", "rmsd_improvement", "r_hat mean vs RMSD improvement"),
            ("r_hat_abs_mean", "tmscore_improvement", "r_hat mean vs TM improvement"),
            ("r_hat_abs_max", "rmsd_improvement", "r_hat max vs RMSD improvement"),
            ("r_hat_abs_max", "tmscore_improvement", "r_hat max vs TM improvement"),
            (
                "r_hat_abs_mean",
                "absolute_rmsd_change",
                "r_hat mean vs absolute RMSD change",
            ),
        )
        available = {
            (str(row["x_variable"]), str(row["y_variable"]))
            for row in correlations
            if row["benchmark"] == benchmark
        }
        for x_name, y_name, label in requested_correlations:
            if (x_name, y_name) in available:
                row = find_correlation(correlations, benchmark, x_name, y_name)
                lines.append(f"{label}: {correlation_text(row)}")
            elif x_name.startswith("r_hat"):
                lines.append(f"{label}: NOT AVAILABLE")

        benchmark_quartiles = [
            row for row in quartiles if row["benchmark"] == benchmark
        ]
        lines.append("baseline RMSD quartiles (descriptive only):")
        for row in benchmark_quartiles:
            if row["group_type"] == "baseline_rmsd":
                lines.append(
                    f"  {row['quartile']}: N={row['num_samples']}, "
                    f"mean improvement={fmt(row['mean_rmsd_improvement'])}, "
                    f"median improvement={fmt(row['median_rmsd_improvement'])}, "
                    f"RMSD improved={float(row['rmsd_improved_percent']):.6f}%"
                )
        lines.append("length quartiles (descriptive only):")
        for row in benchmark_quartiles:
            if row["group_type"] == "length":
                lines.append(
                    f"  {row['quartile']}: N={row['num_samples']}, "
                    f"mean length={fmt(row['mean_length'])}, "
                    f"mean RMSD improvement={fmt(row['mean_rmsd_improvement'])}, "
                    f"mean TM improvement={fmt(row['mean_tmscore_improvement'])}"
                )

        rank_one = {
            (str(row["metric"]), str(row["selection"])): row
            for row in extremes
            if row["benchmark"] == benchmark and int(row["rank"]) == 1
        }
        best_rmsd = rank_one[("rmsd", "top_improvement")]
        worst_rmsd = rank_one[("rmsd", "top_degradation")]
        lines.append(
            "best RMSD improvement: "
            f"{best_rmsd['sample_id']} improvement={fmt(best_rmsd['rmsd_improvement'])}, "
            f"Bit={fmt(best_rmsd['bit_bb_rmsd'])}, FM={fmt(best_rmsd['fm_bb_rmsd'])}"
        )
        lines.append(
            "worst RMSD degradation: "
            f"{worst_rmsd['sample_id']} degradation={fmt(worst_rmsd['degradation'])}, "
            f"Bit={fmt(worst_rmsd['bit_bb_rmsd'])}, FM={fmt(worst_rmsd['fm_bb_rmsd'])}"
        )
        best_tm = rank_one[("tmscore", "top_improvement")]
        worst_tm = rank_one[("tmscore", "top_degradation")]
        lines.append(
            "best TM improvement: "
            f"{best_tm['sample_id']} improvement={fmt(best_tm['tmscore_improvement'])}, "
            f"Bit={fmt(best_tm['bit_bb_tmscore'])}, FM={fmt(best_tm['fm_bb_tmscore'])}"
        )
        lines.append(
            "worst TM degradation: "
            f"{worst_tm['sample_id']} degradation={fmt(worst_tm['degradation'])}, "
            f"Bit={fmt(worst_tm['bit_bb_tmscore'])}, FM={fmt(worst_tm['fm_bb_tmscore'])}"
        )
        lines.append("")

    lines.extend(
        [
            "[BENCHMARK COMPARISON]",
            "Descriptive comparison only; no cross-benchmark superiority test.",
            (
                "benchmark,N,mean_length,mean_Bit_RMSD,mean_FM_RMSD,"
                "mean_RMSD_improvement,median_RMSD_improvement,RMSD_improved_percent,"
                "mean_Bit_TM,mean_FM_TM,mean_TM_improvement,median_TM_improvement,"
                "TM_improved_percent,mean_r_hat_abs_mean,median_r_hat_abs_mean"
            ),
        ]
    )
    for spec in specs:
        summary = benchmark_summaries[spec.key]
        lines.append(
            ",".join(
                [
                    spec.display_name,
                    str(summary["num_samples"]),
                    fmt(summary["mean_length"]),
                    fmt(summary["mean_bit_rmsd"]),
                    fmt(summary["mean_fm_rmsd"]),
                    fmt(summary["mean_rmsd_improvement"]),
                    fmt(summary["median_rmsd_improvement"]),
                    f"{float(summary['rmsd_improved_percent']):.6f}",
                    fmt(summary["mean_bit_tmscore"]),
                    fmt(summary["mean_fm_tmscore"]),
                    fmt(summary["mean_tmscore_improvement"]),
                    fmt(summary["median_tmscore_improvement"]),
                    f"{float(summary['tmscore_improved_percent']):.6f}",
                    fmt(summary["mean_r_hat_abs_mean"]),
                    fmt(summary["median_r_hat_abs_mean"]),
                ]
            )
        )
    lines.extend(
        [
            "",
            "Output files:",
            str(SAMPLES_CSV),
            str(CORRELATIONS_CSV),
            str(QUARTILES_CSV),
            str(EXTREMES_CSV),
            str(SUMMARY_TXT),
            "",
            "FINAL_SUCCESS_FAILURE_ANALYSIS_PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    specs = (
        BenchmarkSpec(
            "cameo2022",
            "CAMEO",
            args.cameo_paired_csv,
            args.cameo_manifest,
            163,
            6.28888164,
            6.19972204,
            91,
            98,
        ),
        BenchmarkSpec(
            "PDB_date",
            "PDB-DATE",
            args.pdb_date_paired_csv,
            args.pdb_date_manifest,
            442,
            3.11376375,
            3.10744498,
            216,
            211,
        ),
    )
    samples_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    correlations: list[dict[str, Any]] = []
    quartiles: list[dict[str, Any]] = []
    extremes: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for spec in specs:
        rows, residuals_available = load_joined_samples(spec)
        validate_reference(spec, rows)
        benchmark_correlations = build_correlations(rows, residuals_available)
        benchmark_quartiles = build_quartiles(rows)
        benchmark_extremes = build_extremes(rows)
        samples_by_benchmark[spec.key] = rows
        correlations.extend(benchmark_correlations)
        quartiles.extend(benchmark_quartiles)
        extremes.extend(benchmark_extremes)
        summaries[spec.key] = benchmark_summary(rows, residuals_available)

    all_samples = [
        row for spec in specs for row in samples_by_benchmark[spec.key]
    ]
    require(len(all_samples) == 163 + 442, "combined sample count mismatch")
    require(len(correlations) == 18, "expected nine correlations per benchmark")
    require(len(quartiles) == 16, "expected eight quartile rows per benchmark")
    require(len(extremes) == 80, "expected forty extreme rows per benchmark")

    summary_text = build_summary_text(
        specs,
        samples_by_benchmark,
        correlations,
        quartiles,
        extremes,
        summaries,
    )
    atomic_write_csv(SAMPLES_CSV, SAMPLE_COLUMNS, all_samples)
    atomic_write_csv(CORRELATIONS_CSV, CORRELATION_COLUMNS, correlations)
    atomic_write_csv(QUARTILES_CSV, QUARTILE_COLUMNS, quartiles)
    atomic_write_csv(EXTREMES_CSV, EXTREME_COLUMNS, extremes)
    atomic_write_text(SUMMARY_TXT, summary_text)
    print(summary_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

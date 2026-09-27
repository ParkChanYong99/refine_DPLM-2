#!/usr/bin/env python3
"""Post-hoc paired statistics for FM-100 versus Matched DDPM (DDIM-100).

This script reads only finalized per-target metric CSVs.  It performs no model
loading, inference, sampling, decoding, alpha selection, or metric evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "generation-results/dplm2_bit_650m_final/ddpm_100nfe"
INPUTS = (
    ("CAMEO", 163, RESULT_ROOT / "cameo_per_target.csv"),
    ("PDB-date", 442, RESULT_ROOT / "pdb_date_per_target.csv"),
)
OUTPUT_ROOT = RESULT_ROOT / "statistics"
STATISTICS_CSV = OUTPUT_ROOT / "fm_vs_ddim100_paired_statistics.csv"
SUMMARY_TXT = OUTPUT_ROOT / "fm_vs_ddim100_paired_statistics.txt"
DIFFERENCES_CSV = OUTPUT_ROOT / "fm_vs_ddim100_per_target_differences.csv"
METADATA_JSON = OUTPUT_ROOT / "metadata.json"

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
CI_PERCENTILES = (2.5, 97.5)
HOLM_ALPHA = 0.05
SENTINEL_TARGET = "7W2P"
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12

REQUIRED_INPUT_COLUMNS = (
    "benchmark", "target_id", "length", "fm_backbone_rmsd",
    "ddim100_backbone_rmsd", "fm_tm_score", "ddim100_tm_score",
    "fm_bb_advantage", "fm_tm_advantage", "ddim100_nfe", "alpha",
)
DIFFERENCE_COLUMNS = (
    "benchmark", "target_id", "length",
    "fm_backbone_rmsd", "ddim100_backbone_rmsd", "fm_bb_advantage",
    "fm_tm_score", "ddim100_tm_score", "fm_tm_advantage",
    "fm_official_sentinel", "ddim100_official_sentinel",
    "included_in_primary", "included_in_7W2P_excluded_sensitivity",
)
STATISTICS_COLUMNS = (
    "comparison", "benchmark", "metric", "analysis_role", "n",
    "mean_fm_advantage", "median_fm_advantage",
    "bootstrap_ci95_lower", "bootstrap_ci95_upper",
    "bootstrap_replicates", "bootstrap_seed",
    "wilcoxon_statistic", "wilcoxon_raw_p", "wilcoxon_holm_p",
    "wilcoxon_holm_significant_0_05", "wilcoxon_alternative",
    "wilcoxon_zero_method", "wilcoxon_correction", "wilcoxon_method",
    "matched_rank_biserial", "fm_better_count", "ddim100_better_count",
    "tied_count", "positive_difference_means", "holm_family",
    "sensitivity_exclusion", "sensitivity_n",
    "sensitivity_mean_fm_advantage",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    text: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8",
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            if columns is None:
                require(text is not None, "missing text output")
                handle.write(text)
            else:
                require(rows is not None, "missing CSV rows")
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def finite(row: Mapping[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"{context}: invalid {column}") from exc
    require(math.isfinite(value), f"{context}: non-finite {column}")
    return value


def official_sentinel(rmsd: float, tm_score: float) -> bool:
    return rmsd == 100.0 and tm_score == 0.0


def read_benchmark(
    benchmark: str, expected_n: int, path: Path,
) -> list[dict[str, Any]]:
    require(path.is_file(), f"input missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)), f"duplicate columns: {path}")
        require(set(REQUIRED_INPUT_COLUMNS).issubset(columns),
                f"input schema mismatch: {path}")
        source_rows = list(reader)
    require(len(source_rows) == expected_n,
            f"{benchmark}: expected N={expected_n}, got {len(source_rows)}")
    seen: set[str] = set()
    differences: list[dict[str, Any]] = []
    for line, source in enumerate(source_rows, start=2):
        target_id = source["target_id"].strip()
        context = f"{path}:{line}:{target_id}"
        require(source["benchmark"] == benchmark, f"{context}: benchmark mismatch")
        require(target_id and target_id not in seen, f"{context}: duplicate/empty ID")
        seen.add(target_id)
        length = int(source["length"])
        require(length > 0, f"{context}: nonpositive length")
        require(int(source["ddim100_nfe"]) == 100, f"{context}: DDIM NFE mismatch")
        require(float(source["alpha"]) == 0.5, f"{context}: alpha mismatch")
        fm_bb = finite(source, "fm_backbone_rmsd", context)
        ddim_bb = finite(source, "ddim100_backbone_rmsd", context)
        fm_tm = finite(source, "fm_tm_score", context)
        ddim_tm = finite(source, "ddim100_tm_score", context)
        bb_advantage = ddim_bb - fm_bb
        tm_advantage = fm_tm - ddim_tm
        require(math.isclose(
            finite(source, "fm_bb_advantage", context), bb_advantage,
            rel_tol=ALLCLOSE_RTOL, abs_tol=ALLCLOSE_ATOL,
        ), f"{context}: stored BB advantage mismatch")
        require(math.isclose(
            finite(source, "fm_tm_advantage", context), tm_advantage,
            rel_tol=ALLCLOSE_RTOL, abs_tol=ALLCLOSE_ATOL,
        ), f"{context}: stored TM advantage mismatch")
        fm_sentinel = official_sentinel(fm_bb, fm_tm)
        ddim_sentinel = official_sentinel(ddim_bb, ddim_tm)
        differences.append({
            "benchmark": benchmark, "target_id": target_id, "length": length,
            "fm_backbone_rmsd": fm_bb, "ddim100_backbone_rmsd": ddim_bb,
            "fm_bb_advantage": bb_advantage, "fm_tm_score": fm_tm,
            "ddim100_tm_score": ddim_tm, "fm_tm_advantage": tm_advantage,
            "fm_official_sentinel": fm_sentinel,
            "ddim100_official_sentinel": ddim_sentinel,
            "included_in_primary": True,
            "included_in_7W2P_excluded_sensitivity": target_id != SENTINEL_TARGET,
        })
    require(len(seen) == expected_n, f"{benchmark}: unique target count mismatch")
    return differences


def bootstrap_mean_ci(
    differences: np.ndarray, rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(
        0, differences.size,
        size=(BOOTSTRAP_REPLICATES, differences.size), endpoint=False,
    )
    means = differences[indices].mean(axis=1)
    require(bool(np.isfinite(means).all()), "non-finite bootstrap means")
    lower, upper = np.percentile(means, CI_PERCENTILES)
    return float(lower), float(upper)


def matched_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    require(nonzero.size > 0, "rank-biserial undefined for all ties")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    value = (positive - negative) / (positive + negative)
    require(-1.0 <= value <= 1.0, "rank-biserial outside [-1,1]")
    return value


def analyze(
    benchmark: str,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    difference_column: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    differences = np.asarray(
        [float(row[difference_column]) for row in rows], dtype=np.float64
    )
    require(differences.shape == (len(rows),) and
            bool(np.isfinite(differences).all()), "invalid paired differences")
    require(bool(np.any(differences != 0.0)), "all paired differences are tied")
    lower, upper = bootstrap_mean_ci(differences, rng)
    test = wilcoxon(
        differences, zero_method="wilcox", correction=False,
        alternative="two-sided", method="auto",
    )
    statistic, raw_p = float(test.statistic), float(test.pvalue)
    require(math.isfinite(statistic) and math.isfinite(raw_p) and
            0.0 <= raw_p <= 1.0, "invalid Wilcoxon result")
    sensitivity_values = np.asarray(
        [float(row[difference_column]) for row in rows
         if row["target_id"] != SENTINEL_TARGET], dtype=np.float64
    )
    if benchmark == "PDB-date":
        require(differences.size == 442 and sensitivity_values.size == 441,
                "7W2P sensitivity count mismatch")
        sensitivity_exclusion: str = SENTINEL_TARGET
        sensitivity_n: int | str = int(sensitivity_values.size)
        sensitivity_mean: float | str = float(sensitivity_values.mean())
    else:
        require(sensitivity_values.size == differences.size,
                "unexpected CAMEO sentinel ID")
        sensitivity_exclusion = ""
        sensitivity_n = ""
        sensitivity_mean = ""
    return {
        "comparison": "FM-100_vs_Matched_DDPM_DDIM-100",
        "benchmark": benchmark, "metric": metric, "analysis_role": "primary",
        "n": int(differences.size),
        "mean_fm_advantage": float(differences.mean()),
        "median_fm_advantage": float(np.median(differences)),
        "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "wilcoxon_statistic": statistic, "wilcoxon_raw_p": raw_p,
        "wilcoxon_holm_p": "", "wilcoxon_holm_significant_0_05": "",
        "wilcoxon_alternative": "two-sided",
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_correction": False, "wilcoxon_method": "auto",
        "matched_rank_biserial": matched_rank_biserial(differences),
        "fm_better_count": int(np.count_nonzero(differences > 0.0)),
        "ddim100_better_count": int(np.count_nonzero(differences < 0.0)),
        "tied_count": int(np.count_nonzero(differences == 0.0)),
        "positive_difference_means": "FM-100 better than Matched DDPM (DDIM-100)",
        "holm_family": "FM-100_vs_DDIM-100_primary_4",
        "sensitivity_exclusion": sensitivity_exclusion,
        "sensitivity_n": sensitivity_n,
        "sensitivity_mean_fm_advantage": sensitivity_mean,
    }


def holm_bonferroni(raw_p_values: Sequence[float]) -> list[float]:
    values = np.asarray(raw_p_values, dtype=np.float64)
    require(values.shape == (4,), "Holm family must contain four tests")
    require(bool(np.isfinite(values).all()) and
            bool(((0.0 <= values) & (values <= 1.0)).all()),
            "invalid raw p-values")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.minimum(
        1.0, np.maximum.accumulate(values[order] * np.arange(4, 0, -1))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def apply_holm(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 4, "primary Holm family must contain four tests")
    adjusted = holm_bonferroni([float(row["wilcoxon_raw_p"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["wilcoxon_holm_p"] = value
        row["wilcoxon_holm_significant_0_05"] = value < HOLM_ALPHA


def sentinel_audit(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    matches = [
        row for row in rows
        if row["benchmark"] == "PDB-date" and row["target_id"] == SENTINEL_TARGET
    ]
    require(len(matches) == 1, "7W2P sentinel row missing/duplicated")
    row = matches[0]
    require(bool(row["fm_official_sentinel"]) and
            bool(row["ddim100_official_sentinel"]),
            "7W2P expected sentinel did not occur in both methods")
    require(float(row["fm_bb_advantage"]) == 0.0 and
            float(row["fm_tm_advantage"]) == 0.0,
            "7W2P sentinel pair should be tied")
    return row


def text_summary(
    statistics_rows: Sequence[Mapping[str, Any]],
    sentinel: Mapping[str, Any],
) -> str:
    lines = [
        "FM_VS_DDIM100_PAIRED_STATISTICS",
        "Post-hoc analysis only; no inference, sampling, decoding, or metric recomputation.",
        "Positive FM advantage: BB RMSD = DDIM-100 - FM-100; "
        "TM-score = FM-100 - DDIM-100.",
        "Bootstrap: 10,000 intact target-pair resamples, seed=42, percentile 95% CI.",
        "Wilcoxon: two-sided, zero_method=wilcox, correction=False, method=auto.",
        "Holm family: CAMEO BB/TM and PDB-date BB/TM.",
        "",
    ]
    significant: list[str] = []
    for row in statistics_rows:
        label = "{} {}".format(row["benchmark"], row["metric"])
        is_significant = bool(row["wilcoxon_holm_significant_0_05"])
        if is_significant:
            significant.append(label)
            interpretation = "a statistically significant paired difference was detected"
        else:
            interpretation = "no statistically significant paired difference was detected"
        lines.extend([
            "[{}] N={}".format(label, row["n"]),
            "mean FM advantage: {:.12g}".format(float(row["mean_fm_advantage"])),
            "median FM advantage: {:.12g}".format(float(row["median_fm_advantage"])),
            "bootstrap 95% CI: [{:.12g}, {:.12g}]".format(
                float(row["bootstrap_ci95_lower"]),
                float(row["bootstrap_ci95_upper"]),
            ),
            "Wilcoxon statistic: {:.12g}".format(float(row["wilcoxon_statistic"])),
            "Wilcoxon raw p: {:.12g}".format(float(row["wilcoxon_raw_p"])),
            "Holm p: {:.12g}".format(float(row["wilcoxon_holm_p"])),
            "matched rank-biserial: {:.12g}".format(
                float(row["matched_rank_biserial"])
            ),
            "FM/DDIM-100/tie: {}/{}/{}".format(
                row["fm_better_count"], row["ddim100_better_count"],
                row["tied_count"],
            ),
            f"interpretation: {interpretation}.",
            "",
        ])
    pdb_rows = {row["metric"]: row for row in statistics_rows
                if row["benchmark"] == "PDB-date"}
    lines.extend([
        "[7W2P sentinel audit]",
        "FM-100 RMSD/TM: {:.12g} / {:.12g}".format(
            float(sentinel["fm_backbone_rmsd"]), float(sentinel["fm_tm_score"])
        ),
        "DDIM-100 RMSD/TM: {:.12g} / {:.12g}".format(
            float(sentinel["ddim100_backbone_rmsd"]),
            float(sentinel["ddim100_tm_score"]),
        ),
        "sentinel methods: FM-100 and Matched DDPM (DDIM-100)",
        "primary inclusion: retained as an official-metric tied pair",
        "reporting-only sensitivity excluding 7W2P, BB mean FM advantage "
        "(N=441): {:.12g}".format(
            float(pdb_rows["backbone_rmsd"]["sensitivity_mean_fm_advantage"])
        ),
        "reporting-only sensitivity excluding 7W2P, TM mean FM advantage "
        "(N=441): {:.12g}".format(
            float(pdb_rows["tm_score"]["sensitivity_mean_fm_advantage"])
        ),
        "",
        "Significant after Holm: " + (", ".join(significant) if significant else "none"),
        "Non-significance is not evidence of equivalence or same quality.",
        "overall: FM_VS_DDIM100_STATS_PASS",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    input_hashes_before = {str(path): sha256_file(path) for _, _, path in INPUTS}
    all_differences: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for benchmark, expected_n, path in INPUTS:
        differences = read_benchmark(benchmark, expected_n, path)
        all_differences.extend(differences)
        statistics_rows.append(analyze(
            benchmark, differences, "backbone_rmsd", "fm_bb_advantage", rng
        ))
        statistics_rows.append(analyze(
            benchmark, differences, "tm_score", "fm_tm_advantage", rng
        ))
    require(len(all_differences) == 605, "total paired count mismatch")
    apply_holm(statistics_rows)
    sentinel = sentinel_audit(all_differences)
    summary = text_summary(statistics_rows, sentinel)
    atomic_write(
        DIFFERENCES_CSV, columns=DIFFERENCE_COLUMNS, rows=all_differences
    )
    atomic_write(
        STATISTICS_CSV, columns=STATISTICS_COLUMNS, rows=statistics_rows
    )
    atomic_write(SUMMARY_TXT, text=summary)
    input_hashes_after = {str(path): sha256_file(path) for _, _, path in INPUTS}
    require(input_hashes_after == input_hashes_before,
            "read-only input artifact changed during analysis")
    metadata = {
        "status": "FM_VS_DDIM100_STATS_PASS",
        "analysis": "post-hoc paired structural-quality statistics",
        "comparison": "FM-100 vs Matched DDPM (DDIM-100)",
        "benchmarks": {"CAMEO": 163, "PDB-date": 442},
        "metrics": ["backbone_rmsd", "tm_score"],
        "positive_difference": {
            "backbone_rmsd": "DDIM-100 minus FM-100",
            "tm_score": "FM-100 minus DDIM-100",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%", "unit": "intact target pair",
        },
        "wilcoxon": {
            "alternative": "two-sided", "zero_method": "wilcox",
            "correction": False, "method": "auto",
        },
        "holm": {
            "family": "four primary benchmark-metric tests", "alpha": HOLM_ALPHA,
        },
        "sentinel": {
            "target_id": SENTINEL_TARGET,
            "fm_official_sentinel": True,
            "ddim100_official_sentinel": True,
            "included_in_primary": True,
            "reporting_only_sensitivity_excludes_target": True,
        },
        "input_sha256": input_hashes_before,
        "output_sha256": {
            str(DIFFERENCES_CSV): sha256_file(DIFFERENCES_CSV),
            str(STATISTICS_CSV): sha256_file(STATISTICS_CSV),
            str(SUMMARY_TXT): sha256_file(SUMMARY_TXT),
        },
        "model_inference_performed": False,
        "sampling_performed": False,
        "metrics_recomputed": False,
        "alpha_tuning_performed": False,
    }
    atomic_write(METADATA_JSON, text=json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("overall: FM_VS_DDIM100_STATS_FAIL", flush=True)
        raise

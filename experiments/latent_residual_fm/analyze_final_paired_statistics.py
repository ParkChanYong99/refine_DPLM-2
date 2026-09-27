#!/usr/bin/env python3
"""Statistical analysis of the final local paired CAMEO and PDB-date results."""

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
from scipy.stats import binomtest, rankdata, wilcoxon


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12
WILCOXON_ALTERNATIVE = "two-sided"
WILCOXON_ZERO_METHOD = "wilcox"
WILCOXON_METHOD = "auto"
HOLM_ALPHA = 0.05

OUTPUT_DIR = Path("generation-results/dplm2_bit_650m_final/statistics")
OUTPUT_CSV = OUTPUT_DIR / "final_paired_statistics.csv"
OUTPUT_TXT = OUTPUT_DIR / "final_paired_statistics.txt"

REQUIRED_COLUMNS = {
    "sample_id",
    "bit_bb_rmsd",
    "fm_bb_rmsd",
    "rmsd_improvement",
    "bit_bb_tmscore",
    "fm_bb_tmscore",
    "tmscore_improvement",
}
OUTPUT_COLUMNS = (
    "benchmark",
    "metric",
    "num_samples",
    "mean_improvement",
    "median_improvement",
    "std_improvement",
    "q1",
    "q3",
    "min",
    "max",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
    "bootstrap_interpretation",
    "bootstrap_resamples",
    "bootstrap_seed",
    "wilcoxon_statistic",
    "wilcoxon_raw_p",
    "wilcoxon_holm_p",
    "wilcoxon_holm_significant_0_05",
    "wilcoxon_alternative",
    "wilcoxon_zero_method",
    "wilcoxon_method",
    "rank_biserial",
    "improved_count",
    "worsened_count",
    "tie_count",
    "improved_percent",
    "binomial_two_sided_p",
    "binomial_analysis_role",
    "mean_ci_excludes_zero",
    "majority_improved",
    "mean_and_median_same_direction",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    path: Path
    expected_samples: int


@dataclass(frozen=True)
class PairedData:
    sample_ids: tuple[str, ...]
    bit_rmsd: np.ndarray
    fm_rmsd: np.ndarray
    rmsd_improvement: np.ndarray
    bit_tmscore: np.ndarray
    fm_tmscore: np.ndarray
    tmscore_improvement: np.ndarray


BENCHMARKS = (
    BenchmarkSpec(
        key="cameo2022",
        display_name="CAMEO",
        path=Path(
            "generation-results/dplm2_bit_650m_final/cameo2022/comparison/"
            "cameo2022_bit_vs_fm_paired.csv"
        ),
        expected_samples=163,
    ),
    BenchmarkSpec(
        key="PDB_date",
        display_name="PDB-DATE",
        path=Path(
            "generation-results/dplm2_bit_650m_final/PDB_date/comparison/"
            "pdb_date_bit_vs_fm_paired.csv"
        ),
        expected_samples=442,
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameo_paired_csv", type=Path, default=BENCHMARKS[0].path)
    parser.add_argument("--pdb_date_paired_csv", type=Path, default=BENCHMARKS[1].path)
    return parser.parse_args()


def parse_finite_float(value: str, column: str, path: Path, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"invalid {column} at {path}:{row_number}: {value!r}"
        ) from exc
    require(math.isfinite(parsed), f"non-finite {column} at {path}:{row_number}")
    return parsed


def read_paired_csv(spec: BenchmarkSpec) -> PairedData:
    require(spec.path.is_file(), f"paired CSV does not exist: {spec.path}")
    with spec.path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        require(not missing, f"{spec.path} missing columns: {sorted(missing)}")
        rows = list(reader)

    require(
        len(rows) == spec.expected_samples,
        f"{spec.display_name} expected N={spec.expected_samples}, got {len(rows)}",
    )
    sample_ids: list[str] = []
    numeric: dict[str, list[float]] = {
        column: [] for column in REQUIRED_COLUMNS if column != "sample_id"
    }
    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        require(sample_id, f"empty sample_id at {spec.path}:{row_number}")
        sample_ids.append(sample_id)
        for column in numeric:
            numeric[column].append(
                parse_finite_float(row[column], column, spec.path, row_number)
            )

    duplicate_ids = sorted(
        sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
    )
    require(not duplicate_ids, f"{spec.display_name} duplicate sample IDs: {duplicate_ids}")
    require(len(set(sample_ids)) == spec.expected_samples, "unique sample count mismatch")

    arrays = {
        column: np.asarray(values, dtype=np.float64)
        for column, values in numeric.items()
    }
    for column, values in arrays.items():
        require(values.shape == (spec.expected_samples,), f"{column} shape mismatch")
        require(bool(np.isfinite(values).all()), f"{column} contains NaN/Inf")

    recomputed_rmsd = arrays["bit_bb_rmsd"] - arrays["fm_bb_rmsd"]
    recomputed_tm = arrays["fm_bb_tmscore"] - arrays["bit_bb_tmscore"]
    require(
        bool(
            np.allclose(
                arrays["rmsd_improvement"],
                recomputed_rmsd,
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
            )
        ),
        f"{spec.display_name} stored RMSD improvement does not match Bit - FM",
    )
    require(
        bool(
            np.allclose(
                arrays["tmscore_improvement"],
                recomputed_tm,
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
            )
        ),
        f"{spec.display_name} stored TM improvement does not match FM - Bit",
    )

    # Downstream analysis uses the improvements recomputed from each intact pair.
    return PairedData(
        sample_ids=tuple(sample_ids),
        bit_rmsd=arrays["bit_bb_rmsd"],
        fm_rmsd=arrays["fm_bb_rmsd"],
        rmsd_improvement=recomputed_rmsd,
        bit_tmscore=arrays["bit_bb_tmscore"],
        fm_tmscore=arrays["fm_bb_tmscore"],
        tmscore_improvement=recomputed_tm,
    )


def bootstrap_mean_ci(
    improvements: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Paired bootstrap: resample sample indices, never Bit/FM observations separately."""
    num_samples = improvements.size
    indices = rng.integers(
        0,
        num_samples,
        size=(BOOTSTRAP_RESAMPLES, num_samples),
        endpoint=False,
    )
    bootstrap_means = improvements[indices].mean(axis=1)
    require(bool(np.isfinite(bootstrap_means).all()), "bootstrap produced NaN/Inf")
    lower, upper = np.percentile(bootstrap_means, CI_PERCENTILES)
    return float(lower), float(upper)


def bootstrap_interpretation(lower: float, upper: float) -> str:
    if lower > 0.0:
        return "mean improvement supported"
    if upper < 0.0:
        return "degradation supported"
    return "mean improvement not clearly supported"


def matched_pairs_rank_biserial(improvements: np.ndarray) -> float:
    """Return (sum positive ranks - sum negative ranks) / sum nonzero ranks.

    Absolute nonzero paired differences receive average ranks for ties, matching
    Wilcoxon's discarded-zero convention. Positive values point toward FM
    improvement because both input improvement definitions use positive=better.
    """
    nonzero = improvements[improvements != 0.0]
    require(nonzero.size > 0, "rank-biserial is undefined when every pair is tied")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0.0].sum())
    negative_rank_sum = float(ranks[nonzero < 0.0].sum())
    denominator = positive_rank_sum + negative_rank_sum
    require(denominator > 0.0, "rank-biserial denominator is zero")
    result = (positive_rank_sum - negative_rank_sum) / denominator
    require(-1.0 <= result <= 1.0, f"rank-biserial out of range: {result}")
    return result


def direction(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def analyze_metric(
    spec: BenchmarkSpec,
    metric: str,
    improvements: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    require(improvements.shape == (spec.expected_samples,), "improvement shape mismatch")
    require(bool(np.isfinite(improvements).all()), "improvements contain NaN/Inf")
    lower, upper = bootstrap_mean_ci(improvements, rng)
    wilcoxon_result = wilcoxon(
        improvements,
        zero_method=WILCOXON_ZERO_METHOD,
        correction=False,
        alternative=WILCOXON_ALTERNATIVE,
        method=WILCOXON_METHOD,
    )
    wilcoxon_statistic = float(wilcoxon_result.statistic)
    wilcoxon_raw_p = float(wilcoxon_result.pvalue)
    require(math.isfinite(wilcoxon_statistic), "non-finite Wilcoxon statistic")
    require(math.isfinite(wilcoxon_raw_p), "non-finite Wilcoxon p-value")

    improved_count = int(np.count_nonzero(improvements > 0.0))
    worsened_count = int(np.count_nonzero(improvements < 0.0))
    tie_count = int(np.count_nonzero(improvements == 0.0))
    require(
        improved_count + worsened_count + tie_count == spec.expected_samples,
        "improvement direction counts do not sum to N",
    )
    non_tie_count = improved_count + worsened_count
    binomial_p = (
        float(
            binomtest(
                improved_count,
                n=non_tie_count,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if non_tie_count
        else 1.0
    )

    mean_improvement = float(np.mean(improvements))
    median_improvement = float(np.median(improvements))
    q1, q3 = np.percentile(improvements, (25.0, 75.0))
    return {
        "benchmark": spec.key,
        "metric": metric,
        "num_samples": spec.expected_samples,
        "mean_improvement": mean_improvement,
        "median_improvement": median_improvement,
        # Sample standard deviation (ddof=1) describes observed pair variation.
        "std_improvement": float(np.std(improvements, ddof=1)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(improvements)),
        "max": float(np.max(improvements)),
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "bootstrap_interpretation": bootstrap_interpretation(lower, upper),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_raw_p": wilcoxon_raw_p,
        "wilcoxon_holm_p": "",
        "wilcoxon_holm_significant_0_05": "",
        "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
        "wilcoxon_zero_method": WILCOXON_ZERO_METHOD,
        "wilcoxon_method": WILCOXON_METHOD,
        "rank_biserial": matched_pairs_rank_biserial(improvements),
        "improved_count": improved_count,
        "worsened_count": worsened_count,
        "tie_count": tie_count,
        "improved_percent": improved_count / spec.expected_samples * 100.0,
        "binomial_two_sided_p": binomial_p,
        "binomial_analysis_role": "secondary analysis; ties excluded; unadjusted",
        "mean_ci_excludes_zero": lower > 0.0 or upper < 0.0,
        "majority_improved": improved_count > spec.expected_samples / 2.0,
        "mean_and_median_same_direction": (
            direction(mean_improvement) == direction(median_improvement)
        ),
    }


def holm_bonferroni(raw_p_values: Sequence[float]) -> list[float]:
    """Return standard step-down Holm-adjusted p-values in original order."""
    p_values = np.asarray(raw_p_values, dtype=np.float64)
    require(p_values.shape == (4,), "Holm correction requires exactly four tests")
    require(bool(np.isfinite(p_values).all()), "Holm input contains NaN/Inf")
    require(bool(((0.0 <= p_values) & (p_values <= 1.0)).all()), "invalid p-value")
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    multipliers = np.arange(p_values.size, 0, -1, dtype=np.float64)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * multipliers))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
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


def format_float(value: Any) -> str:
    return f"{float(value):.12g}"


def build_text_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["[FINAL PAIRED STATISTICS]", ""]
    display_names = {spec.key: spec.display_name for spec in BENCHMARKS}
    metric_display_names = {"rmsd": "RMSD", "tmscore": "TM-SCORE"}
    for row in rows:
        title = (
            f"{display_names[str(row['benchmark'])]} "
            f"{metric_display_names[str(row['metric'])]}"
        )
        lines.extend(
            [
                f"[{title}]",
                f"N: {row['num_samples']}",
                f"mean improvement: {format_float(row['mean_improvement'])}",
                f"median improvement: {format_float(row['median_improvement'])}",
                "bootstrap 95% CI: "
                f"[{format_float(row['bootstrap_ci95_lower'])}, "
                f"{format_float(row['bootstrap_ci95_upper'])}] "
                f"({row['bootstrap_interpretation']})",
                "Wilcoxon settings: alternative=two-sided, "
                "zero_method=wilcox, correction=False, method=auto",
                f"Wilcoxon statistic: {format_float(row['wilcoxon_statistic'])}",
                f"Wilcoxon raw p: {format_float(row['wilcoxon_raw_p'])}",
                "Wilcoxon Holm-adjusted p: "
                f"{format_float(row['wilcoxon_holm_p'])}",
                f"rank-biserial: {format_float(row['rank_biserial'])}",
                "improved/worsened/tie: "
                f"{row['improved_count']}/{row['worsened_count']}/{row['tie_count']} "
                f"(improved {float(row['improved_percent']):.6f}%)",
                "exact binomial p: "
                f"{format_float(row['binomial_two_sided_p'])} "
                "(secondary, two-sided, ties excluded, unadjusted)",
                "",
            ]
        )

    lines.extend(["[INTERPRETATION FLAGS]", ""])
    for row in rows:
        title = (
            f"{display_names[str(row['benchmark'])]} "
            f"{metric_display_names[str(row['metric'])]}"
        )
        lines.extend(
            [
                f"{title}:",
                "mean CI excludes zero: "
                f"{'YES' if row['mean_ci_excludes_zero'] else 'NO'}",
                "Holm-adjusted Wilcoxon p < 0.05: "
                f"{'YES' if row['wilcoxon_holm_significant_0_05'] else 'NO'}",
                f"majority improved: {'YES' if row['majority_improved'] else 'NO'}",
                "mean and median same direction: "
                f"{'YES' if row['mean_and_median_same_direction'] else 'NO'}",
            ]
        )
        if direction(float(row["mean_improvement"])) != direction(
            float(row["median_improvement"])
        ):
            lines.append(f"MEAN_MEDIAN_DIRECTION_MISMATCH: {title}")
        lines.append("")

    lines.extend(
        [
            f"statistics CSV: {OUTPUT_CSV}",
            f"statistics text: {OUTPUT_TXT}",
            "",
            "FINAL_PAIRED_STATISTICS_PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    specs = (
        BenchmarkSpec("cameo2022", "CAMEO", args.cameo_paired_csv, 163),
        BenchmarkSpec("PDB_date", "PDB-DATE", args.pdb_date_paired_csv, 442),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        data = read_paired_csv(spec)
        rows.append(analyze_metric(spec, "rmsd", data.rmsd_improvement, rng))
        rows.append(analyze_metric(spec, "tmscore", data.tmscore_improvement, rng))

    require(len(rows) == 4, "expected exactly four benchmark-metric tests")
    adjusted_p_values = holm_bonferroni(
        [float(row["wilcoxon_raw_p"]) for row in rows]
    )
    for row, adjusted_p in zip(rows, adjusted_p_values):
        row["wilcoxon_holm_p"] = adjusted_p
        row["wilcoxon_holm_significant_0_05"] = adjusted_p < HOLM_ALPHA

    summary_text = build_text_summary(rows)
    atomic_write_csv(OUTPUT_CSV, rows)
    atomic_write_text(OUTPUT_TXT, summary_text)
    print(summary_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

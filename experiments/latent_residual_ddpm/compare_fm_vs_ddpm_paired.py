#!/usr/bin/env python3
"""Post-hoc paired comparison of finalized FM, matched DDPM, and Bit results.

No generation, model loading, or metric recomputation occurs here. Statistical
conventions match analyze_final_paired_statistics.py: paired percentile bootstrap,
two-sided Wilcoxon with discarded zeros, matched rank-biserial, and step-down Holm.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "generation-results/dplm2_bit_650m_final"
OUTPUT_DIR = RESULTS / "ddpm_matched/comparison"
PAIRED_CSV = OUTPUT_DIR / "fm_vs_ddpm_paired.csv"
FM_STATISTICS_CSV = OUTPUT_DIR / "fm_vs_ddpm_paired_statistics.csv"
BIT_STATISTICS_CSV = OUTPUT_DIR / "bit_vs_ddpm_paired_statistics.csv"
STATISTICS_TXT = OUTPUT_DIR / "fm_vs_ddpm_paired_statistics.txt"

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000
CI_PERCENTILES = (2.5, 97.5)
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12
HOLM_ALPHA = 0.05

FM_REQUIRED = (
    "sample_id", "length", "bit_bb_rmsd", "fm_bb_rmsd",
    "rmsd_improvement", "bit_bb_tmscore", "fm_bb_tmscore",
    "tmscore_improvement",
)
DDPM_REQUIRED = (
    "benchmark", "target_id", "length", "bit_backbone_rmsd",
    "ddpm_backbone_rmsd", "bit_tm_score", "ddpm_tm_score",
)
PAIRED_COLUMNS = (
    "benchmark", "target_id", "length", "bit_backbone_rmsd",
    "fm_backbone_rmsd", "ddpm_backbone_rmsd", "bit_tm_score",
    "fm_tm_score", "ddpm_tm_score", "fm_bb_advantage",
    "fm_tm_advantage", "ddpm_bb_advantage_over_bit",
    "ddpm_tm_advantage_over_bit",
)
STATISTICS_COLUMNS = (
    "comparison", "benchmark", "metric", "analysis_role", "n",
    "mean_paired_difference", "median_paired_difference",
    "bootstrap_ci95_lower", "bootstrap_ci95_upper", "bootstrap_resamples",
    "bootstrap_seed", "wilcoxon_statistic", "wilcoxon_raw_p",
    "wilcoxon_holm_p", "wilcoxon_holm_significant_0_05",
    "wilcoxon_alternative", "wilcoxon_zero_method", "wilcoxon_method",
    "rank_biserial", "first_method_better_count",
    "second_method_better_count", "tied_count", "positive_difference_means",
    "holm_family",
)


@dataclass(frozen=True)
class Benchmark:
    name: str
    expected_n: int
    fm_path: Path
    ddpm_path: Path


BENCHMARKS = (
    Benchmark(
        "CAMEO", 163,
        RESULTS / "cameo2022/comparison/cameo2022_bit_vs_fm_paired.csv",
        RESULTS / "ddpm_matched/cameo_per_target.csv",
    ),
    Benchmark(
        "PDB-date", 442,
        RESULTS / "PDB_date/comparison/pdb_date_bit_vs_fm_paired.csv",
        RESULTS / "ddpm_matched/pdb_date_per_target.csv",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_indexed_csv(path: Path, id_column: str, required: Sequence[str],
                     expected_n: int) -> dict[str, dict[str, str]]:
    require(path.is_file(), f"missing finalized artifact: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)), f"duplicate CSV columns: {path}")
        missing = set(required) - set(columns)
        require(not missing, f"missing columns in {path}: {sorted(missing)}")
        indexed: dict[str, dict[str, str]] = {}
        for line, row in enumerate(reader, start=2):
            require(None not in row and all(value is not None for value in row.values()),
                    f"malformed CSV row: {path}:{line}")
            target_id = row[id_column].strip()
            require(target_id, f"empty target ID: {path}:{line}")
            require(target_id not in indexed, f"duplicate target ID {target_id}: {path}:{line}")
            indexed[target_id] = row
    require(len(indexed) == expected_n,
            f"{path}: expected N={expected_n}, got {len(indexed)}")
    return indexed


def finite_float(row: Mapping[str, str], column: str, path: Path,
                 target_id: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid {column} for {target_id} in {path}") from exc
    require(math.isfinite(value), f"non-finite {column} for {target_id} in {path}")
    return value


def positive_length(row: Mapping[str, str], path: Path, target_id: str) -> int:
    try:
        length = int(row["length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid length for {target_id} in {path}") from exc
    require(length > 0, f"nonpositive length for {target_id} in {path}")
    return length


def close(actual: float, expected: float, context: str) -> None:
    require(math.isclose(actual, expected, rel_tol=ALLCLOSE_RTOL,
                         abs_tol=ALLCLOSE_ATOL),
            f"{context}: {actual} != {expected}")


def join_benchmark(spec: Benchmark) -> list[dict[str, Any]]:
    fm = read_indexed_csv(spec.fm_path, "sample_id", FM_REQUIRED, spec.expected_n)
    ddpm = read_indexed_csv(spec.ddpm_path, "target_id", DDPM_REQUIRED, spec.expected_n)
    fm_ids, ddpm_ids = set(fm), set(ddpm)
    require(fm_ids == ddpm_ids,
            f"{spec.name} target mismatch: FM-only={sorted(fm_ids - ddpm_ids)}, "
            f"DDPM-only={sorted(ddpm_ids - fm_ids)}")
    paired: list[dict[str, Any]] = []
    for target_id in sorted(fm_ids):
        f, d = fm[target_id], ddpm[target_id]
        f_length = positive_length(f, spec.fm_path, target_id)
        d_length = positive_length(d, spec.ddpm_path, target_id)
        require(f_length == d_length,
                f"{spec.name} length mismatch for {target_id}: FM={f_length}, DDPM={d_length}")
        expected_benchmark = "cameo2022" if spec.name == "CAMEO" else "PDB_date"
        require(d["benchmark"] == expected_benchmark,
                f"{spec.name} benchmark label mismatch for {target_id}: {d['benchmark']}")

        bit_bb = finite_float(f, "bit_bb_rmsd", spec.fm_path, target_id)
        fm_bb = finite_float(f, "fm_bb_rmsd", spec.fm_path, target_id)
        bit_tm = finite_float(f, "bit_bb_tmscore", spec.fm_path, target_id)
        fm_tm = finite_float(f, "fm_bb_tmscore", spec.fm_path, target_id)
        ddpm_bit_bb = finite_float(d, "bit_backbone_rmsd", spec.ddpm_path, target_id)
        ddpm_bb = finite_float(d, "ddpm_backbone_rmsd", spec.ddpm_path, target_id)
        ddpm_bit_tm = finite_float(d, "bit_tm_score", spec.ddpm_path, target_id)
        ddpm_tm = finite_float(d, "ddpm_tm_score", spec.ddpm_path, target_id)

        # The finalized Bit baseline must be identical across the two artifacts.
        close(ddpm_bit_bb, bit_bb, f"{spec.name} {target_id} Bit BB baseline mismatch")
        close(ddpm_bit_tm, bit_tm, f"{spec.name} {target_id} Bit TM baseline mismatch")
        close(finite_float(f, "rmsd_improvement", spec.fm_path, target_id),
              bit_bb - fm_bb, f"{spec.name} {target_id} FM-Bit BB improvement mismatch")
        close(finite_float(f, "tmscore_improvement", spec.fm_path, target_id),
              fm_tm - bit_tm, f"{spec.name} {target_id} FM-Bit TM improvement mismatch")

        paired.append({
            "benchmark": spec.name, "target_id": target_id, "length": f_length,
            "bit_backbone_rmsd": bit_bb, "fm_backbone_rmsd": fm_bb,
            "ddpm_backbone_rmsd": ddpm_bb, "bit_tm_score": bit_tm,
            "fm_tm_score": fm_tm, "ddpm_tm_score": ddpm_tm,
            "fm_bb_advantage": ddpm_bb - fm_bb,
            "fm_tm_advantage": fm_tm - ddpm_tm,
            "ddpm_bb_advantage_over_bit": bit_bb - ddpm_bb,
            "ddpm_tm_advantage_over_bit": ddpm_tm - bit_tm,
        })
    require(len(paired) == spec.expected_n, f"{spec.name} joined N mismatch")
    return paired


def bootstrap_mean_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Resample intact target-level paired differences, not method observations."""
    indices = rng.integers(0, differences.size,
                           size=(BOOTSTRAP_RESAMPLES, differences.size), endpoint=False)
    means = differences[indices].mean(axis=1)
    require(bool(np.isfinite(means).all()), "non-finite bootstrap mean")
    lower, upper = np.percentile(means, CI_PERCENTILES)
    return float(lower), float(upper)


def matched_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    require(nonzero.size > 0, "rank-biserial undefined for all-tied pairs")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    result = (positive - negative) / (positive + negative)
    require(-1.0 <= result <= 1.0, "rank-biserial out of range")
    return result


def analyze(spec: Benchmark, paired: Sequence[Mapping[str, Any]],
            comparison: str, metric: str, difference_column: str,
            rng: np.random.Generator) -> dict[str, Any]:
    differences = np.asarray([row[difference_column] for row in paired], dtype=np.float64)
    require(differences.shape == (spec.expected_n,), "paired difference shape mismatch")
    require(bool(np.isfinite(differences).all()), "non-finite paired difference")
    require(bool(np.any(differences != 0.0)), "all paired differences are zero")
    lower, upper = bootstrap_mean_ci(differences, rng)
    test = wilcoxon(differences, zero_method="wilcox", correction=False,
                    alternative="two-sided", method="auto")
    statistic, raw_p = float(test.statistic), float(test.pvalue)
    require(math.isfinite(statistic) and math.isfinite(raw_p) and 0 <= raw_p <= 1,
            "invalid Wilcoxon result")
    first_better = int(np.count_nonzero(differences > 0.0))
    second_better = int(np.count_nonzero(differences < 0.0))
    tied = int(np.count_nonzero(differences == 0.0))
    require(first_better + second_better + tied == spec.expected_n,
            "direction counts do not sum to N")
    first, second = ("FM", "DDPM") if comparison == "FM_vs_DDPM" else ("DDPM", "Bit")
    return {
        "comparison": comparison, "benchmark": spec.name, "metric": metric,
        "analysis_role": "primary" if comparison == "FM_vs_DDPM" else "secondary",
        "n": spec.expected_n, "mean_paired_difference": float(differences.mean()),
        "median_paired_difference": float(np.median(differences)),
        "bootstrap_ci95_lower": lower, "bootstrap_ci95_upper": upper,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "wilcoxon_statistic": statistic, "wilcoxon_raw_p": raw_p,
        "wilcoxon_holm_p": "", "wilcoxon_holm_significant_0_05": "",
        "wilcoxon_alternative": "two-sided", "wilcoxon_zero_method": "wilcox",
        "wilcoxon_method": "auto", "rank_biserial": matched_rank_biserial(differences),
        "first_method_better_count": first_better,
        "second_method_better_count": second_better, "tied_count": tied,
        "positive_difference_means": f"{first} better than {second}",
        "holm_family": ("FM_vs_DDPM_primary_4" if comparison == "FM_vs_DDPM"
                        else "Bit_vs_DDPM_secondary_4"),
    }


def holm_bonferroni(raw_p_values: Sequence[float]) -> list[float]:
    p_values = np.asarray(raw_p_values, dtype=np.float64)
    require(p_values.shape == (4,), "Holm family must contain exactly four tests")
    require(bool(np.isfinite(p_values).all()) and bool(((0 <= p_values) & (p_values <= 1)).all()),
            "invalid Holm p-values")
    order = np.argsort(p_values, kind="stable")
    adjusted_sorted = np.minimum(
        1.0, np.maximum.accumulate(p_values[order] * np.arange(4, 0, -1))
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def apply_holm(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == 4, "expected four benchmark-metric rows")
    for row, adjusted in zip(rows, holm_bonferroni([row["wilcoxon_raw_p"] for row in rows])):
        row["wilcoxon_holm_p"] = adjusted
        row["wilcoxon_holm_significant_0_05"] = adjusted < HOLM_ALPHA


def atomic_write(path: Path, columns: Sequence[str] | None,
                 rows: Sequence[Mapping[str, Any]] | None, text: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False,
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


def text_summary(fm_rows: Sequence[Mapping[str, Any]],
                 bit_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "POST-HOC FINAL PAIRED STATISTICAL COMPARISON",
        "No inference, training, or metric recomputation.",
        "Positive FM advantage: BB = DDPM - FM; TM = FM - DDPM.",
        "Positive DDPM advantage over Bit: BB = Bit - DDPM; TM = DDPM - Bit.",
        "Bootstrap: 10,000 target-pair resamples, seed=42, percentile 95% CI.",
        "Wilcoxon: two-sided, zero_method=wilcox, correction=False, method=auto.",
        "Rank-biserial: (positive rank sum - negative rank sum) / nonzero rank sum.",
        "Holm: four FM-vs-DDPM primary tests; separate four-test Bit-vs-DDPM secondary family.",
        "",
    ]
    for title, rows in (("FM vs matched DDPM (primary)", fm_rows),
                        ("Bit vs matched DDPM (secondary)", bit_rows)):
        lines.append(f"[{title}]")
        for row in rows:
            lines.extend([
                f"{row['benchmark']} {row['metric']}: N={row['n']}, "
                f"mean={row['mean_paired_difference']:.12g}, "
                f"median={row['median_paired_difference']:.12g}, "
                f"bootstrap 95% CI=[{row['bootstrap_ci95_lower']:.12g}, "
                f"{row['bootstrap_ci95_upper']:.12g}]",
                f"  Wilcoxon statistic={row['wilcoxon_statistic']:.12g}, "
                f"raw p={row['wilcoxon_raw_p']:.12g}, "
                f"Holm p={row['wilcoxon_holm_p']:.12g}, "
                f"rank-biserial={row['rank_biserial']:.12g}",
                f"  {row['positive_difference_means']}: "
                f"first/second/tied={row['first_method_better_count']}/"
                f"{row['second_method_better_count']}/{row['tied_count']}",
            ])
        lines.append("")
    lines.extend([
        f"FM paired CSVs: {BENCHMARKS[0].fm_path}; {BENCHMARKS[1].fm_path}",
        f"DDPM per-target CSVs: {BENCHMARKS[0].ddpm_path}; {BENCHMARKS[1].ddpm_path}",
        f"Paired output: {PAIRED_CSV}", f"FM statistics: {FM_STATISTICS_CSV}",
        f"Bit statistics: {BIT_STATISTICS_CSV}", f"Text summary: {STATISTICS_TXT}",
        "FM_VS_DDPM_PAIRED_STATISTICS_PASS", "",
    ])
    return "\n".join(lines)


def main() -> int:
    paired_all: list[dict[str, Any]] = []
    fm_rows: list[dict[str, Any]] = []
    bit_rows: list[dict[str, Any]] = []
    fm_rng = np.random.default_rng(BOOTSTRAP_SEED)
    bit_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for spec in BENCHMARKS:
        paired = join_benchmark(spec)
        paired_all.extend(paired)
        fm_rows.append(analyze(spec, paired, "FM_vs_DDPM", "backbone_rmsd",
                               "fm_bb_advantage", fm_rng))
        fm_rows.append(analyze(spec, paired, "FM_vs_DDPM", "tm_score",
                               "fm_tm_advantage", fm_rng))
        bit_rows.append(analyze(spec, paired, "Bit_vs_DDPM", "backbone_rmsd",
                                "ddpm_bb_advantage_over_bit", bit_rng))
        bit_rows.append(analyze(spec, paired, "Bit_vs_DDPM", "tm_score",
                                "ddpm_tm_advantage_over_bit", bit_rng))
    require(len(paired_all) == 605, "total joined target count mismatch")
    apply_holm(fm_rows)
    apply_holm(bit_rows)  # Distinct secondary family; never mixed with FM primary tests.
    summary = text_summary(fm_rows, bit_rows)
    atomic_write(PAIRED_CSV, PAIRED_COLUMNS, paired_all)
    atomic_write(FM_STATISTICS_CSV, STATISTICS_COLUMNS, fm_rows)
    atomic_write(BIT_STATISTICS_CSV, STATISTICS_COLUMNS, bit_rows)
    atomic_write(STATISTICS_TXT, None, None, summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

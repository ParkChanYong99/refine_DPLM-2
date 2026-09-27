#!/usr/bin/env python3
"""Prepare manuscript tables, captions, and Results text from finalized outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


FINAL_RESULTS_DIR = Path("generation-results/dplm2_bit_650m_final/final_results")
DEFAULT_OUTPUT_DIR = Path("generation-results/dplm2_bit_650m_final/manuscript")

MAIN_PERFORMANCE_CSV = FINAL_RESULTS_DIR / "main_performance_table.csv"
PAIRED_STATISTICS_CSV = FINAL_RESULTS_DIR / "paired_statistical_table.csv"
LOCAL_SUMMARY_CSV = FINAL_RESULTS_DIR / "local_bit_vs_fm_summary.csv"
BASELINE_QUARTILES_CSV = FINAL_RESULTS_DIR / "baseline_difficulty_quartiles.csv"
METADATA_TXT = FINAL_RESULTS_DIR / "final_results_metadata.txt"
FIGURE_PDFS = (
    FINAL_RESULTS_DIR / "figures/figure1_local_bit_vs_fm_performance.pdf",
    FINAL_RESULTS_DIR / "figures/figure2_paired_improvement_ci.pdf",
    FINAL_RESULTS_DIR / "figures/figure3_baseline_difficulty.pdf",
)

BENCHMARKS = ("cameo2022", "PDB_date")
BENCHMARK_LABELS = {"cameo2022": "CAMEO", "PDB_date": "PDB-date"}
METHODS = ("Paper Bit", "Paper Bit + RESDIFF", "Local Bit", "Local Bit + Latent FM")
METRICS = ("rmsd", "tmscore")
EXPECTED_N = {"cameo2022": 163, "PDB_date": 442}
EXPECTED_SOURCE = {
    "Paper Bit": "PAPER_REFERENCE",
    "Paper Bit + RESDIFF": "PAPER_REFERENCE",
    "Local Bit": "LOCAL_REPRODUCTION",
    "Local Bit + Latent FM": "LOCAL_REPRODUCTION",
}
EXPECTED_PERFORMANCE = {
    ("cameo2022", "Paper Bit"): (6.4028, 0.8380),
    ("cameo2022", "Paper Bit + RESDIFF"): (6.1781, 0.8428),
    ("cameo2022", "Local Bit"): (6.28888164, 0.841421163),
    ("cameo2022", "Local Bit + Latent FM"): (6.19972204, 0.844954294),
    ("PDB_date", "Paper Bit"): (3.2213, 0.9043),
    ("PDB_date", "Paper Bit + RESDIFF"): (3.0168, 0.9076),
    ("PDB_date", "Local Bit"): (3.11376375, 0.91379618),
    ("PDB_date", "Local Bit + Latent FM"): (3.10744498, 0.91385542),
}
EXPECTED_STATISTICS = {
    ("cameo2022", "rmsd"): (0.0891596035, -0.0704047244, 0.2473303489, 0.649260251577),
    ("cameo2022", "tmscore"): (0.00353313079, -0.0000777252, 0.00703693793, 0.00879188717),
    ("PDB_date", "rmsd"): (0.00631876438, -0.0249448853, 0.0364133559, 1.0),
    ("PDB_date", "tmscore"): (0.0000592400217, -0.00115664778, 0.00122829674, 1.0),
}

MAIN_COLUMNS = {
    "benchmark",
    "method",
    "result_type",
    "num_samples",
    "bb_rmsd",
    "bb_tmscore",
    "rmsd_delta_vs_local_bit",
    "tmscore_delta_vs_local_bit",
    "rmsd_relative_change_percent",
}
STATISTICS_COLUMNS = {
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
}
LOCAL_COLUMNS = {
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
}
QUARTILE_COLUMNS = {
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
}

REFERENCE_RTOL = 1e-9
REFERENCE_ATOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main_performance_csv", type=Path, default=MAIN_PERFORMANCE_CSV)
    parser.add_argument("--paired_statistics_csv", type=Path, default=PAIRED_STATISTICS_CSV)
    parser.add_argument("--local_summary_csv", type=Path, default=LOCAL_SUMMARY_CSV)
    parser.add_argument("--baseline_quartiles_csv", type=Path, default=BASELINE_QUARTILES_CSV)
    parser.add_argument("--metadata_txt", type=Path, default=METADATA_TXT)
    parser.add_argument("--figure1_pdf", type=Path, default=FIGURE_PDFS[0])
    parser.add_argument("--figure2_pdf", type=Path, default=FIGURE_PDFS[1])
    parser.add_argument("--figure3_pdf", type=Path, default=FIGURE_PDFS[2])
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        require(required_columns <= columns, f"{path}: missing columns {sorted(required_columns - columns)}")
        return list(reader)


def finite(row: Mapping[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError(f"{path}: invalid {column}={row.get(column)!r}") from error
    require(math.isfinite(value), f"{path}: non-finite {column}")
    return value


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=REFERENCE_RTOL, abs_tol=REFERENCE_ATOL),
        f"reference mismatch for {label}: expected {expected}, found {actual}",
    )


def close_rounded(actual: float, expected: float, decimals: int, label: str) -> None:
    tolerance = 0.5 * 10.0 ** (-decimals)
    require(abs(actual - expected) <= tolerance, f"rounded reference mismatch for {label}")


def index_rows(
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
    expected_keys: set[tuple[str, ...]],
    path: Path,
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        require(key not in indexed, f"{path}: duplicate row key {key}")
        indexed[key] = row
    require(len(rows) == len(expected_keys), f"{path}: expected {len(expected_keys)} rows, found {len(rows)}")
    require(set(indexed) == expected_keys, f"{path}: unexpected row keys {sorted(set(indexed) ^ expected_keys)}")
    return indexed


def normalized_metadata(path: Path) -> str:
    require(path.is_file(), f"missing metadata: {path}")
    text = path.read_text(encoding="utf-8")
    return " ".join(text.split())


def validate_metadata(metadata: str) -> None:
    required_fragments = (
        "optimizer_step: 100000",
        "FM alpha: 0.50",
        "FM Euler steps: 100",
        "CAMEO evaluation N: 163",
        "PDB-date evaluation N: 442",
        "alpha selection: internal validation only not selected using CAMEO/PDB-date",
        "statistics: paired bootstrap 10000 seed 42 two-sided Wilcoxon Holm correction across 4 tests",
        "post-hoc analysis: descriptive/exploratory only",
        "descriptive only; not used for model or hyperparameter selection",
    )
    for fragment in required_fragments:
        require(fragment in metadata, f"metadata is missing finalized statement: {fragment}")


def validate_pdfs(paths: Sequence[Path]) -> None:
    for path in paths:
        require(path.is_file(), f"missing finalized figure PDF: {path}")
        with path.open("rb") as handle:
            require(handle.read(5) == b"%PDF-", f"invalid PDF header: {path}")


def validate_inputs(args: argparse.Namespace) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
    dict[str, dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
]:
    main_rows = read_csv(args.main_performance_csv, MAIN_COLUMNS)
    statistics_rows = read_csv(args.paired_statistics_csv, STATISTICS_COLUMNS)
    local_rows = read_csv(args.local_summary_csv, LOCAL_COLUMNS)
    quartile_rows = read_csv(args.baseline_quartiles_csv, QUARTILE_COLUMNS)

    main = index_rows(
        main_rows,
        ("benchmark", "method"),
        {(benchmark, method) for benchmark in BENCHMARKS for method in METHODS},
        args.main_performance_csv,
    )
    statistics = index_rows(
        statistics_rows,
        ("benchmark", "metric"),
        {(benchmark, metric) for benchmark in BENCHMARKS for metric in METRICS},
        args.paired_statistics_csv,
    )
    local_by_tuple = index_rows(
        local_rows,
        ("benchmark",),
        {(benchmark,) for benchmark in BENCHMARKS},
        args.local_summary_csv,
    )
    local = {key[0]: row for key, row in local_by_tuple.items()}
    quartiles = index_rows(
        quartile_rows,
        ("benchmark", "quartile"),
        {(benchmark, f"Q{order}") for benchmark in BENCHMARKS for order in range(1, 5)},
        args.baseline_quartiles_csv,
    )

    for key, row in main.items():
        benchmark, method = key
        require(row["result_type"] == EXPECTED_SOURCE[method], f"paper/local source mismatch for {key}")
        rmsd = finite(row, "bb_rmsd", args.main_performance_csv)
        tm = finite(row, "bb_tmscore", args.main_performance_csv)
        expected_rmsd, expected_tm = EXPECTED_PERFORMANCE[key]
        close(rmsd, expected_rmsd, f"{key} BB RMSD")
        close(tm, expected_tm, f"{key} TM-score")
        if method.startswith("Paper"):
            require(row["num_samples"].strip() == "", f"paper reference unexpectedly has paired N: {key}")
        else:
            require(int(row["num_samples"]) == EXPECTED_N[benchmark], f"local N mismatch for {key}")

    numeric_statistics = (
        "num_samples",
        "mean_improvement",
        "median_improvement",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "wilcoxon_raw_p",
        "wilcoxon_holm_p",
        "rank_biserial",
        "improved_count",
        "worsened_count",
        "tie_count",
        "improved_percent",
        "binomial_two_sided_p",
    )
    for key, row in statistics.items():
        benchmark, metric = key
        for column in numeric_statistics:
            finite(row, column, args.paired_statistics_csv)
        require(int(row["num_samples"]) == EXPECTED_N[benchmark], f"statistics N mismatch for {key}")
        mean = finite(row, "mean_improvement", args.paired_statistics_csv)
        lower = finite(row, "bootstrap_ci95_lower", args.paired_statistics_csv)
        upper = finite(row, "bootstrap_ci95_upper", args.paired_statistics_csv)
        holm_p = finite(row, "wilcoxon_holm_p", args.paired_statistics_csv)
        expected_mean, expected_lower, expected_upper, expected_p = EXPECTED_STATISTICS[key]
        close(mean, expected_mean, f"{key} mean improvement")
        close(lower, expected_lower, f"{key} CI lower")
        close(upper, expected_upper, f"{key} CI upper")
        close(holm_p, expected_p, f"{key} Holm p")
        if metric == "rmsd":
            direction_check = finite(
                main[(benchmark, "Local Bit")], "bb_rmsd", args.main_performance_csv
            ) - finite(
                main[(benchmark, "Local Bit + Latent FM")],
                "bb_rmsd",
                args.main_performance_csv,
            )
        else:
            direction_check = finite(
                main[(benchmark, "Local Bit + Latent FM")],
                "bb_tmscore",
                args.main_performance_csv,
            ) - finite(
                main[(benchmark, "Local Bit")], "bb_tmscore", args.main_performance_csv
            )
        close(mean, direction_check, f"{key} improvement direction")
        require(lower <= 0.0 <= upper, f"finalized bootstrap CI must include zero for {key}")
        require(row["bootstrap_ci_excludes_zero"] == "False", f"CI flag mismatch for {key}")
        expected_significant = key == ("cameo2022", "tmscore")
        require((holm_p < 0.05) == expected_significant, f"Holm result mismatch for {key}")
        require(row["holm_significant_0_05"] == str(expected_significant), f"Holm flag mismatch for {key}")

    for benchmark, row in local.items():
        require(int(row["N"]) == EXPECTED_N[benchmark], f"local summary N mismatch for {benchmark}")
        cross_checks = (
            ("bit_rmsd", main[(benchmark, "Local Bit")], "bb_rmsd"),
            ("fm_rmsd", main[(benchmark, "Local Bit + Latent FM")], "bb_rmsd"),
            ("bit_tmscore", main[(benchmark, "Local Bit")], "bb_tmscore"),
            ("fm_tmscore", main[(benchmark, "Local Bit + Latent FM")], "bb_tmscore"),
            ("rmsd_holm_p", statistics[(benchmark, "rmsd")], "wilcoxon_holm_p"),
            ("tm_holm_p", statistics[(benchmark, "tmscore")], "wilcoxon_holm_p"),
        )
        for local_column, other_row, other_column in cross_checks:
            close(
                finite(row, local_column, args.local_summary_csv),
                finite(other_row, other_column, args.main_performance_csv if other_column.startswith("bb_") else args.paired_statistics_csv),
                f"local summary cross-check {benchmark}/{local_column}",
            )

    for benchmark in BENCHMARKS:
        require(
            sum(int(quartiles[(benchmark, f"Q{order}")]["num_samples"]) for order in range(1, 5))
            == EXPECTED_N[benchmark],
            f"quartile counts do not sum to N for {benchmark}",
        )
        for order in range(1, 5):
            row = quartiles[(benchmark, f"Q{order}")]
            for column in QUARTILE_COLUMNS - {"benchmark", "quartile"}:
                finite(row, column, args.baseline_quartiles_csv)
    close_rounded(finite(quartiles[("cameo2022", "Q1")], "mean_rmsd_improvement", args.baseline_quartiles_csv), -0.085066, 6, "CAMEO Q1")
    close_rounded(finite(quartiles[("cameo2022", "Q4")], "mean_rmsd_improvement", args.baseline_quartiles_csv), 0.421299, 6, "CAMEO Q4")
    close_rounded(finite(quartiles[("PDB_date", "Q1")], "mean_rmsd_improvement", args.baseline_quartiles_csv), -0.061178, 6, "PDB-date Q1")
    close_rounded(finite(quartiles[("PDB_date", "Q4")], "mean_rmsd_improvement", args.baseline_quartiles_csv), 0.062787, 6, "PDB-date Q4")

    validate_metadata(normalized_metadata(args.metadata_txt))
    validate_pdfs((args.figure1_pdf, args.figure2_pdf, args.figure3_pdf))
    return main, statistics, local, quartiles


def source_label(method: str) -> str:
    return "Paper reference only" if method.startswith("Paper") else "Local reproduction"


def table1_markdown(main: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    lines = [
        "# Table 1. Folding performance of DPLM-2.1 Bit and latent residual refinement.",
        "",
        "| Method | Result source | CAMEO BB RMSD ↓ | CAMEO TM-score ↑ | PDB-date BB RMSD ↓ | PDB-date TM-score ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cameo, pdb = main[("cameo2022", method)], main[("PDB_date", method)]
        lines.append(
            f"| {method} | {source_label(method)} | {float(cameo['bb_rmsd']):.4f} | "
            f"{float(cameo['bb_tmscore']):.4f} | {float(pdb['bb_rmsd']):.4f} | {float(pdb['bb_tmscore']):.4f} |"
        )
    lines.extend(
        [
            "",
            "*Note.* Paper results are reported reference values and are not locally paired with the proposed method. "
            "Statistical comparisons are performed only between the local Bit baseline and Local Bit + Latent FM.",
            "",
        ]
    )
    return "\n".join(lines)


def table1_latex(main: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    rows = []
    for index, method in enumerate(METHODS):
        if index == 2:
            rows.append(r"\midrule")
        cameo, pdb = main[("cameo2022", method)], main[("PDB_date", method)]
        rows.append(
            f"{method} & {source_label(method)} & {float(cameo['bb_rmsd']):.4f} & "
            f"{float(cameo['bb_tmscore']):.4f} & {float(pdb['bb_rmsd']):.4f} & "
            f"{float(pdb['bb_tmscore']):.4f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Folding performance of DPLM-2.1 Bit and latent residual refinement.}",
            r"\label{tab:main_folding_performance}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Method & Result source & CAMEO BB RMSD $\downarrow$ & CAMEO TM-score $\uparrow$ & PDB-date BB RMSD $\downarrow$ & PDB-date TM-score $\uparrow$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize \textit{Note.} Paper results are reported reference values and are not locally paired with the proposed method. Statistical comparisons are performed only between the local Bit baseline and Local Bit + Latent FM.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )


def display_statistic(value: float) -> str:
    return f"{value:.6f}"


def table2_markdown(statistics: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    lines = [
        "# Table 2. Paired statistical analysis of Local Bit versus Local Bit + Latent FM.",
        "",
        "| Benchmark | Metric | Mean improvement | Median improvement | Bootstrap 95% CI | Holm-adjusted p | Rank-biserial | Improved targets (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for metric in METRICS:
            row = statistics[(benchmark, metric)]
            p_text = f"{float(row['wilcoxon_holm_p']):.4f}"
            if (benchmark, metric) == ("cameo2022", "tmscore"):
                p_text = f"**{p_text}**"
            lines.append(
                f"| {BENCHMARK_LABELS[benchmark]} | {'RMSD' if metric == 'rmsd' else 'TM-score'} | "
                f"{display_statistic(float(row['mean_improvement']))} | "
                f"{display_statistic(float(row['median_improvement']))} | "
                f"[{display_statistic(float(row['bootstrap_ci95_lower']))}, "
                f"{display_statistic(float(row['bootstrap_ci95_upper']))}] | {p_text} | "
                f"{float(row['rank_biserial']):.3f} | {row['improved_count']}/{row['num_samples']} "
                f"({float(row['improved_percent']):.1f}%) |"
            )
    lines.extend(
        [
            "",
            "*Note.* RMSD improvement = Bit − FM; TM-score improvement = FM − Bit. Positive improvement values "
            "favor Latent FM. Bootstrap intervals quantify uncertainty in the "
            "mean paired difference. Holm-adjusted p-values are from two-sided Wilcoxon signed-rank tests. "
            "All four bootstrap confidence intervals include zero.",
            "",
        ]
    )
    return "\n".join(lines)


def table2_latex(statistics: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    rows = []
    for benchmark in BENCHMARKS:
        for metric in METRICS:
            row = statistics[(benchmark, metric)]
            p_text = f"{float(row['wilcoxon_holm_p']):.4f}"
            if (benchmark, metric) == ("cameo2022", "tmscore"):
                p_text = rf"\textbf{{{p_text}}}"
            rows.append(
                f"{BENCHMARK_LABELS[benchmark]} & {'RMSD' if metric == 'rmsd' else 'TM-score'} & "
                f"{display_statistic(float(row['mean_improvement']))} & "
                f"{display_statistic(float(row['median_improvement']))} & "
                f"[{display_statistic(float(row['bootstrap_ci95_lower']))}, "
                f"{display_statistic(float(row['bootstrap_ci95_upper']))}] & {p_text} & "
                f"{float(row['rank_biserial']):.3f} & {row['improved_count']}/{row['num_samples']} "
                f"({float(row['improved_percent']):.1f}\\%) \\\\"
            )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Paired statistical analysis of Local Bit versus Local Bit + Latent FM.}",
            r"\label{tab:paired_statistics}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Benchmark & Metric & Mean improvement & Median improvement & Bootstrap 95\% CI & Holm-adjusted $p$ & Rank-biserial & Improved targets (\%) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize \textit{Note.} RMSD improvement $=$ Bit $-$ FM; TM-score improvement $=$ FM $-$ Bit. Positive improvement values favor Latent FM. Bootstrap intervals quantify uncertainty in the mean paired difference. Holm-adjusted $p$-values are from two-sided Wilcoxon signed-rank tests. All four bootstrap confidence intervals include zero.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )


def make_figure_captions(
    statistics: Mapping[tuple[str, str], Mapping[str, str]],
    quartiles: Mapping[tuple[str, str], Mapping[str, str]],
) -> str:
    cameo_tm = statistics[("cameo2022", "tmscore")]
    cameo_rmsd = statistics[("cameo2022", "rmsd")]
    captions = [
        (
            "Figure 1. Local folding performance of DPLM-2.1 Bit and latent residual refinement. "
            "Local Bit and Local Bit + Latent FM are compared on the same paired local generations for CAMEO "
            "(N=163) and PDB-date (N=442). Lower BB RMSD is better, whereas higher TM-score is better. "
            "Paper Bit + RESDIFF reference results are not included in this figure."
        ),
        (
            "Figure 2. Paired mean improvements for Local Bit + Latent FM relative to Local Bit, with percentile "
            "bootstrap 95% confidence intervals. RMSD improvement is Bit − FM and TM-score improvement is FM − "
            "Bit, so positive values favor Latent FM. Intervals use 10,000 paired resamples with seed 42. Annotated "
            "p-values are from two-sided Wilcoxon signed-rank tests with Holm correction across four tests. The "
            f"CAMEO TM-score paired rank shift was supported (Holm-adjusted p={float(cameo_tm['wilcoxon_holm_p']):.4f}), "
            f"although its bootstrap CI for the mean narrowly included zero [{float(cameo_tm['bootstrap_ci95_lower']):.6f}, "
            f"{float(cameo_tm['bootstrap_ci95_upper']):.6f}]. CAMEO RMSD had a positive mean improvement "
            f"({float(cameo_rmsd['mean_improvement']):.6f} Å), but its CI included zero and its Holm-adjusted "
            "Wilcoxon result was not significant. Neither PDB-date metric showed a statistically supported paired difference."
        ),
        (
            "Figure 3. Mean RMSD improvement (Bit − FM) stratified by baseline Bit RMSD difficulty quartile, where "
            "Q1 contains the lowest baseline Bit RMSD values (relatively easiest) and Q4 the highest (relatively "
            "hardest). CAMEO mean improvement ranged from "
            f"{float(quartiles[('cameo2022', 'Q1')]['mean_rmsd_improvement']):+.6f} Å in Q1 to "
            f"{float(quartiles[('cameo2022', 'Q4')]['mean_rmsd_improvement']):+.6f} Å in Q4; PDB-date ranged from "
            f"{float(quartiles[('PDB_date', 'Q1')]['mean_rmsd_improvement']):+.6f} Å to "
            f"{float(quartiles[('PDB_date', 'Q4')]['mean_rmsd_improvement']):+.6f} Å. This is a POST-HOC "
            "DESCRIPTIVE ANALYSIS and was NOT USED FOR MODEL OR HYPERPARAMETER SELECTION. Because baseline metrics "
            "and change scores are mathematically coupled, this pattern is exploratory and does not establish a causal relationship."
        ),
    ]
    return "\n\n".join(captions) + "\n"


def make_results_paragraphs(
    main: Mapping[tuple[str, str], Mapping[str, str]],
    statistics: Mapping[tuple[str, str], Mapping[str, str]],
    quartiles: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[str]:
    def performance(benchmark: str, method: str, column: str) -> float:
        return float(main[(benchmark, method)][column])

    c_r, c_t = statistics[("cameo2022", "rmsd")], statistics[("cameo2022", "tmscore")]
    p_r, p_t = statistics[("PDB_date", "rmsd")], statistics[("PDB_date", "tmscore")]
    return [
        (
            "Latent residual refinement improved aggregate mean performance on CAMEO. Relative to Local Bit, "
            f"Local Bit + Latent FM changed mean BB RMSD from {performance('cameo2022', 'Local Bit', 'bb_rmsd'):.4f} "
            f"Å to {performance('cameo2022', 'Local Bit + Latent FM', 'bb_rmsd'):.4f} Å and mean TM-score from "
            f"{performance('cameo2022', 'Local Bit', 'bb_tmscore'):.4f} to "
            f"{performance('cameo2022', 'Local Bit + Latent FM', 'bb_tmscore'):.4f} (N=163). Table 1 reports paper "
            "values only as references; they are not locally paired with Latent FM."
        ),
        (
            f"In the paired CAMEO analysis, the mean RMSD improvement was {float(c_r['mean_improvement']):.6f} Å "
            f"(95% bootstrap CI [{float(c_r['bootstrap_ci95_lower']):.6f}, {float(c_r['bootstrap_ci95_upper']):.6f}]; "
            f"Holm-adjusted p={float(c_r['wilcoxon_holm_p']):.4f}). Thus, CAMEO RMSD improved in the aggregate "
            "mean, but the confidence interval included zero and the paired statistical evidence was insufficient. "
            f"For CAMEO TM-score, the mean improvement was {float(c_t['mean_improvement']):.6f} (95% bootstrap CI "
            f"[{float(c_t['bootstrap_ci95_lower']):.6f}, {float(c_t['bootstrap_ci95_upper']):.6f}]). The Holm-adjusted "
            f"paired rank shift was supported (p={float(c_t['wilcoxon_holm_p']):.4f}), although the bootstrap CI "
            "for the mean narrowly included zero."
        ),
        (
            f"On PDB-date (N=442), aggregate differences were negligible: mean BB RMSD changed from "
            f"{performance('PDB_date', 'Local Bit', 'bb_rmsd'):.4f} Å to "
            f"{performance('PDB_date', 'Local Bit + Latent FM', 'bb_rmsd'):.4f} Å, and mean TM-score changed from "
            f"{performance('PDB_date', 'Local Bit', 'bb_tmscore'):.4f} to "
            f"{performance('PDB_date', 'Local Bit + Latent FM', 'bb_tmscore'):.4f}. Neither RMSD (mean paired "
            f"improvement {float(p_r['mean_improvement']):.6f} Å; Holm-adjusted p={float(p_r['wilcoxon_holm_p']):.4f}) "
            f"nor TM-score (mean paired improvement {float(p_t['mean_improvement']):.6f}; Holm-adjusted "
            f"p={float(p_t['wilcoxon_holm_p']):.4f}) showed a statistically supported paired difference."
        ),
        (
            "A post-hoc descriptive stratification suggested larger RMSD gains among targets with poorer Bit "
            "baseline performance. Mean RMSD improvement changed from "
            f"{float(quartiles[('cameo2022', 'Q1')]['mean_rmsd_improvement']):+.6f} Å in CAMEO Q1 to "
            f"{float(quartiles[('cameo2022', 'Q4')]['mean_rmsd_improvement']):+.6f} Å in Q4, and from "
            f"{float(quartiles[('PDB_date', 'Q1')]['mean_rmsd_improvement']):+.6f} Å in PDB-date Q1 to "
            f"{float(quartiles[('PDB_date', 'Q4')]['mean_rmsd_improvement']):+.6f} Å in Q4. This analysis was not "
            "used for model or hyperparameter selection."
        ),
        (
            "Taken together, the Latent FM effect appears benchmark- and target-dependent rather than universal. "
            "The baseline-difficulty pattern is exploratory, is subject to mathematical coupling between the "
            "baseline metric and change score, and should not be interpreted causally. The evidence supports a "
            "CAMEO TM-score paired rank shift, but does not support broad claims of significant RMSD improvement, "
            "consistent improvement on PDB-date, superiority to paper RESDIFF, or complete recovery of quantization loss."
        ),
    ]


def manuscript_notes() -> str:
    return """[MANUSCRIPT RESULTS NOTES]

Paper values:
reference only

Local paired statistics:
Local Bit vs Local Bit + Latent FM only

Final FM:
alpha=0.50
checkpoint_best at optimizer step 100000
Euler steps=100

Evaluation:
CAMEO N=163
PDB-date N=442

Alpha selection:
internal validation only
not test-set selected

Figure 3:
post-hoc descriptive analysis

No new statistical test performed.
"""


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    args = parse_args()
    main_data, statistics, _local, quartiles = validate_inputs(args)
    results_paragraphs = make_results_paragraphs(main_data, statistics, quartiles)
    require(4 <= len(results_paragraphs) <= 6, "Results draft must contain 4-6 paragraphs")

    outputs = {
        args.output_dir / "table1_main_performance.md": table1_markdown(main_data),
        args.output_dir / "table1_main_performance.tex": table1_latex(main_data),
        args.output_dir / "table2_paired_statistics.md": table2_markdown(statistics),
        args.output_dir / "table2_paired_statistics.tex": table2_latex(statistics),
        args.output_dir / "figure_captions.txt": make_figure_captions(statistics, quartiles),
        args.output_dir / "results_section_draft.txt": "\n\n".join(results_paragraphs) + "\n",
        args.output_dir / "manuscript_results_notes.txt": manuscript_notes(),
    }
    for path, content in outputs.items():
        atomic_write(path, content)

    print("[MANUSCRIPT RESULTS PREPARATION]")
    print()
    print("Table 1:")
    print("4 methods × 2 benchmarks")
    print()
    print("Table 2:")
    print("4 local paired benchmark-metric rows")
    print()
    print("Figure captions:")
    print("3")
    print()
    print("Results paragraphs:")
    print(len(results_paragraphs))
    print()
    print("Paper/local separation: PASS")
    print("No new statistical analysis: YES")
    print("No test-set tuning: YES")
    print()
    print("created files:")
    for path in outputs:
        print(path)
    print()
    print("MANUSCRIPT_RESULTS_PREPARATION_PASS")


if __name__ == "__main__":
    main()

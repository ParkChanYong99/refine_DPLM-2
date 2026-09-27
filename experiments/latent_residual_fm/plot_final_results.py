#!/usr/bin/env python3
"""Create publication figures from the finalized result tables only."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("generation-results/dplm2_bit_650m_final/final_results")
PERFORMANCE_CSV = RESULTS_DIR / "figure_benchmark_performance.csv"
STATISTICS_CSV = RESULTS_DIR / "paired_statistical_table.csv"
BASELINE_DIFFICULTY_CSV = RESULTS_DIR / "figure_baseline_difficulty.csv"
OUTPUT_DIR = RESULTS_DIR / "figures"

BENCHMARKS = ("cameo2022", "PDB_date")
BENCHMARK_LABELS = {"cameo2022": "CAMEO", "PDB_date": "PDB-date"}
PERFORMANCE_METRICS = ("BB_RMSD", "TM_SCORE")
STATISTICAL_METRICS = ("rmsd", "tmscore")
QUARTILES = ("Q1", "Q2", "Q3", "Q4")

PERFORMANCE_COLUMNS = {
    "benchmark",
    "metric",
    "bit_value",
    "fm_value",
    "improvement",
    "relative_improvement_percent",
    "better_direction",
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
BASELINE_COLUMNS = {
    "benchmark",
    "quartile",
    "quartile_order",
    "mean_rmsd_improvement",
    "median_rmsd_improvement",
    "rmsd_improved_percent",
}

COLORS = {"bit": "#4C78A8", "fm": "#F58518", "effect": "#3A7D44"}
ALLCLOSE_RTOL = 1e-12
ALLCLOSE_ATOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance_csv", type=Path, default=PERFORMANCE_CSV)
    parser.add_argument("--statistics_csv", type=Path, default=STATISTICS_CSV)
    parser.add_argument("--baseline_difficulty_csv", type=Path, default=BASELINE_DIFFICULTY_CSV)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"input CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        require(
            required_columns <= fieldnames,
            f"{path}: missing columns {sorted(required_columns - fieldnames)}",
        )
        return list(reader)


def finite_float(row: Mapping[str, str], column: str, source: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError(f"{source}: invalid numeric value in {column}: {row.get(column)!r}") from error
    require(math.isfinite(value), f"{source}: NaN/Inf in {column}")
    return value


def unique_index(
    rows: Sequence[dict[str, str]],
    key_columns: Sequence[str],
    expected_keys: set[tuple[str, ...]],
    source: Path,
) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[column] for column in key_columns)
        require(key not in index, f"{source}: duplicate key {key}")
        index[key] = row
    require(len(rows) == len(expected_keys), f"{source}: expected {len(expected_keys)} rows, found {len(rows)}")
    require(set(index) == expected_keys, f"{source}: unexpected keys {sorted(set(index) ^ expected_keys)}")
    return index


def validate_inputs(
    performance_rows: Sequence[dict[str, str]],
    statistical_rows: Sequence[dict[str, str]],
    baseline_rows: Sequence[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
]:
    performance_keys = {(benchmark, metric) for benchmark in BENCHMARKS for metric in PERFORMANCE_METRICS}
    statistical_keys = {(benchmark, metric) for benchmark in BENCHMARKS for metric in STATISTICAL_METRICS}
    baseline_keys = {(benchmark, quartile) for benchmark in BENCHMARKS for quartile in QUARTILES}

    performance = unique_index(performance_rows, ("benchmark", "metric"), performance_keys, args.performance_csv)
    statistics = unique_index(statistical_rows, ("benchmark", "metric"), statistical_keys, args.statistics_csv)
    baseline = unique_index(baseline_rows, ("benchmark", "quartile"), baseline_keys, args.baseline_difficulty_csv)

    for key, row in performance.items():
        bit = finite_float(row, "bit_value", args.performance_csv)
        fm = finite_float(row, "fm_value", args.performance_csv)
        improvement = finite_float(row, "improvement", args.performance_csv)
        finite_float(row, "relative_improvement_percent", args.performance_csv)
        expected_direction = "lower" if key[1] == "BB_RMSD" else "higher"
        expected_improvement = bit - fm if key[1] == "BB_RMSD" else fm - bit
        require(row["better_direction"] == expected_direction, f"{args.performance_csv}: wrong direction for {key}")
        require(
            math.isclose(improvement, expected_improvement, rel_tol=ALLCLOSE_RTOL, abs_tol=ALLCLOSE_ATOL),
            f"{args.performance_csv}: FM improvement direction/value mismatch for {key}",
        )

    numeric_statistical_columns = (
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
    metric_mapping = {"rmsd": "BB_RMSD", "tmscore": "TM_SCORE"}
    for (benchmark, metric), row in statistics.items():
        for column in numeric_statistical_columns:
            finite_float(row, column, args.statistics_csv)
        mean = finite_float(row, "mean_improvement", args.statistics_csv)
        lower = finite_float(row, "bootstrap_ci95_lower", args.statistics_csv)
        upper = finite_float(row, "bootstrap_ci95_upper", args.statistics_csv)
        require(lower <= mean <= upper, f"{args.statistics_csv}: CI does not bracket mean for {(benchmark, metric)}")
        finalized_improvement = finite_float(
            performance[(benchmark, metric_mapping[metric])], "improvement", args.performance_csv
        )
        require(
            math.isclose(mean, finalized_improvement, rel_tol=ALLCLOSE_RTOL, abs_tol=ALLCLOSE_ATOL),
            f"finalized mean improvement mismatch for {(benchmark, metric)}",
        )

    for benchmark in BENCHMARKS:
        orders = []
        for quartile in QUARTILES:
            row = baseline[(benchmark, quartile)]
            order = finite_float(row, "quartile_order", args.baseline_difficulty_csv)
            require(order.is_integer(), f"{args.baseline_difficulty_csv}: non-integer quartile_order")
            orders.append(int(order))
            finite_float(row, "mean_rmsd_improvement", args.baseline_difficulty_csv)
            finite_float(row, "median_rmsd_improvement", args.baseline_difficulty_csv)
            percent = finite_float(row, "rmsd_improved_percent", args.baseline_difficulty_csv)
            require(0.0 <= percent <= 100.0, f"{args.baseline_difficulty_csv}: invalid improved percent")
        require(orders == [1, 2, 3, 4], f"{args.baseline_difficulty_csv}: {benchmark} is not ordered Q1-Q4")

    return performance, statistics, baseline


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, title: str, subject: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"Title": title, "Subject": subject})
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Description": subject},
    )
    plt.close(fig)
    return [pdf_path, png_path]


def add_bar_labels(ax: plt.Axes, bars: Sequence[matplotlib.patches.Patch], precision: int) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.{precision}f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_figure1(performance: Mapping[tuple[str, str], Mapping[str, str]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), constrained_layout=True)
    x = np.arange(len(BENCHMARKS), dtype=float)
    width = 0.34
    panel_specs = (
        ("BB_RMSD", "(a) BB RMSD", "Mean BB RMSD (Å)", 3),
        ("TM_SCORE", "(b) TM-score", "Mean TM-score", 4),
    )
    for ax, (metric, title, ylabel, precision) in zip(axes, panel_specs):
        bit_values = [finite_float(performance[(benchmark, metric)], "bit_value", PERFORMANCE_CSV) for benchmark in BENCHMARKS]
        fm_values = [finite_float(performance[(benchmark, metric)], "fm_value", PERFORMANCE_CSV) for benchmark in BENCHMARKS]
        bit_bars = ax.bar(
            x - width / 2,
            bit_values,
            width,
            label="Local Bit",
            color=COLORS["bit"],
            edgecolor="#222222",
            linewidth=0.7,
            hatch="///",
        )
        fm_bars = ax.bar(
            x + width / 2,
            fm_values,
            width,
            label="Local Bit + Latent FM",
            color=COLORS["fm"],
            edgecolor="#222222",
            linewidth=0.7,
            hatch="...",
        )
        add_bar_labels(ax, bit_bars, precision)
        add_bar_labels(ax, fm_bars, precision)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [BENCHMARK_LABELS[benchmark] for benchmark in BENCHMARKS])
        ax.set_ylim(bottom=0.0, top=max(bit_values + fm_values) * 1.18)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Local Bit vs Latent FM performance", fontsize=13, fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "figure1_local_bit_vs_fm_performance",
        "Local Bit vs Latent FM performance",
        "Local paired comparison. Paper RESDIFF is not included in this figure.",
    )


def make_figure2(statistics: Mapping[tuple[str, str], Mapping[str, str]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), constrained_layout=True)
    x = np.arange(len(BENCHMARKS), dtype=float)
    panel_specs = (
        ("rmsd", "(a) RMSD improvement", "Mean improvement, Bit − FM (Å)"),
        ("tmscore", "(b) TM-score improvement", "Mean improvement, FM − Bit"),
    )
    markers = ("o", "s")
    for ax, (metric, title, ylabel) in zip(axes, panel_specs):
        all_limits = [0.0]
        for position, benchmark, marker in zip(x, BENCHMARKS, markers):
            row = statistics[(benchmark, metric)]
            mean = float(row["mean_improvement"])
            lower = float(row["bootstrap_ci95_lower"])
            upper = float(row["bootstrap_ci95_upper"])
            holm_p = float(row["wilcoxon_holm_p"])
            all_limits.extend((lower, upper))
            ax.errorbar(
                position,
                mean,
                yerr=np.array([[mean - lower], [upper - mean]]),
                fmt=marker,
                markersize=7,
                color=COLORS["effect"],
                markerfacecolor="white",
                markeredgewidth=1.6,
                capsize=5,
                elinewidth=1.4,
                zorder=3,
            )
            ax.annotate(
                f"Holm p={holm_p:.4f}",
                (position, upper),
                xytext=(12 if benchmark == "cameo2022" else 0, 16),
                textcoords="offset points",
                ha="left" if benchmark == "cameo2022" else "center",
                va="bottom",
                fontsize=8,
            )
        span = max(all_limits) - min(all_limits)
        margin = 0.24 * span if span > 0 else 0.05
        ax.set_ylim(min(all_limits) - margin, max(all_limits) + margin * 2.4)
        ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [BENCHMARK_LABELS[benchmark] for benchmark in BENCHMARKS])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    fig.suptitle("Paired mean improvement with bootstrap 95% CI", fontsize=13, fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "figure2_paired_improvement_ci",
        "Paired mean improvement with bootstrap 95% CI",
        "Positive values indicate FM improvement. Existing adjusted p-values and bootstrap intervals only; no new test.",
    )


def make_figure3(baseline: Mapping[tuple[str, str], Mapping[str, str]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5), constrained_layout=True)
    x = np.arange(len(QUARTILES), dtype=float)
    for ax, benchmark, panel in zip(axes, BENCHMARKS, ("(a)", "(b)")):
        means = [float(baseline[(benchmark, quartile)]["mean_rmsd_improvement"]) for quartile in QUARTILES]
        percentages = [float(baseline[(benchmark, quartile)]["rmsd_improved_percent"]) for quartile in QUARTILES]
        ax.plot(
            x,
            means,
            color=COLORS["effect"],
            marker="o" if benchmark == "cameo2022" else "s",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.6,
            linewidth=1.6,
        )
        for position, quartile, mean, percent in zip(x, QUARTILES, means, percentages):
            offset = -3 if benchmark == "PDB_date" and quartile == "Q1" else (9 if mean >= 0 else -11)
            ax.annotate(
                f"{percent:.1f}% improved",
                (position, mean),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=7.5,
            )
        data_min, data_max = min([0.0] + means), max([0.0] + means)
        span = data_max - data_min
        margin = 0.28 * span if span > 0 else 0.05
        ax.set_ylim(data_min - margin * 1.5, data_max + margin)
        ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
        ax.set_title(f"{panel} {BENCHMARK_LABELS[benchmark]}", loc="left", fontweight="bold")
        ax.set_ylabel("Mean RMSD improvement, Bit − FM (Å)")
        ax.set_xticks(x, QUARTILES)
        ax.set_xlabel("Baseline difficulty (Q1 easiest → Q4 hardest)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    fig.suptitle("FM effect by baseline difficulty", fontsize=13, fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "figure3_baseline_difficulty",
        "FM effect by baseline difficulty",
        "Positive values indicate FM improvement (Bit RMSD minus FM RMSD). "
        "POST-HOC DESCRIPTIVE ANALYSIS. NOT USED FOR MODEL OR HYPERPARAMETER SELECTION. No causal claim.",
    )


def print_summary(
    performance: Mapping[tuple[str, str], Mapping[str, str]],
    statistics: Mapping[tuple[str, str], Mapping[str, str]],
    baseline: Mapping[tuple[str, str], Mapping[str, str]],
    created_files: Sequence[Path],
) -> None:
    print("[FINAL RESULTS FIGURE GENERATION]")
    print()
    print("Figure 1:")
    for benchmark in BENCHMARKS:
        label = BENCHMARK_LABELS[benchmark]
        rmsd = performance[(benchmark, "BB_RMSD")]
        tm = performance[(benchmark, "TM_SCORE")]
        print(f"{label} RMSD: Local Bit={float(rmsd['bit_value']):.3f}, Local Bit + Latent FM={float(rmsd['fm_value']):.3f}")
        print(f"{label} TM: Local Bit={float(tm['bit_value']):.4f}, Local Bit + Latent FM={float(tm['fm_value']):.4f}")
    print("Local paired comparison. Paper RESDIFF is not included in this figure.")
    print()
    print("Figure 2:")
    for benchmark in BENCHMARKS:
        label = BENCHMARK_LABELS[benchmark]
        for metric, metric_label in (("rmsd", "RMSD"), ("tmscore", "TM")):
            row = statistics[(benchmark, metric)]
            print(
                f"{label} {metric_label} mean improvement / CI: "
                f"{float(row['mean_improvement']):.10g} / "
                f"[{float(row['bootstrap_ci95_lower']):.10g}, {float(row['bootstrap_ci95_upper']):.10g}]"
            )
    print("Positive = FM improvement (RMSD: Bit − FM; TM-score: FM − Bit).")
    print()
    print("Figure 3:")
    for benchmark in BENCHMARKS:
        values = [float(baseline[(benchmark, quartile)]["mean_rmsd_improvement"]) for quartile in QUARTILES]
        rendered = " -> ".join(f"{value:.6f}" for value in values)
        print(f"{BENCHMARK_LABELS[benchmark]} Q1 -> Q4 RMSD improvement: {rendered}")
    print("Positive = FM improvement (Bit RMSD − FM RMSD).")
    print("POST-HOC DESCRIPTIVE ANALYSIS — NOT USED FOR MODEL OR HYPERPARAMETER SELECTION")
    print()
    print("created files:")
    for path in created_files:
        print(path)
    print()
    print("No new statistical test performed: YES")
    print("Test-set hyperparameter tuning performed: NO")
    print()
    print("FINAL_RESULTS_FIGURES_PASS")


def main() -> None:
    args = parse_args()
    performance_rows = read_csv(args.performance_csv, PERFORMANCE_COLUMNS)
    statistical_rows = read_csv(args.statistics_csv, STATISTICS_COLUMNS)
    baseline_rows = read_csv(args.baseline_difficulty_csv, BASELINE_COLUMNS)
    performance, statistics, baseline = validate_inputs(
        performance_rows, statistical_rows, baseline_rows, args
    )

    apply_publication_style()
    created_files = []
    created_files.extend(make_figure1(performance, args.output_dir))
    created_files.extend(make_figure2(statistics, args.output_dir))
    created_files.extend(make_figure3(baseline, args.output_dir))
    print_summary(performance, statistics, baseline, created_files)


if __name__ == "__main__":
    main()

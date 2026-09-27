#!/usr/bin/env python3
"""Prepare manuscript tables from finalized local and paper-reference artifacts only.

This is formatting/consistency validation, not training, inference, metric
recomputation, bootstrap resampling, or statistical testing.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "generation-results/dplm2_bit_650m_final"
OUTPUT = RESULTS / "manuscript"
FM_STATS = RESULTS / "statistics/final_paired_statistics.csv"
FM_DDPM_STATS = RESULTS / "ddpm_matched/comparison/fm_vs_ddpm_paired_statistics.csv"
BIT_DDPM_STATS = RESULTS / "ddpm_matched/comparison/bit_vs_ddpm_paired_statistics.csv"
FM_DDPM_PAIRS = RESULTS / "ddpm_matched/comparison/fm_vs_ddpm_paired.csv"
DDPM_SUMMARY = RESULTS / "ddpm_matched/summary.csv"
EXISTING_MAIN = RESULTS / "final_results/main_performance_table.csv"
FM_GENERATOR = ROOT / "experiments/latent_residual_fm/generate_final_folding_predictions.py"
BENCHMARKS = (
    ("CAMEO", "cameo2022", 163, "cameo2022"),
    ("PDB-date", "PDB_date", 442, "pdb_date"),
)
METHODS = ("Local Bit", "Local Bit + Latent Residual FM",
           "Local Bit + Matched Residual DDPM")
EXPECTED_LOCAL = {
    "CAMEO": {
        "Bit": (6.28888164, 0.841421163),
        "FM": (6.19972204, 0.844954294),
        "DDPM": (6.19250677, 0.843772619),
    },
    "PDB-date": {
        "Bit": (3.11376375, 0.91379618),
        "FM": (3.10744498, 0.91385542),
        "DDPM": (3.11804019, 0.913043985),
    },
}
EXPECTED_FM_DDPM = {
    ("CAMEO", "backbone_rmsd"): (-0.0072152727952, -0.149566456026,
                                   0.130140540443, 0.877531915555,
                                   0.0139159060302, (86, 77, 0)),
    ("CAMEO", "tm_score"): (0.00118167566929, -0.00219709651935,
                             0.00475046821578, 0.630634811163,
                             0.0906778392937, (90, 73, 0)),
    ("PDB-date", "backbone_rmsd"): (0.0105952068961, -0.0307988598656,
                                      0.0533013142089, 0.246828020448,
                                      0.102676968223, (241, 200, 1)),
    ("PDB-date", "tm_score"): (0.000811435143934, -0.00022016720674,
                                0.0018366161035, 0.476093223402,
                                0.0774566236751, (230, 211, 1)),
}
# DPLM-2.1 paper Table 4 reference values verified against the paper:
# https://arxiv.org/html/2504.11454v3 (Table 4). Paper Bit and RESDIFF are
# also independently checked against EXISTING_MAIN; none are local measurements.
PAPER_TABLE4 = (
    ("DPLM2 Bit", 6.4028, 0.8380, 3.2213, 0.9043),
    ("Bit + RESDIFF (paper-reported only)", 6.1781, 0.8428, 3.0168, 0.9076),
    ("Paper Bit + FM (paper data-space FM)", 6.1825, 0.8414, 2.8697, 0.9099),
    ("Bit + FM + RESDIFF (paper-reported only)", 6.0765, 0.8456, 2.7884, 0.9146),
)
LOCAL_TOL = 1e-7
STRICT_TOL = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"missing finalized artifact: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        require(len(columns) == len(set(columns)), f"duplicate column in {path}")
        missing = set(required) - set(columns)
        require(not missing, f"missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    require(rows, f"empty artifact: {path}")
    for line, row in enumerate(rows, start=2):
        require(None not in row and all(value is not None for value in row.values()),
                f"malformed CSV row: {path}:{line}")
    return rows


def number(row: Mapping[str, str], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid {key}: {context}") from exc
    require(math.isfinite(value), f"non-finite {key}: {context}")
    return value


def integer(row: Mapping[str, str], key: str, context: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid {key}: {context}") from exc


def same(actual: float, expected: float, context: str, tol: float = STRICT_TOL) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol),
            f"{context}: {actual} != {expected}")


def indexed(rows: Sequence[dict[str, str]], key: str, expected_n: int,
            context: str) -> dict[str, dict[str, str]]:
    require(len(rows) == expected_n,
            f"{context}: expected N={expected_n}, got {len(rows)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        target = row[key].strip()
        require(target and target not in result, f"empty/duplicate target {target!r}: {context}")
        result[target] = row
    return result


def unique_stats(path: Path, required: Sequence[str], expected_comparison: str | None = None
                 ) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path, required)
    require(len(rows) == 4, f"expected four finalized tests: {path}")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if expected_comparison is not None:
            require(row["comparison"] == expected_comparison,
                    f"comparison mismatch: {path}")
        key = (row["benchmark"], row["metric"])
        require(key not in result, f"duplicate statistical test {key}: {path}")
        result[key] = row
    return result


def fm_nfe_from_source() -> int:
    """Read the frozen generator's Euler-step constant without importing it."""
    require(FM_GENERATOR.is_file(), f"missing FM generator source: {FM_GENERATOR}")
    tree = ast.parse(FM_GENERATOR.read_text(encoding="utf-8"), filename=str(FM_GENERATOR))
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_EULER_STEPS"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    require(values == [100], f"FM residual NFE must be 100, found {values}")
    return values[0]


def load_local() -> tuple[dict[str, dict[str, tuple[float, float]]],
                          dict[str, dict[str, tuple[float, float]]], int]:
    fm_nfe = fm_nfe_from_source()
    summary_rows = read_csv(DDPM_SUMMARY, (
        "benchmark", "num_samples", "bit_mean_backbone_rmsd",
        "ddpm_mean_backbone_rmsd", "bit_mean_tm_score", "ddpm_mean_tm_score",
        "mean_backbone_improvement", "mean_tm_improvement",
    ))
    require(len(summary_rows) == 2, "DDPM summary must have two benchmarks")
    summary = {row["benchmark"]: row for row in summary_rows}
    require(len(summary) == 2, "duplicate DDPM summary benchmark")
    pair_rows = read_csv(FM_DDPM_PAIRS, (
        "benchmark", "target_id", "length", "bit_backbone_rmsd",
        "fm_backbone_rmsd", "ddpm_backbone_rmsd", "bit_tm_score",
        "fm_tm_score", "ddpm_tm_score",
    ))
    local: dict[str, dict[str, tuple[float, float]]] = {}
    improvements: dict[str, dict[str, tuple[float, float]]] = {}
    for display, key, n, slug in BENCHMARKS:
        fm_path = RESULTS / f"{key}/comparison/{key}_bit_vs_fm_paired.csv"
        if display == "PDB-date":
            fm_path = RESULTS / "PDB_date/comparison/pdb_date_bit_vs_fm_paired.csv"
        fm_summary_path = RESULTS / f"{key}/comparison/{slug}_bit_vs_fm_summary.csv"
        ddpm_path = RESULTS / f"ddpm_matched/{'cameo' if display == 'CAMEO' else 'pdb_date'}_per_target.csv"
        fm_pairs = indexed(read_csv(fm_path, (
            "sample_id", "length", "bit_bb_rmsd", "fm_bb_rmsd",
            "bit_bb_tmscore", "fm_bb_tmscore",
        )), "sample_id", n, f"{display} FM pairs")
        ddpm_pairs = indexed(read_csv(ddpm_path, (
            "target_id", "length", "ddpm_nfe", "bit_backbone_rmsd",
            "ddpm_backbone_rmsd", "bit_tm_score", "ddpm_tm_score",
        )), "target_id", n, f"{display} DDPM pairs")
        combined = indexed([row for row in pair_rows if row["benchmark"] == display],
                           "target_id", n, f"{display} finalized FM-DDPM pairs")
        require(set(fm_pairs) == set(ddpm_pairs) == set(combined),
                f"FM/DDPM paired target IDs mismatch: {display}")
        for target in fm_pairs:
            f, d, p = fm_pairs[target], ddpm_pairs[target], combined[target]
            require(integer(f, "length", target) == integer(d, "length", target)
                    == integer(p, "length", target), f"length mismatch: {display} {target}")
            require(d["benchmark"] == key, f"DDPM benchmark mismatch: {target}")
            require(integer(d, "ddpm_nfe", target) == 1000,
                    f"DDPM residual NFE != 1000: {display} {target}")
            for fm_column, ddpm_column, combined_column in (
                ("bit_bb_rmsd", "bit_backbone_rmsd", "bit_backbone_rmsd"),
                ("fm_bb_rmsd", "ddpm_backbone_rmsd", "fm_backbone_rmsd"),
                ("bit_bb_tmscore", "bit_tm_score", "bit_tm_score"),
                ("fm_bb_tmscore", "ddpm_tm_score", "fm_tm_score"),
            ):
                # FM columns compare to the paired artifact. Only Bit columns
                # additionally compare against the DDPM artifact.
                same(number(f, fm_column, target), number(p, combined_column, target),
                     f"FM/combined {fm_column}: {target}")
                if fm_column.startswith("bit_"):
                    same(number(f, fm_column, target), number(d, ddpm_column, target),
                         f"Bit baseline mismatch: {display} {target}")
            for dcol, pcol in (("ddpm_backbone_rmsd", "ddpm_backbone_rmsd"),
                               ("ddpm_tm_score", "ddpm_tm_score")):
                same(number(d, dcol, target), number(p, pcol, target),
                     f"DDPM/combined {dcol}: {target}")

        fm_summary_rows = read_csv(fm_summary_path, (
            "num_samples", "bit_mean_bb_rmsd", "fm_mean_bb_rmsd",
            "bit_mean_bb_tmscore", "fm_mean_bb_tmscore",
            "mean_rmsd_improvement", "mean_tmscore_improvement",
        ))
        require(len(fm_summary_rows) == 1, f"FM summary row count: {display}")
        fsum = fm_summary_rows[0]
        require(integer(fsum, "num_samples", display) == n,
                f"FM summary N mismatch: {display}")
        require(key in summary, f"DDPM summary missing {key}")
        dsum = summary[key]
        require(integer(dsum, "num_samples", display) == n,
                f"DDPM summary N mismatch: {display}")
        local[display] = {
            "Bit": (number(fsum, "bit_mean_bb_rmsd", display),
                    number(fsum, "bit_mean_bb_tmscore", display)),
            "FM": (number(fsum, "fm_mean_bb_rmsd", display),
                   number(fsum, "fm_mean_bb_tmscore", display)),
            "DDPM": (number(dsum, "ddpm_mean_backbone_rmsd", display),
                     number(dsum, "ddpm_mean_tm_score", display)),
        }
        same(local[display]["Bit"][0], number(dsum, "bit_mean_backbone_rmsd", display),
             f"Bit BB summary mismatch: {display}")
        same(local[display]["Bit"][1], number(dsum, "bit_mean_tm_score", display),
             f"Bit TM summary mismatch: {display}")
        for method, (bb, tm) in local[display].items():
            expected_bb, expected_tm = EXPECTED_LOCAL[display][method]
            same(bb, expected_bb, f"{display} {method} known BB mismatch", LOCAL_TOL)
            same(tm, expected_tm, f"{display} {method} known TM mismatch", LOCAL_TOL)
        improvements[display] = {
            "FM": (number(fsum, "mean_rmsd_improvement", display),
                   number(fsum, "mean_tmscore_improvement", display)),
            "DDPM": (number(dsum, "mean_backbone_improvement", display),
                     number(dsum, "mean_tm_improvement", display)),
        }
        for method in ("FM", "DDPM"):
            bb, tm = improvements[display][method]
            same(bb, local[display]["Bit"][0] - local[display][method][0],
                 f"{display} {method} BB improvement mismatch")
            same(tm, local[display][method][1] - local[display]["Bit"][1],
                 f"{display} {method} TM improvement mismatch")
    require(len(pair_rows) == 605, "finalized FM-DDPM paired CSV total N != 605")
    return local, improvements, fm_nfe


def load_statistics() -> tuple[dict[tuple[str, str], dict[str, str]],
                               dict[tuple[str, str], dict[str, str]],
                               dict[tuple[str, str], dict[str, str]]]:
    fm = unique_stats(FM_STATS, ("benchmark", "metric", "num_samples",
                                 "mean_improvement", "bootstrap_ci95_lower",
                                 "bootstrap_ci95_upper", "wilcoxon_holm_p",
                                 "rank_biserial"))
    fd = unique_stats(FM_DDPM_STATS, ("comparison", "benchmark", "metric", "n",
                                          "mean_paired_difference", "median_paired_difference",
                                          "bootstrap_ci95_lower", "bootstrap_ci95_upper",
                                          "wilcoxon_holm_p", "rank_biserial",
                                          "first_method_better_count",
                                          "second_method_better_count", "tied_count",
                                          "bootstrap_resamples", "bootstrap_seed"),
                      "FM_vs_DDPM")
    bd = unique_stats(BIT_DDPM_STATS, ("comparison", "benchmark", "metric", "n",
                                            "mean_paired_difference", "bootstrap_ci95_lower",
                                            "bootstrap_ci95_upper", "wilcoxon_holm_p",
                                            "rank_biserial", "bootstrap_resamples",
                                            "bootstrap_seed"), "Bit_vs_DDPM")
    require(set(fd) == set(EXPECTED_FM_DDPM), "FM-DDPM test family mismatch")
    require(set(bd) == set(fd), "Bit-DDPM test family mismatch")
    expected_fm_keys = {(key, metric) for _, key, _, _ in BENCHMARKS
                        for metric in ("rmsd", "tmscore")}
    require(set(fm) == expected_fm_keys, "finalized FM-Bit test family mismatch")
    for display, key, n, _ in BENCHMARKS:
        for metric, fm_metric in (("backbone_rmsd", "rmsd"), ("tm_score", "tmscore")):
            frow, fdrow, bdrow = fm[(key, fm_metric)], fd[(display, metric)], bd[(display, metric)]
            require(integer(frow, "num_samples", display) == n and
                    integer(fdrow, "n", display) == n and
                    integer(bdrow, "n", display) == n,
                    f"statistics artifact N mismatch: {display} {metric}")
            for label, row in (("FM-DDPM", fdrow), ("Bit-DDPM", bdrow)):
                require(integer(row, "bootstrap_resamples", label) == 10000 and
                        integer(row, "bootstrap_seed", label) == 42,
                        f"{label} finalized bootstrap protocol mismatch")
                for column in ("bootstrap_ci95_lower", "bootstrap_ci95_upper",
                               "wilcoxon_holm_p", "rank_biserial"):
                    number(row, column, f"{label} {display} {metric}")
            expected = EXPECTED_FM_DDPM[(display, metric)]
            for column, value in zip(("mean_paired_difference", "bootstrap_ci95_lower",
                                      "bootstrap_ci95_upper", "wilcoxon_holm_p",
                                      "rank_biserial"), expected[:5]):
                same(number(fdrow, column, f"{display} {metric}"), value,
                     f"known FM-DDPM {display} {metric} {column}", 1e-9)
            counts = tuple(integer(fdrow, column, f"{display} {metric}") for column in
                           ("first_method_better_count", "second_method_better_count",
                            "tied_count"))
            require(counts == expected[5] and sum(counts) == n,
                    f"FM/DDPM/tied counts mismatch: {display} {metric}")
            require(number(fdrow, "wilcoxon_holm_p", display) > 0.05,
                    f"unexpected significant FM-DDPM test: {display} {metric}")
    cameo_tm = fm[("cameo2022", "tmscore")]
    same(number(cameo_tm, "wilcoxon_holm_p", "CAMEO FM-Bit TM"),
         0.008791887171644853, "FM-Bit CAMEO TM Holm p", 1e-8)
    same(number(cameo_tm, "rank_biserial", "CAMEO FM-Bit TM"),
         0.2765225198264253, "FM-Bit CAMEO TM rank-biserial", 1e-8)
    require(number(cameo_tm, "wilcoxon_holm_p", "CAMEO FM-Bit TM") < 0.05 and
            number(cameo_tm, "rank_biserial", "CAMEO FM-Bit TM") > 0 and
            number(cameo_tm, "bootstrap_ci95_lower", "CAMEO FM-Bit TM") <= 0 <=
            number(cameo_tm, "bootstrap_ci95_upper", "CAMEO FM-Bit TM"),
            "FM-Bit CAMEO TM interpretation mismatch")
    return fm, fd, bd


def load_paper_reference() -> list[tuple[str, float, float, float, float]]:
    rows = read_csv(EXISTING_MAIN, ("benchmark", "method", "result_type",
                                    "bb_rmsd", "bb_tmscore"))
    source = {(row["benchmark"], row["method"]): row for row in rows
              if row["result_type"] == "PAPER_REFERENCE"}
    require(len(source) == 4, "expected four stored Paper Bit/RESDIFF reference rows")
    for paper_row, source_method in ((PAPER_TABLE4[0], "Paper Bit"),
                                     (PAPER_TABLE4[1], "Paper Bit + RESDIFF")):
        for benchmark, bb, tm in (("cameo2022", paper_row[1], paper_row[2]),
                                  ("PDB_date", paper_row[3], paper_row[4])):
            require((benchmark, source_method) in source,
                    f"missing stored paper reference: {benchmark} {source_method}")
            stored = source[(benchmark, source_method)]
            same(number(stored, "bb_rmsd", source_method), bb,
                 f"paper Table 4 BB mismatch: {benchmark} {source_method}")
            same(number(stored, "bb_tmscore", source_method), tm,
                 f"paper Table 4 TM mismatch: {benchmark} {source_method}")
    return list(PAPER_TABLE4)


def local_rows(local: Mapping[str, Mapping[str, tuple[float, float]]], fm_nfe: int
               ) -> list[tuple[str, str, float, float, float, float]]:
    rows = []
    for method, key, nfe in ((METHODS[0], "Bit", "N/A"),
                             (METHODS[1], "FM", str(fm_nfe)),
                             (METHODS[2], "DDPM", "1000")):
        rows.append((method, nfe, *local["CAMEO"][key], *local["PDB-date"][key]))
    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                caption: str, label: str) -> str:
    escaped = lambda value: value.replace("_", "\\_").replace("%", "\\%")
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}",
             "\\hline", " & ".join(escaped(item) for item in headers) + r" \\",
             "\\hline"]
    for row in rows:
        lines.append(" & ".join(escaped(item) for item in row) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}",
                  f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                  "\\end{table}", ""])
    return "\n".join(lines)


def render_local(local: Mapping[str, Mapping[str, tuple[float, float]]], fm_nfe: int
                 ) -> tuple[str, str]:
    headers = ("Method", "Residual sampler NFE", "CAMEO BB RMSD ↓ (N=163)",
               "CAMEO TM-score ↑", "PDB-date BB RMSD ↓ (N=442)",
               "PDB-date TM-score ↑")
    rows = []
    for method, nfe, cb, ct, pb, pt in local_rows(local, fm_nfe):
        rows.append((method, nfe, f"{cb:.4f}", f"{ct:.5f}",
                     f"{pb:.4f}", f"{pt:.5f}"))
    note = ("Bold, if used, indicates the numerically best local mean only; "
            "paired statistical significance is reported separately. "
            "NFE counts residual-model evaluations, not DPLM base-generation iterations.")
    md = ("# Table 3. Local matched method comparison\n\n" +
          markdown_table(headers, rows) + "\n" + note + "\n")
    tex_headers = ("Method", "Residual NFE", "CAMEO BB RMSD", "CAMEO TM",
                   "PDB-date BB RMSD", "PDB-date TM")
    tex_rows = [(row[0], *row[1:]) for row in rows]
    tex = latex_table(tex_headers, tex_rows,
                      "Local matched methods (CAMEO $N=163$; PDB-date $N=442$). "
                      "Lower backbone RMSD and higher TM-score are better. "
                      "Bold, if used, indicates the numerically best local mean only; "
                      "paired statistical significance is reported separately. "
                      "NFE denotes residual-model evaluations, not DPLM iterations.",
                      "tab:local_matched_methods")
    return md, tex


def render_paper(paper: Sequence[tuple[str, float, float, float, float]]
                 ) -> tuple[str, str]:
    headers = ("Paper method", "CAMEO RMSD ↓", "CAMEO TM ↑",
               "PDB-date RMSD ↓", "PDB-date TM ↑")
    rows = [(name, f"{cb:.4f}", f"{ct:.4f}", f"{pb:.4f}", f"{pt:.4f}")
            for name, cb, ct, pb, pt in paper]
    caution = ("Paper-reported reference only; not a local matched comparison. "
               "Paper Bit + FM is the paper data-space FM, not our Latent Residual FM. "
               "Local Matched Residual DDPM is not a RESDIFF reproduction.")
    md = "# Table 4. DPLM-2.1 paper-reported reference results\n\n" + markdown_table(headers, rows) + "\n" + caution + "\n"
    tex = latex_table(("Paper method", "CAMEO RMSD", "CAMEO TM", "PDB-date RMSD", "PDB-date TM"), rows, caution, "tab:paper_reference_methods")
    return md, tex


def render_statistics(fm: Mapping[tuple[str, str], Mapping[str, str]],
                      fd: Mapping[tuple[str, str], Mapping[str, str]],
                      bd: Mapping[tuple[str, str], Mapping[str, str]]
                      ) -> tuple[str, str]:
    headers = ("Comparison", "Benchmark", "Metric", "Mean difference",
               "Bootstrap 95% CI", "Holm p", "Rank-biserial", "Better / worse / tied")
    rows: list[tuple[str, ...]] = []
    for comparison, source, label in (("FM vs DDPM (primary)", fd, "FM / DDPM"),
                                      ("DDPM vs Bit (secondary)", bd, "DDPM / Bit")):
        for display, _, _, _ in BENCHMARKS:
            for metric in ("backbone_rmsd", "tm_score"):
                row = source[(display, metric)]
                counts = "/".join(row[column] for column in
                                  ("first_method_better_count",
                                   "second_method_better_count", "tied_count"))
                rows.append((comparison, display,
                             "BB RMSD" if metric == "backbone_rmsd" else "TM-score",
                             f"{number(row, 'mean_paired_difference', display):+.6f}",
                             f"[{number(row, 'bootstrap_ci95_lower', display):.6f}, "
                             f"{number(row, 'bootstrap_ci95_upper', display):.6f}]",
                             f"{number(row, 'wilcoxon_holm_p', display):.9g}",
                             f"{number(row, 'rank_biserial', display):+.6f}",
                             f"{counts} ({label})"))
    fm_tm = fm[("cameo2022", "tmscore")]
    fm_note = ("Finalized FM-vs-Bit CAMEO TM: Holm p="
               f"{number(fm_tm, 'wilcoxon_holm_p', 'FM-Bit'):.9g}; "
               "significant positive paired rank shift after Holm correction, "
               "while the bootstrap CI for the mean improvement includes zero.")
    caveat = ("FM-vs-DDPM primary tests are all non-significant after Holm correction. "
              "Positive FM advantage: BB=DDPM−FM, TM=FM−DDPM. "
              "DDPM-vs-Bit results are a separate secondary family; positive means DDPM better.")
    md = ("# Local paired method-comparison statistics\n\n" +
          markdown_table(headers, rows) + "\n" + caveat + "\n\n" + fm_note + "\n")
    tex = latex_table(headers, rows, caveat + " " + fm_note,
                      "tab:local_method_comparison_statistics")
    return md, tex


def render_notes(local: Mapping[str, Mapping[str, tuple[float, float]]],
                 improvements: Mapping[str, Mapping[str, tuple[float, float]]],
                 fm: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    lines = [
        "FINAL METHOD COMPARISON NOTES — post-hoc, finalized artifacts only",
        "",
        "Local matched population: CAMEO N=163; PDB-date N=442.",
        "Residual sampler NFE: Bit N/A; Latent Residual FM 100; "
        "Local Matched Residual DDPM 1000 (full ancestral).",
        "NFE counts residual-refinement network evaluations, not DPLM base-generation iterations.",
        "Do not infer a 10x wall-clock speedup from one-tenth the NFE.",
        "",
        "Bit-relative mean change uses BB = Bit RMSD - refined RMSD; "
        "TM = refined TM-score - Bit TM-score. Positive is improvement.",
    ]
    for display, _, _, _ in BENCHMARKS:
        for method in ("FM", "DDPM"):
            bb, tm = improvements[display][method]
            lines.append(f"{display} {method} vs Bit: BB improvement={bb:+.10f} Å; "
                         f"TM improvement={tm:+.12f}.")
    lines.extend([
        "",
        "FM and the matched DDPM baseline showed no statistically significant "
        "difference in final structural quality on the four primary paired comparisons "
        "(all Holm-adjusted p > 0.05).",
        "FM and matched DDPM showed similar observed mean structural quality, "
        "and the four primary paired comparisons did not detect a statistically "
        "significant difference. This does not establish equivalence.",
        "FM used 100 residual-model NFEs, compared with 1000 for full ancestral "
        "DDPM; this is one-tenth the NFE, not a claim of 10x faster wall-clock execution.",
        "",
        "Finalized FM-vs-Bit CAMEO TM-score: significant positive paired rank shift "
        "after Holm correction; bootstrap CI for the mean improvement includes zero.",
        "A significant FM-vs-Bit result and a non-significant DDPM-vs-Bit result "
        "do not imply that FM significantly outperforms DDPM; the direct "
        "FM-vs-DDPM tests are non-significant.",
        "",
        "The Local Matched Residual DDPM did not show improvements of the same "
        "magnitude as the paper-reported RESDIFF reference in our local evaluation.",
        "These local results do not support the hypothesis that the diffusion "
        "formulation alone is sufficient to explain the paper-reported RESDIFF improvements.",
        "Because this is not an exact RESDIFF reproduction or a controlled "
        "paper-vs-local comparison, this result does not refute the paper-reported "
        "RESDIFF findings.",
        "Paper-reported results and local matched results must not be treated as a "
        "controlled head-to-head comparison or used to claim statistical superiority.",
        "Potentially differing or unknown factors: exact residual training corpus; "
        "RESDIFF implementation details; diffusion schedule/sampler details; "
        "training exposure; condition implementation details; "
        "evaluation-generation realization.",
        "Paper Bit + FM denotes paper data-space FM, not our Latent Residual FM.",
        "Paper RESDIFF values are paper-reported reference only.",
        "",
        "Local metrics and inference statistics were read from finalized CSV artifacts; "
        "no new RMSD/TM, bootstrap, or Wilcoxon computation was performed.",
        "All paper-reference values in Table 4 were verified against DPLM-2.1 "
        "Table 4 (https://arxiv.org/html/2504.11454v3) and are reported only as "
        "external paper references, not local measurements.",
        "",
        "FINAL_METHOD_COMPARISON_PREP_PASS",
        "",
    ])
    return "\n".join(lines)


def atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    local, improvements, fm_nfe = load_local()
    fm, fd, bd = load_statistics()
    paper = load_paper_reference()
    local_md, local_tex = render_local(local, fm_nfe)
    paper_md, paper_tex = render_paper(paper)
    stats_md, stats_tex = render_statistics(fm, fd, bd)
    outputs = {
        "table3_local_matched_methods.md": local_md,
        "table3_local_matched_methods.tex": local_tex,
        "table4_paper_reference.md": paper_md,
        "table4_paper_reference.tex": paper_tex,
        "local_method_comparison_statistics.md": stats_md,
        "local_method_comparison_statistics.tex": stats_tex,
        "final_method_comparison_notes.txt": render_notes(local, improvements, fm),
    }
    for filename, contents in outputs.items():
        atomic_write(OUTPUT / filename, contents)
    print("FINAL_METHOD_COMPARISON_PREP_PASS")
    for filename in outputs:
        print(OUTPUT / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

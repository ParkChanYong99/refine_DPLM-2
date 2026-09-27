#!/usr/bin/env python3
"""Create a validated per-protein PDB-date comparison from evaluator runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.latent_residual_fm.compare_cameo_bit_vs_fm import (
    PAIRED_COLUMNS,
    SUMMARY_COLUMNS,
    AggregateMetrics,
    SampleMetrics,
    atomic_write_csv,
    collect_top_samples,
    describe_extremes,
    make_summary,
    read_aggregate,
    require,
    validate_aggregate,
)


EXPECTED_NUM_SAMPLES = 442
DEFAULT_BIT_EVAL_DIR = Path(
    "generation-results/dplm2_bit_650m_final/PDB_date/bit_only/folding/"
    "forward_folding/run_2026-09-06_14-08-38/folding/eval"
)
DEFAULT_FM_EVAL_DIR = Path(
    "generation-results/dplm2_bit_650m_final/PDB_date/"
    "latent_fm_alpha050/folding/forward_folding/"
    "run_2026-09-06_14-32-37/folding/eval"
)
OUTPUT_DIR = Path(
    "generation-results/dplm2_bit_650m_final/PDB_date/comparison"
)
PAIRED_CSV = OUTPUT_DIR / "pdb_date_bit_vs_fm_paired.csv"
SUMMARY_CSV = OUTPUT_DIR / "pdb_date_bit_vs_fm_summary.csv"

PAPER_BIT_RMSD = 3.2213
PAPER_BIT_TMSCORE = 0.9043
PAPER_RESDIFF_RMSD = 3.0168
PAPER_RESDIFF_TMSCORE = 0.9076


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bit_eval_dir", type=Path, default=DEFAULT_BIT_EVAL_DIR)
    parser.add_argument("--fm_eval_dir", type=Path, default=DEFAULT_FM_EVAL_DIR)
    return parser.parse_args()


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
        f"expected exactly {EXPECTED_NUM_SAMPLES} paired PDB-date samples, "
        f"got {len(bit_ids)}",
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


def apply_pdb_date_paper_reference(summary: dict[str, Any]) -> None:
    summary.update(
        {
            "paper_reference_only_bit_rmsd": PAPER_BIT_RMSD,
            "paper_reference_only_bit_tmscore": PAPER_BIT_TMSCORE,
            "paper_reference_only_resdiff_rmsd": PAPER_RESDIFF_RMSD,
            "paper_reference_only_resdiff_tmscore": PAPER_RESDIFF_TMSCORE,
            "paper_reference_notice": (
                "PAPER REFERENCE ONLY; NOT LOCAL PAIRED RESULT"
            ),
        }
    )


def print_summary(
    summary: Mapping[str, Any], paired: Sequence[Mapping[str, Any]]
) -> None:
    print("\n[PDB-DATE PAIRED COMPARISON]\n")
    print(f"num paired samples: {summary['num_samples']}")
    print(f"Bit mean BB RMSD: {float(summary['bit_mean_bb_rmsd']):.9g}")
    print(f"FM mean BB RMSD: {float(summary['fm_mean_bb_rmsd']):.9g}")
    print(
        "absolute RMSD improvement: "
        f"{float(summary['mean_rmsd_improvement']):.9g}"
    )
    print(
        "relative RMSD reduction: "
        f"{float(summary['relative_mean_rmsd_reduction_percent']):.9g}%"
    )
    print(
        "median RMSD improvement: "
        f"{float(summary['median_rmsd_improvement']):.9g}"
    )
    print(f"Bit mean TM-score: {float(summary['bit_mean_bb_tmscore']):.9g}")
    print(f"FM mean TM-score: {float(summary['fm_mean_bb_tmscore']):.9g}")
    print(
        "absolute TM-score improvement: "
        f"{float(summary['mean_tmscore_improvement']):.9g}"
    )
    print(
        "median TM-score improvement: "
        f"{float(summary['median_tmscore_improvement']):.9g}"
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
    print("\n[PAPER REFERENCE ONLY]")
    print("NOT LOCAL PAIRED RESULT")
    print(
        f"Paper Bit: RMSD {PAPER_BIT_RMSD:.4f}, "
        f"TM-score {PAPER_BIT_TMSCORE:.4f}"
    )
    print(
        "Paper Bit + RESDIFF: "
        f"RMSD {PAPER_RESDIFF_RMSD:.4f}, "
        f"TM-score {PAPER_RESDIFF_TMSCORE:.4f}"
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

    bit_aggregate: AggregateMetrics = read_aggregate(args.bit_eval_dir, "Bit")
    fm_aggregate: AggregateMetrics = read_aggregate(args.fm_eval_dir, "FM")
    validate_aggregate("Bit", bit_samples, bit_aggregate)
    validate_aggregate("FM", fm_samples, fm_aggregate)
    paired = make_paired_rows(bit_samples, fm_samples)
    summary = make_summary(paired, bit_aggregate, fm_aggregate)
    apply_pdb_date_paper_reference(summary)

    atomic_write_csv(PAIRED_CSV, PAIRED_COLUMNS, paired)
    atomic_write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, [summary])
    print_summary(summary, paired)
    print("\nPDB_DATE_BIT_VS_FM_PAIRED_COMPARISON_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

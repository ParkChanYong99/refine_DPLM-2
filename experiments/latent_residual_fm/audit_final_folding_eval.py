#!/usr/bin/env python3
"""Read-only audit of prerequisites for the final forward-folding evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


FINAL_ALPHA = 0.50
EXPECTED_OPTIMIZER_STEP = 100_000
EXPECTED_BEST_VAL_LOSS = 0.6716926411918394


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    directory: Path
    metadata_csv: Path
    metadata_data_dir: Path


@dataclass
class DatasetAudit:
    spec: DatasetSpec
    dataset_exists: bool
    aatype_exists: bool
    struct_exists: bool
    preprocessed_exists: bool
    num_sequences: int
    num_struct_sequences: int
    num_preprocessed_files: int
    num_metadata_matches: int
    num_gt_structures: int
    eligible_sequence_count: int
    ready: bool
    reasons: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument(
        "--fm_checkpoint",
        default=(
            "experiments/latent_residual_fm/runs/"
            "fm_modeA_current_main/checkpoint_best.pt"
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                require(current_id, f"empty FASTA header at {path}:{line_number}")
                require(current_id not in records, f"duplicate FASTA ID {current_id}: {path}")
                chunks = []
            else:
                require(current_id is not None, f"sequence before FASTA header: {path}")
                chunks.append(line)
    if current_id is not None:
        require(current_id not in records, f"duplicate FASTA ID {current_id}: {path}")
        records[current_id] = "".join(chunks)
    return records


def resolve_metadata_path(data_dir: Path, raw_path: str | None) -> Path | None:
    if raw_path is None or not str(raw_path).strip():
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else data_dir / path


def find_metadata_rows(
    metadata_csv: Path, target_ids: set[str]
) -> dict[str, dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    found: dict[str, dict[str, str]] = {}
    remaining = set(target_ids)
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require("pdb_name" in (reader.fieldnames or ()), "metadata lacks pdb_name")
        require(
            "processed_path" in (reader.fieldnames or ()),
            "metadata lacks processed_path",
        )
        for row in reader:
            pdb_name = row.get("pdb_name", "")
            if pdb_name in remaining:
                found[pdb_name] = row
                remaining.remove(pdb_name)
                if not remaining:
                    break
    return found


def audit_dataset(spec: DatasetSpec) -> DatasetAudit:
    aatype_path = spec.directory / "aatype.fasta"
    struct_path = spec.directory / "struct.fasta"
    preprocessed_dir = spec.directory / "preprocessed"
    dataset_exists = spec.directory.is_dir()
    aatype_exists = aatype_path.is_file()
    struct_exists = struct_path.is_file()
    preprocessed_exists = preprocessed_dir.is_dir()
    reasons: list[str] = []

    aa_records = read_fasta(aatype_path) if aatype_exists else {}
    struct_records = read_fasta(struct_path) if struct_exists else {}
    preprocessed_files = (
        list(preprocessed_dir.rglob("*.pkl")) if preprocessed_exists else []
    )
    metadata_rows = (
        find_metadata_rows(spec.metadata_csv, set(aa_records))
        if spec.metadata_csv.is_file() and aa_records
        else {}
    )

    resolved_gt = 0
    for sample_id, row in metadata_rows.items():
        processed = resolve_metadata_path(
            spec.metadata_data_dir, row.get("processed_path")
        )
        pdb_fallback = resolve_metadata_path(
            spec.metadata_data_dir, row.get("pdb_path")
        )
        if (processed is not None and processed.is_file()) or (
            pdb_fallback is not None and pdb_fallback.is_file()
        ):
            resolved_gt += 1

    eligible_count = sum(60 <= len(sequence) <= 512 for sequence in aa_records.values())
    if not dataset_exists:
        reasons.append("dataset directory missing")
    if not aatype_exists:
        reasons.append("aatype.fasta missing")
    if not struct_exists:
        reasons.append("struct.fasta missing")
    if not preprocessed_exists:
        reasons.append("preprocessed directory missing")
    if not spec.metadata_csv.is_file():
        reasons.append("metadata CSV missing")
    if not aa_records:
        reasons.append("aatype.fasta has no entries")
    if set(aa_records) != set(struct_records):
        reasons.append("aatype/struct FASTA ID sets differ")
    if len(metadata_rows) != len(aa_records):
        reasons.append("not every FASTA ID has an exact metadata pdb_name match")
    if resolved_gt != len(aa_records):
        reasons.append("not every metadata match resolves to an existing GT file")
    if len(preprocessed_files) < resolved_gt:
        reasons.append("preprocessed file count is smaller than resolved GT count")

    return DatasetAudit(
        spec=spec,
        dataset_exists=dataset_exists,
        aatype_exists=aatype_exists,
        struct_exists=struct_exists,
        preprocessed_exists=preprocessed_exists,
        num_sequences=len(aa_records),
        num_struct_sequences=len(struct_records),
        num_preprocessed_files=len(preprocessed_files),
        num_metadata_matches=len(metadata_rows),
        num_gt_structures=resolved_gt,
        eligible_sequence_count=eligible_count,
        ready=not reasons,
        reasons=reasons,
    )


def require_source_fragments(path: Path, fragments: tuple[str, ...]) -> None:
    source = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in source]
    require(not missing, f"source audit failed for {path}: missing {missing}")


def audit_sources(root: Path) -> None:
    readme = root / "README.md"
    generator = root / "generate_dplm2.py"
    evaluator = root / "src/byprot/utils/protein/evaluator_dplm2.py"
    eval_utils = root / "src/byprot/utils/protein/utils.py"
    superimposition = root / "vendor/openfold/openfold/utils/superimposition.py"
    forward_config = (
        root / "configs/experiment/structok/inference/forward_folding.yaml"
    )
    require_source_fragments(
        readme,
        (
            "task=folding",
            "--max_iter 100",
            "--unmasking_strategy deterministic",
            "--sampling_strategy argmax",
            "-cn forward_folding",
        ),
    )
    require_source_fragments(
        generator,
        (
            "if args.bit_model:",
            "DPLM2Bit.from_pretrained(args.model_name)",
            'parser.add_argument("--bit_model", action="store_true")',
            'save_dir=os.path.join(args.saveto, args.task)',
            'pdb_save_dir = os.path.join(save_dir, "pdb")',
        ),
    )
    require_source_fragments(
        forward_config,
        (
            "task: forward_folding",
            "csv_path: ${.data_dir}/metadata/pdb_afdb_cameo.csv",
        ),
    )
    require_source_fragments(
        evaluator,
        (
            'glob(os.path.join(pdb_folder, "*.pdb"))',
            "metadata_df[metadata_df.pdb_name == pdb_name].iloc[0]",
            'du.read_pkl(row.processed_path)',
            'load_from_pdb(row.pdb_path)',
            'if 60 <= len(feats["aatype"]) <= 512:',
            'true_bb_pos[..., :3, :].reshape(-1, 3)',
            '"Average bb_rmsd_to_gt": top_sample_csv.bb_rmsd_to_gt.mean()',
            '"Average bb_tmscore_to_gt": top_sample_csv.bb_tmscore_to_gt.mean()',
            'os.path.join(output_dir, "forward_fold_metrics.csv")',
        ),
    )
    require_source_fragments(
        eval_utils,
        (
            "from openfold.utils.superimposition import superimpose",
            "from tmtools import tm_align",
            "tm_results = tm_align",
        ),
    )
    require_source_fragments(
        superimposition,
        ("SVDSuperimposer", "Superimposes coordinates onto a reference"),
    )


def audit_checkpoint(path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    reasons: list[str] = []
    metadata: dict[str, Any] = {}
    if not path.is_file():
        return False, metadata, ["FM checkpoint missing"]
    try:
        checkpoint = torch.load(str(path), map_location="cpu", mmap=True)
        for key in ("optimizer_step", "best_step", "best_val_loss"):
            if key not in checkpoint:
                reasons.append(f"FM checkpoint missing {key}")
        if not reasons:
            metadata = {
                "optimizer_step": int(checkpoint["optimizer_step"]),
                "best_step": int(checkpoint["best_step"]),
                "best_val_loss": float(checkpoint["best_val_loss"]),
            }
            if metadata["optimizer_step"] != EXPECTED_OPTIMIZER_STEP:
                reasons.append("FM optimizer_step is not 100000")
            if metadata["best_step"] != EXPECTED_OPTIMIZER_STEP:
                reasons.append("FM best_step is not 100000")
            if not math.isclose(
                metadata["best_val_loss"],
                EXPECTED_BEST_VAL_LOSS,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                reasons.append("FM best_val_loss does not match expected value")
        del checkpoint
    except Exception as exc:
        reasons.append(f"FM checkpoint load failed: {type(exc).__name__}: {exc}")
    return not reasons, metadata, reasons


def print_dataset_audit(audit: DatasetAudit) -> None:
    skipped_by_length = audit.num_sequences - audit.eligible_sequence_count
    eligible_percent = (
        audit.eligible_sequence_count / audit.num_sequences * 100.0
        if audit.num_sequences
        else 0.0
    )
    print(f"[{audit.spec.label} AUDIT]")
    print(f"dataset exists: {'YES' if audit.dataset_exists else 'NO'}")
    print(f"total sequences: {audit.num_sequences}")
    print(f"evaluator eligible sequences (60 <= L <= 512): {audit.eligible_sequence_count}")
    print(f"evaluator skipped by length: {skipped_by_length}")
    print(f"eligible percent: {eligible_percent:.6f}")
    print(f"num struct FASTA entries: {audit.num_struct_sequences}")
    print(f"num preprocessed coordinate files: {audit.num_preprocessed_files}")
    print(f"num metadata exact-ID matches: {audit.num_metadata_matches}")
    print(f"num GT structures: {audit.num_gt_structures}")
    print(f"evaluator-length-eligible sequences: {audit.eligible_sequence_count}")
    print(
        "official evaluator GT resolution: exact metadata pdb_name match -> "
        f"{audit.spec.metadata_data_dir}/<processed_path>; on load exception, "
        "metadata pdb_path fallback when that column/path exists"
    )
    print(f"metadata CSV: {audit.spec.metadata_csv}")
    print(f"ready: {'YES' if audit.ready else 'NO'}")
    for reason in audit.reasons:
        print(f"reason: {reason}")
    print()


def print_official_route() -> None:
    print("[OFFICIAL FORWARD-FOLDING SETTINGS]")
    print("task: folding")
    print("max_iter: 100")
    print("unmasking_strategy: deterministic")
    print("sampling_strategy: argmax")
    print("Bit model CLI: --model_name airkingbd/dplm2_bit_650m --bit_model")
    print("Bit model API selection: DPLM2Bit.from_pretrained(args.model_name)")
    print()
    print("[OFFICIAL METRICS]")
    print("prediction PDB input: <input_fasta_dir>/pdb/*.pdb (copied into evaluator run tree)")
    print("GT lookup: metadata_df.pdb_name exact match to parsed prediction PDB name")
    print("GT load: processed_path pickle; broad-exception fallback to metadata pdb_path")
    print("RMSD atom basis: N/CA/C (atom37 first three atoms); CA RMSD is also computed")
    print("RMSD alignment: OpenFold superimpose -> Bio.SVDSuperimposer SVD alignment")
    print("TM-score: tmtools.tm_align; evaluator records tm_norm_chain2")
    print("aggregate: arithmetic mean for bb_rmsd_to_gt and bb_tmscore_to_gt; no median")
    print("failure handling: metadata/load errors abort; lengths outside 60..512 are skipped")
    print("special metric failures: listed hard-coded IDs receive RMSD=100 and TM-score=0")
    print("existing per-sample top_sample.csv causes that instance to be skipped")
    print("per-sample files: <eval>/length_<L>/<pdb_name>/{sc_results.csv,top_sample.csv}")
    print("aggregate CSV: <eval>/forward_fold_metrics.csv")
    print("JSON output: none in the inspected forward-folding evaluator route")
    print()


def print_layouts() -> None:
    print("[PROPOSED OFFICIAL-COMPATIBLE OUTPUT LAYOUT]")
    for dataset in ("cameo2022", "PDB_date"):
        for variant in ("bit_only", "latent_fm_alpha050"):
            root = Path("generation-results/dplm2_bit_650m_final") / dataset / variant
            print(f"prediction root (--saveto): {root}")
            print(f"  evaluator input_fasta_dir: {root / 'folding'}")
            print(f"  generated FASTA: {root / 'folding/struct_token.fasta'}")
            print(f"  generated PDB directory: {root / 'folding/pdb'}")
    print(
        "evaluator results: <input_fasta_dir>/forward_folding/<inference_subdir>/"
        "folding/eval/forward_fold_metrics.csv"
    )
    print(
        "PDB-date evaluator overrides required: "
        "inference.metadata.data_dir=data-bin/PDB_date "
        "inference.metadata.csv_path=data-bin/metadata/pdb_date.csv"
    )
    print()


def print_paper_references() -> None:
    print("[PAPER REFERENCES — NOT LOCAL REPRODUCTION RESULTS]")
    print("DPLM-2.1 Bit 650M CAMEO: RMSD 6.4028, TMscore 0.8380")
    print("DPLM-2.1 Bit 650M PDB-date: RMSD 3.2213, TMscore 0.9043")
    print("Bit + RESDIFF CAMEO: RMSD 6.1781, TMscore 0.8428")
    print("Bit + RESDIFF PDB-date: RMSD 3.0168, TMscore 0.9076")
    print()


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).resolve()
    audit_sources(root)

    cameo = audit_dataset(
        DatasetSpec(
            label="CAMEO",
            directory=root / "data-bin/cameo2022",
            metadata_csv=root / "data-bin/metadata/pdb_afdb_cameo.csv",
            metadata_data_dir=root / "data-bin",
        )
    )
    pdb_date = audit_dataset(
        DatasetSpec(
            label="PDB-DATE",
            directory=root / "data-bin/PDB_date",
            metadata_csv=root / "data-bin/metadata/pdb_date.csv",
            metadata_data_dir=root / "data-bin/PDB_date",
        )
    )
    checkpoint_path = root / args.fm_checkpoint
    checkpoint_ready, checkpoint_metadata, checkpoint_reasons = audit_checkpoint(
        checkpoint_path
    )

    print_dataset_audit(cameo)
    print_dataset_audit(pdb_date)
    print_official_route()
    print("[FM FINAL CONFIG]")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint loadable: {'YES' if checkpoint_ready else 'NO'}")
    print(f"optimizer_step: {checkpoint_metadata.get('optimizer_step', 'NOT CONFIRMED')}")
    print(f"best_step: {checkpoint_metadata.get('best_step', 'NOT CONFIRMED')}")
    print(f"best_val_loss: {checkpoint_metadata.get('best_val_loss', 'NOT CONFIRMED')}")
    print(f"alpha: {FINAL_ALPHA:.2f}")
    print("inference definition: z_refined = predicted_z_quant + 0.50 * r_hat")
    for reason in checkpoint_reasons:
        print(f"reason: {reason}")
    print()
    print_paper_references()
    print_layouts()

    all_reasons = cameo.reasons + pdb_date.reasons + checkpoint_reasons
    ready = cameo.ready and pdb_date.ready and checkpoint_ready
    print("[FINAL BENCHMARK READINESS]")
    print(f"CAMEO evaluation population: {cameo.eligible_sequence_count}")
    print(f"PDB-date evaluation population: {pdb_date.eligible_sequence_count}")
    if ready:
        print("READY")
        print("\nFINAL_FOLDING_EVAL_AUDIT_PASS")
        return 0
    print("NOT_READY: " + "; ".join(all_reasons))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

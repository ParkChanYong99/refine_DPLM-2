#!/usr/bin/env python3
"""Read-only metric-path diagnostic for finalized CAMEO Bit target 7dz2_C.

No generation, DDPM sampling, tokenizer decode, checkpoint load, or file write
occurs here. Run from the repository root in the dplm environment.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from byprot.datamodules.pdb_dataset import utils as du
from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset
from byprot.utils.protein import utils as evaluator_utils
from byprot.utils.protein.evaluator_dplm2 import load_pdb_by_name
from experiments.latent_residual_ddpm import evaluate_final_ddpm as current
from experiments.latent_residual_fm.generate_final_folding_predictions import read_targets


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ID = "7dz2_C"
TARGET_LENGTH = 284
OLD_EVAL_DIR = Path(
    "generation-results/dplm2_bit_650m_final/cameo2022/bit_only/folding/"
    "forward_folding/run_2026-09-03_09-08-54/folding/eval/length_284/7dz2_C"
)
REFERENCE_CSV = OLD_EVAL_DIR / "top_sample.csv"
OLD_PREDICTION = OLD_EVAL_DIR / "sample.pdb"
OBSERVED_NEW_CA = 3.06991910934
KNOWN_REFERENCE = (3.145063638687134, 3.0601673126220703, 0.9235105897587298)
CLASSIFICATION_ATOL = 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_reference() -> tuple[Path, tuple[float, float, float]]:
    require(REFERENCE_CSV.is_file(), f"reference CSV missing: {REFERENCE_CSV}")
    with REFERENCE_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_path", "folded_path", "length", "ca_rmsd_to_gt",
            "bb_rmsd_to_gt", "bb_tmscore_to_gt",
        }
        require(required.issubset(reader.fieldnames or ()),
                "reference CSV metric/path columns missing")
        rows = list(reader)
    require(len(rows) == 1, "expected exactly one 7dz2_C reference row")
    row = rows[0]
    require(int(row["length"]) == TARGET_LENGTH, "reference length changed")
    prediction = Path(row["sample_path"])
    require(prediction == OLD_PREDICTION,
            f"reference sample_path differs from specified PDB: {prediction}")
    require(Path(row["folded_path"]) == prediction,
            "reference folded_path differs from sample_path")
    require(prediction.is_file(), f"finalized Bit sample PDB missing: {prediction}")
    reference = (
        float(row["ca_rmsd_to_gt"]),
        float(row["bb_rmsd_to_gt"]),
        float(row["bb_tmscore_to_gt"]),
    )
    require(all(math.isfinite(value) for value in reference),
            "reference CSV contains non-finite metric")
    require(all(math.isclose(value, known, rel_tol=0.0, abs_tol=1e-12)
                for value, known in zip(reference, KNOWN_REFERENCE)),
            "7dz2_C reference CSV values differ from the reported artifact")
    return prediction, reference


def resolve_official_gt() -> tuple[object, Path, str, object]:
    # This is the exact metadata loader/path used by current.metric_triplet().
    metadata = current.load_official_metadata("cameo2022")
    rows = metadata.loc[metadata.pdb_name == TARGET_ID]
    require(len(rows) == 1, f"official metadata must contain one {TARGET_ID} row")
    row = rows.iloc[0]
    gt = load_pdb_by_name(TARGET_ID, metadata)
    processed_path = Path(row["processed_path"])
    # load_pdb_by_name tries processed_path first and falls back to pdb_path
    # only if reading/processing the pickle raises. Mirror that choice solely
    # to report which GT path its official call actually selected.
    try:
        processed = PdbDataset.process_chain(du.read_pkl(str(processed_path)))
    except Exception as processed_error:
        fallback = row["pdb_path"]
        require(isinstance(fallback, str) and bool(fallback),
                f"processed GT failed and no pdb_path fallback exists: {processed_error}")
        gt_path = Path(fallback)
        route = f"pdb_path fallback ({type(processed_error).__name__})"
    else:
        require(torch.equal(processed["all_atom_positions"], gt["all_atom_positions"]),
                "processed GT does not match official load_pdb_by_name output")
        gt_path = processed_path
        route = "processed_path pickle"
    require(gt_path.is_file(), f"official GT source missing: {gt_path}")
    require(tuple(gt["all_atom_positions"].shape) == (TARGET_LENGTH, 37, 3),
            "official GT atom37 shape/length changed")
    require(len(gt["aatype"]) == TARGET_LENGTH, "official GT residue count changed")
    return gt, gt_path, route, metadata


def finalized_helper_metrics(prediction: Path, gt: object) -> tuple[float, float, float]:
    # evaluator_dplm2.py: run_evaluation() builds true_bb_pos from
    # all_atom_positions_gt[..., :3, :] and compute_sample_metrics() calls
    # eu.process_folded_outputs(sample_path, folded_output, true_bb_pos).
    # In this reference row folded_path == sample_path, so folded_output=None
    # is the official no-self-consistency/placeholder branch for the same PDB.
    true_bb = gt["all_atom_positions"][..., :3, :].reshape(-1, 3).cpu().numpy()
    frame = evaluator_utils.process_folded_outputs(str(prediction), None, true_bb)
    require(len(frame) == 1, "legacy helper returned multiple metric rows")
    row = frame.iloc[0]
    result = (
        float(row["ca_rmsd_to_gt"]),
        float(row["bb_rmsd_to_gt"]),
        float(row["bb_tmscore_to_gt"]),
    )
    require(all(math.isfinite(value) for value in result),
            "legacy helper returned non-finite metric")
    return result


def classify(current_ca: float, reference_ca: float) -> str:
    if math.isclose(current_ca, OBSERVED_NEW_CA, rel_tol=0.0,
                    abs_tol=CLASSIFICATION_ATOL):
        return "CASE A: existing PDB follows the new value; metric/GT path mismatch"
    if math.isclose(current_ca, reference_ca, rel_tol=0.0,
                    abs_tol=CLASSIFICATION_ATOL):
        return "CASE B: existing PDB matches reference; new Bit coordinates differ"
    return "CASE C: neither value; inspect GT path, masks and residue alignment"


def main() -> int:
    require(Path.cwd().resolve() == REPO_ROOT, "run from /mnt/ext-vol/dplm")
    prediction, reference = read_reference()
    gt, gt_path, gt_route, metadata = resolve_official_gt()
    targets = [target for target in read_targets(current.DATASET_FASTAS["cameo2022"])
               if target.sample_id == TARGET_ID]
    require(len(targets) == 1 and targets[0].length == TARGET_LENGTH,
            "finalized FASTA target/length mismatch")
    predicted = du.parse_pdb_feats("sample", str(prediction))
    require(len(predicted["aatype"]) == TARGET_LENGTH,
            "legacy sample PDB parsed residue count differs")

    # This is precisely the function used by evaluate_final_ddpm.py.
    with torch.inference_mode():
        current_values = current.metric_triplet(prediction, targets[0], metadata)
    # This is the exact metric helper called by the legacy official evaluator;
    # rerunning the full evaluator would write files, so call its helper only.
    legacy_values = finalized_helper_metrics(prediction, gt)

    gt_ca = gt["all_atom_positions"][:, 1, :]
    gt_ca_nonzero = int((gt_ca.abs().sum(-1) > 1e-7).sum())
    prediction_ca_present = int(predicted["atom_mask"][:, 1].sum())
    gt_bb_complete = int(gt["all_atom_mask"][:, :3].bool().all(-1).sum())
    prediction_bb_complete = int(predicted["atom_mask"][:, :3].astype(bool).all(-1).sum())

    print("[FINALIZED BIT METRIC CONSISTENCY: 7dz2_C]")
    print(f"metadata CSV: {current.GT_METADATA['cameo2022'][0]}")
    print(f"GT path: {gt_path}")
    print(f"GT resolution route: {gt_route} via load_pdb_by_name")
    print(f"prediction path: {prediction}")
    print(f"sequence length: {targets[0].length}")
    print(f"GT residue count: {len(gt['aatype'])}")
    print(f"prediction residue count: {len(predicted['aatype'])}")
    print(f"current-helper CA RMSD: {current_values[0]:.15g}")
    print(f"current-helper backbone RMSD: {current_values[1]:.15g}")
    print(f"current-helper TM-score: {current_values[2]:.15g}")
    print(f"legacy/finalized CA RMSD: {legacy_values[0]:.15g}")
    print(f"legacy/finalized backbone RMSD: {legacy_values[1]:.15g}")
    print(f"legacy/finalized TM-score: {legacy_values[2]:.15g}")
    print(f"reference CSV CA RMSD: {reference[0]:.15g}")
    print(f"reference CSV backbone RMSD: {reference[1]:.15g}")
    print(f"reference CSV TM-score: {reference[2]:.15g}")
    print(f"current vs reference CA delta: {current_values[0] - reference[0]:+.15g}")
    print(f"legacy vs reference CA delta: {legacy_values[0] - reference[0]:+.15g}")
    print("CA selection: atom37 CA index 1; prediction parse_pdb_feats bb_positions")
    print("backbone selection: atom37 indices 0,1,2 (N,CA,C)")
    print("residue mask: GT CA coordinate abs-sum > 1e-7; not GT all_atom_mask")
    print("missing atom handling: PDB parser zeroes missing atoms; helper masks by GT CA only")
    print("Kabsch: OpenFold superimpose on GT-CA-valid residues; CA or flattened N/CA/C")
    print("TM-score: tmtools on GT-CA-valid N/CA/C triplets")
    print("chain filtering: parse_pdb_feats default chain_id='A' for sample.pdb")
    print(f"GT CA-valid residues: {gt_ca_nonzero}")
    print(f"prediction CA-present residues: {prediction_ca_present}")
    print(f"GT complete N/CA/C residues: {gt_bb_complete}")
    print(f"prediction complete N/CA/C residues: {prediction_bb_complete}")
    print(f"classification: {classify(current_values[0], reference[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

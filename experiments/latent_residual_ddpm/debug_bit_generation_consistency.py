#!/usr/bin/env python3
"""Diagnose 7dz2_C Bit token generation versus continuous-latent decode.

The only generation is one frozen DPLM folding call for this target. No DDPM
sampling or final benchmark loop occurs. Two temporary PDBs are written under
/tmp solely to reuse the official metric helper; existing artifacts are read
but never changed.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile

import torch
from openfold.utils.superimposition import superimpose

from byprot.datamodules.pdb_dataset import utils as du
from experiments.latent_residual_ddpm import evaluate_final_ddpm as final
from experiments.latent_residual_ddpm.debug_bit_metric_consistency import (
    read_reference,
)
from experiments.latent_residual_fm.generate_final_folding_predictions import (
    output_layout,
    read_manifest,
    read_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "cameo2022"
TARGET_ID = "7dz2_C"
EXPECTED_LENGTH = 284
EXPECTED_FASTA_INDEX = 0
EXPECTED_SEED = 42
METRIC_REPRODUCTION_ATOL = 1e-3  # Diagnostic only; does not change final-evaluator assertions.
COORDINATE_REPRODUCTION_RMS_MAX = 0.02  # Å, after common-mask Kabsch alignment.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(indices: list[int]) -> str:
    canonical = ",".join(str(value) for value in indices).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def describe_tokens(label: str, indices: list[int]) -> None:
    require(len(indices) == EXPECTED_LENGTH, f"{label} token count differs")
    print(f"{label} token count: {len(indices)}")
    print(f"{label} first 20: {indices[:20]}")
    print(f"{label} last 20: {indices[-20:]}")
    print(f"{label} min/max: {min(indices)} / {max(indices)}")
    print(f"{label} SHA256 (canonical raw LFQ indices): {digest(indices)}")


def load_target_and_manifest() -> tuple[object, dict[str, str], list[str]]:
    targets = read_targets(final.DATASET_FASTAS[DATASET])
    require(targets[EXPECTED_FASTA_INDEX].sample_id == TARGET_ID,
            "7dz2_C is not FASTA index zero")
    target = targets[EXPECTED_FASTA_INDEX]
    require(target.length == EXPECTED_LENGTH, "7dz2_C FASTA length differs")
    manifest_path = output_layout(DATASET).manifest
    rows = read_manifest(manifest_path)
    require(EXPECTED_FASTA_INDEX in rows, "7dz2_C missing from finalized manifest")
    row = rows[EXPECTED_FASTA_INDEX]
    require(row["sample_id"] == target.sample_id and
            int(row["fasta_index"]) == target.fasta_index and
            int(row["length"]) == target.length,
            "manifest target identity/length differs from FASTA")
    require(row["aa_sequence"] == target.sequence,
            "manifest AA sequence differs from final-evaluator FASTA input")
    require(int(row["sample_seed"]) == EXPECTED_SEED and
            int(row["base_seed"]) == final.BASE_SEED,
            "manifest seed differs from current final evaluator")
    require(row["dplm_checkpoint"] == final.DPLM_CHECKPOINT and
            int(row["dplm_max_iter"]) == final.DPLM_MAX_ITER and
            row["unmasking_strategy"] == "deterministic" and
            row["sampling_strategy"] == "argmax",
            "manifest generation settings differ from final evaluator")
    require(row["status"] == "success" and
            row["generation_success"].lower() == "true",
            "finalized manifest generation was not successful")
    lexemes = row["struct_token_sequence"].split(",")
    require(len(lexemes) == EXPECTED_LENGTH and
            all(token.isdigit() for token in lexemes),
            "manifest struct_token_sequence is not L numeric vocabulary lexemes")
    return target, row, lexemes


def manifest_combined_ids(model: torch.nn.Module, lexemes: list[str]) -> list[int]:
    # Finalized FM source stores batch_decode(structure_half).split() strings.
    # DPLM2Bit.prepare_for_struct_tokenizer subtracts struct_vocab_offset
    # before LFQ get_codebook_entry. Confirm the vocab lexemes represent those
    # raw LFQ indices rather than guessing whether CSV integers are token IDs.
    offset = int(model.struct_vocab_offset)
    combined: list[int] = []
    for position, lexeme in enumerate(lexemes):
        require(lexeme in model.tokenizer._token_to_id,
                f"manifest lexeme absent from DPLM vocabulary at {position}: {lexeme}")
        token_id = int(model.tokenizer._token_to_id[lexeme])
        require(token_id - offset == int(lexeme),
                f"manifest lexeme is not raw LFQ index at {position}: "
                f"text={lexeme}, combined_id={token_id}, offset={offset}")
        combined.append(token_id)
    return combined


def aligned_coordinate_difference(
    pdb_coords: torch.Tensor,
    decoded_coords: torch.Tensor,
    common_atom_mask: torch.Tensor,
) -> tuple[float, float]:
    require(tuple(pdb_coords.shape) == tuple(decoded_coords.shape) ==
            (EXPECTED_LENGTH, 37, 3), "atom37 coordinate shapes differ")
    require(tuple(common_atom_mask.shape) == (EXPECTED_LENGTH, 37),
            "common atom mask shape differs")
    require(int(common_atom_mask.sum()) >= 3, "too few common atoms for Kabsch")
    reference = pdb_coords.reshape(1, -1, 3).float()
    candidate = decoded_coords.reshape(1, -1, 3).float()
    mask = common_atom_mask.reshape(1, -1)
    aligned, rmsd = superimpose(reference, candidate, mask)
    errors = (aligned[0] - reference[0])[mask[0]]
    return float(rmsd[0]), float(errors.abs().max())


def metrics_match(observed: tuple[float, float, float], reference: tuple[float, float, float]) -> bool:
    return all(math.isclose(value, expected, rel_tol=0.0,
                            abs_tol=METRIC_REPRODUCTION_ATOL)
               for value, expected in zip(observed, reference))


def classify(
    tokens_equal: bool,
    finalized_metrics: tuple[float, float, float],
    reference_metrics: tuple[float, float, float],
    finalized_coordinate_rms: float,
) -> str:
    reproduced = (
        metrics_match(finalized_metrics, reference_metrics)
        and finalized_coordinate_rms <= COORDINATE_REPRODUCTION_RMS_MAX
    )
    if not tokens_equal and reproduced:
        return "CASE G1: generation token mismatch; finalized-token decode reproduces Bit PDB"
    if tokens_equal and finalized_coordinate_rms > COORDINATE_REPRODUCTION_RMS_MAX:
        return "CASE G2: tokens match; decode/PDB behavior differs from finalized Bit"
    if not tokens_equal and not reproduced:
        return "CASE G3: tokens differ and finalized-token decode does not reproduce Bit PDB"
    return "CASE G4: other mismatch or no mismatch; inspect detailed diagnostics"


def main() -> int:
    require(Path.cwd().resolve() == REPO_ROOT, "run from /mnt/ext-vol/dplm")
    target, manifest, lexemes = load_target_and_manifest()
    existing_bit_pdb, reference_metrics = read_reference()
    require(existing_bit_pdb.is_file(), "finalized Bit sample.pdb is missing")
    require(final.FINAL_ALPHA == 0.50 and final.BASE_SEED == EXPECTED_SEED,
            "frozen final evaluator settings changed")
    device = final.resolve_device("cuda")
    final.seed_everything(final.BASE_SEED)
    checks: dict[str, bool] = {}
    model, tokenizer = final.load_models(final.DPLM_CHECKPOINT, device, checks)
    # The final evaluator loads DDPM after DPLM and before its first target.
    # Match this RNG/loading order, but do not call the DDPM sampler or forward.
    ddpm, schedule, checkpoint_info = final.load_ddpm(final.DDPM_CHECKPOINT, device)
    require(not ddpm.training and schedule.num_timesteps == 1000 and
            int(checkpoint_info["optimizer_step"]) == 100000,
            "DDPM load-order checkpoint context differs")

    with torch.inference_mode():
        aatype = torch.tensor(du.seq_to_aatype(target.sequence), dtype=torch.long)
        require(tuple(aatype.shape) == (EXPECTED_LENGTH,),
                "current evaluator AA input length differs")
        input_tokens, partial_mask = final.build_folding_input(model, aatype, device)
        generated, _ = final.timed_call(
            device,
            lambda: model.generate(
                input_tokens=input_tokens,
                max_iter=final.DPLM_MAX_ITER,
                temperature=1.0,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
            ),
        )
        require("output_tokens" in generated, "generation returned no final tokens")
        final_tokens = generated["output_tokens"]
        require(tuple(final_tokens.shape) == (1, 2 * (EXPECTED_LENGTH + 2)),
                "current output_tokens shape differs")
        struct_half, aa_half = final_tokens.chunk(2, dim=1)
        current_combined = struct_half[0, 1:-1].long().tolist()
        offset = int(model.struct_vocab_offset)
        current_raw = [value - offset for value in current_combined]
        require(all(0 <= value < 8192 for value in current_raw),
                "current structure residue token outside LFQ codebook")

        manifest_combined = manifest_combined_ids(model, lexemes)
        finalized_raw = [value - offset for value in manifest_combined]
        require(len(finalized_raw) == len(current_raw) == EXPECTED_LENGTH,
                "manifest/current structure residue counts differ")
        manifest_final_tokens = final_tokens.clone()
        manifest_final_tokens[:, 1:EXPECTED_LENGTH + 1] = torch.tensor(
            manifest_combined, dtype=final_tokens.dtype, device=device
        ).view(1, -1)
        require(torch.equal(manifest_final_tokens[:, EXPECTED_LENGTH + 2:], aa_half),
                "manifest token replacement changed AA-side condition")
        finalized_z, finalized_mask = final.reconstruct_prediction_latent(
            model, manifest_final_tokens
        )
        current_z, current_mask = final.reconstruct_prediction_latent(model, final_tokens)
        expected_shape = (1, EXPECTED_LENGTH, 13)
        require(tuple(finalized_z.shape) == tuple(current_z.shape) == expected_shape,
                "z_quant rank-3 shape differs")
        require(torch.equal(finalized_mask, current_mask) and
                bool(current_mask.bool().all()),
                "manifest/current reconstructed residue masks differ")
        require(bool(torch.isfinite(finalized_z).all()) and
                bool(torch.isfinite(current_z).all()),
                "reconstructed z_quant is non-finite")

        finalized_decoded = tokenizer.detokenize(finalized_z, res_mask=finalized_mask)
        current_decoded = tokenizer.detokenize(current_z, res_mask=current_mask)
        target_aatype = aatype.view(1, EXPECTED_LENGTH).to(device)
        finalized_output = final.finalize_decoder_output(
            "finalized-token Bit", finalized_decoded, target_aatype,
            finalized_mask, target.sample_id, target.length,
        )
        current_output = final.finalize_decoder_output(
            "current-token Bit", current_decoded, target_aatype,
            current_mask, target.sample_id, target.length,
        )

        with tempfile.TemporaryDirectory(prefix="bit-generation-consistency-") as temp_name:
            temp_root = Path(temp_name)
            finalized_dir = temp_root / "finalized_token"
            current_dir = temp_root / "current_token"
            finalized_dir.mkdir()
            current_dir.mkdir()
            tokenizer.output_to_pdb(dict(finalized_output), output_dir=str(finalized_dir))
            tokenizer.output_to_pdb(dict(current_output), output_dir=str(current_dir))
            finalized_pdb = finalized_dir / f"{TARGET_ID}.pdb"
            current_pdb = current_dir / f"{TARGET_ID}.pdb"
            require(finalized_pdb.is_file() and current_pdb.is_file(),
                    "temporary decoder PDB missing")
            gt_metadata = final.load_official_metadata(DATASET)
            finalized_metrics = final.metric_triplet(finalized_pdb, target, gt_metadata)
            current_metrics = final.metric_triplet(current_pdb, target, gt_metadata)

    existing = du.parse_pdb_feats("existing", str(existing_bit_pdb))
    pdb_coords = torch.as_tensor(existing["atom_positions"])
    pdb_mask = torch.as_tensor(existing["atom_mask"]).bool()
    finalized_coords = finalized_output["atom37_positions"][0].detach().cpu()
    current_coords = current_output["atom37_positions"][0].detach().cpu()
    finalized_atom_mask = finalized_output["atom37_mask"][0].detach().cpu().bool()
    current_atom_mask = current_output["atom37_mask"][0].detach().cpu().bool()
    common_mask = pdb_mask & finalized_atom_mask & current_atom_mask
    finalized_rms, finalized_max = aligned_coordinate_difference(
        pdb_coords, finalized_coords, common_mask
    )
    current_rms, current_max = aligned_coordinate_difference(
        pdb_coords, current_coords, common_mask
    )

    different = [index for index, (old, now) in enumerate(zip(finalized_raw, current_raw))
                 if old != now]
    z_different = torch.any(finalized_z != current_z, dim=-1)[0]
    z_abs_max = float((finalized_z - current_z).abs().max())
    token_equal = len(different) == 0
    print("[BIT GENERATION CONSISTENCY DEBUG]")
    print(f"finalized manifest: {output_layout(DATASET).manifest}")
    print(f"target_id: {target.sample_id}; length: {target.length}; "
          f"FASTA index: {target.fasta_index}; seed: {manifest['sample_seed']}")
    print(f"max_iter: {manifest['dplm_max_iter']}; "
          f"unmasking: {manifest['unmasking_strategy']}; "
          f"sampling: {manifest['sampling_strategy']}")
    print(f"AA sequence: {manifest['aa_sequence']}")
    print("AA exact match: True (manifest == FASTA == current evaluator input)")
    print("generation config exact match: True")
    print("manifest token field: struct_token_sequence = batch-decoded structure-half "
          "vocabulary lexemes, not combined-vocabulary IDs")
    print(f"token representation: numeric lexemes map to raw LFQ indices; "
          f"combined ID = raw index + model.struct_vocab_offset ({offset})")
    describe_tokens("finalized", finalized_raw)
    describe_tokens("current", current_raw)
    print(f"same token count: {len(finalized_raw) == len(current_raw)}")
    print(f"token exact match: {token_equal}")
    print(f"different token positions: {len(different)}")
    print(f"mismatch fraction: {len(different) / EXPECTED_LENGTH:.9g}")
    for index in different[:20]:
        print(f"mismatch residue_index_0_based={index} "
              f"finalized={finalized_raw[index]} current={current_raw[index]}")
    print(f"z_quant shapes: {tuple(finalized_z.shape)}, {tuple(current_z.shape)}")
    print(f"z_quant exact: {torch.equal(finalized_z, current_z)}")
    print(f"z_quant allclose (1e-6): "
          f"{torch.allclose(finalized_z, current_z, rtol=1e-6, atol=1e-6)}")
    print(f"z_quant differing residues: {int(z_different.sum())}")
    print(f"z_quant max absolute difference: {z_abs_max:.9g}")
    print(f"existing finalized Bit PDB: {existing_bit_pdb}")
    print(f"common atom37 comparison atoms: {int(common_mask.sum())}")
    print(f"finalized-token decode vs finalized PDB: "
          f"aligned coordinate RMS={finalized_rms:.9g} Å, max abs={finalized_max:.9g} Å")
    print(f"current-token decode vs finalized PDB: "
          f"aligned coordinate RMS={current_rms:.9g} Å, max abs={current_max:.9g} Å")
    print(f"reference metrics CA/BB/TM: {reference_metrics}")
    print(f"finalized-token metrics CA/BB/TM: {finalized_metrics}")
    print(f"current-token metrics CA/BB/TM: {current_metrics}")
    print("classification: " + classify(
        token_equal, finalized_metrics, reference_metrics, finalized_rms
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

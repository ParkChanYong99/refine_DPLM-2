#!/usr/bin/env python3
"""Generate paired Bit-only and fixed-alpha latent-FM folding predictions."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from Bio import SeqIO

from byprot.datamodules.pdb_dataset import protein as protein_utils
from byprot.datamodules.pdb_dataset import utils as du
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    assert_atom37,
    build_folding_input,
    extract_final_token_condition,
    load_fm,
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.flow_matching import euler_sample


FINAL_ALPHA = 0.50
DPLM_CHECKPOINT = "airkingbd/dplm2_bit_650m"
FM_CHECKPOINT = Path(
    "experiments/latent_residual_fm/runs/"
    "fm_modeA_current_main/checkpoint_best.pt"
)
EXPECTED_OPTIMIZER_STEP = 100_000
DPLM_MAX_ITER = 100
DEFAULT_EULER_STEPS = 100
MIN_LENGTH = 60
MAX_LENGTH = 512
OUTPUT_ROOT = Path("generation-results/dplm2_bit_650m_final")
DATASET_FASTAS = {
    "cameo2022": Path("data-bin/cameo2022/aatype.fasta"),
    "PDB_date": Path("data-bin/PDB_date/aatype.fasta"),
}
MANIFEST_COLUMNS = (
    "fasta_index",
    "sample_id",
    "length",
    "eligible",
    "sample_seed",
    "bit_pdb_path",
    "fm_pdb_path",
    "generation_success",
    "r_hat_abs_mean",
    "r_hat_abs_max",
    "status",
    "error",
    "base_seed",
    "num_flow_steps",
    "alpha",
    "dplm_checkpoint",
    "fm_checkpoint",
    "dplm_max_iter",
    "unmasking_strategy",
    "sampling_strategy",
    "struct_token_sequence",
    "aa_sequence",
)
RESUME_SETTING_COLUMNS = (
    "base_seed",
    "num_flow_steps",
    "alpha",
    "dplm_checkpoint",
    "fm_checkpoint",
    "dplm_max_iter",
    "unmasking_strategy",
    "sampling_strategy",
)


@dataclass(frozen=True)
class Target:
    fasta_index: int
    sample_id: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def eligible(self) -> bool:
        return MIN_LENGTH <= self.length <= MAX_LENGTH


@dataclass(frozen=True)
class OutputLayout:
    dataset_root: Path
    bit_folding: Path
    fm_folding: Path
    bit_pdb_dir: Path
    fm_pdb_dir: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_FASTAS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_flow_steps", type=int, default=DEFAULT_EULER_STEPS)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_layout(dataset: str) -> OutputLayout:
    root = OUTPUT_ROOT / dataset
    bit_folding = root / "bit_only" / "folding"
    fm_folding = root / "latent_fm_alpha050" / "folding"
    return OutputLayout(
        dataset_root=root,
        bit_folding=bit_folding,
        fm_folding=fm_folding,
        bit_pdb_dir=bit_folding / "pdb",
        fm_pdb_dir=fm_folding / "pdb",
        manifest=root / "generation_manifest.csv",
    )


def read_targets(path: Path) -> list[Target]:
    require(path.is_file(), f"input FASTA does not exist: {path}")
    targets: list[Target] = []
    seen: set[str] = set()
    for fasta_index, record in enumerate(SeqIO.parse(str(path), "fasta")):
        sample_id = record.name
        require(sample_id, f"empty FASTA ID at index {fasta_index}")
        require(sample_id not in seen, f"duplicate FASTA ID: {sample_id}")
        require(
            Path(sample_id).name == sample_id
            and "/" not in sample_id
            and "\\" not in sample_id,
            f"FASTA ID cannot be used as an exact PDB filename: {sample_id}",
        )
        sequence = str(record.seq)
        require(sequence, f"empty sequence for {sample_id}")
        seen.add(sample_id)
        targets.append(Target(fasta_index, sample_id, sequence))
    require(targets, f"input FASTA has no records: {path}")
    return targets


def settings(args: argparse.Namespace) -> dict[str, str]:
    return {
        "base_seed": str(args.seed),
        "num_flow_steps": str(args.num_flow_steps),
        "alpha": f"{FINAL_ALPHA:.2f}",
        "dplm_checkpoint": DPLM_CHECKPOINT,
        "fm_checkpoint": str(FM_CHECKPOINT),
        "dplm_max_iter": str(DPLM_MAX_ITER),
        "unmasking_strategy": "deterministic",
        "sampling_strategy": "argmax",
    }


def default_manifest_row(
    target: Target, layout: OutputLayout, args: argparse.Namespace
) -> dict[str, Any]:
    bit_path = layout.bit_pdb_dir / f"{target.sample_id}.pdb"
    fm_path = layout.fm_pdb_dir / f"{target.sample_id}.pdb"
    row: dict[str, Any] = {
        "fasta_index": target.fasta_index,
        "sample_id": target.sample_id,
        "length": target.length,
        "eligible": target.eligible,
        "sample_seed": args.seed + target.fasta_index,
        "bit_pdb_path": str(bit_path),
        "fm_pdb_path": str(fm_path),
        "generation_success": False,
        "r_hat_abs_mean": "",
        "r_hat_abs_max": "",
        "status": "pending" if target.eligible else "skipped_length",
        "error": "",
        "struct_token_sequence": "",
        "aa_sequence": target.sequence,
    }
    row.update(settings(args))
    return row


def read_manifest(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS).difference(reader.fieldnames or ())
        require(not missing, f"existing manifest lacks columns: {sorted(missing)}")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            index = int(row["fasta_index"])
            require(index not in rows, f"duplicate fasta_index in manifest: {index}")
            rows[index] = row
    return rows


def validate_existing_manifest(
    existing: Mapping[int, Mapping[str, str]],
    targets: Sequence[Target],
    expected_settings: Mapping[str, str],
) -> None:
    by_index = {target.fasta_index: target for target in targets}
    for index, row in existing.items():
        require(index in by_index, f"manifest fasta_index no longer exists: {index}")
        target = by_index[index]
        require(row["sample_id"] == target.sample_id, f"manifest ID mismatch at {index}")
        require(int(row["length"]) == target.length, f"manifest length mismatch at {index}")
        require(
            int(row["sample_seed"]) == int(expected_settings["base_seed"]) + index,
            f"resume sample seed mismatch for {target.sample_id}",
        )
        for key in RESUME_SETTING_COLUMNS:
            require(
                row[key] == expected_settings[key],
                f"resume setting mismatch for {target.sample_id}: {key} "
                f"was {row[key]!r}, current {expected_settings[key]!r}",
            )


def merge_manifest_rows(
    targets: Sequence[Target],
    existing: Mapping[int, Mapping[str, str]],
    layout: OutputLayout,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for target in targets:
        row = default_manifest_row(target, layout, args)
        if target.fasta_index in existing:
            row.update(existing[target.fasta_index])
        # Identity, paths, and fixed settings always come from the current verified input.
        row.update(default_manifest_row(target, layout, args) | settings(args))
        if target.fasta_index in existing:
            for key in (
                "generation_success",
                "r_hat_abs_mean",
                "r_hat_abs_max",
                "status",
                "error",
                "struct_token_sequence",
                "aa_sequence",
            ):
                row[key] = existing[target.fasta_index][key]
        if not target.eligible:
            row["generation_success"] = False
            row["status"] = "skipped_length"
            row["error"] = ""
        rows[target.fasta_index] = row
    return rows


def atomic_write_csv(path: Path, rows: Mapping[int, Mapping[str, Any]]) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for index in sorted(rows):
                writer.writerow(rows[index])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_fasta(path: Path, records: Sequence[tuple[str, str]]) -> None:
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
            for sample_id, sequence in records:
                handle.write(f">{sample_id}\n{sequence}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def valid_pdb(path: Path, expected_length: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        pdb_text = path.read_text(encoding="utf-8")
        if not any(line.startswith("ATOM") for line in pdb_text.splitlines()):
            return False
        parsed = protein_utils.from_pdb_string(pdb_text)
        positions = torch.as_tensor(parsed.atom_positions)
        atom_mask = torch.as_tensor(parsed.atom_mask)
        return (
            tuple(positions.shape) == (expected_length, 37, 3)
            and tuple(atom_mask.shape) == (expected_length, 37)
            and bool(torch.isfinite(positions).all())
            and bool((atom_mask.sum(dim=-1) > 0).all())
        )
    except Exception:
        return False


def row_has_complete_outputs(row: Mapping[str, Any], target: Target) -> bool:
    return (
        is_true(row["generation_success"])
        and bool(row["struct_token_sequence"])
        and row["aa_sequence"] == target.sequence
        and valid_pdb(Path(str(row["bit_pdb_path"])), target.length)
        and valid_pdb(Path(str(row["fm_pdb_path"])), target.length)
    )


def rebuild_output_fastas(
    rows: Mapping[int, Mapping[str, Any]], layout: OutputLayout
) -> None:
    successful = [
        row
        for index, row in sorted(rows.items())
        if is_true(row["generation_success"])
        and bool(row["struct_token_sequence"])
    ]
    struct_records = [
        (str(row["sample_id"]), str(row["struct_token_sequence"]))
        for row in successful
    ]
    aa_records = [
        (str(row["sample_id"]), str(row["aa_sequence"])) for row in successful
    ]
    # Both variants record the same DPLM discrete prediction.  The FM output is
    # a continuous-latent decode and is not represented as a new discrete token.
    for folding_dir in (layout.bit_folding, layout.fm_folding):
        atomic_write_fasta(folding_dir / "struct_token.fasta", struct_records)
        atomic_write_fasta(folding_dir / "aatype.fasta", aa_records)


def finalize_decoder_output(
    name: str,
    decoded: Mapping[str, torch.Tensor],
    target_aatype: torch.Tensor,
    residue_mask: torch.Tensor,
    sample_id: str,
    length: int,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(decoded)
    assert_atom37(name, output, length)
    require(tuple(output["atom37_mask"].shape) == (1, length, 37), f"{name} atom37 mask shape")
    require(bool(torch.isfinite(output["atom37_mask"]).all()), f"{name} atom37 mask non-finite")
    require(tuple(target_aatype.shape) == (1, length), "target aatype is misaligned")
    require(tuple(residue_mask.shape) == (1, length), "residue mask is misaligned")
    output["aatype"] = target_aatype
    output["atom37_mask"] = output["atom37_mask"] * residue_mask[..., None]
    output["header"] = [sample_id]
    return output


def save_pdb_pair_atomically(
    struct_tokenizer: torch.nn.Module,
    bit_output: Mapping[str, Any],
    fm_output: Mapping[str, Any],
    bit_path: Path,
    fm_path: Path,
    sample_id: str,
    length: int,
) -> None:
    bit_path.parent.mkdir(parents=True, exist_ok=True)
    fm_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bit-pdb-", dir=bit_path.parent) as bit_tmp, tempfile.TemporaryDirectory(
        prefix=".fm-pdb-", dir=fm_path.parent
    ) as fm_tmp:
        struct_tokenizer.output_to_pdb(dict(bit_output), output_dir=bit_tmp)
        struct_tokenizer.output_to_pdb(dict(fm_output), output_dir=fm_tmp)
        temporary_bit = Path(bit_tmp) / f"{sample_id}.pdb"
        temporary_fm = Path(fm_tmp) / f"{sample_id}.pdb"
        require(valid_pdb(temporary_bit, length), f"invalid temporary Bit PDB: {sample_id}")
        require(valid_pdb(temporary_fm, length), f"invalid temporary FM PDB: {sample_id}")
        os.replace(temporary_bit, bit_path)
        os.replace(temporary_fm, fm_path)
    require(valid_pdb(bit_path, length), f"invalid committed Bit PDB: {sample_id}")
    require(valid_pdb(fm_path, length), f"invalid committed FM PDB: {sample_id}")


@torch.inference_mode()
def generate_one(
    *,
    model: torch.nn.Module,
    struct_tokenizer: torch.nn.Module,
    fm: torch.nn.Module,
    target: Target,
    sample_seed: int,
    num_flow_steps: int,
    device: torch.device,
    bit_path: Path,
    fm_path: Path,
) -> dict[str, Any]:
    require(FINAL_ALPHA == 0.50, "FINAL_ALPHA must remain fixed at 0.50")
    try:
        aatype_list = du.seq_to_aatype(target.sequence)
    except ValueError as exc:
        raise AssertionError(f"unsupported amino acid in {target.sample_id}") from exc
    aatype = torch.tensor(aatype_list, dtype=torch.long)
    require(tuple(aatype.shape) == (target.length,), "aatype conversion changed length")

    input_tokens, partial_mask = build_folding_input(model, aatype, device)
    # Exactly one DPLM generation supplies both paired prediction branches.
    generated = model.generate(
        input_tokens=input_tokens,
        max_iter=DPLM_MAX_ITER,
        temperature=1.0,
        unmasking_strategy="deterministic",
        sampling_strategy="argmax",
        partial_masks=partial_mask,
    )
    require("output_tokens" in generated, "generation returned no output_tokens")
    final_tokens = generated["output_tokens"]
    require(
        tuple(final_tokens.shape) == (1, 2 * (target.length + 2)),
        f"final_tokens shape is {tuple(final_tokens.shape)}",
    )

    predicted_z_quant, residue_mask = reconstruct_prediction_latent(model, final_tokens)
    expected_latent_shape = (1, target.length, EXPECTED_CODEBOOK_DIM)
    require(
        tuple(predicted_z_quant.shape) == expected_latent_shape,
        f"predicted_z_quant shape is {tuple(predicted_z_quant.shape)}",
    )
    require(tuple(residue_mask.shape) == (1, target.length), "residue mask shape mismatch")
    require(bool(residue_mask.bool().all()), "generated output has masked-out residues")
    require(bool(torch.isfinite(predicted_z_quant).all()), "predicted_z_quant is non-finite")

    bit_decoded = struct_tokenizer.detokenize(predicted_z_quant, res_mask=residue_mask)

    # The helper performs exactly one extra frozen forward on final output_tokens,
    # drops the embedding state, and returns only aligned structure-side states.
    hidden_states = extract_final_token_condition(model, final_tokens, target.length)
    require(len(hidden_states) == EXPECTED_NUM_LAYERS, "expected 33 hidden states")
    for layer_index, hidden in enumerate(hidden_states):
        require(
            tuple(hidden.shape) == (1, target.length, EXPECTED_HIDDEN_SIZE),
            f"hidden layer {layer_index} shape is {tuple(hidden.shape)}",
        )

    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(sample_seed)
    initial_noise = torch.randn(
        predicted_z_quant.shape,
        dtype=predicted_z_quant.dtype,
        device=device,
        generator=noise_generator,
    )
    r_hat = euler_sample(
        fm,
        predicted_z_quant,
        hidden_states,
        residue_mask,
        num_steps=num_flow_steps,
        initial_noise=initial_noise,
    )
    require(tuple(r_hat.shape) == expected_latent_shape, f"r_hat shape is {tuple(r_hat.shape)}")
    require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")
    z_refined = predicted_z_quant + FINAL_ALPHA * r_hat
    require(
        torch.allclose(
            z_refined,
            predicted_z_quant + 0.50 * r_hat,
            atol=0.0,
            rtol=0.0,
        ),
        "z_refined does not equal predicted_z_quant + 0.50 * r_hat",
    )
    require(bool(torch.isfinite(z_refined).all()), "z_refined is non-finite")
    fm_decoded = struct_tokenizer.detokenize(z_refined, res_mask=residue_mask)

    target_aatype = aatype.view(1, target.length).to(device)
    bit_output = finalize_decoder_output(
        "Bit-only", bit_decoded, target_aatype, residue_mask, target.sample_id, target.length
    )
    fm_output = finalize_decoder_output(
        "FM-refined", fm_decoded, target_aatype, residue_mask, target.sample_id, target.length
    )
    save_pdb_pair_atomically(
        struct_tokenizer,
        bit_output,
        fm_output,
        bit_path,
        fm_path,
        target.sample_id,
        target.length,
    )

    struct_tokens, _ = final_tokens.chunk(2, dim=-1)
    decoded_struct = model.tokenizer.batch_decode(
        struct_tokens, skip_special_tokens=True
    )[0]
    discrete_tokens = decoded_struct.split()
    require(len(discrete_tokens) == target.length, "decoded structure-token length mismatch")
    return {
        "struct_token_sequence": ",".join(discrete_tokens),
        "r_hat_abs_mean": f"{r_hat.float().abs().mean().item():.9g}",
        "r_hat_abs_max": f"{r_hat.float().abs().max().item():.9g}",
    }


def main() -> int:
    args = parse_args()
    require(
        args.num_flow_steps == DEFAULT_EULER_STEPS,
        f"final generation requires {DEFAULT_EULER_STEPS} Euler steps",
    )
    require(args.max_samples is None or args.max_samples > 0, "--max_samples must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    require(FINAL_ALPHA == 0.50, "FINAL_ALPHA must remain fixed at 0.50")
    input_fasta = DATASET_FASTAS[args.dataset]
    targets = read_targets(input_fasta)
    eligible_targets = [target for target in targets if target.eligible]
    requested_targets = (
        eligible_targets
        if args.max_samples is None
        else eligible_targets[: args.max_samples]
    )
    layout = output_layout(args.dataset)
    layout.bit_pdb_dir.mkdir(parents=True, exist_ok=True)
    layout.fm_pdb_dir.mkdir(parents=True, exist_ok=True)

    expected_settings = settings(args)
    existing = read_manifest(layout.manifest)
    validate_existing_manifest(existing, targets, expected_settings)
    rows = merge_manifest_rows(targets, existing, layout, args)
    atomic_write_csv(layout.manifest, rows)

    checks: dict[str, bool] = {}
    model, struct_tokenizer = load_models(DPLM_CHECKPOINT, device, checks)
    dplm_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    require(dplm_trainable == 0 and not model.training, "DPLM is not fully frozen/eval")
    fm, optimizer_step, best_val_loss, best_step = load_fm(FM_CHECKPOINT, device)
    fm_trainable = sum(
        parameter.numel() for parameter in fm.parameters() if parameter.requires_grad
    )
    require(fm_trainable == 0 and not fm.training, "FM is not fully frozen/eval")
    require(optimizer_step == EXPECTED_OPTIMIZER_STEP, "FM checkpoint optimizer step mismatch")
    require(best_step == EXPECTED_OPTIMIZER_STEP, "FM checkpoint best step mismatch")

    generated_count = 0
    resume_skipped_count = 0
    failed_count = 0
    for ordinal, target in enumerate(requested_targets, start=1):
        row = rows[target.fasta_index]
        if not args.overwrite and row_has_complete_outputs(row, target):
            resume_skipped_count += 1
            print(
                f"[{ordinal}/{len(requested_targets)}] resume-skip "
                f"{target.sample_id} L={target.length}"
            )
            continue

        sample_seed = args.seed + target.fasta_index
        try:
            result = generate_one(
                model=model,
                struct_tokenizer=struct_tokenizer,
                fm=fm,
                target=target,
                sample_seed=sample_seed,
                num_flow_steps=args.num_flow_steps,
                device=device,
                bit_path=Path(str(row["bit_pdb_path"])),
                fm_path=Path(str(row["fm_pdb_path"])),
            )
            row.update(result)
            row["generation_success"] = True
            row["status"] = "success"
            row["error"] = ""
            row["aa_sequence"] = target.sequence
            atomic_write_csv(layout.manifest, rows)
            rebuild_output_fastas(rows, layout)
            generated_count += 1
            print(
                f"[{ordinal}/{len(requested_targets)}] generated "
                f"{target.sample_id} L={target.length} seed={sample_seed}"
            )
        except Exception as exc:
            row["generation_success"] = False
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            row["r_hat_abs_mean"] = ""
            row["r_hat_abs_max"] = ""
            row["struct_token_sequence"] = ""
            atomic_write_csv(layout.manifest, rows)
            failed_count += 1
            print(
                f"[{ordinal}/{len(requested_targets)}] FAILED "
                f"{target.sample_id}: {row['error']}"
            )
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rebuild_output_fastas(rows, layout)
    success_count = sum(
        row_has_complete_outputs(rows[target.fasta_index], target)
        for target in requested_targets
    )
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    else:
        peak_allocated = peak_reserved = 0.0

    print("\n[FINAL FOLDING PREDICTION GENERATION]")
    print(f"dataset: {args.dataset}")
    print(f"total FASTA samples: {len(targets)}")
    print(f"eligible samples: {len(eligible_targets)}")
    print(f"skipped by length: {len(targets) - len(eligible_targets)}")
    print(f"requested samples: {len(requested_targets)}")
    print(f"generated samples this run: {generated_count}")
    print(f"success count: {success_count}")
    print(f"resume-skipped count: {resume_skipped_count}")
    print(f"failed count: {failed_count}")
    print(f"Bit output directory: {layout.bit_pdb_dir}")
    print(f"FM output directory: {layout.fm_pdb_dir}")
    print(f"manifest path: {layout.manifest}")
    print(f"FM checkpoint: {FM_CHECKPOINT}")
    print(f"FM checkpoint optimizer_step: {optimizer_step}")
    print(f"FM checkpoint best_val_loss: {best_val_loss:.16g}")
    print(f"alpha: {FINAL_ALPHA:.2f}")
    print(f"Euler steps: {args.num_flow_steps}")
    print(f"DPLM max_iter: {DPLM_MAX_ITER}")
    print(f"DPLM trainable params: {dplm_trainable}")
    print(f"FM trainable params: {fm_trainable}")
    print(f"GPU peak allocated MiB: {peak_allocated:.2f}")
    print(f"GPU peak reserved MiB: {peak_reserved:.2f}")

    all_requested_succeeded = (
        success_count == len(requested_targets)
        and failed_count == 0
        and generated_count + resume_skipped_count == len(requested_targets)
    )
    if all_requested_succeeded:
        print("\nFINAL_FOLDING_PREDICTION_PASS")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

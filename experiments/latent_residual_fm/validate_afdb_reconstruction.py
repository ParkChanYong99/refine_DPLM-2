#!/usr/bin/env python3
"""Validate reconstruction of current tokenizer latents from official AFDB v4.

The script never downloads data.  ``--select_only`` only reads metadata and is
the intended first step for identifying archive members to extract externally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import shutil
import statistics
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SelectedSample:
    row_number: int
    processed_path: str
    basename: str
    archive_member: str
    local_path: Path
    seq_len: int
    aa_seq: str
    struct_ids: tuple[int, ...]
    plddt: np.ndarray


@dataclass
class SampleResult:
    sample: SelectedSample
    available: bool = False
    validated: bool = False
    passed: bool = False
    aa_exact: bool = False
    plddt_allclose: bool = False
    latent_pass: bool = False
    token_accuracy: float | None = None
    bit_accuracy: float | None = None
    mismatch_margins: list[float] = field(default_factory=list)
    residual_abs_mean: float | None = None
    residual_abs_max: float | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata_csv", default="data-bin/metadata/pdb_afdb_cameo.csv"
    )
    parser.add_argument("--pdb_gz_dir", default="/tmp/dplm_afdb_validate")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--select_only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def parse_int(value: str, field_name: str, row_number: int) -> int:
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError):
        raise ValueError(
            f"row {row_number}: invalid {field_name} value {value!r}"
        )


def comma_values(value: str, converter: Any) -> list[Any]:
    return [converter(item) for item in value.split(",") if item.strip()]


def make_sample(row: dict[str, str], row_number: int, pdb_gz_dir: Path) -> SelectedSample:
    processed_path = row["processed_path"].strip()
    filename = Path(processed_path).name
    if not filename.endswith(".pkl"):
        raise ValueError(
            f"row {row_number}: processed_path does not end in .pkl: {processed_path}"
        )
    basename = filename[: -len(".pkl")]
    expected_prefix = "AF-"
    expected_suffix = "-F1-model_v4"
    if not (basename.startswith(expected_prefix) and basename.endswith(expected_suffix)):
        raise ValueError(
            f"row {row_number}: unexpected AFDB v4 basename: {basename}"
        )
    archive_member = f"{basename}.pdb.gz"
    return SelectedSample(
        row_number=row_number,
        processed_path=processed_path,
        basename=basename,
        archive_member=archive_member,
        local_path=pdb_gz_dir / archive_member,
        seq_len=parse_int(row["seq_len"], "seq_len", row_number),
        aa_seq=row["aa_seq"].strip(),
        struct_ids=tuple(comma_values(row["struct_seq"], int)),
        plddt=np.asarray(comma_values(row["plddt"], float), dtype=np.float64),
    )


def select_samples(
    metadata_csv: Path,
    pdb_gz_dir: Path,
    num_samples: int,
    max_length: int,
    seed: int,
) -> tuple[list[SelectedSample], int]:
    """Deterministic reservoir sample without retaining the entire large CSV."""
    if num_samples < 0:
        raise ValueError("--num_samples must be non-negative")
    rng = random.Random(seed)
    reservoir: list[SelectedSample] = []
    candidate_count = 0
    required = {
        "split",
        "processed_path",
        "aa_seq",
        "struct_seq",
        "plddt",
        "seq_len",
    }
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata CSV missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if row.get("split") != "afdb_swissprot":
                continue
            if not all(
                nonempty(row.get(key))
                for key in ("processed_path", "aa_seq", "struct_seq", "plddt")
            ):
                continue
            try:
                seq_len = parse_int(row.get("seq_len", ""), "seq_len", row_number)
            except ValueError:
                continue
            if seq_len > max_length:
                continue
            sample = make_sample(row, row_number, pdb_gz_dir)
            candidate_count += 1
            if len(reservoir) < num_samples:
                reservoir.append(sample)
            elif num_samples:
                replacement = rng.randrange(candidate_count)
                if replacement < num_samples:
                    reservoir[replacement] = sample
    if candidate_count < num_samples:
        raise ValueError(
            f"requested {num_samples} samples but only {candidate_count} candidates matched"
        )
    # Reservoir slots give a deterministic selection, while sorting restores CSV order.
    reservoir.sort(key=lambda sample: sample.row_number)
    return reservoir, candidate_count


def print_selection(samples: list[SelectedSample]) -> None:
    print("=" * 60)
    print("SELECTED AFDB VALIDATION SAMPLES")
    print("=" * 60)
    for index, sample in enumerate(samples, start=1):
        print(f"\nsample {index}")
        print(f"processed_path: {sample.processed_path}")
        print(f"archive_member: {sample.archive_member}")
        print(f"local_path: {sample.local_path}")
        print(f"seq_len: {sample.seq_len}")
    print("\n[ARCHIVE MEMBERS]")
    for sample in samples:
        print(sample.archive_member)


def mismatch_count(left: str, right: str) -> int:
    shared = sum(a != b for a, b in zip(left, right))
    return shared + abs(len(left) - len(right))


def finite_scalar(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def format_float(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.9f}"


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def min_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return min(values) if values else None


def max_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return max(values) if values else None


def validate_one(sample: SelectedSample, tokenizer: Any, device: Any, verbose: bool) -> SampleResult:
    import torch

    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset

    result = SampleResult(sample=sample)
    print("\n" + "=" * 60)
    print(sample.basename)
    print("=" * 60)
    if not sample.local_path.is_file():
        print(f"STATUS: MISSING_PDB_GZ ({sample.local_path})")
        return result
    result.available = True

    try:
        with tempfile.TemporaryDirectory(prefix="dplm_afdb_validate_") as temp_dir:
            pdb_path = Path(temp_dir) / f"{sample.basename}.pdb"
            with gzip.open(sample.local_path, "rb") as source, pdb_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            raw_chain_feats, metadata = du.process_pdb_file(str(pdb_path))

        raw_aatype = np.asarray(raw_chain_feats["aatype"])
        raw_aa_seq = du.aatype_to_seq(raw_aatype)
        raw_plddt = np.asarray(raw_chain_feats["b_factors"], dtype=np.float64)[
            :, du.CA_IDX
        ]
        chain_feats = PdbDataset.process_chain(raw_chain_feats)
        result.validated = True

        aa_mismatches = mismatch_count(raw_aa_seq, sample.aa_seq)
        result.aa_exact = raw_aa_seq == sample.aa_seq
        print(f"raw length: {len(raw_aa_seq)}")
        print(f"CSV aa_seq length: {len(sample.aa_seq)}")
        print(f"AA exact match: {result.aa_exact}")
        print(f"AA mismatch count: {aa_mismatches}")

        lengths_match = (
            len(raw_aa_seq)
            == len(sample.aa_seq)
            == len(sample.struct_ids)
            == len(sample.plddt)
            == sample.seq_len
            == int(chain_feats["all_atom_positions"].shape[0])
        )
        if raw_plddt.shape == sample.plddt.shape:
            plddt_delta = np.abs(raw_plddt - sample.plddt)
            result.plddt_allclose = bool(
                np.allclose(raw_plddt, sample.plddt, atol=1e-3, rtol=0.0)
            )
            plddt_max = float(plddt_delta.max()) if plddt_delta.size else 0.0
            plddt_mean = float(plddt_delta.mean()) if plddt_delta.size else 0.0
        else:
            result.plddt_allclose = False
            plddt_max = None
            plddt_mean = None
        print(f"raw pLDDT length: {len(raw_plddt)}")
        print(f"CSV pLDDT length: {len(sample.plddt)}")
        print(f"pLDDT allclose (atol=1e-3): {result.plddt_allclose}")
        print(f"pLDDT max abs diff: {format_float(plddt_max)}")
        print(f"pLDDT mean abs diff: {format_float(plddt_mean)}")
        print(f"full lengths align: {lengths_match}")

        high_confidence = np.where(sample.plddt > 70)[0]
        if high_confidence.size == 0:
            raise ValueError("no metadata pLDDT value is > 70")
        start = int(high_confidence.min())
        end = int(high_confidence.max()) + 1
        crop_length = end - start
        cropped_aa = sample.aa_seq[start:end]
        cropped_struct_ids = sample.struct_ids[start:end]
        cropped_plddt = sample.plddt[start:end]
        print(f"crop start: {start}")
        print(f"crop end exclusive: {end}")
        print(f"full length: {len(sample.aa_seq)}")
        print(f"cropped length: {crop_length}")
        if verbose:
            print(f"cropped aa_seq: {cropped_aa}")
            print(f"cropped struct_seq length: {len(cropped_struct_ids)}")
            print(f"cropped pLDDT length: {len(cropped_plddt)}")

        atom_positions = chain_feats["all_atom_positions"].unsqueeze(0).to(device)
        mask = chain_feats["res_mask"].unsqueeze(0).to(device)
        seq_length = torch.as_tensor(
            [atom_positions.shape[1]], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            z_cont_full, encoder_feats = tokenizer.encode(
                atom_positions, mask, seq_length
            )
            z_quant_current_full, _, aux = tokenizer.quantize(
                z_cont_full, mask=mask.bool()
            )
            struct_ids_current_full = aux[2]

        expected_full_shape = (1, len(sample.struct_ids), 13)
        latent_finite = bool(
            torch.isfinite(z_cont_full).all()
            and torch.isfinite(encoder_feats).all()
            and torch.isfinite(z_quant_current_full).all()
        )
        shape_ok = tuple(z_cont_full.shape) == expected_full_shape
        ids_shape_ok = tuple(struct_ids_current_full.shape) == (
            1,
            len(sample.struct_ids),
        )
        print(f"z_cont_full shape: {tuple(z_cont_full.shape)}")
        print(f"encoder_feats shape: {tuple(encoder_feats.shape)}")
        print(f"z_quant_current_full shape: {tuple(z_quant_current_full.shape)}")
        print(f"struct_ids_current_full shape: {tuple(struct_ids_current_full.shape)}")
        print(f"current tokenizer output finite: {latent_finite}")
        if not (shape_ok and ids_shape_ok):
            raise ValueError("current tokenizer output does not align with full metadata length")

        stored_ids = torch.tensor(
            sample.struct_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        generated = struct_ids_current_full.long()
        token_match_mask = generated == stored_ids
        token_matches = int(token_match_mask.sum().item())
        token_total = int(stored_ids.numel())
        token_mismatches = token_total - token_matches
        result.token_accuracy = token_matches / token_total if token_total else 1.0
        xor_values = [
            int(current) ^ int(stored)
            for current, stored in zip(
                generated[0].detach().cpu().tolist(), sample.struct_ids
            )
        ]
        flips = [bin(value).count("1") for value in xor_values]
        different_bits = sum(flips)
        total_bits = token_total * 13
        result.bit_accuracy = (
            (total_bits - different_bits) / total_bits if total_bits else 1.0
        )
        mismatch_flips = [count for count in flips if count]
        flip_distribution = Counter(mismatch_flips)
        print(f"token total: {token_total}")
        print(f"token matches: {token_matches}")
        print(f"token mismatches: {token_mismatches}")
        print(f"token accuracy: {result.token_accuracy:.9f}")
        print(f"total LFQ bits: {total_bits}")
        print(f"different bits: {different_bits}")
        print(f"bit accuracy: {result.bit_accuracy:.9f}")
        print(
            "mean flips per mismatching token: "
            + format_float(mean_or_none(mismatch_flips))
        )
        print(f"max flips: {max(mismatch_flips) if mismatch_flips else 0}")
        print(f"Hamming-distance distribution: {dict(sorted(flip_distribution.items()))}")

        with torch.inference_mode():
            z_quant_stored_full = tokenizer.quantize.get_codebook_entry(stored_ids)
        mismatch_bit_mask = z_quant_current_full != z_quant_stored_full
        margins = z_cont_full.abs()[mismatch_bit_mask].detach().cpu().double().tolist()
        result.mismatch_margins = [float(value) for value in margins]
        print(f"mismatch margin mean: {format_float(mean_or_none(margins))}")
        print(
            "mismatch margin median: "
            + format_float(statistics.median(margins) if margins else None)
        )
        print(f"mismatch margin min: {format_float(min_or_none(margins))}")
        print(f"mismatch margin max: {format_float(max_or_none(margins))}")
        for threshold in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50):
            count = sum(value <= threshold for value in margins)
            ratio = count / len(margins) if margins else 0.0
            print(f"|z_cont| <= {threshold:.2f}: {count}/{len(margins)} ({ratio:.6f})")

        # MAIN target: both operands come from the same current tokenizer pass.
        r_quant_full = z_cont_full - z_quant_current_full
        z_cont = z_cont_full[:, start:end]
        z_quant_current = z_quant_current_full[:, start:end]
        r_quant = r_quant_full[:, start:end]
        reconstructed = z_quant_current + r_quant
        residual_finite = bool(torch.isfinite(r_quant).all())
        reconstruction_pass = bool(
            torch.allclose(reconstructed, z_cont, atol=1e-6, rtol=1e-6)
        )
        reconstruction_max_error = float(
            (reconstructed - z_cont).abs().max().item()
        )
        result.residual_abs_mean = float(r_quant.abs().mean().item())
        result.residual_abs_max = float(r_quant.abs().max().item())
        residual_shape_ok = tuple(r_quant.shape) == (1, crop_length, 13)
        print(f"MAIN r_quant shape: {tuple(r_quant.shape)}")
        print(f"MAIN r_quant finite: {residual_finite}")
        print(f"residual abs mean: {result.residual_abs_mean:.9f}")
        print(f"residual abs max: {result.residual_abs_max:.9f}")
        print(f"reconstruction max error: {reconstruction_max_error:.9g}")
        print(f"reconstruction allclose: {reconstruction_pass}")

        current_crop = generated[:, start:end]
        stored_crop = stored_ids[:, start:end]
        crop_different_bits = sum(
            bin(int(current) ^ int(stored)).count("1")
            for current, stored in zip(
                current_crop[0].detach().cpu().tolist(),
                stored_crop[0].detach().cpu().tolist(),
            )
        )
        crop_total_bits = crop_length * 13
        crop_bit_agreement = (
            (crop_total_bits - crop_different_bits) / crop_total_bits
            if crop_total_bits
            else 1.0
        )
        historical_condition_residual_diagnostic = (
            z_cont - z_quant_stored_full[:, start:end]
        )
        print(f"current vs stored bit agreement after crop: {crop_bit_agreement:.9f}")
        print(
            "historical_condition_residual_diagnostic abs mean: "
            f"{historical_condition_residual_diagnostic.abs().mean().item():.9f}"
        )

        result.latent_pass = bool(
            lengths_match
            and latent_finite
            and residual_shape_ok
            and residual_finite
            and reconstruction_pass
        )
        result.passed = bool(
            result.aa_exact
            and result.plddt_allclose
            and lengths_match
            and result.latent_pass
            and result.bit_accuracy is not None
        )
        print(f"SAMPLE STATUS: {'PASS' if result.passed else 'FAIL'}")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"SAMPLE STATUS: FAIL ({result.error})")
        if verbose:
            traceback.print_exc()
    return result


def print_summary(results: list[SampleResult]) -> str:
    available = [result for result in results if result.available]
    validated = [result for result in results if result.validated]
    passed = [result for result in available if result.passed]
    token_accuracies = [
        result.token_accuracy
        for result in validated
        if result.token_accuracy is not None
    ]
    bit_accuracies = [
        result.bit_accuracy
        for result in validated
        if result.bit_accuracy is not None
    ]
    margins = [value for result in validated for value in result.mismatch_margins]
    residual_means = [
        result.residual_abs_mean
        for result in validated
        if finite_scalar(result.residual_abs_mean)
    ]
    residual_maxima = [
        result.residual_abs_max
        for result in validated
        if finite_scalar(result.residual_abs_max)
    ]
    if len(available) == len(results) and all(result.passed for result in available):
        status = "READY_FOR_AFDB_RECONSTRUCTION"
    elif available and passed:
        status = "PARTIAL"
    else:
        status = "FAILED"

    print("\n" + "=" * 60)
    print("AFDB v4 RECONSTRUCTION VALIDATION SUMMARY")
    print("=" * 60)
    print(f"requested samples: {len(results)}")
    print(f"available pdb.gz: {len(available)}")
    print(f"validated samples: {len(validated)}")
    print(f"failed samples: {sum(not result.passed for result in available)}")
    print(f"\nAA exact: {sum(result.aa_exact for result in validated)}/{len(validated)}")
    print(
        f"pLDDT allclose: {sum(result.plddt_allclose for result in validated)}/{len(validated)}"
    )
    print(
        f"latent pipeline PASS: {sum(result.latent_pass for result in validated)}/{len(validated)}"
    )
    print("\nTOKEN AGREEMENT")
    print(f"mean token accuracy: {format_float(mean_or_none(token_accuracies))}")
    print(f"min token accuracy: {format_float(min_or_none(token_accuracies))}")
    print(f"max token accuracy: {format_float(max_or_none(token_accuracies))}")
    print("\nBIT AGREEMENT")
    print(f"mean bit accuracy: {format_float(mean_or_none(bit_accuracies))}")
    print(f"min bit accuracy: {format_float(min_or_none(bit_accuracies))}")
    print(f"max bit accuracy: {format_float(max_or_none(bit_accuracies))}")
    print("\nLFQ MISMATCH MARGIN")
    print(f"mean mismatch |z_cont|: {format_float(mean_or_none(margins))}")
    print(
        "median mismatch |z_cont|: "
        + format_float(statistics.median(margins) if margins else None)
    )
    print("\nQUANTIZATION RESIDUAL")
    print(f"mean residual abs mean: {format_float(mean_or_none(residual_means))}")
    print(f"max residual abs max: {format_float(max_or_none(residual_maxima))}")
    print(f"\nFINAL STATUS: {status}")
    print(
        "NOTE: this reconstructs z_cont_current from the current public tokenizer "
        "and official AFDB v4 coordinates; historical z_cont equivalence is NOT CONFIRMED."
    )
    print("=" * 60)
    return status


def main() -> int:
    args = parse_args()
    metadata_csv = Path(args.metadata_csv)
    pdb_gz_dir = Path(args.pdb_gz_dir)
    samples, candidate_count = select_samples(
        metadata_csv=metadata_csv,
        pdb_gz_dir=pdb_gz_dir,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )
    if args.verbose:
        print(f"eligible AFDB Swiss-Prot candidates: {candidate_count}")
    print_selection(samples)
    if args.select_only:
        return 0

    import torch

    from byprot.models.utils import get_struct_tokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    device = torch.device(args.device)
    tokenizer = get_struct_tokenizer().to(device).eval()
    tokenizer.requires_grad_(False)
    trainable_count = sum(
        parameter.numel() for parameter in tokenizer.parameters() if parameter.requires_grad
    )
    print(f"\nfrozen tokenizer trainable parameter count: {trainable_count}")
    results = [
        validate_one(sample, tokenizer, device, args.verbose) for sample in samples
    ]
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

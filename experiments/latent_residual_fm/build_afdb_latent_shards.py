#!/usr/bin/env python3
"""Build and benchmark sharded AFDB current-quantization residual data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import subprocess
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.latent_residual_fm.build_afdb_latent_dataset import (
    OriginalRow,
    TrainRow,
    load_original_rows,
    select_train_rows,
    shutil_copyfileobj,
)


SCHEMA_VERSION = 1
TOKENIZER_ID = "airkingbd/struct_tokenizer"
FULL_SCOPE_SAMPLES = 176_733
FULL_SCOPE_RESIDUES = 46_844_154
SAMPLE_KEYS = {
    "sample_id",
    "processed_path",
    "archive_member",
    "full_length",
    "crop_start",
    "crop_end",
    "length",
    "aatype",
    "res_mask",
    "struct_ids_current",
    "struct_ids_stored",
    "z_cont",
    "z_quant_current",
    "residual",
    "token_accuracy_vs_stored",
    "bit_accuracy_vs_stored",
}
MANIFEST_COLUMNS = (
    "sample_id",
    "processed_path",
    "archive_member",
    "status",
    "full_length",
    "crop_start",
    "crop_end",
    "length",
    "token_accuracy_vs_stored",
    "bit_accuracy_vs_stored",
    "residual_abs_mean",
    "residual_abs_max",
    "shard_file",
    "index_in_shard",
    "error",
)


@dataclass
class Timings:
    tar_seconds: float = 0.0
    pdb_seconds: float = 0.0
    tokenizer_seconds: float = 0.0
    shard_save_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar_path", default="data-bin/afdb_v4/swissprot_pdb_v4.tar")
    parser.add_argument("--metadata_csv", default="data-bin/metadata/pdb_afdb_cameo.csv")
    parser.add_argument("--train_dir", default="data-bin/pdb_swissprot/train")
    parser.add_argument("--output_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded")
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify_saved", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_manifest_save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_sample(sample: dict[str, Any]) -> None:
    missing = SAMPLE_KEYS.difference(sample)
    if missing:
        raise ValueError(f"sample missing keys: {sorted(missing)}")
    length = int(sample["length"])
    expected = {
        "aatype": ((length,), torch.long),
        "res_mask": ((length,), torch.float32),
        "struct_ids_current": ((length,), torch.long),
        "struct_ids_stored": ((length,), torch.long),
        "z_cont": ((length, 13), torch.float32),
        "z_quant_current": ((length, 13), torch.float32),
        "residual": ((length, 13), torch.float32),
    }
    for key, (shape, dtype) in expected.items():
        value = sample[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{key} is not a tensor")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{key}: expected {shape}/{dtype}, got {tuple(value.shape)}/{value.dtype}"
            )
    for key in ("z_cont", "z_quant_current", "residual"):
        if not bool(torch.isfinite(sample[key]).all()):
            raise ValueError(f"{key} contains non-finite values")
    for key in ("struct_ids_current", "struct_ids_stored"):
        ids = sample[key]
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) > 8191):
            raise ValueError(f"{key} is outside raw LFQ range 0..8191")
    values = set(sample["z_quant_current"].unique().tolist())
    if not values or not values.issubset({-1.0, 1.0}):
        raise ValueError(f"z_quant_current values are not -1/+1: {sorted(values)}")
    if not torch.allclose(
        sample["z_quant_current"] + sample["residual"],
        sample["z_cont"],
        atol=1e-7,
        rtol=1e-7,
    ):
        raise ValueError("latent reconstruction check failed")


def verify_shard(path: Path) -> dict[str, Any]:
    shard = torch.load(path, map_location="cpu")
    required = {"schema_version", "num_samples", "total_residues", "samples"}
    missing = required.difference(shard)
    if missing:
        raise ValueError(f"shard missing keys: {sorted(missing)}")
    if shard["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version: {shard['schema_version']}")
    samples = shard["samples"]
    if not isinstance(samples, list) or shard["num_samples"] != len(samples):
        raise ValueError("shard sample count is inconsistent")
    residues = 0
    for sample in samples:
        validate_sample(sample)
        residues += int(sample["length"])
    if shard["total_residues"] != residues:
        raise ValueError("shard total_residues is inconsistent")
    return shard


def build_sample(
    member_file: Any,
    train: TrainRow,
    original: OriginalRow,
    tokenizer: Any,
    device: torch.device,
    timings: Timings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset

    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pdb") as temporary_pdb:
        with gzip.GzipFile(fileobj=member_file, mode="rb") as decompressed:
            shutil_copyfileobj(decompressed, temporary_pdb)
        temporary_pdb.flush()
        raw_chain_feats, _ = du.process_pdb_file(temporary_pdb.name)
    raw_aatype = np.asarray(raw_chain_feats["aatype"])
    raw_aa_seq = du.aatype_to_seq(raw_aatype)
    raw_plddt = np.asarray(raw_chain_feats["b_factors"], dtype=np.float64)[:, du.CA_IDX]
    full_length = len(raw_aa_seq)
    if raw_aa_seq != original.aa_seq:
        raise ValueError("raw AFDB AA does not exactly match original CSV")
    if not (
        full_length
        == len(original.aa_seq)
        == len(original.struct_ids)
        == len(original.plddt)
        == original.seq_len
    ):
        raise ValueError("raw/original full lengths do not align")
    if raw_plddt.shape != original.plddt.shape or not np.allclose(
        raw_plddt, original.plddt, atol=1e-3, rtol=0.0
    ):
        raise ValueError("raw AFDB CA B-factor pLDDT does not match original CSV")
    valid_idx = np.where(original.plddt > 70)[0]
    if valid_idx.size == 0:
        raise ValueError("original pLDDT contains no value > 70")
    start = int(valid_idx.min())
    end = int(valid_idx.max()) + 1
    if original.aa_seq[start:end] != train.aa_seq:
        raise ValueError("cropped original aa_seq does not match train parquet")
    if tuple(original.struct_ids[start:end]) != train.struct_ids:
        raise ValueError("cropped original struct_seq does not match train parquet")
    if original.plddt[start:end].shape != train.plddt.shape or not np.allclose(
        original.plddt[start:end], train.plddt, atol=1e-3, rtol=0.0
    ):
        raise ValueError("cropped original pLDDT does not match train parquet")
    processed = PdbDataset.process_chain(raw_chain_feats)
    if int(processed["all_atom_positions"].shape[0]) != full_length:
        raise ValueError("processed coordinate length differs from full AFDB length")
    timings.pdb_seconds += time.perf_counter() - started

    atom_positions = processed["all_atom_positions"].unsqueeze(0).to(device)
    full_mask = processed["res_mask"].unsqueeze(0).to(device)
    seq_length = torch.tensor([full_length], dtype=torch.long, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        z_cont_full, encoder_feats = tokenizer.encode(
            atom_positions=atom_positions, mask=full_mask, seq_length=seq_length
        )
        z_quant_full, _, aux = tokenizer.quantize(
            z_cont_full, mask=full_mask.bool()
        )
        ids_current_full = aux[2]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    timings.tokenizer_seconds += time.perf_counter() - started
    if tuple(z_cont_full.shape) != (1, full_length, 13):
        raise ValueError(f"unexpected z_cont_full shape: {tuple(z_cont_full.shape)}")
    if tuple(encoder_feats.shape[:2]) != (1, full_length):
        raise ValueError(f"unexpected encoder feature shape: {tuple(encoder_feats.shape)}")
    if tuple(z_quant_full.shape) != (1, full_length, 13):
        raise ValueError(f"unexpected z_quant_current_full shape: {tuple(z_quant_full.shape)}")
    if tuple(ids_current_full.shape) != (1, full_length):
        raise ValueError(f"unexpected current ID shape: {tuple(ids_current_full.shape)}")

    length = end - start
    z_cont = z_cont_full[0, start:end].float().cpu()
    z_quant_current = z_quant_full[0, start:end].float().cpu()
    struct_ids_current = ids_current_full[0, start:end].long().cpu()
    struct_ids_stored = torch.tensor(original.struct_ids[start:end], dtype=torch.long)
    aatype = processed["aatype"][start:end].long().cpu()
    res_mask = processed["res_mask"][start:end].float().cpu()
    residual = z_cont - z_quant_current
    matches = int((struct_ids_current == struct_ids_stored).sum())
    token_accuracy = matches / length
    different_bits = sum(
        bin(int(current) ^ int(stored)).count("1")
        for current, stored in zip(struct_ids_current.tolist(), struct_ids_stored.tolist())
    )
    bit_accuracy = (length * 13 - different_bits) / (length * 13)
    sample = {
        "sample_id": train.sample_id,
        "processed_path": train.processed_path,
        "archive_member": train.archive_member,
        "full_length": full_length,
        "crop_start": start,
        "crop_end": end,
        "length": length,
        "aatype": aatype,
        "res_mask": res_mask,
        "struct_ids_current": struct_ids_current,
        "struct_ids_stored": struct_ids_stored,
        "z_cont": z_cont,
        "z_quant_current": z_quant_current,
        "residual": residual,
        "token_accuracy_vs_stored": token_accuracy,
        "bit_accuracy_vs_stored": bit_accuracy,
    }
    validate_sample(sample)
    manifest = {
        "sample_id": train.sample_id,
        "processed_path": train.processed_path,
        "archive_member": train.archive_member,
        "status": "SAVED",
        "full_length": full_length,
        "crop_start": start,
        "crop_end": end,
        "length": length,
        "token_accuracy_vs_stored": token_accuracy,
        "bit_accuracy_vs_stored": bit_accuracy,
        "residual_abs_mean": float(residual.abs().mean()),
        "residual_abs_max": float(residual.abs().max()),
        "shard_file": "",
        "index_in_shard": "",
        "error": "",
    }
    return sample, manifest


def read_existing_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(MANIFEST_COLUMNS).difference(reader.fieldnames or ()):
            return {}
        return {
            row["sample_id"]: row
            for row in reader
            if row.get("status") == "SAVED" and row.get("sample_id")
        }


def next_shard_index(output_dir: Path, overwrite: bool) -> int:
    if overwrite:
        return 0
    indexes = []
    for path in output_dir.glob("shard-*.pt"):
        try:
            indexes.append(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return max(indexes) + 1 if indexes else 0


def save_shard(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_dir: Path,
    shard_index: int,
    verify_saved: bool,
    timings: Timings,
) -> tuple[Path, bool]:
    path = output_dir / f"shard-{shard_index:06d}.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "num_samples": len(samples),
        "total_residues": sum(int(sample["length"]) for sample in samples),
        "samples": samples,
    }
    started = time.perf_counter()
    atomic_torch_save(payload, path)
    verified = False
    if verify_saved:
        verify_shard(path)
        verified = True
    timings.shard_save_seconds += time.perf_counter() - started
    for index, row in enumerate(rows):
        row["shard_file"] = path.name
        row["index_in_shard"] = index
    return path, verified


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get("status") == "SAVED" and row.get(key) != ""]


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.9f}"


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    if args.max_samples is None:
        raise ValueError("--max_samples is required; unrestricted full build is disabled")
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    tar_path = Path(args.tar_path)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    if not tar_path.is_file() or not metadata_csv.is_file():
        raise FileNotFoundError("source TAR or metadata CSV is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    meta_path = output_dir / "dataset_meta.json"

    selected = select_train_rows(args)
    originals = load_original_rows(metadata_csv, selected)
    existing = {} if args.overwrite else read_existing_manifest(manifest_path)
    selected_ids = {row.sample_id for row in selected}
    resume_rows = {
        sample_id: row
        for sample_id, row in existing.items()
        if sample_id in selected_ids
        and (output_dir / row.get("shard_file", "")).is_file()
    }
    pending_rows = [row for row in selected if row.sample_id not in resume_rows]
    print(f"selected={len(selected)} resume_saved={len(resume_rows)} pending={len(pending_rows)}")

    from byprot.models.utils import get_struct_tokenizer

    device = torch.device(args.device)
    tokenizer = get_struct_tokenizer(TOKENIZER_ID).to(device).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise RuntimeError("structure tokenizer is not fully frozen")

    by_member = {row.archive_member: row for row in pending_rows}
    pending_members = set(by_member)
    timings = Timings()
    found_new = 0
    manifest_new: dict[str, dict[str, Any]] = {}
    buffer_samples: list[dict[str, Any]] = []
    buffer_rows: list[dict[str, Any]] = []
    shard_index = next_shard_index(output_dir, args.overwrite)
    created_paths: list[Path] = []
    verified_new = 0
    tar_checkpoint = time.perf_counter()
    with tarfile.open(tar_path, mode="r:") as archive:
        iterator = iter(archive)
        while pending_members:
            try:
                member = next(iterator)
            except StopIteration:
                break
            now = time.perf_counter()
            timings.tar_seconds += now - tar_checkpoint
            basename = Path(member.name).name
            if basename not in pending_members:
                tar_checkpoint = time.perf_counter()
                continue
            pending_members.remove(basename)
            found_new += 1
            train = by_member[basename]
            try:
                access_started = time.perf_counter()
                member_file = archive.extractfile(member)
                timings.tar_seconds += time.perf_counter() - access_started
                if member_file is None:
                    raise ValueError("tar member has no extractable content")
                with member_file:
                    sample, row = build_sample(
                        member_file,
                        train,
                        originals[train.processed_path],
                        tokenizer,
                        device,
                        timings,
                    )
                buffer_samples.append(sample)
                buffer_rows.append(row)
                manifest_new[train.sample_id] = row
                print(
                    f"[READY {len(manifest_new)}/{len(pending_rows)}] "
                    f"sample_id={train.sample_id} L={sample['length']} "
                    f"token_acc={sample['token_accuracy_vs_stored']:.9f} "
                    f"bit_acc={sample['bit_accuracy_vs_stored']:.9f}"
                )
                if len(buffer_samples) == args.shard_size:
                    path, verified = save_shard(
                        buffer_samples,
                        buffer_rows,
                        output_dir,
                        shard_index,
                        args.verify_saved,
                        timings,
                    )
                    created_paths.append(path)
                    verified_new += int(verified)
                    print(f"[SHARD SAVED] {path} samples={len(buffer_samples)}")
                    shard_index += 1
                    buffer_samples, buffer_rows = [], []
            except Exception as exc:
                row = {
                    "sample_id": train.sample_id,
                    "processed_path": train.processed_path,
                    "archive_member": train.archive_member,
                    "status": "FAILED",
                    "full_length": "",
                    "crop_start": "",
                    "crop_end": "",
                    "length": "",
                    "token_accuracy_vs_stored": "",
                    "bit_accuracy_vs_stored": "",
                    "residual_abs_mean": "",
                    "residual_abs_max": "",
                    "shard_file": "",
                    "index_in_shard": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                manifest_new[train.sample_id] = row
                print(f"[FAILED] sample_id={train.sample_id} reason={row['error']}")
                if args.verbose:
                    traceback.print_exc()
            tar_checkpoint = time.perf_counter()
    if buffer_samples:
        path, verified = save_shard(
            buffer_samples,
            buffer_rows,
            output_dir,
            shard_index,
            args.verify_saved,
            timings,
        )
        created_paths.append(path)
        verified_new += int(verified)
        print(f"[SHARD SAVED] {path} samples={len(buffer_samples)}")
    for member_name in sorted(pending_members):
        train = by_member[member_name]
        manifest_new[train.sample_id] = {
            "sample_id": train.sample_id,
            "processed_path": train.processed_path,
            "archive_member": train.archive_member,
            "status": "FAILED",
            "full_length": "",
            "crop_start": "",
            "crop_end": "",
            "length": "",
            "token_accuracy_vs_stored": "",
            "bit_accuracy_vs_stored": "",
            "residual_abs_mean": "",
            "residual_abs_max": "",
            "shard_file": "",
            "index_in_shard": "",
            "error": "TAR_MEMBER_NOT_FOUND",
        }

    manifest_rows: list[dict[str, Any]] = []
    for train in selected:
        if train.sample_id in manifest_new:
            manifest_rows.append(manifest_new[train.sample_id])
        else:
            manifest_rows.append(resume_rows[train.sample_id])
    atomic_manifest_save(manifest_rows, manifest_path)

    all_shards = sorted(output_dir.glob("shard-*.pt"))
    verified_shards = 0
    if args.verify_saved:
        for path in all_shards:
            verify_shard(path)
            verified_shards += 1
    saved_rows = [row for row in manifest_rows if row["status"] == "SAVED"]
    failed_rows = [row for row in manifest_rows if row["status"] == "FAILED"]
    lengths = values(saved_rows, "length")
    token = values(saved_rows, "token_accuracy_vs_stored")
    bits = values(saved_rows, "bit_accuracy_vs_stored")
    residual_means = values(saved_rows, "residual_abs_mean")
    residual_maxes = values(saved_rows, "residual_abs_max")
    total_residues = int(sum(lengths))
    total_seconds = time.perf_counter() - total_started
    samples_per_second = len(saved_rows) / total_seconds if total_seconds else 0.0
    residues_per_second = total_residues / total_seconds if total_seconds else 0.0
    total_shard_bytes = sum(path.stat().st_size for path in all_shards)
    bytes_per_sample = total_shard_bytes / len(saved_rows) if saved_rows else 0.0
    bytes_per_residue = total_shard_bytes / total_residues if total_residues else 0.0
    estimated_gib = bytes_per_residue * FULL_SCOPE_RESIDUES / (1024**3)
    estimated_hours = FULL_SCOPE_SAMPLES / samples_per_second / 3600 if samples_per_second else math.inf
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_tar": str(tar_path),
        "source_tar_size": tar_path.stat().st_size,
        "metadata_csv": str(metadata_csv),
        "train_dir": str(args.train_dir),
        "structure_tokenizer": TOKENIZER_ID,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "crop_threshold": 70,
        "important_processing_order": "full_encode_then_crop",
        "residual_definition": "z_cont_current - z_quant_current",
        "shard_size": args.shard_size,
        "selected_samples": len(selected),
        "created_shards": len(all_shards),
        "saved_samples": len(saved_rows),
        "failed_samples": len(failed_rows),
        "torch_version": torch.__version__,
        "git_commit": git_commit(),
        "timing_seconds": {
            "total": total_seconds,
            "tar_scanning_member_access": timings.tar_seconds,
            "pdb_preprocessing": timings.pdb_seconds,
            "tokenizer_encode_quantize": timings.tokenizer_seconds,
            "shard_save": timings.shard_save_seconds,
        },
    }
    atomic_json_save(meta, meta_path)

    expected_shards = math.ceil(len(selected) / args.shard_size)
    if (
        len(selected) == 100
        and found_new + len(resume_rows) == 100
        and len(saved_rows) == 100
        and not failed_rows
        and len(all_shards) == expected_shards == 2
        and (not args.verify_saved or verified_shards == 2)
    ):
        status = "SHARD_BENCHMARK_PASS"
    elif saved_rows:
        status = "SHARD_BENCHMARK_PARTIAL"
    else:
        status = "SHARD_BENCHMARK_FAILED"

    print("\n" + "=" * 60)
    print("AFDB SHARDED LATENT DATASET BENCHMARK")
    print("=" * 60)
    print(f"selected: {len(selected)}")
    print(f"found: {found_new + len(resume_rows)}")
    print(f"saved: {len(saved_rows)}")
    print(f"failed: {len(failed_rows)}")
    print(f"\nshard size: {args.shard_size}")
    print(f"shards created: {len(all_shards)}")
    print(f"verified shards: {verified_shards}")
    print(f"\nlength min: {fmt(min(lengths) if lengths else None)}")
    print(f"length mean: {fmt(statistics.fmean(lengths) if lengths else None)}")
    print(f"length max: {fmt(max(lengths) if lengths else None)}")
    print(f"total residues: {total_residues}")
    print("\nTOKEN AGREEMENT")
    print(f"mean: {fmt(statistics.fmean(token) if token else None)}")
    print(f"min: {fmt(min(token) if token else None)}")
    print(f"max: {fmt(max(token) if token else None)}")
    print("\nBIT AGREEMENT")
    print(f"mean: {fmt(statistics.fmean(bits) if bits else None)}")
    print(f"min: {fmt(min(bits) if bits else None)}")
    print(f"max: {fmt(max(bits) if bits else None)}")
    print("\nRESIDUAL")
    print(f"mean abs mean: {fmt(statistics.fmean(residual_means) if residual_means else None)}")
    print(f"max abs max: {fmt(max(residual_maxes) if residual_maxes else None)}")
    print("\nTIMING")
    print(f"total seconds: {total_seconds:.6f}")
    print(f"TAR scanning / member access seconds: {timings.tar_seconds:.6f}")
    print(f"PDB preprocessing seconds: {timings.pdb_seconds:.6f}")
    print(f"tokenizer encode+quantize seconds: {timings.tokenizer_seconds:.6f}")
    print(f"shard save seconds: {timings.shard_save_seconds:.6f}")
    print(f"samples/sec: {samples_per_second:.9f}")
    print(f"residues/sec: {residues_per_second:.9f}")
    print("\nSTORAGE")
    print(f"number of shard files: {len(all_shards)}")
    print(f"total shard bytes: {total_shard_bytes}")
    print(f"total shard MiB: {total_shard_bytes / (1024**2):.6f}")
    print(f"bytes/sample: {bytes_per_sample:.6f}")
    print(f"bytes/residue: {bytes_per_residue:.6f}")
    print(f"estimated full L<=512 GiB: {estimated_gib:.6f}")
    print("\nFULL BUILD ESTIMATE")
    print(f"target samples: {FULL_SCOPE_SAMPLES}")
    print(f"estimated hours: {estimated_hours:.6f}")
    print("This is a linear benchmark extrapolation; TAR ordering, I/O, and GPU load may change actual runtime.")
    print(f"\noutput dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"dataset meta: {meta_path}")
    print(f"\nFINAL STATUS: {status}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark batched structure-tokenizer inference for AFDB latent shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass, field
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
from experiments.latent_residual_fm.build_afdb_latent_shards import (
    FULL_SCOPE_SAMPLES,
    MANIFEST_COLUMNS,
    SCHEMA_VERSION,
    TOKENIZER_ID,
    atomic_json_save,
    atomic_manifest_save,
    git_commit,
    next_shard_index,
    validate_sample,
    verify_shard,
)


BASELINE_TOTAL_SECONDS = 150.366079
BASELINE_SAMPLES_PER_SECOND = 0.665043609
BASELINE_RESIDUES_PER_SECOND = 194.711467960
OLD_FULL_ESTIMATE_HOURS = 73.818467


@dataclass
class PreparedSample:
    train: TrainRow
    original: OriginalRow
    processed: dict[str, Any]
    full_length: int
    crop_start: int
    crop_end: int


@dataclass
class Timings:
    # These intervals are intentionally independent; they are not summed to
    # reconstruct total wall time, which also includes selection/model loading.
    tar_scan_seconds: float = 0.0
    pdb_preprocess_seconds: float = 0.0
    tokenizer_seconds: float = 0.0
    shard_save_seconds: float = 0.0
    reference_compare_seconds: float = 0.0


@dataclass
class BatchStats:
    sizes: list[int] = field(default_factory=list)
    actual_tokens: list[int] = field(default_factory=list)
    padded_tokens: list[int] = field(default_factory=list)
    oom_events: int = 0
    batch_splits: int = 0


@dataclass
class EquivalenceStats:
    reference_samples: int = 0
    exact_struct_ids_current: int = 0
    z_cont_allclose: int = 0
    z_quant_allclose: int = 0
    residual_allclose: int = 0
    exact_metadata_and_inputs: int = 0
    z_cont_max_diff: float = 0.0
    z_cont_abs_sum: float = 0.0
    z_cont_elements: int = 0
    residual_max_diff: float = 0.0
    residual_abs_sum: float = 0.0
    residual_elements: int = 0
    id_mismatch_samples: int = 0
    id_mismatch_tokens: int = 0
    id_mismatch_bits: int = 0
    id_mismatch_margins: list[float] = field(default_factory=list)


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
    parser.add_argument("--max_batch_tokens", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--batch_buffer_size", type=int, default=64)
    parser.add_argument("--reference_dir", default=None)
    parser.add_argument("--compare_reference", action="store_true")
    return parser.parse_args()


def prepare_member(
    member_file: Any,
    train: TrainRow,
    original: OriginalRow,
    timings: Timings,
) -> PreparedSample:
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
    if int(processed["res_mask"].shape[0]) != full_length:
        raise ValueError("processed mask length differs from full AFDB length")
    timings.pdb_preprocess_seconds += time.perf_counter() - started
    return PreparedSample(train, original, processed, full_length, start, end)


def form_dynamic_batches(
    buffer: list[PreparedSample], max_batch_tokens: int, max_batch_size: int
) -> list[list[PreparedSample]]:
    ordered = sorted(buffer, key=lambda sample: sample.full_length)
    batches: list[list[PreparedSample]] = []
    current: list[PreparedSample] = []
    token_sum = 0
    for sample in ordered:
        would_exceed = current and (
            len(current) + 1 > max_batch_size
            or token_sum + sample.full_length > max_batch_tokens
        )
        if would_exceed:
            batches.append(current)
            current = []
            token_sum = 0
        current.append(sample)
        token_sum += sample.full_length
        # A single sequence over the budget is explicitly allowed.
        if len(current) == 1 and sample.full_length > max_batch_tokens:
            batches.append(current)
            current = []
            token_sum = 0
    if current:
        batches.append(current)
    return batches


def materialize_samples(
    prepared: list[PreparedSample],
    z_cont_batch: torch.Tensor,
    z_quant_batch: torch.Tensor,
    ids_batch: torch.Tensor,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, item in enumerate(prepared):
        full_length = item.full_length
        start, end = item.crop_start, item.crop_end
        length = end - start
        # Remove right-padding first, then apply the historical terminal crop.
        z_cont_full = z_cont_batch[index, :full_length]
        z_quant_full = z_quant_batch[index, :full_length]
        ids_full = ids_batch[index, :full_length]
        z_cont = z_cont_full[start:end].float().cpu()
        z_quant_current = z_quant_full[start:end].float().cpu()
        struct_ids_current = ids_full[start:end].long().cpu()
        struct_ids_stored = torch.tensor(
            item.original.struct_ids[start:end], dtype=torch.long
        )
        aatype = item.processed["aatype"][start:end].long().cpu()
        res_mask = item.processed["res_mask"][start:end].float().cpu()
        residual = z_cont - z_quant_current
        matches = int((struct_ids_current == struct_ids_stored).sum())
        token_accuracy = matches / length
        different_bits = sum(
            bin(int(current) ^ int(stored)).count("1")
            for current, stored in zip(
                struct_ids_current.tolist(), struct_ids_stored.tolist()
            )
        )
        bit_accuracy = (length * 13 - different_bits) / (length * 13)
        sample = {
            "sample_id": item.train.sample_id,
            "processed_path": item.train.processed_path,
            "archive_member": item.train.archive_member,
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
        row = {
            "sample_id": item.train.sample_id,
            "processed_path": item.train.processed_path,
            "archive_member": item.train.archive_member,
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
        output.append((sample, row))
    return output


def encode_with_oom_fallback(
    prepared: list[PreparedSample],
    tokenizer: Any,
    device: torch.device,
    timings: Timings,
    stats: BatchStats,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn

    try:
        batch = collate_fn([item.processed for item in prepared])
        positions = batch["all_atom_positions"].to(device)
        mask = batch["res_mask"].to(device)
        seq_length = batch["seq_length"].long().to(device)
        expected_lengths = torch.tensor(
            [item.full_length for item in prepared], dtype=torch.long, device=device
        )
        if seq_length.shape != expected_lengths.shape or not torch.equal(
            seq_length, expected_lengths
        ):
            raise ValueError(
                f"official collate seq_length mismatch: {seq_length.tolist()} vs {expected_lengths.tolist()}"
            )
        max_length = max(item.full_length for item in prepared)
        if tuple(positions.shape) != (len(prepared), max_length, 37, 3):
            raise ValueError(f"unexpected collated coordinate shape: {tuple(positions.shape)}")
        for index, item in enumerate(prepared):
            if not bool((mask[index, item.full_length :] == 0).all()):
                raise ValueError("official collate produced nonzero padding mask")
        started = time.perf_counter()
        with torch.inference_mode():
            z_cont_batch, encoder_feats = tokenizer.encode(
                atom_positions=positions,
                mask=mask,
                seq_length=seq_length,
            )
            z_quant_batch, _, aux = tokenizer.quantize(
                z_cont_batch, mask=mask.bool()
            )
            ids_batch = aux[2]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        timings.tokenizer_seconds += time.perf_counter() - started
        expected_latent = (len(prepared), max_length, 13)
        if tuple(z_cont_batch.shape) != expected_latent:
            raise ValueError(f"unexpected z_cont batch shape: {tuple(z_cont_batch.shape)}")
        if tuple(z_quant_batch.shape) != expected_latent:
            raise ValueError(f"unexpected z_quant batch shape: {tuple(z_quant_batch.shape)}")
        if tuple(ids_batch.shape) != (len(prepared), max_length):
            raise ValueError(f"unexpected ID batch shape: {tuple(ids_batch.shape)}")
        if tuple(encoder_feats.shape[:2]) != (len(prepared), max_length):
            raise ValueError(f"unexpected encoder feature shape: {tuple(encoder_feats.shape)}")
        stats.sizes.append(len(prepared))
        stats.actual_tokens.append(sum(item.full_length for item in prepared))
        stats.padded_tokens.append(len(prepared) * max_length)
        return (
            materialize_samples(prepared, z_cont_batch, z_quant_batch, ids_batch),
            [],
        )
    except torch.cuda.OutOfMemoryError as exc:
        stats.oom_events += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(prepared) == 1:
            item = prepared[0]
            failed = failed_row(item.train, f"CUDA OOM at batch size 1: {exc}")
            return [], [failed]
        midpoint = len(prepared) // 2
        stats.batch_splits += 1
        left_output, left_failed = encode_with_oom_fallback(
            prepared[:midpoint], tokenizer, device, timings, stats
        )
        right_output, right_failed = encode_with_oom_fallback(
            prepared[midpoint:], tokenizer, device, timings, stats
        )
        return left_output + right_output, left_failed + right_failed


def failed_row(train: TrainRow, error: str) -> dict[str, Any]:
    return {
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
        "error": error,
    }


def atomic_shard_save(
    samples: list[dict[str, Any]], path: Path, timings: Timings
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "num_samples": len(samples),
        "total_residues": sum(int(sample["length"]) for sample in samples),
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    started = time.perf_counter()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    timings.shard_save_seconds += time.perf_counter() - started


def load_reference(reference_dir: Path) -> dict[str, dict[str, Any]]:
    reference: dict[str, dict[str, Any]] = {}
    files = sorted(reference_dir.glob("shard-*.pt"))
    if not files:
        raise FileNotFoundError(f"no reference shards under {reference_dir}")
    for path in files:
        shard = verify_shard(path)
        for sample in shard["samples"]:
            sample_id = sample["sample_id"]
            if sample_id in reference:
                raise ValueError(f"duplicate reference sample_id: {sample_id}")
            reference[sample_id] = sample
    return reference


def compare_one(
    sample: dict[str, Any], reference: dict[str, Any], stats: EquivalenceStats
) -> None:
    stats.reference_samples += 1
    scalar_keys = (
        "sample_id",
        "processed_path",
        "archive_member",
        "full_length",
        "crop_start",
        "crop_end",
        "length",
    )
    exact_tensor_keys = ("aatype", "res_mask", "struct_ids_stored")
    metadata_exact = all(sample[key] == reference[key] for key in scalar_keys)
    input_exact = all(torch.equal(sample[key], reference[key]) for key in exact_tensor_keys)
    if metadata_exact and input_exact:
        stats.exact_metadata_and_inputs += 1
    ids_equal = torch.equal(sample["struct_ids_current"], reference["struct_ids_current"])
    if ids_equal:
        stats.exact_struct_ids_current += 1
    else:
        current = sample["struct_ids_current"]
        old = reference["struct_ids_current"]
        mismatch = current != old
        stats.id_mismatch_samples += 1
        stats.id_mismatch_tokens += int(mismatch.sum())
        stats.id_mismatch_bits += sum(
            bin(int(a) ^ int(b)).count("1")
            for a, b in zip(current[mismatch].tolist(), old[mismatch].tolist())
        )
        bit_mask = sample["z_quant_current"] != reference["z_quant_current"]
        stats.id_mismatch_margins.extend(
            sample["z_cont"].abs()[bit_mask].double().tolist()
        )
    z_cont_diff = (sample["z_cont"] - reference["z_cont"]).abs().double()
    residual_diff = (sample["residual"] - reference["residual"]).abs().double()
    stats.z_cont_max_diff = max(stats.z_cont_max_diff, float(z_cont_diff.max()))
    stats.z_cont_abs_sum += float(z_cont_diff.sum())
    stats.z_cont_elements += z_cont_diff.numel()
    stats.residual_max_diff = max(stats.residual_max_diff, float(residual_diff.max()))
    stats.residual_abs_sum += float(residual_diff.sum())
    stats.residual_elements += residual_diff.numel()
    if torch.allclose(sample["z_cont"], reference["z_cont"], atol=1e-5, rtol=1e-5):
        stats.z_cont_allclose += 1
    if torch.allclose(
        sample["z_quant_current"], reference["z_quant_current"], atol=1e-5, rtol=1e-5
    ):
        stats.z_quant_allclose += 1
    if torch.allclose(sample["residual"], reference["residual"], atol=1e-5, rtol=1e-5):
        stats.residual_allclose += 1


def manifest_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get("status") == "SAVED" and row.get(key) != ""
    ]


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    if args.max_samples is None:
        raise ValueError("--max_samples is required; unrestricted full build is disabled")
    if min(args.shard_size, args.max_batch_tokens, args.max_batch_size, args.batch_buffer_size) <= 0:
        raise ValueError("shard and batching limits must be positive")
    if args.compare_reference and not args.reference_dir:
        raise ValueError("--compare_reference requires --reference_dir")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")
    tar_path = Path(args.tar_path)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    if not tar_path.is_file() or not metadata_csv.is_file():
        raise FileNotFoundError("source TAR or metadata CSV is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    meta_path = output_dir / "dataset_meta.json"

    selected = select_train_rows(args)
    originals = load_original_rows(metadata_csv, selected)
    reference_started = time.perf_counter()
    reference = load_reference(reference_dir) if args.compare_reference else {}
    reference_load_seconds = time.perf_counter() - reference_started
    if args.compare_reference:
        selected_ids = {row.sample_id for row in selected}
        if selected_ids != set(reference):
            raise ValueError(
                f"selected/reference sample_id sets differ: selected={len(selected_ids)} reference={len(reference)}"
            )

    from byprot.models.utils import get_struct_tokenizer

    device = torch.device(args.device)
    tokenizer = get_struct_tokenizer(TOKENIZER_ID).to(device).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise RuntimeError("structure tokenizer is not fully frozen")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    by_member = {row.archive_member: row for row in selected}
    pending_members = set(by_member)
    timings = Timings(reference_compare_seconds=reference_load_seconds)
    batch_stats = BatchStats()
    equivalence = EquivalenceStats()
    buffer: list[PreparedSample] = []
    samples_by_id: dict[str, dict[str, Any]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    found = 0

    def flush_buffer() -> None:
        nonlocal buffer
        for dynamic_batch in form_dynamic_batches(
            buffer, args.max_batch_tokens, args.max_batch_size
        ):
            completed, failed = encode_with_oom_fallback(
                dynamic_batch, tokenizer, device, timings, batch_stats
            )
            for sample, row in completed:
                samples_by_id[sample["sample_id"]] = sample
                rows_by_id[sample["sample_id"]] = row
                if args.compare_reference:
                    started = time.perf_counter()
                    compare_one(sample, reference[sample["sample_id"]], equivalence)
                    timings.reference_compare_seconds += time.perf_counter() - started
                print(
                    f"[BATCHED {len(samples_by_id)}/{len(selected)}] "
                    f"sample_id={sample['sample_id']} L={sample['length']}"
                )
            for row in failed:
                rows_by_id[row["sample_id"]] = row
                print(f"[FAILED] sample_id={row['sample_id']} reason={row['error']}")
        buffer = []

    tar_checkpoint = time.perf_counter()
    with tarfile.open(tar_path, mode="r:") as archive:
        iterator = iter(archive)
        while pending_members:
            try:
                member = next(iterator)
            except StopIteration:
                break
            now = time.perf_counter()
            timings.tar_scan_seconds += now - tar_checkpoint
            basename = Path(member.name).name
            if basename not in pending_members:
                tar_checkpoint = time.perf_counter()
                continue
            pending_members.remove(basename)
            found += 1
            train = by_member[basename]
            try:
                access_started = time.perf_counter()
                member_file = archive.extractfile(member)
                timings.tar_scan_seconds += time.perf_counter() - access_started
                if member_file is None:
                    raise ValueError("tar member has no extractable content")
                with member_file:
                    buffer.append(
                        prepare_member(
                            member_file,
                            train,
                            originals[train.processed_path],
                            timings,
                        )
                    )
                if len(buffer) >= args.batch_buffer_size:
                    flush_buffer()
            except Exception as exc:
                row = failed_row(train, f"{type(exc).__name__}: {exc}")
                rows_by_id[train.sample_id] = row
                print(f"[FAILED] sample_id={train.sample_id} reason={row['error']}")
                if args.verbose:
                    traceback.print_exc()
            tar_checkpoint = time.perf_counter()
    if buffer:
        flush_buffer()
    for member_name in pending_members:
        train = by_member[member_name]
        rows_by_id[train.sample_id] = failed_row(train, "TAR_MEMBER_NOT_FOUND")

    ordered_samples = [
        samples_by_id[row.sample_id] for row in selected if row.sample_id in samples_by_id
    ]
    shard_paths: list[Path] = []
    for shard_index, offset in enumerate(range(0, len(ordered_samples), args.shard_size)):
        shard_samples = ordered_samples[offset : offset + args.shard_size]
        path = output_dir / f"shard-{shard_index:06d}.pt"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"final shard exists; use --overwrite or a fresh output: {path}")
        atomic_shard_save(shard_samples, path, timings)
        shard_paths.append(path)
        for index, sample in enumerate(shard_samples):
            row = rows_by_id[sample["sample_id"]]
            row["shard_file"] = path.name
            row["index_in_shard"] = index
    manifest_rows = [rows_by_id[row.sample_id] for row in selected]
    atomic_manifest_save(manifest_rows, manifest_path)
    verified_shards = 0
    if args.verify_saved:
        for path in shard_paths:
            verify_shard(path)
            verified_shards += 1

    saved_rows = [row for row in manifest_rows if row["status"] == "SAVED"]
    failed_rows = [row for row in manifest_rows if row["status"] == "FAILED"]
    cropped_residues = int(sum(manifest_values(saved_rows, "length")))
    full_residues = int(sum(float(row["full_length"]) for row in saved_rows))
    total_seconds = time.perf_counter() - total_started
    samples_per_second = len(saved_rows) / total_seconds if total_seconds else 0.0
    cropped_residues_per_second = cropped_residues / total_seconds if total_seconds else 0.0
    full_residues_per_second = full_residues / total_seconds if total_seconds else 0.0
    speedup_total = BASELINE_TOTAL_SECONDS / total_seconds if total_seconds else 0.0
    speedup_samples = samples_per_second / BASELINE_SAMPLES_PER_SECOND
    speedup_residues = cropped_residues_per_second / BASELINE_RESIDUES_PER_SECOND
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    total_shard_bytes = sum(path.stat().st_size for path in shard_paths)
    bytes_per_sample = total_shard_bytes / len(saved_rows) if saved_rows else 0.0
    bytes_per_residue = total_shard_bytes / cropped_residues if cropped_residues else 0.0
    new_estimate_hours = (
        FULL_SCOPE_SAMPLES / samples_per_second / 3600 if samples_per_second else math.inf
    )
    time_saved_hours = OLD_FULL_ESTIMATE_HOURS - new_estimate_hours
    padding_overhead = (
        1.0 - sum(batch_stats.actual_tokens) / sum(batch_stats.padded_tokens)
        if batch_stats.padded_tokens
        else 0.0
    )
    z_cont_mean_diff = (
        equivalence.z_cont_abs_sum / equivalence.z_cont_elements
        if equivalence.z_cont_elements
        else 0.0
    )
    residual_mean_diff = (
        equivalence.residual_abs_sum / equivalence.residual_elements
        if equivalence.residual_elements
        else 0.0
    )
    if not args.compare_reference or equivalence.reference_samples != len(selected):
        equivalence_status = "BATCH_EQUIVALENCE_FAILED"
    elif (
        equivalence.exact_metadata_and_inputs == len(selected)
        and equivalence.exact_struct_ids_current == len(selected)
        and equivalence.z_cont_allclose == len(selected)
        and equivalence.z_quant_allclose == len(selected)
        and equivalence.residual_allclose == len(selected)
    ):
        equivalence_status = "BATCH_EQUIVALENCE_PASS"
    elif equivalence.reference_samples:
        equivalence_status = "BATCH_EQUIVALENCE_PARTIAL"
    else:
        equivalence_status = "BATCH_EQUIVALENCE_FAILED"

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
        "important_processing_order": "batch_pad_then_full_encode_then_remove_padding_then_crop",
        "residual_definition": "z_cont_current - z_quant_current",
        "shard_size": args.shard_size,
        "selected_samples": len(selected),
        "created_shards": len(shard_paths),
        "saved_samples": len(saved_rows),
        "failed_samples": len(failed_rows),
        "torch_version": torch.__version__,
        "git_commit": git_commit(),
        "batching_enabled": True,
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "batch_buffer_size": args.batch_buffer_size,
        "reference_dir": str(reference_dir) if reference_dir else None,
        "reference_equivalence_status": equivalence_status,
        "benchmark_timing_seconds": {
            "tar_scan": timings.tar_scan_seconds,
            "pdb_preprocess": timings.pdb_preprocess_seconds,
            "tokenizer": timings.tokenizer_seconds,
            "shard_save": timings.shard_save_seconds,
            "reference_compare": timings.reference_compare_seconds,
            "total": total_seconds,
        },
        "peak_gpu_memory_mib": {
            "allocated": peak_allocated,
            "reserved": peak_reserved,
        },
    }
    atomic_json_save(meta, meta_path)

    if (
        len(selected) == found == len(saved_rows) == 100
        and not failed_rows
        and len(shard_paths) == verified_shards == 2
        and equivalence_status == "BATCH_EQUIVALENCE_PASS"
        and batch_stats.sizes
    ):
        final_status = "BATCH_BENCHMARK_PASS"
    elif saved_rows:
        final_status = "BATCH_BENCHMARK_PARTIAL"
    else:
        final_status = "BATCH_BENCHMARK_FAILED"

    print("\n" + "=" * 60)
    print("AFDB BATCHED TOKENIZER BENCHMARK")
    print("=" * 60)
    print(f"selected: {len(selected)}")
    print(f"found: {found}")
    print(f"saved: {len(saved_rows)}")
    print(f"failed: {len(failed_rows)}")
    print(f"\nshards: {len(shard_paths)}")
    print(f"verified shards: {verified_shards}")
    print("\nBATCHING")
    print(f"tokenizer batches: {len(batch_stats.sizes)}")
    print(f"mean batch size: {statistics.fmean(batch_stats.sizes):.6f}")
    print(f"max batch size: {max(batch_stats.sizes)}")
    print(f"mean tokens/batch: {statistics.fmean(batch_stats.actual_tokens):.6f}")
    print(f"max tokens/batch: {max(batch_stats.actual_tokens)}")
    print(f"padding overhead: {padding_overhead:.9f}")
    print(f"OOM events: {batch_stats.oom_events}")
    print(f"batch splits: {batch_stats.batch_splits}")
    print("\nREFERENCE EQUIVALENCE")
    print(f"reference samples: {equivalence.reference_samples}")
    print(f"exact struct_ids_current: {equivalence.exact_struct_ids_current}")
    print(f"z_cont allclose: {equivalence.z_cont_allclose}")
    print(f"residual allclose: {equivalence.residual_allclose}")
    print(f"max z_cont abs diff: {equivalence.z_cont_max_diff:.9g}")
    print(f"mean z_cont abs diff: {z_cont_mean_diff:.9g}")
    print(f"max residual abs diff: {equivalence.residual_max_diff:.9g}")
    print(f"mean residual abs diff: {residual_mean_diff:.9g}")
    print(f"ID mismatch samples/tokens/bits: {equivalence.id_mismatch_samples}/{equivalence.id_mismatch_tokens}/{equivalence.id_mismatch_bits}")
    if equivalence.id_mismatch_margins:
        print(f"ID mismatch z_cont margin mean: {statistics.fmean(equivalence.id_mismatch_margins):.9g}")
        print(f"ID mismatch z_cont margin max: {max(equivalence.id_mismatch_margins):.9g}")
    print(f"equivalence status: {equivalence_status}")
    print("\nTIMING")
    print(f"tar scan seconds: {timings.tar_scan_seconds:.6f}")
    print(f"PDB preprocess seconds: {timings.pdb_preprocess_seconds:.6f}")
    print(f"tokenizer seconds: {timings.tokenizer_seconds:.6f}")
    print(f"shard save seconds: {timings.shard_save_seconds:.6f}")
    print(f"reference comparison seconds: {timings.reference_compare_seconds:.6f}")
    print(f"total seconds: {total_seconds:.6f}")
    print(f"samples/sec: {samples_per_second:.9f}")
    print(f"cropped residues/sec: {cropped_residues_per_second:.9f}")
    print(f"FULL residues/sec: {full_residues_per_second:.9f}")
    print(f"baseline total seconds: {BASELINE_TOTAL_SECONDS:.6f}")
    print(f"total speedup: {speedup_total:.9f}")
    print(f"samples/sec speedup: {speedup_samples:.9f}")
    print(f"residues/sec speedup: {speedup_residues:.9f}")
    print("\nGPU MEMORY")
    print(f"peak allocated MiB: {peak_allocated:.6f}")
    print(f"peak reserved MiB: {peak_reserved:.6f}")
    print("\nSTORAGE")
    print(f"total shard MiB: {total_shard_bytes / (1024**2):.6f}")
    print(f"bytes/sample: {bytes_per_sample:.6f}")
    print(f"bytes/residue: {bytes_per_residue:.6f}")
    print("\nFULL BUILD ESTIMATE")
    print(f"target samples: {FULL_SCOPE_SAMPLES}")
    print(f"old estimate hours: {OLD_FULL_ESTIMATE_HOURS:.6f}")
    print(f"new estimate hours: {new_estimate_hours:.6f}")
    print(f"time saved hours: {time_saved_hours:.6f}")
    print("This is a linear extrapolation; TAR ordering, I/O, and GPU load may change actual runtime.")
    print(f"\nmanifest: {manifest_path}")
    print(f"dataset meta: {meta_path}")
    print(f"\nFINAL STATUS: {final_status}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

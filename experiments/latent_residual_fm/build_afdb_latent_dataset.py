#!/usr/bin/env python3
"""Build an AFDB-SwissProt pilot dataset of current quantization residuals.

Coordinates are streamed directly from a local AFDB v4 TAR.  No network
access, model training, hidden-state generation, or persistent raw-PDB
extraction is performed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import random
import statistics
import tarfile
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch


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
    "output_path",
    "error",
)

SAVED_KEYS = {
    "sample_id",
    "processed_path",
    "archive_member",
    "full_length",
    "crop_start",
    "crop_end",
    "length",
    "aatype",
    "struct_ids_current",
    "struct_ids_stored",
    "z_cont",
    "z_quant_current",
    "residual",
    "res_mask",
    "token_accuracy_vs_stored",
    "bit_accuracy_vs_stored",
}


@dataclass(frozen=True)
class TrainRow:
    processed_path: str
    aa_seq: str
    struct_ids: tuple[int, ...]
    plddt: np.ndarray

    @property
    def sample_id(self) -> str:
        return Path(self.processed_path).stem

    @property
    def archive_member(self) -> str:
        return f"{self.sample_id}.pdb.gz"


@dataclass(frozen=True)
class OriginalRow:
    processed_path: str
    aa_seq: str
    struct_ids: tuple[int, ...]
    plddt: np.ndarray
    seq_len: int
    modeled_seq_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar_path", default="data-bin/afdb_v4/swissprot_pdb_v4.tar")
    parser.add_argument("--metadata_csv", default="data-bin/metadata/pdb_afdb_cameo.csv")
    parser.add_argument("--train_dir", default="data-bin/pdb_swissprot/train")
    parser.add_argument("--output_dir", default="data-bin/latent_residual_fm/afdb")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_saved", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def parse_sequence(value: Any) -> str:
    if not nonempty(value):
        raise ValueError("empty sequence")
    return str(value).strip()


def parse_comma_values(value: Any, converter: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [converter(item) for item in value]
    if not nonempty(value):
        raise ValueError("empty comma-separated value")
    return [converter(item) for item in str(value).split(",") if item.strip()]


def parse_metadata_int(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"invalid {name}: {value!r}")
    return int(number)


def iter_parquet_rows(train_dir: Path) -> Iterable[dict[str, Any]]:
    files = sorted(train_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {train_dir}")
    required = ("processed_path", "aa_seq", "struct_seq", "plddt")
    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        missing = set(required).difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{parquet_path} missing columns: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=list(required), batch_size=4096):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                yield {column: values[column][index] for column in required}


def select_train_rows(args: argparse.Namespace) -> list[TrainRow]:
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max_samples must be non-negative or omitted")
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("invalid min/max length range")

    candidates: list[TrainRow] = []
    seen: set[str] = set()
    for raw in iter_parquet_rows(Path(args.train_dir)):
        processed_path = str(raw["processed_path"] or "").strip()
        if not processed_path.startswith("afdb_swissprot/"):
            continue
        if processed_path in seen:
            raise ValueError(f"duplicate AFDB processed_path: {processed_path}")
        seen.add(processed_path)
        aa_seq = parse_sequence(raw["aa_seq"])
        if not (args.min_length <= len(aa_seq) <= args.max_length):
            continue
        struct_ids = tuple(parse_comma_values(raw["struct_seq"], int))
        plddt = np.asarray(parse_comma_values(raw["plddt"], float), dtype=np.float64)
        if not (len(aa_seq) == len(struct_ids) == len(plddt)):
            raise ValueError(f"cropped train lengths disagree: {processed_path}")
        candidates.append(TrainRow(processed_path, aa_seq, struct_ids, plddt))

    candidates.sort(key=lambda row: row.processed_path)
    if args.selection == "random":
        random.Random(args.seed).shuffle(candidates)
    if args.max_samples is not None:
        candidates = candidates[: args.max_samples]
    if not candidates:
        raise ValueError("no AFDB train rows matched the selection")
    return candidates


def load_original_rows(metadata_csv: Path, selected: list[TrainRow]) -> dict[str, OriginalRow]:
    targets = {row.processed_path for row in selected}
    matches: dict[str, list[OriginalRow]] = {path: [] for path in targets}
    required = {
        "processed_path",
        "aa_seq",
        "struct_seq",
        "plddt",
        "seq_len",
        "modeled_seq_len",
    }
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata CSV missing columns: {sorted(missing)}")
        for raw in reader:
            path = str(raw.get("processed_path") or "").strip()
            if path not in targets:
                continue
            matches[path].append(
                OriginalRow(
                    processed_path=path,
                    aa_seq=parse_sequence(raw["aa_seq"]),
                    struct_ids=tuple(parse_comma_values(raw["struct_seq"], int)),
                    plddt=np.asarray(parse_comma_values(raw["plddt"], float), dtype=np.float64),
                    seq_len=parse_metadata_int(raw["seq_len"], "seq_len"),
                    modeled_seq_len=parse_metadata_int(raw["modeled_seq_len"], "modeled_seq_len"),
                )
            )
    bad = {path: len(rows) for path, rows in matches.items() if len(rows) != 1}
    if bad:
        raise ValueError(f"original metadata join is not exactly one-to-one: {bad}")
    return {path: rows[0] for path, rows in matches.items()}


def empty_manifest_row(train: TrainRow, output_path: Path) -> dict[str, Any]:
    return {
        "sample_id": train.sample_id,
        "processed_path": train.processed_path,
        "archive_member": train.archive_member,
        "status": "",
        "full_length": "",
        "crop_start": "",
        "crop_end": "",
        "length": "",
        "token_accuracy_vs_stored": "",
        "bit_accuracy_vs_stored": "",
        "residual_abs_mean": "",
        "residual_abs_max": "",
        "output_path": str(output_path),
        "error": "",
    }


def atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def verify_payload(output_path: Path) -> dict[str, Any]:
    payload = torch.load(output_path, map_location="cpu")
    missing = SAVED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"saved payload missing keys: {sorted(missing)}")
    length = int(payload["length"])
    expected = {
        "aatype": ((length,), torch.long),
        "struct_ids_current": ((length,), torch.long),
        "struct_ids_stored": ((length,), torch.long),
        "z_cont": ((length, 13), torch.float32),
        "z_quant_current": ((length, 13), torch.float32),
        "residual": ((length, 13), torch.float32),
        "res_mask": ((length,), torch.float32),
    }
    for key, (shape, dtype) in expected.items():
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{key} is not a tensor")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{key}: expected shape/dtype {shape}/{dtype}, got {tuple(tensor.shape)}/{tensor.dtype}"
            )
    for key in ("z_cont", "z_quant_current", "residual"):
        if not bool(torch.isfinite(payload[key]).all()):
            raise ValueError(f"{key} contains non-finite values")
    for key in ("struct_ids_current", "struct_ids_stored"):
        ids = payload[key]
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) > 8191):
            raise ValueError(f"{key} outside raw LFQ range 0..8191")
    quant_values = set(payload["z_quant_current"].unique().tolist())
    if not quant_values.issubset({-1.0, 1.0}) or not quant_values:
        raise ValueError(f"unexpected z_quant_current values: {sorted(quant_values)}")
    if not torch.allclose(
        payload["z_quant_current"] + payload["residual"],
        payload["z_cont"],
        atol=1e-7,
        rtol=1e-7,
    ):
        raise ValueError("saved latent reconstruction failed")
    return payload


def process_member(
    member_file: Any,
    train: TrainRow,
    original: OriginalRow,
    tokenizer: Any,
    device: torch.device,
    output_path: Path,
    verify_saved: bool,
) -> dict[str, Any]:
    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.datamodules.pdb_dataset.pdb_datamodule import PdbDataset

    row = empty_manifest_row(train, output_path)
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
        raise ValueError("terminal-cropped original aa_seq does not match train parquet")
    if tuple(original.struct_ids[start:end]) != train.struct_ids:
        raise ValueError("terminal-cropped original struct_seq does not match train parquet")
    if original.plddt[start:end].shape != train.plddt.shape or not np.allclose(
        original.plddt[start:end], train.plddt, atol=1e-3, rtol=0.0
    ):
        raise ValueError("terminal-cropped original pLDDT does not match train parquet")

    processed = PdbDataset.process_chain(raw_chain_feats)
    if int(processed["all_atom_positions"].shape[0]) != full_length:
        raise ValueError("repository processed coordinate length differs from full AFDB length")
    atom_positions = processed["all_atom_positions"].unsqueeze(0).to(device)
    full_mask = processed["res_mask"].unsqueeze(0).to(device)
    seq_length = torch.tensor([full_length], dtype=torch.long, device=device)
    with torch.inference_mode():
        z_cont_full, encoder_feats = tokenizer.encode(
            atom_positions=atom_positions,
            mask=full_mask,
            seq_length=seq_length,
        )
        z_quant_full, _, aux = tokenizer.quantize(
            z_cont_full, mask=full_mask.bool()
        )
        ids_current_full = aux[2]
    if tuple(z_cont_full.shape) != (1, full_length, 13):
        raise ValueError(f"unexpected z_cont_full shape: {tuple(z_cont_full.shape)}")
    if tuple(encoder_feats.shape[:2]) != (1, full_length):
        raise ValueError(f"unexpected encoder feature shape: {tuple(encoder_feats.shape)}")
    if tuple(z_quant_full.shape) != (1, full_length, 13):
        raise ValueError(f"unexpected z_quant_current_full shape: {tuple(z_quant_full.shape)}")
    if tuple(ids_current_full.shape) != (1, full_length):
        raise ValueError(f"unexpected current ID shape: {tuple(ids_current_full.shape)}")

    z_cont = z_cont_full[0, start:end].float().cpu()
    z_quant_current = z_quant_full[0, start:end].float().cpu()
    struct_ids_current = ids_current_full[0, start:end].long().cpu()
    struct_ids_stored = torch.tensor(
        original.struct_ids[start:end], dtype=torch.long
    )
    aatype = processed["aatype"][start:end].long().cpu()
    res_mask = processed["res_mask"][start:end].float().cpu()
    length = end - start
    tensors = (
        aatype,
        struct_ids_current,
        struct_ids_stored,
        z_cont,
        z_quant_current,
        res_mask,
    )
    if any(tensor.shape[0] != length for tensor in tensors):
        raise ValueError("cropped tensor lengths do not align")
    residual = z_cont - z_quant_current
    if tuple(residual.shape) != (length, 13) or not bool(torch.isfinite(residual).all()):
        raise ValueError("invalid current quantization residual")
    if not torch.allclose(
        z_quant_current + residual, z_cont, atol=1e-7, rtol=1e-7
    ):
        raise ValueError("current quantization residual reconstruction failed")

    matches = int((struct_ids_current == struct_ids_stored).sum())
    token_accuracy = matches / length
    different_bits = sum(
        bin(int(current) ^ int(stored)).count("1")
        for current, stored in zip(
            struct_ids_current.tolist(), struct_ids_stored.tolist()
        )
    )
    bit_accuracy = (length * 13 - different_bits) / (length * 13)
    residual_abs_mean = float(residual.abs().mean())
    residual_abs_max = float(residual.abs().max())
    payload = {
        "sample_id": train.sample_id,
        "processed_path": train.processed_path,
        "archive_member": train.archive_member,
        "full_length": full_length,
        "crop_start": start,
        "crop_end": end,
        "length": length,
        "aatype": aatype,
        "struct_ids_current": struct_ids_current,
        "struct_ids_stored": struct_ids_stored,
        "z_cont": z_cont,
        "z_quant_current": z_quant_current,
        "residual": residual,
        "res_mask": res_mask,
        "token_accuracy_vs_stored": token_accuracy,
        "bit_accuracy_vs_stored": bit_accuracy,
    }
    atomic_torch_save(payload, output_path)
    if verify_saved:
        verify_payload(output_path)
    row.update(
        status="SAVED",
        full_length=full_length,
        crop_start=start,
        crop_end=end,
        length=length,
        token_accuracy_vs_stored=token_accuracy,
        bit_accuracy_vs_stored=bit_accuracy,
        residual_abs_mean=residual_abs_mean,
        residual_abs_max=residual_abs_max,
    )
    return row


def shutil_copyfileobj(source: Any, target: Any, chunk_size: int = 1024 * 1024) -> None:
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        target.write(chunk)


def atomic_write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get("status") == "SAVED"]


def format_stat(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.9f}"


def print_summary(
    selected: list[TrainRow],
    found: int,
    rows: list[dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    verify_saved: bool,
) -> str:
    saved_rows = [row for row in rows if row["status"] == "SAVED"]
    skipped = sum(row["status"] == "SKIP_EXISTING" for row in rows)
    failed = sum(row["status"] == "FAILED" for row in rows)
    lengths = numeric(rows, "length")
    token = numeric(rows, "token_accuracy_vs_stored")
    bits = numeric(rows, "bit_accuracy_vs_stored")
    residual_means = numeric(rows, "residual_abs_mean")
    residual_maxes = numeric(rows, "residual_abs_max")
    if (
        len(selected) == 10
        and found == 10
        and len(saved_rows) == 10
        and failed == 0
        and (not verify_saved or len(saved_rows) == 10)
    ):
        status = "PILOT_PASS"
    elif saved_rows or skipped:
        status = "PILOT_PARTIAL"
    else:
        status = "PILOT_FAILED"

    print("\n" + "=" * 60)
    print("AFDB LATENT DATASET BUILD SUMMARY")
    print("=" * 60)
    print(f"selected: {len(selected)}")
    print(f"tar members found: {found}")
    print(f"saved: {len(saved_rows)}")
    print(f"skipped existing: {skipped}")
    print(f"failed: {failed}")
    print(f"\nlength min: {format_stat(min(lengths) if lengths else None)}")
    print(f"length mean: {format_stat(statistics.fmean(lengths) if lengths else None)}")
    print(f"length max: {format_stat(max(lengths) if lengths else None)}")
    print("\ntoken accuracy vs stored:")
    print(f"mean: {format_stat(statistics.fmean(token) if token else None)}")
    print(f"min: {format_stat(min(token) if token else None)}")
    print(f"max: {format_stat(max(token) if token else None)}")
    print("\nbit accuracy vs stored:")
    print(f"mean: {format_stat(statistics.fmean(bits) if bits else None)}")
    print(f"min: {format_stat(min(bits) if bits else None)}")
    print(f"max: {format_stat(max(bits) if bits else None)}")
    print("\nresidual abs mean:")
    print(f"mean across samples: {format_stat(statistics.fmean(residual_means) if residual_means else None)}")
    print("\nresidual abs max:")
    print(f"max across samples: {format_stat(max(residual_maxes) if residual_maxes else None)}")
    print(f"\noutput dir: {output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"\nFINAL STATUS: {status}")
    print("=" * 60)
    return status


def main() -> int:
    args = parse_args()
    tar_path = Path(args.tar_path)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")

    selected = select_train_rows(args)
    originals = load_original_rows(metadata_csv, selected)
    print("SELECTED AFDB TRAIN SAMPLES")
    for index, row in enumerate(selected, start=1):
        print(f"{index}: {row.sample_id} L={len(row.aa_seq)} member={row.archive_member}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    from byprot.models.utils import get_struct_tokenizer

    device = torch.device(args.device)
    tokenizer = get_struct_tokenizer().to(device).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise RuntimeError("structure tokenizer is not fully frozen")

    by_member = {row.archive_member: row for row in selected}
    pending = set(by_member)
    rows_by_path: dict[str, dict[str, Any]] = {}
    found = 0
    saved_counter = 0
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            basename = Path(member.name).name
            if basename not in pending:
                continue
            pending.remove(basename)
            found += 1
            train = by_member[basename]
            output_path = output_dir / f"{train.sample_id}.pt"
            if output_path.exists() and not args.overwrite:
                row = empty_manifest_row(train, output_path)
                row["status"] = "SKIP_EXISTING"
                rows_by_path[train.processed_path] = row
                print(f"[SKIP_EXISTING] sample_id={train.sample_id} path={output_path}")
            else:
                try:
                    member_file = archive.extractfile(member)
                    if member_file is None:
                        raise ValueError("tar member has no extractable file content")
                    with member_file:
                        row = process_member(
                            member_file=member_file,
                            train=train,
                            original=originals[train.processed_path],
                            tokenizer=tokenizer,
                            device=device,
                            output_path=output_path,
                            verify_saved=args.verify_saved,
                        )
                    saved_counter += 1
                    print(
                        f"[SAVED {saved_counter}/{len(selected)}] "
                        f"sample_id={train.sample_id} full_L={row['full_length']} "
                        f"crop={row['crop_start']}:{row['crop_end']} L={row['length']} "
                        f"token_acc={float(row['token_accuracy_vs_stored']):.9f} "
                        f"bit_acc={float(row['bit_accuracy_vs_stored']):.9f} "
                        f"res_abs_mean={float(row['residual_abs_mean']):.9f} "
                        f"res_abs_max={float(row['residual_abs_max']):.9f} path={output_path}"
                    )
                    rows_by_path[train.processed_path] = row
                except Exception as exc:
                    row = empty_manifest_row(train, output_path)
                    row["status"] = "FAILED"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    rows_by_path[train.processed_path] = row
                    print(f"[FAILED] sample_id={train.sample_id} reason={row['error']}")
                    if args.verbose:
                        traceback.print_exc()
            if not pending:
                break

    for member_name in sorted(pending):
        train = by_member[member_name]
        row = empty_manifest_row(train, output_dir / f"{train.sample_id}.pt")
        row["status"] = "FAILED"
        row["error"] = "TAR_MEMBER_NOT_FOUND"
        rows_by_path[train.processed_path] = row
        print(f"[FAILED] sample_id={train.sample_id} reason=TAR_MEMBER_NOT_FOUND")
    manifest_rows = [rows_by_path[row.processed_path] for row in selected]
    atomic_write_manifest(manifest_rows, manifest_path)
    print_summary(
        selected, found, manifest_rows, output_dir, manifest_path, args.verify_saved
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

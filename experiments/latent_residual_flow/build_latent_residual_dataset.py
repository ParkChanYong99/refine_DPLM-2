#!/usr/bin/env python3
"""Build independent StructOK latent-residual training samples from structure PKLs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from byprot.utils.protein import residue_constants
from experiments.latent_residual_flow.tokenizer_adapter import (
    StructureTokenizerAdapter,
)


LATENT_DIM = 13
STRUCT_VOCAB_SIZE = 8192
BUILDER_VERSION = "latent_residual_v1"
RESIDUAL_TOL = 1e-6
TOKEN_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    """Parse dataset-builder CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pkl_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage_dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument(
        "--dataset_role", choices=("train", "val", "test", "debug"), required=True
    )
    parser.add_argument("--allow_eval_as_train", action="store_true")
    parser.add_argument("--save_encoder_feats", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs used by this process."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device, warning and falling back when CUDA is absent."""
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def validate_dataset_role(
    input_pkl_dir: Path, dataset_role: str, allow_eval_as_train: bool
) -> None:
    """Prevent known evaluation directories from silently becoming training data."""
    path_text = str(input_pkl_dir).lower()
    is_known_eval = "cameo2022" in path_text or "pdb_date" in path_text
    if dataset_role == "train" and is_known_eval and not allow_eval_as_train:
        raise RuntimeError(
            "refusing to use a CAMEO2022/PDB_date evaluation directory as train data; "
            "pass --allow_eval_as_train only if this is intentional"
        )


def load_pkl(path: Path) -> Mapping[str, Any]:
    """Load one preprocessed CAMEO/PDB pickle without modifying its contents."""
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def get_backbone_atom_indices() -> Tuple[int, int, int, int]:
    """Return atom37 indices for N, CA, C, O in that exact order."""
    return tuple(residue_constants.atom_order[name] for name in ("N", "CA", "C", "O"))


def make_tokenizer_batch(
    raw: Mapping[str, Any], device: torch.device
) -> Dict[str, Tensor]:
    """Convert one PKL into a tokenizer batch with shapes [1,L,...]."""
    if "atom_positions" not in raw or "atom_mask" not in raw:
        raise KeyError("input must contain atom_positions and atom_mask")
    positions = torch.as_tensor(raw["atom_positions"], dtype=torch.float32)
    atom_mask = torch.as_tensor(raw["atom_mask"], dtype=torch.float32)
    if positions.ndim != 3 or tuple(positions.shape[1:]) != (37, 3):
        raise ValueError(f"atom_positions must have shape [L,37,3], got {positions.shape}")
    length = positions.shape[0]
    if atom_mask.shape != (length, 37):
        raise ValueError(f"atom_mask must have shape [{length},37], got {atom_mask.shape}")

    positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    atom_mask = torch.nan_to_num(atom_mask, nan=0.0, posinf=0.0, neginf=0.0)
    backbone_indices = get_backbone_atom_indices()
    if "bb_mask" in raw:
        res_mask = torch.as_tensor(raw["bb_mask"], dtype=torch.float32)
        if res_mask.shape != (length,):
            raise ValueError(f"bb_mask must have shape [{length}], got {res_mask.shape}")
        res_mask = torch.nan_to_num(res_mask, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        res_mask = atom_mask[:, list(backbone_indices)].bool().all(dim=-1).float()
    res_mask = (res_mask > 0).float()

    aatype = torch.as_tensor(
        raw.get("aatype", torch.zeros(length)), dtype=torch.long
    )
    residue_index = torch.as_tensor(
        raw.get("residue_index", torch.arange(length)), dtype=torch.long
    )
    if aatype.shape != (length,):
        raise ValueError(f"aatype must have shape [{length}], got {aatype.shape}")
    if residue_index.shape != (length,):
        raise ValueError(
            f"residue_index must have shape [{length}], got {residue_index.shape}"
        )
    return {
        "all_atom_positions": positions.unsqueeze(0).to(device),
        "all_atom_mask": atom_mask.unsqueeze(0).to(device),
        "res_mask": res_mask.unsqueeze(0).to(device),
        "seq_length": torch.tensor([length], dtype=torch.long, device=device),
        "aatype": aatype.unsqueeze(0).to(device),
        "residue_index": residue_index.unsqueeze(0).to(device),
    }


def safe_output_name(index: int, source_path: Path) -> str:
    """Create ``{index:06d}_{safe_stem}.pt`` without unsafe path characters."""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._")
    return f"{index:06d}_{safe_stem or 'sample'}.pt"


def _max_abs(tensor: Tensor) -> float:
    return float(tensor.abs().max()) if tensor.numel() else 0.0


def _assert_finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        count = int((~torch.isfinite(tensor)).sum())
        raise ValueError(f"{name} contains {count} NaN/Inf values at valid residues")


def validate_latents(
    batch: Mapping[str, Tensor],
    encoder_feats: Tensor,
    e_cont: Tensor,
    e_quant: Tensor,
    struct_ids: Tensor,
    latent_residual: Tensor,
    x_gt_backbone: Tensor,
    adapter: StructureTokenizerAdapter,
    check_encoder_feats: bool,
) -> Tuple[float, float]:
    """Validate raw [1,L,...] tensors on valid residues before zero masking."""
    length = int(batch["seq_length"].item())
    res_mask = batch["res_mask"]
    expected_prefix = (1, length)
    if e_cont.shape != e_quant.shape:
        raise ValueError(f"e_cont {e_cont.shape} != e_quant {e_quant.shape}")
    if e_cont.shape != (1, length, LATENT_DIM):
        raise ValueError(f"e_cont must be [1,{length},13], got {e_cont.shape}")
    if encoder_feats.ndim != 3 or encoder_feats.shape[:2] != expected_prefix:
        raise ValueError(f"encoder_feats must start with [1,{length}], got {encoder_feats.shape}")
    if struct_ids.shape != res_mask.shape:
        raise ValueError(f"struct_ids {struct_ids.shape} != res_mask {res_mask.shape}")
    if latent_residual.shape != e_cont.shape:
        raise ValueError("latent_residual shape must equal e_cont shape")
    if x_gt_backbone.shape != (1, length, 4, 3):
        raise ValueError(f"x_gt_backbone must be [1,{length},4,3]")
    for key in ("all_atom_positions", "all_atom_mask", "aatype", "residue_index"):
        if batch[key].shape[1] != length:
            raise ValueError(f"{key} residue length does not equal {length}")

    valid2 = res_mask.bool()
    if not bool(valid2.any()):
        raise ValueError("sample contains no valid residues")
    valid3 = valid2.unsqueeze(-1).expand_as(e_cont)
    _assert_finite("e_cont", e_cont[valid3])
    _assert_finite("e_quant", e_quant[valid3])
    _assert_finite("latent_residual", latent_residual[valid3])
    _assert_finite("x_gt_backbone", x_gt_backbone[valid2])
    if check_encoder_feats:
        valid_encoder = valid2.unsqueeze(-1).expand_as(encoder_feats)
        _assert_finite("encoder_feats", encoder_feats[valid_encoder])

    valid_ids = struct_ids[valid2]
    token_min, token_max = int(valid_ids.min()), int(valid_ids.max())
    if token_min < 0 or token_max >= STRUCT_VOCAB_SIZE:
        raise ValueError(f"valid struct_ids outside [0,8191]: min={token_min}, max={token_max}")
    residual_error = _max_abs((e_quant + latent_residual - e_cont)[valid3])
    if residual_error > RESIDUAL_TOL:
        raise ValueError(
            f"residual identity max error {residual_error} exceeds {RESIDUAL_TOL}"
        )
    from_ids = adapter.ids_to_quantized_latent(struct_ids)
    token_error = _max_abs((from_ids - e_quant)[valid3])
    if token_error > TOKEN_TOL:
        raise ValueError(f"token round-trip max error {token_error} exceeds {TOKEN_TOL}")
    return residual_error, token_error


def mask_and_build_sample(
    source_path: Path,
    dataset_role: str,
    storage_dtype: torch.dtype,
    save_encoder_feats: bool,
    batch: Mapping[str, Tensor],
    encoder_feats: Tensor,
    e_cont: Tensor,
    e_quant: Tensor,
    struct_ids: Tensor,
    latent_residual: Tensor,
    x_gt_backbone: Tensor,
    atom_mask_backbone: Tensor,
) -> Dict[str, Any]:
    """Zero invalid residues and construct one CPU sample with unbatched shapes."""
    res_mask = batch["res_mask"].float()
    mask3 = res_mask.unsqueeze(-1)
    mask4 = mask3.unsqueeze(-1)
    masked_encoder = encoder_feats * mask3
    sample: Dict[str, Any] = {
        "source_path": str(source_path),
        "dataset_role": dataset_role,
        "seq_length": int(batch["seq_length"].item()),
        "struct_ids": (struct_ids * res_mask.long())[0].long().cpu(),
        "aatype": (batch["aatype"] * res_mask.long())[0].long().cpu(),
        "residue_index": batch["residue_index"][0].long().cpu(),
        "res_mask": res_mask[0].float().cpu(),
        "e_cont": (e_cont * mask3)[0].to(storage_dtype).cpu(),
        "e_quant": (e_quant * mask3)[0].to(storage_dtype).cpu(),
        "latent_residual": (latent_residual * mask3)[0].to(storage_dtype).cpu(),
        "x_gt_backbone": (x_gt_backbone * mask4)[0].to(storage_dtype).cpu(),
        "atom_mask": (atom_mask_backbone * mask3)[0].float().cpu(),
        "metadata": {
            "latent_dim": LATENT_DIM,
            "struct_vocab_size": STRUCT_VOCAB_SIZE,
            "storage_dtype": str(storage_dtype).removeprefix("torch."),
            "builder_version": BUILDER_VERSION,
        },
    }
    if save_encoder_feats:
        sample["encoder_feats"] = masked_encoder[0].to(storage_dtype).cpu()
    return sample


def atomic_torch_save(sample: Mapping[str, Any], output_path: Path) -> None:
    """Save through a sibling .tmp file and atomically replace the final path."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(dict(sample), temporary_path)
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


@dataclass
class RunningLatentStats:
    """Streaming scalar and per-residue statistics for valid [N,13] residuals."""

    value_sum: float = 0.0
    value_square_sum: float = 0.0
    value_abs_sum: float = 0.0
    value_abs_max: float = 0.0
    value_count: int = 0
    residue_l2_sum: float = 0.0
    residue_count: int = 0

    def update(self, valid_residual: Tensor) -> None:
        """Accumulate one [N_valid,13] tensor without retaining its values."""
        values = valid_residual.detach().double()
        self.value_sum += float(values.sum())
        self.value_square_sum += float(values.square().sum())
        self.value_abs_sum += float(values.abs().sum())
        self.value_abs_max = max(self.value_abs_max, _max_abs(values))
        self.value_count += values.numel()
        self.residue_l2_sum += float(torch.linalg.vector_norm(values, dim=-1).sum())
        self.residue_count += values.shape[0]

    def summary(self) -> Dict[str, float]:
        """Return population statistics accumulated across all valid residues."""
        if self.value_count == 0 or self.residue_count == 0:
            return {
                "latent_residual_mean": 0.0,
                "latent_residual_std": 0.0,
                "latent_residual_abs_mean": 0.0,
                "latent_residual_abs_max": 0.0,
                "latent_residual_l2_mean_per_residue": 0.0,
            }
        mean = self.value_sum / self.value_count
        variance = max(0.0, self.value_square_sum / self.value_count - mean * mean)
        return {
            "latent_residual_mean": mean,
            "latent_residual_std": variance**0.5,
            "latent_residual_abs_mean": self.value_abs_sum / self.value_count,
            "latent_residual_abs_max": self.value_abs_max,
            "latent_residual_l2_mean_per_residue": self.residue_l2_sum
            / self.residue_count,
        }


def print_debug_sample(
    source_path: Path,
    output_path: Path,
    batch: Mapping[str, Tensor],
    encoder_feats: Tensor,
    e_cont: Tensor,
    e_quant: Tensor,
    latent_residual: Tensor,
    struct_ids: Tensor,
    x_gt_backbone: Tensor,
    atom_mask_backbone: Tensor,
    residual_error: float,
    token_error: float,
) -> None:
    """Print shapes, dtypes, ranges, checks, and paths for the first valid sample."""
    def describe(name: str, tensor: Tensor) -> None:
        print(
            f"  {name}: shape={list(tensor.shape)} dtype={tensor.dtype} "
            f"min={float(tensor.min()):.6g} max={float(tensor.max()):.6g}"
        )

    print("\nFirst valid sample")
    print(f"  source path: {source_path}")
    print(f"  seq_length: {int(batch['seq_length'].item())}")
    print(f"  all_atom_positions shape: {list(batch['all_atom_positions'].shape)}")
    print(f"  res_mask shape: {list(batch['res_mask'].shape)}")
    for name, tensor in (
        ("encoder_feats", encoder_feats),
        ("e_cont", e_cont),
        ("e_quant", e_quant),
        ("latent_residual", latent_residual),
        ("struct_ids", struct_ids),
    ):
        describe(name, tensor)
    print(f"  x_gt_backbone shape: {list(x_gt_backbone.shape)}")
    print(f"  atom_mask shape: {list(atom_mask_backbone.shape)}")
    print(f"  residual identity max error: {residual_error:.9g}")
    print(f"  token round-trip max error: {token_error:.9g}")
    print(f"  valid residues: {int(batch['res_mask'].bool().sum())}")
    print(f"  output path: {output_path}")


def write_manifest(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write manifest.tsv with one status row per considered source PKL."""
    fields = ("source_path", "status", "output_path", "seq_length", "error_message")
    path = output_dir / "manifest.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    args: argparse.Namespace,
    num_found: int,
    manifest: Sequence[Mapping[str, Any]],
    lengths: Sequence[int],
    total_valid_residues: int,
    latent_stats: RunningLatentStats,
    unique_tokens: Tensor,
) -> Dict[str, Any]:
    """Build the requested JSON-serializable dataset summary."""
    counts = {
        status: sum(row["status"] == status for row in manifest)
        for status in ("ok", "skip_length", "skip_existing", "fail")
    }
    if lengths:
        mean_length = sum(lengths) / len(lengths)
        median_length = median(lengths)
        min_length, max_length = min(lengths), max(lengths)
    else:
        mean_length = median_length = min_length = max_length = 0
    unique_count = int(unique_tokens.sum())
    summary: Dict[str, Any] = {
        "dataset_role": args.dataset_role,
        "input_pkl_dir": str(args.input_pkl_dir),
        "output_dir": str(args.output_dir),
        "num_found": num_found,
        "num_ok": counts["ok"],
        "num_skip_length": counts["skip_length"],
        "num_skip_existing": counts["skip_existing"],
        "num_fail": counts["fail"],
        "storage_dtype": args.storage_dtype,
        "save_encoder_feats": args.save_encoder_feats,
        "total_valid_residues": total_valid_residues,
        "mean_sequence_length": mean_length,
        "median_sequence_length": median_length,
        "min_sequence_length": min_length,
        "max_sequence_length": max_length,
        **latent_stats.summary(),
        "unique_struct_token_count": unique_count,
        "struct_token_utilization_ratio": unique_count / STRUCT_VOCAB_SIZE,
    }
    return summary


@torch.no_grad()
def main() -> None:
    """Convert selected PKLs to independently stored latent-residual samples."""
    args = parse_args()
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num_samples must be positive or omitted")
    if args.max_length < 1:
        raise ValueError("--max_length must be at least 1")
    if not args.input_pkl_dir.is_dir():
        raise NotADirectoryError(args.input_pkl_dir)
    validate_dataset_role(
        args.input_pkl_dir, args.dataset_role, args.allow_eval_as_train
    )
    set_deterministic_seed(args.seed)
    device = resolve_device(args.device)
    storage_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
    }[args.storage_dtype]

    found_paths = sorted(path for path in args.input_pkl_dir.glob(args.pattern) if path.is_file())
    num_found = len(found_paths)
    paths = found_paths[: args.num_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = StructureTokenizerAdapter.load_struct_tokenizer(device)
    adapter.struct_tokenizer.eval()
    for parameter in adapter.struct_tokenizer.parameters():
        parameter.requires_grad_(False)

    manifest: List[Dict[str, Any]] = []
    successful_lengths: List[int] = []
    total_valid_residues = 0
    latent_stats = RunningLatentStats()
    unique_tokens = torch.zeros(STRUCT_VOCAB_SIZE, dtype=torch.bool)
    debug_printed = False
    backbone_indices = list(get_backbone_atom_indices())

    for index, source_path in enumerate(tqdm(paths, desc="latent residuals")):
        output_path = args.output_dir / safe_output_name(index, source_path)
        row: Dict[str, Any] = {
            "source_path": str(source_path),
            "status": "fail",
            "output_path": str(output_path),
            "seq_length": "",
            "error_message": "",
        }
        try:
            raw = load_pkl(source_path)
            if "atom_positions" not in raw:
                raise KeyError("input is missing atom_positions")
            length = int(torch.as_tensor(raw["atom_positions"]).shape[0])
            row["seq_length"] = length
            if length < 1 or length > args.max_length:
                row["status"] = "skip_length"
                manifest.append(row)
                continue
            if output_path.exists() and not args.overwrite:
                row["status"] = "skip_existing"
                manifest.append(row)
                continue

            batch = make_tokenizer_batch(raw, device)
            e_cont, encoder_feats = adapter.encode_continuous(
                batch["all_atom_positions"], batch["res_mask"], batch["seq_length"]
            )
            e_quant, struct_ids = adapter.quantize(e_cont, batch["res_mask"])
            latent_residual = e_cont - e_quant
            x_gt_backbone = batch["all_atom_positions"][:, :, backbone_indices, :]
            atom_mask_backbone = batch["all_atom_mask"][:, :, backbone_indices]
            residual_error, token_error = validate_latents(
                batch,
                encoder_feats,
                e_cont,
                e_quant,
                struct_ids,
                latent_residual,
                x_gt_backbone,
                adapter,
                args.save_encoder_feats,
            )

            valid = batch["res_mask"][0].bool()
            valid_residual = latent_residual[0, valid]
            valid_ids = struct_ids[0, valid].detach().cpu()
            sample = mask_and_build_sample(
                source_path,
                args.dataset_role,
                storage_dtype,
                args.save_encoder_feats,
                batch,
                encoder_feats,
                e_cont,
                e_quant,
                struct_ids,
                latent_residual,
                x_gt_backbone,
                atom_mask_backbone,
            )
            atomic_torch_save(sample, output_path)
            latent_stats.update(valid_residual)
            unique_tokens[valid_ids] = True
            valid_count = int(valid.sum())
            total_valid_residues += valid_count
            successful_lengths.append(length)
            row["status"] = "ok"
            manifest.append(row)

            if args.debug and not debug_printed:
                print_debug_sample(
                    source_path,
                    output_path,
                    batch,
                    encoder_feats,
                    e_cont,
                    e_quant,
                    latent_residual,
                    struct_ids,
                    x_gt_backbone,
                    atom_mask_backbone,
                    residual_error,
                    token_error,
                )
                debug_printed = True
        except Exception as error:
            row["status"] = "fail"
            row["error_message"] = f"{type(error).__name__}: {error}"
            manifest.append(row)
            tqdm.write(f"FAIL {source_path}: {row['error_message']}")

    write_manifest(args.output_dir, manifest)
    summary = build_summary(
        args,
        num_found,
        manifest,
        successful_lengths,
        total_valid_residues,
        latent_stats,
        unique_tokens,
    )
    with (args.output_dir / "dataset_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

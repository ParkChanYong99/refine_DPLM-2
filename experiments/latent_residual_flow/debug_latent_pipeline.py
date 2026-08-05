#!/usr/bin/env python3
"""Read-only oracle experiment for StructOK latent residual interpolation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from build_residual_dataset import (
    get_backbone_atom_indices,
    load_pkl,
    make_batch_from_pkl,
)
from tokenizer_adapter import StructureTokenizerAdapter, load_struct_tokenizer


RESIDUAL_TOL = 1e-6
TOKEN_TOL = 1e-6
DECODER_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_pkl_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_tsv", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return parser.parse_args()


def _finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        bad = int((~torch.isfinite(tensor)).sum())
        raise RuntimeError(f"hard check failed: {name} contains {bad} NaN/Inf values")


def _max_abs(x: Tensor) -> float:
    return float(x.abs().max()) if x.numel() else 0.0


def _rmsd(diff: Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.sum(diff.float().square(), dim=-1))))


def _kabsch_transform(pred_ca: Tensor, target_ca: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return rotation and centroids mapping prediction onto target using CA atoms."""
    if pred_ca.shape[0] < 3:
        raise RuntimeError("at least three valid CA residues are required for Kabsch alignment")
    pred64, target64 = pred_ca.double(), target_ca.double()
    pred_center, target_center = pred64.mean(0), target64.mean(0)
    covariance = (pred64 - pred_center).T @ (target64 - target_center)
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, dtype=torch.float64, device=pred_ca.device)
    correction[-1, -1] = torch.sign(torch.det(u @ vh))
    rotation = u @ correction @ vh
    return rotation, pred_center, target_center


def _coordinate_metrics(
    pred_atom37: Tensor,
    gt_atom37: Tensor,
    gt_atom_mask: Tensor,
    backbone_indices: Sequence[int],
) -> Dict[str, float]:
    n_idx, ca_idx, c_idx, o_idx = [int(x) for x in backbone_indices]
    bb_idx = torch.tensor([n_idx, ca_idx, c_idx, o_idx], device=pred_atom37.device)
    residue_valid = gt_atom_mask[:, bb_idx].bool().all(dim=-1)
    if int(residue_valid.sum()) < 3:
        raise RuntimeError("sample has fewer than three residues with valid N/CA/C/O atoms")
    pred_bb = pred_atom37[residue_valid][:, bb_idx]
    gt_bb = gt_atom37[residue_valid][:, bb_idx]
    pred_ca, gt_ca = pred_bb[:, 1], gt_bb[:, 1]
    rotation, pred_center, gt_center = _kabsch_transform(pred_ca, gt_ca)
    aligned_bb = (pred_bb.double() - pred_center) @ rotation + gt_center
    aligned_ca = aligned_bb[:, 1]
    return {
        "aligned_backbone_rmsd": _rmsd(aligned_bb - gt_bb.double()),
        "aligned_ca_rmsd": _rmsd(aligned_ca - gt_ca.double()),
        "nonaligned_backbone_rmsd": _rmsd(pred_bb - gt_bb),
        "nonaligned_ca_rmsd": _rmsd(pred_ca - gt_ca),
    }


def _shape_dtype(tensor: Tensor) -> str:
    return f"shape={list(tensor.shape)} dtype={tensor.dtype}"


def _range(tensor: Tensor) -> str:
    if not tensor.numel():
        return "empty"
    return f"min={float(tensor.min()):.6g} max={float(tensor.max()):.6g}"


def _gt_tensors(raw: Dict[str, Any], device: torch.device) -> Tuple[Tensor, Tensor]:
    positions = torch.as_tensor(raw["atom_positions"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(raw["atom_mask"], dtype=torch.float32, device=device)
    return positions, mask


def _append(rows: List[Dict[str, Any]], row_type: str, **values: Any) -> None:
    rows.append({"row_type": row_type, **values})


@torch.no_grad()
def evaluate_sample(
    path: Path,
    raw: Dict[str, Any],
    batch: Dict[str, Tensor],
    adapter: StructureTokenizerAdapter,
    alphas: Sequence[float],
    backbone_indices: Sequence[int],
    first: bool,
) -> Tuple[List[Dict[str, Any]], Dict[float, Dict[str, float]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    e_cont, encoder_feats = adapter.encode_continuous(
        batch["all_atom_positions"], batch["res_mask"], batch["seq_length"]
    )
    e_quant, struct_ids = adapter.quantize(e_cont, batch["res_mask"])
    residual = e_cont - e_quant
    e_quant_from_ids = adapter.ids_to_quantized_latent(struct_ids)
    quant_positions, quant_mask = adapter.decode_latent(e_quant, batch["res_mask"])
    id_decoded = adapter.struct_tokenizer.detokenize(struct_ids, res_mask=batch["res_mask"])
    id_positions = id_decoded["atom37_positions"]

    tensors = {
        "encoder_feats": encoder_feats,
        "e_cont": e_cont,
        "e_quant": e_quant,
        "struct_ids": struct_ids,
        "latent_residual": residual,
        "decoded_atom37_positions": quant_positions,
        "decoded_atom37_mask": quant_mask,
    }
    for name, tensor in tensors.items():
        _append(rows, "tensor", sample=path.name, metric=name, value=_shape_dtype(tensor))
    for name in ("e_cont", "e_quant", "latent_residual", "decoded_atom37_positions"):
        _finite(name, tensors[name])
    _finite("coordinates decoded from struct_ids", id_positions)

    valid = batch["res_mask"].bool().unsqueeze(-1).expand_as(e_quant)
    residual_error = _max_abs((e_quant + residual) - e_cont)
    token_error = _max_abs((e_quant_from_ids - e_quant)[valid])
    decoder_diff = id_positions - quant_positions
    decoder_max = _max_abs(decoder_diff)
    decoder_rmsd = _rmsd(decoder_diff.reshape(-1, 3))
    checks = {
        "residual_identity_max_error": residual_error,
        "token_roundtrip_max_error": token_error,
        "decoder_equivalence_max_error": decoder_max,
        "decoder_equivalence_rmsd": decoder_rmsd,
    }
    if residual_error > RESIDUAL_TOL:
        raise RuntimeError(f"hard check failed: residual identity error {residual_error} > {RESIDUAL_TOL}")
    if token_error > TOKEN_TOL:
        raise RuntimeError(f"hard check failed: token round-trip error {token_error} > {TOKEN_TOL}")
    if decoder_max > DECODER_TOL:
        raise RuntimeError(f"hard check failed: decoder equivalence error {decoder_max} > {DECODER_TOL}")
    for metric, value in checks.items():
        _append(rows, "check", sample=path.name, metric=metric, value=value)

    if first:
        print(f"\nFirst sample: {path}")
        for name, tensor in tensors.items():
            print(f"  {name}: {_shape_dtype(tensor)} {_range(tensor)}")

    gt_positions, gt_mask = _gt_tensors(raw, e_cont.device)
    alpha_metrics: Dict[float, Dict[str, float]] = {}
    for alpha in alphas:
        refined = e_quant + float(alpha) * residual
        decoded_positions, _ = adapter.decode_latent(refined, batch["res_mask"])
        _finite(f"decoded coordinates at alpha={alpha}", decoded_positions)
        metrics = _coordinate_metrics(
            decoded_positions[0], gt_positions, gt_mask, backbone_indices
        )
        alpha_metrics[float(alpha)] = metrics
        _append(rows, "sample_alpha", sample=path.name, alpha=float(alpha), **metrics)

    if 0.0 not in alpha_metrics or 1.0 not in alpha_metrics:
        raise ValueError("--alphas must include both 0 and 1 for requested improvements")
    best_alpha = min(alpha_metrics, key=lambda a: alpha_metrics[a]["aligned_ca_rmsd"])
    baseline = alpha_metrics[0.0]["aligned_ca_rmsd"]
    sample_summary = {
        "best_alpha": best_alpha,
        "best_aligned_ca_rmsd": alpha_metrics[best_alpha]["aligned_ca_rmsd"],
        "best_improvement_vs_alpha0": baseline - alpha_metrics[best_alpha]["aligned_ca_rmsd"],
        "alpha1_improvement_vs_alpha0": baseline - alpha_metrics[1.0]["aligned_ca_rmsd"],
    }
    _append(rows, "sample_summary", sample=path.name, **sample_summary)
    print(f"\n{path.name}: best alpha={best_alpha:g}, CA RMSD={sample_summary['best_aligned_ca_rmsd']:.4f}")
    for alpha in alphas:
        m = alpha_metrics[float(alpha)]
        print(f"  alpha={alpha:g} aligned BB/CA={m['aligned_backbone_rmsd']:.4f}/{m['aligned_ca_rmsd']:.4f} nonaligned BB/CA={m['nonaligned_backbone_rmsd']:.4f}/{m['nonaligned_ca_rmsd']:.4f}")
    return rows, alpha_metrics, sample_summary


def write_tsv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}, key=lambda x: (x != "row_type", x))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas must not contain duplicates")
    paths = sorted(args.input_pkl_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no .pkl files found in {args.input_pkl_dir}")

    device = torch.device(args.device)
    adapter = load_struct_tokenizer(device)
    backbone_indices = get_backbone_atom_indices()
    if len(backbone_indices) != 4:
        raise RuntimeError(f"expected N/CA/C/O indices, got {backbone_indices}")

    rows: List[Dict[str, Any]] = []
    all_metrics: Dict[float, List[Dict[str, float]]] = {float(a): [] for a in args.alphas}
    summaries: List[Dict[str, float]] = []
    processed = 0
    for path in paths:
        raw = load_pkl(path)
        length = int(torch.as_tensor(raw["atom_positions"]).shape[0])
        if args.max_length is not None and length > args.max_length:
            continue
        batch = make_batch_from_pkl(raw, device)
        sample_rows, metrics, summary = evaluate_sample(
            path, raw, batch, adapter, args.alphas, backbone_indices, processed == 0
        )
        rows.extend(sample_rows)
        summaries.append(summary)
        for alpha, values in metrics.items():
            all_metrics[alpha].append(values)
        processed += 1
        if args.num_samples is not None and processed >= args.num_samples:
            break
    if processed == 0:
        raise RuntimeError("no samples satisfied --max_length")

    print("\nOverall summary")
    for alpha in args.alphas:
        values = all_metrics[float(alpha)]
        mean_bb = sum(x["aligned_backbone_rmsd"] for x in values) / len(values)
        ca_values = [x["aligned_ca_rmsd"] for x in values]
        mean_ca = sum(ca_values) / len(ca_values)
        median_ca = median(ca_values)
        baseline_values = all_metrics[0.0]
        improved = sum(
            x["aligned_ca_rmsd"] < b["aligned_ca_rmsd"]
            for x, b in zip(values, baseline_values)
        ) / len(values)
        print(f"  alpha={alpha:g}: mean aligned BB={mean_bb:.4f}, mean/median aligned CA={mean_ca:.4f}/{median_ca:.4f}, improved vs alpha=0={improved:.1%}")
        _append(
            rows, "overall_alpha", alpha=float(alpha),
            mean_aligned_backbone_rmsd=mean_bb,
            mean_aligned_ca_rmsd=mean_ca,
            median_aligned_ca_rmsd=median_ca,
            improved_sample_fraction_vs_alpha0=improved,
            num_samples=processed,
        )
    for metric in ("best_improvement_vs_alpha0", "alpha1_improvement_vs_alpha0"):
        values = [x[metric] for x in summaries]
        _append(rows, "overall_summary", metric=metric, mean=sum(values) / len(values), median=median(values), num_samples=processed)
    write_tsv(args.output_tsv, rows)
    print(f"\nWrote {len(rows)} rows to {args.output_tsv}")


if __name__ == "__main__":
    main()

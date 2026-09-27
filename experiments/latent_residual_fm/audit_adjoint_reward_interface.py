#!/usr/bin/env python3
"""Read-only terminal-residual autograd audit on one short internal-val sample.

This checks the reward interface only. It does not implement Adjoint Matching,
train a model, compute final metrics, or write any artifact.
"""

from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

import torch

from byprot.datamodules.pdb_dataset import residue_constants
from byprot.datamodules.pdb_dataset.utils import align_structures
from experiments.latent_residual_fm.debug_interfaces import require, resolve_device, seed_everything
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    load_fm,
    load_validation_sample,
)
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.flow_matching import euler_sample


ALPHA = 0.5
BACKBONE = tuple(residue_constants.atom_order[name] for name in ("N", "CA", "C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded")
    parser.add_argument("--split_csv", default="data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv")
    parser.add_argument("--checkpoint", default="experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt")
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_flow_steps", type=int, default=100)
    return parser.parse_args()


def shortest_val_index(split_csv: Path) -> int:
    with split_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    require(bool(rows), "validation split is empty")
    # Stable tie break: first occurrence in the existing split CSV.
    return min(range(len(rows)), key=lambda index: (int(rows[index]["length"]), index))


def gradient_report(name: str, loss: torch.Tensor, terminal: torch.Tensor, mask: torch.Tensor, *, retain_graph: bool = False) -> bool:
    print(f"{name}:")
    print(f"  loss: {float(loss.detach()):.9g}")
    loss_finite = bool(torch.isfinite(loss.detach()).all())
    print(f"  loss finite: {loss_finite}")
    try:
        grad = torch.autograd.grad(loss, terminal, allow_unused=True, retain_graph=retain_graph)[0]
    except (RuntimeError, ValueError) as exc:
        grad = None
        print(f"  autograd error: {type(exc).__name__}: {exc}")
    print(f"  grad exists: {grad is not None}")
    if grad is None:
        print("  grad shape: None\n  grad finite: False\n  grad abs mean: NA")
        print("  grad norm: NA\n  nonzero fraction: NA\n  NaN count: NA\n  Inf count: NA")
        print("  status: FAIL")
        return False
    valid = mask.bool().unsqueeze(-1).expand_as(grad)
    values = grad[valid]
    finite = bool(torch.isfinite(grad).all())
    nan_count = int(torch.isnan(grad).sum())
    inf_count = int(torch.isinf(grad).sum())
    abs_mean = float(values.abs().mean())
    norm = float(torch.linalg.vector_norm(values))
    nonzero = int((values != 0).sum()) / values.numel()
    passed = loss_finite and finite and nonzero > 0.0 and norm > 0.0
    print(f"  grad shape: {tuple(grad.shape)}")
    print(f"  grad finite: {finite}")
    print(f"  grad abs mean: {abs_mean:.9g}")
    print(f"  grad norm: {norm:.9g}")
    print(f"  nonzero fraction: {nonzero:.9g}")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    return passed


def coordinate_losses(decoded: dict, oracle: dict, res_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    positions = decoded["atom37_positions"]
    target = oracle["atom37_positions"]
    require(positions.shape == target.shape, "decoder/oracle atom37 shape mismatch")
    require(positions.ndim == 4 and positions.shape[-2:] == (37, 3), "invalid atom37 coordinates")
    valid = res_mask.bool().unsqueeze(-1)
    valid = valid & decoded["atom37_mask"].bool() & oracle["atom37_mask"].bool()
    valid = valid[:, :, BACKBONE]
    count = int(valid.sum())
    require(count >= 9, "fewer than three valid backbone residues")
    moving = positions[:, :, BACKBONE, :][valid]
    reference = target[:, :, BACKBONE, :][valid]
    require(bool(torch.isfinite(moving).all() and torch.isfinite(reference).all()), "backbone coordinates are non-finite")
    coord_mse = (moving - reference).square().mean()

    # Use the repository's Kabsch implementation exactly as provided. Its
    # @torch.no_grad decorator is expected to disconnect this candidate.
    batch_indices = torch.zeros(count, device=moving.device, dtype=torch.long)
    aligned, centered_reference, _ = align_structures(
        moving.double(), batch_indices, reference.double()
    )
    aligned_mse = (aligned - centered_reference).square().sum(dim=-1).mean()
    return coord_mse, aligned_mse, count


def audit_terminal(label: str, residual: torch.Tensor, z_quant: torch.Tensor, z_cont: torch.Tensor,
                   res_mask: torch.Tensor, oracle: dict, tokenizer: torch.nn.Module) -> tuple[bool, bool, bool]:
    terminal = residual.detach().clone().requires_grad_(True)
    refined = z_quant + ALPHA * terminal
    denominator = res_mask.sum() * refined.shape[-1]
    require(bool(denominator > 0), "empty latent mask")
    latent_loss = (((refined - z_cont).square()) * res_mask[..., None]).sum() / denominator
    print(f"\nterminal state: {label}")
    latent_ok = gradient_report("latent reward", latent_loss, terminal, res_mask, retain_graph=True)

    # Decoder parameters are frozen, but its forward MUST run with autograd on.
    require(torch.is_grad_enabled(), "decoder forward unexpectedly has grad disabled")
    decoded = tokenizer.detokenize(refined, res_mask=res_mask)
    coord_loss, aligned_loss, atom_count = coordinate_losses(decoded, oracle, res_mask)
    print(f"  valid backbone atoms: {atom_count}")
    coord_ok = gradient_report("coordinate reward", coord_loss, terminal, res_mask)
    aligned_ok = gradient_report("aligned reward", aligned_loss, terminal, res_mask)
    return latent_ok, coord_ok, aligned_ok


def main() -> int:
    args = parse_args()
    require(args.num_flow_steps > 0, "num_flow_steps must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print("ADJOINT_REWARD_INTERFACE_AUDIT", flush=True)
    stage = "load validation sample"
    sample_id = "unknown"
    length = -1
    try:
        val_index = shortest_val_index(Path(args.split_csv))
        sample = load_validation_sample(Path(args.dataset_dir), Path(args.split_csv), val_index)
        sample_id = str(sample["sample_id"])
        length = int(sample["length"])
        print(f"sample: {sample_id} (val_index={val_index})", flush=True)
        print(f"length: {length}", flush=True)
        print("condition: Mode A teacher-forced stored current LFQ IDs")
        print(f"alpha: {ALPHA}")
        z_cont = sample["z_cont"].float().reshape(1, length, 13).to(device)
        z_quant = sample["z_quant_current"].float().reshape(1, length, 13).to(device)
        res_mask = sample["res_mask"].float().reshape(1, length).to(device)
        require(bool(torch.isfinite(z_cont).all() and torch.isfinite(z_quant).all()), "stored latent is non-finite")
        require(bool(res_mask.bool().any()), "stored res_mask is empty")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        stage = "load frozen DPLM/tokenizer"
        extractor = DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        model = extractor.model
        tokenizer = model.struct_tokenizer
        model.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        dplm_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokenizer_trainable = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        assert dplm_trainable == 0, "DPLM has trainable parameters"
        assert tokenizer_trainable == 0, "structure tokenizer has trainable parameters"
        frozen_ok = True
        print(f"frozen parameter audit: PASS (DPLM={dplm_trainable}, tokenizer={tokenizer_trainable})")

        stage = "load finalized FM checkpoint"
        fm, step, _, _ = load_fm(Path(args.checkpoint), device)
        assert sum(p.numel() for p in fm.parameters() if p.requires_grad) == 0
        print(f"FM checkpoint optimizer_step: {step}; FM trainable parameters: 0")
        stage = "construct Mode A condition and sample frozen FM"
        aatype = sample["aatype"].long().reshape(1, length)
        struct_ids = sample["struct_ids_current"].long().reshape(1, length)
        with torch.no_grad():
            hidden = extractor(aatype, struct_ids, res_mask)["hidden_states"]
            require(len(hidden) == 33, "expected 33 FM condition states")
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn(z_quant.shape, device=device, dtype=z_quant.dtype, generator=generator)
            fm_residual = euler_sample(fm, z_quant, hidden, res_mask, args.num_flow_steps, initial_noise=noise)
        del hidden, fm, extractor, noise
        require(bool(torch.isfinite(fm_residual).all()), "FM terminal residual is non-finite")
        gt_residual = z_cont - z_quant

        stage = "decode oracle continuous latent"
        with torch.no_grad():
            oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)
        stage = "differentiate reward candidates"
        results = {}
        for label, residual in (("X1_gt", gt_residual), ("X1_fm", fm_residual)):
            results[label] = audit_terminal(label, residual, z_quant, z_cont, res_mask, oracle, tokenizer)
        primary_ok = all(latent and coord for latent, coord, _ in results.values())
        print(f"\ndecoder gradient path: {'PASS' if all(r[1] for r in results.values()) else 'FAIL'}")
        print("aligned path note: repository align_structures is decorated with @torch.no_grad() ")
        print("  at src/byprot/datamodules/pdb_dataset/utils.py:445; SVD backward is unreachable.")
        print(f"frozen parameter audit: {'PASS' if frozen_ok else 'FAIL'}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            print(f"peak CUDA allocated MiB: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")
            print(f"peak CUDA reserved MiB: {torch.cuda.max_memory_reserved(device) / 1024**2:.2f}")
        else:
            print("peak CUDA allocated MiB: NA\npeak CUDA reserved MiB: NA")
        print("overall requires both terminal states to pass latent and coordinate rewards; aligned reward is optional.")
        print(f"overall: {'ADJOINT_REWARD_INTERFACE_PASS' if primary_ok and frozen_ok else 'ADJOINT_REWARD_INTERFACE_FAIL'}")
        print("ADJOINT_REWARD_INTERFACE_PASS" if primary_ok and frozen_ok else "ADJOINT_REWARD_INTERFACE_FAIL")
        return 0 if primary_ok and frozen_ok else 1
    except Exception as exc:
        print(f"failure point: {stage}; sample={sample_id}; length={length}")
        print(f"error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if device.type == "cuda":
            print(f"peak CUDA allocated MiB: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")
            print(f"peak CUDA reserved MiB: {torch.cuda.max_memory_reserved(device) / 1024**2:.2f}")
        print("overall: ADJOINT_REWARD_INTERFACE_FAIL")
        print("ADJOINT_REWARD_INTERFACE_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit an experiment-local differentiable aligned terminal reward.

One internal validation sample, frozen finalized FM/DPLM/tokenizer, no training.
The proposed reward would be -lambda_reward * aligned_squared_rmsd; this audit
does not select lambda_reward or implement Adjoint Matching.
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import torch

from byprot.datamodules.pdb_dataset import residue_constants
from experiments.latent_residual_fm.debug_interfaces import require, resolve_device, seed_everything
from experiments.latent_residual_fm.debug_prediction_time_fm import load_fm, load_validation_sample
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.flow_matching import euler_sample


SAMPLE_ID = "AF-Q4UJC4-F1-model_v4"
VAL_INDEX = 1426  # The shortest validation item in split_v1.csv; see prior audit.
LENGTH = 57
ALPHA = 0.5
BACKBONE = tuple(residue_constants.atom_order[name] for name in ("N", "CA", "C"))
PERTURBATION_SCALE = 1e-3
NUM_STABILITY_TRIALS = 5


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


def backbone_points(decoded: dict[str, torch.Tensor], oracle: dict[str, torch.Tensor],
                    res_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = decoded["atom37_positions"]
    target = oracle["atom37_positions"]
    require(positions.shape == target.shape, "prediction/oracle atom37 shapes differ")
    require(positions.ndim == 4 and positions.shape[-2:] == (37, 3), "invalid atom37 positions")
    require(res_mask.shape == positions.shape[:2], "residue mask is misaligned")
    pred_backbone = positions[:, :, BACKBONE, :]
    oracle_backbone = target[:, :, BACKBONE, :]
    valid = (res_mask.bool().unsqueeze(-1)
             & decoded["atom37_mask"][:, :, BACKBONE].bool()
             & oracle["atom37_mask"][:, :, BACKBONE].bool())
    require(bool(valid.any()), "no valid backbone atoms")
    require(bool(torch.isfinite(pred_backbone[valid]).all()), "predicted backbone is non-finite")
    require(bool(torch.isfinite(oracle_backbone[valid]).all()), "oracle backbone is non-finite")
    return pred_backbone, oracle_backbone, valid


def aligned_squared_rmsd(pred: torch.Tensor, oracle: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
    """Proper-rotation Kabsch; mean squared 3D distance per valid backbone atom.

    All coordinate arithmetic, including SVD and reflection correction, stays
    in PyTorch/autograd. Use float64 for the small 3x3 covariance and backward.
    """
    require(pred.shape == oracle.shape and pred.shape[-2:] == (3, 3), "bad backbone shapes")
    require(valid.shape == pred.shape[:-1], "bad atom mask shape")
    squared_sum = pred.new_zeros((), dtype=torch.float64)
    atom_count = 0
    for batch in range(pred.shape[0]):
        moving = pred[batch][valid[batch]].double()
        target = oracle[batch][valid[batch]].double()
        count = int(moving.shape[0])
        require(count >= 3, "Kabsch needs at least three valid backbone atoms")
        moving_centered = moving - moving.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        covariance = moving_centered.transpose(0, 1) @ target_centered
        u, _, vh = torch.linalg.svd(covariance)
        determinant = torch.linalg.det(u @ vh)
        one = torch.ones((), dtype=covariance.dtype, device=covariance.device)
        handedness = torch.where(determinant < 0, -one, one)
        correction = torch.diag(torch.stack((one, one, handedness)))
        rotation = u @ correction @ vh
        aligned = moving_centered @ rotation
        squared_sum = squared_sum + (aligned - target_centered).square().sum()
        atom_count += count
    return squared_sum / atom_count


def report_gradient(label: str, loss: torch.Tensor, terminal: torch.Tensor,
                    res_mask: torch.Tensor) -> bool:
    print(f"\n{label} residual:")
    loss_finite = bool(torch.isfinite(loss).all())
    print(f"  aligned loss: {float(loss.detach()):.9g}")
    print(f"  loss finite: {loss_finite}")
    print(f"  aligned RMSD (report only): {math.sqrt(max(0.0, float(loss.detach()))):.9g}")
    try:
        grad = torch.autograd.grad(loss, terminal, allow_unused=True)[0]
    except (RuntimeError, ValueError) as exc:
        grad = None
        print(f"  autograd error: {type(exc).__name__}: {exc}")
    print(f"  grad exists: {grad is not None}")
    if grad is None:
        print("  grad shape: None\n  grad finite: False\n  grad abs mean: NA")
        print("  grad norm: NA\n  nonzero fraction: NA\n  NaN count: NA\n  Inf count: NA")
        print("  status: FAIL")
        return False
    valid = res_mask.bool().unsqueeze(-1).expand_as(grad)
    values = grad[valid]
    grad_finite = bool(torch.isfinite(grad).all())
    nonzero = int((values != 0).sum()) / values.numel()
    norm = float(torch.linalg.vector_norm(values))
    passed = (loss_finite and grad_finite and tuple(grad.shape) == (1, LENGTH, 13)
              and norm > 0.0 and nonzero > 0.0)
    print(f"  grad shape: {tuple(grad.shape)}")
    print(f"  grad finite: {grad_finite}")
    print(f"  grad abs mean: {float(values.abs().mean()):.9g}")
    print(f"  grad norm: {norm:.9g}")
    print(f"  nonzero fraction: {nonzero:.9g}")
    print(f"  NaN count: {int(torch.isnan(grad).sum())}")
    print(f"  Inf count: {int(torch.isinf(grad).sum())}")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    return passed


def make_refined_decode(terminal: torch.Tensor, z_quant: torch.Tensor,
                        res_mask: torch.Tensor, tokenizer: torch.nn.Module) -> dict[str, torch.Tensor]:
    require(torch.is_grad_enabled(), "decoder forward must retain autograd")
    require(terminal.requires_grad, "terminal residual must require grad")
    z_refined = z_quant + ALPHA * terminal
    return tokenizer.detokenize(z_refined, res_mask=res_mask)


def rigid_motion_invariance(pred: torch.Tensor, oracle: torch.Tensor,
                            valid: torch.Tensor) -> bool:
    # Deterministic proper rotation around z plus a nonzero translation.
    angle = torch.tensor(0.731, dtype=torch.float64, device=oracle.device)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    rotation = torch.stack((torch.stack((cosine, -sine, zero)),
                            torch.stack((sine, cosine, zero)),
                            torch.stack((zero, zero, one))))
    translation = torch.tensor([10.0, -4.0, 2.5], dtype=torch.float64, device=oracle.device)
    transformed = oracle.double() @ rotation + translation
    baseline = aligned_squared_rmsd(pred, oracle, valid)
    moved = aligned_squared_rmsd(pred, transformed, valid)
    raw_before = (pred[valid].double() - oracle[valid].double()).square().mean()
    raw_after = (pred[valid].double() - transformed[valid]).square().mean()
    difference = float((baseline - moved).abs())
    tolerance = 1e-6 + 1e-5 * abs(float(baseline))
    passed = bool(torch.isfinite(moved)) and difference <= tolerance
    print("\nrigid-motion invariance:")
    print(f"  aligned original: {float(baseline):.9g}")
    print(f"  aligned moved oracle: {float(moved):.9g}")
    print(f"  absolute difference: {difference:.9g}; tolerance: {tolerance:.9g}")
    print(f"  raw coordinate MSE before/after: {float(raw_before):.9g} / {float(raw_after):.9g}")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    return passed


def svd_stability(fm_residual: torch.Tensor, z_quant: torch.Tensor, oracle: dict,
                  res_mask: torch.Tensor, tokenizer: torch.nn.Module, seed: int) -> bool:
    generator = torch.Generator(device=fm_residual.device).manual_seed(seed + 1000)
    finite_trials = 0
    print("\nSVD backward stability:")
    for index in range(NUM_STABILITY_TRIALS):
        try:
            noise = torch.randn(fm_residual.shape, generator=generator,
                                device=fm_residual.device, dtype=fm_residual.dtype)
            terminal = (fm_residual + PERTURBATION_SCALE * noise).clone().requires_grad_(True)
            decoded = make_refined_decode(terminal, z_quant, res_mask, tokenizer)
            pred, target, valid = backbone_points(decoded, oracle, res_mask)
            loss = aligned_squared_rmsd(pred, target, valid)
            grad = torch.autograd.grad(loss, terminal)[0]
            finite = bool(torch.isfinite(loss).all() and torch.isfinite(grad).all())
            finite = finite and tuple(grad.shape) == (1, LENGTH, 13)
            print(f"  trial {index + 1}: {'PASS' if finite else 'FAIL'}"
                  f"; loss={float(loss.detach()):.9g}; grad_norm={float(grad.norm()):.9g}")
            finite_trials += int(finite)
            del decoded, pred, target, valid, loss, grad, terminal, noise
        except (RuntimeError, ValueError) as exc:
            print(f"  trial {index + 1}: FAIL; {type(exc).__name__}: {exc}")
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
    passed = finite_trials == NUM_STABILITY_TRIALS
    print(f"  finite trials: {finite_trials}/{NUM_STABILITY_TRIALS}")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    return passed


def cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(f"peak CUDA allocated MiB: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")
        print(f"peak CUDA reserved MiB: {torch.cuda.max_memory_reserved(device) / 1024**2:.2f}")
    else:
        print("peak CUDA allocated MiB: NA\npeak CUDA reserved MiB: NA")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print("ADJOINT_ALIGNED_REWARD_AUDIT", flush=True)
    stage = "load internal validation sample"
    try:
        require(args.num_flow_steps > 0, "num_flow_steps must be positive")
        sample = load_validation_sample(Path(args.dataset_dir), Path(args.split_csv), VAL_INDEX)
        require(sample["sample_id"] == SAMPLE_ID and int(sample["length"]) == LENGTH,
                "internal split no longer matches the prior audit sample")
        print(f"sample: {SAMPLE_ID} (val_index={VAL_INDEX})", flush=True)
        print(f"length: {LENGTH}", flush=True)
        z_cont = sample["z_cont"].float().reshape(1, LENGTH, 13).to(device)
        z_quant = sample["z_quant_current"].float().reshape(1, LENGTH, 13).to(device)
        res_mask = sample["res_mask"].float().reshape(1, LENGTH).to(device)
        require(bool(torch.isfinite(z_cont).all() and torch.isfinite(z_quant).all()), "stored latent non-finite")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        stage = "load frozen DPLM and structure tokenizer"
        extractor = DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        model = extractor.model
        tokenizer = model.struct_tokenizer
        model.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        dplm_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tokenizer_trainable = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in tokenizer.decoder.parameters() if p.requires_grad)
        assert dplm_trainable == 0, "DPLM is not frozen"
        assert tokenizer_trainable == 0, "structure tokenizer is not frozen"
        assert decoder_trainable == 0, "structure decoder is not frozen"
        print(f"frozen parameter audit: PASS (DPLM={dplm_trainable}, tokenizer={tokenizer_trainable}, decoder={decoder_trainable})")

        stage = "sample frozen finalized FM"
        fm, step, _, _ = load_fm(Path(args.checkpoint), device)
        assert sum(p.numel() for p in fm.parameters() if p.requires_grad) == 0
        aatype = sample["aatype"].long().reshape(1, LENGTH)
        struct_ids = sample["struct_ids_current"].long().reshape(1, LENGTH)
        with torch.no_grad():  # Only condition extraction and FM sampling.
            hidden = extractor(aatype, struct_ids, res_mask)["hidden_states"]
            require(len(hidden) == 33, "expected 33 Mode A condition states")
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn(z_quant.shape, generator=generator,
                                device=device, dtype=z_quant.dtype)
            fm_residual = euler_sample(fm, z_quant, hidden, res_mask,
                                       args.num_flow_steps, initial_noise=noise)
        del hidden, noise, fm, extractor
        require(bool(torch.isfinite(fm_residual).all()), "FM residual is non-finite")
        print(f"FM checkpoint optimizer_step: {step}; FM trainable parameters: 0")

        stage = "decode continuous-latent oracle"
        with torch.no_grad():  # Oracle is a fixed reference, never the reward path.
            oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)

        stage = "differentiate GT aligned reward"
        gt_terminal = (z_cont - z_quant).clone().requires_grad_(True)
        gt_decoded = make_refined_decode(gt_terminal, z_quant, res_mask, tokenizer)
        pred, target, valid = backbone_points(gt_decoded, oracle, res_mask)
        gt_loss = aligned_squared_rmsd(pred, target, valid)
        gt_ok = report_gradient("GT", gt_loss, gt_terminal, res_mask)
        del gt_decoded, pred, target, valid, gt_loss, gt_terminal

        stage = "differentiate FM aligned reward"
        fm_terminal = fm_residual.clone().requires_grad_(True)
        fm_decoded = make_refined_decode(fm_terminal, z_quant, res_mask, tokenizer)
        pred, target, valid = backbone_points(fm_decoded, oracle, res_mask)
        fm_loss = aligned_squared_rmsd(pred, target, valid)
        fm_ok = report_gradient("FM", fm_loss, fm_terminal, res_mask)
        stage = "rigid-motion invariance"
        invariance_ok = rigid_motion_invariance(pred, target, valid)
        del fm_decoded, pred, target, valid, fm_loss, fm_terminal

        stage = "five perturbed SVD backward trials"
        stability_ok = svd_stability(fm_residual, z_quant, oracle, res_mask,
                                     tokenizer, args.seed)
        decoder_ok = gt_ok and fm_ok
        print(f"\ndecoder gradient path: {'PASS' if decoder_ok else 'FAIL'}")
        print("frozen parameter audit: PASS")
        cuda_peaks(device)
        overall = gt_ok and fm_ok and stability_ok and invariance_ok
        print(f"overall: {'ADJOINT_ALIGNED_REWARD_PASS' if overall else 'ADJOINT_ALIGNED_REWARD_FAIL'}")
        print("ADJOINT_ALIGNED_REWARD_PASS" if overall else "ADJOINT_ALIGNED_REWARD_FAIL")
        return 0 if overall else 1
    except Exception as exc:
        print(f"failure point: {stage}; sample={SAMPLE_ID}; length={LENGTH}")
        print(f"error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        cuda_peaks(device)
        print("overall: ADJOINT_ALIGNED_REWARD_FAIL")
        print("ADJOINT_ALIGNED_REWARD_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

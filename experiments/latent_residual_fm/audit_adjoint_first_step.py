#!/usr/bin/env python3
"""Read-only first-step numerical audit for the frozen latent-residual FM.

A/B/C below are experimental discretization conventions, not confirmed rules
from the Adjoint Matching paper. This script performs no training or backward.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from experiments.latent_residual_fm.audit_adjoint_aligned_reward import (
    aligned_squared_rmsd,
    backbone_points,
)
from experiments.latent_residual_fm.debug_interfaces import require, resolve_device, seed_everything
from experiments.latent_residual_fm.debug_prediction_time_fm import load_fm, load_validation_sample
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.flow_matching import euler_sample


K = 40
H = 1.0 / K
ALPHA = 0.5
NUM_SAMPLES = 8
NUM_SEEDS = 8
REFERENCE_STEPS = 100
CANDIDATES = ("A", "B", "C")
# An intentionally conservative, declared numerical alarm, not a paper rule.
EXPLOSION_ABSOLUTE_FLOOR = 20.0
EXPLOSION_REFERENCE_MULTIPLIER = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded")
    parser.add_argument("--split_csv", default="data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv")
    parser.add_argument("--checkpoint", default="experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt")
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_validation_rows(split_csv: Path) -> list[tuple[int, str, int]]:
    """Select the shortest eight distinct lengths, breaking ties by split order."""
    with split_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require({"split", "sample_id", "length"}.issubset(reader.fieldnames or ()), "bad split CSV")
        validation = [row for row in reader if row["split"] == "val"]
    require(validation, "internal validation split is empty")
    ranked = sorted(enumerate(validation), key=lambda pair: (int(pair[1]["length"]), pair[0]))
    chosen: list[tuple[int, str, int]] = []
    lengths: set[int] = set()
    for index, row in ranked:
        length = int(row["length"])
        if length not in lengths:
            chosen.append((index, row["sample_id"], length))
            lengths.add(length)
        if len(chosen) == NUM_SAMPLES:
            break
    require(len(chosen) == NUM_SAMPLES, "fewer than eight distinct internal validation lengths")
    return chosen


def sigma_offset(t: float) -> float:
    return math.sqrt(2.0 * (1.0 - t + H) / (t + H))


def velocity(fm: torch.nn.Module, x: torch.Tensor, t: float, z_quant: torch.Tensor,
             hidden: tuple[torch.Tensor, ...], res_mask: torch.Tensor) -> torch.Tensor:
    time = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    return fm(x, time, z_quant, hidden, res_mask)


def first_step(candidate: str, fm: torch.nn.Module, x0: torch.Tensor,
               eps0: torch.Tensor, z_quant: torch.Tensor,
               hidden: tuple[torch.Tensor, ...], res_mask: torch.Tensor) -> torch.Tensor:
    if candidate == "A":
        return x0 + H * velocity(fm, x0, 0.0, z_quant, hidden, res_mask)
    t_eval = H if candidate == "B" else H / 2.0
    drift = 2.0 * velocity(fm, x0, t_eval, z_quant, hidden, res_mask) - x0 / t_eval
    return x0 + H * drift + math.sqrt(H) * sigma_offset(t_eval) * eps0


@dataclass
class AuditSeries:
    name: str
    endpoint_elements: list[torch.Tensor] = field(default_factory=list)
    endpoint_residue_norms: list[torch.Tensor] = field(default_factory=list)
    aligned_losses: list[float] = field(default_factory=list)
    finite_trajectories: int = 0
    nan_count: int = 0
    inf_count: int = 0
    explosive_trajectories: int = 0
    max_abs_state: float = 0.0
    max_state_norm: float = 0.0
    step_norm_sum: list[float] = field(default_factory=lambda: [0.0] * (K + 1))
    step_norm_count: list[int] = field(default_factory=lambda: [0] * (K + 1))
    step_norm_max: list[float] = field(default_factory=lambda: [0.0] * (K + 1))
    decode_failures: int = 0

    def record_state(self, step: int, x: torch.Tensor, res_mask: torch.Tensor,
                     still_finite: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_elements = res_mask.bool().unsqueeze(-1)
        active_elements = valid_elements & still_finite[:, None, None]
        self.nan_count += int((torch.isnan(x) & active_elements).sum())
        self.inf_count += int((torch.isinf(x) & active_elements).sum())
        finite_now = still_finite & torch.isfinite(x).all(dim=(1, 2))
        safe = torch.where(finite_now[:, None, None], x, torch.zeros_like(x))
        valid_residues = res_mask.bool() & finite_now[:, None]
        norms = torch.linalg.vector_norm(safe, dim=-1)
        values = norms[valid_residues]
        if values.numel():
            self.step_norm_sum[step] += float(values.sum())
            self.step_norm_count[step] += values.numel()
            step_max = float(values.max())
            self.step_norm_max[step] = max(self.step_norm_max[step], step_max)
            self.max_state_norm = max(self.max_state_norm, step_max)
            self.max_abs_state = max(self.max_abs_state, float(safe[valid_residues].abs().max()))
        explosive_now = ((norms > threshold) & valid_residues).any(dim=1)
        return safe, finite_now, explosive_now

    def record_endpoint(self, x: torch.Tensor, res_mask: torch.Tensor,
                        finite: torch.Tensor, explosive: torch.Tensor) -> None:
        self.finite_trajectories += int(finite.sum())
        self.explosive_trajectories += int(explosive.sum())
        valid = res_mask.bool() & finite[:, None]
        if bool(valid.any()):
            self.endpoint_elements.append(x[valid].float().reshape(-1).cpu())
            self.endpoint_residue_norms.append(torch.linalg.vector_norm(x.float(), dim=-1)[valid].cpu())

    def endpoint_stats(self) -> tuple[float, float, float]:
        if not self.endpoint_elements:
            return (math.nan, math.nan, math.nan)
        elements = torch.cat(self.endpoint_elements)
        norms = torch.cat(self.endpoint_residue_norms)
        return (float(elements.mean()), float(elements.std(unbiased=False)), float(norms.mean()))

    def aligned_stats(self) -> tuple[float, float]:
        if not self.aligned_losses:
            return (math.nan, math.nan)
        return (statistics.mean(self.aligned_losses), statistics.median(self.aligned_losses))


def decode_aligned_losses(series: AuditSeries, endpoints: torch.Tensor, finite: torch.Tensor,
                          z_quant: torch.Tensor, res_mask: torch.Tensor,
                          oracle: dict[str, torch.Tensor], tokenizer: torch.nn.Module) -> None:
    indices = torch.nonzero(finite).flatten()
    if indices.numel() == 0:
        return
    for index in indices.tolist():
        try:
            decoded = tokenizer.detokenize(
                z_quant[index:index + 1] + ALPHA * endpoints[index:index + 1],
                res_mask=res_mask[index:index + 1],
            )
            pred, target, valid = backbone_points(decoded, oracle, res_mask[index:index + 1])
            loss = float(aligned_squared_rmsd(pred, target, valid))
            if not math.isfinite(loss):
                raise ValueError("decoded aligned loss is non-finite")
            series.aligned_losses.append(loss)
        except (RuntimeError, ValueError, AssertionError) as exc:
            series.decode_failures += 1
            print(f"decode warning: {series.name} seed={index}: {type(exc).__name__}: {exc}", flush=True)


def simulate(candidate: str, series: AuditSeries, fm: torch.nn.Module,
             x0: torch.Tensor, eps: torch.Tensor, z_quant: torch.Tensor,
             hidden: tuple[torch.Tensor, ...], res_mask: torch.Tensor,
             threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = res_mask.to(dtype=x0.dtype).unsqueeze(-1)
    finite = torch.ones(x0.shape[0], dtype=torch.bool, device=x0.device)
    explosive = torch.zeros_like(finite)
    x = x0 * mask
    x, finite, flagged = series.record_state(0, x, res_mask, finite, threshold)
    explosive |= flagged
    x = first_step(candidate, fm, x, eps[0], z_quant, hidden, res_mask) * mask
    x, finite, flagged = series.record_state(1, x, res_mask, finite, threshold)
    explosive |= flagged
    for k in range(1, K):
        if not bool(finite.any()):
            break
        t = k * H
        drift = 2.0 * velocity(fm, x, t, z_quant, hidden, res_mask) - x / t
        x = (x + H * drift + math.sqrt(H) * sigma_offset(t) * eps[k]) * mask
        x, finite, flagged = series.record_state(k + 1, x, res_mask, finite, threshold)
        explosive |= flagged
    series.record_endpoint(x, res_mask, finite, explosive)
    return x, finite, explosive


def make_gaussian_batch(seed: int, val_index: int, length: int,
                        device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x0, eps = [], []
    for seed_index in range(NUM_SEEDS):
        generator = torch.Generator(device=device).manual_seed(seed + val_index * 1_000_003 + seed_index)
        x0.append(torch.randn((1, length, 13), generator=generator, device=device, dtype=dtype))
        eps.append(torch.randn((K, length, 13), generator=generator, device=device, dtype=dtype))
    return torch.cat(x0, dim=0), torch.stack(eps, dim=1)


def print_series(series: AuditSeries, total: int) -> None:
    mean, std, norm = series.endpoint_stats()
    aligned_mean, aligned_median = series.aligned_stats()
    print(f"\n{series.name}:")
    print(f"  finite: {series.finite_trajectories}/{total} ({series.finite_trajectories / total:.4f})")
    print(f"  NaN/Inf valid state elements: {series.nan_count}/{series.inf_count}")
    if series.name == "reference":
        print("  max absolute state: NA (reference trajectory not tracked)")
        print("  max state per-residue L2 norm: NA (reference trajectory not tracked)")
        print("  explosive trajectories: NA (reference is the endpoint comparator)")
    else:
        print(f"  max absolute state: {series.max_abs_state:.8g}")
        print(f"  max state per-residue L2 norm: {series.max_state_norm:.8g}")
        print(f"  explosive trajectories: {series.explosive_trajectories}/{total}")
    print(f"  endpoint latent mean/std: {mean:.8g}/{std:.8g}")
    print(f"  endpoint mean per-residue L2 norm: {norm:.8g}")
    print(f"  oracle aligned backbone loss mean/median: {aligned_mean:.8g}/{aligned_median:.8g}"
          f" (n={len(series.aligned_losses)}, decode failures={series.decode_failures})")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print("ADJOINT_FIRST_STEP_AUDIT", flush=True)
    stage = "select internal validation samples"
    try:
        rows = select_validation_rows(Path(args.split_csv))
        total = len(rows) * NUM_SEEDS
        print(f"reference: finalized FM checkpoint, deterministic {REFERENCE_STEPS}-step Euler, same X0")
        print(f"samples: {len(rows)} internal validation samples; selected shortest distinct lengths")
        for index, sample_id, length in rows:
            print(f"  val_index={index} sample_id={sample_id} length={length}")
        print(f"seeds: {NUM_SEEDS} per sample; seed = {args.seed} + val_index*1000003 + seed_index (0..7)")
        print(f"trajectories: {total} per method; K={K}; h={H}; alpha={ALPHA}")
        print("B/C first steps: numerical candidates only, not paper-confirmed rules")
        print("explosion threshold per sample: max(20, 10 * reference endpoint max valid per-residue L2 norm)")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        stage = "load frozen models"
        extractor = DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        dplm = extractor.model.requires_grad_(False).eval()
        tokenizer = dplm.struct_tokenizer.requires_grad_(False).eval()
        fm, checkpoint_step, _, _ = load_fm(Path(args.checkpoint), device)
        require(checkpoint_step == 100_000, "wrong finalized FM checkpoint")
        for name, model in (("DPLM", dplm), ("tokenizer", tokenizer),
                            ("decoder", tokenizer.decoder), ("FM", fm)):
            require(not model.training and all(not p.requires_grad for p in model.parameters()),
                    f"{name} is not frozen and in eval mode")
        print("frozen models: DPLM/tokenizer/decoder/FM; trainable parameters: 0", flush=True)

        series = {name: AuditSeries(name) for name in ("reference", *CANDIDATES)}
        with torch.no_grad():
            for sample_number, (val_index, sample_id, length) in enumerate(rows, start=1):
                stage = f"internal validation sample {sample_number}/{len(rows)} ({sample_id})"
                sample = load_validation_sample(Path(args.dataset_dir), Path(args.split_csv), val_index)
                require(sample["sample_id"] == sample_id and int(sample["length"]) == length,
                        "selected validation row changed")
                z_cont = sample["z_cont"].float().reshape(1, length, 13).to(device)
                zq_one = sample["z_quant_current"].float().reshape(1, length, 13).to(device)
                mask_one = sample["res_mask"].float().reshape(1, length).to(device)
                require(bool(mask_one.bool().any()), "empty validation residue mask")
                require(bool(torch.isfinite(z_cont).all() and torch.isfinite(zq_one).all()),
                        "non-finite stored validation latent")
                aatype = sample["aatype"].long().reshape(1, length)
                struct_ids = sample["struct_ids_current"].long().reshape(1, length)
                hidden_one = extractor(aatype, struct_ids, mask_one)["hidden_states"]
                require(len(hidden_one) == 33, "bad Mode A condition")
                hidden = tuple(h.expand(NUM_SEEDS, -1, -1) for h in hidden_one)
                z_quant = zq_one.expand(NUM_SEEDS, -1, -1)
                res_mask = mask_one.expand(NUM_SEEDS, -1)
                x0, eps = make_gaussian_batch(args.seed, val_index, length, device, zq_one.dtype)
                x0 = x0 * res_mask.unsqueeze(-1)
                eps = eps * res_mask.unsqueeze(0).unsqueeze(-1)

                reference = euler_sample(fm, z_quant, hidden, res_mask, REFERENCE_STEPS,
                                         initial_noise=x0)
                require(bool(torch.isfinite(reference).all()), "reference FM endpoint is non-finite")
                ref_norms = torch.linalg.vector_norm(reference, dim=-1)[res_mask.bool()]
                threshold = max(EXPLOSION_ABSOLUTE_FLOOR,
                                EXPLOSION_REFERENCE_MULTIPLIER * float(ref_norms.max()))
                series["reference"].finite_trajectories += NUM_SEEDS
                series["reference"].endpoint_elements.append(reference[res_mask.bool()].reshape(-1).float().cpu())
                series["reference"].endpoint_residue_norms.append(ref_norms.float().cpu())
                oracle = tokenizer.detokenize(z_cont, res_mask=mask_one)
                decode_aligned_losses(series["reference"], reference, torch.ones(NUM_SEEDS,
                                      dtype=torch.bool, device=device), z_quant, res_mask, oracle, tokenizer)
                for name in CANDIDATES:
                    endpoint, finite, _ = simulate(name, series[name], fm, x0, eps, z_quant,
                                                    hidden, res_mask, threshold)
                    decode_aligned_losses(series[name], endpoint, finite, z_quant,
                                          res_mask, oracle, tokenizer)
                print(f"completed sample {sample_number}/{len(rows)}: {sample_id}, L={length}, "
                      f"explosion_threshold={threshold:.6g}", flush=True)
                del sample, z_cont, zq_one, mask_one, hidden_one, hidden, z_quant, res_mask
                del x0, eps, reference, oracle

        for name in ("reference", *CANDIDATES):
            print_series(series[name], total)
        print("\nper-timestep state norm across valid finite residues (mean/max; t=k*h):")
        print("  k       t          A mean/max           B mean/max           C mean/max")
        for k in range(K + 1):
            parts = []
            for name in CANDIDATES:
                item = series[name]
                mean = item.step_norm_sum[k] / item.step_norm_count[k] if item.step_norm_count[k] else math.nan
                parts.append(f"{mean:10.5g}/{item.step_norm_max[k]:10.5g}")
            print(f"  {k:2d}  {k * H:6.3f}    " + "    ".join(parts))

        ref_mean, ref_std, ref_norm = series["reference"].endpoint_stats()
        require(ref_std > 0 and ref_norm > 0, "degenerate reference endpoint distribution")
        distortion: dict[str, float] = {}
        for name in CANDIDATES:
            item = series[name]
            mean, std, norm = item.endpoint_stats()
            if item.finite_trajectories == total:
                distortion[name] = (abs(mean - ref_mean) / ref_std
                                    + abs(std - ref_std) / ref_std
                                    + abs(norm - ref_norm) / ref_norm)
            else:
                distortion[name] = math.inf  # No ranking on a surviving-only distribution.
        most_stable = min(CANDIDATES, key=lambda name: (
            total - series[name].finite_trajectories,
            series[name].explosive_trajectories,
            series[name].max_state_norm,
        ))
        least_distortion = min(CANDIDATES, key=lambda name: distortion[name])
        least_label = least_distortion if math.isfinite(distortion[least_distortion]) else "NONE"
        passed = any(series[name].finite_trajectories == total
                     and series[name].explosive_trajectories == 0 for name in CANDIDATES)
        print("\nsummary:")
        print("  endpoint distortion score = |mean-ref_mean|/ref_std + "
              "|std-ref_std|/ref_std + |mean_norm-ref_mean_norm|/ref_mean_norm")
        print("  scores: " + ", ".join(f"{name}={distortion[name]:.8g}" for name in CANDIDATES))
        print("  most stable candidate: " + most_stable +
              " (rank: finite trajectory failures, explosive count, max state norm)")
        print("  least endpoint distortion candidate: " + least_label)
        print("  oracle aligned loss is diagnostic only, not a model-selection metric")
        print("  paper exact first-step: NOT CONFIRMED")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            print(f"  peak CUDA allocated MiB: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")
        marker = "ADJOINT_FIRST_STEP_DIAGNOSTIC_PASS" if passed else "ADJOINT_FIRST_STEP_DIAGNOSTIC_FAIL"
        print("overall: " + marker)
        print(marker)
        return 0 if passed else 1
    except Exception as exc:
        print(f"failure point: {stage}; {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("paper exact first-step: NOT CONFIRMED")
        print("ADJOINT_FIRST_STEP_DIAGNOSTIC_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

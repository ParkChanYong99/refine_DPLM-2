#!/usr/bin/env python3
"""Tiny fixed-path FM overfit with real frozen DPLM teacher-forced conditions."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.flow_matching import (
    masked_flow_matching_loss,
    sample_flow_matching_batch,
)
from experiments.latent_residual_fm.fm_model import LatentResidualFM


CHECKPOINT = "airkingbd/dplm2_bit_650m"
LOG_STEPS = (0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199)


@dataclass
class CachedSample:
    sample_id: str
    length: int
    z_quant: torch.Tensor
    residual: torch.Tensor
    res_mask: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    r_t: torch.Tensor
    u_t: torch.Tensor
    t: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        default="data-bin/latent_residual_fm/afdb_l512_sharded/shard-000000.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_counts(model: torch.nn.Module) -> tuple[int, int, int]:
    present = nonzero = invalid = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        present += 1
        nonzero += int(bool((parameter.grad != 0).any()))
        invalid += int(not bool(torch.isfinite(parameter.grad).all()))
    return present, nonzero, invalid


def to_gpu(sample: CachedSample, device: torch.device) -> dict[str, Any]:
    return {
        "z_quant": sample.z_quant.to(device),
        "res_mask": sample.res_mask.to(device),
        "r_t": sample.r_t.to(device),
        "u_t": sample.u_t.to(device),
        "t": sample.t.to(device),
        # Keep the 33 states as a tuple; never stack them.
        "hidden_states": tuple(state.to(device) for state in sample.hidden_states),
    }


def evaluate_sample(
    model: LatentResidualFM,
    sample: CachedSample,
    device: torch.device,
) -> float:
    tensors = to_gpu(sample, device)
    with torch.no_grad():
        prediction = model(
            tensors["r_t"],
            tensors["t"],
            tensors["z_quant"],
            tensors["hidden_states"],
            tensors["res_mask"],
        )
        require(bool(torch.isfinite(prediction).all()), "evaluation prediction is non-finite")
        loss = masked_flow_matching_loss(
            prediction, tensors["u_t"], tensors["res_mask"]
        )
        require(bool(torch.isfinite(loss)), "evaluation loss is non-finite")
        value = float(loss)
    del tensors, prediction, loss
    return value


def default_summary() -> dict[str, Any]:
    return {
        "sample_ids": [],
        "lengths": [],
        "hidden_layers": 0,
        "hidden_dim": 0,
        "precompute_seconds": 0.0,
        "dplm_forward_calls": 0,
        "steps": 0,
        "lr": 0.0,
        "step_losses": {},
        "initial_losses": [],
        "final_losses": [],
        "ratios": [],
        "mean_initial": float("nan"),
        "mean_final": float("nan"),
        "samples_improved": 0,
        "initial_weight_min": float("nan"),
        "initial_weight_max": float("nan"),
        "final_weight_min": float("nan"),
        "final_weight_max": float("nan"),
        "max_weight_change": 0.0,
        "sensitivity_mean": 0.0,
        "sensitivity_max": 0.0,
        "fm_nonzero_gradients": 0,
        "fm_invalid_gradients": -1,
        "dplm_gradient_parameters": -1,
        "training_seconds": 0.0,
        "total_seconds": 0.0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0,
        "final_status": "REAL_CONDITION_OVERFIT_FAILED",
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("REAL-CONDITION FM TINY OVERFIT")
    print("=" * 60)
    print(f"samples: {len(summary['sample_ids'])}")
    print(f"sample IDs: {summary['sample_ids']}")
    print(f"lengths: {summary['lengths']}")
    print("\nDPLM CONDITION CACHE")
    print(f"hidden layers: {summary['hidden_layers']}")
    print(f"hidden dim: {summary['hidden_dim']}")
    print(f"precompute seconds: {summary['precompute_seconds']:.6f}")
    print(f"DPLM forward calls: {summary['dplm_forward_calls']}")
    print("\nTRAINING")
    print(f"steps: {summary['steps']}")
    print("batch size: 1")
    print("optimizer: AdamW(weight_decay=0.01)")
    print(f"learning rate: {summary['lr']}")
    for step in LOG_STEPS:
        value = summary["step_losses"].get(step, float("nan"))
        print(f"step {step} loss: {value:.9f}")
    print("\nPER-SAMPLE")
    for index, (initial, final, ratio) in enumerate(
        zip(summary["initial_losses"], summary["final_losses"], summary["ratios"]),
        start=1,
    ):
        print(f"sample {index} initial/final/ratio: {initial:.9f}/{final:.9f}/{ratio:.9f}")
    print(f"mean initial loss: {summary['mean_initial']:.9f}")
    print(f"mean final loss: {summary['mean_final']:.9f}")
    print(f"samples improved: {summary['samples_improved']}")
    print("\nLAYER MIX")
    print(
        f"initial weights min/max: {summary['initial_weight_min']:.9f}/"
        f"{summary['initial_weight_max']:.9f}"
    )
    print(
        f"final weights min/max: {summary['final_weight_min']:.9f}/"
        f"{summary['final_weight_max']:.9f}"
    )
    print(f"max absolute weight change: {summary['max_weight_change']:.9f}")
    print("\nCONDITION SENSITIVITY")
    print(f"mean abs diff: {summary['sensitivity_mean']:.9f}")
    print(f"max abs diff: {summary['sensitivity_max']:.9f}")
    print("\nGRADIENT")
    print(f"FM nonzero gradient parameters: {summary['fm_nonzero_gradients']}")
    print(f"FM nan/inf gradients: {summary['fm_invalid_gradients']}")
    print(f"DPLM gradient parameters: {summary['dplm_gradient_parameters']}")
    print("\nTIMING")
    print(f"training seconds: {summary['training_seconds']:.6f}")
    print(f"total seconds: {summary['total_seconds']:.6f}")
    print("\nGPU MEMORY")
    print(f"peak allocated MiB: {summary['peak_allocated_mib']:.6f}")
    print(f"peak reserved MiB: {summary['peak_reserved_mib']:.6f}")
    print(f"\nFINAL STATUS: {summary['final_status']}")
    print("=" * 60)


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    require(args.num_samples == 4, "this validated debug must use exactly 4 samples")
    require(args.steps == 200, "this validated debug must run exactly 200 steps")
    require(args.lr > 0, "learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    summary = default_summary()
    summary["steps"] = args.steps
    summary["lr"] = args.lr
    extractor: DPLMConditionExtractor | None = None
    try:
        shard = torch.load(Path(args.shard), map_location="cpu")
        require(shard["num_samples"] == len(shard["samples"]), "bad shard count")
        require(len(shard["samples"]) >= args.num_samples, "not enough shard samples")
        raw_samples = shard["samples"][: args.num_samples]
        summary["sample_ids"] = [sample["sample_id"] for sample in raw_samples]
        summary["lengths"] = [int(sample["length"]) for sample in raw_samples]

        extractor = DPLMConditionExtractor(CHECKPOINT, device)
        dplm = extractor.model
        require(
            all(not parameter.requires_grad for parameter in dplm.parameters()),
            "DPLM is not fully frozen",
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        cached: list[CachedSample] = []
        precompute_started = time.perf_counter()
        for sample in raw_samples:
            length = int(sample["length"])
            aatype = sample["aatype"].long().unsqueeze(0)
            struct_ids = sample["struct_ids_current"].long().unsqueeze(0)
            res_mask_cpu = sample["res_mask"].float().unsqueeze(0)
            with torch.no_grad():
                condition = extractor(aatype, struct_ids, res_mask_cpu)
            summary["dplm_forward_calls"] += 1
            hidden_cpu = tuple(
                state.detach().float().cpu() for state in condition["hidden_states"]
            )
            require(len(hidden_cpu) == 33, "condition cache does not contain 33 layers")
            require(
                all(tuple(state.shape) == (1, length, 1280) for state in hidden_cpu),
                "cached hidden shape mismatch",
            )
            residual_cpu = sample["residual"].float().unsqueeze(0)
            z_quant_cpu = sample["z_quant_current"].float().unsqueeze(0)
            fixed_r0 = torch.randn(
                residual_cpu.shape, dtype=torch.float32, generator=generator
            )
            fixed_t = torch.rand(1, dtype=torch.float32, generator=generator)
            fixed_flow = sample_flow_matching_batch(
                residual_cpu,
                res_mask_cpu,
                r0=fixed_r0,
                t=fixed_t,
            )
            cached.append(
                CachedSample(
                    sample_id=sample["sample_id"],
                    length=length,
                    z_quant=z_quant_cpu,
                    residual=residual_cpu,
                    res_mask=res_mask_cpu,
                    hidden_states=hidden_cpu,
                    r_t=fixed_flow["r_t"],
                    u_t=fixed_flow["u_t"],
                    t=fixed_flow["t"],
                )
            )
            del condition, hidden_cpu, fixed_flow
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary["precompute_seconds"] = time.perf_counter() - precompute_started
        summary["hidden_layers"] = len(cached[0].hidden_states)
        summary["hidden_dim"] = cached[0].hidden_states[0].shape[-1]

        # Frozen DPLM remains available for the final gradient audit but no
        # longer occupies GPU memory and is never forwarded again.
        dplm.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        fm = LatentResidualFM().to(device)
        optimizer_parameters = [parameter for parameter in fm.parameters() if parameter.requires_grad]
        dplm_parameter_ids = {id(parameter) for parameter in dplm.parameters()}
        require(
            all(id(parameter) not in dplm_parameter_ids for parameter in optimizer_parameters),
            "DPLM parameter was included in the optimizer",
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=args.lr, weight_decay=0.01
        )
        initial_weights = fm.layer_weights().detach().cpu().clone()
        summary["initial_weight_min"] = float(initial_weights.min())
        summary["initial_weight_max"] = float(initial_weights.max())

        fm.eval()
        summary["initial_losses"] = [
            evaluate_sample(fm, sample, device) for sample in cached
        ]
        require(
            all(math.isfinite(value) for value in summary["initial_losses"]),
            "initial per-sample loss is non-finite",
        )

        fm.train()
        training_started = time.perf_counter()
        for step in range(args.steps):
            sample = cached[step % len(cached)]
            tensors = to_gpu(sample, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = fm(
                tensors["r_t"],
                tensors["t"],
                tensors["z_quant"],
                tensors["hidden_states"],
                tensors["res_mask"],
            )
            require(bool(torch.isfinite(prediction).all()), "training prediction is non-finite")
            loss = masked_flow_matching_loss(
                prediction, tensors["u_t"], tensors["res_mask"]
            )
            require(bool(torch.isfinite(loss)), "training loss is non-finite")
            loss.backward()
            _, _, invalid = gradient_counts(fm)
            require(invalid == 0, "training gradient contains NaN/Inf")
            optimizer.step()
            loss_value = float(loss.detach())
            if step in LOG_STEPS:
                summary["step_losses"][step] = loss_value
                print(f"step {step} loss: {loss_value:.9f}")
            del tensors, prediction, loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        summary["training_seconds"] = time.perf_counter() - training_started

        fm.eval()
        summary["final_losses"] = [
            evaluate_sample(fm, sample, device) for sample in cached
        ]
        summary["ratios"] = [
            final / initial
            for initial, final in zip(
                summary["initial_losses"], summary["final_losses"]
            )
        ]
        summary["mean_initial"] = statistics.fmean(summary["initial_losses"])
        summary["mean_final"] = statistics.fmean(summary["final_losses"])
        summary["samples_improved"] = sum(
            final < initial
            for initial, final in zip(
                summary["initial_losses"], summary["final_losses"]
            )
        )
        final_weights = fm.layer_weights().detach().cpu()
        summary["final_weight_min"] = float(final_weights.min())
        summary["final_weight_max"] = float(final_weights.max())
        summary["max_weight_change"] = float(
            (final_weights - initial_weights).abs().max()
        )

        first = to_gpu(cached[0], device)
        zero_base = torch.zeros_like(first["hidden_states"][0])
        zero_hidden = tuple(zero_base for _ in range(33))
        with torch.no_grad():
            actual_prediction = fm(
                first["r_t"],
                first["t"],
                first["z_quant"],
                first["hidden_states"],
                first["res_mask"],
            )
            zero_prediction = fm(
                first["r_t"],
                first["t"],
                first["z_quant"],
                zero_hidden,
                first["res_mask"],
            )
        difference = (actual_prediction - zero_prediction).abs()[first["res_mask"].bool()]
        summary["sensitivity_mean"] = float(difference.mean())
        summary["sensitivity_max"] = float(difference.max())
        del first, zero_base, zero_hidden, actual_prediction, zero_prediction, difference

        _, fm_nonzero, fm_invalid = gradient_counts(fm)
        summary["fm_nonzero_gradients"] = fm_nonzero
        summary["fm_invalid_gradients"] = fm_invalid
        dplm_gradients, _, _ = gradient_counts(dplm)
        summary["dplm_gradient_parameters"] = dplm_gradients
        if device.type == "cuda":
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024**2)

        passed = (
            len(cached) == 4
            and all(math.isfinite(value) for value in summary["final_losses"])
            and fm_invalid == 0
            and dplm_gradients == 0
            and summary["mean_final"] < summary["mean_initial"]
            and summary["samples_improved"] >= 3
            and summary["max_weight_change"] > 0
            and summary["sensitivity_max"] > 0
        )
        require(passed, "one or more tiny-overfit success criteria failed")
        summary["final_status"] = "REAL_CONDITION_OVERFIT_PASS"
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if device.type == "cuda" and torch.cuda.is_available():
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    finally:
        summary["total_seconds"] = time.perf_counter() - total_started

    print_summary(summary)
    return 0 if summary["final_status"] == "REAL_CONDITION_OVERFIT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

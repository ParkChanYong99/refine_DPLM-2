#!/usr/bin/env python3
"""Two-sample smoke test for the latent-residual FM core only."""

from __future__ import annotations

import argparse
import math
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from experiments.latent_residual_fm.flow_matching import (
    euler_sample,
    masked_flow_matching_loss,
    sample_flow_matching_batch,
)
from experiments.latent_residual_fm.fm_model import LatentResidualFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        default="data-bin/latent_residual_fm/afdb_l512_sharded/shard-000000.pt",
    )
    parser.add_argument("--device", default="cuda")
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


def load_real_batch(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    shard = torch.load(path, map_location="cpu")
    require(
        {"schema_version", "num_samples", "total_residues", "samples"}.issubset(shard),
        "shard top-level schema is incomplete",
    )
    require(shard["num_samples"] == len(shard["samples"]), "shard sample count mismatch")
    require(len(shard["samples"]) >= 2, "debug shard needs at least two samples")
    samples = shard["samples"][:2]
    lengths = [int(sample["length"]) for sample in samples]
    for sample, length in zip(samples, lengths):
        require(tuple(sample["z_quant_current"].shape) == (length, 13), "bad z_quant shape")
        require(tuple(sample["residual"].shape) == (length, 13), "bad residual shape")
        require(tuple(sample["res_mask"].shape) == (length,), "bad res_mask shape")
        require(sample["z_quant_current"].dtype == torch.float32, "bad z_quant dtype")
        require(sample["residual"].dtype == torch.float32, "bad residual dtype")
        require(sample["res_mask"].dtype == torch.float32, "bad res_mask dtype")
    z_quant = pad_sequence(
        [sample["z_quant_current"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)
    residual = pad_sequence(
        [sample["residual"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)
    res_mask = pad_sequence(
        [sample["res_mask"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)
    return z_quant, residual, res_mask, torch.tensor(lengths)


def count_gradients(model: torch.nn.Module) -> tuple[int, int]:
    nonzero = 0
    invalid = 0
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            invalid += 1
        if bool((parameter.grad != 0).any()):
            nonzero += 1
    return nonzero, invalid


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LATENT RESIDUAL FM CORE DEBUG")
    print("=" * 60)
    print(f"device: {summary['device']}")
    print(f"sample lengths: {summary['sample_lengths']}")
    print(f"batch shape: {summary['batch_shape']}")
    print("\nFM MODEL")
    print(f"trainable parameters: {summary['trainable_parameters']}")
    print(f"output shape: {summary['output_shape']}")
    print(f"output finite: {summary['output_finite']}")
    print("\nFLOW")
    print(f"r_t shape: {summary['r_t_shape']}")
    print(f"u_t shape: {summary['u_t_shape']}")
    print(f"initial loss: {summary['initial_loss']:.9f}")
    print("\nGRADIENT")
    print(f"nonzero gradient parameters: {summary['nonzero_gradient_parameters']}")
    print(f"nan/inf gradients: {summary['invalid_gradients']}")
    print("\nLAYER MIX")
    print(f"num layers: {summary['layer_count']}")
    print(f"weight sum: {summary['weight_sum']:.9f}")
    print(f"min weight: {summary['weight_min']:.9f}")
    print(f"max weight: {summary['weight_max']:.9f}")
    print("\nEULER")
    print(f"steps: {summary['euler_steps']}")
    print(f"output shape: {summary['euler_shape']}")
    print(f"output finite: {summary['euler_finite']}")
    print("\nMICRO OVERFIT")
    print(f"initial loss: {summary['initial_loss']:.9f}")
    print(f"final loss: {summary['final_loss']:.9f}")
    print(f"best loss: {summary['best_loss']:.9f}")
    print(f"loss decreased: {summary['loss_decreased']}")
    print(f"\nFINAL STATUS: {summary['final_status']}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    summary: dict[str, Any] = {
        "device": str(device),
        "sample_lengths": [],
        "batch_shape": None,
        "trainable_parameters": 0,
        "output_shape": None,
        "output_finite": False,
        "r_t_shape": None,
        "u_t_shape": None,
        "initial_loss": float("nan"),
        "nonzero_gradient_parameters": 0,
        "invalid_gradients": -1,
        "layer_count": 33,
        "weight_sum": float("nan"),
        "weight_min": float("nan"),
        "weight_max": float("nan"),
        "euler_steps": 4,
        "euler_shape": None,
        "euler_finite": False,
        "final_loss": float("nan"),
        "best_loss": float("nan"),
        "loss_decreased": False,
        "final_status": "FM_CORE_FAILED",
    }
    try:
        z_quant, residual, res_mask, lengths = load_real_batch(Path(args.shard), device)
        summary["sample_lengths"] = lengths.tolist()
        summary["batch_shape"] = tuple(residual.shape)
        require(z_quant.shape == residual.shape, "z_quant/residual batch mismatch")
        require(res_mask.shape == residual.shape[:2], "mask batch mismatch")
        require(bool(torch.isfinite(z_quant).all()), "z_quant is non-finite")
        require(bool(torch.isfinite(residual).all()), "residual is non-finite")
        print(f"[A] actual shard load PASS: {args.shard}")
        print(
            f"[B] batch shapes: residual={tuple(residual.shape)} "
            f"z_quant={tuple(z_quant.shape)} res_mask={tuple(res_mask.shape)}"
        )

        batch_size, max_length = residual.shape[:2]
        base_hidden = torch.randn(
            batch_size, max_length, 1280, dtype=torch.float32, device=device
        )
        hidden_states = tuple(base_hidden for _ in range(33))
        model = LatentResidualFM().to(device)
        summary["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        flow = sample_flow_matching_batch(residual, res_mask)
        for key in ("r0", "r1", "r_t", "u_t", "t"):
            require(bool(torch.isfinite(flow[key]).all()), f"{key} is non-finite")
        summary["r_t_shape"] = tuple(flow["r_t"].shape)
        summary["u_t_shape"] = tuple(flow["u_t"].shape)
        print(
            f"[C] FM sample PASS: r0={tuple(flow['r0'].shape)} "
            f"r1={tuple(flow['r1'].shape)} r_t={tuple(flow['r_t'].shape)} "
            f"u_t={tuple(flow['u_t'].shape)} t={tuple(flow['t'].shape)}"
        )

        prediction = model(
            flow["r_t"], flow["t"], z_quant, hidden_states, res_mask
        )
        summary["output_shape"] = tuple(prediction.shape)
        summary["output_finite"] = bool(torch.isfinite(prediction).all())
        require(prediction.shape == residual.shape, "FM output shape mismatch")
        require(summary["output_finite"], "FM output is non-finite")
        print(f"[D] forward PASS: v_pred={tuple(prediction.shape)} finite=True")

        loss = masked_flow_matching_loss(prediction, flow["u_t"], res_mask)
        summary["initial_loss"] = float(loss.detach())
        require(bool(torch.isfinite(loss)), "FM loss is non-finite")
        require(float(loss.detach()) > 0, "FM loss is not positive")
        print(f"[E] masked loss PASS: {summary['initial_loss']:.9f}")
        loss.backward()
        nonzero, invalid = count_gradients(model)
        summary["nonzero_gradient_parameters"] = nonzero
        summary["invalid_gradients"] = invalid
        require(nonzero > 0, "no trainable FM parameter has a nonzero gradient")
        require(invalid == 0, "FM gradients contain NaN/Inf")
        print(f"[F] backward PASS: nonzero={nonzero} invalid={invalid}")

        weights = model.layer_weights().detach()
        summary["weight_sum"] = float(weights.sum())
        summary["weight_min"] = float(weights.min())
        summary["weight_max"] = float(weights.max())
        require(tuple(weights.shape) == (33,), "layer mixing weight shape mismatch")
        require(
            math.isclose(summary["weight_sum"], 1.0, rel_tol=1e-6, abs_tol=1e-6),
            "layer mixing weights do not sum to one",
        )
        print(f"[G] layer mix PASS: shape={tuple(weights.shape)} sum={weights.sum().item():.9f}")

        model.eval()
        with torch.inference_mode():
            sampled = euler_sample(
                model,
                z_quant,
                hidden_states,
                res_mask,
                num_steps=summary["euler_steps"],
            )
        summary["euler_shape"] = tuple(sampled.shape)
        summary["euler_finite"] = bool(torch.isfinite(sampled).all())
        require(sampled.shape == residual.shape, "Euler output shape mismatch")
        require(summary["euler_finite"], "Euler output is non-finite")
        print(f"[H] Euler PASS: shape={tuple(sampled.shape)} finite=True")

        # Reuse the initially sampled fixed path for all 20 updates.  This is a
        # learnability smoke test, not a claim about actual training behavior.
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        best_loss = summary["initial_loss"]
        final_loss = summary["initial_loss"]
        print(f"micro-overfit step 0 loss: {summary['initial_loss']:.9f}")
        for step in range(1, 21):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(
                flow["r_t"], flow["t"], z_quant, hidden_states, res_mask
            )
            train_loss = masked_flow_matching_loss(predicted, flow["u_t"], res_mask)
            require(bool(torch.isfinite(train_loss)), "micro-overfit loss is non-finite")
            train_loss.backward()
            _, train_invalid = count_gradients(model)
            require(train_invalid == 0, "micro-overfit gradient contains NaN/Inf")
            optimizer.step()
            final_loss = float(train_loss.detach())
            best_loss = min(best_loss, final_loss)
            if step in (5, 10, 15, 20):
                print(f"micro-overfit step {step} loss: {final_loss:.9f}")
        summary["final_loss"] = final_loss
        summary["best_loss"] = best_loss
        summary["loss_decreased"] = best_loss < summary["initial_loss"]
        require(summary["loss_decreased"], "micro-overfit best loss did not decrease")
        summary["final_status"] = "FM_CORE_PASS"
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()

    print_summary(summary)
    return 0 if summary["final_status"] == "FM_CORE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

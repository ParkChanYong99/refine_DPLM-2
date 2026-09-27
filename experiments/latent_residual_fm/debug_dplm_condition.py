#!/usr/bin/env python3
"""Connect one real AFDB latent sample to frozen DPLM teacher-forced states."""

from __future__ import annotations

import argparse
import math
import random
import traceback
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


def gradient_counts(model: torch.nn.Module) -> tuple[int, int, int]:
    present = nonzero = invalid = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        present += 1
        invalid += int(not bool(torch.isfinite(parameter.grad).all()))
        nonzero += int(bool((parameter.grad != 0).any()))
    return present, nonzero, invalid


def default_summary(device: torch.device) -> dict[str, Any]:
    return {
        "sample_id": "N/A",
        "length": 0,
        "checkpoint": CHECKPOINT,
        "dplm_total_parameters": 0,
        "dplm_trainable_parameters": -1,
        "packed_shape": None,
        "all_hidden_states": 0,
        "transformer_states_used": 0,
        "first_shape": None,
        "middle_shape": None,
        "last_shape": None,
        "hidden_finite": False,
        "fm_trainable_parameters": 0,
        "prediction_shape": None,
        "prediction_finite": False,
        "loss": float("nan"),
        "fm_nonzero_gradients": 0,
        "fm_invalid_gradients": -1,
        "layer_logits_gradient_nonzero": False,
        "dplm_gradient_parameters": -1,
        "sensitivity_mean": 0.0,
        "sensitivity_max": 0.0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0,
        "final_status": "DPLM_FM_CONDITION_FAILED",
        "device": str(device),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("DPLM → LATENT FM CONDITION DEBUG")
    print("=" * 60)
    print(f"sample_id: {summary['sample_id']}")
    print(f"L: {summary['length']}")
    print("\nDPLM")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"total parameters: {summary['dplm_total_parameters']}")
    print(f"trainable parameters: {summary['dplm_trainable_parameters']}")
    print(f"packed token shape: {summary['packed_shape']}")
    print(f"all hidden states: {summary['all_hidden_states']}")
    print(f"transformer states used: {summary['transformer_states_used']}")
    print("\nALIGNED HIDDEN")
    print(f"first layer: {summary['first_shape']}")
    print(f"middle layer: {summary['middle_shape']}")
    print(f"last layer: {summary['last_shape']}")
    print(f"finite: {summary['hidden_finite']}")
    print("\nFM")
    print(f"trainable parameters: {summary['fm_trainable_parameters']}")
    print(f"prediction shape: {summary['prediction_shape']}")
    print(f"prediction finite: {summary['prediction_finite']}")
    print(f"loss: {summary['loss']:.9f}")
    print("\nGRADIENT")
    print(f"FM nonzero gradient parameters: {summary['fm_nonzero_gradients']}")
    print(f"FM nan/inf gradients: {summary['fm_invalid_gradients']}")
    print(f"layer_logits gradient nonzero: {summary['layer_logits_gradient_nonzero']}")
    print(f"DPLM gradient parameters: {summary['dplm_gradient_parameters']}")
    print("\nCONDITION SENSITIVITY")
    print(
        "actual-vs-zero hidden mean abs diff: "
        f"{summary['sensitivity_mean']:.9f}"
    )
    print(
        "actual-vs-zero hidden max abs diff: "
        f"{summary['sensitivity_max']:.9f}"
    )
    print("\nGPU MEMORY")
    print(f"peak allocated MiB: {summary['peak_allocated_mib']:.6f}")
    print(f"peak reserved MiB: {summary['peak_reserved_mib']:.6f}")
    print(f"\nFINAL STATUS: {summary['final_status']}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    summary = default_summary(device)
    try:
        shard = torch.load(Path(args.shard), map_location="cpu")
        require(shard["num_samples"] == len(shard["samples"]), "bad shard count")
        sample = shard["samples"][0]
        summary["sample_id"] = sample["sample_id"]
        summary["length"] = int(sample["length"])
        length = summary["length"]
        aatype = sample["aatype"].long().unsqueeze(0)
        struct_ids = sample["struct_ids_current"].long().unsqueeze(0)
        z_quant = sample["z_quant_current"].float().unsqueeze(0).to(device)
        residual = sample["residual"].float().unsqueeze(0).to(device)
        res_mask = sample["res_mask"].float().unsqueeze(0).to(device)
        require(tuple(aatype.shape) == (1, length), "aatype shape mismatch")
        require(tuple(struct_ids.shape) == (1, length), "struct ID shape mismatch")
        require(tuple(z_quant.shape) == (1, length, 13), "z_quant shape mismatch")
        require(tuple(residual.shape) == (1, length, 13), "residual shape mismatch")
        require(tuple(res_mask.shape) == (1, length), "res_mask shape mismatch")

        extractor = DPLMConditionExtractor(CHECKPOINT, device)
        dplm = extractor.model
        summary["dplm_total_parameters"] = sum(p.numel() for p in dplm.parameters())
        summary["dplm_trainable_parameters"] = sum(
            p.numel() for p in dplm.parameters() if p.requires_grad
        )
        require(summary["dplm_trainable_parameters"] == 0, "DPLM is trainable")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        condition = extractor(aatype, struct_ids, res_mask.cpu())
        hidden_states = condition["hidden_states"]
        summary["packed_shape"] = tuple(condition["packed_tokens"].shape)
        summary["all_hidden_states"] = condition["all_hidden_states_count"]
        summary["transformer_states_used"] = condition["transformer_states_used"]
        summary["first_shape"] = tuple(hidden_states[0].shape)
        summary["middle_shape"] = tuple(hidden_states[len(hidden_states) // 2].shape)
        summary["last_shape"] = tuple(hidden_states[-1].shape)
        summary["hidden_finite"] = all(
            bool(torch.isfinite(state).all()) for state in hidden_states
        )
        require(summary["all_hidden_states"] == 34, "hidden state count is not 34")
        require(summary["transformer_states_used"] == 33, "used state count is not 33")
        require(
            all(tuple(state.shape) == (1, length, 1280) for state in hidden_states),
            "one or more aligned hidden shapes are invalid",
        )
        require(summary["hidden_finite"], "aligned hidden is non-finite")

        fm = LatentResidualFM().to(device)
        summary["fm_trainable_parameters"] = sum(
            p.numel() for p in fm.parameters() if p.requires_grad
        )
        flow = sample_flow_matching_batch(residual, res_mask)
        prediction = fm(
            flow["r_t"], flow["t"], z_quant, hidden_states, res_mask
        )
        summary["prediction_shape"] = tuple(prediction.shape)
        summary["prediction_finite"] = bool(torch.isfinite(prediction).all())
        require(tuple(prediction.shape) == (1, length, 13), "FM prediction shape mismatch")
        require(summary["prediction_finite"], "FM prediction is non-finite")
        loss = masked_flow_matching_loss(prediction, flow["u_t"], res_mask)
        summary["loss"] = float(loss.detach())
        require(bool(torch.isfinite(loss)) and summary["loss"] > 0, "invalid FM loss")

        loss.backward()
        _, nonzero, invalid = gradient_counts(fm)
        summary["fm_nonzero_gradients"] = nonzero
        summary["fm_invalid_gradients"] = invalid
        layer_gradient = fm.layer_logits.grad
        summary["layer_logits_gradient_nonzero"] = bool(
            layer_gradient is not None
            and torch.isfinite(layer_gradient).all()
            and (layer_gradient != 0).any()
        )
        dplm_grad_present, _, _ = gradient_counts(dplm)
        summary["dplm_gradient_parameters"] = dplm_grad_present
        require(nonzero > 0 and invalid == 0, "FM gradient check failed")
        require(summary["layer_logits_gradient_nonzero"], "layer_logits gradient failed")
        require(dplm_grad_present == 0, "a DPLM parameter received a gradient")

        with torch.no_grad():
            zero_hidden = tuple(torch.zeros_like(state) for state in hidden_states)
            zero_prediction = fm(
                flow["r_t"], flow["t"], z_quant, zero_hidden, res_mask
            )
        difference = (prediction.detach() - zero_prediction).abs()
        valid_difference = difference[res_mask.bool()]
        summary["sensitivity_mean"] = float(valid_difference.mean())
        summary["sensitivity_max"] = float(valid_difference.max())
        require(summary["sensitivity_max"] > 0, "FM output ignores hidden condition")

        if device.type == "cuda":
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024**2)
        summary["final_status"] = "DPLM_FM_CONDITION_PASS"
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if device.type == "cuda" and torch.cuda.is_available():
            summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
            summary["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024**2)

    print_summary(summary)
    return 0 if summary["final_status"] == "DPLM_FM_CONDITION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

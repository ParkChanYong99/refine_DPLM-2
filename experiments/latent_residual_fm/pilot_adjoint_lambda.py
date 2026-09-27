#!/usr/bin/env python3
"""Held-out internal-validation pilot for three Adjoint Matching lambdas.

Every lambda starts from fresh finalized FM copies. Pilot checkpoints and
reports are explicitly non-production and are written only below the dedicated
am_lambda_pilot run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm.adjoint_matching_core import (
    adjoint_matching_loss,
    lean_adjoint_step,
    sigma_offset,
)
from experiments.latent_residual_fm.flow_matching import euler_sample


LAMBDA_GRID = (0.25, 0.5, 1.0)
LENGTH_BINS = ((50, 128), (129, 256), (257, 384), (385, 512))
SAMPLES_PER_BIN = 4
LEARNING_RATE_DIAGNOSTIC = 2e-6
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
NUM_UPDATES = 200
K = 40
H = 1.0 / K
ALPHA_REFINE = 0.5
RESIDUAL_DIM = 13
EXPECTED_FINE_PARAMS = 34_946_606
EVALUATION_STEPS = (50, 100, 200)
EARLY_INDICES = tuple(range(30))
LATE_INDICES = tuple(range(30, 40))
SUBSET_SEED = 42
INFERENCE_SEED_NAMESPACE = "adjoint-lambda-pilot-inference-v1"
TRAJECTORY_SEED_NAMESPACE = "adjoint-lambda-pilot-trajectory-v1"
TIE_ATOL = 1e-12
OUTPUT_RELATIVE = Path("experiments/latent_residual_fm/runs/am_lambda_pilot")


@dataclass(frozen=True)
class Selection:
    split: str
    split_index: int
    sample_id: str
    length: int
    length_bin: str
    shard_file: str
    index_in_shard: int


@dataclass
class Bundle:
    selection: Selection
    z_cont: torch.Tensor
    z_quant: torch.Tensor
    res_mask: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    oracle: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TargetMetric:
    sample_id: str
    length: int
    aligned_loss: float
    aligned_sqrt: float
    latent_mse: float


@dataclass(frozen=True)
class ValidationResult:
    lambda_value: float | None
    step: int
    targets: tuple[TargetMetric, ...]
    mean_loss: float
    median_loss: float
    mean_latent_mse: float
    delta_vs_base: float
    improved: int
    worse: int
    tied: int
    parameter_drift: float
    relative_parameter_drift: float
    finite: bool
    seconds: float


@dataclass(frozen=True)
class UpdateMetric:
    step: int
    sample_id: str
    am_loss: float
    terminal_loss: float
    terminal_adjoint_norm: float
    preclip_grad_norm: float
    postclip_grad_norm: float
    clipped: bool
    seconds: float


@dataclass
class LambdaRun:
    lambda_value: float
    validations: list[ValidationResult]
    updates: list[UpdateMetric]
    best_step: int
    best_validation: ValidationResult
    peak_allocated_mib: float
    peak_reserved_mib: float
    frozen_ok: bool
    finite: bool
    seconds: float
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir", default="data-bin/latent_residual_fm/afdb_l512_sharded"
    )
    parser.add_argument(
        "--split_csv",
        default="data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/latent_residual_fm/runs/"
            "fm_modeA_current_main/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(OUTPUT_RELATIVE))
    return parser.parse_args()


def stable_seed(namespace: str, sample_id: str, step: int = 0) -> int:
    payload = f"{namespace}|{sample_id}|{step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified(split_csv: Path, split: str) -> tuple[Selection, ...]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    integration.require(rows, f"{split} split is empty")
    selected: list[Selection] = []
    for lower, upper in LENGTH_BINS:
        candidates = sorted(
            (
                (index, row)
                for index, row in enumerate(rows)
                if lower <= int(row["length"]) <= upper
            ),
            key=lambda item: (int(item[1]["length"]), item[1]["sample_id"]),
        )
        integration.require(
            len(candidates) >= SAMPLES_PER_BIN,
            f"{split} bin {lower}-{upper} has fewer than four samples",
        )
        for split_index, row in candidates[:SAMPLES_PER_BIN]:
            selected.append(
                Selection(
                    split=split,
                    split_index=split_index,
                    sample_id=row["sample_id"],
                    length=int(row["length"]),
                    length_bin=f"{lower}-{upper}",
                    shard_file=row["shard_file"],
                    index_in_shard=int(row["index_in_shard"]),
                )
            )
    integration.require(len(selected) == 16, f"selected {len(selected)} {split} samples")
    integration.require(len({item.sample_id for item in selected}) == 16, "duplicate sample")
    return tuple(selected)


def write_indices(path: Path, selections: Sequence[Selection]) -> None:
    fields = tuple(Selection.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selection in selections:
            writer.writerow(asdict(selection))


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_bundle(
    dataset_dir: Path,
    selection: Selection,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> Bundle:
    shard_path = dataset_dir / selection.shard_file
    shard = torch.load(str(shard_path), map_location="cpu", mmap=True)
    integration.require(isinstance(shard, dict) and "samples" in shard, "invalid shard")
    integration.require(selection.index_in_shard < len(shard["samples"]), "bad shard index")
    sample = shard["samples"][selection.index_in_shard]
    integration.require(sample["sample_id"] == selection.sample_id, "sample ID mismatch")
    integration.require(int(sample["length"]) == selection.length, "sample length mismatch")
    length = selection.length
    z_cont = sample["z_cont"].float().reshape(1, length, RESIDUAL_DIM).to(device)
    z_quant = (
        sample["z_quant_current"]
        .float()
        .reshape(1, length, RESIDUAL_DIM)
        .to(device)
    )
    res_mask = sample["res_mask"].float().reshape(1, length).to(device)
    aatype = sample["aatype"].long().reshape(1, length)
    struct_ids = sample["struct_ids_current"].long().reshape(1, length)
    integration.require(bool(torch.isfinite(z_cont).all()), "z_cont NaN/Inf")
    integration.require(bool(torch.isfinite(z_quant).all()), "z_quant NaN/Inf")
    integration.require(bool(((res_mask == 0) | (res_mask == 1)).all()), "bad mask")
    condition = extractor(aatype, struct_ids, res_mask)["hidden_states"]
    integration.require(len(condition) == 33, "Mode-A layer count mismatch")
    integration.require(
        all(tuple(hidden.shape) == (1, length, 1280) for hidden in condition),
        "Mode-A condition shape mismatch",
    )
    with torch.no_grad():
        oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)
    del shard, sample, aatype, struct_ids
    return Bundle(
        selection=selection,
        z_cont=z_cont,
        z_quant=z_quant,
        res_mask=res_mask,
        hidden_states=tuple(condition),
        oracle=oracle,
    )


def model_versions(module: torch.nn.Module) -> dict[str, int]:
    return {name: parameter._version for name, parameter in module.named_parameters()}


def versions_unchanged(module: torch.nn.Module, versions: dict[str, int]) -> bool:
    current = tuple(module.named_parameters())
    return len(current) == len(versions) and all(
        name in versions and parameter._version == versions[name]
        for name, parameter in current
    )


def models_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return tuple(left_state) == tuple(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def model_pair_initialization(
    base: torch.nn.Module, fine: torch.nn.Module, probe_bundle: Bundle
) -> bool:
    independent = all(
        base_parameter.data_ptr() != fine_parameter.data_ptr()
        for base_parameter, fine_parameter in zip(base.parameters(), fine.parameters())
    )
    trainable = sum(p.numel() for p in fine.parameters() if p.requires_grad)
    generator = torch.Generator(device=probe_bundle.z_quant.device).manual_seed(42)
    state = torch.randn(
        probe_bundle.z_quant.shape,
        generator=generator,
        device=probe_bundle.z_quant.device,
        dtype=probe_bundle.z_quant.dtype,
    ) * probe_bundle.res_mask[..., None]
    time_value = torch.full((1,), 0.5, device=state.device, dtype=state.dtype)
    with torch.no_grad():
        base_output = base(
            state,
            time_value,
            probe_bundle.z_quant,
            probe_bundle.hidden_states,
            probe_bundle.res_mask,
        )
        fine_output = fine(
            state,
            time_value,
            probe_bundle.z_quant,
            probe_bundle.hidden_states,
            probe_bundle.res_mask,
        )
    return bool(
        models_equal(base, fine)
        and independent
        and trainable == EXPECTED_FINE_PARAMS
        and all(not p.requires_grad for p in base.parameters())
        and torch.allclose(base_output, fine_output, rtol=0.0, atol=0.0)
    )


def parameter_drift(
    base: torch.nn.Module, fine: torch.nn.Module
) -> tuple[float, float]:
    difference_squared = 0.0
    base_squared = 0.0
    for base_parameter, fine_parameter in zip(base.parameters(), fine.parameters()):
        difference_norm = float(
            torch.linalg.vector_norm(fine_parameter.detach() - base_parameter.detach())
        )
        base_norm = float(torch.linalg.vector_norm(base_parameter.detach()))
        difference_squared += difference_norm * difference_norm
        base_squared += base_norm * base_norm
    drift = math.sqrt(difference_squared)
    return drift, drift / math.sqrt(base_squared)


def gradient_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()))
            squared += norm * norm
    return math.sqrt(squared)


def terminal_adjoint(
    x_hat_1: torch.Tensor,
    bundle: Bundle,
    tokenizer: torch.nn.Module,
    lambda_value: float,
) -> tuple[float, torch.Tensor]:
    terminal_leaf = x_hat_1.detach().requires_grad_(True)
    z_refined = bundle.z_quant + ALPHA_REFINE * terminal_leaf
    decoded = tokenizer.detokenize(z_refined, res_mask=bundle.res_mask)
    prediction, target, valid = integration.backbone_points(
        decoded, bundle.oracle, bundle.res_mask
    )
    structural_loss = integration.aligned_squared_rmsd(prediction, target, valid)
    cost = lambda_value * structural_loss
    adjoint = torch.autograd.grad(
        cost,
        terminal_leaf,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    adjoint = adjoint * bundle.res_mask[..., None]
    loss_value = float(structural_loss.detach())
    integration.require(math.isfinite(loss_value) and loss_value > 0.0, "bad terminal loss")
    integration.require(bool(torch.isfinite(adjoint).all()), "terminal adjoint NaN/Inf")
    integration.require(float(torch.linalg.vector_norm(adjoint)) > 0.0, "zero adjoint")
    del terminal_leaf, z_refined, decoded, prediction, target, valid, structural_loss, cost
    return loss_value, adjoint


def build_adjoints(
    states: Sequence[torch.Tensor],
    a1: torch.Tensor,
    base_fn: integration.VelocityFn,
    res_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    adjoints: list[torch.Tensor | None] = [None] * (K + 1)
    adjoints[K] = a1.detach()
    for index in range(K, 0, -1):
        current = adjoints[index]
        integration.require(current is not None, f"missing adjoint {index}")
        adjoints[index - 1] = lean_adjoint_step(
            states[index], current, index * H, H, base_fn, res_mask
        )
    resolved = tuple(adjoint for adjoint in adjoints if adjoint is not None)
    integration.require(len(resolved) == K + 1, "adjoint grid incomplete")
    integration.require(
        all(bool(torch.isfinite(adjoint).all()) for adjoint in resolved),
        "adjoint NaN/Inf",
    )
    return resolved


def choose_timesteps(rng: random.Random) -> tuple[int, ...]:
    selected = tuple(sorted(rng.sample(EARLY_INDICES, 10))) + LATE_INDICES
    integration.require(len(selected) == 20, "bad paper H.2 subset")
    return selected


def update_once(
    step: int,
    bundle: Bundle,
    lambda_value: float,
    base: torch.nn.Module,
    fine: torch.nn.Module,
    tokenizer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    subset_rng: random.Random,
    device: torch.device,
) -> UpdateMetric:
    integration.synchronize(device)
    start = time.perf_counter()
    base_fn = integration.velocity_adapter(
        base, bundle.z_quant, bundle.hidden_states, bundle.res_mask
    )
    fine_fn = integration.velocity_adapter(
        fine, bundle.z_quant, bundle.hidden_states, bundle.res_mask
    )
    trajectory_seed = stable_seed(
        TRAJECTORY_SEED_NAMESPACE, bundle.selection.sample_id, step
    )
    states = integration.build_trajectory(fine_fn, bundle.res_mask, trajectory_seed)
    integration.require(len(states) == K + 1, "trajectory grid incomplete")
    x_hat_1 = integration.make_x_hat_1(states, base_fn, bundle.res_mask)
    terminal_loss, a1 = terminal_adjoint(
        x_hat_1, bundle, tokenizer, lambda_value
    )
    adjoints = build_adjoints(states, a1, base_fn, bundle.res_mask)
    selected_timesteps = choose_timesteps(subset_rng)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index in selected_timesteps:
        time_value = torch.tensor(
            index * H, device=device, dtype=bundle.z_quant.dtype
        )
        losses.append(
            adjoint_matching_loss(
                states[index],
                adjoints[index],
                time_value,
                fine_fn,
                base_fn,
                sigma_offset(time_value, H),
                bundle.res_mask,
            )
        )
    loss = torch.stack(losses).sum()
    integration.require(bool(torch.isfinite(loss)), "AM loss NaN/Inf")
    loss_value = float(loss.detach())
    loss.backward()
    preclip = float(
        torch.nn.utils.clip_grad_norm_(
            fine.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True
        )
    )
    postclip = gradient_norm(fine)
    integration.require(math.isfinite(preclip) and math.isfinite(postclip), "grad NaN/Inf")
    integration.require(integration.no_parameter_grads(base), "base received gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    integration.synchronize(device)
    seconds = time.perf_counter() - start
    metric = UpdateMetric(
        step=step,
        sample_id=bundle.selection.sample_id,
        am_loss=loss_value,
        terminal_loss=terminal_loss,
        terminal_adjoint_norm=float(torch.linalg.vector_norm(a1)),
        preclip_grad_norm=preclip,
        postclip_grad_norm=postclip,
        clipped=preclip > GRADIENT_CLIP_NORM,
        seconds=seconds,
    )
    del base_fn, fine_fn, states, x_hat_1, a1, adjoints, losses, loss
    return metric


def masked_latent_mse(
    prediction: torch.Tensor, target: torch.Tensor, res_mask: torch.Tensor
) -> float:
    denominator = res_mask.sum() * prediction.shape[-1]
    integration.require(bool(denominator > 0), "empty latent mask")
    return float(((prediction - target).square() * res_mask[..., None]).sum() / denominator)


def compare_to_base(
    losses: Sequence[float], base_losses: Sequence[float]
) -> tuple[int, int, int]:
    improved = 0
    worse = 0
    tied = 0
    for loss, base_loss in zip(losses, base_losses):
        delta = base_loss - loss
        if abs(delta) <= TIE_ATOL:
            tied += 1
        elif delta > 0.0:
            improved += 1
        else:
            worse += 1
    return improved, worse, tied


def evaluate(
    model: torch.nn.Module,
    bundles: Sequence[Bundle],
    base_result: ValidationResult | None,
    lambda_value: float | None,
    step: int,
    base_model: torch.nn.Module | None,
    device: torch.device,
) -> tuple[ValidationResult, bool]:
    integration.synchronize(device)
    start = time.perf_counter()
    targets: list[TargetMetric] = []
    step_zero_equal = True
    with torch.no_grad():
        for bundle in bundles:
            generator = torch.Generator(device=device).manual_seed(
                stable_seed(INFERENCE_SEED_NAMESPACE, bundle.selection.sample_id)
            )
            initial_noise = torch.randn(
                bundle.z_quant.shape,
                generator=generator,
                device=device,
                dtype=bundle.z_quant.dtype,
            ) * bundle.res_mask[..., None]
            residual = euler_sample(
                model,
                bundle.z_quant,
                bundle.hidden_states,
                bundle.res_mask,
                100,
                initial_noise=initial_noise,
            )
            if base_model is not None:
                base_residual = euler_sample(
                    base_model,
                    bundle.z_quant,
                    bundle.hidden_states,
                    bundle.res_mask,
                    100,
                    initial_noise=initial_noise,
                )
                step_zero_equal = step_zero_equal and torch.allclose(
                    residual, base_residual, rtol=0.0, atol=0.0
                )
                del base_residual
            decoded = model_decode(bundle, residual)
            prediction, target, valid = integration.backbone_points(
                decoded, bundle.oracle, bundle.res_mask
            )
            aligned_tensor = integration.aligned_squared_rmsd(
                prediction, target, valid
            )
            aligned_loss = float(aligned_tensor)
            latent_mse = masked_latent_mse(
                residual, bundle.z_cont - bundle.z_quant, bundle.res_mask
            )
            integration.require(
                math.isfinite(aligned_loss) and math.isfinite(latent_mse),
                "validation NaN/Inf",
            )
            targets.append(
                TargetMetric(
                    sample_id=bundle.selection.sample_id,
                    length=bundle.selection.length,
                    aligned_loss=aligned_loss,
                    aligned_sqrt=math.sqrt(max(0.0, aligned_loss)),
                    latent_mse=latent_mse,
                )
            )
            del initial_noise, residual, decoded, prediction, target, valid, aligned_tensor
    losses = [target.aligned_loss for target in targets]
    if base_result is None:
        delta = 0.0
        improved, worse, tied = 0, 0, len(targets)
    else:
        base_losses = [target.aligned_loss for target in base_result.targets]
        delta = base_result.mean_loss - sum(losses) / len(losses)
        improved, worse, tied = compare_to_base(losses, base_losses)
    drift, relative_drift = (
        (0.0, 0.0)
        if base_model is None
        else parameter_drift(base_model, model)
    )
    integration.synchronize(device)
    result = ValidationResult(
        lambda_value=lambda_value,
        step=step,
        targets=tuple(targets),
        mean_loss=sum(losses) / len(losses),
        median_loss=statistics.median(losses),
        mean_latent_mse=sum(target.latent_mse for target in targets) / len(targets),
        delta_vs_base=delta,
        improved=improved,
        worse=worse,
        tied=tied,
        parameter_drift=drift,
        relative_parameter_drift=relative_drift,
        finite=all(
            math.isfinite(value)
            for target in targets
            for value in (target.aligned_loss, target.latent_mse)
        ),
        seconds=time.perf_counter() - start,
    )
    return result, step_zero_equal


def model_decode(bundle: Bundle, residual: torch.Tensor) -> dict[str, torch.Tensor]:
    # The tokenizer is reachable through the fixed oracle only conceptually;
    # attach it explicitly in main for clear frozen ownership.
    tokenizer = _TOKENIZER_HOLDER[0]
    return tokenizer.detokenize(
        bundle.z_quant + ALPHA_REFINE * residual, res_mask=bundle.res_mask
    )


_TOKENIZER_HOLDER: list[torch.nn.Module] = []


def print_validation(label: str, result: ValidationResult) -> None:
    print(f"{label}:", flush=True)
    print(f"  mean: {result.mean_loss:.9g}", flush=True)
    print(f"  median: {result.median_loss:.9g}", flush=True)
    print(f"  delta vs base: {result.delta_vs_base:.9g}", flush=True)
    print(
        f"  improved/worse/tied: {result.improved}/{result.worse}/{result.tied}",
        flush=True,
    )
    print(f"  mean latent MSE: {result.mean_latent_mse:.9g}", flush=True)
    print(f"  finite: {result.finite}", flush=True)
    for target in result.targets:
        print(
            f"    {target.sample_id} L={target.length}: "
            f"loss={target.aligned_loss:.9g} sqrt={target.aligned_sqrt:.9g} "
            f"latent_mse={target.latent_mse:.9g}",
            flush=True,
        )


def save_pilot_checkpoint(
    path: Path,
    model: torch.nn.Module,
    lambda_value: float,
    validation: ValidationResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "label": "PILOT_NON_PRODUCTION",
        "production_checkpoint": False,
        "lambda_diagnostic": lambda_value,
        "learning_rate_diagnostic": LEARNING_RATE_DIAGNOSTIC,
        "pilot_step": validation.step,
        "validation_mean_aligned_loss": validation.mean_loss,
        "fm_model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    del payload


def lambda_label(value: float) -> str:
    return str(value).replace(".", "p")


def update_summary(metrics: Sequence[UpdateMetric]) -> dict[str, float | int]:
    am_losses = [metric.am_loss for metric in metrics]
    terminal_losses = [metric.terminal_loss for metric in metrics]
    preclip = [metric.preclip_grad_norm for metric in metrics]
    postclip = [metric.postclip_grad_norm for metric in metrics]
    return {
        "am_loss_mean": sum(am_losses) / len(am_losses),
        "am_loss_min": min(am_losses),
        "am_loss_max": max(am_losses),
        "terminal_loss_mean": sum(terminal_losses) / len(terminal_losses),
        "terminal_loss_min": min(terminal_losses),
        "terminal_loss_max": max(terminal_losses),
        "preclip_grad_mean": sum(preclip) / len(preclip),
        "preclip_grad_min": min(preclip),
        "preclip_grad_max": max(preclip),
        "postclip_grad_mean": sum(postclip) / len(postclip),
        "postclip_grad_min": min(postclip),
        "postclip_grad_max": max(postclip),
        "clipped_updates": sum(metric.clipped for metric in metrics),
        "clip_fraction": sum(metric.clipped for metric in metrics) / len(metrics),
        "mean_update_seconds": sum(metric.seconds for metric in metrics) / len(metrics),
    }


def write_summary_csv(
    path: Path,
    base_result: ValidationResult,
    runs: Sequence[LambdaRun],
) -> None:
    fields = (
        "record",
        "lambda",
        "step",
        "mean_aligned_loss",
        "median_aligned_loss",
        "delta_vs_base",
        "improved",
        "worse",
        "tied",
        "mean_latent_mse",
        "parameter_drift",
        "relative_parameter_drift",
        "finite",
        "best_for_lambda",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "record": "base",
                "lambda": "",
                "step": 0,
                "mean_aligned_loss": base_result.mean_loss,
                "median_aligned_loss": base_result.median_loss,
                "delta_vs_base": 0.0,
                "improved": 0,
                "worse": 0,
                "tied": len(base_result.targets),
                "mean_latent_mse": base_result.mean_latent_mse,
                "parameter_drift": 0.0,
                "relative_parameter_drift": 0.0,
                "finite": base_result.finite,
                "best_for_lambda": False,
            }
        )
        for run in runs:
            for validation in run.validations:
                writer.writerow(
                    {
                        "record": "pilot",
                        "lambda": run.lambda_value,
                        "step": validation.step,
                        "mean_aligned_loss": validation.mean_loss,
                        "median_aligned_loss": validation.median_loss,
                        "delta_vs_base": validation.delta_vs_base,
                        "improved": validation.improved,
                        "worse": validation.worse,
                        "tied": validation.tied,
                        "mean_latent_mse": validation.mean_latent_mse,
                        "parameter_drift": validation.parameter_drift,
                        "relative_parameter_drift": validation.relative_parameter_drift,
                        "finite": validation.finite,
                        "best_for_lambda": validation.step == run.best_step,
                    }
                )


def summary_text(
    base_result: ValidationResult,
    runs: Sequence[LambdaRun],
    selected: LambdaRun | None,
    total_seconds: float,
    peak_allocated: float,
    peak_reserved: float,
) -> str:
    lines = [
        "ADJOINT_LAMBDA_PILOT",
        "",
        "PILOT / NON-PRODUCTION",
        f"base mean: {base_result.mean_loss:.9g}",
        f"base median: {base_result.median_loss:.9g}",
    ]
    for run in runs:
        diagnostics = update_summary(run.updates)
        best = run.best_validation
        lines.extend(
            [
                "",
                f"lambda={run.lambda_value}",
                f"best step: {run.best_step}",
                f"mean: {best.mean_loss:.9g}",
                f"median: {best.median_loss:.9g}",
                f"delta vs base: {best.delta_vs_base:.9g}",
                f"improved/worse/tied: {best.improved}/{best.worse}/{best.tied}",
                f"AM loss mean/range: {diagnostics['am_loss_mean']:.9g} "
                f"[{diagnostics['am_loss_min']:.9g}, {diagnostics['am_loss_max']:.9g}]",
                f"terminal L mean/range: {diagnostics['terminal_loss_mean']:.9g} "
                f"[{diagnostics['terminal_loss_min']:.9g}, "
                f"{diagnostics['terminal_loss_max']:.9g}]",
                f"preclip grad mean/range: {diagnostics['preclip_grad_mean']:.9g} "
                f"[{diagnostics['preclip_grad_min']:.9g}, "
                f"{diagnostics['preclip_grad_max']:.9g}]",
                f"postclip grad mean/range: {diagnostics['postclip_grad_mean']:.9g} "
                f"[{diagnostics['postclip_grad_min']:.9g}, "
                f"{diagnostics['postclip_grad_max']:.9g}]",
                f"clipped updates/fraction: {diagnostics['clipped_updates']}/"
                f"{len(run.updates)} ({diagnostics['clip_fraction']:.9g})",
                f"parameter drift: {best.parameter_drift:.9g}",
                f"relative drift: {best.relative_parameter_drift:.9g}",
                f"finite: {run.finite}",
                f"peak CUDA allocated/reserved MiB: "
                f"{run.peak_allocated_mib:.2f}/{run.peak_reserved_mib:.2f}",
                f"runtime seconds: {run.seconds:.3f}",
            ]
        )
    lines.extend([""])
    if selected is None:
        lines.append("LAMBDA_PILOT_NO_CLEAR_IMPROVEMENT")
        lines.append("selected lambda: NONE")
        lines.append("selected pilot step: NONE")
    else:
        lines.append(f"selected lambda: {selected.lambda_value}")
        lines.append(f"selected pilot step: {selected.best_step}")
    lines.extend(
        [
            "production lambda: NOT YET SELECTED/FROZEN",
            f"peak CUDA allocated MiB: {peak_allocated:.2f}",
            f"peak CUDA reserved MiB: {peak_reserved:.2f}",
            f"total runtime seconds: {total_seconds:.3f}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    integration.seed_everything(42)
    device = integration.resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    stage = "startup"
    current_lambda: float | None = None
    current_step = 0
    current_sample = "none"
    total_start = time.perf_counter()
    print("ADJOINT_LAMBDA_PILOT", flush=True)
    try:
        integration.require(device.type == "cuda", "pilot requires CUDA")
        integration.require(
            output_dir.resolve().is_relative_to((REPOSITORY_ROOT / OUTPUT_RELATIVE).resolve()),
            "output_dir must remain below the dedicated pilot directory",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint_digest_before = file_digest(checkpoint_path)

        stage = "select stratified train and validation samples"
        train_selections = select_stratified(Path(args.split_csv), "train")
        validation_selections = select_stratified(Path(args.split_csv), "val")
        integration.require(
            not ({item.sample_id for item in train_selections} & {item.sample_id for item in validation_selections}),
            "train/validation pilot selections overlap",
        )
        write_indices(output_dir / "pilot_train_indices.csv", train_selections)
        write_indices(output_dir / "pilot_validation_indices.csv", validation_selections)
        print("train samples:", flush=True)
        for item in train_selections:
            print(f"  {item.sample_id} length={item.length} bin={item.length_bin}", flush=True)
        print("validation samples:", flush=True)
        for item in validation_selections:
            print(f"  {item.sample_id} length={item.length} bin={item.length_bin}", flush=True)

        metadata = {
            "label": "PILOT_NON_PRODUCTION",
            "status": "RUNNING",
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_digest_before,
            "lambda_grid": list(LAMBDA_GRID),
            "learning_rate_diagnostic": LEARNING_RATE_DIAGNOSTIC,
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_clip": GRADIENT_CLIP_NORM,
            "updates_per_lambda": NUM_UPDATES,
            "physical_batch": 1,
            "gradient_accumulation": 1,
            "am_timestep_policy": "paper_H2_20",
            "per_timestep_loss_clipping": False,
            "alpha_refine": ALPHA_REFINE,
            "production_hyperparameters_frozen": False,
            "train_samples": [asdict(item) for item in train_selections],
            "validation_samples": [asdict(item) for item in validation_selections],
            "completed_lambdas": [],
        }
        write_json(output_dir / "metadata.json", metadata)

        stage = "load frozen DPLM/tokenizer and cache fixed conditions"
        extractor = integration.DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        _TOKENIZER_HOLDER.append(tokenizer)
        dplm_versions = model_versions(dplm)
        tokenizer_versions = model_versions(tokenizer)
        dataset_dir = Path(args.dataset_dir)
        train_bundles = tuple(
            load_bundle(dataset_dir, item, extractor, tokenizer, device)
            for item in train_selections
        )
        validation_bundles = tuple(
            load_bundle(dataset_dir, item, extractor, tokenizer, device)
            for item in validation_selections
        )

        stage = "base held-out validation"
        reference_base, _, _, _ = integration.load_fm(checkpoint_path, device)
        reference_base.requires_grad_(False).eval()
        base_result, _ = evaluate(
            reference_base,
            validation_bundles,
            None,
            None,
            0,
            None,
            device,
        )
        print_validation("base validation", base_result)
        del reference_base
        torch.cuda.empty_cache()

        runs: list[LambdaRun] = []
        global_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        global_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

        for lambda_value in LAMBDA_GRID:
            current_lambda = lambda_value
            current_step = 0
            stage = f"initialize lambda={lambda_value} from finalized checkpoint"
            print(f"\nlambda={lambda_value}: fresh checkpoint initialization", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            run_start = time.perf_counter()
            base, base_step, _, _ = integration.load_fm(checkpoint_path, device)
            fine, fine_step, _, _ = integration.load_fm(checkpoint_path, device)
            integration.require(base_step == fine_step == 100_000, "checkpoint step mismatch")
            base.requires_grad_(False).eval()
            fine.requires_grad_(True).eval()
            integration.require(
                model_pair_initialization(base, fine, train_bundles[0]),
                "fresh base/fine copies differ",
            )
            base_versions = model_versions(base)
            optimizer = torch.optim.AdamW(
                fine.parameters(),
                lr=LEARNING_RATE_DIAGNOSTIC,
                weight_decay=WEIGHT_DECAY,
            )
            subset_rng = random.Random(SUBSET_SEED)
            updates: list[UpdateMetric] = []
            validations: list[ValidationResult] = []
            best_validation: ValidationResult | None = None
            lambda_dir = output_dir / f"lambda_{lambda_label(lambda_value)}"
            checkpoint_output = lambda_dir / "checkpoint_best_pilot.pt"

            for step in range(1, NUM_UPDATES + 1):
                current_step = step
                bundle = train_bundles[(step - 1) % len(train_bundles)]
                current_sample = bundle.selection.sample_id
                stage = f"lambda={lambda_value} AM update {step}"
                metric = update_once(
                    step,
                    bundle,
                    lambda_value,
                    base,
                    fine,
                    tokenizer,
                    optimizer,
                    subset_rng,
                    device,
                )
                updates.append(metric)
                if step == 1 or step % 10 == 0:
                    print(
                        f"  update {step}: sample={metric.sample_id} "
                        f"AM={metric.am_loss:.7g} terminal={metric.terminal_loss:.7g} "
                        f"grad={metric.preclip_grad_norm:.7g}->{metric.postclip_grad_norm:.7g} "
                        f"clipped={metric.clipped} seconds={metric.seconds:.3f}",
                        flush=True,
                    )
                if step in EVALUATION_STEPS:
                    stage = f"lambda={lambda_value} held-out validation step {step}"
                    validation, _ = evaluate(
                        fine,
                        validation_bundles,
                        base_result,
                        lambda_value,
                        step,
                        base,
                        device,
                    )
                    validations.append(validation)
                    print_validation(f"lambda={lambda_value} step {step}", validation)
                    if best_validation is None or (
                        validation.mean_loss,
                        validation.step,
                    ) < (best_validation.mean_loss, best_validation.step):
                        best_validation = validation
                        save_pilot_checkpoint(
                            checkpoint_output, fine, lambda_value, validation
                        )

            integration.require(best_validation is not None, "no validation checkpoint")
            optimizer.zero_grad(set_to_none=True)
            finite = bool(
                all(validation.finite for validation in validations)
                and all(
                    all(
                        math.isfinite(value)
                        for value in (
                            metric.am_loss,
                            metric.terminal_loss,
                            metric.terminal_adjoint_norm,
                            metric.preclip_grad_norm,
                            metric.postclip_grad_norm,
                        )
                    )
                    for metric in updates
                )
                and all(bool(torch.isfinite(parameter).all()) for parameter in fine.parameters())
            )
            frozen_ok = bool(
                versions_unchanged(base, base_versions)
                and integration.no_parameter_grads(base)
                and integration.no_parameter_grads(dplm)
                and integration.no_parameter_grads(tokenizer)
            )
            integration.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            global_peak_allocated = max(global_peak_allocated, peak_allocated)
            global_peak_reserved = max(global_peak_reserved, peak_reserved)
            run = LambdaRun(
                lambda_value=lambda_value,
                validations=validations,
                updates=updates,
                best_step=best_validation.step,
                best_validation=best_validation,
                peak_allocated_mib=peak_allocated,
                peak_reserved_mib=peak_reserved,
                frozen_ok=frozen_ok,
                finite=finite,
                seconds=time.perf_counter() - run_start,
                checkpoint_path=str(checkpoint_output),
            )
            runs.append(run)
            diagnostics = update_summary(updates)
            print(
                f"lambda={lambda_value} complete: best_step={run.best_step} "
                f"best_mean={best_validation.mean_loss:.9g} "
                f"delta={best_validation.delta_vs_base:.9g} "
                f"improved/worse/tied={best_validation.improved}/"
                f"{best_validation.worse}/{best_validation.tied} "
                f"clip_fraction={diagnostics['clip_fraction']:.3f} "
                f"relative_drift={best_validation.relative_parameter_drift:.9g}",
                flush=True,
            )
            metadata["completed_lambdas"].append(
                {
                    "lambda": lambda_value,
                    "best_step": run.best_step,
                    "best_mean": best_validation.mean_loss,
                    "checkpoint": str(checkpoint_output),
                }
            )
            write_json(output_dir / "metadata.json", metadata)
            del optimizer, base, fine
            torch.cuda.empty_cache()

        stage = "select pilot lambda and write summaries"
        eligible = [
            run
            for run in runs
            if run.best_validation.mean_loss < base_result.mean_loss
            and run.best_validation.improved > run.best_validation.worse
            and run.finite
        ]
        selected_run = (
            min(
                eligible,
                key=lambda run: (
                    run.best_validation.mean_loss,
                    run.lambda_value,
                ),
            )
            if eligible
            else None
        )
        checkpoint_digest_after = file_digest(checkpoint_path)
        global_frozen_ok = bool(
            checkpoint_digest_before == checkpoint_digest_after
            and versions_unchanged(dplm, dplm_versions)
            and versions_unchanged(tokenizer, tokenizer_versions)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
            and all(run.frozen_ok for run in runs)
        )
        all_finite = base_result.finite and all(run.finite for run in runs)
        total_seconds = time.perf_counter() - total_start
        write_summary_csv(output_dir / "lambda_pilot_summary.csv", base_result, runs)
        text = summary_text(
            base_result,
            runs,
            selected_run,
            total_seconds,
            global_peak_allocated,
            global_peak_reserved,
        )
        (output_dir / "lambda_pilot_summary.txt").write_text(text, encoding="utf-8")
        metadata.update(
            {
                "status": "PASS" if all_finite and global_frozen_ok else "FAIL",
                "base_validation_mean": base_result.mean_loss,
                "base_validation_median": base_result.median_loss,
                "selected_lambda": None if selected_run is None else selected_run.lambda_value,
                "selected_pilot_step": None if selected_run is None else selected_run.best_step,
                "selection_status": (
                    "LAMBDA_PILOT_NO_CLEAR_IMPROVEMENT"
                    if selected_run is None
                    else "PILOT_LAMBDA_SELECTED_NON_PRODUCTION"
                ),
                "production_lambda_selected": False,
                "source_checkpoint_unchanged": checkpoint_digest_before
                == checkpoint_digest_after,
                "frozen_audit": global_frozen_ok,
                "finite_audit": all_finite,
                "peak_cuda_allocated_mib": global_peak_allocated,
                "peak_cuda_reserved_mib": global_peak_reserved,
                "total_runtime_seconds": total_seconds,
            }
        )
        write_json(output_dir / "metadata.json", metadata)

        print("\nADJOINT_LAMBDA_PILOT", flush=True)
        print(f"base validation mean: {base_result.mean_loss:.9g}", flush=True)
        print(f"base validation median: {base_result.median_loss:.9g}", flush=True)
        for run in runs:
            diagnostics = update_summary(run.updates)
            best = run.best_validation
            print(f"lambda={run.lambda_value}:", flush=True)
            print(f"  best step: {run.best_step}", flush=True)
            print(f"  mean: {best.mean_loss:.9g}", flush=True)
            print(f"  median: {best.median_loss:.9g}", flush=True)
            print(f"  delta vs base: {best.delta_vs_base:.9g}", flush=True)
            print(f"  improved/worse/tied: {best.improved}/{best.worse}/{best.tied}", flush=True)
            print(f"  clip fraction: {diagnostics['clip_fraction']:.9g}", flush=True)
            print(f"  relative drift: {best.relative_parameter_drift:.9g}", flush=True)
            print(
                f"  AM loss mean/range: {diagnostics['am_loss_mean']:.9g} "
                f"[{diagnostics['am_loss_min']:.9g}, {diagnostics['am_loss_max']:.9g}]",
                flush=True,
            )
            print(
                f"  terminal L mean/range: {diagnostics['terminal_loss_mean']:.9g} "
                f"[{diagnostics['terminal_loss_min']:.9g}, {diagnostics['terminal_loss_max']:.9g}]",
                flush=True,
            )
            print(
                f"  preclip grad mean/range: {diagnostics['preclip_grad_mean']:.9g} "
                f"[{diagnostics['preclip_grad_min']:.9g}, {diagnostics['preclip_grad_max']:.9g}]",
                flush=True,
            )
            print(
                f"  postclip grad mean/range: {diagnostics['postclip_grad_mean']:.9g} "
                f"[{diagnostics['postclip_grad_min']:.9g}, {diagnostics['postclip_grad_max']:.9g}]",
                flush=True,
            )
            print(f"  NaN/Inf free: {run.finite}", flush=True)
            print(
                f"  peak CUDA MiB allocated/reserved: "
                f"{run.peak_allocated_mib:.2f}/{run.peak_reserved_mib:.2f}",
                flush=True,
            )
            print(f"  runtime seconds: {run.seconds:.3f}", flush=True)
        if selected_run is None:
            print("LAMBDA_PILOT_NO_CLEAR_IMPROVEMENT", flush=True)
            print("selected lambda: NONE", flush=True)
            print("selected pilot step: NONE", flush=True)
        else:
            print(f"selected lambda: {selected_run.lambda_value}", flush=True)
            print(f"selected pilot step: {selected_run.best_step}", flush=True)
        print("production lambda: NOT YET SELECTED/FROZEN", flush=True)
        print(f"frozen audit: {'PASS' if global_frozen_ok else 'FAIL'}", flush=True)
        print(f"NaN/Inf audit: {'PASS' if all_finite else 'FAIL'}", flush=True)
        print(
            f"peak CUDA MiB allocated/reserved: "
            f"{global_peak_allocated:.2f}/{global_peak_reserved:.2f}",
            flush=True,
        )
        print(f"total runtime seconds: {total_seconds:.3f}", flush=True)
        overall = all_finite and global_frozen_ok and len(runs) == len(LAMBDA_GRID)
        marker = "ADJOINT_LAMBDA_PILOT_PASS" if overall else "ADJOINT_LAMBDA_PILOT_FAIL"
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if overall else 1
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        failure = {
            "label": "PILOT_NON_PRODUCTION",
            "status": "FAIL",
            "failure_stage": stage,
            "failure_lambda": current_lambda,
            "failure_step": current_step,
            "failure_sample": current_sample,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "production_lambda_selected": False,
        }
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "metadata.json", failure)
        except Exception:
            pass
        print(f"failure point: {stage}", flush=True)
        print(f"failure lambda/step/sample: {current_lambda}/{current_step}/{current_sample}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("production lambda: NOT SELECTED", flush=True)
        print("overall: ADJOINT_LAMBDA_PILOT_FAIL", flush=True)
        print("ADJOINT_LAMBDA_PILOT_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

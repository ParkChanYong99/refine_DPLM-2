#!/usr/bin/env python3
"""Audit Adjoint Matching lambda and clipping scales on internal validation.

The audit keeps the finalized base and its exact trainable copy at identical
weights. It accumulates exact per-timestep backward gradients for diagnostic
aggregates, but creates no optimizer, performs no update, and writes no model.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# This read-only import supplies the already validated model adapters,
# trajectory construction, terminal-state semantics, and CUDA import bootstrap.
from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm.adjoint_matching_core import (
    adjoint_matching_loss,
    lean_adjoint_step,
    sigma_offset,
)


LENGTH_BINS = ((50, 128), (129, 256), (257, 384), (385, 512))
SAMPLES_PER_BIN = 2
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0)
CLIPPING_CONSTANTS = (0.4, 0.8, 1.6, 3.2, 6.4)
PAPER_CLIPPING_CONSTANT = 1.6
SUBSET_SELECTION_SEED = 42
BASE_TRAJECTORY_SEED = 42
K = 40
H = 1.0 / K
ALPHA_REFINE = 0.5
RESIDUAL_DIM = 13
EXPECTED_FINE_PARAMS = 34_946_606
STATE_NORM_LIMIT = 1e3
ADJOINT_NORM_LIMIT = 1e6

EARLY_INDICES = tuple(range(30))
LATE_INDICES = tuple(range(30, 40))
_subset_rng = random.Random(SUBSET_SELECTION_SEED)
SELECTED_EARLY_INDICES = tuple(sorted(_subset_rng.sample(EARLY_INDICES, 10)))
PAPER_SUBSET_INDICES = SELECTED_EARLY_INDICES + LATE_INDICES


@dataclass(frozen=True)
class SelectedSample:
    validation_index: int
    sample_id: str
    length: int
    bin_label: str


@dataclass(frozen=True)
class GradientAggregate:
    global_norm: float
    maximum_tensor_norm: float
    finite: bool


@dataclass(frozen=True)
class LambdaResult:
    sample: SelectedSample
    lambda_value: float
    struct_loss: float
    terminal_adjoint_norm: float
    adjoint_norms: tuple[float, ...]
    timestep_losses: tuple[float, ...]
    all_loss: float
    subset_loss: float
    clipped_loss: float
    all_gradient: GradientAggregate
    subset_gradient: GradientAggregate
    clipped_gradient: GradientAggregate
    paper_overall_clip_fraction: float
    paper_early_clip_fraction: float
    paper_late_clip_fraction: float
    sensitivity: tuple[tuple[float, float, float, float], ...]
    peak_allocated_mib: float
    peak_reserved_mib: float
    nan_count: int
    inf_count: int
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="data-bin/latent_residual_fm/afdb_l512_sharded",
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
    return parser.parse_args()


def select_samples(split_csv: Path) -> tuple[SelectedSample, ...]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        validation_rows = [
            row for row in csv.DictReader(handle) if row.get("split") == "val"
        ]
    integration.require(validation_rows, "validation split is empty")
    selected: list[SelectedSample] = []
    for lower, upper in LENGTH_BINS:
        candidates = [
            (index, row)
            for index, row in enumerate(validation_rows)
            if lower <= int(row["length"]) <= upper
        ]
        integration.require(
            len(candidates) >= SAMPLES_PER_BIN,
            f"length bin {lower}-{upper} has fewer than two validation samples",
        )
        for validation_index, row in candidates[:SAMPLES_PER_BIN]:
            selected.append(
                SelectedSample(
                    validation_index=validation_index,
                    sample_id=row["sample_id"],
                    length=int(row["length"]),
                    bin_label=f"{lower}-{upper}",
                )
            )
    integration.require(len(selected) == 8, f"selected {len(selected)} samples")
    return tuple(selected)


def load_sample(
    args: argparse.Namespace,
    selected: SelectedSample,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = integration.load_validation_sample(
        Path(args.dataset_dir), Path(args.split_csv), selected.validation_index
    )
    integration.require(sample["sample_id"] == selected.sample_id, "sample ID changed")
    integration.require(int(sample["length"]) == selected.length, "sample length changed")
    length = selected.length
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
    integration.require(bool(torch.isfinite(z_cont).all()), "z_cont contains NaN/Inf")
    integration.require(bool(torch.isfinite(z_quant).all()), "z_quant contains NaN/Inf")
    integration.require(bool(((res_mask == 0) | (res_mask == 1)).all()), "nonbinary mask")
    return sample, z_cont, z_quant, res_mask, aatype, struct_ids


def models_exactly_equal(base: torch.nn.Module, fine: torch.nn.Module) -> bool:
    base_state = base.state_dict()
    fine_state = fine.state_dict()
    return tuple(base_state) == tuple(fine_state) and all(
        torch.equal(base_state[name], fine_state[name]) for name in base_state
    )


def terminal_loss_and_unit_adjoint(
    x_hat_1: torch.Tensor,
    z_quant: torch.Tensor,
    res_mask: torch.Tensor,
    tokenizer: torch.nn.Module,
    oracle: dict[str, torch.Tensor],
) -> tuple[float, torch.Tensor]:
    terminal_leaf = x_hat_1.detach().requires_grad_(True)
    integration.require(torch.is_grad_enabled(), "terminal decoder graph is disabled")
    z_refined = z_quant + ALPHA_REFINE * terminal_leaf
    decoded = tokenizer.detokenize(z_refined, res_mask=res_mask)
    prediction, target, valid = integration.backbone_points(
        decoded, oracle, res_mask
    )
    loss = integration.aligned_squared_rmsd(prediction, target, valid)
    unit_adjoint = torch.autograd.grad(
        loss,
        terminal_leaf,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    unit_adjoint = unit_adjoint * res_mask[..., None]
    loss_value = float(loss.detach())
    integration.require(math.isfinite(loss_value) and loss_value > 0.0, "bad L_struct")
    integration.require(bool(torch.isfinite(unit_adjoint).all()), "bad terminal adjoint")
    integration.require(float(torch.linalg.vector_norm(unit_adjoint)) > 0.0, "zero terminal adjoint")
    del terminal_leaf, z_refined, decoded, prediction, target, valid, loss
    return loss_value, unit_adjoint


def build_unit_adjoints(
    states: Sequence[torch.Tensor],
    unit_a1: torch.Tensor,
    base_fn: integration.VelocityFn,
    res_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    adjoints: list[torch.Tensor | None] = [None] * (K + 1)
    adjoints[K] = unit_a1.detach()
    for index in range(K, 0, -1):
        current = adjoints[index]
        integration.require(current is not None, f"missing adjoint index {index}")
        adjoints[index - 1] = lean_adjoint_step(
            states[index], current, index * H, H, base_fn, res_mask
        )
    resolved = tuple(value for value in adjoints if value is not None)
    integration.require(len(resolved) == K + 1, "incomplete adjoint grid")
    return resolved


def state_and_adjoint_audit(
    states: Sequence[torch.Tensor],
    unit_adjoints: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
) -> tuple[float, float]:
    expected_shape = (1, res_mask.shape[1], RESIDUAL_DIM)
    padding = ~res_mask.bool().unsqueeze(-1).expand(expected_shape)
    max_state_norm = 0.0
    for state in states:
        integration.require(tuple(state.shape) == expected_shape, "state shape mismatch")
        integration.require(not state.requires_grad and state.grad is None, "state graph retained")
        integration.require(bool(torch.isfinite(state).all()), "state contains NaN/Inf")
        if bool(padding.any()):
            integration.require(not bool((state[padding] != 0).any()), "nonzero padded state")
        max_state_norm = max(max_state_norm, float(torch.linalg.vector_norm(state, dim=-1).max()))
    max_adjoint_norm = 0.0
    for adjoint in unit_adjoints:
        integration.require(tuple(adjoint.shape) == expected_shape, "adjoint shape mismatch")
        integration.require(not adjoint.requires_grad and adjoint.grad is None, "adjoint graph retained")
        integration.require(bool(torch.isfinite(adjoint).all()), "adjoint contains NaN/Inf")
        if bool(padding.any()):
            integration.require(not bool((adjoint[padding] != 0).any()), "nonzero padded adjoint")
        max_adjoint_norm = max(max_adjoint_norm, float(torch.linalg.vector_norm(adjoint)))
    integration.require(max_state_norm <= STATE_NORM_LIMIT, "state explosion")
    integration.require(max_adjoint_norm <= ADJOINT_NORM_LIMIT, "adjoint explosion")
    return max_state_norm, max_adjoint_norm


def aggregate_stats(accumulators: Sequence[torch.Tensor]) -> GradientAggregate:
    squared_norm = 0.0
    maximum = 0.0
    finite = True
    for gradient in accumulators:
        tensor_finite = bool(torch.isfinite(gradient).all())
        finite = finite and tensor_finite
        if tensor_finite:
            norm = float(torch.linalg.vector_norm(gradient))
            squared_norm += norm * norm
            maximum = max(maximum, norm)
    return GradientAggregate(math.sqrt(squared_norm), maximum, finite)


def clip_fraction(
    losses: Sequence[float], indices: Sequence[int], threshold: float
) -> float:
    return sum(losses[index] > threshold for index in indices) / len(indices)


def regression_diagnostics(
    states: Sequence[torch.Tensor],
    adjoints: Sequence[torch.Tensor],
    lambda_value: float,
    base_fn: integration.VelocityFn,
    fine_fn: integration.VelocityFn,
    v_finetune: torch.nn.Module,
    res_mask: torch.Tensor,
) -> tuple[
    tuple[float, ...],
    float,
    float,
    float,
    GradientAggregate,
    GradientAggregate,
    GradientAggregate,
    int,
    int,
]:
    parameters = [p for p in v_finetune.parameters() if p.requires_grad]
    integration.require(
        sum(p.numel() for p in parameters) == EXPECTED_FINE_PARAMS,
        "unexpected fine parameter count",
    )
    all_accumulators = [torch.zeros_like(p) for p in parameters]
    subset_accumulators = [torch.zeros_like(p) for p in parameters]
    clipped_accumulators = [torch.zeros_like(p) for p in parameters]
    paper_threshold = PAPER_CLIPPING_CONSTANT * lambda_value * lambda_value
    timestep_losses: list[float] = []
    nan_counter = torch.zeros((), dtype=torch.int64, device=res_mask.device)
    inf_counter = torch.zeros((), dtype=torch.int64, device=res_mask.device)

    v_finetune.zero_grad(set_to_none=True)
    for index in range(K):
        time_value = torch.tensor(
            index * H, device=res_mask.device, dtype=res_mask.dtype
        )
        loss = adjoint_matching_loss(
            states[index],
            adjoints[index],
            time_value,
            fine_fn,
            base_fn,
            sigma_offset(time_value, H),
            res_mask,
        )
        loss_value = float(loss.detach())
        nan_counter.add_(int(math.isnan(loss_value)))
        inf_counter.add_(int(math.isinf(loss_value)))
        timestep_losses.append(loss_value)
        loss.backward()
        with torch.no_grad():
            for parameter, all_acc, subset_acc, clipped_acc in zip(
                parameters,
                all_accumulators,
                subset_accumulators,
                clipped_accumulators,
            ):
                integration.require(parameter.grad is not None, "missing fine gradient")
                nan_counter.add_(torch.isnan(parameter.grad).sum())
                inf_counter.add_(torch.isinf(parameter.grad).sum())
                all_acc.add_(parameter.grad)
                if index in PAPER_SUBSET_INDICES:
                    subset_acc.add_(parameter.grad)
                if loss_value <= paper_threshold:
                    clipped_acc.add_(parameter.grad)
        v_finetune.zero_grad(set_to_none=True)
        del loss

    all_loss = sum(timestep_losses)
    subset_loss = sum(timestep_losses[index] for index in PAPER_SUBSET_INDICES)
    clipped_loss = sum(min(value, paper_threshold) for value in timestep_losses)
    all_stats = aggregate_stats(all_accumulators)
    subset_stats = aggregate_stats(subset_accumulators)
    clipped_stats = aggregate_stats(clipped_accumulators)
    nan_count = int(nan_counter)
    inf_count = int(inf_counter)
    del all_accumulators, subset_accumulators, clipped_accumulators
    v_finetune.zero_grad(set_to_none=True)
    return (
        tuple(timestep_losses),
        all_loss,
        subset_loss,
        clipped_loss,
        all_stats,
        subset_stats,
        clipped_stats,
        nan_count,
        inf_count,
    )


def summarize(values: Sequence[float]) -> str:
    return (
        f"mean={sum(values) / len(values):.9g} "
        f"range=[{min(values):.9g}, {max(values):.9g}]"
    )


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pooled_fraction(
    results: Sequence[LambdaResult], constant: float, indices: Sequence[int]
) -> float:
    clipped = 0
    total = 0
    for result in results:
        threshold = constant * result.lambda_value * result.lambda_value
        clipped += sum(result.timestep_losses[index] > threshold for index in indices)
        total += len(indices)
    return clipped / total


def clipping_behavior(overall: float, early: float, late: float) -> str:
    # Diagnostic interpretation only: no production constant is selected.
    if overall >= 0.50 or late >= 0.50:
        return "too aggressive"
    if overall <= 0.01 or early == 0.0:
        return "too weak"
    return "reasonable"


def main() -> int:
    args = parse_args()
    integration.seed_everything(BASE_TRAJECTORY_SEED)
    device = integration.resolve_device(args.device)
    stage = "startup"
    current_sample: SelectedSample | None = None
    total_start = time.perf_counter()
    print("ADJOINT_TRAINING_SCALE_AUDIT", flush=True)
    try:
        integration.require(device.type == "cuda", "this audit requires CUDA")
        torch.backends.cudnn.benchmark = False
        selected_samples = select_samples(Path(args.split_csv))
        print("samples:", flush=True)
        for selected in selected_samples:
            print(
                f"  val_index={selected.validation_index} sample={selected.sample_id} "
                f"length={selected.length} bin={selected.bin_label} "
                f"trajectory_seed={BASE_TRAJECTORY_SEED + selected.validation_index}",
                flush=True,
            )
        print(f"lengths: {[sample.length for sample in selected_samples]}", flush=True)
        print(f"H.2 subset selection seed: {SUBSET_SELECTION_SEED}", flush=True)
        print(
            "H.2 selected times: "
            f"{[round(index * H, 3) for index in PAPER_SUBSET_INDICES]}",
            flush=True,
        )

        stage = "load frozen DPLM/tokenizer"
        extractor = integration.DPLMConditionExtractor(
            args.dplm_checkpoint, device=device
        )
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()

        stage = "load independent FM copies"
        v_base, base_step, _, _ = integration.load_fm(Path(args.checkpoint), device)
        v_finetune, fine_step, _, _ = integration.load_fm(
            Path(args.checkpoint), device
        )
        integration.require(base_step == fine_step == 100_000, "checkpoint step mismatch")
        v_base.requires_grad_(False).eval()
        v_finetune.requires_grad_(True).eval()
        initialization_ok = models_exactly_equal(v_base, v_finetune)
        initialization_ok = initialization_ok and all(
            base_parameter.data_ptr() != fine_parameter.data_ptr()
            for base_parameter, fine_parameter in zip(
                v_base.parameters(), v_finetune.parameters()
            )
        )
        initialization_ok = initialization_ok and (
            sum(p.numel() for p in v_finetune.parameters() if p.requires_grad)
            == EXPECTED_FINE_PARAMS
        )
        integration.require(initialization_ok, "base/fine initialization mismatch")
        print("base/fine initialization: PASS", flush=True)

        results: list[LambdaResult] = []
        sample_max_state_norms: dict[str, float] = {}
        sample_max_unit_adjoint_norms: dict[str, float] = {}
        output_max_differences: dict[str, float] = {}

        for sample_number, selected in enumerate(selected_samples, start=1):
            current_sample = selected
            stage = "load internal validation sample"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            print(
                f"sample {sample_number}/8: {selected.sample_id} "
                f"length={selected.length}",
                flush=True,
            )
            sample, z_cont, z_quant, res_mask, aatype, struct_ids = load_sample(
                args, selected, device
            )

            stage = "extract Mode-A condition"
            condition_result = extractor(aatype, struct_ids, res_mask)
            hidden_states = condition_result["hidden_states"]
            integration.require(len(hidden_states) == 33, "condition layer count mismatch")
            integration.require(
                all(
                    tuple(hidden.shape) == (1, selected.length, 1280)
                    for hidden in hidden_states
                ),
                "condition shape mismatch",
            )
            base_fn = integration.velocity_adapter(
                v_base, z_quant, hidden_states, res_mask
            )
            fine_fn = integration.velocity_adapter(
                v_finetune, z_quant, hidden_states, res_mask
            )

            stage = "build detached stochastic trajectory"
            trajectory_seed = BASE_TRAJECTORY_SEED + selected.validation_index
            states = integration.build_trajectory(fine_fn, res_mask, trajectory_seed)
            integration.require(len(states) == K + 1, "trajectory indexing mismatch")
            x_hat_1 = integration.make_x_hat_1(states, base_fn, res_mask)
            with torch.no_grad():
                base_probe = base_fn(states[0], torch.tensor(0.5, device=device))
                fine_probe = fine_fn(states[0], torch.tensor(0.5, device=device))
            output_max_difference = float((base_probe - fine_probe).abs().max())
            output_max_differences[selected.sample_id] = output_max_difference
            integration.require(output_max_difference == 0.0, "base/fine outputs differ")

            stage = "decode oracle and terminal structural gradient"
            with torch.no_grad():
                oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)
            struct_loss, unit_a1 = terminal_loss_and_unit_adjoint(
                x_hat_1, z_quant, res_mask, tokenizer, oracle
            )

            stage = "unit lean-adjoint recursion"
            unit_adjoints = build_unit_adjoints(states, unit_a1, base_fn, res_mask)
            max_state_norm, max_unit_adjoint_norm = state_and_adjoint_audit(
                states, unit_adjoints, res_mask
            )
            sample_max_state_norms[selected.sample_id] = max_state_norm
            sample_max_unit_adjoint_norms[selected.sample_id] = max_unit_adjoint_norm
            integration.synchronize(device)
            pre_lambda_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            pre_lambda_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

            for lambda_value in LAMBDA_GRID:
                stage = f"lambda={lambda_value} regression backward"
                print(f"  lambda={lambda_value}: running 40 timestep backward passes", flush=True)
                integration.require(
                    models_exactly_equal(v_base, v_finetune),
                    "model weights changed before lambda diagnostic",
                )
                v_finetune.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                adjoints = tuple(
                    (lambda_value * adjoint).detach() for adjoint in unit_adjoints
                )
                adjoint_norms = tuple(
                    float(torch.linalg.vector_norm(adjoint)) for adjoint in adjoints
                )
                (
                    timestep_losses,
                    all_loss,
                    subset_loss,
                    clipped_loss,
                    all_gradient,
                    subset_gradient,
                    clipped_gradient,
                    nan_count,
                    inf_count,
                ) = regression_diagnostics(
                    states,
                    adjoints,
                    lambda_value,
                    base_fn,
                    fine_fn,
                    v_finetune,
                    res_mask,
                )
                paper_threshold = PAPER_CLIPPING_CONSTANT * lambda_value**2
                paper_overall = clip_fraction(
                    timestep_losses, range(K), paper_threshold
                )
                paper_early = clip_fraction(
                    timestep_losses, EARLY_INDICES, paper_threshold
                )
                paper_late = clip_fraction(
                    timestep_losses, LATE_INDICES, paper_threshold
                )
                sensitivity = tuple(
                    (
                        constant,
                        clip_fraction(
                            timestep_losses,
                            range(K),
                            constant * lambda_value**2,
                        ),
                        clip_fraction(
                            timestep_losses,
                            EARLY_INDICES,
                            constant * lambda_value**2,
                        ),
                        clip_fraction(
                            timestep_losses,
                            LATE_INDICES,
                            constant * lambda_value**2,
                        ),
                    )
                    for constant in CLIPPING_CONSTANTS
                )
                integration.synchronize(device)
                peak_allocated = max(
                    pre_lambda_peak_allocated,
                    torch.cuda.max_memory_allocated(device) / 1024**2,
                )
                peak_reserved = max(
                    pre_lambda_peak_reserved,
                    torch.cuda.max_memory_reserved(device) / 1024**2,
                )
                finite = bool(
                    nan_count == 0
                    and inf_count == 0
                    and all(math.isfinite(value) for value in timestep_losses)
                    and all(math.isfinite(value) for value in adjoint_norms)
                    and math.isfinite(all_loss)
                    and math.isfinite(subset_loss)
                    and math.isfinite(clipped_loss)
                    and all_gradient.finite
                    and subset_gradient.finite
                    and clipped_gradient.finite
                    and all_gradient.global_norm > 0.0
                    and subset_gradient.global_norm > 0.0
                )
                result = LambdaResult(
                    sample=selected,
                    lambda_value=lambda_value,
                    struct_loss=struct_loss,
                    terminal_adjoint_norm=adjoint_norms[K],
                    adjoint_norms=adjoint_norms,
                    timestep_losses=timestep_losses,
                    all_loss=all_loss,
                    subset_loss=subset_loss,
                    clipped_loss=clipped_loss,
                    all_gradient=all_gradient,
                    subset_gradient=subset_gradient,
                    clipped_gradient=clipped_gradient,
                    paper_overall_clip_fraction=paper_overall,
                    paper_early_clip_fraction=paper_early,
                    paper_late_clip_fraction=paper_late,
                    sensitivity=sensitivity,
                    peak_allocated_mib=peak_allocated,
                    peak_reserved_mib=peak_reserved,
                    nan_count=nan_count,
                    inf_count=inf_count,
                    finite=finite,
                )
                results.append(result)
                print(
                    f"    L_struct={struct_loss:.6g} a1_norm={result.terminal_adjoint_norm:.6g} "
                    f"all_loss={all_loss:.6g} all_grad={all_gradient.global_norm:.6g} "
                    f"subset_loss={subset_loss:.6g} subset_grad={subset_gradient.global_norm:.6g} "
                    f"clipped_loss={clipped_loss:.6g} clipped_grad={clipped_gradient.global_norm:.6g} "
                    f"max_param_grad={all_gradient.maximum_tensor_norm:.6g} finite={finite}",
                    flush=True,
                )
                print(
                    f"    paper_c1.6_clip overall={percentage(paper_overall)} "
                    f"early={percentage(paper_early)} late10={percentage(paper_late)} "
                    f"NaN={nan_count} Inf={inf_count} "
                    f"peak_MiB={peak_allocated:.2f}/{peak_reserved:.2f}",
                    flush=True,
                )
                print(
                    f"    adjoint norms t=0..1: {[float(f'{value:.7g}') for value in adjoint_norms]}",
                    flush=True,
                )
                print(
                    f"    unclipped losses t=0..0.975: {[float(f'{value:.7g}') for value in timestep_losses]}",
                    flush=True,
                )
                integration.require(
                    models_exactly_equal(v_base, v_finetune),
                    "model weights changed after lambda diagnostic",
                )
                del adjoints

            v_finetune.zero_grad(set_to_none=True)
            del (
                sample,
                z_cont,
                z_quant,
                res_mask,
                aatype,
                struct_ids,
                condition_result,
                hidden_states,
                base_fn,
                fine_fn,
                states,
                x_hat_1,
                oracle,
                unit_a1,
                unit_adjoints,
                base_probe,
                fine_probe,
            )
            torch.cuda.empty_cache()

        stage = "aggregate report"
        integration.require(len(results) == 8 * len(LAMBDA_GRID), "result grid incomplete")
        frozen_ok = bool(
            integration.no_parameter_grads(v_base)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
            and models_exactly_equal(v_base, v_finetune)
        )
        all_finite = all(result.finite for result in results)
        subset_viable = all(
            result.subset_gradient.finite
            and result.subset_gradient.global_norm > 0.0
            and math.isfinite(result.subset_loss)
            and result.subset_loss > 0.0
            for result in results
        )

        print("\nPer-lambda summary:", flush=True)
        for lambda_value in LAMBDA_GRID:
            group = [result for result in results if result.lambda_value == lambda_value]
            print(f"lambda: {lambda_value}", flush=True)
            print(f"  terminal L_struct: {summarize([r.struct_loss for r in group])}", flush=True)
            print(
                f"  terminal adjoint norm: "
                f"{summarize([r.terminal_adjoint_norm for r in group])}",
                flush=True,
            )
            print(f"  unclipped AM loss: {summarize([r.all_loss for r in group])}", flush=True)
            print(
                f"  unclipped grad norm: "
                f"{summarize([r.all_gradient.global_norm for r in group])}",
                flush=True,
            )
            print(
                f"  maximum parameter-tensor grad norm: "
                f"{summarize([r.all_gradient.maximum_tensor_norm for r in group])}",
                flush=True,
            )
            print(f"  paper-20-step loss: {summarize([r.subset_loss for r in group])}", flush=True)
            print(
                f"  paper-20-step grad norm: "
                f"{summarize([r.subset_gradient.global_norm for r in group])}",
                flush=True,
            )
            print(f"  clipped AM loss: {summarize([r.clipped_loss for r in group])}", flush=True)
            print(
                f"  clipped grad norm: "
                f"{summarize([r.clipped_gradient.global_norm for r in group])}",
                flush=True,
            )
            print(
                f"  paper LCT clip fraction: "
                f"{percentage(sum(r.paper_overall_clip_fraction for r in group) / len(group))}",
                flush=True,
            )
            print(
                f"  early clip fraction: "
                f"{percentage(sum(r.paper_early_clip_fraction for r in group) / len(group))}",
                flush=True,
            )
            print(
                f"  late-10 clip fraction: "
                f"{percentage(sum(r.paper_late_clip_fraction for r in group) / len(group))}",
                flush=True,
            )
            print(f"  finite: {all(r.finite for r in group)}", flush=True)
            print(
                f"  peak allocated MiB: "
                f"{summarize([r.peak_allocated_mib for r in group])}",
                flush=True,
            )
            print(
                f"  peak reserved MiB: "
                f"{summarize([r.peak_reserved_mib for r in group])}",
                flush=True,
            )

        print("\nPer-sample adjoint behavior at lambda=1:", flush=True)
        for result in (r for r in results if r.lambda_value == 1.0):
            print(
                f"  {result.sample.sample_id} length={result.sample.length}: "
                f"a0={result.adjoint_norms[0]:.9g} "
                f"a1={result.adjoint_norms[-1]:.9g} "
                f"min={min(result.adjoint_norms):.9g} "
                f"max={max(result.adjoint_norms):.9g}",
                flush=True,
            )

        print("\nClipping sensitivity (pooled sample/lambda timesteps):", flush=True)
        print("c | overall clipped | early clipped | late-10 clipped", flush=True)
        for constant in CLIPPING_CONSTANTS:
            overall_fraction = pooled_fraction(results, constant, range(K))
            early_fraction = pooled_fraction(results, constant, EARLY_INDICES)
            late_fraction = pooled_fraction(results, constant, LATE_INDICES)
            print(
                f"{constant:.1f} | {percentage(overall_fraction)} | "
                f"{percentage(early_fraction)} | {percentage(late_fraction)}",
                flush=True,
            )

        paper_overall = pooled_fraction(results, PAPER_CLIPPING_CONSTANT, range(K))
        paper_early = pooled_fraction(results, PAPER_CLIPPING_CONSTANT, EARLY_INDICES)
        paper_late = pooled_fraction(results, PAPER_CLIPPING_CONSTANT, LATE_INDICES)
        paper_behavior = clipping_behavior(paper_overall, paper_early, paper_late)
        stable_lambdas = [
            lambda_value
            for lambda_value in LAMBDA_GRID
            if all(
                result.finite
                for result in results
                if result.lambda_value == lambda_value
            )
        ]
        stable_range = (
            f"{min(stable_lambdas)}-{max(stable_lambdas)} on diagnostic grid"
            if stable_lambdas
            else "none"
        )
        max_allocated = max(result.peak_allocated_mib for result in results)
        max_reserved = max(result.peak_reserved_mib for result in results)
        total_seconds = time.perf_counter() - total_start
        print(f"\nstable lambda range: {stable_range}", flush=True)
        print(
            f"paper timestep subset numerically viable: "
            f"{'YES' if subset_viable else 'NO'}",
            flush=True,
        )
        print(
            f"paper c=1.6 clipping behavior: {paper_behavior} "
            f"(overall={percentage(paper_overall)}, early={percentage(paper_early)}, "
            f"late-10={percentage(paper_late)})",
            flush=True,
        )
        print(f"peak CUDA allocated MiB: {max_allocated:.2f}", flush=True)
        print(f"peak CUDA reserved MiB: {max_reserved:.2f}", flush=True)
        print(f"total runtime seconds: {total_seconds:.3f}", flush=True)
        print(f"frozen model gradient routing: {'PASS' if frozen_ok else 'FAIL'}", flush=True)
        print("production lambda: NOT YET SELECTED", flush=True)
        print("production clipping constant: NOT YET SELECTED", flush=True)

        overall = bool(
            initialization_ok
            and all_finite
            and subset_viable
            and frozen_ok
            and len(stable_lambdas) == len(LAMBDA_GRID)
        )
        marker = (
            "ADJOINT_TRAINING_SCALE_AUDIT_PASS"
            if overall
            else "ADJOINT_TRAINING_SCALE_AUDIT_FAIL"
        )
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if overall else 1
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        sample_text = (
            "none"
            if current_sample is None
            else f"{current_sample.sample_id} length={current_sample.length}"
        )
        print(f"failure point: {stage}", flush=True)
        print(f"failure sample: {sample_text}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("production lambda: NOT YET SELECTED", flush=True)
        print("production clipping constant: NOT YET SELECTED", flush=True)
        print("overall: ADJOINT_TRAINING_SCALE_AUDIT_FAIL", flush=True)
        print("ADJOINT_TRAINING_SCALE_AUDIT_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

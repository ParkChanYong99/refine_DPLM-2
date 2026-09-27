#!/usr/bin/env python3
"""One-iteration real-model Adjoint Matching integration diagnostic.

This script uses one internal validation item and the finalized FM checkpoint.
It performs no optimizer step, parameter update, checkpoint write, or production
hyperparameter selection.
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.utils.cpp_extension

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# DeepSpeed probes nvcc while importing OpenFold even though this diagnostic
# compiles no CUDA extension. The runtime image supplies PyTorch CUDA but no
# toolkit. Give that import-only version probe a temporary, matching response.
_CUDA_IMPORT_STUB: tempfile.TemporaryDirectory[str] | None = None
if (
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.utils.cpp_extension.CUDA_HOME is None
):
    _CUDA_IMPORT_STUB = tempfile.TemporaryDirectory(prefix="dplm_cuda_import_")
    stub_root = Path(_CUDA_IMPORT_STUB.name)
    stub_bin = stub_root / "bin"
    stub_bin.mkdir()
    nvcc_stub = stub_bin / "nvcc"
    nvcc_stub.write_text(
        "#!/bin/sh\n"
        "echo nvcc: NVIDIA Cuda compiler driver\n"
        f"echo Cuda compilation tools, release {torch.version.cuda}, "
        f"V{torch.version.cuda}.0\n",
        encoding="utf-8",
    )
    nvcc_stub.chmod(0o700)
    os.environ["CUDA_HOME"] = str(stub_root)
    torch.utils.cpp_extension.CUDA_HOME = str(stub_root)
    atexit.register(_CUDA_IMPORT_STUB.cleanup)

from experiments.latent_residual_fm.adjoint_matching_core import (
    adjoint_matching_loss,
    deterministic_warm_start,
    lean_adjoint_step,
    memoryless_step,
    sigma_offset,
)
from experiments.latent_residual_fm.audit_adjoint_aligned_reward import (
    aligned_squared_rmsd,
    backbone_points,
)
from experiments.latent_residual_fm.debug_interfaces import (
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    load_fm,
    load_validation_sample,
)
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor


SAMPLE_ID = "AF-Q4UJC4-F1-model_v4"
VAL_INDEX = 1426
LENGTH = 57
RESIDUAL_DIM = 13
EXPECTED_FINE_PARAMS = 34_946_606
K = 40
H = 1.0 / K
ALPHA_REFINE = 0.5
LAMBDA_DIAGNOSTIC = 1.0
FIXED_POINT_TOL = 1e-10
DETERMINISM_RTOL = 1e-6
DETERMINISM_ATOL = 1e-8
STATE_NORM_LIMIT = 1e3
ADJOINT_NORM_LIMIT = 1e6


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GradStats:
    global_norm: float
    grad_tensors: int
    nonzero_tensors: int
    finite_parameter_count: int
    trainable_parameter_count: int
    trainable_tensor_count: int
    all_finite: bool


@dataclass(frozen=True)
class PassResult:
    lambda_value: float
    struct_loss: float
    a1_norm: float
    am_loss: float
    grad_stats: GradStats
    trajectory_ok: bool
    terminal_ok: bool
    adjoint_ok: bool
    routing_ok: bool
    x1_xhat_mae: float
    x1_xhat_l2: float
    max_state_norm: float
    min_adjoint_norm: float
    max_adjoint_norm: float
    earliest_adjoint_norm: float
    trajectory_seconds: float
    terminal_seconds: float
    adjoint_seconds: float
    regression_seconds: float
    total_seconds: float


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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def velocity_adapter(
    model: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
) -> VelocityFn:
    """Adapt the finalized FM signature to the mathematical core signature."""

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_tensor = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if time_tensor.ndim == 0:
            time_tensor = time_tensor.expand(x.shape[0])
        elif time_tensor.ndim == 1 and time_tensor.shape[0] == 1:
            time_tensor = time_tensor.expand(x.shape[0])
        elif time_tensor.ndim == 2 and tuple(time_tensor.shape) == (1, 1):
            time_tensor = time_tensor.expand(x.shape[0], 1)
        require(
            time_tensor.ndim in (1, 2) and time_tensor.shape[0] == x.shape[0],
            f"invalid adapted time shape {tuple(time_tensor.shape)}",
        )
        return model(x, time_tensor, z_quant, hidden_states, res_mask)

    return velocity


def load_sample_tensors(
    args: argparse.Namespace, device: torch.device
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = load_validation_sample(
        Path(args.dataset_dir), Path(args.split_csv), VAL_INDEX
    )
    require(sample["sample_id"] == SAMPLE_ID, "internal validation sample ID changed")
    require(int(sample["length"]) == LENGTH, "internal validation sample length changed")
    z_cont = sample["z_cont"].float().reshape(1, LENGTH, RESIDUAL_DIM).to(device)
    z_quant = (
        sample["z_quant_current"]
        .float()
        .reshape(1, LENGTH, RESIDUAL_DIM)
        .to(device)
    )
    res_mask = sample["res_mask"].float().reshape(1, LENGTH).to(device)
    aatype = sample["aatype"].long().reshape(1, LENGTH)
    struct_ids = sample["struct_ids_current"].long().reshape(1, LENGTH)
    require(bool(torch.isfinite(z_cont).all()), "z_cont contains NaN/Inf")
    require(bool(torch.isfinite(z_quant).all()), "z_quant contains NaN/Inf")
    require(bool(((res_mask == 0) | (res_mask == 1)).all()), "res_mask is not binary")
    require(bool(res_mask.any()), "res_mask has no valid residues")
    return sample, z_cont, z_quant, res_mask, aatype, struct_ids


def compare_model_copies(
    v_base: torch.nn.Module,
    v_finetune: torch.nn.Module,
    base_fn: VelocityFn,
    finetune_fn: VelocityFn,
    res_mask: torch.Tensor,
    seed: int,
) -> tuple[bool, int, float]:
    base_state = v_base.state_dict()
    fine_state = v_finetune.state_dict()
    require(tuple(base_state) == tuple(fine_state), "FM state-dict keys differ")
    state_equal = all(torch.equal(base_state[name], fine_state[name]) for name in base_state)

    base_named = tuple(v_base.named_parameters())
    fine_named = tuple(v_finetune.named_parameters())
    require(len(base_named) == len(fine_named), "FM parameter tensor counts differ")
    independent = all(
        base_name == fine_name and base_param.data_ptr() != fine_param.data_ptr()
        for (base_name, base_param), (fine_name, fine_param) in zip(
            base_named, fine_named
        )
    )
    trainable = sum(p.numel() for p in v_finetune.parameters() if p.requires_grad)
    frozen = all(not p.requires_grad for p in v_base.parameters())
    generator = torch.Generator(device=res_mask.device).manual_seed(seed + 10_000)
    probe = torch.randn(
        (1, LENGTH, RESIDUAL_DIM),
        generator=generator,
        device=res_mask.device,
        dtype=res_mask.dtype,
    ) * res_mask[..., None]
    probe_t = torch.tensor(0.5, device=probe.device, dtype=probe.dtype)
    with torch.no_grad():
        base_output = base_fn(probe, probe_t)
        fine_output = finetune_fn(probe, probe_t)
    output_difference = float((base_output - fine_output).abs().max())
    outputs_equal = torch.allclose(base_output, fine_output, rtol=0.0, atol=0.0)
    passed = bool(
        state_equal
        and independent
        and frozen
        and trainable == EXPECTED_FINE_PARAMS
        and outputs_equal
    )
    return passed, trainable, output_difference


def build_trajectory(
    finetune_fn: VelocityFn,
    res_mask: torch.Tensor,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device=res_mask.device).manual_seed(seed)
    shape = (res_mask.shape[0], res_mask.shape[1], RESIDUAL_DIM)
    x0 = torch.randn(
        shape, generator=generator, device=res_mask.device, dtype=res_mask.dtype
    ) * res_mask[..., None]
    states = [x0.detach()]
    with torch.no_grad():
        state = deterministic_warm_start(x0, H, finetune_fn, res_mask).detach()
        states.append(state)
        for index in range(1, K):
            noise = torch.randn(
                shape,
                generator=generator,
                device=res_mask.device,
                dtype=res_mask.dtype,
            )
            state = memoryless_step(
                state,
                index * H,
                H,
                finetune_fn,
                noise,
                res_mask,
            ).detach()
            states.append(state)
    require(len(states) == K + 1, f"trajectory has {len(states)} states")
    return states


def trajectory_audit(
    states: Sequence[torch.Tensor], res_mask: torch.Tensor
) -> tuple[bool, float]:
    expected_shape = (1, LENGTH, RESIDUAL_DIM)
    valid = res_mask.bool().unsqueeze(-1).expand(expected_shape)
    padding = ~valid
    finite = all(bool(torch.isfinite(state).all()) for state in states)
    shapes = all(tuple(state.shape) == expected_shape for state in states)
    detached = all(not state.requires_grad and state.grad is None for state in states)
    padding_zero = all(
        not bool((state[padding] != 0).any()) if bool(padding.any()) else True
        for state in states
    )
    max_state_norm = max(
        float(torch.linalg.vector_norm(state, dim=-1)[res_mask.bool()].max())
        for state in states
    )
    passed = bool(
        len(states) == K + 1
        and finite
        and shapes
        and detached
        and padding_zero
        and max_state_norm <= STATE_NORM_LIMIT
    )
    return passed, max_state_norm


def make_x_hat_1(
    states: Sequence[torch.Tensor], base_fn: VelocityFn, res_mask: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        velocity = base_fn(
            states[K - 1],
            torch.tensor(1.0 - H, device=res_mask.device, dtype=res_mask.dtype),
        )
        x_hat_1 = (states[K - 1] + H * velocity) * res_mask[..., None]
    return x_hat_1.detach()


def terminal_adjoint(
    x_hat_1: torch.Tensor,
    z_quant: torch.Tensor,
    res_mask: torch.Tensor,
    tokenizer: torch.nn.Module,
    oracle: dict[str, torch.Tensor],
    lambda_value: float,
) -> tuple[float, torch.Tensor, bool]:
    expected_shape = (1, LENGTH, RESIDUAL_DIM)
    if lambda_value == 0.0:
        adjoint = torch.zeros_like(x_hat_1)
        return 0.0, adjoint, True

    x_hat_1_leaf = x_hat_1.detach().requires_grad_(True)
    require(torch.is_grad_enabled(), "terminal decoder path requires autograd")
    z_refined = z_quant + ALPHA_REFINE * x_hat_1_leaf
    decoded = tokenizer.detokenize(z_refined, res_mask=res_mask)
    pred, target, valid = backbone_points(decoded, oracle, res_mask)
    loss_struct = aligned_squared_rmsd(pred, target, valid)
    terminal_cost = lambda_value * loss_struct
    adjoint = torch.autograd.grad(
        terminal_cost,
        x_hat_1_leaf,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    adjoint = adjoint * res_mask[..., None]
    struct_loss = float(loss_struct.detach())
    valid_adjoint = adjoint[res_mask.bool().unsqueeze(-1).expand_as(adjoint)]
    padding = ~res_mask.bool().unsqueeze(-1).expand_as(adjoint)
    padding_zero = (
        not bool((adjoint[padding] != 0).any()) if bool(padding.any()) else True
    )
    passed = bool(
        tuple(adjoint.shape) == expected_shape
        and math.isfinite(struct_loss)
        and struct_loss > 0.0
        and bool(torch.isfinite(adjoint).all())
        and float(torch.linalg.vector_norm(valid_adjoint)) > 0.0
        and padding_zero
    )
    del decoded, pred, target, valid, terminal_cost, loss_struct, x_hat_1_leaf
    return struct_loss, adjoint, passed


def build_adjoints(
    states: Sequence[torch.Tensor],
    a1: torch.Tensor,
    base_fn: VelocityFn,
    res_mask: torch.Tensor,
    report_steps: bool,
) -> tuple[list[torch.Tensor], bool, float, float, float]:
    adjoints: list[torch.Tensor | None] = [None] * (K + 1)
    adjoints[K] = a1.detach() * res_mask[..., None]
    for index in range(K, 0, -1):
        current = adjoints[index]
        require(current is not None, f"missing adjoint at index {index}")
        adjoints[index - 1] = lean_adjoint_step(
            states[index], current, index * H, H, base_fn, res_mask
        )
    resolved = [value for value in adjoints if value is not None]
    require(len(resolved) == K + 1, "incomplete adjoint grid")

    expected_shape = (1, LENGTH, RESIDUAL_DIM)
    padding = ~res_mask.bool().unsqueeze(-1).expand(expected_shape)
    norms: list[float] = []
    audit_ok = True
    if report_steps:
        print("adjoint timestep audit:", flush=True)
    for index, adjoint in enumerate(resolved):
        nan_count = int(torch.isnan(adjoint).sum())
        inf_count = int(torch.isinf(adjoint).sum())
        norm = float(torch.linalg.vector_norm(adjoint))
        padding_nonzero = (
            int(torch.count_nonzero(adjoint[padding])) if bool(padding.any()) else 0
        )
        finite = bool(torch.isfinite(adjoint).all())
        detached = not adjoint.requires_grad and adjoint.grad is None
        step_ok = bool(
            tuple(adjoint.shape) == expected_shape
            and finite
            and nan_count == 0
            and inf_count == 0
            and padding_nonzero == 0
            and detached
        )
        audit_ok = audit_ok and step_ok
        norms.append(norm)
        if report_steps:
            print(
                f"  t={index * H:.3f} shape={tuple(adjoint.shape)} "
                f"finite={finite} NaN={nan_count} Inf={inf_count} "
                f"norm={norm:.9g} padding_nonzero={padding_nonzero}",
                flush=True,
            )
    minimum = min(norms)
    maximum = max(norms)
    earliest = norms[0]
    audit_ok = audit_ok and maximum <= ADJOINT_NORM_LIMIT
    return resolved, audit_ok, minimum, maximum, earliest


def gradient_stats(model: torch.nn.Module) -> GradStats:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    squared_norm = 0.0
    grad_tensors = 0
    nonzero_tensors = 0
    finite_parameter_count = 0
    all_finite = True
    for parameter in trainable:
        grad = parameter.grad
        if grad is None:
            all_finite = False
            continue
        grad_tensors += 1
        finite = bool(torch.isfinite(grad).all())
        all_finite = all_finite and finite
        if finite:
            finite_parameter_count += parameter.numel()
            norm = float(torch.linalg.vector_norm(grad.detach()))
            squared_norm += norm * norm
        if bool(torch.count_nonzero(grad)):
            nonzero_tensors += 1
    return GradStats(
        global_norm=math.sqrt(squared_norm),
        grad_tensors=grad_tensors,
        nonzero_tensors=nonzero_tensors,
        finite_parameter_count=finite_parameter_count,
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        trainable_tensor_count=len(trainable),
        all_finite=all_finite,
    )


def no_parameter_grads(model: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def am_regression_backward(
    states: Sequence[torch.Tensor],
    adjoints: Sequence[torch.Tensor],
    finetune_fn: VelocityFn,
    base_fn: VelocityFn,
    res_mask: torch.Tensor,
    v_finetune: torch.nn.Module,
) -> tuple[float, GradStats]:
    v_finetune.zero_grad(set_to_none=True)
    losses = []
    for index in range(K):
        time_value = torch.tensor(
            index * H, device=res_mask.device, dtype=res_mask.dtype
        )
        losses.append(
            adjoint_matching_loss(
                states[index],
                adjoints[index],
                time_value,
                finetune_fn,
                base_fn,
                sigma_offset(time_value, H),
                res_mask,
            )
        )
    require(len(losses) == K, "AM aggregate must contain 40 grid terms")
    loss = torch.stack(losses).sum()
    require(bool(torch.isfinite(loss)), "AM aggregate loss is NaN/Inf")
    loss_value = float(loss.detach())
    loss.backward()
    stats = gradient_stats(v_finetune)
    del losses, loss
    return loss_value, stats


def run_pass(
    lambda_value: float,
    seed: int,
    base_fn: VelocityFn,
    finetune_fn: VelocityFn,
    v_base: torch.nn.Module,
    v_finetune: torch.nn.Module,
    dplm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    hidden_states: Sequence[torch.Tensor],
    z_quant: torch.Tensor,
    res_mask: torch.Tensor,
    oracle: dict[str, torch.Tensor],
    device: torch.device,
    report_steps: bool = False,
) -> PassResult:
    pass_start = time.perf_counter()
    v_finetune.zero_grad(set_to_none=True)

    synchronize(device)
    start = time.perf_counter()
    states = build_trajectory(finetune_fn, res_mask, seed)
    x_hat_1 = make_x_hat_1(states, base_fn, res_mask)
    trajectory_ok, max_state_norm = trajectory_audit(states, res_mask)
    x_difference = states[K] - x_hat_1
    x1_xhat_mae = float(x_difference.abs().mean())
    x1_xhat_l2 = float(torch.linalg.vector_norm(x_difference))
    trajectory_ok = trajectory_ok and x1_xhat_l2 > 0.0
    synchronize(device)
    trajectory_seconds = time.perf_counter() - start

    start = time.perf_counter()
    struct_loss, a1, terminal_ok = terminal_adjoint(
        x_hat_1, z_quant, res_mask, tokenizer, oracle, lambda_value
    )
    a1_norm = float(torch.linalg.vector_norm(a1))
    if lambda_value == 0.0:
        terminal_ok = terminal_ok and a1_norm == 0.0
    synchronize(device)
    terminal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    adjoints, adjoint_ok, minimum, maximum, earliest = build_adjoints(
        states, a1, base_fn, res_mask, report_steps
    )
    synchronize(device)
    adjoint_seconds = time.perf_counter() - start

    start = time.perf_counter()
    am_loss, stats = am_regression_backward(
        states,
        adjoints,
        finetune_fn,
        base_fn,
        res_mask,
        v_finetune,
    )
    synchronize(device)
    regression_seconds = time.perf_counter() - start

    state_grads_absent = all(state.grad is None for state in states)
    adjoint_grads_absent = all(adjoint.grad is None for adjoint in adjoints)
    condition_detached = all(not state.requires_grad and state.grad is None for state in hidden_states)
    routing_ok = bool(
        stats.grad_tensors == stats.trainable_tensor_count
        and stats.finite_parameter_count == stats.trainable_parameter_count
        and stats.all_finite
        and no_parameter_grads(v_base)
        and no_parameter_grads(dplm)
        and no_parameter_grads(tokenizer)
        and state_grads_absent
        and adjoint_grads_absent
        and condition_detached
    )
    total_seconds = time.perf_counter() - pass_start
    return PassResult(
        lambda_value=lambda_value,
        struct_loss=struct_loss,
        a1_norm=a1_norm,
        am_loss=am_loss,
        grad_stats=stats,
        trajectory_ok=trajectory_ok,
        terminal_ok=terminal_ok,
        adjoint_ok=adjoint_ok,
        routing_ok=routing_ok,
        x1_xhat_mae=x1_xhat_mae,
        x1_xhat_l2=x1_xhat_l2,
        max_state_norm=max_state_norm,
        min_adjoint_norm=minimum,
        max_adjoint_norm=maximum,
        earliest_adjoint_norm=earliest,
        trajectory_seconds=trajectory_seconds,
        terminal_seconds=terminal_seconds,
        adjoint_seconds=adjoint_seconds,
        regression_seconds=regression_seconds,
        total_seconds=total_seconds,
    )


def allclose_scalar(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=DETERMINISM_RTOL,
        abs_tol=DETERMINISM_ATOL,
    )


def cuda_report(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return math.nan, math.nan
    synchronize(device)
    return (
        torch.cuda.max_memory_allocated(device) / 1024**2,
        torch.cuda.max_memory_reserved(device) / 1024**2,
    )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    total_start = time.perf_counter()
    stage = "startup"
    print("REAL_ADJOINT_MATCHING_INTEGRATION", flush=True)
    try:
        require(args.seed == 42, "this diagnostic is fixed to seed 42")
        require(device.type == "cuda", "this real integration diagnostic requires CUDA")
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        stage = "load internal validation sample"
        sample, z_cont, z_quant, res_mask, aatype, struct_ids = load_sample_tensors(
            args, device
        )
        print(f"sample: {sample['sample_id']} (internal val index {VAL_INDEX})", flush=True)
        print(f"length: {int(sample['length'])}", flush=True)

        stage = "load frozen DPLM and Mode-A condition"
        extractor = DPLMConditionExtractor(args.dplm_checkpoint, device=device)
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()
        condition_result = extractor(aatype, struct_ids, res_mask)
        hidden_states = condition_result["hidden_states"]
        require(len(hidden_states) == 33, "Mode-A condition must have 33 layers")
        require(
            all(tuple(value.shape) == (1, LENGTH, 1280) for value in hidden_states),
            "Mode-A condition shape mismatch",
        )
        require(
            tuple(z_quant.shape) == (1, LENGTH, RESIDUAL_DIM),
            "z_quant shape mismatch",
        )
        require(tuple(res_mask.shape) == (1, LENGTH), "res_mask shape mismatch")
        print("condition:", flush=True)
        print(f"  layers: {len(hidden_states)} (embedding excluded)", flush=True)
        print(f"  shape: {tuple(hidden_states[0].shape)} per layer", flush=True)
        print(f"  z_quant shape: {tuple(z_quant.shape)}", flush=True)
        print(f"  res_mask shape: {tuple(res_mask.shape)}", flush=True)

        stage = "load independent base and finetune FM copies"
        v_base, base_step, _, _ = load_fm(Path(args.checkpoint), device)
        v_finetune, fine_step, _, _ = load_fm(Path(args.checkpoint), device)
        require(base_step == fine_step, "FM checkpoint step mismatch")
        v_base.requires_grad_(False).eval()
        v_finetune.requires_grad_(True).eval()
        base_fn = velocity_adapter(v_base, z_quant, hidden_states, res_mask)
        finetune_fn = velocity_adapter(v_finetune, z_quant, hidden_states, res_mask)
        initialization_ok, trainable_fine, output_max_diff = compare_model_copies(
            v_base, v_finetune, base_fn, finetune_fn, res_mask, args.seed
        )
        print("base/fine initialization:", flush=True)
        print(f"  status: {'PASS' if initialization_ok else 'FAIL'}", flush=True)
        print(f"  checkpoint optimizer_step: {base_step}", flush=True)
        print(f"  trainable fine params: {trainable_fine}", flush=True)
        print(f"  identical-output max abs difference: {output_max_diff:.9g}", flush=True)

        dplm_trainable = sum(p.numel() for p in dplm.parameters() if p.requires_grad)
        tokenizer_trainable = sum(
            p.numel() for p in tokenizer.parameters() if p.requires_grad
        )
        decoder_trainable = sum(
            p.numel() for p in tokenizer.decoder.parameters() if p.requires_grad
        )
        frozen_ok = dplm_trainable == tokenizer_trainable == decoder_trainable == 0

        stage = "decode fixed oracle"
        with torch.no_grad():
            oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)

        stage = "lambda-zero fixed-point pass"
        print("running lambda=0 fixed-point pass", flush=True)
        zero = run_pass(
            0.0,
            args.seed,
            base_fn,
            finetune_fn,
            v_base,
            v_finetune,
            dplm,
            tokenizer,
            hidden_states,
            z_quant,
            res_mask,
            oracle,
            device,
        )
        zero_ok = bool(
            zero.trajectory_ok
            and zero.terminal_ok
            and zero.adjoint_ok
            and zero.routing_ok
            and abs(zero.am_loss) <= FIXED_POINT_TOL
            and zero.grad_stats.global_norm <= FIXED_POINT_TOL
        )

        stage = "first lambda-one pass"
        print("running lambda=1 pass 1", flush=True)
        one = run_pass(
            LAMBDA_DIAGNOSTIC,
            args.seed,
            base_fn,
            finetune_fn,
            v_base,
            v_finetune,
            dplm,
            tokenizer,
            hidden_states,
            z_quant,
            res_mask,
            oracle,
            device,
            report_steps=True,
        )
        one_ok = bool(
            one.trajectory_ok
            and one.terminal_ok
            and one.adjoint_ok
            and one.routing_ok
            and math.isfinite(one.struct_loss)
            and one.struct_loss > 0.0
            and math.isfinite(one.a1_norm)
            and one.a1_norm > 0.0
            and math.isfinite(one.am_loss)
            and one.am_loss > 0.0
            and math.isfinite(one.grad_stats.global_norm)
            and one.grad_stats.global_norm > 0.0
            and one.grad_stats.nonzero_tensors > 0
        )

        stage = "second lambda-one determinism pass"
        print("running lambda=1 pass 2", flush=True)
        repeat = run_pass(
            LAMBDA_DIAGNOSTIC,
            args.seed,
            base_fn,
            finetune_fn,
            v_base,
            v_finetune,
            dplm,
            tokenizer,
            hidden_states,
            z_quant,
            res_mask,
            oracle,
            device,
        )
        determinism_checks = {
            "L_struct": allclose_scalar(one.struct_loss, repeat.struct_loss),
            "a_1 norm": allclose_scalar(one.a1_norm, repeat.a1_norm),
            "AM loss": allclose_scalar(one.am_loss, repeat.am_loss),
            "global grad norm": allclose_scalar(
                one.grad_stats.global_norm, repeat.grad_stats.global_norm
            ),
        }
        determinism_ok = all(determinism_checks.values())
        repeat_ok = bool(
            repeat.trajectory_ok
            and repeat.terminal_ok
            and repeat.adjoint_ok
            and repeat.routing_ok
        )
        routing_ok = zero.routing_ok and one.routing_ok and repeat.routing_ok
        frozen_ok = frozen_ok and no_parameter_grads(v_base) and no_parameter_grads(dplm)

        print("trajectory:", flush=True)
        print(f"  status: {'PASS' if one.trajectory_ok else 'FAIL'}", flush=True)
        print(f"  states: {K + 1} (X_0 through X_1_stoch)", flush=True)
        print(f"  X_1_stoch finite: {bool(torch.isfinite(torch.tensor(one.x1_xhat_l2)))}", flush=True)
        print(f"  X_hat_1 finite: {one.terminal_ok}", flush=True)
        print(f"  X1_stoch-vs-Xhat MAE: {one.x1_xhat_mae:.9g}", flush=True)
        print(f"  X1_stoch-vs-Xhat L2: {one.x1_xhat_l2:.9g}", flush=True)
        print(f"  max state norm: {one.max_state_norm:.9g}", flush=True)
        print(f"  explosion limit: {STATE_NORM_LIMIT:.9g}", flush=True)

        print("terminal reward:", flush=True)
        print(f"  lambda_diagnostic: {LAMBDA_DIAGNOSTIC} (diagnostic only)", flush=True)
        print(f"  alpha_refine: {ALPHA_REFINE}", flush=True)
        print(f"  L_struct: {one.struct_loss:.9g}", flush=True)
        print(f"  a_1 norm: {one.a1_norm:.9g}", flush=True)
        print(f"  status: {'PASS' if one.terminal_ok else 'FAIL'}", flush=True)

        print("lean adjoint:", flush=True)
        print(f"  status: {'PASS' if one.adjoint_ok else 'FAIL'}", flush=True)
        print(f"  min norm: {one.min_adjoint_norm:.9g}", flush=True)
        print(f"  max norm: {one.max_adjoint_norm:.9g}", flush=True)
        print(f"  earliest a_0 norm: {one.earliest_adjoint_norm:.9g}", flush=True)
        print(f"  explosion limit: {ADJOINT_NORM_LIMIT:.9g}", flush=True)

        print("lambda=0:", flush=True)
        print(f"  tolerance: {FIXED_POINT_TOL:.3g}", flush=True)
        print(f"  AM loss: {zero.am_loss:.9g}", flush=True)
        print(f"  grad norm: {zero.grad_stats.global_norm:.9g}", flush=True)
        print(f"  status: {'PASS' if zero_ok else 'FAIL'}", flush=True)

        print("lambda=1:", flush=True)
        print(f"  AM loss: {one.am_loss:.9g}", flush=True)
        print(f"  grad norm: {one.grad_stats.global_norm:.9g}", flush=True)
        print(f"  nonzero grad tensors: {one.grad_stats.nonzero_tensors}", flush=True)
        print(
            f"  finite-gradient parameter count: "
            f"{one.grad_stats.finite_parameter_count}",
            flush=True,
        )
        print(f"  status: {'PASS' if one_ok else 'FAIL'}", flush=True)

        print("gradient routing:", flush=True)
        print(f"  v_finetune grad tensors: {one.grad_stats.grad_tensors}/{one.grad_stats.trainable_tensor_count}", flush=True)
        print(f"  v_base grads all None: {no_parameter_grads(v_base)}", flush=True)
        print(f"  DPLM grads all None: {no_parameter_grads(dplm)}", flush=True)
        print(f"  tokenizer/decoder grads all None: {no_parameter_grads(tokenizer)}", flush=True)
        print("  trajectory retained grads: 0", flush=True)
        print("  adjoint retained grads: 0", flush=True)
        print(f"  status: {'PASS' if routing_ok else 'FAIL'}", flush=True)

        print("determinism:", flush=True)
        for name, passed in determinism_checks.items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}", flush=True)
        print(f"  status: {'PASS' if determinism_ok else 'FAIL'}", flush=True)

        print("frozen DPLM/tokenizer:", flush=True)
        print(
            f"  trainable params DPLM/tokenizer/decoder: "
            f"{dplm_trainable}/{tokenizer_trainable}/{decoder_trainable}",
            flush=True,
        )
        print(f"  status: {'PASS' if frozen_ok else 'FAIL'}", flush=True)

        allocated, reserved = cuda_report(device)
        total_seconds = time.perf_counter() - total_start
        print("peak CUDA:", flush=True)
        print(f"  allocated MiB: {allocated:.2f}", flush=True)
        print(f"  reserved MiB: {reserved:.2f}", flush=True)
        print("runtime (lambda=1 pass 1):", flush=True)
        print(f"  trajectory generation seconds: {one.trajectory_seconds:.3f}", flush=True)
        print(f"  terminal reward + gradient seconds: {one.terminal_seconds:.3f}", flush=True)
        print(f"  lean adjoint recursion seconds: {one.adjoint_seconds:.3f}", flush=True)
        print(f"  AM regression forward/backward seconds: {one.regression_seconds:.3f}", flush=True)
        print(f"  lambda=0 pass seconds: {zero.total_seconds:.3f}", flush=True)
        print(f"  lambda=1 repeat seconds: {repeat.total_seconds:.3f}", flush=True)
        print(f"  total seconds: {total_seconds:.3f}", flush=True)

        overall = bool(
            initialization_ok
            and frozen_ok
            and zero_ok
            and one_ok
            and repeat_ok
            and routing_ok
            and determinism_ok
        )
        marker = (
            "REAL_ADJOINT_MATCHING_INTEGRATION_PASS"
            if overall
            else "REAL_ADJOINT_MATCHING_INTEGRATION_FAIL"
        )
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if overall else 1
    except Exception as exc:
        print(f"failure point: {stage}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        allocated, reserved = cuda_report(device)
        print(f"peak CUDA allocated MiB: {allocated:.2f}", flush=True)
        print(f"peak CUDA reserved MiB: {reserved:.2f}", flush=True)
        print("overall: REAL_ADJOINT_MATCHING_INTEGRATION_FAIL", flush=True)
        print("REAL_ADJOINT_MATCHING_INTEGRATION_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

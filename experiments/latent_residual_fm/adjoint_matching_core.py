"""Reusable mathematical primitives for synthetic Adjoint Matching validation.

This module contains no trainer, optimizer, production hyperparameters, model
loading, or dataset access. Its equations follow the experiment specification:
the local deterministic warm-start, memoryless Flow-Matching drift, lean-adjoint
Euler VJP, and Algorithm 1 regression residual.
"""

from __future__ import annotations

from typing import Callable

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _floating_tensor(
    value: float | torch.Tensor, reference: torch.Tensor, name: str
) -> torch.Tensor:
    tensor = (
        value.to(device=reference.device)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if tensor.dtype != reference.dtype:
        tensor = tensor.to(dtype=reference.dtype)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _broadcast_state_coefficient(
    coefficient: torch.Tensor, state: torch.Tensor, name: str
) -> torch.Tensor:
    if coefficient.ndim == 0:
        return coefficient
    candidates = [coefficient]
    if coefficient.ndim < state.ndim:
        candidates.insert(
            0,
            coefficient.reshape(
                *coefficient.shape, *([1] * (state.ndim - coefficient.ndim))
            ),
        )
    for candidate in candidates:
        try:
            torch.broadcast_shapes(state.shape, candidate.shape)
            return candidate
        except RuntimeError:
            pass
    raise ValueError(
        f"{name} shape {tuple(coefficient.shape)} cannot broadcast to state shape "
        f"{tuple(state.shape)}"
    )


def _mask_for_state(mask: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones((), dtype=state.dtype, device=state.device)
    mask_tensor = mask.to(device=state.device)
    if not bool(torch.isfinite(mask_tensor).all()):
        raise ValueError("mask must be finite")
    if not bool(((mask_tensor == 0) | (mask_tensor == 1)).all()):
        raise ValueError("mask entries must be binary")
    if mask_tensor.shape == state.shape:
        expanded = mask_tensor
    elif state.ndim >= 1 and mask_tensor.shape == state.shape[:-1]:
        expanded = mask_tensor.unsqueeze(-1)
    else:
        raise ValueError(
            f"mask shape {tuple(mask_tensor.shape)} is incompatible with state shape "
            f"{tuple(state.shape)}"
        )
    return expanded.to(dtype=state.dtype)


def _positive_step(
    h: float | torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    step = _floating_tensor(h, reference, "h")
    if step.numel() != 1 or not bool(step > 0):
        raise ValueError("h must be a positive scalar")
    return step.reshape(())


def _call_velocity(
    fn: VelocityFn, x: torch.Tensor, t: torch.Tensor, name: str
) -> torch.Tensor:
    output = fn(x, t)
    if not isinstance(output, torch.Tensor) or output.shape != x.shape:
        shape = None if not isinstance(output, torch.Tensor) else tuple(output.shape)
        raise ValueError(
            f"{name} must return state shape {tuple(x.shape)}, got {shape}"
        )
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"{name} returned non-finite values")
    return output


def sigma_offset(t: torch.Tensor, h: float | torch.Tensor) -> torch.Tensor:
    """Appendix H.1 offset schedule: sqrt(2*(1-t+h)/(t+h)).

    t must be floating-point, finite, and in [0, 1]. The result preserves
    its dtype and device. Unlike the theoretical schedule, the offset schedule
    is finite at t=0 when h>0.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")
    if not t.is_floating_point():
        raise TypeError("t must have a floating-point dtype")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("t must be finite")
    if not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError("t must lie in [0, 1]")
    step = _positive_step(h, t)
    result = torch.sqrt(2 * (1 - t + step) / (t + step))
    if not bool(torch.isfinite(result).all()):
        raise ValueError("sigma_offset produced non-finite values")
    return result


def base_drift(
    x: torch.Tensor, t: float | torch.Tensor, v_base: VelocityFn
) -> torch.Tensor:
    """Return 2*v_base(x,t) - x/t for strictly positive t."""
    time = _floating_tensor(t, x, "t")
    if not bool((time > 0).all()):
        raise ValueError("base_drift requires t > 0")
    denominator = _broadcast_state_coefficient(time, x, "t")
    velocity = _call_velocity(v_base, x, time, "v_base")
    drift = 2 * velocity - x / denominator
    if not bool(torch.isfinite(drift).all()):
        raise ValueError("base_drift produced non-finite values")
    return drift


def deterministic_warm_start(
    x0: torch.Tensor,
    h: float | torch.Tensor,
    v_finetune: VelocityFn,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local 0->h policy: one deterministic FM Euler step, with no noise."""
    step = _positive_step(h, x0)
    state_mask = _mask_for_state(mask, x0)
    state = x0 * state_mask
    zero = torch.zeros((), dtype=x0.dtype, device=x0.device)
    velocity = _call_velocity(v_finetune, state, zero, "v_finetune") * state_mask
    result = (state + step * velocity) * state_mask
    if not bool(torch.isfinite(result).all()):
        raise ValueError("deterministic_warm_start produced non-finite values")
    return result


def memoryless_step(
    x: torch.Tensor,
    t: float | torch.Tensor,
    h: float | torch.Tensor,
    v_finetune: VelocityFn,
    noise: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One memoryless Flow-Matching Euler-Maruyama step for t>0."""
    if noise.shape != x.shape:
        raise ValueError("noise must have the same shape as x")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("noise must be finite")
    time = _floating_tensor(t, x, "t")
    if not bool((time > 0).all()):
        raise ValueError(
            "memoryless_step requires t > 0; use deterministic_warm_start at t=0"
        )
    step = _positive_step(h, x)
    state_mask = _mask_for_state(mask, x)
    state = x * state_mask
    masked_noise = noise * state_mask
    denominator = _broadcast_state_coefficient(time, state, "t")
    velocity = (
        _call_velocity(v_finetune, state, time, "v_finetune") * state_mask
    )
    sigma = _broadcast_state_coefficient(
        sigma_offset(time, step), state, "sigma"
    )
    result = (
        state
        + step * (2 * velocity - state / denominator)
        + torch.sqrt(step) * sigma * masked_noise
    ) * state_mask
    if not bool(torch.isfinite(result).all()):
        raise ValueError("memoryless_step produced non-finite values")
    return result


def control_from_velocities(
    v_finetune_out: torch.Tensor,
    v_base_out: torch.Tensor,
    sigma: float | torch.Tensor,
    *,
    detach_base: bool = True,
) -> torch.Tensor:
    """Map FM velocity difference to control: 2/sigma * (v_ft-v_base)."""
    if v_finetune_out.shape != v_base_out.shape:
        raise ValueError("velocity outputs must have identical shapes")
    coefficient = _floating_tensor(sigma, v_finetune_out, "sigma")
    if not bool((coefficient > 0).all()):
        raise ValueError("sigma must be positive")
    coefficient = _broadcast_state_coefficient(
        coefficient, v_finetune_out, "sigma"
    )
    base_value = v_base_out.detach() if detach_base else v_base_out
    control = (2 / coefficient) * (v_finetune_out - base_value)
    if not bool(torch.isfinite(control).all()):
        raise ValueError("control mapping produced non-finite values")
    return control


def lean_adjoint_step(
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    t: float | torch.Tensor,
    h: float | torch.Tensor,
    v_base_fn: VelocityFn,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq. 216 Euler recursion using a VJP of the frozen base drift.

    A temporary state leaf is created only for J_x[b_base]^T a. The function
    never calls backward and therefore never accumulates parameter gradients.
    The returned adjoint is detached.
    """
    if x_t.shape != a_t.shape:
        raise ValueError("x_t and a_t must have identical shapes")
    step = _positive_step(h, x_t)
    time = _floating_tensor(t, x_t, "t")
    if not bool((time > 0).all()):
        raise ValueError("lean_adjoint_step requires t > 0")
    state_mask = _mask_for_state(mask, x_t)
    state_value = x_t.detach() * state_mask
    adjoint = a_t.detach() * state_mask
    with torch.enable_grad():
        state_leaf = state_value.requires_grad_(True)
        drift = base_drift(state_leaf, time, v_base_fn) * state_mask
        vjp = torch.autograd.grad(
            outputs=drift,
            inputs=state_leaf,
            grad_outputs=adjoint,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
    result = (adjoint + step * vjp) * state_mask
    if not bool(torch.isfinite(result).all()):
        raise ValueError("lean_adjoint_step produced non-finite values")
    return result.detach()


def adjoint_matching_residual(
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    t: float | torch.Tensor,
    v_finetune_fn: VelocityFn,
    v_base_fn: VelocityFn,
    sigma: float | torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-timestep Eq. 217 residual with explicit stop-gradient routing."""
    if x_t.shape != a_t.shape:
        raise ValueError("x_t and a_t must have identical shapes")
    time = _floating_tensor(t, x_t, "t").detach()
    sigma_value = _floating_tensor(sigma, x_t, "sigma").detach()
    state_mask = _mask_for_state(mask, x_t)
    state = x_t.detach() * state_mask
    adjoint = a_t.detach() * state_mask
    finetune_velocity = (
        _call_velocity(v_finetune_fn, state, time, "v_finetune") * state_mask
    )
    with torch.no_grad():
        base_velocity = (
            _call_velocity(v_base_fn, state, time, "v_base") * state_mask
        )
    control = control_from_velocities(
        finetune_velocity, base_velocity, sigma_value, detach_base=True
    )
    sigma_broadcast = _broadcast_state_coefficient(
        sigma_value, state, "sigma"
    )
    residual = (control + sigma_broadcast * adjoint) * state_mask
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("adjoint_matching_residual produced non-finite values")
    return residual


def masked_mean_square(
    values: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean square over all entries, or valid residues and last channel."""
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if mask is None:
        return values.square().mean()
    if values.ndim < 2 or mask.shape != values.shape[:-1]:
        raise ValueError("mask must have shape values.shape[:-1]")
    state_mask = _mask_for_state(mask, values)
    valid = state_mask.sum() * values.shape[-1]
    if not bool(valid.detach() > 0):
        raise ValueError("mask contains no valid positions")
    return (values.square() * state_mask).sum() / valid


def adjoint_matching_loss(
    x_t: torch.Tensor,
    a_t: torch.Tensor,
    t: float | torch.Tensor,
    v_finetune_fn: VelocityFn,
    v_base_fn: VelocityFn,
    sigma: float | torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked one-timestep Algorithm 1 / Eq. 217 regression loss."""
    residual = adjoint_matching_residual(
        x_t, a_t, t, v_finetune_fn, v_base_fn, sigma, mask
    )
    return masked_mean_square(residual, mask)


__all__ = [
    "adjoint_matching_loss",
    "adjoint_matching_residual",
    "base_drift",
    "control_from_velocities",
    "deterministic_warm_start",
    "lean_adjoint_step",
    "masked_mean_square",
    "memoryless_step",
    "sigma_offset",
]

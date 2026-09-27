"""Gaussian-source conditional Flow Matching utilities."""

from __future__ import annotations

from typing import Sequence

import torch


def _validate_residual_mask(residual: torch.Tensor, res_mask: torch.Tensor) -> None:
    if residual.ndim != 3 or residual.shape[-1] != 13:
        raise ValueError(f"residual must have shape [B,L,13], got {tuple(residual.shape)}")
    if res_mask.shape != residual.shape[:2]:
        raise ValueError("res_mask must have shape [B,L]")
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("residual contains non-finite values")


def sample_flow_matching_batch(
    residual: torch.Tensor,
    res_mask: torch.Tensor,
    *,
    r0: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the Gaussian-source linear FM path and its target velocity."""
    _validate_residual_mask(residual, res_mask)
    batch_size = residual.shape[0]
    if r0 is None:
        r0 = torch.randn(
            residual.shape,
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        )
    if r0.shape != residual.shape:
        raise ValueError("r0 must have the same shape as residual")
    if t is None:
        t = torch.rand(
            batch_size,
            dtype=residual.dtype,
            device=residual.device,
            generator=generator,
        )
    if t.ndim == 2 and t.shape == (batch_size, 1):
        t = t[:, 0]
    if t.shape != (batch_size,):
        raise ValueError("t must have shape [B] or [B,1]")
    if not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError("t must lie in [0,1]")

    mask = res_mask.to(dtype=residual.dtype)[..., None]
    r1 = residual * mask
    r0 = r0 * mask
    t_broadcast = t[:, None, None]
    r_t = ((1.0 - t_broadcast) * r0 + t_broadcast * r1) * mask
    u_t = (r1 - r0) * mask
    return {"r0": r0, "r1": r1, "r_t": r_t, "u_t": u_t, "t": t}


def masked_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    res_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact masked mean squared velocity loss over residues and 13 bits."""
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target velocity shapes must match")
    _validate_residual_mask(target_velocity, res_mask)
    denominator = res_mask.to(dtype=predicted_velocity.dtype).sum() * 13
    if not bool(denominator.detach() > 0):
        raise ValueError("masked FM loss denominator is zero")
    squared_error = (predicted_velocity - target_velocity).square()
    return (squared_error * res_mask[..., None]).sum() / denominator


def euler_sample(
    model: torch.nn.Module,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    num_steps: int,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit Euler integration of the learned residual velocity from 0 to 1."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    _validate_residual_mask(z_quant, res_mask)
    if initial_noise is None:
        residual = torch.randn_like(z_quant)
    else:
        if initial_noise.shape != z_quant.shape:
            raise ValueError("initial_noise must have the same shape as z_quant")
        residual = initial_noise.clone()
    mask = res_mask.to(dtype=z_quant.dtype)[..., None]
    residual = residual * mask
    dt = 1.0 / num_steps
    batch_size = z_quant.shape[0]
    for step in range(num_steps):
        t = torch.full(
            (batch_size,),
            step / num_steps,
            dtype=z_quant.dtype,
            device=z_quant.device,
        )
        velocity = model(residual, t, z_quant, hidden_states, res_mask)
        residual = (residual + dt * velocity) * mask
    return residual


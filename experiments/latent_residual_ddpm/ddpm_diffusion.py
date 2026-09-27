"""Mathematical core for the pre-registered matched residual DDPM baseline.

This module intentionally contains no residual network, reverse sampler,
training loop, dataset access, or DPLM integration.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


RESIDUAL_DIM = 13
DEFAULT_NUM_TIMESTEPS = 1_000
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 2e-2

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_latent(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 3 or tensor.shape[-1] != RESIDUAL_DIM:
        raise ValueError(
            f"{name} must have shape [B,L,{RESIDUAL_DIM}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and residue dimensions")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    _require_finite(tensor, name)


def _mask_like(
    res_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if res_mask.ndim != 2 or tuple(res_mask.shape) != tuple(reference.shape[:2]):
        raise ValueError(
            f"res_mask must have shape {tuple(reference.shape[:2])}, "
            f"got {tuple(res_mask.shape)}"
        )
    if res_mask.is_floating_point():
        _require_finite(res_mask, "res_mask")
    mask = res_mask.to(device=reference.device, dtype=reference.dtype)
    if not bool((mask >= 0).all()):
        raise ValueError("res_mask must be non-negative")
    return mask


class DDPMSchedule(nn.Module):
    """Linear DDPM schedule stored entirely as non-trainable buffers.

    Public timesteps use the mathematical convention t in {1, ..., T}.
    Schedule tensors use normal zero-based Python storage, so mathematical
    timestep t selects tensor index t - 1.

    Coefficients are constructed and validated in float64. Standard
    nn.Module.to(device=..., dtype=...) can subsequently move/cast all
    registered buffers for model use without introducing trainable parameters.
    """

    def __init__(
        self,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
    ) -> None:
        super().__init__()
        if not isinstance(num_timesteps, int) or isinstance(num_timesteps, bool):
            raise TypeError("num_timesteps must be an integer")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError("beta endpoints must satisfy 0 < beta_start < beta_end < 1")

        self.num_timesteps = num_timesteps
        betas = torch.linspace(
            beta_start,
            beta_end,
            steps=num_timesteps,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            (torch.ones(1, dtype=torch.float64), alpha_bars[:-1]),
            dim=0,
        )
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        coefficients = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "alpha_bars_prev": alpha_bars_prev,
            "posterior_variance": posterior_variance,
            "sqrt_alpha_bars": sqrt_alpha_bars,
            "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        }
        for name, coefficient in coefficients.items():
            if coefficient.shape != (num_timesteps,):
                raise AssertionError(f"{name} has shape {tuple(coefficient.shape)}")
            _require_finite(coefficient, name)
        if not bool(((betas > 0.0) & (betas < 1.0)).all()):
            raise AssertionError("all beta values must lie strictly between zero and one")
        if not bool(((alphas > 0.0) & (alphas < 1.0)).all()):
            raise AssertionError("all alpha values must lie strictly between zero and one")
        if num_timesteps > 1 and not bool((alpha_bars[1:] < alpha_bars[:-1]).all()):
            raise AssertionError("alpha_bars must be strictly decreasing")
        if not (float(alpha_bars[0]) < 1.0 and float(alpha_bars[-1]) > 0.0):
            raise AssertionError("alpha_bar endpoints are invalid")
        if float(posterior_variance[0]) != 0.0:
            raise AssertionError("posterior variance at mathematical t=1 must be zero")
        if not bool((posterior_variance >= 0.0).all()):
            raise AssertionError("posterior variance must be non-negative")

        for name, coefficient in coefficients.items():
            self.register_buffer(name, coefficient, persistent=True)

    def mathematical_timesteps_to_indices(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Validate mathematical timesteps and return zero-based indices."""
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}")
        if timesteps.numel() == 0:
            raise ValueError("timesteps must be non-empty")
        if timesteps.dtype not in _INTEGER_DTYPES:
            raise TypeError("timesteps must use an integer dtype")
        if not bool(((timesteps >= 1) & (timesteps <= self.num_timesteps)).all()):
            raise ValueError(
                f"mathematical timesteps must lie in [1,{self.num_timesteps}]"
            )
        return timesteps.to(dtype=torch.long) - 1

    def extract(
        self,
        coefficient: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Gather a schedule coefficient and return broadcast shape [B,1,1]."""
        if coefficient.ndim != 1 or coefficient.shape[0] != self.num_timesteps:
            raise ValueError("coefficient must be a length-T schedule buffer")
        indices = self.mathematical_timesteps_to_indices(timesteps)
        selected = coefficient.index_select(0, indices.to(coefficient.device))
        return selected.to(device=like.device, dtype=like.dtype).view(-1, 1, 1)


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: Optional[torch.Tensor],
    schedule: DDPMSchedule,
    res_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Sample q(x_t | x_0) for mathematical timesteps in [1, T].

    When a residue mask is supplied, x0, sampled noise/epsilon, and x_t are
    all zero at padding positions. The returned epsilon is therefore the exact
    masked epsilon-prediction target.
    """
    if not isinstance(schedule, DDPMSchedule):
        raise TypeError("schedule must be a DDPMSchedule")
    _validate_latent(x0, "x0")
    if not isinstance(timesteps, torch.Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
        raise ValueError(
            f"timesteps must have shape [{x0.shape[0]}], got {tuple(timesteps.shape)}"
        )
    schedule.mathematical_timesteps_to_indices(timesteps)

    if noise is None:
        noise = torch.randn_like(x0)
    else:
        _validate_latent(noise, "noise")
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")
        if noise.device != x0.device or noise.dtype != x0.dtype:
            raise ValueError("noise and x0 must have the same dtype and device")

    if res_mask is None:
        masked_x0 = x0
        epsilon = noise
        mask_3d = None
    else:
        mask_3d = _mask_like(res_mask, x0)[..., None]
        masked_x0 = x0 * mask_3d
        epsilon = noise * mask_3d

    sqrt_alpha_bar = schedule.extract(
        schedule.sqrt_alpha_bars,
        timesteps,
        like=x0,
    )
    sqrt_one_minus_alpha_bar = schedule.extract(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        like=x0,
    )
    x_t = sqrt_alpha_bar * masked_x0 + sqrt_one_minus_alpha_bar * epsilon
    if mask_3d is not None:
        x_t = x_t * mask_3d
    _require_finite(x_t, "x_t")
    return {
        "x_t": x_t,
        "epsilon": epsilon,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
    }


def masked_epsilon_mse(
    epsilon_pred: torch.Tensor,
    epsilon: torch.Tensor,
    res_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact masked epsilon MSE over residues and all 13 latent dimensions."""
    _validate_latent(epsilon_pred, "epsilon_pred")
    _validate_latent(epsilon, "epsilon")
    if epsilon_pred.shape != epsilon.shape:
        raise ValueError("epsilon_pred and epsilon must have identical shapes")
    if epsilon_pred.device != epsilon.device or epsilon_pred.dtype != epsilon.dtype:
        raise ValueError("epsilon_pred and epsilon must have the same dtype and device")
    mask = _mask_like(res_mask, epsilon_pred)
    denominator = mask.sum() * RESIDUAL_DIM
    if not bool(denominator.detach() > 0):
        raise ValueError("masked epsilon MSE denominator is zero")
    squared_error = (epsilon_pred - epsilon).square()
    return (squared_error * mask[..., None]).sum() / denominator

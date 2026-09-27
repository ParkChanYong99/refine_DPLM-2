"""Full ancestral sampler for the Local Matched Residual DDPM baseline.

Public timesteps follow the mathematical convention ``1 <= t <= T``.  The
primary sampler evaluates every timestep in reverse order and deliberately
contains no DDIM path, timestep skipping, clipping, guidance, or alpha scaling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import (
    DDPMSchedule,
    RESIDUAL_DIM,
)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_latent(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or tensor.shape[-1] != RESIDUAL_DIM:
        raise ValueError(
            f"{name} must have shape [B,L,{RESIDUAL_DIM}], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and residue dimensions")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    _require_finite(tensor, name)


def _mask_like(res_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(res_mask, torch.Tensor):
        raise TypeError("res_mask must be a torch.Tensor")
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


def _validate_conditioning(
    x_t: torch.Tensor,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
) -> None:
    _validate_latent(z_quant, "z_quant")
    if z_quant.shape != x_t.shape:
        raise ValueError("z_quant must have the same shape as x_t")
    if z_quant.device != x_t.device or z_quant.dtype != x_t.dtype:
        raise ValueError("z_quant and x_t must have the same dtype and device")
    if not isinstance(hidden_states, Sequence):
        raise TypeError("hidden_states must be a sequence")
    if len(hidden_states) != 33:
        raise ValueError(f"expected 33 Transformer outputs, got {len(hidden_states)}")
    for index, state in enumerate(hidden_states):
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"hidden_states[{index}] must be a torch.Tensor")
        if state.ndim != 3 or tuple(state.shape[:2]) != tuple(x_t.shape[:2]):
            raise ValueError(
                f"hidden_states[{index}] must have leading shape "
                f"{tuple(x_t.shape[:2])}, got {tuple(state.shape)}"
            )
        if state.device != x_t.device:
            raise ValueError(f"hidden_states[{index}] must be on {x_t.device}")


def _randn_like(
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def ddpm_reverse_step(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    x_t: torch.Tensor,
    timestep: int,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Perform one fixed-variance ancestral epsilon-prediction reverse step.

    ``timestep`` is a mathematical timestep, not a zero-based buffer index.
    The returned dictionary exposes the mean and existing schedule variance for
    direct diagnostics while ``x_prev`` is the state used by the full sampler.
    At mathematical timestep one, ``noise`` is intentionally ignored.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(schedule, DDPMSchedule):
        raise TypeError("schedule must be a DDPMSchedule")
    if not isinstance(timestep, int) or isinstance(timestep, bool):
        raise TypeError("timestep must be a mathematical integer")
    if not 1 <= timestep <= schedule.num_timesteps:
        raise ValueError(
            f"mathematical timestep must lie in [1,{schedule.num_timesteps}]"
        )

    _validate_latent(x_t, "x_t")
    _validate_conditioning(x_t, z_quant, hidden_states)
    mask = _mask_like(res_mask, x_t)
    mask_3d = mask[..., None]
    masked_x_t = x_t * mask_3d

    batch_timesteps = torch.full(
        (x_t.shape[0],),
        fill_value=timestep,
        dtype=torch.long,
        device=x_t.device,
    )
    # Explicitly exercise the shared mathematical t -> storage t-1 convention.
    schedule.mathematical_timesteps_to_indices(batch_timesteps)
    tau = torch.full(
        (x_t.shape[0],),
        fill_value=float(timestep) / schedule.num_timesteps,
        dtype=x_t.dtype,
        device=x_t.device,
    )

    model.eval()
    epsilon_pred = model(
        x_t=masked_x_t,
        t=tau,
        z_quant=z_quant,
        hidden_states=hidden_states,
        res_mask=mask,
    )
    if not isinstance(epsilon_pred, torch.Tensor):
        raise TypeError("model output must be a torch.Tensor")
    if epsilon_pred.shape != x_t.shape:
        raise ValueError("epsilon prediction must have the same shape as x_t")
    if epsilon_pred.device != x_t.device or epsilon_pred.dtype != x_t.dtype:
        raise ValueError("epsilon prediction and x_t must have the same dtype and device")
    epsilon_pred = epsilon_pred * mask_3d
    _require_finite(epsilon_pred, "epsilon_pred")

    alpha_t = schedule.extract(schedule.alphas, batch_timesteps, like=x_t)
    beta_t = schedule.extract(schedule.betas, batch_timesteps, like=x_t)
    sqrt_one_minus_alpha_bar_t = schedule.extract(
        schedule.sqrt_one_minus_alpha_bars,
        batch_timesteps,
        like=x_t,
    )
    posterior_variance_t = schedule.extract(
        schedule.posterior_variance,
        batch_timesteps,
        like=x_t,
    )

    posterior_mean = (
        masked_x_t
        - beta_t / sqrt_one_minus_alpha_bar_t * epsilon_pred
    ) / torch.sqrt(alpha_t)
    posterior_mean = posterior_mean * mask_3d
    _require_finite(posterior_mean, "posterior_mean")

    if timestep == 1:
        x_prev = posterior_mean
    else:
        if noise is None:
            reverse_noise = torch.randn_like(x_t)
        else:
            _validate_latent(noise, "noise")
            if noise.shape != x_t.shape:
                raise ValueError("noise must have the same shape as x_t")
            if noise.device != x_t.device or noise.dtype != x_t.dtype:
                raise ValueError("noise and x_t must have the same dtype and device")
            reverse_noise = noise
        reverse_noise = reverse_noise * mask_3d
        x_prev = posterior_mean + torch.sqrt(posterior_variance_t) * reverse_noise
        x_prev = x_prev * mask_3d

    _require_finite(x_prev, "x_prev")
    return {
        "x_prev": x_prev,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance_t,
        "epsilon_pred": epsilon_pred,
        "tau": tau,
        "timesteps": batch_timesteps,
    }


@torch.inference_mode()
def sample_residual_ddpm(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    *,
    return_metadata: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Run the complete ancestral chain ``T, T-1, ..., 1``.

    With the primary ``T=1000`` schedule this performs exactly 1000 model
    evaluations.  If ``generator`` is supplied, its single stream is consumed
    sequentially for a generated initial state and every ``t > 1`` reverse
    noise draw, without using the global RNG.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(schedule, DDPMSchedule):
        raise TypeError("schedule must be a DDPMSchedule")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    _validate_latent(z_quant, "z_quant")
    _validate_conditioning(z_quant, z_quant, hidden_states)
    mask = _mask_like(res_mask, z_quant)
    mask_3d = mask[..., None]

    if initial_noise is None:
        x_t = _randn_like(z_quant, generator)
    else:
        _validate_latent(initial_noise, "initial_noise")
        if initial_noise.shape != z_quant.shape:
            raise ValueError("initial_noise must have the same shape as z_quant")
        if initial_noise.device != z_quant.device or initial_noise.dtype != z_quant.dtype:
            raise ValueError(
                "initial_noise and z_quant must have the same dtype and device"
            )
        x_t = initial_noise
    x_t = x_t * mask_3d

    model.eval()
    timestep_order: list[int] = []
    for timestep in range(schedule.num_timesteps, 0, -1):
        reverse_noise = (
            _randn_like(x_t, generator) if timestep > 1 else None
        )
        step = ddpm_reverse_step(
            model=model,
            schedule=schedule,
            x_t=x_t,
            timestep=timestep,
            z_quant=z_quant,
            hidden_states=hidden_states,
            res_mask=mask,
            noise=reverse_noise,
        )
        x_t = step["x_prev"]
        timestep_order.append(timestep)

    r_hat = x_t * mask_3d
    _require_finite(r_hat, "r_hat")
    if return_metadata:
        return r_hat, {
            "nfe": len(timestep_order),
            "timesteps": tuple(timestep_order),
            "sampler": "full_ancestral_ddpm",
        }
    return r_hat

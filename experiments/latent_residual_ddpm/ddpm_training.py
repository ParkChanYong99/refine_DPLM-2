"""Single-batch training computation for the Local Matched Residual DDPM."""

from __future__ import annotations

from typing import Sequence

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import (
    DDPMSchedule,
    masked_epsilon_mse,
    q_sample,
)


def ddpm_training_step(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    x0: torch.Tensor,
    z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute one epsilon-prediction DDPM training batch.

    This function deliberately performs no optimizer, scheduler, gradient
    accumulation, validation, checkpoint, or data-loading work. Public integer
    timesteps use the mathematical convention ``1 <= t <= T``. The residual
    network receives normalized floating-point time ``tau = t / T``.
    """
    if not isinstance(schedule, DDPMSchedule):
        raise TypeError("schedule must be a DDPMSchedule")
    if not isinstance(x0, torch.Tensor):
        raise TypeError("x0 must be a torch.Tensor")

    batch_size = x0.shape[0] if x0.ndim > 0 else 0
    if timesteps is None:
        timesteps = torch.randint(
            low=1,
            high=schedule.num_timesteps + 1,
            size=(batch_size,),
            dtype=torch.long,
            device=x0.device,
        )
    if noise is None:
        noise = torch.randn_like(x0)

    diffusion = q_sample(
        x0=x0,
        timesteps=timesteps,
        noise=noise,
        schedule=schedule,
        res_mask=res_mask,
    )
    tau = timesteps.to(device=x0.device, dtype=x0.dtype) / schedule.num_timesteps
    epsilon_pred = model(
        x_t=diffusion["x_t"],
        t=tau,
        z_quant=z_quant,
        hidden_states=hidden_states,
        res_mask=res_mask,
    )
    loss = masked_epsilon_mse(epsilon_pred, diffusion["epsilon"], res_mask)

    return {
        "loss": loss,
        "x_t": diffusion["x_t"],
        "epsilon": diffusion["epsilon"],
        "epsilon_pred": epsilon_pred,
        "timesteps": timesteps,
        "tau": tau,
    }

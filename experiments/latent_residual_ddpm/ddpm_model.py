"""Parameter-matched epsilon-prediction model for latent residual DDPM."""

from __future__ import annotations

from typing import Sequence

import torch

from experiments.latent_residual_fm.fm_model import LatentResidualFM


class LatentResidualDDPM(LatentResidualFM):
    """DDPM epsilon model with the finalized FM network architecture.

    The inherited modules, parameter names, initialization, masking, condition
    construction, and time embedding are intentionally identical to
    :class:`LatentResidualFM`. Only the semantic interpretation differs: the
    first input is a noised residual ``x_t`` and the output predicts the sampled
    noise ``epsilon``.

    ``t`` is normalized DDPM time ``tau = mathematical_timestep / T`` and must
    therefore be supplied as a floating-point tensor with shape ``[B]`` or
    ``[B,1]`` by the future training or sampling caller.
    """

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon with shape ``[B,L,13]`` and zeroed padding."""
        epsilon_pred = super().forward(
            x_t,
            t,
            z_quant,
            hidden_states,
            res_mask,
        )
        return epsilon_pred

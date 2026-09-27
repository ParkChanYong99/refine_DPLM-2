"""Core latent-residual FM network.

The adaptive-normalization blocks here are an explicit experiment-local
AdaLN-style design.  They are not claimed to reproduce an unpublished ResDiff
implementation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return standard sinusoidal embeddings for continuous ``t`` in [0, 1]."""
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]
    if t.ndim != 1:
        raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")
    half = dim // 2
    if half == 0:
        raise ValueError("timestep embedding dimension must be at least 2")
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    ).to(dtype=t.dtype)
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AdaLNResidualBlock(nn.Module):
    """Residue-wise MLP block modulated by per-residue condition and time."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        adaptive = self.norm(x) * (1.0 + scale) + shift
        update = self.mlp(adaptive)
        x = x + torch.sigmoid(gate) * update
        return x * mask[..., None]


class LatentResidualFM(nn.Module):
    """Velocity model for 13-dimensional LFQ quantization residuals.

    ``hidden_states`` must contain the 33 Transformer layer outputs with the
    embedding/pre-layer state already excluded and structure residues already
    aligned.  DPLM token packing and modality slicing deliberately live outside
    this module.
    """

    def __init__(
        self,
        residual_dim: int = 13,
        condition_dim: int = 1280,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_hidden_states: int = 33,
    ) -> None:
        super().__init__()
        self.residual_dim = residual_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_hidden_states = num_hidden_states

        self.layer_logits = nn.Parameter(torch.zeros(num_hidden_states))
        self.z_quant_projection = nn.Linear(residual_dim, condition_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_projection = nn.Linear(residual_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            AdaLNResidualBlock(hidden_dim) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, residual_dim)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def _validate_inputs(
        self,
        r_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> None:
        if r_t.ndim != 3 or r_t.shape[-1] != self.residual_dim:
            raise ValueError(
                f"r_t must have shape [B,L,{self.residual_dim}], got {tuple(r_t.shape)}"
            )
        if z_quant.shape != r_t.shape:
            raise ValueError("z_quant must have the same shape as r_t")
        if res_mask.shape != r_t.shape[:2]:
            raise ValueError("res_mask must have shape [B,L]")
        if t.ndim not in (1, 2) or t.shape[0] != r_t.shape[0]:
            raise ValueError("t must have shape [B] or [B,1]")
        if t.ndim == 2 and t.shape[1] != 1:
            raise ValueError("rank-2 t must have shape [B,1]")
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} Transformer outputs, got {len(hidden_states)}"
            )
        expected_hidden = (*r_t.shape[:2], self.condition_dim)
        for index, state in enumerate(hidden_states):
            if tuple(state.shape) != expected_hidden:
                raise ValueError(
                    f"hidden_states[{index}] must have shape {expected_hidden}, got {tuple(state.shape)}"
                )

    def forward(
        self,
        r_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(r_t, t, z_quant, hidden_states, res_mask)
        mask = res_mask.to(dtype=r_t.dtype)
        weights = self.layer_weights()

        # Avoid torch.stack([33,B,L,1280]): accumulate the weighted states while
        # retaining gradients for only the 33 learned scalar weights.
        h_mix = hidden_states[0] * weights[0]
        for index in range(1, self.num_hidden_states):
            h_mix = h_mix + hidden_states[index] * weights[index]

        condition_1280 = h_mix + self.z_quant_projection(z_quant)
        condition = self.condition_projection(condition_1280)
        time_embedding = self.time_mlp(
            sinusoidal_timestep_embedding(t.to(dtype=r_t.dtype), self.hidden_dim)
        )
        condition = condition + time_embedding[:, None, :]

        x = self.input_projection(r_t) * mask[..., None]
        for block in self.blocks:
            x = block(x, condition, mask)
        velocity = self.output_projection(self.output_norm(x))
        return velocity * mask[..., None]


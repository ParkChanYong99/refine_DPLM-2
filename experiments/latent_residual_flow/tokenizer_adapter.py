"""Small, read-only adapter around the frozen StructOK structure tokenizer."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from byprot.models.utils import get_struct_tokenizer


class StructureTokenizerAdapter:
    """Expose the continuous, quantized, ID, and decoder sides of VQModel."""

    def __init__(self, struct_tokenizer: nn.Module) -> None:
        self.struct_tokenizer = struct_tokenizer

    @classmethod
    def load_struct_tokenizer(
        cls, device: str | torch.device = "cpu"
    ) -> "StructureTokenizerAdapter":
        """Load, move, freeze, and wrap the tokenizer for inference only."""
        model = get_struct_tokenizer().to(torch.device(device)).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model)

    @torch.no_grad()
    def encode_continuous(
        self,
        all_atom_positions: Tensor,
        res_mask: Tensor,
        seq_length: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return pre-LFQ ``e_cont [B,L,13]`` and GVP features [B,L,512]."""
        e_cont, encoder_feats = self.struct_tokenizer.encode(
            all_atom_positions, res_mask, seq_length
        )
        return e_cont, encoder_feats

    @torch.no_grad()
    def quantize(self, e_cont: Tensor, res_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """LFQ-quantize a continuous latent and return raw tokenizer IDs."""
        e_quant, _, (_, _, struct_ids) = self.struct_tokenizer.quantize(
            e_cont, mask=res_mask.bool()
        )
        return e_quant, struct_ids

    @torch.no_grad()
    def ids_to_quantized_latent(self, struct_ids: Tensor) -> Tensor:
        """Map raw LFQ IDs in [0,8191] to their 13-dimensional code vectors."""
        if struct_ids.ndim != 2:
            raise ValueError(f"struct_ids must have shape [B,L], got {struct_ids.shape}")
        if struct_ids.numel() and (
            int(struct_ids.min()) < 0 or int(struct_ids.max()) > 8191
        ):
            raise ValueError("struct_ids must be raw tokenizer IDs in [0,8191]")
        return self.struct_tokenizer.quantize.get_codebook_entry(struct_ids)

    @torch.no_grad()
    def decode_latent(self, latent: Tensor, res_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode a rank-3 latent [B,L,13] through VQModel.detokenize()."""
        if latent.ndim != 3 or latent.shape[-1] != 13:
            raise ValueError(f"latent must have shape [B,L,13], got {latent.shape}")
        decoded = self.struct_tokenizer.detokenize(latent, res_mask=res_mask)
        return decoded["atom37_positions"], decoded["atom37_mask"]


def load_struct_tokenizer(device: str | torch.device = "cpu") -> StructureTokenizerAdapter:
    """Module-level convenience wrapper for the adapter classmethod."""
    return StructureTokenizerAdapter.load_struct_tokenizer(device)

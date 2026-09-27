"""Frozen clean teacher-forced DPLM-2 Bit condition extraction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from byprot.datamodules.pdb_dataset import utils as du
from byprot.models.dplm2 import DPLM2Bit


class DPLMConditionExtractor(nn.Module):
    """Return 33 residue-aligned structure-side Transformer states.

    The current validated protocol is intentionally batch-size one.  Inputs are
    clean current LFQ IDs and repository aatype indices; no diffusion masking or
    generation is performed.
    """

    def __init__(
        self,
        checkpoint: str = "airkingbd/dplm2_bit_650m",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.model = DPLM2Bit.from_pretrained(checkpoint)
        self.model.requires_grad_(False).eval().to(self.device)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("DPLM2Bit is not fully frozen")

    @property
    def hidden_size(self) -> int:
        return int(self.model.net.config.hidden_size)

    @property
    def num_transformer_layers(self) -> int:
        return int(self.model.net.config.num_hidden_layers)

    def pack_clean_teacher_forced_inputs(
        self,
        aatype: torch.Tensor,
        struct_ids_current: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if aatype.ndim != 2 or struct_ids_current.ndim != 2 or res_mask.ndim != 2:
            raise ValueError("aatype, struct_ids_current, and res_mask must be rank-2 [B,L]")
        if aatype.shape != struct_ids_current.shape or aatype.shape != res_mask.shape:
            raise ValueError("aatype, struct_ids_current, and res_mask shapes must match")
        if aatype.shape[0] != 1:
            raise ValueError("the current validated condition extractor supports batch size 1 only")
        if struct_ids_current.numel() and (
            int(struct_ids_current.min()) < 0 or int(struct_ids_current.max()) > 8191
        ):
            raise ValueError("struct_ids_current must contain raw LFQ IDs in [0,8191]")

        batch_size, length = aatype.shape
        raw_ids = struct_ids_current.long().to(self.device)
        struct_middle = raw_ids + self.model.struct_vocab_offset
        struct_bos = torch.full(
            (batch_size, 1),
            self.model.struct_bos_id,
            dtype=torch.long,
            device=self.device,
        )
        struct_eos = torch.full(
            (batch_size, 1),
            self.model.struct_eos_id,
            dtype=torch.long,
            device=self.device,
        )
        struct_tokens = torch.cat((struct_bos, struct_middle, struct_eos), dim=1)

        # Repository-authoritative aatype convention -> sequence, then the same
        # DPLM2Tokenizer special-token strings used by generate_dplm2.py.
        aa_sequence = du.aatype_to_seq(aatype[0].detach().cpu().tolist())
        aa_string = (
            self.model.tokenizer.aa_cls_token
            + aa_sequence
            + self.model.tokenizer.aa_eos_token
        )
        aa_encoded = self.model.tokenizer.batch_encode_plus(
            [aa_string],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        aa_tokens = aa_encoded["input_ids"].long().to(self.device)
        expected_half_shape = (batch_size, length + 2)
        if tuple(struct_tokens.shape) != expected_half_shape:
            raise AssertionError(f"structure half shape is {tuple(struct_tokens.shape)}")
        if tuple(aa_tokens.shape) != expected_half_shape:
            raise AssertionError(f"AA half shape is {tuple(aa_tokens.shape)}")
        if int(aa_tokens[0, 0]) != self.model.aa_bos_id:
            raise AssertionError("AA BOS token mismatch")
        if int(aa_tokens[0, -1]) != self.model.aa_eos_id:
            raise AssertionError("AA EOS token mismatch")
        if not torch.equal(
            struct_tokens[:, 1:-1] - self.model.struct_vocab_offset, raw_ids
        ):
            raise AssertionError("structure offset packing does not recover raw LFQ IDs")
        packed_tokens = torch.cat((struct_tokens, aa_tokens), dim=1)
        return {
            "packed_tokens": packed_tokens,
            "struct_tokens": struct_tokens,
            "aa_tokens": aa_tokens,
        }

    def forward(
        self,
        aatype: torch.Tensor,
        struct_ids_current: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> dict[str, Any]:
        packed = self.pack_clean_teacher_forced_inputs(
            aatype, struct_ids_current, res_mask
        )
        packed_tokens = packed["packed_tokens"]
        with torch.no_grad():
            outputs = self.model(packed_tokens)
        all_hidden_states = outputs.get("all_hidden_states")
        if all_hidden_states is None:
            raise RuntimeError("DPLM forward returned no all_hidden_states")
        expected_count = self.num_transformer_layers + 1
        if len(all_hidden_states) != expected_count:
            raise AssertionError(
                f"hidden tuple length {len(all_hidden_states)} != {expected_count}"
            )
        if self.num_transformer_layers != 33:
            raise AssertionError(
                f"expected 33 Transformer layers, got {self.num_transformer_layers}"
            )

        struct_tokens = packed["struct_tokens"]
        batch_size, half_length = struct_tokens.shape
        residue_positions = struct_tokens.ne(self.model.struct_bos_id) & struct_tokens.ne(
            self.model.struct_eos_id
        )
        counts = residue_positions.sum(dim=1)
        if not bool(torch.all(counts == half_length - 2)):
            raise AssertionError(f"BOS/EOS residue selection counts: {counts.tolist()}")
        expected_full_shape = (
            batch_size,
            packed_tokens.shape[1],
            self.hidden_size,
        )
        aligned: list[torch.Tensor] = []
        for index, state in enumerate(all_hidden_states[1:], start=1):
            if tuple(state.shape) != expected_full_shape:
                raise AssertionError(
                    f"all_hidden_states[{index}] shape {tuple(state.shape)} != {expected_full_shape}"
                )
            structure_state, _ = state.chunk(2, dim=1)
            residue_state = structure_state[residue_positions].view(
                batch_size, half_length - 2, self.hidden_size
            )
            if not bool(torch.isfinite(residue_state).all()):
                raise ValueError(f"aligned Transformer state {index - 1} is non-finite")
            aligned.append(residue_state.detach())
        if len(aligned) != 33:
            raise AssertionError(f"aligned Transformer state count is {len(aligned)}")
        return {
            "hidden_states": tuple(aligned),
            "packed_tokens": packed_tokens,
            "all_hidden_states_count": len(all_hidden_states),
            "transformer_states_used": len(aligned),
            "full_hidden_shape": expected_full_shape,
        }


#!/usr/bin/env python3
"""Connect one real AFDB train sample to one matched-DDPM training step."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback
from typing import Sequence

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import (
    DDPMSchedule,
    masked_epsilon_mse,
)
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_ddpm.ddpm_training import ddpm_training_step
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.train_latent_residual_fm import (
    DPLM_CHECKPOINT,
    Timings,
    load_shard,
    load_split,
    sample_tensors,
)


DATASET_DIR = Path("data-bin/latent_residual_fm/afdb_l512_sharded")
SPLIT_CSV = DATASET_DIR / "splits" / "split_v1.csv"
NUM_TIMESTEPS = 1_000
BETA_START = 1e-4
BETA_END = 2e-2
TIMESTEP = 500
EXPECTED_DDPM_TRAINABLE_PARAMETERS = 34_946_606
EXPECTED_HIDDEN_STATES = 33
EXPECTED_HIDDEN_WIDTH = 1280
RESIDUAL_DIM = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parameter_count(model: torch.nn.Module, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def all_finite(tensors: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def isolated_model_initialization(
    device: torch.device, seed: int
) -> LatentResidualDDPM:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        model = LatentResidualDDPM().to(device)
    return model


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")

    print("[DDPM TRAINING INTEGRATION DEBUG]")
    train_rows, _ = load_split(SPLIT_CSV)
    require(bool(train_rows), "train split is empty")
    row = min(train_rows, key=lambda item: (item.length, item.sample_id))
    selection = "minimum train-split length; sample_id lexical tie-break"
    timings = Timings()
    shard = load_shard(DATASET_DIR / row.shard_file, timings, device)
    require(row.index_in_shard < len(shard["samples"]), "index outside shard")
    sample = shard["samples"][row.index_in_shard]
    require(sample["sample_id"] == row.sample_id, "split/shard sample ID mismatch")
    require(int(sample["length"]) == row.length, "split/shard length mismatch")

    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype, struct_ids, z_quant, x0, res_mask = sample_tensors(sample, device)
    stored_residual = sample["residual"].float().view(1, length, RESIDUAL_DIM).to(device)
    x0_is_stored_residual = torch.equal(x0, stored_residual)
    require(x0_is_stored_residual, "x0 differs from the stored residual")
    del stored_residual
    del sample, shard

    expected_residual_shape = (1, length, RESIDUAL_DIM)
    expected_token_shape = (1, length)
    require(tuple(aatype.shape) == expected_token_shape, "aatype shape mismatch")
    require(
        tuple(struct_ids.shape) == expected_token_shape,
        "struct_ids_current shape mismatch",
    )
    require(tuple(z_quant.shape) == expected_residual_shape, "z_quant shape mismatch")
    require(tuple(x0.shape) == expected_residual_shape, "x0/residual shape mismatch")
    require(tuple(res_mask.shape) == expected_token_shape, "res_mask shape mismatch")
    require(
        all_finite((aatype, struct_ids, z_quant, x0, res_mask)),
        "one or more sample tensors are non-finite",
    )

    extractor = DPLMConditionExtractor(DPLM_CHECKPOINT, device)
    extractor.eval().requires_grad_(False)
    dplm = extractor.model
    tokenizer = dplm.struct_tokenizer
    tokenizer.eval().requires_grad_(False)
    dplm_trainable = parameter_count(dplm, trainable_only=True)
    tokenizer_trainable = parameter_count(tokenizer, trainable_only=True)
    require(dplm_trainable == 0, "DPLM trainable parameter count is not zero")
    require(
        tokenizer_trainable == 0,
        "structure tokenizer trainable parameter count is not zero",
    )

    with torch.no_grad():
        condition = extractor(aatype, struct_ids, res_mask)
    hidden_states = tuple(condition["hidden_states"])
    require(
        condition["all_hidden_states_count"] == EXPECTED_HIDDEN_STATES + 1,
        "embedding-plus-Transformer hidden-state count mismatch",
    )
    require(
        condition["transformer_states_used"] == EXPECTED_HIDDEN_STATES,
        "Transformer hidden-state count mismatch",
    )
    require(
        len(hidden_states) == EXPECTED_HIDDEN_STATES,
        "aligned hidden-state count mismatch",
    )
    expected_hidden_shape = (1, length, EXPECTED_HIDDEN_WIDTH)
    require(
        all(tuple(state.shape) == expected_hidden_shape for state in hidden_states),
        "one or more aligned hidden-state shapes are invalid",
    )
    require(all_finite(hidden_states), "one or more aligned hidden states are non-finite")
    require(
        all(not state.requires_grad and state.grad_fn is None for state in hidden_states),
        "frozen DPLM condition is attached to an autograd graph",
    )

    packed_tokens = condition["packed_tokens"]
    expected_half_length = length + 2
    require(
        tuple(packed_tokens.shape) == (1, 2 * expected_half_length),
        "packed DPLM token shape mismatch",
    )
    packed_struct_tokens = packed_tokens[:, :expected_half_length]
    packed_aa_tokens = packed_tokens[:, expected_half_length:]
    aa_tokens = packed_aa_tokens[:, 1:-1]
    recovered_raw_ids = (
        packed_struct_tokens[:, 1:-1] - dplm.struct_vocab_offset
    )
    token_consistency = torch.equal(
        recovered_raw_ids.cpu(), struct_ids.long().cpu()
    )
    require(token_consistency, "packed structure IDs do not recover stored raw LFQ IDs")
    require(tuple(aa_tokens.shape) == expected_token_shape, "AA token shape mismatch")

    # The condition tensors are detached. Move the frozen DPLM/tokenizer off GPU
    # before constructing the trainable DDPM to fit the single-GPU debug budget.
    dplm.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    schedule = DDPMSchedule(
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ).to(device)
    ddpm_model = isolated_model_initialization(device, args.seed)
    ddpm_model.train()
    ddpm_trainable = parameter_count(ddpm_model, trainable_only=True)
    require(
        ddpm_trainable == EXPECTED_DDPM_TRAINABLE_PARAMETERS,
        f"DDPM trainable parameter count is {ddpm_trainable}",
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    deterministic_noise = torch.randn(
        x0.shape,
        dtype=x0.dtype,
        device=device,
        generator=generator,
    )
    timesteps = torch.tensor([TIMESTEP], dtype=torch.long, device=device)
    result = ddpm_training_step(
        model=ddpm_model,
        schedule=schedule,
        x0=x0,
        z_quant=z_quant,
        hidden_states=hidden_states,
        res_mask=res_mask,
        timesteps=timesteps,
        noise=deterministic_noise,
    )

    require(tuple(result["x_t"].shape) == expected_residual_shape, "x_t shape mismatch")
    require(
        tuple(result["epsilon"].shape) == expected_residual_shape,
        "epsilon shape mismatch",
    )
    require(
        tuple(result["epsilon_pred"].shape) == expected_residual_shape,
        "epsilon_pred shape mismatch",
    )
    require(tuple(result["timesteps"].shape) == (1,), "timestep shape mismatch")
    require(tuple(result["tau"].shape) == (1,), "tau shape mismatch")
    require(
        torch.allclose(result["tau"], torch.tensor([0.5], device=device)),
        f"tau is {result['tau'].detach().cpu().tolist()}, expected [0.5]",
    )
    require(result["loss"].ndim == 0, "loss is not scalar")
    require(bool(torch.isfinite(result["loss"])), "loss is non-finite")

    valid = res_mask.bool()
    padding = ~valid
    valid_finite = all_finite(
        (
            result["x_t"][valid],
            result["epsilon"][valid],
            result["epsilon_pred"][valid],
        )
    )
    padding_zero = all(
        bool((tensor[padding] == 0).all())
        for tensor in (
            result["x_t"],
            result["epsilon"],
            result["epsilon_pred"],
        )
    )
    require(valid_finite, "a valid-position DDPM tensor is non-finite")
    require(padding_zero, "a padding-position DDPM tensor is nonzero")

    mask = res_mask.to(dtype=x0.dtype)[..., None]
    alpha_bar = schedule.alpha_bars[TIMESTEP - 1].to(dtype=x0.dtype)
    x_t_manual = (
        torch.sqrt(alpha_bar) * (x0 * mask)
        + torch.sqrt(1.0 - alpha_bar) * (result["epsilon"] * mask)
    ) * mask
    formula_allclose = torch.allclose(result["x_t"], x_t_manual)
    require(formula_allclose, "q_sample output does not match the manual formula")
    direct_loss = masked_epsilon_mse(
        result["epsilon_pred"], result["epsilon"], res_mask
    )
    loss_allclose = torch.allclose(result["loss"], direct_loss)
    require(loss_allclose, "training-step loss does not match masked_epsilon_mse")

    result["loss"].backward()
    ddpm_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in ddpm_model.parameters()
        if parameter.requires_grad
    )
    dplm_gradients_absent = all(parameter.grad is None for parameter in dplm.parameters())
    tokenizer_gradients_absent = all(
        parameter.grad is None for parameter in tokenizer.parameters()
    )
    require(ddpm_gradients_finite, "DDPM gradient is missing or non-finite")
    require(dplm_gradients_absent, "a frozen DPLM parameter received a gradient")
    require(
        tokenizer_gradients_absent,
        "a frozen structure-tokenizer parameter received a gradient",
    )

    print(f"sample selection: {selection}")
    print(f"sample_id: {sample_id}")
    print(f"length: {length}")
    print(f"struct_ids_current: {tuple(struct_ids.shape)}")
    print(f"aatype: {tuple(aatype.shape)}")
    print(f"aa tokens: {tuple(aa_tokens.shape)}")
    print(f"token consistency: {token_consistency}")
    print(f"x0: {tuple(x0.shape)}")
    print(f"x0 equals stored residual: {x0_is_stored_residual}")
    print(f"z_quant: {tuple(z_quant.shape)}")
    print(f"res_mask: {tuple(res_mask.shape)}")
    print(f"sample tensors finite: {all_finite((z_quant, x0, res_mask))}")
    print(f"hidden states: {expected_hidden_shape}")
    print(f"hidden count: {len(hidden_states)}")
    print(f"hidden width: {hidden_states[0].shape[-1]}")
    print(f"DDPM parameters: {ddpm_trainable}")
    print(f"DPLM trainable parameters: {dplm_trainable}")
    print(f"Tokenizer trainable parameters: {tokenizer_trainable}")
    print(f"timestep: {int(result['timesteps'].item())}")
    print(f"tau: {float(result['tau'].item())}")
    print(f"x_t: {tuple(result['x_t'].shape)}")
    print(f"epsilon: {tuple(result['epsilon'].shape)}")
    print(f"epsilon_pred: {tuple(result['epsilon_pred'].shape)}")
    print(f"loss: {float(result['loss'].detach())}")
    print(f"valid tensors finite: {valid_finite}")
    print(f"padding positions: {int(padding.sum())}")
    print(f"padding output zero: {padding_zero}")
    print(f"q_sample formula allclose: {formula_allclose}")
    print(f"loss allclose: {loss_allclose}")
    print(f"DDPM gradients finite: {ddpm_gradients_finite}")
    print(f"DPLM gradients absent: {dplm_gradients_absent}")
    print(f"Tokenizer gradients absent: {tokenizer_gradients_absent}")
    print("DDPM_TRAINING_INTEGRATION_PASS")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[DDPM_TRAINING_INTEGRATION_FAIL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

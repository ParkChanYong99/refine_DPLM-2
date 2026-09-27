#!/usr/bin/env python3
"""Four-sample tiny-overfit diagnostic for the matched residual DDPM."""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path
import traceback

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
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
NUM_SAMPLES = 4
NUM_TIMESTEPS = 1_000
FIXED_TIMESTEPS = (100, 300, 600, 900)
TINY_OVERFIT_LR = 1e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_MAX_NORM = 1.0
MAX_OPTIMIZER_STEPS = 500
LOG_STEPS = (0, 1, 10, 25, 50, 100, 200, 300, 400, 500)
EXPECTED_DDPM_PARAMETERS = 34_946_606
EXPECTED_HIDDEN_STATES = 33
EXPECTED_HIDDEN_WIDTH = 1280
RESIDUAL_DIM = 13


@dataclass(frozen=True)
class CachedSample:
    sample_id: str
    length: int
    x0: torch.Tensor
    z_quant: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    res_mask: torch.Tensor
    timestep: torch.Tensor
    noise: torch.Tensor
    noise_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def gradients_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def parameters_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


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


def cache_training_samples(
    device: torch.device,
    base_seed: int,
) -> tuple[list[CachedSample], int, int, bool, bool]:
    train_rows, _ = load_split(SPLIT_CSV)
    require(len(train_rows) >= NUM_SAMPLES, "train split has fewer than four samples")
    selected_rows = sorted(
        train_rows,
        key=lambda row: (row.length, row.sample_id),
    )[:NUM_SAMPLES]
    timings = Timings()

    extractor = DPLMConditionExtractor(DPLM_CHECKPOINT, device)
    extractor.eval().requires_grad_(False)
    dplm = extractor.model
    tokenizer = dplm.struct_tokenizer
    tokenizer.eval().requires_grad_(False)
    dplm_trainable = parameter_count(dplm)
    tokenizer_trainable = parameter_count(tokenizer)
    require(dplm_trainable == 0, "DPLM is not fully frozen")
    require(tokenizer_trainable == 0, "structure tokenizer is not fully frozen")

    cached: list[CachedSample] = []
    for sample_index, (row, timestep) in enumerate(
        zip(selected_rows, FIXED_TIMESTEPS)
    ):
        shard = load_shard(DATASET_DIR / row.shard_file, timings, device)
        require(row.index_in_shard < len(shard["samples"]), "index outside shard")
        sample = shard["samples"][row.index_in_shard]
        require(sample["sample_id"] == row.sample_id, "split/shard sample ID mismatch")
        require(int(sample["length"]) == row.length, "split/shard length mismatch")

        length = int(sample["length"])
        aatype, struct_ids, z_quant, x0, res_mask = sample_tensors(sample, device)
        stored_residual = sample["residual"].float().view(1, length, RESIDUAL_DIM)
        require(
            torch.equal(x0.detach().cpu(), stored_residual),
            "x0 differs from the stored residual",
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
            "aligned hidden-state shape mismatch",
        )
        require(
            all(
                not state.requires_grad
                and state.grad_fn is None
                and bool(torch.isfinite(state).all())
                for state in hidden_states
            ),
            "a cached condition is non-finite or attached to autograd",
        )

        noise_seed = base_seed + sample_index
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(noise_seed)
        noise = torch.randn(
            (1, length, RESIDUAL_DIM),
            dtype=torch.float32,
            generator=noise_generator,
        )
        cached.append(
            CachedSample(
                sample_id=str(sample["sample_id"]),
                length=length,
                x0=x0.detach().float().cpu(),
                z_quant=z_quant.detach().float().cpu(),
                hidden_states=tuple(
                    state.detach().float().cpu() for state in hidden_states
                ),
                res_mask=res_mask.detach().float().cpu(),
                timestep=torch.tensor([timestep], dtype=torch.long),
                noise=noise,
                noise_seed=noise_seed,
            )
        )
        del sample, shard, condition, hidden_states
        del aatype, struct_ids, z_quant, x0, res_mask, stored_residual

    dplm_gradients_absent = all(parameter.grad is None for parameter in dplm.parameters())
    tokenizer_gradients_absent = all(
        parameter.grad is None for parameter in tokenizer.parameters()
    )
    require(dplm_gradients_absent, "a DPLM parameter received a gradient")
    require(tokenizer_gradients_absent, "a tokenizer parameter received a gradient")

    # Cached tensors are detached CPU copies, so the frozen extractor and its
    # owned DPLM/tokenizer modules are no longer needed during optimization.
    dplm.to("cpu")
    del tokenizer, dplm, extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        cached,
        dplm_trainable,
        tokenizer_trainable,
        dplm_gradients_absent,
        tokenizer_gradients_absent,
    )


def sample_to_device(
    sample: CachedSample,
    device: torch.device,
) -> dict[str, object]:
    return {
        "x0": sample.x0.to(device),
        "z_quant": sample.z_quant.to(device),
        "hidden_states": tuple(state.to(device) for state in sample.hidden_states),
        "res_mask": sample.res_mask.to(device),
        "timesteps": sample.timestep.to(device),
        "noise": sample.noise.to(device),
    }


def evaluate_fixed_set(
    model: LatentResidualDDPM,
    schedule: DDPMSchedule,
    cached: list[CachedSample],
    device: torch.device,
) -> list[float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for sample in cached:
            tensors = sample_to_device(sample, device)
            result = ddpm_training_step(
                model=model,
                schedule=schedule,
                x0=tensors["x0"],
                z_quant=tensors["z_quant"],
                hidden_states=tensors["hidden_states"],
                res_mask=tensors["res_mask"],
                timesteps=tensors["timesteps"],
                noise=tensors["noise"],
            )
            value = float(result["loss"])
            require(math.isfinite(value), f"non-finite evaluation loss for {sample.sample_id}")
            losses.append(value)
            del tensors, result
    model.train()
    return losses


def mean(values: list[float]) -> float:
    require(bool(values), "cannot average an empty sequence")
    return sum(values) / len(values)


def maximum_parameter_change(
    model: LatentResidualDDPM,
    initial_parameters: dict[str, torch.Tensor],
) -> float:
    maximum = 0.0
    for name, parameter in model.named_parameters():
        initial = initial_parameters[name]
        change = float((parameter.detach().cpu() - initial).abs().max())
        maximum = max(maximum, change)
    return maximum


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")

    print("[DDPM TINY OVERFIT DEBUG]")
    print("This is a tiny-overfit diagnostic only.")
    print("Fixed timestep/noise are used only to verify trainability.")
    print("The full matched DDPM training will use:")
    print("t ~ Uniform{1,...,T}")
    print("epsilon ~ N(0,I) for every training example/step.")

    (
        cached,
        dplm_trainable,
        tokenizer_trainable,
        dplm_gradients_absent,
        tokenizer_gradients_absent,
    ) = cache_training_samples(device, args.seed)
    require(len(cached) == NUM_SAMPLES, "cached sample count is not four")

    schedule = DDPMSchedule(
        num_timesteps=NUM_TIMESTEPS,
        beta_start=1e-4,
        beta_end=0.02,
    ).to(device)
    model = isolated_model_initialization(device, args.seed)
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    ddpm_parameters = sum(parameter.numel() for parameter in trainable_parameters)
    require(
        ddpm_parameters == EXPECTED_DDPM_PARAMETERS,
        f"DDPM trainable parameter count is {ddpm_parameters}",
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=TINY_OVERFIT_LR,
        weight_decay=WEIGHT_DECAY,
    )
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    initial_losses = evaluate_fixed_set(model, schedule, cached, device)
    initial_mean_loss = mean(initial_losses)
    require(math.isfinite(initial_mean_loss), "initial mean loss is non-finite")
    require(
        all(value > 0.0 for value in initial_losses),
        "initial per-sample loss must be positive for reduction diagnostics",
    )
    logged_mean_losses: dict[int, float] = {0: initial_mean_loss}
    print(f"step 0: {initial_mean_loss:.9f}")
    first_gradient_norm = float("nan")
    all_gradients_finite = True
    all_parameters_finite = True

    for optimizer_step in range(1, MAX_OPTIMIZER_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for sample in cached:
            tensors = sample_to_device(sample, device)
            result = ddpm_training_step(
                model=model,
                schedule=schedule,
                x0=tensors["x0"],
                z_quant=tensors["z_quant"],
                hidden_states=tensors["hidden_states"],
                res_mask=tensors["res_mask"],
                timesteps=tensors["timesteps"],
                noise=tensors["noise"],
            )
            loss = result["loss"]
            require(
                bool(torch.isfinite(loss)),
                f"non-finite training loss at optimizer step {optimizer_step}",
            )
            (loss / NUM_SAMPLES).backward()
            accumulated_loss += float(loss.detach()) / NUM_SAMPLES
            del tensors, result, loss

        require(
            math.isfinite(accumulated_loss),
            f"non-finite accumulated loss at optimizer step {optimizer_step}",
        )
        pre_clip_finite = gradients_finite(trainable_parameters)
        require(
            pre_clip_finite,
            f"missing or non-finite gradient before clipping at step {optimizer_step}",
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=GRADIENT_CLIP_MAX_NORM,
        )
        require(
            bool(torch.isfinite(gradient_norm)),
            f"non-finite gradient norm at optimizer step {optimizer_step}",
        )
        post_clip_finite = gradients_finite(trainable_parameters)
        require(
            post_clip_finite,
            f"non-finite gradient after clipping at step {optimizer_step}",
        )
        if optimizer_step == 1:
            first_gradient_norm = float(gradient_norm)
        all_gradients_finite = (
            all_gradients_finite and pre_clip_finite and post_clip_finite
        )
        optimizer.step()
        step_parameters_finite = parameters_finite(trainable_parameters)
        require(
            step_parameters_finite,
            f"non-finite model parameter after optimizer step {optimizer_step}",
        )
        all_parameters_finite = all_parameters_finite and step_parameters_finite

        if optimizer_step in LOG_STEPS:
            step_losses = evaluate_fixed_set(model, schedule, cached, device)
            step_mean_loss = mean(step_losses)
            require(
                math.isfinite(step_mean_loss),
                f"non-finite logged mean loss at step {optimizer_step}",
            )
            logged_mean_losses[optimizer_step] = step_mean_loss
            print(f"step {optimizer_step}: {step_mean_loss:.9f}")

    final_losses = evaluate_fixed_set(model, schedule, cached, device)
    final_mean_loss = mean(final_losses)
    max_parameter_change = maximum_parameter_change(model, initial_parameters)
    mean_reduction = (initial_mean_loss - final_mean_loss) / initial_mean_loss * 100.0
    per_sample_reductions = [
        (initial - final) / initial * 100.0
        for initial, final in zip(initial_losses, final_losses)
    ]

    all_losses_finite = all(
        math.isfinite(value)
        for value in initial_losses + final_losses + list(logged_mean_losses.values())
    )
    parameter_update_occurred = max_parameter_change > 0.0
    mean_improved = final_mean_loss < initial_mean_loss
    every_sample_improved = all(
        final < initial for initial, final in zip(initial_losses, final_losses)
    )
    pass_conditions = {
        "all losses finite": all_losses_finite,
        "gradients finite": all_gradients_finite,
        "parameters finite": all_parameters_finite,
        "parameter update occurred": parameter_update_occurred,
        "final mean below initial mean": mean_improved,
        "all four samples improved": every_sample_improved,
    }
    require(all(pass_conditions.values()), "one or more tiny-overfit PASS conditions failed")

    print(f"train samples: {len(cached)}")
    print("sample selection: first 4 by ascending (length, sample_id) in train split")
    for index, sample in enumerate(cached):
        print(f"sample {index}:")
        print(f"  id: {sample.sample_id}")
        print(f"  length: {sample.length}")
        print(f"  timestep: {int(sample.timestep.item())}")
        print(f"  noise_seed: {sample.noise_seed}")
    print(f"DPLM trainable parameters: {dplm_trainable}")
    print(f"Tokenizer trainable parameters: {tokenizer_trainable}")
    print(f"DPLM gradients absent after caching: {dplm_gradients_absent}")
    print(f"Tokenizer gradients absent after caching: {tokenizer_gradients_absent}")
    print(f"DDPM parameters: {ddpm_parameters}")
    print("optimizer: AdamW")
    print(f"learning rate: {TINY_OVERFIT_LR}")
    print(f"weight decay: {WEIGHT_DECAY}")
    print(f"gradient clipping: {GRADIENT_CLIP_MAX_NORM}")
    print(f"effective tiny batch: {NUM_SAMPLES}")
    print("optimizer step: mean gradient of all 4 fixed samples, then one update")
    print(f"max optimizer steps: {MAX_OPTIMIZER_STEPS}")
    print(f"initial per-sample losses: {initial_losses}")
    print(f"initial mean loss: {initial_mean_loss:.9f}")
    for step in LOG_STEPS[1:]:
        print(f"logged step {step}: {logged_mean_losses[step]:.9f}")
    print(f"final per-sample losses: {final_losses}")
    print(f"final mean loss: {final_mean_loss:.9f}")
    print(f"mean loss reduction: {mean_reduction:.6f}%")
    print(f"per-sample reductions: {per_sample_reductions}")
    print(f"max parameter change: {max_parameter_change:.9g}")
    print(f"parameters finite: {all_parameters_finite}")
    print(f"first pre-clip gradient norm: {first_gradient_norm:.9f}")
    print(f"gradients finite: {all_gradients_finite}")
    print("PASS condition:")
    for name, passed in pass_conditions.items():
        print(f"  {name}: {passed}")
    print("DDPM_TINY_OVERFIT_PASS")


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[DDPM_TINY_OVERFIT_FAIL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

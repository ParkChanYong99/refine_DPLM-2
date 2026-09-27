#!/usr/bin/env python3
"""Single-sample prediction-time integration debug for the matched DDPM."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_ddpm.ddpm_sampling import sample_residual_ddpm
from experiments.latent_residual_fm.debug_interfaces import (
    EXPECTED_CODEBOOK_DIM,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
    load_models,
    require,
    resolve_device,
    seed_everything,
)
from experiments.latent_residual_fm.debug_prediction_time_fm import (
    GENERATION_MAX_ITER,
    assert_atom37,
    build_folding_input,
    extract_final_token_condition,
    load_validation_sample,
    reconstruct_prediction_latent,
)


DEFAULT_DATASET_DIR = "data-bin/latent_residual_fm/afdb_l512_sharded"
DEFAULT_SPLIT_CSV = (
    "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main"
)
DEFAULT_DPLM_CHECKPOINT = "airkingbd/dplm2_bit_650m"
EXPECTED_OPTIMIZER_STEP = 100_000
EXPECTED_BEST_STEP = 100_000
EXPECTED_BEST_VAL_LOSS = 0.150297817
EXPECTED_DDPM_PARAMETERS = 34_946_606
NUM_TIMESTEPS = 1_000
BETA_START = 1e-4
BETA_END = 2e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dplm_checkpoint", default=DEFAULT_DPLM_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--val_index", type=int, default=0)
    parser.add_argument(
        "--skip_repeatability_check",
        action="store_true",
        help="Skip the second full 1000-NFE DDPM run.",
    )
    parser.add_argument(
        "--skip_decode",
        action="store_true",
        help="Skip the optional continuous-latent tokenizer decode diagnostic.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function: Any) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - started


def all_grads_none(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def load_ddpm(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LatentResidualDDPM, DDPMSchedule, dict[str, float | int]]:
    require(checkpoint_path.is_file(), f"checkpoint_best not found: {checkpoint_path}")
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", mmap=True
    )
    require(isinstance(checkpoint, dict), "checkpoint_best payload is not a dictionary")
    required = {
        "ddpm_model_state_dict",
        "ddpm_schedule_state_dict",
        "optimizer_step",
        "best_step",
        "best_val_loss",
        "ddpm_schedule_configuration",
        "ddpm_architecture_configuration",
    }
    missing = required.difference(checkpoint)
    require(not missing, f"checkpoint_best missing keys: {sorted(missing)}")

    optimizer_step = int(checkpoint["optimizer_step"])
    best_step = int(checkpoint["best_step"])
    best_val_loss = float(checkpoint["best_val_loss"])
    require(
        optimizer_step == EXPECTED_OPTIMIZER_STEP,
        f"expected optimizer_step={EXPECTED_OPTIMIZER_STEP}, got {optimizer_step}",
    )
    require(
        best_step == EXPECTED_BEST_STEP,
        f"expected best_step={EXPECTED_BEST_STEP}, got {best_step}",
    )
    require(math.isfinite(best_val_loss), "checkpoint best_val_loss is non-finite")
    require(
        math.isclose(
            best_val_loss,
            EXPECTED_BEST_VAL_LOSS,
            rel_tol=0.0,
            abs_tol=5e-10,
        ),
        "checkpoint best_val_loss does not round to 0.150297817",
    )

    expected_schedule_configuration = {
        "num_timesteps": NUM_TIMESTEPS,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "schedule": "linear",
        "prediction_type": "epsilon",
    }
    require(
        checkpoint["ddpm_schedule_configuration"]
        == expected_schedule_configuration,
        "checkpoint_best DDPM schedule configuration mismatch",
    )

    ddpm = LatentResidualDDPM()
    parameter_count = sum(parameter.numel() for parameter in ddpm.parameters())
    require(
        parameter_count == EXPECTED_DDPM_PARAMETERS,
        f"DDPM parameter count is {parameter_count}, expected {EXPECTED_DDPM_PARAMETERS}",
    )
    saved_architecture = checkpoint["ddpm_architecture_configuration"]
    require(
        int(saved_architecture.get("total_parameters", -1))
        == EXPECTED_DDPM_PARAMETERS,
        "checkpoint_best architecture parameter count mismatch",
    )
    ddpm.load_state_dict(checkpoint["ddpm_model_state_dict"], strict=True)

    schedule = DDPMSchedule(
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )
    schedule.load_state_dict(checkpoint["ddpm_schedule_state_dict"], strict=True)
    require(not list(schedule.parameters()), "DDPM schedule has trainable parameters")

    ddpm.requires_grad_(False).eval().to(device)
    schedule.eval().to(device)
    require(not ddpm.training, "DDPM is not in eval mode")
    require(all_grads_none(ddpm), "DDPM has gradients immediately after load")
    del checkpoint
    return ddpm, schedule, {
        "optimizer_step": optimizer_step,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "parameter_count": parameter_count,
    }


def generator_state_snapshot(device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).clone()
    return torch.random.get_rng_state().clone()


def run_sampler(
    ddpm: torch.nn.Module,
    schedule: DDPMSchedule,
    predicted_z_quant: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    res_mask: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], float, bool]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    global_rng_before = generator_state_snapshot(device)
    sampled, seconds = timed_call(
        device,
        lambda: sample_residual_ddpm(
            model=ddpm,
            schedule=schedule,
            z_quant=predicted_z_quant,
            hidden_states=hidden_states,
            res_mask=res_mask,
            initial_noise=None,
            generator=generator,
            return_metadata=True,
        ),
    )
    r_hat, metadata = sampled
    global_rng_unchanged = torch.equal(
        global_rng_before, generator_state_snapshot(device)
    )
    return r_hat, metadata, seconds, global_rng_unchanged


def latent_diagnostics(
    r_hat: torch.Tensor,
    gt_residual: torch.Tensor,
    res_mask: torch.Tensor,
) -> dict[str, float]:
    require(r_hat.shape == gt_residual.shape, "GT residual shape mismatch")
    require(res_mask.shape == r_hat.shape[:2], "diagnostic mask shape mismatch")
    valid = res_mask.bool()
    require(bool(valid.any()), "diagnostic mask has no valid residues")
    predicted = r_hat.float()[valid]
    target = gt_residual.float()[valid]
    difference = predicted - target
    cosine = F.cosine_similarity(
        predicted.reshape(1, -1), target.reshape(1, -1), dim=-1
    )[0]
    return {
        "mean": float(predicted.mean()),
        "std": float(predicted.std(unbiased=False)),
        "min": float(predicted.min()),
        "max": float(predicted.max()),
        "l2_per_residue_mean": float(predicted.norm(dim=-1).mean()),
        "gt_mse": float(difference.square().mean()),
        "gt_cosine_similarity": float(cosine),
        "gt_l2_error_per_residue_mean": float(difference.norm(dim=-1).mean()),
    }


def main() -> int:
    args = parse_args()
    require(args.val_index >= 0, "--val_index must be non-negative")
    require(args.base_seed >= 0, "--base_seed must be non-negative")
    device = resolve_device(args.device)
    seed_everything(args.base_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset_dir = Path(args.dataset_dir)
    split_csv = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint_best.pt"
    sample = load_validation_sample(dataset_dir, split_csv, args.val_index)
    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype = sample["aatype"].long()
    gt_residual = sample["residual"].float().view(
        1, length, EXPECTED_CODEBOOK_DIM
    ).to(device)
    stored_mask = sample["res_mask"].float().view(1, length).to(device)
    require(bool(torch.isfinite(gt_residual).all()), "GT residual is non-finite")

    checks: dict[str, bool] = {}
    dplm, struct_tokenizer = load_models(args.dplm_checkpoint, device, checks)
    ddpm, schedule, checkpoint_metadata = load_ddpm(checkpoint_path, device)
    dplm_trainable = sum(
        parameter.numel() for parameter in dplm.parameters() if parameter.requires_grad
    )
    tokenizer_trainable = sum(
        parameter.numel()
        for parameter in struct_tokenizer.parameters()
        if parameter.requires_grad
    )
    require(dplm_trainable == 0, "DPLM trainable parameter count is not zero")
    require(tokenizer_trainable == 0, "tokenizer trainable parameter count is not zero")
    require(all_grads_none(dplm), "DPLM has gradients before inference")
    require(all_grads_none(struct_tokenizer), "tokenizer has gradients before inference")

    total_started = time.perf_counter()
    with torch.inference_mode():
        input_tokens, partial_mask = build_folding_input(dplm, aatype, device)
        generated, generation_seconds = timed_call(
            device,
            lambda: dplm.generate(
                input_tokens=input_tokens,
                max_iter=GENERATION_MAX_ITER,
                temperature=1.0,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax",
                partial_masks=partial_mask,
            ),
        )
        require("output_tokens" in generated, "generation returned no output_tokens")
        final_tokens = generated["output_tokens"]
        require(
            tuple(final_tokens.shape) == (1, 2 * (length + 2)),
            f"generated output token shape is {tuple(final_tokens.shape)}",
        )
        final_struct_tokens, _ = final_tokens.chunk(2, dim=1)
        generated_structure_tokens = final_struct_tokens[:, 1:-1]
        require(
            tuple(generated_structure_tokens.shape) == (1, length),
            "generated structure-token residue shape mismatch",
        )

        predicted_z_quant, res_mask = reconstruct_prediction_latent(
            dplm, final_tokens
        )
        expected_latent_shape = (1, length, EXPECTED_CODEBOOK_DIM)
        require(
            tuple(predicted_z_quant.shape) == expected_latent_shape,
            f"predicted z_quant shape is {tuple(predicted_z_quant.shape)}",
        )
        require(tuple(res_mask.shape) == (1, length), "predicted res_mask misaligned")
        require(bool(res_mask.bool().all()), "predicted sample contains padding residues")
        require(
            bool(torch.isfinite(predicted_z_quant).all()),
            "predicted z_quant is non-finite",
        )

        hidden_states, extra_forward_seconds = timed_call(
            device,
            lambda: extract_final_token_condition(dplm, final_tokens, length),
        )
        require(len(hidden_states) == EXPECTED_NUM_LAYERS, "expected 33 hidden states")
        for index, hidden in enumerate(hidden_states):
            require(
                tuple(hidden.shape) == (1, length, EXPECTED_HIDDEN_SIZE),
                f"hidden state {index} shape is {tuple(hidden.shape)}",
            )

        metric_mask = res_mask.float() * stored_mask
        require(bool(metric_mask.bool().all()), "stored/generated residue masks differ")
        sample_seed = args.base_seed + args.val_index
        r_hat, sampler_metadata, sampling_seconds, global_rng_unchanged = run_sampler(
            ddpm,
            schedule,
            predicted_z_quant,
            hidden_states,
            res_mask,
            sample_seed,
            device,
        )
        require(
            sampler_metadata["nfe"] == NUM_TIMESTEPS,
            f"expected NFE={NUM_TIMESTEPS}, got {sampler_metadata['nfe']}",
        )
        require(
            sampler_metadata["timesteps"] == tuple(range(NUM_TIMESTEPS, 0, -1)),
            "DDPM reverse timestep order mismatch",
        )
        require(tuple(r_hat.shape) == expected_latent_shape, "r_hat shape mismatch")
        require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")
        require(
            bool((r_hat * (1.0 - res_mask[..., None])).eq(0).all()),
            "r_hat padding is nonzero",
        )
        require(not r_hat.requires_grad and r_hat.grad_fn is None, "r_hat has a grad graph")
        require(global_rng_unchanged, "DDPM sampler changed global RNG with a Generator")

        z_refined_alpha1 = predicted_z_quant + r_hat
        require(
            tuple(z_refined_alpha1.shape) == expected_latent_shape,
            "z_refined_alpha1 shape mismatch",
        )
        require(
            bool(torch.isfinite(z_refined_alpha1).all()),
            "z_refined_alpha1 is non-finite",
        )

        repeat_allclose: bool | None = None
        repeat_exact: bool | None = None
        repeat_seconds = 0.0
        if not args.skip_repeatability_check:
            repeated, repeated_metadata, repeat_seconds, repeat_rng_unchanged = run_sampler(
                ddpm,
                schedule,
                predicted_z_quant,
                hidden_states,
                res_mask,
                sample_seed,
                device,
            )
            require(repeated_metadata["nfe"] == NUM_TIMESTEPS, "repeat NFE mismatch")
            repeat_exact = torch.equal(r_hat, repeated)
            repeat_allclose = torch.allclose(r_hat, repeated, atol=0.0, rtol=0.0)
            require(repeat_exact and repeat_allclose, "same-seed DDPM sampling differs")
            require(repeat_rng_unchanged, "repeat sampling changed global RNG")

        bit_only = refined_alpha1 = None
        if not args.skip_decode:
            bit_only = struct_tokenizer.detokenize(
                predicted_z_quant, res_mask=res_mask
            )
            refined_alpha1 = struct_tokenizer.detokenize(
                z_refined_alpha1, res_mask=res_mask
            )
            assert_atom37("Bit-only", bit_only, length)
            assert_atom37("DDPM alpha=1 diagnostic", refined_alpha1, length)

    synchronize(device)
    total_inference_seconds = time.perf_counter() - total_started
    require(all_grads_none(dplm), "DPLM has gradients after inference")
    require(all_grads_none(struct_tokenizer), "tokenizer has gradients after inference")
    require(all_grads_none(ddpm), "DDPM has gradients after sampling")
    require(not ddpm.training, "DDPM left eval mode")

    diagnostics = latent_diagnostics(r_hat, gt_residual, metric_mask)
    valid_quant = predicted_z_quant[res_mask.bool()].detach().float()

    print("\n[INTERNAL VALIDATION SAMPLE]")
    print(f"val_index: {args.val_index}")
    print(f"sample_id: {sample_id}")
    print(f"length: {length}")
    print("dataset role: internal AFDB validation only")
    print("CAMEO/PDB-date used: False")

    print("\n[DPLM PREDICTION-TIME CONDITION]")
    print(f"generation max_iter: {GENERATION_MAX_ITER}")
    print("unmasking strategy: deterministic")
    print("sampling strategy: argmax")
    print(f"final generated output token shape: {tuple(final_tokens.shape)}")
    print(f"generated structure token shape: {tuple(generated_structure_tokens.shape)}")
    print("extra final-token frozen forward: True")
    print("all hidden states count: 34 (validated by reused helper)")
    print(f"used Transformer hidden states: {len(hidden_states)}")
    print(f"aligned hidden shape: {tuple(hidden_states[-1].shape)}")
    print(f"predicted z_quant shape: {tuple(predicted_z_quant.shape)}")
    print(f"predicted z_quant finite: {bool(torch.isfinite(predicted_z_quant).all())}")
    print(f"predicted z_quant unique values: {torch.unique(valid_quant).cpu().tolist()}")
    print(
        "predicted z_quant value range: "
        f"[{valid_quant.min().item():.9g}, {valid_quant.max().item():.9g}]"
    )

    print("\n[DDPM CHECKPOINT]")
    print(f"path: {checkpoint_path}")
    print("state_dict strict load: True")
    print(f"optimizer_step: {checkpoint_metadata['optimizer_step']}")
    print(f"best_step: {checkpoint_metadata['best_step']}")
    print(f"best_val_loss: {checkpoint_metadata['best_val_loss']:.9f}")
    print(f"DDPM parameters: {checkpoint_metadata['parameter_count']}")
    print(f"DDPM eval mode: {not ddpm.training}")

    print("\n[FULL ANCESTRAL DDPM]")
    print(f"seed: {sample_seed}")
    print(f"reverse order: {NUM_TIMESTEPS} -> 1")
    print(f"NFE: {sampler_metadata['nfe']}")
    print(f"r_hat shape: {tuple(r_hat.shape)}")
    print(f"r_hat finite: {bool(torch.isfinite(r_hat).all())}")
    print(f"r_hat mean: {diagnostics['mean']:.9g}")
    print(f"r_hat std: {diagnostics['std']:.9g}")
    print(f"r_hat min: {diagnostics['min']:.9g}")
    print(f"r_hat max: {diagnostics['max']:.9g}")
    print(f"r_hat L2 norm per residue mean: {diagnostics['l2_per_residue_mean']:.9g}")
    print(f"global RNG unchanged by local Generator sampling: {global_rng_unchanged}")

    print("\n[GT RESIDUAL DIAGNOSTIC ONLY; NOT USED AS MODEL INPUT]")
    print(f"GT residual shape: {tuple(gt_residual.shape)}")
    print(f"MSE(r_hat, GT residual): {diagnostics['gt_mse']:.9g}")
    print(f"cosine similarity: {diagnostics['gt_cosine_similarity']:.9g}")
    print(
        "mean L2 error per residue: "
        f"{diagnostics['gt_l2_error_per_residue_mean']:.9g}"
    )

    print("\n[CANONICAL FULL-RESIDUAL INTERFACE DIAGNOSTIC]")
    print("DIAGNOSTIC ONLY")
    print("NOT SELECTED ALPHA")
    print("alpha: 1.0")
    print(f"z_refined_alpha1 shape: {tuple(z_refined_alpha1.shape)}")
    print(f"z_refined_alpha1 finite: {bool(torch.isfinite(z_refined_alpha1).all())}")
    print(f"continuous latent decode performed: {not args.skip_decode}")
    if bit_only is not None and refined_alpha1 is not None:
        print(f"Bit-only atom37 shape: {tuple(bit_only['atom37_positions'].shape)}")
        print(
            "DDPM alpha=1 diagnostic atom37 shape: "
            f"{tuple(refined_alpha1['atom37_positions'].shape)}"
        )

    print("\n[FREEZE AND REPEATABILITY]")
    print(f"DPLM trainable params: {dplm_trainable}")
    print(f"tokenizer trainable params: {tokenizer_trainable}")
    print(f"DPLM gradients absent: {all_grads_none(dplm)}")
    print(f"tokenizer gradients absent: {all_grads_none(struct_tokenizer)}")
    print(f"DDPM gradients absent: {all_grads_none(ddpm)}")
    print(f"DDPM output grad graph absent: {r_hat.grad_fn is None}")
    print(f"repeatability check performed: {not args.skip_repeatability_check}")
    if repeat_exact is not None and repeat_allclose is not None:
        print(f"same-seed r_hat exact: {repeat_exact}")
        print(f"same-seed r_hat allclose: {repeat_allclose}")

    print("\n[TIMING; DIAGNOSTIC ONLY, NOT AN OFFICIAL BENCHMARK]")
    print(f"DPLM generation seconds: {generation_seconds:.6f}")
    print(f"extra forward seconds: {extra_forward_seconds:.6f}")
    print(f"DDPM 1000-step sampling seconds: {sampling_seconds:.6f}")
    if not args.skip_repeatability_check:
        print(f"DDPM repeat sampling seconds: {repeat_seconds:.6f}")
    print(f"total inference seconds: {total_inference_seconds:.6f}")

    print("\nDDPM_INFERENCE_INTEGRATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

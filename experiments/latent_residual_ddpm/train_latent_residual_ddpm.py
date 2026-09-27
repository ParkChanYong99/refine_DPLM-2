#!/usr/bin/env python3
"""Train the Mode-A Local Matched Residual DDPM with frozen DPLM conditions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import traceback
from typing import Any, Sequence

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_ddpm.ddpm_training import ddpm_training_step
from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.train_latent_residual_fm import (
    DPLM_CHECKPOINT,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    METRIC_COLUMNS,
    Timings,
    atomic_torch_save,
    deterministic_subset,
    dplm_gradient_parameters,
    extract_hidden,
    gpu_peaks,
    gradients_finite,
    group_rows,
    iter_training_epoch,
    iter_validation_samples,
    layer_mix_metrics,
    load_split,
    make_scheduler,
    model_digest,
    open_metrics,
    read_metric_history,
    restore_rng_state,
    rng_state,
    sample_tensors,
    seed_everything,
    synchronize,
    validation_seed,
)


DEFAULT_DATASET_DIR = "data-bin/latent_residual_fm/afdb_l512_sharded"
DEFAULT_SPLIT_CSV = (
    "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv"
)
DEFAULT_OUTPUT_DIR = "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main"
MATCHED_DDPM_SPEC_SHA256 = (
    "cb81f2932e6690528b764619a56fa62176f0adfb93a28eb6076ddb7336972a31"
)
EXPECTED_DDPM_PARAMETERS = 34_946_606
PHYSICAL_BATCH_SIZE = 1
NUM_TIMESTEPS = 1_000
BETA_START = 1e-4
BETA_END = 2e-2
PREDICTION_TYPE = "epsilon"
CHECKPOINT_SELECTION_METRIC = "deterministic_internal_val_masked_epsilon_mse"
TRAINING_TIMESTEP_POLICY = "independent Uniform{1,...,1000} per training sample"
TRAINING_NOISE_POLICY = "independent Gaussian N(0,I) per training sample"
VALIDATION_POLICY = {
    "name": "stable_sample_identity_sha256_v1",
    "base_seed": 42,
    "identity": "sample_id",
    "seed_derivation": "SHA256(f'{base_seed}:{sample_id}') first 8 bytes little-endian, 63-bit",
    "draw_order": "one Uniform{1,...,T} timestep, then one Gaussian epsilon tensor",
    "global_rng_unchanged": True,
}
DEFAULT_WANDB_TAGS = (
    "latent-residual",
    "matched-ddpm",
    "modeA",
    "current-token",
    "dplm2-bit-650m",
    "afdb",
    "l512",
)
RESUME_MATCH_FIELDS = (
    "dataset_dir",
    "split_csv",
    "seed",
    "lr",
    "final_lr",
    "weight_decay",
    "warmup_steps",
    "max_optimizer_steps",
    "grad_accum_steps",
    "grad_clip",
    "val_every",
    "save_every",
    "log_every",
    "train_max_samples",
    "val_max_samples",
)


def primary_protocol() -> dict[str, Any]:
    return {
        "dataset_dir": DEFAULT_DATASET_DIR,
        "split_csv": DEFAULT_SPLIT_CSV,
        "train_samples": EXPECTED_TRAIN,
        "val_samples": EXPECTED_VAL,
        "seed": 42,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "grad_accum_steps": 8,
        "effective_batch_size": 8,
        "max_optimizer_steps": 100_000,
        "warmup_steps": 2_000,
        "lr": 1e-4,
        "final_lr": 1e-5,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "log_every": 100,
        "val_every": 5_000,
        "save_every": 5_000,
        "num_timesteps": NUM_TIMESTEPS,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "prediction_type": PREDICTION_TYPE,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--max_optimizer_steps", type=int, default=100_000)
    parser.add_argument("--stop_after_optimizer_steps", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_every", type=int, default=5_000)
    parser.add_argument("--val_max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", default="bumgye99-konkuk-university")
    parser.add_argument("--wandb_project", default="dplm2-ddpm-refiner")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("lr", "final_lr", "weight_decay", "grad_clip"):
        require(getattr(args, name) >= 0, f"--{name} must be non-negative")
    for name in (
        "max_optimizer_steps",
        "grad_accum_steps",
        "val_every",
        "log_every",
        "save_every",
    ):
        require(getattr(args, name) > 0, f"--{name} must be positive")
    require(args.lr > 0 and args.final_lr > 0, "learning rates must be positive")
    require(args.final_lr <= args.lr, "--final_lr must not exceed --lr")
    require(args.warmup_steps >= 0, "--warmup_steps must be non-negative")
    require(
        args.warmup_steps <= args.max_optimizer_steps,
        "--warmup_steps must not exceed --max_optimizer_steps",
    )
    if args.stop_after_optimizer_steps is not None:
        require(
            0 < args.stop_after_optimizer_steps <= args.max_optimizer_steps,
            "--stop_after_optimizer_steps must be in [1,max_optimizer_steps]",
        )
    for name in ("train_max_samples", "val_max_samples"):
        value = getattr(args, name)
        require(value is None or value > 0, f"--{name} must be positive when set")


def schedule_configuration() -> dict[str, Any]:
    return {
        "num_timesteps": NUM_TIMESTEPS,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "schedule": "linear",
        "prediction_type": PREDICTION_TYPE,
    }


def architecture_configuration(model: LatentResidualDDPM) -> dict[str, Any]:
    return {
        "class": "LatentResidualDDPM",
        "residual_dim": model.residual_dim,
        "condition_dim": model.condition_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_hidden_states": model.num_hidden_states,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def deterministic_validation_corruption(
    residual: torch.Tensor,
    schedule: DDPMSchedule,
    base_seed: int,
    sample_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable per-sample validation timestep/noise without global RNG use.

    Policy: derive a local seed with the finalized FM SHA256 sample-identity
    policy, then draw one discrete timestep followed by the epsilon tensor.
    """
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(validation_seed(base_seed, sample_id))
    timesteps = torch.randint(
        low=1,
        high=schedule.num_timesteps + 1,
        size=(residual.shape[0],),
        dtype=torch.long,
        device=residual.device,
        generator=generator,
    )
    noise = torch.randn(
        residual.shape,
        dtype=residual.dtype,
        device=residual.device,
        generator=generator,
    )
    return timesteps, noise


def validate(
    model: LatentResidualDDPM,
    schedule: DDPMSchedule,
    extractor: DPLMConditionExtractor,
    grouped: dict[str, list[Any]],
    dataset_dir: Path,
    seed: int,
    timings: Timings,
    device: torch.device,
) -> tuple[float, int, float]:
    started = time.perf_counter()
    model.eval()
    extractor.eval()
    total_loss = 0.0
    samples = 0
    with torch.no_grad():
        for row, sample in iter_validation_samples(
            grouped, dataset_dir, timings, device
        ):
            aatype, struct_ids, z_quant, residual, res_mask = sample_tensors(
                sample, device
            )
            hidden = extract_hidden(
                extractor, aatype, struct_ids, res_mask, timings, device
            )
            timesteps, noise = deterministic_validation_corruption(
                residual, schedule, seed, row.sample_id
            )
            result = ddpm_training_step(
                model=model,
                schedule=schedule,
                x0=residual,
                z_quant=z_quant,
                hidden_states=hidden,
                res_mask=res_mask,
                timesteps=timesteps,
                noise=noise,
            )
            loss = result["loss"]
            require(bool(torch.isfinite(loss)), "validation loss is non-finite")
            total_loss += float(loss)
            samples += 1
            del aatype, struct_ids, z_quant, residual, res_mask
            del hidden, timesteps, noise, result, loss
    require(samples > 0, "validation set is empty")
    synchronize(device)
    elapsed = time.perf_counter() - started
    timings.validation += elapsed
    model.train()
    return total_loss / samples, samples, elapsed


def checkpoint_payload(
    model: LatentResidualDDPM,
    schedule: DDPMSchedule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    optimizer_step: int,
    micro_step: int,
    epoch: int,
    epoch_position: int,
    samples_seen: int,
    best_val_loss: float,
    best_step: int,
    initial_ddpm_digest: str,
    wandb_run_id: str | None,
) -> dict[str, Any]:
    payload = {
        "ddpm_model_state_dict": model.state_dict(),
        "ddpm_schedule_state_dict": schedule.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "epoch": epoch,
        "epoch_position": epoch_position,
        "samples_seen": samples_seen,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "training_configuration": vars(args).copy(),
        "ddpm_schedule_configuration": schedule_configuration(),
        "ddpm_architecture_configuration": architecture_configuration(model),
        "validation_policy": VALIDATION_POLICY.copy(),
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "matched_ddpm_spec_sha256": MATCHED_DDPM_SPEC_SHA256,
        "split_path": args.split_csv,
        "dataset_path": args.dataset_dir,
        "dplm_checkpoint_identifier": DPLM_CHECKPOINT,
        "initial_ddpm_digest": initial_ddpm_digest,
    }
    if wandb_run_id is not None:
        payload["wandb_run_id"] = wandb_run_id
    payload.update(rng_state())
    return payload


def guard_resume_compatibility(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    model: LatentResidualDDPM,
    schedule: DDPMSchedule,
) -> None:
    require(
        checkpoint["dplm_checkpoint_identifier"] == DPLM_CHECKPOINT,
        "resume DPLM checkpoint mismatch",
    )
    require(checkpoint["split_path"] == args.split_csv, "resume split mismatch")
    require(checkpoint["dataset_path"] == args.dataset_dir, "resume dataset mismatch")
    stored = checkpoint.get("training_configuration", {})
    if stored.get("max_optimizer_steps") != args.max_optimizer_steps:
        raise RuntimeError(
            "scheduler horizon mismatch: "
            f"checkpoint max_optimizer_steps={stored.get('max_optimizer_steps')}, "
            f"current max_optimizer_steps={args.max_optimizer_steps}"
        )
    for field in RESUME_MATCH_FIELDS:
        require(
            stored.get(field) == getattr(args, field),
            f"resume training configuration mismatch: {field}",
        )
    require(
        checkpoint.get("ddpm_schedule_configuration") == schedule_configuration(),
        "resume DDPM schedule/prediction mismatch",
    )
    require(
        checkpoint.get("ddpm_architecture_configuration")
        == architecture_configuration(model),
        "resume DDPM architecture mismatch",
    )
    require(
        checkpoint.get("validation_policy") == VALIDATION_POLICY,
        "resume deterministic validation policy mismatch",
    )
    require(
        checkpoint.get("checkpoint_selection_metric")
        == CHECKPOINT_SELECTION_METRIC,
        "resume checkpoint-selection metric mismatch",
    )
    require(
        checkpoint.get("matched_ddpm_spec_sha256") == MATCHED_DDPM_SPEC_SHA256,
        "resume frozen specification digest mismatch",
    )
    stored_schedule = checkpoint.get("ddpm_schedule_state_dict", {})
    current_schedule = schedule.state_dict()
    require(
        set(stored_schedule) == set(current_schedule),
        "resume DDPM schedule buffer keys mismatch",
    )
    for name, current in current_schedule.items():
        saved = stored_schedule[name]
        require(
            torch.equal(saved.detach().cpu(), current.detach().cpu()),
            f"resume DDPM schedule coefficient mismatch: {name}",
        )


def resolved_wandb_tags(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys((*DEFAULT_WANDB_TAGS, *(args.wandb_tags or ()))))


def wandb_configuration(
    args: argparse.Namespace,
    train_samples: int,
    val_samples: int,
    extractor: DPLMConditionExtractor,
    model: LatentResidualDDPM,
) -> dict[str, Any]:
    return {
        "EXPERIMENT": {
            "experiment": "latent_residual_ddpm",
            "method": "Local Matched Residual DDPM",
            "mode": "Mode-A Current",
            "residual_definition": "z_cont_current - z_quant_current",
            "prediction_type": PREDICTION_TYPE,
            "spec_sha256": MATCHED_DDPM_SPEC_SHA256,
            "DPLM_checkpoint": DPLM_CHECKPOINT,
            "DPLM_frozen": True,
            "hidden_layers": extractor.num_transformer_layers,
            "hidden_dim": extractor.hidden_size,
            "hidden_modality": "structure",
            "hidden_embedding_state": False,
        },
        "DATA": {
            "dataset_dir": args.dataset_dir,
            "split_csv": args.split_csv,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_max_samples": args.train_max_samples,
            "val_max_samples": args.val_max_samples,
            "seed": args.seed,
            "max_length": 512,
        },
        "DDPM_MODEL": architecture_configuration(model),
        "DDPM_SCHEDULE": schedule_configuration(),
        "TRAINING": {
            "physical_batch_size": PHYSICAL_BATCH_SIZE,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.grad_accum_steps,
            "lr": args.lr,
            "final_lr": args.final_lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_optimizer_steps": args.max_optimizer_steps,
            "grad_clip": args.grad_clip,
            "val_every": args.val_every,
            "save_every": args.save_every,
            "log_every": args.log_every,
            "training_timestep": TRAINING_TIMESTEP_POLICY,
            "training_noise": TRAINING_NOISE_POLICY,
            "objective": "masked epsilon MSE",
        },
        "VALIDATION_POLICY": VALIDATION_POLICY.copy(),
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
    }


def initialize_wandb(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    stored_run_id: str | None,
) -> Any:
    import wandb

    run_name = args.wandb_run_name or (
        f"latentDDPM-modeA-current-seed{args.seed}-"
        f"{args.max_optimizer_steps}steps"
    )
    kwargs: dict[str, Any] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": run_name,
        "tags": resolved_wandb_tags(args),
        "notes": args.wandb_notes,
        "mode": args.wandb_mode,
        "config": configuration,
    }
    if stored_run_id is not None:
        kwargs.update({"id": stored_run_id, "resume": "must"})
    return wandb.init(**kwargs)


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LOCAL MATCHED RESIDUAL DDPM TRAINING")
    print("=" * 60)
    print(f"train available: {summary['train_available']}")
    print(f"val available: {summary['val_available']}")
    print(f"train subset: {summary['train_subset']}")
    print(f"val subset: {summary['val_subset']}")
    print(f"DPLM trainable params: {summary['dplm_trainable_params']}")
    print(f"Tokenizer trainable params: {summary['tokenizer_trainable_params']}")
    print(f"DDPM trainable params: {summary['ddpm_trainable_params']}")
    print(f"optimizer steps: {summary['optimizer_steps']}")
    print(f"micro steps: {summary['micro_steps']}")
    print(f"samples seen: {summary['samples_seen']}")
    print(f"initial train loss: {summary['initial_train_loss']:.9f}")
    print(f"final train loss: {summary['final_train_loss']:.9f}")
    print(f"best val loss: {summary['best_val_loss']:.9f}")
    print(f"best step: {summary['best_step']}")
    print(f"DPLM gradient params: {summary['dplm_gradient_params']}")
    print(f"DDPM parameters changed: {summary['ddpm_parameters_changed']}")
    print(f"checkpoint_last exists: {summary['last_exists']}")
    print(f"checkpoint_best exists: {summary['best_exists']}")
    print(f"metrics exists: {summary['metrics_exists']}")
    print(f"DPLM condition seconds: {summary['timings'].dplm_condition:.6f}")
    print(f"DDPM forward/backward seconds: {summary['timings'].fm_forward_backward:.6f}")
    print(f"optimizer seconds: {summary['timings'].optimizer:.6f}")
    print(f"validation seconds: {summary['timings'].validation:.6f}")
    print(f"shard load seconds: {summary['timings'].shard_load:.6f}")
    print(f"total seconds: {summary['total_seconds']:.6f}")
    print(f"samples/sec: {summary['samples_per_second']:.9f}")
    print(f"GPU peak allocated MiB: {summary['peak_allocated_mib']:.6f}")
    print(f"GPU peak reserved MiB: {summary['peak_reserved_mib']:.6f}")
    print(f"FINAL STATUS: {summary['final_status']}")
    print("=" * 60)


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats(device)

    dataset_dir = Path(args.dataset_dir)
    split_path = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "checkpoint_best.pt"
    metrics_path = output_dir / "metrics.csv"
    timings = Timings()
    summary: dict[str, Any] = {
        "train_available": 0,
        "val_available": 0,
        "train_subset": 0,
        "val_subset": 0,
        "dplm_trainable_params": -1,
        "tokenizer_trainable_params": -1,
        "ddpm_trainable_params": 0,
        "optimizer_steps": 0,
        "micro_steps": 0,
        "samples_seen": 0,
        "initial_train_loss": float("nan"),
        "final_train_loss": float("nan"),
        "best_val_loss": float("inf"),
        "best_step": -1,
        "dplm_gradient_params": -1,
        "ddpm_parameters_changed": False,
        "last_exists": False,
        "best_exists": False,
        "metrics_exists": False,
        "timings": timings,
        "total_seconds": 0.0,
        "samples_per_second": 0.0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0,
        "final_status": "TRAINING_FAILED",
    }
    metrics_handle = None
    wandb_run = None
    wandb_run_id: str | None = None
    try:
        train_all, val_all = load_split(split_path)
        summary["train_available"] = len(train_all)
        summary["val_available"] = len(val_all)
        train_rows = deterministic_subset(train_all, args.train_max_samples, args.seed)
        val_rows = list(val_all[: args.val_max_samples]) if args.val_max_samples else val_all
        require(train_rows and val_rows, "train or validation subset is empty")
        summary["train_subset"] = len(train_rows)
        summary["val_subset"] = len(val_rows)
        train_groups = group_rows(train_rows)
        val_groups = group_rows(val_rows)

        extractor = DPLMConditionExtractor(DPLM_CHECKPOINT, device)
        extractor.eval().requires_grad_(False)
        tokenizer = extractor.model.struct_tokenizer
        tokenizer.eval().requires_grad_(False)
        summary["dplm_trainable_params"] = sum(
            parameter.numel()
            for parameter in extractor.model.parameters()
            if parameter.requires_grad
        )
        summary["tokenizer_trainable_params"] = sum(
            parameter.numel()
            for parameter in tokenizer.parameters()
            if parameter.requires_grad
        )
        require(summary["dplm_trainable_params"] == 0, "DPLM is not frozen")
        require(summary["tokenizer_trainable_params"] == 0, "tokenizer is not frozen")

        schedule = DDPMSchedule(
            num_timesteps=NUM_TIMESTEPS,
            beta_start=BETA_START,
            beta_end=BETA_END,
        ).to(device)
        require(not list(schedule.parameters()), "DDPM schedule has trainable parameters")
        model = LatentResidualDDPM().to(device)
        model.train()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        summary["ddpm_trainable_params"] = sum(
            parameter.numel() for parameter in parameters
        )
        require(
            summary["ddpm_trainable_params"] == EXPECTED_DDPM_PARAMETERS,
            "DDPM parameter count mismatch",
        )
        frozen_ids = {id(parameter) for parameter in extractor.model.parameters()}
        require(
            all(id(parameter) not in frozen_ids for parameter in parameters),
            "a frozen DPLM/tokenizer parameter entered the optimizer set",
        )
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = make_scheduler(
            optimizer,
            args.lr,
            args.final_lr,
            args.warmup_steps,
            args.max_optimizer_steps,
        )
        initial_ddpm_digest = model_digest(model)
        optimizer_step = micro_step = epoch = epoch_position = samples_seen = 0
        best_val_loss = float("inf")
        best_step = -1
        checkpoint: dict[str, Any] | None = None
        process_stop_step = (
            args.max_optimizer_steps
            if args.stop_after_optimizer_steps is None
            else args.stop_after_optimizer_steps
        )

        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location=device)
            guard_resume_compatibility(checkpoint, args, model, schedule)
            checkpoint_lr = float(
                checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
            )
            model.load_state_dict(checkpoint["ddpm_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            current_lr = float(optimizer.param_groups[0]["lr"])
            optimizer_step = int(checkpoint["optimizer_step"])
            micro_step = int(checkpoint["micro_step"])
            epoch = int(checkpoint["epoch"])
            epoch_position = int(checkpoint["epoch_position"])
            samples_seen = int(checkpoint["samples_seen"])
            best_val_loss = float(checkpoint["best_val_loss"])
            best_step = int(checkpoint["best_step"])
            initial_ddpm_digest = str(checkpoint["initial_ddpm_digest"])
            restore_rng_state(checkpoint)
            print("[RESUME]")
            print(f"optimizer_step={optimizer_step}")
            print(f"checkpoint_lr={checkpoint_lr:.17g}")
            print(f"current_lr={current_lr:.17g}")
            print(f"max_optimizer_steps={args.max_optimizer_steps}")
            if abs(checkpoint_lr - current_lr) > 1e-12:
                raise RuntimeError("resume optimizer LR mismatch")
            require(
                optimizer_step < process_stop_step,
                "resume checkpoint already reached the requested stop step",
            )

        if args.wandb:
            saved_rng_state = rng_state()
            stored_run_id = (
                checkpoint.get("wandb_run_id") if checkpoint is not None else None
            )
            wandb_run = initialize_wandb(
                args,
                wandb_configuration(
                    args, len(train_all), len(val_all), extractor, model
                ),
                stored_run_id,
            )
            restore_rng_state(saved_rng_state)
            wandb_run_id = str(wandb_run.id)
            wandb_run.summary["checkpoint_selection_metric"] = (
                CHECKPOINT_SELECTION_METRIC
            )

        initial_train_loss: float | None = None
        final_train_loss = float("nan")
        val_losses: dict[int, float] = {}
        if args.resume is not None:
            initial_train_loss, final_train_loss, val_losses = read_metric_history(
                metrics_path, optimizer_step
            )
        metrics_handle, metrics_writer = open_metrics(
            metrics_path, resume=args.resume is not None
        )
        optimizer.zero_grad(set_to_none=True)

        while optimizer_step < process_stop_step:
            accumulation_count = 0
            accumulated_loss = 0.0
            epoch_finished = True
            for row, sample in iter_training_epoch(
                train_groups,
                dataset_dir,
                args.seed,
                epoch,
                epoch_position,
                timings,
                device,
            ):
                epoch_finished = False
                aatype, struct_ids, z_quant, residual, res_mask = sample_tensors(
                    sample, device
                )
                hidden = extract_hidden(
                    extractor, aatype, struct_ids, res_mask, timings, device
                )
                synchronize(device)
                ddpm_started = time.perf_counter()
                result = ddpm_training_step(
                    model=model,
                    schedule=schedule,
                    x0=residual,
                    z_quant=z_quant,
                    hidden_states=hidden,
                    res_mask=res_mask,
                )
                loss = result["loss"]
                require(bool(torch.isfinite(loss)), "training loss is non-finite")
                loss.backward()
                synchronize(device)
                timings.fm_forward_backward += time.perf_counter() - ddpm_started
                require(gradients_finite(parameters), "DDPM gradient contains NaN/Inf")
                require(
                    dplm_gradient_parameters(extractor) == 0,
                    "a frozen DPLM/tokenizer parameter received a gradient",
                )
                accumulation_count += 1
                accumulated_loss += float(loss.detach())
                micro_step += 1
                samples_seen += 1
                epoch_position += 1
                del aatype, struct_ids, z_quant, residual, res_mask
                del hidden, result, loss

                at_epoch_end = epoch_position == len(train_rows)
                should_update = (
                    accumulation_count == args.grad_accum_steps or at_epoch_end
                )
                if not should_update:
                    continue

                synchronize(device)
                optimizer_started = time.perf_counter()
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulation_count)
                require(
                    gradients_finite(parameters),
                    "averaged DDPM gradient contains NaN/Inf",
                )
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    parameters, args.grad_clip
                )
                require(bool(torch.isfinite(grad_norm_tensor)), "gradient norm is non-finite")
                grad_norm = float(grad_norm_tensor)
                used_lr = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                synchronize(device)
                timings.optimizer += time.perf_counter() - optimizer_started
                optimizer_step += 1
                train_loss = accumulated_loss / accumulation_count
                if initial_train_loss is None:
                    initial_train_loss = train_loss
                final_train_loss = train_loss
                accumulation_count = 0
                accumulated_loss = 0.0

                val_loss: float | None = None
                improved = False
                wandb_metrics: dict[str, Any] = {}
                reached_requested_stop = (
                    args.stop_after_optimizer_steps is not None
                    and optimizer_step == process_stop_step
                )
                if optimizer_step % args.val_every == 0 or reached_requested_stop:
                    val_loss, val_samples, val_elapsed = validate(
                        model,
                        schedule,
                        extractor,
                        val_groups,
                        dataset_dir,
                        int(VALIDATION_POLICY["base_seed"]),
                        timings,
                        device,
                    )
                    require(math.isfinite(val_loss), "validation loss is non-finite")
                    val_losses[optimizer_step] = val_loss
                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        best_step = optimizer_step
                    print(
                        f"[VAL] step={optimizer_step} samples={val_samples} "
                        f"loss={val_loss:.9f} best={best_val_loss:.9f} "
                        f"time={val_elapsed:.3f}"
                    )
                    if wandb_run is not None:
                        wandb_metrics.update(
                            {
                                "val/loss": val_loss,
                                "val/best_loss": best_val_loss,
                                "val/best_step": best_step,
                                "val/num_samples": val_samples,
                                "val/time_sec": val_elapsed,
                                "checkpoint/is_best": improved,
                            }
                        )
                        wandb_metrics.update(layer_mix_metrics(model))
                        wandb_run.summary["best_val_loss"] = best_val_loss
                        wandb_run.summary["best_step"] = best_step

                allocated, reserved = gpu_peaks(device)
                metrics_writer.writerow(
                    {
                        "wall_time_sec": f"{time.perf_counter() - total_started:.6f}",
                        "optimizer_step": optimizer_step,
                        "micro_step": micro_step,
                        "epoch": epoch,
                        "train_loss": f"{train_loss:.9f}",
                        "learning_rate": f"{used_lr:.12g}",
                        "grad_norm": f"{grad_norm:.9f}",
                        "val_loss": "" if val_loss is None else f"{val_loss:.9f}",
                        "samples_seen": samples_seen,
                        "gpu_peak_allocated_mib": f"{allocated:.6f}",
                        "gpu_peak_reserved_mib": f"{reserved:.6f}",
                    }
                )
                metrics_handle.flush()

                if optimizer_step % args.log_every == 0:
                    print(
                        f"[TRAIN] step={optimizer_step} epoch={epoch} "
                        f"loss={train_loss:.9f} lr={used_lr:.9g} "
                        f"grad_norm={grad_norm:.9f} samples_seen={samples_seen}"
                    )
                    require(
                        dplm_gradient_parameters(extractor) == 0,
                        "periodic DPLM gradient audit failed",
                    )
                    if wandb_run is not None:
                        elapsed = time.perf_counter() - total_started
                        wandb_metrics.update(
                            {
                                "train/loss": train_loss,
                                "train/lr": used_lr,
                                "train/learning_rate": used_lr,
                                "train/grad_norm": grad_norm,
                                "optimizer_step": optimizer_step,
                                "micro_step": micro_step,
                                "epoch": epoch,
                                "train/optimizer_step": optimizer_step,
                                "train/micro_step": micro_step,
                                "train/epoch": epoch,
                                "train/samples_seen": samples_seen,
                                "performance/samples_per_sec": samples_seen / elapsed,
                                "gpu/peak_allocated_mib": allocated,
                                "gpu/peak_reserved_mib": reserved,
                            }
                        )

                if wandb_run is not None and wandb_metrics:
                    wandb_metrics.update(
                        {
                            "timing/dplm_condition_sec": timings.dplm_condition,
                            "timing/ddpm_forward_backward_sec": timings.fm_forward_backward,
                            "timing/optimizer_sec": timings.optimizer,
                            "timing/shard_load_sec": timings.shard_load,
                            "timing/validation_sec": timings.validation,
                            "timing/total_sec": time.perf_counter() - total_started,
                        }
                    )
                    wandb_run.log(wandb_metrics, step=optimizer_step)

                should_save = (
                    optimizer_step % args.save_every == 0
                    or optimizer_step == args.max_optimizer_steps
                    or reached_requested_stop
                )
                if should_save or improved:
                    payload = checkpoint_payload(
                        model,
                        schedule,
                        optimizer,
                        scheduler,
                        args,
                        optimizer_step,
                        micro_step,
                        epoch,
                        epoch_position,
                        samples_seen,
                        best_val_loss,
                        best_step,
                        initial_ddpm_digest,
                        wandb_run_id,
                    )
                    if should_save:
                        atomic_torch_save(payload, last_path)
                    if improved:
                        atomic_torch_save(payload, best_path)
                    print(
                        f"[SAVE] last={should_save and last_path.is_file()} "
                        f"best={best_path.is_file()}"
                    )
                    del payload

                if optimizer_step >= process_stop_step:
                    epoch_finished = False
                    break
                if at_epoch_end:
                    epoch_finished = True
                    break

            if optimizer_step >= process_stop_step:
                break
            require(epoch_finished or epoch_position == len(train_rows), "epoch ended early")
            require(epoch_position == len(train_rows), "epoch position is inconsistent")
            epoch += 1
            epoch_position = 0

        require(initial_train_loss is not None, "no optimizer step was executed")
        summary["optimizer_steps"] = optimizer_step
        summary["micro_steps"] = micro_step
        summary["samples_seen"] = samples_seen
        summary["initial_train_loss"] = initial_train_loss
        summary["final_train_loss"] = final_train_loss
        summary["best_val_loss"] = best_val_loss
        summary["best_step"] = best_step
        summary["dplm_gradient_params"] = dplm_gradient_parameters(extractor)
        summary["ddpm_parameters_changed"] = model_digest(model) != initial_ddpm_digest
        summary["last_exists"] = last_path.is_file()
        summary["best_exists"] = best_path.is_file()
        summary["metrics_exists"] = metrics_path.is_file()
        passed = (
            optimizer_step == process_stop_step
            and math.isfinite(initial_train_loss)
            and math.isfinite(final_train_loss)
            and math.isfinite(best_val_loss)
            and best_step > 0
            and summary["last_exists"]
            and summary["best_exists"]
            and summary["metrics_exists"]
            and summary["dplm_gradient_params"] == 0
            and summary["ddpm_parameters_changed"]
        )
        require(passed, "one or more training completion criteria failed")
        summary["final_status"] = "DDPM_TRAINING_COMPLETE"
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        if wandb_run is not None:
            wandb_run.finish()
        summary["total_seconds"] = time.perf_counter() - total_started
        summary["samples_per_second"] = (
            summary["samples_seen"] / summary["total_seconds"]
            if summary["total_seconds"] > 0
            else 0.0
        )
        allocated, reserved = gpu_peaks(device)
        summary["peak_allocated_mib"] = allocated
        summary["peak_reserved_mib"] = reserved
        print_summary(summary)
    return 0 if summary["final_status"] == "DDPM_TRAINING_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

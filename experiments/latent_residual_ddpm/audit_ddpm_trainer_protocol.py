#!/usr/bin/env python3
"""Read-only protocol audit for the Local Matched Residual DDPM trainer."""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import traceback

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_ddpm.train_latent_residual_ddpm import (
    BETA_END,
    BETA_START,
    CHECKPOINT_SELECTION_METRIC,
    DEFAULT_DATASET_DIR,
    DEFAULT_SPLIT_CSV,
    EXPECTED_DDPM_PARAMETERS,
    MATCHED_DDPM_SPEC_SHA256,
    NUM_TIMESTEPS,
    PHYSICAL_BATCH_SIZE,
    PREDICTION_TYPE,
    TRAINING_NOISE_POLICY,
    TRAINING_TIMESTEP_POLICY,
    VALIDATION_POLICY,
    deterministic_validation_corruption,
    parse_args,
    primary_protocol,
)
from experiments.latent_residual_fm.fm_model import LatentResidualFM
from experiments.latent_residual_fm.train_latent_residual_fm import (
    DPLM_CHECKPOINT,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    load_split,
)


FINAL_FM_CHECKPOINT = Path(
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/checkpoint_best.pt"
)
TRAINER_PATH = Path(
    "experiments/latent_residual_ddpm/train_latent_residual_ddpm.py"
)
SPEC_PATH = Path("experiments/latent_residual_ddpm/MATCHED_DDPM_SPEC.md")
FM_SHARED_FIELDS = (
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def run() -> None:
    protocol = primary_protocol()
    trainer_defaults = vars(parse_args([]))
    require(SPEC_PATH.is_file(), "frozen specification is missing")
    actual_spec_digest = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    require(
        actual_spec_digest == MATCHED_DDPM_SPEC_SHA256,
        "frozen specification SHA-256 mismatch",
    )
    require(Path(DEFAULT_DATASET_DIR).is_dir(), "dataset directory is missing")
    require(Path(DEFAULT_SPLIT_CSV).is_file(), "split CSV is missing")
    train_rows, val_rows = load_split(Path(DEFAULT_SPLIT_CSV))
    require(len(train_rows) == EXPECTED_TRAIN, "train count mismatch")
    require(len(val_rows) == EXPECTED_VAL, "validation count mismatch")

    require(FINAL_FM_CHECKPOINT.is_file(), "finalized FM checkpoint is missing")
    fm_checkpoint = torch.load(
        str(FINAL_FM_CHECKPOINT),
        map_location="cpu",
        mmap=True,
    )
    fm_configuration = fm_checkpoint.get("training_configuration", {})
    expected_fm = {
        "dataset_dir": DEFAULT_DATASET_DIR,
        "split_csv": DEFAULT_SPLIT_CSV,
        "seed": protocol["seed"],
        "lr": 1e-4,
        "final_lr": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 2_000,
        "max_optimizer_steps": 100_000,
        "grad_accum_steps": 8,
        "grad_clip": 1.0,
        "val_every": 5_000,
        "save_every": 5_000,
        "log_every": 100,
        "train_max_samples": None,
        "val_max_samples": None,
    }
    for field in FM_SHARED_FIELDS:
        require(
            fm_configuration.get(field) == expected_fm[field],
            f"finalized FM checkpoint protocol mismatch: {field}",
        )

    shared_protocol_mapping = {
        "dataset_dir": protocol["dataset_dir"],
        "split_csv": protocol["split_csv"],
        "seed": 42,
        "lr": protocol["lr"],
        "final_lr": protocol["final_lr"],
        "weight_decay": protocol["weight_decay"],
        "warmup_steps": protocol["warmup_steps"],
        "max_optimizer_steps": protocol["max_optimizer_steps"],
        "grad_accum_steps": protocol["grad_accum_steps"],
        "grad_clip": protocol["grad_clip"],
        "val_every": protocol["val_every"],
        "save_every": protocol["save_every"],
        "log_every": protocol["log_every"],
        "train_max_samples": None,
        "val_max_samples": None,
    }
    for field in FM_SHARED_FIELDS:
        require(
            shared_protocol_mapping[field] == fm_configuration[field],
            f"DDPM/FM shared setting mismatch: {field}",
        )
        require(
            trainer_defaults[field] == shared_protocol_mapping[field],
            f"DDPM trainer CLI default mismatch: {field}",
        )

    torch.manual_seed(42)
    fm = LatentResidualFM()
    torch.manual_seed(42)
    ddpm = LatentResidualDDPM()
    fm_total, fm_trainable = parameter_counts(fm)
    ddpm_total, ddpm_trainable = parameter_counts(ddpm)
    require(
        fm_total == fm_trainable == EXPECTED_DDPM_PARAMETERS,
        "finalized FM parameter count mismatch",
    )
    require(
        ddpm_total == ddpm_trainable == EXPECTED_DDPM_PARAMETERS,
        "DDPM parameter count mismatch",
    )
    fm_topology = [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in fm.named_parameters()
    ]
    ddpm_topology = [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in ddpm.named_parameters()
    ]
    require(fm_topology == ddpm_topology, "FM/DDPM parameter topology mismatch")

    schedule = DDPMSchedule(
        num_timesteps=NUM_TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
    )
    require(not list(schedule.parameters()), "DDPM schedule is trainable")
    require(schedule.betas.numel() == NUM_TIMESTEPS, "DDPM T mismatch")
    require(
        math.isclose(float(schedule.betas[0]), BETA_START, rel_tol=0.0, abs_tol=1e-15),
        "beta_1 mismatch",
    )
    require(
        math.isclose(float(schedule.betas[-1]), BETA_END, rel_tol=0.0, abs_tol=1e-15),
        "beta_T mismatch",
    )

    residual = torch.zeros(1, 7, 13)
    global_rng_before = torch.get_rng_state().clone()
    timestep_a, noise_a = deterministic_validation_corruption(
        residual, schedule, VALIDATION_POLICY["base_seed"], "audit-sample"
    )
    timestep_b, noise_b = deterministic_validation_corruption(
        residual, schedule, VALIDATION_POLICY["base_seed"], "audit-sample"
    )
    global_rng_after = torch.get_rng_state()
    require(torch.equal(timestep_a, timestep_b), "validation timestep is not stable")
    require(torch.equal(noise_a, noise_b), "validation noise is not stable")
    require(
        torch.equal(global_rng_before, global_rng_after),
        "validation corruption changed global RNG",
    )

    trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
    trainer_text = trainer_source.lower()
    require("cameo" not in trainer_text, "CAMEO appears in trainer source")
    require("pdb-date" not in trainer_text, "PDB-date appears in trainer source")
    require(
        "extractor.eval().requires_grad_(false)" in trainer_text,
        "trainer does not explicitly freeze DPLM extractor",
    )
    require(
        "tokenizer.eval().requires_grad_(false)" in trainer_text,
        "trainer does not explicitly freeze structure tokenizer",
    )
    syntax_tree = ast.parse(trainer_source)
    training_step_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ddpm_training_step"
    ]
    keyword_sets = [{keyword.arg for keyword in node.keywords} for node in training_step_calls]
    require(
        any("timesteps" not in keys and "noise" not in keys for keys in keyword_sets),
        "training does not use the random ddpm_training_step corruption path",
    )
    require(
        any("timesteps" in keys and "noise" in keys for keys in keyword_sets),
        "validation does not pass explicit deterministic timestep/noise",
    )
    required_checkpoint_keys = (
        "ddpm_model_state_dict",
        "ddpm_schedule_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "optimizer_step",
        "micro_step",
        "epoch",
        "epoch_position",
        "samples_seen",
        "best_val_loss",
        "best_step",
        "training_configuration",
        "ddpm_schedule_configuration",
        "ddpm_architecture_configuration",
        "validation_policy",
        "split_path",
        "dataset_path",
        "dplm_checkpoint_identifier",
        "initial_ddpm_digest",
        "matched_ddpm_spec_sha256",
    )
    for key in required_checkpoint_keys:
        quoted_key = chr(34) + key + chr(34)
        require(quoted_key in trainer_source, f"checkpoint key missing: {key}")
    require(
        "payload.update(rng_state())" in trainer_source,
        "checkpoint does not include Python/NumPy/Torch RNG states",
    )
    require(CHECKPOINT_SELECTION_METRIC == "deterministic_internal_val_masked_epsilon_mse", "best metric mismatch")
    require(PREDICTION_TYPE == "epsilon", "prediction type mismatch")

    print("[MATCHED DDPM TRAINER PROTOCOL AUDIT]")
    print(f"frozen spec SHA-256: {actual_spec_digest}")
    print(f"Dataset path: {DEFAULT_DATASET_DIR}")
    print(f"split path: {DEFAULT_SPLIT_CSV}")
    print(f"train count: {len(train_rows)}")
    print(f"val count: {len(val_rows)}")
    print(f"DPLM identifier: {DPLM_CHECKPOINT}")
    print(f"DDPM residual dim: {ddpm.residual_dim}")
    print(f"DDPM condition dim: {ddpm.condition_dim}")
    print(f"DDPM hidden dim: {ddpm.hidden_dim}")
    print(f"DDPM blocks: {ddpm.num_layers}")
    print(f"DDPM hidden states: {ddpm.num_hidden_states}")
    print(f"DDPM parameter count: {ddpm_trainable}")
    print(f"DDPM schedule T: {schedule.num_timesteps}")
    print(f"DDPM beta1: {float(schedule.betas[0]):.17g}")
    print(f"DDPM betaT: {float(schedule.betas[-1]):.17g}")
    print(f"physical batch: {PHYSICAL_BATCH_SIZE}")
    print(f"gradient accumulation: {protocol['grad_accum_steps']}")
    print(f"effective batch: {protocol['effective_batch_size']}")
    print(f"max optimizer steps: {protocol['max_optimizer_steps']}")
    print(f"warmup optimizer steps: {protocol['warmup_steps']}")
    print(f"peak LR: {protocol['lr']}")
    print(f"final LR: {protocol['final_lr']}")
    print(f"weight decay: {protocol['weight_decay']}")
    print(f"gradient clip: {protocol['grad_clip']}")
    print(f"log interval: {protocol['log_every']}")
    print(f"validation interval: {protocol['val_every']}")
    print(f"save interval: {protocol['save_every']}")
    print(f"training timestep: {TRAINING_TIMESTEP_POLICY}")
    print(f"training noise: {TRAINING_NOISE_POLICY}")
    print("training objective: masked epsilon MSE")
    print(f"validation deterministic policy: {VALIDATION_POLICY}")
    print(f"best checkpoint metric: {CHECKPOINT_SELECTION_METRIC}")
    print("CAMEO/PDB-date absent from train/validation: True")
    print("shared finalized FM settings matched: True")
    print("DDPM-specific differences:")
    print("  discrete t in 1..1000")
    print("  Gaussian epsilon and q_sample forward corruption")
    print("  epsilon-prediction target")
    print("MATCHED_DDPM_TRAINER_PROTOCOL_AUDIT_PASS")


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"[MATCHED_DDPM_TRAINER_PROTOCOL_AUDIT_FAIL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

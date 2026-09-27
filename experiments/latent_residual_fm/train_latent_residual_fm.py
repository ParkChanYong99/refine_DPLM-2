#!/usr/bin/env python3
"""Train the Mode-A latent residual FM with frozen DPLM teacher forcing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from experiments.latent_residual_fm.dplm_condition import DPLMConditionExtractor
from experiments.latent_residual_fm.flow_matching import (
    masked_flow_matching_loss,
    sample_flow_matching_batch,
)
from experiments.latent_residual_fm.fm_model import LatentResidualFM


DPLM_CHECKPOINT = "airkingbd/dplm2_bit_650m"
EXPECTED_TRAIN = 174_685
EXPECTED_VAL = 2_048
DEFAULT_WANDB_TAGS = (
    "latent-residual",
    "flow-matching",
    "modeA",
    "current-token",
    "dplm2-bit-650m",
    "afdb",
    "l512",
)
METRIC_COLUMNS = (
    "wall_time_sec",
    "optimizer_step",
    "micro_step",
    "epoch",
    "train_loss",
    "learning_rate",
    "grad_norm",
    "val_loss",
    "samples_seen",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
)


@dataclass(frozen=True)
class SplitRow:
    sample_id: str
    shard_file: str
    index_in_shard: int
    length: int
    split: str


@dataclass
class Timings:
    dplm_condition: float = 0.0
    fm_forward_backward: float = 0.0
    optimizer: float = 0.0
    validation: float = 0.0
    shard_load: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="data-bin/latent_residual_fm/afdb_l512_sharded",
    )
    parser.add_argument(
        "--split_csv",
        default="data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments/latent_residual_fm/runs/fm_modeA_current",
    )
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
    parser.add_argument("--val_every", type=int, default=2_000)
    parser.add_argument("--val_max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb_entity", default="bumgye99-konkuk-university"
    )
    parser.add_argument("--wandb_project", default="dplm2-fm-refiner")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "lr",
        "final_lr",
        "weight_decay",
        "grad_clip",
    ):
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
    require(args.warmup_steps >= 0, "--warmup_steps must be non-negative")
    require(
        args.final_lr <= args.lr,
        "--final_lr must not exceed --lr for linear decay",
    )
    require(
        args.warmup_steps <= args.max_optimizer_steps,
        "--warmup_steps must not exceed --max_optimizer_steps",
    )
    if args.stop_after_optimizer_steps is not None:
        require(
            0 < args.stop_after_optimizer_steps <= args.max_optimizer_steps,
            "--stop_after_optimizer_steps must be positive and not exceed "
            "--max_optimizer_steps",
        )
    for name in ("train_max_samples", "val_max_samples"):
        value = getattr(args, name)
        require(value is None or value > 0, f"--{name} must be positive when set")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_peaks(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    scale = 1024**2
    return (
        torch.cuda.max_memory_allocated(device) / scale,
        torch.cuda.max_memory_reserved(device) / scale,
    )


def load_split(path: Path) -> tuple[list[SplitRow], list[SplitRow]]:
    required = {"sample_id", "shard_file", "index_in_shard", "length", "split"}
    rows: list[SplitRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        require(not missing, f"split CSV missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            require(raw["split"] in {"train", "val"}, f"bad split at line {line_number}")
            index = int(raw["index_in_shard"])
            length = int(raw["length"])
            require(index >= 0 and length > 0, f"bad index/length at line {line_number}")
            rows.append(
                SplitRow(
                    sample_id=raw["sample_id"],
                    shard_file=raw["shard_file"],
                    index_in_shard=index,
                    length=length,
                    split=raw["split"],
                )
            )
    require(len({row.sample_id for row in rows}) == len(rows), "duplicate split sample IDs")
    train = [row for row in rows if row.split == "train"]
    val = [row for row in rows if row.split == "val"]
    require(len(train) == EXPECTED_TRAIN, f"expected {EXPECTED_TRAIN} train rows")
    require(len(val) == EXPECTED_VAL, f"expected {EXPECTED_VAL} val rows")
    return train, val


def deterministic_subset(
    rows: Sequence[SplitRow], maximum: int | None, seed: int
) -> list[SplitRow]:
    if maximum is None or maximum >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(rows), size=maximum, replace=False))
    return [rows[int(index)] for index in selected]


def group_rows(rows: Sequence[SplitRow]) -> dict[str, list[SplitRow]]:
    grouped: dict[str, list[SplitRow]] = defaultdict(list)
    for row in rows:
        grouped[row.shard_file].append(row)
    return dict(grouped)


def shuffled_epoch_groups(
    grouped: dict[str, list[SplitRow]], seed: int, epoch: int
) -> list[tuple[str, list[SplitRow]]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    shard_files = sorted(grouped)
    rng.shuffle(shard_files)
    result: list[tuple[str, list[SplitRow]]] = []
    for shard_file in shard_files:
        rows = list(grouped[shard_file])
        rng.shuffle(rows)
        result.append((shard_file, rows))
    return result


def load_shard(path: Path, timings: Timings, device: torch.device) -> dict[str, Any]:
    synchronize(device)
    started = time.perf_counter()
    shard = torch.load(str(path), map_location="cpu", mmap=True)
    timings.shard_load += time.perf_counter() - started
    require(isinstance(shard, dict), f"invalid shard payload: {path}")
    require("samples" in shard and "num_samples" in shard, f"invalid shard keys: {path}")
    require(shard["num_samples"] == len(shard["samples"]), f"bad shard count: {path}")
    return shard


def iter_training_epoch(
    grouped: dict[str, list[SplitRow]],
    dataset_dir: Path,
    seed: int,
    epoch: int,
    skip: int,
    timings: Timings,
    device: torch.device,
) -> Iterator[tuple[SplitRow, dict[str, Any]]]:
    position = 0
    for shard_file, rows in shuffled_epoch_groups(grouped, seed, epoch):
        next_position = position + len(rows)
        if next_position <= skip:
            position = next_position
            continue
        start = max(0, skip - position)
        shard = load_shard(dataset_dir / shard_file, timings, device)
        samples = shard["samples"]
        for row in rows[start:]:
            require(row.index_in_shard < len(samples), f"index outside {shard_file}")
            sample = samples[row.index_in_shard]
            require(sample["sample_id"] == row.sample_id, "split/shard sample ID mismatch")
            require(int(sample["length"]) == row.length, "split/shard length mismatch")
            yield row, sample
        del samples, shard
        position = next_position


def iter_validation_samples(
    grouped: dict[str, list[SplitRow]],
    dataset_dir: Path,
    timings: Timings,
    device: torch.device,
) -> Iterator[tuple[SplitRow, dict[str, Any]]]:
    for shard_file in sorted(grouped):
        rows = sorted(grouped[shard_file], key=lambda row: row.index_in_shard)
        shard = load_shard(dataset_dir / shard_file, timings, device)
        samples = shard["samples"]
        for row in rows:
            require(row.index_in_shard < len(samples), f"index outside {shard_file}")
            sample = samples[row.index_in_shard]
            require(sample["sample_id"] == row.sample_id, "split/shard sample ID mismatch")
            yield row, sample
        del samples, shard


def sample_tensors(
    sample: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(sample["length"])
    aatype = sample["aatype"].long().view(1, length)
    struct_ids = sample["struct_ids_current"].long().view(1, length)
    z_quant = sample["z_quant_current"].float().view(1, length, 13).to(device)
    residual = sample["residual"].float().view(1, length, 13).to(device)
    res_mask = sample["res_mask"].float().view(1, length).to(device)
    require(bool(torch.isfinite(residual).all()), "residual is non-finite")
    return aatype, struct_ids, z_quant, residual, res_mask


def extract_hidden(
    extractor: DPLMConditionExtractor,
    aatype: torch.Tensor,
    struct_ids: torch.Tensor,
    res_mask: torch.Tensor,
    timings: Timings,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        condition = extractor(aatype, struct_ids, res_mask)
    synchronize(device)
    timings.dplm_condition += time.perf_counter() - started
    hidden = condition["hidden_states"]
    require(len(hidden) == 33, "DPLM condition must contain 33 Transformer outputs")
    return hidden


def validation_seed(global_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def validate(
    fm: LatentResidualFM,
    extractor: DPLMConditionExtractor,
    grouped: dict[str, list[SplitRow]],
    dataset_dir: Path,
    seed: int,
    timings: Timings,
    device: torch.device,
) -> tuple[float, int, float]:
    started = time.perf_counter()
    fm.eval()
    extractor.eval()
    total_loss = 0.0
    samples = 0
    with torch.no_grad():
        for row, sample in iter_validation_samples(
            grouped, dataset_dir, timings, device
        ):
            aatype, struct_ids, z_quant, residual, res_mask = sample_tensors(sample, device)
            hidden = extract_hidden(
                extractor, aatype, struct_ids, res_mask, timings, device
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(validation_seed(seed, row.sample_id))
            flow = sample_flow_matching_batch(
                residual, res_mask, generator=generator
            )
            prediction = fm(
                flow["r_t"], flow["t"], z_quant, hidden, res_mask
            )
            loss = masked_flow_matching_loss(prediction, flow["u_t"], res_mask)
            require(bool(torch.isfinite(loss)), "validation loss is non-finite")
            total_loss += float(loss)
            samples += 1
            del aatype, struct_ids, z_quant, residual, res_mask
            del hidden, flow, prediction, loss, generator
    require(samples > 0, "validation set is empty")
    synchronize(device)
    elapsed = time.perf_counter() - started
    timings.validation += elapsed
    fm.train()
    return total_loss / samples, samples, elapsed


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    final_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    final_ratio = final_lr / lr

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if max_steps == warmup_steps:
            return final_ratio
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 + progress * (final_ratio - 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dplm_gradient_parameters(extractor: DPLMConditionExtractor) -> int:
    return sum(parameter.grad is not None for parameter in extractor.model.parameters())


def gradients_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"]:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["torch_cuda_rng_state"]]
        )


def checkpoint_payload(
    fm: LatentResidualFM,
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
    initial_fm_digest: str,
    wandb_run_id: str | None,
) -> dict[str, Any]:
    payload = {
        "fm_model_state_dict": fm.state_dict(),
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
        "split_path": args.split_csv,
        "dplm_checkpoint_identifier": DPLM_CHECKPOINT,
        "initial_fm_digest": initial_fm_digest,
    }
    if wandb_run_id is not None:
        payload["wandb_run_id"] = wandb_run_id
    payload.update(rng_state())
    return payload


def open_metrics(path: Path, resume: bool) -> tuple[Any, csv.DictWriter]:
    if resume:
        require(path.is_file(), "resume requested but metrics.csv is missing")
        handle = path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
    else:
        require(not path.exists(), f"refusing to overwrite existing metrics: {path}")
        handle = path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        handle.flush()
    return handle, writer


def read_metric_history(
    path: Path, through_step: int
) -> tuple[float | None, float, dict[int, float]]:
    initial: float | None = None
    final = float("nan")
    validation: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["optimizer_step"])
            if step > through_step:
                continue
            train_loss = float(row["train_loss"])
            if initial is None:
                initial = train_loss
            final = train_loss
            if row["val_loss"]:
                validation[step] = float(row["val_loss"])
    return initial, final, validation


def resolved_wandb_tags(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys((*DEFAULT_WANDB_TAGS, *(args.wandb_tags or ()))))


def wandb_configuration(
    args: argparse.Namespace,
    train_samples: int,
    val_samples: int,
    extractor: DPLMConditionExtractor,
    fm: LatentResidualFM,
    fm_parameters: int,
) -> dict[str, Any]:
    return {
        "EXPERIMENT": {
            "experiment": "latent_residual_fm",
            "method": "Gaussian-source conditional Flow Matching",
            "mode": "Mode-A Current",
            "residual_definition": "z_cont_current - z_quant_current",
            "source": "Gaussian",
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
        "FM_MODEL": {
            "FM_parameters": fm_parameters,
            "residual_dim": fm.residual_dim,
            "condition_dim": fm.condition_dim,
            "hidden_dim": fm.hidden_dim,
            "num_layers": fm.num_layers,
        },
        "TRAINING": {
            "physical_batch_size": 1,
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
        },
        "checkpoint_selection_metric": "val/loss",
    }


def initialize_wandb(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    stored_run_id: str | None,
) -> Any:
    import wandb

    run_name = args.wandb_run_name or (
        f"latentFM-modeA-current-seed{args.seed}-"
        f"{args.max_optimizer_steps}steps"
    )
    init_kwargs: dict[str, Any] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": run_name,
        "tags": resolved_wandb_tags(args),
        "notes": args.wandb_notes,
        "mode": args.wandb_mode,
        "config": configuration,
    }
    if stored_run_id is not None:
        init_kwargs.update({"id": stored_run_id, "resume": "must"})
    return wandb.init(**init_kwargs)


def layer_mix_metrics(fm: LatentResidualFM) -> dict[str, float]:
    with torch.no_grad():
        weights = fm.layer_weights().detach().float().cpu()
    metrics = {
        "layer_mix/min": float(weights.min()),
        "layer_mix/max": float(weights.max()),
        "layer_mix/entropy": float(-(weights * weights.log()).sum()),
    }
    metrics.update(
        {
            f"layer_mix/layer_{index:02d}": float(weight)
            for index, weight in enumerate(weights, start=1)
        }
    )
    return metrics


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LATENT RESIDUAL FM TRAINING SMOKE TEST")
    print("=" * 60)
    print("\nDATA")
    print(f"train available: {summary['train_available']}")
    print(f"val available: {summary['val_available']}")
    print(f"smoke train subset: {summary['train_subset']}")
    print(f"smoke val subset: {summary['val_subset']}")
    print("\nMODEL")
    print(f"DPLM trainable params: {summary['dplm_trainable_params']}")
    print(f"FM trainable params: {summary['fm_trainable_params']}")
    print("\nTRAIN")
    print(f"optimizer steps: {summary['optimizer_steps']}")
    print(f"micro steps: {summary['micro_steps']}")
    print(f"samples seen: {summary['samples_seen']}")
    print(f"grad accumulation: {summary['grad_accumulation']}")
    print(f"\ninitial logged train loss: {summary['initial_train_loss']:.9f}")
    print(f"final logged train loss: {summary['final_train_loss']:.9f}")
    print("\nVALIDATION")
    print(f"step 50 val loss: {summary['val_losses'].get(50, float('nan')):.9f}")
    print(f"step 100 val loss: {summary['val_losses'].get(100, float('nan')):.9f}")
    print(f"best val loss: {summary['best_val_loss']:.9f}")
    print(f"best step: {summary['best_step']}")
    print("\nCHECKPOINT")
    print(f"last exists: {summary['last_exists']}")
    print(f"best exists: {summary['best_exists']}")
    print(f"metrics exists: {summary['metrics_exists']}")
    print("\nGRADIENT")
    print(f"DPLM gradient params: {summary['dplm_gradient_params']}")
    print(f"FM parameters changed: {summary['fm_parameters_changed']}")
    print("\nTIMING")
    print(f"DPLM condition seconds: {summary['timings'].dplm_condition:.6f}")
    print(f"FM forward/backward seconds: {summary['timings'].fm_forward_backward:.6f}")
    print(f"optimizer seconds: {summary['timings'].optimizer:.6f}")
    print(f"validation seconds: {summary['timings'].validation:.6f}")
    print(f"shard load seconds: {summary['timings'].shard_load:.6f}")
    print(f"total seconds: {summary['total_seconds']:.6f}")
    print(f"samples/sec: {summary['samples_per_second']:.9f}")
    print("\nGPU")
    print(f"peak allocated MiB: {summary['peak_allocated_mib']:.6f}")
    print(f"peak reserved MiB: {summary['peak_reserved_mib']:.6f}")
    print(f"\nFINAL STATUS:\n{summary['final_status']}")
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
        "fm_trainable_params": 0,
        "optimizer_steps": 0,
        "micro_steps": 0,
        "samples_seen": 0,
        "grad_accumulation": args.grad_accum_steps,
        "initial_train_loss": float("nan"),
        "final_train_loss": float("nan"),
        "val_losses": {},
        "best_val_loss": float("inf"),
        "best_step": -1,
        "last_exists": False,
        "best_exists": False,
        "metrics_exists": False,
        "dplm_gradient_params": -1,
        "fm_parameters_changed": False,
        "timings": timings,
        "total_seconds": 0.0,
        "samples_per_second": 0.0,
        "peak_allocated_mib": 0.0,
        "peak_reserved_mib": 0.0,
        "final_status": "SMOKE_TRAIN_FAILED",
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
        summary["train_subset"] = len(train_rows)
        summary["val_subset"] = len(val_rows)
        require(train_rows and val_rows, "train or validation subset is empty")
        train_groups = group_rows(train_rows)
        val_groups = group_rows(val_rows)

        extractor = DPLMConditionExtractor(DPLM_CHECKPOINT, device)
        extractor.eval().requires_grad_(False)
        summary["dplm_trainable_params"] = sum(
            parameter.numel()
            for parameter in extractor.model.parameters()
            if parameter.requires_grad
        )
        require(summary["dplm_trainable_params"] == 0, "DPLM is not frozen")

        fm = LatentResidualFM().to(device)
        fm.train()
        parameters = [parameter for parameter in fm.parameters() if parameter.requires_grad]
        summary["fm_trainable_params"] = sum(parameter.numel() for parameter in parameters)
        require(parameters, "FM has no trainable parameters")
        dplm_parameter_ids = {id(parameter) for parameter in extractor.model.parameters()}
        require(
            all(id(parameter) not in dplm_parameter_ids for parameter in parameters),
            "DPLM parameter included in optimizer parameters",
        )
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = make_scheduler(
            optimizer,
            args.lr,
            args.final_lr,
            args.warmup_steps,
            args.max_optimizer_steps,
        )
        initial_fm_digest = model_digest(fm)
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
            require(
                checkpoint["dplm_checkpoint_identifier"] == DPLM_CHECKPOINT,
                "resume DPLM checkpoint mismatch",
            )
            require(checkpoint["split_path"] == args.split_csv, "resume split mismatch")
            checkpoint_configuration = checkpoint.get("training_configuration", {})
            checkpoint_horizon = checkpoint_configuration.get("max_optimizer_steps")
            if checkpoint_horizon != args.max_optimizer_steps:
                raise RuntimeError(
                    "scheduler horizon mismatch: "
                    f"checkpoint max_optimizer_steps={checkpoint_horizon}, "
                    f"current max_optimizer_steps={args.max_optimizer_steps}"
                )
            checkpoint_lr = float(
                checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
            )
            fm.load_state_dict(checkpoint["fm_model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            current_lr = float(optimizer.param_groups[0]["lr"])
            optimizer_step = int(checkpoint["optimizer_step"])
            micro_step = int(checkpoint["micro_step"])
            epoch = int(checkpoint["epoch"])
            epoch_position = int(checkpoint.get("epoch_position", 0))
            samples_seen = int(checkpoint["samples_seen"])
            best_val_loss = float(checkpoint["best_val_loss"])
            best_step = int(checkpoint.get("best_step", -1))
            initial_fm_digest = checkpoint["initial_fm_digest"]
            restore_rng_state(checkpoint)
            print("[RESUME]")
            print(f"optimizer_step={optimizer_step}")
            print(f"checkpoint_lr={checkpoint_lr:.17g}")
            print(f"current_lr={current_lr:.17g}")
            print(f"max_optimizer_steps={args.max_optimizer_steps}")
            if abs(checkpoint_lr - current_lr) > 1e-12:
                raise RuntimeError(
                    "resume optimizer LR mismatch: "
                    f"checkpoint_lr={checkpoint_lr:.17g}, current_lr={current_lr:.17g}"
                )

        if args.wandb:
            saved_rng_state = rng_state()
            stored_wandb_run_id = (
                checkpoint.get("wandb_run_id") if checkpoint is not None else None
            )
            wandb_run = initialize_wandb(
                args,
                wandb_configuration(
                    args,
                    len(train_all),
                    len(val_all),
                    extractor,
                    fm,
                    summary["fm_trainable_params"],
                ),
                stored_wandb_run_id,
            )
            restore_rng_state(saved_rng_state)
            wandb_run_id = str(wandb_run.id)
            wandb_run.summary["checkpoint_selection_metric"] = "val/loss"

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
                aatype, struct_ids, z_quant, residual, res_mask = sample_tensors(sample, device)
                hidden = extract_hidden(
                    extractor, aatype, struct_ids, res_mask, timings, device
                )
                synchronize(device)
                fm_started = time.perf_counter()
                flow = sample_flow_matching_batch(residual, res_mask)
                prediction = fm(
                    flow["r_t"], flow["t"], z_quant, hidden, res_mask
                )
                loss = masked_flow_matching_loss(prediction, flow["u_t"], res_mask)
                require(bool(torch.isfinite(loss)), "training loss is non-finite")
                loss.backward()
                synchronize(device)
                timings.fm_forward_backward += time.perf_counter() - fm_started
                require(gradients_finite(parameters), "FM gradient contains NaN/Inf")
                require(
                    dplm_gradient_parameters(extractor) == 0,
                    "a DPLM parameter received a gradient",
                )
                accumulation_count += 1
                accumulated_loss += float(loss.detach())
                micro_step += 1
                samples_seen += 1
                epoch_position += 1
                del aatype, struct_ids, z_quant, residual, res_mask
                del hidden, flow, prediction, loss

                at_epoch_end = epoch_position == len(train_rows)
                should_update = accumulation_count == args.grad_accum_steps or at_epoch_end
                if not should_update:
                    continue

                synchronize(device)
                optimizer_started = time.perf_counter()
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulation_count)
                require(gradients_finite(parameters), "averaged gradient contains NaN/Inf")
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
                wandb_step_metrics: dict[str, Any] = {}
                reached_requested_stop = (
                    args.stop_after_optimizer_steps is not None
                    and optimizer_step == process_stop_step
                )
                if optimizer_step % args.val_every == 0 or reached_requested_stop:
                    val_loss, val_samples, val_elapsed = validate(
                        fm,
                        extractor,
                        val_groups,
                        dataset_dir,
                        args.seed,
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
                        wandb_step_metrics.update(
                            {
                                "val/loss": val_loss,
                                "val/best_loss": best_val_loss,
                                "val/best_step": best_step,
                                "val/num_samples": val_samples,
                                "val/time_sec": val_elapsed,
                                "checkpoint/is_best": improved,
                            }
                        )
                        wandb_step_metrics.update(layer_mix_metrics(fm))
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
                        wandb_step_metrics.update(
                            {
                                "train/loss": train_loss,
                                "train/learning_rate": used_lr,
                                "train/grad_norm": grad_norm,
                                "train/optimizer_step": optimizer_step,
                                "train/micro_step": micro_step,
                                "train/epoch": epoch,
                                "train/samples_seen": samples_seen,
                                "performance/samples_per_sec": samples_seen / elapsed,
                                "gpu/peak_allocated_mib": allocated,
                                "gpu/peak_reserved_mib": reserved,
                            }
                        )

                if wandb_run is not None and wandb_step_metrics:
                    wandb_step_metrics.update(
                        {
                            "timing/dplm_condition_sec": timings.dplm_condition,
                            "timing/fm_forward_backward_sec": timings.fm_forward_backward,
                            "timing/optimizer_sec": timings.optimizer,
                            "timing/shard_load_sec": timings.shard_load,
                            "timing/validation_sec": timings.validation,
                            "timing/total_sec": time.perf_counter() - total_started,
                        }
                    )
                    wandb_run.log(wandb_step_metrics, step=optimizer_step)

                should_save = (
                    optimizer_step % args.save_every == 0
                    or optimizer_step == args.max_optimizer_steps
                    or reached_requested_stop
                )
                if should_save or improved:
                    payload = checkpoint_payload(
                        fm,
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
                        initial_fm_digest,
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
        summary["val_losses"] = val_losses
        summary["best_val_loss"] = best_val_loss
        summary["best_step"] = best_step
        summary["dplm_gradient_params"] = dplm_gradient_parameters(extractor)
        summary["fm_parameters_changed"] = model_digest(fm) != initial_fm_digest
        summary["last_exists"] = last_path.is_file()
        summary["best_exists"] = best_path.is_file()
        summary["metrics_exists"] = metrics_path.is_file()
        smoke_configuration = (
            args.max_optimizer_steps == 100
            and args.grad_accum_steps == 4
            and args.train_max_samples == 512
            and args.val_max_samples == 64
            and args.val_every == 50
        )
        common_passed = (
            optimizer_step == process_stop_step
            and math.isfinite(initial_train_loss)
            and math.isfinite(final_train_loss)
            and summary["last_exists"]
            and summary["metrics_exists"]
            and summary["dplm_gradient_params"] == 0
            and summary["fm_parameters_changed"]
        )
        if smoke_configuration:
            passed = (
                common_passed
                and optimizer_step == 100
                and micro_step == 400
                and samples_seen == 400
                and 50 in val_losses
                and 100 in val_losses
                and all(math.isfinite(value) for value in val_losses.values())
                and summary["best_exists"]
            )
            require(passed, "one or more smoke success criteria failed")
            summary["final_status"] = "SMOKE_TRAIN_PASS"
        else:
            require(common_passed, "one or more training completion criteria failed")
            summary["final_status"] = "TRAINING_COMPLETE"
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if metrics_handle is not None:
            metrics_handle.close()
        summary["total_seconds"] = time.perf_counter() - total_started
        if summary["samples_seen"]:
            summary["samples_per_second"] = (
                summary["samples_seen"] / summary["total_seconds"]
            )
        summary["peak_allocated_mib"], summary["peak_reserved_mib"] = gpu_peaks(device)
        if wandb_run is not None:
            if math.isfinite(summary["best_val_loss"]):
                wandb_run.summary["best_val_loss"] = summary["best_val_loss"]
                wandb_run.summary["best_step"] = summary["best_step"]
            wandb_run.summary["final_optimizer_step"] = summary["optimizer_steps"]
            wandb_run.summary["samples_seen"] = summary["samples_seen"]
            wandb_run.summary["peak_gpu_allocated_mib"] = summary[
                "peak_allocated_mib"
            ]
            wandb_run.summary["peak_gpu_reserved_mib"] = summary[
                "peak_reserved_mib"
            ]
            wandb_run.finish()
        print_summary(summary)
    return (
        0
        if summary["final_status"] == "SMOKE_TRAIN_PASS"
        or (
            summary["final_status"] == "TRAINING_COMPLETE"
            and args.stop_after_optimizer_steps is not None
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

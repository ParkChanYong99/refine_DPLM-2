#!/usr/bin/env python3
"""Tiny four-protein Adjoint Matching optimization diagnostic.

This is the first diagnostic that intentionally calls optimizer.step. It never
overwrites or saves the finalized FM checkpoint and does not select production
hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.latent_residual_fm import debug_adjoint_matching_integration as integration
from experiments.latent_residual_fm.adjoint_matching_core import (
    adjoint_matching_loss,
    lean_adjoint_step,
    sigma_offset,
)
from experiments.latent_residual_fm.flow_matching import euler_sample


LAMBDA_DIAGNOSTIC = 1.0
LEARNING_RATE_DIAGNOSTIC = 1e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
NUM_UPDATES = 200
K = 40
H = 1.0 / K
NUM_AM_TIMESTEPS = 20
ALPHA_REFINE = 0.5
RESIDUAL_DIM = 13
EXPECTED_FINE_PARAMS = 34_946_606
EVALUATION_STEPS = (0, 25, 50, 100, 200)
EARLY_INDICES = tuple(range(30))
LATE_INDICES = tuple(range(30, 40))
SUBSET_SEED = 42
TRAJECTORY_SEED_BASE = 100_000
INFERENCE_SEED_BASE = 42_000


@dataclass(frozen=True)
class TrainSelection:
    train_index: int
    sample_id: str
    length: int
    shard_file: str
    index_in_shard: int


@dataclass
class TrainBundle:
    selection: TrainSelection
    z_cont: torch.Tensor
    z_quant: torch.Tensor
    res_mask: torch.Tensor
    aatype: torch.Tensor
    struct_ids: torch.Tensor
    oracle: dict[str, torch.Tensor]


@dataclass(frozen=True)
class EvaluationValue:
    structural_loss: float
    structural_rmsd: float
    latent_mse: float


@dataclass(frozen=True)
class EvaluationCheckpoint:
    step: int
    values: tuple[EvaluationValue, ...]
    mean_structural_loss: float
    mean_latent_mse: float
    parameter_drift: float
    relative_parameter_drift: float
    seconds: float


@dataclass(frozen=True)
class TrainingValue:
    step: int
    sample_index: int
    am_loss: float
    terminal_structural_loss: float
    terminal_adjoint_norm: float
    preclip_gradient_norm: float
    postclip_gradient_norm: float
    clipped: bool
    seconds: float
    optimizer_seconds: float


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
        "--checkpoint",
        default=(
            "experiments/latent_residual_fm/runs/"
            "fm_modeA_current_main/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--dplm_checkpoint", default="airkingbd/dplm2_bit_650m")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def select_shortest_train(split_csv: Path) -> tuple[TrainSelection, ...]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    integration.require(rows, "AFDB train split is empty")
    ranked = sorted(
        enumerate(rows), key=lambda item: (int(item[1]["length"]), item[1]["sample_id"])
    )
    selected: list[TrainSelection] = []
    seen: set[str] = set()
    for train_index, row in ranked:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        selected.append(
            TrainSelection(
                train_index=train_index,
                sample_id=sample_id,
                length=int(row["length"]),
                shard_file=row["shard_file"],
                index_in_shard=int(row["index_in_shard"]),
            )
        )
        if len(selected) == 4:
            break
    integration.require(len(selected) == 4, "fewer than four distinct train proteins")
    return tuple(selected)


def load_train_bundle(
    dataset_dir: Path,
    selection: TrainSelection,
    tokenizer: torch.nn.Module,
    device: torch.device,
) -> TrainBundle:
    shard_path = dataset_dir / selection.shard_file
    shard = torch.load(str(shard_path), map_location="cpu", mmap=True)
    integration.require(isinstance(shard, dict) and "samples" in shard, "invalid shard")
    integration.require(
        selection.index_in_shard < len(shard["samples"]), "train shard index out of range"
    )
    sample = shard["samples"][selection.index_in_shard]
    integration.require(sample["sample_id"] == selection.sample_id, "train sample ID mismatch")
    integration.require(int(sample["length"]) == selection.length, "train length mismatch")
    length = selection.length
    z_cont = sample["z_cont"].float().reshape(1, length, RESIDUAL_DIM).to(device)
    z_quant = (
        sample["z_quant_current"]
        .float()
        .reshape(1, length, RESIDUAL_DIM)
        .to(device)
    )
    res_mask = sample["res_mask"].float().reshape(1, length).to(device)
    aatype = sample["aatype"].long().reshape(1, length)
    struct_ids = sample["struct_ids_current"].long().reshape(1, length)
    integration.require(bool(torch.isfinite(z_cont).all()), "z_cont contains NaN/Inf")
    integration.require(bool(torch.isfinite(z_quant).all()), "z_quant contains NaN/Inf")
    integration.require(bool(((res_mask == 0) | (res_mask == 1)).all()), "bad mask")
    with torch.no_grad():
        oracle = tokenizer.detokenize(z_cont, res_mask=res_mask)
    del shard, sample
    return TrainBundle(
        selection=selection,
        z_cont=z_cont,
        z_quant=z_quant,
        res_mask=res_mask,
        aatype=aatype,
        struct_ids=struct_ids,
        oracle=oracle,
    )


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return tuple(left_state) == tuple(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def snapshot_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def snapshot_equal(module: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return tuple(current) == tuple(snapshot) and all(
        torch.equal(current[name].detach().cpu(), snapshot[name]) for name in current
    )


def parameter_versions(module: torch.nn.Module) -> dict[str, int]:
    return {name: parameter._version for name, parameter in module.named_parameters()}


def versions_equal(module: torch.nn.Module, versions: dict[str, int]) -> bool:
    return all(
        name in versions and parameter._version == versions[name]
        for name, parameter in module.named_parameters()
    ) and len(tuple(module.named_parameters())) == len(versions)


def parameter_drift(
    base: torch.nn.Module, fine: torch.nn.Module
) -> tuple[float, float]:
    difference_squared = 0.0
    base_squared = 0.0
    for base_parameter, fine_parameter in zip(base.parameters(), fine.parameters()):
        difference_norm = float(
            torch.linalg.vector_norm(fine_parameter.detach() - base_parameter.detach())
        )
        base_norm = float(torch.linalg.vector_norm(base_parameter.detach()))
        difference_squared += difference_norm * difference_norm
        base_squared += base_norm * base_norm
    drift = math.sqrt(difference_squared)
    denominator = math.sqrt(base_squared)
    return drift, drift / denominator


def global_gradient_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()))
            squared += norm * norm
    return math.sqrt(squared)


def condition_for_bundle(
    extractor: integration.DPLMConditionExtractor,
    bundle: TrainBundle,
) -> tuple[torch.Tensor, ...]:
    result = extractor(bundle.aatype, bundle.struct_ids, bundle.res_mask)
    hidden_states = result["hidden_states"]
    length = bundle.selection.length
    integration.require(len(hidden_states) == 33, "Mode-A condition layer mismatch")
    integration.require(
        all(tuple(hidden.shape) == (1, length, 1280) for hidden in hidden_states),
        "Mode-A condition shape mismatch",
    )
    return hidden_states


def terminal_loss_and_adjoint(
    x_hat_1: torch.Tensor,
    bundle: TrainBundle,
    tokenizer: torch.nn.Module,
) -> tuple[float, torch.Tensor]:
    terminal_leaf = x_hat_1.detach().requires_grad_(True)
    integration.require(torch.is_grad_enabled(), "terminal reward autograd disabled")
    z_refined = bundle.z_quant + ALPHA_REFINE * terminal_leaf
    decoded = tokenizer.detokenize(z_refined, res_mask=bundle.res_mask)
    prediction, target, valid = integration.backbone_points(
        decoded, bundle.oracle, bundle.res_mask
    )
    structural_loss = integration.aligned_squared_rmsd(prediction, target, valid)
    terminal_cost = LAMBDA_DIAGNOSTIC * structural_loss
    adjoint = torch.autograd.grad(
        terminal_cost,
        terminal_leaf,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0].detach()
    adjoint = adjoint * bundle.res_mask[..., None]
    loss_value = float(structural_loss.detach())
    integration.require(math.isfinite(loss_value), "terminal loss is NaN/Inf")
    integration.require(loss_value > 0.0, "terminal loss is zero")
    integration.require(bool(torch.isfinite(adjoint).all()), "terminal adjoint NaN/Inf")
    integration.require(float(torch.linalg.vector_norm(adjoint)) > 0.0, "zero terminal adjoint")
    del terminal_leaf, z_refined, decoded, prediction, target, valid, structural_loss, terminal_cost
    return loss_value, adjoint


def lean_adjoints(
    states: Sequence[torch.Tensor],
    a1: torch.Tensor,
    base_fn: integration.VelocityFn,
    res_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    adjoints: list[torch.Tensor | None] = [None] * (K + 1)
    adjoints[K] = a1.detach()
    for index in range(K, 0, -1):
        current = adjoints[index]
        integration.require(current is not None, f"missing adjoint {index}")
        adjoints[index - 1] = lean_adjoint_step(
            states[index], current, index * H, H, base_fn, res_mask
        )
    resolved = tuple(value for value in adjoints if value is not None)
    integration.require(len(resolved) == K + 1, "adjoint indexing mismatch")
    integration.require(
        all(bool(torch.isfinite(adjoint).all()) for adjoint in resolved),
        "adjoint contains NaN/Inf",
    )
    return resolved


def select_am_timesteps(rng: random.Random) -> tuple[int, ...]:
    early = tuple(sorted(rng.sample(EARLY_INDICES, 10)))
    selected = early + LATE_INDICES
    integration.require(len(selected) == NUM_AM_TIMESTEPS, "bad H.2 subset")
    return selected


def train_update(
    update: int,
    sample_index: int,
    bundle: TrainBundle,
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    v_base: torch.nn.Module,
    v_finetune: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    subset_rng: random.Random,
    device: torch.device,
) -> TrainingValue:
    integration.synchronize(device)
    update_start = time.perf_counter()
    hidden_states = condition_for_bundle(extractor, bundle)
    base_fn = integration.velocity_adapter(
        v_base, bundle.z_quant, hidden_states, bundle.res_mask
    )
    fine_fn = integration.velocity_adapter(
        v_finetune, bundle.z_quant, hidden_states, bundle.res_mask
    )
    trajectory_seed = TRAJECTORY_SEED_BASE + update
    states = integration.build_trajectory(fine_fn, bundle.res_mask, trajectory_seed)
    integration.require(len(states) == K + 1, "trajectory indexing mismatch")
    x_hat_1 = integration.make_x_hat_1(states, base_fn, bundle.res_mask)
    terminal_loss, a1 = terminal_loss_and_adjoint(x_hat_1, bundle, tokenizer)
    adjoints = lean_adjoints(states, a1, base_fn, bundle.res_mask)
    selected_timesteps = select_am_timesteps(subset_rng)

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index in selected_timesteps:
        time_value = torch.tensor(
            index * H, device=bundle.res_mask.device, dtype=bundle.res_mask.dtype
        )
        losses.append(
            adjoint_matching_loss(
                states[index],
                adjoints[index],
                time_value,
                fine_fn,
                base_fn,
                sigma_offset(time_value, H),
                bundle.res_mask,
            )
        )
    am_loss = torch.stack(losses).sum()
    integration.require(bool(torch.isfinite(am_loss)), "AM loss is NaN/Inf")
    am_loss_value = float(am_loss.detach())
    am_loss.backward()
    preclip_norm_tensor = torch.nn.utils.clip_grad_norm_(
        v_finetune.parameters(),
        GRADIENT_CLIP_NORM,
        error_if_nonfinite=True,
    )
    preclip_norm = float(preclip_norm_tensor)
    postclip_norm = global_gradient_norm(v_finetune)
    integration.require(math.isfinite(preclip_norm), "preclip gradient norm NaN/Inf")
    integration.require(math.isfinite(postclip_norm), "postclip gradient norm NaN/Inf")
    clipped = preclip_norm > GRADIENT_CLIP_NORM
    integration.require(integration.no_parameter_grads(v_base), "v_base received gradients")
    integration.require(
        all(not state.requires_grad and state.grad is None for state in states),
        "trajectory retained gradients",
    )
    integration.require(
        all(not adjoint.requires_grad and adjoint.grad is None for adjoint in adjoints),
        "adjoint retained gradients",
    )
    integration.synchronize(device)
    optimizer_start = time.perf_counter()
    optimizer.step()
    integration.synchronize(device)
    optimizer_seconds = time.perf_counter() - optimizer_start
    update_seconds = time.perf_counter() - update_start
    value = TrainingValue(
        step=update,
        sample_index=sample_index,
        am_loss=am_loss_value,
        terminal_structural_loss=terminal_loss,
        terminal_adjoint_norm=float(torch.linalg.vector_norm(a1)),
        preclip_gradient_norm=preclip_norm,
        postclip_gradient_norm=postclip_norm,
        clipped=clipped,
        seconds=update_seconds,
        optimizer_seconds=optimizer_seconds,
    )
    optimizer.zero_grad(set_to_none=True)
    del hidden_states, base_fn, fine_fn, states, x_hat_1, a1, adjoints, losses, am_loss
    return value


def masked_latent_mse(
    prediction: torch.Tensor, target: torch.Tensor, res_mask: torch.Tensor
) -> float:
    denominator = res_mask.sum() * prediction.shape[-1]
    integration.require(bool(denominator > 0), "empty latent MSE mask")
    value = ((prediction - target).square() * res_mask[..., None]).sum() / denominator
    return float(value)


def evaluate_checkpoint(
    step: int,
    bundles: Sequence[TrainBundle],
    extractor: integration.DPLMConditionExtractor,
    tokenizer: torch.nn.Module,
    v_base: torch.nn.Module,
    v_finetune: torch.nn.Module,
    device: torch.device,
) -> tuple[EvaluationCheckpoint, bool]:
    integration.synchronize(device)
    start = time.perf_counter()
    values: list[EvaluationValue] = []
    initial_equal = True
    with torch.no_grad():
        for sample_index, bundle in enumerate(bundles):
            hidden_states = condition_for_bundle(extractor, bundle)
            generator = torch.Generator(device=device).manual_seed(
                INFERENCE_SEED_BASE + sample_index
            )
            initial_noise = torch.randn(
                bundle.z_quant.shape,
                generator=generator,
                device=device,
                dtype=bundle.z_quant.dtype,
            ) * bundle.res_mask[..., None]
            predicted_residual = euler_sample(
                v_finetune,
                bundle.z_quant,
                hidden_states,
                bundle.res_mask,
                100,
                initial_noise=initial_noise,
            )
            if step == 0:
                base_residual = euler_sample(
                    v_base,
                    bundle.z_quant,
                    hidden_states,
                    bundle.res_mask,
                    100,
                    initial_noise=initial_noise,
                )
                initial_equal = initial_equal and torch.allclose(
                    predicted_residual, base_residual, rtol=0.0, atol=0.0
                )
                del base_residual
            decoded = tokenizer.detokenize(
                bundle.z_quant + ALPHA_REFINE * predicted_residual,
                res_mask=bundle.res_mask,
            )
            prediction, target, valid = integration.backbone_points(
                decoded, bundle.oracle, bundle.res_mask
            )
            structural_loss_tensor = integration.aligned_squared_rmsd(
                prediction, target, valid
            )
            structural_loss = float(structural_loss_tensor)
            target_residual = bundle.z_cont - bundle.z_quant
            latent_mse = masked_latent_mse(
                predicted_residual, target_residual, bundle.res_mask
            )
            integration.require(
                math.isfinite(structural_loss) and math.isfinite(latent_mse),
                "evaluation contains NaN/Inf",
            )
            values.append(
                EvaluationValue(
                    structural_loss=structural_loss,
                    structural_rmsd=math.sqrt(max(0.0, structural_loss)),
                    latent_mse=latent_mse,
                )
            )
            del (
                hidden_states,
                initial_noise,
                predicted_residual,
                decoded,
                prediction,
                target,
                valid,
                structural_loss_tensor,
                target_residual,
            )
    drift, relative_drift = parameter_drift(v_base, v_finetune)
    integration.synchronize(device)
    seconds = time.perf_counter() - start
    checkpoint = EvaluationCheckpoint(
        step=step,
        values=tuple(values),
        mean_structural_loss=sum(value.structural_loss for value in values) / len(values),
        mean_latent_mse=sum(value.latent_mse for value in values) / len(values),
        parameter_drift=drift,
        relative_parameter_drift=relative_drift,
        seconds=seconds,
    )
    return checkpoint, initial_equal


def print_evaluation(
    checkpoint: EvaluationCheckpoint, bundles: Sequence[TrainBundle]
) -> None:
    label = "initial deterministic structural loss" if checkpoint.step == 0 else f"step {checkpoint.step}"
    print(f"\n{label}:", flush=True)
    for bundle, value in zip(bundles, checkpoint.values):
        print(
            f"  {bundle.selection.sample_id}: aligned_loss={value.structural_loss:.9g} "
            f"sqrt={value.structural_rmsd:.9g} latent_mse={value.latent_mse:.9g}",
            flush=True,
        )
    print(f"  mean aligned loss: {checkpoint.mean_structural_loss:.9g}", flush=True)
    print(f"  mean latent MSE: {checkpoint.mean_latent_mse:.9g}", flush=True)
    print(
        f"  parameter drift: {checkpoint.parameter_drift:.9g} "
        f"relative={checkpoint.relative_parameter_drift:.9g}",
        flush=True,
    )
    print(f"  evaluation seconds: {checkpoint.seconds:.3f}", flush=True)


def range_text(values: Sequence[float]) -> str:
    return f"[{min(values):.9g}, {max(values):.9g}]"


def main() -> int:
    args = parse_args()
    integration.seed_everything(42)
    device = integration.resolve_device(args.device)
    stage = "startup"
    current_update = 0
    current_sample = "none"
    total_start = time.perf_counter()
    print("ADJOINT_TINY_OVERFIT", flush=True)
    try:
        integration.require(device.type == "cuda", "this diagnostic requires CUDA")
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint_path = Path(args.checkpoint)
        checkpoint_digest_before = checkpoint_digest(checkpoint_path)

        stage = "select shortest AFDB train proteins"
        selections = select_shortest_train(Path(args.split_csv))
        print("samples:", flush=True)
        for selection in selections:
            print(
                f"  train_index={selection.train_index} "
                f"sample={selection.sample_id} length={selection.length}",
                flush=True,
            )
        print(f"lengths: {[selection.length for selection in selections]}", flush=True)
        print("diagnostic config:", flush=True)
        print(f"  lambda: {LAMBDA_DIAGNOSTIC}", flush=True)
        print(f"  lr: {LEARNING_RATE_DIAGNOSTIC}", flush=True)
        print(f"  weight decay: {WEIGHT_DECAY}", flush=True)
        print(f"  updates: {NUM_UPDATES}", flush=True)
        print("  physical batch / accumulation: 1 / 1", flush=True)
        print("  AM timesteps: paper H.2-style 20 (10 sampled early + final 10)", flush=True)
        print(f"  timestep subset seed: {SUBSET_SEED}", flush=True)
        print("  per-timestep loss clipping: disabled", flush=True)
        print(f"  global gradient clipping: {GRADIENT_CLIP_NORM}", flush=True)
        print("  all values above are diagnostic only", flush=True)

        stage = "load frozen DPLM/tokenizer"
        extractor = integration.DPLMConditionExtractor(
            args.dplm_checkpoint, device=device
        )
        dplm = extractor.model
        tokenizer = dplm.struct_tokenizer
        dplm.requires_grad_(False).eval()
        tokenizer.requires_grad_(False).eval()

        stage = "load base and finetune FM copies"
        v_base, base_step, _, _ = integration.load_fm(checkpoint_path, device)
        v_finetune, fine_step, _, _ = integration.load_fm(checkpoint_path, device)
        integration.require(base_step == fine_step == 100_000, "checkpoint step mismatch")
        v_base.requires_grad_(False).eval()
        v_finetune.requires_grad_(True).eval()
        initialization_ok = state_dict_equal(v_base, v_finetune)
        initialization_ok = initialization_ok and all(
            base_parameter.data_ptr() != fine_parameter.data_ptr()
            for base_parameter, fine_parameter in zip(
                v_base.parameters(), v_finetune.parameters()
            )
        )
        initialization_ok = initialization_ok and (
            sum(p.numel() for p in v_finetune.parameters() if p.requires_grad)
            == EXPECTED_FINE_PARAMS
        )
        integration.require(initialization_ok, "base/fine initialization mismatch")
        base_snapshot = snapshot_state(v_base)
        dplm_versions = parameter_versions(dplm)
        tokenizer_versions = parameter_versions(tokenizer)

        stage = "load four train samples"
        bundles = tuple(
            load_train_bundle(Path(args.dataset_dir), selection, tokenizer, device)
            for selection in selections
        )

        stage = "step-zero deterministic evaluation"
        evaluations: dict[int, EvaluationCheckpoint] = {}
        initial_evaluation, initial_output_equal = evaluate_checkpoint(
            0, bundles, extractor, tokenizer, v_base, v_finetune, device
        )
        integration.require(initial_output_equal, "step-0 base/fine inference differs")
        evaluations[0] = initial_evaluation
        print_evaluation(initial_evaluation, bundles)

        stage = "create diagnostic AdamW"
        optimizer = torch.optim.AdamW(
            v_finetune.parameters(),
            lr=LEARNING_RATE_DIAGNOSTIC,
            weight_decay=WEIGHT_DECAY,
        )
        subset_rng = random.Random(SUBSET_SEED)
        training_values: list[TrainingValue] = []

        for update in range(1, NUM_UPDATES + 1):
            current_update = update
            sample_index = (update - 1) % len(bundles)
            bundle = bundles[sample_index]
            current_sample = bundle.selection.sample_id
            stage = f"AM optimizer update {update}"
            training_value = train_update(
                update,
                sample_index,
                bundle,
                extractor,
                tokenizer,
                v_base,
                v_finetune,
                optimizer,
                subset_rng,
                device,
            )
            training_values.append(training_value)
            if update == 1 or update % 10 == 0:
                print(
                    f"update {update}: sample={current_sample} "
                    f"AM_loss={training_value.am_loss:.9g} "
                    f"terminal_L={training_value.terminal_structural_loss:.9g} "
                    f"a1_norm={training_value.terminal_adjoint_norm:.9g} "
                    f"grad={training_value.preclip_gradient_norm:.9g}->"
                    f"{training_value.postclip_gradient_norm:.9g} "
                    f"clipped={training_value.clipped}",
                    flush=True,
                )
            if update in EVALUATION_STEPS[1:]:
                stage = f"deterministic evaluation at step {update}"
                evaluation, _ = evaluate_checkpoint(
                    update,
                    bundles,
                    extractor,
                    tokenizer,
                    v_base,
                    v_finetune,
                    device,
                )
                evaluations[update] = evaluation
                print_evaluation(evaluation, bundles)

        stage = "final frozen and success audit"
        optimizer.zero_grad(set_to_none=True)
        checkpoint_digest_after = checkpoint_digest(checkpoint_path)
        base_unchanged = snapshot_equal(v_base, base_snapshot)
        dplm_unchanged = versions_equal(dplm, dplm_versions)
        tokenizer_unchanged = versions_equal(tokenizer, tokenizer_versions)
        frozen_grads_absent = bool(
            integration.no_parameter_grads(v_base)
            and integration.no_parameter_grads(dplm)
            and integration.no_parameter_grads(tokenizer)
        )
        checkpoint_unchanged = checkpoint_digest_before == checkpoint_digest_after
        frozen_ok = bool(
            base_unchanged
            and dplm_unchanged
            and tokenizer_unchanged
            and frozen_grads_absent
            and checkpoint_unchanged
        )
        parameter_finite = all(
            bool(torch.isfinite(parameter).all()) for parameter in v_finetune.parameters()
        )
        training_finite = all(
            all(
                math.isfinite(value)
                for value in (
                    record.am_loss,
                    record.terminal_structural_loss,
                    record.terminal_adjoint_norm,
                    record.preclip_gradient_norm,
                    record.postclip_gradient_norm,
                )
            )
            for record in training_values
        )
        evaluation_finite = all(
            math.isfinite(value.structural_loss) and math.isfinite(value.latent_mse)
            for checkpoint in evaluations.values()
            for value in checkpoint.values
        )
        finite_ok = parameter_finite and training_finite and evaluation_finite

        initial = evaluations[0]
        final = evaluations[NUM_UPDATES]
        improved = sum(
            final_value.structural_loss < initial_value.structural_loss
            for initial_value, final_value in zip(initial.values, final.values)
        )
        mean_improved = final.mean_structural_loss < initial.mean_structural_loss
        clipped_updates = sum(record.clipped for record in training_values)
        preclip_norms = [record.preclip_gradient_norm for record in training_values]
        postclip_norms = [record.postclip_gradient_norm for record in training_values]
        am_losses = [record.am_loss for record in training_values]
        terminal_losses = [record.terminal_structural_loss for record in training_values]
        adjoint_norms = [record.terminal_adjoint_norm for record in training_values]
        update_seconds = [record.seconds for record in training_values]
        optimizer_seconds = [record.optimizer_seconds for record in training_values]
        evaluation_seconds = sum(checkpoint.seconds for checkpoint in evaluations.values())
        integration.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        total_seconds = time.perf_counter() - total_start

        print("\nfinal improved / worse:", flush=True)
        for bundle, initial_value, final_value in zip(
            bundles, initial.values, final.values
        ):
            direction = (
                "improved"
                if final_value.structural_loss < initial_value.structural_loss
                else "worse"
            )
            print(
                f"  {bundle.selection.sample_id}: {direction} "
                f"{initial_value.structural_loss:.9g} -> {final_value.structural_loss:.9g}",
                flush=True,
            )
        print(f"  improved: {improved}/4", flush=True)
        print(
            f"  mean: {initial.mean_structural_loss:.9g} -> "
            f"{final.mean_structural_loss:.9g}",
            flush=True,
        )

        print("training:", flush=True)
        print(f"  initial AM loss: {am_losses[0]:.9g}", flush=True)
        print(f"  final AM loss: {am_losses[-1]:.9g}", flush=True)
        print(f"  AM loss range: {range_text(am_losses)}", flush=True)
        print(f"  terminal L_struct range: {range_text(terminal_losses)}", flush=True)
        print(f"  terminal adjoint norm range: {range_text(adjoint_norms)}", flush=True)
        print(f"  pre-clip grad norm range: {range_text(preclip_norms)}", flush=True)
        print(f"  post-clip grad norm range: {range_text(postclip_norms)}", flush=True)
        print(f"  clipped updates: {clipped_updates}/{NUM_UPDATES}", flush=True)
        print(
            f"  parameter drift: {final.parameter_drift:.9g} "
            f"relative={final.relative_parameter_drift:.9g}",
            flush=True,
        )

        print("frozen audit:", flush=True)
        print(f"  v_base unchanged: {base_unchanged}", flush=True)
        print(f"  DPLM unchanged: {dplm_unchanged}", flush=True)
        print(f"  tokenizer/decoder unchanged: {tokenizer_unchanged}", flush=True)
        print(f"  frozen grads all None: {frozen_grads_absent}", flush=True)
        print(f"  finalized checkpoint SHA256 unchanged: {checkpoint_unchanged}", flush=True)
        print(f"  status: {'PASS' if frozen_ok else 'FAIL'}", flush=True)
        print(f"NaN/Inf: {'PASS' if finite_ok else 'FAIL'}", flush=True)
        print("peak CUDA:", flush=True)
        print(f"  allocated MiB: {peak_allocated:.2f}", flush=True)
        print(f"  reserved MiB: {peak_reserved:.2f}", flush=True)
        print("runtime:", flush=True)
        print(f"  mean AM update seconds: {sum(update_seconds) / len(update_seconds):.3f}", flush=True)
        print(
            f"  mean optimizer.step seconds: "
            f"{sum(optimizer_seconds) / len(optimizer_seconds):.6f}",
            flush=True,
        )
        print(f"  evaluation seconds: {evaluation_seconds:.3f}", flush=True)
        print(f"  total seconds: {total_seconds:.3f}", flush=True)
        print("production lambda: NOT SELECTED", flush=True)
        print("production learning rate: NOT SELECTED", flush=True)
        print("production timestep/clipping policy: NOT SELECTED", flush=True)

        overall = bool(
            mean_improved
            and improved >= 3
            and finite_ok
            and frozen_ok
            and initialization_ok
            and initial_output_equal
        )
        marker = (
            "ADJOINT_TINY_OVERFIT_PASS"
            if overall
            else "ADJOINT_TINY_OVERFIT_FAIL"
        )
        print(f"overall: {marker}", flush=True)
        print(marker, flush=True)
        return 0 if overall else 1
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        print(f"failure point: {stage}", flush=True)
        print(f"failure update: {current_update}", flush=True)
        print(f"failure sample: {current_sample}", flush=True)
        print(f"error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("production hyperparameters: NOT SELECTED", flush=True)
        print("overall: ADJOINT_TINY_OVERFIT_FAIL", flush=True)
        print("ADJOINT_TINY_OVERFIT_FAIL", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

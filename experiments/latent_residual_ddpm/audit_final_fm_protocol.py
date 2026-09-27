#!/usr/bin/env python3
"""Read-only audit of the finalized FM protocol for a matched DDPM baseline."""

from __future__ import annotations

import ast
import csv
import gc
import hashlib
import math
import random
import re
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from experiments.latent_residual_fm.fm_model import LatentResidualFM


RUN_DIR = Path("experiments/latent_residual_fm/runs/fm_modeA_current_main")
BEST_CHECKPOINT = RUN_DIR / "checkpoint_best.pt"
LAST_CHECKPOINT = RUN_DIR / "checkpoint_last.pt"
METRICS_CSV = RUN_DIR / "metrics.csv"
TRAIN_LOG = RUN_DIR / "train.log"
FM_TRAIN_SOURCE = Path("experiments/latent_residual_fm/train_latent_residual_fm.py")
ALPHA_SOURCE = Path("experiments/latent_residual_fm/evaluate_internal_val_alpha_sweep.py")
ALPHA100_SOURCE = Path("experiments/latent_residual_fm/evaluate_internal_val_alpha_sweep_100.py")
ALPHA_RESULTS = RUN_DIR / "internal_val_alpha_sweep_100.csv"
ALPHA_INDICES = RUN_DIR / "internal_val_alpha_sweep_100_indices.csv"
MATCHED_SPEC = Path("experiments/latent_residual_ddpm/MATCHED_DDPM_SPEC.md")
OUTPUT_DIR = Path("generation-results/dplm2_bit_650m_final/ddpm_protocol_audit")
OUTPUT_TXT = OUTPUT_DIR / "final_fm_protocol_audit.txt"
OUTPUT_CSV = OUTPUT_DIR / "final_fm_protocol_audit.csv"

EXPECTED_MAX_STEPS = 100_000
EXPECTED_VAL_INTERVAL = 5_000
EXPECTED_SAVE_INTERVAL = 5_000
EXPECTED_TRAIN = 174_685
EXPECTED_VAL = 2_048
EXPECTED_GRAD_ACCUM = 8
EXPECTED_SAMPLES_SEEN = 799_988
EXPECTED_PARAMETERS = 34_946_606
EXPECTED_BEST_STEP = 100_000
EXPECTED_BEST_LOSS = 0.6716926411918394
EXPECTED_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_ALPHA_N = 100
EXPECTED_ALPHA_SEED = 42
EXPECTED_SELECTED_ALPHA = 0.5

STATUSES = {"CONFIRMED", "NOT CONFIRMED", "MISMATCH"}


@dataclass(frozen=True)
class AuditRecord:
    item: str
    status: str
    value: str
    source: str
    details: str
    critical: bool = True

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid audit status: {self.status}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def status_for(condition: bool) -> str:
    return "CONFIRMED" if condition else "MISMATCH"


def text_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, (list, tuple)):
        return ",".join(text_value(item) for item in value)
    return str(value)


def add_check(
    records: list[AuditRecord],
    item: str,
    actual: Any,
    expected: Any,
    source: Path | str,
    *,
    details: str = "",
    critical: bool = True,
) -> None:
    records.append(
        AuditRecord(
            item=item,
            status=status_for(actual == expected),
            value=text_value(actual),
            source=str(source),
            details=details or f"expected={text_value(expected)}",
            critical=critical,
        )
    )


def add_close_check(
    records: list[AuditRecord],
    item: str,
    actual: float,
    expected: float,
    source: Path | str,
    *,
    abs_tol: float,
    details: str = "",
    critical: bool = True,
) -> None:
    matched = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=abs_tol)
    records.append(
        AuditRecord(
            item=item,
            status=status_for(matched),
            value=text_value(actual),
            source=str(source),
            details=details or f"expected={expected:.17g}, abs_tol={abs_tol:g}",
            critical=critical,
        )
    )


def load_checkpoint_after_key_discovery(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing checkpoint: {path}")
    kwargs = {"map_location": "cpu", "mmap": True}
    try:
        payload = torch.load(str(path), weights_only=False, **kwargs)
    except TypeError:
        payload = torch.load(str(path), **kwargs)
    require(isinstance(payload, dict), f"{path}: checkpoint is not a dict")
    print(f"{label} checkpoint keys:")
    print(",".join(sorted(str(key) for key in payload)))
    required = {
        "fm_model_state_dict",
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
        "split_path",
        "dplm_checkpoint_identifier",
    }
    missing = required - set(payload)
    require(not missing, f"{path}: missing checkpoint keys after discovery: {sorted(missing)}")
    require(isinstance(payload["training_configuration"], dict), "training_configuration is not a dict")
    return payload


def require_config(config: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = set(keys) - set(config)
    require(not missing, f"checkpoint training_configuration missing keys: {sorted(missing)}")


def read_split_counts(path: Path) -> tuple[int, int]:
    require(path.is_file(), f"missing split CSV: {path}")
    train = val = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require("split" in (reader.fieldnames or ()), "split CSV lacks split column")
        for row in reader:
            if row["split"] == "train":
                train += 1
            elif row["split"] == "val":
                val += 1
            else:
                raise AssertionError(f"unexpected split value: {row['split']}")
    return train, val


def read_metrics(path: Path) -> tuple[list[int], list[tuple[int, float]], dict[str, str], int]:
    require(path.is_file(), f"missing metrics CSV: {path}")
    validation_rows: list[tuple[int, float]] = []
    final_rows: dict[int, dict[str, str]] = {}
    row_count = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"optimizer_step", "micro_step", "epoch", "val_loss", "samples_seen"}
        require(required <= set(reader.fieldnames or ()), "metrics CSV schema mismatch")
        for row in reader:
            row_count += 1
            step = int(row["optimizer_step"])
            final_rows[step] = row
            if row["val_loss"].strip():
                loss = float(row["val_loss"])
                require(math.isfinite(loss), f"non-finite validation loss at step {step}")
                validation_rows.append((step, loss))
    require(final_rows, "metrics CSV is empty")
    unique_validation = sorted({step for step, _ in validation_rows})
    return unique_validation, validation_rows, final_rows[max(final_rows)], row_count


def extract_logged_integer(log_text: str, label: str) -> int | None:
    matches = re.findall(rf"^{re.escape(label)}:\s*(\d+)\s*$", log_text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def extract_literal_assignment(path: Path, variable: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: literal assignment {variable} not found")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_alpha_indices(path: Path) -> list[int]:
    require(path.is_file(), f"missing alpha index artifact: {path}")
    result: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["selection_order", "val_index"], "alpha index schema mismatch")
        for expected_order, row in enumerate(reader):
            require(int(row["selection_order"]) == expected_order, "selection_order is not contiguous")
            result.append(int(row["val_index"]))
    return result


def read_alpha_results(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing alpha result artifact: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"val_index", "sample_id", "alpha", "ca_rmsd", "bb_rmsd"}
        require(required <= set(reader.fieldnames or ()), "alpha result schema mismatch")
        rows = list(reader)
    for row in rows:
        for column in ("val_index", "alpha", "ca_rmsd", "bb_rmsd"):
            require(math.isfinite(float(row[column])), f"non-finite alpha result: {column}")
    return rows


def arithmetic_mean(values: list[float]) -> float:
    require(values, "cannot average an empty list")
    return math.fsum(values) / len(values)


def write_outputs(records: list[AuditRecord], summary_lines: list[str], ready: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = [
        {
            "item": record.item,
            "status": record.status,
            "value": record.value,
            "source": record.source,
            "details": record.details,
            "critical": str(record.critical),
        }
        for record in records
    ]

    csv_fd, csv_temp = tempfile.mkstemp(prefix=".final_fm_protocol_audit.", suffix=".csv", dir=OUTPUT_DIR)
    txt_fd, txt_temp = tempfile.mkstemp(prefix=".final_fm_protocol_audit.", suffix=".txt", dir=OUTPUT_DIR)
    os.close(csv_fd)
    os.close(txt_fd)
    try:
        with Path(csv_temp).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("item", "status", "value", "source", "details", "critical"),
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        status = "FINAL_FM_PROTOCOL_AUDIT_PASS" if ready else "FINAL_FM_PROTOCOL_AUDIT_NOT_READY"
        body = [
            "[FINAL FM PROTOCOL AUDIT FOR MATCHED DDPM]",
            "",
            *summary_lines,
            "",
            "[ITEMIZED AUDIT]",
        ]
        body.extend(
            f"{record.item}\t{record.status}\t{record.value}\t{record.source}\t{record.details}"
            for record in records
        )
        body.extend(["", status, ""])
        Path(txt_temp).write_text("\n".join(body), encoding="utf-8")
        os.replace(csv_temp, OUTPUT_CSV)
        os.replace(txt_temp, OUTPUT_TXT)
    finally:
        for temporary in (csv_temp, txt_temp):
            if os.path.exists(temporary):
                os.unlink(temporary)


def run_audit() -> tuple[list[AuditRecord], list[str]]:
    records: list[AuditRecord] = []
    best = load_checkpoint_after_key_discovery(BEST_CHECKPOINT, "best")
    last = load_checkpoint_after_key_discovery(LAST_CHECKPOINT, "last")
    best_config = best["training_configuration"]
    last_config = last["training_configuration"]
    config_keys = (
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
        "train_max_samples",
        "val_max_samples",
    )
    require_config(best_config, config_keys)
    require_config(last_config, config_keys)
    add_check(records, "best_and_last_training_configuration", best_config, last_config, BEST_CHECKPOINT)

    scalar_expectations = (
        ("checkpoint_optimizer_step", int(best["optimizer_step"]), EXPECTED_MAX_STEPS),
        ("best_optimizer_step", int(best["best_step"]), EXPECTED_BEST_STEP),
        ("max_optimizer_steps", int(best_config["max_optimizer_steps"]), EXPECTED_MAX_STEPS),
        ("gradient_accumulation", int(best_config["grad_accum_steps"]), EXPECTED_GRAD_ACCUM),
        ("validation_interval", int(best_config["val_every"]), EXPECTED_VAL_INTERVAL),
        ("save_interval", int(best_config["save_every"]), EXPECTED_SAVE_INTERVAL),
        ("warmup_steps", int(best_config["warmup_steps"]), 2_000),
        ("seed", int(best_config["seed"]), 42),
        ("samples_seen", int(best["samples_seen"]), EXPECTED_SAMPLES_SEEN),
    )
    for item, actual, expected in scalar_expectations:
        add_check(records, item, actual, expected, BEST_CHECKPOINT)
    add_check(records, "checkpoint_last_optimizer_step", int(last["optimizer_step"]), EXPECTED_MAX_STEPS, LAST_CHECKPOINT)
    add_check(records, "checkpoint_micro_step", int(best["micro_step"]), EXPECTED_SAMPLES_SEEN, BEST_CHECKPOINT)
    records.append(
        AuditRecord("checkpoint_epoch", "CONFIRMED", str(best["epoch"]), str(BEST_CHECKPOINT), "actual saved state")
    )
    records.append(
        AuditRecord(
            "checkpoint_epoch_position",
            "CONFIRMED",
            str(best["epoch_position"]),
            str(BEST_CHECKPOINT),
            "actual saved state",
        )
    )
    add_close_check(records, "peak_base_lr", float(best_config["lr"]), 1e-4, BEST_CHECKPOINT, abs_tol=0.0)
    add_close_check(records, "final_lr", float(best_config["final_lr"]), 1e-5, BEST_CHECKPOINT, abs_tol=0.0)
    add_close_check(records, "weight_decay", float(best_config["weight_decay"]), 0.01, BEST_CHECKPOINT, abs_tol=0.0)
    add_close_check(records, "gradient_clipping", float(best_config["grad_clip"]), 1.0, BEST_CHECKPOINT, abs_tol=0.0)

    trainer_text = FM_TRAIN_SOURCE.read_text(encoding="utf-8")
    add_check(
        records,
        "physical_batch_size",
        '"physical_batch_size": 1' in trainer_text and "yield row, sample" in trainer_text,
        True,
        FM_TRAIN_SOURCE,
        details="one sample is yielded per micro-step; W&B configuration records physical_batch_size=1",
    )
    add_check(
        records,
        "nominal_effective_batch",
        int(best_config["grad_accum_steps"]),
        EXPECTED_GRAD_ACCUM,
        FM_TRAIN_SOURCE,
        details="physical batch 1 × gradient accumulation",
    )
    add_check(
        records,
        "scheduler_type",
        "torch.optim.lr_scheduler.LambdaLR" in trainer_text,
        True,
        FM_TRAIN_SOURCE,
        details="LambdaLR with warmup followed by linear decay",
    )
    scheduler_state = best["scheduler_state_dict"]
    require(isinstance(scheduler_state, dict), "scheduler_state_dict is not a dict")
    records.append(
        AuditRecord(
            "scheduler_state_keys",
            "CONFIRMED",
            ",".join(sorted(scheduler_state)),
            str(BEST_CHECKPOINT),
            "actual serialized scheduler state",
            critical=False,
        )
    )
    if "last_epoch" in scheduler_state:
        add_check(
            records,
            "scheduler_last_epoch",
            int(scheduler_state["last_epoch"]),
            EXPECTED_MAX_STEPS,
            BEST_CHECKPOINT,
        )
    else:
        records.append(
            AuditRecord("scheduler_last_epoch", "NOT CONFIRMED", "", str(BEST_CHECKPOINT), "key absent")
        )

    dataset_path = Path(str(best_config["dataset_dir"]))
    split_path = Path(str(best_config["split_csv"]))
    add_check(
        records,
        "dataset_path",
        str(dataset_path),
        "data-bin/latent_residual_fm/afdb_l512_sharded",
        BEST_CHECKPOINT,
    )
    add_check(
        records,
        "split_path",
        str(split_path),
        "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv",
        BEST_CHECKPOINT,
    )
    add_check(records, "checkpoint_split_path", str(best["split_path"]), str(split_path), BEST_CHECKPOINT)
    train_count, val_count = read_split_counts(split_path)
    add_check(records, "train_count", train_count, EXPECTED_TRAIN, split_path)
    add_check(records, "validation_count", val_count, EXPECTED_VAL, split_path)
    add_check(records, "train_subset_is_full", best_config["train_max_samples"], None, BEST_CHECKPOINT)
    add_check(records, "validation_subset_is_full", best_config["val_max_samples"], None, BEST_CHECKPOINT)

    model = LatentResidualFM()
    load_result = model.load_state_dict(best["fm_model_state_dict"], strict=True)
    require(not load_result.missing_keys and not load_result.unexpected_keys, "FM state_dict strict load failed")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    state_parameters = sum(tensor.numel() for tensor in best["fm_model_state_dict"].values())
    add_check(records, "fm_total_parameters", total_parameters, EXPECTED_PARAMETERS, FM_TRAIN_SOURCE)
    add_check(records, "fm_trainable_parameters", trainable_parameters, EXPECTED_PARAMETERS, FM_TRAIN_SOURCE)
    add_check(records, "fm_state_parameter_elements", state_parameters, EXPECTED_PARAMETERS, BEST_CHECKPOINT)
    architecture = {
        "residual_dim": model.residual_dim,
        "condition_dim": model.condition_dim,
        "hidden_dim": model.hidden_dim,
        "number_of_blocks": model.num_layers,
        "dplm_hidden_layers_used": model.num_hidden_states,
    }
    expected_architecture = {
        "residual_dim": 13,
        "condition_dim": 1280,
        "hidden_dim": 1024,
        "number_of_blocks": 6,
        "dplm_hidden_layers_used": 33,
    }
    for item, expected in expected_architecture.items():
        add_check(records, f"fm_architecture_{item}", architecture[item], expected, FM_TRAIN_SOURCE)
    del model

    log_text = TRAIN_LOG.read_text(encoding="utf-8") if TRAIN_LOG.is_file() else ""
    logged_dplm_trainable = extract_logged_integer(log_text, "DPLM trainable params")
    source_freeze_guards = (
        "extractor.eval().requires_grad_(False)" in trainer_text
        and 'summary["dplm_trainable_params"] == 0' in trainer_text
        and "dplm_gradient_parameters(extractor) == 0" in trainer_text
    )
    if logged_dplm_trainable is None:
        records.append(
            AuditRecord(
                "dplm_trainable_parameters",
                "NOT CONFIRMED",
                "",
                str(TRAIN_LOG),
                "final log value absent",
            )
        )
    else:
        add_check(
            records,
            "dplm_trainable_parameters",
            logged_dplm_trainable,
            0,
            TRAIN_LOG,
            details=f"trainer freeze/gradient guards present={source_freeze_guards}",
        )
        if not source_freeze_guards:
            records.append(
                AuditRecord(
                    "dplm_freeze_source_guards",
                    "MISMATCH",
                    str(source_freeze_guards),
                    str(FM_TRAIN_SOURCE),
                    "required freeze and gradient guards",
                )
            )

    validation_steps, validation_rows, final_metric_row, metrics_row_count = read_metrics(METRICS_CSV)
    expected_validation_steps = list(
        range(EXPECTED_VAL_INTERVAL, EXPECTED_MAX_STEPS + 1, EXPECTED_VAL_INTERVAL)
    )
    add_check(records, "validation_optimizer_steps", validation_steps, expected_validation_steps, METRICS_CSV)
    records.append(
        AuditRecord(
            "metrics_row_count",
            "CONFIRMED",
            str(metrics_row_count),
            str(METRICS_CSV),
            "informational; interrupted/resumed branches may leave duplicate optimizer-step rows",
            critical=False,
        )
    )
    min_step, min_loss = min(validation_rows, key=lambda pair: pair[1])
    add_check(records, "metrics_best_step", min_step, int(best["best_step"]), METRICS_CSV)
    add_close_check(
        records,
        "best_validation_loss",
        float(best["best_val_loss"]),
        EXPECTED_BEST_LOSS,
        BEST_CHECKPOINT,
        abs_tol=1e-12,
    )
    add_close_check(
        records,
        "metrics_min_validation_loss",
        min_loss,
        float(best["best_val_loss"]),
        METRICS_CSV,
        abs_tol=5e-10,
        details="metrics.csv stores validation loss rounded to 9 decimals",
    )
    add_check(
        records,
        "checkpoint_selection_criterion",
        (
            "improved = val_loss < best_val_loss" in trainer_text
            and "validation_seed(seed, row.sample_id)" in trainer_text
            and 'masked_flow_matching_loss(prediction, flow["u_t"], res_mask)' in trainer_text
        ),
        True,
        FM_TRAIN_SOURCE,
        details="strict lowest deterministic internal-validation masked FM velocity MSE",
    )

    epoch = int(best["epoch"])
    epoch_position = int(best["epoch_position"])
    exposure_from_position = epoch * train_count + epoch_position
    add_check(
        records,
        "sample_exposure_epoch_consistency",
        int(best["samples_seen"]),
        exposure_from_position,
        BEST_CHECKPOINT,
    )
    full_epochs = epoch
    deficit = EXPECTED_MAX_STEPS * EXPECTED_GRAD_ACCUM - int(best["samples_seen"])
    expected_deficit = full_epochs * (
        EXPECTED_GRAD_ACCUM - (train_count % EXPECTED_GRAD_ACCUM)
    )
    add_check(
        records,
        "epoch_boundary_partial_accumulation_deficit",
        deficit,
        expected_deficit,
        FM_TRAIN_SOURCE,
        details=(
            f"{full_epochs} completed epoch boundaries × "
            f"({EXPECTED_GRAD_ACCUM} - {train_count % EXPECTED_GRAD_ACCUM}) samples"
        ),
    )
    add_check(
        records,
        "final_metrics_samples_seen",
        int(final_metric_row["samples_seen"]),
        int(best["samples_seen"]),
        METRICS_CSV,
    )

    alpha_grid = tuple(float(value) for value in extract_literal_assignment(ALPHA_SOURCE, "ALPHAS"))
    alpha_total = int(extract_literal_assignment(ALPHA100_SOURCE, "EXPECTED_VALIDATION_SAMPLES"))
    alpha_n = int(extract_literal_assignment(ALPHA100_SOURCE, "EXPECTED_SELECTED_SAMPLES"))
    add_check(records, "alpha_grid", alpha_grid, EXPECTED_ALPHA_GRID, ALPHA_SOURCE)
    add_check(records, "alpha_selection_population", alpha_total, EXPECTED_VAL, ALPHA100_SOURCE)
    add_check(records, "alpha_selection_n", alpha_n, EXPECTED_ALPHA_N, ALPHA100_SOURCE)

    indices = read_alpha_indices(ALPHA_INDICES)
    recreated_indices = random.Random(EXPECTED_ALPHA_SEED).sample(range(alpha_total), alpha_n)
    add_check(records, "alpha_selection_indices", indices, recreated_indices, ALPHA_INDICES)
    add_check(records, "alpha_unique_indices", len(set(indices)), EXPECTED_ALPHA_N, ALPHA_INDICES)
    index_hash = sha256_file(ALPHA_INDICES)
    records.append(
        AuditRecord(
            "alpha_indices_sha256",
            "CONFIRMED",
            index_hash,
            str(ALPHA_INDICES),
            "existing artifact recorded by path/hash; not copied",
        )
    )

    alpha_rows = read_alpha_results(ALPHA_RESULTS)
    expected_alpha_rows = EXPECTED_ALPHA_N * len(EXPECTED_ALPHA_GRID)
    add_check(records, "alpha_result_rows", len(alpha_rows), expected_alpha_rows, ALPHA_RESULTS)
    result_indices_in_order = list(dict.fromkeys(int(row["val_index"]) for row in alpha_rows))
    add_check(records, "alpha_result_index_order", result_indices_in_order, indices, ALPHA_RESULTS)
    observed_alphas = tuple(sorted({float(row["alpha"]) for row in alpha_rows}))
    add_check(records, "alpha_result_grid", observed_alphas, EXPECTED_ALPHA_GRID, ALPHA_RESULTS)
    rows_by_index: dict[int, list[dict[str, str]]] = {}
    for row in alpha_rows:
        rows_by_index.setdefault(int(row["val_index"]), []).append(row)
    per_index_complete = all(
        len(rows_by_index.get(index, ())) == len(EXPECTED_ALPHA_GRID)
        and {float(row["alpha"]) for row in rows_by_index[index]} == set(EXPECTED_ALPHA_GRID)
        and len({row["sample_id"] for row in rows_by_index[index]}) == 1
        for index in indices
    )
    add_check(
        records,
        "alpha_rows_complete_per_index",
        per_index_complete,
        True,
        ALPHA_RESULTS,
        details="each selected index has one sample ID and exactly five unique alpha rows",
    )
    alpha_means = {
        alpha: arithmetic_mean(
            [float(row["ca_rmsd"]) for row in alpha_rows if float(row["alpha"]) == alpha]
        )
        for alpha in alpha_grid
    }
    selected_alpha = min(alpha_grid, key=lambda alpha: alpha_means[alpha])
    add_close_check(
        records,
        "selected_alpha",
        selected_alpha,
        EXPECTED_SELECTED_ALPHA,
        ALPHA_RESULTS,
        abs_tol=0.0,
    )

    alpha_source_text = ALPHA_SOURCE.read_text(encoding="utf-8")
    alpha100_source_text = ALPHA100_SOURCE.read_text(encoding="utf-8")
    add_check(
        records,
        "alpha_selection_seed",
        (
            'parser.add_argument("--selection_seed", type=int, default=42)' in alpha100_source_text
            and "random.Random(selection_seed).sample(" in alpha100_source_text
        ),
        True,
        ALPHA100_SOURCE,
        details="seed=42; random.Random(seed).sample(range(2048),100)",
    )
    add_check(
        records,
        "alpha_noise_base_seed",
        'parser.add_argument("--noise_base_seed", type=int, default=42)' in alpha100_source_text,
        True,
        ALPHA100_SOURCE,
        details="noise_base_seed=42",
    )
    add_check(
        records,
        "alpha_internal_validation_paths",
        (
            "data-bin/latent_residual_fm/afdb_l512_sharded" in alpha100_source_text
            and "splits/split_v1.csv" in alpha100_source_text
        ),
        True,
        ALPHA100_SOURCE,
        details="selection data and split are internal AFDB validation artifacts",
    )
    add_check(
        records,
        "alpha_primary_criterion",
        'primary_alpha = min(ALPHAS, key=lambda alpha: summaries[alpha]["ca_mean"])'
        in alpha100_source_text,
        True,
        ALPHA100_SOURCE,
        details="lowest mean CA RMSD",
    )
    add_check(
        records,
        "alpha_secondary_not_primary",
        (
            "SECONDARY DIAGNOSTICS:" in alpha100_source_text
            and alpha100_source_text.count("primary_alpha = min(") == 1
        ),
        True,
        ALPHA100_SOURCE,
        details="median CA, improved %, worst degradation, and backbone metrics are reported diagnostics",
    )
    add_check(
        records,
        "alpha_tie_rule",
        alpha_grid == tuple(sorted(alpha_grid)) and "min(ALPHAS, key=" in alpha100_source_text,
        True,
        ALPHA100_SOURCE,
        details="Python min returns first grid entry on an exact primary-metric tie; ALPHAS is ascending",
    )
    add_check(
        records,
        "alpha_decoder",
        (
            "struct_tokenizer.detokenize(z_alpha" in alpha_source_text
            and "kabsch_atom_rmsd(" in alpha_source_text
            and "CA_INDEX" in alpha_source_text
        ),
        True,
        ALPHA_SOURCE,
        details="frozen continuous-latent detokenize; Kabsch CA RMSD primary metric",
    )
    add_check(
        records,
        "alpha_sampling_seed_policy",
        "generator.manual_seed(base_seed + val_index)" in alpha_source_text,
        True,
        ALPHA_SOURCE,
        details="noise_base_seed=42 plus validation index",
    )
    selection_paths = " ".join(
        (str(best_config["dataset_dir"]), str(best_config["split_csv"]), str(ALPHA_RESULTS), str(ALPHA_INDICES))
    ).lower()
    source_lower = (alpha_source_text + alpha100_source_text).lower()
    no_cameo = "cameo" not in selection_paths and "cameo" not in source_lower
    no_pdb_date = (
        "pdb_date" not in selection_paths
        and "pdb-date" not in selection_paths
        and "pdb_date" not in source_lower
        and "pdb-date" not in source_lower
    )
    add_check(records, "cameo_used_in_alpha_selection", "NO" if no_cameo else "YES", "NO", ALPHA_SOURCE)
    add_check(records, "pdb_date_used_in_alpha_selection", "NO" if no_pdb_date else "YES", "NO", ALPHA_SOURCE)

    spec_text = MATCHED_SPEC.read_text(encoding="utf-8")
    add_check(
        records,
        "matched_spec_validation_interval",
        "Validation interval | 5,000 optimizer steps" in spec_text,
        True,
        MATCHED_SPEC,
    )
    add_check(
        records,
        "matched_spec_save_interval",
        "Save interval | 5,000 optimizer steps" in spec_text,
        True,
        MATCHED_SPEC,
    )

    run_artifacts = sorted(path.name for path in RUN_DIR.iterdir() if path.is_file())
    records.append(
        AuditRecord(
            "run_directory_artifacts",
            "CONFIRMED",
            ",".join(run_artifacts),
            str(RUN_DIR),
            "read-only inventory",
            critical=False,
        )
    )

    summary_lines = [
        f"checkpoint: {BEST_CHECKPOINT}",
        f"optimizer step: {best['optimizer_step']}",
        f"best step: {best['best_step']}",
        f"best validation loss: {best['best_val_loss']}",
        f"max optimizer steps: {best_config['max_optimizer_steps']}",
        f"micro step: {best['micro_step']}",
        f"epoch: {best['epoch']}",
        "",
        f"validation interval: {best_config['val_every']}",
        f"save interval: {best_config['save_every']}",
        f"validation optimizer steps: {','.join(str(step) for step in validation_steps)}",
        "",
        "physical batch: 1",
        f"gradient accumulation: {best_config['grad_accum_steps']}",
        f"effective batch: {best_config['grad_accum_steps']}",
        "",
        f"samples seen: {best['samples_seen']}",
        f"sample exposure explanation: 800000 - {deficit} from {full_epochs} epoch-boundary partial accumulations",
        "",
        f"warmup: {best_config['warmup_steps']}",
        f"peak LR: {best_config['lr']}",
        f"final LR: {best_config['final_lr']}",
        "",
        f"FM trainable parameters: {trainable_parameters}",
        f"DPLM trainable parameters: {logged_dplm_trainable if logged_dplm_trainable is not None else 'NOT CONFIRMED'}",
        "",
        f"alpha grid: {text_value(alpha_grid)}",
        f"alpha selection N: {alpha_n}",
        f"alpha selection seed: {EXPECTED_ALPHA_SEED}",
        "alpha selection population: internal AFDB validation, total 2048",
        "primary criterion: lowest mean CA RMSD",
        f"selected alpha: {selected_alpha}",
        "tie rule: first alpha in ascending ALPHAS order on exact tie",
        "decoder: frozen continuous-latent structure tokenizer detokenize",
        "sampling seed policy: noise_base_seed(42) + val_index",
        f"selection indices artifact: {ALPHA_INDICES}",
        f"selection indices SHA-256: {index_hash}",
        "",
        "CAMEO used in selection: NO",
        "PDB-date used in selection: NO",
    ]
    del best, last
    gc.collect()
    return records, summary_lines


def main() -> int:
    records: list[AuditRecord] = []
    summary_lines: list[str] = []
    try:
        records, summary_lines = run_audit()
    except Exception as error:
        records.append(
            AuditRecord(
                "audit_runtime",
                "NOT CONFIRMED",
                "",
                "audit_final_fm_protocol.py",
                f"{type(error).__name__}: {error}",
            )
        )
        summary_lines.append(f"audit error: {type(error).__name__}: {error}")

    blockers = [
        record
        for record in records
        if record.critical and record.status in {"NOT CONFIRMED", "MISMATCH"}
    ]
    ready = not blockers
    summary_lines.extend(
        [
            "",
            "Protocol mismatches:",
            *(f"- {record.item}: {record.status} ({record.details})" for record in blockers),
        ]
    )
    if not blockers:
        summary_lines.append("- none")

    write_outputs(records, summary_lines, ready)
    print("[FINAL FM PROTOCOL AUDIT FOR MATCHED DDPM]")
    print()
    for line in summary_lines:
        print(line)
    print()
    print(f"created: {OUTPUT_TXT}")
    print(f"created: {OUTPUT_CSV}")
    print()
    final_status = "FINAL_FM_PROTOCOL_AUDIT_PASS" if ready else "FINAL_FM_PROTOCOL_AUDIT_NOT_READY"
    print(final_status)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

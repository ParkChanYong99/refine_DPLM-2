#!/usr/bin/env python3
"""Select matched-DDPM residual alpha on the fixed FM internal-val 100."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Sequence

import torch

from experiments.latent_residual_ddpm.debug_inference_integration import (
    DEFAULT_DPLM_CHECKPOINT,
    EXPECTED_BEST_STEP,
    EXPECTED_DDPM_PARAMETERS,
    EXPECTED_OPTIMIZER_STEP,
    NUM_TIMESTEPS,
    all_grads_none,
    load_ddpm,
    run_sampler,
    synchronize,
    timed_call,
)
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
    build_folding_input,
    extract_final_token_condition,
    load_validation_sample,
    reconstruct_prediction_latent,
)
from experiments.latent_residual_fm.evaluate_internal_val_alpha_sweep_100 import (
    count_validation_rows,
)
from experiments.latent_residual_fm.evaluate_internal_val_coordinate import (
    BACKBONE_INDICES,
    CA_INDEX,
    common_residue_mask,
    kabsch_atom_rmsd,
    validate_decoder_output,
)


ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
EXPECTED_VALIDATION_POPULATION = 2_048
EXPECTED_SELECTION_SIZE = 100
EXPECTED_INDICES_SHA256 = (
    "ae8001d053d778745c8560a36f60753d408e6db962bbcac749ac30216bd5ed21"
)
DEFAULT_DATASET_DIR = "data-bin/latent_residual_fm/afdb_l512_sharded"
DEFAULT_SPLIT_CSV = (
    "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main"
)
DEFAULT_INDICES_CSV = (
    "experiments/latent_residual_fm/runs/fm_modeA_current_main/"
    "internal_val_alpha_sweep_100_indices.csv"
)
RESULT_NAME = "internal_val_alpha_sweep_100.csv"
SUMMARY_TEXT_NAME = "internal_val_alpha_sweep_100_summary.txt"
SUMMARY_CSV_NAME = "internal_val_alpha_sweep_100_summary.csv"
METADATA_NAME = "internal_val_alpha_sweep_100_metadata.json"

RESULT_COLUMNS = (
    "selection_order",
    "val_index",
    "sample_id",
    "length",
    "seed",
    "alpha",
    "ca_rmsd",
    "backbone_rmsd",
    "ddpm_nfe",
    "dplm_generation_seconds",
    "extra_forward_seconds",
    "ddpm_sampling_seconds",
    "decode_metric_seconds",
    "target_total_seconds",
)
SUMMARY_COLUMNS = (
    "alpha",
    "num_samples",
    "mean_ca_rmsd",
    "median_ca_rmsd",
    "mean_backbone_rmsd",
    "median_backbone_rmsd",
    "selected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--indices_csv", default=DEFAULT_INDICES_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dplm_checkpoint", default=DEFAULT_DPLM_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_indices(path: Path, split_csv: Path) -> tuple[list[int], str]:
    require(path.is_file(), f"FM indices artifact not found: {path}")
    digest = sha256_file(path)
    require(
        digest == EXPECTED_INDICES_SHA256,
        f"FM indices artifact SHA256 mismatch: {digest}",
    )
    validation_population = count_validation_rows(split_csv)
    require(
        validation_population == EXPECTED_VALIDATION_POPULATION,
        f"expected {EXPECTED_VALIDATION_POPULATION} validation rows, "
        f"got {validation_population}",
    )

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(
            tuple(reader.fieldnames or ()) == ("selection_order", "val_index"),
            "FM indices artifact columns changed",
        )
        rows = list(reader)
    require(len(rows) == EXPECTED_SELECTION_SIZE, "FM index count must be 100")
    selected: list[int] = []
    for expected_order, row in enumerate(rows):
        selection_order = int(row["selection_order"])
        val_index = int(row["val_index"])
        require(selection_order == expected_order, "FM selection_order is not contiguous")
        require(
            0 <= val_index < validation_population,
            f"FM val_index out of range: {val_index}",
        )
        selected.append(val_index)
    require(len(set(selected)) == EXPECTED_SELECTION_SIZE, "FM val indices are not unique")
    return selected, digest


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_metadata(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, float | int],
    indices_digest: str,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    return {
        "method": "Local Matched Residual DDPM",
        "dataset_role": "internal AFDB validation only",
        "dataset_dir": args.dataset_dir,
        "split_csv": args.split_csv,
        "validation_population": EXPECTED_VALIDATION_POPULATION,
        "selection_size": EXPECTED_SELECTION_SIZE,
        "selected_val_indices": list(selected_indices),
        "indices_artifact_path": args.indices_csv,
        "indices_artifact_sha256": indices_digest,
        "ddpm_checkpoint_path": str(checkpoint_path),
        "optimizer_step": int(checkpoint_metadata["optimizer_step"]),
        "best_step": int(checkpoint_metadata["best_step"]),
        "best_val_loss": float(checkpoint_metadata["best_val_loss"]),
        "ddpm_parameters": int(checkpoint_metadata["parameter_count"]),
        "sampler": "full ancestral DDPM",
        "ddpm_timesteps": NUM_TIMESTEPS,
        "ddpm_nfe": NUM_TIMESTEPS,
        "alpha_grid": list(ALPHAS),
        "seed_policy": "42 + val_index",
        "base_seed": args.base_seed,
        "dplm_checkpoint": args.dplm_checkpoint,
        "generation_max_iter": GENERATION_MAX_ITER,
        "unmasking_strategy": "deterministic",
        "sampling_strategy": "argmax",
        "selection_metric": "lowest mean CA RMSD",
        "tie_rule": "exact tie keeps first alpha in ascending grid",
        "tm_score": "not computed; finalized FM internal alpha sweep did not compute TM-score",
        "cameo_or_pdb_date_used": False,
    }


def validate_or_create_metadata(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            stored = json.load(handle)
        require(stored == expected, "resume metadata differs from current fixed settings")
    else:
        atomic_write_json(path, expected)


def validate_numeric_row(row: dict[str, Any]) -> None:
    for column in (
        "selection_order",
        "val_index",
        "length",
        "seed",
        "alpha",
        "ca_rmsd",
        "backbone_rmsd",
        "ddpm_nfe",
        "dplm_generation_seconds",
        "extra_forward_seconds",
        "ddpm_sampling_seconds",
        "decode_metric_seconds",
        "target_total_seconds",
    ):
        require(math.isfinite(float(row[column])), f"non-finite partial value: {column}")


def load_partial_rows(
    path: Path,
    selected_indices: Sequence[int],
    base_seed: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    if not path.exists():
        return [], set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == RESULT_COLUMNS, "partial CSV columns changed")
        rows = list(reader)

    selected_set = set(selected_indices)
    counts = Counter(int(row["val_index"]) for row in rows)
    require(set(counts).issubset(selected_set), "partial CSV contains an unselected val_index")
    require(
        all(count == len(ALPHAS) for count in counts.values()),
        "partial CSV contains an incomplete target",
    )
    order_by_index = {index: order for order, index in enumerate(selected_indices)}
    for row in rows:
        validate_numeric_row(row)
        val_index = int(row["val_index"])
        require(
            int(row["selection_order"]) == order_by_index[val_index],
            "partial selection_order mismatch",
        )
        require(int(row["seed"]) == base_seed + val_index, "partial seed mismatch")
        require(int(row["ddpm_nfe"]) == NUM_TIMESTEPS, "partial NFE mismatch")
    for val_index in counts:
        target_rows = [row for row in rows if int(row["val_index"]) == val_index]
        require(
            [float(row["alpha"]) for row in target_rows] == list(ALPHAS),
            f"partial alpha order mismatch for val_index={val_index}",
        )
        require(
            len({row["sample_id"] for row in target_rows}) == 1,
            f"partial sample_id mismatch for val_index={val_index}",
        )
    rows.sort(key=lambda row: (int(row["selection_order"]), float(row["alpha"])))
    return rows, set(counts)


def evaluate_decodes(
    struct_tokenizer: torch.nn.Module,
    z_cont: torch.Tensor,
    predicted_z_quant: torch.Tensor,
    r_hat: torch.Tensor,
    diagnostic_mask: torch.Tensor,
    length: int,
) -> list[dict[str, float]]:
    oracle_decode = struct_tokenizer.detokenize(z_cont, res_mask=diagnostic_mask)
    validate_decoder_output("oracle continuous", oracle_decode, length)
    direct_bit_decode = struct_tokenizer.detokenize(
        predicted_z_quant, res_mask=diagnostic_mask
    )
    validate_decoder_output("direct Bit-only", direct_bit_decode, length)

    alpha_latents: dict[float, torch.Tensor] = {}
    alpha_decodes: dict[float, dict[str, torch.Tensor]] = {}
    for alpha in ALPHAS:
        z_alpha = predicted_z_quant + alpha * r_hat
        require(bool(torch.isfinite(z_alpha).all()), f"alpha={alpha:.2f} latent is non-finite")
        alpha_latents[alpha] = z_alpha
        decoded = struct_tokenizer.detokenize(z_alpha, res_mask=diagnostic_mask)
        validate_decoder_output(f"alpha={alpha:.2f}", decoded, length)
        alpha_decodes[alpha] = decoded

    require(
        torch.equal(alpha_latents[0.0], predicted_z_quant),
        "alpha=0 latent is not exactly predicted_z_quant",
    )
    alpha0_decode = alpha_decodes[0.0]
    require(
        torch.equal(alpha0_decode["atom37_mask"], direct_bit_decode["atom37_mask"]),
        "alpha=0 and direct Bit-only atom masks differ",
    )
    require(
        torch.allclose(
            alpha0_decode["atom37_positions"],
            direct_bit_decode["atom37_positions"],
            atol=0.0,
            rtol=0.0,
        ),
        "alpha=0 and direct Bit-only coordinates differ",
    )

    all_decodes = (oracle_decode,) + tuple(alpha_decodes[alpha] for alpha in ALPHAS)
    ca_metric_mask = common_residue_mask(all_decodes, diagnostic_mask, (CA_INDEX,))
    backbone_metric_mask = common_residue_mask(
        all_decodes, diagnostic_mask, BACKBONE_INDICES
    )
    metrics: list[dict[str, float]] = []
    for alpha in ALPHAS:
        metrics.append(
            {
                "alpha": alpha,
                "ca_rmsd": kabsch_atom_rmsd(
                    alpha_decodes[alpha],
                    oracle_decode,
                    ca_metric_mask,
                    (CA_INDEX,),
                    f"alpha={alpha:.2f} CA",
                ),
                "backbone_rmsd": kabsch_atom_rmsd(
                    alpha_decodes[alpha],
                    oracle_decode,
                    backbone_metric_mask,
                    BACKBONE_INDICES,
                    f"alpha={alpha:.2f} backbone",
                ),
            }
        )
    return metrics


def evaluate_target(
    *,
    dplm: torch.nn.Module,
    struct_tokenizer: torch.nn.Module,
    ddpm: torch.nn.Module,
    schedule: torch.nn.Module,
    sample: dict[str, Any],
    selection_order: int,
    val_index: int,
    base_seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    target_started = time.perf_counter()
    sample_id = str(sample["sample_id"])
    length = int(sample["length"])
    aatype = sample["aatype"].long()
    z_cont = sample["z_cont"].float().view(
        1, length, EXPECTED_CODEBOOK_DIM
    ).to(device)
    diagnostic_mask = sample["res_mask"].float().view(1, length).to(device)
    require(bool(torch.isfinite(z_cont).all()), "stored z_cont is non-finite")

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
        f"generated token shape is {tuple(final_tokens.shape)}",
    )
    predicted_z_quant, generated_mask = reconstruct_prediction_latent(dplm, final_tokens)
    require(
        tuple(predicted_z_quant.shape) == (1, length, EXPECTED_CODEBOOK_DIM),
        f"predicted z_quant shape is {tuple(predicted_z_quant.shape)}",
    )
    require(tuple(generated_mask.shape) == (1, length), "generated mask is misaligned")
    require(bool(generated_mask.bool().all()), "generated sample has masked residues")
    require(bool(torch.isfinite(predicted_z_quant).all()), "predicted z_quant is non-finite")

    hidden_states, extra_forward_seconds = timed_call(
        device,
        lambda: extract_final_token_condition(dplm, final_tokens, length),
    )
    require(len(hidden_states) == EXPECTED_NUM_LAYERS, "expected 33 hidden states")
    for index, hidden in enumerate(hidden_states):
        require(
            tuple(hidden.shape) == (1, length, EXPECTED_HIDDEN_SIZE),
            f"hidden state {index} shape mismatch",
        )

    require(
        torch.equal(generated_mask.bool(), diagnostic_mask.bool()),
        "stored and generated residue masks differ",
    )
    sample_seed = base_seed + val_index
    r_hat, sampler_metadata, sampling_seconds, global_rng_unchanged = run_sampler(
        ddpm,
        schedule,
        predicted_z_quant,
        hidden_states,
        generated_mask,
        sample_seed,
        device,
    )
    require(global_rng_unchanged, "local DDPM Generator changed global RNG")
    require(sampler_metadata["nfe"] == NUM_TIMESTEPS, "DDPM NFE is not 1000")
    require(
        sampler_metadata["timesteps"] == tuple(range(NUM_TIMESTEPS, 0, -1)),
        "DDPM reverse timestep order mismatch",
    )
    require(tuple(r_hat.shape) == tuple(predicted_z_quant.shape), "r_hat shape mismatch")
    require(bool(torch.isfinite(r_hat).all()), "r_hat is non-finite")

    metric_rows, decode_metric_seconds = timed_call(
        device,
        lambda: evaluate_decodes(
            struct_tokenizer,
            z_cont,
            predicted_z_quant,
            r_hat,
            diagnostic_mask,
            length,
        ),
    )
    target_total_seconds = time.perf_counter() - target_started
    rows: list[dict[str, Any]] = []
    for metric in metric_rows:
        row = {
            "selection_order": selection_order,
            "val_index": val_index,
            "sample_id": sample_id,
            "length": length,
            "seed": sample_seed,
            "alpha": metric["alpha"],
            "ca_rmsd": metric["ca_rmsd"],
            "backbone_rmsd": metric["backbone_rmsd"],
            "ddpm_nfe": int(sampler_metadata["nfe"]),
            "dplm_generation_seconds": generation_seconds,
            "extra_forward_seconds": extra_forward_seconds,
            "ddpm_sampling_seconds": sampling_seconds,
            "decode_metric_seconds": decode_metric_seconds,
            "target_total_seconds": target_total_seconds,
        }
        validate_numeric_row(row)
        rows.append(row)
    require([row["alpha"] for row in rows] == list(ALPHAS), "target alpha grid mismatch")
    print(
        f"[target {selection_order + 1}/{EXPECTED_SELECTION_SIZE}] "
        f"val_index={val_index} sample_id={sample_id} L={length} "
        f"seed={sample_seed} DDPM_calls=1 NFE={sampler_metadata['nfe']}"
    )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    summaries: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        require(len(alpha_rows) == EXPECTED_SELECTION_SIZE, f"alpha={alpha} N mismatch")
        ca_values = [float(row["ca_rmsd"]) for row in alpha_rows]
        backbone_values = [float(row["backbone_rmsd"]) for row in alpha_rows]
        summaries.append(
            {
                "alpha": alpha,
                "num_samples": len(alpha_rows),
                "mean_ca_rmsd": statistics.fmean(ca_values),
                "median_ca_rmsd": statistics.median(ca_values),
                "mean_backbone_rmsd": statistics.fmean(backbone_values),
                "median_backbone_rmsd": statistics.median(backbone_values),
                "selected": False,
            }
        )
    # ALPHAS is ascending, so Python min retains the smaller alpha on an exact tie.
    selected_alpha = min(
        ALPHAS,
        key=lambda alpha: next(
            float(row["mean_ca_rmsd"])
            for row in summaries
            if float(row["alpha"]) == alpha
        ),
    )
    for row in summaries:
        row["selected"] = float(row["alpha"]) == selected_alpha
    return summaries, selected_alpha


def unique_target_timing(rows: Sequence[dict[str, Any]], column: str) -> float:
    by_index: dict[int, float] = {}
    for row in rows:
        val_index = int(row["val_index"])
        value = float(row[column])
        if val_index in by_index:
            require(by_index[val_index] == value, f"repeated timing differs: {column}")
        else:
            by_index[val_index] = value
    require(len(by_index) == EXPECTED_SELECTION_SIZE, f"timing N mismatch: {column}")
    return sum(by_index.values())


def summary_text(
    summaries: Sequence[dict[str, Any]],
    selected_alpha: float,
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    current_run_seconds: float,
) -> str:
    lines = [
        "LOCAL MATCHED RESIDUAL DDPM INTERNAL-VALIDATION ALPHA SELECTION",
        "",
        "[FIXED PROTOCOL]",
        f"DDPM checkpoint path: {metadata['ddpm_checkpoint_path']}",
        f"optimizer step: {metadata['optimizer_step']}",
        f"best step: {metadata['best_step']}",
        f"DDPM parameters: {metadata['ddpm_parameters']}",
        f"DDPM sampler: {metadata['sampler']}",
        f"DDPM NFE: {metadata['ddpm_nfe']}",
        f"alpha grid: {metadata['alpha_grid']}",
        f"validation population: {metadata['validation_population']}",
        f"selection N: {metadata['selection_size']}",
        f"indices artifact path: {metadata['indices_artifact_path']}",
        f"indices artifact SHA256: {metadata['indices_artifact_sha256']}",
        f"seed policy: {metadata['seed_policy']}",
        f"DPLM checkpoint: {metadata['dplm_checkpoint']}",
        f"generation max_iter: {metadata['generation_max_iter']}",
        "generation: deterministic argmax",
        f"selection metric: {metadata['selection_metric']}",
        f"tie rule: {metadata['tie_rule']}",
        f"TM-score: {metadata['tm_score']}",
        f"CAMEO/PDB-date used: {metadata['cameo_or_pdb_date_used']}",
        "",
        "[ALPHA SUMMARY]",
    ]
    for row in summaries:
        lines.extend(
            [
                f"alpha={float(row['alpha']):.2f}",
                f"  N: {row['num_samples']}",
                f"  mean CA RMSD: {float(row['mean_ca_rmsd']):.9g}",
                f"  median CA RMSD: {float(row['median_ca_rmsd']):.9g}",
                f"  mean backbone RMSD: {float(row['mean_backbone_rmsd']):.9g}",
                f"  median backbone RMSD: {float(row['median_backbone_rmsd']):.9g}",
            ]
        )
    lines.extend(
        [
            "",
            "[SELECTION]",
            f"selected alpha: {selected_alpha:.2f}",
            "selection criterion: lowest mean CA RMSD",
            "tie rule: exact tie keeps the smaller alpha in ascending grid order",
            "",
            "[TIMING; DIAGNOSTIC ONLY, NOT AN OFFICIAL BENCHMARK]",
            f"total DPLM generation seconds: {unique_target_timing(rows, 'dplm_generation_seconds'):.6f}",
            f"total extra forward seconds: {unique_target_timing(rows, 'extra_forward_seconds'):.6f}",
            f"total DDPM sampling seconds: {unique_target_timing(rows, 'ddpm_sampling_seconds'):.6f}",
            f"total decode/metric seconds: {unique_target_timing(rows, 'decode_metric_seconds'):.6f}",
            f"total target-processing seconds: {unique_target_timing(rows, 'target_total_seconds'):.6f}",
            f"current invocation elapsed seconds: {current_run_seconds:.6f}",
            "",
        ]
    )
    return "\n".join(lines)


def validate_complete_rows(
    rows: Sequence[dict[str, Any]], selected_indices: Sequence[int]
) -> None:
    require(len(rows) == EXPECTED_SELECTION_SIZE * len(ALPHAS), "expected 500 rows")
    counts = Counter(int(row["val_index"]) for row in rows)
    require(set(counts) == set(selected_indices), "completed val_index set mismatch")
    require(all(count == len(ALPHAS) for count in counts.values()), "target row count mismatch")
    for selection_order, val_index in enumerate(selected_indices):
        target_rows = [row for row in rows if int(row["val_index"]) == val_index]
        require(
            [float(row["alpha"]) for row in target_rows] == list(ALPHAS),
            f"final alpha order mismatch for val_index={val_index}",
        )
        require(
            all(int(row["selection_order"]) == selection_order for row in target_rows),
            "final selection_order mismatch",
        )
        require(all(int(row["ddpm_nfe"]) == NUM_TIMESTEPS for row in target_rows), "NFE mismatch")
        for row in target_rows:
            validate_numeric_row(dict(row))


def main() -> int:
    invocation_started = time.perf_counter()
    args = parse_args()
    require(args.base_seed == 42, "fairness protocol requires --base_seed 42")
    require(GENERATION_MAX_ITER == 100, "folding generation requires max_iter=100")
    require(ALPHAS == (0.00, 0.25, 0.50, 0.75, 1.00), "alpha grid changed")

    dataset_dir = Path(args.dataset_dir)
    split_csv = Path(args.split_csv)
    indices_path = Path(args.indices_csv)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint_best.pt"
    result_path = output_dir / RESULT_NAME
    summary_text_path = output_dir / SUMMARY_TEXT_NAME
    summary_csv_path = output_dir / SUMMARY_CSV_NAME
    metadata_path = output_dir / METADATA_NAME

    selected_indices, indices_digest = load_fixed_indices(indices_path, split_csv)
    device = resolve_device(args.device)
    seed_everything(args.base_seed)
    checks: dict[str, bool] = {}
    dplm, struct_tokenizer = load_models(args.dplm_checkpoint, device, checks)
    ddpm, schedule, checkpoint_metadata = load_ddpm(checkpoint_path, device)
    require(
        int(checkpoint_metadata["optimizer_step"]) == EXPECTED_OPTIMIZER_STEP,
        "checkpoint optimizer step mismatch",
    )
    require(int(checkpoint_metadata["best_step"]) == EXPECTED_BEST_STEP, "best step mismatch")
    require(
        int(checkpoint_metadata["parameter_count"]) == EXPECTED_DDPM_PARAMETERS,
        "DDPM parameter count mismatch",
    )
    expected_metadata = run_metadata(
        args,
        checkpoint_path,
        checkpoint_metadata,
        indices_digest,
        selected_indices,
    )
    validate_or_create_metadata(metadata_path, expected_metadata)
    rows, completed = load_partial_rows(result_path, selected_indices, args.base_seed)
    print(f"fixed FM indices SHA256: {indices_digest}")
    print(f"validation population: {EXPECTED_VALIDATION_POPULATION}")
    print(f"selected validation indices: {len(selected_indices)}")
    print(f"resume completed targets: {len(completed)}")

    with torch.inference_mode():
        for selection_order, val_index in enumerate(selected_indices):
            if val_index in completed:
                print(
                    f"[resume skip {selection_order + 1}/{EXPECTED_SELECTION_SIZE}] "
                    f"val_index={val_index}"
                )
                continue
            sample = load_validation_sample(dataset_dir, split_csv, val_index)
            target_rows = evaluate_target(
                dplm=dplm,
                struct_tokenizer=struct_tokenizer,
                ddpm=ddpm,
                schedule=schedule,
                sample=sample,
                selection_order=selection_order,
                val_index=val_index,
                base_seed=args.base_seed,
                device=device,
            )
            rows.extend(target_rows)
            rows.sort(key=lambda row: (int(row["selection_order"]), float(row["alpha"])))
            atomic_write_csv(result_path, RESULT_COLUMNS, rows)
            completed.add(val_index)
            del sample, target_rows

    synchronize(device)
    require(all_grads_none(dplm), "DPLM accumulated gradients")
    require(all_grads_none(struct_tokenizer), "tokenizer accumulated gradients")
    require(all_grads_none(ddpm), "DDPM accumulated gradients")
    validate_complete_rows(rows, selected_indices)
    summaries, selected_alpha = summarize(rows)
    atomic_write_csv(summary_csv_path, SUMMARY_COLUMNS, summaries)
    current_run_seconds = time.perf_counter() - invocation_started
    report = summary_text(
        summaries,
        selected_alpha,
        expected_metadata,
        rows,
        current_run_seconds,
    )
    atomic_write_text(summary_text_path, report)

    print("\n" + report.rstrip())
    print(f"per-target CSV: {result_path}")
    print(f"summary CSV: {summary_csv_path}")
    print(f"summary text: {summary_text_path}")
    print(f"resume metadata: {metadata_path}")
    print("\nDDPM_INTERNAL_VAL_ALPHA_SWEEP_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

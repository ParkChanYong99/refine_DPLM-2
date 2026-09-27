#!/usr/bin/env python3
"""Read-only, CPU-only preflight for the frozen final matched-DDPM evaluation.

This audit inspects source, metadata, the trained checkpoint, and existing FM/Bit
artifacts. It never generates tokens, samples a DDPM trajectory, decodes a
structure, calculates target metrics, or writes an output file.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from experiments.latent_residual_ddpm import evaluate_final_ddpm as final
from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_fm import generate_final_folding_predictions as fm
from experiments.latent_residual_fm.compare_cameo_bit_vs_fm import (
    collect_top_samples,
    read_aggregate,
    validate_aggregate,
)


REPO = Path(__file__).resolve().parents[2]
FINAL_SOURCE = REPO / "experiments/latent_residual_ddpm/evaluate_final_ddpm.py"
SAMPLER_SOURCE = REPO / "experiments/latent_residual_ddpm/ddpm_sampling.py"
TRAIN_SOURCE = REPO / "experiments/latent_residual_ddpm/train_latent_residual_ddpm.py"
SELECT_SOURCE = REPO / "experiments/latent_residual_ddpm/select_internal_val_alpha.py"
FM_SOURCE = REPO / "experiments/latent_residual_fm/generate_final_folding_predictions.py"
EXPECTED_ARCHITECTURE = {
    "class": "LatentResidualDDPM",
    "residual_dim": 13,
    "condition_dim": 1280,
    "hidden_dim": 1024,
    "num_layers": 6,
    "num_hidden_states": 33,
    "total_parameters": 34_946_606,
    "trainable_parameters": 34_946_606,
}
EXPECTED_SCHEDULE = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "schedule": "linear",
    "prediction_type": "epsilon",
}
KNOWN_BIT = {
    "cameo2022": (6.28888164, 0.841421163),
    "PDB_date": (3.11376375, 0.91379618),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_source(path: Path) -> tuple[str, ast.Module]:
    require(path.is_file(), f"required source missing: {path}")
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def function_source(source: str, tree: ast.Module, name: str) -> str:
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
    require(len(functions) == 1, f"expected exactly one {name} definition")
    segment = ast.get_source_segment(source, functions[0])
    require(segment is not None, f"cannot inspect {name} source")
    return segment


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def contains(source: str, fragment: str, label: str) -> None:
    require(compact(fragment) in compact(source), f"source contract missing: {label}")


def static_protocol_audit() -> None:
    source, tree = parse_source(FINAL_SOURCE)
    fm_source, fm_tree = parse_source(FM_SOURCE)
    sampler_source, sampler_tree = parse_source(SAMPLER_SOURCE)
    train_source, train_tree = parse_source(TRAIN_SOURCE)
    select_source, select_tree = parse_source(SELECT_SOURCE)

    require(final.FINAL_ALPHA == 0.50 and final.BASE_SEED == 42,
            "final alpha/base seed changed")
    require(final.EXPECTED_COUNTS == {"cameo2022": 163, "PDB_date": 442},
            "final target counts changed")
    require(final.DDPM_CHECKPOINT == Path(
        "experiments/latent_residual_ddpm/runs/ddpm_modeA_current_main/checkpoint_best.pt"),
        "final DDPM checkpoint path changed")
    require(final.OUTPUT_ROOT == Path(
        "generation-results/dplm2_bit_650m_final/ddpm_matched"),
        "final output root changed")
    require(final.DPLM_CHECKPOINT == fm.DPLM_CHECKPOINT == "airkingbd/dplm2_bit_650m",
            "DPLM checkpoint differs from finalized FM")
    require(final.DPLM_MAX_ITER == fm.DPLM_MAX_ITER == 100,
            "DPLM max_iter differs from finalized FM")
    require(final.MIN_LENGTH == fm.MIN_LENGTH == 60 and
            final.MAX_LENGTH == fm.MAX_LENGTH == 512,
            "length eligibility differs from finalized FM")
    require(final.DATASET_FASTAS == fm.DATASET_FASTAS,
            "final FASTA paths differ from finalized FM")
    require(final.NUM_TIMESTEPS == 1000, "final DDPM T/NFE changed")
    require(final.PREDICTION_SOURCE == "finalized_generation_manifest",
            "final prediction source is not the frozen manifest")
    require(final.REFERENCE_EVAL_DIRS["cameo2022"] == final.CAMEO_REFERENCE_EVAL
            and final.REFERENCE_EVAL_DIRS["PDB_date"] == final.PDB_DATE_REFERENCE_EVAL,
            "final Bit reference paths changed")

    target_fn = function_source(source, tree, "target_roster")
    for fragment in (
        "targets = read_targets(DATASET_FASTAS[dataset])",
        "existing = read_manifest(fm_layout.manifest)",
        "row = existing[target.fasta_index]",
        'int(row["sample_seed"]) == BASE_SEED + target.fasta_index',
        "set(reference) == {target.sample_id for target in eligible}",
    ):
        contains(target_fn, fragment, "FM target/order/seed/Bit roster")
    contains(function_source(fm_source, fm_tree, "read_targets"),
             'enumerate(SeqIO.parse(str(path), "fasta"))', "zero-based FM FASTA index")
    contains(function_source(fm_source, fm_tree, "settings"),
             '"base_seed": str(args.seed)', "FM base seed")
    contains(function_source(fm_source, fm_tree, "default_manifest_row"),
             '"sample_seed": args.seed + target.fasta_index', "FM sample seed")

    eval_fn = function_source(source, tree, "evaluate_target")
    manifest_fn = function_source(source, tree, "manifest_raw_indices")
    for fragment in (
        'row["struct_token_sequence"].split(",")',
        "len(lexemes) == target.length",
        "0 <= index < 8192",
    ):
        contains(manifest_fn, fragment, "frozen raw LFQ manifest tokens")
    target_fn = function_source(source, tree, "target_roster")
    for fragment in (
        "manifest_raw_indices(row, target)",
        "PAIRED_REFERENCE_CSVS[dataset]",
        "reference_metric_tuple(target, reference)",
        "sha256_file(fm_layout.manifest)",
    ):
        contains(target_fn, fragment, "frozen target/Bit artifact provenance")
    for fragment in (
        "sample_seed = BASE_SEED + target.fasta_index",
        "aatype = torch.tensor(du.seq_to_aatype(target.sequence)",
        "with torch.inference_mode():",
        "build_folding_input(dplm, aatype, device)",
        'dplm.tokenizer._token_to_id[lexeme]',
        "token_id - offset == raw_index",
        "final_tokens[:, 1:target.length + 1] =",
        "torch.equal(final_tokens[partial_mask], input_tokens[partial_mask])",
        "reconstruct_prediction_latent(dplm, final_tokens)",
        "extract_final_token_condition(dplm, final_tokens, target.length)",
        "len(hidden_states) == EXPECTED_NUM_LAYERS == 33",
        "bit_metrics = reference_metric_tuple(target, reference)",
        "run_sampler(",
        'sampling_metadata["nfe"] == NUM_TIMESTEPS == 1000',
        'sampling_metadata["timesteps"] == tuple(range(1000, 0, -1))',
        "z_refined = predicted_z_quant + FINAL_ALPHA * r_hat",
        "tokenizer.detokenize(z_refined, res_mask=residue_mask)",
        "save_ddpm_pdb_atomically(",
        "ddpm_metrics = metric_triplet(ddpm_path, target, gt_metadata)",
        '"manifest_token_sha256": token_digest',
    ):
        contains(eval_fn, fragment, f"frozen-manifest inference: {fragment[:55]}")
    require("tokenizer.detokenize(predicted_z_quant" not in eval_fn,
            "final evaluation recreates a Bit baseline decode")
    require("bit_metrics = metric_triplet(" not in eval_fn,
            "final evaluation recalculates the frozen Bit baseline")
    generate_calls = [
        item for item in ast.walk(tree)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        and item.func.attr == "generate"
    ]
    require(not generate_calls, "final evaluator performs new DPLM generation")
    metric_fn = function_source(source, tree, "metric_triplet")
    contains(metric_fn, "load_pdb_by_name(target.sample_id, gt_metadata)",
             "official GT loader")
    contains(metric_fn, "evaluator_utils.process_folded_outputs(str(path), None, true_bb)",
             "official CA/BB/TM metric helper")
    for fragment in (
        '"ca_improvement": bit_metrics[0] - ddpm_metrics[0]',
        '"backbone_improvement": bit_metrics[1] - ddpm_metrics[1]',
        '"tm_improvement": ddpm_metrics[2] - bit_metrics[2]',
    ):
        contains(eval_fn, fragment, "positive paired-change sign")

    run_sampler_source, run_sampler_tree = parse_source(
        REPO / "experiments/latent_residual_ddpm/debug_inference_integration.py")
    runner_fn = function_source(run_sampler_source, run_sampler_tree, "run_sampler")
    contains(runner_fn, "generator = torch.Generator(device=device)", "local DDPM RNG")
    contains(runner_fn, "generator.manual_seed(seed)", "target-specific DDPM seed")
    contains(runner_fn, "sample_residual_ddpm(", "full DDPM sampler call")
    sampler_fn = function_source(sampler_source, sampler_tree, "sample_residual_ddpm")
    contains(sampler_fn, "for timestep in range(schedule.num_timesteps, 0, -1)",
             "complete ancestral 1000-to-1 traversal")
    contains(sampler_fn, '"nfe": len(timestep_order)', "full sampler NFE metadata")
    contains(sampler_fn, "generator=generator", "sampler local generator") if False else None
    forbidden = {"ddim", "eta", "dynamic_threshold", "clip", "clamp", "skip_timesteps"}
    for label, node in (("final evaluation", tree), ("DDPM sampler", sampler_tree)):
        names = {item.id.lower() for item in ast.walk(node) if isinstance(item, ast.Name)}
        names |= {item.attr.lower() for item in ast.walk(node) if isinstance(item, ast.Attribute)}
        names |= {item.arg.lower() for item in ast.walk(node)
                  if isinstance(item, ast.keyword) and item.arg is not None}
        require(not names.intersection(forbidden),
                f"{label} contains forbidden sampler option: {sorted(names.intersection(forbidden))}")

    metadata_fn = function_source(source, tree, "protocol_metadata")
    for key in (
        "ddpm_checkpoint_sha256", "optimizer_step", "best_val_loss", "alpha",
        "sampler", "T", "NFE", "eligible_targets", "fm_generation_manifest_sha256",
        "seed_policy", "metric_source_sha256", "dplm_checkpoint",
        "generation_max_iter_provenance", "CAMEO_PDB_date_used_for_tuning",
        "prediction_source", "prediction_manifest_path", "prediction_manifest_sha256",
        "new_dplm_generation", "frozen_manifest_tokens_reused",
        "reference_bit_top_samples_sha256", "extra_frozen_dplm_forward",
    ):
        contains(metadata_fn, f'"{key}"', f"resume metadata key {key}")
    contains(metadata_fn, '"new_dplm_generation": False', "no new DPLM generation metadata")
    contains(metadata_fn, '"frozen_manifest_tokens_reused": True', "frozen tokens metadata")
    contains(function_source(source, tree, "load_or_create_metadata"),
             "stored == expected", "strict resume metadata equality")
    contains(function_source(source, tree, "load_or_create_metadata"),
             'stored.get("prediction_source") != PREDICTION_SOURCE',
             "legacy regeneration metadata is rejected")
    load_partial_fn = function_source(source, tree, "load_partial")
    for fragment in (
        "validate_row(row, target, dataset, reference, manifest[index])",
        "valid_pdb(ddpm_path, target.length)",
    ):
        contains(load_partial_fn, fragment, "partial result validation")
    main_fn = function_source(source, tree, "main")
    for fragment in (
        "load_or_create_metadata(", "load_partial(", "evaluate_target(",
        "atomic_write_csv(result_path, PER_TARGET_COLUMNS, ordered)",
    ):
        contains(main_fn, fragment, "resume-before-evaluate / atomic per-target CSV")
    require(compact(main_fn).index("load_partial(") < compact(main_fn).index("evaluate_target("),
            "partial CSV is not loaded before target evaluation")
    require(compact(main_fn).index("evaluate_target(") < compact(main_fn).index(
        "atomic_write_csv(result_path,PER_TARGET_COLUMNS,ordered)"),
        "completed target is not atomically persisted")
    parse_args_fn = function_source(source, tree, "parse_args")
    require("--alpha" not in parse_args_fn and "--sampler" not in parse_args_fn,
            "final test exposes alpha or sampler tuning CLI")
    final_alpha_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(name, ast.Name) and name.id == "FINAL_ALPHA"
                for target in node.targets
                for name in ast.walk(target)
            ):
                final_alpha_assignments.append(node)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if isinstance(node.target, ast.Name) and node.target.id == "FINAL_ALPHA":
                final_alpha_assignments.append(node)
    require(len(final_alpha_assignments) == 1,
            "final alpha is reassigned or its declaration is missing")
    contains(function_source(source, tree, "validate_frozen_alpha"),
             'float(selected[0]["alpha"]) == FINAL_ALPHA', "internal alpha lock")
    contains(function_source(select_source, select_tree, "run_metadata"),
             '"cameo_or_pdb_date_used": False', "alpha selection has no test data")
    require("data-bin/cameo" not in train_source.lower() and
            "data-bin/pdb_date" not in train_source.lower(),
            "training source references test FASTA/data")
    require("data-bin/cameo" not in select_source.lower() and
            "data-bin/pdb_date" not in select_source.lower(),
            "alpha-selection source references test FASTA/data")
    require("selection_metric" not in eval_fn and "best_alpha" not in eval_fn,
            "test-time branch contains alpha selection")


def checkpoint_audit() -> tuple[int, int, float, int]:
    path = REPO / final.DDPM_CHECKPOINT
    require(path.is_file(), f"DDPM checkpoint missing: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    require(isinstance(checkpoint, dict), "DDPM checkpoint is not a dictionary")
    required = {
        "ddpm_model_state_dict", "ddpm_schedule_state_dict",
        "ddpm_architecture_configuration", "ddpm_schedule_configuration",
        "optimizer_step", "best_step", "best_val_loss", "split_path",
        "dataset_path", "checkpoint_selection_metric", "training_configuration",
    }
    require(required.issubset(checkpoint),
            f"checkpoint missing keys: {sorted(required.difference(checkpoint))}")
    optimizer_step = int(checkpoint["optimizer_step"])
    best_step = int(checkpoint["best_step"])
    best_val_loss = float(checkpoint["best_val_loss"])
    require(optimizer_step == best_step == 100_000, "checkpoint step is not 100000")
    require(math.isfinite(best_val_loss) and
            math.isclose(best_val_loss, 0.150297817, rel_tol=0.0, abs_tol=5e-10),
            "checkpoint best val loss does not round to 0.150297817")
    require(checkpoint["ddpm_architecture_configuration"] == EXPECTED_ARCHITECTURE,
            "checkpoint DDPM architecture differs from frozen specification")
    require(checkpoint["ddpm_schedule_configuration"] == EXPECTED_SCHEDULE,
            "checkpoint DDPM schedule differs from full ancestral protocol")
    model = LatentResidualDDPM()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    require(parameters == 34_946_606, "instantiated DDPM parameter count differs")
    model.load_state_dict(checkpoint["ddpm_model_state_dict"], strict=True)
    schedule = DDPMSchedule(num_timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    schedule.load_state_dict(checkpoint["ddpm_schedule_state_dict"], strict=True)
    require(schedule.num_timesteps == 1000, "loaded DDPM horizon differs")
    require(checkpoint["checkpoint_selection_metric"] ==
            "deterministic_internal_val_masked_epsilon_mse",
            "checkpoint was selected using another metric")
    require(checkpoint["dataset_path"] ==
            "data-bin/latent_residual_fm/afdb_l512_sharded" and
            checkpoint["split_path"] ==
            "data-bin/latent_residual_fm/afdb_l512_sharded/splits/split_v1.csv",
            "training checkpoint used a non-internal dataset/split")
    require(checkpoint["training_configuration"].get("max_optimizer_steps") == 100_000,
            "training horizon differs from 100k")
    return optimizer_step, best_step, best_val_loss, parameters


def alpha_audit() -> None:
    path = REPO / final.ALPHA_SELECTION_ARTIFACT
    require(path.is_file(), f"internal alpha artifact missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require({"alpha", "num_samples", "mean_ca_rmsd", "selected"}.issubset(
            reader.fieldnames or ()), "internal alpha CSV schema changed")
        rows = list(reader)
    require(len(rows) == 5, "internal alpha grid has wrong size")
    require([float(row["alpha"]) for row in rows] ==
            [0.0, 0.25, 0.5, 0.75, 1.0], "internal alpha grid changed")
    require(all(int(row["num_samples"]) == 100 for row in rows),
            "alpha-selection target count changed")
    selected = [row for row in rows if row["selected"].strip().lower() == "true"]
    require(len(selected) == 1 and float(selected[0]["alpha"]) == 0.50,
            "internal validation did not select alpha=0.50")
    require(float(selected[0]["mean_ca_rmsd"]) ==
            min(float(row["mean_ca_rmsd"]) for row in rows),
            "selected alpha is not lowest mean CA RMSD")
    metadata_path = path.with_name("internal_val_alpha_sweep_100_metadata.json")
    require(metadata_path.is_file(), "internal alpha-selection metadata missing")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    require(metadata["dataset_role"] == "internal AFDB validation only" and
            metadata["cameo_or_pdb_date_used"] is False,
            "alpha selection metadata indicates test leakage")


def targets_and_baseline_audit() -> dict[str, tuple[int, float, float]]:
    output: dict[str, tuple[int, float, float]] = {}
    for dataset, expected_count in final.EXPECTED_COUNTS.items():
        targets = fm.read_targets(fm.DATASET_FASTAS[dataset])
        eligible = [target for target in targets if 60 <= target.length <= 512]
        require(len(eligible) == expected_count,
                f"{dataset} eligible target count changed: {len(eligible)}")
        manifest_path = fm.output_layout(dataset).manifest
        manifest = fm.read_manifest(manifest_path)
        require(list(manifest) == list(range(len(targets))),
                f"{dataset} FM manifest index/order differs from FASTA")
        for target in targets:
            row = manifest[target.fasta_index]
            require(row["sample_id"] == target.sample_id and
                    row["aa_sequence"] == target.sequence and
                    int(row["length"]) == target.length,
                    f"{dataset} FM manifest target differs at {target.fasta_index}")
            require(int(row["sample_seed"]) == 42 + target.fasta_index,
                    f"{dataset} FM manifest seed differs for {target.sample_id}")
            if target.eligible:
                final.manifest_raw_indices(row, target)
            if target in eligible:
                require(row["status"] == "success" and
                        row["generation_success"].lower() == "true",
                        f"{dataset} finalized FM target incomplete: {target.sample_id}")
        label = f"{dataset} Local Bit"
        reference_dir = final.REFERENCE_EVAL_DIRS[dataset]
        reference = collect_top_samples(reference_dir, label)
        require(len(reference) == expected_count and
                set(reference) == {target.sample_id for target in eligible},
                f"{dataset} Local Bit target IDs differ from FM eligible roster")
        for target in eligible:
            require(reference[target.sample_id].length == target.length,
                    f"{dataset} Local Bit target length differs: {target.sample_id}")
        aggregate = read_aggregate(reference_dir, label)
        validate_aggregate(label, reference, aggregate)
        known_bb, known_tm = KNOWN_BIT[dataset]
        require(math.isclose(aggregate.mean_bb_rmsd, known_bb, abs_tol=1e-5) and
                math.isclose(aggregate.mean_bb_tmscore, known_tm, abs_tol=1e-5),
                f"{dataset} wrong Local Bit reference artifact")
        output[dataset] = (len(eligible), aggregate.mean_bb_rmsd,
                           aggregate.mean_bb_tmscore)
    return output


def resume_output_audit() -> list[Path]:
    root = final.OUTPUT_ROOT
    require(root != fm.OUTPUT_ROOT and root.is_relative_to(fm.OUTPUT_ROOT),
            "final DDPM output overlaps finalized FM root")
    expected = [
        root / "cameo_per_target.csv", root / "pdb_date_per_target.csv",
        root / "summary.csv", root / "summary.txt",
        root / "cameo2022" / final.METADATA_NAME,
        root / "PDB_date" / final.METADATA_NAME,
    ]
    require(len(set(expected)) == len(expected), "output artifact paths collide")
    require(final.PER_TARGET_COLUMNS[:6] ==
            ("benchmark", "target_id", "target_index", "length", "seed",
             "manifest_token_sha256"),
            "per-target output identity/token digest schema changed")
    require({"bit_ca_rmsd", "ddpm_ca_rmsd", "ca_improvement",
             "bit_backbone_rmsd", "ddpm_backbone_rmsd", "backbone_improvement",
             "bit_tm_score", "ddpm_tm_score", "tm_improvement", "ddpm_nfe",
             "alpha"}.issubset(final.PER_TARGET_COLUMNS),
            "per-target output metric schema incomplete")
    for dataset in final.EXPECTED_COUNTS:
        metadata_path = root / dataset / final.METADATA_NAME
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
            require(stored.get("prediction_source") == final.PREDICTION_SOURCE and
                    stored.get("new_dplm_generation") is False and
                    stored.get("frozen_manifest_tokens_reused") is True,
                    f"legacy/incompatible resume metadata remains: {metadata_path}; "
                    "archive it manually before frozen-manifest evaluation")
    return expected


def main() -> int:
    print("[FINAL DDPM EVALUATION PREFLIGHT AUDIT]")
    static_protocol_audit()
    optimizer_step, best_step, best_val_loss, parameters = checkpoint_audit()
    alpha_audit()
    baselines = targets_and_baseline_audit()
    artifacts = resume_output_audit()
    print(f"checkpoint: {final.DDPM_CHECKPOINT}")
    print(f"optimizer step: {optimizer_step}")
    print(f"best step: {best_step}")
    print(f"best val loss: {best_val_loss:.9f}")
    print(f"DDPM params: {parameters:,}")
    print("sampler: full ancestral DDPM")
    print("T: 1000")
    print("NFE: 1000")
    print("alpha: 0.50 (selected on internal validation only)")
    for dataset, (count, bb_rmsd, tm_score) in baselines.items():
        print(f"{dataset} target count: {count}")
        print(f"{dataset} Local Bit backbone RMSD: {bb_rmsd:.9g}")
        print(f"{dataset} Local Bit TM-score: {tm_score:.9g}")
    print("target order matches finalized FM: True")
    print("seed policy: 42 + zero-based FASTA target index; matches FM manifest")
    print(f"DPLM checkpoint: {final.DPLM_CHECKPOINT}")
    print(f"generation max_iter: {final.DPLM_MAX_ITER}")
    print("generation provenance: deterministic argmax, max_iter=100 (no new generation)")
    print("prediction source: finalized_generation_manifest")
    print("new DPLM generation: False; frozen manifest tokens reused: True")
    print("extra frozen DPLM forward: True; 33 structure-side hidden states")
    print("Bit baseline artifact: finalized Local Bit top_sample.csv and forward_fold_metrics.csv")
    print("Bit target consistency: PASS")
    print("metrics: official CA RMSD, backbone RMSD, TM-score")
    print("paired-change sign: Bit-DDPM RMSD; DDPM-Bit TM-score")
    print("resume metadata complete: True")
    print("test-time tuning absent: True")
    print("CAMEO/PDB-date tuning leakage absent: True")
    print(f"output root: {final.OUTPUT_ROOT}")
    for path in artifacts:
        print(f"expected output artifact: {path}")
    print("FINAL_DDPM_EVALUATION_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

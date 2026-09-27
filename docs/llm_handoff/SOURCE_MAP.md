# Handoff Source Map

This file records the main uploaded sources used to build `PROJECT_CONTEXT.md`.

## Manuscript / protocol

- `methods_section_draft(1).md`
  - detailed finalized local Methods
- `results_section_final_draft(1).md`
  - local results, paired inference, claim boundaries
- `local_method_comparison_statistics(1).md`
  - primary/secondary paired statistical results
- `MATCHED_DDPM_SPEC(2).md`
  - historical pre-registration/design specification

## FM implementation

- `build_afdb_latent_shards(2).py`
- `audit_final_fm_protocol(1).py`
- `train_latent_residual_fm.py`
- `flow_matching.py`
- `dplm_condition.py`
- `fm_model.py`
- `generate_final_folding_predictions(1).py`

## DDPM / DDIM implementation

- `ddpm_training(1).py`
- `ddpm_diffusion(1).py`
- `ddpm_model(1).py`
- `train_latent_residual_ddpm(1).py`
- `evaluate_final_ddpm(1).py`
- `evaluate_ddpm_100nfe(1).py`
- `benchmark_fm_vs_ddpm_latency.py`
- `benchmark_fm_vs_ddim100_interleaved.py`

## Final compact artifacts

- `metadata(1).json`
- `summary(1).txt`
- `latency_summary.txt`
- `interleaved_summary.txt`

## External papers

- `DPLM-2(2).pdf`
  - DPLM-2 paper
- `DPLM2.1(3).pdf`
  - DPLM-2.1 / "Elucidating the Design Space of Multimodal Protein Language Models"

## Important precedence note

`MATCHED_DDPM_SPEC(2).md` is historical design intent. Later implemented Python code, final Methods/Results, metadata, and result summaries supersede its initial "design only" execution status.

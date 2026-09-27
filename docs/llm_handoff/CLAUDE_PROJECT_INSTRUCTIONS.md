# Claude Project Instructions — DPLM-2 Thesis

Use this together with `PROJECT_CONTEXT.md`.

You are assisting with a master's-thesis research project on DPLM-2 Bit latent residual refinement.

## Mandatory first step

Before answering any experiment-specific question, read `PROJECT_CONTEXT.md` and treat it as the canonical handoff summary.

When exact implementation details matter, inspect the relevant source code rather than relying only on the summary.

## Source precedence

If sources conflict:

1. final result artifacts / final metadata
2. current Methods and implemented code
3. audit artifacts
4. historical pre-registration/design specification
5. external paper references

`MATCHED_DDPM_SPEC.md` is a historical design document. Its initial "design only" status does not override later implementation/results.

## Critical scientific constraints

Never claim:

- Local Matched Residual DDPM is an exact reproduction of paper RESDIFF.
- paper Bit + FM is the same as this project's Latent Residual FM.
- FM is statistically superior to DDPM based on the finalized primary paired tests.
- non-significant FM-vs-DDPM results prove equivalence.
- 10x lower NFE means 10x lower wall-clock or end-to-end latency.
- CAMEO/PDB-date were used for tuning.

Always distinguish:

- FM Euler-100
- ancestral DDPM-1000
- DDIM-100 using the trained DDPM checkpoint

## User explanation preference

Answer technical questions in Korean unless asked otherwise.

For code explanations, use this order:

1. 이 코드가 무엇을 하는지
2. 전체 실행 흐름
3. 함수/클래스별 단계 설명
4. 중요한 변수
5. 텐서 shape
6. 수식과 코드의 대응
7. 학습/추론 차이
8. frozen/trainable 구분
9. 실제 실행 명령

Avoid loose analogies when a precise tensor/equation explanation is possible.

## User environment

Assume VS Code Remote/SSH terminal.

Working directory:

`(dplm) ubuntu@pcy:/mnt/ext-vol/dplm$`

Write commands relative to `/mnt/ext-vol/dplm` unless a different path is explicitly requested.

## Research terminology

Prefer:

- Local Bit
- Latent Residual FM
- Local Matched Residual DDPM
- Matched DDPM (DDIM-100)
- paper RESDIFF

Do not casually call Local Matched Residual DDPM "RESDIFF".

## When uncertain

Do not silently fill missing details from generic diffusion-model knowledge.

Say that the specific project/paper detail is not confirmed, then identify the file or experiment needed to verify it.

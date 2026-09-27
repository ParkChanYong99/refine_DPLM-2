# ChatGPT ↔ Claude DPLM-2 Shared-Project Setup Guide

This guide creates a practical shared source-of-truth workflow.

Recommended architecture:

- **GitHub private repository**: code + canonical Markdown experiment context
- **Optional Google Drive**: papers, manuscript drafts, presentation assets, large result documents
- **ChatGPT Project and Claude Project**: both consume the same canonical sources

The ChatGPT and Claude project conversations themselves do not need to be synchronized.

---

# 1. Put the handoff files in the local DPLM repository

In VS Code terminal:

```bash
cd /mnt/ext-vol/dplm

mkdir -p docs/llm_handoff
```

Copy these downloaded files into:

```text
/mnt/ext-vol/dplm/docs/llm_handoff/
```

Recommended names:

```text
docs/llm_handoff/PROJECT_CONTEXT.md
docs/llm_handoff/CLAUDE_PROJECT_INSTRUCTIONS.md
docs/llm_handoff/SYNC_SETUP_GUIDE.md
```

Check:

```bash
cd /mnt/ext-vol/dplm
ls -lh docs/llm_handoff
```

---

# 2. Keep canonical project files in Git

First inspect repository status:

```bash
cd /mnt/ext-vol/dplm
git status
```

If this directory is already a Git repository, do not run `git init` again.

Add the handoff files:

```bash
git add docs/llm_handoff/
git commit -m "Add DPLM experiment handoff context for shared LLM workflow"
```

If you already have a private GitHub remote:

```bash
git remote -v
git push
```

If you do not yet have a GitHub repository, create a **private** repository on GitHub, then connect it:

```bash
cd /mnt/ext-vol/dplm

git remote add origin <YOUR_PRIVATE_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

Do not commit checkpoints, datasets, generated structures, secrets, tokens, or huge result folders unless you intentionally use Git LFS and understand the storage implications.

---

# 3. Recommended `.gitignore`

Review before applying to make sure it matches your repository.

Example entries:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo

# Environments
.env
.venv/
venv/

# Secrets
*.key
*.pem

# Model/data artifacts
*.pt
*.pth
*.ckpt

# Large datasets
data-bin/

# Generated predictions / bulky results
generation-results/

# W&B local artifacts
wandb/
```

Important: if a needed summary/metadata file is inside a normally ignored result directory, copy the compact canonical summary into a tracked `docs/results/` or `docs/llm_handoff/` directory rather than tracking all generated outputs.

---

# 4. Recommended tracked source-of-truth layout

```text
/mnt/ext-vol/dplm/
├── docs/
│   ├── llm_handoff/
│   │   ├── PROJECT_CONTEXT.md
│   │   ├── CLAUDE_PROJECT_INSTRUCTIONS.md
│   │   └── SYNC_SETUP_GUIDE.md
│   ├── methods_section_draft.md
│   ├── results_section_final_draft.md
│   ├── local_method_comparison_statistics.md
│   └── results/
│       ├── compute_matched_100nfe_summary.txt
│       ├── latency_summary.txt
│       └── interleaved_summary.txt
│
├── experiments/
│   ├── latent_residual_fm/
│   └── latent_residual_ddpm/
└── ...
```

You do not have to use this exact layout, but having stable canonical paths helps both LLMs.

---

# 5. Add the repository to Claude Project

In Claude:

1. Create/open the thesis Project.
2. Connect GitHub.
3. Select the private DPLM repository.
4. Add the relevant repository content/project knowledge.
5. Put the contents of `CLAUDE_PROJECT_INSTRUCTIONS.md` into the Claude Project instructions if the UI supports project-level instructions.
6. Ensure `docs/llm_handoff/PROJECT_CONTEXT.md` is available to the project.

At the start of a new Claude research thread, use a prompt like:

```text
먼저 docs/llm_handoff/PROJECT_CONTEXT.md를 읽고 현재 실험 상태를 정리해줘.
그 다음 질문에서는 해당 문서를 기본 전제로 사용하되,
정확한 구현이 필요한 경우 실제 Python 코드를 확인해줘.
특히 Local Matched Residual DDPM을 paper RESDIFF와 동일하다고 표현하지 마.
```

---

# 6. Keep ChatGPT on the same source of truth

For ChatGPT, keep the same canonical files in this Project or connect the shared storage/repository when available.

When a result changes, update the canonical Markdown/result summaries rather than relying only on an old conversation.

Recommended update order after a new experiment:

1. raw experiment completes
2. produce compact metadata/summary artifact
3. update `methods_section_draft.md` if protocol changed
4. update `results_section_final_draft.md`
5. update `PROJECT_CONTEXT.md`
6. commit all canonical changes together

Example:

```bash
cd /mnt/ext-vol/dplm

git add \
  docs/llm_handoff/PROJECT_CONTEXT.md \
  docs/methods_section_draft.md \
  docs/results_section_final_draft.md \
  docs/results/

git commit -m "Update canonical DPLM experiment state"
git push
```

Then refresh/sync the repository in Claude.

---

# 7. Optional Google Drive layer

Use Drive mainly for:

- DPLM-2 / DPLM-2.1 PDFs
- thesis manuscript
- PPT
- figures
- documents that benefit from direct collaborative editing

Suggested Drive layout:

```text
DPLM-2-Thesis/
├── 00_CONTEXT/
├── 01_PAPERS/
├── 02_METHODS/
├── 03_RESULTS/
└── 04_PRESENTATION/
```

Keep code canonical in Git rather than maintaining separate manually copied Python versions in Drive.

---

# 8. What should be synchronized after each experiment?

Do not try to synchronize every generated file.

Synchronize compact, authoritative artifacts:

```text
PROJECT_CONTEXT.md
methods_section_draft.md
results_section_final_draft.md
local_method_comparison_statistics.md
summary.txt
latency_summary.txt
interleaved_summary.txt
relevant metadata.json
experiment Python source
```

Keep large intermediate datasets/checkpoints on the research machine unless there is a specific reason to share them.

---

# 9. Versioning rule

When a conclusion changes, commit the context and result source in the same Git commit.

Good:

```text
commit:
- evaluate_ddpm_100nfe.py changed
- summary.txt changed
- PROJECT_CONTEXT.md changed
```

Bad:

```text
code changed today
PROJECT_CONTEXT updated a week later
```

The second pattern causes ChatGPT/Claude to use stale experimental assumptions.

---

# 10. Suggested Git workflow

Before starting work:

```bash
cd /mnt/ext-vol/dplm
git pull --ff-only
git status
```

After verified changes:

```bash
git add <verified-files>
git diff --cached
git commit -m "Describe the verified experiment change"
git push
```

Do not commit an experimental conclusion until the corresponding output artifact has been checked.

---

# 11. Verification prompt to run in Claude

After linking the repository, ask Claude:

```text
PROJECT_CONTEXT.md와 실제 구현 파일을 비교해서 아래 항목을 표로 검증해줘.

1. base checkpoint
2. residual target
3. FM source/interpolation/target
4. hidden-state 범위
5. conditioning
6. network dimensions and parameter count
7. Mode-A teacher forcing
8. FM inference NFE
9. DDPM T / beta schedule / prediction target
10. DDPM ancestral inference NFE
11. DDIM-100이 재학습인지 inference-only 변경인지
12. alpha
13. dataset/split
14. CAMEO/PDB-date counts
15. 최종 FM-vs-DDIM latency

각 항목은:
- PROJECT_CONTEXT 기재값
- 코드/결과 파일 근거
- 일치 여부
- 주의사항
으로 정리해줘.

추측하지 말고 파일로 확인되지 않는 부분은 NOT CONFIRMED라고 표시해줘.
```

If that verification passes, Claude has enough project state to work in parallel with ChatGPT.

---

# 12. Operating principle

Use **Git files as the shared memory**, not model-to-model conversation transfer.

This gives:

- version history
- reproducibility
- precise code provenance
- less drift between ChatGPT and Claude
- an easy way to update both systems after every experiment

`PROJECT_CONTEXT.md` should remain concise enough to read at the beginning of a thread, but authoritative enough to prevent the major experimental distinctions from being lost.

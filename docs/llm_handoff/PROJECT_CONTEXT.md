# DPLM-2 Thesis Experiment — Project Context / Handoff

> **Purpose**
>
> This file is the canonical handoff document for continuing the user's DPLM-2 thesis experiments in another LLM/project (e.g. Claude).
> Read this file before answering experiment-specific questions.
>
> **Repository working directory**
>
> `/mnt/ext-vol/dplm`
>
> **Current scope covered by this handoff**
>
> Frozen DPLM-2 Bit → latent quantization residual refinement → Latent Residual Flow Matching (FM) → parameter/condition-matched residual DDPM baseline → compute-matched DDIM-100 evaluation → structural-quality and latency comparison.

---

## 0. Source-of-truth rules

Use the following precedence when sources appear to conflict.

1. **Final experimental result artifacts**  
   `results_section_final_draft.md`, `summary.txt`, `interleaved_summary.txt`, `latency_summary.txt`, final metadata JSONs.
2. **Finalized Methods / implemented code**  
   `methods_section_draft.md` and the actual FM/DDPM/DDIM Python implementation.
3. **Audit artifacts / protocol metadata**
4. **Pre-registration / design specification**  
   `MATCHED_DDPM_SPEC.md`
5. **External DPLM-2 / DPLM-2.1 paper values**

Important: `MATCHED_DDPM_SPEC.md` begins with "Status: design only" because it was written before implementation. Later files show that the matched DDPM and DDIM-100 evaluations were implemented and completed. Do **not** conclude that DDPM was never implemented merely from that old status line.

Paper values are external references. They must never silently replace locally measured values.

---

# 1. Research question

The local experiment studies whether the structural information lost by LFQ quantization in the DPLM-2 Bit folding pipeline can be refined in latent space using a lightweight residual generative model.

The central local comparison is:

- **Local Bit**
- **Local Bit + Latent Residual Flow Matching (FM)**
- **Local Bit + Matched Residual DDPM**
- Supplementary compute-matched comparison: **FM-100 vs DDPM checkpoint sampled with DDIM-100**

The local residual module is deliberately lightweight relative to retraining the base protein language model.

---

# 2. Base model and frozen components

Base checkpoint:

`airkingbd/dplm2_bit_650m`

The following remain frozen:

- DPLM-2 Bit base model
- structure tokenizer
- structure decoder

They are excluded from the residual-model optimizer.

The residual models train only the local refinement stack.

DPLM-2 Bit is used as the folding base model that predicts structure tokens from an amino-acid sequence.

---

# 3. What is being refined?

For a protein of length `L`:

- continuous tokenizer latent:
  `z_cont ∈ R^[B,L,13]`
- LFQ-quantized latent:
  `z_quant ∈ R^[B,L,13]`

The local target is the **latent quantization residual**:

`r = z_cont - z_quant`

At prediction time:

`z_refined = z_quant + alpha * r_hat`

The frozen structure decoder converts `z_refined` to an atom37 structure.

Final fixed scale:

`alpha = 0.50`

This is:

- latent-space refinement of LFQ quantization error

This is **not**:

- coordinate-residual Flow Matching
- the DPLM-2.1 paper's data-space Bit + FM method

---

# 4. Conditioning from frozen DPLM-2 Bit

The local residual model uses:

1. quantized latent `z_quant`
2. hidden representations from frozen DPLM-2 Bit

DPLM returns an embedding/pre-layer state plus 33 Transformer-layer outputs.

The local protocol uses:

`all_hidden_states[1:]`

Therefore:

- embedding/pre-layer state is excluded
- 33 Transformer outputs are retained
- only the **structure-side half** is used
- structure BOS/EOS positions are removed
- amino-acid-side hidden states are not directly passed to the residual model

Each retained layer state:

`h_l ∈ R^[B,L,1280]`

Learned layer mixture:

`a_l = softmax(w)_l`

`h_mix = Σ_l a_l h_l`

Quantized latent projection:

`W_quant : 13 -> 1280`

Condition:

`c_1280 = h_mix + W_quant(z_quant)`

A learned projection then maps:

`1280 -> 1024`

A sinusoidal time embedding followed by a 2-layer MLP is added to the condition.

The same conditioning topology is used for local FM and matched DDPM.

---

# 5. Residual refinement network

Both FM and matched DDPM use the same model capacity/topology.

Key dimensions:

- residual input: 13
- residual output: 13
- DPLM hidden condition: 1280
- internal hidden width: 1024
- residual blocks: 6
- DPLM layer-mixture logits: 33

Architecture components:

- learned 33-way softmax layer mixture
- `Linear(13,1280)` quantized-latent projection
- condition projection `1280 -> 1024`
- residual input projection `13 -> 1024`
- sinusoidal time embedding
- 2-layer time MLP
- six AdaLN-style residue-wise residual MLP blocks
- output LayerNorm
- final `1024 -> 13` projection

Each FM and DDPM residual network has:

`34,946,606 trainable parameters`

### AdaLN-style block

For each block:

1. non-affine LayerNorm on current hidden state
2. condition produces scale, shift, gate
3. modulation:
   `LN(x) * (1 + scale) + shift`
4. Linear → SiLU → Linear MLP
5. gated residual update:
   `x + sigmoid(gate) * update`
6. residue mask reapplied

The local AdaLN-style implementation is experiment-specific. Do not claim it exactly reproduces unpublished paper RESDIFF internals.

---

# 6. Mode-A / current teacher forcing

"Mode-A/current teacher forcing" is a **local implementation term**, not terminology attributed to the DPLM-2.1 paper.

During residual-model training and internal validation, each AFDB record supplies:

- amino-acid type
- current ground-truth LFQ structure IDs
- `z_quant,current`
- `z_cont,current`
- residual `z_cont,current - z_quant,current`

The frozen DPLM receives clean current ground-truth structure tokens and ground-truth amino-acid information in the packed forward input.

The condition itself is not additionally corrupted.

However, the residual trajectory is stochastic:

- FM input is an interpolated Gaussian-to-data residual state
- DDPM input is a forward-diffused residual state

At folding inference, the condition is no longer teacher-forced. It comes from the **final structure tokens predicted by DPLM** and their reconstructed `z_quant`.

Therefore a training-to-inference exposure gap exists in the conditioning protocol. Do not assign a causal performance effect to this gap unless separately demonstrated.

---

# 7. Latent Residual Flow Matching

Target:

`r1 = r = z_cont - z_quant`

Source:

`r0 ~ N(0, I)`

Continuous time:

`t ~ Uniform[0,1]`

Linear interpolation:

`r_t = (1-t) r0 + t r1`

Target velocity:

`u = r1 - r0`

Network:

`u_theta(r_t, t, c)`

Loss:

masked MSE over valid residues and 13 latent channels.

Conceptually:

`L_FM = sum(mask * (u_pred - u)^2) / (13 * number_of_valid_residues)`

Padding is zeroed throughout the FM path and network.

### FM inference

- start from newly drawn Gaussian residual
- integrate learned ODE from `t=0` to `t=1`
- explicit Euler sampler
- final benchmark: **100 Euler steps**
- therefore residual-model **NFE = 100**

The sampled final residual is `r_hat`.

No zero-source trajectory, coordinate loss, or coordinate-residual model is used.

---

# 8. Prediction-time DPLM/FМ pipeline

For each target amino-acid sequence:

1. Frozen DPLM-2 Bit performs folding generation.
2. DPLM generation uses:
   - `max_iter = 100`
   - deterministic unmasking
   - argmax token sampling
3. Final predicted structure tokens are retained.
4. Predicted tokens are converted to `z_quant`.
5. One **additional frozen DPLM forward pass** is run on the final token sequence.
6. Extract 33 post-embedding, structure-side hidden states aligned to residues.
7. FM samples `r_hat` with 100 Euler steps.
8. Refined latent:
   `z_quant + 0.50 * r_hat`
9. Frozen decoder outputs atom37 coordinates/PDB.

Ground-truth structure tokens are not used to build the prediction-time condition.

For final FM benchmarks the residual seed is target-specific, based on base seed 42 plus zero-based FASTA record index.

---

# 9. Local Matched Residual DDPM

The DDPM was constructed as a **local matched baseline** to isolate the effect of generative objective/sampler while holding major experimental factors fixed.

It is **not an exact reproduction of DPLM-2.1 RESDIFF**.

Shared with FM:

- base checkpoint
- frozen DPLM/tokenizer/decoder
- latent residual target
- dataset and split
- Mode-A training condition
- 33 structure-side hidden states
- embedding state exclusion
- layer-mixture topology
- z_quant conditioning
- residual-network architecture/capacity
- optimizer family/hyperparameters
- training update budget
- residue masking
- final base-model token realization for target-paired evaluation

### DDPM forward process

Clean target:

`x0 = r = z_cont - z_quant`

Sample:

`t ~ Uniform{1,...,1000}`

`epsilon ~ N(0,I)`

Forward corruption:

`x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon`

Training schedule:

- `T = 1000`
- linear beta
- `beta_start = 1e-4`
- `beta_end = 0.02`

Time sent to the shared model interface:

`tau = t / T`

The DDPM network predicts:

`epsilon_hat`

rather than FM velocity.

Loss:

masked epsilon-prediction MSE with the same valid-residue / 13-channel normalization as FM.

Not used:

- learned variance
- coordinate loss
- SNR weighting
- x0 prediction
- self-conditioning

---

# 10. Native DDPM inference

Primary matched DDPM inference is ancestral sampling.

- starts from Gaussian `x_T`
- traverses all timesteps `1000 -> 1`
- epsilon model evaluated once per reverse step
- fresh Gaussian reverse noise is added for `t > 1`
- no new noise at `t = 1`
- full sampler NFE = **1000**

The result `r_hat` enters the same:

`z_quant + alpha * r_hat`

decoder path.

Important:

FM-100 vs ancestral DDPM-1000 is **not compute matched**.

NFE refers only to residual-network calls. It does not count DPLM base generation and does not imply wall-clock speed scales exactly in proportion to NFE.

---

# 11. DDIM-100 supplementary compute-matched evaluation

The trained matched-DDPM checkpoint is reused.

No DDPM retraining occurs.

Accelerated sampler:

- DDIM
- `eta = 0`
- deterministic
- training diffusion horizon remains `T = 1000`
- only inference is respaced
- DDIM residual-model NFE = **100**
- FM residual-model NFE = **100**

Therefore FM-100 vs DDIM-100 is the supplementary **compute-matched NFE comparison**.

Do not describe DDIM-100 as a separately trained model.

The final audit reports:

- exact NFE: PASS
- deterministic: PASS
- NaN/Inf: PASS
- same frozen Bit targets: PASS

---

# 12. Residual training dataset

Local residual dataset source:

AFDB v4 SwissProt structure source.

Saved latent dataset:

- **176,733 proteins**
- **46,844,154 residues**
- saved lengths observed: **50–512**

Each record includes information needed to form the residual target, including current LFQ structure IDs and paired continuous/quantized tokenizer latents.

This is the dataset for the **local residual modules**.

Do not claim this is necessarily:

- the entire DPLM-2 Bit pretraining corpus
- the exact paper RESDIFF training corpus

---

# 13. Train/validation split

`split_v1`

- train: **174,685 proteins**
- internal validation: **2,048 proteins**
- seed: **42**
- no recorded train/validation ID overlap
- length-stratified across:
  - 50–128
  - 129–256
  - 257–384
  - 385–512

FM and matched DDPM use the same stored dataset and split.

CAMEO and PDB-date are external final evaluation sets and were not used to select:

- objective
- schedule
- checkpoint
- alpha

---

# 14. Optimization protocol

Shared FM/DDPM training protocol:

- physical batch size: 1 protein
- gradient accumulation: 8
- nominal effective batch: 8
- optimizer: AdamW
- base/peak LR: `1e-4`
- final LR: `1e-5`
- weight decay: `0.01`
- AdamW betas/epsilon: PyTorch defaults
- warmup: 2,000 optimizer steps
- total optimizer steps: 100,000
- post-warmup: linear LR decay
- gradient clipping: global norm 1.0
- full internal validation: every 5,000 optimizer steps
- checkpoint save interval: every 5,000 optimizer steps

Both final runs reached:

`100,000 optimizer steps`

Both best checkpoints were at:

`step 100,000`

Checkpoint selection is method-specific:

- FM: deterministic masked velocity MSE
- DDPM: deterministic masked epsilon MSE

Do not compare absolute validation-loss magnitudes between FM and DDPM because the targets differ.

---

# 15. Alpha selection

Candidate grid:

`{0, 0.25, 0.50, 0.75, 1.00}`

Selection population:

- fixed 100 samples from the internal 2,048-protein AFDB validation split
- selection seed 42
- same validation-index artifact reused across methods

Selection metric:

mean C-alpha RMSD between:

- decoded `z_quant,predicted + alpha * r_hat`
- oracle decoded stored `z_cont`

This is an internal latent-refinement selection criterion, not final CAMEO/PDB-date RMSD against native coordinates.

Selected:

- FM alpha = **0.50**
- DDPM alpha = **0.50**

No second alpha search was performed on CAMEO/PDB-date.

---

# 16. Final structural evaluation protocol

Final eligible populations:

- **CAMEO: 163 targets**
- **PDB-date: 442 targets**
- official/evaluator length filter: **60–512 residues**

Primary metrics:

- backbone RMSD (BB RMSD), lower is better
- TM-score, higher is better

BB RMSD uses backbone `N, CA, C` atoms after structural superposition.

For fair local pairing:

- FM evaluation generated a single DPLM base prediction per target.
- Final structure tokens were stored in a manifest.
- DDPM evaluation reused the **exact same frozen final Bit token predictions**.
- DDPM did not rerun DPLM generation for the paired comparison.

Therefore local Bit/FM/DDPM refinements are aligned to the same base prediction realization.

---

# 17. Main local structural results

## CAMEO (N=163)

| Method | Mean BB RMSD (Å) | Mean TM-score |
|---|---:|---:|
| Local Bit | 6.2889 | 0.84142 |
| Latent Residual FM-100 | 6.1997 | 0.84495 |
| Matched Residual DDPM ancestral-1000 | 6.1925 | 0.84377 |

Bit-relative mean change:

FM:
- BB RMSD improvement: `+0.0891596035 Å`
- TM improvement: `+0.003533130792`

DDPM:
- BB RMSD improvement: `+0.0963748763 Å`
- TM improvement: `+0.002351455122`

## PDB-date (N=442)

| Method | Mean BB RMSD (Å) | Mean TM-score |
|---|---:|---:|
| Local Bit | 3.1138 | 0.91380 |
| Latent Residual FM-100 | 3.1074 | 0.91386 |
| Matched Residual DDPM ancestral-1000 | 3.1180 | 0.91304 |

Bit-relative mean change:

FM:
- BB RMSD improvement: `+0.0063187644 Å`
- TM improvement: `+0.000059240022`

DDPM:
- BB RMSD improvement: `-0.0042764425 Å`
- TM improvement: `-0.000752195122`

Interpret means descriptively. Do not infer significance from mean ordering alone.

---

# 18. Primary paired FM vs DDPM statistics

Difference orientation:

- BB RMSD FM advantage:
  `DDPM - FM`
- TM-score FM advantage:
  `FM - DDPM`

Positive values favor FM.

| Benchmark | Metric | Mean FM advantage | Bootstrap 95% CI | Holm-adjusted p |
|---|---|---:|---|---:|
| CAMEO | BB RMSD | -0.007215 Å | [-0.149566, 0.130141] | 0.8775 |
| CAMEO | TM-score | +0.001182 | [-0.002197, 0.004750] | 0.6306 |
| PDB-date | BB RMSD | +0.010595 Å | [-0.030799, 0.053301] | 0.2468 |
| PDB-date | TM-score | +0.000811 | [-0.000220, 0.001837] | 0.4761 |

All four bootstrap intervals include zero.

All four direct FM-vs-DDPM primary tests are non-significant after Holm correction.

Correct interpretation:

- No statistically significant FM-vs-DDPM difference was detected in these four primary paired tests.

Incorrect interpretations:

- "FM is statistically superior to DDPM."
- "FM and DDPM are statistically equivalent."

Failure to reject a difference is not an equivalence test.

---

# 19. FM vs Bit statistical note

For FM-vs-Bit CAMEO TM-score:

- Holm-adjusted p = `0.00879188717`
- significant positive paired rank shift after Holm correction
- bootstrap CI for the **mean** TM-score improvement still includes zero

These are different inferential summaries and should be reported together.

Do not infer from this that FM significantly outperformed DDPM. The direct FM-vs-DDPM tests are the relevant tests for that question and were non-significant.

---

# 20. NFE result

Native residual samplers:

- FM Euler: **100 NFE**
- ancestral matched DDPM: **1000 NFE**

Thus FM uses one-tenth as many residual-model function evaluations as native ancestral DDPM.

Allowed statement:

> FM used 100 residual-model NFEs versus 1000 for ancestral matched DDPM, while the primary paired tests did not detect a structural-quality difference between them.

Not allowed:

> FM is 10x faster.

NFE is not wall-clock time.

---

# 21. Compute-matched structural quality: FM-100 vs DDIM-100

Both methods use 100 residual-network evaluations.

## CAMEO (N=163)

- FM-100 BB RMSD: **6.19972204**
- FM-100 TM: **0.844954294**
- DDIM-100 BB RMSD: **6.21658677**
- DDIM-100 TM: **0.843924419**

Ancestral-1000 DDPM reference:

- BB RMSD: **6.19250677**
- TM: **0.843772619**

## PDB-date (N=442)

- FM-100 BB RMSD: **3.10744498**
- FM-100 TM: **0.91385542**
- DDIM-100 BB RMSD: **3.12709310**
- DDIM-100 TM: **0.913761325**

Ancestral-1000 DDPM reference:

- BB RMSD: **3.11804019**
- TM: **0.913043985**

This comparison is compute-matched in residual **NFE**, not necessarily total FLOPs or whole-pipeline wall time.

---

# 22. Same-session interleaved residual latency: FM-100 vs DDIM-100

Hardware / run:

- GPU: NVIDIA RTX 2000 Ada Generation
- dtype: torch.float32
- targets: 16
- repeats: 5
- batch size: 1
- FM NFE: 100
- DDIM NFE: 100
- same-session: YES
- interleaved: YES

## FM-100

- mean: `400.305789182 ms`
- median: **399.847084424 ms**
- median ms/NFE: `3.998470844`
- median residues/s: `484.596165830`

## Matched DDPM (DDIM-100)

- mean: `437.005612723 ms`
- median: **436.039869441 ms**
- median ms/NFE: `4.360398694`
- median residues/s: `439.225726820`

Comparison:

- median target-wise DDIM/FM ratio: **1.091421832**
- aggregate median DDIM/FM ratio: **1.090516566**
- median paired difference: **36.446723156 ms**

Interpretation:

At equal NFE=100 in this 16-target, 5-repeat, batch-1 residual-sampling benchmark, FM residual sampling was approximately 9% faster by the aggregate/target-wise median latency ratios.

This is a measured residual-sampler latency result under the stated hardware/protocol. Do not automatically generalize it to all GPUs, batch sizes, protein populations, or full end-to-end folding.

---

# 23. Earlier compute-matched latency artifact

A separate 16-target, 3-repeat artifact reported:

- FM-100 median residual: `422.845473047 ms`
- DDIM-100 median residual: `460.416638060 ms`
- ancestral-1000 reference median residual: `6434.198538074 ms`
- DDIM/FM aggregate median ratio: `1.088853180`

Prefer the later **same-session interleaved 5-repeat** benchmark when presenting the clean FM-100 vs DDIM-100 latency comparison, because it was explicitly run interleaved in the same session and repeated five times.

---

# 24. Relationship to DPLM-2.1 paper methods

These distinctions are mandatory.

### Local Latent Residual FM

Our experiment:

- predicts residual in the 13-D LFQ latent space
- target is `z_cont - z_quant`

DPLM-2.1 paper Bit + FM:

- external paper method
- described in this project as a different **data-space FM** approach

Do **not** call them the same method.

### Local Matched Residual DDPM

Our local baseline:

- high-level residual DDPM comparison
- condition/model capacity deliberately matched to local FM

It is **not** an exact paper RESDIFF reproduction.

Unknown/unconfirmed paper implementation details must not be invented.

---

# 25. External DPLM-2.1 paper reference values

These are contextual references only, not locally paired measurements.

| Paper row | CAMEO RMSD | CAMEO TM | PDB-date RMSD | PDB-date TM |
|---|---:|---:|---:|---:|
| Bit | 6.4028 | 0.8380 | 3.2213 | 0.9043 |
| Bit + RESDIFF | 6.1781 | 0.8428 | 3.0168 | 0.9076 |
| Bit + FM | 6.1825 | 0.8414 | 2.8697 | 0.9099 |
| Bit + FM + RESDIFF | 6.0765 | 0.8456 | 2.7884 | 0.9146 |

Do not attach local paired p-values, target-wise win rates, or causal conclusions to these paper rows.

Differences in training, implementation, conditioning, sampling, and evaluation realization prevent treating paper-vs-local absolute values as a controlled head-to-head experiment.

---

# 26. Claims supported by current results

Supported:

- On CAMEO, both local FM and local matched ancestral DDPM improved the reported mean BB RMSD and TM-score relative to Local Bit.
- On PDB-date, FM changed the mean metrics only slightly relative to Local Bit.
- Matched ancestral DDPM did not improve either PDB-date mean metric relative to Local Bit.
- Direct FM-vs-DDPM paired tests did not detect significant differences after Holm correction.
- FM used 100 residual NFE versus 1000 for ancestral DDPM.
- This NFE reduction occurred without a **detected** structural-quality loss in the specified direct primary comparisons.
- At equal 100 NFE, the final interleaved residual-latency benchmark measured an approximately 1.09x DDIM/FM latency ratio (FM lower latency) on the stated 16-target RTX 2000 Ada setup.
- DDIM-100 is an inference-only acceleration of the trained DDPM checkpoint.

---

# 27. Claims NOT supported

Do not claim:

- FM is statistically superior to DDPM.
- FM and DDPM are statistically equivalent.
- FM is universally better than DDPM.
- FM is 10x faster in wall-clock time.
- A 10x NFE reduction means a 10x end-to-end speedup.
- Local Matched Residual DDPM exactly reproduces DPLM-2.1 RESDIFF.
- Local results refute the paper RESDIFF results.
- Paper Bit + FM and Local Latent Residual FM are the same method.
- The paper RESDIFF's exact beta schedule / exact hidden-state range / exact AdaLN implementation are known unless directly supported by a source.
- CAMEO/PDB-date were used to tune alpha, sampler, checkpoint, or schedule.

---

# 28. Key code files

Expected repository-relative names:

## FM

- `experiments/latent_residual_fm/build_afdb_latent_shards.py`
  - builds sharded AFDB latent-residual dataset
- `experiments/latent_residual_fm/train_latent_residual_fm.py`
  - main Mode-A latent residual FM training
- `experiments/latent_residual_fm/flow_matching.py`
  - Gaussian-source FM interpolation, loss, Euler sampler
- `experiments/latent_residual_fm/dplm_condition.py`
  - frozen DPLM condition extraction
- `experiments/latent_residual_fm/fm_model.py`
  - 13-D residual network / AdaLN-style blocks
- `experiments/latent_residual_fm/generate_final_folding_predictions.py`
  - final paired Bit/FM folding generation

## DDPM / DDIM

- `experiments/latent_residual_ddpm/ddpm_training.py`
  - single-batch epsilon-prediction DDPM training computation
- `experiments/latent_residual_ddpm/ddpm_diffusion.py`
  - DDPM schedule and forward diffusion
- `experiments/latent_residual_ddpm/ddpm_model.py`
  - parameter-matched DDPM model using FM architecture
- `experiments/latent_residual_ddpm/train_latent_residual_ddpm.py`
  - matched DDPM training
- `experiments/latent_residual_ddpm/evaluate_final_ddpm.py`
  - final ancestral DDPM evaluation using frozen Bit manifests
- `experiments/latent_residual_ddpm/evaluate_ddpm_100nfe.py`
  - DDIM-100 compute-matched evaluation
- `experiments/latent_residual_ddpm/benchmark_fm_vs_ddpm_latency.py`
  - native FM-100 vs ancestral DDPM-1000 latency benchmark
- `experiments/latent_residual_ddpm/benchmark_fm_vs_ddim100_interleaved.py`
  - final same-session interleaved FM-100 vs DDIM-100 latency benchmark
- `experiments/latent_residual_ddpm/MATCHED_DDPM_SPEC.md`
  - historical pre-registration/design document; not the final execution state

---

# 29. Key result / manuscript files

- `methods_section_draft.md`
  - current detailed local Methods description
- `results_section_final_draft.md`
  - finalized local Results narrative and supported/not-supported claims
- `local_method_comparison_statistics.md`
  - paired FM/DDPM/DDPM-vs-Bit statistics
- `summary.txt`
  - compute-matched FM-100 vs DDIM-100 structural summary
- `latency_summary.txt`
  - earlier compute-matched latency summary
- `interleaved_summary.txt`
  - final same-session interleaved FM-100 vs DDIM-100 residual latency summary
- DDIM metadata JSON
  - audit of checkpoint, NFE, determinism, device, timestep sequence, etc.

---

# 30. Environment

Known finalized DDIM audit environment:

- GPU: NVIDIA RTX 2000 Ada Generation
- PyTorch: `2.2.0+cu121`
- torch CUDA: `12.1`
- residual inference dtype: `torch.float32`

User working environment:

`(dplm) ubuntu@pcy:/mnt/ext-vol/dplm$`

When giving execution commands, assume VS Code Remote/SSH terminal opened at:

`/mnt/ext-vol/dplm`

---

# 31. How to explain code to the user

When the user provides or asks about code:

1. state what the file/code is for
2. explain execution flow step-by-step
3. identify important variables/classes/functions
4. give tensor shapes where relevant
5. explain equations tied to exact code
6. distinguish training vs inference
7. distinguish DPLM base generation vs residual sampling
8. explicitly identify what is frozen vs trainable
9. for commands, use the VS Code terminal and working directory:
   `/mnt/ext-vol/dplm`

Prefer Korean for explanations unless the user requests another language.

---

# 32. Terminology

Use these names consistently:

- **Local Bit**
- **Latent Residual FM** / **Local Bit + Latent Residual FM**
- **Local Matched Residual DDPM** / **Matched Residual DDPM**
- **Matched DDPM (DDIM-100)** for the accelerated 100-NFE inference result
- **paper RESDIFF** only when referring to the external DPLM-2.1 paper method

Avoid shortening local matched DDPM to "RESDIFF" because that falsely implies exact reproduction.

---

# 33. Recommended reasoning discipline for future analysis

Before making a thesis claim:

1. Identify whether the number is local or paper-reported.
2. Identify sampler:
   - FM Euler-100
   - DDPM ancestral-1000
   - DDIM-100
3. Identify whether the comparison is:
   - native sampler comparison
   - equal-NFE compute-matched comparison
   - structural-quality comparison
   - latency-only comparison
4. Check whether the same target/base-prediction realization was reused.
5. Check whether statistical evidence exists for the requested conclusion.
6. If direct significance was not detected, do not transform descriptive mean ordering into a superiority claim.
7. Never infer exact unpublished RESDIFF details.

---

# 34. Minimal experiment summary

The project freezes DPLM-2 Bit and refines the 13-D LFQ quantization residual `z_cont - z_quant` using a lightweight conditional residual model. Latent Residual FM learns a Gaussian-to-residual velocity field and samples with 100 Euler evaluations. A parameter- and condition-matched DDPM baseline learns epsilon prediction with a 1000-step linear-beta diffusion process and uses 1000 ancestral reverse evaluations. The local matched structural comparisons did not detect significant FM-vs-DDPM differences across the four primary paired benchmark/metric tests, while FM required one-tenth the residual NFE of ancestral DDPM. A supplementary equal-NFE comparison reuses the same trained DDPM checkpoint with deterministic DDIM-100. In the final same-session interleaved 16-target latency benchmark, FM-100 had median residual latency 399.85 ms versus 436.04 ms for DDIM-100, an aggregate ratio of about 1.09. This supports an efficiency observation under the measured setup, not a universal superiority or exact RESDIFF-reproduction claim.

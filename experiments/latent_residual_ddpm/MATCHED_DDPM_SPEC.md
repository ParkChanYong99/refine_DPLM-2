# Local Matched Residual DDPM Specification

Status: design only. No DDPM code, training, inference, or test-set evaluation is authorized by this document.

## 1. Goal

Train a **Local Matched Residual DDPM** under conditions matched as closely as possible to the finalized local latent residual Flow Matching (FM) experiment, enabling the following local comparison:

1. Local Bit
2. Local Bit + Matched Residual DDPM
3. Local Bit + Latent FM

The DDPM is a local baseline. It must not be called an exact reproduction of paper RESDIFF.

## 2. Why this baseline is needed

The existing local experiment replaces residual diffusion with Gaussian-source conditional FM. A matched DDPM separates the effect of the generative objective and sampler from the effects of the dataset, frozen DPLM, conditioning representation, residual target, model capacity, optimizer, and training budget.

Paper Bit + RESDIFF remains a reported reference only. Since it is not a locally reproduced, target-paired result, it is not an eligible member of local paired statistical tests.

## 3. Frozen/shared components

The following are fixed across local FM and matched DDPM:

| Component | Fixed specification | Repository basis |
|---|---|---|
| Base checkpoint | `airkingbd/dplm2_bit_650m` | `train_latent_residual_fm.py::DPLM_CHECKPOINT` and final metadata |
| DPLM | frozen, `eval()`, zero trainable parameters, excluded from optimizer | `DPLMConditionExtractor` and FM trainer |
| Structure tokenizer/decoder | frozen | finalized local pipeline |
| Dataset | `data-bin/latent_residual_fm/afdb_l512_sharded` | FM trainer default |
| Split | `split_v1.csv`; train 174,685, validation 2,048; seed 42 | FM trainer assertions |
| Training condition | clean teacher-forced current GT structure tokens plus GT AA tokens | `dplm_condition.py` |
| Hidden states | structure modality only; 33 Transformer layer outputs; embedding/pre-layer state excluded | `all_hidden_states[1:]` in `dplm_condition.py` |
| Latent/residual width | 13 | FM model and loss |
| DPLM hidden width | 1280 | validated FM interface/model default |
| Trainable conditioning | 33-way softmax layer mixture plus trainable `z_quant` projection | `LatentResidualFM.forward` |

CAMEO and PDB-date are excluded from training, validation, checkpoint selection, residual-scale selection, schedule selection, and all other hyperparameter decisions.

## 4. Residual target

For each current training example:

\[
x_0 = r = z_{\mathrm{cont,current}} - z_{\mathrm{quant,current}},
\qquad x_0 \in \mathbb{R}^{B\times L\times 13}.
\]

The stored dataset residual is used directly. There is no coordinate residual, target renormalization, clipping, or alternate latent pairing. Padding is zeroed with the same residue mask used by FM.

## 5. Conditioning

The matched DDPM uses the same condition definition as the implemented FM:

\[
a_i = \operatorname{softmax}(w)_i,
\qquad
h_{\mathrm{mix}} = \sum_{i=1}^{33} a_i h_i,
\]

\[
c_{1280} = W_{\mathrm{quant}}(z_{\mathrm{quant,current}}) + h_{\mathrm{mix}}.
\]

Here every `h_i` has shape `[B,L,1280]`, comes from the structure-side half of a clean frozen DPLM forward, and corresponds to a Transformer layer rather than the embedding state. The same trainable `layer_logits`, `z_quant_projection`, and `condition_projection` topology must be used.

Training is Mode-A/current teacher forcing. GT current `struct_ids` are allowed only in training and internal validation condition extraction. They must never replace predicted final structure tokens in prediction-time folding evaluation.

## 6. DDPM forward process

Pre-register `T=1000` discrete diffusion steps indexed by `t in {1,...,T}`. For every training sample:

\[
t \sim \operatorname{Uniform}\{1,\ldots,T\}, \qquad
\epsilon \sim \mathcal{N}(0,I),
\]

\[
\alpha_t = 1-\beta_t, \qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s,
\]

\[
x_t = \sqrt{\bar\alpha_t}\,x_0 +
      \sqrt{1-\bar\alpha_t}\,\epsilon.
\]

Noise, `x_t`, model output, and loss contributions at padded residues are masked. Training `t` and `epsilon` are sampled independently per example. The model receives normalized time `tau=t/T in (0,1]`, preserving the FM time-embedding interface and parameter count.

## 7. DDPM training objective

The baseline is fixed to epsilon prediction:

\[
\hat\epsilon =
\epsilon_\theta(x_t,\tau\mid z_{\mathrm{quant,current}},h_{1:33}),
\]

\[
\mathcal{L}_{\mathrm{DDPM}} =
\frac{
\sum_{b,l,d}m_{b,l}
(\hat\epsilon_{b,l,d}-\epsilon_{b,l,d})^2
}{
13\sum_{b,l}m_{b,l}
}.
\]

The mask factor in the numerator is implemented as `m[b,l] * (epsilon_hat - epsilon)^2`. The denominator and residue mask semantics must exactly match `masked_flow_matching_loss`. No SNR weighting, velocity prediction, `x_0` prediction, learned variance, auxiliary coordinate loss, or self-conditioning is part of the primary baseline.

## 8. Noise schedule

### Pre-registered choice

Use a linear beta schedule:

\[
\beta_t = 10^{-4} +
\frac{t-1}{T-1}(2\times10^{-2}-10^{-4}),
\qquad T=1000.
\]

This is a **standard matched DDPM baseline choice** made for simplicity, reproducibility, and a conventional full-noise terminal state. It is not claimed to be paper RESDIFF's schedule.

The schedule tensors should be computed once in float64 for numerical verification, then registered as non-trainable buffers in the model/sampler dtype as appropriate. Assertions must cover `0 < beta_t < 1`, monotonic `alpha_bar_t`, finite coefficients, and expected length `T`.

**NOT CONFIRMED:** the exact paper RESDIFF beta schedule and exact paper `T`. No complete public ResDiff implementation or configuration was found in this repository.

## 9. Model architecture

The residual network should be parameter-matched to `LatentResidualFM`:

- residual input/output dimension: 13;
- condition dimension: 1280;
- hidden dimension: 1024;
- six residue-wise AdaLN-style residual MLP blocks;
- 33 learned layer-mixture logits with softmax;
- `Linear(13,1280)` quantized-latent projection;
- the same condition projection, sinusoidal time embedding width, time MLP, input projection, output normalization, and output projection;
- the same mask applications at input, every residual block, and output.

Only the output meaning changes from FM velocity to DDPM noise. The noise schedule has no trainable parameters. No learned variance head is added. At implementation time, instantiate FM and DDPM models and assert exact equality of total and trainable parameter counts. If code sharing is introduced, it must not change the finalized FM implementation or checkpoint.

The repository FM AdaLN block is explicitly experiment-local. Exact paper RESDIFF AdaLN details are **NOT CONFIRMED**.

## 10. Training configuration

| Setting | Matched DDPM value |
|---|---:|
| Optimizer steps | 100,000 |
| Physical batch size | 1 |
| Gradient accumulation | 8 |
| Nominal effective batch | 8 |
| Optimizer | AdamW |
| AdamW betas / epsilon | PyTorch defaults: (0.9, 0.999) / (10^{-8}) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Peak/base learning rate | (10^{-4}) |
| Final learning rate | (10^{-5}) |
| Warmup | first 2,000 optimizer steps |
| Post-warmup schedule | linear decay to final LR at step 100,000 |
| Validation interval | 5,000 optimizer steps |
| Save interval | 5,000 optimizer steps |
| Seed | 42 |
| Numeric training path | float32, matching the current FM trainer |

Reuse the FM split loader, deterministic shard/sample ordering, epoch-boundary accumulation behavior, scheduler formula, checkpoint/RNG restoration semantics, metrics schema where applicable, and optimizer-step boundary stopping behavior.

The checked-in FM trainer currently has CLI defaults of 2,000 for validation/save intervals, while this matched protocol requires 5,000. Whether the finalized FM run explicitly used 5,000 is **NOT CONFIRMED** by `final_results_metadata.txt`. Before DDPM implementation, audit the saved FM checkpoint's `training_configuration`; if it differs, stop and resolve the protocol rather than silently claiming identical selection opportunities.

## 11. Validation/checkpoint selection

- Run full internal validation only.
- Generate validation (t,epsilon) deterministically from seed 42 and stable sample identity, analogous to the FM validation hash policy.
- Reuse the same fixed (t,epsilon) for a sample at every validation event, so checkpoint comparisons are not driven by validation-noise resampling.
- Primary checkpoint metric: lowest full-validation masked epsilon-prediction MSE.
- Validate at every 5,000 optimizer steps and at the terminal step.
- Save `checkpoint_last` at every save boundary; save `checkpoint_best` only on strict validation-loss improvement. An exact tie retains the earlier checkpoint.
- Store model, optimizer, scheduler, optimizer/micro step, epoch position, samples seen, best metric/step, RNG states, schedule parameters, (T), prediction type, and full configuration.
- Resume must reject scheduler-horizon, DDPM schedule, (T), architecture, dataset, or split mismatch.

FM velocity loss and DDPM epsilon loss have different targets and scales. Their absolute validation-loss values must **not** be compared across methods. Each loss selects checkpoints only within its own training run.

## 12. Inference protocol

The DPLM portion must reuse the finalized prediction-time protocol:

1. Generate final structure tokens from the AA sequence using frozen DPLM2Bit with `max_iter=100`, deterministic unmasking, and argmax sampling.
2. Reconstruct predicted `z_quant` with shape `[1,L,13]`.
3. Run an extra frozen DPLM forward on the final generated output tokens.
4. Exclude the embedding state, retain `all_hidden_states[1:]`, select only structure-side states, and residue-align all 33 tensors.
5. Put the DDPM in `eval()` and run under `torch.inference_mode()`.
6. Draw `x_T ~ N(0,I)` with a deterministic per-target seed.
7. Apply the full reverse chain from `T` to 1.
8. Set `r_hat = x_0`, then decode `z_refined = z_quant + alpha * r_hat` with the frozen continuous-latent decoder.

For the fixed-variance ancestral sampler:

\[
\mu_\theta(x_t,t,c)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t,c)\right),
\]

\[
\tilde\beta_t =
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
\]

For `t > 1`, sample from the stated posterior mean and variance with fresh Gaussian noise; for `t = 1`, use `x_0 = mu_theta` without added noise. Do not clip the predicted residual, because the target is an unbounded continuous residual. Do not use teacher-forced GT structure tokens or generation-time intermediate hidden states at inference.

## 13. DDPM residual scaling policy

Three defensible choices exist:

### A. Canonical alpha = 1.0

**Advantage:** directly tests the DDPM's intended reconstruction `z_quant + r_hat`, with no calibration.  
**Disadvantage:** FM was deployed with internally selected `alpha=0.50`, so only DDPM would be denied calibration; decoder sensitivity and residual over-amplitude could confound the generator comparison.

### B. Select DDPM alpha on internal validation

**Advantage:** gives both generators the same opportunity for post-generation scale calibration without touching either test set.  
**Disadvantage:** adds one selection stage and makes the selected DDPM alpha potentially different from FM's.

### C. Force FM's alpha = 0.50

**Advantage:** identical numeric scale in the final addition.  
**Disadvantage:** `0.50` was selected for FM, not DDPM; forcing it can favor or penalize DDPM and is not intrinsically fair.

### Recommended policy

Choose **B**. Reuse the existing FM internal-validation alpha protocol and its pre-registered grid:

\[
\alpha\in\{0.00,0.25,0.50,0.75,1.00\}.
\]

Use the same internal validation population/indices, deterministic seeds, decoder, masks, metric, aggregation, and tie rule used to select final FM alpha. Select exactly once before CAMEO/PDB-date evaluation. If the exact prior FM selection population or rule cannot be recovered from artifacts, mark it **NOT CONFIRMED** and resolve it before running DDPM; do not construct a new rule after observing DDPM or test results. Report the canonical (alpha=1.0) internal diagnostic separately if useful, but only the preselected alpha enters the final test comparison.

No alpha sweep is authorized at this specification stage.

## 14. Fairness against FM

### 14.1 What is identical?

Base checkpoint, frozen DPLM/tokenizer/decoder, latent residual target, dataset and split, clean teacher-forced training condition, 33 structure-side layer states, layer mixture, quantized-latent conditioning, residual network dimensions/topology, optimizer family and hyperparameters, LR horizon, seed, physical/accumulated batch, residue mask, optimizer-step budget, validation population, and final DPLM folding generation must be identical.

### 14.2 What must differ intrinsically?

FM samples continuous `t ~ U[0,1]`, constructs a linear Gaussian-to-data path, predicts velocity, and integrates an ODE. DDPM samples discrete `t ~ Uniform{1,...,T}`, corrupts clean data according to `alpha_bar_t`, predicts Gaussian noise, and runs a discrete reverse diffusion chain. Their loss magnitudes and native sampler dynamics are not directly comparable.

### 14.3 Is 100k optimizer steps fair?

It is the primary matched training-update budget and is reasonable because architectures and batch handling are matched. It does not guarantee equal convergence, gradient difficulty, or exact FLOPs. Report optimizer steps, actual samples seen, wall-clock time, and training compute separately. Do not extend DDPM training based on test performance.

### 14.4 How should FM Euler 100 and DDPM T sampling be compared?

The primary, pre-registered quality comparison uses each fixed native sampler: FM Euler 100 model evaluations versus DDPM ancestral 1000 evaluations. It is explicitly **not compute-matched**. Report neural function evaluations (NFE), per-target wall time, and peak memory.

An optional secondary compute-matched result may use a pre-registered 100-evaluation DDIM-style schedule with fixed timestep spacing and `eta=0`, but it must be labeled “accelerated DDPM, 100 NFE,” must not replace the primary ancestral result, and must be specified before any test run. No accelerated sampler is part of the first implementation unless separately approved.

### 14.5 Can parameter counts be matched?

Yes. Both inputs and outputs are 13-D, and the same conditioning/time/AdaLN topology can be retained. Fixed DDPM schedule buffers add zero trainable parameters. Exact trainable-count equality is an implementation acceptance criterion.

### 14.6 Is training sample exposure identical?

It is identical only if the same iterator, shuffle seed, epoch-boundary accumulation, and stopping semantics are reused. Equal optimizer steps alone are insufficient. Record and assert the final `samples_seen` against the finalized FM run before claiming equality.

### 14.7 Discrete DDPM time versus continuous FM time

DDPM samples one of 1000 discrete corruption levels uniformly and trains an epsilon target whose scale/meaning varies with SNR. FM samples a continuous scalar uniformly and trains a velocity along a Gaussian-source linear path. Matching the time-embedding network does not make these objectives identical; the difference is the intended experimental variable.

### 14.8 How is checkpoint selection matched?

Both methods use full, deterministic internal validation at the same pre-registered opportunities and select the lowest method-specific validation loss. DDPM uses diffusion epsilon MSE; FM uses velocity MSE. The two absolute values cannot be compared.

### 14.9 Are CAMEO/PDB-date final-test only?

**YES.** Neither benchmark may affect checkpoint, alpha, schedule, `T`, sampler, seed policy, architecture, stopping time, or any other choice.

## 15. Sampling-compute comparison

Every result must state:

| Quantity | FM primary | Matched DDPM primary |
|---|---:|---:|
| Sampler | explicit Euler | ancestral DDPM |
| Nominal steps / NFE | 100 / 100 | 1000 / 1000 |
| Stochastic source | initial Gaussian | initial Gaussian plus reverse-step noise |
| Compute-matched? | No | No |

Measure wall time after common DPLM generation separately for residual generation and for end-to-end folding. Use the same device, dtype, target order, warm-up policy, synchronization, and timing boundaries. These measurements describe compute; they do not permit sampler tuning on CAMEO/PDB-date.

## 16. Relationship to paper RESDIFF

The local DDPM intentionally follows the high-level residual-diffusion description recorded by the project's prior DPLM-2.1 paper audit: residual prediction in quantized latent space, conditioning on `z_quant` and learned layer-wise hidden mixtures, a lightweight six-layer 1024-hidden network, DDPM-style noise prediction, 100k updates, and the published LR/warmup schedule.

However, high-level correspondence is not exact reproduction. The repository-wide search found no complete public ResDiff implementation, and `fm_model.py` explicitly identifies its AdaLN-style block as experiment-local. Therefore the method name must remain **Local Matched Residual DDPM** or **Matched Residual DDPM Baseline**.

Paper RESDIFF numbers may appear only as “PAPER REFERENCE ONLY.” No local paired p-value, win rate, or target-level difference may be attached to them.

## 17. Confirmed vs NOT CONFIRMED paper details

“Confirmed” below means present in the project's prior paper audit/specification; it does not imply that a complete official implementation exists locally.

| Detail | Status | Evidence / interpretation |
|---|---|---|
| Residual target `z_cont - z_quant` | CONFIRMED | Prior paper audit in `experiments/latent_residual_flow/CODEX_SPEC.md` |
| Conditioning combines projected `z_quant` and learned layer-wise hidden mixture | CONFIRMED | Prior paper audit; same form implemented by local FM |
| Layer weights use a learnable softmax | CONFIRMED | Prior paper audit |
| Lightweight six-layer MLP, hidden size 1024 | CONFIRMED | Prior paper audit |
| DDPM-style epsilon/noise prediction | CONFIRMED | Prior paper audit |
| Training horizon 100,000 steps | CONFIRMED | Prior paper audit |
| Warmup 2,000; peak LR `1e-4`; final LR `1e-5` | CONFIRMED | Prior paper audit |
| Exact beta/noise schedule | **NOT CONFIRMED** | No local official config/code |
| Exact diffusion `T` | **NOT CONFIRMED** | No local official config/code |
| Exact AdaLN equations, gating, normalization, and time parameterization | **NOT CONFIRMED** | Local FM block is experiment-specific |
| Exact training condition: teacher-forced current tokens versus prediction-aware tokens | **NOT CONFIRMED** | No official ResDiff implementation found |
| Exact hidden-state subset, modality slice, and embedding-state inclusion | **NOT CONFIRMED** | No official ResDiff implementation found |
| Complete paper training dataset and sample-order/exposure behavior | **NOT CONFIRMED** | Not recoverable from local public implementation |
| Reverse variance, clipping, guidance, EMA, and sampling behavior | **NOT CONFIRMED** | No complete public ResDiff sampler/config |
| Complete public ResDiff code behavior | **NOT CONFIRMED** | Repository search found no implementation |

The linear `1e-4 -> 0.02`, `T=1000` schedule and fixed posterior-variance sampler in this document are local baseline decisions, not inferred paper settings.

## 18. Final planned comparison table

| Method | Residual generator | Result role | Paired local comparison |
|---|---|---|---|
| Local Bit | None | Local baseline | Yes |
| Local Bit + Matched DDPM | DDPM epsilon prediction | New local baseline | Yes, against the same Local Bit generations |
| Local Bit + Latent FM | Flow Matching velocity prediction | Finalized local method | Yes, against the same Local Bit generations |
| Paper Bit + RESDIFF | Paper residual diffusion | PAPER REFERENCE ONLY; not locally paired | No |

Final local reporting must include aggregate BB RMSD/TM-score, target-paired differences against the shared Local Bit baseline, pre-registered paired statistics, residual-sampler NFE/time, selected internal-validation alpha, and sample/accounting checks. A direct paired DDPM-versus-FM comparison is allowed only when both use the identical target population and shared DPLM generations; it must never be presented as a comparison against paper RESDIFF.

## 19. Leakage/test-set rules

1. CAMEO and PDB-date are final test only.
2. All DDPM schedule, `T`, architecture, loss, validation seeds, sampler, and alpha grid are frozen before loading test outcomes.
3. Alpha is selected using internal AFDB validation only.
4. DDPM choices must not be adjusted to approach or exceed finalized FM test numbers.
5. Test subgroup, target length, difficulty quartile, or failure pattern must not drive tuning.
6. Failed targets and evaluator length filtering must follow the same pre-existing evaluator semantics for all local methods.
7. Any deviation after test access invalidates confirmatory claims and must be labeled exploratory.
8. Paper RESDIFF remains reference-only and outside local paired statistics.

## 20. Implementation checklist

No item below is authorized in the present design-only stage. The next implementation request should proceed in this order:

- [ ] Read-only audit the finalized FM checkpoint configuration for validation/save intervals, scheduler horizon, actual samples seen, and trainable parameter count.
- [ ] Freeze this specification and record its digest before DDPM training or test access.
- [ ] Implement schedule utilities with coefficient/shape/finite unit tests.
- [ ] Implement epsilon batch construction and exact masked MSE tests, including padding invariance.
- [ ] Implement a DDPM residual model with the FM-matched topology; assert exact trainable parameter-count equality.
- [ ] Reuse `DPLMConditionExtractor`, dataset/split loading, shuffling, accumulation, scheduler, checkpoint, RNG, and resume guards without changing existing FM sources.
- [ ] Add validation noise/timestep determinism tests and confirm checkpoint selection uses only DDPM validation loss.
- [ ] Add reverse-step algebra, `t=1` no-noise, mask, reproducibility, and finite-output tests.
- [ ] Run only a separately approved tiny smoke test before committing the 100k configuration.
- [ ] Train once under the frozen 100k protocol; do not extend based on results.
- [ ] Select DDPM alpha once using the recovered, identical internal-validation alpha protocol.
- [ ] Freeze checkpoint, alpha, sampler, and seeds before CAMEO/PDB-date.
- [ ] Reuse final predicted tokens and the extra frozen DPLM forward for final evaluation.
- [ ] Verify identical test IDs, DPLM generations, evaluator inclusion, and GT structures across Local Bit, DDPM, and FM.
- [ ] Report native quality and sampling compute transparently; keep optional accelerated DDPM separate.
- [ ] Apply paired statistics only to locally paired methods and keep paper RESDIFF reference-only.

Until the unresolved FM runtime interval audit and prior alpha-selection artifact audit are complete, those details remain **NOT CONFIRMED** and implementation should not begin.

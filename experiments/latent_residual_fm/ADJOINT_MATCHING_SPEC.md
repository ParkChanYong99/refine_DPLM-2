# Adjoint Matching for the finalized latent-residual FM: implementation specification (not frozen)

Status: **design only; local first-step convention selected, overall AM protocol not frozen**. This document maps the paper to the local 13-dimensional residual FM. It authorizes no trainer, sampler, fine-tuning, checkpoint write, or test-set evaluation. Do **not** create `SPEC_FREEZE` until the remaining protocol decisions are resolved.

## Evidence and equation numbering

- Primary paper: [ICLR 2025 Adjoint Matching camera-ready PDF](https://openreview.net/pdf?id=xQBRrtQM8u), especially main Eqs. (25)–(27), Appendix Eq. (144), Algorithm 1 / Eqs. (215)–(217), and Appendix H.1. The [arXiv v5 HTML](https://arxiv.org/html/2409.08861) has the same relevant formulas but numbers them (37)–(42) and calls the experimental-details section G.1. Equation numbers below follow the ICLR PDF.
- Local source, inspected read-only: [`fm_model.py`](fm_model.py), [`flow_matching.py`](flow_matching.py), [`dplm_condition.py`](dplm_condition.py), [`train_latent_residual_fm.py`](train_latent_residual_fm.py), and [`generate_final_folding_predictions.py`](generate_final_folding_predictions.py). The finalized checkpoint is [`runs/fm_modeA_current_main/checkpoint_best.pt`](runs/fm_modeA_current_main/checkpoint_best.pt); read-only inspection found `optimizer_step=best_step=100000`, the expected split path and DPLM identifier, and 34,946,606 parameter elements. A fresh `LatentResidualFM()` has exactly 34,946,606 trainable parameters.
- Reward feasibility, inspected read-only: [`audit_adjoint_reward_interface.py`](audit_adjoint_reward_interface.py) passed the `X1 -> frozen decoder -> coordinates` gradient; [`audit_adjoint_aligned_reward.py`](audit_adjoint_aligned_reward.py) passed differentiable aligned-loss gradients for both GT and FM residuals, 5/5 perturbed SVD backpropagations, and a rigid-motion invariance check. The repository's separate `align_structures` helper has `@torch.no_grad()` and is **not** the reward implementation.
- First-step numerical diagnostic: [`audit_adjoint_first_step.py`](audit_adjoint_first_step.py) tested three explicit local conventions on eight short internal-validation proteins (lengths 57-65), eight Gaussian seeds each, 64 trajectories per candidate. A/B/C were all finite (64/64), with zero NaN, Inf, or explosive trajectories. This is a numerical feasibility check, not evidence of a paper-exact rule.
- No official AM Flow Matching implementation was found in this workspace. The authors' [public `soc-fine-tuning-sd` repository](https://github.com/microsoft/soc-fine-tuning-sd) describes a Stable Diffusion/DDIM scheduler and AM trainer, not the first-step discretization for this local FM. Its DDIM behavior is not evidence for resolving the FM `1/t` singularity.

## Frozen base and trainable copy

`v_base(X,t,c)` is the exact `LatentResidualFM` state loaded from `checkpoint_best.pt` in eval mode with every parameter frozen. It is never updated. `v_finetune(X,t,c)` is a **separate, same-architecture exact copy** of that state, initially with the same 34,946,606 trainable-parameter layout. Only `v_finetune` may later receive AM optimizer updates. Before any first update, compare every named parameter and buffer against `v_base` for equality and compare both velocity outputs on identical `(X,t,c,res_mask)` for exact/allclose agreement; no random reinitialization is allowed. The DPLM-2 Bit model and the structure tokenizer/decoder remain frozen.

The conditioning tuple `c` is exactly the finalized **Mode-A/current, teacher-forced** condition: stored `aatype` and current raw LFQ `struct_ids_current` are packed with BOS/EOS and the structure vocabulary offset; a frozen DPLM forward supplies the 33 structure-side Transformer outputs **excluding** its embedding state, aligned to `[B,L,1280]`. Stored `z_quant_current` and `res_mask` accompany that tuple. Current `DPLMConditionExtractor` validates batch size one, so `B=1` is the supported initial implementation boundary; larger batches require a separate alignment audit. Do not replace this with predicted-token or generation-intermediate hidden states during Mode-A training. Final inference uses the existing generated-token condition path, as the base FM already does.

The population is the same saved AFDB latent split as the finalized FM: 176,733 total, 174,685 train, 2,048 internal validation. No CAMEO or PDB-date data enter training, pilot selection, or the specification decisions.

## Residual state and memoryless schedule

The AM state `X_t` is the **13-dimensional LFQ quantization residual**, shape `[B,L,13]`. It is neither atom coordinates, `z_quant`, nor DPLM token logits. The local FM training path is

```text
X_0 = r_0 ~ N(0,I)
X_1 = r_1 = z_cont - z_quant
X_t = (1-t) r_0 + t r_1
alpha_t = t,  beta_t = 1-t,  alpha_dot_t = 1,  beta_dot_t = -1.
```

Those equalities come from `flow_matching.sample_flow_matching_batch`; its target velocity is `r_1-r_0`. They define the pretraining interpolation, **not** an AM stochastic trajectory generated from paired GT residuals.

| Schedule | Definition | Status |
| --- | --- | --- |
| Theoretical distinguished memoryless FM schedule | `sigma_theory(t)^2 = 2 beta_t [(alpha_dot_t/alpha_t) beta_t - beta_dot_t] = 2(1-t)/t` | Confirmed by the paper's memoryless-FM relation and Eq. (144); singular at `t=0`. |
| Paper's experimental offset | `K=40`, `h=1/K=0.025`, `sigma_offset(t) = sqrt(2(1-t+h)/(t+h))` | Confirmed in Appendix H.1; finite at the grid endpoints. |
| **Primary local AM noise schedule** | The paper's `sigma_offset` for stochastic forward updates at `t=h,2h,...,1-h` | Chosen for a future implementation; the local first interval has no stochastic noise, and the update at `t=1-h` produces the regular stochastic terminal state. Do not silently substitute another schedule. |

The offset is a finite-step empirical modification of the theoretical schedule. Do not claim the exact continuous-time memoryless theorem for the offset formula without a separate derivation. **Offsetting `sigma` does not offset or define `alpha_dot/alpha = 1/t`.** The literal Algorithm 1 forward drift is undefined at `t=0`; the following local numerical convention handles that interval without claiming to recover the paper implementation:

```text
PAPER_EXACT_FIRST_STEP_HANDLING = NOT CONFIRMED
LOCAL_PRIMARY_FIRST_STEP_POLICY = DETERMINISTIC_WARM_START
```

In the first interval only, with `h=0.025`, use the original FM ODE Euler step and add no stochastic noise:

```text
X_h = X_0 + h v_finetune(X_0,0,c),   X_0 ~ N(0,I).
```

Initially `v_finetune == v_base`; later the same convention would use the temporarily held `v_finetune`. From `t=h` onward, switch to the memoryless stochastic update below. Candidate A avoids inserting an arbitrary surrogate time into the singular `-X/t` term. In the internal diagnostic, all A/B/C candidates were finite for 64/64 trajectories and their aggregate endpoint mean, standard deviation, and mean residue norm were nearly identical to one another and to the deterministic 100-step base-FM reference. Candidate C had a somewhat larger maximum state norm (9.97081 versus 9.02097 for A and 9.02090 for B). The diagnostic does not establish that A is theoretically exact or empirically superior; it identifies no paper-exact first-step rule.

## Algorithm 1 trajectory and masking

After the deterministic `0 -> h` warm-start, for valid residue coordinates at `t=h,2h,...,1-h`, map Eq. (215) to the local velocity call `v_finetune(X_t,t,z_quant,hidden_states,res_mask)`:

```text
X_{t+h} = X_t
          + h [2 v_finetune(X_t,t,c) - (1/t) X_t]
          + sqrt(h) sigma_offset(t) epsilon_t,
epsilon_t ~ N(0,I),   X_0 ~ N(0,I).
```

Each state, Gaussian draw, and velocity has shape `[B,L,13]`. Apply `res_mask[...,None]` to initial noise, every later noise draw, each velocity, and the updated state, including the warm-start; padding entries must remain **exactly zero**. The existing FM forward already masks velocity and the deterministic Euler sampler masks states, but a future stochastic trajectory must explicitly mask its added noise and each update. Do not evaluate an undefined `0/0` on padded or valid elements at `t=0` and then mask it afterward. In particular, execute the regular stochastic update at `t=1-h=0.975` and store its result as `X_1_stoch`. The complete detached trajectory is

```text
X_0, X_h, X_2h, ..., X_{1-h}, X_1_stoch.
```

`X_1_stoch` remains the regular terminal trajectory state. The separate H.1 noiseless point `X_hat_1`, defined below, does not replace it.

Trajectory generation uses the current, temporarily held `v_finetune` field with **stop-gradient** states and no autograd graph through the full simulation. Stored `X_t` is detached before lean-adjoint VJPs and regression. This is the `X ~ p^{stopgrad(u)}` treatment in Eq. (25); the pretraining conditional linear interpolation is not substituted for this on-policy trajectory.

## Frozen-base drift and lean adjoint

From Eq. (144), the frozen base drift is

```text
b_base(X_t,t,c) = 2 v_base(X_t,t,c) - (1/t) X_t,    t>0.
```

The state Jacobian in the lean adjoint is **of `b_base`, never of `v_finetune`**. For no intermediate state cost (`f=0`), paper Eqs. (26)–(27), Euler Eq. (216), and Appendix H.1 give the following local terminal semantics:

```text
TERMINAL_STATE_SEMANTICS = CONFIRMED_FROM_PAPER_H1
reward-gradient point: X_hat_1
t=1 VJP state: X_1_stoch

a_tilde_1 = grad_{X_hat_1} g(X_hat_1) = -grad_{X_hat_1} R(X_hat_1),
a_tilde_{1-h} = a_tilde_1
                  + h J_X[b_base(X_1_stoch,1,c)]^T a_tilde_1,
a_tilde_{t-h} = a_tilde_t
                + h J_X[b_base(X_t,t,c)]^T a_tilde_t,
                t=1-h,1-2h,...,h.
```

The backward grid is `t=1,0.975,0.95,...,0.025`. The first VJP evaluates the base-drift Jacobian at regular stochastic `X_1_stoch`, not at `X_hat_1`; `t=1` is finite. Every later VJP uses the corresponding detached regular stochastic `X_t`. The future PyTorch operation for each VJP is conceptually `torch.autograd.grad(outputs=b_base, inputs=X, grad_outputs=a_tilde, ...)`, with a fresh state leaf requiring grad for that *single* VJP. Keep `v_base` parameters `requires_grad=False`; this operation must not accumulate parameter gradients in them. After each recursion step, stop-gradient the stored adjoint and zero padded entries. Condition `c`, `z_quant`, and the stochastic trajectory states are frozen inputs. The forward `0 -> h` step uses the declared local warm-start while the paper-exact first-step handling remains unconfirmed. No recursion code is implemented here.

## Terminal structural reward and sign

The primary candidate uses the already audited experiment-local Kabsch loss from `audit_adjoint_aligned_reward.py`. Let `D(z,m)` be the frozen structure tokenizer's **continuous-latent** `detokenize` path returning atom37 positions and masks. Let the oracle be `D(z_cont,res_mask)`; it is not native/raw coordinates. For the separate noiseless reward-evaluation residual `X_hat_1`, set

```text
alpha_refine = 0.5
z_refined(X_hat_1) = z_quant + 0.5 X_hat_1
P = N/CA/C atoms of D(z_refined(X_hat_1), res_mask)
Q = N/CA/C atoms of D(z_cont, res_mask)
L_struct(X_hat_1) = sum_valid_atoms ||Kabsch(P)_atom - Q_atom||_2^2 / N_valid_atoms.
```

Kabsch uses only the intersection of the residue mask and both decoder atom masks, removes centroids, performs a `torch.linalg.svd` rotation with reflection correction, and uses squared 3D distance **per valid atom** (no square root in the differentiable loss). The decoder remains parameter-frozen but its forward on `z_refined` must run with autograd enabled. No `tmtools` TM-score, evaluator RMSD, or repository `align_structures` (`@torch.no_grad`) may replace this path. The prior audits demonstrate a nonzero finite `dL_struct/dX_1` for both GT and finalized-FM residuals; they do not establish a useful reward scale.

The sign convention is fixed explicitly. With `lambda_reward > 0` but **not yet numerically selected**,

```text
R(X_hat_1) = -lambda_reward L_struct(X_hat_1)             (higher is better)
g(X_hat_1) = -R(X_hat_1) = lambda_reward L_struct(X_hat_1) (SOC terminal cost)
a_tilde_1 = grad_{X_hat_1} g(X_hat_1)
          = -grad_{X_hat_1} R(X_hat_1)
          = +lambda_reward grad_{X_hat_1} L_struct(X_hat_1).
```

This follows the paper's `f=0, g=-r` convention and Eq. (27)/Algorithm 1 terminal sign. A negative sign on `grad L_struct` here would reverse the intended reward preference.

**Primary local endpoint policy:** follow the paper's Appendix H.1 noiseless terminal estimate, whose displayed formula is confirmed:

```text
X_hat_1 = X_{1-h} + h v_base(X_{1-h},1-h,c),   h=0.025.
a_tilde_1 = grad_X g(X) evaluated at X = X_hat_1.
```

`X_hat_1` is a **residual state**; apply the same terminal decoder mapping `z_quant+0.5 X_hat_1` to evaluate `g`. Appendix H.1 simultaneously retains a regular stochastic final state, evaluates the terminal reward gradient at the separate noiseless estimate, and starts backward adjoint integration at `t=1`. Therefore the local implementation retains both values and assigns them distinct roles:

```text
regular stochastic final trajectory state: X_1_stoch
noiseless terminal reward-gradient point:  X_hat_1
terminal adjoint value:                     grad_{X_hat_1} g(X_hat_1)
t=1 base-drift Jacobian/VJP state:           X_1_stoch
```

Do not use `X_hat_1` as the `t=1` VJP state, and do not evaluate the terminal structural reward at `X_1_stoch`. Future code should use the explicit variable names `x_1_stoch` and `x_hat_1` to prevent this distinction from being lost. The earlier terminal timestep / `X_1` state ambiguity is **RESOLVED** by `TERMINAL_STATE_SEMANTICS = CONFIRMED_FROM_PAPER_H1`.

## AM regression objective and gradient ownership

For stored trajectory states, Eq. (144) gives `u_t = sqrt(2/[beta_t*((alpha_dot_t/alpha_t)*beta_t-beta_dot_t)])*(v_finetune-v_base) = (2/sigma_theory(t))*(v_finetune-v_base)` for the theoretical schedule; with `alpha_t=t`, this coefficient is `sqrt(2t/(1-t))`. Algebraically, when the full drift difference is `2(v_finetune-v_base)`, its control form is `u_t=(2/sigma(t))*(v_finetune-v_base)`. With the selected practical `sigma_offset`, the Algorithm 1 Eq. (217) residual is

```text
E_t = (2/sigma_offset(t))
      [v_finetune(X_t,t,c) - stopgrad(v_base(X_t,t,c))]
      + sigma_offset(t) stopgrad(a_tilde_t),
L_AM = sum_{t in {0,h,...,1-h}} masked_mean_valid_[B,L,13](E_t^2).
```

For each time, `masked_mean_valid` is the sum of `E_t^2 * res_mask[...,None]` divided by `13 * sum(res_mask)`; retain the sum over time as in Eq. (217). This is a declared local normalization, **not** the paper's unnormalized vector norm: because proteins have different valid lengths, it changes relative example weights, not just a global constant. Record its effect on loss and gradient scale in later formula diagnostics. The coefficient **`2/sigma`** and the **plus** sign before `sigma * a_tilde` are fixed by Eq. (217); do not replace them with the pretraining FM MSE. `X_t`, `a_tilde_t`, and `v_base` output are stop-gradient. Gradient from `L_AM` flows **only** into `v_finetune` parameters. The practical offset coefficient is an empirical paper choice; Eq. (144)'s exact theoretical derivation uses the unoffset memoryless coefficient.

Algorithm 1 sums over all 40 stored times. The paper's separate Appendix H.2 reports subsampling those times and H.3 reports loss-term clipping in experiments. Neither is silently adopted here; the primary mathematical objective is the full Eq. (217) masked sum. Whether those practical modifications are necessary for the local run remains a pre-implementation protocol decision.

## Later pilot, inference, and comparison boundaries

- `lambda_reward = NOT YET SELECTED`. Only a later internal-validation pilot may choose it, from a small set declared **before** looking at pilot outcomes and only after formula/unit and small-overfit diagnostics pass. No numerical grid is set here. No CAMEO/PDB-date tuning.
- `AM learning rate = TO BE SELECTED`; `AM total updates = TO BE SELECTED`; `AM grad accumulation = TO BE SELECTED`. Paper image-model settings and the original FM 100k schedule are not automatically transferred to this single-GPU protein experiment.
- The intended frozen final inference is the existing **deterministic FM ODE** (`sigma=0`) using `v_finetune`, 100 Euler steps, generated predicted `z_quant`/final-token hidden condition, and `z_refined=predicted_z_quant+0.5*AM_residual` through the frozen decoder. The paper allows a different post-training sampling noise coefficient, including zero; this does not make the *training* trajectory deterministic. Any AM-specific alpha policy must be declared and frozen using internal validation before test evaluation; no test-driven alpha sweep.
- A future final evaluation compares Local Bit, finalized Latent Residual FM, Local Matched Residual DDPM, and FM+AM. The primary paired contrast is **FM+AM versus the same finalized FM** on the same targets and inference protocol. Do not treat literature RESDIFF results as locally paired.
- Keep CAMEO and PDB-date untouched until the implementation, lambda, AM checkpoint, inference protocol, and alpha policy are frozen. No checkpoint or existing result artifact is changed by this specification.

## Open items / not yet frozen

1. **`PAPER_EXACT_FIRST_STEP_HANDLING = NOT CONFIRMED`; `LOCAL_PRIMARY_FIRST_STEP_POLICY = DETERMINISTIC_WARM_START`.** Eq. (215) uses `alpha_dot/alpha=1/t` at its `t=0` first step; H.1 offsets `sigma` only. Neither the paper's displayed FM formula nor an available official FM implementation specifies the exact finite first-step drift. Candidate A is the selected local numerical convention after the 64-trajectory diagnostic, not a recovered paper implementation. Do **not** infer a paper-exact drift from `sigma_offset`.
2. **Theory versus experimental details:** the theoretical `sigma_theory` is singular at zero; H.1 uses the finite offset, H.2 subsets time steps, and H.3 clips loss terms. The author repository inspected is DDIM-focused, not a local FM first-step implementation. The offset's exact memoryless guarantee and whether H.2/H.3 are needed locally are not confirmed. The required valid-length masked mean also changes example weights relative to the paper's raw norm. Primary Eq. (217) currently means the full 40-time masked sum, without clipping or subsampling; revisit only through an explicit, separately documented protocol decision.
3. **`lambda_reward = NOT YET SELECTED`.** Feasible gradients establish an interface, not a balanced reward/control scale. Select only with the internal-validation-only pilot described above.
4. **Optimizer schedule = TO BE SELECTED.** Learning rate, total updates, grad accumulation, numerical precision, and resource budget need implementation diagnostics; do not inherit the base FM 100k schedule or image-model hyperparameters without evidence.

No `SPEC_FREEZE`, trainer, memoryless sampler, optimizer step, or fine-tuning is part of this document.

# 3 Methods

### 3.1 Base Model and Residual Refinement Setting

We studied refinement of the structure-tokenizer latent produced by a frozen DPLM-2 Bit folding model (`airkingbd/dplm2_bit_650m`). The base model generated structure tokens conditioned on an amino-acid sequence; a separate, fixed structure tokenizer mapped structures into continuous latents and their lookup-free-quantized (LFQ) counterparts. Only the trainable local residual-refinement stack—including the 33-way layer-mixture logits, quantized-latent and condition projections, the time MLP operating on sinusoidal time embeddings, and the residual blocks with their input/output layers—was optimized. DPLM-2 Bit and the complete structure tokenizer/decoder remained frozen and excluded from optimization. During residual-model training, the tokenizer-derived quantities had already been stored in the latent dataset, so the tokenizer was not part of the trainable computation graph. The frozen DPLM was evaluated without gradients.

For a protein with \(L\) residues, let \(z_{\mathrm{cont}}\in\mathbb{R}^{B\times L\times13}\) denote the tokenizer's continuous latent and \(z_{\mathrm{quant}}\in\mathbb{R}^{B\times L\times13}\) its LFQ-quantized latent. The training target was the *latent quantization residual*

\[
r=z_{\mathrm{cont}}-z_{\mathrm{quant}}.
\]

At prediction time, a sampled residual \(\hat r\) was combined with the latent reconstructed from **predicted** Bit tokens using a fixed scale \(\alpha\):

\[
z_{\mathrm{refined}}=z_{\mathrm{quant}}+\alpha\hat r.
\]

The frozen structure decoder then decoded this rank-three continuous latent into an atom37 structure. An unmodified \(z_{\mathrm{quant}}\) supplied the corresponding local Bit-only branch. This is latent-space refinement of LFQ quantization error, not coordinate-residual flow matching. It is also distinct from the DPLM-2.1 paper's data-space Bit + FM method.

### 3.2 Conditioning from Frozen DPLM-2 Bit

The residual model received both the quantized latent and hidden representations from the frozen DPLM-2 Bit checkpoint. The DPLM forward pass returned an embedding or pre-layer state followed by 33 Transformer-layer outputs. We excluded the embedding state by taking `all_hidden_states[1:]`. DPLM packs structure and amino-acid modalities into separate sequence halves. From each Transformer output we selected only the structure half, removed the structure beginning- and end-of-sequence positions, and aligned the remaining states to residues. Thus every retained layer state \(h_\ell\), \(\ell=1,\ldots,33\), had shape \(B\times L\times1280\); amino-acid-side hidden states were not supplied to the residual network.

The model learned one scalar logit \(w_\ell\) per layer. Its normalized mixture weights and latent-plus-hidden condition were

\[
a_\ell=\operatorname{softmax}(w)_\ell,\qquad
h_{\mathrm{mix}}=\sum_{\ell=1}^{33}a_\ell h_\ell,\qquad
c_{1280}=h_{\mathrm{mix}}+W_{\mathrm{quant}}(z_{\mathrm{quant}}),
\]

where \(W_{\mathrm{quant}}\) is a learned linear map from 13 to 1280 channels. A further learned linear projection mapped \(c_{1280}\) to the network's 1024-channel internal condition \(c\). The 33-way softmax weights and both projections were trained with the residual network; the DPLM hidden states themselves were not. A sinusoidal embedding of model time was processed by a two-layer MLP and added to this projected condition before the adaptive residual blocks. This same conditioning topology was used for both local FM and matched DDPM.

### 3.3 Latent Residual Flow Matching

The FM model learned a velocity field on the 13-dimensional latent residual, with a Gaussian rather than a zero-valued source. For each training example, define \(r_1=r\), sample \(r_0\sim\mathcal N(0,I)\), and sample \(t\sim\mathcal U[0,1]\). The straight interpolation and target velocity were

\[
r_t=(1-t)r_0+tr_1,\qquad u=r_1-r_0.
\]

The conditional network \(u_\theta(r_t,t,c)\) predicted \(u\). For a residue-validity mask \(m_{b,l}\), the implemented objective was the mean squared velocity error over valid residues and all 13 latent channels:

\[
\mathcal L_{\mathrm{FM}}=
\frac{\sum_{b,l,d}m_{b,l}\bigl[u_\theta(r_t,t,c)_{b,l,d}-u_{b,l,d}\bigr]^2}
{13\sum_{b,l}m_{b,l}}.
\]

The source, target, interpolation, and velocity were zeroed at padded positions. The mask also zeroed the network input, each block output, and final prediction at padding. For inference, the learned ordinary differential equation was integrated from \(t=0\) to \(t=1\) by explicit Euler steps, starting from a newly drawn Gaussian residual. The final state was \(\hat r\). No zero-source trajectory, coordinate loss, or coordinate-residual model was used in this experiment.

### 3.4 Residual Refinement Network

Both local residual methods used the same residue-wise network topology. The residual input and output each had 13 channels; the incoming DPLM hidden condition had 1280 channels and was projected to a hidden width of 1024. The network contained six adaptive-layer-normalization-style residual MLP blocks. It also contained the learned 33-layer hidden-state mixture, the 13-to-1280 quantized-latent projection, an input projection, a sinusoidal time embedding followed by a two-layer time MLP, an output normalization, and a final 1024-to-13 projection. Each FM and DDPM network had 34,946,606 trainable parameters.

Within a block, non-affine layer normalization was applied to the current hidden vector. A linear transformation of the residue-specific condition produced scale, shift, and gate vectors. The normalized vector was modulated as \(\operatorname{LN}(x)(1+\mathrm{scale})+\mathrm{shift}\), passed through a linear–SiLU–linear MLP, and added back to \(x\) after multiplication by \(\operatorname{sigmoid}(\mathrm{gate})\). The residue mask was applied after each block. The final normalized, projected output was also masked. DDPM inherited the FM module and parameter layout without changing this architecture; only the interpretation of its noisy residual input, time, and 13-channel output differed. Accordingly, model capacity and condition construction were matched, although the learned weights were separately optimized for their respective objectives.

### 3.5 Training Condition and Mode-A Protocol

We call the local training-conditioning arrangement *Mode-A/current teacher forcing*. This is an implementation-specific protocol name, not a term attributed to the DPLM-2.1 paper. Each AFDB training record provided its amino-acid type, current ground-truth structure LFQ IDs, \(z_{\mathrm{quant,current}}\), and the stored residual \(z_{\mathrm{cont,current}}-z_{\mathrm{quant,current}}\). The frozen DPLM received the amino-acid and **clean current ground-truth structure tokens** in a packed forward input. Structure-side Transformer outputs were extracted as described above. The quantized-latent branch also used the record's current ground-truth latent. No additional corruption or generation was applied to this *condition* during residual-model training or validation.

This does not mean the residual-model inputs were noise-free: FM trained on interpolated Gaussian-to-data residual states, whereas DDPM trained on forward-diffused residuals. The distinction is between a clean, teacher-forced DPLM/token condition and a stochastic residual trajectory. At folding inference, the condition instead came from the final structure tokens predicted by DPLM and from their reconstructed quantized latent. Thus training and prediction conditions differed. We describe this teacher-forcing-to-prediction exposure gap as a protocol property, without assigning it a causal effect on structural performance in Methods.

### 3.6 FM Inference

For each target sequence, the frozen DPLM-2 Bit model performed folding generation with 100 iterations, deterministic unmasking, and argmax token sampling. The returned **final** output tokens supplied the predicted structure LFQ IDs and \(z_{\mathrm{quant}}\). The implementation then performed one *additional frozen DPLM forward pass on those final tokens*. From this forward it extracted the 33 post-embedding Transformer outputs and aligned their structure-side residues. Hidden states saved during an earlier generation update were not reused: they need not correspond to the eventual final token sequence, whereas the extra forward explicitly conditions the refiner on that sequence.

The trained FM network was evaluated without gradients. A target-specific Gaussian \(r_0\) was generated from a fixed seed, and 100 explicit Euler steps produced \(\hat r\), requiring 100 residual-network evaluations (NFE). For the final benchmark, the seed was \(42+\) the zero-based FASTA record index; this preserves reproducibility while not reusing one noise tensor for every protein. The refinement scale was fixed at \(\alpha=0.50\) from the separate internal-validation procedure in Section 3.10. The frozen structure tokenizer's continuous-latent decoder processed both \(z_{\mathrm{quant}}\) and \(z_{\mathrm{quant}}+0.50\hat r\), producing the paired Bit-only and FM-refined atom37 predictions and their PDB outputs. Neither the FM model nor the decoder was optimized at inference, and no ground-truth structure tokens were used to construct the prediction-time condition.

### 3.7 Matched Residual DDPM Baseline

The Local Matched Residual DDPM used the same clean target \(x_0=r=z_{\mathrm{cont}}-z_{\mathrm{quant}}\), residue mask, frozen base model, hidden-state conditioning, and residual-network topology as FM. Its forward process sampled an integer \(t\) uniformly from \(\{1,\ldots,T\}\) and Gaussian \(\epsilon\sim\mathcal N(0,I)\):

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s).
\]

The local schedule used \(T=1000\) and linearly spaced \(\beta_t\) from \(10^{-4}\) to \(0.02\). The residual network received normalized time \(\tau=t/T\) through the same sinusoidal time-embedding interface and predicted \(\epsilon\), rather than FM velocity. Its loss was masked epsilon-prediction MSE with the same normalization \(13\sum_{b,l}m_{b,l}\) as the FM loss. Both the noised state and the epsilon target were masked at padded positions. No learned variance or additional coordinate objective was used.

Primary DDPM inference began with Gaussian \(x_T\) and traversed every timestep from 1000 to 1. At each step the epsilon prediction set the ancestral posterior mean. The Local Matched Residual DDPM used the fixed posterior variance

\[
\tilde{\beta}_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,
\qquad \bar\alpha_0=1.
\]

For \(t>1\), fresh Gaussian reverse noise was scaled by \(\sqrt{\tilde{\beta}_t}\); at \(t=1\), no additional noise was added. This full chain required 1000 residual-network evaluations, with no timestep skipping, clipping, or DDIM shortcut. The resulting \(\hat r\) entered the same \(z_{\mathrm{quant}}+\alpha\hat r\) decoder path. We matched the local training data and split, residual target, frozen DPLM/tokenizer, condition representation, network capacity, optimization-step budget, and final target/base-prediction realization. The native 100-step Euler and 1000-step ancestral samplers were **not NFE-matched**. NFE counts calls to the residual network, not DPLM generation work, and does not imply a tenfold wall-clock difference. This posterior-variance rule and sampler define the Local Matched Residual DDPM baseline, not the paper's RESDIFF sampler; the local baseline is not an exact RESDIFF reproduction.

### 3.8 Training Data and Split

The residual-refinement dataset was prepared from the AFDB v4 SwissProt structure source. Its saved latent records comprised 176,733 proteins and 46,844,154 residues, with observed saved lengths from 50 to 512 residues. Each record contained the current structure LFQ IDs and the paired continuous and quantized tokenizer latents needed to form the 13-channel residual target. These are the inputs for the *local residual modules*, not a claim about the full corpus used to pretrain DPLM-2 Bit or the corpus used by paper RESDIFF.

The fixed `split_v1` partition contained 174,685 training proteins and 2,048 internal-validation proteins. It was generated deterministically with seed 42 and stratification by length intervals 50–128, 129–256, 257–384, and 385–512 residues; its recorded train/validation IDs did not overlap. Both FM and DDPM used this same stored dataset and split. Internal validation served model-checkpoint selection and the later refinement-scale study. The external CAMEO and PDB-date evaluation sets were not part of the residual-model training or internal-validation partition, and were not used to choose the objective, schedule, checkpoint, or \(\alpha\).

### 3.9 Optimization

The two residual models used the same optimizer protocol. Each micro-step processed one protein; gradients were accumulated over eight micro-steps for a nominal effective batch of eight. At an epoch boundary, the trainer also performed an optimizer update with the remaining smaller accumulation group, dividing gradients by the actual number of micro-steps in that group. Consequently, eight is the nominal rather than an invariant per-update batch size. Optimization covered the complete local residual-refinement stack: layer-mixture logits, quantized-latent and condition projections, the time MLP, residual blocks, and input/output layers; DPLM-2 Bit and the complete structure tokenizer/decoder were excluded. The optimizer was AdamW with learning rate \(10^{-4}\) and weight decay \(0.01\); the implementation used AdamW's default moment and epsilon settings. The learning-rate scheduler warmed up linearly over 2,000 optimizer steps and then decayed linearly toward \(10^{-5}\) over a fixed horizon of 100,000 optimizer steps. Gradients were clipped to global norm 1.0 before each update.

Both final training runs reached 100,000 optimizer steps. Full internal-validation loss was evaluated every 5,000 optimizer steps; checkpoints were also saved at 5,000-step intervals, with the best checkpoint selected by a strict decrease in that method's deterministic validation loss. The FM criterion was masked velocity MSE; the DDPM criterion was masked epsilon-prediction MSE, using fixed sample-specific validation noise and, for DDPM, a fixed validation timestep. These loss values measure different prediction targets and were not compared numerically across methods. The final best checkpoints were both at step 100,000. The frozen DPLM was used to extract condition features but received no optimizer updates.

### 3.10 Alpha Selection

The residual scale \(\alpha\) was chosen before the external folding evaluations. For each method, we evaluated the same fixed set of 100 indices from the 2,048-protein internal AFDB validation split. The FM indices were sampled once using selection seed 42 and saved; the DDPM selection reused that indices artifact. We evaluated the ascending grid \(\{0,0.25,0.50,0.75,1.00\}\), where \(\alpha=0\) is the Bit-only continuous-latent decode. For each internal-validation sample, the stored \(z_{\mathrm{cont}}\) was decoded by the frozen structure tokenizer as the oracle continuous-latent reference, and each \(z_\alpha=z_{\mathrm{quant,predicted}}+\alpha\hat r\) was decoded by that same tokenizer. The primary selection quantity was the mean C-alpha RMSD between each alpha decode and its oracle continuous-latent decode across the fixed 100 samples, not RMSD to raw/native benchmark coordinates. The candidate with the lowest mean was selected, with an exact tie resolved in favor of the smaller \(\alpha\). The selected value was \(0.50\) for both FM and DDPM.

The selection scripts used prediction-time DPLM folding rather than the teacher-forced training condition. FM drew a deterministic, per-validation-index Gaussian source using base seed 42 plus the validation index and used 100 Euler steps. DDPM used the corresponding index-based seed for its full 1000-step ancestral chain. Each method's \(\alpha\) grid was applied to its sampled residual without using CAMEO or PDB-date targets. The external test sets did not trigger a second scale search or a method-specific retuning after their results were observed.

### 3.11 Evaluation Protocol

The final local comparison used the eligible forward-folding populations of CAMEO (163 targets) and PDB-date (442 targets), applying the evaluator's 60–512-residue length filter. Targets outside this interval were skipped under the official evaluator protocol rather than counted as evaluation failures. Local Bit, Latent Residual FM, and Local Matched Residual DDPM were compared on matched target identifiers and lengths. The primary structural metrics were backbone RMSD, for which lower is better, and TM-score, for which higher is better. Unlike internal alpha selection, these final CAMEO/PDB-date metrics compared decoded predictions against benchmark ground-truth structures. Backbone RMSD used the N, CA, and C atoms after structural superposition; TM-score used `tmtools.tm_align` and the evaluator's `tm_norm_chain2` value.

For the FM comparison, a single DPLM folding generation per target supplied both the Bit-only and FM branches. The generated final structure tokens were recorded in a manifest along with target identity and sampling settings. For the DDPM comparison, the final evaluator reused the **exact final Bit token predictions from that existing manifest**. It reconstructed their quantized latents and performed the extra frozen forward for final-token hidden states, but did not rerun DPLM generation. This protocol held the base-prediction realization fixed across local refinement methods, preventing a new Bit sample from entering the DDPM-versus-FM contrast. DDPM refinement used the fixed \(\alpha=0.50\) and its target-specific sampler seed; Bit predictions and evaluation targets remained aligned across the compared per-target artifacts. The paper-reported Bit + FM and RESDIFF rows were external context, not locally target-paired methods in this comparison.

### 3.12 Statistical Analysis

Statistical comparisons were made after joining finalized per-target records by benchmark target ID and checking complete one-to-one coverage and length agreement. The primary local FM-versus-DDPM analysis contained four tests: backbone RMSD and TM-score in each of CAMEO and PDB-date. We oriented paired differences so positive values favored FM: \(\Delta_{\mathrm{BB}}=\mathrm{RMSD}_{\mathrm{DDPM}}-\mathrm{RMSD}_{\mathrm{FM}}\) and \(\Delta_{\mathrm{TM}}=\mathrm{TM}_{\mathrm{FM}}-\mathrm{TM}_{\mathrm{DDPM}}\). We reported the sample size, mean and median difference, and counts of targets favoring FM, favoring DDPM, or tied.

Uncertainty in the arithmetic mean paired difference was summarized by a two-sided percentile 95% confidence interval from 10,000 target-pair bootstrap resamples with seed 42. The bootstrap resampled paired differences, not the two methods independently. A two-sided Wilcoxon signed-rank test was applied to each vector of paired differences, discarding zero differences (`zero_method="wilcox"`), with no continuity correction and automatic exact/asymptotic method selection. The matched rank-biserial effect size was the sum of positive absolute-difference ranks minus the sum of negative ranks, divided by the sum of all nonzero ranks; ties in absolute magnitude received average ranks. Holm's step-down adjustment was applied across the four FM-versus-DDPM primary tests. DDPM-versus-Bit comparisons were analyzed with the same procedures but in a separate four-test secondary Holm family, oriented so positive differences favored DDPM. Existing FM-versus-Bit paired analyses were treated as a separate comparison rather than merged into either correction family. C-alpha RMSD was not included in the four-test primary family. These procedures describe the analysis plan; interpretation of the resulting estimates and tests belongs to Results and Discussion.

## Implementation points requiring author confirmation

No unresolved implementation points were identified for the finalized local FM and matched-DDPM pipelines. Exact implementation details of the paper RESDIFF system that are not publicly specified remain unresolved and are not attributed to the local DDPM baseline.

<!-- Author-review note: remove this confirmation section before incorporating the Methods draft into the final thesis. -->

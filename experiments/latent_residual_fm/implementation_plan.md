# DPLM-2.1 Bit 650M latent residual Flow Matching implementation plan

Scope: replace only the generative process used for the 13-dimensional LFQ quantization residual with Gaussian-source conditional Flow Matching. This is not the existing coordinate residual experiment and not the paper's Section 3.3 data-space FM. No implementation is performed in this phase.

## 1. Confirmed repository interfaces

### Structure tokenizer and LFQ

- `src/byprot/models/utils.py:387-412`, `get_struct_tokenizer(model_name_or_path, eval_mode=True)`: constructs `VQModel`, loads `dplm2_struct_tokenizer.ckpt`, freezes the complete tokenizer with `requires_grad_(False)`, and returns eval mode by default.
- `src/byprot/models/structok/structok_lfq.py:32-87`, `VQModel.__init__`: `codebook_embed_dim` comes from the tokenizer config. `pre_quant` maps encoder width to codebook width; `post_quant` maps codebook width to decoder width and then applies a 4-layer transformer.
- `src/byprot/models/structok/structok_lfq.py:138-159`, `VQModel.encode`: returns `(pre_quant, encoder_feats)`. `pre_quant`, not `encoder_feats`, is `z_cont`; it is masked by `mask[..., None]`.
- `src/byprot/models/structok/structok_lfq.py:121-123`, `VQModel.quantize`: is an `LFQ` module invoked as `quantize(pre_quant, mask=res_mask.bool())` and returns `(z_quant, loss, (_, _, struct_ids))`.
- `src/byprot/models/structok/modules/lfq.py:133-199`: an 8192-code LFQ has `codebook_dim=log2(8192)=13`; its codebook contains bit vectors mapped to `-1/+1`.
- `src/byprot/models/structok/modules/lfq.py:272-317,363-389`: `LFQ.forward` sign-quantizes each component, preserves input shape after packing/unpacking, and returns indices reshaped like `mask`.
- `src/byprot/models/structok/modules/lfq.py:218-239`, `LFQ.get_codebook_entry`: maps raw LFQ IDs `[B,L]` to float code vectors `[B,L,13]`.
- `src/byprot/models/structok/structok_lfq.py:161-176`, `VQModel.decode`: accepts a float latent `quant` and passes its post-quant representation into the structure decoder.
- `src/byprot/models/structok/structok_lfq.py:225-255`, `VQModel.detokenize`: rank-2 input is treated as raw LFQ IDs; rank-3 input bypasses lookup and is treated directly as continuous latent. It returns `atom37_positions`, `atom37_mask`, `aatype`, `residue_index`, and `plddt`.

### DPLM-2 Bit

- `src/byprot/models/dplm2/dplm2.py:119-167`, `from_pretrained`: the Hugging Face checkpoint config selects the concrete network class; the wrapper is constructed around the loaded network.
- `src/byprot/models/dplm2/dplm2_bit.py:38-85`, `DPLM2Bit`: initializes the structure tokenizer and defines `struct_vocab_offset = 36`.
- `src/byprot/models/dplm2/dplm2_bit.py:87-147`, `DPLM2Bit.forward`: splits the combined input into equal structure and amino-acid halves, embeds raw LFQ structure codes after subtracting offset 36, concatenates structure first and amino acid second, and always calls the Bit LM with `output_hidden_states=True`.
- `src/byprot/models/dplm2/modules/dplm2_bit_modeling_esm.py:388-444`: the hidden-state tuple contains the pre-layer hidden tensor and one tensor after every transformer layer. Therefore its length is `num_hidden_layers + 1`.
- `src/byprot/models/dplm2/modules/dplm2_bit_modeling_esm.py:672-712`: returns `last_hidden_state` and, when requested, `all_hidden_states`; structure and amino-acid logits are computed from the first and second sequence halves respectively.
- Cached exact checkpoint config `/home/ubuntu/.cache/huggingface/hub/models--airkingbd--dplm2_bit_650m/snapshots/40cf303d79335c4c8f1b13d08fbc3187e8ba9222/config.json`: `hidden_size=1280`, `num_hidden_layers=33`, `codebook_embed_dim=13`. These values must still be read from `model.net.config` at runtime rather than hard-coded.
- `src/byprot/models/dplm2/dplm2_bit.py:260-319`, `forward_decoder`: exposes `all_hidden_states` internally.
- `src/byprot/models/dplm2/dplm2_bit.py:342-432`, `generate`: keeps internal hidden states during iteration but returns only `output_tokens`, `res_mask`, and `final_struct_feature`.
- `src/byprot/models/dplm2/dplm2_bit.py:434-461`, `prepare_for_struct_tokenizer`: selects the structure half, removes structure BOS/EOS, subtracts offset 36, replaces padded raw IDs with zero, performs LFQ lookup, and returns `[B,S-2,13]` quantized features.
- `generate_dplm2.py:16-103`: folding input explicitly adds BOS/EOS independently to both modalities and concatenates the structure half before the amino-acid half.
- `generate_dplm2.py:288-326,366-443`: official generation decodes `final_struct_feature` via rank-3 `detokenize`, confirming that this generated feature is a continuous decoder input.

### Evaluation and absent components

- `src/byprot/utils/protein/evaluator_dplm2.py:448-487,555 onward`: provides the official forward-folding evaluation route.
- `src/byprot/utils/protein/evaluator_dplm2.py:1074-1092`: aggregates backbone RMSD-to-GT and TM-score-to-GT.
- Repository-wide search found no AdaLN implementation and no public ResDiff/residual-diffusion implementation. The only matches were the experiment specification itself.

## 2. Exact tensor pipeline

### Ground-truth residual target

```text
all_atom_positions [B,L,37,3]
res_mask          [B,L]
seq_length        [B]
  -> VQModel.encode
encoder_feats     [B,L,512]
  -> pre_quant
z_cont            [B,L,13]
  -> LFQ.forward(mask=res_mask.bool())
z_quant           [B,L,13], components exactly -1/+1 in the forward value
struct_ids        [B,L], raw LFQ range [0,8191]
r_target          [B,L,13] = z_cont - z_quant
```

`encoder_feats [B,L,512]` is not the residual target space. Only `pre_quant [B,L,13]` can be subtracted from and added to `z_quant`.

### Frozen DPLM folding condition

Let `S=L+2` before padding for one example.

```text
sequence residues [B,L]
  -> add AA BOS/EOS and batch padding
AA tokens         [B,S]
structure BOS + L masks + EOS, padded
struct tokens     [B,S]
  -> concat(struct, AA)
input_ids         [B,2S]
  -> frozen DPLM2Bit.generate
predicted combined IDs [B,2S]
  -> take first half, remove BOS/EOS, subtract 36, mask padding
predicted raw LFQ IDs  [B,Lmax]
  -> LFQ.get_codebook_entry
predicted z_quant      [B,Lmax,13]
```

For conditioning hidden states, run one additional frozen `DPLM2Bit.forward(final_output_tokens)` after generation. It returns:

```text
all_hidden_states: tuple length 34; each [B,2S,1280]
last_hidden_state:                    [B,2S,1280]
```

Then take each tensor's first half `[B,S,1280]`, remove structure BOS/EOS with the same position mask, and retain padding mask `res_mask`, yielding residue-aligned states `[B,Lmax,1280]`.

The extra final forward is necessary because public `generate()` does not return hidden states, and its internally retained tensor was computed from the decoder input before the final sampled-token update.

### Residual FM and decoding

```text
r_t              [B,L,13]
t                [B] -> broadcast/embedded per residue
z_quant          [B,L,13]
aligned h_i      34 tensors [B,L,1280] if embedding state is included
res_mask         [B,L]
  -> LatentResidualFlowMatcher
velocity         [B,L,13]
  -> Euler ODE from t=0 to 1
r_hat            [B,L,13]
z_refined        [B,L,13] = z_quant + r_hat
  -> VQModel.detokenize(rank-3 latent, res_mask)
atom37_positions [B,L,37,3]
atom37_mask      [B,L,37]
```

## 3. DPLM hidden-state alignment

1. Both modalities are tokenized as `BOS + L residues + EOS`, then padded separately to the same batch maximum and concatenated as `[structure, amino acid]` (`generate_dplm2.py:30-103`). Thus each hidden tensor is `[B,2S,H]`.
2. Split every hidden tensor with `h_struct, h_aa = h.chunk(2, dim=1)`. The structure condition candidate is `h_struct`; selecting structure rather than AA or combining both is an experiment-level choice and is **NOT CONFIRMED** by the public ResDiff code because that code is absent.
3. Build the residue selection from final structure IDs exactly as `prepare_for_struct_tokenizer`: exclude `struct_bos_id` and `struct_eos_id`. Preserve `res_mask` to exclude batch padding. Do not assume a fixed `[:,1:-1]` slice without first asserting the tokenizer layout and equal per-example BOS/EOS count.
4. Apply the identical selection to predicted raw LFQ IDs, every hidden-state tensor, and the latent mask. Assert `aligned_hidden.shape[:2] == z_quant.shape[:2] == res_mask.shape` before FM use.
5. The tuple has 34 entries for the exact 33-layer model: entry 0 is the input to transformer layer 0 (embedding-derived state); entries 1..33 are layer outputs, with entry 33 final-layer-normalized. Whether the paper's sum over layers includes entry 0 is **NOT CONFIRMED**. Make this an explicit config (`include_embedding_state`) and use only a paper-confirmed setting when available.
6. For generation-aware inference, perform a final no-grad forward on the final combined token IDs. Do not use `generate()`'s absent final hidden return or assume the last internal state corresponds to the returned sampled bits.
7. Padding is removed from aggregation/loss through `res_mask`; tensors may remain padded for batching. BOS/EOS must never enter `[B,L,13]` FM positions.

## 4. Condition construction

Preserve the paper's form:

```text
a = softmax(w, dim=0)                         [K]
h_mix = sum_i a_i * h_i                       [B,L,H]
z_feat = W_quant(z_quant)                     [B,L,H]
c = h_mix + z_feat                            [B,L,H]
```

Here `H=model.net.config.hidden_size` (1280 for the cached exact checkpoint), and `K` is 33 or 34 depending on the unresolved embedding-state convention. `W_quant` is a learned `Linear(13,H)`. `w`, `W_quant`, time embedding, FM MLP, and AdaLN parameters are trainable; DPLM, structure encoder, LFQ, and structure decoder remain frozen and evaluated under `torch.no_grad()`.

Use a six-block residue-wise MLP baseline with hidden size 1024, matching the paper's stated lightweight ResDiff capacity as closely as possible. Since no repository AdaLN implementation exists, the future minimal block should contain normalization without affine parameters, a time/condition projection producing per-residue shift and scale (and optionally a residual gate), an MLP transform, and a gated residual connection. Exact AdaLN parameterization used by paper ResDiff is **NOT CONFIRMED**; it must be isolated behind the FM module rather than attributed to existing source.

Mask FM activations after each block or at minimum before output, and compute loss only over valid residues. Verify freezing with: all base parameters have `requires_grad=False`; after backward all base `.grad is None`; every named trainable parameter belongs to the FM/condition module.

## 5. FM mathematical definition

For every valid residue and all 13 latent dimensions:

```text
r1 = z_cont - z_quant
r0 ~ N(0,I)
t  ~ Uniform(0,1)
rt = (1-t)r0 + t r1
u_t = r1-r0
L_FM = masked_mean(||v_theta(rt,t,c)-u_t||^2)
```

Implementation semantics: sample one `t` per example as `[B]`, broadcast to `[B,1,1]`, and sample `r0` with `torch.randn_like(r1)`. A precise masked scalar is

```text
sum((v_pred-u_t)^2 * res_mask[...,None])
------------------------------------------------
sum(res_mask) * 13
```

with a nonzero-denominator assertion. This is Gaussian-source latent residual FM, not zero-source coordinate FM.

## 6. Inference pipeline

1. Freeze/eval DPLM-2 Bit and the complete structure tokenizer.
2. Build folding inputs exactly as `initialize_conditional_generation` does and generate final combined IDs.
3. Derive predicted raw LFQ IDs, `res_mask`, and `z_quant [B,L,13]` using the same offset/BOS/EOS rules as `prepare_for_struct_tokenizer`.
4. Run one final frozen forward on final combined IDs; align the chosen hidden tuple entries to the same residue positions.
5. Initialize `r=torch.randn_like(z_quant)` and zero invalid positions.
6. With `dt=1/num_flow_steps`, for `step=0..num_flow_steps-1`, evaluate `t=step/num_flow_steps`, compute `v`, and update `r += dt*v`; re-mask invalid positions after every update.
7. Set `r_hat=r`, `z_refined=z_quant+r_hat`, and decode through `struct_tokenizer.detokenize(z_refined, res_mask=res_mask)`.
8. Keep ODE sampler separate from the model interface so a later solver or Adjoint Matching experiment can be added without changing model conditioning. Do not implement Adjoint Matching in the first experiment.

The decoder accepts arbitrary rank-3 float latent values structurally; whether unconstrained FM outputs remain in the decoder's well-trained latent distribution is **NOT CONFIRMED**. Log latent statistics and compare `z_quant`, oracle `z_cont`, and `z_refined`; do not silently clip without a separately specified ablation.

## 7. Training dataset feasibility

### Locally confirmed files

- `data-bin/pdb_swissprot`: two train parquet shards and one valid parquet shard. README metadata reports 220,354 train examples and 128 valid examples. Columns include `processed_path`, `struct_seq`, `aa_seq`, `gvp_feat_path`, and metadata; coordinates are not embedded as a declared parquet field.
- `data-bin/cameo2022`: `aatype.fasta`, `struct.fasta`, and local `preprocessed/*.pkl` coordinate features.
- `data-bin/PDB_date`: `aatype.fasta`, `struct.fasta`, and local nested `preprocessed/**/*.pkl` coordinate features.
- `src/byprot/datamodules/pdb_dataset/pdb_datamodule.py:523-560`: the coordinate loader formats `processed_path`, reads the external pickle, then calls `PdbDataset.process_chain`.
- `src/byprot/datamodules/dataset/tokenized_protein.py:266-316`: the DPLM training dataset consumes only `struct_seq` and `aa_seq`; it does not load coordinates.

### Feasibility conclusion

- CAMEO 2022 and PDB date contain locally available GT pickles and are feasible for read-only smoke/oracle construction and evaluation.
- Full PDB+SwissProt FM training is currently **NOT CONFIRMED / not locally feasible from the parquet alone**. The parquet stores `processed_path`; sampled visible values such as `cameo2022/pdb3/preprocessed/7dz2_C.pkl` do not resolve at that relative location in this workspace. Some matching evaluation pickles exist elsewhere, but that does not establish availability of the 220k training coordinate corpus.
- Before dataset implementation, audit every `processed_path` against a user-supplied root and report resolvable counts. Never infer or rewrite missing roots. If coordinate pickles are supplied, use the existing `PdbDataset.process_chain`/`collate_fn` contract to obtain `all_atom_positions`, `res_mask`, `seq_length`, and `aatype`.
- Do not train on CAMEO/PDB-date evaluation targets merely because their coordinates are local.

## 8. Unconfirmed issues

- **NOT CONFIRMED:** paper ResDiff training uses Mode A teacher-forced GT `z_quant,h`, Mode B generated/prediction-aware `z_quant,h`, or another pairing. No official ResDiff implementation was found.
- **NOT CONFIRMED:** for Mode A, how transformer hidden states are obtained and paired with GT quantized tokens (clean teacher-forced tokens versus noised tokens/timestep).
- **NOT CONFIRMED:** whether the condition uses structure-half states, AA-half states, both modalities, or a fusion.
- **NOT CONFIRMED:** whether the layer mixture includes the embedding state (34 tensors) or only 33 transformer layer outputs.
- **NOT CONFIRMED:** exact AdaLN architecture and residual-module implementation used in the paper.
- **NOT CONFIRMED:** availability of the full PDB+SwissProt coordinate/GVP files referenced by parquet paths.
- **NOT CONFIRMED:** whether paper reference checkpoints/configuration needed to reproduce the exact Table 4 ResDiff numbers are public and locally available.
- **NOT CONFIRMED:** optimal Euler step count, latent regularization, or clipping. These must be explicit ablations, not assumptions.

Mode A learns quantization-error recovery:

```text
GT coordinates -> z_cont -> LFQ z_quant -> r_target
```

Mode B learns prediction-aware refinement:

```text
sequence -> generated predicted z_quant and final hidden states -> refinement target
```

These are not interchangeable. For the first fair ResDiff comparison, block implementation at the dataset/conditioning stage until the paper or an authoritative artifact confirms the training condition. If it remains unavailable, label Mode A and Mode B as separate experimental baselines; do not mix their samples or claims.

## 9. Proposed new files

All future work remains isolated under `experiments/latent_residual_fm/`:

```text
experiments/latent_residual_fm/
    implementation_plan.md       # this analysis
    debug_interfaces.py           # read-only shape/alignment/freeze assertions
    model.py                      # condition builder + LatentResidualFlowMatcher
    flow.py                       # FM path/loss and decoupled Euler sampler
    dataset.py                    # coordinate rows and masks; explicit Mode A/B
    build_dataset.py              # optional cached latent/condition production
    train.py                      # single-GPU configurable trainer
    sample_folding.py             # DPLM generation, final forward, FM, decode
    evaluate_folding.py           # adapter to official evaluator/metrics
    configs/
        baseline.yaml
```

Do not modify or copy the coordinate-residual logic from `train_fm_refiner.py` or `build_residual_dataset.py`. Future CLI/config must expose `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_steps`, `warmup_steps`, `num_flow_steps`, `num_workers`, and `precision`. Default physical batch must fit one RTX 2000 Ada; paper batch 240 is only an effective-batch reference.

## 10. Implementation order

1. Add `debug_interfaces.py`: load exact frozen checkpoints; print runtime `H`, layer count, tuple length; assert vocabulary offset and token ranges; run one local CAMEO pickle through encode, LFQ, residual, rank-3 decode; verify atom37 shapes.
2. Add a final-token hidden alignment test: one short folding generation, extra final forward, split/remove BOS/EOS, and assert equality of all residue axes and masks. Also prove that public `generate()` lacks hidden states.
3. Resolve the training-condition blocker from paper/authoritative artifacts. Record explicit Mode A or Mode B; otherwise implement them as separately named experiments, never a silent hybrid.
4. Audit PDB+SwissProt `processed_path` resolvability. Stop dataset construction if GT coordinates are unavailable; do not use evaluation sets as training substitutes.
5. Implement `model.py` and `flow.py` with unit tests for shapes, softmax layer weights, padding invariance, exact masked loss, Euler determinism under a seed, and no base gradients.
6. Implement a tiny overfit test on a few training-authorized coordinate examples. Check loss decrease and oracle reconstruction comparisons (`z_quant` decode versus `z_cont` decode).
7. Build the authorized training dataset/cache with provenance, checkpoint identity, dtype, mask, and conditioning-mode metadata. Avoid caching all 34 full 1280-wide states unless storage is measured and justified.
8. Train the 6-layer/1024 baseline with configurable accumulation, precision, warmup, LR schedule, and checkpoints. Log latent statistics and effective batch size.
9. Implement folding inference using predicted `z_quant`, extra final hidden forward, Euler FM sampling, continuous-latent decode, and seeded multiple samples if specified.
10. Evaluate identical CAMEO 2022 and PDB-date targets with the official evaluator: DPLM-2 Bit baseline versus Bit + Latent Residual FM; report RMSD and TM-score. Keep paper ResDiff and Section 3.3 data-space FM numbers as references, not locally reproduced results unless actually reproduced.

| Item                                         | Confirmed? | Source | Shape / value |
| -------------------------------------------- | ---------- | ------ | ------------- |
| z_cont access                                | Yes | `structok_lfq.py:138-159`, `VQModel.encode` first return | `[B,L,13]` pre-quant latent |
| z_quant access                               | Yes | `structok_lfq.py:121-123`; `lfq.py:272-389` | `[B,L,13]`, forward values `-1/+1` |
| latent dimension                             | Yes | `lfq.py:153-160`; cached Bit config | `13 = log2(8192)` |
| DPLM hidden states                           | Yes | `dplm2_bit.py:139-147`; Bit modeling `388-444,672-712` | tuple of `[B,2S,H]`; final `[B,2S,H]` |
| DPLM hidden size                             | Yes | cached exact `dplm2_bit_650m/config.json`; runtime source is `model.net.config.hidden_size` | `1280` for cached exact checkpoint |
| number of hidden layers                      | Yes | cached exact config; Bit modeling `360,388-444` | 33 transformer layers; 34 tuple entries |
| structure modality slice                     | Yes for tensor ordering; conditioning choice NOT CONFIRMED | `dplm2_bit.py:115-137`; Bit modeling `703-705` | first half `[B,S,H]`; use in ResDiff condition NOT CONFIRMED |
| BOS/EOS handling                             | Yes | `generate_dplm2.py:30-103`; `dplm2_bit.py:438-459` | per modality `S=L+2`; remove structure BOS/EOS, mask padding |
| LFQ token offset                             | Yes | `dplm2_bit.py:80-85,124-128,321-340,438-459` | combined structure ID = raw LFQ ID + 36 |
| decoder continuous latent input              | Yes | `structok_lfq.py:161-176,225-255`; `generate_dplm2.py:416-423` | rank-3 `[B,L,13]` -> atom37 `[B,L,37,3]` |
| training coordinate availability             | NOT CONFIRMED for PDB+SwissProt; yes for local eval sets | local directory audit; dataset README; PDB loader `523-560` | parquet has paths, not coordinates; CAMEO/PDB-date have `.pkl` |
| official ResDiff implementation availability | No public implementation found | repository-wide `AdaLN/ResDiff/residual diffusion` search | NOT CONFIRMED outside this repository |

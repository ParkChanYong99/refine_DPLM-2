# Latent residual flow: repository map

## Scope and terminology

This map is based on the repository at `/mnt/ext-vol/dplm` on 2026-08-04. No existing source file was changed during the analysis. In particular, `train_fm_refiner.py` and `build_residual_dataset.py` were only read.

For this experiment, the repository's closest concrete definitions are:

- `encoder_feats`: the frozen GVP/ESM-IF1 encoder output, shape `[B, L, 512]`.
- `e_cont`: `VQModel.encode()`'s `pre_quant`, i.e. the output of `pre_quant` immediately before `LFQ.forward()`, shape `[B, L, 13]`.
- `e_quant`: the first return value of `LFQ.forward()`, shape `[B, L, 13]`, with LFQ values `-1/+1` (and a straight-through estimator during training).
- `latent_residual = e_cont - e_quant`, shape `[B, L, 13]`.
- `refined_latent = e_quant + predicted_residual`, shape `[B, L, 13]`.

The value that should be called `e_cont` for the stated objective is `pre_quant`, not the raw 512-dimensional `encoder_feats`: `pre_quant` and `e_quant` are in the same 13-dimensional LFQ space and can therefore be subtracted and added directly.

## End-to-end path at a glance

```text
atom37 coordinates [B,L,37,3]
  -> VQModel.encode
  -> GVPTransformerEncoderWrapper2.forward
     -> ESM-IF1 GVP encoder using backbone atoms [:,:,:3,:]
  -> encoder_feats [B,L,512]
  -> VQModel.pre_quant
  -> e_cont/pre_quant [B,L,13]
  -> LFQ.forward
     -> e_quant [B,L,13] and struct token indices [B,L]
  -> VQModel.decode(e_quant or refined_latent, aatype, mask)
     -> post_quant MLP [13 -> 128]
     -> 4-layer TransformerEncoder [B,L,128]
     -> ESMFoldStructureDecoder(emb_s=[B,L,128], emb_z=None, ...)
  -> final_atom_positions [B,L,37,3]
```

## 1. `get_struct_tokenizer()` and checkpoint loading

Definition: `src/byprot/models/utils.py:387-412`, function `get_struct_tokenizer(model_name_or_path="airkingbd/struct_tokenizer", eval_mode=True)`.

Loading sequence:

1. Imports `VQModel` from `byprot.models.structok.structok_lfq` (`utils.py:390`).
2. If `model_name_or_path` exists locally, uses `<path>/.hydra` as `root_path` (`utils.py:392-394`).
3. Otherwise calls Hugging Face `snapshot_download(repo_id=model_name_or_path)` and uses the returned snapshot directory directly (`utils.py:395`).
4. Loads `<root_path>/config.yaml` with `load_yaml_config` (`utils.py:396`).
5. Instantiates `VQModel(**cfg)` (`utils.py:397`). The downloaded config is therefore expected to contain the constructor-level `encoder_config`, `decoder_config`, and `codebook_config`, rather than the full training Hydra tree.
6. Loads `<root_path>/dplm2_struct_tokenizer.ckpt` on CPU (`utils.py:398-401`).
7. Applies the state dict with `strict=False`, prints missing/unexpected keys, freezes every parameter with `requires_grad_(False)`, and returns eval mode by default (`utils.py:402-412`).

Both DPLM-2 wrappers lazily obtain this tokenizer through `MultimodalDiffusionProteinLanguageModel.struct_tokenizer` at `src/byprot/models/dplm2/dplm2.py:197-204`. The 650M experiment configs set `struct_tokenizer.exp_path: airkingbd/struct_tokenizer` at `configs/experiment/dplm2/dplm2_650m.yaml:54-55` and `configs/experiment/dplm2/dplm2_bit_650m.yaml:54-55`. `DPLM2Bit.__init__` forces early loading because it reads `self.struct_tokenizer.codebook_embed_dim` (`dplm2_bit.py:47-51`).

Runtime verification in this workspace restored `airkingbd/struct_tokenizer` with 0 missing and 0 unexpected keys.

## 2. Actual structure encoder forward path

The concrete model class is `VQModel` (`src/byprot/models/structok/structok_lfq.py:32-113`). Its encoder is `GVPTransformerEncoderWrapper2` (`structok_lfq.py:23,55`).

`VQModel.forward()` calls `VQModel.encode()` with `batch["all_atom_positions"]`, `batch["res_mask"]`, and `batch["seq_length"]` (`structok_lfq.py:113-123`). `tokenize()` uses the same `encode()` path (`structok_lfq.py:214-223`).

Inside `VQModel.encode()` (`structok_lfq.py:138-159`):

1. It constructs a length-based `padding_mask` from `seq_length` (`143-146`).
2. It calls `self.encoder(backb_positions=atom_positions, mask=mask, padding_mask=padding_mask)` (`147-151`).
3. `GVPTransformerEncoderWrapper2.forward()` is at `src/byprot/models/structok/modules/gvp_encoder.py:70-89`.
4. It takes only `backb_positions[:, :, :3, :]` (`gvp_encoder.py:72`), i.e. N/CA/C from atom37 input. It combines residue and padding masks, fills invalid coordinates with NaN, and passes coordinates, padding mask, and confidence into the ESM-IF1 GVP encoder (`73-82`).
5. The ESM encoder returns sequence-first features; the wrapper publishes `encoder_out["out"] = encoder_out["encoder_out"][0].transpose(0, 1)` (`84`), producing batch-first `[B,L,512]`.
6. `VQModel.encode()` explicitly detaches this encoder output (`structok_lfq.py:147-151`), applies `pre_quant`, and zeroes it at masked residues (`153-159`).

`GVPTransformerEncoderWrapper2.__init__()` loads `esm.pretrained.esm_if1_gvp4_t16_142M_UR50()` and uses its encoder (`gvp_encoder.py:53-66`). The encoder embedding width is obtained dynamically; the tested checkpoint produces 512.

## 3. Continuous latent immediately before LFQ

Function: `VQModel.encode()` at `src/byprot/models/structok/structok_lfq.py:138-159`.

```python
e_cont, encoder_feats = struct_tokenizer.encode(
    atom_positions=batch["all_atom_positions"],  # [B,L,37,3]
    mask=batch["res_mask"],                      # [B,L]
    seq_length=batch["seq_length"],              # [B]
)
```

`e_cont` is named `pre_quant` in source. `self.pre_quant` is `LayerNorm(512) -> Linear(512,13) -> ReLU -> Linear(13,13)` for the loaded tokenizer (`structok_lfq.py:64-70`). The configured LFQ dimension is 13 because the codebook has 8192 = 2^13 entries (`configs/experiment/structok/structok_lfq_8k_pdb_swissprot_c512.yaml:52-57`; the downloaded checkpoint was also runtime-verified as 13-dimensional).

Shape: `[B,L,13]`, floating point. Masked positions are multiplied by zero at `structok_lfq.py:158`.

There is no public method specifically named `get_continuous_latent`; `encode()` is the existing direct access point.

## 4. LFQ quantized latent

Function: `LFQ.forward()` at `src/byprot/models/structok/modules/lfq.py:272-395`.

```python
e_quant, quantizer_loss, (_, _, struct_ids) = struct_tokenizer.quantize(
    e_cont,
    mask=batch["res_mask"].bool(),
)
```

Input `[B,L,13]` is internally reshaped to `[B,L,1,13]` (`lfq.py:287-291`). Quantization is componentwise sign quantization using `torch.where(x > 0, +1, -1)` (`293-296`). The straight-through expression is applied at `363`, dimensions are merged back at `367-371`, and the first returned value is `[B,L,13]` (`389`). Token indices are reconstructed as `[B,L]` at `381-383`.

Shapes and dtypes verified on the CAMEO sample:

- `e_quant`: `[1,286,13]`, `float32`.
- `struct_ids`: `[1,286]`, `int64`.
- `latent_residual`: `[1,286,13]`, `float32`.

Mask caveat: `mask` governs losses and index reshaping, but the quantized vector itself is generated by sign thresholding. Since masked `e_cont` is zero, its raw quantized vector becomes all `-1`, corresponding to token index 0. The structure decoder separately receives `res_mask`, so invalid residues are masked downstream.

## 5. Token index to quantized latent path

The canonical conversion is `LFQ.get_codebook_entry(x)` at `src/byprot/models/structok/modules/lfq.py:218-239`:

```text
struct_ids [B,L] int64
  -> x.unsqueeze(-1) & big-endian bit mask [13]
  -> boolean bits [B,L,13]
  -> bits * 2 - 1
  -> e_quant [B,L,13] with values -1/+1
```

The big-endian mask is registered in `LFQ.__init__()` at `lfq.py:180-185`. `VQModel.detokenize()` invokes `get_codebook_entry` for 2-D token input (`structok_lfq.py:225-229`). `VQModel.get_decoder_features()` does the same at `structok_lfq.py:195-212`.

In DPLM-2 Bit, combined-vocabulary structure IDs include `struct_vocab_offset = 36` (`src/byprot/models/dplm2/dplm2_bit.py:80-85`). The offset is subtracted before codebook lookup in `DPLM2Bit.forward()` (`124-128`) and `prepare_for_struct_tokenizer()` (`438-459`). Conversely, sampled LFQ bits are reduced to an integer and offset 36 is added at `dplm2_bit.py:321-340`.

Important distinction: raw structure-tokenizer indices are in `[0,8191]`; DPLM-2's shared vocabulary representation uses its own special tokens/offset. Do not pass offset DPLM IDs directly to `LFQ.get_codebook_entry()`.

## 6. Exact structure decoder input format

There are two levels:

### `VQModel.decode()` public model-level input

Defined at `src/byprot/models/structok/structok_lfq.py:161-176`:

- `quant`: float tensor `[B,L,13]`; this can be `e_quant` or `refined_latent`.
- `aatype`: integer tensor `[B,L]`.
- `mask`: binary/float tensor `[B,L]`.
- `decoder_kwargs`: dict forwarded as keyword arguments.

`decode()` applies `post_quant["mlp"]`, mapping 13 to decoder `input_dim=128`, then a 4-layer, 8-head `TransformerEncoder` (`structok_lfq.py:75-87,161-167`; config at `configs/experiment/structok/structok_lfq_8k_pdb_swissprot_c512.yaml:60-72`).

### `ESMFoldStructureDecoder.forward()` low-level input

Called at `structok_lfq.py:168-175`, definition at `src/byprot/models/structok/modules/folding_utils/decoder.py:185-260`:

- `emb_s`: float `[B,L,128]` after `post_quant`.
- `emb_z`: `None` in this path; when attention-map mode is disabled, the decoder creates zero pair features `[B,L,L,32]` (`decoder.py:243-249`).
- `mask`: `[B,L]`.
- `aa`: int `[B,L]`.
- `esmaa`: int `[B,L]`.
- optional `residx`, `masking_pattern`, `num_recycles`, `return_features_only`.

The tokenizer normally supplies all-zero `aatype` during `detokenize()` (`structok_lfq.py:241-246`). The decoder outputs `final_atom_positions [B,L,37,3]` at `decoder.py:327-334` along with masks, confidence, frames, and other structure-module outputs.

## 7. Can the decoder accept continuous latent directly?

Yes, confirmed both statically and at runtime, provided it has shape `[B,L,13]` in the tokenizer's pre/post-LFQ latent space.

Two supported routes exist:

```python
# Most explicit route
raw_decoder_out = struct_tokenizer.decode(
    quant=refined_latent,
    aatype=torch.zeros(B, L, dtype=torch.long, device=device),
    mask=res_mask,
)

# Convenience route; returns normalized detokenize output keys
decoder_out = struct_tokenizer.detokenize(
    refined_latent,       # ndim == 3, so no codebook lookup
    res_mask=res_mask,
)
```

`detokenize()` explicitly treats a rank-3 input as `quant = struct_tokens` (`structok_lfq.py:225-230`) and then calls `decode()` (`245-247`). Runtime testing passed `e_quant [1,286,13]` through this route and obtained atom37 coordinates `[1,286,37,3]`.

The argument name `struct_tokens` is misleading for rank-3 inputs, but no token-ID conversion occurs in that branch. There is no check that the last dimension equals 13; a wrong last dimension will fail later in `post_quant["mlp"]`.

## 8. Internal `detokenize(struct_ids)` path

`VQModel.detokenize()` is at `src/byprot/models/structok/structok_lfq.py:225-255`.

For `struct_ids [B,L]`:

1. `LFQ.get_codebook_entry(struct_ids)` produces `[B,L,13]` (`225-228`).
2. If absent, `res_mask` defaults to all ones `[B,L]` (`237-240`).
3. It creates all-zero `_aatypes [B,L]` with `int64` dtype (`241-243`).
4. Calls `VQModel.decode(quant, _aatypes, res_mask, kwargs)` (`245-247`).
5. `decode()` applies the post-quant MLP/transformer and invokes `ESMFoldStructureDecoder` as described above.
6. The raw decoder result is reduced/renamed to (`248-255`):
   - `atom37_positions = final_atom_positions`, `[B,L,37,3]`;
   - `atom37_mask = atom37_atom_exists`, `[B,L,37]`;
   - `aatype = argmax(lm_logits)`, `[B,L]`;
   - `residue_index`, `[B,L]`;
   - `plddt`, expected `[B,L,37]` from the decoder head.

For rank-3 `[B,L,13]` input, step 1 is skipped and the input is used as `quant` directly.

## 9. DPLM-2 and DPLM-2 Bit final hidden state

### DPLM-2 650M

`MultimodalDiffusionProteinLanguageModel.forward()` builds embeddings and calls `self.net(...)` at `src/byprot/models/dplm2/dplm2.py:229-270`. The underlying LM wrapper returns:

```python
{
    "logits": logits,
    "last_hidden_state": sequence_output,
}
```

at `src/byprot/models/dplm2/modules/dplm2_modeling_esm.py:651-686`; `sequence_output = outputs[0]` at lines 679-685. Therefore:

```python
net_out = dplm2(input_ids)
h_final = net_out["last_hidden_state"]  # [B, 2*S, H]
h_struct, h_aa = h_final.chunk(2, dim=1) # each [B,S,H]
```

Here `S` includes the modality's BOS/EOS/padding positions in the tokenized batch. The iterative decoder also exposes the same tensor as `hidden_states=net_out["last_hidden_state"]` at `dplm2.py:477-552`, but `generate()` does not include it in its final return (`dplm2.py:741-827`). For residual training, direct `forward()` or a deliberate capture inside the generation loop is therefore required.

### DPLM-2 Bit 650M

`DPLM2Bit.forward()` is at `src/byprot/models/dplm2/dplm2_bit.py:87-147`. It calls the Bit LM with `output_hidden_states=True` (`139-145`). The underlying wrapper returns both:

- `last_hidden_state = sequence_output`, `[B,2*S,H]` (`src/byprot/models/dplm2/modules/dplm2_bit_modeling_esm.py:701-711`);
- `all_hidden_states = outputs["hidden_states"]` when requested (`710-711`).

Thus the final state is directly `net_out["last_hidden_state"]`; the last entry of `all_hidden_states` should represent the same final encoder output, but using `last_hidden_state` avoids dependence on tuple conventions. `DPLM2Bit.forward_decoder()` currently propagates only `all_hidden_states` (`dplm2_bit.py:311-319`), and `generate()` ultimately returns `final_struct_feature`, which is LFQ codebook latent `[B,S-2,13]`, not the 650M transformer hidden state (`dplm2_bit.py:419-461`).

The exact `H` is read from the downloaded model's `config.hidden_size`. It is expected to be 1280 for the 650M ESM-family checkpoint, but this repository's YAML does not pin the numeric value. **Confirmation required:** print `model.net.config.hidden_size` for the exact DPLM-2 and Bit checkpoints used in the experiment rather than hard-coding 1280.

Also confirm which positions condition the refiner: structure half only, amino-acid half only, or fused/both. The repository establishes tensor ordering (structure half followed by amino-acid half) but not the new experiment's conditioning choice.

## 10. Existing FAPE loss

Primary implementation: `src/byprot/models/structok/modules/loss.py`.

- `compute_fape()` at lines `174-244`: transforms predicted and target points into local frame coordinates, computes clamped/scaled distances, applies frame/position masks, and normalizes.
- `backbone_loss()` at lines `247-302`: constructs predicted/ground-truth `Rigid` frames and calls `compute_fape`, optionally mixing clamped and unclamped FAPE.
- `sidechain_loss()` at lines `305-352`: resolves alternate naming, flattens sidechain frames/atoms, and calls `compute_fape`.
- `fape_loss()` at lines `355-379`: weighted backbone plus optional sidechain loss.
- `StructureVQLoss` dispatches the configured `fape` term near `loss.py:1802-1806`; default configuration entries are near `loss.py:1895` onward.

The decoder raw output retains `sm["frames"]` and `sm["positions"]` at `src/byprot/models/structok/modules/folding_utils/decoder.py:327-334`. Note that `VQModel.detokenize()` discards `sm` when it builds its compact return dict (`structok_lfq.py:248-255`). Training with the repository FAPE wrapper therefore needs `VQModel.decode()`'s raw output, not only the compact `detokenize()` result, plus the ground-truth batch features expected by `fape_loss()`.

**Confirmation required:** the new latent-refiner training objective (latent MSE/flow loss only versus an added coordinate/FAPE term) is not defined by existing code and should be specified before implementation.

## 11. Relevant file/class/function index

| Purpose | File | Symbol / lines |
|---|---|---|
| Tokenizer factory/checkpoint | `src/byprot/models/utils.py` | `get_struct_tokenizer`, 387-412 |
| Tokenizer model | `src/byprot/models/structok/structok_lfq.py` | `VQModel`, 32 onward |
| Full encode/quantize/decode | same | `forward`, 113-136 |
| Continuous latent | same | `encode`, 138-159 |
| Continuous decoder entry | same | `decode`, 161-176 |
| Quantize then decode | same | `quantize_and_decode`, 178-193 |
| Tokenization | same | `tokenize`, 214-223 |
| Token/latent detokenization | same | `detokenize`, 225-255 |
| Structure encoder wrapper | `src/byprot/models/structok/modules/gvp_encoder.py` | `GVPTransformerEncoderWrapper2`, 53-89 |
| LFQ construction/index lookup | `src/byprot/models/structok/modules/lfq.py` | `LFQ.__init__`, 133-200; `get_codebook_entry`, 218-239 |
| LFQ forward | same | `LFQ.forward`, 272-395 |
| Post-quant transformer | `src/byprot/models/structok/modules/nn.py` | `TransformerEncoder`, 15-90 |
| Structure decoder | `src/byprot/models/structok/modules/folding_utils/decoder.py` | `ESMFoldStructureDecoder`, 63 onward; `forward`, 185-334 |
| Tokenizer dimensions | `configs/experiment/structok/structok_lfq_8k_pdb_swissprot_c512.yaml` | model config, 50-72 |
| DPLM-2 wrapper/hidden state | `src/byprot/models/dplm2/dplm2.py` | `forward`, 229-270; `forward_decoder`, 477-552 |
| DPLM-2 LM output | `src/byprot/models/dplm2/modules/dplm2_modeling_esm.py` | LM `forward`, 651-686 |
| Bit wrapper/latent conversion | `src/byprot/models/dplm2/dplm2_bit.py` | `forward`, 87-147; `prepare_for_struct_tokenizer`, 438-461 |
| Bit LM output | `src/byprot/models/dplm2/modules/dplm2_bit_modeling_esm.py` | LM `forward`, 672-712 |
| FAPE | `src/byprot/models/structok/modules/loss.py` | `compute_fape`, 174-244; `fape_loss`, 355-379 |
| CAMEO pkl adapter (reference only) | `build_residual_dataset.py` | `load_pkl`, 45-48; `make_batch_from_pkl`, 99 onward; existing tokenize/detokenize call, 203-253 |

## 12. Minimal CAMEO shape check

The following is a read-only probe. It does not edit either existing script. It reuses the existing pkl adapter and runs one sample through encoder, LFQ, and rank-3 latent detokenization.

```bash
cd /mnt/ext-vol/dplm
PYTHONPATH=src conda run -n dplm python - <<'PY'
import torch

from build_residual_dataset import load_pkl, make_batch_from_pkl
from byprot.models.utils import get_struct_tokenizer

pkl_path = "data-bin/cameo2022/preprocessed/7dz2_C.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_struct_tokenizer().to(device).eval()
batch = make_batch_from_pkl(load_pkl(pkl_path), device)

with torch.no_grad():
    e_cont, encoder_feats = model.encode(
        batch["all_atom_positions"],
        batch["res_mask"],
        batch["seq_length"],
    )
    e_quant, _, (_, _, struct_ids) = model.quantize(
        e_cont, mask=batch["res_mask"].bool()
    )
    latent_residual = e_cont - e_quant
    decoded = model.detokenize(e_quant, res_mask=batch["res_mask"])

print("inputs:", batch["all_atom_positions"].shape, batch["res_mask"].shape)
print("encoder_feats:", encoder_feats.shape, encoder_feats.dtype)
print("e_cont:", e_cont.shape, e_cont.dtype)
print("e_quant:", e_quant.shape, e_quant.dtype)
print("struct_ids:", struct_ids.shape, struct_ids.dtype)
print("latent_residual:", latent_residual.shape)
print("decoded atom37:", decoded["atom37_positions"].shape)
PY
```

Observed in this workspace for `7dz2_C.pkl` (length 286):

```text
inputs:             [1,286,37,3], [1,286]
encoder_feats:      [1,286,512] float32
e_cont:             [1,286,13] float32
e_quant:            [1,286,13] float32
struct_ids:         [1,286] int64 (observed range 0..8125)
latent_residual:    [1,286,13]
decoded atom37:     [1,286,37,3]
decoded atom37 mask:[1,286,37]
```

The pkl itself contains `atom_positions [286,37,3] float64`, `atom_mask [286,37]`, `aatype [286]`, `bb_mask [286]`, and related metadata. The adapter converts coordinates/masks to float32 and adds batch dimension.

## Remaining items that require an experiment-level decision

- Exact DPLM checkpoint hidden width should be read from `model.net.config.hidden_size` at runtime for both 650M variants; do not rely on the expected value 1280.
- Decide how BOS/EOS are removed when aligning DPLM hidden positions `[B,S,...]` with tokenizer latents `[B,L,13]`. `DPLM2Bit.prepare_for_struct_tokenizer()` demonstrates BOS/EOS removal at `dplm2_bit.py:438-459`, but the new refiner dataset path is not yet defined.
- Decide whether the refiner conditions on the structure-half final state, amino-acid-half state, both, or another pooling/fusion. This cannot be inferred from the repository.
- Decide whether refined latent values remain unconstrained or are regularized/clipped. The decoder accepts arbitrary `[B,L,13]` floats, but the checkpoint was trained on LFQ values near exactly `-1/+1` at its decoder input.
- Decide whether training uses only latent residual regression/flow matching or additionally decodes and applies FAPE. Existing FAPE infrastructure supports the latter through raw `decode()` output, but the requested comparison does not itself specify the loss composition.

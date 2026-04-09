# Tensor Layout & GGUF Integration

## Current state

The Qwen3.5-27B GGUF has 1 MTP layer at index 64 (the model has 64 + 1 = 65 layers total). Loader references:

- `src/llama-model.cpp:1789` — reads `LLM_KV_NEXTN_PREDICT_LAYERS` into `hparams.nextn_predict_layers`
- `src/llama-model.cpp:5493-5532` — iterates `i >= n_layer - nextn_predict_layers` and loads per-layer `nextn.*` tensors

So the loader *already* supports `nextn_predict_layers > 1`; the only reason it currently ends at 1 is that `convert_hf_to_gguf.py` writes a single MTP block and the upstream HF checkpoint only has one.

## Target layout for N=4

Layers 64, 65, 66, 67. GGUF tensor names (per existing `LLM_TENSOR_NEXTN_*` in `src/llama-arch.cpp`):

```
blk.64.nextn.enorm.weight               [5120]
blk.64.nextn.hnorm.weight               [5120]
blk.64.nextn.eh_proj.weight             [5120, 10240]     # 2*n_embd -> n_embd
blk.64.nextn.shared_head_norm.weight    [5120]
blk.64.attn_norm.weight                 [5120]
blk.64.attn_q.weight                    [5120, 5120]
blk.64.attn_k.weight                    [5120, 1024]      # GQA
blk.64.attn_v.weight                    [5120, 1024]
blk.64.attn_output.weight               [5120, 5120]
blk.64.attn_q_norm.weight               [128]
blk.64.attn_k_norm.weight               [128]
blk.64.ffn_norm.weight                  [5120]
blk.64.ffn_gate.weight                  [5120, 17408]
blk.64.ffn_up.weight                    [5120, 17408]
blk.64.ffn_down.weight                  [17408, 5120]
# optional (absent when mtp_use_dedicated_embeddings=False):
# blk.64.nextn.embed_tokens.weight
# blk.64.nextn.shared_head_head.weight

blk.65.nextn.*    # head_2, identical shapes
blk.66.nextn.*    # head_3
blk.67.nextn.*    # head_4
```

**KV metadata**:
```
qwen3.nextn_predict_layers = 4
```

That's the only new metadata. Everything else is reused.

## Storage estimate

Per head (Q4_K_M):
- attn q/k/v/o: (5120·5120 + 5120·1024 + 5120·1024 + 5120·5120) bytes @ ~0.56 B/param ≈ 29 MB
- ffn gate/up/down: 3 · 5120 · 17408 @ ~0.56 B/param ≈ 150 MB
- eh_proj: 10240 · 5120 @ 0.56 ≈ 29 MB
- norms: < 1 MB
- **Per head: ~210 MB Q4_K_M**
- **N=4 total added: ~840 MB**

Current model is ~16 GB → +5.25%. Acceptable.

## `convert_hf_to_gguf.py` changes

The current converter treats the single MTP layer as a one-off. Generalization:

1. HF checkpoint layout (hypothetical, post-training): the trained weights live as `mtp.layers.0`, `mtp.layers.1`, `mtp.layers.2`, `mtp.layers.3` in a safetensors shard named `mtp_heads.safetensors`.
2. Loop:
```python
for k in range(num_mtp_layers):
    gguf_block_idx = n_main_layers + k       # 64 + k
    for hf_name, gguf_suffix in MTP_TENSOR_MAP.items():
        src = state_dict[f"mtp.layers.{k}.{hf_name}"]
        tensor_name = f"blk.{gguf_block_idx}.{gguf_suffix}"
        gguf_writer.add_tensor(tensor_name, src.numpy())
```
3. Write `qwen3.nextn_predict_layers = num_mtp_layers`.
4. Shared weights (`mtp.norm`, `mtp.pre_fc_norm_embedding`, `mtp.fc`) — reconsider. Currently a single `eh_proj`/`enorm`/`hnorm` per block. Per-head versions are cleaner than trying to share norms across heads.

## Quantization

The existing `llama-quantize` pipeline handles `blk.N.nextn.*` tensors as ordinary block tensors — the added heads will be quantized Q4_K_M identically to the main model. No special casing needed as long as the `LLM_TENSOR_NEXTN_*` entries in `src/llama-arch.cpp` already list the quantization tier (they do — inherited from the existing MTP path).

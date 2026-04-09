# Per-Position MTP Heads — Architecture Spec

**Status**: design-only, pre-training. No GPU spent.
**Target model**: Qwen3.5-27B Q4_K_M (hybrid DeltaNet + full-attention), current branch `mtp-dispatch` @ `b070bed01`.
**Baseline**: plain decode 17.90 tok/s; K=1 vanilla MTP 7.64 tok/s (0.43×). Single-head spec loses on this model because the draft pass costs almost as much as a main pass.

## Goal

Replace the single MTP head with **N independent heads** (N=4), each predicting a *fixed relative offset* from a shared main-model hidden state. DeepSeek V3 style.

```
main fwd -> h_t
  head_1(h_t, tok_{t})             -> logits for t+1
  head_2(h_t, tok_{t+1}_pred)      -> logits for t+2
  head_3(h_t, tok_{t+2}_pred)      -> logits for t+3
  head_4(h_t, tok_{t+3}_pred)      -> logits for t+4
verify (t+1..t+4) in ONE T=5 main pass -> commit up to 5 tokens
```

One main forward amortizes over K head forwards. Heads are cheap (single transformer block each).

## Per-head structure

Each `head_k` is a single dense transformer block identical in shape to the existing `mtp.layers.0` in `~/mlx-fork/mtp_weights.safetensors`:

- `input_layernorm` (RMS, [n_embd])
- `self_attn.{q,k,v,o}_proj` + q_norm, k_norm (GQA: n_head=40, n_head_kv=8, head_dim=128)
- `post_attention_layernorm`
- `mlp.{gate,up,down}_proj` (intermediate=17408)
- `eh_proj` [2*n_embd, n_embd] — fuses `concat(hnorm(h_t), enorm(embed(prev_tok)))` down to n_embd
- `hnorm`, `enorm` — pre-concat RMS norms
- **Shared** with main model: `embed_tokens`, final `norm`, `lm_head`

### Shared vs dedicated LM head tradeoff

| Option | Params added per head | Quality | Notes |
|--------|----------------------|---------|-------|
| **Shared LM head** (recommended) | 0 | +0 | matches current Qwen3.5 setup (`mtp_use_dedicated_embeddings: False`); head_k output goes through main `norm` + main `lm_head` |
| Dedicated LM head | 5120 * 151936 ≈ 778M fp16 / 195M Q4 | marginal | only justified if heads diverge in output distribution; DeepSeek V3 shares |

**Decision**: share. Per-head params ≈ 480M fp16 / 120M Q4. N=4 → ~480M Q4 added to the 16GB model ≈ 3% overhead.

## Training objective

Per training example (sequence of length L):

1. Run frozen main model once, collect all hidden states `h_1..h_L` at the final layer (pre-norm, same tap point that feeds the existing MTP head).
2. For each position t and each head k ∈ {1..N}:
   - Input: `h_t`, `embed(token_{t+k-1})` (the previous committed token along the chain — at train time this is the ground-truth token, at inference it's the head_{k-1} argmax)
   - Target: `token_{t+k}`
   - Loss: cross-entropy via shared `norm` + `lm_head`

Total loss = Σ_k w_k · CE_k. Use w_k = 1.0 for all heads (DeepSeek V3 uses uniform weights).

**Teacher forcing**: heads trained with ground-truth previous tokens. At inference, the chain uses argmax from the previous head — this introduces train-test skew. Mitigation: scheduled sampling in last 10% of training (20% probability of replacing GT with head_{k-1} argmax). Optional; DeepSeek V3 skipped this and it still worked.

## N=4 justification

- DeepSeek V3 uses 4.
- Acceptance rate decays roughly geometrically: if p = per-token acceptance ≈ 0.75, then expected accepted per cycle ≈ p + p² + p³ + p⁴ = 2.05 tokens drafted accepted, + 1 bonus = 3.05 committed.
- N=6+ hits diminishing returns (`p⁶` ≈ 0.18) while still costing 6 head forwards.
- Memory: N=4 heads ≈ 480MB Q4, fits comfortably.

## Critical risks on this hybrid model

1. **DeltaNet recurrent state in the main model is NOT captured by `h_t` alone.** The current MTP head inherits the main model's final-layer hidden `h_t`, which contains the *attention* output at position t — but DeltaNet layers also update a hidden recurrent state `S_t` that depends on the full token history. Each head_k acts as a dense-attention block, so it does NOT need recurrent state for its own compute, but **it implicitly assumes `h_t` summarizes everything up to t**. This is the same assumption the existing single MTP head makes, so it's not *new* risk — but it means **accept rates will be lower than on pure-attention models** (our 7.64 tok/s baseline already proves the single head struggles).

2. **Per-head cost vs main pass cost.** The diagnostic that killed single-head MTP: "draft pass costs about as much as a main pass." This is because on a hybrid model, `llama_decode(T=1)` is dominated by fixed overhead (KV cache bookkeeping, DeltaNet state update, graph allocation) rather than FLOPs. If a *head-only* graph can be built that bypasses the main model layers entirely and only runs the single MTP block (as `LLM_GRAPH_TYPE_MTP` already does in `src/models/qwen35.cpp:21`), head cost should be ~10-15ms vs main pass ~56ms. This is the load-bearing assumption of the whole design. **Verify empirically in Phase 0 before training**: instrument the existing `build_mtp_head` path and measure head-only wall clock.

3. **KV cache for each head.** Each head has its own self-attention and therefore its own KV cache growing at rate 1/main. With N=4 heads and sequence length 2048, that's 4 × (2048 × 8 × 128 × 2 × 2 bytes) ≈ 16 MB — negligible. Implementation note: the heads' KV caches must be rolled back on misprediction the same way the main KV is rolled back today.

4. **Shared `h_t` for all heads: stale hidden.** head_4 uses the same `h_t` as head_1 even though it's predicting 4 positions ahead. The DeepSeek V3 paper shows this still works because the head learns to compensate using `embed(token_{t+3})` as the dominant signal. But on our hybrid model the mismatch may hurt more.

## Cost model

Let main forward = M ms, head forward = H ms, acceptance per position = p.

- Cycle time = M + K·H
- Expected committed tokens per cycle = 1 + Σ_{k=1}^{K} p^k (bonus + geometric accepts)
- Throughput = committed / cycle_time

At M=56ms, H=12ms, K=4, p=0.7:
- cycle = 56 + 48 = 104 ms
- committed = 1 + 0.7 + 0.49 + 0.343 + 0.24 = 2.77
- tok/s = 2.77 / 0.104 = **26.6 tok/s** → **1.49× over plain 17.9 tok/s**

Optimistic (p=0.8): committed = 3.36, tok/s = 32.3 → **1.80×**.
Pessimistic (p=0.6): committed = 2.30, tok/s = 22.1 → **1.24×**.

**These numbers assume H=12ms.** If H ≈ M (as today with single-head spec), throughput is **worse** than plain regardless of p.

## Non-goals

- No new MTP loss formulation, no auxiliary losses beyond per-head CE.
- No dedicated LM heads.
- No tree search beyond the existing verify path.
- No training run in this task — scripts delivered, user decides whether to execute.

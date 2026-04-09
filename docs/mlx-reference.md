# MLX Reference — What the 1.68× Actually Is

## TL;DR — the MLX win is NOT per-position heads.

Searched:
- `~/optimizations/qwen-mtp-inference/` (exists, 40+ files)
- `~/mlx-fork/mtp_*.py`, `~/mlx-fork/mtp_weights*.safetensors`

The safetensors file `~/mlx-fork/mtp_weights_vanilla.safetensors` contains exactly **one** MTP block: `mtp.layers.0.*` (self-attn, MLP, norms, `mtp.fc` eh-projection). There is no `mtp.layers.1` or higher. The MLX model has a **single trained MTP head**, same as our llama.cpp GGUF.

## What the 1.68× actually comes from

File: `~/optimizations/qwen-mtp-inference/stacked_v2/stacked_v2.py`

The technique is **stacked speculative decoding**:

1. **Adaptive MTP chain** (`adaptive_mtp.py`): recurrently feed the single MTP head its own previous output to draft 1-6 positions before verification. Chain length is gated by confidence (`max(softmax) >= threshold`). This is *not* per-position heads — it's the same head applied K times, each call using the previous head output's hidden as the new `prev_hidden` input.

2. **0.8B draft model stacked on top**: when the MTP chain commits to length 2, the small 0.8B Qwen model drafts position 3, so the main model's T=4 verify can commit up to 4 tokens.

3. **Confidence gating**: low-confidence positions use T=2 (short verify), high-confidence use T=3/T=4. This amortizes the verify cost.

Relevant signals from the code:
- `c1 = mx.max(mtp1_probs); if c1.item() >= confidence_threshold: chain`
- `mtp2_logits, h_mtp2 = mtp_head(h_mtp1, embed_d1, lm_head)` ← **same mtp_head**, second call with NEW h from first call
- Stacked 0.8B draft: `draft_model(d1.reshape(1,1), cache=d_cache)`

## Why this matters for our design

The 1.68× on MLX is achieved **without** training new heads. It's algorithmic — chained recurrent application of the existing head + a second small draft model.

This is cheaper and less risky than per-position heads. **Before committing Phase 1+2 GPU ($165)**, strongly consider replicating MLX's technique in llama.cpp:

1. **Chained MTP draft** (0 training cost): modify `llama_mtp_draft` to loop K times, feeding `h_mtp_{k-1}` (the post-MTP hidden, not the shared main-model hidden) as input to call k. This requires capturing the intermediate hidden from `build_mtp_head` — currently the head returns only logits, but the internal residual-stream hidden is available right before the LM projection.
2. **Stacked 0.8B**: use an existing small Qwen as a `-md` draft model, but only invoke it when the MTP chain commits to length ≥ 2.
3. **Confidence gating**: compute `softmax(logits).max()` on each draft, skip chain if below threshold.

**If chained single-head approach reaches the goal, per-position heads become unnecessary.** If chained still loses because on our hybrid model the MTP draft call itself is too expensive (the bug that motivated this whole task), then per-position heads are the real fix — but only if Phase 0 confirms head_fwd ≪ main_fwd.

## The honest hierarchy of techniques

Ranked by risk / cost to ship:

| Approach | Training cost | Code change | Expected speedup |
|----------|--------------|-------------|------------------|
| 1. Chained single-head MTP (MLX-style) | $0 | moderate | 1.3-1.5x IF head_fwd ≪ main_fwd |
| 2. + Stacked 0.8B draft | $0 | larger | 1.5-1.7x (MLX-observed) |
| 3. Per-position heads (this doc) | ~$165 | moderate | 1.4-1.8x IF training lands |
| 4. All of the above combined | ~$165 | large | speculative |

**Recommendation**: do Phase 0 first. If head_fwd is cheap, try approach 1 before approach 3. If head_fwd is not cheap, neither approach works and spec decode is dead on this hybrid model without a different primitive (e.g. pruning MTP attention to be single-layer-linear).

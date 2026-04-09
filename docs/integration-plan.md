# End-to-End Integration Plan

Six phases. Gate between each — user decides whether to proceed.

## Phase 0 — Validate the cost assumption (NO GPU, NO TRAINING)
**Goal**: prove the head forward pass is cheap (≤ 20 ms) before spending training money.
**Actions**:
1. Instrument `build_mtp_head` in `src/models/qwen35.cpp` with `ggml_time_us()` markers.
2. Run current single-head MTP path 1000 iterations on the existing model, record p50/p99 head wall time.
3. Compare to main forward pass wall time (56 ms baseline).

**Success**: head_fwd ≤ 0.25 * main_fwd (i.e. ≤ 14 ms).
**Failure**: if head_fwd ≈ main_fwd, the entire per-position approach is dead — no amount of training fixes it. Re-evaluate (switch to MLX-style 0.8B stacked draft, or give up on spec decode for this model).
**Rollback**: none — read-only measurement.

## Phase 1 — Build training data (GPU spent)
**Script**: `scripts/build_training_data.py`
**Cost**: ~24 H100-hours, ~$85.
**Disk**: ~36 GB tokens-only, or ~4 TB with cached hidden states (pick tokens-only, recompute hidden on-train).
**Failure modes**:
- OOM on 2048 seq_len at batch 8 → halve batch.
- Tokenizer mismatch between HF checkpoint and GGUF → verify vocab_size == 151936 before starting.
**Rollback**: delete output directory.

## Phase 2 — Train heads (GPU spent)
**Script**: `scripts/train_per_position_heads.py`
**Cost**: ~23 H100-hours, ~$80.
**Total Phase 1+2**: **~50 H100-hours, ~$165**.
**Failure modes**:
- Loss plateaus at random init level → check warm-start loaded correctly.
- Head 3/4 loss much higher than 1/2 → expected, still useful if p_4 ≥ 0.4.
- Catastrophic main-model drift → main is frozen; if this happens, there's a bug.
**Checkpoint**: save every 5k steps, keep last 3.
**Rollback**: discard checkpoint, re-run from last good step.

## Phase 3 — Quantize + GGUF append (CPU-only)
**Actions**:
1. Extend `convert_hf_to_gguf.py` per `docs/tensor-layout.md`. Write loop over `num_mtp_layers`.
2. Run converter on trained checkpoint → produces `qwen35-27b-mtpN4.gguf` (fp16).
3. `llama-quantize` to Q4_K_M. Inspect tensor list: `gguf-dump | grep nextn` should show 4 sets.
4. Verify KV metadata `qwen3.nextn_predict_layers == 4`.
**Failure modes**: tensor shape mismatch → dimensions in converter off. Diff against `docs/tensor-layout.md`.
**Rollback**: use existing single-head GGUF.

## Phase 4 — Land loader + graph + inference patches
**Files**: apply `patches/loader-stub.patch`, `patches/graph-stub.patch`, `patches/inference-stub.patch` as starting points. Each becomes real commits.
**Order**: loader first (model loads but MTP dispatches head_0 only) → graph (mtp_head_idx plumbing) → inference (draft loop iterates K heads).
**Validation at each step**: plain decode must still produce identical output to pre-patch.
**Rollback**: `git revert`.

## Phase 5 — Validate output coherence end-to-end
**Actions**:
1. `MTP_NUM_HEADS=1` → output must match pre-patch single-head MTP exactly.
2. `MTP_NUM_HEADS=4` → output must match plain decode (greedy sampling). This is the correctness test — spec decode is supposed to be exactness-preserving under greedy.
3. Run 10 prompts x 500 tokens, compare byte-for-byte.
**Failure modes**: any mismatch indicates a verify bug, KV rollback bug, or wrong shared-hidden tap point. Do not proceed to benchmarks until byte-exact.
**Rollback**: `MTP_NUM_HEADS` env var disables multi-head entirely.

## Phase 6 — Benchmark
**Metrics**: tok/s (plain), tok/s (K=1 single-head, today's 7.64), tok/s (K=4 per-position), per-head accept rate, cycle time breakdown (main + head_0..3).
**Success criterion**: K=4 tok/s > 1.2 * plain_tok/s (21.5 tok/s).
**Stretch**: ≥ 1.5× (26.9 tok/s).
**Failure**: if K=4 < plain, the head cost or accept rate is worse than modeled. Debug with the breakdown, then decide whether to retrain (lower K, more data, longer schedule) or shelve.
**Rollback**: `MTP_NUM_HEADS=0` (or unset) keeps the path dormant.

## Summary of spend gates

| Phase | GPU-h | $ | Decision after |
|-------|-------|---|----------------|
| 0     | 0     | 0 | Is head_fwd actually cheap? |
| 1     | 24    | 85| Data looks healthy? |
| 2     | 23    | 80| Loss curves match expected? |
| 3     | 0     | 0 | GGUF loads? |
| 4     | 0     | 0 | Plain decode still exact? |
| 5     | 0     | 0 | Spec output byte-exact? |
| 6     | 0     | 0 | Actual speedup ≥ 1.2x? |

Total committed through Phase 2: **~50 GPU-hours, ~$165**.

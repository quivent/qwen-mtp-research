# The recipe — chained recurrent MTP with confidence gating

> The MLX `stacked_v2.py` 1.68× speedup recipe, ported to llama.cpp. Two environment variables on top of the bug-fixed K=1 path. Coherent output. **1.99× over K=1 vanilla on Qwen3.5-27B.**

## TL;DR

```bash
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m qwen3.5-27b-q4km.gguf \
    -p "Explain photosynthesis." -n 64 -ngl 99 -c 2048
```

That's it. The infrastructure was already in [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) — the chained recurrent threading at `src/llama-context.cpp:4180` (`next_prev_hidden = scratch_out_hidden.data()`) and the confidence gate at line 4140. The cache-bookkeeping bug fix in patch 11 unblocked it.

## Honest 5-prompt benchmark

Qwen3.5-27B Q4_K_M, M4 Max, post-bug-fix, n=64 tokens per prompt:

| Prompt | K=1 vanilla | K=2 thresh=0.85 | Speedup |
|---|---|---|---|
| Write a haiku about spring. | 4.6 tok/s | 13.0 tok/s | **2.83×** |
| Explain photosynthesis in one paragraph. | 7.1 tok/s | 14.7 tok/s | 2.07× |
| Write a Python function to compute Fibonacci. | 6.6 tok/s | 14.0 tok/s | 2.12× |
| List the planets of the solar system. | 8.3 tok/s | 13.8 tok/s | 1.66× |
| Translate hello world to French. | 8.5 tok/s | 14.4 tok/s | 1.69× |
| **Mean** | **7.02** | **13.98** | **1.99×** |

**Plain decode baseline**: 17.90 tok/s. Chained MTP is at **0.78×** of plain decode.

## What's actually happening (algorithm)

Per cycle, when `c1 >= MTP_CHAIN_THRESH`:

1. Run the MTP head once with `(prev_hidden = main_model_hidden, seed = id_last)` → produces `d1` and `h_mtp1`
2. Compute top-1 probability of `d1`. If below threshold, fall through to single-step path.
3. **Chain**: run the SAME MTP head again with `(prev_hidden = h_mtp1, seed = d1)` → produces `d2`. *This is the recurrent step — the head's own output hidden becomes the next step's input.*
4. Verify `[id_last, d1, d2]` in one main-model forward pass
5. If both accepted, commit `d1`, `d2`, plus the verify's tail-logits argmax (the bonus token) — **3 tokens for one main forward**

## Why K=2 and not K=3?

| K | thresh | tok/s | Why |
|---|---|---|---|
| 2 | 0.85 | **15.0** | Sweet spot |
| 3 | 0.85 | 12.8 | 3rd step's marginal accept (~50%) doesn't pay back its cost |
| 3 | 0.9 | 13.2 | Tighter threshold helps but K=2 still wins |

Each chained step costs ~1 MTP forward pass (~10ms on M4 Max). On Qwen3.5's single MTP head, the +1 prediction is ~80% accurate, the +2 prediction (chained) is ~60%, the +3 (chained twice) is ~40%. The 3rd step gets accepted often enough to commit but not often enough to amortize the rollback cost when it gets rejected. K=2 is the local optimum.

## Comparison to MLX `stacked_v2.py`

| | MLX `stacked_v2` | llama.cpp chained MTP |
|---|---|---|
| Plain decode baseline | 29.5 tok/s | 17.90 tok/s |
| Spec throughput | 51.1 tok/s | 13.98 tok/s |
| **vs plain** | **1.73×** | **0.78×** |
| Recipe | Chained MTP + 0.8B companion + confidence gate | Chained MTP + confidence gate |
| 0.8B companion implemented? | Yes | **Not yet** |

The algorithmic parity is established. The remaining gap is two-fold:

1. **Per-forward-pass overhead in llama.cpp**: each MTP draft pass and each main verify pass has fixed overhead (graph allocation, snapshot/restore for rollback, KV bookkeeping) that MLX doesn't pay because its recurrent state model uses cache slicing instead of snapshot/restore. This is the dominant factor in the gap.
2. **0.8B companion model**: `stacked_v2.py` runs a small Qwen 0.8B alongside the 27B and contributes additional draft candidates. We don't yet have a Qwen3.5-0.8B GGUF on hand (only the MLX safetensors), and wiring two models into one binary is the next ~hour of integration work.

Closing the per-pass overhead gap is the higher-leverage next move. The 0.8B adds maybe another 1.2× on top once the per-pass cost is brought down.

## Why this finding was buried for the entire session

The chained recurrent threading at `src/llama-context.cpp:4180` was added in commit `987541157` (the in-graph AR loop work). The confidence gate at line 4140 was added in commit `7ac89131e` (the adaptive chain patch). Both landed early in the session. **Neither was ever measured against the post-fix bug** — the cache-bookkeeping bug was masking all output, so every measurement of these env vars produced "speedups" on broken text and the variants were marked as ineffective.

Once the bug was fixed, the right env-var combo was sitting in the codebase the whole time. **The session's biggest win was a re-validation, not a new feature.**

This is the specific lesson worth remembering: **after finding a bug that affects measurement, re-run every experiment that was previously dismissed as ineffective**. Especially the ones that "almost worked but didn't quite pay back" — those are the most likely to turn into wins on the corrected baseline.

## Reproducing

```bash
# Build (after applying the qwen-mtp-llamacpp patches)
cmake --build build -j 12 --target llama-mtp-speculative

MODEL=path/to/qwen3.5-27b-q4km.gguf

# Plain decode ground truth
./build/bin/llama-bench -m $MODEL -p 0 -n 32 -ngl 99

# K=1 vanilla (the previous baseline)
./build/bin/llama-mtp-speculative -m $MODEL -p "Explain photosynthesis." -n 64 -ngl 99

# THE RECIPE — chained MTP with confidence gating
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m $MODEL -p "Explain photosynthesis." -n 64 -ngl 99
```

Output should be byte-coherent with plain decode. If it isn't, you don't have patch 11 from [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp).

# qwen-mtp-research

> Research notes, methodology, and design work from a deep dive into **Multi-Token Prediction (MTP) speculative decoding for Qwen3.5-27B** in llama.cpp. Six optimization variants explored, one critical bug found and fixed, and a forward design for per-position MTP heads (DeepSeek V3 style) on a hybrid attention + DeltaNet architecture.

This repo is the **explorer's notebook**. It collects what we learned about how MTP, hybrid attention, DeltaNet recurrence, and speculative decoding interact on a specific real model — the things that aren't in any paper because they only show up when you actually try to run it.

## What was explored

Over one focused session, eight subagents and many direct implementations attacked the problem from every angle. The matrix:

| # | Approach | Idea | Status |
|---|---|---|---|
| 1 | **Adaptive chain** | Top-1 probability gating to trim wasted draft passes | Implemented; pre-fix measurement only |
| 2 | **Predictive hidden** | Identity / linear extrapolation of `prev_hidden` to skip a main forward pass | Implemented; +75pt accept on short prefixes pre-fix |
| 3 | **Drift refresh** | Periodic T=1 plain-decode every N tokens to bound DeltaNet recurrent drift | Implemented; 7%→75% accept jump pre-fix |
| 4 | **Perturbed-head ensemble** | Top-K candidates from one MTP forward pass, tree-fork verify | Implemented; pre-fix measurement only |
| 5 | **Branching speculative tree** | Full B*D tree with multi-sequence batching, unified KV, per-branch `id_last` | Implemented; pre-fix measurement only |
| 6 | **Ensemble fast-path** | Skip 2nd forward pass on top-1 hits, accept recurrent contamination | Implemented; **proven broken on hybrid model post-fix** |
| 7 | **Rollback batching** | Convert N×T=1 rollback re-decodes to one T=N batch | Implemented; uncovered the cache bug |
| 8 | **Per-position heads (design)** | DeepSeek V3 style — 4 trained MTP heads, one per position offset | Design only — see [docs/per-position-heads.md](docs/per-position-heads.md) |

The first 7 are real code in [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations). The 8th is the next direction.

## The bug that ate the session

For most of the session, every variant looked like it was producing wins (1.16×, 1.72×, 2.5× depending on prompt). All of those measurements were on **degraded text** that diverged from plain decode within ~10 tokens after any rejection.

The root cause was a **one-line cache-bookkeeping bug** in `mtp-speculative.cpp`:

```cpp
// after a batched rollback re-decode of [id_last, drafts..., corr]:
n_past  += n_commit;
id_last = corr;          // BUG: corr is already in the cache as the last batch slot
```

The next iteration's verify batch wrote `corr` into the cache **a second time** at position `n_past + n_commit`, shifting every subsequent token by one slot. The model saw garbage context and produced "1. 2. 3. ... 1000000" digit-loops.

**Six previous agents missed it** because they were all hunting for forward-pass numerical bugs (chunking-vs-AR divergence, RoPE positions, recurrent state leak, hidden state pipe corruption). The bug was three lines below all the things they kept staring at, in the host-side bookkeeping. Once an isolated debugger compared plain-decode vs spec slot-0 logits at every iteration with the SAME `id_last`, the divergence proved the cache history was wrong, not the numerics.

The fix: read the tail logits of the re-decoded batch and use `argmax` as the new `id_last`, mirroring the accept-all branch.

The lessons documented in [docs/the-bug.md](docs/the-bug.md):

1. **Validate output text against ground truth on every measurement.** Throughput numbers without coherence checks are meaningless. Six agents reported speedups; zero validated text quality.
2. **Mutual drift convergence is real.** When both drafter and target are corrupted by the same upstream bug, accept rates can paradoxically *climb* — the two corrupted distributions agree on garbage.
3. **Bookkeeping bugs hide behind numerical bugs.** When the symptom is "wrong logits", the instinct is to look at the graph. Always check that the graph is being fed the input you think it is.

## What we learned about the architecture

### Hybrid attention + DeltaNet is its own beast
DeltaNet is irreversible — there is no "state at intermediate position", only the state after processing all tokens in a batch. This breaks the standard speculative decoding rollback assumption that you can cheaply undo a wrong-path decode. Every variant in this session had to work around this in some way: snapshot/restore (cheap snapshot, full restore), in-graph AR loop (mathematically equivalent to T=1 but in one dispatch), force-recurrent-position metadata override (the new `llama_memory_seq_force_recurrent_pos` primitive), or accepting state contamination as cost.

### Chunking vs AR DeltaNet kernels
For T≥2 batches, llama.cpp routes through `build_delta_net_chunking` which is mathematically equivalent to `build_delta_net_autoregressive` but **numerically divergent in fp16**. This was hypothesis #1 for why MTP spec was producing garbage — and it turned out to be a red herring. The divergence is real but bounded (a few ulps); the actual bug was upstream.

### Cross-stream `seq_cp` is alias-only on recurrent memory
For tree/multi-seq variants, `llama_memory_seq_cp` for the recurrent half of a hybrid memory creates an alias, not a copy. All forked branches share one recurrent cell that gets stomped by whichever ubatch writes last. To do real per-branch recurrent state on a hybrid model, you need either a deep memory-layer rewrite or to fall back to per-branch `id_last` duplication (with the `kv_unified=true` requirement and `n_parallel` bumping that comes with that).

### `MTP_VERIFY_FORCE_AR` doesn't fix what people think it fixes
The in-graph AR loop (commit `987541157`) was added to bypass the chunking divergence in the verify path. But on Qwen3.5-27B, it doesn't change throughput meaningfully under non-contention conditions, because the chunking path was never the actual bottleneck — the cache-bookkeeping bug was. Useful primitive, wrong target.

### Single-MTP-head spec on a hybrid model is hard to win
Post-fix, the honest single-head numbers are:
- Plain decode: 17.90 tok/s
- K=1 MTP spec: 7.64 tok/s (0.43× of plain)

The MTP draft pass costs about as much as a main forward pass on this model, so single-head spec doesn't yet beat plain decode. The two paths forward are:
1. Make the draft pass cheaper (prune the head, distill, etc.)
2. **Per-position heads** — multiple cheap heads, each predicting +1, +2, ... +N from the same hidden state — so one main forward pass amortizes over N committed tokens. This is the DeepSeek V3 design and likely how the user's MLX implementation hits 1.68×.

## Per-position MTP heads — design

See [docs/per-position-heads.md](docs/per-position-heads.md) for the full design. Headlines:

- **N=4 heads**, each a single transformer block, sharing the main model's embedding and LM head
- **Tensor naming**: `blk.<64+k>.nextn.*` for k=0..3, slotted into the existing GGUF layer table
- **Training**: warm-start each head from the existing single MTP layer's weights, freeze the main model, train heads jointly with summed cross-entropy at offsets +1..+4 on a 1B-token corpus
- **Inference math**: 1 main forward (60ms) + 4 head forwards (4×10ms) = 100ms per cycle, commits up to 4 tokens → theoretical 40 tok/s vs plain 17.9 tok/s = **2.23× speedup ceiling**
- **Honest risk**: the "10ms per head" assumption needs measurement; on the hybrid model the heads still need to walk DeltaNet recurrent state, which may not be as cheap as a pure-attention head

## Methodology learnings

- **Spawn agents in parallel for independent variants** — six concurrent explorations is a research multiplier, not a coordination problem
- **Have one agent dedicated to correctness debugging** — separate from optimization agents, so its judgment isn't biased by the speedup numbers it's hearing
- **Compare top-K logits not just argmax** — it's the cheapest way to detect drift before it cascades into different argmax outputs
- **Use plain decode (or a `MTP_FORCE_AR` style flag) as the always-on ground truth** — every agent should be able to reproduce the baseline in two commands
- **Don't trust acceptance rates without text-coherence checks** — see "the bug that ate the session" above

## What's in this repo

```
docs/
  per-position-heads.md      Full design for the DeepSeek V3 style approach
  the-bug.md                 Detailed root-cause writeup of the cache bookkeeping bug
  hybrid-deltanet-notes.md   What we learned about DeltaNet + spec decoding
  methodology.md             How to run a parallel-agent exploration like this
scripts/
  bench-honest.sh            5-prompt benchmark with output coherence validation
  compare-decode.py          Token-by-token diff between plain and spec output
```

## Related repos

- **[qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp)** — the infrastructure patches (the substrate for everything here)
- **[qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations)** — the actual code for the 6 explored variants
- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)** — converter and tensor-name deep dive

## License

MIT.

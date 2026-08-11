<div align="center">

```
  ___  __  __ _____ ____  
 / _ \|  \/  |_   _|  _ \ 
| | | | |\/| | | | | |_) |
| |_| | |  | | | | |  __/ 
 \__\_\_|  |_| |_| |_|    
  R E S E A R C H
```

**Research notes, methodology, and design work for Multi-Token Prediction speculative decoding.**

*Exploring optimization variants, bug fixes, and per-position MTP heads for Qwen3.5-27B in llama.cpp*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [✨ What was explored](#-what-was-explored)
- [🐛 The Bug That Ate the Session](#-the-bug-that-ate-the-session)
- [🏗️ Architectural Discoveries](#️-architectural-discoveries)
- [⚡ The MLX Truth & Recipe](#-the-mlx-truth--recipe)
- [🔮 Per-Position Heads Design](#-per-position-heads-design)
- [📖 Methodology Learnings](#-methodology-learnings)
- [📁 Repository Structure](#-repository-structure)
- [🔗 Related Repositories](#-related-repositories)
- [📄 License](#-license)

---

## 🎯 Overview

This repository serves as the **explorer's notebook** for a deep dive into Multi-Token Prediction (MTP) speculative decoding for Qwen3.5-27B in llama.cpp. It covers six optimization variants, the discovery and fix of a critical bug, and a forward design for per-position MTP heads (DeepSeek V3 style) on a hybrid attention + DeltaNet architecture.

It collects the practical realities of interacting with MTP, hybrid attention, DeltaNet recurrence, and speculative decoding—insights rarely found in papers.

---

## ✨ What was explored

Over one focused session, eight subagents and many direct implementations attacked the problem from every angle:

| # | Approach | Idea | Status |
|---|---|---|---|
| 1 | **Adaptive chain** | Top-1 probability gating to trim wasted draft passes | Implemented; pre-fix measurement only |
| 2 | **Predictive hidden** | Identity / linear extrapolation of `prev_hidden` to skip a main forward pass | Implemented; +75pt accept on short prefixes pre-fix |
| 3 | **Drift refresh** | Periodic T=1 plain-decode every N tokens to bound DeltaNet recurrent drift | Implemented; 7%→75% accept jump pre-fix |
| 4 | **Perturbed-head ensemble** | Top-K candidates from one MTP forward pass, tree-fork verify | Implemented; pre-fix measurement only |
| 5 | **Branching speculative tree** | Full B*D tree with multi-sequence batching, unified KV, per-branch `id_last` | Implemented; pre-fix measurement only |
| 6 | **Ensemble fast-path** | Skip 2nd forward pass on top-1 hits, accept recurrent contamination | Implemented; **proven broken on hybrid model post-fix** |
| 7 | **Rollback batching** | Convert N×T=1 rollback re-decodes to one T=N batch | Implemented; uncovered the cache bug |
| 8 | **Per-position heads (design)** | DeepSeek V3 style — 4 trained MTP heads, one per position offset | Design only — see `docs/per-position-heads.md` |

> [!NOTE]
> The first 7 variants are real code located in the [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations) repo.

---

## 🐛 The Bug That Ate the Session

For most of the session, every variant appeared to produce massive speedups (1.16×, 1.72×, 2.5×). However, all these measurements were on **degraded text** that quickly diverged from plain decode.

> [!WARNING]
> The root cause was a one-line cache-bookkeeping bug in `mtp-speculative.cpp`:

```cpp
// after a batched rollback re-decode of [id_last, drafts..., corr]:
n_past  += n_commit;
id_last = corr;          // BUG: corr is already in the cache as the last batch slot
```

The next iteration's verify batch wrote `corr` into the cache a second time, shifting subsequent tokens and feeding garbage context to the model. 

### Key Lessons
1. **Validate output text against ground truth.** Throughput numbers without coherence checks are meaningless.
2. **Mutual drift convergence is real.** When drafter and target are corrupted by the same bug, accept rates can paradoxically *climb*.
3. **Bookkeeping bugs hide behind numerical bugs.** Always verify the inputs fed to the graph.

---

## 🏗️ Architectural Discoveries

### Hybrid attention + DeltaNet is its own beast
DeltaNet is irreversible. Every variant had to workaround this via snapshot/restore, in-graph AR loops, or force-recurrent-position metadata overrides.

### Chunking vs AR DeltaNet kernels
Chunking is numerically divergent in fp16, but this divergence is bounded and wasn't the root cause of the MTP spec outputting garbage.

### Cross-stream `seq_cp` is alias-only on recurrent memory
`llama_memory_seq_cp` creates an alias, not a copy, for hybrid memory recurrent cells.

### Single-MTP-head spec on a hybrid model is hard to win
Post-fix single-head numbers:
- Plain decode: 17.90 tok/s
- K=1 MTP spec: 7.64 tok/s (0.43× of plain)

---

## ⚡ The MLX Truth & Recipe

The MLX implementation hitting **1.68×** over baseline does **NOT** use per-position trained heads. The checkpoint contains one MTP block. 

The strategy (`stacked_v2.py`):
1. Chained recurrent application of the single MTP head
2. Small (~0.8B) companion draft model
3. Confidence gating
4. Zero training cost

**Status in llama.cpp port**: Delivers 1.99× over K=1 vanilla with these environment variables:

```bash
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m qwen3.5-27b-q4km.gguf \
    -p "Explain photosynthesis." -n 64 -ngl 99
```

---

## 🔮 Per-Position Heads Design

If the chained approach doesn't scale, a DeepSeek V3 style design is the alternative.

> [!IMPORTANT]
> **Phase 0 instrumentation**: If `head_fwd ≈ main_fwd`, per-position heads CANNOT win regardless of accept rate. The fixed overhead per draft pass is the dominant cost. Phase 0 is a kill-or-proceed gate.

Highlights if Phase 0 passes:
- **N=4 heads** sharing main embedding and LM head
- **Training**: 1B-token corpus, freezing main model
- **Inference**: ~40 tok/s theoretical vs plain 17.9 tok/s (2.23× speedup ceiling)

---

## 📖 Methodology Learnings

- Spawn agents in parallel for independent variants
- Dedicate one agent to correctness debugging
- Compare top-K logits, not just argmax
- Use plain decode as the always-on ground truth
- Don't trust acceptance rates without text-coherence checks

---

## 📁 Repository Structure

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

---

## 🔗 Related Repositories

- **[qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp)** — infrastructure patches
- **[qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations)** — explored variants
- **[qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors)** — converter and tensor-name deep dive

---

## 📄 License

MIT.

#!/usr/bin/env python3
"""
build_training_data.py — Per-position MTP head training data extractor.

DESIGN ONLY. Do not run without explicit user approval — requires ~100 GPU-hours
on an A100/H100 to process 1B tokens.

Produces: parquet shards containing (h_t, target_{t+1..t+N}) pairs for every
position t in the corpus, where h_t is the final-layer hidden state of the
frozen main model at position t.

Usage (planned):
    python build_training_data.py \\
        --hf-model ~/models/qwen35-27b-hf/ \\
        --corpus fineweb-edu-10B \\
        --num-tokens 1000000000 \\
        --num-heads 4 \\
        --output ~/data/qwen35-mtp-train/ \\
        --batch-size 8 \\
        --seq-len 2048

Output format (per shard, parquet):
    - hidden_state : float16 [n_embd=5120]            (frozen main model h_t)
    - targets      : int32   [num_heads=4]            (token_{t+1}..token_{t+N})
    - prev_tokens  : int32   [num_heads=4]            (token_t, token_{t+1}, ..., token_{t+N-1})
                                                      — head_k trains on embed(prev_tokens[k])
    - seq_id       : int64                            (for debugging)
    - pos          : int32                            (position within sequence)

Storage math at 1B tokens:
    row size ≈ 5120*2 + 4*4 + 4*4 + 8 + 4 = ~10.3 KB
    1e9 * 10.3 KB = ~10.3 TB uncompressed
    With zstd parquet column compression (hidden states dominate, fp16 entropy ~6 bits):
    realistic: ~4 TB on disk

    ALTERNATIVE: store only token ids (4 + 4*4 + 4*4 = 36 B/row → 36 GB total)
    and RECOMPUTE hidden states on-the-fly during training.
    This is the recommended path: hidden states at scale are too big.

Recommended corpus:
    - FineWeb-Edu (HuggingFaceFW/fineweb-edu) 10B sample — high-quality filtered web.
    - Mix 80% FineWeb-Edu + 10% code (StarCoder subset) + 10% math (OpenWebMath).
    - Tokenize with Qwen3.5 tokenizer, pack into 2048-length sequences.

Implementation sketch (not executed):
"""
import argparse
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True)
    p.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--num-tokens", type=int, default=1_000_000_000)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--shard-rows", type=int, default=100_000)
    p.add_argument("--store-hidden", action="store_true",
                   help="Store hidden states (10x disk). Default: tokens only.")
    return p.parse_args()

def main():
    args = parse_args()
    # 1. Load frozen main model (bf16 on GPU).
    #    from transformers import AutoModelForCausalLM, AutoTokenizer
    #    tok   = AutoTokenizer.from_pretrained(args.hf_model)
    #    model = AutoModelForCausalLM.from_pretrained(args.hf_model,
    #                                                 torch_dtype=torch.bfloat16,
    #                                                 device_map="cuda")
    #    model.eval()
    #
    # 2. Stream corpus, tokenize, pack into seq_len windows.
    #    ds = load_dataset(args.corpus, split="train", streaming=True)
    #    buffer = []
    #    for row in ds:
    #        buffer.extend(tok(row["text"]).input_ids)
    #        while len(buffer) >= args.seq_len:
    #            yield buffer[:args.seq_len]; buffer = buffer[args.seq_len:]
    #
    # 3. For each batch of seq_len sequences:
    #    with torch.no_grad():
    #        out = model(input_ids, output_hidden_states=True)
    #        h   = out.hidden_states[-1]   # [B, T, n_embd] — pre-lm_head, post-final-norm
    #        # NOTE: We want the tap point BEFORE final norm to match the
    #        # existing MTP head which does its own normalization. Use
    #        # output_hidden_states=True and take the last BLOCK output, not
    #        # post-norm. See transformers Qwen3ForCausalLM internals.
    #
    # 4. For each position t in [0, T-N):
    #       row = {
    #         'hidden_state': h[:, t, :].cpu().numpy().astype(np.float16),   # optional
    #         'prev_tokens' : input_ids[:, t:t+N].cpu().numpy(),
    #         'targets'     : input_ids[:, t+1:t+N+1].cpu().numpy(),
    #       }
    #       writer.write_row(row)
    #
    # 5. Flush shards every --shard-rows rows.
    print(f"[DESIGN-ONLY] Would process {args.num_tokens:,} tokens "
          f"from {args.corpus} using {args.hf_model}")
    print(f"Estimated GPU-hours on H100: {args.num_tokens / 1e9 * 80:.0f}h")
    print(f"Estimated disk (tokens only): {args.num_tokens * 36 / 1e9:.1f} GB")
    print(f"Estimated disk (with hidden): {args.num_tokens * 10300 / 1e9:.0f} GB")
    print("NOT RUNNING. Exit.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

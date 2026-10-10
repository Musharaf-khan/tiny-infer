"""Phase 4: what paging buys. Same GPU, same kernels, same fixed KV pool; only the allocation policy differs.

    paged       blocks allocated as tokens arrive (waste < 1 block per sequence)
    contiguous  ceil(max_model_len / block_size) blocks reserved per request at admission

All requests are submitted at once with max_num_seqs very high, so the KV pool is the only limit on
how many run concurrently. Reports peak concurrent sequences, mean KV waste (allocated-but-unused
slots / allocated slots), preemptions and throughput. Also prints the closed-form numbers.

    python bench/kv_capacity.py --kv-cache-gb 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make_workload  # noqa: E402
from tiny_infer.config import EngineConfig  # noqa: E402
from tiny_infer.engine import LLMEngine  # noqa: E402
from tiny_infer.model import LlamaModel  # noqa: E402
from tiny_infer.sequence import SamplingParams  # noqa: E402


def run(model, tok, mode, args, workload):
    cfg = EngineConfig(model=args.model, kv_mode=mode, kv_cache_gb=args.kv_cache_gb, max_num_seqs=1024,
                       device=args.device, dtype=args.dtype,
                       max_model_len=args.max_model_len, block_size=args.block_size,
                       use_custom_kernels=args.custom_kernels)
    eng = LLMEngine(cfg, model=model, tokenizer=tok)
    for p, m in workload:
        eng.add_request(p, SamplingParams(max_tokens=m, ignore_eos=True))
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    sync()
    dt = time.perf_counter() - t0
    st = eng.stats()
    out_tokens = sum(m for _, m in workload)
    r = {"mode": mode, "num_blocks": eng.bm.num_blocks, "peak_concurrent_seqs": st["peak_running"],
         "mean_kv_waste": st["mean_kv_waste"], "preemptions": st["preemptions"], "seconds": dt,
         "output_tok_s": out_tokens / dt, "steps": st["steps"]}
    del eng
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--kv-cache-gb", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--input-len", type=int, nargs=2, default=[128, 512])
    p.add_argument("--output-len", type=int, nargs=2, default=[64, 256])
    p.add_argument("--custom-kernels", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default="results/kv_capacity.json")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = torch.float16 if a.dtype == "float16" else torch.float32
    model = LlamaModel.from_pretrained(a.model, a.device, dt, use_custom_kernels=a.custom_kernels)
    wl = make_workload(a.num_prompts, tuple(a.input_len), tuple(a.output_len), vocab=model.cfg.vocab_size)

    per_tok = model.cfg.kv_bytes_per_token(dt)
    pool_tokens = int(a.kv_cache_gb * 2**30 // (per_tok * a.block_size)) * a.block_size
    mean_len = sum(len(p) + m for p, m in wl) / len(wl)
    theory = {
        "kv_bytes_per_token": per_tok, "pool_tokens": pool_tokens,
        "contiguous_max_seqs": pool_tokens // a.max_model_len,
        "paged_max_seqs_at_mean_final_len": int(pool_tokens // (mean_len + a.block_size / 2)),
        "mean_final_len": mean_len,
        "contiguous_waste_at_mean_len": 1 - mean_len / a.max_model_len,
    }
    print("theory:", json.dumps(theory, indent=1))
    results = [run(model, tok, m, a, wl) for m in ("contiguous", "paged")]
    for r in results:
        print(r)
    c, pg = results
    summary = {"theory": theory, "runs": results,
               "concurrency_gain": pg["peak_concurrent_seqs"] / max(c["peak_concurrent_seqs"], 1),
               "throughput_gain": pg["output_tok_s"] / c["output_tok_s"]}
    print(f"\npaged vs contiguous: {summary['concurrency_gain']:.1f}x peak concurrency, "
          f"{summary['throughput_gain']:.2f}x throughput, KV waste {c['mean_kv_waste']:.1%} -> {pg['mean_kv_waste']:.1%}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()

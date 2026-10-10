"""Phase 1 baseline: HuggingFace generate(), in process, no server in the way.

  1. concurrency 1: TTFT, TPOT, decode tok/s, peak VRAM (the "before" every number is compared to)
  2. static batching: B padded prompts in one generate() call. Throughput rises with B, but every
     request waits for the longest prompt to prefill and the longest output to finish.

    python bench/hf_baseline.py --batch-sizes 1 4 16 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import make_workload  # noqa: E402


class TimingStreamer:
    def __init__(self):
        self.times, self.first = [], True

    def put(self, value):
        if self.first:            # first put() is the prompt
            self.first = False
            return
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())

    def end(self):
        pass


def load(model_name, dtype):
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tiny_infer.hf_utils import load_causal_lm
    return AutoTokenizer.from_pretrained(model_name), load_causal_lm(model_name, dtype, "cuda")


@torch.inference_mode()
def single_stream(model, workload):
    rows = []
    for prompt, max_new in workload:
        st = TimingStreamer()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=max_new, min_new_tokens=max_new,
                       do_sample=False, streamer=st, pad_token_id=0)
        ttft = st.times[0] - t0
        tpot = (st.times[-1] - st.times[0]) / (len(st.times) - 1)
        rows.append({"prompt_len": len(prompt), "out": len(st.times), "ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3})
    return rows


@torch.inference_mode()
def static_batch(model, workload, B):
    """Left-padded batch of B prompts, one generate() call; output len = max of the batch (static batching)."""
    batch = workload[:B]
    L = max(len(p) for p, _ in batch)
    ids = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.long)
    for i, (p, _) in enumerate(batch):
        ids[i, L - len(p):] = torch.tensor(p)
        mask[i, L - len(p):] = 1
    max_new = max(m for _, m in batch)
    useful = sum(m for _, m in batch)               # tokens the users actually asked for
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=max_new, min_new_tokens=max_new,
                   do_sample=False, pad_token_id=0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"batch": B, "seconds": dt, "useful_output_tok_s": useful / dt, "generated_tok_s": B * max_new / dt,
            "wasted_fraction": 1 - useful / (B * max_new), "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--input-len", type=int, nargs=2, default=[128, 512])
    p.add_argument("--output-len", type=int, nargs=2, default=[64, 256])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--out", default="results/hf_baseline.json")
    a = p.parse_args()

    tok, model = load(a.model, torch.float16)
    weights_gib = torch.cuda.memory_allocated() / 2**30
    wl = make_workload(max(a.num_prompts, max(a.batch_sizes)), tuple(a.input_len), tuple(a.output_len),
                       vocab=model.config.vocab_size)
    single_stream(model, wl[:1])                      # warm-up
    torch.cuda.reset_peak_memory_stats()
    rows = single_stream(model, wl[:a.num_prompts])
    res = {
        "model": a.model, "gpu": torch.cuda.get_device_name(), "weights_gib": weights_gib,
        "c1_ttft_ms_p50": statistics.median(r["ttft_ms"] for r in rows),
        "c1_tpot_ms_p50": statistics.median(r["tpot_ms"] for r in rows),
        "c1_decode_tok_s": 1000 / statistics.median(r["tpot_ms"] for r in rows),
        "c1_peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "single_stream": rows, "static_batching": [],
    }
    print(f"HF generate, concurrency 1: TTFT p50 {res['c1_ttft_ms_p50']:.1f} ms | TPOT p50 {res['c1_tpot_ms_p50']:.2f} ms"
          f" | {res['c1_decode_tok_s']:.1f} tok/s | peak VRAM {res['c1_peak_vram_gib']:.2f} GiB")
    for B in a.batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        try:
            r = static_batch(model, wl, B)
        except torch.cuda.OutOfMemoryError:
            r = {"batch": B, "error": "OOM"}
            torch.cuda.empty_cache()
        res["static_batching"].append(r)
        print("static batch", r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()

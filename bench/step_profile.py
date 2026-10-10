"""Where does a decode step's time go? Evidence for the vLLM gap analysis.

Runs the engine at a fixed decode batch, profiles a window of pure-decode steps with torch.profiler,
and splits the step into: GEMMs, attention, norms, other GPU kernels, and time the GPU sits idle
waiting on the CPU (Python scheduling, input building, kernel-launch overhead). Also compares the step
to the roofline floor: weights + KV bytes / 320 GB/s.

    python bench/step_profile.py --batch 32 --custom-kernels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_infer.config import EngineConfig  # noqa: E402
from tiny_infer.engine import LLMEngine  # noqa: E402
from tiny_infer.model import LlamaModel  # noqa: E402
from tiny_infer.sequence import SamplingParams  # noqa: E402

T4_GBS = 320e9


def bucket(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "gemv", "cutlass", "sm75_xmma", "turing", "volta_h", "ampere_h", "cublas")):
        return "gemm"
    if any(k in n for k in ("attention", "fmha", "flash", "sdpa", "softmax", "paged_decode", "combine_partitions")):
        return "attention"
    if any(k in n for k in ("rms_norm", "norm")):
        return "norm"
    if any(k in n for k in ("index", "gather", "scatter", "copy", "cat", "repeat")):
        return "copy/gather"
    return "elementwise/other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--custom-kernels", action="store_true")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = LlamaModel.from_pretrained(a.model, "cuda", torch.float16)
    eng = LLMEngine(EngineConfig(model=a.model, kv_cache_gb=4.0, max_num_seqs=a.batch,
                                 use_custom_kernels=a.custom_kernels), model=model, tokenizer=tok)
    g = torch.Generator().manual_seed(0)
    for _ in range(a.batch):
        eng.add_request(torch.randint(3, model.cfg.vocab_size, (a.prompt_len,), generator=g).tolist(),
                        SamplingParams(max_tokens=a.steps + 64, ignore_eos=True))
    while any(s.num_uncomputed > 1 for s in eng.scheduler.running) or eng.scheduler.waiting:
        eng.step()                                   # finish all prefills
    for _ in range(5):
        eng.step()                                   # warm decode
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(a.steps):
            eng.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.steps
    buckets: dict[str, float] = {}
    for ev in prof.key_averages():
        dt = getattr(ev, "device_time_total", None)
        if dt is None:
            dt = getattr(ev, "cuda_time_total", 0)
        if dt and ev.key and not ev.key.startswith(("cuda", "aten::", "Memcpy", "Memset")) and ev.device_type.name == "CUDA":
            buckets[bucket(ev.key)] = buckets.get(bucket(ev.key), 0.0) + dt / 1e3 / a.steps
    gpu = sum(buckets.values())
    ctx = a.prompt_len + 5 + a.steps // 2
    weight_bytes = sum(t.numel() * t.element_size() for L in model.layers for k, t in L.items()) \
        + model.lm_head.numel() * 2
    kv_bytes = a.batch * ctx * model.cfg.kv_bytes_per_token(torch.float16)
    floor_ms = (weight_bytes + kv_bytes) / T4_GBS * 1e3
    res = {"batch": a.batch, "custom_kernels": a.custom_kernels, "wall_step_ms": wall * 1e3,
           "gpu_busy_ms": gpu, "gpu_idle_ms (CPU/launch overhead)": wall * 1e3 - gpu,
           "roofline_floor_ms": floor_ms, "pct_of_roofline": floor_ms / (wall * 1e3) * 100,
           "breakdown_ms": dict(sorted(buckets.items(), key=lambda kv: -kv[1]))}
    print(json.dumps(res, indent=2))
    print("\ntop kernels:")
    try:
        print(prof.key_averages().table(sort_by="device_time_total", row_limit=15))
    except Exception:  # noqa: BLE001  (older torch)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    out = a.out or f"results/step_profile_b{a.batch}{'_kernels' if a.custom_kernels else ''}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()

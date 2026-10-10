"""Phase 5: what the Project 1 kernels buy, measured twice.

  A. op level: each kernel vs the PyTorch path it replaces, at TinyLlama's decode shapes, with the
     effective bandwidth against the T4's 320 GB/s (decode ops are memory-bound, so GB/s is the score);
  B. end to end: one engine, fixed decode batch, same requests, kernels off vs on; reports the
     decode-step time (= TPOT under load) and output tok/s.

    python bench/kernel_bench.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_infer import ops  # noqa: E402

T4_GBS = 320.0


def cuda_time(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(5):
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / iters)
    return statistics.median(times)          # ms


def op_bench():
    rows = []
    r = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float16)
    H = 2048
    for T in (1, 8, 32, 128):
        x, w, res = r(T, H), r(H), r(T, H)
        ref = cuda_time(lambda: ops.fused_add_rms_norm_ref(x, res, w, 1e-5))
        ker = cuda_time(lambda: ops.fused_add_rms_norm(x, res, w, 1e-5, True))
        nbytes = 4 * T * H * 2 + H * 2                       # read x, res, w; write h, y
        rows.append({"op": "fused_add_rms_norm", "shape": f"T={T},H={H}", "torch_us": ref * 1e3,
                     "kernel_us": ker * 1e3, "speedup": ref / ker, "kernel_GBs": nbytes / ker / 1e6})
    hq, hkv, hd, bs = 32, 4, 64, 16
    for B, ctx in ((1, 512), (1, 2048), (8, 1024), (32, 512), (32, 1024), (64, 1024)):
        nb = B * ctx // bs + 8
        kc, vc = r(nb, bs, hkv, hd), r(nb, bs, hkv, hd)
        bt = torch.randperm(nb, device="cuda")[: B * (ctx // bs)].view(B, -1).to(torch.int32)
        lens = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
        q = r(B, hq, hd)
        ref = cuda_time(lambda: ops.paged_decode_attention_ref(q, kc, vc, bt, lens, hd ** -0.5, ctx), iters=50)
        kv_bytes = 2 * B * ctx * hkv * hd * 2                # K and V, each read once
        row = {"op": "paged_decode_attention", "shape": f"B={B},ctx={ctx}", "torch_us": ref * 1e3}
        for ps in (128, 512):
            k = cuda_time(lambda: ops.paged_decode_attention(q, kc, vc, bt, lens, hd ** -0.5, ctx, True, ps), iters=50)
            row[f"kernel_p{ps}_us"] = k * 1e3
            row[f"kernel_p{ps}_GBs"] = kv_bytes / k / 1e6
        best = min(row["kernel_p128_us"], row["kernel_p512_us"])          # microseconds
        gbs = kv_bytes / (best * 1e-6) / 1e9
        row.update(kernel_us=best, speedup=row["torch_us"] / best, kernel_GBs=gbs, pct_of_T4_peak=gbs / T4_GBS * 100)
        rows.append(row)
    return rows


def e2e_bench(model_name, batch, prompt_len, out_len):
    from transformers import AutoTokenizer
    from tiny_infer.config import EngineConfig
    from tiny_infer.engine import LLMEngine
    from tiny_infer.model import LlamaModel
    from tiny_infer.sequence import SamplingParams
    tok = AutoTokenizer.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(model_name, "cuda", torch.float16)
    res = {}
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(3, model.cfg.vocab_size, (prompt_len,), generator=g).tolist() for _ in range(batch)]
    outs = {}
    for kern in (False, True, False, True):          # interleaved to cancel clock/thermal drift
        cfg = EngineConfig(model=model_name, use_custom_kernels=kern, kv_cache_gb=4.0, max_num_seqs=batch)
        eng = LLMEngine(cfg, model=model, tokenizer=tok)
        seqs = eng.generate(prompts, SamplingParams(max_tokens=out_len, ignore_eos=True))
        decode_steps = [t for d, p, t in eng.step_times if p == 0 and d == batch]
        key = "kernels" if kern else "torch"
        res.setdefault(key, []).append(statistics.median(decode_steps) * 1e3)
        outs[key] = [s.output_ids for s in seqs]
        del eng
        torch.cuda.empty_cache()
    t, k = statistics.median(res["torch"]), statistics.median(res["kernels"])
    agree = sum(a == b for a, b in zip(outs["torch"], outs["kernels"])) / batch
    return {"batch": batch, "prompt_len": prompt_len, "out_len": out_len,
            "decode_step_ms_torch": t, "decode_step_ms_kernels": k, "tpot_speedup": t / k,
            "output_tok_s_torch": batch / t * 1e3, "output_tok_s_kernels": batch / k * 1e3,
            "greedy_outputs_identical_fraction": agree}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--e2e-batches", type=int, nargs="+", default=[1, 16, 32])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=256)
    p.add_argument("--out", default="results/kernel_bench.json")
    a = p.parse_args()
    ops.load_kernels()
    from tiny_infer.kernels import self_test
    errs = self_test()
    print("self-test worst max|err| =", max(errs.values()))
    rows = op_bench()
    for r in rows:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    e2e = [e2e_bench(a.model, b, a.prompt_len, a.out_len) for b in a.e2e_batches]
    for r in e2e:
        print(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"self_test": errs, "ops": rows, "e2e": e2e, "gpu": torch.cuda.get_device_name()},
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()

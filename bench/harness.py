"""Serving benchmark client for any OpenAI-compatible /v1/completions endpoint (tiny-infer, naive HF, vLLM).

Methodology (mirrors `vllm bench serve`):
  * prompts are sent as TOKEN IDS with ignore_eos=true, so every system sees identical input lengths
    and produces exactly max_tokens outputs: no system "wins" by stopping early;
  * closed loop at fixed concurrency C (C workers, each sends its next request when the last finishes),
    or open loop at a Poisson request rate;
  * client-side timing from the SSE stream: TTFT = first token chunk, ITL = gaps between chunks,
    TPOT = (E2E - TTFT) / (output_tokens - 1);
  * percentiles, not means; prefill-side (input tok/s) and decode-side (output tok/s) throughput reported
    separately, never blended into one "tokens/sec".

    python bench/harness.py --url http://localhost:8000 --concurrency 1 4 16 32 --num-prompts 64 --label engine
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class RequestResult:
    prompt_len: int
    max_tokens: int
    ok: bool = False
    error: str = ""
    start: float = 0.0
    ttft: float = 0.0
    e2e: float = 0.0
    output_tokens: int = 0
    itls: list[float] = field(default_factory=list)


def make_workload(n: int, input_len: tuple[int, int], output_len: tuple[int, int], vocab: int = 32000,
                  seed: int = 0, shared_prefix: int = 0) -> list[tuple[list[int], int]]:
    """n (prompt_token_ids, max_tokens) pairs; lengths uniform in [lo, hi]. Token ids avoid specials (<3).
    shared_prefix > 0 gives every prompt the same first tokens (a 'system prompt' for prefix caching)."""
    rng = random.Random(seed)
    prefix = [rng.randrange(3, vocab) for _ in range(shared_prefix)]
    out = []
    for _ in range(n):
        L = rng.randint(*input_len)
        body = [rng.randrange(3, vocab) for _ in range(max(L - 1 - shared_prefix, 1))]
        out.append(([1] + prefix + body, rng.randint(*output_len)))      # 1 = BOS for Llama tokenizers
    return out


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: list[int], max_tokens: int
                      ) -> RequestResult:
    r = RequestResult(prompt_len=len(prompt), max_tokens=max_tokens)
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               "ignore_eos": True, "stream_options": {"include_usage": True}}
    r.start = time.perf_counter()
    last, chunks, usage_tokens = None, 0, None
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=payload) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if "error" in d:
                    r.error = str(d["error"])
                    return r
                now = time.perf_counter()
                if d.get("usage"):
                    usage_tokens = d["usage"].get("completion_tokens")
                if not d.get("choices"):
                    continue                     # usage-only chunk (vLLM)
                chunks += 1
                if last is None:
                    r.ttft = now - r.start
                else:
                    r.itls.append(now - last)
                last = now
        r.e2e = time.perf_counter() - r.start
        r.output_tokens = usage_tokens if usage_tokens is not None else chunks
        r.ok = r.output_tokens > 0
    except Exception as e:  # noqa: BLE001
        r.error = f"{type(e).__name__}: {e}"
    return r


class GpuSampler:
    """Polls nvidia-smi for SM utilisation and memory while a run is in flight."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.util, self.mem = interval, [], []
        self.enabled = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()

    def __enter__(self):
        if self.enabled:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits", "-i", "0"],
                                     capture_output=True, text=True, timeout=2).stdout.strip().split(",")
                self.util.append(float(out[0]))
                self.mem.append(float(out[1]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *a):
        self._stop.set()


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(results: list[RequestResult], duration: float, label: str, concurrency, gpu: GpuSampler | None) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok]
    tpot = [(r.e2e - r.ttft) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    itl = [x for r in ok for x in r.itls]
    e2e = [r.e2e for r in ok]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_len for r in ok)
    s = {
        "label": label, "concurrency": concurrency, "num_requests": len(results), "completed": len(ok),
        "error_rate": 1 - len(ok) / max(len(results), 1), "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_tok_s": out_tok / duration,                      # decode-side, system-wide
        "input_tok_s": in_tok / duration,                        # prefill-side, system-wide
        "per_user_decode_tok_s": statistics.mean(1 / t for t in tpot) if tpot else float("nan"),
        "total_output_tokens": out_tok, "total_input_tokens": in_tok,
    }
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2e", e2e)):
        s[f"{name}_ms_p50"] = pct(xs, 50) * 1e3
        s[f"{name}_ms_p99"] = pct(xs, 99) * 1e3
        s[f"{name}_ms_mean"] = (statistics.mean(xs) * 1e3) if xs else float("nan")
    if gpu and gpu.util:
        s["gpu_util_mean"] = statistics.mean(gpu.util)
        s["gpu_mem_peak_mib"] = max(gpu.mem)
    errs = [r.error for r in results if not r.ok and r.error]
    if errs:
        s["first_error"] = errs[0][:300]
    return s


async def run_closed_loop(url, model, workload, concurrency, timeout):
    queue = list(enumerate(workload))
    results: list[RequestResult | None] = [None] * len(workload)
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), limits=limits) as client:
        async def worker():
            while queue:
                i, (p, m) = queue.pop(0)
                results[i] = await one_request(client, url, model, p, m)
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        return results, time.perf_counter() - t0


async def run_open_loop(url, model, workload, rate, timeout, seed=0):
    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout),
                                 limits=httpx.Limits(max_connections=1000)) as client:
        tasks, t0 = [], time.perf_counter()
        for p, m in workload:
            tasks.append(asyncio.create_task(one_request(client, url, model, p, m)))
            await asyncio.sleep(rng.expovariate(rate))
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - t0


def print_table(rows: list[dict]):
    hdr = ("label", "conc", "ok", "out tok/s", "in tok/s", "req/s", "TTFT p50", "TTFT p99", "TPOT p50",
           "TPOT p99", "ITL p99", "E2E p50", "GPU%")
    print(" | ".join(f"{h:>10}" for h in hdr))
    for s in rows:
        vals = (s["label"][:10], s["concurrency"], f"{s['completed']}/{s['num_requests']}", f"{s['output_tok_s']:.1f}",
                f"{s['input_tok_s']:.0f}", f"{s['request_throughput']:.2f}", f"{s['ttft_ms_p50']:.0f}ms",
                f"{s['ttft_ms_p99']:.0f}ms", f"{s['tpot_ms_p50']:.1f}ms", f"{s['tpot_ms_p99']:.1f}ms",
                f"{s['itl_ms_p99']:.1f}ms", f"{s['e2e_ms_p50'] / 1e3:.2f}s",
                f"{s.get('gpu_util_mean', float('nan')):.0f}")
        print(" | ".join(f"{str(v):>10}" for v in vals))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="tiny-infer", help="model name sent in the request (vLLM needs the real one)")
    p.add_argument("--label", default="run")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 32])
    p.add_argument("--request-rate", type=float, default=None, help="open-loop Poisson rate (req/s) instead")
    p.add_argument("--num-prompts", type=int, default=None, help="per concurrency level (default 4*C, min 16)")
    p.add_argument("--input-len", type=int, nargs=2, default=[128, 512])
    p.add_argument("--output-len", type=int, nargs=2, default=[64, 256])
    p.add_argument("--shared-prefix", type=int, default=0)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default="results/serving.jsonl")
    a = p.parse_args(argv)

    rows = []
    levels = [a.request_rate] if a.request_rate else a.concurrency
    for c in levels:
        n = a.num_prompts or max(16, 4 * int(c if not a.request_rate else 16))
        wl = make_workload(n, tuple(a.input_len), tuple(a.output_len), a.vocab, a.seed, a.shared_prefix)
        if a.warmup:
            asyncio.run(run_closed_loop(a.url, a.model, [(wl[0][0][:32], 8)] * a.warmup, 1, a.timeout))
        with GpuSampler() as gpu:
            if a.request_rate:
                res, dur = asyncio.run(run_open_loop(a.url, a.model, wl, a.request_rate, a.timeout, a.seed))
            else:
                res, dur = asyncio.run(run_closed_loop(a.url, a.model, wl, int(c), a.timeout))
        s = summarize(res, dur, a.label, f"rate={c}" if a.request_rate else int(c), gpu)
        s.update(input_len=a.input_len, output_len=a.output_len, shared_prefix=a.shared_prefix)
        try:
            s["server_stats"] = httpx.get(f"{a.url}/stats", timeout=5).json()
        except Exception:  # noqa: BLE001  (vLLM has no /stats)
            pass
        rows.append(s)
        print_table([s])
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(s) + "\n")
    print("\n== summary")
    print_table(rows)


if __name__ == "__main__":
    main()

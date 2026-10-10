"""Turn results/*.json[l] into results/REPORT.md (the README tables) and results/*.png charts.

    python bench/make_report.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

ORDER = ["naive", "engine", "engine+kernels", "vllm"]
NAMES = {"naive": "Naive server (HF generate, 1 at a time)", "engine": "tiny-infer (batched + paged)",
         "engine+kernels": "tiny-infer + Project 1 kernels", "vllm": "vLLM (official)"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results")
    a = p.parse_args()
    d = a.dir
    serving = load_jsonl(os.path.join(d, "serving.jsonl"))
    latest = {}
    for s in serving:                                        # last run of each (label, concurrency) wins
        latest[(s["label"], s["concurrency"])] = s
    by_label = defaultdict(list)
    ckey = lambda c: (0, c, "") if isinstance(c, (int, float)) else (1, 0, str(c))
    for (lab, c), s in sorted(latest.items(), key=lambda kv: (kv[0][0], ckey(kv[0][1]))):
        by_label[lab].append(s)

    hf = json.load(open(os.path.join(d, "hf_baseline.json"))) if os.path.exists(os.path.join(d, "hf_baseline.json")) else None
    kv = json.load(open(os.path.join(d, "kv_capacity.json"))) if os.path.exists(os.path.join(d, "kv_capacity.json")) else None
    kb = json.load(open(os.path.join(d, "kernel_bench.json"))) if os.path.exists(os.path.join(d, "kernel_bench.json")) else None

    L = ["# tiny-infer results", ""]
    gpu = (hf or {}).get("gpu") or (kb or {}).get("gpu") or "GPU"
    L += [f"Hardware: {gpu}. Model: TinyLlama-1.1B-Chat-v1.0, FP16. Workload: prompts of 128-512 tokens, "
          "outputs of 64-256 tokens (ignore_eos, so every system generates the same tokens), closed loop at fixed "
          "concurrency, client-side timing over HTTP/SSE.", ""]

    # ---------------------------------------------------------------- main table
    L += ["## Serving comparison", "",
          "| System | Concurrency | Output tok/s | p50 TTFT | p99 TTFT | p50 TPOT | p99 TPOT | Peak running seqs |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if hf:
        L.append(f"| HF generate() in-process | 1 | {fmt(hf['c1_decode_tok_s'])} | {fmt(hf['c1_ttft_ms_p50'], 0)} ms | – | "
                 f"{fmt(hf['c1_tpot_ms_p50'], 2)} ms | – | 1 |")
        for sb in hf.get("static_batching", []):
            if "error" in sb:
                L.append(f"| HF static batch | {sb['batch']} | OOM | | | | | |")
            else:
                L.append(f"| HF static batch (padded) | {sb['batch']} | {fmt(sb['useful_output_tok_s'])} | – | – | – | – | {sb['batch']} |")
    is_extra = lambda l: l.startswith(("prefix", "chunk", "nochunk"))
    labels = [l for l in ORDER if l in by_label] + [l for l in by_label if l not in ORDER and not is_extra(l)]
    for lab in labels:
        for s in by_label[lab]:
            peak = (s.get("server_stats") or {}).get("peak_running") or (1 if lab == "naive" else None)
            L.append(f"| {NAMES.get(lab, lab)} | {s['concurrency']} | {fmt(s['output_tok_s'])} | {fmt(s['ttft_ms_p50'], 0)} ms | "
                     f"{fmt(s['ttft_ms_p99'], 0)} ms | {fmt(s['tpot_ms_p50'], 2)} ms | {fmt(s['tpot_ms_p99'], 2)} ms | "
                     f"{peak if peak is not None else '–'} |")
    L.append("")

    # ---------------------------------------------------------------- ratios
    def at(lab, c):
        return latest.get((lab, c))
    conc = sorted({c for (_, c) in latest if isinstance(c, int)})
    if conc:
        L += ["## Ratios at equal concurrency", "", "| Concurrency | engine / naive | (engine+kernels) / engine | best tiny-infer / vLLM |",
              "|---:|---:|---:|---:|"]
        for c in conc:
            n, e, k, v = at("naive", c), at("engine", c), at("engine+kernels", c), at("vllm", c)
            r1 = e["output_tok_s"] / n["output_tok_s"] if e and n else None
            r2 = k["output_tok_s"] / e["output_tok_s"] if k and e else None
            best = max([x["output_tok_s"] for x in (e, k) if x], default=None)
            r3 = best / v["output_tok_s"] if best and v else None
            sfx = lambda v, nd, unit: "–" if v is None else f"{v:,.{nd}f}{unit}"
            L.append(f"| {c} | {sfx(r1, 2, 'x')} | {sfx(r2, 3, 'x')} | {sfx(r3 * 100 if r3 else None, 0, '%')} |")
        L.append("")

    # ---------------------------------------------------------------- KV capacity
    if kv:
        t = kv["theory"]
        L += ["## Paged vs contiguous KV cache (same GPU, same 1 GiB pool, same kernels)", "",
              f"KV cost: {t['kv_bytes_per_token']:,} bytes/token. Pool: {t['pool_tokens']:,} token slots.", "",
              "| Allocation | Peak concurrent sequences | Mean KV waste | Preemptions | Output tok/s |", "|---|---:|---:|---:|---:|"]
        for r in kv["runs"]:
            L.append(f"| {r['mode']} | {r['peak_concurrent_seqs']} | {r['mean_kv_waste']:.1%} | {r['preemptions']} | {fmt(r['output_tok_s'])} |")
        L += ["", f"Paged runs **{kv['concurrency_gain']:.1f}x** more sequences at once and delivers "
              f"**{kv['throughput_gain']:.2f}x** the throughput. Closed form: contiguous fits {t['contiguous_max_seqs']} "
              f"sequences (it reserves max_model_len each, {t['contiguous_waste_at_mean_len']:.0%} of it unused at the mean "
              f"final length {t['mean_final_len']:.0f}); paged fits ~{t['paged_max_seqs_at_mean_final_len']}.", ""]

    # ---------------------------------------------------------------- kernels
    if kb:
        L += ["## Project 1 kernels in the decode path", "", "### Op level (T4, FP16, TinyLlama shapes)", "",
              "| Op | Shape | PyTorch | Kernel | Speedup | Kernel GB/s |", "|---|---|---:|---:|---:|---:|"]
        for r in kb["ops"]:
            L.append(f"| {r['op']} | {r['shape']} | {fmt(r['torch_us'])} µs | {fmt(r['kernel_us'])} µs | "
                     f"{fmt(r['speedup'], 2)}x | {fmt(r['kernel_GBs'], 0)} |")
        L += ["", "### End to end (decode step = TPOT at that batch size)", "",
              "| Decode batch | Step, PyTorch ops | Step, Project 1 kernels | TPOT speedup | Output tok/s (kernels) | Greedy outputs identical |",
              "|---:|---:|---:|---:|---:|---:|"]
        for r in kb["e2e"]:
            L.append(f"| {r['batch']} | {fmt(r['decode_step_ms_torch'], 2)} ms | {fmt(r['decode_step_ms_kernels'], 2)} ms | "
                     f"{fmt(r['tpot_speedup'], 3)}x | {fmt(r['output_tok_s_kernels'])} | {r['greedy_outputs_identical_fraction']:.0%} |")
        L.append("")

    # ---------------------------------------------------------------- extras (prefix caching / chunked prefill)
    extras = [l for l in by_label if is_extra(l)]
    if extras:
        L += ["## Prefix caching and chunked prefill", "", "| Run | Concurrency | p50 TTFT | p99 TTFT | p99 ITL | Output tok/s | Prefix hit rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for lab in extras:
            for s in by_label[lab]:
                hit = (s.get("server_stats") or {}).get("prefix_hit_rate")
                L.append(f"| {lab} | {s['concurrency']} | {fmt(s['ttft_ms_p50'], 0)} ms | {fmt(s['ttft_ms_p99'], 0)} ms | "
                         f"{fmt(s['itl_ms_p99'], 1)} ms | {fmt(s['output_tok_s'])} | {fmt(hit * 100 if hit is not None else None, 0)}% |")
        L.append("")

    # ---------------------------------------------------------------- step profile
    import glob
    profs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "step_profile_*.json")))]
    if profs:
        L += ["## Where a decode step goes (torch.profiler)", "",
              "| Batch | Kernels | Wall ms/step | GPU busy | GPU idle (CPU/launch) | GEMM-class | copy/gather | attention | norm | other | Roofline floor | % of floor |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in profs:
            b = r["breakdown_ms"]
            L.append(f"| {r['batch']} | {'on' if r['custom_kernels'] else 'off'} | {r['wall_step_ms']:.1f} | {r['gpu_busy_ms']:.1f} | "
                     f"{r['gpu_idle_ms (CPU/launch overhead)']:.1f} | {b.get('gemm', 0):.1f} | {b.get('copy/gather', 0):.1f} | "
                     f"{b.get('attention', 0):.1f} | {b.get('norm', 0):.1f} | {b.get('elementwise/other', 0):.1f} | "
                     f"{r['roofline_floor_ms']:.1f} | {r['pct_of_roofline']:.0f}% |")
        L += ["", "GEMM-class includes the batched matmuls inside the PyTorch attention fallback (SDPA's math path on sm_75).", ""]

    # ---------------------------------------------------------------- correctness
    corr = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "correctness_*.json")))]
    if corr:
        L += ["## Correctness vs HF generate() (TinyLlama, greedy)", "",
              "| dtype | Kernels | Token-identical prompts | Divergences (top-2 logit gap) | Result |", "|---|---|---:|---|---|"]
        for c in corr:
            dv = ", ".join(f"p{x['prompt']}@t{x['token']} ({x['top2_gap']:.2f})" for x in c["divergences"]) or "none"
            L.append(f"| {c['dtype']} | {'on' if c['custom_kernels'] else 'off'} | {c['identical']}/{c['prompts']} "
                     f"({c['tokens_per_prompt']} tokens each) | {dv} | {'PASS' if c['pass'] else 'FAIL'} |")
        L.append("")

    out = os.path.join(d, "REPORT.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")

    # ---------------------------------------------------------------- charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab in labels:
        rows = [s for s in by_label[lab] if isinstance(s["concurrency"], int)]
        if lab in ORDER and rows:
            xs = [s["concurrency"] for s in rows]
            axes[0].plot(xs, [s["output_tok_s"] for s in rows], "o-", label=NAMES[lab])
            axes[1].plot(xs, [s["ttft_ms_p50"] for s in rows], "o-", label=NAMES[lab])
            axes[2].plot(xs, [s["tpot_ms_p50"] for s in rows], "o-", label=NAMES[lab])
    for ax, t, yl in zip(axes, ["Output throughput", "p50 TTFT", "p50 TPOT"], ["tokens/s", "ms", "ms"]):
        ax.set_title(t); ax.set_xlabel("concurrent users"); ax.set_ylabel(yl); ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "serving.png"), dpi=140)
    print("wrote", os.path.join(d, "serving.png"))


if __name__ == "__main__":
    main()

"""Correctness oracle on the REAL model: tiny-infer greedy output vs HF generate(), token for token.

  fp32: must be identical on every prompt (same math, only scheduling differs).
  fp16: rounding order differs between kernels (cuBLAS tile choice depends on batch size, SDPA backend
        choice, our RMSNorm reduction order), so a sequence may diverge where the top-2 logits are
        within fp16 noise. Every divergence is reported with HF's top-2 logit gap at that step;
        a divergence with a large gap is a real bug and fails the check.

    python scripts/check_correctness.py --dtype float32
    python scripts/check_correctness.py --dtype float16 --custom-kernels
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_infer.config import EngineConfig  # noqa: E402
from tiny_infer.engine import LLMEngine  # noqa: E402
from tiny_infer.model import LlamaModel  # noqa: E402
from tiny_infer.sequence import SamplingParams  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Explain the difference between a process and a thread in simple terms.",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "List three benefits of regular exercise:",
    "Translate to Spanish: 'Where is the train station?'",
    "The key idea behind PagedAttention is",
    "Q: What is 17 * 23?\nA:",
    "Write a haiku about GPUs.",
    "In 1969, Apollo 11",
    "<|user|>\nWhat is continuous batching?</s>\n<|assistant|>\n",
    "A" * 40,
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--custom-kernels", action="store_true")
    p.add_argument("--margin", type=float, default=0.5, help="fp16: max allowed top-2 logit gap at a divergence")
    a = p.parse_args()

    from transformers import AutoTokenizer
    from tiny_infer.hf_utils import load_causal_lm
    dt = torch.float32 if a.dtype == "float32" else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = load_causal_lm(a.model, dt, "cuda")

    expected, scores = [], []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            out = hf.generate(ids, max_new_tokens=a.max_new, min_new_tokens=a.max_new, do_sample=False,
                              output_scores=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
            expected.append(out.sequences[0, ids.shape[1]:].tolist())
            scores.append([s[0].float() for s in out.scores])

    model = LlamaModel.from_hf_model(hf, "cuda", dt, use_custom_kernels=a.custom_kernels)
    del hf
    torch.cuda.empty_cache()
    cfg = EngineConfig(model=a.model, dtype=a.dtype, kv_cache_gb=2.0, use_custom_kernels=a.custom_kernels,
                       max_num_batched_tokens=64)          # small budget: exercises chunked prefill too
    eng = LLMEngine(cfg, model=model, tokenizer=tok)
    seqs = eng.generate(PROMPTS, SamplingParams(max_tokens=a.max_new, ignore_eos=True))

    exact, bad, divergences = 0, 0, []
    for i, (s, e) in enumerate(zip(seqs, expected)):
        got = s.output_ids
        if got == e:
            exact += 1
            continue
        j = next(k for k in range(len(e)) if got[k] != e[k])
        top2 = scores[i][j].topk(2).values
        gap = float(top2[0] - top2[1])
        # fp32: only an exact tie may flip (cuBLAS picks different tiles for batch 1 vs batch N)
        near_tie = gap <= (a.margin if a.dtype == "float16" else 1e-3)
        bad += 0 if near_tie else 1
        divergences.append({"prompt": i, "token": j, "top2_gap": gap, "near_tie": near_tie})
        print(f"prompt {i}: diverges at token {j}; HF top-2 logit gap there = {gap:.3f} "
              f"({'near-tie: fp16 rounding' if near_tie else 'REAL MISMATCH'})")
    print(f"\n{exact}/{len(PROMPTS)} prompts token-identical to HF ({a.dtype}, "
          f"kernels={'on' if a.custom_kernels else 'off'}, {a.max_new} tokens each)")
    import json
    os.makedirs("results", exist_ok=True)
    out = f"results/correctness_{a.dtype}{'_kernels' if a.custom_kernels else ''}.json"
    json.dump({"dtype": a.dtype, "custom_kernels": a.custom_kernels, "prompts": len(PROMPTS),
               "tokens_per_prompt": a.max_new, "identical": exact, "divergences": divergences,
               "pass": bad == 0}, open(out, "w"), indent=2)
    if bad:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()

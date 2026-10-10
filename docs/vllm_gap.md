# Why tiny-infer is slower than vLLM, and what would close the gap

> Ratios come from `results/REPORT.md` (runbook step 14); the time breakdown comes from `results/step_profile_*.json` (step 11b). Cells marked `—` are filled in after the T4 run.

## The ratio

| Concurrency | tiny-infer (best) out tok/s | vLLM out tok/s | tiny-infer / vLLM |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 16 | — | — | — |
| 64 | — | — | — |

## Where a decode step goes (batch 32, T4)

| Component | ms / step | Notes |
|---|---:|---|
| Wall time per step | — | |
| GPU busy | — | |
| → GEMMs (QKV, O, gate/up, down, lm_head) | — | cuBLAS; these read all 2.2 GB of weights every step |
| → attention | — | |
| → norms | — | |
| → copies / gathers / elementwise | — | RoPE, `index_copy_` KV writes, SiLU·mul, embedding |
| **GPU idle (CPU-side: Python scheduling, input building, ~300 kernel launches)** | — | |
| Roofline floor (weights + KV bytes ÷ 320 GB/s) | — | |

## The mechanisms, largest expected effect first

The step profile above is the evidence for each claim below. Where a row is small, the matching claim is downgraded.

1. **CUDA graphs: the decode step is launch-bound at small batch.**
   - One TinyLlama decode step in tiny-infer is about 15 kernel launches per layer × 22 layers, around 300 launches in all.
   - With the CPU overhead of each eager PyTorch launch, that can rival the 6.9 ms weight-read floor itself.
   - vLLM captures each decode batch size as a CUDA graph and replays it with a single launch.
   - *Measured as:* the "GPU idle" row. This is also why tiny-infer's batch-1 TPOT is furthest from vLLM's.
2. **Kernel fusion depth.** vLLM fuses:
   - RoPE with the K/V cache write (`reshape_and_cache`);
   - SiLU·mul;
   - residual-add + RMSNorm (as tiny-infer now does);
   - and, through `torch.compile`, much of the remaining elementwise glue.

   tiny-infer still runs RoPE as about 6 elementwise kernels, `index_copy_` twice per layer, and SiLU and mul separately. *Measured as:* the "copies/elementwise" row.
3. **CPU/GPU overlap.** tiny-infer builds the next step's inputs only after `.tolist()` has synchronised on the sampled tokens, so the GPU idles while Python schedules. vLLM overlaps the two:
   - API server and detokenisation in separate processes;
   - a persistent batch with incremental input updates;
   - async scheduling.
4. **Attention kernels for prefill.** tiny-infer's chunked prefill over a cached context gathers K/V and runs SDPA with a mask. vLLM's paged prefill kernels read the block table directly. This matters for TTFT under prefix hits and long chunked prompts. It matters little for decode throughput.
5. **Scheduler sophistication.** These policies affect tail latency more than steady-state throughput:
   - vLLM's token budget, priority between prefill and decode, preemption, and admission were tuned on real traces;
   - tiny-infer uses FCFS with decode first, recompute preemption, and a fixed 2,048-token budget.

## What would close it (and expected size)

| Change | Closes | Effort |
|---|---|---|
| CUDA-graph the decode step per batch bucket (static input buffers, graph replay) | most of "GPU idle" at small batch | medium: needs fixed-address inputs and padded batch sizes |
| Fused RoPE + cache write; fused SiLU·mul | most of the elementwise row | small: two short kernels |
| Overlap: build step N+1 inputs while step N runs; move detokenisation out of the event loop | the CPU share at large batch | medium |
| `torch.compile` the layer body | some of 1 and 2 at once | small to try, flaky to debug |

## The one-sentence version (for interviews)

"At batch 1 my engine is launch-bound: about 300 small kernel launches per token against a 7 ms weight-read floor. At high batch it's GEMM-bound like vLLM, so the gap narrows. CUDA graphs plus two more fusions would close most of what's left, and the step profile shows exactly how many milliseconds each one is worth."

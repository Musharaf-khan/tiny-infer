# Why tiny-infer is slower than vLLM, and what would close the gap

Measured on a Tesla T4 with vLLM 0.31.0 (`--dtype half`, prefix caching off). Same harness and workload for both systems.

## The ratio

| Concurrency | tiny-infer out tok/s | vLLM out tok/s | tiny-infer / vLLM | tiny-infer TPOT | vLLM TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.4 | 100.3 | 36% | 26.5 ms | 9.6 ms |
| 4 | 106.3 | 276.0 | 38% | 32.7 ms | 12.3 ms |
| 16 | 256.6 | 759.2 | 34% | 55.2 ms | 18.0 ms |
| 32 | 311.2 | 1,018.9 | 31% | 94.7 ms | 27.1 ms |
| 64 | 340.9 | 1,331.1 | 26% | 178.8 ms | 43.1 ms |

TTFT is close: at p50 tiny-infer is within about 1.5–2x of vLLM at every load, and it is faster at 1 user (43 ms vs 69 ms). **The gap is almost entirely decode, and it widens with batch size.** That pattern points to a per-sequence cost in the decode step, not a fixed one.

## Where a decode step goes (batch 32, T4, PyTorch ops)

| Component | ms / step | Notes |
|---|---:|---|
| Wall time per step | 79.5 | vLLM's TPOT at 32 users is 27.1 ms |
| GPU busy | 65.7 | |
| → copy / gather | **31.6** | gather of K/V out of the block pool + 8x `repeat_interleave` for GQA, every layer |
| → GEMM-class | 31.2 | weight GEMMs **plus** SDPA's batched matmuls (sm_75 has no FlashAttention, so SDPA uses its math path) |
| → elementwise / other | 2.8 | RoPE, SiLU·mul, residual adds |
| GPU idle (CPU-side: Python scheduling, input building, launches) | 13.8 | |
| Roofline floor (weights + KV ÷ 320 GB/s) | 7.7 | tiny-infer runs at 9.6% of it |

Here is what this predicts for the kernel rerun:

- The paged-attention kernel reads K/V once through the block table, so it should remove most of the 31.6 ms copy bucket and the attention part of the GEMM-class bucket.
- At batch 64 and about 600 tokens of context, the PyTorch path materialises roughly 0.3 GB of K/V copies per layer before attention. The kernel reads about 0.04 GB (64 × 600 × 4 KV heads × 64 × 2 bytes × K and V).

## The mechanisms, largest expected effect first

The step profile above is the evidence for each claim below. Where a row is small, the matching claim is downgraded.

0. **Decode attention through gather + repeat (measured: 31.6 of 65.7 GPU-ms at batch 32).** This is the largest single item, and it is the one Phase 5's kernel targets. vLLM reads paged K/V in-kernel and never materialises a dense copy.
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

"My engine reaches 26–38% of vLLM's throughput on a T4, and the gap is almost all decode. The profiler showed half of each decode step's GPU time going to copying K/V out of the paged cache and repeating it for GQA, which my paged-attention kernel removes. What's left is launch overhead at small batch (14 ms of GPU idle per step, which CUDA graphs fix) and fusion depth."

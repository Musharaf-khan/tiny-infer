# Why tiny-infer is slower than vLLM, and what would close the gap

Measured on a Tesla T4 with vLLM 0.31.0 (`--dtype half`, prefix caching off). Both systems ran the same harness, workload and concurrency levels. Raw data: `results/serving.jsonl` and `results/step_profile_*.json`.

## The ratio

| Users | tiny-infer, PyTorch ops | tiny-infer + kernels | vLLM | **kernels / vLLM** | TPOT p50: kernels vs vLLM | TTFT p50: kernels vs vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.4 | 58.6 | 100.3 | **58%** | 16.9 vs 9.6 ms | 40 vs 69 ms |
| 4 | 106.3 | 178.4 | 276.0 | **65%** | 19.3 vs 12.3 ms | 65 vs 92 ms |
| 16 | 256.6 | 530.8 | 759.2 | **70%** | 25.6 vs 18.0 ms | 82 vs 104 ms |
| 32 | 311.2 | 714.9 | 1,018.9 | **70%** | 38.4 vs 27.1 ms | 109 vs 135 ms |
| 64 | 340.9 | 938.9 | 1,331.1 | **71%** | 61.8 vs 43.1 ms | 180 vs 232 ms |

Throughput is in output tok/s.

- **Before the kernels: 26–38% of vLLM. After: 58–71%.**
- tiny-infer's TTFT is lower than vLLM's at every load, at both p50 and p99 (p99 at 64 users: 2.63 s vs 2.95 s). A likely cause, not yet isolated: decode-first scheduling with a 2,048-token budget admits new prompts quickly.
- **The whole remaining gap is decode (TPOT).** tiny-infer's TPOT is 1.76x vLLM's at 1 user and about 1.42x from 16 users up.

## Step 1: the gap that the kernels closed

Batch-32 decode step, `torch.profiler`, offline engine:

| Component | PyTorch ops | Project 1 kernels |
|---|---:|---:|
| Wall time per step | 79.8 ms | **29.2 ms** |
| GPU busy | 66.5 | 26.6 |
| → copy/gather (K/V out of the block pool + 8x GQA repeat) | **32.0** | **0.5** |
| → GEMM-class (weights + SDPA math-path matmuls) | 31.6 | 11.9 (weights only) |
| → attention kernel | – | 13.2 |
| → norms + elementwise | 2.9 | 1.0 |
| GPU idle (CPU-side) | 13.3 | 2.5 |
| Roofline floor (weights + KV ÷ 320 GB/s) | 7.7 | 7.7 |
| % of roofline | 9.6% | 26% |

This was the largest single gap to vLLM. vLLM never materialises a dense K/V copy, and now tiny-infer doesn't either.

## Step 2: what is left, largest first

1. **Small-batch launch overhead → CUDA graphs.**
   - At batch 1 the offline decode step takes 16.6 ms; the weight-read floor is 6.9 ms and vLLM's TPOT is 9.6 ms.
   - A tiny-infer step still launches roughly 15 kernels per layer, about 300 in all, each from eager PyTorch. vLLM captures each decode batch size as a CUDA graph and replays it with one launch.
   - This is the most likely explanation for the batch-1 gap. It isn't yet isolated with a batch-1 profile, which is the next measurement to take.
2. **The attention kernel at 15% of bandwidth.** At batch 64, attention is 25.7 of 39.4 GPU-ms, and the kernel moves 49 GB/s against 320 available. Measured headroom already exists: 128-token partitions beat the engine's current 512 by 1.5x at 64 × 1,024. Beyond that, the kernel needs more bytes in flight per SM (wider loads, more warps per partition).
3. **Serving-stack overhead.** At 64 users the offline engine's decode step is 44.9 ms, but the served TPOT is 61.8 ms. Two things account for the difference:
   - Prefills of arriving requests are interleaved into decode steps (closed-loop clients keep sending new prompts).
   - Detokenisation and HTTP work run in the same Python process as the engine thread and share its GIL.

   vLLM puts the API server, the detokeniser and the engine core in separate processes.
4. **Fusion depth.** RoPE is still about 6 elementwise kernels plus two `index_copy_` cache writes per layer, and SiLU and mul are separate. vLLM fuses RoPE with the cache write and SiLU·mul. After the kernels this is about 1 ms per step, so it is small on its own; it also shrinks item 1's launch count.
5. **Weight GEMMs at 54% of bandwidth** (2.05 GB in 11.9 ms). These are cuBLAS FP16 GEMMs with M = batch on sm_75. vLLM uses the same cuBLAS for these on a T4, so this part is not a vLLM advantage. It is the floor both systems share.

## What would close it

| Change | Targets | Evidence it matters | Effort |
|---|---|---|---|
| 128-token attention partitions (one-line default change) | item 2 | op-level 1.4–3.4x faster than 512 in every shape | trivial; needs an end-to-end rerun |
| CUDA-graph the decode step per batch bucket | item 1 | batch-1 step 16.6 ms vs 6.9 ms floor | medium: static input buffers, padded batch sizes |
| Engine in its own process; detokenise off the event loop | item 3 | 44.9 ms offline vs 61.8 ms served at batch 64 | medium |
| Fused RoPE + cache write, fused SiLU·mul | item 4 (+ fewer launches for item 1) | ~1 ms/step, plus fewer launches | small |

## The one-sentence version (for interviews)

"Out of the box my engine reached 26–38% of vLLM on a T4. The profiler showed half of every decode step was copying K/V out of the paged cache and repeating it for GQA. My paged-attention kernel removed that, which took the engine to 58–71% of vLLM and cut a batch-32 decode step from 80 to 29 ms. What's left is decode-only: launch overhead at small batch, which CUDA graphs fix; an attention kernel still at 15% of bandwidth; and serving-stack overhead from running everything in one Python process."

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

This was the largest single gap to vLLM. vLLM ran its own Triton attention kernel on this GPU (`TRITON_ATTN`, per `results/vllm_startup_evidence.txt`), and tiny-infer now reads the paged cache in-kernel too.

## Step 2: what is left, largest first

Each item cites the measured number it rests on. Where the *cause* is inferred rather than isolated, the item says so.

1. **Small-batch overhead → CUDA graphs** *(cause inferred, not isolated)*.
   - **Measured:** at batch 1 the offline decode step takes **16.6 ms** (`results/kernel_bench.json`, e2e batch 1). vLLM's TPOT is **9.6 ms** (`results/serving.jsonl`).
   - **Floor:** reading the weights once takes about **6.9 ms** (2.05 GiB from `results/hf_baseline.json` ÷ 320 GB/s).
   - **Evidence on vLLM's side:** vLLM captured CUDA graphs for 35 piecewise and 19 full batch sizes and ran `torch.compile` (`results/vllm_startup_evidence.txt`). tiny-infer launches every op eagerly from Python.
   - **Why it's only inferred:** GPU idle time was measured only at batch 32 (2.5 ms) and batch 64 (4.1 ms) with kernels (`results/step_profile_*.json`). A batch-1 profile is the next measurement; it would confirm or refute this item.
2. **The attention kernel at 15% of bandwidth** *(measured)*.
   - At batch 64, attention is **25.7 of 39.4 GPU-ms** (`results/step_profile_b64_kernels.json`).
   - At 64 × 1,024 the kernel moves **48.9 GB/s, 15.3%** of the T4's 320 GB/s (`results/kernel_bench.json`).
   - Headroom already measured: 128-token partitions beat the engine's current 512 by **1.48–3.38x** across shapes. Beyond that, the kernel needs more bytes in flight per SM.
3. **Serving-stack overhead** *(gap measured, split between causes not measured)*.
   - **Measured:** at 64 users the offline decode step is **44.9 ms** (`kernel_bench.json`), while the served TPOT is **61.8 ms** (`serving.jsonl`).
   - **Likely contributors:**
     - prefills of newly arriving requests interleaved into decode steps;
     - detokenisation and HTTP handling sharing one Python process and GIL with the engine thread.
   - **vLLM's layout:** its log shows the API server and engine core in separate processes (APIServer pid 15951, EngineCore pid 16133).
4. **Unfused glue** *(measured, small)*.
   - RoPE, SiLU·mul and the two per-layer `index_copy_` cache writes are separate PyTorch ops (`tiny_infer/model.py`).
   - Together with the residual adds they cost **0.8 ms** per batch-32 step (elementwise/other bucket).
   - Small on its own, but each op is also one more launch for item 1.
5. **Weight GEMMs at about 54% of bandwidth** *(measured)*.
   - About 2.07 GB of layer and `lm_head` weights stream through in **11.9 ms** (about 174 GB/s), at batch 32 (`step_profile_b32_kernels.json`).
   - The 2.07 GB is the weight total minus the 131 MB embedding table, which a decode step only indexes.
   - These are cuBLAS FP16 GEMMs with M = batch. I didn't profile vLLM's kernels, so whether vLLM does better here is not measured.

## What would close it

| Change | Targets | Evidence it matters | Effort |
|---|---|---|---|
| 128-token attention partitions (one-line default change) | item 2 | op-level 1.4–3.4x faster than 512 in every shape | trivial; needs an end-to-end rerun |
| CUDA-graph the decode step per batch bucket | item 1 | batch-1 step 16.6 ms vs 6.9 ms floor | medium: static input buffers, padded batch sizes |
| Engine in its own process; detokenise off the event loop | item 3 | 44.9 ms offline vs 61.8 ms served at batch 64 | medium |
| Fused RoPE + cache write, fused SiLU·mul | item 4 (+ fewer launches for item 1) | 0.8 ms/step measured, plus fewer launches | small |

## The one-sentence version (for interviews)

"Out of the box my engine reached 26–38% of vLLM on a T4. The profiler showed half of every decode step was copying K/V out of the paged cache and repeating it for GQA. My paged-attention kernel removed that, which took the engine to 58–71% of vLLM and cut a batch-32 decode step from 80 to 29 ms. What's left is decode-only. My attention kernel still runs at 15% of bandwidth. Serving through one Python process adds 17 ms per step at 64 users. And at batch 1 I'm 2.4x above the weight-read floor, which I attribute to eager launch overhead, since vLLM uses CUDA graphs; a batch-1 profile would confirm it."

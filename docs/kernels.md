# Phase 5: Project 1 kernels inside the engine

Two kernels from [cuda-kernel-lab](https://github.com/Musharaf-khan/cuda-kernel-lab) were adapted for the decode path of a real serving engine. They are registered as PyTorch custom ops (`torch.ops.tiny_infer.*`) and measured both in isolation and end to end.

> Measured on a Tesla T4 (`results/kernel_bench.json`, `results/step_profile_*.json`). Every kernel passed the startup self-test against its PyTorch reference (worst max |error| 2.0e-3 in FP16).

## Kernel 1: fused residual-add + RMSNorm (FP16)

**Lineage:** `kernels/06_layernorm`. In Project 1, v3 (register-cached) ran 1.42x faster than `F.layer_norm`, and the fused add + LayerNorm ran 1.44x faster than PyTorch eager.

**What it replaces.** Each Llama layer runs two residual-add + RMSNorm pairs (44 pairs per step for TinyLlama, plus a final norm). In PyTorch, HF's `LlamaRMSNorm` plus the residual add is a chain of small kernels:

1. add
2. cast to fp32
3. pow
4. mean
5. add eps
6. rsqrt
7. mul
8. cast back
9. mul by weight

Each launch reads and writes the activation through HBM.

**What changed from Project 1:**

| Project 1 (LayerNorm, FP32) | tiny-infer (RMSNorm, FP16) |
|---|---|
| two statistics (mean, variance) | one (sum of squares); no mean to subtract |
| `float4` loads | 16-byte `uint4` loads = 8 halves per load; hidden 2,048 = exactly 1 vector/thread at 256 threads |
| out-of-place | **in place** (vLLM convention): `residual ← x + residual`, `x ← norm(residual)·w` |
| exact two-pass | rounding sequence that mirrors HF step by step: fp16 add → fp32 stats → fp16 cast → fp16 weight-multiply |

**Roofline.** One read of x and the residual, one write of h and y, about 2 FLOP/byte. The kernel is purely memory-bound, so the right score is GB/s against the 320 GB/s peak.

The decode shapes are a catch: T = batch-size rows × 2,048 → only 20–516 KB per call at batch 1–32 (read x, residual, weight; write h, y). At those sizes **launch overhead dominates**, not bandwidth. Collapsing the 9 PyTorch ops of the reference implementation into 1 launch is the real win; GB/s per launch matters less.

| Shape (T × 2,048) | PyTorch | Kernel | Speedup | GB/s |
|---|---:|---:|---:|---:|
| 1 | 133.1 µs | 12.9 µs | **10.3x** | 1.6 |
| 8 | 140.7 µs | 13.2 µs | 10.6x | 10.2 |
| 32 | 132.7 µs | 12.6 µs | 10.5x | 41.9 |
| 128 | 165.9 µs | 12.4 µs | **13.4x** | 170.0 |

The PyTorch time is flat at about 133 µs from 1 to 32 rows: it is dominated by the per-op overhead of the 9 separate PyTorch ops. The kernel is flat at about 13 µs: a single launch whose data fits in one wave. Only at 128 rows does bandwidth start to show, at 170 GB/s (53% of peak).

## Kernel 2: GQA-aware paged decode attention (split-K)

**Lineage:** `kernels/07_attention` (FlashAttention-style fused attention, v2 reached 94.5-95.8% of PyTorch FP32 SDPA) and the online-softmax helpers in `common/reduce.cuh`.

**Why the prefill kernel doesn't carry over.** Project 1's kernel assigned 64 query rows per block and reused each K/V tile, staged in shared memory, across those 64 rows. Decode has **one** query row per head, so:

- there is no reuse to exploit: every K/V byte is used by exactly one dot product per query head;
- arithmetic intensity is about 1 FLOP/byte, far below the T4 ridge of about 200, so the kernel is purely a streaming problem;
- with batch 1 and 4 KV heads there are only 4 (sequence, kv-head) pairs, which is 4 blocks for 40 SMs.

**What the kernel does about each:**

1. **GQA-shared loads.** TinyLlama has 32 query heads over 4 KV heads, so G = 8. One block handles one KV head and all 8 of its query heads, so each K/V row is read once and used 8 times. The PyTorch path gathers the K/V into a dense copy and `repeat_interleave`s it 8x before SDPA, so it writes and re-reads roughly 9x the KV bytes.
2. **Paged addressing in the kernel.** `slot = block_table[t / 16] * 16 + t % 16`, so there is no gather and no dense copy. A warp reads one 128-byte K row (64 halves, 2 per lane): fully coalesced.
3. **Split-K over the context (Flash-Decoding).** Each 512-token partition of the context gets its own block, which fixes occupancy at small batch sizes. A second small kernel merges the partitions with the online-softmax rule (the same `(max, sum)` rescaling as `online_merge` in Project 1): `w_p = sum_p · exp(max_p − max)`.

**Inside a block (4 warps):**

- **Scores:** warp w takes tokens w, w+4, w+8, …. Each lane loads its 2 elements of the K row, computes partial dot products for all 8 heads, and does a `warp_sum`. The scores go to shared memory.
- **Softmax:** `block_max`, then exp and `block_sum`, per head.
- **Output:** the same token loop accumulates P·V into registers (8 heads × 2 elements per lane). The warps' partial outputs are summed through shared memory.

Compiled for sm_75: 46-64 registers, **0 spills**, 24 KB of shared memory for G = 8 with 512-token partitions.

| Batch × context | PyTorch (gather+SDPA) | Kernel | Speedup | Kernel GB/s (KV bytes) |
|---|---:|---:|---:|---:|
| 1 × 512 | 441 µs | 133 µs | 3.3x | 4.0 |
| 1 × 2,048 | 507 µs | 76 µs | **6.6x** | 27.5 |
| 8 × 1,024 | 1,247 µs | 211 µs | 5.9x | 39.7 |
| 32 × 512 | 2,415 µs | 369 µs | 6.5x | 45.4 |
| 32 × 1,024 | 4,587 µs | 689 µs | **6.7x** | 48.7 |
| 64 × 1,024 | 9,043 µs | 1,373 µs | 6.6x | 48.9 |

"Kernel" is the faster partition size, which was 128 tokens in every shape.

**The split-K (Flash-Decoding) effect, measured.** Cutting the context into 128-token partitions instead of 512-token ones made the kernel faster in every shape. The gain is largest when there is least parallelism:

| Shape | 512-token partitions | 128-token partitions |
|---|---:|---:|
| 1 × 512 | 448 µs | 133 µs (3.4x) |
| 1 × 2,048 | 213 µs | 76 µs (2.8x) |
| 64 × 1,024 | 2,025 µs | 1,373 µs (1.5x) |

At batch 1 the grid is only (1 sequence × 4 KV heads × partitions) blocks, so smaller partitions mean more of the 40 SMs get work.

**The honest part.** 49 GB/s is 15% of the T4's 320 GB/s, so the kernel is not yet bandwidth-bound. Two likely reasons:
- each 4-warp block streams its partition one 128-byte row per warp at a time, with little memory-level parallelism per SM;
- the softmax runs 8 block-wide reductions, one per query head.

The engine currently calls the kernel with 512-token partitions; switching to 128 is the measured next step.

## End to end

The decode-step time at a fixed decode batch, with kernels off vs on, is TPOT under that load. Runs are interleaved off/on/off/on to cancel thermal and clock drift (the T4 is power-capped at 70 W, which Project 1 also hit).

| Decode batch | PyTorch ops | Project 1 kernels | TPOT speedup |
|---:|---:|---:|---:|
| 1 | 24.45 ms | 16.58 ms | **1.47x** |
| 16 | 50.44 ms | 20.17 ms | **2.50x** |
| 32 | 85.97 ms | 30.08 ms | **2.86x** |
| 64 | 156.38 ms | 44.85 ms | **3.49x** |

In this run (512-token prompts, 256 output tokens, interleaved off/on/off/on), greedy output was **identical with and without the kernels in 100% of sequences** at every batch size.

The same kernels in the serving sweep (HTTP, mixed lengths):

| Users | Output tok/s, PyTorch ops | Output tok/s, kernels | Gain |
|---:|---:|---:|---:|
| 1 | 36.4 | 58.6 | 1.61x |
| 4 | 106.3 | 178.4 | 1.68x |
| 16 | 256.6 | 530.8 | 2.07x |
| 32 | 311.2 | 714.9 | 2.30x |
| 64 | 340.9 | 938.9 | **2.75x** |

**Profiler view of one batch-32 decode step:**

| Component | PyTorch ops | Kernels |
|---|---:|---:|
| Wall time | 79.8 ms | 29.2 ms |
| copy/gather | 32.0 ms | 0.5 ms |
| GEMM-class | 31.6 ms | 11.9 ms |
| attention | – | 13.2 ms |
| norms | in "other" | 0.17 ms |
| GPU idle | 13.3 ms | 2.5 ms |

GPU idle shrank too, because the step now launches far fewer kernels.

**Why the end-to-end gain (1.5–3.5x) is smaller than the op-level gain (up to 13x): Amdahl.**

After the kernels, the batch-32 decode step is 26.6 GPU-ms:
- 11.9 ms of weight GEMMs, which the kernels don't touch. About 2.07 GB of layer and `lm_head` weights in 11.9 ms is about 174 GB/s, 54% of peak (weight total from `results/hf_baseline.json` minus the 131 MB embedding table).
- 13.2 ms of attention.
- about 1.5 ms of everything else.

The norm kernel's 10x applies to a share that was already small. The attention kernel's 6.6x applies to the largest share. Before the kernels, that share was copying plus SDPA's math-path matmuls, which made up most of the 66.5 GPU-ms.

This is also why the end-to-end gain grows with batch: the attention share grows with batch × context, while the weight-GEMM share stays almost flat.

## What I'd change next

- Fuse RoPE with the K/V cache write (removes a read and write of K plus the `index_copy_`).
- Fuse SiLU·mul into one kernel.
- Use a tensor-core (`mma.sync`) score loop for large G or long context. At G = 8 the dot products are tiny, so the kernel stays memory-bound and this isn't worth it on TinyLlama.

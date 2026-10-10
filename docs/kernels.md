# Phase 5: Project 1 kernels inside the engine

Two kernels from [cuda-kernel-lab](https://github.com/Musharaf-khan/cuda-kernel-lab) were adapted for the decode path of a real serving engine. They are registered as PyTorch custom ops (`torch.ops.tiny_infer.*`) and measured both in isolation and end to end.

> Numbers marked `—` are filled in from `results/kernel_bench.json` after the T4 run (runbook step 11).

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

The decode shapes are a catch: T = batch size rows (1-64) × 2,048 → only 4-256 KiB per call. At those sizes **launch overhead dominates**, not bandwidth. Collapsing about 9 launches into 1 is the real win; GB/s per launch matters less.

| Shape (T × 2,048) | PyTorch | Kernel | Speedup | GB/s |
|---|---:|---:|---:|---:|
| 1 | — | — | — | — |
| 32 | — | — | — | — |
| 128 | — | — | — | — |

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
| 1 × 2,048 | — | — | — | — |
| 32 × 1,024 | — | — | — | — |
| 64 × 1,024 | — | — | — | — |

## End to end

The decode-step time at a fixed decode batch, with kernels off vs on, is TPOT under that load. Runs are interleaved off/on/off/on to cancel thermal and clock drift (the T4 is power-capped at 70 W, which Project 1 also hit).

| Decode batch | PyTorch ops | Project 1 kernels | TPOT speedup |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 32 | — | — | — |
| 64 | — | — | — |

**Why the end-to-end gain is smaller than the op-level gain (Amdahl).** At decode, most of the GPU time should be the weight-reading GEMMs (the step profile gives the measured share), which are untouched here; cuBLAS already runs them near the bandwidth roofline. The kernels can only shrink the remaining share:

- norms;
- attention, which grows with batch × context, so the attention kernel matters more at large batch and long context;
- the gather copies the attention kernel removes.

`bench/step_profile.py` measures each share directly.

## What I'd change next

- Fuse RoPE with the K/V cache write (removes a read and write of K plus the `index_copy_`).
- Fuse SiLU·mul into one kernel.
- Use a tensor-core (`mma.sync`) score loop for large G or long context. At G = 8 the dot products are tiny, so the kernel stays memory-bound and this isn't worth it on TinyLlama.

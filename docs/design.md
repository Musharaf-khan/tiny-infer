# tiny-infer design notes

Each decision below gives what was chosen, the alternatives, and why. These are also the answers to the "why did you…" interview questions.

## 1. The two jobs inside one request

| | Prefill | Decode |
|---|---|---|
| Work | the whole prompt in one forward pass | one token per step |
| GEMM shape | `[prompt_len × hidden] × W` | `[batch × hidden] × W` |
| Arithmetic intensity of the GEMMs | about `prompt_len` FLOP/byte | about `batch` FLOP/byte |
| Bound by (T4 ridge ≈ 65 TFLOP/s ÷ 320 GB/s ≈ 200 FLOP/byte) | **compute** once the prompt is ≳ 200 tokens | **memory bandwidth** until batch ≳ 200 |
| Governs | TTFT | TPOT / ITL |

Here is the consequence for TinyLlama on a T4. One decode step must read all ~2.2 GB of weights, which takes at least 2.2 GB ÷ 320 GB/s ≈ **6.9 ms**, whether the step serves one user or sixty-four. Reading the weights once and serving B users with them multiplies decode throughput by about B, until KV reads and compute catch up. That is why batching exists. It also explains why batching helps decode far more than it helps prefill, which is already compute-bound.

## 2. KV-cache math

```text
bytes/token = 2 (K,V) × n_layers × n_kv_heads × head_dim × dtype_bytes
TinyLlama   = 2 × 22 × 4 × 64 × 2 = 22,528 B ≈ 22 KiB/token  → 44 MiB per 2,048-token sequence
```

GQA (32 query heads sharing 4 KV heads) makes the cache 8x smaller than full multi-head attention would. `scripts/kv_math.py` does this calculation for any HF config.

## 3. The packed batch

All of a step's new tokens are concatenated into one `[T, hidden]` tensor, decode tokens first and then prefill chunks. There is no padding.

- Every linear layer (fused QKV, O, fused gate/up, down) runs **once** over all T rows, so the weights are read once per step.
- Attention is the only part that needs per-sequence structure.
- `lm_head` runs only on the rows that will be sampled. For a 2,048-token prefill, that skips a `[2047 × 2048] × [2048 × 32000]` GEMM.

Why not pad to `[B, max_len]`? With mixed lengths, padding wastes compute in proportion to the length variance. HF static batching pays exactly this cost, and `bench/hf_baseline.py` reports it as `wasted_fraction`.

## 4. The one rule

`Sequence.token_ids` holds the prompt plus generated tokens, and `num_computed_tokens` counts the tokens whose K/V is already in the cache. Each step computes `token_ids[num_computed : num_computed + n]` and samples only when it reaches the end.

| Mechanism | n | starting num_computed |
|---|---|---|
| decode | 1 | len − 1 |
| prefill | prompt length | 0 |
| chunked prefill | ≤ budget, several steps | 0, then grows |
| prefix-cache hit | the remainder | hit length (a multiple of block_size) |
| recompute after preemption | everything (prompt + outputs so far) | reset to 0 |

Since there is one code path, there is one oracle test per mechanism. Every one of them must reproduce HF's tokens exactly.

## 5. Scheduler (`scheduler.py`)

These are iteration-level policies, decided again at every step:

1. **Decode first, oldest first.** Running users get their next token before anyone new is admitted, which keeps TPOT stable under load.
2. **Memory pressure → preempt the newest** (recompute style, like vLLM's default). The victim's blocks are freed, `num_computed` is reset to 0, and the victim goes back to the **front** of the queue with its generated tokens kept. Its greedy continuation is unchanged; the oracle tests this.
   - *Alternative: swapping to CPU.* It keeps work done, but on a 1B model recomputing a prompt is cheaper than moving its KV over PCIe.
   - *Alternative: refusing new work.* That is not enough, because sequences grow after they are admitted.
3. **Continue chunked prefills** with whatever budget remains.
4. **Admit FCFS** while all of the following hold:
   - the pool can hold the whole prompt, minus prefix hits, plus a 1% watermark;
   - `running < max_num_seqs`;
   - budget remains;
   - nobody was preempted in this step.

   A request at the head of the queue that doesn't fit blocks the queue (no skipping), so large requests can't starve.
5. **Token budget** (`max_num_batched_tokens`, default 2,048) caps each step's work. It bounds each step's latency, so a decode is never stuck behind a whole 2K-token prefill: that is chunked prefill.

**Admission policy, precisely:** "fits" means `free_blocks + evictable_cached_blocks − blocks_needed ≥ watermark`, where `blocks_needed = ceil(prompt_tokens / 16) − cached_prefix_blocks`. In paged mode it is deliberately **optimistic**: it does not reserve room for future output, and preemption handles the cases where the optimism fails. In contiguous mode it is **pessimistic**: `ceil(max_model_len / 16)` blocks are reserved up front, so it never preempts but it wastes memory.

## 6. Block manager (`block_manager.py`)

- **Pool:** `num_blocks` physical blocks, each holding 16 tokens × 4 KV heads × 64 dims × (K and V) × 22 layers.
- **Block table per sequence:** logical block i → physical block id. The physical ids don't need to be contiguous, which is the virtual-memory analogy.
- **Allocation:** a new block is taken only when the last one is full, so waste is at most 15 token slots per sequence. Blocks come back to the free list the moment a sequence finishes.

Example with three sequences sharing a pool of 8 blocks (block size 4):

```text
seq A (6 tokens):  [ 0 | 1 ]          A0 = tokens 0-3 in phys 0, A1 = tokens 4-5 in phys 1 (2 slots free)
seq B (3 tokens):  [ 2 ]
seq C (9 tokens):  [ 3 | 4 | 5 ]      phys 5 holds 1 token
free list:         6, 7
B finishes → phys 2 returns to the free list and the next allocation can use it
```

**Prefix caching:**

- A **full** block is identified by `hash(parent_block_hash, its 16 token ids)`. Chaining in the parent hash makes the key encode the whole prefix, so the same 16 tokens after a different history never match.
- When a new prompt starts with cached blocks, those physical blocks are mapped into its table (`ref_count += 1`) and their prefill is skipped. If the whole prompt is cached, the last block is still recomputed, because logits are needed for the first output token.
- A block whose ref_count drops to 0 keeps its contents and goes on an LRU list. It counts as free and is evicted only when the free list runs out.

**Copy-on-write:**

- Sharing is safe without CoW because only **full** blocks are shared, and sequences only append into their last, partial block. So nobody ever writes into a shared block.
- CoW becomes necessary when a sequence **forks** with a partially filled shared block, as in parallel sampling (`n > 1`) or beam search. At the first divergent write, the block is copied: a new block is allocated, the contents copied, the writer's table entry remapped, and the old block's ref decremented.

## 7. Attention paths (`model.py`)

| Rows | Path |
|---|---|
| decode (n = 1) | **paged decode attention**: reads K/V through the block table. PyTorch reference = gather + repeat + SDPA; CUDA kernel = no gather (§8). |
| prefill, nothing cached | SDPA, causal, directly on the fresh K/V (the same call HF makes) |
| prefill chunk / prefix hit | gather the cached context through the block table + causal mask with offset |

New K/V are written into the pool with one `index_copy_` per layer, using a `slot_mapping` (physical block × 16 + offset) computed on the CPU.

## 8. Custom kernels (`csrc/`)

See [kernels.md](kernels.md) for the full write-up. They are registered with `TORCH_LIBRARY(tiny_infer, …)` as custom ops whose schemas mark mutated arguments (`Tensor(a!)`). They are JIT-built with `torch.utils.cpp_extension.load`. At startup they are self-tested against the PyTorch references, and the engine refuses to serve if a kernel disagrees.

## 9. CUDA graphs for decode steps (`cuda_graphs.py`)

**Why.** The batch-1 decode step takes 16.6 ms against a 6.9 ms weight-read floor (see [vllm_gap.md](vllm_gap.md)). vLLM captured CUDA graphs and used `torch.compile`; tiny-infer launches every op eagerly. The launch overhead is the suspected cause, inferred not isolated: a batch-1 profile would confirm it. CUDA graphs remove the per-op CPU launch cost by replaying one captured launch sequence per step.

**What landed (G2 slice 1).** `CUDAGraphRunner` captures one graph per batch-size bucket (powers of two up to `max_num_seqs`). Each bucket keeps static input buffers; per step the real `input_ids` and attention metadata are copied in, the graph is replayed, and logits are read from the bucket's static output buffer (only the first `nd` rows are valid, and the caller must sample before the next replay overwrites them). All graphs share one CUDA graph memory pool.

- **Decode-only.** The engine replays only when the step has no prefill tokens (`npf == 0`); prefill and chunked-prefill steps stay eager. A batch larger than the biggest bucket falls back to the eager forward.
- **Padding is idempotent.** Rows beyond the real batch duplicate row 0, so their K/V write writes the same token to the same slot twice: harmless, and it keeps the captured shapes fixed.
- **Captured context is worst-case.** The captured `max_decode_context` is `max_model_len`, which is correct for every step because empty partitions are ignored by the kernel merge, but short contexts pay the worst-case partition count. Context-length bucketing is the follow-up (see §11).
- **Opt-in, default off.** `--cuda-graphs` / `use_cuda_graphs`; it requires the custom-kernel path and CUDA, and falls back to eager on CPU with a warning.
- **CPU verification.** `tests/test_cuda_graphs.py` covers bucketing, padding, refusal on non-CUDA devices, and the engine flag falling back on CPU while staying token-identical to HF. Full suite: 46 passed, 1 skipped (2026-11-17).
- **GPU verification pending (Colab T4).** Capture/replay has not yet run on a GPU, so no speedup number is claimed here. The follow-up is to measure decode step time with `--cuda-graphs` on the T4 at batch 1/4/16/32 versus eager, then add context-length bucketing, and only then consider default-on.

## 10. Correctness strategy

- **The oracle:** greedy output must be identical to HF `generate()`.
- **fp32 on CPU (tests):** exact match is required for 10 configurations.
- **fp32 on GPU (real model):** exact match is required, except where HF's own top-2 logits are tied to within 1e-3.
- **fp16 with kernels:** different reduction orders (cuBLAS picks different GEMM tiles for batch 1 and batch 32; the RMSNorm sum order differs) can flip an argmax only where the top-2 logits are within fp16 noise. Every divergence is reported with its logit gap, and a gap above 0.5 fails the check.

## 11. What I'd do next (ordered by expected gain)

1. ~~CUDA graphs for decode steps, captured per batch-size bucket.~~ **Done (slice 1, §9):** capture/replay for decode-only steps, opt-in, GPU verification pending on the T4. Follow-up: measure decode step time with `--cuda-graphs` at batch 1/4/16/32 versus eager, then add context-length bucketing so short contexts do not pay the worst-case partition count; only then consider default-on.
2. Prepare the next step's inputs on the CPU while the GPU runs this one (async scheduling).
3. Fused RoPE + KV-cache write kernel, and fused SiLU·mul.
4. A FlashAttention-style kernel for chunked prefill over paged context, instead of gather + SDPA.
5. An FP8/INT8 KV cache, for 2x the concurrency. The T4 has no FP8 math, so it would be storage-only with dequant inside the attention kernel.

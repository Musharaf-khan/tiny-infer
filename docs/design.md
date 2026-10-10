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

## 9. Correctness strategy

- **The oracle:** greedy output must be identical to HF `generate()`.
- **fp32 on CPU (tests):** exact match is required for 10 configurations.
- **fp32 on GPU (real model):** exact match is required, except where HF's own top-2 logits are tied to within 1e-3.
- **fp16 with kernels:** different reduction orders (cuBLAS picks different GEMM tiles for batch 1 and batch 32; the RMSNorm sum order differs) can flip an argmax only where the top-2 logits are within fp16 noise. Every divergence is reported with its logit gap, and a gap above 0.5 fails the check.

## 10. What I'd do next (ordered by expected gain)

1. CUDA graphs for decode steps, captured per batch-size bucket. This removes the CPU launch overhead; see [vllm_gap.md](vllm_gap.md) for how big that overhead is.
2. Prepare the next step's inputs on the CPU while the GPU runs this one (async scheduling).
3. Fused RoPE + KV-cache write kernel, and fused SiLU·mul.
4. A FlashAttention-style kernel for chunked prefill over paged context, instead of gather + SDPA.
5. An FP8/INT8 KV cache, for 2x the concurrency. The T4 has no FP8 math, so it would be storage-only with dequant inside the attention kernel.

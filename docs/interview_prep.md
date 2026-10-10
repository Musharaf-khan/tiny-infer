# Expert checklist: answers grounded in this codebase

Practise saying each answer out loud in under two minutes, then whiteboard it.

## 1. Whiteboard the iteration-level scheduler

```text
every step:
  1. decode  — running seqs, oldest first: reserve 1 slot each; if no block: preempt NEWEST (free, requeue front)
  2. prefill — continue partial (chunked) prefills with the leftover token budget
  3. admit   — FCFS while (pool fits prompt − prefix hits + watermark) & running < max_num_seqs & budget left
  4. run ONE forward over all scheduled tokens (packed, no padding) → sample → append
  5. retire finished → blocks freed immediately → the next waiting request joins next step
```

Points to make:

- The batch is rebuilt every iteration, so no one waits for the slowest member.
- Decode goes first to protect TPOT.
- The token budget bounds the latency of each step (chunked prefill).

## 2. Paged KV: block table lifecycle

| Event | What happens |
|---|---|
| Admit | allocate `ceil(prompt/16)` blocks; prefix hits are mapped in with ref++ |
| Every decode step | if `len(tokens) > 16 × len(table)`, take one block from the free list (or evict the LRU cached block) |
| Finish | ref−− on each block; at ref 0 → free list, or LRU cache if the block was hashed |
| Preempt | same as finish, but the sequence goes back to the front of the queue with `num_computed = 0` |

Waste per sequence is at most 15 token slots, compared with `max_len − actual` when allocation is contiguous. CoW is needed only when a forked sequence shares a **partial** block; see `design.md §6`.

## 3. KV math for any model

`2 × L × n_kv × d × bytes`:

| Model | Per token | Context × users | Total |
|---|---:|---:|---:|
| TinyLlama | 22 KiB | 2K tokens | 44 MiB per sequence |
| Llama-3-8B | 128 KiB | 8K tokens × 16 users | 16 GiB |
| Llama-3-70B | 320 KiB | 8K tokens × 32 users | 80 GiB |

For Llama-3-70B that is 80 GiB of cache alone, more than its 140 GB of weights leave on two 80 GB GPUs, which is why GQA and KV quantisation exist. Run `scripts/kv_math.py --preset …` to check.

## 4. Prefill vs decode on the roofline

| | Prefill | Decode |
|---|---|---|
| GEMM intensity | about prompt_len FLOP/byte | about batch FLOP/byte |
| Where it sits (T4 ridge ≈ 200) | right of the ridge: compute-bound | left of the ridge: bandwidth-bound |
| Consequence | TTFT ∝ prompt FLOPs | TPOT is flat (~weights ÷ BW ≈ 7 ms on T4 for TinyLlama) until batch ≈ 200, so throughput rises with batch almost for free |
| Attention | intensity grows with prompt length | about 1 FLOP/byte regardless of batch, and grows with context; this is what Flash-Decoding parallelises |

## 5. Continuous vs static batching

**Static batching:**
- B requests start together and the batch ends when the longest one does.
- Utilisation over a batch is `Σ actual_len / (B × max_len)`, and new arrivals wait for the whole batch.
- It also pads every prompt to the longest one.

**Continuous batching:**
- Membership is re-evaluated at every token.
- A finished slot is refilled on the very next step.
- The GPU always works at the current live batch size, and TTFT for newcomers is about one step plus their own prefill, not a whole batch's duration.

Evidence in this repo: `bench/hf_baseline.py` (static, with `wasted_fraction`) against the engine sweep.

## 6. Why tiny-infer is slower than vLLM

The full answer is in [vllm_gap.md](vllm_gap.md):

1. CUDA graphs (launch-bound at small batch)
2. Fusion depth (RoPE + cache write, SiLU·mul, compiled glue)
3. CPU/GPU overlap (separate processes, async scheduling)
4. Paged prefill kernels

Back each one with a number from `step_profile`.

## 7. "How would you serve 10x more users on the same GPU?" (in this order)

1. **Batching.** Continuous batching uses decode's spare FLOPs. Throughput rises roughly linearly with batch until compute or KV bandwidth binds.
2. **Paging.** It removes the reservation waste, which on this workload means about 4x more sequences in the same pool (runbook step 10).
3. **Prefix caching.** A shared system prompt is stored once and its prefill skipped. This cuts memory and TTFT for agents and RAG.
4. **Quantisation.**
   - INT8/FP8 KV halves the cache, which doubles concurrency again.
   - INT4 weights shrink the 2.2 GB per-step read, which cuts TPOT.
5. Then shorter contexts or sliding windows, speculative decoding (lower TPOT per user), and finally more GPUs.

## 8. System design: "serve model X at Y QPS under Z ms p99"

1. **Per-request work.** Avg prompt P and output O tokens → prefill FLOPs ≈ 2·N_params·P; decode needs O steps.
2. **Latency SLOs.**
   - TTFT p99 ≤ Z₁ → bounds prompt length per step and needs chunked prefill.
   - TPOT ≤ Z₂ → bounds batch size, since TPOT grows once you leave the bandwidth regime.
3. **Memory.**
   - Concurrency needed: by Little's law, concurrent users ≈ QPS × E2E latency.
   - KV needed = concurrency × (P + O) × bytes/token.
   - Check that KV + weights + activations ≤ VRAM × 0.9. If not: quantise the KV, shard with TP, or add replicas.
4. **Throughput per replica.**
   - Decode tok/s at batch B ≈ B / TPOT(B).
   - Replicas = Y × O / per-replica tok/s, plus headroom for p99.
5. **Topology.**
   - Load balancer with prefix-affinity routing, which raises cache hits.
   - Separate prefill and decode pools if prefill interference blows the TPOT p99 (disaggregation).
   - Autoscale on queue depth and KV usage, not CPU.
6. **Measure.** Use a harness like `bench/harness.py` with an open-loop Poisson arrival rate (`--request-rate`). Closed-loop tests hide queueing.

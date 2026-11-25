# Expert checklist: answers grounded in this codebase

Practise saying each answer out loud in under two minutes, then whiteboard it.

**Status: refreshed Nov 25, 2026 (D4).** Since the Oct 15 baseline, four gap
items have landed as first slices: G1 (128-token attention partitions, now the
default, engine-level measured on the T4 Nov 14), G2 slice 1 (opt-in CUDA-graph
capture for decode steps), G3 slice 1 (engine in its own process over localhost
IPC), and G4 slice 1 (fused RoPE + KV-cache write kernel). GPU measurement of
the last three is still pending on the Colab T4, so no speedup is claimed for
them. Test suite: 76 passed, 1 skipped on CPU. Headline: **58-71% of vLLM
0.31.0 throughput** in the only same-sweep comparison; see §0 and §9 below.
Full sourced numbers live in [PROJECT_BRIEF.md](PROJECT_BRIEF.md); the gap
attribution lives in [vllm_gap.md](vllm_gap.md).

## 0. The 60-second story (say this first, then go deep)

> "Out of the box my engine reached 26-38% of vLLM on a T4. The profiler
> showed half of every decode step was copying K/V out of the paged cache and
> repeating it for GQA. My paged-attention kernel removed that, which took the
> engine to 58-71% of vLLM and cut a batch-32 decode step from 80 to 29 ms.
> Switching the attention partitions from 512 to 128 tokens then gained
> another 10-16% end to end, measured on the same T4 in November. Since then
> three more slices have landed, all waiting on GPU measurement: CUDA graphs
> for decode steps, the engine split into its own process, and a fused RoPE
> plus KV-cache write kernel. What's left is decode-only. My attention kernel
> still runs at 15% of bandwidth. Serving through one Python process adds
> 17 ms per step at 64 users. And at batch 1 I'm 2.4x above the weight-read
> floor, which I attribute to eager launch overhead, since vLLM uses CUDA
> graphs; a batch-1 profile would confirm it."

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

Attention partitions are 128 tokens by default (G1, Nov 14): the kernel bench
measured 128 as 1.48-3.38x faster than 512 at the op level, and the serving
sweep measured 1.10-1.16x end to end on the T4. `--attention-partition-size 512`
switches back.

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

## 6. Why tiny-infer is still slower than vLLM

The full answer is in [vllm_gap.md](vllm_gap.md). The gap is entirely in
decode (TPOT). Every item below has seen a landed slice since Oct 15, but the
GPU measurement for each landed slice is still pending on the Colab T4, so
quote the gap, not a fix:

1. **Attention kernel at 15% of bandwidth (measured).** At batch 64,
   attention is 25.7 of 39.4 GPU-ms, at 48.9 GB/s of 320. G1 closed the
   measured headroom: 128-token partitions beat 512 by 1.48-3.38x at the op
   level and 1.10-1.16x end to end. Beyond that the kernel needs more bytes
   in flight per SM. Next: a FlashAttention-style kernel for chunked prefill
   (G6), still unbuilt.
2. **Serving-stack overhead (gap measured, causes not split).** The decode
   step takes 44.9 ms offline but TPOT is 61.8 ms through the HTTP server at
   64 users. **G3 slice 1 landed Nov 20:** the engine can now run in its own
   process over localhost IPC (`--engine-process`), with detokenisation in
   the server process. Whether the split closes any of the 44.9 -> 61.8 ms
   gap is **not yet measured** on the GPU. Follow-up: measure TPOT/throughput
   through HTTP with `--engine-process` vs in-process, then add reconnect
   handling for a dropped engine connection.
3. **Small-batch launch overhead (cause inferred, not isolated).** The
   batch-1 decode step takes 16.6 ms against a 6.9 ms weight-read floor;
   vLLM's TPOT is 9.6 ms. **G2 slice 1 landed Nov 17:** CUDA graphs are
   captured per batch-size bucket for decode-only steps (`--cuda-graphs`,
   opt-in, eager fallback; see `design.md §9`). Whether graphs close the
   batch-1 gap is **not yet measured** on the GPU. A batch-1 profile would
   confirm the cause. Follow-up: measure decode step time with
   `--cuda-graphs` at batch 1/4/16/32 vs eager, then context-length bucketing
   so short contexts do not pay the worst-case partition count.
4. **Fusion depth (partly landed).** **G4 slice 1 landed Nov 24:** a fused
   RoPE + KV-cache write kernel (`tiny_infer/csrc/rope_kv_write.cu`),
   dispatched through the op layer with the exact pre-G4 PyTorch code as the
   reference oracle; CPU behaviour unchanged and the engine stays
   token-identical to HF. Its GPU effect is **not yet measured**. The other
   half of G4, fused SiLU-mul, is still unfused glue at 0.8 ms per batch-32
   step: that is the next code slice (see §9).

Back each one with a number from `step_profile`, and say "pending GPU
verification" whenever the slice has not run on the T4 yet.

## 7. "How would you serve 10x more users on the same GPU?" (in this order)

1. **Batching.** Continuous batching uses decode's spare FLOPs. Throughput rises roughly linearly with batch until compute or KV bandwidth binds.
2. **Paging.** It removes the reservation waste, which on this workload measured 6.2x more concurrent sequences in the same 1 GiB pool (23 → 143; `results/kv_capacity.json`).
3. **Prefix caching.** A shared system prompt is stored once and its prefill skipped. This cuts memory and TTFT for agents and RAG.
4. **Quantisation.**
   - INT8/FP8 KV halves the cache, which doubles concurrency again.
   - INT4 weights shrink the 2.2 GB per-step read, which cuts TPOT.
5. Then shorter contexts or sliding windows, speculative decoding (lower TPOT per user), and finally more GPUs.

Note what is already done in this repo, so you do not pitch it as future work:
128-token attention partitions are the default (G1, measured 1.10-1.16x end to
end), CUDA graphs and a separate engine process exist as opt-in flags
(`--cuda-graphs`, `--engine-process`) with GPU measurement still pending, and
the fused RoPE + KV-cache write kernel landed Nov 24 (GPU measurement pending;
fused SiLU-mul is the remaining half).

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

## 9. Numbers to quote (all measured on a Colab T4 unless noted)

| Claim | Number | Source |
|---|---|---|
| Throughput vs vLLM 0.31.0, same sweep | 58-71% (26-38% without kernels) | PROJECT_BRIEF.md §9.1 |
| TTFT vs vLLM | lower at p50 and p99 at every load (p99 at 64 users: 2.63 s vs 2.95 s) | PROJECT_BRIEF.md §9.1 |
| Batch-32 decode step, kernels off -> on | 80 ms -> 29 ms (2.75x serving throughput at 64 users) | PROJECT_BRIEF.md §9.6 |
| G1: 128 vs 512 partitions, engine-level | 1.10-1.16x output tok/s (Nov 14, T4) | PROJECT_BRIEF.md §9.7 |
| G1: 128 vs 512 partitions, op-level | 1.48-3.38x | kernel bench |
| Paged KV capacity in 1 GiB pool | 23 -> 143 concurrent sequences (6.2x) | `results/kv_capacity.json` |
| Test suite | 76 passed, 1 skipped on CPU | `tests/` |
| Greedy output vs HF `generate()` | token-identical, with and without kernels | PROJECT_BRIEF.md §10 |
| Fused add+RMSNorm vs PyTorch | 10.3-10.6x; paged attention 3.3-6.7x; KV-copy per step 32.0 -> 0.5 ms | docs/kernels.md, PROJECT_BRIEF.md §9.5 |

**What NOT to claim:**

- Do not quote 68-84% of vLLM. That is a cross-run read mixing two sweeps on
  different days: an indication, not a measurement. The measured figure is
  58-71%.
- Do not claim any speedup for CUDA graphs, the separate engine process, or
  the fused RoPE + KV-cache write kernel. All three landed as code slices
  with GPU verification pending on the Colab T4.
- Do not round in a way that inflates: 58-71% must not become "~70%", and
  2.75x must not become "3x".
- Never call tiny-infer deployed, in production, or a website. It is a
  working prototype of an inference server.

## 10. The next gap in the story

The next code slice is the other half of G4: **fused SiLU-mul** (unfused glue,
0.8 ms per batch-32 step). After that, the story needs GPU measurements, not
more code: measure the fused RoPE + KV-cache write kernel vs the unfused path,
measure `--cuda-graphs` decode step time at batch 1/4/16/32 vs eager, and
measure TPOT/throughput through HTTP with `--engine-process` vs in-process to
see whether the split closes any of the 44.9 -> 61.8 ms gap. Only then consider
default-on for either flag. Longer term: async scheduling (G5, prepare the
next step's inputs on the CPU while the GPU runs this one) and a
FlashAttention-style kernel for chunked prefill (G6). See `design.md §11` and
PROJECT_BRIEF.md §12.

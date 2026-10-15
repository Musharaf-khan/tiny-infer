# tiny-infer: complete project brief

**Author:** Musharaf Khan Pathan · **Repo:** https://github.com/Musharaf-khan/tiny-infer · **Status as of 2026-10-15:** working prototype, all benchmarks measured on an NVIDIA Tesla T4.

---

## 0. Instructions for an AI reading this document

This document is the single source of truth about tiny-infer. Use it to write resume bullets, cover letters, LinkedIn posts, interview answers or project summaries.

**Rules:**

1. **Use only numbers that appear in this document**, with the context given next to them (GPU, model, batch size or user count). Never round in a way that inflates a number: 58–71% must not become "~70%", and 2.75x must not become "3x".
2. **Respect section 13 ("What NOT to claim")**. Several tempting claims are false or not yet proven.
3. **Describe it as a "working prototype" or "from-scratch inference engine/server"**, never as "deployed", "in production", a "product" or a "website".
4. **The author built it as a solo project.** It is the second project in a three-project GPU-systems plan. Project 1 is "cuda-kernel-lab"; see section 2.
5. **When space is short, prefer the three headline results:**
   - 58–71% of vLLM's throughput;
   - 6.2x more concurrent requests from paging;
   - CUDA kernels cutting a decode step from 80 to 29 ms (2.75x serving throughput at 64 users).

---

## 1. One-paragraph summary

tiny-infer is an LLM inference server built from scratch in Python and CUDA. It is the kind of system that sits between a language model and its users, like a miniature version of vLLM. It has:

- an iteration-level **continuous batching** scheduler;
- a **paged KV cache** (PagedAttention-style block manager) with **prefix caching** and **recompute preemption**;
- **chunked prefill**;
- an **OpenAI-compatible HTTP API** with token streaming over server-sent events;
- two **custom CUDA kernels** written by the author: a GQA-aware paged decode-attention kernel and a fused residual-add + RMSNorm kernel. They are registered as PyTorch custom ops and plugged into the decode path.

The author benchmarked it on a single NVIDIA T4 serving TinyLlama-1.1B (FP16) against three baselines, all through one harness:
- HuggingFace `generate()`;
- a naive one-request-at-a-time server;
- **vLLM 0.31.0**.

It reaches **58–71% of vLLM's throughput**, with lower time-to-first-token than vLLM at every load tested. Profiling showed half of each decode step was spent copying the KV cache; the author's CUDA kernel removed that copy, which is the project's main finding.

---

## 2. Context: the "kernel → engine → tokens/sec" story

- **Project 1, cuda-kernel-lab** (https://github.com/Musharaf-khan/cuda-kernel-lab): 25 hand-written CUDA kernel versions across 8 problems on a T4, each profiled with Nsight Compute. Highlights:
  - SGEMM at 101.3% of cuBLAS (autotuned, warp tiling);
  - LayerNorm 1.42x faster than `F.layer_norm`;
  - fused residual-add + LayerNorm 1.44x faster than PyTorch eager;
  - FlashAttention-style fused attention at 94.5–95.8% of PyTorch FP32 SDPA.
- **Project 2, tiny-infer** (this project): puts Project 1's kernel techniques inside a real serving engine and measures what they change end to end, in tokens per second.
- **The narrative:** "Project 1 proved my kernels beat PyTorch ops in isolation. Project 2 asked whether that matters inside a real serving system. Profiling showed the real bottleneck was a memory copy, and a kernel that removed it gave a 2.75x end-to-end throughput gain."

Two kernels were adapted from Project 1:
- The register-cached LayerNorm and its fused add+LayerNorm became an FP16 fused add+RMSNorm.
- The prefill-oriented FlashAttention kernel and its online-softmax helpers became a decode-oriented paged attention kernel. Section 7 explains why decode needed a different design.

---

## 3. Glossary (for readers without an ML-systems background)

| Term | Meaning |
|---|---|
| **Inference server** | Software that loads a trained model and answers requests from many users. |
| **Prefill** | Processing the whole prompt in one parallel pass. Compute-bound. Determines time to first token. |
| **Decode** | Generating output one token at a time. Each step re-reads all model weights, so it is memory-bandwidth-bound. Determines time per output token. |
| **KV cache** | Stored attention keys/values for every past token, so they aren't recomputed. For TinyLlama: 2 × 22 layers × 4 KV heads × 64 dims × 2 bytes = **22,528 bytes per token** (44 MiB for a 2,048-token sequence). |
| **TTFT** | Time to first token (latency until streaming starts). |
| **TPOT** | Time per output token after the first (streaming speed). |
| **ITL** | Inter-token latency (gap between streamed tokens). |
| **Continuous batching** | Rebuilding the batch every decode step, so finished requests leave immediately and new ones join immediately. (Orca, OSDI '22.) |
| **Static batching** | A batch starts and ends together; everyone waits for the longest request. |
| **Paged KV cache / PagedAttention** | The KV cache is split into fixed 16-token blocks allocated on demand, with a per-request block table mapping logical to physical blocks, like OS virtual memory. (vLLM, SOSP '23.) |
| **Contiguous allocation** | Reserving the maximum length's worth of cache per request up front (wasteful). |
| **Prefix caching** | Requests sharing a prompt prefix (e.g. a system prompt) reuse the same cached blocks. |
| **Chunked prefill** | Splitting long prompts across several steps so running streams aren't stalled. |
| **Preemption (recompute)** | When memory runs out, evicting a request's cache and recomputing it later. |
| **GQA** | Grouped-query attention. TinyLlama has 32 query heads sharing 4 KV heads (groups of 8). |
| **Split-K / Flash-Decoding** | Splitting a long context across GPU thread blocks, then merging partial softmax results. |
| **Roofline floor** | The minimum time set by memory bandwidth: bytes that must be read ÷ 320 GB/s on a T4. |
| **vLLM** | The leading open-source LLM serving engine, used as the industry benchmark. |

---

## 4. Architecture

```text
HTTP clients ──► FastAPI server (/v1/completions, SSE streaming, incremental detokenisation)
                     │  add / abort requests
                     ▼
               AsyncEngine (background thread; per-request asyncio queues)
                     │
                     ▼
   ┌──────────────── LLMEngine.step() — one iteration ────────────────┐
   │ Scheduler ──► BlockManager ──► Model runner ──► Llama forward ──► Sampler │
   │ (who runs)    (KV blocks)      (pack tokens)    (+ CUDA kernels)          │
   └──────────────────────────────────────────────────────────────────┘
                     │ sampled tokens
                     ▼
               streamed back to each client
```

### The one rule that drives everything

Every request (a `Sequence`) keeps:
- `token_ids`: the prompt plus generated tokens;
- `num_computed_tokens`: how many of those already have K/V in the cache.

Each step computes the uncomputed suffix, up to a token budget, and samples a token only when it reaches the end. Every feature is a special case of this rule:

| Feature | Expressed as |
|---|---|
| Decode | 1 new token per step |
| Prefill | the whole prompt in one step |
| Chunked prefill | the prompt over several partial steps |
| Prefix-cache hit | `num_computed_tokens` starts above 0 |
| Preemption | `num_computed_tokens` reset to 0, then recomputed |

One code path covers all of them, so one correctness oracle tests all of them.

### Scheduler loop (every iteration)

1. **Decode first.** Each running request, oldest first, gets one token of budget. If the KV pool is full, the **newest** request is preempted: its blocks are freed, and it goes back to the front of the queue with its generated tokens kept.
2. **Continue chunked prefills** with the remaining budget.
3. **Admit new requests**, first come first served, while three conditions hold:
   - the KV pool can hold the prompt (minus prefix-cache hits) plus a 1% watermark;
   - running requests are below `max_num_seqs` (128);
   - token budget remains.

   No admissions happen in a step that preempted.
4. **Run one forward pass** over all scheduled tokens, packed into a single tensor with no padding.
5. **Sample, append, and retire finished requests.** Their blocks are freed immediately.

The per-step token budget (`max_num_batched_tokens`, default 2,048) bounds step latency.

---

## 5. Components (file by file)

| File | Lines | What it does |
|---|---:|---|
| `tiny_infer/config.py` | 100 | `ModelConfig` (read from a HuggingFace config) with KV-cache math (`kv_bytes_per_token`, `kv_math`); `EngineConfig` (all engine knobs). |
| `tiny_infer/sequence.py` | 87 | `Sequence` (token_ids, num_computed_tokens, block table, timing) and `SamplingParams` (max_tokens, temperature, top_p, stop tokens, ignore_eos). |
| `tiny_infer/block_manager.py` | 192 | Paged KV block manager:<br>• free list and reference counts<br>• per-sequence block tables<br>• **paged** (on-demand) and **contiguous** (reserve max length) modes<br>• **prefix caching**: full blocks hashed as `hash(parent_hash, 16 token ids)`, shared via ref counts, kept in an LRU list after release and evicted only when the free list is empty<br>• rollback on allocation failure |
| `tiny_infer/scheduler.py` | 135 | Iteration-level scheduler: decode-first, chunked prefill, FCFS admission, newest-first recompute preemption, abort. |
| `tiny_infer/model.py` | 187 | Llama forward pass over a **packed batch** with fused QKV and fused gate/up weights:<br>• K/V written into the block pool via a slot mapping<br>• decode rows → paged decode attention<br>• fresh prefill → causal SDPA<br>• chunked/prefix-hit prefill → gather from cache with an offset causal mask<br>• `lm_head` run only on rows being sampled<br>• RoPE tables built exactly like HuggingFace's. |
| `tiny_infer/ops.py` | 110 | PyTorch reference implementations that match HuggingFace's Llama numerics (RMSNorm, fused add+RMSNorm, paged decode attention), plus dispatchers that route to the CUDA kernels when enabled. |
| `tiny_infer/kernels.py` | 70 | JIT-builds the CUDA extension (`torch.utils.cpp_extension`) and **self-tests** every kernel against its reference. The engine refuses to start if any kernel disagrees. |
| `tiny_infer/csrc/rms_norm.cu` | 124 | Fused residual-add + RMSNorm CUDA kernel (FP16). |
| `tiny_infer/csrc/paged_attention.cu` | 266 | GQA-aware paged decode-attention CUDA kernel with split-K, plus a partition-combine kernel. |
| `tiny_infer/csrc/reduce.cuh` | 60 | Warp-shuffle and block reductions (carried over from Project 1). |
| `tiny_infer/csrc/ops.cpp` | 27 | Registers the kernels as PyTorch custom ops via `TORCH_LIBRARY(tiny_infer, …)`, with mutation-annotated schemas (`Tensor(a!)`). |
| `tiny_infer/engine.py` | 237 | `LLMEngine`. Its `step()` runs schedule → prepare inputs → forward → sample → retire. It also sizes the KV pool by profiling peak activation memory and collects stats (peak concurrency, KV waste, preemptions, per-step timing). |
| `tiny_infer/async_engine.py` | 96 | Runs the engine loop in a background thread and streams tokens to asyncio consumers. Handles client disconnects (aborts) and reports errors per request. |
| `tiny_infer/server.py` | 250 | FastAPI server: OpenAI-compatible `/v1/completions` (streaming and non-streaming), `/health`, `/stats`, `/v1/models`. Two backends: `engine` (tiny-infer) and `naive` (one HF `generate()` at a time, for comparison). |
| `tiny_infer/hf_utils.py` | 23 | Version-safe HuggingFace model loading, verified to load the requested dtype. |
| `bench/harness.py` | — | Benchmark client for **any** OpenAI-compatible endpoint (used for tiny-infer, the naive server and vLLM alike). |
| `bench/hf_baseline.py` | — | HuggingFace `generate()` baseline (concurrency 1, plus padded static batching). |
| `bench/kv_capacity.py` | — | Paged vs contiguous on the same fixed 1 GiB pool. |
| `bench/kernel_bench.py` | — | Kernel op-level and end-to-end benchmarks. |
| `bench/step_profile.py` | — | `torch.profiler` breakdown of one decode step. |
| `bench/make_report.py` | — | Generates `results/REPORT.md` and charts from raw results. |
| `scripts/` | — | Real-model correctness check (`check_correctness.py`), server benchmark orchestration, live 8-user streaming demo (`demo.py`), KV-math calculator (`kv_math.py`). |
| `notebooks/` | — | Colab T4 runbooks that produce every number in `results/`. |
| `tests/` | — | 29 automated tests (section 10). |

Totals: about 1,500 lines of engine Python, about 480 lines of CUDA/C++, and about 1,700 lines of benchmarks, scripts and tests. CI runs the tests on every push with GitHub Actions. A Dockerfile is included.

**Server flags:**
- `--custom-kernels`
- `--kv-mode paged|contiguous`
- `--kv-cache-gb`
- `--prefix-caching`
- `--no-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--block-size`
- `--backend engine|naive`

---

## 6. Key design decisions (and why)

1. **Packed batch, no padding.** All new tokens of a step form one `[T, hidden]` tensor. Each weight matrix is read once per step for all users, which is why batching helps decode. Padding would waste compute in proportion to length variance.
2. **Decode-first scheduling.** It protects the streaming speed of users who are already running.
3. **Optimistic admission plus recompute preemption.** Paged mode admits a request if its *prompt* fits, not its maximum length; preemption handles the overflow. On a 1B model, recomputing is cheaper than swapping the cache to CPU.
4. **FCFS with head-of-line blocking.** A large request at the front of the queue is never starved by smaller ones skipping ahead.
5. **Prefix-cache hashing over full blocks only, with chained hashes.** Shared blocks are never written, so copy-on-write is never needed. CoW would be needed only for forking (beam search or parallel sampling), which isn't implemented.
6. **Kernels as `torch.library` custom ops with a startup self-test.** The engine refuses to serve if a kernel disagrees with the PyTorch reference.
7. **Correctness oracle from day one.** Every feature must reproduce HuggingFace's greedy tokens exactly (section 10).
8. **One benchmark harness for every system.** Prompts are sent as token IDs with `ignore_eos`, so every system does identical work. Closed-loop clients at fixed concurrency; client-side timing over HTTP/SSE; p50 and p99 reported; prefill and decode throughput kept separate.

---

## 7. The CUDA kernels in detail

### Kernel 1: fused residual-add + RMSNorm (FP16), `csrc/rms_norm.cu`

- **Replaces:** the residual add plus HuggingFace's `LlamaRMSNorm`, which is **9 separate PyTorch ops** in the reference code (add, cast, pow, mean, add-eps, rsqrt, multiply, cast back, weight multiply). Each generated token runs **45 norm calls**: 44 use the fused add+RMSNorm kernel, and the very first layer's input norm uses a plain RMSNorm variant of the same kernel.
- **Design:**
  - one thread block per token row;
  - the row is loaded **once** into registers with 16-byte vector loads (8 halves per load);
  - sum of squares reduced with warp shuffles and one shared-memory hop;
  - output written once;
  - in place, vLLM-style (`residual ← x + residual`, `x ← norm(residual) · w`).
- **Numerics:** the rounding order mirrors HuggingFace's step by step: FP16 add → FP32 statistics → FP16 cast → FP16 weight multiply.
- **Compile:** 20–80 registers per variant, **0 spills** (sm_75).
- **Measured (T4):** **10.3–13.4x faster** than the PyTorch path. At 32 rows: 132.7 µs → 12.6 µs. The gain is almost entirely launch overhead (one launch instead of nine ops), because at decode sizes a call moves only 20–516 KB.

### Kernel 2: GQA-aware paged decode attention with split-K, `csrc/paged_attention.cu`

**Why Project 1's attention kernel didn't carry over.** It was built for prefill: 64 query rows per thread block, reusing each K/V tile across rows. Decode has **one** query row per head, so:
- there is no reuse, and the job is pure streaming (about 1 FLOP/byte);
- with batch 1 and 4 KV heads there are only 4 work items for the T4's 40 SMs.

**Three design moves:**
1. **GQA-shared loads.** One block serves one KV head and all 8 query heads that share it, so each K/V row is read from memory once and used 8 times. The PyTorch fallback instead gathered K/V into a dense copy and repeated it 8x.
2. **Paged addressing inside the kernel.** Token t lives at `block_table[t / 16] * 16 + t % 16`, so there is no gather and no dense copy. Each warp reads one 128-byte K row, fully coalesced.
3. **Split-K over the context (Flash-Decoding).** Each context partition gets its own thread block, and a second kernel merges the partial softmaxes with the online-softmax rescaling rule (from Project 1's `online_merge`).

**Inside a block (4 warps):**
- per-token dot products for all 8 heads, warp-reduced;
- scores in shared memory;
- block-wide max and sum per head;
- P·V accumulated in registers;
- warp partials summed through shared memory.

**Compile:** 45–64 registers, **0 spills**, 24 KB shared memory (G = 8, 512-token partitions).

**Measured (T4):** **3.3–6.7x faster** than gather + repeat + SDPA. At 32 sequences × 1,024 context: 4.59 ms → 0.69 ms. It peaks at **48.9 GB/s, about 15% of the T4's 320 GB/s**: correct and fast, but not yet bandwidth-bound.

**Split-K, measured:** 128-token partitions were **1.48–3.38x faster** than 512-token ones in every shape tested. The gain is biggest at batch 1, where parallelism is scarce. The engine still uses 512 by default; switching is a measured next step.

**Correctness of both kernels:** the startup self-test passes, with a worst max abs error of **2.0e-3** in FP16. It covers ragged lengths, shuffled physical blocks, GQA groups of 8 and 1, head dims 64 and 128, and both partition sizes.

---

## 8. Benchmark methodology

| Setting | Value |
|---|---|
| Hardware | 1× NVIDIA **Tesla T4** (Turing, sm_75, 320 GB/s, ~15 GiB), Google Colab |
| Model | **TinyLlama/TinyLlama-1.1B-Chat-v1.0**, FP16 (weights ≈ 2.05 GiB) |
| Main serving workload | prompts **128–512** tokens, outputs **64–256** tokens, `ignore_eos` (identical work for every system), closed-loop clients at fixed concurrency **1/4/16/32/64** |
| Requests per level | max(16, 4 × concurrency), i.e. 16 / 16 / 64 / 128 / 256 at 1 / 4 / 16 / 32 / 64 users; the naive server used 16 per level |
| vLLM | **v0.31.0**, `--dtype half`, `--max-model-len 2048`, `--max-num-seqs 128`, `--gpu-memory-utilization 0.90`, prefix caching off. Its own log shows it used Triton attention, `torch.compile` and CUDA graphs. |
| Timing | client side, from the SSE stream (TTFT, TPOT, ITL, end-to-end), p50 and p99 |
| Raw data | `results/serving.jsonl`, `hf_baseline.json`, `kv_capacity.json`, `kernel_bench.json`, `step_profile_*.json`, `vllm_startup_evidence.txt` |

**Caveat to state if asked:** these results come from two Colab T4 sessions on the same GPU model, not one.
- Session 1: the naive server, tiny-infer without kernels, vLLM, paging, prefix caching and chunked prefill.
- Session 2: tiny-infer with kernels, the kernel benchmarks and the profiles.

The kernel on/off end-to-end comparison in `kernel_bench.json` was run interleaved (off/on/off/on) within a single session, which makes it the most controlled kernel number.

---

## 9. Measured results (all on a T4)

### 9.1 Serving throughput and latency

| System | Users | Output tok/s | p50 TTFT | p99 TTFT | p50 TPOT |
|---|---:|---:|---:|---:|---:|
| HF `generate()` in-process | 1 | 35.6 | 76 ms | – | 28.1 ms |
| HF static batching (offline, padded) | 16 / 32 | 230.5 / 253.7 | – | – | – |
| Naive server (1 request at a time) | 1 | 32.9 | 82 ms | 112 ms | 29.0 ms |
| | 4 | 32.9 | 15.5 s | 18.3 s | 29.8 ms |
| | 16 | 33.1 | 36.7 s | **77.0 s** | 29.0 ms |
| tiny-infer, PyTorch ops | 1 | 36.4 | 43 ms | 54 ms | 26.5 ms |
| | 4 | 106.3 | 92 ms | 213 ms | 32.7 ms |
| | 16 | 256.6 | 144 ms | 837 ms | 55.2 ms |
| | 32 | 311.2 | 234 ms | 1.58 s | 94.7 ms |
| | 64 | 340.9 | 420 ms | 3.27 s | 178.8 ms |
| **tiny-infer + CUDA kernels** | 1 | **58.6** | 40 ms | 49 ms | 16.9 ms |
| | 4 | **178.4** | 65 ms | 162 ms | 19.3 ms |
| | 16 | **530.8** | 82 ms | 691 ms | 25.6 ms |
| | 32 | **714.9** | 109 ms | 1.30 s | 38.4 ms |
| | 64 | **938.9** | 180 ms | 2.63 s | 61.8 ms |
| vLLM 0.31.0 | 1 | 100.3 | 69 ms | 92 ms | 9.6 ms |
| | 4 | 276.0 | 92 ms | 216 ms | 12.3 ms |
| | 16 | 759.2 | 104 ms | 825 ms | 18.0 ms |
| | 32 | 1,018.9 | 135 ms | 1.46 s | 27.1 ms |
| | 64 | 1,331.1 | 232 ms | 2.95 s | 43.1 ms |

**Derived ratios:**

| Comparison | 1 user | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| tiny-infer + kernels ÷ vLLM (throughput) | 58% | 65% | 70% | 70% | 71% |
| tiny-infer without kernels ÷ vLLM | 36% | 38% | 34% | 31% | 26% |
| kernels ÷ no kernels (throughput) | 1.61x | 1.68x | 2.07x | 2.30x | **2.75x** |
| tiny-infer (no kernels) ÷ naive server | 1.11x | 3.23x | 7.75x | – | – |
| TPOT, tiny-infer + kernels ÷ vLLM | 1.76x | 1.57x | 1.42x | 1.42x | 1.43x |

**Other comparisons:**
- **tiny-infer + kernels vs the naive server at 16 users:** 530.8 ÷ 33.1 = **16x** throughput. The naive server was only measured up to 16 users.
- **TTFT vs vLLM:** tiny-infer + kernels is lower at p50 *and* p99 at every load. Example at 64 users: p50 180 vs 232 ms, p99 2.63 vs 2.95 s.
- **vs offline HF static batching** (all requests submitted together, padded): tiny-infer without kernels is 1.11x faster at 16 and 1.23x at 32, while also streaming and admitting new requests every step.

### 9.2 Paged vs contiguous KV cache

Same 1 GiB pool (47,648 token slots), 256 requests submitted at once, kernels off.

| Allocation | Peak concurrent requests | Mean KV waste | Preemptions | Output tok/s |
|---|---:|---:|---:|---:|
| Contiguous (reserve 2,048 tokens each) | 23 | 79.7% | 0 | 314.9 |
| **Paged (16-token blocks on demand)** | **143** | **1.5%** | 21 | **381.5** |

- **6.2x** more concurrent requests and **1.21x** throughput.
- The closed-form prediction for contiguous allocation (23) matched exactly.
- The vLLM paper reports waste dropping from 60–80% to under 4%. This result is consistent with that.

### 9.3 Prefix caching (16 users, shared 1,024-token system prompt, 84% of prompt tokens hit the cache)

| | p50 TTFT | p99 TTFT | p99 ITL | Output tok/s |
|---|---:|---:|---:|---:|
| Off | 376 ms | 2,714 ms | 304 ms | 120.4 |
| **On** | **209 ms** | **707 ms** | **130 ms** | **170.5** |

Result: **3.8x** lower p99 TTFT, **1.8x** lower p50 TTFT, **1.42x** throughput.

### 9.4 Chunked prefill (16 users, prompts 64–1,900 tokens): a trade-off, reported honestly

| | p50 TTFT | p99 TTFT | p99 ITL | Output tok/s |
|---|---:|---:|---:|---:|
| No chunking | 363 ms | 2,673 ms | 303 ms | 123.8 |
| Chunked (512-token budget) | 691 ms | 4,036 ms | **169 ms** | 122.6 |

- Chunking cut the **p99 inter-token latency of running streams by 44%**, but **raised TTFT**: it protects users already streaming at the expense of new arrivals.
- Do **not** claim chunked prefill "improved latency" without this qualification.

### 9.5 Where a decode step goes (`torch.profiler`, offline engine, 512-token prompts)

| Batch 32 | Wall time | GPU busy | GPU idle | KV copy/gather | GEMM-class | Attention kernel | Roofline floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch ops | 79.8 ms | 66.5 ms | 13.3 ms | **32.0 ms** | 31.6 ms | – | 7.7 ms |
| **CUDA kernels** | **29.2 ms** | 26.6 ms | 2.5 ms | **0.5 ms** | 11.9 ms | 13.2 ms | 7.7 ms |

- **Half the GPU time was copying K/V.** The kernels cut it from 32.0 to 0.5 ms, and the step went from 79.8 to 29.2 ms.
- The step moved from 9.6% to 26% of the bandwidth roofline.
- At batch 64 with kernels: 43.5 ms wall, of which attention is 25.7 of 39.4 GPU-ms.

### 9.6 Kernel end-to-end (offline decode step, kernels off vs on, interleaved, 512-token prompts, 256 output tokens)

| Decode batch | PyTorch ops | CUDA kernels | Speedup | Greedy output identical |
|---:|---:|---:|---:|---:|
| 1 | 24.45 ms | 16.58 ms | 1.47x | 100% |
| 16 | 50.44 ms | 20.17 ms | 2.50x | 100% |
| 32 | 85.97 ms | 30.08 ms | 2.86x | 100% |
| 64 | 156.38 ms | 44.85 ms | **3.49x** | 100% |

The gain grows with batch because attention's share of the step grows with batch × context, while the weight-reading GEMMs, which the kernels don't touch, stay roughly constant. That is Amdahl's law.

---

## 10. Correctness: what is proven, and by which check

| # | Check | What it compares | Status |
|---|---|---|---|
| 1 | Test-suite oracle | tiny-infer vs HF `generate()`, token for token, on a small random Llama (GQA, RoPE, RMSNorm, SwiGLU), FP32, CPU | **Passed, 29/29 tests, green in CI** |
| 2 | Kernels vs PyTorch ops | tiny-infer with vs without the CUDA kernels on real TinyLlama, 256 greedy tokens at batch 1/16/32/64 | **Passed: 100% identical** |
| 3 | Kernel self-test | each kernel vs its PyTorch reference on random data | **Passed: worst error 2.0e-3** |
| 4 | Real-model oracle | tiny-infer vs HF `generate()` on real TinyLlama, 12 prompts | **Pending re-run** (see below) |

**What check 1 covers:** these configurations:
- batched and paged;
- unfused weights;
- no chunking;
- a 7-token budget (many chunks);
- contiguous allocation;
- one sequence at a time;
- 16-token blocks;
- forced recompute-preemption;
- prefix-cache hits;
- EOS handling.

Plus scheduler policy, block-manager invariants, HTTP/SSE streaming with concurrent clients, and regression tests.

**Check 4 history.**
- **First run: invalid.** The script told HF to suppress end-of-sequence (`min_new_tokens`) while the engine could still emit it, so 4 of 12 prompts "diverged" at identical positions in FP32, FP16 and FP16 + kernels.
- **Fix:** both sides now stop at EOS naturally, and a regression test reproduces the original failure.
- **Pending:** the fixed check has not been re-run on the GPU yet.

**The 29 tests:**
- 8 block manager;
- 11 HF-equivalence and correctness;
- 4 scheduler;
- 4 HTTP server and harness;
- 2 regression.

---

## 11. Bugs found and fixed (good interview material)

1. **Kernel self-test crash.** The self-test built block tables larger than its own block pool, so `--custom-kernels` crashed at startup on the first GPU run. Fixed, with a regression test.
2. **SSE parsing bug in the benchmark client.** `httpx.aiter_lines()` splits on Unicode line separators (U+2028, U+0085). vLLM emits these raw inside JSON strings, so 3 of 416 vLLM requests were misparsed. The fix splits only on `\n`, with a regression test. vLLM's numbers were not re-run; the effect is under 1%.
3. **Correctness-oracle EOS bug.** Described in section 10. It was diagnosed by noticing the divergences were *identical* across FP32, FP16 and kernels, which pointed to the harness, not the numerics. The cause was then reproduced on CPU.
4. **Benchmark unit bug.** A µs-vs-ms mix-up made the kernel benchmark report bandwidth 1000x too low. Found during review and fixed; the stored results were corrected from the raw timings.
5. **Accidental Vercel deployments.** The author's Vercel account auto-imported the repo and tried to build a GPU server as a website, failing on every push. Disabled with `vercel.json`.

---

## 12. Why tiny-infer is still slower than vLLM (gap analysis)

After the kernels, the remaining gap is entirely in decode (TPOT). Each cause, with its evidence:

1. **Attention kernel at 15% of bandwidth** *(measured)*. At batch 64, attention is 25.7 of 39.4 GPU-ms, at 48.9 GB/s of 320. Smaller split-K partitions are already measured to be 1.48–3.38x faster.
2. **Serving-stack overhead** *(gap measured, causes not split)*. The decode step takes 44.9 ms offline but TPOT is 61.8 ms through the HTTP server at 64 users. In tiny-infer, detokenisation and HTTP share one Python process with the engine. vLLM's log shows separate API-server and engine processes.
3. **Small-batch launch overhead** *(cause inferred, not isolated)*. The batch-1 decode step takes 16.6 ms against a 6.9 ms weight-read floor; vLLM's TPOT is 9.6 ms. vLLM captured CUDA graphs and used `torch.compile`, while tiny-infer launches every op eagerly. A batch-1 profile would confirm this.
4. **Unfused glue** *(measured, small)*. RoPE, SiLU·mul and the KV-cache writes are separate ops: 0.8 ms per batch-32 step.

**Next steps, in order:**
1. 128-token attention partitions;
2. CUDA graphs for decode;
3. separate the engine process from the server;
4. fused RoPE + cache write.

---

## 13. What NOT to claim (important)

| Don't say | Why | Say instead |
|---|---|---|
| "Deployed", "in production", "live website", "product" | It runs only while someone starts a GPU; there is no hosted service | "working prototype", "from-scratch inference server" |
| "Token-identical to HuggingFace on TinyLlama" | The real-model check (section 10, #4) is pending re-run | "verified token-identical to HuggingFace in a 29-test suite"; "kernels leave greedy output unchanged" |
| "Matches / beats vLLM" | It reaches 58–71% of vLLM's throughput | "58–71% of vLLM's throughput, with lower TTFT" |
| "~70% of vLLM" or "3x faster" | Inflated rounding | "58–71%", "2.75x" |
| "Kernels are bandwidth-optimal" | The attention kernel is at ~15% of peak bandwidth | "3.3–6.7x faster than the PyTorch path" |
| "Chunked prefill reduced latency" | It raised TTFT | "cut p99 inter-token latency 44% at the cost of higher TTFT" |
| "Multi-GPU", "distributed", "tensor parallel", "speculative decoding", "quantization", "CUDA graphs" | Not implemented | — |
| "16x faster than vLLM" or "16x than HuggingFace" | 16x is vs the naive one-at-a-time server | "16x a one-request-at-a-time server" |
| "6.2x throughput" | 6.2x is concurrent *requests*; throughput was 1.21x | "6.2x more concurrent requests in the same memory" |
| "Served real users" | It was benchmarked with a load generator | "benchmarked at up to 64 concurrent clients" |

---

## 14. Skills and keywords demonstrated

- **Languages / tools:** Python, CUDA C++, PyTorch (custom ops via `torch.library`, `torch.utils.cpp_extension`, `torch.profiler`), FastAPI, asyncio, server-sent events, HuggingFace Transformers, pytest, GitHub Actions, Docker, Google Colab.
- **LLM inference systems:** continuous batching, iteration-level scheduling, PagedAttention / paged KV cache, KV-cache memory math, prefix caching, chunked prefill, preemption, prefill vs decode, TTFT/TPOT/ITL, OpenAI-compatible API.
- **GPU programming:** memory-bound kernel design, coalesced vector loads, warp-shuffle reductions, shared memory, online softmax, Flash-Decoding / split-K, GQA-aware attention, register-pressure control (0 spills), roofline analysis.
- **Engineering practice:** benchmarking methodology, profiling-driven optimisation, correctness oracles, regression testing, honest comparison against an industry baseline (vLLM), root-causing measurement bugs.

---

## 15. Resume content

### Project header

**tiny-infer: LLM Inference Engine from Scratch** | Python, CUDA, PyTorch, FastAPI | github.com/Musharaf-khan/tiny-infer

### Recommended bullets (3)

- Built an LLM inference server from scratch with continuous batching, a paged KV cache and an OpenAI-compatible streaming API. On a single NVIDIA T4 it serves TinyLlama-1.1B at up to **939 tok/s**, **16x** a one-request-at-a-time server, and reaches **58–71% of vLLM's throughput** with lower time-to-first-token.
- Designed a paged KV-cache block manager with prefix sharing and recompute preemption. It fits **6.2x more concurrent requests** in the same GPU memory, cuts wasted cache memory from **79.7% to 1.5%**, and lowers p99 time-to-first-token **3.8x** with prefix caching.
- Profiled decoding and found **half of GPU time** spent copying the KV cache. Replaced it with custom CUDA kernels (GQA-aware paged decode attention with split-K, fused residual-add + RMSNorm) registered as PyTorch ops, cutting a decode step from **80 to 29 ms** and raising throughput **2.75x** with unchanged output.

### Short version (2 bullets)

- Built an LLM inference server from scratch (continuous batching, paged KV cache, custom CUDA kernels) reaching **58–71% of vLLM's throughput** on a single T4, **16x** a one-request-at-a-time server.
- Wrote CUDA paged-attention and fused RMSNorm kernels after profiling showed half of decode time was KV-cache copying, cutting a decode step from **80 to 29 ms** (**2.75x** throughput).

### One-line version

- Built a from-scratch LLM inference server with continuous batching, a paged KV cache and custom CUDA kernels, reaching **58–71% of vLLM's throughput** on a single T4 GPU.

### Variants by target role

**AI infrastructure / inference engineer:**
- Implemented iteration-level continuous batching with decode-first scheduling, chunked prefill and recompute preemption. It reached **7.75x** the throughput of a sequential server at 16 users without kernels and **16x** with them, keeping median time-to-first-token under 150 ms.
- Benchmarked against **vLLM 0.31.0** with a single harness and attributed the remaining gap with profiler data: attention kernel at 15% of bandwidth, single-process serving overhead, and no CUDA graphs.

**GPU / CUDA engineer:**
- Wrote a GQA-aware paged decode-attention CUDA kernel (Flash-Decoding split-K, warp-shuffle reductions, 0 register spills), **3.3–6.7x** faster than PyTorch's gather + SDPA path on a T4.
- Fused residual-add + RMSNorm into one FP16 kernel matching HuggingFace's rounding order, **10–13x** faster than the 9-op PyTorch sequence at decode sizes.

**General software engineer:**
- Built a 29-test correctness suite verifying token-identical output against HuggingFace across batching, paging, chunking, prefix caching and preemption, running in CI on every push.
- Root-caused three measurement bugs, including a correctness check that hid EOS and a Unicode SSE parsing bug that broke 3 of 416 benchmark requests, before publishing results.

### 30-second interview pitch

"I built tiny-infer, a small vLLM from scratch: continuous batching, a paged KV cache, prefix caching and a streaming OpenAI-style API. Then I plugged in CUDA kernels from my previous project. On a T4 with TinyLlama, paging alone fit 6.2 times more users in the same memory. Profiling showed half of every decode step was copying the KV cache, so I wrote a paged-attention kernel that reads the cache in place. That cut a decode step from 80 to 29 milliseconds and took me from about a third of vLLM's throughput to 58–71%. The rest of the gap is decode-side, and I can show you with profiler numbers where it goes."

---

## 16. Quick-reference facts table

| Fact | Value | Source file |
|---|---|---|
| GPU / model | Tesla T4 / TinyLlama-1.1B-Chat-v1.0 FP16 | `results/hf_baseline.json` |
| KV cache per token | 22,528 bytes | `results/kv_capacity.json` |
| Peak throughput | 938.9 tok/s (64 users, with kernels) | `results/serving.jsonl` |
| vs vLLM | 58% / 65% / 70% / 70% / 71% at 1 / 4 / 16 / 32 / 64 users | `results/serving.jsonl` |
| vs vLLM before kernels | 26–38% | `results/serving.jsonl` |
| vs naive server | 16x at 16 users (with kernels); 7.75x (without) | `results/serving.jsonl` |
| Kernel serving gain | 1.61x (1 user) to 2.75x (64 users) | `results/serving.jsonl` |
| Decode step, batch 32 | 79.8 → 29.2 ms | `results/step_profile_b32*.json` |
| KV copy time, batch 32 | 32.0 → 0.5 ms | `results/step_profile_b32*.json` |
| Offline decode-step speedup | 1.47x / 2.50x / 2.86x / 3.49x at batch 1 / 16 / 32 / 64 | `results/kernel_bench.json` |
| RMSNorm kernel | 10.3–13.4x | `results/kernel_bench.json` |
| Attention kernel | 3.3–6.7x; peak 48.9 GB/s (15% of T4 bandwidth) | `results/kernel_bench.json` |
| Paging | 23 → 143 concurrent (6.2x); waste 79.7% → 1.5%; 1.21x throughput | `results/kv_capacity.json` |
| Prefix caching | p99 TTFT 2,714 → 707 ms (3.8x); 84% hit rate | `results/serving.jsonl` |
| Chunked prefill | p99 ITL −44%; TTFT higher | `results/serving.jsonl` |
| Naive server collapse | p99 TTFT 77.0 s at 16 users; flat 33 tok/s | `results/serving.jsonl` |
| Tests | 29, all passing in CI | `tests/`, GitHub Actions |
| Kernel numerical error | worst 2.0e-3 (FP16) | `results/kernel_bench.json` |
| vLLM version | 0.31.0 (CUDA graphs, torch.compile, Triton attention) | `results/vllm_startup_evidence.txt` |
| Code size | ~1,500 lines engine Python, ~480 CUDA/C++, ~1,700 bench/tests | repository |

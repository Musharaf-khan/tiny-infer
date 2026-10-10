# tiny-infer

A from-scratch LLM inference server: an iteration-level **continuous batching** scheduler, a **paged KV cache** with prefix caching, an OpenAI-compatible **streaming** API, and my own **CUDA kernels** from [cuda-kernel-lab](https://github.com/Musharaf-khan/cuda-kernel-lab) in the decode path. It is benchmarked with one harness against HuggingFace `generate()` and against vLLM.

**The chain:** [cuda-kernel-lab](https://github.com/Musharaf-khan/cuda-kernel-lab) produced kernels that beat PyTorch ops in isolation. tiny-infer puts them inside a serving engine and measures what they change end to end, in tokens per second.

**Headline (Tesla T4):**

| Comparison | Result |
|---|---|
| vs a one-request-at-a-time server, 16 users | **16x the throughput** |
| Paging vs contiguous KV allocation | **6.2x more concurrent users** in the same memory |
| Project 1 kernels in the engine | **2.75x faster** at 64 users |
| vs vLLM | **58–71% of vLLM's throughput**, with lower median TTFT |

**Model:** TinyLlama-1.1B-Chat, FP16. **GPU:** one NVIDIA T4 (Turing, sm_75, 320 GB/s, 15 GiB). **Code:** about 1,500 lines of engine Python and 480 lines of CUDA/C++, plus 1,700 lines of benchmarks, scripts and tests. **Correctness:** see [the correctness table](#correctness-what-is-proven-and-by-which-check). The test suite proves token-identical output to HuggingFace on a small random Llama, and on the T4 the CUDA kernels leave real TinyLlama's output unchanged. The real-model check against HuggingFace is pending a re-run.

**What it is:** a working prototype of a single-GPU LLM inference server: an engine plus an HTTP API, run on a GPU you start yourself. It is not a hosted service or a website.

---

## Results (Tesla T4, FP16, measured)

All numbers come from [`notebooks/t4_runbook.ipynb`](notebooks/t4_runbook.ipynb) on a Colab T4. Raw data and the auto-generated report are in [`results/`](results/REPORT.md).

**Workload:** prompts of 128–512 tokens and outputs of 64–256 tokens. Prompts are sent as token IDs with `ignore_eos`, so every system generates the same tokens. The client runs a closed loop at fixed concurrency and times everything over HTTP/SSE.

### Serving comparison (same workload, same client, same GPU)

| System | Concurrency | Output tok/s | p50 TTFT | p99 TTFT | p50 TPOT | GPU util |
|---|---:|---:|---:|---:|---:|---:|
| HF `generate()` in-process (baseline) | 1 | 35.6 | 76 ms | – | 28.1 ms | – |
| HF static batching (padded, offline) | 16 / 32 | 230.5 / 253.7 | – | – | – | – |
| Naive server (one `generate()` at a time) | 1 | 32.9 | 82 ms | 112 ms | 29.0 ms | 47% |
| | 4 | 32.9 | **15.5 s** | 18.3 s | 29.8 ms | 47% |
| | 16 | 33.1 | **36.7 s** | **77.0 s** | 29.0 ms | 47% |
| **tiny-infer** (continuous batching + paged KV) | 1 | 36.4 | 43 ms | 54 ms | 26.5 ms | 47% |
| | 4 | 106.3 | 92 ms | 213 ms | 32.7 ms | 58% |
| | 16 | 256.6 | 144 ms | 837 ms | 55.2 ms | 77% |
| | 32 | 311.2 | 234 ms | 1.58 s | 94.7 ms | 83% |
| | 64 | 340.9 | 420 ms | 3.27 s | 178.8 ms | 87% |
| **tiny-infer + Project 1 kernels** | 1 | **58.6** | 40 ms | 49 ms | 16.9 ms | 86% |
| | 4 | **178.4** | 65 ms | 162 ms | 19.3 ms | 84% |
| | 16 | **530.8** | 82 ms | 691 ms | 25.6 ms | 77% |
| | 32 | **714.9** | 109 ms | 1.30 s | 38.4 ms | 74% |
| | 64 | **938.9** | 180 ms | 2.63 s | 61.8 ms | 70% |
| **vLLM 0.31.0** (`--dtype half`, prefix caching off) | 1 | 100.3 | 69 ms | 92 ms | 9.6 ms | 99% |
| | 4 | 276.0 | 92 ms | 216 ms | 12.3 ms | 98% |
| | 16 | 759.2 | 104 ms | 825 ms | 18.0 ms | 99% |
| | 32 | 1,018.9 | 135 ms | 1.46 s | 27.1 ms | 99% |
| | 64 | 1,331.1 | 232 ms | 2.95 s | 43.1 ms | 99% |

**What the table shows:**

- **Continuous batching vs the naive server.**
  - The naive server's throughput is flat at about 33 tok/s whatever the load. Its TTFT grows with queue depth instead, reaching a p99 of 77 s at 16 users. That is the Phase 2 collapse.
  - tiny-infer serves the same load at **3.2x the naive server's throughput at 4 users and 7.75x at 16 users**, with p50 TTFT staying under 150 ms.
  - Against an idealised HF static batch, where all requests arrive together and run offline, tiny-infer is **1.11x faster at 16 users and 1.23x at 32**. It also streams, and admits new requests every step instead of every batch.
- **The Project 1 kernels made the engine 1.6x faster at 1 user and 2.75x faster at 64 users.** They removed the K/V copying that the profiler showed was eating half of every decode step.
- **vs vLLM: tiny-infer with kernels reaches 58–71% of vLLM's throughput** (58% at 1 user, 70% at 16, 71% at 64). Without the kernels it reached 26–38%.
  - TTFT p50 is *lower* than vLLM's at every load measured (e.g. 82 ms vs 104 ms at 16 users, 180 ms vs 232 ms at 64).
  - What remains of the gap is decode (TPOT 61.8 vs 43.1 ms at 64 users). [docs/vllm_gap.md](docs/vllm_gap.md) attributes it with profiler numbers.

![Throughput, TTFT and TPOT vs concurrency on a T4](results/serving.png)

### Paged vs contiguous KV cache (1 GiB pool, same kernels, 256 requests submitted at once)

| Allocation | Peak concurrent sequences | Mean KV waste | Preemptions | Output tok/s |
|---|---:|---:|---:|---:|
| contiguous (reserve `max_model_len` = 2,048) | 23 | 79.7% | 0 | 314.9 |
| **paged** (16-token blocks, on demand) | **143** | **1.5%** | 21 | **381.5** |

**Paging fit 6.2x more sequences in the same memory and cut KV waste from 79.7% to 1.5%.** The vLLM paper reports going from 60–80% waste to under 4%.

The closed form predicted 23 sequences for contiguous allocation, which matched exactly. For paged allocation it predicted about 96 at the mean *final* length. Paged peaked higher, at 143, because sequences early in their output are still short. That optimism is why paged mode needed 21 recompute preemptions; contiguous mode, which reserves the maximum up front, needed none.

### Differentiators: prefix caching and chunked prefill (16 users)

| Run | p50 TTFT | p99 TTFT | p99 ITL | Output tok/s |
|---|---:|---:|---:|---:|
| 1,024-token shared system prompt, prefix caching **off** | 376 ms | 2,714 ms | 304 ms | 120.4 |
| same, prefix caching **on** (84% of prompt tokens hit) | **209 ms** | **707 ms** | **130 ms** | **170.5** |
| prompts of 64–1,900 tokens, no chunking | 363 ms | 2,673 ms | 303 ms | 123.8 |
| same, chunked prefill (512-token budget) | 691 ms | 4,036 ms | **169 ms** | 122.6 |

- **Prefix caching:** 3.8x lower p99 TTFT, 1.8x lower p50 TTFT, and 1.42x the throughput.
- **Chunked prefill is a trade-off.** It cut the p99 inter-token latency of running streams by 44%, because no decode waits behind a 1,900-token prefill. But it raised TTFT, because a 512-token budget spreads every long prompt over several steps. In other words, it protects users who are already streaming at the expense of newcomers. That is the reason vLLM tunes the budget instead of fixing it.

### Where a decode step goes (batch 32, `torch.profiler`)

| | Wall time | GPU busy | GPU idle (CPU-side) | Copy / gather | GEMM-class | Attention kernel | Roofline floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch ops | 79.8 ms | 66.5 ms | 13.3 ms | **32.0 ms** | 31.6 ms | – | 7.7 ms |
| **Project 1 kernels** | **29.2 ms** | 26.6 ms | 2.5 ms | **0.5 ms** | 11.9 ms | 13.2 ms | 7.7 ms |

**Without the kernels, half the GPU time was copying K/V.** The PyTorch decode path gathers each sequence's K/V out of the block pool and `repeat_interleave`s it 8x for GQA before SDPA. SDPA then runs its math path on sm_75 (the GEMM-class bucket includes its batched matmuls). The paged-attention kernel reads K/V once through the block table and shares each row across the 8 query heads of a KV head. That leaves the decode step 2.7x shorter, at 26% of the bandwidth roofline (up from 9.6%).

### Project 1 kernels in the decode path

| Kernel | Replaces | Op speedup (T4) | End-to-end |
|---|---|---:|---|
| fused residual-add + RMSNorm (FP16) | `x + residual` then HF `LlamaRMSNorm` (9 separate PyTorch ops in the reference code) | **10.3–13.4x** (133 → 13 µs at batch 32) | – |
| GQA paged decode attention (Flash-Decoding split-K) | gather-from-blocks + `repeat_interleave` + SDPA | **3.3–6.7x** (4.59 → 0.69 ms at 32 seqs × 1,024 ctx) | – |
| **both, in the engine** | | | decode step **1.47x / 2.50x / 2.86x / 3.49x** faster at batch 1 / 16 / 32 / 64; greedy output identical with and without kernels at every batch size |

- **The RMSNorm speedup is launch overhead, not bandwidth.** At decode shapes (batch 1–32) a call moves only 20–516 KB, and one launch replaces 9 separate PyTorch ops.
- **The attention speedup grows with batch × context,** because that's what the gather path copies. Its end-to-end effect therefore grows with load.
- **The attention kernel is now the largest item at batch 64** (25.7 of 39.4 GPU-ms). It reaches 49 GB/s, which is 15% of the T4's 320 GB/s. 128-token partitions beat 512-token ones in every measured shape (1.4–3.4x), but the engine still uses 512 because 128 hasn't been measured end to end yet. That is the next step.

Write-up: [docs/kernels.md](docs/kernels.md). Gap to vLLM, with a profiler breakdown: [docs/vllm_gap.md](docs/vllm_gap.md).

---

## Architecture

```mermaid
flowchart LR
    C[HTTP clients<br/>OpenAI /v1/completions<br/>SSE streaming] --> S[FastAPI server<br/>detokenize per token]
    S -- add / abort --> AE[AsyncEngine<br/>background thread]
    AE --> SCH
    subgraph Engine step loop
        SCH[Scheduler<br/>decode first, then chunked prefill,<br/>then FCFS admission; preempt newest] --> BM[Block manager<br/>free list, ref counts,<br/>block tables, prefix-cache LRU]
        SCH --> MR[Model runner<br/>pack all new tokens into one batch]
        MR --> M[Llama forward<br/>one GEMM per layer for the whole batch]
        M --> K[Kernels<br/>fused add+RMSNorm<br/>paged decode attention]
        M --> KV[(KV block pool<br/>layers x blocks x 16 x kv_heads x 64)]
        K --> KV
        M --> SM[Sampler]
    end
    SM -- tokens --> AE -- per-request queues --> S
```

**One rule drives everything.** Each sequence has `token_ids` (prompt plus generated tokens) and `num_computed_tokens` (how many of those already have K/V in the cache). Every step processes the uncomputed suffix, up to the token budget. When a step reaches the end of `token_ids`, one token is sampled. Each feature is a special case of that rule:

| Feature | In terms of the rule |
|---|---|
| decode | 1 new token |
| prefill | the whole prompt |
| chunked prefill | several partial steps |
| prefix-cache hit | `num_computed_tokens` starts above 0 |
| preemption | `num_computed_tokens` is reset to 0 and the sequence is recomputed |

They all share one code path, and the test-suite oracle (check 1 below) covers each of them.

```text
loop forever:                                       # tiny_infer/scheduler.py
    decode:  for each running seq (oldest first): reserve a slot for its next token;
             if the pool is full, preempt the NEWEST seq (free its blocks, requeue it at the front)
    prefill: continue partially-prefilled seqs with the remaining token budget
    admit:   while the pool can hold the next waiting prompt (minus prefix hits): admit it (FCFS)
    run ONE forward pass over all scheduled tokens; sample; append; retire finished seqs at once
```

Design notes, with every decision and the reason for it: [docs/design.md](docs/design.md).

## Quickstart

**Colab (T4, no setup):** open [`notebooks/t4_runbook.ipynb`](notebooks/t4_runbook.ipynb) in Colab, pick the T4 runtime, then choose Run all.

**Any CUDA machine:**

```bash
git clone https://github.com/Musharaf-khan/tiny-infer && cd tiny-infer
pip install -r requirements.txt                      # torch with CUDA assumed
python -m tiny_infer.server --custom-kernels         # :8000, OpenAI-compatible
curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "<|user|>\nWhat is a KV cache?</s>\n<|assistant|>\n", "max_tokens": 128, "stream": true}'
python scripts/demo.py --users 8                     # live multi-user streaming view
```

**Docker:** `docker build -t tiny-infer . && docker run --gpus all -p 8000:8000 tiny-infer`

**Offline API:**

```python
from tiny_infer import EngineConfig, LLMEngine, SamplingParams
eng = LLMEngine(EngineConfig(use_custom_kernels=True))
for s in eng.generate(["The capital of France is"] * 4, SamplingParams(max_tokens=32, temperature=0)):
    print(eng.tokenizer.decode(s.output_ids))
```

Useful server flags:

| Flag | Effect |
|---|---|
| `--kv-mode contiguous` | the "before" allocator |
| `--kv-cache-gb 1` | fixed KV pool |
| `--prefix-caching` | turn on prefix caching |
| `--no-chunked-prefill` | turn off chunked prefill |
| `--max-num-batched-tokens 512` | set the per-step token budget |
| `--backend naive` | run the Phase 2 server |

## Benchmarks: how each number is produced

| Script | Measures |
|---|---|
| `bench/harness.py` | Any OpenAI endpoint: TTFT, TPOT, ITL, E2E (p50 and p99), output and input tok/s reported separately, req/s, error rate, GPU utilisation. Prompts are sent as token IDs with `ignore_eos`, so every system does identical work. |
| `bench/hf_baseline.py` | HF `generate()` at concurrency 1 (TTFT, TPOT, peak VRAM), plus padded static batching |
| `bench/kv_capacity.py` | paged vs contiguous on the same fixed pool: peak concurrency, KV waste, preemptions, throughput |
| `bench/kernel_bench.py` | each kernel vs the PyTorch path it replaces (µs, GB/s against 320 GB/s), plus end-to-end decode-step delta |
| `bench/step_profile.py` | torch.profiler breakdown of one decode step: GEMM, attention, norm, copies, idle GPU (CPU overhead), and the roofline floor |
| `scripts/check_correctness.py` | real-model oracle against HF: fp32 must be identical; in fp16 every divergence is reported with its top-2 logit gap |
| `bench/make_report.py` | `results/*.json` → `results/REPORT.md` and charts |

## Correctness: what is proven, and by which check

Four separate checks, each proving something different. Only the first three have passed so far.

| Check | What it compares | Where it runs | Status |
|---|---|---|---|
| **1. Test-suite oracle** | tiny-infer vs HF `generate()`, token for token, on a small random Llama (GQA, RoPE, RMSNorm, SwiGLU) in FP32 | CPU, `pytest`, GitHub Actions on every push | **Passed: 29/29 tests** |
| **2. Kernels vs PyTorch ops** | tiny-infer *with* the CUDA kernels vs tiny-infer *without* them, on real TinyLlama: 256 greedy tokens at batch 1, 16, 32 and 64 | T4 (`results/kernel_bench.json`) | **Passed: 100% of sequences identical** |
| **3. Kernel self-test** | each CUDA kernel vs its PyTorch reference on random tensors (ragged lengths, shuffled blocks, GQA 8 and 1) | T4, on every engine start with `--custom-kernels` | **Passed: worst max abs error 2.0e-3** (FP16) |
| **4. Real-model oracle** | tiny-infer vs HF `generate()` on real TinyLlama: 12 prompts, FP32 / FP16 / FP16 + kernels | T4, `scripts/check_correctness.py` | **Not yet established** (see below) |

**What this does and does not prove.**
- Check 1 proves the engine's logic matches HF on every code path:
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
- Check 2 proves the CUDA kernels don't change the real model's output.
- Neither check alone proves that tiny-infer reproduces HF on the *real* model's weights. That is check 4's job.

**Check 4 history.**
- **First run: invalid.** The script passed `min_new_tokens` to HF, which suppresses EOS, while the engine could still pick it. As a result, 4 of 12 prompts "diverged" at identical positions in all three modes.
- **Fix:** both sides now stop at EOS naturally. A CPU regression test (`test_eos_handling_matches_hf`) reproduces the failure and passes with the fix.
- **Pending:** the fixed script has **not yet been re-run on the T4**. Until it is, this README makes no claim that tiny-infer matches HF token for token on real TinyLlama.

Run the tests:

```bash
python -m pytest tests -q      # 29 tests, CPU, ~15 s
```

The other tests cover:
- block-manager invariants;
- scheduler policy;
- real HTTP/SSE streaming with concurrent clients;
- the SSE parser and the kernel self-test construction.

The engine refuses to serve if any kernel fails its startup self-test.

## Honest scope

This is a single-GPU engine with Python-level scheduling.

**Not included:**
- tensor or pipeline parallelism
- disaggregated prefill/decode
- CUDA graphs
- speculative decoding
- quantisation
- production hardening such as autoscaling or fault tolerance

**Not supported:**
- copy-on-write forking, because nothing forks. Only full blocks are shared, so a shared block is never written. CoW would be needed for parallel sampling and beam search; it's explained in [docs/design.md](docs/design.md).

**The vLLM gap is the expected result, not a surprise.** The reasons are measured in [docs/vllm_gap.md](docs/vllm_gap.md).

## Layout

```text
tiny_infer/
  config.py          ModelConfig (+ KV math), EngineConfig
  sequence.py        Sequence, SamplingParams: the token_ids / num_computed_tokens rule
  block_manager.py   paged + contiguous allocation, ref counts, prefix cache (hash chain + LRU)
  scheduler.py       iteration-level scheduler: decode-first, chunked prefill, preemption
  model.py           Llama forward over a packed batch; paged KV writes; prefill/decode attention split
  ops.py             reference ops (HF-exact) + dispatch to the CUDA custom ops
  kernels.py         JIT build + self-test of the CUDA extension
  csrc/              rms_norm.cu, paged_attention.cu, reduce.cuh, ops.cpp (TORCH_LIBRARY registration)
  engine.py          LLMEngine.step(): schedule → prepare → forward → sample → retire
  async_engine.py    engine loop thread ↔ asyncio streams
  server.py          OpenAI /v1/completions (+SSE), naive HF backend for Phase 2
bench/  scripts/  tests/  notebooks/t4_runbook.ipynb  docs/
```

## References

The design draws on the following work:

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP '23)
- Yu et al., *Orca* (OSDI '22), which introduced iteration-level scheduling
- Dao et al., *Flash-Decoding*

Structure was also cross-checked against several open-source engines (code was read and compared, not copied):

- nano-vllm
- ember
- neel429/llm-inference-engine

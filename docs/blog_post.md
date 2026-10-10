# I built a tiny vLLM from scratch (and put my own CUDA kernels in it)

*All numbers are from a Tesla T4. Raw data is in the repo's `results/` folder.*

In my last post I wrote 25 CUDA kernels on a free T4 and got matrix multiply to cuBLAS parity. Kernels that beat PyTorch in a microbenchmark are nice. But the question that kept bugging me was simpler: **does any of it show up when a real user asks a real model a question?**

To answer it I needed a real serving engine to put the kernels in. So I wrote one: tiny-infer, about 1,500 lines of Python and 480 of CUDA. It serves TinyLlama-1.1B over an OpenAI-compatible streaming API.

## An LLM request is two jobs in a trench coat

When you send a prompt, the model does two very different things.

- **Prefill** reads your whole prompt in one big parallel pass and builds the KV cache. It's a fat matrix multiply and keeps the GPU's math units busy. It decides how long you wait for the first token.
- **Decode** produces one token at a time. Each token needs a pass over *all* the weights, which is 2.2 GB for TinyLlama, to do very little math. A T4 moves 320 GB/s, so one decode step can't take less than about 7 ms, however many FLOPs the GPU has.

That second fact is the whole game. **Reading the weights costs the same whether one user or sixty-four users share the step.** Serving them together is nearly free, up to a point. Serving them one at a time wastes almost the entire GPU.

## Step 1: the naive server (built to lose)

One request, one `generate()` call, one lock. At 1 user it's fine. At 16 users the 16th person waits for 15 other complete answers before getting a first token: p99 time-to-first-token went from 112 ms at 1 user to 77 seconds at 16, while throughput stayed flat at 33 tok/s.

## Step 2: continuous batching

The fix sounds obvious: put everyone in the same batch. The obvious version, static batching, has a catch. The batch only ends when its **longest** member finishes, so everyone who finished early leaves a seat empty. Newcomers wait for the whole batch.

Continuous batching (from the Orca paper) rebuilds the batch **every token**:

```text
every step:
  give each running user 1 token of work      (decode first: keep streams smooth)
  continue any half-done long prompts          (chunked prefill)
  admit new users while memory allows          (first come, first served)
  ONE forward pass for everybody → one new token each
  finished users leave now; their seats are refilled next step
```

All of a step's tokens go into one packed tensor. Every weight matrix is read once and multiplied against everyone's tokens. Result at 16 users: **7.75x** the throughput of the naive server, with median time-to-first-token at 144 ms instead of 37 seconds.

## Step 3: the memory problem, and paging

Each token a user has seen costs 22 KiB of KV cache: 2 × 22 layers × 4 KV heads × 64 dims × 2 bytes. The simple design reserves room for the *maximum* length up front, 2,048 tokens or 44 MiB per user. Most answers are far shorter, so most of that reservation sits empty, and you run out of memory while memory is free.

PagedAttention borrows the operating-system trick: cut the cache into 16-token **blocks**, give each user a **block table** (logical block → physical block), and allocate only when the current block fills. Waste drops to under one block per user.

Same GPU, same 1 GiB pool, same kernels, only the allocator changed:

| | Users at once | KV waste |
|---|---:|---:|
| reserve max length | 23 | 79.7% |
| paged | 143 | 1.5% |

Paging also made two more features cheap:
- **Prefix caching:** identical system prompts share physical blocks.
- **Preemption:** when memory runs out mid-answer, the newest request gives its blocks back and is recomputed later.

## Step 4: my kernels, in the hot path

Two kernels from the last project went into decode, adapted to it:

1. **Fused residual-add + RMSNorm.** It replaces 9 separate PyTorch ops with one kernel launch, 45 times per token (twice per layer, plus the final norm).
2. **A paged decode-attention kernel.** My old attention kernel was built for prefill, with many query rows sharing each K/V tile. Decode has *one* query row, so there's nothing to share, and the job turned into pure streaming. Three things mattered:
   - reading K/V straight through the block table;
   - letting all 8 query heads that share a KV head (grouped-query attention) use each K/V row loaded once;
   - splitting long contexts across thread blocks (Flash-Decoding) so 40 SMs have work even at batch 1.

Before writing them, I profiled one decode step at 32 users: 80 ms, and **32 ms of it was copying**. My PyTorch fallback gathered every sequence's K/V out of the blocks and repeated it 8 times so the attention call saw one K/V per query head. The kernel reads each K/V row once, straight through the block table, and shares it across the 8 heads.

| | PyTorch ops | My kernels |
|---|---:|---:|
| decode step at 32 users | 80 ms | **29 ms** |
| copying K/V | 32 ms | 0.5 ms |
| output tok/s at 64 users | 341 | **939** |

At the op level, the RMSNorm kernel is 10x faster, all of it launch overhead. One launch replaces nine ops, and at decode sizes the data is tiny. The attention kernel is up to 6.7x faster. Output tokens were identical with and without the kernels at every batch size.

The end-to-end gain is smaller than the op-level gain, and Amdahl's law says why: about half of the remaining step is the weight-reading GEMMs, which my kernels don't touch.

## Step 5: losing to vLLM, honestly

Before the kernels, tiny-infer ran at 26–38% of vLLM's throughput. After them, **58–71%**, and its time-to-first-token is actually lower than vLLM's. The remaining gap is all decode, and the profiler gives three reasons:

1. **Overhead at small batch.** At batch 1 my decode step is 16.6 ms against a 6.9 ms weight-read floor. vLLM's log shows it captured CUDA graphs. My engine launches every op eagerly from Python, so that's the likely cause; a batch-1 profile would prove it.
2. **My attention kernel still uses only 15% of the T4's bandwidth.** Smaller split-K partitions are already measured to be faster.
3. **Everything runs in one Python process.** The decode step takes 45 ms offline but 62 ms behind the HTTP server at 64 users.

The full breakdown is in the repo.

## What I'd tell myself at the start

- **Measure before you build.** The benchmark harness came before the engine. Every claim has a script.
- **Have a correctness oracle from day one.** Every feature, including preemption and recompute, must reproduce HuggingFace's tokens exactly. It caught bugs that throughput numbers never would.
- **Decode is a bandwidth problem.** Almost every serving trick (batching, paging, GQA, quantised KV) is a way to read fewer bytes per useful token.

Code, benchmarks and the T4 notebook that produces every number: github.com/Musharaf-khan/tiny-infer

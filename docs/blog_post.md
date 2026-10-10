# I built a tiny vLLM from scratch (and put my own CUDA kernels in it)

*Draft. Replace every `[X]` with a measured number from `results/REPORT.md` before publishing.*

In my last post I wrote 25 CUDA kernels on a free T4 and got matrix multiply to cuBLAS parity. Kernels that beat PyTorch in a microbenchmark are nice. But the question that kept bugging me was simpler: **does any of it show up when a real user asks a real model a question?**

To answer it I needed a real serving engine to put the kernels in. So I wrote one: tiny-infer, about 1,500 lines of Python and 480 of CUDA. It serves TinyLlama-1.1B over an OpenAI-compatible streaming API.

## An LLM request is two jobs in a trench coat

When you send a prompt, the model does two very different things.

- **Prefill** reads your whole prompt in one big parallel pass and builds the KV cache. It's a fat matrix multiply and keeps the GPU's math units busy. It decides how long you wait for the first token.
- **Decode** produces one token at a time. Each token needs a pass over *all* the weights, which is 2.2 GB for TinyLlama, to do very little math. A T4 moves 320 GB/s, so one decode step can't take less than about 7 ms, however many FLOPs the GPU has.

That second fact is the whole game. **Reading the weights costs the same whether one user or sixty-four users share the step.** Serving them together is nearly free, up to a point. Serving them one at a time wastes almost the entire GPU.

## Step 1: the naive server (built to lose)

One request, one `generate()` call, one lock. At 1 user it's fine. At 16 users the 16th person waits for 15 other complete answers before getting a first token: p99 TTFT went from [X] ms to [X] s.

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

All of a step's tokens go into one packed tensor. Every weight matrix is read once and multiplied against everyone's tokens. Result at 32 users: **[X]x** the throughput of the naive server.

## Step 3: the memory problem, and paging

Each token a user has seen costs 22 KiB of KV cache: 2 × 22 layers × 4 KV heads × 64 dims × 2 bytes. The simple design reserves room for the *maximum* length up front, 2,048 tokens or 44 MiB per user. Most answers are far shorter, so most of that reservation sits empty, and you run out of memory while memory is free.

PagedAttention borrows the operating-system trick: cut the cache into 16-token **blocks**, give each user a **block table** (logical block → physical block), and allocate only when the current block fills. Waste drops to under one block per user.

Same GPU, same 1 GiB pool, same kernels, only the allocator changed:

| | Users at once | KV waste |
|---|---:|---:|
| reserve max length | [X] | [X]% |
| paged | [X] | [X]% |

Paging also made two more features cheap:
- **Prefix caching:** identical system prompts share physical blocks.
- **Preemption:** when memory runs out mid-answer, the newest request gives its blocks back and is recomputed later.

## Step 4: my kernels, in the hot path

Two kernels from the last project went into decode, adapted to it:

1. **Fused residual-add + RMSNorm.** It replaces a chain of about 9 tiny PyTorch kernels with 1, run 45 times per token.
2. **A paged decode-attention kernel.** My old attention kernel was built for prefill, with many query rows sharing each K/V tile. Decode has *one* query row, so there's nothing to share, and the job turned into pure streaming. Three things mattered:
   - reading K/V straight through the block table;
   - letting all 8 query heads that share a KV head (grouped-query attention) use each K/V row loaded once;
   - splitting long contexts across thread blocks (Flash-Decoding) so 40 SMs have work even at batch 1.

Op level: [X]x and [X]x. End to end: decode steps got **[X]%** faster. That's smaller than the op-level number, and Amdahl's law says why. Most of a decode step is the weight-reading GEMMs, which my kernels don't touch.

## Step 5: losing to vLLM, honestly

vLLM is [X]x faster than tiny-infer at [X] users. The profiler says where it goes.

At small batch my engine is **launch-bound**: about 300 small kernel launches per token, and the GPU idles [X] ms per step waiting on Python. vLLM records the whole decode step as a CUDA graph and replays it with one launch. It also fuses more (RoPE with the cache write, SiLU with the multiply) and overlaps CPU work with GPU work. The full breakdown, and what would close the gap, is in the repo.

## What I'd tell myself at the start

- **Measure before you build.** The benchmark harness came before the engine. Every claim has a script.
- **Have a correctness oracle from day one.** Every feature, including preemption and recompute, must reproduce HuggingFace's tokens exactly. It caught bugs that throughput numbers never would.
- **Decode is a bandwidth problem.** Almost every serving trick (batching, paging, GQA, quantised KV) is a way to read fewer bytes per useful token.

Code, benchmarks and the T4 notebook that produces every number: github.com/Musharaf-khan/tiny-infer

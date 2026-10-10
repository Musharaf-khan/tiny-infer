# Launch kit: resume bullets, LinkedIn post, demo recording

Every number below comes from `results/` (Tesla T4 runs).

## Resume bullets (measured on a Tesla T4; pick 2-3)

- Built **tiny-infer**, a from-scratch LLM inference server with an iteration-level continuous-batching scheduler, a paged KV cache with prefix caching, and an OpenAI-compatible streaming API. On a single T4 it serves TinyLlama-1.1B at **up to 939 tok/s** (64 users), **16x a sequential HF server** at 16 users, and reaches **58–71% of vLLM's throughput**, with lower median time-to-first-token.
- Designed a paged KV block manager (16-token blocks, ref-counted prefix sharing, LRU eviction, recompute preemption). It fit **6.2x more concurrent sequences** than contiguous allocation in the same 1 GiB pool, cut KV waste from **79.7% to 1.5%**, and prefix caching lowered **p99 TTFT 3.8x** on shared-system-prompt traffic.
- Profiled the decode step, found **half of GPU time going to K/V gather/copy**, and replaced it with my own CUDA kernels (GQA-aware paged decode attention with Flash-Decoding split-K, and fused residual-add + RMSNorm) registered as PyTorch custom ops. That cut a batch-32 decode step from **80 to 29 ms**, raised serving throughput **2.75x at 64 users**, and left greedy output unchanged.

## LinkedIn post

> I built a tiny vLLM from scratch, and then put my own CUDA kernels inside it.
>
> Last project, I wrote 25 CUDA kernels and got SGEMM to cuBLAS parity on a T4. The question I couldn't answer was: does any of it matter once there's a real serving system around it?
>
> So I built tiny-infer: continuous batching, a paged KV cache, prefix caching, and a streaming OpenAI-compatible API, all from scratch in Python + CUDA.
>
> On one T4 with TinyLlama-1.1B:
> • 16x the throughput of a one-request-at-a-time server at 16 users
> • 6.2x more users in the same memory, just from paging (KV waste 80% → 1.5%)
> • the profiler showed half of every decode step was copying the KV cache, so I wrote a paged-attention kernel to kill it: 80 ms → 29 ms per step, 2.75x throughput at 64 users
> • vLLM is still faster: I'm at 58–71% of it. The README breaks down exactly where the rest goes (CUDA graphs, an attention kernel at 15% of bandwidth, single-process serving).
>
> The best lesson: decode is a memory-bandwidth problem. Batching works because the weights are read once per step whether you serve 1 user or 64. And my biggest speedup came from deleting a copy, not from clever math.
>
> Code + benchmarks: github.com/Musharaf-khan/tiny-infer
> #CUDA #LLM #inference #vLLM #GPU

## Demo recording (60-90 s)

1. Colab or a GPU box: `python -m tiny_infer.server --custom-kernels &`
2. `python scripts/demo.py --users 8 --max-tokens 200`: 8 users streaming at once, with live per-user TTFT and tok/s and the system total at the top.
3. Then run `--users 1` to show the per-user rate barely drops while the total goes up about 8x. That is the batching story in one clip.
4. Record with OBS or QuickTime and convert to a GIF for the README (`ffmpeg -i demo.mov -vf "fps=12,scale=900:-1" demo.gif`).

## Blog post

Draft: [blog_post.md](blog_post.md). Publish it on dev.to next to the Project 1 post and link the two posts to each other.

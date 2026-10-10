# Launch kit: resume bullets, LinkedIn post, demo recording

Replace every `[X]` with a number from `results/REPORT.md`. Do not ship a bullet whose number you haven't measured.

## Resume bullets (pick 2-3)

- Built **tiny-infer**, a from-scratch LLM inference server with iteration-level continuous batching, a paged KV cache with prefix caching, and an OpenAI-compatible streaming API. It serves TinyLlama-1.1B at **[X] tok/s** on a single T4, **[X]x** a sequential HF `generate()` server at 32 concurrent users, with greedy output verified token-identical to HuggingFace.
- Designed a paged KV block manager (16-token blocks, ref-counted prefix sharing, LRU eviction, recompute preemption). It ran **[X]x** more concurrent sequences than contiguous allocation in the same 1 GiB pool, cutting KV waste from **[X]%** to **[X]%**.
- Integrated my own CUDA kernels (fused residual-add + RMSNorm; a GQA-aware paged decode-attention kernel with Flash-Decoding split-K) into the decode path as PyTorch custom ops, for a **[X]%** decode-step (TPOT) reduction at batch 32. Benchmarked against vLLM (**[X]%** of its throughput) and attributed the gap with a profiler breakdown (CUDA graphs, fusion depth, CPU/GPU overlap).

## LinkedIn post

> I built a tiny vLLM from scratch, and then put my own CUDA kernels inside it.
>
> Last project, I wrote 25 CUDA kernels and got SGEMM to cuBLAS parity on a T4. The question I couldn't answer was: does any of it matter once there's a real serving system around it?
>
> So I built tiny-infer: continuous batching, a paged KV cache, prefix caching, and a streaming OpenAI-compatible API. My fused RMSNorm and a new paged decode-attention kernel run in the decode path.
>
> On one T4 with TinyLlama-1.1B:
> • [X]x the throughput of a one-request-at-a-time server at 32 users
> • [X]x more concurrent users from paging alone (same memory)
> • my kernels: [X]% faster per decode step
> • vLLM is still [X]x faster than me, and the README explains exactly why (CUDA graphs, launch overhead, fusion), with a profiler breakdown
>
> Every output is checked token-for-token against HuggingFace.
>
> The best lesson: decode is a memory-bandwidth problem. Batching works because the weights are read once per step whether you serve 1 user or 64.
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

# tiny-infer results

Hardware: Tesla T4. Model: TinyLlama-1.1B-Chat-v1.0, FP16. Workload: prompts of 128-512 tokens, outputs of 64-256 tokens (ignore_eos, so every system generates the same tokens), closed loop at fixed concurrency, client-side timing over HTTP/SSE.

## Serving comparison

| System | Concurrency | Output tok/s | p50 TTFT | p99 TTFT | p50 TPOT | p99 TPOT | Peak running seqs |
|---|---:|---:|---:|---:|---:|---:|---:|
| HF generate() in-process | 1 | 35.6 | 76 ms | – | 28.12 ms | – | 1 |
| HF static batch (padded) | 1 | 29.8 | – | – | – | – | 1 |
| HF static batch (padded) | 4 | 102.1 | – | – | – | – | 4 |
| HF static batch (padded) | 16 | 230.5 | – | – | – | – | 16 |
| HF static batch (padded) | 32 | 253.7 | – | – | – | – | 32 |
| Naive server (HF generate, 1 at a time) | 1 | 32.9 | 82 ms | 112 ms | 28.97 ms | 32.96 ms | 1 |
| Naive server (HF generate, 1 at a time) | 4 | 32.9 | 15,529 ms | 18,281 ms | 29.77 ms | 33.08 ms | 1 |
| Naive server (HF generate, 1 at a time) | 16 | 33.1 | 36,736 ms | 77,035 ms | 28.97 ms | 32.19 ms | 1 |
| tiny-infer (batched + paged) | 1 | 36.4 | 43 ms | 54 ms | 26.46 ms | 31.59 ms | 1 |
| tiny-infer (batched + paged) | 4 | 106.3 | 92 ms | 213 ms | 32.71 ms | 35.31 ms | 4 |
| tiny-infer (batched + paged) | 16 | 256.6 | 144 ms | 837 ms | 55.23 ms | 58.85 ms | 16 |
| tiny-infer (batched + paged) | 32 | 311.2 | 234 ms | 1,582 ms | 94.72 ms | 99.69 ms | 32 |
| tiny-infer (batched + paged) | 64 | 340.9 | 420 ms | 3,265 ms | 178.76 ms | 190.89 ms | 64 |
| vLLM (official) | 1 | 100.3 | 69 ms | 92 ms | 9.63 ms | 9.74 ms | – |
| vLLM (official) | 4 | 276.0 | 92 ms | 216 ms | 12.27 ms | 12.53 ms | – |
| vLLM (official) | 16 | 759.2 | 104 ms | 825 ms | 17.99 ms | 20.38 ms | – |
| vLLM (official) | 32 | 1,018.9 | 135 ms | 1,456 ms | 27.14 ms | 32.39 ms | – |
| vLLM (official) | 64 | 1,331.1 | 232 ms | 2,951 ms | 43.13 ms | 53.41 ms | – |

## Ratios at equal concurrency

| Concurrency | engine / naive | (engine+kernels) / engine | best tiny-infer / vLLM |
|---:|---:|---:|---:|
| 1 | 1.11x | – | 36% |
| 4 | 3.23x | – | 38% |
| 16 | 7.75x | – | 34% |
| 32 | – | – | 31% |
| 64 | – | – | 26% |

## Paged vs contiguous KV cache (same GPU, same 1 GiB pool, same kernels)

KV cost: 22,528 bytes/token. Pool: 47,648 token slots.

| Allocation | Peak concurrent sequences | Mean KV waste | Preemptions | Output tok/s |
|---|---:|---:|---:|---:|
| contiguous | 23 | 79.7% | 0 | 314.9 |
| paged | 143 | 1.5% | 21 | 381.5 |

Paged runs **6.2x** more sequences at once and delivers **1.21x** the throughput. Closed form: contiguous fits 23 sequences (it reserves max_model_len each, 76% of it unused at the mean final length 483); paged fits ~96.

## Prefix caching and chunked prefill

| Run | Concurrency | p50 TTFT | p99 TTFT | p99 ITL | Output tok/s | Prefix hit rate |
|---|---:|---:|---:|---:|---:|---:|
| chunk-512 | 16 | 691 ms | 4,036 ms | 169.3 ms | 122.6 | 0% |
| nochunk | 16 | 363 ms | 2,673 ms | 302.8 ms | 123.8 | 0% |
| prefix-off | 16 | 376 ms | 2,714 ms | 303.6 ms | 120.4 | 0% |
| prefix-on | 16 | 209 ms | 707 ms | 130.4 ms | 170.5 | 84% |

## Where a decode step goes (torch.profiler)

| Batch | Kernels | Wall ms/step | GPU busy | GPU idle (CPU/launch) | GEMM-class | copy/gather | attention | norm | other | Roofline floor | % of floor |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | off | 79.5 | 65.7 | 13.8 | 31.2 | 31.6 | 0.0 | 0.0 | 2.8 | 7.7 | 10% |

GEMM-class includes the batched matmuls inside the PyTorch attention fallback (SDPA's math path on sm_75).

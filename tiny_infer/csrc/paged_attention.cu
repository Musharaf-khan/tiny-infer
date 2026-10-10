// Paged decode attention (FP16 in, FP32 accumulate), one new query token per sequence:
//     o[b, h] = softmax(q[b, h] . K[b]^T * scale) V[b]       K/V read straight out of the block pool
//
// Lineage: cuda-kernel-lab/kernels/07_attention (Project 1) was a PREFILL kernel: 64 query rows per
// block, K/V tiles staged in shared memory and reused by every row, online softmax. Decode is a
// different shape: ONE query row per head, so there is nothing to reuse K/V across and the job is
// pure streaming. The arithmetic intensity is ~1 FLOP/byte, far left of the T4 ridge point, so the
// only things that matter are (a) reading each K/V byte once and (b) keeping enough blocks in flight.
// Three decisions follow:
//
//  1. GQA-shared loads. TinyLlama has 32 query heads but only 4 KV heads (G = 8 query heads per KV
//     head). One block handles one KV head and ALL G query heads that use it, so each K/V row is read
//     from HBM once instead of G times. (The gather+SDPA fallback materialises the K/V copy and
//     repeat_interleaves it G times.)
//  2. Paged addressing. The block table maps token t -> (physical block, offset); no dense K/V copy.
//  3. Split-K over the context (Flash-Decoding). With batch 1 and 4 KV heads there would be only 4
//     blocks for 40 SMs. The context is cut into partitions of PARTITION tokens, one block each; a
//     second tiny kernel merges the partial softmaxes with the same (max, sum) rescaling as online
//     softmax: w_p = sum_p * exp(max_p - max).
//
// Inside a block: each warp takes every NWARPS-th token. A lane owns HD/32 consecutive elements of
// the head (HD = 64: one half2 = 4 bytes, so a warp reads one 128-byte K row: fully coalesced),
// computes partial dot products for all G heads, and warp-shuffle-reduces them. Scores sit in shared
// memory, softmax is reduced block-wide, then the same token loop accumulates P.V and the warps'
// partial outputs are summed through shared memory.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "reduce.cuh"

namespace tiny_infer {

constexpr int NWARPS = 4;
constexpr int NTHREADS = NWARPS * 32;

template <int EPL>
__device__ __forceinline__ void load_row(const __half* p, float (&r)[EPL]) {
    static_assert(EPL % 2 == 0, "EPL must be even");
#pragma unroll
    for (int e = 0; e < EPL; e += 2) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(p + e));
        r[e] = f.x;
        r[e + 1] = f.y;
    }
}

template <int HD, int G>
__global__ void __launch_bounds__(NTHREADS)
paged_decode_attention_kernel(__half* __restrict__ out,               // [B, Hq, HD]
                              float* __restrict__ part_out,           // [B, Hq, P, HD]   (P > 1)
                              float* __restrict__ part_max,           // [B, Hq, P]
                              float* __restrict__ part_sum,           // [B, Hq, P]
                              const __half* __restrict__ q,           // [B, Hq, HD]
                              const __half* __restrict__ k_cache,     // [NB, BS, Hkv, HD]
                              const __half* __restrict__ v_cache,
                              const int* __restrict__ block_tables,   // [B, max_blocks]
                              const int* __restrict__ context_lens,   // [B]
                              int max_blocks, int num_kv_heads, int block_size, int partition, float scale) {
    constexpr int EPL = HD / 32;                       // head elements per lane
    const int b = blockIdx.x, kvh = blockIdx.y, p = blockIdx.z, P = gridDim.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int num_q_heads = num_kv_heads * G;
    const int ctx = context_lens[b];
    const int start = p * partition;

    if (start >= ctx) {                                // empty partition (shorter sequence in the batch)
        if (threadIdx.x < G) {
            const size_t i = ((size_t)b * num_q_heads + kvh * G + threadIdx.x) * P + p;
            part_max[i] = -INFINITY;
            part_sum[i] = 0.0f;
        }
        return;
    }
    const int n = min(partition, ctx - start);

    extern __shared__ float smem[];
    float* logits = smem;                              // [G][partition]
    float* red = smem + G * partition;                 // [NWARPS][G][HD]
    __shared__ float s_sum[G];

    // Query rows for the G heads of this KV head, pre-scaled, in registers.
    float qr[G][EPL];
#pragma unroll
    for (int g = 0; g < G; g++) {
        load_row<EPL>(q + ((size_t)b * num_q_heads + kvh * G + g) * HD + lane * EPL, qr[g]);
#pragma unroll
        for (int e = 0; e < EPL; e++) qr[g][e] *= scale;
    }
    const int* table = block_tables + (size_t)b * max_blocks;
    const size_t row_stride = (size_t)num_kv_heads * HD;          // one token slot in the pool

    // ---- scores: one K row load serves all G heads
    for (int t = warp; t < n; t += NWARPS) {
        const int tok = start + t;
        const size_t slot = (size_t)table[tok / block_size] * block_size + tok % block_size;
        float kr[EPL];
        load_row<EPL>(k_cache + slot * row_stride + (size_t)kvh * HD + lane * EPL, kr);
#pragma unroll
        for (int g = 0; g < G; g++) {
            float d = 0.0f;
#pragma unroll
            for (int e = 0; e < EPL; e++) d += qr[g][e] * kr[e];
            d = warp_sum(d);
            if (lane == 0) logits[g * partition + t] = d;
        }
    }
    __syncthreads();

    // ---- softmax over this partition, per head
    float gmax[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        float m = -INFINITY;
        for (int t = threadIdx.x; t < n; t += NTHREADS) m = fmaxf(m, logits[g * partition + t]);
        m = block_max(m);
        float s = 0.0f;
        for (int t = threadIdx.x; t < n; t += NTHREADS) {
            const float e = expf(logits[g * partition + t] - m);
            logits[g * partition + t] = e;
            s += e;
        }
        s = block_sum(s);                              // ends with __syncthreads: exp values visible
        gmax[g] = m;
        if (threadIdx.x == 0) s_sum[g] = s;
    }

    // ---- P.V: one V row load serves all G heads
    float acc[G][EPL];
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
        for (int e = 0; e < EPL; e++) acc[g][e] = 0.0f;
    for (int t = warp; t < n; t += NWARPS) {
        const int tok = start + t;
        const size_t slot = (size_t)table[tok / block_size] * block_size + tok % block_size;
        float vr[EPL];
        load_row<EPL>(v_cache + slot * row_stride + (size_t)kvh * HD + lane * EPL, vr);
#pragma unroll
        for (int g = 0; g < G; g++) {
            const float pg = logits[g * partition + t];
#pragma unroll
            for (int e = 0; e < EPL; e++) acc[g][e] += pg * vr[e];
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
        for (int e = 0; e < EPL; e++) red[(warp * G + g) * HD + lane * EPL + e] = acc[g][e];
    __syncthreads();

    // ---- sum the warps' partial outputs, normalise, write
    for (int i = threadIdx.x; i < G * HD; i += NTHREADS) {
        const int g = i / HD, d = i % HD;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; w++) v += red[(w * G + g) * HD + d];
        v /= s_sum[g];
        const int h = kvh * G + g;
        if (P == 1) {
            out[((size_t)b * num_q_heads + h) * HD + d] = __float2half_rn(v);
        } else {
            const size_t pi = ((size_t)b * num_q_heads + h) * P + p;
            part_out[pi * HD + d] = v;
        }
    }
    if (P > 1 && threadIdx.x < G) {
        const size_t pi = ((size_t)b * num_q_heads + kvh * G + threadIdx.x) * P + p;
#pragma unroll
        for (int g = 0; g < G; g++)            // constant-index the register array (no local memory)
            if (g == (int)threadIdx.x) part_max[pi] = gmax[g];
        part_sum[pi] = s_sum[threadIdx.x];
    }
}

// Merge the P partial results of each (sequence, head): the online-softmax merge rule.
template <int HD>
__global__ void combine_partitions_kernel(__half* __restrict__ out, const float* __restrict__ part_out,
                                          const float* __restrict__ part_max, const float* __restrict__ part_sum,
                                          const int* __restrict__ context_lens, int P, int partition) {
    const int b = blockIdx.x, h = blockIdx.y, Hq = gridDim.y, d = threadIdx.x;
    const int used = min(P, (context_lens[b] + partition - 1) / partition);
    const size_t base = ((size_t)b * Hq + h) * P;
    float m = -INFINITY;
    for (int p = 0; p < used; p++) m = fmaxf(m, part_max[base + p]);
    float num = 0.0f, den = 0.0f;
    for (int p = 0; p < used; p++) {
        const float w = part_sum[base + p] * expf(part_max[base + p] - m);
        num += w * part_out[(base + p) * HD + d];
        den += w;
    }
    out[((size_t)b * Hq + h) * HD + d] = __float2half_rn(num / den);
}

void paged_decode_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                            const torch::Tensor& context_lens, double scale, int64_t max_context_len,
                            int64_t partition_size) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.is_contiguous() && q.dim() == 3,
                "q must be contiguous CUDA fp16 [B, Hq, HD]");
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf && v_cache.scalar_type() == at::kHalf &&
                k_cache.is_contiguous() && v_cache.is_contiguous(), "caches must be contiguous fp16");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt && block_tables.is_contiguous(), "block_tables: int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt && context_lens.is_contiguous(), "context_lens: int32");
    const int B = q.size(0), Hq = q.size(1), HD = q.size(2);
    const int block_size = k_cache.size(1), Hkv = k_cache.size(2);
    TORCH_CHECK(k_cache.size(3) == HD && Hq % Hkv == 0, "head shape mismatch");
    const int G = Hq / Hkv;
    if (B == 0) return;

    int partition = partition_size > 0 ? (int)partition_size : (int)max_context_len;
    partition = std::max(32, std::min(partition, ((int)max_context_len + 31) / 32 * 32));
    const int P = ((int)max_context_len + partition - 1) / partition;
    const size_t smem = (size_t)(G * partition + NWARPS * G * HD) * sizeof(float);
    TORCH_CHECK(smem <= 48 * 1024, "paged_decode_attention: partition too large for shared memory (",
                smem, " bytes); use partition_size <= 512");

    auto fopts = q.options().dtype(at::kFloat);
    torch::Tensor part_out, part_max, part_sum;
    if (P > 1) {
        part_out = torch::empty({B, Hq, P, HD}, fopts);
        part_max = torch::empty({B, Hq, P}, fopts);
        part_sum = torch::empty({B, Hq, P}, fopts);
    }
    float* po = P > 1 ? part_out.data_ptr<float>() : nullptr;
    float* pm = P > 1 ? part_max.data_ptr<float>() : nullptr;
    float* ps = P > 1 ? part_sum.data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    const __half* kp = reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>());
    const __half* vp = reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>());
    const int* bt = block_tables.data_ptr<int>();
    const int* cl = context_lens.data_ptr<int>();
    const int max_blocks = block_tables.size(1);
    const dim3 grid(B, Hkv, P);

#define TI_ATTN(HD_, G_)                                                                              \
    paged_decode_attention_kernel<HD_, G_><<<grid, NTHREADS, smem, stream>>>(                       \
        o, po, pm, ps, qp, kp, vp, bt, cl, max_blocks, Hkv, block_size, partition, (float)scale)
#define TI_ATTN_G(HD_)                                                                                \
    switch (G) {                                                                                      \
        case 1: TI_ATTN(HD_, 1); break;                                                               \
        case 2: TI_ATTN(HD_, 2); break;                                                               \
        case 4: TI_ATTN(HD_, 4); break;                                                               \
        case 8: TI_ATTN(HD_, 8); break;                                                               \
        default: TORCH_CHECK(false, "unsupported GQA group size ", G);                                \
    }
    if (HD == 64) { TI_ATTN_G(64) }
    else if (HD == 128) { TI_ATTN_G(128) }
    else TORCH_CHECK(false, "unsupported head_dim ", HD);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (P > 1) {
        if (HD == 64)
            combine_partitions_kernel<64><<<dim3(B, Hq), 64, 0, stream>>>(o, po, pm, ps, cl, P, partition);
        else
            combine_partitions_kernel<128><<<dim3(B, Hq), 128, 0, stream>>>(o, po, pm, ps, cl, P, partition);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#undef TI_ATTN_G
#undef TI_ATTN
}

}  // namespace tiny_infer

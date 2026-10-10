// RMSNorm and fused residual-add + RMSNorm for FP16 activations.
//
// Lineage: cuda-kernel-lab/kernels/06_layernorm (Project 1), v3 "register-cached" and the fused
// add + LayerNorm kernel. Same idea, Llama's norm:
//   - one block per token row; the row is loaded ONCE into registers as 16-byte vectors (8 halves),
//   - statistics are reduced with warp shuffles + one shared-memory hop (block_sum),
//   - the output is written once. HBM traffic = the minimum: read x (and residual), write y (and h).
// RMSNorm drops the mean, so there is a single reduction (sum of squares) instead of two.
//
// Numerics mirror HF LlamaRMSNorm exactly, step by step:
//   h   = fp16(x + residual)                  (torch adds two fp16 tensors, rounds once)
//   var = mean(float(h)^2);  n = fp16(float(h) * rsqrt(var + eps))
//   y   = fp16(float(w) * float(n))           (weight multiply happens in fp16 after the cast)
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "reduce.cuh"

namespace tiny_infer {

constexpr int VEC = 8;   // halves per 16-byte load

union Pack8 {
    uint4 u;
    __half h[VEC];
};

// MAXV = max 16-byte vectors per thread kept in registers. hidden <= MAXV * VEC * blockDim.
template <int MAXV, bool FUSED_ADD>
__global__ void rms_norm_kernel(__half* out,                          // [T, H] (== x when FUSED_ADD: no __restrict__)
                                const __half* x,                      // [T, H]
                                __half* __restrict__ residual,        // [T, H] in/out when FUSED_ADD
                                const __half* __restrict__ weight,    // [H]
                                int hidden, float eps) {
    const int nvec = hidden / VEC;
    const size_t base = (size_t)blockIdx.x * nvec;
    const uint4* x4 = reinterpret_cast<const uint4*>(x) + base;
    uint4* r4 = reinterpret_cast<uint4*>(residual) + base;
    uint4* o4 = reinterpret_cast<uint4*>(out) + base;
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);

    Pack8 v[MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < MAXV; j++) {
        const int idx = j * blockDim.x + threadIdx.x;
        if (idx < nvec) {
            v[j].u = x4[idx];
            if (FUSED_ADD) {
                Pack8 r;
                r.u = r4[idx];
#pragma unroll
                for (int e = 0; e < VEC; e++)
                    v[j].h[e] = __float2half_rn(__half2float(v[j].h[e]) + __half2float(r.h[e]));
                r4[idx] = v[j].u;                       // h goes back out for the next residual
            }
#pragma unroll
            for (int e = 0; e < VEC; e++) {
                const float f = __half2float(v[j].h[e]);
                ss += f * f;
            }
        }
    }
    const float rstd = rsqrtf(block_sum(ss) / hidden + eps);
#pragma unroll
    for (int j = 0; j < MAXV; j++) {
        const int idx = j * blockDim.x + threadIdx.x;
        if (idx < nvec) {
            Pack8 w, o;
            w.u = w4[idx];
#pragma unroll
            for (int e = 0; e < VEC; e++) {
                const __half n = __float2half_rn(__half2float(v[j].h[e]) * rstd);
                o.h[e] = __float2half_rn(__half2float(w.h[e]) * __half2float(n));
            }
            o4[idx] = o.u;
        }
    }
}

template <bool FUSED_ADD>
static void launch(torch::Tensor& out, const torch::Tensor& x, torch::Tensor* residual,
                   const torch::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "rms_norm: x must be a CUDA fp16 tensor");
    TORCH_CHECK(weight.scalar_type() == at::kHalf && weight.is_contiguous(), "rms_norm: weight must be fp16");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "rms_norm: tensors must be contiguous");
    const int hidden = x.size(-1);
    TORCH_CHECK(hidden % VEC == 0, "rms_norm: hidden size must be a multiple of 8");
    TORCH_CHECK(weight.numel() == hidden, "rms_norm: weight size mismatch");
    const int rows = x.numel() / hidden;
    if (rows == 0) return;
    const int nvec = hidden / VEC;
    // 256 threads: hidden 2048 (TinyLlama) = exactly one 16-byte vector per thread
    const int threads = std::max(32, std::min(256, ((nvec + 31) / 32) * 32));
    const int maxv = (nvec + threads - 1) / threads;
    TORCH_CHECK(maxv <= 8, "rms_norm: hidden size too large (max 16384)");
    auto stream = at::cuda::getCurrentCUDAStream();
    __half* o = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half* xi = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* r = residual ? reinterpret_cast<__half*>(residual->data_ptr<at::Half>()) : nullptr;
    const __half* w = reinterpret_cast<const __half*>(weight.data_ptr<at::Half>());
    const float e = static_cast<float>(eps);
#define TI_LAUNCH(MV) rms_norm_kernel<MV, FUSED_ADD><<<rows, threads, 0, stream>>>(o, xi, r, w, hidden, e)
    if (maxv <= 1) TI_LAUNCH(1);
    else if (maxv <= 2) TI_LAUNCH(2);
    else if (maxv <= 4) TI_LAUNCH(4);
    else TI_LAUNCH(8);
#undef TI_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rms_norm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, double eps) {
    launch<false>(out, x, nullptr, weight, eps);
}

// In place: residual <- x + residual; x <- RMSNorm(residual) * weight
void fused_add_rms_norm(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& weight, double eps) {
    TORCH_CHECK(residual.is_contiguous() && residual.scalar_type() == at::kHalf && residual.sizes() == x.sizes(),
                "fused_add_rms_norm: residual must match x");
    launch<true>(x, x, &residual, weight, eps);
}

}  // namespace tiny_infer

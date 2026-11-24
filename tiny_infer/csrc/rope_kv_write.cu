// Fused RoPE + KV-cache write (G4 slice 1).
//
// Per layer, the unfused path did three separate passes over q/k plus two scatters:
//   q, k = rope(q, k, positions)                       (cos/sin gather, rotate_half cat, muls)
//   k_cache[slot_mapping] = k;  v_cache[slot_mapping] = v     (two index_copy_ scatters)
// This kernel does it in one pass: one block per (token, head).
//   - q heads (0..Hq-1): RoPE in place.
//   - kv heads (0..Hkv-1): RoPE k in place, then write the RoPE'd k and the raw v into the
//     paged cache at slot_mapping[t].
// Each thread owns one pair (j, j + hd/2):
//   out[j]   = x[j]*cos[j] - x[j+h]*sin[j]
//   out[j+h] = x[j+h]*cos[j] + x[j]*sin[j]
// (The cos/sin tables duplicate the first half in the second half, exactly as HF builds them,
// so cos[j] == cos[j+h] and only the first half is ever read.)
// The math is done in fp32 and rounded once to fp16, the same rounding policy as rms_norm.cu;
// the PyTorch reference in ops.py (rope_kv_write_ref) is the oracle, and kernels.self_test
// checks the kernel against it.
//
// GPU verification pending (Colab T4): this VM has no GPU, so the kernel is written and
// CPU-tested here; the self-test runs it on CUDA when a GPU is present, and the engine
// refuses to serve with custom kernels on CUDA if that self-test fails.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/all.h>

namespace tiny_infer {

__global__ void rope_kv_write_kernel(__half* __restrict__ q,              // [T, Hq, hd] in/out
                                     __half* __restrict__ k,              // [T, Hkv, hd] in/out
                                     const __half* __restrict__ v,        // [T, Hkv, hd]
                                     const __half* __restrict__ cos,      // [max_pos, hd]
                                     const __half* __restrict__ sin,      // [max_pos, hd]
                                     const int64_t* __restrict__ positions,      // [T]
                                     const int64_t* __restrict__ slot_mapping,   // [T]
                                     __half* __restrict__ k_cache,        // [NB, BS, Hkv, hd]
                                     __half* __restrict__ v_cache,        // [NB, BS, Hkv, hd]
                                     int num_q_heads, int num_kv_heads, int head_dim) {
    const int half = head_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;
    const int t = blockIdx.x / total_heads;
    const int h = blockIdx.x % total_heads;
    const int j = threadIdx.x;                       // 0..half-1
    const int64_t pos = positions[t];
    const float c = __half2float(cos[pos * head_dim + j]);
    const float s = __half2float(sin[pos * head_dim + j]);
    if (h < num_q_heads) {
        __half* row = q + ((size_t)t * num_q_heads + h) * head_dim;
        const float x0 = __half2float(row[j]);
        const float x1 = __half2float(row[j + half]);
        row[j] = __float2half_rn(x0 * c - x1 * s);
        row[j + half] = __float2half_rn(x1 * c + x0 * s);
    } else {
        const int kh = h - num_q_heads;
        __half* krow = k + ((size_t)t * num_kv_heads + kh) * head_dim;
        const float x0 = __half2float(krow[j]);
        const float x1 = __half2float(krow[j + half]);
        const float y0 = x0 * c - x1 * s;
        const float y1 = x1 * c + x0 * s;
        krow[j] = __float2half_rn(y0);
        krow[j + half] = __float2half_rn(y1);
        const int64_t slot = slot_mapping[t];
        __half* kc = k_cache + ((size_t)slot * num_kv_heads + kh) * head_dim;
        __half* vc = v_cache + ((size_t)slot * num_kv_heads + kh) * head_dim;
        const __half* vrow = v + ((size_t)t * num_kv_heads + kh) * head_dim;
        kc[j] = __float2half_rn(y0);
        kc[j + half] = __float2half_rn(y1);
        vc[j] = vrow[j];
        vc[j + half] = vrow[j + half];
    }
}

void rope_kv_write(torch::Tensor& q, torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& cos, const torch::Tensor& sin,
                   const torch::Tensor& positions, const torch::Tensor& slot_mapping,
                   torch::Tensor& k_cache, torch::Tensor& v_cache) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "rope_kv_write: q must be a CUDA fp16 tensor");
    TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kHalf, "rope_kv_write: k must be a CUDA fp16 tensor");
    TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kHalf, "rope_kv_write: v must be a CUDA fp16 tensor");
    TORCH_CHECK(cos.is_cuda() && cos.scalar_type() == at::kHalf, "rope_kv_write: cos must be CUDA fp16");
    TORCH_CHECK(sin.is_cuda() && sin.scalar_type() == at::kHalf, "rope_kv_write: sin must be CUDA fp16");
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong,
                "rope_kv_write: positions must be a CUDA int64 tensor");
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.scalar_type() == at::kLong,
                "rope_kv_write: slot_mapping must be a CUDA int64 tensor");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.scalar_type() == at::kHalf, "rope_kv_write: k_cache must be CUDA fp16");
    TORCH_CHECK(v_cache.is_cuda() && v_cache.scalar_type() == at::kHalf, "rope_kv_write: v_cache must be CUDA fp16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "rope_kv_write: q, k, v must be contiguous");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "rope_kv_write: cos, sin must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "rope_kv_write: caches must be contiguous");
    const int T = q.size(0);
    const int Hq = q.size(1), Hkv = k.size(1), hd = q.size(2);
    TORCH_CHECK(k.size(0) == T && v.sizes() == k.sizes(), "rope_kv_write: k/v shape mismatch with q");
    TORCH_CHECK(positions.numel() == T && slot_mapping.numel() == T, "rope_kv_write: positions/slot_mapping length");
    TORCH_CHECK(cos.size(1) == hd && sin.size(1) == hd, "rope_kv_write: cos/sin head_dim mismatch");
    TORCH_CHECK(k_cache.size(2) == Hkv && v_cache.sizes() == k_cache.sizes(), "rope_kv_write: cache shape mismatch");
    TORCH_CHECK(hd % 2 == 0 && hd >= 2 && hd <= 512, "rope_kv_write: head_dim must be even, 2..512");
    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    rope_kv_write_kernel<<<T * (Hq + Hkv), hd / 2, 0, stream>>>(
        reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sin.data_ptr<at::Half>()),
        positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v_cache.data_ptr<at::Half>()),
        Hq, Hkv, hd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tiny_infer

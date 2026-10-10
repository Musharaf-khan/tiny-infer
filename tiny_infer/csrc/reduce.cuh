// Warp- and block-level reductions, carried over from cuda-kernel-lab/common/reduce.cuh (Project 1):
//   1. each warp reduces its 32 values with register shuffles (no shared memory, no barrier)
//   2. lane 0 of each warp writes one partial to shared memory
//   3. warp 0 reduces the partials, and the result is broadcast to every thread
#pragma once
#include <cmath>

namespace tiny_infer {

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

// Every thread of the block must call these (they contain __syncthreads()).
// blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarps) ? partial[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    v = partial[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarps) ? partial[lane] : -INFINITY;
        v = warp_max(v);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    v = partial[0];
    __syncthreads();
    return v;
}

}  // namespace tiny_infer

// Registers the CUDA kernels as PyTorch custom ops: torch.ops.tiny_infer.<name>.
// Schemas mark mutated arguments with Tensor(a!) so torch.compile / functionalization see the writes.
#include <torch/all.h>
#include <torch/library.h>

namespace tiny_infer {
void rms_norm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& weight, double eps);
void fused_add_rms_norm(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& weight, double eps);
void paged_decode_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                            const torch::Tensor& context_lens, double scale, int64_t max_context_len,
                            int64_t partition_size);
}  // namespace tiny_infer

TORCH_LIBRARY(tiny_infer, m) {
    m.def("rms_norm(Tensor(a!) out, Tensor x, Tensor weight, float eps) -> ()");
    m.def("fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, float eps) -> ()");
    m.def("paged_decode_attention(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor v_cache, "
          "Tensor block_tables, Tensor context_lens, float scale, int max_context_len, "
          "int partition_size) -> ()");
}

TORCH_LIBRARY_IMPL(tiny_infer, CUDA, m) {
    m.impl("rms_norm", &tiny_infer::rms_norm);
    m.impl("fused_add_rms_norm", &tiny_infer::fused_add_rms_norm);
    m.impl("paged_decode_attention", &tiny_infer::paged_decode_attention);
}

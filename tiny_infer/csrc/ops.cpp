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
void rope_kv_write(torch::Tensor& q, torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& cos, const torch::Tensor& sin,
                   const torch::Tensor& positions, const torch::Tensor& slot_mapping,
                   torch::Tensor& k_cache, torch::Tensor& v_cache);
}  // namespace tiny_infer

TORCH_LIBRARY(tiny_infer, m) {
    m.def("rms_norm(Tensor(a!) out, Tensor x, Tensor weight, float eps) -> ()");
    m.def("fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, float eps) -> ()");
    m.def("paged_decode_attention(Tensor(a!) out, Tensor q, Tensor k_cache, Tensor v_cache, "
          "Tensor block_tables, Tensor context_lens, float scale, int max_context_len, "
          "int partition_size) -> ()");
    m.def("rope_kv_write(Tensor(a!) q, Tensor(b!) k, Tensor v, Tensor cos, Tensor sin, "
          "Tensor positions, Tensor slot_mapping, Tensor(c!) k_cache, Tensor(d!) v_cache) -> ()");
}

TORCH_LIBRARY_IMPL(tiny_infer, CUDA, m) {
    m.impl("rms_norm", &tiny_infer::rms_norm);
    m.impl("fused_add_rms_norm", &tiny_infer::fused_add_rms_norm);
    m.impl("paged_decode_attention", &tiny_infer::paged_decode_attention);
    m.impl("rope_kv_write", &tiny_infer::rope_kv_write);
}

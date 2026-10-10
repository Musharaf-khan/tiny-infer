"""The three ops the decode path spends its non-GEMM time in, each with two implementations:

    *_ref    plain PyTorch, numerically identical to HuggingFace's Llama (the correctness oracle)
    kernel   Project 1 CUDA kernels, registered as torch.library custom ops (torch.ops.tiny_infer.*)

The model calls the dispatchers below; use_custom=True routes to the CUDA kernels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_KERNELS_LOADED = False


# ---------------------------------------------------------------------------- references
def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Exactly HF LlamaRMSNorm: statistics in fp32, cast back, then scale in the model dtype."""
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(dt)


def fused_add_rms_norm_ref(x, residual, w, eps):
    residual = x + residual
    return rms_norm_ref(residual, w, eps), residual


def paged_decode_attention_ref(q, k_cache, v_cache, block_tables, context_lens, scale, max_context_len):
    """Gather each sequence's K/V out of the block pool into a padded dense tensor, then SDPA.

    q: [B, Hq, hd]; k_cache/v_cache: [num_blocks, block_size, Hkv, hd];
    block_tables: int32 [B, max_blocks]; context_lens: int32 [B]. Returns [B, Hq, hd].
    This materializes a [B, Hq, L, hd] copy of the cache: the extra HBM traffic the kernel avoids.
    """
    B, Hq, hd = q.shape
    _, bs, Hkv, _ = k_cache.shape
    L = int(max_context_len)
    pos = torch.arange(L, device=q.device)
    slots = block_tables.long()[:, pos // bs] * bs + pos % bs                  # [B, L]
    k = k_cache.view(-1, Hkv, hd)[slots].transpose(1, 2)                         # [B, Hkv, L, hd]
    v = v_cache.view(-1, Hkv, hd)[slots].transpose(1, 2)
    g = Hq // Hkv
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    lens = context_lens.long()
    if bool((lens == L).all()):
        mask = None                                                              # no padding: same call HF makes
    else:
        mask = (pos[None, :] < lens[:, None])[:, None, None, :]                  # [B, 1, 1, L]
    o = F.scaled_dot_product_attention(q.unsqueeze(2), k, v, attn_mask=mask, scale=scale)
    return o.squeeze(2)


# ---------------------------------------------------------------------------- kernel loading
def load_kernels(verbose: bool = False) -> bool:
    """JIT-compile and register the CUDA ops (first call ~1-2 min; cached afterwards)."""
    global _KERNELS_LOADED
    if _KERNELS_LOADED:
        return True
    if hasattr(torch.ops, "tiny_infer") and hasattr(torch.ops.tiny_infer, "rms_norm"):
        _KERNELS_LOADED = True
        return True
    from .kernels import build_extension
    build_extension(verbose=verbose)
    _KERNELS_LOADED = True
    return True


def kernels_available() -> bool:
    return _KERNELS_LOADED


def _use_kernel(use_custom: bool, *tensors) -> bool:
    if not use_custom:
        return False
    if not _KERNELS_LOADED:
        raise RuntimeError("use_custom_kernels=True but kernels are not loaded; call ops.load_kernels()")
    return all(t.is_cuda and t.dtype == torch.float16 for t in tensors)


# ---------------------------------------------------------------------------- dispatchers
def rms_norm(x, w, eps, use_custom=False):
    if _use_kernel(use_custom, x, w):
        x = x.contiguous()
        out = torch.empty_like(x)
        torch.ops.tiny_infer.rms_norm(out, x, w, eps)
        return out
    return rms_norm_ref(x, w, eps)


def fused_add_rms_norm(x, residual, w, eps, use_custom=False):
    """Returns (norm(x + residual), x + residual). The kernel does it in place in one pass."""
    if _use_kernel(use_custom, x, residual, w):
        x, residual = x.contiguous(), residual.contiguous()
        torch.ops.tiny_infer.fused_add_rms_norm(x, residual, w, eps)
        return x, residual
    return fused_add_rms_norm_ref(x, residual, w, eps)


def paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale, max_context_len,
                           use_custom=False, partition_size: int = 512):
    if _use_kernel(use_custom, q, k_cache, v_cache):
        q = q.contiguous()
        out = torch.empty_like(q)
        torch.ops.tiny_infer.paged_decode_attention(
            out, q, k_cache, v_cache, block_tables, context_lens, float(scale), int(max_context_len),
            int(partition_size))
        return out
    return paged_decode_attention_ref(q, k_cache, v_cache, block_tables, context_lens, scale, max_context_len)

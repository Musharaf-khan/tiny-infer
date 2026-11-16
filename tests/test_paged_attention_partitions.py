"""T1: CPU tests for the 128-token paged-attention partitions (G1).

The CUDA kernel itself can only run on a GPU, so on this CPU-only VM the
verifiable slice is:

1. Partition invariance: a split-K reference (chunk the context into
   ``partition_size``-token pieces, per-partition softmax, online-softmax
   merge) must agree with the unpartitioned SDPA reference
   (``ops.paged_decode_attention_ref``) for ragged lengths, shuffled physical
   blocks, and GQA ratios 8 and 1, at both 128 and 512 tokens. If the
   partition size changed the math, the kernel's split-K would be wrong
   no matter how fast it is.
2. Plumbing: 128 is the default everywhere (EngineConfig, LlamaModel,
   ops dispatcher, server CLI), and 512 flows through as the switch-back.
3. Kernel self-test: ``kernels.self_test()`` runs inside pytest when a CUDA
   device is present, and is skipped on CPU-only machines. GPU verification
   of the kernel itself is still pending on Colab T4.
"""
import inspect

import pytest
import torch
import torch.nn.functional as F

from tiny_infer import ops
from tiny_infer.config import EngineConfig
from tiny_infer.model import LlamaModel
from tiny_infer.server import parse_args

# Same shapes as kernels.self_test: ragged lengths (1 token, unaligned 17,
# mid 300, long 1100, max 2048), GQA 8 (32:4, 16:2) and GQA 1 (8:8).
CASES = [
    (32, 4, 64),    # GQA 8
    (8, 8, 64),     # GQA 1 (MHA)
    (16, 2, 128),   # GQA 8, wider head
]
LENS = [1, 17, 300, 1100, 2048]
BLOCK_SIZE = 16
PARTITIONS = (128, 512)


def _paged_inputs(hq, hkv, hd, seed=0):
    """Random paged KV with shuffled physical blocks, one physical block per
    (sequence, logical block), exactly like kernels.self_test."""
    g = torch.Generator().manual_seed(seed)
    B = len(LENS)
    lens = torch.tensor(LENS, dtype=torch.int32)
    maxb = (int(lens.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = B * maxb + 16
    kc = torch.randn(nb, BLOCK_SIZE, hkv, hd, generator=g)
    vc = torch.randn(nb, BLOCK_SIZE, hkv, hd, generator=g)
    perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))[: B * maxb]
    bt = perm.view(B, maxb).to(torch.int32)
    q = torch.randn(B, hq, hd, generator=g)
    return q, kc, vc, bt, lens


def _partitioned_ref(q, k_cache, v_cache, block_tables, context_lens, scale,
                     max_context_len, partition_size):
    """Split-K reference: gather each sequence's K/V, split the context into
    partition_size-token chunks, softmax each chunk, merge with the
    online-softmax (running-max) rescale. Mirrors what the CUDA kernel does;
    the unpartitioned SDPA ref is the oracle it must match."""
    B, Hq, hd = q.shape
    _, bs, Hkv, _ = k_cache.shape
    L = int(max_context_len)
    pos = torch.arange(L)
    slots = block_tables.long()[:, pos // bs] * bs + pos % bs
    k = k_cache.view(-1, Hkv, hd)[slots].transpose(1, 2)   # [B, Hkv, L, hd]
    v = v_cache.view(-1, Hkv, hd)[slots].transpose(1, 2)
    g = Hq // Hkv
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    lens = context_lens.long()
    out = torch.zeros(B, Hq, hd)
    for b in range(B):
        length = int(lens[b])
        m = torch.full((Hq,), float("-inf"))
        l = torch.zeros(Hq)
        acc = torch.zeros(Hq, hd)
        for s in range(0, length, partition_size):
            e = min(s + partition_size, length)
            scores = (q[b].unsqueeze(1) @ k[b, :, s:e].transpose(1, 2)).squeeze(1) * scale   # [Hq, e-s]
            m_p = scores.max(dim=-1).values
            p = torch.exp(scores - m_p[:, None])
            l_p = p.sum(dim=-1)
            acc_p = (p.unsqueeze(1) @ v[b, :, s:e]).squeeze(1)
            m_new = torch.maximum(m, m_p)
            alpha_old = torch.exp(m - m_new)
            alpha_new = torch.exp(m_p - m_new)
            acc = acc * alpha_old[:, None] + acc_p * alpha_new[:, None]
            l = l * alpha_old + l_p * alpha_new
            m = m_new
        out[b] = acc / l[:, None]
    return out


@pytest.mark.parametrize("hq,hkv,hd", CASES)
@pytest.mark.parametrize("partition_size", PARTITIONS)
def test_partitioned_attention_matches_unpartitioned_ref(hq, hkv, hd, partition_size):
    """128- and 512-token partitions must not change the attention result:
    ragged lengths, shuffled blocks, GQA 8 and 1."""
    q, kc, vc, bt, lens = _paged_inputs(hq, hkv, hd)
    ref = ops.paged_decode_attention_ref(q, kc, vc, bt, lens, hd ** -0.5, int(lens.max()))
    got = _partitioned_ref(q, kc, vc, bt, lens, hd ** -0.5, int(lens.max()), partition_size)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(got).all()


def test_partition_sizes_agree_with_each_other():
    """128 vs 512 partitions agree with each other, not just with the ref,
    on the GQA-8 shape (the production default is 128)."""
    q, kc, vc, bt, lens = _paged_inputs(32, 4, 64)
    a = _partitioned_ref(q, kc, vc, bt, lens, 64 ** -0.5, int(lens.max()), 128)
    b = _partitioned_ref(q, kc, vc, bt, lens, 64 ** -0.5, int(lens.max()), 512)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_partition_boundary_lengths():
    """Lengths sitting exactly on, and one token either side of, a 128-token
    partition boundary (128, 129, 255, 256, 257) must not lose or double a token."""
    g = torch.Generator().manual_seed(3)
    lens_list = [128, 129, 255, 256, 257]
    B = len(lens_list)
    lens = torch.tensor(lens_list, dtype=torch.int32)
    maxb = (int(lens.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = B * maxb + 16
    kc = torch.randn(nb, BLOCK_SIZE, 4, 64, generator=g)
    vc = torch.randn(nb, BLOCK_SIZE, 4, 64, generator=g)
    perm = torch.randperm(nb, generator=torch.Generator().manual_seed(3))[: B * maxb]
    bt = perm.view(B, maxb).to(torch.int32)
    q = torch.randn(B, 32, 64, generator=g)
    ref = ops.paged_decode_attention_ref(q, kc, vc, bt, lens, 64 ** -0.5, int(lens.max()))
    for ps in PARTITIONS:
        got = _partitioned_ref(q, kc, vc, bt, lens, 64 ** -0.5, int(lens.max()), ps)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- plumbing
def test_partition_size_default_is_128_everywhere():
    assert EngineConfig().attention_partition_size == 128
    assert inspect.signature(LlamaModel.__init__).parameters["attention_partition_size"].default == 128
    assert inspect.signature(ops.paged_decode_attention).parameters["partition_size"].default == 128
    assert parse_args([]).attention_partition_size == 128


def test_partition_size_512_switchback_flows_to_model(hf_model):
    """The 512 switch-back reaches the model, and on CPU the engine still
    generates identical tokens (both sizes go through the same ref here;
    the kernel path is GPU-only, verification pending on Colab T4)."""
    from conftest import make_engine, random_prompts
    from tiny_infer.sequence import SamplingParams

    prompts = random_prompts(3)
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    eng128 = make_engine(hf_model)
    eng512 = make_engine(hf_model, attention_partition_size=512)
    assert eng128.model.attention_partition_size == 128
    assert eng512.model.attention_partition_size == 512
    got128 = [s.output_ids for s in eng128.generate(prompts, params)]
    got512 = [s.output_ids for s in eng512.generate(prompts, params)]
    assert got128 == got512


# ---------------------------------------------------------------- GPU kernel
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this VM")
def test_kernel_self_test_passes_on_gpu():
    """The kernel self-test (same ragged/GQA/partition cases, fp16) must pass
    on a real GPU; skipped on CPU-only machines. GPU verification pending
    on Colab T4 for this run's CPU-side changes."""
    from tiny_infer.kernels import self_test

    errs = self_test()
    assert errs, "self_test returned no cases"
    worst = max(errs.values())
    assert worst < 2e-2, f"kernel self-test worst error {worst:.2e}: {errs}"

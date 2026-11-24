"""G4 slice 1: the fused RoPE + KV-cache write.

This VM has no GPU, so the CUDA kernel itself cannot run here. These tests pin the behaviour
the kernel must match: the dispatcher (reference path) against the pre-G4 code
(LlamaModel._rope + two index_copy_ scatters), and the whole model forward, now routed
through the dispatcher, token-identical to HF. The kernel is checked against the same
reference in kernels.self_test, which runs on CUDA when a GPU is present (and the engine
refuses to serve with custom kernels on CUDA if that self-test fails).
"""
import pytest
import torch

from conftest import hf_greedy, make_engine, make_hf_tiny_llama, random_prompts
from tiny_infer import ops
from tiny_infer.model import LlamaModel
from tiny_infer.sequence import SamplingParams


def _cos_sin(hd, max_pos=512, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _case(hq, hkv, hd, T=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, hq, hd, generator=g)
    k = torch.randn(T, hkv, hd, generator=g)
    v = torch.randn(T, hkv, hd, generator=g)
    cos, sin = _cos_sin(hd)
    positions = torch.randint(0, 512, (T,), generator=g)
    nb, bs = 64, 16
    slots = torch.randperm(nb * bs, generator=g)[:T]      # shuffled physical slots
    kc = torch.zeros(nb, bs, hkv, hd)
    vc = torch.zeros(nb, bs, hkv, hd)
    return q, k, v, cos, sin, positions, slots, kc, vc


def _oracle_model(hd):
    """A LlamaModel whose _rope is the pre-G4 production code, retabled for this head_dim."""
    m = LlamaModel.from_hf_model(make_hf_tiny_llama(), "cpu", torch.float32)
    m.cos, m.sin = _cos_sin(hd)
    return m


def _oracle(q, k, v, model, positions, slots, kc, vc):
    """The exact pre-G4 lines from LlamaModel.forward."""
    q_o, k_o = model._rope(q.clone(), k.clone(), positions)
    kc_o, vc_o = kc.clone(), vc.clone()
    kc_o.view(-1, k.shape[1], k.shape[2]).index_copy_(0, slots, k_o)
    vc_o.view(-1, v.shape[1], v.shape[2]).index_copy_(0, slots, v.contiguous())
    return q_o, k_o, kc_o, vc_o


@pytest.mark.parametrize("hq,hkv,hd", [(32, 4, 64), (8, 8, 64), (16, 2, 128), (4, 1, 64)])
def test_dispatcher_matches_pre_g4_code(hq, hkv, hd):
    q, k, v, cos, sin, positions, slots, kc, vc = _case(hq, hkv, hd)
    model = _oracle_model(hd)
    q_o, k_o, kc_o, vc_o = _oracle(q, k, v, model, positions, slots, kc, vc)
    q_d, k_d = ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions, slots,
                                 kc, vc)
    assert torch.equal(q_d, q_o) and torch.equal(k_d, k_o)
    assert torch.equal(kc, kc_o) and torch.equal(vc, vc_o)


@pytest.mark.parametrize("hq,hkv,hd", [(32, 4, 64), (8, 8, 128)])
def test_cache_slots_land_exactly(hq, hkv, hd):
    """K lands RoPE'd and V lands raw, at exactly the mapped slots, and nothing else moves."""
    q, k, v, cos, sin, positions, slots, kc, vc = _case(hq, hkv, hd)
    model = _oracle_model(hd)
    q_o, k_o, _, _ = _oracle(q, k, v, model, positions, slots, kc, vc)
    ops.rope_kv_write(q, k, v, cos, sin, positions, slots, kc, vc)
    flat_k, flat_v = kc.view(-1, hkv, hd), vc.view(-1, hkv, hd)
    assert torch.equal(flat_k[slots], k_o)          # RoPE'd k at the mapped slots
    assert torch.equal(flat_v[slots], v)            # raw v, never rotated
    untouched = torch.ones(flat_k.shape[0], dtype=torch.bool)
    untouched[slots] = False
    assert (flat_k[untouched] == 0).all() and (flat_v[untouched] == 0).all()


def test_single_token_and_table_edge_positions():
    """T=1 (every decode step) and positions at the last table row."""
    q, k, v, cos, sin, _, slots, kc, vc = _case(8, 2, 64, T=1, seed=3)
    positions = torch.tensor([511])                 # last row of the cos/sin tables
    model = _oracle_model(64)
    q_o, k_o, kc_o, vc_o = _oracle(q, k, v, model, positions, slots, kc, vc)
    q_d, k_d = ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions, slots,
                                 kc, vc)
    assert torch.equal(q_d, q_o) and torch.equal(k_d, k_o)
    assert torch.equal(kc, kc_o) and torch.equal(vc, vc_o)


@pytest.mark.skipif(ops.kernels_available(), reason="custom kernels are loaded on this machine")
def test_custom_flag_without_loaded_kernels_raises():
    """use_custom=True with nothing loaded is a loud error, never a silent fallback."""
    q, k, v, cos, sin, positions, slots, kc, vc = _case(8, 2, 64, T=4, seed=4)
    with pytest.raises(RuntimeError, match="kernels are not loaded"):
        ops.rope_kv_write(q, k, v, cos, sin, positions, slots, kc, vc, use_custom=True)


def test_forward_through_dispatcher_matches_hf(hf_model):
    """The model forward now routes RoPE + cache write through the dispatcher; greedy
    decoding must still be token-identical to HF (chunked prefill forces several steps)."""
    prompts = random_prompts(6, seed=11)
    expected = [hf_greedy(hf_model, p, 12) for p in prompts]
    eng = make_engine(hf_model, max_num_batched_tokens=7)
    got = [s.output_ids for s in eng.generate(
        prompts, SamplingParams(max_tokens=12, ignore_eos=True))]
    assert got == expected

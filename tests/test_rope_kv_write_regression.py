"""T4: regression tests for the fused RoPE + KV-cache write (G4).

G4 slice 1 pinned the dispatcher against the pre-G4 code. These tests pin it against an
independent oracle instead: RoPE computed from first principles (the per-pair rotation
formula, with angles built straight from theta), so a bug shared by _rope and the
reference path cannot hide. They also cover what the G4 tests did not: a multi-step
decode loop accumulating into one cache (the shape decode actually runs in), slot
overwrite, position 0 as identity, and a long engine run against HF where attention
reads the cache the dispatcher built up over many steps.
"""
import pytest
import torch

from conftest import hf_greedy, make_engine, make_hf_tiny_llama, random_prompts
from tiny_infer import ops
from tiny_infer.sequence import SamplingParams


def _cos_sin(hd, max_pos=512, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _independent_rope(x, positions, theta=10000.0):
    """RoPE from first principles: per-pair rotation, angles straight from theta.

    For each pair (j, j + h): out[j] = x[j]*cos(a) - x[j+h]*sin(a),
    out[j+h] = x[j+h]*cos(a) + x[j]*sin(a), with a = pos * theta**(-2j/hd).
    Deliberately not built from the model's cos/sin tables or _rotate_half, so it
    catches a bug shared by those two paths. Angles are evaluated in fp32, and two
    fp32 evaluations of cos/sin at a large angle differ by ~1e-5 (argument-reduction
    sensitivity grows with position: measured max diff 3.9e-05 at position 400),
    so compare with atol=1e-4, not exact equality; a wrong rotation would be O(1) off.
    """
    h = x.shape[-1] // 2
    j = torch.arange(h, dtype=torch.float32)
    angles = positions.float()[:, None] * theta ** (-2.0 * j / x.shape[-1])  # [T, h]
    c, s = angles.cos(), angles.sin()
    x0, x1 = x[..., :h], x[..., h:]
    return torch.cat([x0 * c[:, None, :] - x1 * s[:, None, :],
                      x1 * c[:, None, :] + x0 * s[:, None, :]], dim=-1)


def _case(hq, hkv, hd, T=37, positions=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, hq, hd, generator=g)
    k = torch.randn(T, hkv, hd, generator=g)
    v = torch.randn(T, hkv, hd, generator=g)
    cos, sin = _cos_sin(hd)
    if positions is None:
        positions = torch.randint(0, 512, (T,), generator=g)
    nb, bs = 64, 16
    slots = torch.randperm(nb * bs, generator=g)[:T]      # shuffled physical slots
    kc = torch.zeros(nb, bs, hkv, hd)
    vc = torch.zeros(nb, bs, hkv, hd)
    return q, k, v, cos, sin, positions, slots, kc, vc


@pytest.mark.parametrize("hq,hkv,hd", [(32, 4, 64), (8, 8, 128), (4, 1, 64)])
def test_dispatcher_matches_independent_rope_oracle(hq, hkv, hd):
    """Structured positions (0, 1, a spread, and the last table row) against the
    from-first-principles oracle: q/k outputs, and the cache contents at the slots."""
    T = 16
    positions = torch.tensor([0, 1, 2, 5, 17, 64, 100, 127, 128, 200, 255, 256, 300, 400, 500, 511])
    q, k, v, cos, sin, positions, slots, kc, vc = _case(hq, hkv, hd, T=T,
                                                         positions=positions, seed=7)
    q_in, k_in = q.clone(), k.clone()
    q_d, k_d = ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions, slots,
                                 kc, vc)
    assert torch.allclose(q_d, _independent_rope(q_in, positions), atol=1e-4)
    assert torch.allclose(k_d, _independent_rope(k_in, positions), atol=1e-4)
    flat_k, flat_v = kc.view(-1, hkv, hd), vc.view(-1, hkv, hd)
    assert torch.allclose(flat_k[slots], _independent_rope(k_in, positions), atol=1e-4)
    assert torch.equal(flat_v[slots], v)          # v lands raw, never rotated
    untouched = torch.ones(flat_k.shape[0], dtype=torch.bool)
    untouched[slots] = False
    assert (flat_k[untouched] == 0).all() and (flat_v[untouched] == 0).all()


def test_position_zero_is_identity():
    """At position 0, cos = 1 and sin = 0, so RoPE must pass q/k through unchanged."""
    q, k, v, cos, sin, _, slots, kc, vc = _case(8, 2, 64, T=5, seed=8)
    positions = torch.zeros(5, dtype=torch.int64)
    q_d, k_d = ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions, slots,
                                 kc, vc)
    assert torch.equal(q_d, q) and torch.equal(k_d, k)
    assert torch.equal(kc.view(-1, 2, 64)[slots], k)


def test_decode_loop_accumulates_without_clobbering():
    """Thirty T=1 decode steps into one cache, positions 0..29: after every step, every
    slot written so far still holds exactly its own step's RoPE'd k and raw v, and no
    unwritten slot has moved. A later write must never clobber an earlier token."""
    hq, hkv, hd = 8, 2, 64
    cos, sin = _cos_sin(hd)
    g = torch.Generator().manual_seed(9)
    nb, bs = 64, 16
    slots = torch.randperm(nb * bs, generator=g)[:30]
    kc = torch.zeros(nb, bs, hkv, hd)
    vc = torch.zeros(nb, bs, hkv, hd)
    written_k, written_v = {}, {}
    for step in range(30):
        q = torch.randn(1, hq, hd, generator=g)
        k = torch.randn(1, hkv, hd, generator=g)
        v = torch.randn(1, hkv, hd, generator=g)
        positions = torch.tensor([step])
        q_d, k_d = ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions,
                                     slots[step:step + 1], kc, vc)
        assert torch.allclose(q_d, _independent_rope(q, positions), atol=1e-4)
        assert torch.allclose(k_d, _independent_rope(k, positions), atol=1e-4)
        written_k[int(slots[step])] = _independent_rope(k, positions)[0]
        written_v[int(slots[step])] = v[0]
        flat_k, flat_v = kc.view(-1, hkv, hd), vc.view(-1, hkv, hd)
        for slot, exp_k in written_k.items():
            assert torch.allclose(flat_k[slot], exp_k, atol=1e-4), f"slot {slot} clobbered at step {step}"
            assert torch.equal(flat_v[slot], written_v[slot])
        untouched = torch.ones(flat_k.shape[0], dtype=torch.bool)
        untouched[slots[:step + 1]] = False
        assert (flat_k[untouched] == 0).all() and (flat_v[untouched] == 0).all()


def test_slot_overwrite_replaces():
    """Writing to a slot twice replaces the first write exactly (a sequence restarted
    into a reused slot must not see stale K/V)."""
    q, k, v, cos, sin, positions, slots, kc, vc = _case(8, 2, 64, T=1, seed=10)
    slot = slots[:1]
    ops.rope_kv_write(q.clone(), k.clone(), v.clone(), cos, sin, positions, slot, kc, vc)
    q2, k2, v2, _, _, positions2, _, _, _ = _case(8, 2, 64, T=1, seed=11)
    ops.rope_kv_write(q2.clone(), k2.clone(), v2.clone(), cos, sin, positions2, slot, kc, vc)
    flat_k, flat_v = kc.view(-1, 2, 64), vc.view(-1, 2, 64)
    assert torch.allclose(flat_k[slot[0]], _independent_rope(k2, positions2)[0], atol=1e-4)
    assert torch.equal(flat_v[slot[0]], v2[0])


def test_long_generation_matches_hf(hf_model):
    """48 greedy tokens with chunked prefill (max_num_batched_tokens=7) token-identical
    to HF. This is the end-to-end regression for the fused write: attention reads the
    cache the dispatcher built up over dozens of prefill and decode steps, at positions
    far past the 12-token G4 check."""
    prompts = random_prompts(4, seed=23)
    expected = [hf_greedy(hf_model, p, 48) for p in prompts]
    eng = make_engine(hf_model, max_num_batched_tokens=7)
    got = [s.output_ids for s in eng.generate(
        prompts, SamplingParams(max_tokens=48, ignore_eos=True))]
    assert got == expected

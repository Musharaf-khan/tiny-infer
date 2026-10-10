"""Correctness oracle: greedy decoding must be token-identical to HuggingFace generate().

Every engine feature that changes HOW tokens are computed (batching, paging, chunking, prefix hits,
preemption + recompute) must not change WHICH tokens come out.
"""
import pytest
import torch

from conftest import hf_greedy, make_engine, random_prompts
from tiny_infer.sequence import SamplingParams

MAX_NEW = 24


@pytest.fixture(scope="module")
def reference(hf_model):
    prompts = random_prompts(10)
    return prompts, [hf_greedy(hf_model, p, MAX_NEW) for p in prompts]


def run(engine, prompts, max_new=MAX_NEW):
    params = SamplingParams(max_tokens=max_new, ignore_eos=True)
    return [s.output_ids for s in engine.generate(prompts, params)]


CONFIGS = {
    "batched_paged": {},
    "unfused_weights": {"fuse_weights": False},
    "no_chunking": {"enable_chunked_prefill": False},
    "tiny_budget_chunked": {"max_num_batched_tokens": 7},          # forces many prefill chunks
    "contiguous": {"kv_mode": "contiguous", "num_blocks": 1024},
    "one_at_a_time": {"max_num_seqs": 1},
    "block_16": {"block_size": 16, "num_blocks": 64},
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_matches_hf(hf_model, reference, name):
    prompts, expected = reference
    got = run(make_engine(hf_model, **CONFIGS[name]), prompts)
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"[{name}] prompt {i} diverged at token {next(j for j in range(len(e)) if g[j] != e[j])}"


def test_preemption_recompute_matches_hf(hf_model, reference):
    """A pool too small for everyone forces recompute-preemption mid-generation."""
    prompts, expected = reference
    eng = make_engine(hf_model, num_blocks=40, block_size=4)
    got = run(eng, prompts)
    assert eng.scheduler.num_preemptions > 0, "test did not exercise preemption"
    assert got == expected


def test_prefix_caching_matches_hf(hf_model):
    shared = random_prompts(1, 30, 31, seed=7)[0]                 # a shared "system prompt"
    prompts = [shared + p for p in random_prompts(6, 2, 10, seed=8)]
    expected = [hf_greedy(hf_model, p, 12) for p in prompts]
    eng = make_engine(hf_model, enable_prefix_caching=True)
    first = run(eng, prompts[:1], 12)                             # warm the cache
    rest = run(eng, prompts[1:], 12)
    assert first + rest == expected
    assert eng.bm.stats()["prefix_hit_rate"] > 0.5


def test_sampling_runs(hf_model):
    eng = make_engine(hf_model)
    seqs = eng.generate(random_prompts(3), SamplingParams(max_tokens=8, temperature=0.8, top_p=0.9, ignore_eos=True))
    assert all(s.num_output_tokens == 8 for s in seqs)

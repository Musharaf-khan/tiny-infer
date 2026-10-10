"""Shared fixtures: a small, randomly initialised Llama with TinyLlama's architecture (GQA 32:4 ratio
scaled down, RoPE, RMSNorm, SwiGLU). Runs on CPU in fp32 so the HF comparison is exact."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_infer.config import EngineConfig  # noqa: E402
from tiny_infer.engine import LLMEngine  # noqa: E402
from tiny_infer.model import LlamaModel  # noqa: E402


def make_hf_tiny_llama(seed: int = 0, num_kv_heads: int = 2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=344, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=num_kv_heads, max_position_embeddings=512,
                      rms_norm_eps=1e-5, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
                      pad_token_id=0, initializer_range=0.08)
    m = LlamaForCausalLM(cfg).eval()
    # random norm weights so a wrong norm can't hide behind gamma = 1
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.6, 1.4)
    return m


@pytest.fixture(scope="session")
def hf_model():
    return make_hf_tiny_llama()


def make_engine(hf_model, **overrides) -> LLMEngine:
    kw = dict(model="", device="cpu", dtype="float32", block_size=4, max_model_len=256,
              max_num_seqs=16, max_num_batched_tokens=64, num_blocks=256)
    kw.update(overrides)
    cfg = EngineConfig(**kw)
    model = LlamaModel.from_hf_model(hf_model, "cpu", torch.float32, fuse_weights=cfg.fuse_weights)
    return LLMEngine(cfg, model=model, tokenizer=None)


def hf_greedy(hf_model, prompt_ids, max_new):
    out = hf_model.generate(torch.tensor([prompt_ids]), max_new_tokens=max_new, do_sample=False,
                            min_new_tokens=max_new, pad_token_id=0, eos_token_id=None)
    return out[0, len(prompt_ids):].tolist()


def random_prompts(n, lo=3, hi=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(lo, hi, (1,), generator=g))
        out.append(torch.randint(3, 512, (L,), generator=g).tolist())
    return out

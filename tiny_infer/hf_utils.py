"""Version-safe HF loading. transformers renamed `torch_dtype` to `dtype` in 4.56; older versions silently
treat an unknown `dtype=` as a config attribute and load fp32, so pick the right name explicitly and verify."""
from __future__ import annotations

import torch


def _dtype_kw() -> str:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_causal_lm(name: str, dtype: torch.dtype, device: str | None = None):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(name, **{_dtype_kw(): dtype}, low_cpu_mem_usage=True)
    if device:
        model = model.to(device)
    model.eval()
    got = next(model.parameters()).dtype
    if got != dtype:
        raise RuntimeError(f"loaded {name} as {got}, expected {dtype}")
    return model

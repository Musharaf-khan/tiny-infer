"""Regressions found on the first T4 run."""
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))


def test_sse_parser_keeps_unicode_line_separators_inside_json():
    """vLLM streams raw U+2028 inside JSON strings; splitting on it corrupted 3 of 416 vLLM requests."""
    from harness import _sse_lines

    class FakeResp:
        async def aiter_text(self):
            payload = 'data: {"choices": [{"text": "a b\u0085c"}]}\n\ndata: [DONE]\n\n'
            for i in range(0, len(payload), 7):          # arbitrary chunk boundaries
                yield payload[i:i + 7]

    async def collect():
        return [l async for l in _sse_lines(FakeResp())]

    lines = [l for l in asyncio.run(collect()) if l]
    assert lines == ['data: {"choices": [{"text": "a b\u0085c"}]}', "data: [DONE]"]


def test_kernel_self_test_cases_are_constructible():
    """The self-test's block tables must fit its block pool (the first T4 run crashed here)."""
    import inspect
    from tiny_infer import kernels, ops
    src = inspect.getsource(kernels.self_test)
    assert "nb = B * maxb" in src
    bs, B = 16, 5
    lens = torch.tensor([1, 17, 300, 1100, 2048], dtype=torch.int32)
    maxb = (int(lens.max()) + bs - 1) // bs
    nb = B * maxb + 16
    bt = torch.randperm(nb)[: B * maxb].view(B, maxb).to(torch.int32)
    kc, vc = torch.randn(nb, bs, 4, 64), torch.randn(nb, bs, 4, 64)
    out = ops.paged_decode_attention_ref(torch.randn(B, 32, 64), kc, vc, bt, lens, 0.125, int(lens.max()))
    assert out.shape == (B, 32, 64) and torch.isfinite(out).all()

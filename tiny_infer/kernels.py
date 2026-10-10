"""Build the CUDA extension (JIT, cached under ~/.cache/torch_extensions) and self-test it."""
from __future__ import annotations

import os

import torch

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
SOURCES = ["ops.cpp", "rms_norm.cu", "paged_attention.cu"]


def _arch_flags() -> list[str]:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in os.environ.get("TINY_INFER_ARCHS", "75").split(",")]


def build_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load
    return load(
        name="tiny_infer_C",
        sources=[os.path.join(CSRC, s) for s in SOURCES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"] + _arch_flags(),
        is_python_module=False,
        verbose=verbose,
    )


@torch.inference_mode()
def self_test(device: str = "cuda", seed: int = 0) -> dict:
    """Compare every kernel against its PyTorch reference on random data; returns max abs errors."""
    from . import ops
    ops.load_kernels()
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=device, dtype=torch.float16)
    res = {}
    # norms
    for hidden in (2048, 4096, 128):
        x, w, resid = r(37, hidden), r(hidden) * 0.2 + 1, r(37, hidden)
        res[f"rms_norm[{hidden}]"] = (ops.rms_norm(x, w, 1e-5, True).float()
                                      - ops.rms_norm_ref(x, w, 1e-5).float()).abs().max().item()
        y_ref, h_ref = ops.fused_add_rms_norm_ref(x, resid, w, 1e-5)
        y, h = ops.fused_add_rms_norm(x.clone(), resid.clone(), w, 1e-5, True)
        res[f"fused_add_rms_norm[{hidden}]"] = max((y.float() - y_ref.float()).abs().max().item(),
                                                   (h.float() - h_ref.float()).abs().max().item())
    # paged decode attention: ragged lengths, shuffled physical blocks, GQA 8 and 1, split / no split
    for hq, hkv, hd in ((32, 4, 64), (8, 8, 64), (16, 2, 128)):
        bs, nb, B = 16, 512, 5
        kc, vc = r(nb, bs, hkv, hd), r(nb, bs, hkv, hd)
        lens = torch.tensor([1, 17, 300, 1100, 2048], dtype=torch.int32, device=device)
        maxb = (int(lens.max()) + bs - 1) // bs
        perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))[: B * maxb]
        bt = perm.view(B, maxb).to(torch.int32).to(device)
        q = r(B, hq, hd)
        ref = ops.paged_decode_attention_ref(q, kc, vc, bt, lens, hd ** -0.5, int(lens.max()))
        for ps in (512, 128):
            out = ops.paged_decode_attention(q, kc, vc, bt, lens, hd ** -0.5, int(lens.max()), True, ps)
            res[f"paged_attn[hq={hq},hkv={hkv},hd={hd},part={ps}]"] = (out.float() - ref.float()).abs().max().item()
    return res


if __name__ == "__main__":
    errs = self_test()
    worst = max(errs.values())
    for k, v in errs.items():
        print(f"{k:48s} max|err| = {v:.2e}")
    print("PASS" if worst < 2e-2 else "FAIL", f"(worst {worst:.2e}; fp16 output ulp near 1.0 is 9.8e-4)")

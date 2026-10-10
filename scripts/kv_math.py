"""KV-cache VRAM math for any model, from its HF config (or a built-in preset when offline).

    python scripts/kv_math.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --context 2048 --users 32
    python scripts/kv_math.py --preset llama3-8b --context 8192 --users 64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from tiny_infer.config import ModelConfig  # noqa: E402

PRESETS = {  # layers, q heads, kv heads, head_dim, hidden, intermediate, vocab, params (B)
    "tinyllama-1.1b": (22, 32, 4, 64, 2048, 5632, 32000, 1.1),
    "llama2-7b": (32, 32, 32, 128, 4096, 11008, 32000, 6.7),
    "llama3-8b": (32, 32, 8, 128, 4096, 14336, 128256, 8.0),
    "llama3-70b": (80, 64, 8, 128, 8192, 28672, 128256, 70.6),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--gpu-gib", type=float, default=15.0, help="VRAM (T4 = 15)")
    a = p.parse_args()
    params_b = None
    if a.preset or not a.model:
        L, H, KV, D, hid, inter, voc, params_b = PRESETS[a.preset or "tinyllama-1.1b"]
        cfg = ModelConfig(vocab_size=voc, hidden_size=hid, intermediate_size=inter, num_layers=L, num_heads=H,
                          num_kv_heads=KV, head_dim=D)
    else:
        from transformers import AutoConfig
        cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(a.model))
    dt = torch.float16 if a.dtype == "float16" else torch.float32
    m = cfg.kv_math(a.context, a.users, dt)
    print(f"bytes/token = 2 x {cfg.num_layers} layers x {cfg.num_kv_heads} kv_heads x {cfg.head_dim} head_dim x "
          f"{2 if a.dtype == 'float16' else 4} B = {m['bytes_per_token']:,} B ({m['bytes_per_token'] / 1024:.1f} KiB)")
    print(f"per sequence @ {a.context} tokens = {m['bytes_per_seq'] / 2**20:,.1f} MiB")
    print(f"{a.users} users @ {a.context} tokens = {m['total_gib']:,.2f} GiB of KV cache")
    if params_b:
        w = params_b * 1e9 * (2 if a.dtype == "float16" else 4) / 2**30
        free = a.gpu_gib * 0.9 - w
        print(f"weights ~{w:.1f} GiB -> ~{free:.1f} GiB left on a {a.gpu_gib:.0f} GiB GPU = "
              f"{max(free, 0) * 2**30 / m['bytes_per_seq']:.0f} full-context sequences")
    if cfg.num_kv_heads < cfg.num_heads:
        print(f"GQA saves {cfg.num_heads // cfg.num_kv_heads}x vs full multi-head KV ({cfg.num_heads} kv heads)")


if __name__ == "__main__":
    main()

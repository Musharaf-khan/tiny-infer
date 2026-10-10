"""Model and engine configuration, plus the KV-cache math every interview asks about."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

DTYPE_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}


@dataclass
class ModelConfig:
    """The handful of Llama-family hyperparameters the engine needs."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = False
    eos_token_id: int | list[int] | None = None

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:  # transformers >= 5 keeps it in rope_parameters
            rp = getattr(hf_config, "rope_parameters", None) or {}
            rope_theta = rp.get("rope_theta", 10000.0)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling and rope_scaling.get("rope_type", rope_scaling.get("type")) not in (None, "default"):
            raise NotImplementedError(f"rope scaling {rope_scaling} is not supported by tiny-infer")
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            head_dim=head_dim,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=float(rope_theta),
            max_position_embeddings=hf_config.max_position_embeddings,
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            eos_token_id=hf_config.eos_token_id,
        )

    # ---------------------------------------------------------------- KV-cache math
    def kv_bytes_per_token(self, dtype: torch.dtype = torch.float16) -> int:
        """bytes/token = 2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    def kv_math(self, context_len: int, users: int, dtype: torch.dtype = torch.float16) -> dict:
        per_tok = self.kv_bytes_per_token(dtype)
        per_seq = per_tok * context_len
        return {
            "bytes_per_token": per_tok,
            "bytes_per_seq": per_seq,
            "total_bytes": per_seq * users,
            "total_gib": per_seq * users / 2**30,
        }


@dataclass
class EngineConfig:
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "float16"
    device: str = "cuda"
    block_size: int = 16
    max_model_len: int = 2048
    max_num_seqs: int = 128               # cap on concurrently running sequences
    max_num_batched_tokens: int = 2048    # per-step token budget (decode tokens + prefill chunk tokens)
    kv_mode: str = "paged"                # "paged" (allocate on demand) | "contiguous" (reserve max_model_len)
    kv_cache_gb: float | None = None      # fixed KV pool size; None = derive from gpu_memory_utilization
    gpu_memory_utilization: float = 0.90
    num_blocks: int | None = None         # override (tests / CPU)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    use_custom_kernels: bool = False      # Project 1 CUDA kernels in the decode path
    fuse_weights: bool = True             # concat q/k/v and gate/up into single GEMMs
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "half": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32, "float": torch.float32}[self.dtype]


if __name__ == "__main__":
    # Worked example from the project brief: TinyLlama 1.1B
    cfg = ModelConfig(vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_layers=22,
                      num_heads=32, num_kv_heads=4, head_dim=64)
    m = cfg.kv_math(context_len=2048, users=1)
    print(f"TinyLlama: {m['bytes_per_token']:,} B/token, {m['bytes_per_seq']/2**20:.1f} MiB per 2K-token sequence")

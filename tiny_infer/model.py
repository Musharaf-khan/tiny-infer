"""Llama-family forward pass over a PACKED batch with a paged KV cache.

The engine flattens every scheduled sequence's new tokens into one [T, hidden] tensor:

    [ d0 d1 d2 ... d(D-1) | p0 p0 p0 ... | p1 p1 ... ]
      decode tokens (1/seq)  prefill chunk  prefill chunk

Every linear layer (QKV, O, gate/up, down, lm_head) runs ONCE over all T tokens: that is where
batching pays off, because those GEMMs are bound by reading the weights, and the weights are read
once per step instead of once per sequence. Only attention needs per-sequence structure:

    decode rows   -> paged decode attention (reads K/V straight from the block pool)
    prefill rows  -> causal attention per sequence (over cached context + the new chunk)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import ops
from .config import ModelConfig


@dataclass
class AttentionMetadata:
    num_decode: int                       # first num_decode packed tokens are decode tokens (1 per seq)
    decode_block_tables: torch.Tensor | None   # int32 [num_decode, max_blocks]
    decode_context_lens: torch.Tensor | None   # int32 [num_decode], context length INCLUDING the new token
    max_decode_context: int
    # one entry per prefill sequence: (packed start, new tokens, tokens already cached, block table int32)
    prefills: list[tuple[int, int, int, torch.Tensor]]
    slot_mapping: torch.Tensor            # int64 [T]: physical cache slot for each new token
    positions: torch.Tensor               # int64 [T]
    logits_indices: torch.Tensor          # int64 [S]: packed rows whose logits are sampled


class KVCache:
    """One K and one V tensor per layer: [num_blocks, block_size, num_kv_heads, head_dim]."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype: torch.dtype,
                 device: torch.device):
        shape = (num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        self.num_blocks, self.block_size = num_blocks, block_size

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        return cfg.kv_bytes_per_token(dtype) * block_size


class LlamaModel:
    def __init__(self, cfg: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device,
                 dtype: torch.dtype, fuse_weights: bool = True, use_custom_kernels: bool = False):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.use_custom_kernels = use_custom_kernels
        self.fuse = fuse_weights
        w = {k: v.to(device=device, dtype=dtype).contiguous() for k, v in weights.items()}
        self.embed = w["model.embed_tokens.weight"]
        self.final_norm = w["model.norm.weight"]
        self.lm_head = self.embed if cfg.tie_word_embeddings or "lm_head.weight" not in w else w["lm_head.weight"]
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            L = {
                "in_norm": w[p + "input_layernorm.weight"],
                "post_norm": w[p + "post_attention_layernorm.weight"],
                "o": w[p + "self_attn.o_proj.weight"],
                "down": w[p + "mlp.down_proj.weight"],
            }
            q, k, v = (w[p + f"self_attn.{n}_proj.weight"] for n in "qkv")
            g, u = w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]
            if fuse_weights:
                L["qkv"] = torch.cat([q, k, v], 0).contiguous()
                L["gate_up"] = torch.cat([g, u], 0).contiguous()
            else:
                L.update(q=q, k=k, v=v, gate=g, up=u)
            self.layers.append(L)
        del w
        # RoPE tables, built exactly the way HF's LlamaRotaryEmbedding does (fp32, then cast)
        hd = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float().to(device) / hd))
        pos = torch.arange(max(cfg.max_position_embeddings, 1), device=device, dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos, self.sin = emb.cos().to(dtype), emb.sin().to(dtype)
        self.scale = hd ** -0.5

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_hf_model(cls, hf_model, device, dtype, **kw) -> "LlamaModel":
        cfg = ModelConfig.from_hf(hf_model.config)
        sd = {k: v for k, v in hf_model.state_dict().items()}
        return cls(cfg, sd, torch.device(device), dtype, **kw)

    @classmethod
    def from_pretrained(cls, name: str, device, dtype, **kw) -> "LlamaModel":
        from .hf_utils import load_causal_lm
        hf = load_causal_lm(name, dtype)
        m = cls.from_hf_model(hf, device, dtype, **kw)
        del hf
        return m

    # ------------------------------------------------------------------ pieces
    def _norm(self, x, w):
        return ops.rms_norm(x, w, self.cfg.rms_norm_eps, self.use_custom_kernels)

    def _add_norm(self, x, residual, w):
        return ops.fused_add_rms_norm(x, residual, w, self.cfg.rms_norm_eps, self.use_custom_kernels)

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def _rope(self, q, k, positions):
        cos = self.cos[positions].unsqueeze(1)      # [T, 1, hd]
        sin = self.sin[positions].unsqueeze(1)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    def _attention(self, q, k, v, layer: int, kv: KVCache, md: AttentionMetadata):
        """q: [T, Hq, hd]; k, v: [T, Hkv, hd] (already written to the cache). Returns [T, Hq*hd]."""
        cfg = self.cfg
        out = torch.empty_like(q)
        nd = md.num_decode
        if nd:
            out[:nd] = ops.paged_decode_attention(
                q[:nd], kv.k[layer], kv.v[layer], md.decode_block_tables, md.decode_context_lens,
                self.scale, md.max_decode_context, self.use_custom_kernels)
        groups = cfg.num_heads // cfg.num_kv_heads
        bs = kv.block_size
        for start, n, ctx, table in md.prefills:
            qs = q[start:start + n].transpose(0, 1).unsqueeze(0)            # [1, Hq, n, hd]
            if ctx == 0:     # whole prompt in this step: attend to the fresh K/V directly
                ks = k[start:start + n].transpose(0, 1).unsqueeze(0)
                vs = v[start:start + n].transpose(0, 1).unsqueeze(0)
                ks, vs = ks.repeat_interleave(groups, 1), vs.repeat_interleave(groups, 1)
                o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=self.scale)
            else:            # chunk / prefix hit: attend to cached context + this chunk
                total = ctx + n
                pos = torch.arange(total, device=q.device)
                slots = table.long()[pos // bs] * bs + pos % bs
                ks = kv.k[layer].view(-1, cfg.num_kv_heads, cfg.head_dim)[slots]
                vs = kv.v[layer].view(-1, cfg.num_kv_heads, cfg.head_dim)[slots]
                ks = ks.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, 1)
                vs = vs.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, 1)
                qpos = torch.arange(ctx, total, device=q.device)
                mask = pos[None, :] <= qpos[:, None]                         # causal with offset
                o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask, scale=self.scale)
            out[start:start + n] = o.squeeze(0).transpose(0, 1)
        return out.reshape(q.shape[0], -1)

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, md: AttentionMetadata, kv: KVCache) -> torch.Tensor:
        cfg = self.cfg
        T = input_ids.shape[0]
        hq, hkv, hd = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim, cfg.head_dim
        residual = F.embedding(input_ids, self.embed)
        x = None
        for i, L in enumerate(self.layers):
            if x is None:
                x = self._norm(residual, L["in_norm"])
            else:
                x, residual = self._add_norm(x, residual, L["in_norm"])
            if self.fuse:
                q, k, v = F.linear(x, L["qkv"]).split([hq, hkv, hkv], dim=-1)
            else:
                q, k, v = F.linear(x, L["q"]), F.linear(x, L["k"]), F.linear(x, L["v"])
            q = q.reshape(T, cfg.num_heads, hd)
            k = k.reshape(T, cfg.num_kv_heads, hd)
            v = v.reshape(T, cfg.num_kv_heads, hd)
            q, k = self._rope(q, k, md.positions)
            kv.k[i].view(-1, cfg.num_kv_heads, hd).index_copy_(0, md.slot_mapping, k)
            kv.v[i].view(-1, cfg.num_kv_heads, hd).index_copy_(0, md.slot_mapping, v.contiguous())
            x = F.linear(self._attention(q, k, v, i, kv, md), L["o"])
            x, residual = self._add_norm(x, residual, L["post_norm"])
            if self.fuse:
                g, u = F.linear(x, L["gate_up"]).chunk(2, dim=-1)
            else:
                g, u = F.linear(x, L["gate"]), F.linear(x, L["up"])
            x = F.linear(F.silu(g) * u, L["down"])
        idx = md.logits_indices          # only rows we sample from need the final norm + lm_head
        x, _ = self._add_norm(x[idx], residual[idx], self.final_norm)
        return F.linear(x, self.lm_head).float()

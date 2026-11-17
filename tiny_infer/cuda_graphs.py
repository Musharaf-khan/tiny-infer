"""CUDA-graph capture and replay for decode-only steps (G2).

Why: a batch-1 decode step is 16.6 ms against a 6.9 ms weight-read floor, so most of
the step is CPU launch overhead: dozens of small kernel launches per layer, each paying
Python + dispatcher + launch latency while the GPU idles. Capturing the whole forward
into one graph replaces all of that with a single replay.

Design (the vLLM shape, cut down to what this engine needs):

- One graph per batch-size bucket (powers of two up to max_num_seqs). A decode-only
  step with nd sequences runs the smallest bucket >= nd.
- All per-step inputs live in static buffers the graph captured pointers to: input ids,
  positions, slot mapping, block tables, context lens. Each step copies the real values
  in (device-to-device, no sync) and replays.
- Padded rows duplicate row 0 (same token, position, slot, block table, context len),
  so their K/V write lands on the same cache slot with the same values: idempotent.
  Their logits are discarded (only the first nd rows are returned).
- The captured max_decode_context is max_model_len (worst case). The paged-attention
  kernel writes -inf/0 for partitions past a sequence's real context length and the
  merge only reads partitions that cover real tokens, so a larger captured max is
  correct, just more partition blocks than a short context needs. Context-length
  bucketing (capture per context range) is the follow-up; it is not in this slice.
- Graphs share one CUDA graph memory pool, so capturing all buckets costs roughly the
  largest bucket's activations, not the sum.
- Decode-only steps only (no prefill rows). Prefill/chunked steps, and any step the
  runner cannot serve, fall back to the eager forward.
- Requires the custom-kernel path: the PyTorch reference attention calls bool() on a
  GPU tensor (a device-to-host sync), which CUDA graph capture forbids. The engine
  only builds the runner when use_custom_kernels is set; the runner raises otherwise.

What this slice does NOT do: no GPU measurement yet (this VM has no GPU), so the
speedup is unverified. GPU verification pending (Colab T4).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .model import AttentionMetadata

log = logging.getLogger("tiny_infer")


def graph_batch_sizes(max_num_seqs: int) -> list[int]:
    """Power-of-two batch buckets up to max_num_seqs (inclusive)."""
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    sizes = []
    b = 1
    while b < max_num_seqs:
        sizes.append(b)
        b *= 2
    sizes.append(max_num_seqs)
    return sizes


def bucket_for(n: int, buckets: list[int]) -> int | None:
    """Smallest bucket that fits n sequences, or None if n exceeds the largest."""
    for b in buckets:
        if n <= b:
            return b
    return None


def pad_decode_rows(t: torch.Tensor, bucket: int) -> torch.Tensor:
    """Pad a [n, ...] decode tensor to [bucket, ...] by repeating row 0.

    Padded rows must duplicate a real row (not zeros): a zero token/position/slot would
    write garbage K/V into the cache and attend over a nonsense context. Row 0's values
    make the padded row's K/V write identical to row 0's, so it is idempotent, and its
    logits are discarded by the caller.
    """
    n = t.shape[0]
    if n == 0:
        raise ValueError("cannot pad an empty decode batch")
    if n > bucket:
        raise ValueError(f"batch of {n} does not fit bucket {bucket}")
    if n == bucket:
        return t
    return torch.cat([t, t[:1].expand(bucket - n, *t.shape[1:])], dim=0)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_logits: torch.Tensor          # [bucket, vocab]; rewritten by every replay
    bufs: dict[str, torch.Tensor]        # static input buffers, captured by pointer
    md: AttentionMetadata                # static metadata, built on the buffers


class CUDAGraphRunner:
    """One captured CUDA graph per batch-size bucket, for decode-only steps."""

    def __init__(self, model, cfg):
        if model.device.type != "cuda":
            raise RuntimeError("CUDAGraphRunner requires a CUDA device")
        if not getattr(model, "use_custom_kernels", False):
            raise RuntimeError(
                "CUDAGraphRunner requires use_custom_kernels=True: the reference attention "
                "syncs to the host (bool() on a GPU tensor), which graph capture forbids")
        self.model = model
        self.device = model.device
        # Worst-case context: correct for every step (see module docstring), at the cost of
        # the maximum partition count. Block-table width is the worst case too.
        self.max_context = cfg.max_model_len
        self.max_blocks = -(-cfg.max_model_len // cfg.block_size)
        self.buckets = graph_batch_sizes(cfg.max_num_seqs)
        self._pool = torch.cuda.graphs.graph_pool_handle()   # shared: buckets reuse the memory
        self._graphs: dict[int, _GraphEntry] = {}

    # ------------------------------------------------------------------ internals
    def _fill(self, bufs: dict[str, torch.Tensor], input_ids: torch.Tensor, md: AttentionMetadata):
        """Copy this step's decode values into a bucket's static buffers (no sync)."""
        bucket = bufs["input_ids"].shape[0]
        bufs["input_ids"].copy_(pad_decode_rows(input_ids, bucket))
        bufs["positions"].copy_(pad_decode_rows(md.positions, bucket))
        bufs["slot_mapping"].copy_(pad_decode_rows(md.slot_mapping, bucket))
        w = md.decode_block_tables.shape[1]
        bufs["block_tables"][:, :w].copy_(pad_decode_rows(md.decode_block_tables, bucket))
        bufs["context_lens"].copy_(pad_decode_rows(md.decode_context_lens, bucket))

    def _capture(self, bucket: int, input_ids: torch.Tensor, md: AttentionMetadata,
                 kv) -> _GraphEntry:
        dev = self.device
        bufs = {
            "input_ids": torch.zeros(bucket, dtype=torch.long, device=dev),
            "positions": torch.zeros(bucket, dtype=torch.long, device=dev),
            "slot_mapping": torch.zeros(bucket, dtype=torch.long, device=dev),
            "block_tables": torch.zeros((bucket, self.max_blocks), dtype=torch.int32, device=dev),
            "context_lens": torch.zeros(bucket, dtype=torch.int32, device=dev),
            "logits_indices": torch.arange(bucket, dtype=torch.long, device=dev),
        }
        self._fill(bufs, input_ids, md)
        static_md = AttentionMetadata(
            num_decode=bucket, decode_block_tables=bufs["block_tables"],
            decode_context_lens=bufs["context_lens"], max_decode_context=self.max_context,
            prefills=[], slot_mapping=bufs["slot_mapping"], positions=bufs["positions"],
            logits_indices=bufs["logits_indices"])
        # Warm up on a side stream (weights/caches hot, no stray work on the capture stream),
        # then capture the forward into the shared pool.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model.forward(bufs["input_ids"], static_md, kv)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            static_logits = self.model.forward(bufs["input_ids"], static_md, kv)
        log.info("CUDA graph captured: batch bucket %d (max_context=%d)", bucket, self.max_context)
        return _GraphEntry(graph=g, static_logits=static_logits, bufs=bufs, md=static_md)

    # ------------------------------------------------------------------ entry point
    def run(self, input_ids: torch.Tensor, md: AttentionMetadata, kv) -> torch.Tensor | None:
        """Replay the graph for a decode-only step.

        Returns logits [nd, vocab] as a view into the bucket's static buffer: the caller
        must consume it (sample) before the next replay overwrites it. Returns None when
        no bucket covers this batch size; the caller then falls back to the eager forward.
        """
        nd = input_ids.shape[0]
        bucket = bucket_for(nd, self.buckets)
        if bucket is None:
            return None
        entry = self._graphs.get(bucket)
        if entry is None:
            entry = self._capture(bucket, input_ids, md, kv)
            self._graphs[bucket] = entry
        else:
            self._fill(entry.bufs, input_ids, md)
        entry.graph.replay()
        return entry.static_logits[:nd]

"""Paged KV-cache bookkeeping: virtual memory for the KV cache.

The GPU holds one big pool of fixed-size blocks (block_size tokens each). Every sequence owns a
block table, a list mapping its logical block i (tokens i*bs .. i*bs+bs-1) to a physical block id.
Physical blocks need not be contiguous, so the only waste is the unfilled tail of each sequence's
last block (< block_size tokens).

Two allocation policies share the same pool so they can be compared on equal terms:

    paged       allocate a block only when the previous one fills (vLLM)
    contiguous  reserve ceil(max_model_len / bs) blocks at admission, the way systems that
                pre-allocate a max-length KV buffer per request do (HF static batching, Orca-Max)

Prefix caching (optional): a FULL block is identified by hash(parent_hash, its tokens). When a new
request's prompt starts with blocks already in the cache, it maps those physical blocks into its own
table (ref_count += 1) and skips their prefill. Blocks whose ref_count drops to 0 stay cached in an
LRU list and are evicted only when the free list runs dry.

Copy-on-write: sequences only ever write into their LAST, partially filled block, and only FULL
blocks are shared, so shared blocks are read-only and CoW never triggers here. It becomes necessary
for parallel sampling / beam search, where a forked sequence shares a partially filled block.
"""
from __future__ import annotations

from collections import OrderedDict, deque

from .sequence import Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class NoFreeBlocksError(RuntimeError):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, mode: str = "paged", max_model_len: int = 2048,
                 enable_prefix_caching: bool = False, watermark: float = 0.01):
        assert mode in ("paged", "contiguous"), mode
        if mode == "contiguous" and enable_prefix_caching:
            raise ValueError("prefix caching requires paged mode")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.mode = mode
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        self.watermark_blocks = int(watermark * num_blocks)
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.ref_count = [0] * num_blocks
        # prefix cache
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()   # cached, ref_count == 0, LRU order
        # stats
        self.prefix_hit_tokens = 0
        self.prefix_query_tokens = 0

    # ------------------------------------------------------------------ capacity
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks) + len(self.evictable)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def blocks_needed_at_admission(self, seq: Sequence) -> int:
        if self.mode == "contiguous":
            return cdiv(self.max_model_len, self.block_size)
        return cdiv(seq.num_tokens, self.block_size)

    def can_allocate(self, seq: Sequence) -> bool:
        need = self.blocks_needed_at_admission(seq)
        if self.enable_prefix_caching:
            need -= self._count_cached_prefix_blocks(seq)
        return self.num_free_blocks - need >= self.watermark_blocks

    # ------------------------------------------------------------------ allocation
    def _pop_free_block(self) -> int:
        if self.free_blocks:
            b = self.free_blocks.popleft()
        elif self.evictable:
            b, _ = self.evictable.popitem(last=False)          # evict least recently used cached block
            h = self.block_to_hash.pop(b)
            del self.hash_to_block[h]
        else:
            raise NoFreeBlocksError
        assert self.ref_count[b] == 0
        self.ref_count[b] = 1
        return b

    def _block_hash(self, parent: int | None, tokens: list[int]) -> int:
        return hash((parent, tuple(tokens)))

    def _prefix_hashes(self, seq: Sequence) -> list[int]:
        hashes, parent, bs = [], None, self.block_size
        for i in range(seq.num_tokens // bs):
            parent = self._block_hash(parent, seq.token_ids[i * bs:(i + 1) * bs])
            hashes.append(parent)
        return hashes

    def _count_cached_prefix_blocks(self, seq: Sequence) -> int:
        n = 0
        for h in self._prefix_hashes(seq):
            if h not in self.hash_to_block:
                break
            n += 1
        if n * self.block_size >= seq.num_tokens:   # must leave >= 1 token to compute for logits
            n -= 1
        return max(n, 0)

    def allocate(self, seq: Sequence) -> None:
        """Give a waiting sequence its initial blocks (and any prefix-cache hits)."""
        assert not seq.block_table
        table: list[int] = []
        seq.block_hashes = []
        if self.enable_prefix_caching:
            hashes = self._prefix_hashes(seq)
            n_hit = self._count_cached_prefix_blocks(seq)
            for h in hashes[:n_hit]:
                b = self.hash_to_block[h]
                if self.ref_count[b] == 0:
                    self.evictable.pop(b)
                self.ref_count[b] += 1
                table.append(b)
                seq.block_hashes.append(h)
            seq.num_computed_tokens = n_hit * self.block_size
            seq.num_cached_prompt_tokens = seq.num_computed_tokens
            self.prefix_hit_tokens += seq.num_computed_tokens
            self.prefix_query_tokens += seq.num_tokens
        need = self.blocks_needed_at_admission(seq) - len(table)
        try:
            for _ in range(need):
                table.append(self._pop_free_block())
        except NoFreeBlocksError:
            seq.block_table = table
            self.free(seq)
            raise
        seq.block_table = table

    def can_append_slot(self, seq: Sequence) -> bool:
        return len(seq.block_table) * self.block_size >= seq.num_tokens or self.num_free_blocks > 0

    def append_slot(self, seq: Sequence) -> None:
        """Make sure every token of seq (including the one about to be written) has a slot."""
        while len(seq.block_table) * self.block_size < seq.num_tokens:
            if self.mode == "contiguous":
                raise RuntimeError("contiguous sequence outgrew its max_model_len reservation")
            seq.block_table.append(self._pop_free_block())

    def free(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.ref_count[b] -= 1
            assert self.ref_count[b] >= 0
            if self.ref_count[b] == 0:
                if b in self.block_to_hash:
                    self.evictable[b] = None                 # keep contents for future prefix hits
                else:
                    self.free_blocks.append(b)
        seq.block_table = []
        seq.block_hashes = []

    def register_full_blocks(self, seq: Sequence) -> None:
        """After a step: publish newly completed full blocks to the prefix cache."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        full = min(seq.num_computed_tokens // bs, len(seq.block_table))
        while len(seq.block_hashes) < full:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            h = self._block_hash(parent, seq.token_ids[i * bs:(i + 1) * bs])
            b = seq.block_table[i]
            if h not in self.hash_to_block and b not in self.block_to_hash:
                self.hash_to_block[h] = b
                self.block_to_hash[b] = h
            seq.block_hashes.append(h)

    # ------------------------------------------------------------------ helpers
    def slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": self.num_used_blocks,
            "free_blocks": len(self.free_blocks),
            "cached_evictable_blocks": len(self.evictable),
            "prefix_hit_rate": (self.prefix_hit_tokens / self.prefix_query_tokens) if self.prefix_query_tokens else 0.0,
        }

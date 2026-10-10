import pytest

from tiny_infer.block_manager import BlockManager, NoFreeBlocksError
from tiny_infer.sequence import SamplingParams, Sequence


def seq(n, start=10):
    return Sequence(list(range(start, start + n)), SamplingParams())


def test_paged_allocates_on_demand_and_frees():
    bm = BlockManager(10, 4)
    s = seq(5)                     # 5 tokens -> 2 blocks
    bm.allocate(s)
    assert len(s.block_table) == 2 and bm.num_free_blocks == 8
    s.token_ids += [1, 2, 3]       # 8 tokens: still fits in 2 blocks
    bm.append_slot(s)
    assert len(s.block_table) == 2
    s.token_ids.append(4)          # 9th token -> third block
    bm.append_slot(s)
    assert len(s.block_table) == 3
    bm.free(s)
    assert bm.num_free_blocks == 10 and all(r == 0 for r in bm.ref_count)


def test_waste_is_bounded_by_one_block():
    bm = BlockManager(100, 16)
    seqs = [seq(n) for n in (1, 17, 33, 100)]
    for s in seqs:
        bm.allocate(s)
    used = bm.num_used_blocks * 16
    live = sum(s.num_tokens for s in seqs)
    assert used - live < 16 * len(seqs)


def test_contiguous_reserves_max_len():
    bm = BlockManager(64, 16, mode="contiguous", max_model_len=256)
    s = seq(5)
    bm.allocate(s)
    assert len(s.block_table) == 16
    assert bm.can_allocate(seq(5)) is True
    for _ in range(3):
        bm.allocate(seq(5))
    assert not bm.can_allocate(seq(5))        # 4 x 16 blocks = pool exhausted


def test_block_tables_three_sequences_share_pool():
    bm = BlockManager(8, 4, watermark=0)
    a, b, c = seq(6), seq(3), seq(9)
    for s in (a, b, c):
        bm.allocate(s)
    physical = a.block_table + b.block_table + c.block_table
    assert len(set(physical)) == len(physical) == 6   # no block owned twice
    b_blocks = list(b.block_table)
    bm.free(b)
    for _ in range(2):                           # drain the never-used blocks first
        bm.allocate(seq(4))
    d = seq(4)
    bm.allocate(d)
    assert d.block_table == b_blocks             # freed block is immediately reusable by anyone


def test_out_of_blocks_raises_and_rolls_back():
    bm = BlockManager(2, 4, watermark=0)
    with pytest.raises(NoFreeBlocksError):
        bm.allocate(seq(12))
    assert bm.num_free_blocks == 2


def test_prefix_cache_hit_shares_blocks_and_survives_free():
    bm = BlockManager(16, 4, enable_prefix_caching=True, watermark=0)
    a = seq(10)
    bm.allocate(a)
    a.num_computed_tokens = 10
    bm.register_full_blocks(a)                # 2 full blocks published
    bm.free(a)
    assert len(bm.evictable) == 2             # cached, reusable, but counted as free
    b = Sequence(list(range(10, 20)) + [999], SamplingParams())
    assert bm.can_allocate(b)
    bm.allocate(b)
    assert b.num_computed_tokens == 8         # 2 blocks x 4 tokens skipped
    assert all(bm.ref_count[x] == 1 for x in b.block_table[:2]) and not bm.evictable


def test_full_prefix_hit_still_computes_last_token():
    bm = BlockManager(16, 4, enable_prefix_caching=True, watermark=0)
    a = seq(8)
    bm.allocate(a)
    a.num_computed_tokens = 8
    bm.register_full_blocks(a)
    b = seq(8)                                # identical prompt, exactly 2 full blocks
    bm.allocate(b)
    assert b.num_computed_tokens == 4         # last block recomputed so we have logits
    assert b.block_table[0] == a.block_table[0] and b.block_table[1] != a.block_table[1]
    assert bm.ref_count[a.block_table[0]] == 2


def test_eviction_reuses_cached_blocks_when_free_list_empty():
    bm = BlockManager(3, 4, enable_prefix_caching=True, watermark=0)
    a = seq(8)
    bm.allocate(a)
    a.num_computed_tokens = 8
    bm.register_full_blocks(a)
    bm.free(a)
    big = seq(12, start=500)                  # needs all 3 blocks -> must evict the cached two
    bm.allocate(big)
    assert len(big.block_table) == 3 and not bm.hash_to_block

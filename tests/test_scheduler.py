from tiny_infer.block_manager import BlockManager
from tiny_infer.config import EngineConfig
from tiny_infer.scheduler import Scheduler
from tiny_infer.sequence import SamplingParams, Sequence


def mk(num_blocks=64, block_size=4, **kw):
    cfg = EngineConfig(block_size=block_size, num_blocks=num_blocks, **kw)
    bm = BlockManager(num_blocks, block_size, mode=cfg.kv_mode, max_model_len=cfg.max_model_len, watermark=0)
    return Scheduler(cfg, bm)


def fake_step(sched, out):
    """Pretend the model ran: advance computed tokens, append a token to sequences that finished prefill."""
    for s, n in out.scheduled:
        s.num_computed_tokens += n
        if s.num_computed_tokens == s.num_tokens:
            s.token_ids.append(7)


def test_admits_fcfs_and_respects_max_num_seqs():
    sch = mk(max_num_seqs=2, max_num_batched_tokens=100)
    seqs = [Sequence([1] * 5, SamplingParams()) for _ in range(3)]
    for s in seqs:
        sch.add(s)
    out = sch.schedule()
    assert [s for s, _ in out.scheduled] == seqs[:2]
    assert sch.waiting[0] is seqs[2]


def test_finished_slot_is_reused_next_step():
    sch = mk(max_num_seqs=1, max_num_batched_tokens=100)
    a, b = Sequence([1] * 5, SamplingParams()), Sequence([1] * 5, SamplingParams())
    sch.add(a); sch.add(b)
    fake_step(sch, sch.schedule())
    sch.finish(a, "length")                      # a leaves...
    out = sch.schedule()
    assert [s for s, _ in out.scheduled] == [b]  # ...and b joins on the very next iteration


def test_decode_first_then_chunked_prefill_within_budget():
    sch = mk(max_num_batched_tokens=10)
    a = Sequence([1] * 4, SamplingParams())
    sch.add(a)
    fake_step(sch, sch.schedule())               # a prefilled, now decoding
    long = Sequence([2] * 30, SamplingParams())
    sch.add(long)
    out = sch.schedule()
    assert out.scheduled[0] == (a, 1)            # decode first
    assert out.scheduled[1] == (long, 9)         # prefill gets the rest of the budget
    fake_step(sch, out)
    out = sch.schedule()
    assert (long, 9) in out.scheduled            # chunk 2 continues


def test_preempts_newest_when_out_of_blocks():
    sch = mk(num_blocks=4, block_size=4, max_num_batched_tokens=100)
    a, b = Sequence([1] * 8, SamplingParams()), Sequence([1] * 8, SamplingParams())
    sch.add(a); sch.add(b)
    fake_step(sch, sch.schedule())               # both prefilled: 4/4 blocks used, 9 tokens each
    out = sch.schedule()                         # both need a 3rd block, none free
    assert out.preempted == [b]
    assert [s for s, _ in out.scheduled] == [a]
    assert b.num_computed_tokens == 0 and sch.waiting[0] is b and b.num_tokens == 9

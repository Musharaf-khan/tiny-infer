"""G2: CUDA-graph bucketing, padding, and the CPU fallback.

The capture/replay itself needs a GPU, so on this VM the tests cover everything
around it: bucket selection, the row-0 padding rule, the static-buffer fill that
capture replays from, the no-bucket fallback, the runner refusing non-CUDA,
and the engine accepting the flag on CPU and staying eager (token-identical to HF).
GPU verification of capture/replay itself is pending on his Colab T4.
"""
import pytest
import torch

from conftest import hf_greedy, make_engine, random_prompts
from tiny_infer.config import EngineConfig
from tiny_infer.cuda_graphs import CUDAGraphRunner, bucket_for, graph_batch_sizes, pad_decode_rows
from tiny_infer.model import AttentionMetadata, KVCache, LlamaModel
from tiny_infer.sequence import SamplingParams


def _bare_runner(buckets):
    """A CUDAGraphRunner without __init__ (which requires CUDA).

    _fill and run's no-bucket path are device-agnostic tensor ops, so they can be
    exercised on CPU; only capture/replay needs a GPU.
    """
    r = CUDAGraphRunner.__new__(CUDAGraphRunner)
    r.buckets = buckets
    r._graphs = {}
    return r


def _fill_inputs(n=3, w=2):
    """A 3-row decode step plus static buffers for bucket 4, block-table width 5."""
    md = AttentionMetadata(
        num_decode=n,
        decode_block_tables=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
        decode_context_lens=torch.tensor([7, 8, 9], dtype=torch.int32),
        max_decode_context=256, prefills=[],
        slot_mapping=torch.tensor([10, 20, 30], dtype=torch.long),
        positions=torch.tensor([5, 6, 7], dtype=torch.long),
        logits_indices=torch.arange(n, dtype=torch.long))
    bufs = {
        "input_ids": torch.zeros(4, dtype=torch.long),
        "positions": torch.zeros(4, dtype=torch.long),
        "slot_mapping": torch.zeros(4, dtype=torch.long),
        "block_tables": torch.zeros((4, 5), dtype=torch.int32),
        "context_lens": torch.zeros(4, dtype=torch.int32),
    }
    return md, bufs


def test_graph_batch_sizes():
    assert graph_batch_sizes(128) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert graph_batch_sizes(16) == [1, 2, 4, 8, 16]
    assert graph_batch_sizes(10) == [1, 2, 4, 8, 10]      # non-power-of-two max is a bucket itself
    assert graph_batch_sizes(1) == [1]
    with pytest.raises(ValueError):
        graph_batch_sizes(0)


def test_bucket_for():
    buckets = graph_batch_sizes(128)
    assert bucket_for(1, buckets) == 1
    assert bucket_for(3, buckets) == 4
    assert bucket_for(4, buckets) == 4
    assert bucket_for(128, buckets) == 128
    assert bucket_for(129, buckets) is None


def test_pad_decode_rows_duplicates_row_zero():
    t = torch.arange(6).reshape(3, 2)
    padded = pad_decode_rows(t, 5)
    assert padded.shape == (5, 2)
    assert torch.equal(padded[:3], t)
    assert torch.equal(padded[3], t[0])       # padded rows repeat row 0, never zeros
    assert torch.equal(padded[4], t[0])
    assert torch.equal(pad_decode_rows(t, 3), t)
    with pytest.raises(ValueError):
        pad_decode_rows(t, 2)
    with pytest.raises(ValueError):
        pad_decode_rows(torch.empty(0, 2), 4)


def test_runner_refuses_non_cuda(hf_model):
    model = LlamaModel.from_hf_model(hf_model, "cpu", torch.float32, fuse_weights=True,
                                     use_custom_kernels=True)
    kv = KVCache(model.cfg, 8, 4, torch.float32, torch.device("cpu"))
    cfg = EngineConfig(model="", device="cpu", max_num_seqs=16, max_model_len=256, block_size=4)
    with pytest.raises(RuntimeError, match="CUDA"):
        CUDAGraphRunner(model, cfg)


def test_engine_flag_falls_back_on_cpu(hf_model):
    """use_cuda_graphs=True on CPU: no runner, still eager, still token-identical to HF."""
    eng = make_engine(hf_model, use_cuda_graphs=True)
    assert eng.graph_runner is None
    prompts = random_prompts(3)
    got = [s.output_ids for s in eng.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True))]
    assert got == [hf_greedy(hf_model, p, 8) for p in prompts]


def test_engine_default_flag_off(hf_model):
    eng = make_engine(hf_model)
    assert eng.cfg.use_cuda_graphs is False
    assert eng.graph_runner is None


def test_server_cli_accepts_cuda_graphs():
    from tiny_infer.server import parse_args
    assert parse_args(["--cuda-graphs"]).cuda_graphs is True
    assert parse_args([]).cuda_graphs is False


# ------------------------------------------------------------------ T2: capture-path construction, CPU-safe

def test_graph_batch_sizes_small_max():
    assert graph_batch_sizes(2) == [1, 2]
    assert graph_batch_sizes(3) == [1, 2, 3]      # non-power-of-two max is a bucket itself
    assert bucket_for(1, []) is None              # no buckets at all: nothing fits


def test_pad_decode_rows_1d_and_identity():
    ids = torch.tensor([11, 22, 33], dtype=torch.long)
    padded = pad_decode_rows(ids, 4)
    assert padded.shape == (4,)
    assert padded.dtype == torch.long
    assert torch.equal(padded, torch.tensor([11, 22, 33, 11]))   # padded row repeats row 0
    t = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    assert pad_decode_rows(t, 3) is t              # exact fit: same tensor, no copy
    three_d = torch.arange(24).reshape(2, 3, 4)
    assert pad_decode_rows(three_d, 3).shape == (3, 3, 4)


def test_fill_copies_decode_values_into_static_buffers():
    """_fill is what capture replays from: every input copied in, row-0 padded.

    Block tables are narrower than the captured width (w=2 of max_blocks=5), so
    only the first w columns are written and the rest stay zero; and the buffers
    are mutated in place, so the pointers the graph captured stay valid.
    """
    runner = _bare_runner([1, 2, 4])
    md, bufs = _fill_inputs()
    orig = dict(bufs)                              # identity check below
    runner._fill(bufs, torch.tensor([11, 22, 33], dtype=torch.long), md)
    assert torch.equal(bufs["input_ids"], torch.tensor([11, 22, 33, 11]))
    assert torch.equal(bufs["positions"], torch.tensor([5, 6, 7, 5]))
    assert torch.equal(bufs["slot_mapping"], torch.tensor([10, 20, 30, 10]))
    assert torch.equal(bufs["context_lens"], torch.tensor([7, 8, 9, 7], dtype=torch.int32))
    assert torch.equal(bufs["block_tables"], torch.tensor(
        [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [5, 6, 0, 0, 0], [1, 2, 0, 0, 0]], dtype=torch.int32))
    for k, v in bufs.items():
        assert v is orig[k]                        # same tensors: captured pointers valid


def test_fill_exact_bucket_is_a_plain_copy():
    runner = _bare_runner([4])
    _, bufs = _fill_inputs()                       # bucket-4 static buffers, n == bucket below
    # n == bucket for a 4-row step: reuse the same buffers, no padding rows.
    md4 = AttentionMetadata(
        num_decode=4,
        decode_block_tables=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32),
        decode_context_lens=torch.tensor([7, 8, 9, 10], dtype=torch.int32),
        max_decode_context=256, prefills=[],
        slot_mapping=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        positions=torch.tensor([5, 6, 7, 8], dtype=torch.long),
        logits_indices=torch.arange(4, dtype=torch.long))
    runner._fill(bufs, torch.tensor([11, 22, 33, 44], dtype=torch.long), md4)
    assert torch.equal(bufs["input_ids"], torch.tensor([11, 22, 33, 44]))
    assert torch.equal(bufs["positions"], torch.tensor([5, 6, 7, 8]))


def test_run_returns_none_when_no_bucket_fits():
    """Batch bigger than the largest bucket: run() returns None, caller stays eager."""
    runner = _bare_runner([1, 2, 4])
    assert runner.run(torch.arange(5), None, None) is None
    assert runner._graphs == {}                   # nothing captured on the fallback path


def test_runner_refuses_non_cuda_regardless_of_kernel_flag(hf_model):
    """The device check comes first: non-CUDA is refused even without custom kernels."""
    model = LlamaModel.from_hf_model(hf_model, "cpu", torch.float32, fuse_weights=True,
                                     use_custom_kernels=False)
    cfg = EngineConfig(model="", device="cpu", max_num_seqs=16, max_model_len=256, block_size=4)
    with pytest.raises(RuntimeError, match="CUDA"):
        CUDAGraphRunner(model, cfg)

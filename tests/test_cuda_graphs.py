"""G2: CUDA-graph bucketing, padding, and the CPU fallback.

The capture/replay itself needs a GPU, so on this VM the tests cover everything
around it: bucket selection, the row-0 padding rule, the runner refusing non-CUDA,
and the engine accepting the flag on CPU and staying eager (token-identical to HF).
GPU verification of capture/replay itself is pending on his Colab T4.
"""
import pytest
import torch

from conftest import hf_greedy, make_engine, random_prompts
from tiny_infer.config import EngineConfig
from tiny_infer.cuda_graphs import CUDAGraphRunner, bucket_for, graph_batch_sizes, pad_decode_rows
from tiny_infer.model import KVCache, LlamaModel
from tiny_infer.sequence import SamplingParams


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

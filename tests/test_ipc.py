"""G3 slice 1: the engine runs in its own process, reached over JSON-lines IPC.

Covers the protocol (order, multiplexing, abort, errors, stats) against a fake
engine, then the real worker subprocess and the real HTTP server, both
token-identical to the HF oracle on the tiny CPU model.
"""
import asyncio
import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from conftest import hf_greedy, random_prompts
from tiny_infer.engine import StepEvent
from tiny_infer.ipc import RemoteEngine, serve_engine
from tiny_infer.sequence import SamplingParams, Sequence


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeEngine:
    """Deterministic stand-in for LLMEngine: one token per step per sequence."""
    def __init__(self, tokens_per_request=5, step_delay=0.0):
        self.tokens_per_request = tokens_per_request
        self.step_delay = step_delay
        self._seqs: dict[str, Sequence] = {}
        self._emitted: dict[str, int] = {}

    def add_request(self, prompt, params=None, request_id=None, arrival_time=None):
        params = params or SamplingParams()
        ids = list(prompt) if isinstance(prompt, list) else [ord(c) for c in prompt]
        if not ids:
            raise ValueError("empty prompt")
        seq = Sequence(ids, params, request_id, arrival_time)
        self._seqs[seq.request_id] = seq
        self._emitted[seq.request_id] = 0
        return seq

    def abort(self, request_id):
        self._seqs.pop(request_id, None)

    def has_unfinished(self):
        return bool(self._seqs)

    def step(self):
        if self.step_delay:
            time.sleep(self.step_delay)
        events = []
        for rid, seq in list(self._seqs.items()):
            n = self._emitted[rid] + 1
            self._emitted[rid] = n
            finished = n >= seq.params.max_tokens
            if finished:
                seq.finish_reason = "length"
                del self._seqs[rid]
            events.append(StepEvent(seq, 1000 + n, finished))
        return events

    def stats(self):
        return {"backend": "fake", "served": 1}


@pytest.fixture
def ipc_pair():
    eng = FakeEngine()
    port = free_port()
    t = threading.Thread(target=serve_engine, args=(eng, "127.0.0.1", port), daemon=True)
    t.start()
    client = RemoteEngine("127.0.0.1", port)
    yield client, eng
    client.close()
    t.join(timeout=10)
    assert not t.is_alive(), "engine worker thread did not exit after close"


async def collect(client, prompt, max_tokens, rid):
    out, prompt_tokens, finish = [], None, None
    async for ev in client.generate(prompt, SamplingParams(max_tokens=max_tokens, temperature=0,
                                                           ignore_eos=True), rid):
        if ev.error:
            return out, prompt_tokens, "error:" + ev.error
        if ev.token_id is None and not ev.finished:
            prompt_tokens = ev.prompt_tokens
        elif ev.token_id is not None:
            out.append(ev.token_id)
        if ev.finished:
            finish = ev.finish_reason
    return out, prompt_tokens, finish


def test_single_request_tokens_in_order(ipc_pair):
    client, _ = ipc_pair
    out, prompt_tokens, finish = asyncio.run(collect(client, [5, 6, 7], 5, "r1"))
    assert prompt_tokens == 3
    assert out == [1001, 1002, 1003, 1004, 1005]
    assert finish == "length"


def test_str_prompt(ipc_pair):
    client, _ = ipc_pair
    out, prompt_tokens, finish = asyncio.run(collect(client, "hello", 3, "r1"))
    assert prompt_tokens == 5 and out == [1001, 1002, 1003] and finish == "length"


def test_concurrent_requests_multiplexed(ipc_pair):
    client, _ = ipc_pair

    async def both():
        return await asyncio.gather(collect(client, [1], 4, "a"),
                                    collect(client, [2, 3], 4, "b"))

    (a_out, a_pt, a_fin), (b_out, b_pt, b_fin) = asyncio.run(both())
    assert (a_out, a_pt, a_fin) == ([1001, 1002, 1003, 1004], 1, "length")
    assert (b_out, b_pt, b_fin) == ([1001, 1002, 1003, 1004], 2, "length")


def test_abort_stops_stream(ipc_pair):
    client, eng = ipc_pair

    async def partial():
        n = 0
        async for ev in client.generate([9], SamplingParams(max_tokens=50, temperature=0,
                                                            ignore_eos=True), "r1"):
            if ev.token_id is not None:
                n += 1
            if n == 2:
                break  # client goes away mid-stream: generate() must abort server-side
        return n

    assert asyncio.run(partial()) == 2
    for _ in range(100):
        if not eng._seqs:
            break
        time.sleep(0.05)
    assert not eng._seqs, "aborted request still running in the engine"


def test_validation_error_propagates(ipc_pair):
    client, _ = ipc_pair
    _, _, finish = asyncio.run(collect(client, [], 5, "r1"))
    assert finish == "error:empty prompt"


def test_stats_round_trip(ipc_pair):
    client, _ = ipc_pair
    assert client.stats() == {"backend": "fake", "served": 1}


# ------------------------------------------------------------------ real worker
@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, hf_model):
    d = tmp_path_factory.mktemp("tiny-model")
    hf_model.save_pretrained(d)
    # offline: the tokenizer is already in the local HF cache; without this,
    # from_pretrained retries the hub for files the cached snapshot lacks
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok.save_pretrained(d)
    return str(d)


def incremental_decode(tok, ids):
    """Mirror tiny_infer.server's streaming detokenisation (see server.py events()).

    The random-weight test model emits token ids that decode to invalid UTF-8,
    and the server's incremental decode is not identical to a single full decode
    on such input (a trailing replacement char can swallow a real char: full
    decode gives '...he\uFFFD\uFFFD', incremental gives '...he~'). That is
    pre-existing server behaviour, not the IPC path, so the HTTP test compares
    against the incremental decode of the expected ids.
    """
    out_ids, sent = [], ""
    for t in ids:
        out_ids.append(t)
        text = tok.decode(out_ids, skip_special_tokens=True)
        if not text.endswith("\ufffd"):
            sent = text
    return sent


def worker_cmd(tiny_model_dir, port):
    return [sys.executable, "-m", "tiny_infer.engine_worker",
            "--model", tiny_model_dir, "--device", "cpu", "--dtype", "float32",
            "--block-size", "4", "--max-model-len", "256", "--max-num-seqs", "8",
            "--max-num-batched-tokens", "64",
            "--host", "127.0.0.1", "--port", str(port)]


def test_worker_subprocess_token_identical(tiny_model_dir, hf_model):
    port = free_port()
    proc = subprocess.Popen(worker_cmd(tiny_model_dir, port))
    try:
        client = RemoteEngine("127.0.0.1", port, connect_timeout=120)
        prompt = random_prompts(1, seed=3)[0]
        out, prompt_tokens, finish = asyncio.run(collect(client, prompt, 10, "e2e-1"))
        assert out == hf_greedy(hf_model, prompt, 10)
        assert prompt_tokens == len(prompt) and finish == "length"
        assert client.stats()["steps"] >= 1
        client.close()
    finally:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("engine worker did not exit after its server closed")
        assert proc.returncode == 0


def test_http_server_with_remote_engine(tiny_model_dir, hf_model):
    from tiny_infer.server import EngineBackend, build_app, parse_args
    args = parse_args(["--backend", "engine", "--engine-process",
                       "--model", tiny_model_dir, "--device", "cpu", "--dtype", "float32",
                       "--block-size", "4", "--max-model-len", "256", "--max-num-seqs", "8",
                       "--max-num-batched-tokens", "64"])
    backend = EngineBackend(args)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(build_app(backend), host="127.0.0.1", port=port,
                                        log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                httpx.get(f"{url}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        prompt = random_prompts(1, seed=3)[0]
        r = httpx.post(f"{url}/v1/completions",
                       json={"prompt": prompt, "max_tokens": 10, "temperature": 0,
                             "ignore_eos": True}, timeout=120).json()
        expected = hf_greedy(hf_model, prompt, 10)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tiny_model_dir)
        assert r["choices"][0]["text"] == incremental_decode(tok, expected)
        assert r["usage"]["completion_tokens"] == 10
        assert r["usage"]["prompt_tokens"] == len(prompt)
        assert r["choices"][0]["finish_reason"] == "length"
        assert httpx.get(f"{url}/stats", timeout=10).json()["steps"] >= 1
    finally:
        srv.should_exit = True
        backend.shutdown()


# ------------------------------------------------- T3: transport boundary
# G3 slice 1 covered order, multiplexing, abort, validation errors, stats and
# token-identical end-to-end runs. What it did not cover: long streams with no
# token loss across the boundary, malformed or unknown wire input, a dropped
# connection mid-stream, and the stats timeout path. Those are this slice.

def start_pair(tokens_per_request=5, step_delay=0.0):
    """ipc_pair without the fixture, for tests that need a bigger FakeEngine."""
    eng = FakeEngine(tokens_per_request, step_delay)
    port = free_port()
    t = threading.Thread(target=serve_engine, args=(eng, "127.0.0.1", port), daemon=True)
    t.start()
    return RemoteEngine("127.0.0.1", port), eng, t


def stop_pair(client, t):
    client.close()
    t.join(timeout=10)
    assert not t.is_alive(), "engine worker thread did not exit after close"


def test_long_stream_no_token_loss():
    client, _, t = start_pair(tokens_per_request=60)
    try:
        out, prompt_tokens, finish = asyncio.run(collect(client, [5, 6, 7], 60, "r-long"))
        assert prompt_tokens == 3
        assert out == list(range(1001, 1061)), "tokens lost, duplicated or reordered"
        assert finish == "length"
    finally:
        stop_pair(client, t)


def test_concurrent_long_streams_no_cross_talk():
    client, _, t = start_pair(tokens_per_request=20)
    try:
        async def all_four():
            return await asyncio.gather(
                collect(client, [1], 20, "a"), collect(client, [2], 20, "b"),
                collect(client, [3], 20, "c"), collect(client, [4], 20, "d"))
        results = asyncio.run(all_four())
        for (out, pt, fin), n in zip(results, (1, 2, 3, 4)):
            assert out == list(range(1001, 1021)), f"request {n} lost or mixed tokens"
            assert pt == 1 and fin == "length"
    finally:
        stop_pair(client, t)


def test_abort_unknown_request_id_is_noop(ipc_pair):
    client, _ = ipc_pair
    client._send({"type": "abort", "request_id": "never-existed"})
    out, _, finish = asyncio.run(collect(client, [5], 3, "r1"))
    assert out == [1001, 1002, 1003] and finish == "length"


def raw_server(port, handler):
    """Accept one connection and drive it by hand (no engine behind it)."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.settimeout(5)
        rf, wf = conn.makefile("rb"), conn.makefile("wb")
        try:
            handler(rf, wf)
        except OSError:
            pass
        finally:
            for f in (wf, rf):
                try:
                    f.close()
                except OSError:
                    pass
            conn.close()
            srv.close()
    threading.Thread(target=serve, daemon=True).start()


def test_malformed_and_unknown_wire_lines_ignored():
    port = free_port()

    def handler(rf, wf):
        rf.readline()  # the "add" for r1
        wf.write(b"this is not json\n")
        wf.write(b'{"type": "mystery", "request_id": "r1"}\n')
        wf.write(b'{"type": "prompt", "request_id": "r1", "prompt_tokens": 2}\n')
        wf.write(b'{"type": "token", "request_id": "r1", "token_id": 42}\n')
        wf.write(b'{"type": "done", "request_id": "r1", "finish_reason": "length"}\n')
        wf.flush()
        rf.readline()  # shutdown (or EOF when the client closes)

    raw_server(port, handler)
    client = RemoteEngine("127.0.0.1", port)
    try:
        out, prompt_tokens, finish = asyncio.run(collect(client, [5, 6], 1, "r1"))
        assert prompt_tokens == 2 and out == [42] and finish == "length"
    finally:
        client.close()


def test_connection_loss_mid_stream_errors_live_request():
    port = free_port()

    def handler(rf, wf):
        rf.readline()  # the "add" for r1
        wf.write(b'{"type": "prompt", "request_id": "r1", "prompt_tokens": 1}\n')
        wf.write(b'{"type": "token", "request_id": "r1", "token_id": 7}\n')
        wf.flush()
        # close without a done: the client must surface the loss, not hang

    raw_server(port, handler)
    client = RemoteEngine("127.0.0.1", port)
    try:
        out, _, finish = asyncio.run(collect(client, [5], 10, "r1"))
        assert out == [7]
        assert finish == "error:engine connection lost"
        assert client.stats() == {"error": "engine connection lost"}
    finally:
        client.close()


def test_stats_timeout_when_engine_never_answers():
    port = free_port()

    def handler(rf, wf):
        rf.readline()  # the "stats" request; never answered
        time.sleep(3)

    raw_server(port, handler)
    client = RemoteEngine("127.0.0.1", port)
    try:
        assert client.stats(timeout=0.5) == {"error": "engine stats timeout"}
    finally:
        client.close()


def test_close_with_live_request_shuts_worker_down():
    # Slow steps: the engine loop steps as fast as possible, so an instant fake
    # engine finishes a 50-token stream before the test can close mid-flight.
    client, eng, t = start_pair(tokens_per_request=50, step_delay=0.2)
    done = threading.Event()

    def run_stream():
        asyncio.run(collect(client, [5], 50, "r-live"))
        done.set()

    threading.Thread(target=run_stream, daemon=True).start()
    for _ in range(100):
        if eng._seqs:
            break
        time.sleep(0.05)
    assert eng._seqs, "request never started before close"
    client.close()
    t.join(timeout=10)
    assert not t.is_alive(), "worker did not exit after close with a live request"
    assert done.wait(timeout=5), "live stream never ended after close"

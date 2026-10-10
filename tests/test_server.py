"""End-to-end: real uvicorn server (tiny random model on CPU) + the benchmark harness over HTTP/SSE."""
import asyncio
import json
import socket
import threading
import time
import types

import httpx
import pytest
import uvicorn

from conftest import hf_greedy, make_engine, random_prompts


class ToyTokenizer:
    """Token id n <-> the word 'w<n>' (enough to exercise streaming detokenisation)."""
    eos_token_id = 2

    def encode(self, text):
        return [int(w[1:]) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f" w{i}" for i in ids if not (skip_special_tokens and i < 3))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def server(hf_model):
    from tiny_infer.async_engine import AsyncEngine
    from tiny_infer.server import build_app
    eng = make_engine(hf_model, max_num_seqs=8)
    eng.tokenizer = ToyTokenizer()
    backend = types.SimpleNamespace(name="tiny-infer", tokenizer=eng.tokenizer, engine=eng,
                                    aengine=AsyncEngine(eng), stats=eng.stats)
    backend.stream = lambda prompt, params, rid: backend.aengine.generate(prompt, params, rid)
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(build_app(backend), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}", backend
    srv.should_exit = True
    backend.aengine.shutdown()


def test_non_streaming_matches_hf(server, hf_model):
    url, _ = server
    prompt = random_prompts(1, seed=3)[0]
    r = httpx.post(f"{url}/v1/completions", json={"prompt": prompt, "max_tokens": 10, "temperature": 0,
                                                    "ignore_eos": True}, timeout=60).json()
    expected = hf_greedy(hf_model, prompt, 10)
    assert r["choices"][0]["text"] == ToyTokenizer().decode(expected)
    assert r["usage"]["completion_tokens"] == 10 and r["usage"]["prompt_tokens"] == len(prompt)


def test_streaming_sse_one_chunk_per_token(server):
    url, _ = server
    chunks = []
    with httpx.stream("POST", f"{url}/v1/completions",
                      json={"prompt": "w5 w6 w7", "max_tokens": 6, "temperature": 0, "stream": True,
                            "ignore_eos": True, "stream_options": {"include_usage": True}}, timeout=60) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                chunks.append(line[5:].strip())
    assert chunks[-1] == "[DONE]"
    body = [json.loads(c) for c in chunks[:-1]]
    assert len(body) == 6
    assert body[-1]["choices"][0]["finish_reason"] == "length" and body[-1]["usage"]["completion_tokens"] == 6


def test_validation_error_is_reported(server):
    url, _ = server
    r = httpx.post(f"{url}/v1/completions", json={"prompt": "w5", "max_tokens": 100000}, timeout=30)
    assert r.status_code == 400


def test_harness_concurrent_requests(server):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
    from harness import make_workload, run_closed_loop, summarize
    url, backend = server
    wl = make_workload(12, (5, 40), (4, 12), vocab=512)
    res, dur = asyncio.run(run_closed_loop(url, "tiny-infer", wl, 6, 120))
    s = summarize(res, dur, "smoke", 6, None)
    assert s["completed"] == 12 and s["error_rate"] == 0
    assert s["total_output_tokens"] == sum(m for _, m in wl)
    assert backend.engine.peak_running > 1          # requests really were batched together

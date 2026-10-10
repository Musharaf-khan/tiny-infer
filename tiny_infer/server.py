"""OpenAI-compatible HTTP server: POST /v1/completions (streaming via SSE).

Two backends behind the same API, so one benchmark client measures both:

    --backend naive    Phase 2: one request -> one HF generate() call, serialised by a lock.
                       Deliberately dumb; it exists to be beaten.
    --backend engine   Phases 3-5: tiny-infer (continuous batching + paged KV + optional kernels).

    python -m tiny_infer.server --backend engine --kv-cache-gb 1 --custom-kernels
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .sequence import SamplingParams

log = logging.getLogger("tiny_infer.server")


# ----------------------------------------------------------------------------- backends
class EngineBackend:
    def __init__(self, args):
        from .async_engine import AsyncEngine
        from .config import EngineConfig
        from .engine import LLMEngine
        cfg = EngineConfig(model=args.model, dtype=args.dtype, device=args.device, block_size=args.block_size,
                           max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                           max_num_batched_tokens=args.max_num_batched_tokens, kv_mode=args.kv_mode,
                           kv_cache_gb=args.kv_cache_gb, gpu_memory_utilization=args.gpu_memory_utilization,
                           enable_chunked_prefill=not args.no_chunked_prefill,
                           enable_prefix_caching=args.prefix_caching, use_custom_kernels=args.custom_kernels)
        self.engine = LLMEngine(cfg)
        self.tokenizer = self.engine.tokenizer
        self.aengine = AsyncEngine(self.engine)
        self.name = "tiny-infer"

    async def stream(self, prompt, params: SamplingParams, rid: str):
        async for ev in self.aengine.generate(prompt, params, rid):
            yield ev

    def stats(self):
        return self.engine.stats()


class _IdStreamer:
    """HF generate() streamer that hands token ids to an asyncio queue (first put() is the prompt)."""

    def __init__(self, loop, q):
        self.loop, self.q, self.first = loop, q, True

    def put(self, value):
        if self.first:
            self.first = False
            return
        for t in value.flatten().tolist():
            self.loop.call_soon_threadsafe(self.q.put_nowait, ("tok", t))

    def end(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, ("end", None))


class NaiveBackend:
    def __init__(self, args):
        from transformers import AutoTokenizer
        from .hf_utils import load_causal_lm
        dt = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = load_causal_lm(args.model, dt, args.device)
        self.device = args.device
        self.lock = threading.Lock()   # one generate() at a time: the GPU serves exactly one user
        self.max_model_len = args.max_model_len
        self.name = "naive-hf"
        self.served = 0

    async def stream(self, prompt, params: SamplingParams, rid: str):
        from .async_engine import TokenEvent
        from .sequence import Sequence
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if len(ids) + params.max_tokens > self.max_model_len:
            yield TokenEvent(None, True, "error", error="prompt + max_tokens > max_model_len")
            return
        yield TokenEvent(None, False, prompt_tokens=len(ids))
        loop, q = asyncio.get_running_loop(), asyncio.Queue()
        eos = self.tokenizer.eos_token_id

        def run():
            with self.lock:
                kw = dict(max_new_tokens=params.max_tokens, streamer=_IdStreamer(loop, q),
                          pad_token_id=eos, do_sample=params.temperature > 0)
                if params.temperature > 0:
                    kw.update(temperature=params.temperature, top_p=params.top_p)
                if params.ignore_eos:
                    kw.update(min_new_tokens=params.max_tokens)
                try:
                    with torch.inference_mode():
                        self.model.generate(torch.tensor([ids], device=self.device), **kw)
                except Exception as e:      # e.g. CUDA OOM
                    loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))
                self.served += 1

        threading.Thread(target=run, daemon=True).start()
        n = 0
        while True:
            kind, val = await q.get()
            if kind == "err":
                yield TokenEvent(None, True, "error", error=val)
                return
            if kind == "end":
                yield TokenEvent(None, True, "length" if n >= params.max_tokens else "stop")
                return
            n += 1
            yield TokenEvent(val, False)

    def stats(self):
        s = {"backend": "naive", "served": self.served}
        if torch.cuda.is_available():
            s["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        return s


# ----------------------------------------------------------------------------- app
def build_app(backend) -> FastAPI:
    app = FastAPI(title="tiny-infer")
    tok = backend.tokenizer

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": backend.name}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": backend.name, "object": "model"}]}

    @app.get("/stats")
    async def stats():
        return backend.stats()

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            if len(prompt) != 1:
                raise HTTPException(400, "batched prompts are not supported; send one request per prompt")
            prompt = prompt[0]
        if not prompt:
            raise HTTPException(400, "prompt is required")
        params = SamplingParams(max_tokens=int(body.get("max_tokens", 16)),
                                temperature=float(body.get("temperature", 1.0)),
                                top_p=float(body.get("top_p", 1.0)),
                                ignore_eos=bool(body.get("ignore_eos", False)))
        rid = f"cmpl-{uuid.uuid4().hex[:16]}"
        created, model_name = int(time.time()), body.get("model", backend.name)
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

        def chunk(text, finish=None, usage=None):
            d = {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                 "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}
            if usage is not None:
                d["usage"] = usage
            return f"data: {json.dumps(d)}\n\n"

        async def events():
            out_ids, sent, prompt_tokens = [], "", 0
            async for ev in backend.stream(prompt, params, rid):
                if ev.error:
                    yield ("error", ev.error)
                    return
                if ev.token_id is None and not ev.finished:
                    prompt_tokens = ev.prompt_tokens
                    continue
                delta = ""
                if ev.token_id is not None:
                    out_ids.append(ev.token_id)
                    text = tok.decode(out_ids, skip_special_tokens=True)
                    if not text.endswith("�"):        # wait for the rest of a multi-byte char
                        delta, sent = text[len(sent):], text
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(out_ids),
                         "total_tokens": prompt_tokens + len(out_ids)}
                yield ("token" if not ev.finished else "final", (delta, ev.finish_reason if ev.finished else None, usage))
                if ev.finished:
                    return

        if stream:
            async def sse():
                async for kind, val in events():
                    if kind == "error":
                        yield f"data: {json.dumps({'error': {'message': val}})}\n\n"
                        break
                    delta, finish, usage = val
                    yield chunk(delta, finish, usage if (finish and include_usage) else None)
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text, finish, usage = "", None, None
        async for kind, val in events():
            if kind == "error":
                raise HTTPException(400, val)
            text += val[0]
            finish, usage = val[1], val[2]
        return JSONResponse({"id": rid, "object": "text_completion", "created": created, "model": model_name,
                             "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                             "usage": usage})

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="tiny-infer OpenAI-compatible server")
    p.add_argument("--backend", choices=["engine", "naive"], default="engine")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--kv-mode", choices=["paged", "contiguous"], default="paged")
    p.add_argument("--kv-cache-gb", type=float, default=None,
                   help="fixed KV pool size (default: fill gpu_memory_utilization)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--custom-kernels", action="store_true", help="use the Project 1 CUDA kernels")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    backend = EngineBackend(args) if args.backend == "engine" else NaiveBackend(args)
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

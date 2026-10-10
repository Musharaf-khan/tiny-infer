"""Runs LLMEngine.step() in a background thread and streams tokens to asyncio consumers.

The engine loop never waits on HTTP: requests are handed over through a thread-safe queue, and each
step's tokens are pushed back to the right request's asyncio.Queue with call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass

from .engine import LLMEngine
from .sequence import SamplingParams


@dataclass
class TokenEvent:
    token_id: int | None
    finished: bool
    finish_reason: str | None = None
    prompt_tokens: int = 0
    error: str | None = None


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="engine-loop", daemon=True)
        self._thread.start()

    def _emit(self, rid: str, ev: TokenEvent) -> None:
        entry = self._streams.get(rid)
        if entry:
            loop, q = entry
            loop.call_soon_threadsafe(q.put_nowait, ev)
            if ev.finished:
                self._streams.pop(rid, None)

    def _loop(self) -> None:
        eng = self.engine
        while not self._stop:
            # drain new requests / aborts
            while True:
                try:
                    kind, rid, payload = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "add":
                    prompt, params, arrival = payload
                    try:
                        seq = eng.add_request(prompt, params, request_id=rid, arrival_time=arrival)
                        self._emit(rid, TokenEvent(None, False, prompt_tokens=seq.prompt_len))
                    except Exception as e:   # validation errors go back to the client
                        self._emit(rid, TokenEvent(None, True, "error", error=str(e)))
                else:
                    eng.abort(rid)
                    self._streams.pop(rid, None)
            if not eng.has_unfinished():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                continue
            try:
                events = eng.step()
            except Exception as e:           # surface engine crashes to every waiting client
                for rid in list(self._streams):
                    self._emit(rid, TokenEvent(None, True, "error", error=f"engine error: {e}"))
                raise
            for ev in events:
                self._emit(ev.seq.request_id, TokenEvent(ev.token_id, ev.finished, ev.seq.finish_reason))

    async def generate(self, prompt, params: SamplingParams, request_id: str):
        """Async iterator of TokenEvent for one request."""
        q: asyncio.Queue = asyncio.Queue()
        self._streams[request_id] = (asyncio.get_running_loop(), q)
        self._inbox.put(("add", request_id, (prompt, params, time.perf_counter())))
        self._wake.set()
        try:
            while True:
                ev = await q.get()
                yield ev
                if ev.finished:
                    return
        finally:
            if request_id in self._streams:          # client went away mid-stream
                self._inbox.put(("abort", request_id, None))
                self._wake.set()

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()

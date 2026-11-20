"""Engine <-> server IPC (G3 slice 1): the engine runs in its own process.

Wire format: one JSON object per line over a localhost TCP socket. The HTTP
server (tiny_infer.server) owns the engine worker process
(tiny_infer.engine_worker) and talks to it through RemoteEngine, a drop-in
for AsyncEngine.generate().

Client -> engine:
  {"type": "add", "request_id": str, "prompt": str | list[int],
   "params": {"max_tokens": int, "temperature": float, "top_p": float,
              "stop_token_ids": list[int], "ignore_eos": bool}}
  {"type": "abort", "request_id": str}
  {"type": "stats", "id": str}
  {"type": "shutdown"}

Engine -> client:
  {"type": "prompt", "request_id": str, "prompt_tokens": int}
  {"type": "token", "request_id": str, "token_id": int}
  {"type": "done", "request_id": str, "finish_reason": str}
  {"type": "error", "request_id": str, "error": str}
  {"type": "stats", "id": str, "stats": {...}}

The worker serves a single connection (its server's) and exits when that
connection closes. There is no reconnect yet: if the socket drops, every live
request gets an error event and the client reports the connection as lost.
That is a follow-up, not this slice.
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
import uuid

from .async_engine import AsyncEngine, TokenEvent
from .sequence import SamplingParams

log = logging.getLogger("tiny_infer.ipc")


def _params_to_dict(p: SamplingParams) -> dict:
    return {"max_tokens": p.max_tokens, "temperature": p.temperature, "top_p": p.top_p,
            "stop_token_ids": list(p.stop_token_ids), "ignore_eos": p.ignore_eos}


# ----------------------------------------------------------------------------- client
class RemoteEngine:
    """Client side of the engine IPC. Same generate() shape as AsyncEngine."""

    def __init__(self, host: str, port: int, sock: socket.socket | None = None,
                 connect_timeout: float = 120.0):
        self.host, self.port = host, port
        self._sock = sock or self._connect_with_retry(host, port, connect_timeout)
        self._rf = self._sock.makefile("rb")
        self._wf = self._sock.makefile("wb")
        self._streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._streams_lock = threading.Lock()
        self._pending_stats: dict[str, list] = {}   # id -> [threading.Event, stats-or-None]
        self._send_lock = threading.Lock()
        self._dead = False
        self._reader = threading.Thread(target=self._read_loop, name="engine-ipc-reader", daemon=True)
        self._reader.start()

    @staticmethod
    def _connect_with_retry(host, port, timeout):
        deadline = time.time() + timeout
        last_err: OSError | None = None
        while time.time() < deadline:
            try:
                return socket.create_connection((host, port), timeout=2)
            except OSError as e:
                last_err = e
                time.sleep(0.2)
        raise RuntimeError(f"could not connect to engine at {host}:{port}: {last_err}")

    # -- sending ---------------------------------------------------------------
    def _send(self, msg: dict) -> None:
        if self._dead:
            return
        data = (json.dumps(msg) + "\n").encode()
        try:
            with self._send_lock:
                self._wf.write(data)
                self._wf.flush()
        except OSError:
            self._dead = True

    # -- receiving -------------------------------------------------------------
    def _read_loop(self) -> None:
        try:
            for line in self._rf:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(msg)
        except (OSError, ValueError):
            pass
        finally:
            self._on_disconnect()

    def _dispatch(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "stats":
            entry = self._pending_stats.get(msg.get("id", ""))
            if entry is not None:
                entry[1] = msg.get("stats", {})
                entry[0].set()
            return
        rid = msg.get("request_id")
        if t == "prompt":
            ev = TokenEvent(None, False, prompt_tokens=msg.get("prompt_tokens", 0))
        elif t == "token":
            ev = TokenEvent(msg.get("token_id"), False)
        elif t == "done":
            ev = TokenEvent(None, True, msg.get("finish_reason"))
        elif t == "error":
            ev = TokenEvent(None, True, "error", error=msg.get("error", "engine error"))
        else:
            return
        with self._streams_lock:
            entry = self._streams.get(rid)
            if t in ("done", "error"):
                self._streams.pop(rid, None)
        if entry is not None:
            loop, q = entry
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:
                pass  # loop already closed; the consumer is gone

    def _on_disconnect(self) -> None:
        self._dead = True
        with self._streams_lock:
            streams = list(self._streams.items())
            self._streams.clear()
        for rid, (loop, q) in streams:
            try:
                loop.call_soon_threadsafe(
                    q.put_nowait, TokenEvent(None, True, "error", error="engine connection lost"))
            except RuntimeError:
                pass  # loop already closed; the consumer is gone

    # -- API (mirrors AsyncEngine) ---------------------------------------------
    async def generate(self, prompt, params: SamplingParams | None = None, request_id: str = ""):
        """Async iterator of TokenEvent for one request."""
        if self._dead:
            raise RuntimeError("engine connection lost")
        params = params or SamplingParams()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._streams_lock:
            self._streams[request_id] = (loop, q)
        self._send({"type": "add", "request_id": request_id, "prompt": prompt,
                    "params": _params_to_dict(params)})
        try:
            while True:
                ev = await q.get()
                yield ev
                if ev.finished:
                    return
        finally:
            with self._streams_lock:
                if request_id in self._streams:
                    del self._streams[request_id]
                    self._send({"type": "abort", "request_id": request_id})

    def stats(self, timeout: float = 5.0) -> dict:
        """Engine stats over IPC. Sync with a bounded wait: this is a diagnostics
        endpoint, and the round trip is normally sub-millisecond on localhost."""
        if self._dead:
            return {"error": "engine connection lost"}
        rid = uuid.uuid4().hex[:8]
        ev = threading.Event()
        with self._streams_lock:
            self._pending_stats[rid] = [ev, None]
        self._send({"type": "stats", "id": rid})
        if not ev.wait(timeout):
            with self._streams_lock:
                self._pending_stats.pop(rid, None)
            return {"error": "engine stats timeout"}
        with self._streams_lock:
            return self._pending_stats.pop(rid)[1] or {}

    def close(self) -> None:
        try:
            self._send({"type": "shutdown"})
        except OSError:
            pass
        self._dead = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._reader.join(timeout=2)


# ----------------------------------------------------------------------------- worker side
async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                         aengine: AsyncEngine) -> None:
    write_lock = asyncio.Lock()
    tasks: dict[str, asyncio.Task] = {}

    async def send(msg: dict) -> None:
        async with write_lock:
            writer.write((json.dumps(msg) + "\n").encode())
            await writer.drain()

    async def forward(request_id: str, prompt, params: SamplingParams) -> None:
        try:
            async for ev in aengine.generate(prompt, params, request_id):
                if ev.error:
                    await send({"type": "error", "request_id": request_id, "error": ev.error})
                elif ev.token_id is not None:
                    await send({"type": "token", "request_id": request_id, "token_id": ev.token_id})
                    if ev.finished:
                        await send({"type": "done", "request_id": request_id,
                                    "finish_reason": ev.finish_reason})
                elif ev.finished:
                    await send({"type": "done", "request_id": request_id,
                                "finish_reason": ev.finish_reason})
                else:
                    await send({"type": "prompt", "request_id": request_id,
                                "prompt_tokens": ev.prompt_tokens})
        except asyncio.CancelledError:
            raise
        except OSError:
            pass  # client went away mid-stream; the read loop below ends the session
        finally:
            tasks.pop(request_id, None)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break  # client disconnected
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "add":
                rid = msg["request_id"]
                tasks[rid] = asyncio.create_task(
                    forward(rid, msg["prompt"], SamplingParams(**msg["params"])))
            elif t == "abort":
                task = tasks.pop(msg["request_id"], None)
                if task is not None:
                    task.cancel()
            elif t == "stats":
                await send({"type": "stats", "id": msg["id"], "stats": aengine.engine.stats()})
            elif t == "shutdown":
                break
    finally:
        if tasks:
            for task in tasks.values():
                task.cancel()
            # let the cancellations land (each generate() sends its abort) before
            # the handler returns, so no task is torn down with the event loop
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        try:
            writer.close()
        except OSError:
            pass


async def _run_server(engine, host: str, port: int, connect_timeout: float = 120.0) -> None:
    aengine = AsyncEngine(engine)
    connected, done = asyncio.Event(), asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connected.set()
        try:
            await _handle_client(reader, writer, aengine)
        finally:
            done.set()

    server = await asyncio.start_server(handler, host, port)
    log.info("engine worker listening on %s:%d", host, port)
    async with server:
        try:
            await asyncio.wait_for(connected.wait(), timeout=connect_timeout)
        except TimeoutError:
            log.warning("engine worker: no server connected in %.0fs; exiting", connect_timeout)
            aengine.shutdown()
            return
        await done.wait()
    aengine.shutdown()


def serve_engine(engine, host: str, port: int, connect_timeout: float = 120.0) -> None:
    """Serve one engine over IPC until its server disconnects, then return."""
    asyncio.run(_run_server(engine, host, port, connect_timeout))

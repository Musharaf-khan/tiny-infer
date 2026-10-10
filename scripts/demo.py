"""Terminal demo for the screen recording: N users stream at once, with live per-user and total tok/s.

    python -m tiny_infer.server --custom-kernels &        # in one terminal
    python scripts/demo.py --users 8                      # in another
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time

import httpx

PROMPTS = [
    "Explain continuous batching to a five-year-old.",
    "Write a short poem about GPUs running out of memory.",
    "What is a KV cache and why does it grow during generation?",
    "Give three tips for writing fast CUDA kernels.",
    "Describe the city of Chicago in two sentences.",
    "Why is LLM decoding memory-bandwidth bound?",
    "Write a Python function that reverses a linked list.",
    "Summarize the plot of a heist movie you just made up.",
]


async def user(client, url, i, prompt, max_tokens, state):
    chat = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    st = state[i] = {"text": "", "n": 0, "t0": time.perf_counter(), "ttft": None, "done": False}
    async with client.stream("POST", f"{url}/v1/completions", json={
            "prompt": chat, "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.9, "stream": True}) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if not d.get("choices"):
                continue
            st["n"] += 1
            st["ttft"] = st["ttft"] or time.perf_counter() - st["t0"]
            st["text"] = (st["text"] + d["choices"][0]["text"]).replace("\n", " ")
    st["done"] = True


async def render(state, users, t0):
    width = shutil.get_terminal_size((120, 40)).columns
    while True:
        lines = ["\x1b[H\x1b[2J\x1b[1mtiny-infer live demo\x1b[0m  "
                 f"{users} concurrent users | total {sum(s['n'] for s in state.values()) / (time.perf_counter() - t0):6.1f} tok/s\n"]
        for i in range(users):
            s = state.get(i)
            if not s:
                continue
            el = time.perf_counter() - s["t0"]
            rate = s["n"] / max(el - (s["ttft"] or 0), 1e-6) if s["n"] > 1 else 0.0
            tag = "\x1b[32mdone\x1b[0m" if s["done"] else f"{rate:5.1f} t/s"
            ttft = f"{s['ttft'] * 1e3:5.0f}ms" if s["ttft"] else "  ... "
            head = f"user {i}  TTFT {ttft}  {s['n']:4d} tok  {tag} | "
            lines.append(head + s["text"][-(width - len(head) + 9):])
        print("\n".join(lines), flush=True)
        if state and all(s["done"] for s in state.values()) and len(state) == users:
            return
        await asyncio.sleep(0.1)


async def main_async(a):
    state: dict = {}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=600) as client:
        tasks = [user(client, a.url, i, PROMPTS[i % len(PROMPTS)], a.max_tokens, state) for i in range(a.users)]
        await asyncio.gather(render(state, a.users, t0), *tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--users", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=200)
    asyncio.run(main_async(p.parse_args()))

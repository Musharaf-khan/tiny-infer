"""Start a server, wait for /health, run the harness sweep, stop the server. One call per table row group.

    python scripts/run_serving_bench.py --label engine --server "python -m tiny_infer.server" -- --concurrency 1 4 16 32
    python scripts/run_serving_bench.py --label vllm --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --server "vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype half --port 8000" -- --concurrency 1 4 16 32
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))


def main():
    if "--" in sys.argv:
        i = sys.argv.index("--")
        own, harness_args = sys.argv[1:i], sys.argv[i + 1:]
    else:
        own, harness_args = sys.argv[1:], []
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--server", required=True, help="server command line")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="tiny-infer", help="model field sent in requests")
    p.add_argument("--startup-timeout", type=float, default=900)
    a = p.parse_args(own)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", f"server_{a.label}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(shlex.split(a.server), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                            start_new_session=True)
    url = f"http://localhost:{a.port}"
    t0 = time.time()
    try:
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"server exited during startup, see {log_path}:\n" + open(log_path).read()[-3000:])
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > a.startup_timeout:
                raise SystemExit("server did not become healthy in time")
            time.sleep(2)
        print(f"[{a.label}] server healthy after {time.time() - t0:.0f}s")
        from harness import main as harness_main
        harness_main(["--url", url, "--label", a.label, "--model", a.model_name,
                      "--out", os.path.join(ROOT, "results", "serving.jsonl")] + harness_args)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(3)          # let the GPU memory go before the next server starts


if __name__ == "__main__":
    main()

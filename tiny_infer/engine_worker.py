"""Engine worker process (G3): owns the LLMEngine, serves one server over IPC.

    python -m tiny_infer.engine_worker --model ... --device ... --host 127.0.0.1 --port 8123

Started by tiny_infer.server when --engine-process is given; not normally run
by hand. Exits when its server disconnects (or sends "shutdown").
"""
from __future__ import annotations

import logging

from .engine import LLMEngine
from .ipc import serve_engine
from .server import make_engine_config, parse_args

log = logging.getLogger("tiny_infer.engine_worker")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    engine = LLMEngine(make_engine_config(args))
    log.info("engine worker ready, serving on %s:%d", args.host, args.port)
    serve_engine(engine, args.host, args.port)


if __name__ == "__main__":
    main()

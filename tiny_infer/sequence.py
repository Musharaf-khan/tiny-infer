"""A request as the engine sees it.

One uniform rule drives prefill, decode, chunked prefill, prefix caching and preemption:

    token_ids            = prompt + generated tokens
    num_computed_tokens  = how many of those already have K/V in the cache

Each step processes token_ids[num_computed : num_computed + n]. When that reaches the end of
token_ids, the last position's logits are sampled and one token is appended. Decode is n = 1;
a whole-prompt prefill is n = len(prompt); a chunked prefill is several smaller n; a prefix-cache
hit simply starts num_computed above 0; a recompute-preemption resets it to 0.
"""
from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field

_ids = itertools.count()


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0          # 0 = greedy
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False          # benchmarks: force exactly max_tokens output tokens


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    def __init__(self, prompt_ids: list[int], params: SamplingParams, request_id: str | None = None,
                 arrival_time: float | None = None):
        self.seq_id = next(_ids)
        self.request_id = request_id or f"seq-{self.seq_id}"
        self.prompt_len = len(prompt_ids)
        self.token_ids: list[int] = list(prompt_ids)
        self.params = params
        self.status = SeqStatus.WAITING
        self.num_computed_tokens = 0
        self.block_table: list[int] = []
        self.block_hashes: list[int] = []     # prefix-caching hash chain of full blocks
        self.num_cached_prompt_tokens = 0     # prefix-cache hit length (stats)
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        # timing (perf_counter seconds)
        self.arrival_time = arrival_time if arrival_time is not None else time.perf_counter()
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.token_times: list[float] = []
        self.finish_time: float | None = None

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len:]

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_uncomputed(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """True while more than one token is waiting for its K/V (prompt, or recompute after preemption)."""
        return self.num_uncomputed > 1

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def __repr__(self) -> str:
        return (f"Seq({self.request_id}, {self.status.value}, tokens={self.num_tokens}, "
                f"computed={self.num_computed_tokens}, blocks={len(self.block_table)})")

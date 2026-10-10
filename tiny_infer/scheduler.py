"""Iteration-level scheduler (Orca / vLLM continuous batching).

Every engine step:

    1. decode     every running sequence that is generating gets 1 token of budget. If the KV pool
                  is out of blocks, PREEMPT the newest running sequence (free its blocks; it goes back
                  to the front of the queue and is recomputed later) until the older one fits.
    2. prefill    sequences part-way through a chunked prefill continue with the remaining budget.
    3. admit      waiting sequences join, FCFS, while the KV pool can hold their prompt (minus any
                  prefix-cache hit), the running set is below max_num_seqs and token budget remains.

The batch is rebuilt from scratch each step, so a finished sequence frees its slot and blocks at
once and a queued one starts on the very next iteration: no waiting for the slowest member of a
static batch. Decode-first ordering keeps TPOT smooth for running users; chunked prefill keeps one
long prompt from stalling them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .config import EngineConfig
from .sequence import Sequence, SeqStatus


@dataclass
class SchedulerOutput:
    scheduled: list[tuple[Sequence, int]] = field(default_factory=list)   # (seq, new tokens this step)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager):
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        # with chunking off a whole prompt must fit in one step's budget
        self.budget = cfg.max_num_batched_tokens if cfg.enable_chunked_prefill \
            else max(cfg.max_num_batched_tokens, cfg.max_model_len)

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        """Recompute-style preemption: drop the KV cache, keep the generated tokens."""
        self.running.remove(seq)
        self.bm.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)
        out.preempted.append(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.budget
        scheduled_ids: set[int] = set()

        # 1. decode: one token for each running sequence that is generating (oldest first)
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            if seq.num_uncomputed != 1:
                i += 1
                continue
            while not self.bm.can_append_slot(seq):
                victim = self.running[-1]
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq.status != SeqStatus.RUNNING:
                continue          # seq preempted itself; running[i] is now the next one
            self.bm.append_slot(seq)
            out.scheduled.append((seq, 1))
            scheduled_ids.add(seq.seq_id)
            budget -= 1
            i += 1

        # 2. continue partially-prefilled sequences
        for seq in self.running:
            if budget <= 0:
                break
            if seq.seq_id in scheduled_ids or seq.num_uncomputed <= 1:
                continue
            n = min(seq.num_uncomputed, budget)
            out.scheduled.append((seq, n))
            scheduled_ids.add(seq.seq_id)
            budget -= n

        # 3. admit new sequences, FCFS; no admissions in a step that had to preempt
        while (self.waiting and budget > 0 and not out.preempted
               and len(self.running) < self.cfg.max_num_seqs):
            seq = self.waiting[0]
            if not self.cfg.enable_chunked_prefill and seq.num_uncomputed > budget:
                break
            if not self.bm.can_allocate(seq):
                break
            self.waiting.popleft()
            self.bm.allocate(seq)                         # may set num_computed via prefix-cache hit
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            n = min(seq.num_uncomputed, budget)
            out.scheduled.append((seq, n))
            budget -= n
        return out

    def finish(self, seq: Sequence, reason: str) -> None:
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        self.bm.free(seq)
        self.running.remove(seq)

    def abort(self, request_id: str) -> None:
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.status, seq.finish_reason = SeqStatus.FINISHED, "abort"
        for seq in list(self.running):
            if seq.request_id == request_id:
                self.finish(seq, "abort")

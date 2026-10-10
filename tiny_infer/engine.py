"""LLMEngine: scheduler -> block manager -> model runner -> sampler, one iteration per step()."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from . import ops
from .block_manager import BlockManager, cdiv
from .config import EngineConfig, ModelConfig
from .model import AttentionMetadata, KVCache, LlamaModel
from .scheduler import Scheduler, SchedulerOutput
from .sequence import SamplingParams, Sequence, SeqStatus

log = logging.getLogger("tiny_infer")


@dataclass
class StepEvent:
    seq: Sequence
    token_id: int
    finished: bool


class LLMEngine:
    def __init__(self, cfg: EngineConfig, model: LlamaModel | None = None, tokenizer=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.use_custom_kernels:
            ops.load_kernels()
            if self.device.type == "cuda":     # refuse to serve with a kernel that disagrees with the reference
                from .kernels import self_test
                errs = self_test(str(self.device))
                bad = {k: v for k, v in errs.items() if not v < 2e-2}
                if bad:
                    raise RuntimeError(f"custom kernel self-test failed: {bad}")
                log.info("custom kernels: self-test passed (worst max|err| %.1e)", max(errs.values()))
        if model is None:
            model = LlamaModel.from_pretrained(cfg.model, self.device, cfg.torch_dtype,
                                               fuse_weights=cfg.fuse_weights,
                                               use_custom_kernels=cfg.use_custom_kernels)
        self.model = model
        self.model.use_custom_kernels = cfg.use_custom_kernels
        self.mcfg: ModelConfig = model.cfg
        if tokenizer is None and cfg.model:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.tokenizer = tokenizer
        eos = self.mcfg.eos_token_id if tokenizer is None or tokenizer.eos_token_id is None else tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
        if cfg.max_model_len > self.mcfg.max_position_embeddings:
            raise ValueError("max_model_len exceeds the model's max_position_embeddings")

        num_blocks = self._determine_num_blocks()
        self.kv = KVCache(self.mcfg, num_blocks, cfg.block_size, cfg.torch_dtype, self.device)
        self.bm = BlockManager(num_blocks, cfg.block_size, mode=cfg.kv_mode, max_model_len=cfg.max_model_len,
                               enable_prefix_caching=cfg.enable_prefix_caching)
        self.scheduler = Scheduler(cfg, self.bm)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        # stats
        self.num_steps = 0
        self.peak_running = 0
        self.frag_samples: list[float] = []
        self.step_times: list[tuple[int, int, float]] = []    # (decode tokens, prefill tokens, seconds)
        log.info("KV cache: %d blocks x %d tokens = %.2f GiB (%s)", num_blocks, cfg.block_size,
                 num_blocks * KVCache.bytes_per_block(self.mcfg, cfg.block_size, cfg.torch_dtype) / 2**30,
                 cfg.kv_mode)

    # ------------------------------------------------------------------ memory sizing
    def _determine_num_blocks(self) -> int:
        cfg = self.cfg
        bpb = KVCache.bytes_per_block(self.mcfg, cfg.block_size, cfg.torch_dtype)
        if cfg.num_blocks:
            return cfg.num_blocks
        if cfg.kv_cache_gb:
            return int(cfg.kv_cache_gb * 2**30 // bpb)
        if self.device.type != "cuda":
            return cdiv(cfg.max_model_len, cfg.block_size) * 8
        # profile one worst-case step to measure activation memory, give the rest to the KV pool
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n = min(cfg.max_num_batched_tokens, cfg.max_model_len)
        tmp = KVCache(self.mcfg, cdiv(n, cfg.block_size), cfg.block_size, cfg.torch_dtype, self.device)
        md = AttentionMetadata(
            num_decode=0, decode_block_tables=None, decode_context_lens=None, max_decode_context=0,
            prefills=[(0, n, 0, torch.arange(cdiv(n, cfg.block_size), dtype=torch.int32, device=self.device))],
            slot_mapping=torch.arange(n, device=self.device), positions=torch.arange(n, device=self.device),
            logits_indices=torch.tensor([n - 1], device=self.device))
        self.model.forward(torch.zeros(n, dtype=torch.long, device=self.device), md, tmp)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del tmp
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(self.device).total_memory
        avail = total * cfg.gpu_memory_utilization - peak
        if avail <= bpb:
            raise RuntimeError("no GPU memory left for the KV cache")
        return int(avail // bpb)

    # ------------------------------------------------------------------ requests
    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None,
                    request_id: str | None = None, arrival_time: float | None = None) -> Sequence:
        params = params or SamplingParams()
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) + params.max_tokens > self.cfg.max_model_len:
            raise ValueError(f"prompt ({len(ids)}) + max_tokens ({params.max_tokens}) > max_model_len "
                             f"({self.cfg.max_model_len})")
        seq = Sequence(ids, params, request_id, arrival_time)
        self.scheduler.add(seq)
        return seq

    def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    # ------------------------------------------------------------------ one iteration
    def _prepare(self, sched: SchedulerOutput):
        bs = self.cfg.block_size
        decode = [(s, n) for s, n in sched.scheduled if n == 1]
        prefill = [(s, n) for s, n in sched.scheduled if n > 1]
        order = decode + prefill
        ids, pos, slots, logits_idx, sample_seqs = [], [], [], [], []
        for s, n in order:
            start = s.num_computed_tokens
            for p in range(start, start + n):
                ids.append(s.token_ids[p])
                pos.append(p)
                slots.append(self.bm.slot(s, p))
            if start + n == s.num_tokens:            # reaches the end: sample from the last row
                logits_idx.append(len(ids) - 1)
                sample_seqs.append(s)
        dev = self.device
        tt = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)
        nd = len(decode)
        if nd:
            max_blocks = max(len(s.block_table) for s, _ in decode)
            bt = [s.block_table + [0] * (max_blocks - len(s.block_table)) for s, _ in decode]
            ctx = [s.num_computed_tokens + 1 for s, _ in decode]
            dbt, dctx, maxctx = tt(bt, torch.int32), tt(ctx, torch.int32), max(ctx)
        else:
            dbt = dctx = None
            maxctx = 0
        prefills, cursor = [], nd
        for s, n in prefill:
            prefills.append((cursor, n, s.num_computed_tokens, tt(s.block_table, torch.int32)))
            cursor += n
        md = AttentionMetadata(num_decode=nd, decode_block_tables=dbt, decode_context_lens=dctx,
                               max_decode_context=maxctx, prefills=prefills, slot_mapping=tt(slots),
                               positions=tt(pos), logits_indices=tt(logits_idx))
        return tt(ids), md, order, sample_seqs, nd, len(ids) - nd

    def _sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        greedy = all(s.params.temperature == 0 for s in seqs)
        if greedy:
            return logits.argmax(-1).tolist()
        out = []
        for row, s in zip(logits, seqs):
            p = s.params
            if p.temperature == 0:
                out.append(int(row.argmax()))
                continue
            probs = torch.softmax(row / p.temperature, -1)
            if p.top_p < 1.0:
                sp, si = probs.sort(descending=True)
                keep = sp.cumsum(-1) - sp < p.top_p
                sp = sp * keep
                out.append(int(si[torch.multinomial(sp / sp.sum(), 1, generator=self.generator)]))
            else:
                out.append(int(torch.multinomial(probs, 1, generator=self.generator)))
        return out

    def step(self) -> list[StepEvent]:
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return []
        t0 = time.perf_counter()
        for s, _ in sched.scheduled:
            if s.first_scheduled_time is None:
                s.first_scheduled_time = t0
        input_ids, md, order, sample_seqs, nd, npf = self._prepare(sched)
        logits = self.model.forward(input_ids, md, self.kv)
        tokens = self._sample(logits, sample_seqs) if sample_seqs else []   # .tolist() syncs the GPU
        now = time.perf_counter()
        self.step_times.append((nd, npf, now - t0))
        self.num_steps += 1

        for s, n in order:
            s.num_computed_tokens += n
            self.bm.register_full_blocks(s)
        events = []
        for s, tok in zip(sample_seqs, tokens):
            s.token_ids.append(tok)
            if s.first_token_time is None:
                s.first_token_time = now
            s.token_times.append(now)
            reason = None
            if not s.params.ignore_eos and tok in self.eos_ids:
                reason = "stop"
            elif tok in s.params.stop_token_ids:
                reason = "stop"
            elif s.num_output_tokens >= s.params.max_tokens or s.num_tokens >= self.cfg.max_model_len:
                reason = "length"
            if reason:
                s.finish_time = now
                self.scheduler.finish(s, reason)
            events.append(StepEvent(s, tok, reason is not None))
        # stats: concurrency and KV waste (allocated-but-unused token slots)
        running = self.scheduler.running
        self.peak_running = max(self.peak_running, len(running))
        used_slots = self.bm.num_used_blocks * self.cfg.block_size
        if used_slots and not self.cfg.enable_prefix_caching:
            live = sum(s.num_tokens for s in running)
            self.frag_samples.append(1.0 - live / used_slots)
        return events

    # ------------------------------------------------------------------ offline helper
    def generate(self, prompts: list[str | list[int]], params: SamplingParams | list[SamplingParams] | None = None
                 ) -> list[Sequence]:
        plist = params if isinstance(params, list) else [params or SamplingParams()] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_unfinished():
            self.step()
        return seqs

    def stats(self) -> dict:
        d = self.bm.stats()
        d.update(steps=self.num_steps, peak_running=self.peak_running,
                 preemptions=self.scheduler.num_preemptions,
                 mean_kv_waste=(sum(self.frag_samples) / len(self.frag_samples)) if self.frag_samples else 0.0,
                 kv_mode=self.cfg.kv_mode, custom_kernels=self.cfg.use_custom_kernels)
        return d

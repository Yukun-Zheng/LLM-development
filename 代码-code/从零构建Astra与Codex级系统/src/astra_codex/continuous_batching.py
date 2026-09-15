from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .model import DecoderOnlyTransformer
from .page_aware_decode import HeterogeneousPageAwareDecodeReference
from .physical_prefix_cache import PhysicalPrefixCacheEngine
from .scheduler import ReferenceRequestScheduler, RequestMetrics, RequestStatus


@dataclass(slots=True)
class ContinuousRequest:
    request_id: str
    prompt_ids: torch.Tensor
    generated_tokens: list[int] = field(default_factory=list)
    pending_token: int | None = None


@dataclass(frozen=True, slots=True)
class ContinuousIteration:
    decode_request_ids: tuple[str, ...]
    prefill_request_ids: tuple[str, ...]
    emitted_tokens: dict[str, int]


class ContinuousBatchingReferenceEngine:
    """Scheduler + prefix cache + page-aware decode reference serving loop.

    One iteration has two explicit lanes:

    1. active requests perform one heterogeneous page-aware decode step;
    2. waiting requests are then admitted for prefill, even while older requests
       remain in DECODE state.

    This demonstrates the lifecycle behind continuous batching without pretending
    that prefill/decode are already fused into one production GPU kernel. Prefill
    remains sequential in this reference implementation so physical-prefix reuse
    is easy to inspect. Decode QKV/FFN is batched and attention reads each live
    request's physical block table directly.

    The sampled token emitted at time t becomes that request's ``pending_token``;
    it is consumed into K/V on the next decode iteration. A request that reaches
    EOS or ``max_new_tokens`` is terminal immediately and its physical cache is
    released because the final emitted token never needs another model step.
    """

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        *,
        total_blocks: int = 256,
        block_size: int = 16,
        max_batch_size: int = 8,
        max_prefill_tokens: int = 4096,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.scheduler = ReferenceRequestScheduler(
            max_batch_size=max_batch_size,
            max_prefill_tokens=max_prefill_tokens,
        )
        self.prefix_cache = PhysicalPrefixCacheEngine(
            model,
            total_blocks=total_blocks,
            block_size=block_size,
        )
        self.decode_executor = HeterogeneousPageAwareDecodeReference(
            model,
            self.prefix_cache.pool,
        )
        self.eos_token_id = eos_token_id
        self.requests: dict[str, ContinuousRequest] = {}

    def submit(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        now: float | None = None,
    ) -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id: {request_id}")
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have shape [1,T] with T > 0")
        if prompt_ids.shape[1] + max_new_tokens > self.model.config.max_seq_len:
            raise ValueError("prompt + requested generation exceeds max_seq_len")
        parameter = next(self.model.parameters())
        prompt_ids = prompt_ids.to(device=parameter.device, dtype=torch.long).clone()
        self.requests[request_id] = ContinuousRequest(request_id, prompt_ids)
        self.scheduler.submit(
            request_id,
            prompt_tokens=int(prompt_ids.shape[1]),
            max_new_tokens=max_new_tokens,
            now=now,
        )

    def cancel(self, request_id: str, *, now: float | None = None) -> None:
        self.scheduler.cancel(request_id, now=now)
        request = self.requests[request_id]
        request.pending_token = None
        if request_id in self.prefix_cache.pool.allocator.requests:
            self.prefix_cache.release_request(request_id)

    def _is_eos(self, token: int) -> bool:
        return self.eos_token_id is not None and token == self.eos_token_id

    def _record_emitted_token(
        self,
        request_id: str,
        token: int,
        *,
        now: float,
    ) -> None:
        request = self.requests[request_id]
        request.generated_tokens.append(token)
        self.scheduler.mark_token(
            request_id,
            now=now,
            finished=self._is_eos(token),
        )
        state = self.scheduler.requests[request_id]
        if state.status is RequestStatus.FINISHED:
            request.pending_token = None
            if request_id in self.prefix_cache.pool.allocator.requests:
                self.prefix_cache.release_request(request_id)
        else:
            request.pending_token = token

    @staticmethod
    def _greedy(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B,V]")
        return logits.argmax(dim=-1)

    @torch.inference_mode()
    def step(self, *, now: float | None = None) -> ContinuousIteration | None:
        timestamp = time.time() if now is None else now
        emitted: dict[str, int] = {}
        decode_ids: tuple[str, ...] = ()
        prefill_ids: tuple[str, ...] = ()

        decode_batch = self.scheduler.next_decode_batch()
        if decode_batch is not None:
            decode_ids = decode_batch.request_ids
            pending: list[int] = []
            for request_id in decode_ids:
                token = self.requests[request_id].pending_token
                if token is None:
                    raise RuntimeError(
                        f"decoding request {request_id!r} has no pending token"
                    )
                pending.append(token)
            device = next(self.model.parameters()).device
            input_tokens = torch.tensor(pending, dtype=torch.long, device=device)[:, None]
            logits = self.decode_executor.decode_batch(list(decode_ids), input_tokens)
            sampled = self._greedy(logits).tolist()
            for request_id, token in zip(decode_ids, sampled, strict=True):
                value = int(token)
                emitted[request_id] = value
                self._record_emitted_token(request_id, value, now=timestamp)

        # Admission happens after the decode lane, so a request that arrived
        # while old requests are decoding does not have to wait for all of them
        # to finish. This is interleaved reference scheduling, not fused prefill.
        prefill_batch = self.scheduler.next_prefill_batch()
        if prefill_batch is not None:
            prefill_ids = prefill_batch.request_ids
            for request_id in prefill_ids:
                request = self.requests[request_id]
                result = self.prefix_cache.prefill(request_id, request.prompt_ids)
                self.scheduler.mark_prefill_complete((request_id,), now=timestamp)
                token = int(self._greedy(result.logits)[0])
                emitted[request_id] = token
                self._record_emitted_token(request_id, token, now=timestamp)

        if not decode_ids and not prefill_ids:
            return None
        return ContinuousIteration(decode_ids, prefill_ids, emitted)

    def run_until_idle(
        self,
        *,
        start_time: float | None = None,
        tick_seconds: float = 1.0,
        max_iterations: int = 10_000,
    ) -> list[ContinuousIteration]:
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive")
        timestamp = time.time() if start_time is None else start_time
        iterations: list[ContinuousIteration] = []
        for _ in range(max_iterations):
            item = self.step(now=timestamp)
            if item is None:
                return iterations
            iterations.append(item)
            timestamp += tick_seconds
        raise RuntimeError("continuous batching engine exceeded max_iterations")

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        return tuple(self.requests[request_id].generated_tokens)

    def metrics(self, request_id: str) -> RequestMetrics:
        return self.scheduler.metrics(request_id)

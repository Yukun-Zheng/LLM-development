from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class BatchKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(slots=True)
class RequestState:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_time: float
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: int = 0
    prefill_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    kind: BatchKind
    request_ids: tuple[str, ...]
    token_count: int


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float | None
    tpot_s: float | None
    total_latency_s: float | None


class ReferenceRequestScheduler:
    """Deterministic decode-first scheduler for teaching continuous batching.

    The scheduler models request lifecycle and admission independently from the
    model executor. A later engine will consume ``ScheduledBatch`` objects and
    perform batched prefill/decode. Keeping this layer separate makes queueing,
    fairness and serving metrics inspectable before kernel optimization.
    """

    def __init__(
        self,
        *,
        max_batch_size: int = 8,
        max_prefill_tokens: int = 4096,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be positive")
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.requests: dict[str, RequestState] = {}

    def submit(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        max_new_tokens: int,
        now: float | None = None,
    ) -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id: {request_id}")
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.requests[request_id] = RequestState(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            arrival_time=time.time() if now is None else now,
        )

    def cancel(self, request_id: str, *, now: float | None = None) -> None:
        request = self.requests[request_id]
        if request.status in {RequestStatus.FINISHED, RequestStatus.CANCELLED}:
            raise RuntimeError(f"request already terminal: {request.status.value}")
        request.status = RequestStatus.CANCELLED
        request.finish_time = time.time() if now is None else now

    def next_batch(self) -> ScheduledBatch | None:
        decode = [
            request
            for request in self.requests.values()
            if request.status is RequestStatus.DECODE
        ]
        decode.sort(key=lambda item: (item.arrival_time, item.request_id))
        if decode:
            chosen = decode[: self.max_batch_size]
            return ScheduledBatch(
                BatchKind.DECODE,
                tuple(item.request_id for item in chosen),
                len(chosen),
            )

        waiting = [
            request
            for request in self.requests.values()
            if request.status is RequestStatus.WAITING
        ]
        waiting.sort(key=lambda item: (item.arrival_time, item.request_id))
        chosen: list[RequestState] = []
        total_tokens = 0
        for request in waiting:
            if len(chosen) >= self.max_batch_size:
                break
            if total_tokens + request.prompt_tokens > self.max_prefill_tokens:
                continue
            chosen.append(request)
            total_tokens += request.prompt_tokens

        if not chosen:
            return None
        return ScheduledBatch(
            BatchKind.PREFILL,
            tuple(item.request_id for item in chosen),
            total_tokens,
        )

    def mark_prefill_complete(
        self, request_ids: tuple[str, ...], *, now: float | None = None
    ) -> None:
        timestamp = time.time() if now is None else now
        for request_id in request_ids:
            request = self.requests[request_id]
            if request.status is not RequestStatus.WAITING:
                raise RuntimeError(
                    f"request {request_id} is not waiting: {request.status.value}"
                )
            request.prefill_time = timestamp
            request.status = RequestStatus.DECODE

    def mark_token(
        self,
        request_id: str,
        *,
        now: float | None = None,
        finished: bool = False,
    ) -> None:
        timestamp = time.time() if now is None else now
        request = self.requests[request_id]
        if request.status is not RequestStatus.DECODE:
            raise RuntimeError(
                f"request {request_id} is not decoding: {request.status.value}"
            )
        request.generated_tokens += 1
        if request.first_token_time is None:
            request.first_token_time = timestamp
        request.last_token_time = timestamp
        if finished or request.generated_tokens >= request.max_new_tokens:
            request.status = RequestStatus.FINISHED
            request.finish_time = timestamp

    def metrics(self, request_id: str) -> RequestMetrics:
        request = self.requests[request_id]
        ttft = (
            None
            if request.first_token_time is None
            else request.first_token_time - request.arrival_time
        )
        if request.generated_tokens <= 1 or request.first_token_time is None:
            tpot = None
        else:
            assert request.last_token_time is not None
            tpot = (request.last_token_time - request.first_token_time) / (
                request.generated_tokens - 1
            )
        total_latency = (
            None
            if request.finish_time is None
            else request.finish_time - request.arrival_time
        )
        return RequestMetrics(
            request_id=request.request_id,
            prompt_tokens=request.prompt_tokens,
            generated_tokens=request.generated_tokens,
            ttft_s=ttft,
            tpot_s=tpot,
            total_latency_s=total_latency,
        )

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .agent import Message, ModelBackend
from .runtime_queue import DurableWorkQueue, WorkStatus
from .steering import DurableSteeringQueue
from .structured import ToolSpec


@dataclass(slots=True)
class LeaseHeartbeat:
    """Explicit lease-renewal primitive for long-running workers.

    A work item that takes longer than its original lease must renew ownership;
    otherwise another worker may legally reclaim it. This class makes that
    lifecycle visible instead of hiding it in a background thread.
    """

    queue: DurableWorkQueue
    item_id: str
    worker_id: str
    lease_seconds: float

    def renew(self, *, now: float | None = None) -> float:
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else now
        current = self.queue.get(self.item_id)
        if current.status is not WorkStatus.LEASED:
            raise RuntimeError(f"work item is not leased: {current.status.value}")
        if current.lease_owner != self.worker_id:
            raise PermissionError(
                f"lease belongs to {current.lease_owner!r}, not {self.worker_id!r}"
            )
        if current.lease_until is not None and current.lease_until < timestamp:
            raise RuntimeError("cannot renew an already expired lease")

        lease_until = timestamp + self.lease_seconds
        self.queue.connection.execute(
            """
            UPDATE work_items
            SET lease_until = ?, updated_at = ?
            WHERE item_id = ? AND status = ? AND lease_owner = ?
            """,
            (
                lease_until,
                timestamp,
                self.item_id,
                WorkStatus.LEASED.value,
                self.worker_id,
            ),
        )
        return lease_until


class ControlledBackend:
    """ModelBackend decorator for worker heartbeat + durable live steering.

    Before every model sampling call it can:

    1. renew the current worker lease;
    2. atomically consume pending steering messages for the thread;
    3. append those messages to the same transcript seen by the harness.

    This means steering submitted while a tool is executing becomes visible to
    the *next* sampling step without restarting the turn.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        heartbeat: Callable[[], object] | None = None,
        steering_queue: DurableSteeringQueue | None = None,
        thread_id: str | None = None,
        steering_prefix: str = "[Steering update] ",
    ) -> None:
        if steering_queue is not None and not thread_id:
            raise ValueError("thread_id is required when steering_queue is set")
        self.backend = backend
        self.heartbeat = heartbeat
        self.steering_queue = steering_queue
        self.thread_id = thread_id
        self.steering_prefix = steering_prefix

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        if self.heartbeat is not None:
            self.heartbeat()

        if self.steering_queue is not None:
            assert self.thread_id is not None
            pending = self.steering_queue.consume_pending(self.thread_id)
            for steering in pending:
                messages.append(
                    Message("user", self.steering_prefix + steering.content)
                )

        return self.backend.generate(messages, tools)

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .agent import Message, ModelBackend
from .runtime_queue import DurableWorkQueue
from .steering import DurableSteeringQueue
from .structured import ToolSpec


@dataclass(slots=True)
class LeaseHeartbeat:
    """Explicit lease-renewal primitive for long-running workers."""

    queue: DurableWorkQueue
    item_id: str
    worker_id: str
    lease_seconds: float

    def renew(self, *, now: float | None = None) -> float:
        return self.queue.renew_lease(
            self.item_id,
            self.worker_id,
            lease_seconds=self.lease_seconds,
            now=now,
        )


class BackgroundLeaseHeartbeat:
    """Renew a lease on a dedicated thread using a thread-local SQLite handle.

    This protects a worker while one model/tool call blocks longer than the
    foreground sampling cadence. The runner opens its own ``DurableWorkQueue``
    connection instead of sharing the main thread's SQLite connection.

    ``last_error`` is retained rather than thrown from the background thread; a
    caller can inspect it after the context closes and decide whether to fail the
    enclosing turn. The runtime currently uses this as a liveness guard, not a
    distributed fencing-token protocol.
    """

    def __init__(
        self,
        queue_path: str,
        item_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        interval_seconds: float | None = None,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        interval = lease_seconds / 3 if interval_seconds is None else interval_seconds
        if interval <= 0:
            raise ValueError("interval_seconds must be positive")
        if interval >= lease_seconds:
            raise ValueError("heartbeat interval must be shorter than lease_seconds")
        self.queue_path = queue_path
        self.item_id = item_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.renewals = 0
        self.last_error: Exception | None = None

    def _run(self) -> None:
        try:
            with DurableWorkQueue(self.queue_path) as queue:
                while not self._stop.wait(self.interval_seconds):
                    queue.renew_lease(
                        self.item_id,
                        self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                    self.renewals += 1
        except Exception as exc:
            self.last_error = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("background heartbeat is already running")
        self._stop.clear()
        self.last_error = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-heartbeat-{self.item_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
            self._thread = None

    def __enter__(self) -> "BackgroundLeaseHeartbeat":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ControlledBackend:
    """ModelBackend decorator for foreground heartbeat + durable live steering.

    Before every model sampling call it can:

    1. renew the current worker lease;
    2. atomically consume pending steering messages for the thread;
    3. append those messages to the same transcript seen by the harness.

    The explicit foreground heartbeat remains useful for deterministic tests and
    event visibility. ``BackgroundLeaseHeartbeat`` covers long blocking calls
    between these boundaries.
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

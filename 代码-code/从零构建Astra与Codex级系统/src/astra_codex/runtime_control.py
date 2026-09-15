from __future__ import annotations

import threading
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

    Starting a heartbeat has a two-phase arming protocol:

    1. synchronously renew the lease on a short-lived caller-thread connection;
    2. start the background thread, have it renew once immediately, then signal
       ``ready`` before ``start`` returns.

    The caller therefore does not enter a long blocking model/tool call merely
    because a heartbeat thread was *scheduled*; it waits until a renewal has
    actually succeeded. This closes the startup race exposed by short-lease CI
    tests and mirrors the more general distributed-systems rule that liveness
    protection must be active before protected work begins.

    The runner opens its own ``DurableWorkQueue`` connection instead of sharing
    the main thread's SQLite connection. ``last_error`` is retained rather than
    thrown asynchronously. This is still a liveness guard, not a fencing-token
    or consensus protocol.
    """

    def __init__(
        self,
        queue_path: str,
        item_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        interval_seconds: float | None = None,
        startup_timeout_seconds: float = 5.0,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        interval = lease_seconds / 3 if interval_seconds is None else interval_seconds
        if interval <= 0:
            raise ValueError("interval_seconds must be positive")
        if interval >= lease_seconds:
            raise ValueError("heartbeat interval must be shorter than lease_seconds")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        self.queue_path = queue_path
        self.item_id = item_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval
        self.startup_timeout_seconds = startup_timeout_seconds
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self.renewals = 0
        self.last_error: Exception | None = None

    def _renew_once(self, queue: DurableWorkQueue) -> None:
        queue.renew_lease(
            self.item_id,
            self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        self.renewals += 1

    def _run(self) -> None:
        try:
            with DurableWorkQueue(self.queue_path) as queue:
                # Do not wait for the first interval. A heartbeat thread that has
                # merely started but has never renewed is not yet protecting the
                # lease.
                self._renew_once(queue)
                self._ready.set()
                while not self._stop.wait(self.interval_seconds):
                    self._renew_once(queue)
        except Exception as exc:
            self.last_error = exc
            self._ready.set()
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("background heartbeat is already running")
        self._stop.clear()
        self._ready.clear()
        self.last_error = None
        self.renewals = 0

        # Arm synchronously first. This makes thread scheduling latency unable to
        # consume the entire lease before the heartbeat worker even opens SQLite.
        with DurableWorkQueue(self.queue_path) as queue:
            self._renew_once(queue)

        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-heartbeat-{self.item_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=self.startup_timeout_seconds):
            self.close()
            raise RuntimeError("background heartbeat failed to become ready")
        if self.last_error is not None:
            error = self.last_error
            self.close()
            raise RuntimeError(f"background heartbeat failed to arm: {error}") from error

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

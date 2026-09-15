from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ModelBackend
from .artifacts import ArtifactStore
from .codex_harness import CodexHarness, CodexTurnResult, HarnessEvent
from .durable import DurableThreadStore, TERMINAL_THREAD_STATUSES, ThreadStatus
from .event_stream import DurableEventStream
from .runtime_control import BackgroundLeaseHeartbeat, ControlledBackend, LeaseHeartbeat
from .runtime_queue import DurableWorkQueue, WorkItem, WorkStatus
from .steering import DurableSteeringQueue
from .tool_journal import (
    DurableToolJournal,
    JournaledToolExecutor,
    TurnScopedJournaledTools,
)
from .tools import ToolRegistry


@dataclass(frozen=True, slots=True)
class RuntimeExecutionRecord:
    work_item_id: str
    thread_id: str
    turn_id: str | None
    status: str
    final_answer: str | None
    model_steps: int | None


class DurableAgentRuntime:
    """Reference composition of durable thread, work, tool and steering state.

    The durable pipeline is:

        submission -> thread event -> work item -> worker lease -> turn/checkpoint
        -> journaled tool calls -> final checkpoint -> turn completion -> work ACK

    The runtime owns a durable steering inbox, content-addressed artifact store
    and an append-only *control-plane* event feed. ``DurableThreadStore`` remains
    the source of truth for reconstructing thread state; ``DurableEventStream``
    is a replayable integration surface for UIs, IDEs and remote clients.

    A foreground heartbeat is emitted at model sampling boundaries. For normal
    real-time execution (``now is None``), a separate background heartbeat also
    keeps the work lease alive while one model or tool call blocks longer than a
    sampling interval. Synthetic-clock tests intentionally skip the background
    thread so time remains deterministic.

    Completed tool calls are keyed by the stable work-item id and call index. If
    a worker disappears and the lease is later reclaimed, completed calls can be
    replayed from the journal without repeating their side effects. A tool call
    left only in STARTED state is *in doubt* and is not retried by default.

    This still does not provide universal exactly-once external side effects or
    an OS security sandbox. Remote side effects may need external idempotency or
    reconciliation, and process/filesystem/network isolation remains a separate
    subsystem.
    """

    def __init__(
        self,
        state_dir: str | Path,
        backend: ModelBackend,
        tools: ToolRegistry,
        *,
        max_model_steps: int = 32,
        retryable_in_doubt_tools: frozenset[str] = frozenset(),
    ) -> None:
        root = Path(state_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.state_dir = root
        self.thread_store = DurableThreadStore(root / "threads.sqlite")
        self.work_queue = DurableWorkQueue(root / "work.sqlite")
        self.tool_journal = DurableToolJournal(root / "tool-journal.sqlite")
        self.steering_queue = DurableSteeringQueue(root / "steering.sqlite")
        self.artifact_store = ArtifactStore(root / "artifacts")
        self.event_stream = DurableEventStream(root / "events.sqlite")
        self.journaled_tools = JournaledToolExecutor(
            tools,
            self.tool_journal,
            retryable_in_doubt_tools=retryable_in_doubt_tools,
        )
        self.backend = backend
        self.max_model_steps = max_model_steps

    def _emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        now: float | None = None,
    ) -> int:
        return self.event_stream.append(
            topic,
            payload,
            thread_id=thread_id,
            turn_id=turn_id,
            now=now,
        )

    def create_thread(self, thread_id: str | None = None) -> str:
        created = self.thread_store.create_thread(thread_id)
        self._emit("thread.created", {"threadId": created}, thread_id=created)
        return created

    def submit(
        self,
        thread_id: str,
        content: str,
        *,
        item_id: str | None = None,
        now: float | None = None,
    ) -> str:
        self.thread_store.submit(thread_id, content)
        created = self.work_queue.enqueue(
            thread_id,
            "turn",
            {"content": content},
            item_id=item_id,
            now=now,
        )
        self._emit(
            "thread.submitted",
            {"workItemId": created, "content": content},
            thread_id=thread_id,
            now=now,
        )
        return created

    def steer(
        self,
        thread_id: str,
        content: str,
        *,
        steering_id: str | None = None,
        now: float | None = None,
    ) -> str:
        projection = self.thread_store.project(thread_id)
        if projection.status in TERMINAL_THREAD_STATUSES:
            raise RuntimeError(
                f"cannot steer terminal thread: {projection.status.value}"
            )
        created = self.steering_queue.submit(
            thread_id,
            content,
            steering_id=steering_id,
            now=now,
        )
        self._emit(
            "thread.steering_submitted",
            {"steeringId": created, "content": content},
            thread_id=thread_id,
            now=now,
        )
        return created

    def snapshot_artifact(
        self,
        thread_id: str,
        source: str | Path,
        *,
        kind: str = "file",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        now: float | None = None,
    ) -> str:
        self.thread_store.project(thread_id)
        created = self.artifact_store.snapshot_file(
            thread_id,
            source,
            kind=kind,
            metadata=metadata,
            artifact_id=artifact_id,
            now=now,
        )
        record = self.artifact_store.get(created)
        self._emit(
            "artifact.created",
            {
                "artifactId": created,
                "kind": record.kind,
                "sha256": record.sha256,
                "sizeBytes": record.size_bytes,
            },
            thread_id=thread_id,
            now=now,
        )
        return created

    @staticmethod
    def _checkpoint_state(projection) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        checkpoint = projection.last_checkpoint
        if checkpoint is None:
            return {}
        state = checkpoint.get("state")
        return state if isinstance(state, dict) else {}

    def _finalize_finished_checkpoint(
        self,
        item: WorkItem,
        worker_id: str,
        turn_id: str,
        state: dict[str, Any],
        *,
        now: float | None,
    ) -> RuntimeExecutionRecord:
        final_answer = str(state.get("final_answer", ""))
        model_steps = int(state.get("model_steps", 0))
        stopped_by_limit = bool(state.get("stopped_by_limit", False))

        self.thread_store.complete_turn(
            item.thread_id,
            turn_id,
            summary=final_answer,
        )
        if stopped_by_limit:
            self.work_queue.fail(
                item.item_id,
                worker_id,
                {"reason": "model_step_limit", "recovered_from_checkpoint": True},
                now=now,
            )
            self.thread_store.fail_thread(
                item.thread_id, "model step limit reached"
            )
            status = "failed_step_limit_recovered"
        else:
            self.work_queue.ack(
                item.item_id,
                worker_id,
                {
                    "final_answer": final_answer,
                    "recovered_from_checkpoint": True,
                },
                now=now,
            )
            status = "completed_from_checkpoint"

        self._emit(
            "work.finished",
            {
                "workItemId": item.item_id,
                "workerId": worker_id,
                "status": status,
                "recoveredFromCheckpoint": True,
            },
            thread_id=item.thread_id,
            turn_id=turn_id,
            now=now,
        )
        return RuntimeExecutionRecord(
            work_item_id=item.item_id,
            thread_id=item.thread_id,
            turn_id=turn_id,
            status=status,
            final_answer=final_answer,
            model_steps=model_steps,
        )

    def run_one(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        now: float | None = None,
        allowed_thread_ids: set[str] | None = None,
    ) -> RuntimeExecutionRecord | None:
        """Run one eligible turn, optionally fenced to an allowed thread set."""

        item = self.work_queue.claim(
            worker_id,
            lease_seconds=lease_seconds,
            now=now,
            kinds={"turn"},
            thread_ids=allowed_thread_ids,
        )
        if item is None:
            return None

        self._emit(
            "work.claimed",
            {
                "workItemId": item.item_id,
                "workerId": worker_id,
                "leaseUntil": item.lease_until,
            },
            thread_id=item.thread_id,
            now=now,
        )

        projection = self.thread_store.project(item.thread_id)
        if projection.status in TERMINAL_THREAD_STATUSES:
            if item.status is WorkStatus.LEASED:
                self.work_queue.cancel(item.item_id, now=now)
            self._emit(
                "work.cancelled",
                {"workItemId": item.item_id, "reason": "thread_terminal"},
                thread_id=item.thread_id,
                now=now,
            )
            return RuntimeExecutionRecord(
                work_item_id=item.item_id,
                thread_id=item.thread_id,
                turn_id=None,
                status="thread_terminal",
                final_answer=None,
                model_steps=None,
            )

        checkpoint_state = self._checkpoint_state(projection)
        recovering = projection.status is ThreadStatus.RUNNING

        if recovering:
            if checkpoint_state.get("work_item_id") != item.item_id:
                self.work_queue.release(item.item_id, worker_id, now=now)
                self._emit(
                    "work.deferred",
                    {"workItemId": item.item_id, "reason": "thread_not_ready"},
                    thread_id=item.thread_id,
                    turn_id=projection.active_turn_id,
                    now=now,
                )
                return RuntimeExecutionRecord(
                    work_item_id=item.item_id,
                    thread_id=item.thread_id,
                    turn_id=projection.active_turn_id,
                    status="deferred_thread_not_ready",
                    final_answer=None,
                    model_steps=None,
                )
            if projection.active_turn_id is None:
                raise RuntimeError("running thread has no active turn id")
            turn_id = projection.active_turn_id
            if checkpoint_state.get("phase") == "turn_finished":
                return self._finalize_finished_checkpoint(
                    item,
                    worker_id,
                    turn_id,
                    checkpoint_state,
                    now=now,
                )
        elif projection.status is ThreadStatus.READY:
            self.thread_store.checkpoint(
                item.thread_id,
                {"phase": "claimed", "work_item_id": item.item_id},
            )
            turn_id = self.thread_store.start_turn(item.thread_id)
            self.thread_store.checkpoint(
                item.thread_id,
                {
                    "phase": "turn_started",
                    "work_item_id": item.item_id,
                    "turn_id": turn_id,
                },
            )
            self._emit(
                "turn.opened",
                {"workItemId": item.item_id, "workerId": worker_id},
                thread_id=item.thread_id,
                turn_id=turn_id,
                now=now,
            )
        else:
            self.work_queue.release(item.item_id, worker_id, now=now)
            self._emit(
                "work.deferred",
                {"workItemId": item.item_id, "reason": projection.status.value},
                thread_id=item.thread_id,
                turn_id=projection.active_turn_id,
                now=now,
            )
            return RuntimeExecutionRecord(
                work_item_id=item.item_id,
                thread_id=item.thread_id,
                turn_id=projection.active_turn_id,
                status="deferred_thread_not_ready",
                final_answer=None,
                model_steps=None,
            )

        content = str(item.payload.get("content", ""))
        scoped_tools = TurnScopedJournaledTools(
            self.journaled_tools,
            scope=item.item_id,
        )
        heartbeat = LeaseHeartbeat(
            self.work_queue,
            item.item_id,
            worker_id,
            lease_seconds,
        )

        def beat() -> float:
            lease_until = heartbeat.renew(now=now) if now is not None else heartbeat.renew()
            self._emit(
                "work.heartbeat",
                {"workItemId": item.item_id, "leaseUntil": lease_until},
                thread_id=item.thread_id,
                turn_id=turn_id,
                now=now,
            )
            return lease_until

        controlled_backend = ControlledBackend(
            self.backend,
            heartbeat=beat,
            steering_queue=self.steering_queue,
            thread_id=item.thread_id,
        )

        def publish_harness_event(event: HarnessEvent) -> None:
            self._emit(
                f"harness.{event.kind.value}",
                dict(event.payload),
                thread_id=item.thread_id,
                turn_id=turn_id,
            )

        harness = CodexHarness(
            controlled_backend,
            scoped_tools,  # type: ignore[arg-type]
            event_sink=publish_harness_event,
            max_model_steps=self.max_model_steps,
        )
        background = (
            BackgroundLeaseHeartbeat(
                str(self.work_queue.path),
                item.item_id,
                worker_id,
                lease_seconds=lease_seconds,
            )
            if now is None
            else None
        )

        try:
            if background is not None:
                background.start()
            try:
                result: CodexTurnResult = harness.run_turn(content)
            finally:
                if background is not None:
                    background.close()

            if background is not None:
                self._emit(
                    "work.background_heartbeat_summary",
                    {
                        "workItemId": item.item_id,
                        "renewals": background.renewals,
                        "error": None
                        if background.last_error is None
                        else f"{type(background.last_error).__name__}: {background.last_error}",
                    },
                    thread_id=item.thread_id,
                    turn_id=turn_id,
                )
                if background.last_error is not None:
                    raise RuntimeError(
                        f"background lease heartbeat failed: {background.last_error}"
                    )

            self.thread_store.checkpoint(
                item.thread_id,
                {
                    "phase": "turn_finished",
                    "work_item_id": item.item_id,
                    "turn_id": turn_id,
                    "final_answer": result.final_answer,
                    "model_steps": result.model_steps,
                    "stopped_by_limit": result.stopped_by_limit,
                    "recovered_turn": recovering,
                },
            )
            self.thread_store.complete_turn(
                item.thread_id,
                turn_id,
                summary=result.final_answer,
            )
            if result.stopped_by_limit:
                self.work_queue.fail(
                    item.item_id,
                    worker_id,
                    {"reason": "model_step_limit"},
                    now=now,
                )
                self.thread_store.fail_thread(
                    item.thread_id, "model step limit reached"
                )
                status = "failed_step_limit"
            else:
                self.work_queue.ack(
                    item.item_id,
                    worker_id,
                    {
                        "final_answer": result.final_answer,
                        "recovered_turn": recovering,
                    },
                    now=now,
                )
                status = "completed_recovered" if recovering else "completed"
            self._emit(
                "work.finished",
                {
                    "workItemId": item.item_id,
                    "workerId": worker_id,
                    "status": status,
                    "modelSteps": result.model_steps,
                },
                thread_id=item.thread_id,
                turn_id=turn_id,
                now=now,
            )
            return RuntimeExecutionRecord(
                work_item_id=item.item_id,
                thread_id=item.thread_id,
                turn_id=turn_id,
                status=status,
                final_answer=result.final_answer,
                model_steps=result.model_steps,
            )
        except Exception as exc:
            if background is not None:
                background.close()
            self.work_queue.fail(
                item.item_id,
                worker_id,
                {"exception": type(exc).__name__, "message": str(exc)},
                now=now,
            )
            self.thread_store.fail_thread(
                item.thread_id,
                f"{type(exc).__name__}: {exc}",
            )
            self._emit(
                "work.failed",
                {
                    "workItemId": item.item_id,
                    "workerId": worker_id,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
                thread_id=item.thread_id,
                turn_id=turn_id,
                now=now,
            )
            raise

    def close(self) -> None:
        self.thread_store.close()
        self.work_queue.close()
        self.tool_journal.close()
        self.steering_queue.close()
        self.artifact_store.close()
        self.event_stream.close()

    def __enter__(self) -> "DurableAgentRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

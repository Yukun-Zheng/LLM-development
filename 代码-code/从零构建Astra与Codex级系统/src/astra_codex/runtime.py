from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ModelBackend
from .artifacts import ArtifactStore
from .codex_harness import CodexHarness, CodexTurnResult
from .durable import DurableThreadStore, TERMINAL_THREAD_STATUSES, ThreadStatus
from .runtime_control import ControlledBackend, LeaseHeartbeat
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

    The runtime now also owns a durable steering inbox and a content-addressed
    artifact store. Before every model sampling step, ``ControlledBackend``
    renews the worker lease and injects steering messages that arrived while the
    turn was executing.

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
        self.thread_store = DurableThreadStore(root / "threads.sqlite")
        self.work_queue = DurableWorkQueue(root / "work.sqlite")
        self.tool_journal = DurableToolJournal(root / "tool-journal.sqlite")
        self.steering_queue = DurableSteeringQueue(root / "steering.sqlite")
        self.artifact_store = ArtifactStore(root / "artifacts")
        self.journaled_tools = JournaledToolExecutor(
            tools,
            self.tool_journal,
            retryable_in_doubt_tools=retryable_in_doubt_tools,
        )
        self.backend = backend
        self.max_model_steps = max_model_steps

    def create_thread(self, thread_id: str | None = None) -> str:
        return self.thread_store.create_thread(thread_id)

    def submit(
        self,
        thread_id: str,
        content: str,
        *,
        item_id: str | None = None,
        now: float | None = None,
    ) -> str:
        self.thread_store.submit(thread_id, content)
        return self.work_queue.enqueue(
            thread_id,
            "turn",
            {"content": content},
            item_id=item_id,
            now=now,
        )

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
        return self.steering_queue.submit(
            thread_id,
            content,
            steering_id=steering_id,
            now=now,
        )

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
        return self.artifact_store.snapshot_file(
            thread_id,
            source,
            kind=kind,
            metadata=metadata,
            artifact_id=artifact_id,
            now=now,
        )

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
    ) -> RuntimeExecutionRecord | None:
        item = self.work_queue.claim(
            worker_id,
            lease_seconds=lease_seconds,
            now=now,
            kinds={"turn"},
        )
        if item is None:
            return None

        projection = self.thread_store.project(item.thread_id)
        if projection.status in TERMINAL_THREAD_STATUSES:
            if item.status is WorkStatus.LEASED:
                self.work_queue.cancel(item.item_id, now=now)
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
        else:
            self.work_queue.release(item.item_id, worker_id, now=now)
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
            return heartbeat.renew(now=now) if now is not None else heartbeat.renew()

        controlled_backend = ControlledBackend(
            self.backend,
            heartbeat=beat,
            steering_queue=self.steering_queue,
            thread_id=item.thread_id,
        )
        harness = CodexHarness(
            controlled_backend,
            scoped_tools,  # type: ignore[arg-type]
            max_model_steps=self.max_model_steps,
        )

        try:
            result: CodexTurnResult = harness.run_turn(content)
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
            return RuntimeExecutionRecord(
                work_item_id=item.item_id,
                thread_id=item.thread_id,
                turn_id=turn_id,
                status=status,
                final_answer=result.final_answer,
                model_steps=result.model_steps,
            )
        except Exception as exc:
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
            raise

    def close(self) -> None:
        self.thread_store.close()
        self.work_queue.close()
        self.tool_journal.close()
        self.steering_queue.close()
        self.artifact_store.close()

    def __enter__(self) -> "DurableAgentRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ModelBackend
from .codex_harness import CodexHarness, CodexTurnResult
from .durable import DurableThreadStore, TERMINAL_THREAD_STATUSES, ThreadStatus
from .runtime_queue import DurableWorkQueue, WorkItem, WorkStatus
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
    """Reference composition of ThreadStore + WorkQueue + TurnExecutor.

    The durable pipeline is now:

        submission -> thread event -> work item -> worker lease -> turn/checkpoint
        -> journaled tool calls -> final checkpoint -> turn completion -> work ACK

    Completed tool calls are keyed by the stable work-item id and call index. If
    a worker disappears and the lease is later reclaimed, completed calls can be
    replayed from the journal without repeating their side effects. A tool call
    left only in STARTED state is *in doubt* and is not retried by default.

    This still does not provide universal exactly-once external side effects.
    An external operation that committed before the process crashed but before
    its COMPLETED journal row was written requires tool-specific reconciliation
    or an external idempotency/transaction mechanism.
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
            # Record work ownership before TURN_STARTED. This closes most of the
            # crash window between queue claim and durable turn creation.
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
        harness = CodexHarness(
            self.backend,
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
            # This catches ordinary executor exceptions. A hard process death
            # cannot execute this block; that is why lease reclaim + durable
            # checkpoints + tool journal exist independently.
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

    def __enter__(self) -> "DurableAgentRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

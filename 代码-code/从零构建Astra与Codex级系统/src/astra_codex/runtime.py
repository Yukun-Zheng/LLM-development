from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import ModelBackend
from .codex_harness import CodexHarness, CodexTurnResult
from .durable import DurableThreadStore, TERMINAL_THREAD_STATUSES, ThreadStatus
from .runtime_queue import DurableWorkQueue, WorkStatus
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

    This is the first layer where the previously independent durability
    primitives form one runnable pipeline:

        submission -> persistent thread event -> durable work item -> worker
        lease -> turn start -> Codex-style executor -> checkpoint -> turn
        completion -> work ACK

    Recovery boundary: if a worker dies *after* TURN_STARTED but before the turn
    completes, the work lease can be reclaimed but this reference runtime does
    not automatically re-run side-effecting tools. Exactly-once side effects
    require idempotency keys / durable tool execution records, which remain a
    later milestone.
    """

    def __init__(
        self,
        state_dir: str | Path,
        backend: ModelBackend,
        tools: ToolRegistry,
        *,
        max_model_steps: int = 32,
    ) -> None:
        root = Path(state_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.thread_store = DurableThreadStore(root / "threads.sqlite")
        self.work_queue = DurableWorkQueue(root / "work.sqlite")
        self.harness = CodexHarness(
            backend,
            tools,
            max_model_steps=max_model_steps,
        )

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

        if projection.status is not ThreadStatus.READY:
            # Another turn currently owns this thread. Put the work back instead
            # of creating concurrent turns for one ordered conversation.
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
        turn_id = self.thread_store.start_turn(item.thread_id)
        result: CodexTurnResult | None = None
        try:
            result = self.harness.run_turn(content)
            self.thread_store.checkpoint(
                item.thread_id,
                {
                    "work_item_id": item.item_id,
                    "final_answer": result.final_answer,
                    "model_steps": result.model_steps,
                    "stopped_by_limit": result.stopped_by_limit,
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
                    {"final_answer": result.final_answer},
                    now=now,
                )
                status = "completed"
            return RuntimeExecutionRecord(
                work_item_id=item.item_id,
                thread_id=item.thread_id,
                turn_id=turn_id,
                status=status,
                final_answer=result.final_answer,
                model_steps=result.model_steps,
            )
        except Exception as exc:
            # Persist the failure before surfacing it so an operator can inspect
            # the thread/work state after a crash-like executor exception.
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

    def __enter__(self) -> "DurableAgentRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

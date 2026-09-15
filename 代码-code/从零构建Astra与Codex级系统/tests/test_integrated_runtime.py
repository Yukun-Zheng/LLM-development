from __future__ import annotations

from dataclasses import dataclass

from astra_codex.agent import ScriptedBackend
from astra_codex.durable import DurableThreadStore, ThreadStatus
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_queue import WorkStatus
from astra_codex.structured import ToolSpec
from astra_codex.tool_journal import TurnScopedJournaledTools
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class CountingTool:
    calls: int = 0

    spec = ToolSpec(
        name="counter",
        description="Count executions.",
        parameters={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(True, str(self.calls), {"calls": self.calls})


def test_integrated_runtime_persists_submission_turn_checkpoint_and_ack(tmp_path) -> None:
    state = tmp_path / "state"
    with DurableAgentRuntime(
        state,
        ScriptedBackend(["done"]),
        ToolRegistry([]),
    ) as runtime:
        thread_id = runtime.create_thread("thr_integrated")
        item_id = runtime.submit(thread_id, "finish the task", item_id="work_1", now=1.0)
        record = runtime.run_one("worker_a", now=2.0)

        assert record is not None
        assert record.status == "completed"
        assert record.final_answer == "done"
        assert runtime.work_queue.get(item_id).status is WorkStatus.COMPLETED
        projection = runtime.thread_store.project(thread_id)
        assert projection.status is ThreadStatus.READY
        assert projection.active_turn_id is None
        assert projection.last_checkpoint is not None
        assert projection.last_checkpoint["state"]["final_answer"] == "done"
        assert projection.last_checkpoint["state"]["phase"] == "turn_finished"

    with DurableThreadStore(state / "threads.sqlite") as reopened:
        projection = reopened.project("thr_integrated")
        assert projection.status is ThreadStatus.READY
        assert projection.submissions == ("finish the task",)
        assert projection.last_checkpoint is not None
        assert projection.last_checkpoint["state"]["work_item_id"] == "work_1"


def test_integrated_runtime_processes_multiple_submissions_in_order(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["first done", "second done"]),
        ToolRegistry([]),
    ) as runtime:
        thread_id = runtime.create_thread("thr_order")
        runtime.submit(thread_id, "first", item_id="work_a", now=1.0)
        runtime.submit(thread_id, "second", item_id="work_b", now=2.0)

        first = runtime.run_one("worker", now=3.0)
        second = runtime.run_one("worker", now=4.0)

        assert first is not None and first.work_item_id == "work_a"
        assert second is not None and second.work_item_id == "work_b"
        assert first.final_answer == "first done"
        assert second.final_answer == "second done"
        assert runtime.thread_store.project(thread_id).submissions == ("first", "second")


def test_integrated_runtime_cancels_pending_work_for_terminal_thread(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        thread_id = runtime.create_thread("thr_cancelled")
        item_id = runtime.submit(thread_id, "do not run", item_id="work_cancel", now=1.0)
        runtime.thread_store.cancel_thread(thread_id, "user stop")

        record = runtime.run_one("worker", now=2.0)

        assert record is not None
        assert record.status == "thread_terminal"
        assert runtime.work_queue.get(item_id).status is WorkStatus.CANCELLED


def test_integrated_runtime_marks_step_limit_as_failed(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(['{"tool":"missing","arguments":{}}']),
        ToolRegistry([]),
        max_model_steps=1,
    ) as runtime:
        thread_id = runtime.create_thread("thr_limit")
        item_id = runtime.submit(thread_id, "loop", item_id="work_limit", now=1.0)

        record = runtime.run_one("worker", now=2.0)

        assert record is not None
        assert record.status == "failed_step_limit"
        assert runtime.work_queue.get(item_id).status is WorkStatus.FAILED
        assert runtime.thread_store.project(thread_id).status is ThreadStatus.FAILED


def test_reclaimed_turn_replays_completed_tool_without_duplicate_side_effect(tmp_path) -> None:
    state = tmp_path / "state"
    original_tool = CountingTool()

    # Simulate a worker that acquired the work item, started the turn, completed
    # the first side effect, and then died before the turn could finish.
    with DurableAgentRuntime(
        state,
        ScriptedBackend([]),
        ToolRegistry([original_tool]),
    ) as crashed_runtime:
        thread_id = crashed_runtime.create_thread("thr_recovery")
        item_id = crashed_runtime.submit(
            thread_id, "increment once", item_id="work_recovery", now=0.0
        )
        claimed = crashed_runtime.work_queue.claim(
            "worker_a", lease_seconds=10.0, now=1.0, kinds={"turn"}
        )
        assert claimed is not None
        crashed_runtime.thread_store.checkpoint(
            thread_id, {"phase": "claimed", "work_item_id": item_id}
        )
        turn_id = crashed_runtime.thread_store.start_turn(thread_id, "turn_recovery")
        crashed_runtime.thread_store.checkpoint(
            thread_id,
            {
                "phase": "turn_started",
                "work_item_id": item_id,
                "turn_id": turn_id,
            },
        )
        scoped = TurnScopedJournaledTools(
            crashed_runtime.journaled_tools, scope=item_id
        )
        first = scoped.execute("counter", {})
        assert first.ok
        assert original_tool.calls == 1

    # After the lease expires, a new process/worker re-runs the turn. The first
    # tool call uses the same work-item/call-index key and is replayed from the
    # persistent journal; the fresh tool body must not execute again.
    fresh_tool = CountingTool()
    with DurableAgentRuntime(
        state,
        ScriptedBackend(
            ['{"tool":"counter","arguments":{}}', "recovered and finished"]
        ),
        ToolRegistry([fresh_tool]),
    ) as recovered_runtime:
        record = recovered_runtime.run_one(
            "worker_b", lease_seconds=10.0, now=12.0
        )

        assert record is not None
        assert record.status == "completed_recovered"
        assert record.turn_id == "turn_recovery"
        assert fresh_tool.calls == 0
        assert recovered_runtime.work_queue.get(item_id).status is WorkStatus.COMPLETED
        assert recovered_runtime.thread_store.project(thread_id).status is ThreadStatus.READY


def test_turn_finished_checkpoint_can_finalize_without_rerunning_model(tmp_path) -> None:
    state = tmp_path / "state"
    with DurableAgentRuntime(
        state,
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as crashed_runtime:
        thread_id = crashed_runtime.create_thread("thr_final_checkpoint")
        item_id = crashed_runtime.submit(
            thread_id, "finish", item_id="work_final_checkpoint", now=0.0
        )
        claimed = crashed_runtime.work_queue.claim(
            "worker_a", lease_seconds=10.0, now=1.0, kinds={"turn"}
        )
        assert claimed is not None
        crashed_runtime.thread_store.checkpoint(
            thread_id, {"phase": "claimed", "work_item_id": item_id}
        )
        turn_id = crashed_runtime.thread_store.start_turn(
            thread_id, "turn_final_checkpoint"
        )
        crashed_runtime.thread_store.checkpoint(
            thread_id,
            {
                "phase": "turn_finished",
                "work_item_id": item_id,
                "turn_id": turn_id,
                "final_answer": "already computed",
                "model_steps": 4,
                "stopped_by_limit": False,
            },
        )

    # Empty ScriptedBackend proves no model sampling is needed to finish a turn
    # whose final checkpoint was already persisted before the crash.
    with DurableAgentRuntime(
        state,
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as recovered_runtime:
        record = recovered_runtime.run_one("worker_b", now=12.0)

        assert record is not None
        assert record.status == "completed_from_checkpoint"
        assert record.final_answer == "already computed"
        assert record.model_steps == 4
        assert recovered_runtime.work_queue.get(item_id).status is WorkStatus.COMPLETED
        assert recovered_runtime.thread_store.project(thread_id).status is ThreadStatus.READY

from __future__ import annotations

from astra_codex.agent import ScriptedBackend
from astra_codex.durable import DurableThreadStore, ThreadStatus
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_queue import WorkStatus
from astra_codex.tools import ToolRegistry


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

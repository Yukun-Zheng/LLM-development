from __future__ import annotations

from dataclasses import dataclass

import pytest

from astra_codex.agent import ScriptedBackend
from astra_codex.codex_harness import CodexHarness, EventKind
from astra_codex.durable import DurableThreadStore, ThreadStatus
from astra_codex.evaluation import aggregate_metrics, summarize_codex_turn
from astra_codex.security import (
    GuardedToolExecutor,
    PermissionDecision,
    PermissionProfile,
)
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class CountingTool:
    calls: int = 0

    spec = ToolSpec(
        name="counter",
        description="Increment a counter.",
        parameters={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(True, str(self.calls))


def test_durable_thread_replays_after_process_restart(tmp_path) -> None:
    db = tmp_path / "thread.sqlite"

    with DurableThreadStore(db) as store:
        thread_id = store.create_thread("thr_test")
        store.submit(thread_id, "fix the bug")
        turn_id = store.start_turn(thread_id, "turn_1")
        store.checkpoint(thread_id, {"phase": "tests-running", "attempt": 1})
        before = store.project(thread_id)
        assert before.status is ThreadStatus.RUNNING
        assert before.active_turn_id == turn_id

    with DurableThreadStore(db) as reopened:
        replayed = reopened.project("thr_test")
        assert replayed.status is ThreadStatus.RUNNING
        assert replayed.active_turn_id == "turn_1"
        assert replayed.submissions == ("fix the bug",)
        assert replayed.last_checkpoint is not None
        assert replayed.last_checkpoint["state"]["attempt"] == 1
        reopened.complete_turn("thr_test", "turn_1", summary="tests passed")
        reopened.pause("thr_test", "waiting for review")
        assert reopened.project("thr_test").status is ThreadStatus.PAUSED
        reopened.resume("thr_test")
        reopened.complete_thread("thr_test")
        assert reopened.project("thr_test").status is ThreadStatus.COMPLETED


def test_durable_thread_rejects_invalid_transition(tmp_path) -> None:
    with DurableThreadStore(tmp_path / "thread.sqlite") as store:
        thread_id = store.create_thread()
        store.start_turn(thread_id, "turn_1")
        with pytest.raises(RuntimeError):
            store.start_turn(thread_id, "turn_2")
        with pytest.raises(RuntimeError):
            store.complete_thread(thread_id)


def test_durable_thread_fork_replays_independent_prefix(tmp_path) -> None:
    with DurableThreadStore(tmp_path / "thread.sqlite") as store:
        parent = store.create_thread("thr_parent")
        store.submit(parent, "investigate bug")
        turn = store.start_turn(parent, "turn_parent")
        store.checkpoint(parent, {"phase": "diagnosis"})
        fork_point = store.project(parent).last_event_id

        child = store.fork_thread(
            parent,
            new_thread_id="thr_child",
            through_event_id=fork_point,
        )
        child_state = store.project(child)
        assert child_state.parent_thread_id == parent
        assert child_state.parent_event_id == fork_point
        assert child_state.status is ThreadStatus.RUNNING
        assert child_state.active_turn_id == turn
        assert child_state.submissions == ("investigate bug",)
        assert child_state.last_checkpoint is not None
        assert child_state.last_checkpoint["state"]["phase"] == "diagnosis"

        store.cancel_thread(child, "explore another branch")
        assert store.project(child).status is ThreadStatus.CANCELLED
        assert store.project(parent).status is ThreadStatus.RUNNING


def test_cancelled_thread_is_terminal(tmp_path) -> None:
    with DurableThreadStore(tmp_path / "thread.sqlite") as store:
        thread_id = store.create_thread("thr_cancel")
        store.submit(thread_id, "long task")
        store.cancel_thread(thread_id, "user requested stop")
        state = store.project(thread_id)
        assert state.status is ThreadStatus.CANCELLED
        assert state.cancellation_reason == "user requested stop"
        with pytest.raises(RuntimeError):
            store.submit(thread_id, "should be rejected")
        with pytest.raises(RuntimeError):
            store.checkpoint(thread_id, {"late": True})


def test_permission_deny_prevents_underlying_tool_execution() -> None:
    tool = CountingTool()
    guarded = GuardedToolExecutor(
        ToolRegistry([tool]),
        PermissionProfile(denied_tools=frozenset({"counter"})),
    )

    result = guarded.execute("counter", {})

    assert not result.ok
    assert tool.calls == 0
    assert result.metadata is not None
    assert result.metadata["permission_decision"] == PermissionDecision.DENY.value


def test_permission_approval_is_enforced_before_dispatch() -> None:
    tool = CountingTool()
    denied = GuardedToolExecutor(
        ToolRegistry([tool]),
        PermissionProfile(approval_tools=frozenset({"counter"})),
        approval_callback=lambda _tool, _arguments: False,
    )
    assert not denied.execute("counter", {}).ok
    assert tool.calls == 0

    allowed = GuardedToolExecutor(
        ToolRegistry([tool]),
        PermissionProfile(approval_tools=frozenset({"counter"})),
        approval_callback=lambda _tool, _arguments: True,
    )
    assert allowed.execute("counter", {}).ok
    assert tool.calls == 1


def test_trajectory_metrics_separate_runtime_completion_from_verification() -> None:
    tool = CountingTool()
    registry = ToolRegistry([tool])
    backend = ScriptedBackend(
        [
            '{"tool":"counter","arguments":{}}',
            "done",
        ]
    )
    result = CodexHarness(backend, registry).run_turn("do one thing")

    metrics = summarize_codex_turn(
        "case-1",
        result,
        verifier_passed=True,
        wall_time_s=2.5,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
    )

    assert metrics.success
    assert metrics.tool_calls == 1
    assert metrics.tool_failures == 0
    assert metrics.model_steps == 2
    assert any(event.kind is EventKind.TOOL_COMPLETED for event in result.events)

    failed_verification = summarize_codex_turn(
        "case-2", result, verifier_passed=False
    )
    assert not failed_verification.success


def test_trajectory_aggregate_reports_cost_and_success_rate() -> None:
    registry = ToolRegistry([])
    success = CodexHarness(ScriptedBackend(["done"]), registry).run_turn("finish")
    stopped = CodexHarness(
        ScriptedBackend(['{"tool":"missing","arguments":{}}']),
        registry,
        max_model_steps=1,
    ).run_turn("never finish")

    aggregate = aggregate_metrics(
        [
            summarize_codex_turn("ok", success, cost_usd=0.02, wall_time_s=1.0),
            summarize_codex_turn("stop", stopped, cost_usd=0.03, wall_time_s=3.0),
        ]
    )

    assert aggregate.cases == 2
    assert aggregate.success_rate == 0.5
    assert aggregate.total_cost_usd == pytest.approx(0.05)
    assert aggregate.mean_wall_time_s == pytest.approx(2.0)

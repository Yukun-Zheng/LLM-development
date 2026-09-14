from pathlib import Path

import pytest

from astra_codex.multi_agent import SequentialCoordinator, WorkerResult
from astra_codex.planning import PlanGraph, PlanStep, StepStatus
from astra_codex.verification import (
    CommandVerifier,
    CompositeVerifier,
    FileExistsVerifier,
    Verdict,
)


def test_plan_graph_dependencies_and_completion() -> None:
    plan = PlanGraph(
        [
            PlanStep("inspect", "inspect repository"),
            PlanStep("edit", "edit implementation", depends_on=("inspect",)),
            PlanStep("verify", "run tests", depends_on=("edit",)),
        ]
    )

    assert [step.id for step in plan.ready()] == ["inspect"]
    plan.start("inspect")
    plan.complete("inspect", {"files": ["a.py"]})
    assert [step.id for step in plan.ready()] == ["edit"]

    plan.start("edit")
    plan.complete("edit")
    plan.start("verify")
    plan.complete("verify", {"tests": "passed"})
    assert plan.is_complete
    assert plan.terminal


def test_plan_graph_rejects_cycles_and_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PlanGraph([PlanStep("x", "a"), PlanStep("x", "b")])

    with pytest.raises(ValueError, match="cycle"):
        PlanGraph(
            [
                PlanStep("a", "a", depends_on=("b",)),
                PlanStep("b", "b", depends_on=("a",)),
            ]
        )


def test_failure_blocks_downstream_step() -> None:
    plan = PlanGraph(
        [
            PlanStep("a", "first"),
            PlanStep("b", "second", depends_on=("a",)),
        ]
    )
    plan.start("a")
    plan.fail("a", "boom")
    assert plan.steps["a"].status is StepStatus.FAILED
    assert plan.steps["b"].status is StepStatus.BLOCKED
    assert plan.terminal


def test_external_verifiers(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("ok", encoding="utf-8")

    file_result = FileExistsVerifier(tmp_path, "artifact.txt").verify()
    assert file_result.verdict is Verdict.PASS

    cmd_result = CommandVerifier(
        tmp_path,
        ["python", "-c", "from pathlib import Path; assert Path('artifact.txt').read_text() == 'ok'"],
    ).verify()
    assert cmd_result.verdict is Verdict.PASS

    combined = CompositeVerifier(
        [FileExistsVerifier(tmp_path, "artifact.txt"), CommandVerifier(tmp_path, ["python", "-c", "pass"])]
    ).verify()
    assert combined.passed


def test_sequential_coordinator_records_worker_evidence() -> None:
    plan = PlanGraph(
        [
            PlanStep("inspect", "inspect"),
            PlanStep("implement", "implement", depends_on=("inspect",)),
        ]
    )
    coordinator = SequentialCoordinator(plan)

    def worker(step: PlanStep) -> WorkerResult:
        return WorkerResult(step.id, True, f"finished {step.id}", {"worker": "test"})

    results = coordinator.run_until_terminal(worker)
    assert [result.task_id for result in results] == ["inspect", "implement"]
    assert plan.is_complete
    assert plan.steps["implement"].evidence["worker"] == "test"

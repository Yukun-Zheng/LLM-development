from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .planning import PlanGraph, PlanStep


@dataclass(frozen=True, slots=True)
class WorkerResult:
    task_id: str
    ok: bool
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)


Worker = Callable[[PlanStep], WorkerResult]


class SequentialCoordinator:
    """A deterministic coordinator over a PlanGraph.

    This is deliberately *not* a parallel scheduler yet.  Its purpose is to
    make task assignment, evidence handoff, failure propagation, and coordinator
    state explicit before adding concurrency and worktree workers.
    """

    def __init__(self, plan: PlanGraph) -> None:
        self.plan = plan
        self.results: list[WorkerResult] = []

    def run_one_ready(self, worker: Worker) -> WorkerResult | None:
        ready = self.plan.ready()
        if not ready:
            return None
        step = ready[0]
        self.plan.start(step.id)
        try:
            result = worker(step)
        except Exception as exc:  # coordinator converts worker failure into state
            result = WorkerResult(step.id, False, f"worker raised {type(exc).__name__}: {exc}")

        if result.task_id != step.id:
            self.plan.fail(step.id, f"worker returned mismatched task id {result.task_id!r}")
            mismatch = WorkerResult(
                step.id,
                False,
                "worker returned a result for the wrong task",
                {"returned_task_id": result.task_id},
            )
            self.results.append(mismatch)
            return mismatch

        if result.ok:
            self.plan.complete(step.id, {"summary": result.summary, **result.evidence})
        else:
            self.plan.fail(step.id, result.summary)
        self.results.append(result)
        return result

    def run_until_terminal(self, worker: Worker, *, max_dispatches: int = 100) -> list[WorkerResult]:
        dispatches = 0
        while not self.plan.terminal and dispatches < max_dispatches:
            result = self.run_one_ready(worker)
            if result is None:
                break
            dispatches += 1
        return list(self.results)

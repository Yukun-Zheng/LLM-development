from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(slots=True)
class PlanStep:
    """One explicit unit of work in an agent task graph."""

    id: str
    description: str
    depends_on: tuple[str, ...] = ()
    status: StepStatus = StepStatus.PENDING
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class PlanGraph:
    """Small inspectable DAG for closed-loop agent planning.

    The plan is intentionally data, not prose. A caller can persist, inspect,
    re-plan, or schedule this structure without scraping a natural-language
    checklist out of a model response.
    """

    def __init__(self, steps: Iterable[PlanStep]) -> None:
        step_list = list(steps)
        if not step_list:
            raise ValueError("plan must contain at least one step")
        ids = [step.id for step in step_list]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        self.steps = {step.id: step for step in step_list}
        self._validate_dependencies()
        self._validate_acyclic()

    def _validate_dependencies(self) -> None:
        ids = set(self.steps)
        for step in self.steps.values():
            missing = [dep for dep in step.depends_on if dep not in ids]
            if missing:
                raise ValueError(f"step {step.id!r} has missing dependencies: {missing}")
            if step.id in step.depends_on:
                raise ValueError(f"step {step.id!r} cannot depend on itself")

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError("plan contains a dependency cycle")
            visiting.add(step_id)
            for dep in self.steps[step_id].depends_on:
                visit(dep)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in self.steps:
            visit(step_id)

    def ready(self) -> list[PlanStep]:
        ready: list[PlanStep] = []
        for step in self.steps.values():
            if step.status is not StepStatus.PENDING:
                continue
            deps = [self.steps[dep] for dep in step.depends_on]
            if any(dep.status in {StepStatus.FAILED, StepStatus.BLOCKED} for dep in deps):
                step.status = StepStatus.BLOCKED
                continue
            if all(dep.status is StepStatus.COMPLETED for dep in deps):
                ready.append(step)
        return ready

    def start(self, step_id: str) -> PlanStep:
        step = self.steps[step_id]
        if step not in self.ready():
            raise ValueError(f"step {step_id!r} is not ready")
        step.status = StepStatus.RUNNING
        return step

    def complete(self, step_id: str, evidence: dict[str, Any] | None = None) -> None:
        step = self.steps[step_id]
        if step.status is not StepStatus.RUNNING:
            raise ValueError(f"step {step_id!r} is not running")
        step.status = StepStatus.COMPLETED
        step.evidence = dict(evidence or {})
        step.error = None

    def fail(self, step_id: str, error: str) -> None:
        step = self.steps[step_id]
        if step.status is not StepStatus.RUNNING:
            raise ValueError(f"step {step_id!r} is not running")
        step.status = StepStatus.FAILED
        step.error = error
        # Calling ready() propagates blocked state to downstream pending steps.
        self.ready()

    @property
    def is_complete(self) -> bool:
        return all(step.status is StepStatus.COMPLETED for step in self.steps.values())

    @property
    def terminal(self) -> bool:
        return all(
            step.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.BLOCKED}
            for step in self.steps.values()
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": step.id,
                "description": step.description,
                "depends_on": list(step.depends_on),
                "status": step.status.value,
                "evidence": step.evidence,
                "error": step.error,
            }
            for step in self.steps.values()
        ]

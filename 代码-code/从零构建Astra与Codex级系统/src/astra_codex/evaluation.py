from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .codex_harness import CodexTurnResult, EventKind


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    case_id: str
    success: bool
    verifier_passed: bool | None
    model_steps: int
    tool_calls: int
    tool_failures: int
    approvals_requested: int
    stopped_by_limit: bool
    wall_time_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    cases: int
    success_rate: float
    mean_model_steps: float
    mean_tool_calls: float
    mean_tool_failures: float
    mean_approvals_requested: float
    mean_wall_time_s: float | None
    total_cost_usd: float | None


def summarize_codex_turn(
    case_id: str,
    result: CodexTurnResult,
    *,
    verifier_passed: bool | None = None,
    wall_time_s: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> TrajectoryMetrics:
    """Convert a Codex-style turn into comparable capability metrics.

    Unit tests answer whether code behaves as specified. These metrics answer a
    different question: what happened during a task trajectory and at what
    execution cost. The function intentionally keeps token/cost fields optional
    so local open-weight runs and API-backed runs can share one record format.
    """

    tool_completed = [
        event for event in result.events if event.kind is EventKind.TOOL_COMPLETED
    ]
    tool_failures = sum(not bool(event.payload.get("ok", False)) for event in tool_completed)
    approvals = sum(
        event.kind is EventKind.APPROVAL_REQUESTED for event in result.events
    )

    runtime_completed = not result.stopped_by_limit
    verifier_ok = verifier_passed is not False
    success = runtime_completed and verifier_ok

    return TrajectoryMetrics(
        case_id=case_id,
        success=success,
        verifier_passed=verifier_passed,
        model_steps=result.model_steps,
        tool_calls=len(tool_completed),
        tool_failures=tool_failures,
        approvals_requested=approvals,
        stopped_by_limit=result.stopped_by_limit,
        wall_time_s=wall_time_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def aggregate_metrics(records: Iterable[TrajectoryMetrics]) -> AggregateMetrics:
    rows = list(records)
    if not rows:
        raise ValueError("at least one evaluation record is required")

    wall_times = [row.wall_time_s for row in rows if row.wall_time_s is not None]
    costs = [row.cost_usd for row in rows if row.cost_usd is not None]

    return AggregateMetrics(
        cases=len(rows),
        success_rate=sum(row.success for row in rows) / len(rows),
        mean_model_steps=mean(row.model_steps for row in rows),
        mean_tool_calls=mean(row.tool_calls for row in rows),
        mean_tool_failures=mean(row.tool_failures for row in rows),
        mean_approvals_requested=mean(row.approvals_requested for row in rows),
        mean_wall_time_s=mean(wall_times) if wall_times else None,
        total_cost_usd=sum(costs) if costs else None,
    )

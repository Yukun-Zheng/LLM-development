from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .codex_harness import CodexTurnResult
from .evaluation import TrajectoryMetrics, aggregate_metrics, summarize_codex_turn


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    goal: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Grade:
    passed: bool
    score: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    case: BenchmarkCase
    grade: Grade
    trajectory: TrajectoryMetrics
    final_answer: str


Executor = Callable[[BenchmarkCase], CodexTurnResult]
Grader = Callable[[BenchmarkCase, CodexTurnResult], Grade]


class ExactAnswerGrader:
    """Small deterministic grader useful for harness tests and toy labs."""

    def __init__(self, expected: dict[str, str]) -> None:
        self.expected = expected

    def __call__(self, case: BenchmarkCase, result: CodexTurnResult) -> Grade:
        if case.case_id not in self.expected:
            raise KeyError(f"no expected answer for case: {case.case_id}")
        target = self.expected[case.case_id]
        passed = result.final_answer.strip() == target.strip()
        return Grade(
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"expected": target, "observed": result.final_answer},
        )


class BenchmarkHarness:
    """Execute cases, grade final evidence, and emit trajectory metrics.

    This is an evaluation substrate, not a benchmark by itself. Real adapters
    must supply environment setup, executor behavior and graders that inspect
    final state/artifacts instead of trusting model self-report.
    """

    def __init__(self, executor: Executor, grader: Grader) -> None:
        self.executor = executor
        self.grader = grader

    def run_case(self, case: BenchmarkCase) -> BenchmarkRecord:
        started = time.perf_counter()
        result = self.executor(case)
        elapsed = time.perf_counter() - started
        grade = self.grader(case, result)
        trajectory = summarize_codex_turn(
            case.case_id,
            result,
            verifier_passed=grade.passed,
            wall_time_s=elapsed,
        )
        return BenchmarkRecord(
            case=case,
            grade=grade,
            trajectory=trajectory,
            final_answer=result.final_answer,
        )

    def run_suite(self, cases: list[BenchmarkCase]) -> list[BenchmarkRecord]:
        return [self.run_case(case) for case in cases]

    @staticmethod
    def aggregate(records: list[BenchmarkRecord]):  # type: ignore[no-untyped-def]
        return aggregate_metrics(record.trajectory for record in records)

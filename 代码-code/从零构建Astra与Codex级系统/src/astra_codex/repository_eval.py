from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentRun, ModelBackend
from .coding import build_coding_agent


@dataclass(frozen=True, slots=True)
class RepositoryFixture:
    """A small reproducible coding task with an external final-state grader."""

    case_id: str
    goal: str
    files: dict[str, str]
    verify_argv: tuple[str, ...]
    expected_files: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class RepositoryGrade:
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    expected_files_ok: bool


@dataclass(frozen=True, slots=True)
class RepositoryEvalRecord:
    case_id: str
    grade: RepositoryGrade
    agent_run: AgentRun
    wall_time_s: float
    tool_calls: int
    tool_failures: int


class RepositoryFixtureHarness:
    """Level-1 local coding benchmark with deterministic external grading.

    The agent may inspect/edit/run commands, but success is computed *after* the
    agent stops by executing ``verify_argv`` independently and optionally
    checking final file contents. A final answer that claims success cannot make
    a failed repository pass.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def materialize(self, fixture: RepositoryFixture) -> Path:
        case_root = self.root / fixture.case_id
        if case_root.exists():
            raise FileExistsError(f"fixture root already exists: {case_root}")
        case_root.mkdir(parents=True)
        for relative, content in fixture.files.items():
            target = (case_root / relative).resolve()
            if case_root != target and case_root not in target.parents:
                raise PermissionError(f"fixture path escapes root: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        # A real Git repository makes the same filesystem usable by GitTool.
        subprocess.run(
            ["git", "init", "-q"],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=case_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return case_root

    def grade(self, fixture: RepositoryFixture, case_root: Path) -> RepositoryGrade:
        completed = subprocess.run(
            list(fixture.verify_argv),
            cwd=case_root,
            text=True,
            capture_output=True,
            timeout=fixture.timeout_s,
        )
        expected_files_ok = True
        for relative, expected in fixture.expected_files.items():
            target = (case_root / relative).resolve()
            if case_root != target and case_root not in target.parents:
                expected_files_ok = False
                break
            if not target.exists() or target.read_text(encoding="utf-8") != expected:
                expected_files_ok = False
                break
        return RepositoryGrade(
            passed=completed.returncode == 0 and expected_files_ok,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            expected_files_ok=expected_files_ok,
        )

    @staticmethod
    def _trajectory_counts(run: AgentRun) -> tuple[int, int]:
        calls = 0
        failures = 0
        for message in run.messages:
            if message.role != "tool":
                continue
            calls += 1
            try:
                observation = json.loads(message.content)
            except json.JSONDecodeError:
                failures += 1
                continue
            if not bool(observation.get("ok", False)):
                failures += 1
        return calls, failures

    def run(
        self,
        fixture: RepositoryFixture,
        backend: ModelBackend,
        *,
        max_steps: int = 40,
    ) -> RepositoryEvalRecord:
        case_root = self.materialize(fixture)
        agent = build_coding_agent(backend, case_root, max_steps=max_steps)
        started = time.perf_counter()
        agent_run = agent.run(fixture.goal)
        wall_time = time.perf_counter() - started
        grade = self.grade(fixture, case_root)
        tool_calls, tool_failures = self._trajectory_counts(agent_run)
        return RepositoryEvalRecord(
            case_id=fixture.case_id,
            grade=grade,
            agent_run=agent_run,
            wall_time_s=wall_time,
            tool_calls=tool_calls,
            tool_failures=tool_failures,
        )

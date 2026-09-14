from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verdict: Verdict
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS


class Verifier(Protocol):
    def verify(self) -> VerificationResult: ...


class FileExistsVerifier:
    def __init__(self, root: str | Path, relative_path: str | Path) -> None:
        self.root = Path(root).resolve()
        self.relative_path = Path(relative_path)

    def verify(self) -> VerificationResult:
        candidate = (self.root / self.relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return VerificationResult(
                Verdict.FAIL,
                "verification path escapes the allowed root",
                {"path": str(candidate)},
            )
        exists = candidate.exists()
        return VerificationResult(
            Verdict.PASS if exists else Verdict.FAIL,
            "file exists" if exists else "file does not exist",
            {"path": str(candidate)},
        )


class CommandVerifier:
    """Run an argv-style verification command without invoking a shell."""

    def __init__(
        self,
        root: str | Path,
        argv: list[str] | tuple[str, ...],
        *,
        timeout: float = 60.0,
    ) -> None:
        if not argv:
            raise ValueError("verification command cannot be empty")
        self.root = Path(root).resolve()
        self.argv = tuple(argv)
        self.timeout = timeout

    def verify(self) -> VerificationResult:
        try:
            completed = subprocess.run(
                self.argv,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return VerificationResult(
                Verdict.UNKNOWN,
                f"verification command timed out after {self.timeout}s",
                {"argv": list(self.argv), "stdout": exc.stdout, "stderr": exc.stderr},
            )
        evidence = {
            "argv": list(self.argv),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        return VerificationResult(
            Verdict.PASS if completed.returncode == 0 else Verdict.FAIL,
            "command passed" if completed.returncode == 0 else "command failed",
            evidence,
        )


class CompositeVerifier:
    def __init__(self, verifiers: list[Verifier], *, require_all: bool = True) -> None:
        if not verifiers:
            raise ValueError("CompositeVerifier requires at least one verifier")
        self.verifiers = list(verifiers)
        self.require_all = require_all

    def verify(self) -> VerificationResult:
        results = [verifier.verify() for verifier in self.verifiers]
        passed = [result.passed for result in results]
        ok = all(passed) if self.require_all else any(passed)
        if ok:
            verdict = Verdict.PASS
        elif any(result.verdict is Verdict.UNKNOWN for result in results):
            verdict = Verdict.UNKNOWN
        else:
            verdict = Verdict.FAIL
        return VerificationResult(
            verdict,
            "all verifiers passed" if self.require_all and ok else (
                "at least one verifier passed" if ok else "verification requirements not met"
            ),
            {
                "results": [
                    {
                        "verdict": result.verdict.value,
                        "summary": result.summary,
                        "evidence": result.evidence,
                    }
                    for result in results
                ]
            },
        )

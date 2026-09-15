from __future__ import annotations

import sys

from astra_codex.agent import ScriptedBackend
from astra_codex.repository_eval import RepositoryFixture, RepositoryFixtureHarness


def fixture() -> RepositoryFixture:
    return RepositoryFixture(
        case_id="fix_add",
        goal="Fix the add function and verify it.",
        files={
            "solution.py": "def add(a, b):\n    return a - b\n",
        },
        verify_argv=(
            sys.executable,
            "-c",
            "import solution; assert solution.add(2, 3) == 5",
        ),
        expected_files={"solution.py": "def add(a, b):\n    return a + b\n"},
    )


def test_repository_fixture_grades_final_state_not_claim(tmp_path) -> None:
    harness = RepositoryFixtureHarness(tmp_path)
    backend = ScriptedBackend(["I fixed it and all tests pass."])

    record = harness.run(fixture(), backend)

    assert not record.grade.passed
    assert record.grade.returncode != 0
    assert not record.grade.expected_files_ok
    assert record.tool_calls == 0


def test_repository_fixture_passes_after_real_edit_and_verification(tmp_path) -> None:
    harness = RepositoryFixtureHarness(tmp_path)
    backend = ScriptedBackend(
        [
            '{"tool":"edit","arguments":{"path":"solution.py","old":"return a - b","new":"return a + b"}}',
            '{"tool":"shell","arguments":{"command":"python -c \'import solution; assert solution.add(2, 3) == 5\'"}}',
            "Fixed the implementation and verified the repository state.",
        ]
    )

    record = harness.run(fixture(), backend)

    assert record.grade.passed
    assert record.grade.returncode == 0
    assert record.grade.expected_files_ok
    assert record.tool_calls == 2
    assert record.tool_failures == 0
    assert (tmp_path / "fix_add" / "solution.py").read_text() == (
        "def add(a, b):\n    return a + b\n"
    )

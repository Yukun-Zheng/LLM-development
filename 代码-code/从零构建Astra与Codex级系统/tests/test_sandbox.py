from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from astra_codex.sandbox import (
    RestrictedSubprocessSandbox,
    SandboxExecTool,
    SandboxLimits,
    SandboxPolicy,
)
from astra_codex.tools import ToolRegistry


def _python_sandbox(tmp_path, **limit_overrides):
    limits = SandboxLimits(**limit_overrides)
    executable = Path(sys.executable).name
    policy = SandboxPolicy(
        workspace_root=tmp_path,
        allowed_executables=frozenset({executable}),
        env_allowlist=frozenset({"PATH", "LANG", "LC_ALL", "PYTHONUNBUFFERED"}),
    )
    return RestrictedSubprocessSandbox(policy, limits=limits)


def test_sandbox_runs_argv_without_inheriting_unapproved_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ASTRA_CODEX_TEST_SECRET", "do-not-leak")
    sandbox = _python_sandbox(tmp_path)
    result = sandbox.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.getcwd()); "
                "print(os.getenv('ASTRA_CODEX_TEST_SECRET', '<missing>'))"
            ),
        ]
    )

    assert result.ok
    lines = result.stdout.strip().splitlines()
    assert Path(lines[0]) == tmp_path.resolve()
    assert lines[1] == "<missing>"


def test_sandbox_rejects_cwd_escape_and_disallowed_executable(tmp_path) -> None:
    sandbox = _python_sandbox(tmp_path)
    with pytest.raises(PermissionError, match="cwd escapes workspace"):
        sandbox.run([sys.executable, "-c", "print('x')"], cwd="../")

    with pytest.raises(PermissionError, match="executable is not allowed"):
        sandbox.run(["sh", "-c", "echo unsafe"])


def test_sandbox_rejects_unapproved_environment_key(tmp_path) -> None:
    sandbox = _python_sandbox(tmp_path)
    with pytest.raises(PermissionError, match="environment key is not allowed"):
        sandbox.run(
            [sys.executable, "-c", "print('x')"],
            env={"AWS_SECRET_ACCESS_KEY": "should-not-enter-child"},
        )


def test_sandbox_wall_timeout_kills_process_group(tmp_path) -> None:
    sandbox = _python_sandbox(tmp_path, wall_time_s=0.08)
    result = sandbox.run(
        [sys.executable, "-c", "import time; time.sleep(0.4); print('late')"]
    )

    assert result.timed_out
    assert not result.ok
    assert "late" not in result.stdout


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not Path("/proc/self/status").exists(),
    reason="Linux /proc is required to observe PR_SET_NO_NEW_PRIVS",
)
def test_linux_sandbox_sets_no_new_privs(tmp_path) -> None:
    sandbox = _python_sandbox(tmp_path)
    code = (
        "from pathlib import Path; "
        "line=next(x for x in Path('/proc/self/status').read_text().splitlines() "
        "if x.startswith('NoNewPrivs:')); print(line.split(':',1)[1].strip())"
    )
    result = sandbox.run([sys.executable, "-c", code])

    assert result.ok
    assert result.stdout.strip() == "1"
    assert result.no_new_privs_requested


def test_sandbox_exec_tool_surfaces_enforcement_metadata(tmp_path) -> None:
    sandbox = _python_sandbox(tmp_path)
    registry = ToolRegistry([SandboxExecTool(sandbox)])
    result = registry.execute(
        "sandbox_exec",
        {"argv": [sys.executable, "-c", "print('verified')"]},
    )

    assert result.ok
    assert result.output == "verified"
    assert result.metadata is not None
    assert result.metadata["timed_out"] is False
    assert result.metadata["security_boundary"] == "restricted-process-not-namespace-container"

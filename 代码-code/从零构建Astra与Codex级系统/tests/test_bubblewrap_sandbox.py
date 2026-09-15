from __future__ import annotations

import shutil
import socket
import threading
from pathlib import Path

import pytest

from astra_codex.bubblewrap_sandbox import (
    BubblewrapExecTool,
    BubblewrapPolicy,
    BubblewrapSandbox,
)
from astra_codex.sandbox import SandboxLimits
from astra_codex.tools import ToolRegistry


BWRAP = shutil.which("bwrap")
pytestmark = pytest.mark.skipif(BWRAP is None, reason="bubblewrap is not installed")
SYSTEM_PYTHON = "/usr/bin/python3"


def _sandbox(workspace: Path, *, writable: bool = True, network: bool = True):
    """Use shared networking unless a test is specifically about net namespaces.

    GitHub-hosted runners currently allow bubblewrap mount/user/PID namespaces
    but may deny configuring a new loopback interface. Filesystem isolation must
    therefore remain independently testable instead of being hidden behind a
    network-namespace policy failure.
    """

    policy = BubblewrapPolicy(
        workspace_root=workspace,
        allowed_guest_executables=frozenset({SYSTEM_PYTHON}),
        workspace_writable=writable,
        network_enabled=network,
    )
    return BubblewrapSandbox(
        policy,
        bwrap_path=BWRAP,
        limits=SandboxLimits(wall_time_s=5.0, cpu_time_s=3),
    )


def test_bubblewrap_workspace_write_and_host_secret_isolation(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    host_secret = outside / "secret.txt"
    host_secret.write_text("HOST-SECRET", encoding="utf-8")

    sandbox = _sandbox(workspace)
    code = (
        "from pathlib import Path; "
        f"p=Path({str(host_secret)!r}); "
        "print('secret_visible=' + str(p.exists())); "
        "Path('created.txt').write_text('sandbox-output', encoding='utf-8')"
    )
    result = sandbox.run([SYSTEM_PYTHON, "-c", code])

    assert result.ok, result.stderr
    assert "secret_visible=False" in result.stdout
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "sandbox-output"


def test_bubblewrap_readonly_workspace_rejects_write(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace, writable=False)
    result = sandbox.run(
        [
            SYSTEM_PYTHON,
            "-c",
            "from pathlib import Path; Path('forbidden.txt').write_text('x')",
        ]
    )

    assert not result.ok
    assert not (workspace / "forbidden.txt").exists()


def test_bubblewrap_network_namespace_cannot_reach_host_loopback_when_supported(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(1.5)
    port = int(listener.getsockname()[1])
    accepted: list[bool] = []

    def accept_once() -> None:
        try:
            connection, _address = listener.accept()
        except TimeoutError:
            accepted.append(False)
            return
        except OSError:
            return
        else:
            accepted.append(True)
            connection.close()

    thread = threading.Thread(target=accept_once, daemon=True)
    thread.start()
    try:
        sandbox = _sandbox(workspace, network=False)
        code = (
            "import socket; "
            "s=socket.socket(); s.settimeout(0.5); "
            f"print(s.connect_ex(('127.0.0.1',{port})))"
        )
        result = sandbox.run([SYSTEM_PYTHON, "-c", code])
        if "Failed RTM_NEWADDR: Operation not permitted" in result.stderr:
            pytest.skip(
                "host runner denies bubblewrap loopback setup; network isolation remains unvalidated here"
            )
        assert result.ok, result.stderr
        assert int(result.stdout.strip()) != 0
        thread.join(timeout=2.0)
        assert accepted == [False]
    finally:
        listener.close()


def test_bubblewrap_rejects_unapproved_guest_executable(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace)
    with pytest.raises(PermissionError, match="guest executable is not allowed"):
        sandbox.run(["/usr/bin/env"])


def test_bubblewrap_tool_reports_namespace_boundary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace)
    registry = ToolRegistry([BubblewrapExecTool(sandbox)])
    result = registry.execute(
        "sandbox_exec",
        {"argv": [SYSTEM_PYTHON, "-c", "print('inside-namespace')"]},
    )

    assert result.ok, result.output
    assert "inside-namespace" in result.output
    assert result.metadata is not None
    assert result.metadata["network_enabled"] is True
    assert result.metadata["security_boundary"] == "linux-bubblewrap-namespaces-shared-kernel"

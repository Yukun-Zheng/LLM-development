from __future__ import annotations

import json
import os
import shutil
import socket
import threading
from pathlib import Path

import pytest

from astra_codex.agent import ScriptedBackend
from astra_codex.coding import build_coding_agent
from astra_codex.docker_sandbox import (
    DockerSandbox,
    DockerSandboxExecTool,
    DockerSandboxPolicy,
)
from astra_codex.sandbox import SandboxLimits
from astra_codex.tools import ToolRegistry


DOCKER = shutil.which("docker")
ENABLED = os.environ.get("ASTRA_CODEX_DOCKER_SECURITY_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not ENABLED or DOCKER is None,
    reason="Docker security tests run only in the dedicated sandbox-security job",
)
IMAGE = os.environ.get("ASTRA_CODEX_DOCKER_IMAGE", "python:3.12-slim")
GUEST_PYTHON = "/usr/local/bin/python"


def _sandbox(workspace: Path, *, writable: bool = True, network: bool = False):
    policy = DockerSandboxPolicy(
        workspace_root=workspace,
        image=IMAGE,
        allowed_guest_executables=frozenset({GUEST_PYTHON}),
        workspace_writable=writable,
        network_enabled=network,
        memory_bytes=256 * 1024 * 1024,
        pids_limit=32,
        cpus=1.0,
    )
    return DockerSandbox(
        policy,
        docker_path=DOCKER,
        limits=SandboxLimits(wall_time_s=10.0, max_output_chars=50_000),
    )


def test_docker_command_fixes_security_sensitive_runtime_flags(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace, writable=False, network=False)
    command, guest_cwd = sandbox.build_docker_argv(
        [GUEST_PYTHON, "-c", "print('x')"]
    )

    assert guest_cwd == "/workspace"
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges:true"
    assert command[command.index("--pids-limit") + 1] == "32"
    mount = command[command.index("--mount") + 1]
    assert str(workspace.resolve()) in mount
    assert "target=/workspace" in mount
    assert mount.endswith(",readonly")
    assert IMAGE in command


def test_docker_hides_host_secret_but_allows_declared_workspace_write(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "host-secret.txt"
    secret.write_text("HOST-SECRET", encoding="utf-8")

    sandbox = _sandbox(workspace)
    code = (
        "from pathlib import Path; "
        f"p=Path({str(secret)!r}); "
        "print('secret_visible=' + str(p.exists())); "
        "Path('/workspace/created.txt').write_text('container-output')"
    )
    result = sandbox.run([GUEST_PYTHON, "-c", code])

    assert result.ok, result.stderr
    assert "secret_visible=False" in result.stdout
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "container-output"


def test_docker_readonly_workspace_and_rootfs_reject_writes(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace, writable=False)

    workspace_write = sandbox.run(
        [
            GUEST_PYTHON,
            "-c",
            "from pathlib import Path; Path('/workspace/forbidden.txt').write_text('x')",
        ]
    )
    root_write = sandbox.run(
        [
            GUEST_PYTHON,
            "-c",
            "from pathlib import Path; Path('/etc/forbidden.txt').write_text('x')",
        ]
    )

    assert not workspace_write.ok
    assert not root_write.ok
    assert not (workspace / "forbidden.txt").exists()


def test_docker_network_none_cannot_reach_host_loopback(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
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
        result = sandbox.run([GUEST_PYTHON, "-c", code])
        assert result.ok, result.stderr
        assert int(result.stdout.strip()) != 0
        thread.join(timeout=2.5)
        assert accepted == [False]
    finally:
        listener.close()


def test_docker_does_not_inherit_parent_secret_environment(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ASTRA_CODEX_CONTAINER_SECRET", "do-not-leak")
    sandbox = _sandbox(workspace)
    result = sandbox.run(
        [
            GUEST_PYTHON,
            "-c",
            "import os; print(os.getenv('ASTRA_CODEX_CONTAINER_SECRET','<missing>'))",
        ]
    )

    assert result.ok, result.stderr
    assert result.stdout.strip() == "<missing>"


def test_docker_drops_capabilities_and_sets_no_new_privileges(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace)
    code = (
        "from pathlib import Path; "
        "lines=Path('/proc/self/status').read_text().splitlines(); "
        "d={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in lines if ':' in x}; "
        "print(d['CapEff']); print(d['NoNewPrivs'])"
    )
    result = sandbox.run([GUEST_PYTHON, "-c", code])

    assert result.ok, result.stderr
    cap_eff, no_new_privs = result.stdout.strip().splitlines()
    assert int(cap_eff, 16) == 0
    assert no_new_privs == "1"


def test_docker_tool_records_exact_container_boundary(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = _sandbox(workspace)
    registry = ToolRegistry([DockerSandboxExecTool(sandbox)])
    result = registry.execute(
        "sandbox_exec",
        {"argv": [GUEST_PYTHON, "-c", "print('container-verified')"]},
    )

    assert result.ok, result.output
    assert "container-verified" in result.output
    assert result.metadata is not None
    assert result.metadata["network_enabled"] is False
    assert result.metadata["read_only_root"] is True
    assert result.metadata["security_boundary"] == "docker-container-shared-kernel"
    assert str(result.metadata["image_id"]).startswith("sha256:")


def test_coding_agent_can_execute_inside_docker_sandbox(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    sandbox = _sandbox(tmp_path)
    call = json.dumps(
        {
            "tool": "sandbox_exec",
            "arguments": {
                "argv": [GUEST_PYTHON, "-c", "print('agent-in-container')"],
            },
        }
    )
    backend = ScriptedBackend([call, "container verification complete"])
    agent = build_coding_agent(
        backend,
        tmp_path,
        execution_sandbox=sandbox,
        max_steps=3,
    )

    tool_names = {spec.name for spec in agent.tools.specs}
    assert "sandbox_exec" in tool_names
    assert "shell" not in tool_names
    run = agent.run("verify command execution")
    assert run.final_answer == "container verification complete"
    observations = [m.content for m in run.messages if m.role == "tool"]
    assert len(observations) == 1
    assert "agent-in-container" in observations[0]

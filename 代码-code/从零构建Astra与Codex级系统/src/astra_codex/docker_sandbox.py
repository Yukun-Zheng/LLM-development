from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .sandbox import SandboxExecution, SandboxLimits
from .structured import ToolSpec
from .tools import ToolResult


@dataclass(frozen=True, slots=True)
class DockerSandboxPolicy:
    """Policy for a Docker-backed local container execution boundary.

    The backend deliberately fixes security-sensitive Docker flags rather than
    accepting arbitrary ``docker run`` options from the model. The container:

    - uses a read-only image root filesystem by default;
    - bind-mounts only the declared workspace at ``/workspace``;
    - uses ``--network none`` by default;
    - drops all Linux capabilities;
    - enables Docker's no-new-privileges security option;
    - applies PID, memory and CPU limits;
    - runs as the host caller's uid/gid instead of root;
    - receives only an explicit environment mapping.

    This is materially stronger than the restricted subprocess runner, but it
    still shares the host kernel and Docker daemon. It is not a VM boundary and
    does not protect against kernel/container-runtime vulnerabilities.
    """

    workspace_root: Path
    image: str = "python:3.12-slim"
    allowed_guest_executables: frozenset[str] = field(
        default_factory=lambda: frozenset({"/usr/local/bin/python"})
    )
    workspace_writable: bool = True
    network_enabled: bool = False
    read_only_root: bool = True
    pids_limit: int = 64
    memory_bytes: int = 256 * 1024 * 1024
    cpus: float = 1.0
    tmpfs_bytes: int = 64 * 1024 * 1024
    env: dict[str, str] = field(
        default_factory=lambda: {
            "HOME": "/tmp/home",
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
        }
    )

    def __post_init__(self) -> None:
        root = Path(self.workspace_root).resolve()
        object.__setattr__(self, "workspace_root", root)
        if not self.image or self.image.startswith("-"):
            raise ValueError("image must be a non-empty Docker image reference")
        if not self.allowed_guest_executables:
            raise ValueError("allowed_guest_executables cannot be empty")
        for executable in self.allowed_guest_executables:
            path = PurePosixPath(executable)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "allowed_guest_executables must contain absolute guest paths"
                )
        if self.pids_limit <= 0:
            raise ValueError("pids_limit must be positive")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.tmpfs_bytes <= 0:
            raise ValueError("tmpfs_bytes must be positive")
        if any(not key or "=" in key for key in self.env):
            raise ValueError("environment keys must be non-empty names")


class DockerSandbox:
    """Reference container sandbox driven through the Docker CLI.

    Model-controlled values never become Docker options: the image, mounts,
    network mode and resource limits all come from ``DockerSandboxPolicy``.
    The model can select only an allowlisted guest executable and its argv.
    """

    def __init__(
        self,
        policy: DockerSandboxPolicy,
        *,
        limits: SandboxLimits | None = None,
        docker_path: str | None = None,
    ) -> None:
        resolved = docker_path or shutil.which("docker")
        if resolved is None:
            raise FileNotFoundError("docker executable was not found")
        self.docker_path = str(Path(resolved).resolve())
        self.policy = policy
        self.limits = limits or SandboxLimits(wall_time_s=60.0)
        self.policy.workspace_root.mkdir(parents=True, exist_ok=True)

    def _validate_argv(self, argv: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if not argv:
            raise ValueError("argv cannot be empty")
        if not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        executable = argv[0]
        if executable not in self.policy.allowed_guest_executables:
            raise PermissionError(
                f"guest executable is not allowed by Docker sandbox policy: {executable}"
            )
        return tuple(argv)

    def _resolve_cwd(self, cwd: str | Path | None) -> str:
        root = self.policy.workspace_root
        host = root if cwd is None else (root / cwd).resolve()
        if host != root and root not in host.parents:
            raise PermissionError(f"sandbox cwd escapes workspace: {cwd}")
        if not host.exists() or not host.is_dir():
            raise FileNotFoundError(host)
        relative = host.relative_to(root)
        guest = PurePosixPath("/workspace") / PurePosixPath(relative.as_posix())
        return str(guest)

    def build_docker_argv(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str | Path | None = None,
    ) -> tuple[list[str], str]:
        checked = self._validate_argv(argv)
        guest_cwd = self._resolve_cwd(cwd)
        uid = os.getuid() if hasattr(os, "getuid") else 65534
        gid = os.getgid() if hasattr(os, "getgid") else 65534

        mount_spec = (
            "type=bind,source="
            + str(self.policy.workspace_root)
            + ",target=/workspace"
        )
        if not self.policy.workspace_writable:
            # Docker --mount uses the flag token `readonly`; `rw` is not a
            # valid field. Read-write is the default and therefore omitted.
            mount_spec += ",readonly"

        command = [
            self.docker_path,
            "run",
            "--rm",
            "--init",
            "--user",
            f"{uid}:{gid}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self.policy.pids_limit),
            "--memory",
            str(self.policy.memory_bytes),
            "--cpus",
            str(self.policy.cpus),
            "--workdir",
            guest_cwd,
            "--mount",
            mount_spec,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.policy.tmpfs_bytes}",
        ]
        if self.policy.read_only_root:
            command.append("--read-only")
        command.extend(["--network", "bridge" if self.policy.network_enabled else "none"])
        for key, value in sorted(self.policy.env.items()):
            command.extend(["--env", f"{key}={value}"])
        command.extend([self.policy.image, *checked])
        return command, guest_cwd

    def image_id(self) -> str | None:
        completed = subprocess.run(
            [
                self.docker_path,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self.policy.image,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        value = completed.stdout.strip()
        return value or None

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        stdin: str | None = None,
    ) -> SandboxExecution:
        command, guest_cwd = self.build_docker_argv(argv, cwd=cwd)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(stdin, timeout=self.limits.wall_time_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:  # pragma: no cover
                process.kill()
            stdout, stderr = process.communicate()

        max_chars = self.limits.max_output_chars
        truncated = len(stdout) > max_chars or len(stderr) > max_chars
        return SandboxExecution(
            argv=tuple(argv),
            cwd=guest_cwd,
            returncode=int(process.returncode),
            stdout=stdout[:max_chars],
            stderr=stderr[:max_chars],
            timed_out=timed_out,
            output_truncated=truncated,
            no_new_privs_requested=True,
        )


class DockerSandboxExecTool:
    spec = ToolSpec(
        name="sandbox_exec",
        description=(
            "Run one argv-style command in the Docker container sandbox. "
            "The image, mounts, network policy and resource limits are fixed by host policy."
        ),
        parameters={
            "type": "object",
            "required": ["argv"],
            "additionalProperties": False,
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
            },
        },
    )

    def __init__(self, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        result = self.sandbox.run(arguments["argv"], cwd=arguments.get("cwd"))
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return ToolResult(
            result.ok,
            output.rstrip(),
            {
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "output_truncated": result.output_truncated,
                "cwd": result.cwd,
                "argv": list(result.argv),
                "image": self.sandbox.policy.image,
                "image_id": self.sandbox.image_id(),
                "network_enabled": self.sandbox.policy.network_enabled,
                "workspace_writable": self.sandbox.policy.workspace_writable,
                "read_only_root": self.sandbox.policy.read_only_root,
                "security_boundary": "docker-container-shared-kernel",
            },
        )

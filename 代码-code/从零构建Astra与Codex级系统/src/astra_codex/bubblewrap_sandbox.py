from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX import fallback
    resource = None  # type: ignore[assignment]

from .sandbox import SandboxExecution, SandboxLimits
from .structured import ToolSpec
from .tools import ToolResult


@dataclass(frozen=True, slots=True)
class BubblewrapPolicy:
    """Namespace-level policy for the Linux bubblewrap reference backend.

    Unlike ``RestrictedSubprocessSandbox``, this backend builds a new mount,
    PID, IPC, UTS and (by default) network namespace. It constructs a minimal
    filesystem view instead of bind-mounting the host root.

    This is stronger isolation, but it is still not a VM/security proof. The
    host kernel is shared and bubblewrap availability/configuration is an
    external dependency.
    """

    workspace_root: Path
    allowed_guest_executables: frozenset[str]
    workspace_writable: bool = True
    network_enabled: bool = False
    env: dict[str, str] = field(
        default_factory=lambda: {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp/home",
            "LANG": "C.UTF-8",
        }
    )

    def __post_init__(self) -> None:
        root = Path(self.workspace_root).resolve()
        object.__setattr__(self, "workspace_root", root)
        if not self.allowed_guest_executables:
            raise ValueError("allowed_guest_executables cannot be empty")
        for executable in self.allowed_guest_executables:
            path = PurePosixPath(executable)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "allowed_guest_executables must contain absolute guest paths"
                )
        if any(not key or "=" in key for key in self.env):
            raise ValueError("environment keys must be non-empty names")


class BubblewrapSandbox:
    """Linux namespace sandbox using the external ``bwrap`` executable.

    Host filesystem visibility is intentionally constructed from a tiny set:

    - read-only ``/usr`` for system binaries/libraries;
    - synthetic ``/bin``, ``/lib`` and ``/lib64`` symlinks into ``/usr``;
    - proc/dev/tmpfs mounts created inside the namespace;
    - one workspace bind at ``/workspace``;
    - a minimal synthetic ``/etc`` with optional dynamic-linker cache/config.

    ``--unshare-all`` provides mount/PID/IPC/UTS/user/cgroup/network isolation;
    ``--share-net`` is added only when explicitly requested.
    """

    def __init__(
        self,
        policy: BubblewrapPolicy,
        *,
        limits: SandboxLimits | None = None,
        bwrap_path: str | None = None,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("BubblewrapSandbox requires a POSIX host")
        resolved_bwrap = bwrap_path or shutil.which("bwrap")
        if resolved_bwrap is None:
            raise FileNotFoundError("bwrap executable was not found")
        self.bwrap_path = str(Path(resolved_bwrap).resolve())
        self.policy = policy
        self.limits = limits or SandboxLimits()
        self.policy.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_cwd(self, cwd: str | Path | None) -> tuple[Path, str]:
        root = self.policy.workspace_root
        host = root if cwd is None else (root / cwd).resolve()
        if host != root and root not in host.parents:
            raise PermissionError(f"sandbox cwd escapes workspace: {cwd}")
        if not host.exists() or not host.is_dir():
            raise FileNotFoundError(host)
        relative = host.relative_to(root)
        guest = PurePosixPath("/workspace") / PurePosixPath(relative.as_posix())
        return host, str(guest)

    def _validate_argv(self, argv: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if not argv:
            raise ValueError("argv cannot be empty")
        if not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        executable = argv[0]
        if executable not in self.policy.allowed_guest_executables:
            raise PermissionError(
                f"guest executable is not allowed by sandbox policy: {executable}"
            )
        return tuple(argv)

    def _apply_limits(self) -> None:
        os.umask(0o077)
        if resource is None:
            return
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (self.limits.cpu_time_s, self.limits.cpu_time_s),
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (self.limits.max_file_bytes, self.limits.max_file_bytes),
        )
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (self.limits.max_open_files, self.limits.max_open_files),
        )
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (self.limits.max_processes, self.limits.max_processes),
            )
        if self.limits.max_memory_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.limits.max_memory_bytes, self.limits.max_memory_bytes),
            )
        if hasattr(resource, "RLIMIT_CORE"):
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    @staticmethod
    def _optional_ro_bind(argv: list[str], source: str, target: str) -> None:
        if Path(source).exists():
            argv.extend(["--ro-bind", source, target])

    def build_bwrap_argv(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str | Path | None = None,
    ) -> tuple[list[str], str]:
        checked = self._validate_argv(argv)
        _host_cwd, guest_cwd = self._resolve_cwd(cwd)

        command: list[str] = [
            self.bwrap_path,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
        ]
        if self.policy.network_enabled:
            command.append("--share-net")

        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/bin",
                "/bin",
                "--symlink",
                "usr/lib",
                "/lib",
            ]
        )
        if Path("/usr/lib64").exists():
            command.extend(["--symlink", "usr/lib64", "/lib64"])

        command.extend(["--dir", "/etc"])
        self._optional_ro_bind(command, "/etc/ld.so.cache", "/etc/ld.so.cache")
        self._optional_ro_bind(command, "/etc/ld.so.conf", "/etc/ld.so.conf")

        workspace_flag = "--bind" if self.policy.workspace_writable else "--ro-bind"
        command.extend(
            [
                workspace_flag,
                str(self.policy.workspace_root),
                "/workspace",
                "--chdir",
                guest_cwd,
                "--clearenv",
            ]
        )
        for key, value in sorted(self.policy.env.items()):
            command.extend(["--setenv", key, value])

        # All capabilities inside the user namespace are unnecessary for the
        # workloads demonstrated here. Bubblewrap accepts ALL as a capability
        # set token on current Linux builds.
        command.extend(["--cap-drop", "ALL", "--"])
        command.extend(checked)
        return command, guest_cwd

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        stdin: str | None = None,
    ) -> SandboxExecution:
        command, guest_cwd = self.build_bwrap_argv(argv, cwd=cwd)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=self._apply_limits,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(stdin, timeout=self.limits.wall_time_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()

        max_chars = self.limits.max_output_chars
        truncated = len(stdout) > max_chars or len(stderr) > max_chars
        stdout = stdout[:max_chars]
        stderr = stderr[:max_chars]
        return SandboxExecution(
            argv=tuple(argv),
            cwd=guest_cwd,
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            output_truncated=truncated,
            # bwrap uses a user namespace and capability drop; this field belongs
            # to the simpler restricted-process backend, so do not overstate it.
            no_new_privs_requested=False,
        )


class BubblewrapExecTool:
    spec = ToolSpec(
        name="sandbox_exec",
        description=(
            "Run one argv-style command inside the Linux bubblewrap namespace sandbox."
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

    def __init__(self, sandbox: BubblewrapSandbox) -> None:
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
                "network_enabled": self.sandbox.policy.network_enabled,
                "workspace_writable": self.sandbox.policy.workspace_writable,
                "security_boundary": "linux-bubblewrap-namespaces-shared-kernel",
            },
        )

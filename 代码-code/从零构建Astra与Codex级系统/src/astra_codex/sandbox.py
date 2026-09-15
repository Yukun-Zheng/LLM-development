from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # POSIX-only resource controls; import is unavailable on Windows.
    import resource
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    resource = None  # type: ignore[assignment]

from .structured import ToolSpec
from .tools import ToolResult


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Resource limits for one restricted subprocess.

    These limits are deliberately conservative teaching defaults. They bound
    resource consumption, but they do not create a filesystem or network
    namespace. A production sandbox needs an OS/container boundary in addition
    to this runner.
    """

    wall_time_s: float = 60.0
    cpu_time_s: int = 30
    max_file_bytes: int = 64 * 1024 * 1024
    max_open_files: int = 256
    max_processes: int = 64
    max_memory_bytes: int | None = None
    max_output_chars: int = 200_000

    def __post_init__(self) -> None:
        if self.wall_time_s <= 0:
            raise ValueError("wall_time_s must be positive")
        if self.cpu_time_s <= 0:
            raise ValueError("cpu_time_s must be positive")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        if self.max_processes <= 0:
            raise ValueError("max_processes must be positive")
        if self.max_memory_bytes is not None and self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive when set")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")


@dataclass(frozen=True, slots=True)
class SandboxExecution:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    no_new_privs_requested: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Explicit policy for the level-1 restricted process runner.

    Enforced here:
    - argv execution without a shell;
    - executable-name allowlist;
    - cwd must remain inside ``workspace_root``;
    - environment allowlist (secrets are not inherited by default);
    - wall-clock timeout;
    - POSIX rlimits where available;
    - Linux ``PR_SET_NO_NEW_PRIVS`` where requested.

    Not enforced here:
    - filesystem mount namespace / chroot;
    - network namespace;
    - seccomp syscall filtering;
    - container/VM boundary;
    - credential broker isolation.

    This distinction is intentional. The project must not call an application
    policy wrapper a complete OS sandbox.
    """

    workspace_root: Path
    allowed_executables: frozenset[str]
    env_allowlist: frozenset[str] = field(
        default_factory=lambda: frozenset({"PATH", "LANG", "LC_ALL", "PYTHONUNBUFFERED"})
    )
    fixed_env: dict[str, str] = field(default_factory=dict)
    require_no_new_privs: bool = True

    def __post_init__(self) -> None:
        root = Path(self.workspace_root).resolve()
        object.__setattr__(self, "workspace_root", root)
        if not self.allowed_executables:
            raise ValueError("allowed_executables cannot be empty")
        if any(not item or "/" in item or "\\" in item for item in self.allowed_executables):
            raise ValueError("allowed_executables must contain executable basenames only")
        forbidden_fixed = set(self.fixed_env) - set(self.env_allowlist)
        if forbidden_fixed:
            raise ValueError(
                "fixed_env contains keys outside env_allowlist: "
                + ", ".join(sorted(forbidden_fixed))
            )


class RestrictedSubprocessSandbox:
    """Level-1 restricted subprocess runner with enforceable local guards.

    The name *sandbox* here refers to a restricted process execution layer, not
    a claim of full hostile-code containment. See ``SandboxPolicy`` for the
    exact boundary. The next security layer is a namespace/container backend.
    """

    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.policy = policy
        self.limits = limits or SandboxLimits()
        self.policy.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        root = self.policy.workspace_root
        candidate = root if cwd is None else (root / cwd).resolve()
        if candidate != root and root not in candidate.parents:
            raise PermissionError(f"sandbox cwd escapes workspace: {cwd}")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not candidate.is_dir():
            raise NotADirectoryError(candidate)
        return candidate

    def _validate_argv(self, argv: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if not argv:
            raise ValueError("argv cannot be empty")
        if not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        executable = Path(argv[0]).name
        if executable not in self.policy.allowed_executables:
            raise PermissionError(f"executable is not allowed by sandbox policy: {executable}")
        return tuple(argv)

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        allowed = self.policy.env_allowlist
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env.update(self.policy.fixed_env)
        for key, value in (extra or {}).items():
            if key not in allowed:
                raise PermissionError(f"environment key is not allowed: {key}")
            if not isinstance(value, str):
                raise ValueError(f"environment value must be string: {key}")
            env[key] = value
        # PATH is needed when argv[0] is a basename. Keep a deterministic
        # fallback rather than inheriting every caller environment variable.
        env.setdefault("PATH", os.defpath)
        env["PWD"] = str(self.policy.workspace_root)
        return env

    @staticmethod
    def _set_no_new_privs() -> None:
        # Linux prctl(PR_SET_NO_NEW_PRIVS, 1). This blocks gaining additional
        # privilege through execve (e.g. setuid binaries). It is not seccomp.
        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_no_new_privs = 38
        result = libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0)
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

    def _preexec(self) -> None:
        os.umask(0o077)
        if resource is not None:
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

        if self.policy.require_no_new_privs and sys.platform.startswith("linux"):
            self._set_no_new_privs()

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
    ) -> SandboxExecution:
        checked_argv = self._validate_argv(argv)
        resolved_cwd = self._resolve_cwd(cwd)
        child_env = self._build_env(env)

        popen_kwargs: dict[str, Any] = {
            "args": list(checked_argv),
            "cwd": resolved_cwd,
            "env": child_env,
            "text": True,
            "stdin": subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "start_new_session": True,
        }
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = self._preexec

        process = subprocess.Popen(**popen_kwargs)
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
            else:  # pragma: no cover - Windows fallback
                process.kill()
            stdout, stderr = process.communicate()

        max_chars = self.limits.max_output_chars
        truncated = len(stdout) > max_chars or len(stderr) > max_chars
        stdout = stdout[:max_chars]
        stderr = stderr[:max_chars]
        return SandboxExecution(
            argv=checked_argv,
            cwd=str(resolved_cwd),
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            output_truncated=truncated,
            no_new_privs_requested=(
                self.policy.require_no_new_privs and sys.platform.startswith("linux")
            ),
        )


class SandboxExecTool:
    """Agent tool exposing argv-only execution through RestrictedSubprocessSandbox."""

    spec = ToolSpec(
        name="sandbox_exec",
        description=(
            "Run one argv-style command through the restricted subprocess sandbox. "
            "Shell operators such as pipes/redirection are not interpreted."
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

    def __init__(self, sandbox: RestrictedSubprocessSandbox) -> None:
        self.sandbox = sandbox

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        argv = arguments["argv"]
        cwd = arguments.get("cwd")
        execution = self.sandbox.run(argv, cwd=cwd)
        output = execution.stdout
        if execution.stderr:
            output += ("\n" if output else "") + execution.stderr
        return ToolResult(
            execution.ok,
            output.rstrip(),
            {
                "returncode": execution.returncode,
                "timed_out": execution.timed_out,
                "output_truncated": execution.output_truncated,
                "cwd": execution.cwd,
                "argv": list(execution.argv),
                "no_new_privs_requested": execution.no_new_privs_requested,
                "security_boundary": "restricted-process-not-namespace-container",
            },
        )

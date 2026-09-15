from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeAlias

from .agent import Agent, ModelBackend
from .bubblewrap_sandbox import BubblewrapExecTool, BubblewrapSandbox
from .context import ContextStore
from .docker_sandbox import DockerSandbox, DockerSandboxExecTool
from .editing import ExactEditTool
from .instructions import ProjectInstructionResolver
from .repo_map import RepoMapTool
from .sandbox import RestrictedSubprocessSandbox, SandboxExecTool
from .tools import FilesystemTool, GitTool, ShellTool, ToolRegistry


ExecutionSandbox: TypeAlias = (
    RestrictedSubprocessSandbox | BubblewrapSandbox | DockerSandbox
)


CODING_SYSTEM_PROMPT = """You are a repository coding agent.
Work empirically: inspect before editing, edit the smallest coherent surface,
run the relevant tests or checks, inspect failures, and iterate.
Never claim that tests pass unless an execution observation shows they pass.
Prefer exact-edit for localized modifications because it fails safely when the
expected context is ambiguous. Use repo_map before blind repository-wide reads.
End with a concise summary of changes and verification performed.
Tool calls must be exactly one JSON object:
{"tool":"<tool-name>","arguments":{...}}
"""


def _execution_tool(sandbox: ExecutionSandbox):  # type: ignore[no-untyped-def]
    if isinstance(sandbox, DockerSandbox):
        return DockerSandboxExecTool(sandbox), "docker-container"
    if isinstance(sandbox, BubblewrapSandbox):
        return BubblewrapExecTool(sandbox), "bubblewrap-namespace"
    if isinstance(sandbox, RestrictedSubprocessSandbox):
        return SandboxExecTool(sandbox), "restricted-process"
    raise TypeError(f"unsupported execution sandbox: {type(sandbox).__name__}")


def build_coding_agent(
    backend: ModelBackend,
    repository_root: str | Path,
    *,
    working_directory: str | Path | None = None,
    instruction_max_bytes: int = 32_768,
    fallback_instruction_filenames: Iterable[str] = (),
    context_store: ContextStore | None = None,
    execution_sandbox: ExecutionSandbox | None = None,
    max_steps: int = 40,
) -> Agent:
    """Construct a repository coding loop with scoped instructions and pluggable execution.

    ``AGENTS.md`` project instructions are resolved from the nearest marked
    project root to ``working_directory`` using ``ProjectInstructionResolver``.
    The exact same resolved text is inserted into the system prompt. If
    ``context_store`` is supplied, every model-visible instruction is also
    persisted as a typed ``INSTRUCTION`` fragment carrying source/scope/
    truncation provenance.

    Execution deliberately exposes a progression of security boundaries:

    - no sandbox: legacy ``ShellTool`` / ``bash -lc`` for earliest teaching;
    - ``RestrictedSubprocessSandbox``: argv/env/resource/no-new-privs guards;
    - ``BubblewrapSandbox``: Linux namespace backend where host policy permits;
    - ``DockerSandbox``: container boundary with fixed host-side Docker policy.

    All sandboxed modes expose exactly one model-facing tool name,
    ``sandbox_exec``. The model cannot choose Docker/bubblewrap flags; those are
    fixed by the host-side sandbox policy. Tool metadata states the exact
    boundary so a higher-level policy/verifier can distinguish restricted
    process execution from a container.
    """

    root = Path(repository_root).resolve()
    cwd = root if working_directory is None else Path(working_directory).resolve()
    try:
        cwd.relative_to(root)
    except ValueError as exc:
        raise ValueError("working_directory must be inside repository_root") from exc

    if execution_sandbox is not None and execution_sandbox.policy.workspace_root != root:
        raise ValueError("execution_sandbox workspace_root must equal repository_root")

    resolver = ProjectInstructionResolver(
        fallback_filenames=fallback_instruction_filenames,
        max_bytes=instruction_max_bytes,
    )
    resolved = resolver.resolve(cwd)
    if context_store is not None:
        context_store.add_resolved_instructions(resolved)

    system_prompt = CODING_SYSTEM_PROMPT
    if resolved.sources:
        system_prompt += (
            "\n\n# Project instructions\n"
            "The following repository-scoped instructions are authoritative for "
            "this working directory:\n\n"
            + resolved.text
        )

    if execution_sandbox is None:
        execution_tool = ShellTool(root)
    else:
        execution_tool, boundary = _execution_tool(execution_sandbox)
        system_prompt += (
            "\n\n# Sandboxed command execution\n"
            f"Execution boundary: {boundary}. "
            "Arbitrary commands are available only through sandbox_exec. Pass argv as "
            "a JSON string array; shell operators, pipelines and redirection are not "
            "interpreted by the host launcher. Security-sensitive sandbox/container "
            "configuration is fixed by host policy, not by model output. A successful "
            "exit code is evidence only for that command, not proof of overall task correctness."
        )

    tools = ToolRegistry(
        [
            RepoMapTool(root),
            FilesystemTool(root),
            ExactEditTool(root),
            execution_tool,
            GitTool(root),
        ]
    )
    return Agent(
        backend,
        tools,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )

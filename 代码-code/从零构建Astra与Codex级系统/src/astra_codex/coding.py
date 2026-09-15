from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .agent import Agent, ModelBackend
from .context import ContextStore
from .editing import ExactEditTool
from .instructions import ProjectInstructionResolver
from .repo_map import RepoMapTool
from .sandbox import RestrictedSubprocessSandbox, SandboxExecTool
from .tools import FilesystemTool, GitTool, ShellTool, ToolRegistry


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


def build_coding_agent(
    backend: ModelBackend,
    repository_root: str | Path,
    *,
    working_directory: str | Path | None = None,
    instruction_max_bytes: int = 32_768,
    fallback_instruction_filenames: Iterable[str] = (),
    context_store: ContextStore | None = None,
    execution_sandbox: RestrictedSubprocessSandbox | None = None,
    max_steps: int = 40,
) -> Agent:
    """Construct a transparent repository coding loop with scoped instructions.

    ``AGENTS.md`` project instructions are resolved from the nearest marked
    project root to ``working_directory`` using ``ProjectInstructionResolver``.
    The exact same resolved text is inserted into the system prompt. If
    ``context_store`` is supplied, every model-visible instruction is also
    persisted as a typed ``INSTRUCTION`` fragment carrying source/scope/
    truncation provenance.

    Execution has two deliberately distinct modes:

    - default teaching mode: the legacy ``ShellTool`` executes ``bash -lc``;
    - restricted mode: pass ``execution_sandbox`` and arbitrary commands are
      exposed only through argv-based ``sandbox_exec``.

    The restricted process runner enforces cwd/executable/env/resource guards
    and Linux ``no_new_privs`` but is still not a filesystem/network namespace
    or container boundary. The distinction remains explicit in tool metadata.
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
        execution_tool = SandboxExecTool(execution_sandbox)
        system_prompt += (
            "\n\n# Restricted command execution\n"
            "Arbitrary commands are available only through sandbox_exec. Pass argv as "
            "a JSON string array. Shell operators, pipelines and redirection are not "
            "interpreted. A successful exit code is still evidence only for that "
            "specific command, not proof of overall task correctness."
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

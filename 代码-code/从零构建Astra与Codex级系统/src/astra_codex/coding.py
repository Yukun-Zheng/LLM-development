from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .agent import Agent, ModelBackend
from .context import ContextStore
from .editing import ExactEditTool
from .instructions import ProjectInstructionResolver
from .repo_map import RepoMapTool
from .tools import FilesystemTool, GitTool, ShellTool, ToolRegistry


CODING_SYSTEM_PROMPT = """You are a repository coding agent.
Work empirically: inspect before editing, edit the smallest coherent surface,
run the relevant tests or checks, inspect failures, and iterate.
Never claim that tests pass unless the shell observation shows they pass.
Prefer exact-edit for localized modifications because it fails safely when the
expected context is ambiguous.  Use repo_map before blind repository-wide reads.
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
    max_steps: int = 40,
) -> Agent:
    """Construct a transparent repository coding loop with scoped instructions.

    ``AGENTS.md`` project instructions are resolved from the nearest marked
    project root to ``working_directory`` using ``ProjectInstructionResolver``.
    The exact same resolved text is inserted into the system prompt. If
    ``context_store`` is supplied, every model-visible instruction is also
    persisted as a typed ``INSTRUCTION`` fragment carrying source/scope/
    truncation provenance.

    Current primitives:
    - repo_map: cheap structural orientation;
    - filesystem: precise reads/writes/search;
    - edit: exact old->new replacement with ambiguity checks;
    - shell: tests/builds/commands;
    - git: status/diff/history.

    Later stages add unified-diff application, language servers, parallel
    worktree orchestration, browser/computer adapters, and stronger sandbox
    isolation.
    """

    root = Path(repository_root).resolve()
    cwd = root if working_directory is None else Path(working_directory).resolve()
    try:
        cwd.relative_to(root)
    except ValueError as exc:
        raise ValueError("working_directory must be inside repository_root") from exc

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

    tools = ToolRegistry(
        [
            RepoMapTool(root),
            FilesystemTool(root),
            ExactEditTool(root),
            ShellTool(root),
            GitTool(root),
        ]
    )
    return Agent(
        backend,
        tools,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )

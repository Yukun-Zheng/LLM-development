from __future__ import annotations

from pathlib import Path

from .agent import Agent, ModelBackend
from .editing import ExactEditTool
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
    max_steps: int = 40,
) -> Agent:
    """Construct a transparent Codex-class repository loop.

    Current primitives:
    - repo_map: cheap structural orientation;
    - filesystem: precise reads/writes/search;
    - edit: exact old->new replacement with ambiguity checks;
    - shell: tests/builds/commands;
    - git: status/diff/history.

    Later stages add unified-diff application, language servers, worktree worker
    orchestration, browser/computer adapters, and stronger sandbox isolation.
    """

    root = Path(repository_root).resolve()
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
        system_prompt=CODING_SYSTEM_PROMPT,
        max_steps=max_steps,
    )

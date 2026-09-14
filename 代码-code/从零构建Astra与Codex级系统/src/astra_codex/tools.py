from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .structured import ToolSpec, validate_schema


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] | None = None


class Tool(Protocol):
    spec: ToolSpec

    def run(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool: {name}")
        try:
            validate_schema(arguments, tool.spec.parameters)
            return tool.run(arguments)
        except Exception as exc:  # agent should observe failures, not crash the loop
            return ToolResult(False, f"{type(exc).__name__}: {exc}")


class RootedPaths:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"path escapes tool root: {relative}")
        return candidate


class FilesystemTool:
    spec = ToolSpec(
        name="filesystem",
        description="Read, write, list, and search text files under the repository root.",
        parameters={
            "type": "object",
            "required": ["action"],
            "additionalProperties": True,
            "properties": {
                "action": {"type": "string", "enum": ["read", "write", "list", "search"]},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "query": {"type": "string"},
            },
        },
    )

    def __init__(self, root: str | Path, *, max_read_chars: int = 100_000) -> None:
        self.paths = RootedPaths(root)
        self.max_read_chars = max_read_chars

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments["action"]
        path = arguments.get("path", ".")
        target = self.paths.resolve(path)

        if action == "read":
            text = target.read_text(encoding="utf-8")
            clipped = text[: self.max_read_chars]
            return ToolResult(True, clipped, {"truncated": len(clipped) != len(text)})

        if action == "write":
            content = arguments.get("content")
            if not isinstance(content, str):
                raise ValueError("write requires string content")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(True, f"wrote {len(content)} chars to {path}")

        if action == "list":
            if not target.is_dir():
                raise NotADirectoryError(path)
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult(True, "\n".join(entries))

        if action == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query:
                raise ValueError("search requires non-empty query")
            pattern = re.compile(query)
            matches: list[str] = []
            for file in target.rglob("*") if target.is_dir() else [target]:
                if not file.is_file() or file.stat().st_size > 2_000_000:
                    continue
                try:
                    lines = file.read_text(encoding="utf-8").splitlines()
                except (UnicodeDecodeError, OSError):
                    continue
                for number, line in enumerate(lines, 1):
                    if pattern.search(line):
                        rel = file.relative_to(self.paths.root)
                        matches.append(f"{rel}:{number}: {line}")
                        if len(matches) >= 200:
                            return ToolResult(True, "\n".join(matches), {"truncated": True})
            return ToolResult(True, "\n".join(matches), {"truncated": False})

        raise ValueError(f"unsupported action: {action}")


class ShellTool:
    spec = ToolSpec(
        name="shell",
        description="Run a shell command inside the repository working directory.",
        parameters={
            "type": "object",
            "required": ["command"],
            "additionalProperties": False,
            "properties": {"command": {"type": "string"}},
        },
    )

    def __init__(self, root: str | Path, *, timeout: float = 60.0) -> None:
        self.root = Path(root).resolve()
        self.timeout = timeout

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments["command"]
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            env={**os.environ, "PWD": str(self.root)},
        )
        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        return ToolResult(
            completed.returncode == 0,
            output.rstrip(),
            {"returncode": completed.returncode},
        )


class GitTool:
    spec = ToolSpec(
        name="git",
        description="Inspect Git status, diff, log, or show from the current repository.",
        parameters={
            "type": "object",
            "required": ["action"],
            "additionalProperties": True,
            "properties": {
                "action": {"type": "string", "enum": ["status", "diff", "log", "show"]},
                "target": {"type": "string"},
            },
        },
    )

    def __init__(self, root: str | Path, *, timeout: float = 30.0) -> None:
        self.root = Path(root).resolve()
        self.timeout = timeout

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments["action"]
        args = {
            "status": ["git", "status", "--short"],
            "diff": ["git", "diff", "--", arguments.get("target", ".")],
            "log": ["git", "log", "--oneline", "-20"],
            "show": ["git", "show", arguments.get("target", "HEAD")],
        }[action]
        completed = subprocess.run(
            args,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        output = completed.stdout + completed.stderr
        return ToolResult(completed.returncode == 0, output.rstrip(), {"returncode": completed.returncode})


def result_as_observation(result: ToolResult) -> str:
    return json.dumps(
        {"ok": result.ok, "output": result.output, "metadata": result.metadata},
        ensure_ascii=False,
    )

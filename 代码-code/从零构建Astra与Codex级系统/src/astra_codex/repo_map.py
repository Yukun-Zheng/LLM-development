from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .structured import ToolSpec
from .tools import ToolResult


@dataclass(frozen=True, slots=True)
class Symbol:
    path: str
    line: int
    kind: str
    name: str


def python_symbols(path: Path, root: Path) -> list[Symbol]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    rel = str(path.relative_to(root))
    symbols: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(Symbol(rel, node.lineno, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(rel, node.lineno, "class", node.name))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(Symbol(rel, child.lineno, "method", f"{node.name}.{child.name}"))
    return symbols


def markdown_headings(path: Path, root: Path) -> list[Symbol]:
    rel = str(path.relative_to(root))
    symbols: list[Symbol] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return symbols
    in_fence = False
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            if title:
                symbols.append(Symbol(rel, number, f"h{level}", title))
    return symbols


def build_repo_map(root: str | Path, *, max_files: int = 5_000) -> list[Symbol]:
    root = Path(root).resolve()
    symbols: list[Symbol] = []
    seen = 0
    ignored = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ignored.intersection(path.parts):
            continue
        seen += 1
        if seen > max_files:
            break
        if path.suffix == ".py":
            symbols.extend(python_symbols(path, root))
        elif path.suffix in {".md", ".markdown"}:
            symbols.extend(markdown_headings(path, root))
    return symbols


def render_repo_map(symbols: list[Symbol], *, limit: int = 1_000) -> str:
    return "\n".join(
        f"{symbol.path}:{symbol.line} [{symbol.kind}] {symbol.name}"
        for symbol in symbols[:limit]
    )


class RepoMapTool:
    spec = ToolSpec(
        name="repo_map",
        description="List top-level Python symbols and Markdown headings with file/line locations.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}},
        },
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        symbols = build_repo_map(self.root)
        query = arguments.get("query", "").lower().strip()
        if query:
            symbols = [
                symbol
                for symbol in symbols
                if query in symbol.name.lower() or query in symbol.path.lower()
            ]
        return ToolResult(True, render_repo_map(symbols), {"count": len(symbols)})

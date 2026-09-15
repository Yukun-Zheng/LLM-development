from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_AGENTS_FILENAME = "AGENTS.md"
LOCAL_OVERRIDE_FILENAME = "AGENTS.override.md"
DEFAULT_PROJECT_ROOT_MARKERS = (".git",)


@dataclass(frozen=True, slots=True)
class InstructionSource:
    source_path: Path
    scope_directory: Path
    candidate_name: str
    contents: str
    truncated: bool
    bytes_loaded: int


@dataclass(frozen=True, slots=True)
class ResolvedInstructions:
    project_root: Path
    cwd: Path
    sources: tuple[InstructionSource, ...]
    max_bytes: int

    @property
    def text(self) -> str:
        return "\n\n".join(source.contents for source in self.sources)

    @property
    def bytes_loaded(self) -> int:
        return sum(source.bytes_loaded for source in self.sources)


class ProjectInstructionResolver:
    """Clean-room AGENTS.md hierarchy resolver with source provenance.

    Semantics intentionally mirror the public Codex design at a small,
    inspectable level:

    * find the nearest ancestor project root using configured markers;
    * if no marker is found, inspect only cwd;
    * an empty marker list disables parent traversal;
    * walk root -> cwd, never above the project root;
    * in each directory prefer AGENTS.override.md, then AGENTS.md, then ordered
      fallback names, selecting at most one instruction file per directory;
    * enforce one total byte budget across the complete hierarchy;
    * preserve exact source/scope/truncation provenance for every fragment.

    This implementation uses the local filesystem only. It does not reproduce
    Codex's remote environment filesystem abstraction or trust gating.
    """

    def __init__(
        self,
        *,
        project_root_markers: Iterable[str] | None = None,
        fallback_filenames: Iterable[str] = (),
        max_bytes: int = 32_768,
    ) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")
        markers = (
            tuple(DEFAULT_PROJECT_ROOT_MARKERS)
            if project_root_markers is None
            else tuple(project_root_markers)
        )
        if any(not marker for marker in markers):
            raise ValueError("project root markers cannot contain empty names")
        fallbacks: list[str] = []
        for name in fallback_filenames:
            if not name:
                continue
            if name not in fallbacks and name not in {
                LOCAL_OVERRIDE_FILENAME,
                DEFAULT_AGENTS_FILENAME,
            }:
                fallbacks.append(name)
        self.project_root_markers = markers
        self.fallback_filenames = tuple(fallbacks)
        self.max_bytes = max_bytes

    @property
    def candidate_filenames(self) -> tuple[str, ...]:
        return (
            LOCAL_OVERRIDE_FILENAME,
            DEFAULT_AGENTS_FILENAME,
            *self.fallback_filenames,
        )

    def find_project_root(self, cwd: str | Path) -> Path:
        current = Path(cwd).resolve()
        if not current.is_dir():
            raise NotADirectoryError(current)
        if not self.project_root_markers:
            return current

        cursor = current
        while True:
            if any((cursor / marker).exists() for marker in self.project_root_markers):
                return cursor
            parent = cursor.parent
            if parent == cursor:
                return current
            cursor = parent

    @staticmethod
    def _directories_from_root(root: Path, cwd: Path) -> tuple[Path, ...]:
        root = root.resolve()
        cwd = cwd.resolve()
        try:
            relative = cwd.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"cwd {cwd} is outside project root {root}") from exc

        directories = [root]
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            directories.append(cursor)
        return tuple(directories)

    def discover(self, cwd: str | Path) -> tuple[Path, tuple[Path, ...]]:
        cwd_path = Path(cwd).resolve()
        root = self.find_project_root(cwd_path)
        found: list[Path] = []
        for directory in self._directories_from_root(root, cwd_path):
            for name in self.candidate_filenames:
                candidate = directory / name
                if candidate.is_file():
                    found.append(candidate)
                    break
        return root, tuple(found)

    def resolve(self, cwd: str | Path) -> ResolvedInstructions:
        cwd_path = Path(cwd).resolve()
        root, paths = self.discover(cwd_path)
        remaining = self.max_bytes
        sources: list[InstructionSource] = []

        for path in paths:
            if remaining == 0:
                break
            raw = path.read_bytes()
            loaded = raw[:remaining]
            truncated = len(raw) > remaining
            remaining -= len(loaded)
            text = loaded.decode("utf-8", errors="replace")
            if not text.strip():
                continue
            sources.append(
                InstructionSource(
                    source_path=path,
                    scope_directory=path.parent,
                    candidate_name=path.name,
                    contents=text,
                    truncated=truncated,
                    bytes_loaded=len(loaded),
                )
            )

        return ResolvedInstructions(
            project_root=root,
            cwd=cwd_path,
            sources=tuple(sources),
            max_bytes=self.max_bytes,
        )

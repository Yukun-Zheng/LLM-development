from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Worktree:
    name: str
    path: Path
    branch: str


class WorktreeManager:
    """Thin explicit wrapper over Git worktrees for parallel coding agents."""

    def __init__(self, repository_root: str | Path, workspace_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _git(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout.strip()

    def create(self, name: str, *, base: str = "HEAD") -> Worktree:
        if not name or any(ch in name for ch in "/\\\0"):
            raise ValueError("worktree name must be a simple path component")
        branch = f"agent/{name}"
        path = self.workspace_root / name
        if path.exists():
            raise FileExistsError(path)
        self._git("worktree", "add", "-b", branch, str(path), base)
        return Worktree(name, path, branch)

    def remove(self, worktree: Worktree, *, force: bool = False, delete_branch: bool = False) -> None:
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree.path))
        self._git(*args)
        if delete_branch:
            self._git("branch", "-D", worktree.branch)

    def diff(self, worktree: Worktree, *, base: str = "HEAD") -> str:
        completed = subprocess.run(
            ["git", "diff", base],
            cwd=worktree.path,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip())
        return completed.stdout

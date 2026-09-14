from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .structured import ToolSpec
from .tools import RootedPaths, ToolResult


@dataclass(frozen=True, slots=True)
class ExactEdit:
    path: str
    old: str
    new: str


def apply_exact_edit(root: str | Path, edit: ExactEdit) -> int:
    """Replace exactly one textual occurrence and return the replacement count.

    Exact-match editing is intentionally boring and auditable.  It is safer for
    an educational coding agent than silently applying a fuzzy patch to the
    wrong location.  A later stage can add a real unified-diff parser.
    """

    paths = RootedPaths(root)
    target = paths.resolve(edit.path)
    text = target.read_text(encoding="utf-8")
    count = text.count(edit.old)
    if count != 1:
        raise ValueError(
            f"expected old text exactly once in {edit.path}, found {count}; "
            "read more context and retry with a more specific edit"
        )
    target.write_text(text.replace(edit.old, edit.new, 1), encoding="utf-8")
    return 1


class ExactEditTool:
    spec = ToolSpec(
        name="edit",
        description="Apply one exact old->new text replacement in a repository file.",
        parameters={
            "type": "object",
            "required": ["path", "old", "new"],
            "additionalProperties": False,
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
        },
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        edit = ExactEdit(
            path=arguments["path"],
            old=arguments["old"],
            new=arguments["new"],
        )
        apply_exact_edit(self.root, edit)
        return ToolResult(True, f"edited {edit.path}")

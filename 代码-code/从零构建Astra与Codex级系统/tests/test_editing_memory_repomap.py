from pathlib import Path

import pytest

from astra_codex.editing import ExactEdit, apply_exact_edit
from astra_codex.memory import EventMemory
from astra_codex.repo_map import build_repo_map


def test_exact_edit_is_ambiguity_safe(tmp_path: Path) -> None:
    file = tmp_path / "a.py"
    file.write_text("x = 1\ny = 2\n", encoding="utf-8")
    apply_exact_edit(tmp_path, ExactEdit("a.py", "y = 2", "y = 3"))
    assert "y = 3" in file.read_text(encoding="utf-8")

    file.write_text("same\nsame\n", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_exact_edit(tmp_path, ExactEdit("a.py", "same", "new"))


def test_repo_map_extracts_python_symbols(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "class A:\n    def f(self):\n        pass\n\ndef g():\n    pass\n",
        encoding="utf-8",
    )
    symbols = build_repo_map(tmp_path)
    names = {symbol.name for symbol in symbols}
    assert {"A", "A.f", "g"} <= names


def test_event_memory_persists_and_searches(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    with EventMemory(db) as memory:
        memory.append("user", "fix cache parity")
        memory.append("tool", "pytest passed")
    with EventMemory(db) as memory:
        assert memory.recent(2)[0].content == "fix cache parity"
        assert memory.search("pytest")[0].content == "pytest passed"

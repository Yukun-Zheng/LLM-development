from __future__ import annotations

from dataclasses import dataclass, field

from astra_codex.agent import Message
from astra_codex.coding import build_coding_agent
from astra_codex.instructions import ProjectInstructionResolver
from astra_codex.structured import ToolSpec


@dataclass
class RecordingBackend:
    output: str
    calls: list[list[Message]] = field(default_factory=list)

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        del tools
        self.calls.append(list(messages))
        return self.output


def test_instruction_resolver_collects_root_to_cwd_with_provenance(tmp_path) -> None:
    root = tmp_path / "repo"
    nested = root / "packages" / "api"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("root rules", encoding="utf-8")
    (root / "packages" / "AGENTS.md").write_text("package rules", encoding="utf-8")
    (nested / "AGENTS.md").write_text("api rules", encoding="utf-8")

    resolved = ProjectInstructionResolver().resolve(nested)

    assert resolved.project_root == root.resolve()
    assert [source.contents for source in resolved.sources] == [
        "root rules",
        "package rules",
        "api rules",
    ]
    assert [source.scope_directory for source in resolved.sources] == [
        root.resolve(),
        (root / "packages").resolve(),
        nested.resolve(),
    ]
    assert resolved.text == "root rules\n\npackage rules\n\napi rules"


def test_override_wins_then_fallback_is_used_only_when_primary_missing(tmp_path) -> None:
    root = tmp_path / "repo"
    child = root / "child"
    child.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("ordinary", encoding="utf-8")
    (root / "AGENTS.override.md").write_text("override", encoding="utf-8")
    (root / "WORKFLOW.md").write_text("root fallback ignored", encoding="utf-8")
    (child / "WORKFLOW.md").write_text("child fallback", encoding="utf-8")

    resolved = ProjectInstructionResolver(
        fallback_filenames=["WORKFLOW.md"]
    ).resolve(child)

    assert [(source.candidate_name, source.contents) for source in resolved.sources] == [
        ("AGENTS.override.md", "override"),
        ("WORKFLOW.md", "child fallback"),
    ]


def test_global_byte_budget_truncates_later_instruction_and_preserves_source(tmp_path) -> None:
    root = tmp_path / "repo"
    child = root / "child"
    child.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_bytes(b"12345")
    (child / "AGENTS.md").write_bytes(b"abcdefghij")

    resolved = ProjectInstructionResolver(max_bytes=8).resolve(child)

    assert resolved.bytes_loaded == 8
    assert len(resolved.sources) == 2
    assert resolved.sources[0].contents == "12345"
    assert resolved.sources[0].truncated is False
    assert resolved.sources[1].contents == "abc"
    assert resolved.sources[1].truncated is True
    assert resolved.sources[1].source_path == child.resolve() / "AGENTS.md"


def test_no_project_marker_considers_only_cwd(tmp_path) -> None:
    parent = tmp_path / "parent"
    cwd = parent / "child"
    cwd.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("must not leak down", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("local only", encoding="utf-8")

    resolved = ProjectInstructionResolver().resolve(cwd)

    assert resolved.project_root == cwd.resolve()
    assert [source.contents for source in resolved.sources] == ["local only"]


def test_empty_root_marker_list_disables_parent_traversal(tmp_path) -> None:
    root = tmp_path / "repo"
    cwd = root / "nested"
    cwd.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("root ignored", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("cwd", encoding="utf-8")

    resolved = ProjectInstructionResolver(project_root_markers=[]).resolve(cwd)

    assert resolved.project_root == cwd.resolve()
    assert [source.contents for source in resolved.sources] == ["cwd"]


def test_resolver_never_walks_above_nearest_project_root(tmp_path) -> None:
    outer = tmp_path / "outer"
    root = outer / "repo"
    cwd = root / "nested"
    cwd.mkdir(parents=True)
    (outer / "AGENTS.md").write_text("outer must not appear", encoding="utf-8")
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("repo", encoding="utf-8")

    resolved = ProjectInstructionResolver().resolve(cwd)

    assert [source.contents for source in resolved.sources] == ["repo"]


def test_coding_agent_injects_only_in_scope_project_instructions(tmp_path) -> None:
    root = tmp_path / "repo"
    cwd = root / "packages" / "api"
    sibling = root / "packages" / "web"
    cwd.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("ROOT-INSTRUCTION", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("API-INSTRUCTION", encoding="utf-8")
    (sibling / "AGENTS.md").write_text("WEB-MUST-NOT-APPEAR", encoding="utf-8")

    backend = RecordingBackend("done")
    agent = build_coding_agent(backend, root, working_directory=cwd)
    run = agent.run("inspect the repository")

    assert run.final_answer == "done"
    system_message = backend.calls[0][0]
    assert system_message.role == "system"
    assert "ROOT-INSTRUCTION" in system_message.content
    assert "API-INSTRUCTION" in system_message.content
    assert "WEB-MUST-NOT-APPEAR" not in system_message.content

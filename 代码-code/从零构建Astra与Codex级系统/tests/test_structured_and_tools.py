from pathlib import Path

import pytest

from astra_codex.agent import ScriptedBackend
from astra_codex.coding import build_coding_agent
from astra_codex.structured import SchemaError, parse_tool_call, validate_schema
from astra_codex.tools import FilesystemTool


def test_parse_tool_call() -> None:
    call = parse_tool_call('{"tool":"filesystem","arguments":{"action":"read","path":"a.txt"}}')
    assert call is not None
    assert call.tool == "filesystem"
    assert call.arguments["path"] == "a.txt"


def test_schema_subset() -> None:
    schema = {
        "type": "object",
        "required": ["x"],
        "additionalProperties": False,
        "properties": {"x": {"type": "integer"}},
    }
    validate_schema({"x": 1}, schema)
    with pytest.raises(SchemaError):
        validate_schema({"x": "bad"}, schema)


def test_filesystem_cannot_escape_root(tmp_path: Path) -> None:
    tool = FilesystemTool(tmp_path)
    result = tool.run({"action": "write", "path": "hello.txt", "content": "hi"})
    assert result.ok
    assert (tmp_path / "hello.txt").read_text() == "hi"
    with pytest.raises(PermissionError):
        tool.paths.resolve("../escape.txt")


def test_coding_agent_observe_act_loop(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()  # only to make the fixture visibly repo-like
    backend = ScriptedBackend(
        [
            '{"tool":"filesystem","arguments":{"action":"write","path":"answer.txt","content":"42"}}',
            '{"tool":"filesystem","arguments":{"action":"read","path":"answer.txt"}}',
            "Created answer.txt and verified its contents are 42.",
        ]
    )
    agent = build_coding_agent(backend, tmp_path, max_steps=5)
    run = agent.run("Create answer.txt containing 42 and verify it.")
    assert not run.stopped_by_limit
    assert (tmp_path / "answer.txt").read_text() == "42"
    assert "verified" in run.final_answer.lower()

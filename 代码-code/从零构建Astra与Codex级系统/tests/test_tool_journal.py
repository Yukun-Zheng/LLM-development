from __future__ import annotations

from dataclasses import dataclass

import pytest

from astra_codex.structured import ToolSpec
from astra_codex.tool_journal import (
    DurableToolJournal,
    ExecutionStatus,
    JournaledToolExecutor,
)
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class CountingTool:
    name: str = "counter"
    calls: int = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Increment a counter and echo an optional value.",
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {"value": {"type": "integer"}},
            },
        )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.calls += 1
        value = int(arguments.get("value", self.calls))
        return ToolResult(True, str(value), {"calls": self.calls})


def test_completed_execution_replays_without_duplicate_tool_body(tmp_path) -> None:
    tool = CountingTool()
    db = tmp_path / "tools.sqlite"
    with DurableToolJournal(db) as journal:
        executor = JournaledToolExecutor(ToolRegistry([tool]), journal)
        first = executor.execute("turn-1:tool-1", "counter", {"value": 7})
        second = executor.execute("turn-1:tool-1", "counter", {"value": 7})

        assert first.ok and second.ok
        assert first.output == second.output == "7"
        assert tool.calls == 1
        assert second.metadata is not None
        assert second.metadata["replayed_from_journal"] is True
        assert journal.get("turn-1:tool-1").status is ExecutionStatus.COMPLETED


def test_completed_result_survives_process_restart(tmp_path) -> None:
    tool = CountingTool()
    db = tmp_path / "tools.sqlite"
    with DurableToolJournal(db) as journal:
        executor = JournaledToolExecutor(ToolRegistry([tool]), journal)
        executor.execute("stable-key", "counter", {"value": 3})
        assert tool.calls == 1

    fresh_tool = CountingTool()
    with DurableToolJournal(db) as reopened:
        executor = JournaledToolExecutor(ToolRegistry([fresh_tool]), reopened)
        replay = executor.execute("stable-key", "counter", {"value": 3})

        assert replay.ok
        assert replay.output == "3"
        assert fresh_tool.calls == 0
        assert replay.metadata is not None
        assert replay.metadata["replayed_from_journal"] is True


def test_started_record_blocks_blind_retry_for_non_idempotent_tool(tmp_path) -> None:
    tool = CountingTool()
    with DurableToolJournal(tmp_path / "tools.sqlite") as journal:
        journal.begin("in-doubt", "counter", {"value": 5})
        executor = JournaledToolExecutor(ToolRegistry([tool]), journal)

        result = executor.execute("in-doubt", "counter", {"value": 5})

        assert not result.ok
        assert tool.calls == 0
        assert result.metadata is not None
        assert result.metadata["in_doubt"] is True
        assert journal.get("in-doubt").status is ExecutionStatus.STARTED


def test_explicitly_retryable_in_doubt_tool_can_complete(tmp_path) -> None:
    tool = CountingTool()
    with DurableToolJournal(tmp_path / "tools.sqlite") as journal:
        journal.begin("retry-key", "counter", {"value": 9})
        executor = JournaledToolExecutor(
            ToolRegistry([tool]),
            journal,
            retryable_in_doubt_tools=frozenset({"counter"}),
        )

        result = executor.execute("retry-key", "counter", {"value": 9})

        assert result.ok
        assert result.output == "9"
        assert tool.calls == 1
        record = journal.get("retry-key")
        assert record is not None
        assert record.status is ExecutionStatus.COMPLETED


def test_idempotency_key_cannot_be_rebound_to_different_arguments(tmp_path) -> None:
    tool = CountingTool()
    with DurableToolJournal(tmp_path / "tools.sqlite") as journal:
        executor = JournaledToolExecutor(ToolRegistry([tool]), journal)
        executor.execute("same-key", "counter", {"value": 1})

        with pytest.raises(ValueError, match="reused"):
            executor.execute("same-key", "counter", {"value": 2})

        assert tool.calls == 1


def test_idempotency_key_cannot_be_rebound_to_different_tool(tmp_path) -> None:
    first = CountingTool(name="counter")
    second = CountingTool(name="other")
    with DurableToolJournal(tmp_path / "tools.sqlite") as journal:
        executor = JournaledToolExecutor(ToolRegistry([first, second]), journal)
        executor.execute("same-key", "counter", {})

        with pytest.raises(ValueError, match="reused"):
            executor.execute("same-key", "other", {})

        assert first.calls == 1
        assert second.calls == 0

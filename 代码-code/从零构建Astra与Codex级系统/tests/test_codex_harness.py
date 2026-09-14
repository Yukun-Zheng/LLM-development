from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra_codex.agent import ScriptedBackend
from astra_codex.codex_harness import (
    ApprovalDecision,
    CodexHarness,
    EventKind,
    SandboxPolicy,
    TurnSettings,
)
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class CountingEchoTool:
    calls: int = 0

    spec = ToolSpec(
        name="echo",
        description="Return the supplied text.",
        parameters={
            "type": "object",
            "required": ["text"],
            "additionalProperties": False,
            "properties": {"text": {"type": "string"}},
        },
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls += 1
        return ToolResult(True, str(arguments["text"]), {"calls": self.calls})


def test_turn_executes_tool_then_follows_up_until_final_answer() -> None:
    tool = CountingEchoTool()
    backend = ScriptedBackend(
        [
            '{"tool":"echo","arguments":{"text":"hello"}}',
            "done",
        ]
    )
    harness = CodexHarness(backend, ToolRegistry([tool]))

    result = harness.run_turn("say hello through the tool")

    assert result.final_answer == "done"
    assert result.model_steps == 2
    assert tool.calls == 1
    assert [event.kind for event in result.events] == [
        EventKind.TURN_STARTED,
        EventKind.MODEL_OUTPUT,
        EventKind.TOOL_STARTED,
        EventKind.TOOL_COMPLETED,
        EventKind.MODEL_OUTPUT,
        EventKind.TURN_COMPLETED,
    ]
    assert '"output": "hello"' in result.messages[-2].content


def test_approval_denial_becomes_observation_and_tool_does_not_run() -> None:
    tool = CountingEchoTool()
    backend = ScriptedBackend(
        [
            '{"tool":"echo","arguments":{"text":"blocked"}}',
            "I observed that approval was denied.",
        ]
    )
    harness = CodexHarness(
        backend,
        ToolRegistry([tool]),
        approval_handler=lambda _name, _arguments: ApprovalDecision.DENY,
    )

    result = harness.run_turn(
        "try an approval-gated action",
        settings=TurnSettings(
            sandbox_policy=SandboxPolicy.READ_ONLY,
            approval_required_tools=frozenset({"echo"}),
        ),
    )

    assert tool.calls == 0
    assert result.final_answer == "I observed that approval was denied."
    assert EventKind.APPROVAL_REQUESTED in [event.kind for event in result.events]
    assert EventKind.APPROVAL_DECIDED in [event.kind for event in result.events]
    assert EventKind.TOOL_STARTED not in [event.kind for event in result.events]
    assert "approval denied for tool: echo" in result.messages[-2].content


def test_approval_allow_executes_tool() -> None:
    tool = CountingEchoTool()
    backend = ScriptedBackend(
        [
            '{"tool":"echo","arguments":{"text":"approved"}}',
            "finished",
        ]
    )
    harness = CodexHarness(
        backend,
        ToolRegistry([tool]),
        approval_handler=lambda _name, _arguments: ApprovalDecision.ALLOW,
    )

    result = harness.run_turn(
        "perform the action",
        settings=TurnSettings(approval_required_tools=frozenset({"echo"})),
    )

    assert tool.calls == 1
    assert result.final_answer == "finished"
    decision_events = [
        event for event in result.events if event.kind is EventKind.APPROVAL_DECIDED
    ]
    assert decision_events[0].payload["decision"] == "allow"


def test_model_step_limit_is_an_explicit_stop_condition() -> None:
    tool = CountingEchoTool()
    backend = ScriptedBackend(
        ['{"tool":"echo","arguments":{"text":"again"}}'] * 2
    )
    harness = CodexHarness(backend, ToolRegistry([tool]), max_model_steps=2)

    result = harness.run_turn("keep going")

    assert result.stopped_by_limit is True
    assert result.model_steps == 2
    assert result.events[-1].kind is EventKind.TURN_STOPPED

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .agent import Message, ModelBackend
from .structured import parse_tool_call, render_tool_specs
from .tools import ToolRegistry, ToolResult, result_as_observation


class ApprovalDecision(str, Enum):
    """A deliberately small approval result for the educational harness."""

    ALLOW = "allow"
    DENY = "deny"


class SandboxPolicy(str, Enum):
    """Model-visible sandbox policy metadata.

    The policy is descriptive in this module. Real OS/container enforcement
    belongs in the sandbox layer and must not be confused with a prompt string.
    """

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class EventKind(str, Enum):
    TURN_STARTED = "turn_started"
    MODEL_OUTPUT = "model_output"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TURN_COMPLETED = "turn_completed"
    TURN_STOPPED = "turn_stopped"


@dataclass(frozen=True, slots=True)
class TurnSettings:
    """Per-turn execution policy.

    This mirrors the *separation of concerns* visible in the open-source Codex
    harness: model choice, approval policy and sandbox policy are turn/session
    settings, not hidden inside the model itself. It is not an API-compatible
    copy of OpenAI Codex types.
    """

    sandbox_policy: SandboxPolicy = SandboxPolicy.WORKSPACE_WRITE
    approval_required_tools: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    kind: EventKind
    payload: dict[str, object]


@dataclass(slots=True)
class CodexTurnResult:
    final_answer: str
    messages: list[Message]
    events: list[HarnessEvent]
    model_steps: int
    stopped_by_limit: bool = False


ApprovalHandler = Callable[[str, dict[str, object]], ApprovalDecision]
HarnessEventSink = Callable[[HarnessEvent], None]


DEFAULT_CODEX_STYLE_SYSTEM_PROMPT = """You are a repository execution agent.
Return either a final answer or exactly one JSON tool call:
{"tool":"<tool-name>","arguments":{...}}
Tool observations are authoritative environment feedback.
Never claim that an action succeeded merely because you intended to run it.
"""


class CodexHarness:
    """A clean-room educational Codex-style turn loop.

    The design is informed by public OpenAI Codex source code and protocol
    documentation, especially the Task/Turn loop, event stream, approval gate,
    tool execution and follow-up sampling pattern. The implementation below is
    independently written in Python and intentionally much smaller than the
    production Rust codebase.

    ``event_sink`` receives each event when it happens. This keeps the turn
    executor independent from any particular UI or transport while allowing a
    durable runtime to expose live progress through a control-plane event feed.

    Crucially, this class does *not* implement a security sandbox. A sandbox
    policy is carried as state so the later sandbox module can enforce it at the
    execution boundary.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tools: ToolRegistry,
        *,
        approval_handler: ApprovalHandler | None = None,
        event_sink: HarnessEventSink | None = None,
        system_prompt: str = DEFAULT_CODEX_STYLE_SYSTEM_PROMPT,
        max_model_steps: int = 32,
    ) -> None:
        if max_model_steps <= 0:
            raise ValueError("max_model_steps must be positive")
        self.backend = backend
        self.tools = tools
        self.approval_handler = approval_handler
        self.event_sink = event_sink
        self.system_prompt = system_prompt
        self.max_model_steps = max_model_steps

    def _record(
        self,
        events: list[HarnessEvent],
        kind: EventKind,
        payload: dict[str, object],
    ) -> HarnessEvent:
        event = HarnessEvent(kind, payload)
        events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)
        return event

    def run_turn(
        self,
        user_input: str,
        *,
        settings: TurnSettings | None = None,
    ) -> CodexTurnResult:
        settings = settings or TurnSettings()
        tool_contract = render_tool_specs(self.tools.specs)
        messages = [
            Message(
                "system",
                self.system_prompt
                + "\nSandbox policy: "
                + settings.sandbox_policy.value
                + "\nAvailable tools:\n"
                + tool_contract,
            ),
            Message("user", user_input),
        ]
        events: list[HarnessEvent] = []
        self._record(
            events,
            EventKind.TURN_STARTED,
            {
                "sandbox_policy": settings.sandbox_policy.value,
                "approval_required_tools": sorted(settings.approval_required_tools),
            },
        )

        for model_step in range(1, self.max_model_steps + 1):
            response = self.backend.generate(messages, self.tools.specs)
            messages.append(Message("assistant", response))
            self._record(
                events,
                EventKind.MODEL_OUTPUT,
                {"model_step": model_step, "content": response},
            )

            call = parse_tool_call(response)
            if call is None:
                self._record(
                    events,
                    EventKind.TURN_COMPLETED,
                    {"model_steps": model_step, "final_answer": response},
                )
                return CodexTurnResult(response, messages, events, model_step)

            if call.tool in settings.approval_required_tools:
                self._record(
                    events,
                    EventKind.APPROVAL_REQUESTED,
                    {"tool": call.tool, "arguments": call.arguments},
                )
                decision = (
                    self.approval_handler(call.tool, call.arguments)
                    if self.approval_handler is not None
                    else ApprovalDecision.DENY
                )
                self._record(
                    events,
                    EventKind.APPROVAL_DECIDED,
                    {"tool": call.tool, "decision": decision.value},
                )
                if decision is ApprovalDecision.DENY:
                    denied = ToolResult(
                        False,
                        f"approval denied for tool: {call.tool}",
                        {"approved": False},
                    )
                    messages.append(Message("tool", result_as_observation(denied)))
                    continue

            self._record(
                events,
                EventKind.TOOL_STARTED,
                {"tool": call.tool, "arguments": call.arguments},
            )
            result = self.tools.execute(call.tool, call.arguments)
            self._record(
                events,
                EventKind.TOOL_COMPLETED,
                {
                    "tool": call.tool,
                    "ok": result.ok,
                    "metadata": result.metadata,
                },
            )
            messages.append(Message("tool", result_as_observation(result)))

        final = "Stopped because max_model_steps was reached before the turn completed."
        self._record(
            events,
            EventKind.TURN_STOPPED,
            {"reason": "model_step_limit", "model_steps": self.max_model_steps},
        )
        return CodexTurnResult(
            final,
            messages,
            events,
            self.max_model_steps,
            stopped_by_limit=True,
        )

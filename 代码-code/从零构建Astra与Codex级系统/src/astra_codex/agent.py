from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .structured import ToolSpec, parse_tool_call, render_tool_specs
from .tools import ToolRegistry, result_as_observation


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


class ModelBackend(Protocol):
    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str: ...


@dataclass(slots=True)
class AgentRun:
    final_answer: str
    messages: list[Message]
    steps: int
    stopped_by_limit: bool = False


DEFAULT_SYSTEM_PROMPT = """You are an execution agent.
You can either return a final natural-language answer or exactly one JSON tool call:
{"tool":"<tool-name>","arguments":{...}}
Use tools when the task requires observing or changing the environment.
After each tool call, inspect the observation before deciding the next action.
Do not claim an action succeeded unless an observation verifies it.
"""


class Agent:
    """The smallest inspectable observe-think-act loop.

    The backend may be our own LM runtime, a public checkpoint adapter, or a
    scripted fake used in unit tests.  Tool execution and state transitions do
    not depend on a particular model vendor.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tools: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, goal: str) -> AgentRun:
        tool_contract = render_tool_specs(self.tools.specs)
        messages = [
            Message("system", self.system_prompt + "\nAvailable tools:\n" + tool_contract),
            Message("user", goal),
        ]

        for step in range(1, self.max_steps + 1):
            response = self.backend.generate(messages, self.tools.specs)
            messages.append(Message("assistant", response))
            call = parse_tool_call(response)
            if call is None:
                return AgentRun(response, messages, step)

            result = self.tools.execute(call.tool, call.arguments)
            messages.append(Message("tool", result_as_observation(result)))

        return AgentRun(
            "Stopped because max_steps was reached before a final answer.",
            messages,
            self.max_steps,
            stopped_by_limit=True,
        )


class ScriptedBackend:
    """Deterministic backend for testing the agent loop without trained weights."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.index = 0

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        del messages, tools
        if self.index >= len(self.outputs):
            raise RuntimeError("ScriptedBackend ran out of outputs")
        output = self.outputs[self.index]
        self.index += 1
        return output

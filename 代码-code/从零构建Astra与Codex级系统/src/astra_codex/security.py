from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .tools import ToolRegistry, ToolResult


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionProfile:
    """Deterministic tool policy evaluated before execution.

    This is an enforceable application-layer permission gate: a denied tool is
    not dispatched to ``ToolRegistry``. It is still not an OS/container sandbox;
    subprocess, filesystem, network and secret isolation require a lower-level
    security boundary.
    """

    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    approval_tools: frozenset[str] = field(default_factory=frozenset)
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    default: PermissionDecision = PermissionDecision.REQUIRE_APPROVAL

    def decide(self, tool_name: str) -> PermissionDecision:
        if tool_name in self.denied_tools:
            return PermissionDecision.DENY
        if tool_name in self.approval_tools:
            return PermissionDecision.REQUIRE_APPROVAL
        if tool_name in self.allowed_tools:
            return PermissionDecision.ALLOW
        return self.default


ApprovalCallback = Callable[[str, dict[str, object]], bool]


class GuardedToolExecutor:
    """Permission-enforcing facade over ``ToolRegistry``.

    The important property is ordering:

        proposal -> policy -> optional approval -> dispatch

    A DENY decision, or a rejected approval, returns an observation without
    invoking the underlying tool body.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        profile: PermissionProfile,
        *,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.registry = registry
        self.profile = profile
        self.approval_callback = approval_callback

    @property
    def specs(self):  # type: ignore[no-untyped-def]
        return self.registry.specs

    def execute(self, tool_name: str, arguments: dict[str, object]) -> ToolResult:
        decision = self.profile.decide(tool_name)
        policy_metadata = {
            "permission_decision": decision.value,
            "tool": tool_name,
        }

        if decision is PermissionDecision.DENY:
            return ToolResult(
                False,
                f"permission policy denied tool: {tool_name}",
                policy_metadata,
            )

        if decision is PermissionDecision.REQUIRE_APPROVAL:
            approved = (
                self.approval_callback(tool_name, arguments)
                if self.approval_callback is not None
                else False
            )
            if not approved:
                return ToolResult(
                    False,
                    f"approval denied for tool: {tool_name}",
                    {**policy_metadata, "approved": False},
                )
            policy_metadata["approved"] = True

        result = self.registry.execute(tool_name, arguments)
        return ToolResult(
            result.ok,
            result.output,
            {**result.metadata, **policy_metadata},
        )

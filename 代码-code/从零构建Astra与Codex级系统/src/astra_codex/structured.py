from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]


class SchemaError(ValueError):
    pass


def validate_schema(value: Any, schema: dict[str, Any], *, path: str = "$" ) -> None:
    """Validate a deliberately small JSON-Schema subset used by our tools."""

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} is not in enum {schema['enum']!r}")

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise SchemaError(f"{path}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise SchemaError(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], path=f"{path}.{key}")
            elif additional is False:
                raise SchemaError(f"{path}: unexpected key {key!r}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise SchemaError(f"{path}: expected array")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            validate_schema(item, item_schema, path=f"{path}[{index}]")
        return

    checks = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    if expected in checks and not isinstance(value, checks[expected]):
        raise SchemaError(f"{path}: expected {expected}")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse the explicit educational protocol:

    {"tool": "filesystem", "arguments": {"action": "read", "path": "README.md"}}

    If the model returns ordinary prose, return None and treat it as a final
    answer.  Production systems usually use grammar-constrained decoding or an
    API-native tool-call channel; this parser keeps the protocol inspectable.
    """

    candidate = strip_code_fence(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"tool", "arguments"}:
        return None
    if not isinstance(payload["tool"], str) or not isinstance(payload["arguments"], dict):
        return None
    return ToolCall(tool=payload["tool"], arguments=payload["arguments"])


def render_tool_specs(specs: list[ToolSpec]) -> str:
    payload = [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in specs
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)

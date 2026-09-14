from __future__ import annotations

"""A deliberately small, inspectable MCP teaching implementation.

This module targets the 2026-07-28 stateless protocol direction and implements
only a tiny subset needed to understand the wire semantics:

- JSON-RPC 2.0 request / response envelopes;
- server/discover;
- tools/list;
- tools/call;
- an in-process transport used by tests.

It is NOT a production MCP SDK. Authentication, HTTP/stdio transports, streaming,
extensions, resources, prompts, and the rest of the specification are future
layers. The point is to make protocol mechanics visible before using an SDK.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from .tools import ToolRegistry

MCP_PROTOCOL_VERSION = "2026-07-28"


class MCPProtocolError(ValueError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class JSONRPCRequest:
    id: str | int
    method: str
    params: dict[str, Any]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }
        if self.meta:
            payload["params"] = {**self.params, "_meta": self.meta}
        return payload


@dataclass(frozen=True, slots=True)
class JSONRPCResponse:
    id: str | int | None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            payload["error"] = self.error
        else:
            payload["result"] = self.result
        return payload


class MCPTransport(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class MCPServer:
    """Stateless MCP server adapter over the project's ToolRegistry."""

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        name: str = "astra-codex-educational-server",
        version: str = "0.1.0",
    ) -> None:
        self.tools = tools
        self.name = name
        self.version = version

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("id")
        try:
            request = self._parse_request(payload)
            result = self._dispatch(request.method, request.params)
            return JSONRPCResponse(request.id, result=result).to_dict()
        except MCPProtocolError as exc:
            error = {"code": exc.code, "message": exc.message}
            if exc.data is not None:
                error["data"] = exc.data
            return JSONRPCResponse(request_id, error=error).to_dict()
        except Exception as exc:  # keep unexpected server failures visible
            return JSONRPCResponse(
                request_id,
                error={"code": -32603, "message": f"Internal error: {type(exc).__name__}: {exc}"},
            ).to_dict()

    def _parse_request(self, payload: dict[str, Any]) -> JSONRPCRequest:
        if not isinstance(payload, dict):
            raise MCPProtocolError(-32600, "Invalid Request")
        if payload.get("jsonrpc") != "2.0":
            raise MCPProtocolError(-32600, "jsonrpc must be '2.0'")
        if "id" not in payload or not isinstance(payload["id"], (str, int)):
            raise MCPProtocolError(-32600, "educational subset requires a string/integer id")
        method = payload.get("method")
        if not isinstance(method, str):
            raise MCPProtocolError(-32600, "method must be a string")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise MCPProtocolError(-32602, "params must be an object")
        meta = params.get("_meta")
        clean_params = {key: value for key, value in params.items() if key != "_meta"}
        return JSONRPCRequest(payload["id"], method, clean_params, meta if isinstance(meta, dict) else None)

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "server/discover":
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": {"tools": {"list": True, "call": True}},
            }

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "inputSchema": spec.parameters,
                    }
                    for spec in self.tools.specs
                ]
            }

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise MCPProtocolError(-32602, "tools/call requires string name")
            if not isinstance(arguments, dict):
                raise MCPProtocolError(-32602, "tools/call arguments must be an object")
            result = self.tools.execute(name, arguments)
            return {
                "content": [{"type": "text", "text": result.output}],
                "isError": not result.ok,
                "structuredContent": {
                    "ok": result.ok,
                    "metadata": result.metadata,
                },
            }

        raise MCPProtocolError(-32601, f"Method not found: {method}")


class InProcessMCPTransport:
    """Transport used to test protocol semantics before stdio/HTTP."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.server.handle(payload)


class MCPClient:
    def __init__(
        self,
        transport: MCPTransport,
        *,
        name: str = "astra-codex-educational-client",
        version: str = "0.1.0",
    ) -> None:
        self.transport = transport
        self.name = name
        self.version = version
        self._next_id = 1

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        payload = JSONRPCRequest(
            request_id,
            method,
            params or {},
            meta={
                "io.modelcontextprotocol/clientInfo": {
                    "name": self.name,
                    "version": self.version,
                }
            },
        ).to_dict()
        response = self.transport.request(payload)
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise MCPProtocolError(-32000, "invalid response envelope", response)
        if "error" in response:
            error = response["error"]
            raise MCPProtocolError(
                int(error.get("code", -32000)),
                str(error.get("message", "unknown MCP error")),
                error.get("data"),
            )
        return response.get("result")

    def discover(self) -> dict[str, Any]:
        result = self._call("server/discover")
        if not isinstance(result, dict):
            raise MCPProtocolError(-32000, "server/discover returned non-object")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._call("tools/list")
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            raise MCPProtocolError(-32000, "tools/list returned invalid result")
        return result["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise MCPProtocolError(-32000, "tools/call returned non-object")
        return result

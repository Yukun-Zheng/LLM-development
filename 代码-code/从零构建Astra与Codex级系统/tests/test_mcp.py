from pathlib import Path

import pytest

from astra_codex.mcp import InProcessMCPTransport, MCPClient, MCPProtocolError, MCPServer
from astra_codex.tools import FilesystemTool, ToolRegistry


def build_client(root: Path) -> MCPClient:
    registry = ToolRegistry([FilesystemTool(root)])
    server = MCPServer(registry)
    return MCPClient(InProcessMCPTransport(server))


def test_mcp_discover(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    info = client.discover()
    assert info["protocolVersion"] == "2026-07-28"
    assert info["capabilities"]["tools"]["call"] is True


def test_mcp_list_tools(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    tools = client.list_tools()
    assert [tool["name"] for tool in tools] == ["filesystem"]
    assert tools[0]["inputSchema"]["type"] == "object"


def test_mcp_call_tool_round_trip(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    write = client.call_tool(
        "filesystem",
        {"action": "write", "path": "hello.txt", "content": "hello MCP"},
    )
    assert write["isError"] is False
    read = client.call_tool("filesystem", {"action": "read", "path": "hello.txt"})
    assert read["isError"] is False
    assert read["content"][0]["text"] == "hello MCP"


def test_mcp_unknown_rpc_method_becomes_jsonrpc_error(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    with pytest.raises(MCPProtocolError) as exc_info:
        client._call("unknown/method")  # noqa: SLF001 - intentional protocol test
    assert exc_info.value.code == -32601


def test_mcp_tool_failure_is_result_observation(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    result = client.call_tool("filesystem", {"action": "read", "path": "missing.txt"})
    assert result["isError"] is True
    assert "FileNotFoundError" in result["content"][0]["text"]

from __future__ import annotations

import pytest

from astra_codex.agent import ScriptedBackend
from astra_codex.app_server import AgentAppClient, AgentAppServer, AppServerError, InProcessAppTransport
from astra_codex.control_auth import BearerTokenAuthorizer, Principal
from astra_codex.http_app_server import HTTPAppTransport, LocalHTTPAppServer
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.tools import ToolRegistry


def _authorizer() -> BearerTokenAuthorizer:
    authorizer = BearerTokenAuthorizer()
    authorizer.register(
        "admin-secret",
        Principal("admin"),
    )
    authorizer.register(
        "thread-a-reader",
        Principal(
            "reader-a",
            allowed_methods=frozenset({"server/discover", "thread/get", "event/poll"}),
            thread_ids=frozenset({"thr_a"}),
        ),
    )
    authorizer.register(
        "thread-a-worker",
        Principal(
            "worker-a",
            allowed_methods=frozenset({"runtime/runOne"}),
            thread_ids=frozenset({"thr_a"}),
        ),
    )
    return authorizer


def test_authenticated_app_server_rejects_missing_and_invalid_tokens(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state", ScriptedBackend([]), ToolRegistry([])
    ) as runtime:
        server = AgentAppServer(runtime, authorizer=_authorizer())

        missing = AgentAppClient(InProcessAppTransport(server))
        with pytest.raises(AppServerError) as exc_info:
            missing.call("server/discover")
        assert exc_info.value.code == -32001

        invalid = AgentAppClient(
            InProcessAppTransport(server, bearer_token="wrong-secret")
        )
        with pytest.raises(AppServerError) as exc_info:
            invalid.call("server/discover")
        assert exc_info.value.code == -32001


def test_thread_scope_blocks_cross_thread_reads_unscoped_event_feed_and_global_worker(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state", ScriptedBackend([]), ToolRegistry([])
    ) as runtime:
        runtime.create_thread("thr_a")
        runtime.create_thread("thr_b")
        server = AgentAppServer(runtime, authorizer=_authorizer())
        reader = AgentAppClient(
            InProcessAppTransport(server, bearer_token="thread-a-reader")
        )

        assert reader.call("thread/get", {"threadId": "thr_a"})["threadId"] == "thr_a"
        assert reader.call("event/poll", {"threadId": "thr_a"})["events"]

        with pytest.raises(AppServerError) as exc_info:
            reader.call("thread/get", {"threadId": "thr_b"})
        assert exc_info.value.code == -32003

        with pytest.raises(AppServerError) as exc_info:
            reader.call("event/poll")
        assert exc_info.value.code == -32003

        with pytest.raises(AppServerError) as exc_info:
            reader.call("thread/submit", {"threadId": "thr_a", "content": "no write"})
        assert exc_info.value.code == -32003

        scoped_worker = AgentAppClient(
            InProcessAppTransport(server, bearer_token="thread-a-worker")
        )
        with pytest.raises(AppServerError) as exc_info:
            scoped_worker.call("runtime/runOne", {"workerId": "worker"})
        assert exc_info.value.code == -32003


def test_admin_principal_can_use_global_worker_endpoint(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["done"]),
        ToolRegistry([]),
    ) as runtime:
        server = AgentAppServer(runtime, authorizer=_authorizer())
        admin = AgentAppClient(
            InProcessAppTransport(server, bearer_token="admin-secret")
        )
        admin.call("thread/create", {"threadId": "thr_admin"})
        admin.call(
            "thread/submit",
            {"threadId": "thr_admin", "content": "finish", "itemId": "work_admin"},
        )
        result = admin.call("runtime/runOne", {"workerId": "admin-worker"})
        assert result["status"] == "completed"
        assert result["finalAnswer"] == "done"


def test_http_transport_carries_bearer_token_and_enforces_policy(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        runtime.create_thread("thr_a")
        prototype = AgentAppServer(runtime, authorizer=_authorizer())
        with LocalHTTPAppServer(prototype) as http_server:
            anonymous = AgentAppClient(HTTPAppTransport(http_server.rpc_url))
            with pytest.raises(AppServerError) as exc_info:
                anonymous.call("server/discover")
            assert exc_info.value.code == -32001

            reader = AgentAppClient(
                HTTPAppTransport(
                    http_server.rpc_url,
                    bearer_token="thread-a-reader",
                )
            )
            projection = reader.call("thread/get", {"threadId": "thr_a"})
            assert projection["threadId"] == "thr_a"

            with pytest.raises(AppServerError) as exc_info:
                reader.call("thread/get", {"threadId": "thr_b"})
            assert exc_info.value.code == -32003

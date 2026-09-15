from __future__ import annotations

from astra_codex.agent import ScriptedBackend
from astra_codex.app_server import AgentAppClient, AgentAppServer, AppServerError, InProcessAppTransport
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.tools import ToolRegistry


def test_app_server_thread_lifecycle_and_run_one(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["done"]),
        ToolRegistry([]),
    ) as runtime:
        client = AgentAppClient(InProcessAppTransport(AgentAppServer(runtime)))

        discovered = client.call("server/discover")
        assert "thread/create" in discovered["methods"]
        assert "thread/steer" in discovered["methods"]

        created = client.call("thread/create", {"threadId": "thr_app"})
        assert created["threadId"] == "thr_app"
        assert created["status"] == "ready"

        submitted = client.call(
            "thread/submit",
            {"threadId": "thr_app", "content": "finish", "itemId": "work_app"},
        )
        assert submitted["workItemId"] == "work_app"

        result = client.call(
            "runtime/runOne",
            {"workerId": "worker_app", "leaseSeconds": 30.0},
        )
        assert result["status"] == "completed"
        assert result["finalAnswer"] == "done"

        projection = client.call("thread/get", {"threadId": "thr_app"})
        assert projection["status"] == "ready"
        assert projection["submissions"] == ["finish"]


def test_app_server_exposes_durable_steering_and_artifact_metadata(tmp_path) -> None:
    source = tmp_path / "report.txt"
    source.write_text("verified output", encoding="utf-8")

    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        client = AgentAppClient(InProcessAppTransport(AgentAppServer(runtime)))
        client.call("thread/create", {"threadId": "thr_assets"})

        steering = client.call(
            "thread/steer",
            {
                "threadId": "thr_assets",
                "content": "use the latest requirements",
                "steeringId": "steer_app",
            },
        )
        assert steering["steeringId"] == "steer_app"
        assert runtime.steering_queue.pending("thr_assets")[0].content == (
            "use the latest requirements"
        )

        runtime.snapshot_artifact(
            "thr_assets",
            source,
            kind="report",
            artifact_id="art_app",
        )
        listed = client.call("artifact/list", {"threadId": "thr_assets"})
        assert listed["artifacts"][0]["artifactId"] == "art_app"
        assert listed["artifacts"][0]["kind"] == "report"
        assert listed["artifacts"][0]["sizeBytes"] == len(b"verified output")


def test_app_server_fork_preserves_parent_provenance(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        client = AgentAppClient(InProcessAppTransport(AgentAppServer(runtime)))
        client.call("thread/create", {"threadId": "thr_parent"})
        client.call(
            "thread/submit",
            {"threadId": "thr_parent", "content": "parent task", "itemId": "work_parent"},
        )

        child = client.call(
            "thread/fork",
            {"threadId": "thr_parent", "newThreadId": "thr_child"},
        )
        assert child["threadId"] == "thr_child"
        assert child["parentThreadId"] == "thr_parent"
        assert child["parentEventId"] is not None
        assert child["submissions"] == ["parent task"]


def test_app_server_returns_protocol_errors_and_client_raises(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        server = AgentAppServer(runtime)
        invalid = server.handle({"jsonrpc": "2.0", "id": 1, "method": "thread/get", "params": {}})
        assert invalid["error"]["code"] == -32602

        client = AgentAppClient(InProcessAppTransport(server))
        try:
            client.call("missing/method")
        except AppServerError as exc:
            assert exc.code == -32601
        else:
            raise AssertionError("expected AppServerError")

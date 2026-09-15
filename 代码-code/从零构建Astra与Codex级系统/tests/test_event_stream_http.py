from __future__ import annotations

from dataclasses import dataclass

from astra_codex.agent import ScriptedBackend
from astra_codex.app_server import AgentAppClient, AgentAppServer, InProcessAppTransport
from astra_codex.event_stream import DurableEventStream
from astra_codex.http_app_server import HTTPAppTransport, LocalHTTPAppServer
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(True, str(arguments["text"]), {"echoed": True})


def test_durable_event_stream_replays_from_cursor_after_restart(tmp_path) -> None:
    path = tmp_path / "events.sqlite"
    with DurableEventStream(path) as stream:
        first = stream.append("thread.created", {"threadId": "thr"}, thread_id="thr", now=1.0)
        second = stream.append("turn.started", {"turnId": "turn"}, thread_id="thr", turn_id="turn", now=2.0)
        assert stream.high_watermark == second
        assert [event.event_id for event in stream.read(after_id=0)] == [first, second]

    with DurableEventStream(path) as reopened:
        remaining = reopened.read(after_id=first)
        assert len(remaining) == 1
        assert remaining[0].event_id == second
        assert remaining[0].topic == "turn.started"
        assert remaining[0].turn_id == "turn"


def test_runtime_publishes_harness_events_and_app_server_polls_by_cursor(tmp_path) -> None:
    backend = ScriptedBackend(
        ['{"tool":"echo","arguments":{"text":"hello"}}', "done"]
    )
    with DurableAgentRuntime(
        tmp_path / "state",
        backend,
        ToolRegistry([EchoTool()]),
    ) as runtime:
        thread_id = runtime.create_thread("thr_events")
        runtime.submit(thread_id, "use echo then finish", item_id="work_events", now=1.0)
        record = runtime.run_one("worker", now=2.0)
        assert record is not None
        assert record.status == "completed"

        client = AgentAppClient(InProcessAppTransport(AgentAppServer(runtime)))
        first_page = client.call(
            "event/poll",
            {"threadId": thread_id, "afterEventId": 0, "limit": 5},
        )
        second_page = client.call(
            "event/poll",
            {
                "threadId": thread_id,
                "afterEventId": first_page["nextAfterEventId"],
                "limit": 100,
            },
        )

        events = first_page["events"] + second_page["events"]
        event_ids = [item["eventId"] for item in events]
        topics = [item["topic"] for item in events]
        assert event_ids == sorted(event_ids)
        assert len(event_ids) == len(set(event_ids))
        assert first_page["highWatermark"] == second_page["highWatermark"]
        assert {
            "thread.created",
            "thread.submitted",
            "work.claimed",
            "turn.opened",
            "harness.turn_started",
            "harness.model_output",
            "harness.tool_started",
            "harness.tool_completed",
            "harness.turn_completed",
            "work.finished",
        }.issubset(set(topics))

        tool_only = client.call(
            "event/poll",
            {
                "threadId": thread_id,
                "topics": ["harness.tool_started", "harness.tool_completed"],
            },
        )
        assert [item["topic"] for item in tool_only["events"]] == [
            "harness.tool_started",
            "harness.tool_completed",
        ]


def test_real_loopback_http_jsonrpc_controls_runtime_and_replays_events(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["finished over HTTP"]),
        ToolRegistry([]),
    ) as runtime:
        app_server = AgentAppServer(runtime)
        with LocalHTTPAppServer(app_server) as http_server:
            client = AgentAppClient(HTTPAppTransport(http_server.rpc_url))

            created = client.call("thread/create", {"threadId": "thr_http"})
            assert created["threadId"] == "thr_http"

            submitted = client.call(
                "thread/submit",
                {
                    "threadId": "thr_http",
                    "content": "finish the task",
                    "itemId": "work_http",
                },
            )
            assert submitted["workItemId"] == "work_http"

            execution = client.call(
                "runtime/runOne",
                {"workerId": "worker_http", "leaseSeconds": 30.0},
            )
            assert execution["status"] == "completed"
            assert execution["finalAnswer"] == "finished over HTTP"

            events = client.call(
                "event/poll",
                {"threadId": "thr_http", "afterEventId": 0},
            )
            assert events["highWatermark"] >= events["nextAfterEventId"] > 0
            assert events["events"][-1]["topic"] == "work.finished"

            projection = client.call("thread/get", {"threadId": "thr_http"})
            assert projection["status"] == "ready"

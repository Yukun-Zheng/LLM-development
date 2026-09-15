from __future__ import annotations

import threading
import time
from urllib.error import HTTPError

import pytest

from astra_codex.control_auth import BearerTokenAuthorizer, Principal
from astra_codex.event_stream import DurableEventStream
from astra_codex.sse_events import LocalSSEEventServer, SSEEventClient


def test_sse_replays_by_cursor_and_last_event_id_without_duplicates(tmp_path) -> None:
    path = tmp_path / "events.sqlite"
    with DurableEventStream(path) as stream:
        first = stream.append(
            "thread.created",
            {"threadId": "thr_a"},
            thread_id="thr_a",
            now=1.0,
        )
        second = stream.append(
            "harness.model_output",
            {"content": "step-1"},
            thread_id="thr_a",
            turn_id="turn_a",
            now=2.0,
        )
        third = stream.append(
            "work.finished",
            {"status": "completed"},
            thread_id="thr_a",
            turn_id="turn_a",
            now=3.0,
        )

    with LocalSSEEventServer(path, poll_interval_s=0.01) as server:
        client = SSEEventClient(server.base_url)
        first_page = client.read(
            after_event_id=0,
            thread_id="thr_a",
            max_events=2,
            stream_timeout_s=0.5,
        )
        assert [message.event_id for message in first_page] == [first, second]
        assert [message.event for message in first_page] == [
            "thread.created",
            "harness.model_output",
        ]

        resumed = client.read(
            after_event_id=second,
            thread_id="thr_a",
            max_events=10,
            stream_timeout_s=0.2,
            use_last_event_id_header=True,
        )
        assert [message.event_id for message in resumed] == [third]
        assert resumed[0].data["payload"]["status"] == "completed"


def test_sse_thread_and_topic_filters_apply_before_delivery(tmp_path) -> None:
    path = tmp_path / "events.sqlite"
    with DurableEventStream(path) as stream:
        stream.append("harness.model_output", {"n": 1}, thread_id="thr_a")
        tool_event = stream.append(
            "harness.tool_completed",
            {"ok": True},
            thread_id="thr_a",
            turn_id="turn_a",
        )
        stream.append("harness.tool_completed", {"ok": True}, thread_id="thr_b")

    with LocalSSEEventServer(path, poll_interval_s=0.01) as server:
        messages = SSEEventClient(server.base_url).read(
            thread_id="thr_a",
            topics=("harness.tool_completed",),
            max_events=10,
            stream_timeout_s=0.15,
        )

    assert [message.event_id for message in messages] == [tool_event]
    assert messages[0].data["threadId"] == "thr_a"


def test_sse_live_delivery_waits_for_new_durable_event(tmp_path) -> None:
    path = tmp_path / "events.sqlite"
    # Create the schema before starting the streaming reader.
    with DurableEventStream(path):
        pass

    received: list[object] = []
    error: list[BaseException] = []

    with LocalSSEEventServer(path, poll_interval_s=0.01) as server:
        client = SSEEventClient(server.base_url, timeout_s=2.0)

        def reader() -> None:
            try:
                received.extend(
                    client.read(
                        thread_id="thr_live",
                        max_events=1,
                        stream_timeout_s=1.0,
                    )
                )
            except BaseException as exc:  # surface background assertion failures
                error.append(exc)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.08)
        with DurableEventStream(path) as stream:
            event_id = stream.append(
                "harness.tool_started",
                {"tool": "search"},
                thread_id="thr_live",
                turn_id="turn_live",
            )
        thread.join(timeout=2.0)

    assert not error
    assert not thread.is_alive()
    assert len(received) == 1
    message = received[0]
    assert message.event_id == event_id  # type: ignore[attr-defined]
    assert message.event == "harness.tool_started"  # type: ignore[attr-defined]
    assert message.data["turnId"] == "turn_live"  # type: ignore[attr-defined]


def test_sse_reuses_bearer_auth_and_thread_scope(tmp_path) -> None:
    path = tmp_path / "events.sqlite"
    with DurableEventStream(path) as stream:
        stream.append("thread.created", {}, thread_id="thr_a")
        stream.append("thread.created", {}, thread_id="thr_b")

    authorizer = BearerTokenAuthorizer()
    authorizer.register(
        "reader-a",
        Principal(
            "reader-a",
            allowed_methods=frozenset({"event/poll"}),
            thread_ids=frozenset({"thr_a"}),
        ),
    )

    with LocalSSEEventServer(
        path,
        authorizer=authorizer,
        poll_interval_s=0.01,
    ) as server:
        anonymous = SSEEventClient(server.base_url)
        with pytest.raises(HTTPError) as exc_info:
            anonymous.read(
                thread_id="thr_a",
                max_events=1,
                stream_timeout_s=0.1,
            )
        assert exc_info.value.code == 401

        reader = SSEEventClient(server.base_url, bearer_token="reader-a")
        allowed = reader.read(
            thread_id="thr_a",
            max_events=1,
            stream_timeout_s=0.1,
        )
        assert len(allowed) == 1
        assert allowed[0].data["threadId"] == "thr_a"

        with pytest.raises(HTTPError) as exc_info:
            reader.read(
                thread_id="thr_b",
                max_events=1,
                stream_timeout_s=0.1,
            )
        assert exc_info.value.code == 403

        with pytest.raises(HTTPError) as exc_info:
            reader.read(max_events=1, stream_timeout_s=0.1)
        assert exc_info.value.code == 403

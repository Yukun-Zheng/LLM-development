from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from .control_auth import AuthenticationError, AuthorizationError, BearerTokenAuthorizer
from .event_stream import DurableEventStream, RuntimeEvent


def _bearer_token(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _event_payload(event: RuntimeEvent) -> dict[str, Any]:
    return {
        "eventId": event.event_id,
        "topic": event.topic,
        "payload": event.payload,
        "threadId": event.thread_id,
        "turnId": event.turn_id,
        "createdAt": event.created_at,
    }


@dataclass(frozen=True, slots=True)
class SSEMessage:
    event_id: int
    event: str
    data: dict[str, Any]


class LocalSSEEventServer:
    """Loopback SSE projection of the durable runtime event stream.

    Each HTTP handler opens its own ``DurableEventStream`` SQLite handle, so the
    threaded server never shares a SQLite connection across threads. SSE is only
    a delivery mechanism: durable ordering/replay remains owned by the event
    store. A reconnecting client can resume from ``Last-Event-ID`` or an
    explicit ``afterEventId`` query parameter.

    ``maxEvents`` and ``timeoutSeconds`` are included to make the teaching
    transport deterministic in tests and scripts. Production streaming would
    normally keep the connection open and add stronger backpressure/retention
    policies.
    """

    def __init__(
        self,
        event_db_path: str | Path,
        *,
        authorizer: BearerTokenAuthorizer | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        poll_interval_s: float = 0.05,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.event_db_path = str(Path(event_db_path).resolve())
        self.authorizer = authorizer
        self.poll_interval_s = poll_interval_s
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                parsed = urlparse(self.path)
                if parsed.path != "/events":
                    self.send_error(404)
                    return
                try:
                    options = outer._parse_options(
                        parse_qs(parsed.query), self.headers.get("Last-Event-ID")
                    )
                    if outer.authorizer is not None:
                        principal = outer.authorizer.authenticate(
                            _bearer_token(self.headers.get("Authorization"))
                        )
                        outer.authorizer.authorize(
                            principal,
                            "event/poll",
                            {"threadId": options["thread_id"]}
                            if options["thread_id"] is not None
                            else {},
                        )
                except AuthenticationError as exc:
                    self._write_error(401, str(exc))
                    return
                except AuthorizationError as exc:
                    self._write_error(403, str(exc))
                    return
                except ValueError as exc:
                    self._write_error(400, str(exc))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                cursor = int(options["after_id"])
                sent = 0
                deadline = time.monotonic() + float(options["timeout_s"])
                topics = options["topics"]
                max_events = int(options["max_events"])

                with DurableEventStream(outer.event_db_path) as stream:
                    while sent < max_events and time.monotonic() < deadline:
                        rows = stream.read(
                            after_id=cursor,
                            limit=max_events - sent,
                            thread_id=options["thread_id"],
                            topics=topics,
                        )
                        if not rows:
                            time.sleep(outer.poll_interval_s)
                            continue
                        for event in rows:
                            self._write_event(event)
                            cursor = event.event_id
                            sent += 1
                            if sent >= max_events:
                                break

                # A comment is SSE-safe and gives clients a clean EOF marker in
                # this finite reference transport without inventing a runtime event.
                self.wfile.write(b": stream-end\n\n")
                self.wfile.flush()

            def _write_event(self, event: RuntimeEvent) -> None:
                data = json.dumps(
                    _event_payload(event), ensure_ascii=False, separators=(",", ":")
                )
                wire = (
                    f"id: {event.event_id}\n"
                    f"event: {event.topic}\n"
                    f"data: {data}\n\n"
                ).encode("utf-8")
                self.wfile.write(wire)
                self.wfile.flush()

            def _write_error(self, status: int, message: str) -> None:
                body = message.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread: threading.Thread | None = None

    @staticmethod
    def _one(query: dict[str, list[str]], key: str, default: str) -> str:
        values = query.get(key)
        return default if not values else values[-1]

    def _parse_options(
        self,
        query: dict[str, list[str]],
        last_event_id: str | None,
    ) -> dict[str, Any]:
        explicit_after = self._one(query, "afterEventId", "")
        cursor_text = explicit_after or last_event_id or "0"
        after_id = int(cursor_text)
        max_events = int(self._one(query, "maxEvents", "100"))
        timeout_s = float(self._one(query, "timeoutSeconds", "30"))
        thread_id = self._one(query, "threadId", "") or None
        topic_values = query.get("topic", [])

        if after_id < 0:
            raise ValueError("afterEventId cannot be negative")
        if max_events <= 0 or max_events > 1000:
            raise ValueError("maxEvents must be between 1 and 1000")
        if timeout_s <= 0 or timeout_s > 300:
            raise ValueError("timeoutSeconds must be in (0, 300]")
        if any(not topic for topic in topic_values):
            raise ValueError("topic values cannot be empty")
        return {
            "after_id": after_id,
            "max_events": max_events,
            "timeout_s": timeout_s,
            "thread_id": thread_id,
            "topics": set(topic_values) if topic_values else None,
        }

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/events"

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("SSE event server is already running")
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="astra-codex-sse-events",
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None

    def __enter__(self) -> "LocalSSEEventServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SSEEventClient:
    """Finite synchronous SSE reader used by tests and teaching examples."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url
        self.bearer_token = bearer_token
        self.timeout_s = timeout_s

    def read(
        self,
        *,
        after_event_id: int = 0,
        thread_id: str | None = None,
        topics: tuple[str, ...] = (),
        max_events: int = 100,
        stream_timeout_s: float = 1.0,
        use_last_event_id_header: bool = False,
    ) -> tuple[SSEMessage, ...]:
        query: list[tuple[str, str]] = [
            ("maxEvents", str(max_events)),
            ("timeoutSeconds", str(stream_timeout_s)),
        ]
        if not use_last_event_id_header:
            query.append(("afterEventId", str(after_event_id)))
        if thread_id is not None:
            query.append(("threadId", thread_id))
        query.extend(("topic", topic) for topic in topics)

        headers: dict[str, str] = {"Accept": "text/event-stream"}
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if use_last_event_id_header:
            headers["Last-Event-ID"] = str(after_event_id)

        request = Request(
            self.base_url + "?" + urlencode(query),
            headers=headers,
            method="GET",
        )
        messages: list[SSEMessage] = []
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - explicit loopback reference transport
            current_id: int | None = None
            current_event = "message"
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if current_id is not None and data_lines:
                        payload = json.loads("\n".join(data_lines))
                        messages.append(
                            SSEMessage(current_id, current_event, payload)
                        )
                    current_id = None
                    current_event = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, separator, value = line.partition(":")
                if not separator:
                    continue
                value = value.lstrip(" ")
                if field == "id":
                    current_id = int(value)
                elif field == "event":
                    current_event = value
                elif field == "data":
                    data_lines.append(value)
        return tuple(messages)

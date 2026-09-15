"""Durable A2A Task-update journal and reference SubscribeToTask SSE binding.

Pinned normative source:
https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

A2A v1 defines ``GET /tasks/{id}:subscribe`` as a streaming Task subscription.
This reference layer persists Task snapshots independently of a live subscriber,
then exposes them as ``StreamResponse.task`` values over SSE.

The optional SSE ``Last-Event-ID`` replay cursor used here is a transport-level
teaching extension. It is not claimed to be a normative A2A request field.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .a2a import (
    TERMINAL_A2A_STATES,
    A2AAgentCard,
    A2AHandler,
    A2ASendMessageRequest,
    A2AService,
    A2ATask,
    A2ATaskStore,
)
from .a2a_http import A2AHTTPError
from .a2a_streaming import A2AStreamResponse


@dataclass(frozen=True, slots=True)
class A2ATaskUpdate:
    update_id: int
    task_id: str
    task: A2ATask
    created_at: float


class A2ATaskUpdateJournal:
    """Append-only durable Task snapshot feed for subscriptions/replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS a2a_task_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_a2a_task_updates_task "
            "ON a2a_task_updates(task_id, update_id)"
        )

    @staticmethod
    def _encode(task: A2ATask) -> str:
        return json.dumps(
            task.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def append(self, task: A2ATask, *, now: float | None = None) -> int:
        encoded = self._encode(task)
        latest = self.connection.execute(
            """
            SELECT update_id, task_json
            FROM a2a_task_updates
            WHERE task_id = ?
            ORDER BY update_id DESC LIMIT 1
            """,
            (task.id,),
        ).fetchone()
        if latest is not None and str(latest[1]) == encoded:
            return int(latest[0])
        cursor = self.connection.execute(
            """
            INSERT INTO a2a_task_updates(task_id, task_json, created_at)
            VALUES (?, ?, ?)
            """,
            (task.id, encoded, time.time() if now is None else now),
        )
        return int(cursor.lastrowid)

    def read(
        self,
        task_id: str,
        *,
        after_id: int = 0,
        limit: int = 100,
    ) -> tuple[A2ATaskUpdate, ...]:
        if after_id < 0:
            raise ValueError("after_id cannot be negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self.connection.execute(
            """
            SELECT update_id, task_id, task_json, created_at
            FROM a2a_task_updates
            WHERE task_id = ? AND update_id > ?
            ORDER BY update_id ASC
            LIMIT ?
            """,
            (task_id, after_id, limit),
        ).fetchall()
        return tuple(
            A2ATaskUpdate(
                update_id=int(row[0]),
                task_id=str(row[1]),
                task=A2ATask.from_dict(json.loads(str(row[2]))),
                created_at=float(row[3]),
            )
            for row in rows
        )

    def latest_id(self, task_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(update_id), 0) FROM a2a_task_updates WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "A2ATaskUpdateJournal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JournaledA2AService(A2AService):
    """A2AService that mirrors externally visible Task snapshots to a journal."""

    def __init__(
        self,
        card: A2AAgentCard,
        store: A2ATaskStore,
        journal: A2ATaskUpdateJournal,
        *,
        handler: A2AHandler | None = None,
    ) -> None:
        super().__init__(card, store, handler=handler)
        self.journal = journal

    def send_message(self, request: A2ASendMessageRequest) -> A2ATask:
        task = super().send_message(request)
        # Non-immediate SendMessage dispatches through self.process_task(), whose
        # override below already records pre/final state. append() de-duplicates
        # the returned final snapshot, so the explicit append remains safe.
        self.journal.append(task)
        return task

    def process_task(self, task_id: str) -> A2ATask:
        before = self.store.get(task_id)
        self.journal.append(before)
        after = super().process_task(task_id)
        self.journal.append(after)
        return after

    def cancel_task(self, task_id: str) -> A2ATask:
        task = super().cancel_task(task_id)
        self.journal.append(task)
        return task


def _encode_task_sse(update: A2ATaskUpdate) -> bytes:
    payload = json.dumps(
        A2AStreamResponse(task=update.task).to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {update.update_id}\ndata: {payload}\n\n".encode("utf-8")


class LocalA2ASubscriptionHTTPServer:
    """Reference ``SubscribeToTask`` server backed by a durable update journal."""

    def __init__(
        self,
        task_store_path: str | Path,
        update_journal_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        poll_interval_s: float = 0.02,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.task_store_path = Path(task_store_path)
        self.update_journal_path = Path(update_journal_path)
        self.poll_interval_s = poll_interval_s
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlparse(self.path)
                path = parsed.path
                prefix = "/tasks/"
                suffix = ":subscribe"
                if not path.startswith(prefix) or not path.endswith(suffix):
                    self._write_json(404, {"error": {"message": "not found"}})
                    return
                task_id = path[len(prefix) : -len(suffix)]
                if not task_id:
                    self._write_json(400, {"error": {"message": "task id is required"}})
                    return
                try:
                    after_id = int(self.headers.get("Last-Event-ID", "0"))
                    if after_id < 0:
                        raise ValueError
                except ValueError:
                    self._write_json(
                        400,
                        {"error": {"message": "Last-Event-ID must be non-negative integer"}},
                    )
                    return

                with A2ATaskStore(outer.task_store_path) as store:
                    try:
                        current = store.get(task_id)
                    except KeyError as exc:
                        self._write_json(404, {"error": {"message": str(exc)}})
                        return
                if after_id == 0 and current.status.state in TERMINAL_A2A_STATES:
                    # The v1 proto explicitly documents SubscribeToTask on an
                    # already-terminal Task as UnsupportedOperationError.
                    self._write_json(
                        409,
                        {"error": {"message": "cannot subscribe to terminal A2A task"}},
                    )
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                cursor = after_id
                with A2ATaskUpdateJournal(outer.update_journal_path) as journal:
                    while True:
                        updates = journal.read(task_id, after_id=cursor)
                        if updates:
                            for update in updates:
                                self.wfile.write(_encode_task_sse(update))
                                self.wfile.flush()
                                cursor = update.update_id
                                if update.task.status.state in TERMINAL_A2A_STATES:
                                    return
                        time.sleep(outer.poll_interval_s)

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = HTTPServer((host, port), Handler)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A2A subscription server is already running")
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="astra-codex-a2a-subscribe",
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=5.0)
        self.server.server_close()
        self.thread = None

    def __enter__(self) -> "LocalA2ASubscriptionHTTPServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class A2ASubscriptionHTTPClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def subscribe(
        self,
        task_id: str,
        *,
        last_event_id: int = 0,
    ) -> Iterator[tuple[int, A2AStreamResponse]]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id:
            headers["Last-Event-ID"] = str(last_event_id)
        request = Request(
            self.base_url + f"/tasks/{quote(task_id, safe='')}:subscribe",
            headers=headers,
            method="GET",
        )
        try:
            response = urlopen(request, timeout=self.timeout_s)  # noqa: S310 - explicit loopback/reference client
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                message = str(payload.get("error", {}).get("message", exc.reason))
            except Exception:
                message = str(exc.reason)
            raise A2AHTTPError(exc.code, message) from exc

        def iterator() -> Iterator[tuple[int, A2AStreamResponse]]:
            current_id: int | None = None
            data_lines: list[str] = []
            try:
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line == "":
                        if current_id is not None and data_lines:
                            payload = json.loads("\n".join(data_lines))
                            if not isinstance(payload, dict):
                                raise A2AHTTPError(502, "subscription SSE data must be object")
                            yield current_id, A2AStreamResponse.from_dict(payload)
                        current_id = None
                        data_lines.clear()
                    elif line.startswith("id:"):
                        current_id = int(line[len("id:") :].strip())
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
            finally:
                response.close()

        return iterator()

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ThreadStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DurableEventType(str, Enum):
    THREAD_CREATED = "thread_created"
    USER_SUBMISSION = "user_submission"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    CHECKPOINT_CREATED = "checkpoint_created"
    THREAD_PAUSED = "thread_paused"
    THREAD_RESUMED = "thread_resumed"
    THREAD_COMPLETED = "thread_completed"
    THREAD_FAILED = "thread_failed"


@dataclass(frozen=True, slots=True)
class DurableEvent:
    id: int
    thread_id: str
    turn_id: str | None
    event_type: DurableEventType
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class ThreadProjection:
    """State reconstructed only from the append-only event log."""

    thread_id: str
    status: ThreadStatus
    active_turn_id: str | None
    last_event_id: int
    submissions: tuple[str, ...]
    last_checkpoint: dict[str, Any] | None
    failure_reason: str | None


class DurableThreadStore:
    """SQLite event store for long-running agent threads.

    The event log is the source of truth. ``project`` rebuilds current state by
    replaying events, so a new Python process can reopen the same database and
    resume a task without relying on in-memory objects.

    This module is intentionally a small reference implementation. It is not a
    distributed queue, lease manager, or production transaction coordinator.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                turn_id TEXT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(thread_id)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_events_thread_id "
            "ON thread_events(thread_id, id)"
        )
        self.connection.commit()

    def create_thread(self, thread_id: str | None = None) -> str:
        thread_id = thread_id or f"thr_{uuid.uuid4().hex}"
        created_at = _utc_now()
        try:
            self.connection.execute(
                "INSERT INTO threads(thread_id, created_at) VALUES (?, ?)",
                (thread_id, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"thread already exists: {thread_id}") from exc
        self.connection.commit()
        self.append_event(thread_id, DurableEventType.THREAD_CREATED)
        return thread_id

    def _thread_exists(self, thread_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return row is not None

    def append_event(
        self,
        thread_id: str,
        event_type: DurableEventType,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> int:
        if not self._thread_exists(thread_id):
            raise KeyError(f"unknown thread: {thread_id}")
        cursor = self.connection.execute(
            """
            INSERT INTO thread_events(thread_id, turn_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                turn_id,
                event_type.value,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                _utc_now(),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def events(self, thread_id: str) -> list[DurableEvent]:
        if not self._thread_exists(thread_id):
            raise KeyError(f"unknown thread: {thread_id}")
        rows = self.connection.execute(
            """
            SELECT id, thread_id, turn_id, event_type, payload_json, created_at
            FROM thread_events
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()
        return [
            DurableEvent(
                id=int(row[0]),
                thread_id=str(row[1]),
                turn_id=row[2],
                event_type=DurableEventType(row[3]),
                payload=json.loads(row[4]),
                created_at=str(row[5]),
            )
            for row in rows
        ]

    def project(self, thread_id: str) -> ThreadProjection:
        events = self.events(thread_id)
        if not events:
            raise RuntimeError(f"thread has no creation event: {thread_id}")

        status = ThreadStatus.READY
        active_turn_id: str | None = None
        submissions: list[str] = []
        last_checkpoint: dict[str, Any] | None = None
        failure_reason: str | None = None

        for event in events:
            if event.event_type is DurableEventType.USER_SUBMISSION:
                submissions.append(str(event.payload.get("content", "")))
            elif event.event_type is DurableEventType.TURN_STARTED:
                status = ThreadStatus.RUNNING
                active_turn_id = event.turn_id
            elif event.event_type is DurableEventType.TURN_COMPLETED:
                status = ThreadStatus.READY
                active_turn_id = None
            elif event.event_type is DurableEventType.CHECKPOINT_CREATED:
                last_checkpoint = dict(event.payload)
            elif event.event_type is DurableEventType.THREAD_PAUSED:
                status = ThreadStatus.PAUSED
            elif event.event_type is DurableEventType.THREAD_RESUMED:
                status = ThreadStatus.READY
            elif event.event_type is DurableEventType.THREAD_COMPLETED:
                status = ThreadStatus.COMPLETED
                active_turn_id = None
            elif event.event_type is DurableEventType.THREAD_FAILED:
                status = ThreadStatus.FAILED
                active_turn_id = None
                failure_reason = str(event.payload.get("reason", ""))

        return ThreadProjection(
            thread_id=thread_id,
            status=status,
            active_turn_id=active_turn_id,
            last_event_id=events[-1].id,
            submissions=tuple(submissions),
            last_checkpoint=last_checkpoint,
            failure_reason=failure_reason,
        )

    def submit(self, thread_id: str, content: str) -> int:
        projection = self.project(thread_id)
        if projection.status in {ThreadStatus.COMPLETED, ThreadStatus.FAILED}:
            raise RuntimeError(f"cannot submit to terminal thread: {projection.status.value}")
        return self.append_event(
            thread_id,
            DurableEventType.USER_SUBMISSION,
            {"content": content},
        )

    def start_turn(self, thread_id: str, turn_id: str | None = None) -> str:
        projection = self.project(thread_id)
        if projection.status is not ThreadStatus.READY:
            raise RuntimeError(f"thread is not ready: {projection.status.value}")
        turn_id = turn_id or f"turn_{uuid.uuid4().hex}"
        self.append_event(
            thread_id,
            DurableEventType.TURN_STARTED,
            {"previous_event_id": projection.last_event_id},
            turn_id=turn_id,
        )
        return turn_id

    def complete_turn(
        self,
        thread_id: str,
        turn_id: str,
        *,
        summary: str | None = None,
    ) -> int:
        projection = self.project(thread_id)
        if projection.status is not ThreadStatus.RUNNING:
            raise RuntimeError("cannot complete a turn when the thread is not running")
        if projection.active_turn_id != turn_id:
            raise ValueError(
                f"active turn is {projection.active_turn_id!r}, not {turn_id!r}"
            )
        return self.append_event(
            thread_id,
            DurableEventType.TURN_COMPLETED,
            {"summary": summary} if summary is not None else {},
            turn_id=turn_id,
        )

    def checkpoint(self, thread_id: str, state: dict[str, Any]) -> int:
        projection = self.project(thread_id)
        if projection.status in {ThreadStatus.COMPLETED, ThreadStatus.FAILED}:
            raise RuntimeError("cannot checkpoint a terminal thread")
        return self.append_event(
            thread_id,
            DurableEventType.CHECKPOINT_CREATED,
            {"state": state, "after_event_id": projection.last_event_id},
            turn_id=projection.active_turn_id,
        )

    def pause(self, thread_id: str, reason: str = "") -> int:
        projection = self.project(thread_id)
        if projection.status not in {ThreadStatus.READY, ThreadStatus.RUNNING}:
            raise RuntimeError(f"cannot pause thread in state {projection.status.value}")
        return self.append_event(
            thread_id,
            DurableEventType.THREAD_PAUSED,
            {"reason": reason},
            turn_id=projection.active_turn_id,
        )

    def resume(self, thread_id: str) -> int:
        projection = self.project(thread_id)
        if projection.status is not ThreadStatus.PAUSED:
            raise RuntimeError(f"thread is not paused: {projection.status.value}")
        return self.append_event(thread_id, DurableEventType.THREAD_RESUMED)

    def complete_thread(self, thread_id: str) -> int:
        projection = self.project(thread_id)
        if projection.status is ThreadStatus.RUNNING:
            raise RuntimeError("complete the active turn before completing the thread")
        if projection.status in {ThreadStatus.COMPLETED, ThreadStatus.FAILED}:
            raise RuntimeError(f"thread already terminal: {projection.status.value}")
        return self.append_event(thread_id, DurableEventType.THREAD_COMPLETED)

    def fail_thread(self, thread_id: str, reason: str) -> int:
        projection = self.project(thread_id)
        if projection.status in {ThreadStatus.COMPLETED, ThreadStatus.FAILED}:
            raise RuntimeError(f"thread already terminal: {projection.status.value}")
        return self.append_event(
            thread_id,
            DurableEventType.THREAD_FAILED,
            {"reason": reason},
            turn_id=projection.active_turn_id,
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DurableThreadStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

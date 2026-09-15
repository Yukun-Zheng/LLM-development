from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_id: int
    topic: str
    payload: dict[str, Any]
    thread_id: str | None
    turn_id: str | None
    created_at: float


class DurableEventStream:
    """Append-only runtime event stream with cursor-based replay.

    This store is deliberately separate from ``DurableThreadStore``. The thread
    event log is the source of truth for reconstructing thread state; this stream
    is an integration/control-plane feed for UIs, IDEs and external clients.

    Consumers keep the last ``event_id`` they observed and resume with
    ``after_id``. That makes reconnect/replay deterministic without requiring a
    live websocket connection.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                thread_id TEXT,
                turn_id TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runtime_events_thread "
            "ON runtime_events(thread_id, event_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runtime_events_topic "
            "ON runtime_events(topic, event_id)"
        )

    def append(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        now: float | None = None,
    ) -> int:
        if not topic:
            raise ValueError("event topic cannot be empty")
        timestamp = time.time() if now is None else now
        cursor = self.connection.execute(
            """
            INSERT INTO runtime_events(topic, payload_json, thread_id, turn_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                topic,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                thread_id,
                turn_id,
                timestamp,
            ),
        )
        return int(cursor.lastrowid)

    def read(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        thread_id: str | None = None,
        topics: Iterable[str] | None = None,
    ) -> tuple[RuntimeEvent, ...]:
        if after_id < 0:
            raise ValueError("after_id cannot be negative")
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")

        query = """
            SELECT event_id, topic, payload_json, thread_id, turn_id, created_at
            FROM runtime_events
            WHERE event_id > ?
        """
        params: list[object] = [after_id]
        if thread_id is not None:
            query += " AND thread_id = ?"
            params.append(thread_id)
        topic_values = sorted(set(topics or ()))
        if topic_values:
            placeholders = ",".join("?" for _ in topic_values)
            query += f" AND topic IN ({placeholders})"
            params.extend(topic_values)
        query += " ORDER BY event_id ASC LIMIT ?"
        params.append(limit)

        rows = self.connection.execute(query, params).fetchall()
        return tuple(
            RuntimeEvent(
                event_id=int(row[0]),
                topic=str(row[1]),
                payload=json.loads(str(row[2])),
                thread_id=None if row[3] is None else str(row[3]),
                turn_id=None if row[4] is None else str(row[4]),
                created_at=float(row[5]),
            )
            for row in rows
        )

    @property
    def high_watermark(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(event_id), 0) FROM runtime_events"
        ).fetchone()
        assert row is not None
        return int(row[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DurableEventStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

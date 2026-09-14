from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    role: str
    content: str
    metadata: dict[str, Any]


class EventMemory:
    """SQLite-backed append-only event memory for long-running agent tasks.

    This is not 'human memory'.  It is an engineering substrate: persistent
    observations, decisions, notes, and searchable task history that can survive
    process restarts and context compaction.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.connection.commit()

    def append(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO events(role, content, metadata_json) VALUES (?, ?, ?)",
            (role, content, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent(self, limit: int = 50) -> list[Event]:
        rows = self.connection.execute(
            "SELECT id, role, content, metadata_json FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Event(row[0], row[1], row[2], json.loads(row[3]))
            for row in reversed(rows)
        ]

    def search(self, query: str, limit: int = 50) -> list[Event]:
        rows = self.connection.execute(
            """
            SELECT id, role, content, metadata_json
            FROM events
            WHERE content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
        return [Event(row[0], row[1], row[2], json.loads(row[3])) for row in rows]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EventMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

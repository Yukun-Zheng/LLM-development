from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SteeringStatus(str, Enum):
    PENDING = "pending"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SteeringMessage:
    steering_id: str
    thread_id: str
    content: str
    status: SteeringStatus
    created_at: float
    consumed_at: float | None


class DurableSteeringQueue:
    """Persistent user-steering inbox for long-running agent threads.

    Steering is intentionally separate from the model context. A message is
    first persisted, then atomically claimed as consumed by the turn executor,
    and only then injected into the next model step. This provides a durable
    substrate for "change direction while the agent is working" semantics.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS steering_messages (
                steering_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                consumed_at REAL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_steering_pending "
            "ON steering_messages(thread_id, status, created_at, steering_id)"
        )

    @staticmethod
    def _row_to_message(row: tuple[object, ...]) -> SteeringMessage:
        return SteeringMessage(
            steering_id=str(row[0]),
            thread_id=str(row[1]),
            content=str(row[2]),
            status=SteeringStatus(str(row[3])),
            created_at=float(row[4]),
            consumed_at=None if row[5] is None else float(row[5]),
        )

    def submit(
        self,
        thread_id: str,
        content: str,
        *,
        steering_id: str | None = None,
        now: float | None = None,
    ) -> str:
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not content:
            raise ValueError("steering content cannot be empty")
        timestamp = time.time() if now is None else now
        steering_id = steering_id or f"steer_{uuid.uuid4().hex}"
        self.connection.execute(
            """
            INSERT INTO steering_messages(
                steering_id, thread_id, content, status, created_at, consumed_at
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                steering_id,
                thread_id,
                content,
                SteeringStatus.PENDING.value,
                timestamp,
            ),
        )
        return steering_id

    def get(self, steering_id: str) -> SteeringMessage:
        row = self.connection.execute(
            """
            SELECT steering_id, thread_id, content, status, created_at, consumed_at
            FROM steering_messages WHERE steering_id = ?
            """,
            (steering_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown steering message: {steering_id}")
        return self._row_to_message(row)

    def pending(self, thread_id: str) -> tuple[SteeringMessage, ...]:
        rows = self.connection.execute(
            """
            SELECT steering_id, thread_id, content, status, created_at, consumed_at
            FROM steering_messages
            WHERE thread_id = ? AND status = ?
            ORDER BY created_at ASC, steering_id ASC
            """,
            (thread_id, SteeringStatus.PENDING.value),
        ).fetchall()
        return tuple(self._row_to_message(row) for row in rows)

    def consume_pending(
        self,
        thread_id: str,
        *,
        now: float | None = None,
    ) -> tuple[SteeringMessage, ...]:
        timestamp = time.time() if now is None else now
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self.connection.execute(
                """
                SELECT steering_id, thread_id, content, status, created_at, consumed_at
                FROM steering_messages
                WHERE thread_id = ? AND status = ?
                ORDER BY created_at ASC, steering_id ASC
                """,
                (thread_id, SteeringStatus.PENDING.value),
            ).fetchall()
            ids = [str(row[0]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self.connection.execute(
                    f"""
                    UPDATE steering_messages
                    SET status = ?, consumed_at = ?
                    WHERE steering_id IN ({placeholders}) AND status = ?
                    """,
                    [
                        SteeringStatus.CONSUMED.value,
                        timestamp,
                        *ids,
                        SteeringStatus.PENDING.value,
                    ],
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

        return tuple(
            SteeringMessage(
                steering_id=str(row[0]),
                thread_id=str(row[1]),
                content=str(row[2]),
                status=SteeringStatus.CONSUMED,
                created_at=float(row[4]),
                consumed_at=timestamp,
            )
            for row in rows
        )

    def cancel(self, steering_id: str) -> None:
        current = self.get(steering_id)
        if current.status is not SteeringStatus.PENDING:
            raise RuntimeError(f"steering message is {current.status.value}")
        self.connection.execute(
            "UPDATE steering_messages SET status = ? WHERE steering_id = ?",
            (SteeringStatus.CANCELLED.value, steering_id),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DurableSteeringQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

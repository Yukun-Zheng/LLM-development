from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class WorkStatus(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WorkItem:
    item_id: str
    thread_id: str
    kind: str
    payload: dict[str, Any]
    status: WorkStatus
    lease_owner: str | None
    lease_until: float | None
    result: dict[str, Any] | None


class DurableWorkQueue:
    """SQLite-backed work queue with lease, retry and cancellation semantics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work_items (
                item_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_until REAL,
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_work_items_claim "
            "ON work_items(status, lease_until, created_at)"
        )

    def _row_to_item(self, row: tuple[object, ...]) -> WorkItem:
        return WorkItem(
            item_id=str(row[0]),
            thread_id=str(row[1]),
            kind=str(row[2]),
            payload=json.loads(str(row[3])),
            status=WorkStatus(str(row[4])),
            lease_owner=None if row[5] is None else str(row[5]),
            lease_until=None if row[6] is None else float(row[6]),
            result=None if row[7] is None else json.loads(str(row[7])),
        )

    def enqueue(
        self,
        thread_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        item_id: str | None = None,
        now: float | None = None,
    ) -> str:
        now = time.time() if now is None else now
        item_id = item_id or f"work_{uuid.uuid4().hex}"
        self.connection.execute(
            """
            INSERT INTO work_items(
                item_id, thread_id, kind, payload_json, status,
                lease_owner, lease_until, result_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (
                item_id,
                thread_id,
                kind,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                WorkStatus.PENDING.value,
                now,
                now,
            ),
        )
        return item_id

    def get(self, item_id: str) -> WorkItem:
        row = self.connection.execute(
            """
            SELECT item_id, thread_id, kind, payload_json, status,
                   lease_owner, lease_until, result_json
            FROM work_items WHERE item_id = ?
            """,
            (item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown work item: {item_id}")
        return self._row_to_item(row)

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: float | None = None,
        kinds: set[str] | None = None,
    ) -> WorkItem | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time.time() if now is None else now

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            query = """
                SELECT item_id
                FROM work_items
                WHERE (
                    status = ?
                    OR (status = ? AND lease_until <= ?)
                )
            """
            params: list[object] = [
                WorkStatus.PENDING.value,
                WorkStatus.LEASED.value,
                now,
            ]
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(sorted(kinds))
            query += " ORDER BY created_at ASC, item_id ASC LIMIT 1"

            row = self.connection.execute(query, params).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None

            item_id = str(row[0])
            self.connection.execute(
                """
                UPDATE work_items
                SET status = ?, lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (
                    WorkStatus.LEASED.value,
                    worker_id,
                    now + lease_seconds,
                    now,
                    item_id,
                ),
            )
            self.connection.execute("COMMIT")
            return self.get(item_id)
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _finish(
        self,
        item_id: str,
        worker_id: str,
        status: WorkStatus,
        result: dict[str, Any] | None,
        *,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        current = self.get(item_id)
        if current.status is not WorkStatus.LEASED:
            raise RuntimeError(f"work item is not leased: {current.status.value}")
        if current.lease_owner != worker_id:
            raise PermissionError(
                f"lease belongs to {current.lease_owner!r}, not {worker_id!r}"
            )
        self.connection.execute(
            """
            UPDATE work_items
            SET status = ?, result_json = ?, lease_owner = NULL,
                lease_until = NULL, updated_at = ?
            WHERE item_id = ?
            """,
            (
                status.value,
                None
                if result is None
                else json.dumps(result, ensure_ascii=False, sort_keys=True),
                now,
                item_id,
            ),
        )

    def ack(
        self,
        item_id: str,
        worker_id: str,
        result: dict[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        self._finish(item_id, worker_id, WorkStatus.COMPLETED, result, now=now)

    def fail(
        self,
        item_id: str,
        worker_id: str,
        result: dict[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        self._finish(item_id, worker_id, WorkStatus.FAILED, result, now=now)

    def release(
        self,
        item_id: str,
        worker_id: str,
        *,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        current = self.get(item_id)
        if current.status is not WorkStatus.LEASED:
            raise RuntimeError(f"work item is not leased: {current.status.value}")
        if current.lease_owner != worker_id:
            raise PermissionError(
                f"lease belongs to {current.lease_owner!r}, not {worker_id!r}"
            )
        self.connection.execute(
            """
            UPDATE work_items
            SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
            WHERE item_id = ?
            """,
            (WorkStatus.PENDING.value, now, item_id),
        )

    def cancel(self, item_id: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        current = self.get(item_id)
        if current.status in {
            WorkStatus.COMPLETED,
            WorkStatus.FAILED,
            WorkStatus.CANCELLED,
        }:
            raise RuntimeError(f"work item already terminal: {current.status.value}")
        self.connection.execute(
            """
            UPDATE work_items
            SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
            WHERE item_id = ?
            """,
            (WorkStatus.CANCELLED.value, now, item_id),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DurableWorkQueue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

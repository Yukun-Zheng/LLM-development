from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    trace_id: str
    work_item_id: str
    thread_id: str
    created_at: float


@dataclass(frozen=True, slots=True)
class RolloutTraceEvent:
    event_id: int
    trace_id: str
    ordinal: int
    kind: str
    turn_id: str | None
    source_event_id: int | None
    artifact_id: str | None
    parent_event_ids: tuple[int, ...]
    payload: dict[str, Any]
    prev_hash: str
    event_hash: str
    created_at: float


class RolloutTraceStore:
    """Append-only, hash-chained provenance for one durable work item.

    ``DurableEventStream`` is optimized for control-plane replay. This store has
    a different contract: it builds an evidence bundle for one work item/turn by
    linking model/tool events, steering, artifacts and verifier outcomes.

    The SHA-256 chain is *tamper-evident*, not a cryptographic signature. A
    process that can rewrite both rows and hashes can forge history; production
    audit would additionally anchor digests in an external trusted log.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rollout_traces (
                trace_id TEXT PRIMARY KEY,
                work_item_id TEXT NOT NULL UNIQUE,
                thread_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rollout_trace_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                kind TEXT NOT NULL,
                turn_id TEXT,
                source_event_id INTEGER,
                artifact_id TEXT,
                parent_event_ids_json TEXT NOT NULL DEFAULT '[]',
                payload_json TEXT NOT NULL DEFAULT '{}',
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(trace_id) REFERENCES rollout_traces(trace_id),
                UNIQUE(trace_id, ordinal)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_rollout_trace_events_trace "
            "ON rollout_trace_events(trace_id, ordinal)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_rollout_traces_thread "
            "ON rollout_traces(thread_id, created_at, trace_id)"
        )
        self.connection.commit()

    @staticmethod
    def trace_id_for_work_item(work_item_id: str) -> str:
        if not work_item_id:
            raise ValueError("work_item_id cannot be empty")
        return f"trace_{work_item_id}"

    def ensure_trace(
        self,
        work_item_id: str,
        thread_id: str,
        *,
        created_at: float | None = None,
    ) -> RolloutTrace:
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        existing = self.get_by_work_item(work_item_id)
        if existing is not None:
            if existing.thread_id != thread_id:
                raise ValueError(
                    f"work item {work_item_id!r} is already bound to thread "
                    f"{existing.thread_id!r}, not {thread_id!r}"
                )
            return existing
        trace_id = self.trace_id_for_work_item(work_item_id)
        timestamp = time.time() if created_at is None else created_at
        self.connection.execute(
            """
            INSERT INTO rollout_traces(trace_id, work_item_id, thread_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (trace_id, work_item_id, thread_id, timestamp),
        )
        self.connection.commit()
        return RolloutTrace(trace_id, work_item_id, thread_id, timestamp)

    def get(self, trace_id: str) -> RolloutTrace:
        row = self.connection.execute(
            """
            SELECT trace_id, work_item_id, thread_id, created_at
            FROM rollout_traces WHERE trace_id = ?
            """,
            (trace_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown rollout trace: {trace_id}")
        return RolloutTrace(str(row[0]), str(row[1]), str(row[2]), float(row[3]))

    def get_by_work_item(self, work_item_id: str) -> RolloutTrace | None:
        row = self.connection.execute(
            """
            SELECT trace_id, work_item_id, thread_id, created_at
            FROM rollout_traces WHERE work_item_id = ?
            """,
            (work_item_id,),
        ).fetchone()
        if row is None:
            return None
        return RolloutTrace(str(row[0]), str(row[1]), str(row[2]), float(row[3]))

    @staticmethod
    def _canonical_event_bytes(
        *,
        trace_id: str,
        ordinal: int,
        kind: str,
        turn_id: str | None,
        source_event_id: int | None,
        artifact_id: str | None,
        parent_event_ids: tuple[int, ...],
        payload: dict[str, Any],
        prev_hash: str,
        created_at: float,
    ) -> bytes:
        return json.dumps(
            {
                "trace_id": trace_id,
                "ordinal": ordinal,
                "kind": kind,
                "turn_id": turn_id,
                "source_event_id": source_event_id,
                "artifact_id": artifact_id,
                "parent_event_ids": list(parent_event_ids),
                "payload": payload,
                "prev_hash": prev_hash,
                "created_at": created_at,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _event_hash(cls, **kwargs: Any) -> str:
        return hashlib.sha256(cls._canonical_event_bytes(**kwargs)).hexdigest()

    def append(
        self,
        work_item_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        source_event_id: int | None = None,
        artifact_id: str | None = None,
        parent_event_ids: tuple[int, ...] = (),
        created_at: float | None = None,
    ) -> int:
        if not kind:
            raise ValueError("trace event kind cannot be empty")
        trace = self.get_by_work_item(work_item_id)
        if trace is None:
            if thread_id is None:
                raise KeyError(
                    f"trace for work item {work_item_id!r} does not exist and thread_id was not supplied"
                )
            trace = self.ensure_trace(work_item_id, thread_id, created_at=created_at)
        elif thread_id is not None and trace.thread_id != thread_id:
            raise ValueError("trace/thread mismatch")

        for parent_id in parent_event_ids:
            row = self.connection.execute(
                "SELECT trace_id FROM rollout_trace_events WHERE event_id = ?",
                (parent_id,),
            ).fetchone()
            if row is None or str(row[0]) != trace.trace_id:
                raise ValueError(
                    f"parent event {parent_id} does not belong to trace {trace.trace_id}"
                )

        last = self.connection.execute(
            """
            SELECT ordinal, event_hash FROM rollout_trace_events
            WHERE trace_id = ? ORDER BY ordinal DESC LIMIT 1
            """,
            (trace.trace_id,),
        ).fetchone()
        ordinal = 0 if last is None else int(last[0]) + 1
        prev_hash = "0" * 64 if last is None else str(last[1])
        timestamp = time.time() if created_at is None else created_at
        normalized_payload = payload or {}
        digest = self._event_hash(
            trace_id=trace.trace_id,
            ordinal=ordinal,
            kind=kind,
            turn_id=turn_id,
            source_event_id=source_event_id,
            artifact_id=artifact_id,
            parent_event_ids=parent_event_ids,
            payload=normalized_payload,
            prev_hash=prev_hash,
            created_at=timestamp,
        )
        cursor = self.connection.execute(
            """
            INSERT INTO rollout_trace_events(
                trace_id, ordinal, kind, turn_id, source_event_id, artifact_id,
                parent_event_ids_json, payload_json, prev_hash, event_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                ordinal,
                kind,
                turn_id,
                source_event_id,
                artifact_id,
                json.dumps(parent_event_ids, ensure_ascii=False),
                json.dumps(normalized_payload, ensure_ascii=False, sort_keys=True),
                prev_hash,
                digest,
                timestamp,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def events(self, trace_id: str) -> list[RolloutTraceEvent]:
        self.get(trace_id)
        rows = self.connection.execute(
            """
            SELECT event_id, trace_id, ordinal, kind, turn_id, source_event_id,
                   artifact_id, parent_event_ids_json, payload_json,
                   prev_hash, event_hash, created_at
            FROM rollout_trace_events
            WHERE trace_id = ? ORDER BY ordinal ASC
            """,
            (trace_id,),
        ).fetchall()
        return [
            RolloutTraceEvent(
                event_id=int(row[0]),
                trace_id=str(row[1]),
                ordinal=int(row[2]),
                kind=str(row[3]),
                turn_id=None if row[4] is None else str(row[4]),
                source_event_id=None if row[5] is None else int(row[5]),
                artifact_id=None if row[6] is None else str(row[6]),
                parent_event_ids=tuple(int(item) for item in json.loads(str(row[7]))),
                payload=json.loads(str(row[8])),
                prev_hash=str(row[9]),
                event_hash=str(row[10]),
                created_at=float(row[11]),
            )
            for row in rows
        ]

    def list_thread(self, thread_id: str) -> list[RolloutTrace]:
        rows = self.connection.execute(
            """
            SELECT trace_id, work_item_id, thread_id, created_at
            FROM rollout_traces WHERE thread_id = ?
            ORDER BY created_at ASC, trace_id ASC
            """,
            (thread_id,),
        ).fetchall()
        return [
            RolloutTrace(str(row[0]), str(row[1]), str(row[2]), float(row[3]))
            for row in rows
        ]

    def verify_integrity(self, trace_id: str) -> bool:
        events = self.events(trace_id)
        prev_hash = "0" * 64
        for expected_ordinal, event in enumerate(events):
            if event.ordinal != expected_ordinal or event.prev_hash != prev_hash:
                return False
            expected_hash = self._event_hash(
                trace_id=event.trace_id,
                ordinal=event.ordinal,
                kind=event.kind,
                turn_id=event.turn_id,
                source_event_id=event.source_event_id,
                artifact_id=event.artifact_id,
                parent_event_ids=event.parent_event_ids,
                payload=event.payload,
                prev_hash=event.prev_hash,
                created_at=event.created_at,
            )
            if expected_hash != event.event_hash:
                return False
            prev_hash = event.event_hash
        return True

    def export_work_item(self, work_item_id: str) -> dict[str, Any]:
        trace = self.get_by_work_item(work_item_id)
        if trace is None:
            raise KeyError(f"unknown rollout trace for work item: {work_item_id}")
        events = self.events(trace.trace_id)
        return {
            "schemaVersion": 1,
            "traceId": trace.trace_id,
            "workItemId": trace.work_item_id,
            "threadId": trace.thread_id,
            "createdAt": trace.created_at,
            "integrityVerified": self.verify_integrity(trace.trace_id),
            "headHash": events[-1].event_hash if events else "0" * 64,
            "events": [
                {
                    "eventId": event.event_id,
                    "ordinal": event.ordinal,
                    "kind": event.kind,
                    "turnId": event.turn_id,
                    "sourceEventId": event.source_event_id,
                    "artifactId": event.artifact_id,
                    "parentEventIds": list(event.parent_event_ids),
                    "payload": event.payload,
                    "prevHash": event.prev_hash,
                    "eventHash": event.event_hash,
                    "createdAt": event.created_at,
                }
                for event in events
            ],
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RolloutTraceStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

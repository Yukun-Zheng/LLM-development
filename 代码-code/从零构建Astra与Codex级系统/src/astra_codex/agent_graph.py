from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class AgentStatus(str, Enum):
    ACTIVE = "active"
    BUSY = "busy"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_AGENT_STATUSES = {
    AgentStatus.COMPLETED,
    AgentStatus.FAILED,
    AgentStatus.CANCELLED,
}


_ALLOWED_AGENT_TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.ACTIVE: frozenset(
        {
            AgentStatus.BUSY,
            AgentStatus.WAITING,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.BUSY: frozenset(
        {
            AgentStatus.ACTIVE,
            AgentStatus.WAITING,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.WAITING: frozenset(
        {
            AgentStatus.ACTIVE,
            AgentStatus.BUSY,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
    AgentStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class AgentNode:
    agent_id: str
    thread_id: str
    role: str
    parent_agent_id: str | None
    status: AgentStatus
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_id: int
    agent_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: float


class MessageStatus(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    ACKED = "acked"
    CANCELLED = "cancelled"


TERMINAL_MESSAGE_STATUSES = {MessageStatus.ACKED, MessageStatus.CANCELLED}


@dataclass(frozen=True, slots=True)
class AgentMessage:
    message_id: str
    sender_agent_id: str
    recipient_agent_id: str
    kind: str
    payload: dict[str, Any]
    status: MessageStatus
    lease_owner: str | None
    lease_until: float | None
    result: dict[str, Any] | None
    created_at: float
    updated_at: float


class PersistentAgentGraph:
    """SQLite-backed AgentGraph plus durable point-to-point mailbox.

    This module deliberately separates *agent topology* from model prompting.
    Parent/child relationships, agent lifecycle and message ownership survive a
    process restart. Mail delivery uses the same lease/reclaim principle as the
    main durable work queue so a crashed mailbox consumer does not permanently
    lose a message.

    It is still a single-node reference implementation. SQLite serialization is
    useful for making semantics inspectable, but this is not a distributed actor
    system, consensus protocol or exactly-once messaging service.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                parent_agent_id TEXT,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(parent_agent_id) REFERENCES agents(agent_id)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_parent "
            "ON agents(parent_agent_id, created_at, agent_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agents_thread "
            "ON agents(thread_id, created_at, agent_id)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_events_agent "
            "ON agent_events(agent_id, event_id)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                message_id TEXT PRIMARY KEY,
                sender_agent_id TEXT NOT NULL,
                recipient_agent_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_until REAL,
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(sender_agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY(recipient_agent_id) REFERENCES agents(agent_id)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_messages_claim "
            "ON agent_messages(recipient_agent_id, status, lease_until, created_at, message_id)"
        )

    @staticmethod
    def _agent_from_row(row: tuple[object, ...]) -> AgentNode:
        return AgentNode(
            agent_id=str(row[0]),
            thread_id=str(row[1]),
            role=str(row[2]),
            parent_agent_id=None if row[3] is None else str(row[3]),
            status=AgentStatus(str(row[4])),
            metadata=json.loads(str(row[5])),
            created_at=float(row[6]),
            updated_at=float(row[7]),
        )

    @staticmethod
    def _message_from_row(row: tuple[object, ...]) -> AgentMessage:
        return AgentMessage(
            message_id=str(row[0]),
            sender_agent_id=str(row[1]),
            recipient_agent_id=str(row[2]),
            kind=str(row[3]),
            payload=json.loads(str(row[4])),
            status=MessageStatus(str(row[5])),
            lease_owner=None if row[6] is None else str(row[6]),
            lease_until=None if row[7] is None else float(row[7]),
            result=None if row[8] is None else json.loads(str(row[8])),
            created_at=float(row[9]),
            updated_at=float(row[10]),
        )

    def _append_agent_event(
        self,
        agent_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> int:
        timestamp = time.time() if now is None else now
        cursor = self.connection.execute(
            """
            INSERT INTO agent_events(agent_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                agent_id,
                event_type,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                timestamp,
            ),
        )
        return int(cursor.lastrowid)

    def create_agent(
        self,
        thread_id: str,
        role: str,
        *,
        agent_id: str | None = None,
        parent_agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> str:
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not role:
            raise ValueError("role cannot be empty")
        if parent_agent_id is not None:
            parent = self.get_agent(parent_agent_id)
            if parent.status in TERMINAL_AGENT_STATUSES:
                raise RuntimeError(
                    f"cannot spawn child from terminal agent: {parent.status.value}"
                )
            if parent.thread_id != thread_id:
                raise ValueError("parent and child agents must belong to the same thread")
        timestamp = time.time() if now is None else now
        agent_id = agent_id or f"agent_{uuid.uuid4().hex}"
        try:
            self.connection.execute(
                """
                INSERT INTO agents(
                    agent_id, thread_id, role, parent_agent_id, status,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    thread_id,
                    role,
                    parent_agent_id,
                    AgentStatus.ACTIVE.value,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    timestamp,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"agent already exists: {agent_id}") from exc
        self._append_agent_event(
            agent_id,
            "agent.created",
            {
                "threadId": thread_id,
                "role": role,
                "parentAgentId": parent_agent_id,
                "metadata": metadata or {},
            },
            now=timestamp,
        )
        return agent_id

    def get_agent(self, agent_id: str) -> AgentNode:
        row = self.connection.execute(
            """
            SELECT agent_id, thread_id, role, parent_agent_id, status,
                   metadata_json, created_at, updated_at
            FROM agents WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown agent: {agent_id}")
        return self._agent_from_row(row)

    def set_status(
        self,
        agent_id: str,
        status: AgentStatus,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> None:
        current = self.get_agent(agent_id)
        if status is current.status:
            return
        if status not in _ALLOWED_AGENT_TRANSITIONS[current.status]:
            raise RuntimeError(
                f"invalid agent transition: {current.status.value} -> {status.value}"
            )
        timestamp = time.time() if now is None else now
        self.connection.execute(
            "UPDATE agents SET status = ?, updated_at = ? WHERE agent_id = ?",
            (status.value, timestamp, agent_id),
        )
        self._append_agent_event(
            agent_id,
            "agent.status_changed",
            {
                "from": current.status.value,
                "to": status.value,
                "reason": reason,
            },
            now=timestamp,
        )

    def children(self, agent_id: str) -> tuple[AgentNode, ...]:
        self.get_agent(agent_id)
        rows = self.connection.execute(
            """
            SELECT agent_id, thread_id, role, parent_agent_id, status,
                   metadata_json, created_at, updated_at
            FROM agents WHERE parent_agent_id = ?
            ORDER BY created_at ASC, agent_id ASC
            """,
            (agent_id,),
        ).fetchall()
        return tuple(self._agent_from_row(row) for row in rows)

    def descendants(self, agent_id: str) -> tuple[AgentNode, ...]:
        self.get_agent(agent_id)
        ordered: list[AgentNode] = []
        queue = list(self.children(agent_id))
        while queue:
            current = queue.pop(0)
            ordered.append(current)
            queue.extend(self.children(current.agent_id))
        return tuple(ordered)

    def lineage(self, agent_id: str) -> tuple[AgentNode, ...]:
        current = self.get_agent(agent_id)
        lineage = [current]
        while current.parent_agent_id is not None:
            current = self.get_agent(current.parent_agent_id)
            lineage.append(current)
        lineage.reverse()
        return tuple(lineage)

    def agents_for_thread(self, thread_id: str) -> tuple[AgentNode, ...]:
        rows = self.connection.execute(
            """
            SELECT agent_id, thread_id, role, parent_agent_id, status,
                   metadata_json, created_at, updated_at
            FROM agents WHERE thread_id = ?
            ORDER BY created_at ASC, agent_id ASC
            """,
            (thread_id,),
        ).fetchall()
        return tuple(self._agent_from_row(row) for row in rows)

    def events(self, agent_id: str) -> tuple[AgentEvent, ...]:
        self.get_agent(agent_id)
        rows = self.connection.execute(
            """
            SELECT event_id, agent_id, event_type, payload_json, created_at
            FROM agent_events WHERE agent_id = ? ORDER BY event_id ASC
            """,
            (agent_id,),
        ).fetchall()
        return tuple(
            AgentEvent(
                event_id=int(row[0]),
                agent_id=str(row[1]),
                event_type=str(row[2]),
                payload=json.loads(str(row[3])),
                created_at=float(row[4]),
            )
            for row in rows
        )

    def cancel_subtree(
        self,
        agent_id: str,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> tuple[str, ...]:
        nodes = [self.get_agent(agent_id), *self.descendants(agent_id)]
        # Cancel leaves before parents so a coordinator's state never suggests
        # the parent is terminal while descendants remain active.
        cancelled: list[str] = []
        for node in reversed(nodes):
            current = self.get_agent(node.agent_id)
            if current.status in TERMINAL_AGENT_STATUSES:
                continue
            self.set_status(
                current.agent_id,
                AgentStatus.CANCELLED,
                reason=reason,
                now=now,
            )
            cancelled.append(current.agent_id)
        return tuple(cancelled)

    def send(
        self,
        sender_agent_id: str,
        recipient_agent_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        message_id: str | None = None,
        now: float | None = None,
    ) -> str:
        sender = self.get_agent(sender_agent_id)
        recipient = self.get_agent(recipient_agent_id)
        if sender.thread_id != recipient.thread_id:
            raise ValueError("cross-thread agent messages are not allowed in this reference runtime")
        if sender.status in TERMINAL_AGENT_STATUSES:
            raise RuntimeError("terminal sender cannot send new messages")
        if recipient.status in TERMINAL_AGENT_STATUSES:
            raise RuntimeError("cannot send to terminal recipient")
        if not kind:
            raise ValueError("message kind cannot be empty")
        timestamp = time.time() if now is None else now
        message_id = message_id or f"msg_{uuid.uuid4().hex}"
        try:
            self.connection.execute(
                """
                INSERT INTO agent_messages(
                    message_id, sender_agent_id, recipient_agent_id, kind,
                    payload_json, status, lease_owner, lease_until, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    message_id,
                    sender_agent_id,
                    recipient_agent_id,
                    kind,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    MessageStatus.PENDING.value,
                    timestamp,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"message already exists: {message_id}") from exc
        self._append_agent_event(
            sender_agent_id,
            "mailbox.sent",
            {
                "messageId": message_id,
                "recipientAgentId": recipient_agent_id,
                "kind": kind,
            },
            now=timestamp,
        )
        self._append_agent_event(
            recipient_agent_id,
            "mailbox.received",
            {
                "messageId": message_id,
                "senderAgentId": sender_agent_id,
                "kind": kind,
            },
            now=timestamp,
        )
        return message_id

    def get_message(self, message_id: str) -> AgentMessage:
        row = self.connection.execute(
            """
            SELECT message_id, sender_agent_id, recipient_agent_id, kind,
                   payload_json, status, lease_owner, lease_until, result_json,
                   created_at, updated_at
            FROM agent_messages WHERE message_id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown agent message: {message_id}")
        return self._message_from_row(row)

    def inbox(
        self,
        recipient_agent_id: str,
        *,
        statuses: Iterable[MessageStatus] | None = None,
    ) -> tuple[AgentMessage, ...]:
        self.get_agent(recipient_agent_id)
        status_values = tuple(status.value for status in statuses or ())
        query = """
            SELECT message_id, sender_agent_id, recipient_agent_id, kind,
                   payload_json, status, lease_owner, lease_until, result_json,
                   created_at, updated_at
            FROM agent_messages WHERE recipient_agent_id = ?
        """
        params: list[object] = [recipient_agent_id]
        if status_values:
            placeholders = ",".join("?" for _ in status_values)
            query += f" AND status IN ({placeholders})"
            params.extend(status_values)
        query += " ORDER BY created_at ASC, message_id ASC"
        rows = self.connection.execute(query, params).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    def claim_message(
        self,
        recipient_agent_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        kinds: set[str] | None = None,
        now: float | None = None,
    ) -> AgentMessage | None:
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        recipient = self.get_agent(recipient_agent_id)
        if recipient.status in TERMINAL_AGENT_STATUSES:
            return None
        timestamp = time.time() if now is None else now

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            query = """
                SELECT message_id FROM agent_messages
                WHERE recipient_agent_id = ?
                  AND (
                    status = ?
                    OR (status = ? AND lease_until <= ?)
                  )
            """
            params: list[object] = [
                recipient_agent_id,
                MessageStatus.PENDING.value,
                MessageStatus.LEASED.value,
                timestamp,
            ]
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(sorted(kinds))
            query += " ORDER BY created_at ASC, message_id ASC LIMIT 1"
            row = self.connection.execute(query, params).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            message_id = str(row[0])
            self.connection.execute(
                """
                UPDATE agent_messages
                SET status = ?, lease_owner = ?, lease_until = ?, updated_at = ?
                WHERE message_id = ?
                """,
                (
                    MessageStatus.LEASED.value,
                    worker_id,
                    timestamp + lease_seconds,
                    timestamp,
                    message_id,
                ),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

        claimed = self.get_message(message_id)
        self._append_agent_event(
            recipient_agent_id,
            "mailbox.claimed",
            {
                "messageId": message_id,
                "workerId": worker_id,
                "leaseUntil": claimed.lease_until,
            },
            now=timestamp,
        )
        return claimed

    def ack_message(
        self,
        message_id: str,
        worker_id: str,
        *,
        result: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        current = self.get_message(message_id)
        if current.status is not MessageStatus.LEASED:
            raise RuntimeError(f"message is not leased: {current.status.value}")
        if current.lease_owner != worker_id:
            raise PermissionError(
                f"message lease belongs to {current.lease_owner!r}, not {worker_id!r}"
            )
        timestamp = time.time() if now is None else now
        self.connection.execute(
            """
            UPDATE agent_messages
            SET status = ?, lease_owner = NULL, lease_until = NULL,
                result_json = ?, updated_at = ?
            WHERE message_id = ?
            """,
            (
                MessageStatus.ACKED.value,
                None
                if result is None
                else json.dumps(result, ensure_ascii=False, sort_keys=True),
                timestamp,
                message_id,
            ),
        )
        self._append_agent_event(
            current.recipient_agent_id,
            "mailbox.acked",
            {"messageId": message_id, "workerId": worker_id, "result": result},
            now=timestamp,
        )

    def release_message(
        self,
        message_id: str,
        worker_id: str,
        *,
        now: float | None = None,
    ) -> None:
        current = self.get_message(message_id)
        if current.status is not MessageStatus.LEASED:
            raise RuntimeError(f"message is not leased: {current.status.value}")
        if current.lease_owner != worker_id:
            raise PermissionError(
                f"message lease belongs to {current.lease_owner!r}, not {worker_id!r}"
            )
        timestamp = time.time() if now is None else now
        self.connection.execute(
            """
            UPDATE agent_messages
            SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
            WHERE message_id = ?
            """,
            (MessageStatus.PENDING.value, timestamp, message_id),
        )

    def cancel_message(self, message_id: str, *, now: float | None = None) -> None:
        current = self.get_message(message_id)
        if current.status in TERMINAL_MESSAGE_STATUSES:
            raise RuntimeError(f"message already terminal: {current.status.value}")
        timestamp = time.time() if now is None else now
        self.connection.execute(
            """
            UPDATE agent_messages
            SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
            WHERE message_id = ?
            """,
            (MessageStatus.CANCELLED.value, timestamp, message_id),
        )
        self._append_agent_event(
            current.recipient_agent_id,
            "mailbox.cancelled",
            {"messageId": message_id},
            now=timestamp,
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PersistentAgentGraph":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

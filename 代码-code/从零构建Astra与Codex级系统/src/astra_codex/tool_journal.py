from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .tools import ToolRegistry, ToolResult


class ExecutionStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    idempotency_key: str
    tool_name: str
    arguments_hash: str
    status: ExecutionStatus
    result: ToolResult | None


def _arguments_hash(arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DurableToolJournal:
    """Persistent journal for tool-dispatch idempotency and in-doubt detection.

    A journal cannot magically provide exactly-once semantics for an arbitrary
    external side effect. The important contract is more conservative:

    * COMPLETED + same key/tool/args -> replay stored result, do not execute.
    * STARTED without a stored result -> execution is *in doubt* after a crash.
      Non-idempotent tools must not be blindly retried.

    Tool-specific reconciliation or an external transaction/idempotency API is
    required to resolve truly in-doubt non-idempotent operations.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_executions (
                idempotency_key TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                result_ok INTEGER,
                result_output TEXT,
                result_metadata_json TEXT
            )
            """
        )
        self.connection.commit()

    def get(self, idempotency_key: str) -> ToolExecutionRecord | None:
        row = self.connection.execute(
            """
            SELECT idempotency_key, tool_name, arguments_hash, status,
                   result_ok, result_output, result_metadata_json
            FROM tool_executions WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        result: ToolResult | None = None
        if row[4] is not None:
            result = ToolResult(
                bool(row[4]),
                str(row[5] or ""),
                None if row[6] is None else json.loads(str(row[6])),
            )
        return ToolExecutionRecord(
            idempotency_key=str(row[0]),
            tool_name=str(row[1]),
            arguments_hash=str(row[2]),
            status=ExecutionStatus(str(row[3])),
            result=result,
        )

    def begin(
        self,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionRecord:
        if not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        digest = _arguments_hash(arguments)
        existing = self.get(idempotency_key)
        if existing is not None:
            self._validate_match(existing, tool_name, digest)
            return existing
        self.connection.execute(
            """
            INSERT INTO tool_executions(
                idempotency_key, tool_name, arguments_hash, status
            ) VALUES (?, ?, ?, ?)
            """,
            (
                idempotency_key,
                tool_name,
                digest,
                ExecutionStatus.STARTED.value,
            ),
        )
        self.connection.commit()
        record = self.get(idempotency_key)
        assert record is not None
        return record

    def complete(self, idempotency_key: str, result: ToolResult) -> None:
        existing = self.get(idempotency_key)
        if existing is None:
            raise KeyError(f"unknown tool execution: {idempotency_key}")
        if existing.status is ExecutionStatus.COMPLETED:
            raise RuntimeError(f"tool execution already completed: {idempotency_key}")
        self.connection.execute(
            """
            UPDATE tool_executions
            SET status = ?, result_ok = ?, result_output = ?, result_metadata_json = ?
            WHERE idempotency_key = ?
            """,
            (
                ExecutionStatus.COMPLETED.value,
                int(result.ok),
                result.output,
                None
                if result.metadata is None
                else json.dumps(result.metadata, ensure_ascii=False, sort_keys=True),
                idempotency_key,
            ),
        )
        self.connection.commit()

    @staticmethod
    def _validate_match(
        record: ToolExecutionRecord,
        tool_name: str,
        digest: str,
    ) -> None:
        if record.tool_name != tool_name or record.arguments_hash != digest:
            raise ValueError(
                "idempotency key was reused for different tool name or arguments"
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DurableToolJournal":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JournaledToolExecutor:
    """Tool executor that prevents blind duplicate side effects.

    ``retryable_in_doubt_tools`` must contain only tools whose operation is
    actually safe to repeat under the same arguments, or tools backed by an
    external idempotency mechanism. The default is conservative: retry none.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        journal: DurableToolJournal,
        *,
        retryable_in_doubt_tools: frozenset[str] = frozenset(),
    ) -> None:
        self.registry = registry
        self.journal = journal
        self.retryable_in_doubt_tools = retryable_in_doubt_tools

    @property
    def specs(self):  # type: ignore[no-untyped-def]
        return self.registry.specs

    def execute(
        self,
        idempotency_key: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        digest = _arguments_hash(arguments)
        existing = self.journal.get(idempotency_key)
        if existing is not None:
            self.journal._validate_match(existing, tool_name, digest)
            if existing.status is ExecutionStatus.COMPLETED:
                assert existing.result is not None
                metadata = dict(existing.result.metadata or {})
                metadata["replayed_from_journal"] = True
                metadata["idempotency_key"] = idempotency_key
                return ToolResult(
                    existing.result.ok,
                    existing.result.output,
                    metadata,
                )
            if tool_name not in self.retryable_in_doubt_tools:
                return ToolResult(
                    False,
                    "tool execution is in-doubt after a previous STARTED record; "
                    "automatic retry is blocked",
                    {
                        "in_doubt": True,
                        "idempotency_key": idempotency_key,
                        "tool": tool_name,
                    },
                )
        else:
            self.journal.begin(idempotency_key, tool_name, arguments)

        result = self.registry.execute(tool_name, arguments)
        self.journal.complete(idempotency_key, result)
        metadata = dict(result.metadata or {})
        metadata["idempotency_key"] = idempotency_key
        return ToolResult(result.ok, result.output, metadata)

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .instructions import ResolvedInstructions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FragmentKind(str, Enum):
    RAW_EVENT = "raw_event"
    NOTE = "note"
    SUMMARY = "summary"
    ARTIFACT = "artifact"
    RETRIEVAL = "retrieval"
    INSTRUCTION = "instruction"


@dataclass(frozen=True, slots=True)
class ContextFragment:
    fragment_id: str
    kind: FragmentKind
    content: str
    source: str
    parent_ids: tuple[str, ...]
    metadata: dict[str, Any]
    created_at: str


class ContextStore:
    """Persistent provenance-aware context fragments.

    Raw observations are never overwritten by compaction. A summary is a new
    fragment that points back to its parent fragments, so any model-visible
    context item can be traced to its source material.

    Project instructions can also be recorded as typed ``INSTRUCTION`` fragments
    before they are inserted into a model prompt. That makes prompt construction
    auditable: a later rollout can recover the exact file, scope directory and
    truncation state that produced each instruction segment.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS context_fragments (
                fragment_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS compactions (
                compaction_id TEXT PRIMARY KEY,
                output_fragment_id TEXT NOT NULL,
                input_ids_json TEXT NOT NULL,
                policy TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def add(
        self,
        kind: FragmentKind,
        content: str,
        *,
        source: str,
        parent_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        fragment_id: str | None = None,
    ) -> str:
        fragment_id = fragment_id or f"ctx_{uuid.uuid4().hex}"
        for parent_id in parent_ids:
            self.get(parent_id)
        self.connection.execute(
            """
            INSERT INTO context_fragments(
                fragment_id, kind, content, source, parent_ids_json, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fragment_id,
                kind.value,
                content,
                source,
                json.dumps(parent_ids, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                _utc_now(),
            ),
        )
        self.connection.commit()
        return fragment_id

    def add_resolved_instructions(
        self,
        resolved: ResolvedInstructions,
        *,
        source: str = "project_instructions",
    ) -> tuple[str, ...]:
        """Persist every model-visible project instruction with provenance."""

        fragment_ids: list[str] = []
        for order, instruction in enumerate(resolved.sources):
            fragment_ids.append(
                self.add(
                    FragmentKind.INSTRUCTION,
                    instruction.contents,
                    source=source,
                    metadata={
                        "order": order,
                        "source_path": str(instruction.source_path),
                        "scope_directory": str(instruction.scope_directory),
                        "candidate_name": instruction.candidate_name,
                        "truncated": instruction.truncated,
                        "bytes_loaded": instruction.bytes_loaded,
                        "project_root": str(resolved.project_root),
                        "cwd": str(resolved.cwd),
                        "max_bytes": resolved.max_bytes,
                    },
                )
            )
        return tuple(fragment_ids)

    def get(self, fragment_id: str) -> ContextFragment:
        row = self.connection.execute(
            """
            SELECT fragment_id, kind, content, source, parent_ids_json, metadata_json, created_at
            FROM context_fragments WHERE fragment_id = ?
            """,
            (fragment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown context fragment: {fragment_id}")
        return ContextFragment(
            fragment_id=str(row[0]),
            kind=FragmentKind(row[1]),
            content=str(row[2]),
            source=str(row[3]),
            parent_ids=tuple(json.loads(row[4])),
            metadata=json.loads(row[5]),
            created_at=str(row[6]),
        )

    def compact(
        self,
        input_ids: list[str],
        summary: str,
        *,
        policy: str,
        source: str = "compactor",
    ) -> str:
        if not input_ids:
            raise ValueError("compaction requires at least one input fragment")
        for fragment_id in input_ids:
            self.get(fragment_id)
        output_id = self.add(
            FragmentKind.SUMMARY,
            summary,
            source=source,
            parent_ids=tuple(input_ids),
            metadata={"compaction_policy": policy},
        )
        self.connection.execute(
            """
            INSERT INTO compactions(
                compaction_id, output_fragment_id, input_ids_json, policy, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"cmp_{uuid.uuid4().hex}",
                output_id,
                json.dumps(input_ids, ensure_ascii=False),
                policy,
                _utc_now(),
            ),
        )
        self.connection.commit()
        return output_id

    def lineage(self, fragment_id: str) -> list[ContextFragment]:
        ordered: list[ContextFragment] = []
        visited: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)
            current = self.get(current_id)
            for parent_id in current.parent_ids:
                visit(parent_id)
            ordered.append(current)

        visit(fragment_id)
        return ordered

    def select_for_context(
        self,
        *,
        max_chars: int,
        kinds: set[FragmentKind] | None = None,
    ) -> list[ContextFragment]:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        rows = self.connection.execute(
            """
            SELECT fragment_id FROM context_fragments
            ORDER BY rowid DESC
            """
        ).fetchall()
        chosen: list[ContextFragment] = []
        used = 0
        for (fragment_id,) in rows:
            fragment = self.get(str(fragment_id))
            if kinds is not None and fragment.kind not in kinds:
                continue
            size = len(fragment.content)
            if chosen and used + size > max_chars:
                continue
            if not chosen and size > max_chars:
                continue
            chosen.append(fragment)
            used += size
            if used >= max_chars:
                break
        chosen.reverse()
        return chosen

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ContextStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

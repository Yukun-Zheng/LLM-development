from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    thread_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any]
    created_at: float


class ArtifactStore:
    """Content-addressed artifact metadata plus immutable file snapshots.

    Long-running agents should not reduce all work products to chat messages.
    This store snapshots files into a managed directory and records provenance
    metadata so verifiers/reviewers can inspect exactly which bytes a trajectory
    produced.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.root / "artifacts.sqlite")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                object_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_thread "
            "ON artifacts(thread_id, created_at, artifact_id)"
        )
        self.connection.commit()

    @staticmethod
    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def put_bytes(
        self,
        thread_id: str,
        data: bytes,
        *,
        kind: str,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        now: float | None = None,
    ) -> str:
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not kind:
            raise ValueError("artifact kind cannot be empty")
        digest = self._digest(data)
        object_path = self.objects / digest[:2] / digest[2:]
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not object_path.exists():
            object_path.write_bytes(data)
        elif object_path.read_bytes() != data:
            raise RuntimeError("sha256 collision or corrupted artifact object")

        artifact_id = artifact_id or f"art_{uuid.uuid4().hex}"
        timestamp = time.time() if now is None else now
        self.connection.execute(
            """
            INSERT INTO artifacts(
                artifact_id, thread_id, kind, object_path, sha256,
                size_bytes, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                thread_id,
                kind,
                str(object_path.relative_to(self.root)),
                digest,
                len(data),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                timestamp,
            ),
        )
        self.connection.commit()
        return artifact_id

    def snapshot_file(
        self,
        thread_id: str,
        source: str | Path,
        *,
        kind: str = "file",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        now: float | None = None,
    ) -> str:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        merged = {"source_path": str(source_path), **(metadata or {})}
        return self.put_bytes(
            thread_id,
            source_path.read_bytes(),
            kind=kind,
            metadata=merged,
            artifact_id=artifact_id,
            now=now,
        )

    def get(self, artifact_id: str) -> ArtifactRecord:
        row = self.connection.execute(
            """
            SELECT artifact_id, thread_id, kind, object_path, sha256,
                   size_bytes, metadata_json, created_at
            FROM artifacts WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return ArtifactRecord(
            artifact_id=str(row[0]),
            thread_id=str(row[1]),
            kind=str(row[2]),
            path=str((self.root / str(row[3])).resolve()),
            sha256=str(row[4]),
            size_bytes=int(row[5]),
            metadata=json.loads(str(row[6])),
            created_at=float(row[7]),
        )

    def read_bytes(self, artifact_id: str) -> bytes:
        record = self.get(artifact_id)
        data = Path(record.path).read_bytes()
        if self._digest(data) != record.sha256:
            raise RuntimeError(f"artifact checksum mismatch: {artifact_id}")
        return data

    def list_thread(self, thread_id: str) -> tuple[ArtifactRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT artifact_id FROM artifacts
            WHERE thread_id = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (thread_id,),
        ).fetchall()
        return tuple(self.get(str(row[0])) for row in rows)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

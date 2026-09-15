"""Source-first educational subset of the A2A v1.0 protocol.

Primary normative source used for this module:
https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

The goal is to make the protocol's *data model and task lifecycle* inspectable
before introducing an official SDK.  This is deliberately not a complete A2A
implementation: streaming RPCs, push notifications, signatures, OAuth/mTLS,
full transport bindings, extensions and conformance certification are outside
this reference layer.

Implemented v1 concepts:

* AgentCard + supportedInterfaces (protocolVersion lives on each interface);
* Message / Part / Artifact;
* Task / TaskStatus / exact v1 TaskState strings;
* SendMessage / GetTask / ListTasks / CancelTask teaching dispatcher;
* SQLite persistence and restart recovery;
* deferred ``return_immediately`` processing through an explicit runner.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

A2A_PROTOCOL_VERSION = "1.0"
A2A_SOURCE_COMMIT = "6d6640c29b102f7a8d23784901351b5d2454fe71"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class A2AProtocolError(ValueError):
    pass


class A2ARole(str, Enum):
    ROLE_UNSPECIFIED = "ROLE_UNSPECIFIED"
    ROLE_USER = "ROLE_USER"
    ROLE_AGENT = "ROLE_AGENT"


class A2ATaskState(str, Enum):
    """Exact A2A v1 TaskState JSON enum spellings."""

    TASK_STATE_UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
    TASK_STATE_WORKING = "TASK_STATE_WORKING"
    TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
    TASK_STATE_FAILED = "TASK_STATE_FAILED"
    TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
    TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    TASK_STATE_REJECTED = "TASK_STATE_REJECTED"
    TASK_STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


TERMINAL_A2A_STATES = frozenset(
    {
        A2ATaskState.TASK_STATE_COMPLETED,
        A2ATaskState.TASK_STATE_FAILED,
        A2ATaskState.TASK_STATE_CANCELED,
        A2ATaskState.TASK_STATE_REJECTED,
    }
)

INTERRUPTED_A2A_STATES = frozenset(
    {
        A2ATaskState.TASK_STATE_INPUT_REQUIRED,
        A2ATaskState.TASK_STATE_AUTH_REQUIRED,
    }
)


@dataclass(frozen=True, slots=True)
class A2APart:
    """A2A v1 ``Part`` oneof content.

    Proto JSON serializes the selected oneof member directly (``text``, ``raw``,
    ``url`` or ``data``).  There is intentionally no legacy inline ``kind``
    discriminator in this reference representation.
    """

    text: str | None = None
    raw: bytes | None = None
    url: str | None = None
    data: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        populated = sum(
            value is not None for value in (self.text, self.raw, self.url, self.data)
        )
        if populated != 1:
            raise A2AProtocolError(
                "A2A Part requires exactly one of text/raw/url/data"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.text is not None:
            payload["text"] = self.text
        elif self.raw is not None:
            payload["raw"] = base64.b64encode(self.raw).decode("ascii")
        elif self.url is not None:
            payload["url"] = self.url
        else:
            payload["data"] = self.data
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.filename is not None:
            payload["filename"] = self.filename
        if self.media_type is not None:
            payload["mediaType"] = self.media_type
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2APart":
        if not isinstance(payload, dict):
            raise A2AProtocolError("Part must be an object")
        content_keys = [key for key in ("text", "raw", "url", "data") if key in payload]
        if len(content_keys) != 1:
            raise A2AProtocolError(
                "Part JSON must contain exactly one of text/raw/url/data"
            )
        kwargs: dict[str, Any] = {
            "metadata": dict(payload.get("metadata") or {}),
            "filename": payload.get("filename"),
            "media_type": payload.get("mediaType"),
        }
        key = content_keys[0]
        value = payload[key]
        if key == "raw":
            if not isinstance(value, str):
                raise A2AProtocolError("raw Part content must be base64 text")
            try:
                kwargs["raw"] = base64.b64decode(value, validate=True)
            except ValueError as exc:
                raise A2AProtocolError("invalid base64 raw Part") from exc
        else:
            kwargs[key] = value
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class A2AMessage:
    message_id: str
    role: A2ARole
    parts: tuple[A2APart, ...]
    context_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()
    reference_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.message_id:
            raise A2AProtocolError("message_id is required")
        if not self.parts:
            raise A2AProtocolError("message requires at least one Part")
        if self.role is A2ARole.ROLE_UNSPECIFIED:
            raise A2AProtocolError("message role cannot be ROLE_UNSPECIFIED")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messageId": self.message_id,
            "role": self.role.value,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.context_id is not None:
            payload["contextId"] = self.context_id
        if self.task_id is not None:
            payload["taskId"] = self.task_id
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.extensions:
            payload["extensions"] = list(self.extensions)
        if self.reference_task_ids:
            payload["referenceTaskIds"] = list(self.reference_task_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2AMessage":
        try:
            role = A2ARole(str(payload["role"]))
            parts_raw = payload["parts"]
            message_id = str(payload["messageId"])
        except (KeyError, ValueError, TypeError) as exc:
            raise A2AProtocolError("invalid A2A Message") from exc
        if not isinstance(parts_raw, list):
            raise A2AProtocolError("Message.parts must be an array")
        return cls(
            message_id=message_id,
            role=role,
            parts=tuple(A2APart.from_dict(part) for part in parts_raw),
            context_id=(
                None if payload.get("contextId") is None else str(payload["contextId"])
            ),
            task_id=None if payload.get("taskId") is None else str(payload["taskId"]),
            metadata=dict(payload.get("metadata") or {}),
            extensions=tuple(str(item) for item in payload.get("extensions", [])),
            reference_task_ids=tuple(
                str(item) for item in payload.get("referenceTaskIds", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class A2AArtifact:
    artifact_id: str
    parts: tuple[A2APart, ...]
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise A2AProtocolError("artifact_id is required")
        if not self.parts:
            raise A2AProtocolError("artifact requires at least one Part")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.name:
            payload["name"] = self.name
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.extensions:
            payload["extensions"] = list(self.extensions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2AArtifact":
        parts = payload.get("parts")
        if not isinstance(parts, list):
            raise A2AProtocolError("Artifact.parts must be an array")
        return cls(
            artifact_id=str(payload.get("artifactId", "")),
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            parts=tuple(A2APart.from_dict(part) for part in parts),
            metadata=dict(payload.get("metadata") or {}),
            extensions=tuple(str(item) for item in payload.get("extensions", [])),
        )


@dataclass(frozen=True, slots=True)
class A2ATaskStatus:
    state: A2ATaskState
    message: A2AMessage | None = None
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state.value,
            "timestamp": self.timestamp,
        }
        if self.message is not None:
            payload["message"] = self.message.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2ATaskStatus":
        try:
            state = A2ATaskState(str(payload["state"]))
        except (KeyError, ValueError) as exc:
            raise A2AProtocolError("invalid TaskStatus.state") from exc
        message_raw = payload.get("message")
        return cls(
            state=state,
            message=(
                None
                if message_raw is None
                else A2AMessage.from_dict(dict(message_raw))
            ),
            timestamp=str(payload.get("timestamp") or _utc_now()),
        )


@dataclass(frozen=True, slots=True)
class A2ATask:
    id: str
    context_id: str
    status: A2ATaskStatus
    artifacts: tuple[A2AArtifact, ...] = ()
    history: tuple[A2AMessage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
        }
        if self.artifacts:
            payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        if self.history:
            payload["history"] = [message.to_dict() for message in self.history]
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2ATask":
        return cls(
            id=str(payload["id"]),
            context_id=str(payload.get("contextId", "")),
            status=A2ATaskStatus.from_dict(dict(payload["status"])),
            artifacts=tuple(
                A2AArtifact.from_dict(item) for item in payload.get("artifacts", [])
            ),
            history=tuple(
                A2AMessage.from_dict(item) for item in payload.get("history", [])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class A2AAgentInterface:
    url: str
    protocol_binding: str
    protocol_version: str = A2A_PROTOCOL_VERSION
    tenant: str | None = None

    def __post_init__(self) -> None:
        if not self.url or not self.protocol_binding or not self.protocol_version:
            raise A2AProtocolError("AgentInterface url/binding/version are required")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "url": self.url,
            "protocolBinding": self.protocol_binding,
            "protocolVersion": self.protocol_version,
        }
        if self.tenant is not None:
            payload["tenant"] = self.tenant
        return payload


@dataclass(frozen=True, slots=True)
class A2AAgentSkill:
    id: str
    name: str
    description: str
    tags: tuple[str, ...]
    examples: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
        }
        if self.examples:
            payload["examples"] = list(self.examples)
        if self.input_modes:
            payload["inputModes"] = list(self.input_modes)
        if self.output_modes:
            payload["outputModes"] = list(self.output_modes)
        return payload


@dataclass(frozen=True, slots=True)
class A2AAgentCard:
    name: str
    description: str
    supported_interfaces: tuple[A2AAgentInterface, ...]
    version: str
    default_input_modes: tuple[str, ...]
    default_output_modes: tuple[str, ...]
    skills: tuple[A2AAgentSkill, ...]
    streaming: bool = False
    push_notifications: bool = False
    extended_agent_card: bool = False
    documentation_url: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.description or not self.version:
            raise A2AProtocolError("AgentCard name/description/version are required")
        if not self.supported_interfaces:
            raise A2AProtocolError("AgentCard requires supportedInterfaces")
        if not self.default_input_modes or not self.default_output_modes:
            raise A2AProtocolError("AgentCard requires default input/output modes")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "supportedInterfaces": [
                interface.to_dict() for interface in self.supported_interfaces
            ],
            "version": self.version,
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": self.push_notifications,
                "extendedAgentCard": self.extended_agent_card,
            },
            "defaultInputModes": list(self.default_input_modes),
            "defaultOutputModes": list(self.default_output_modes),
            "skills": [skill.to_dict() for skill in self.skills],
        }
        if self.documentation_url is not None:
            payload["documentationUrl"] = self.documentation_url
        return payload


@dataclass(frozen=True, slots=True)
class A2ASendMessageConfiguration:
    accepted_output_modes: tuple[str, ...] = ()
    history_length: int | None = None
    return_immediately: bool = False

    def __post_init__(self) -> None:
        if self.history_length is not None and self.history_length < 0:
            raise A2AProtocolError("history_length cannot be negative")


@dataclass(frozen=True, slots=True)
class A2ASendMessageRequest:
    message: A2AMessage
    configuration: A2ASendMessageConfiguration = field(
        default_factory=A2ASendMessageConfiguration
    )
    tenant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class A2AExecutionResult:
    state: A2ATaskState = A2ATaskState.TASK_STATE_COMPLETED
    message: A2AMessage | None = None
    artifacts: tuple[A2AArtifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class A2ATaskStore:
    """SQLite persistence for the teaching Task model."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS a2a_tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_context "
            "ON a2a_tasks(context_id, updated_at, task_id)"
        )
        self.connection.commit()

    def put(self, task: A2ATask, *, create_only: bool = False) -> None:
        now = _utc_now()
        encoded = json.dumps(
            task.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        existing = self.connection.execute(
            "SELECT 1 FROM a2a_tasks WHERE task_id = ?", (task.id,)
        ).fetchone()
        if existing is None:
            self.connection.execute(
                """
                INSERT INTO a2a_tasks(task_id, context_id, task_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task.id, task.context_id, encoded, now, now),
            )
        elif create_only:
            raise ValueError(f"A2A task already exists: {task.id}")
        else:
            self.connection.execute(
                """
                UPDATE a2a_tasks SET context_id = ?, task_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (task.context_id, encoded, now, task.id),
            )
        self.connection.commit()

    def get(self, task_id: str) -> A2ATask:
        row = self.connection.execute(
            "SELECT task_json FROM a2a_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown A2A task: {task_id}")
        return A2ATask.from_dict(json.loads(str(row[0])))

    def list(
        self,
        *,
        context_id: str | None = None,
        states: set[A2ATaskState] | None = None,
    ) -> tuple[A2ATask, ...]:
        query = "SELECT task_json FROM a2a_tasks"
        params: list[object] = []
        if context_id is not None:
            query += " WHERE context_id = ?"
            params.append(context_id)
        query += " ORDER BY created_at ASC, task_id ASC"
        tasks = tuple(
            A2ATask.from_dict(json.loads(str(row[0])))
            for row in self.connection.execute(query, params).fetchall()
        )
        if states is None:
            return tasks
        return tuple(task for task in tasks if task.status.state in states)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "A2ATaskStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


A2AHandler = Callable[[A2ATask, A2AMessage], A2AExecutionResult]


class A2AService:
    """Minimal v1 task service using official operation names.

    ``handle_operation`` deliberately models the normative RPC names
    ``SendMessage``, ``GetTask``, ``ListTasks`` and ``CancelTask`` without
    pretending to implement every official JSON/HTTP/gRPC binding.
    """

    def __init__(
        self,
        card: A2AAgentCard,
        store: A2ATaskStore,
        *,
        handler: A2AHandler | None = None,
    ) -> None:
        self.card = card
        self.store = store
        self.handler = handler

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _agent_message(
        text: str,
        *,
        task_id: str,
        context_id: str,
        message_id: str | None = None,
    ) -> A2AMessage:
        return A2AMessage(
            message_id=message_id or A2AService._new_id("msg"),
            role=A2ARole.ROLE_AGENT,
            parts=(A2APart(text=text),),
            task_id=task_id,
            context_id=context_id,
        )

    def _with_status(
        self,
        task: A2ATask,
        state: A2ATaskState,
        *,
        message: A2AMessage | None = None,
        artifacts: tuple[A2AArtifact, ...] | None = None,
        history: tuple[A2AMessage, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> A2ATask:
        return A2ATask(
            id=task.id,
            context_id=task.context_id,
            status=A2ATaskStatus(state, message),
            artifacts=task.artifacts if artifacts is None else artifacts,
            history=task.history if history is None else history,
            metadata=task.metadata if metadata is None else metadata,
        )

    def send_message(self, request: A2ASendMessageRequest) -> A2ATask:
        message = request.message
        if message.role is not A2ARole.ROLE_USER:
            raise A2AProtocolError("SendMessage requires ROLE_USER input")

        if message.task_id is not None:
            task = self.store.get(message.task_id)
            if task.status.state in TERMINAL_A2A_STATES:
                raise A2AProtocolError(
                    f"cannot continue terminal task: {task.status.state.value}"
                )
            if message.context_id is not None and message.context_id != task.context_id:
                raise A2AProtocolError("message contextId does not match task contextId")
            normalized = A2AMessage(
                message_id=message.message_id,
                role=message.role,
                parts=message.parts,
                context_id=task.context_id,
                task_id=task.id,
                metadata=message.metadata,
                extensions=message.extensions,
                reference_task_ids=message.reference_task_ids,
            )
            task = A2ATask(
                id=task.id,
                context_id=task.context_id,
                status=A2ATaskStatus(A2ATaskState.TASK_STATE_SUBMITTED),
                artifacts=task.artifacts,
                history=task.history + (normalized,),
                metadata={**task.metadata, **request.metadata},
            )
            self.store.put(task)
        else:
            task_id = self._new_id("task")
            context_id = message.context_id or self._new_id("ctx")
            normalized = A2AMessage(
                message_id=message.message_id,
                role=message.role,
                parts=message.parts,
                context_id=context_id,
                task_id=task_id,
                metadata=message.metadata,
                extensions=message.extensions,
                reference_task_ids=message.reference_task_ids,
            )
            task = A2ATask(
                id=task_id,
                context_id=context_id,
                status=A2ATaskStatus(A2ATaskState.TASK_STATE_SUBMITTED),
                history=(normalized,),
                metadata=dict(request.metadata),
            )
            self.store.put(task, create_only=True)

        if request.configuration.return_immediately:
            return self._history_limited(task, request.configuration.history_length)
        processed = self.process_task(task.id)
        return self._history_limited(
            processed, request.configuration.history_length
        )

    def process_task(self, task_id: str) -> A2ATask:
        task = self.store.get(task_id)
        if task.status.state in TERMINAL_A2A_STATES:
            return task
        if task.status.state not in {
            A2ATaskState.TASK_STATE_SUBMITTED,
            *INTERRUPTED_A2A_STATES,
        }:
            raise A2AProtocolError(
                f"task cannot be processed from state: {task.status.state.value}"
            )
        if not task.history:
            raise A2AProtocolError("task has no input message")

        working = self._with_status(task, A2ATaskState.TASK_STATE_WORKING)
        self.store.put(working)
        latest_user = next(
            (
                message
                for message in reversed(working.history)
                if message.role is A2ARole.ROLE_USER
            ),
            None,
        )
        if latest_user is None:
            raise A2AProtocolError("task has no ROLE_USER message")

        if self.handler is None:
            response = self._agent_message(
                "Task accepted by the educational A2A runtime.",
                task_id=working.id,
                context_id=working.context_id,
            )
            result = A2AExecutionResult(message=response)
        else:
            try:
                result = self.handler(working, latest_user)
            except Exception as exc:
                response = self._agent_message(
                    f"handler failed: {type(exc).__name__}: {exc}",
                    task_id=working.id,
                    context_id=working.context_id,
                )
                failed = self._with_status(
                    working,
                    A2ATaskState.TASK_STATE_FAILED,
                    message=response,
                    history=working.history + (response,),
                )
                self.store.put(failed)
                return failed

        if result.state in {
            A2ATaskState.TASK_STATE_UNSPECIFIED,
            A2ATaskState.TASK_STATE_SUBMITTED,
            A2ATaskState.TASK_STATE_WORKING,
            A2ATaskState.TASK_STATE_CANCELED,
        }:
            raise A2AProtocolError(
                "handler result must be COMPLETED/FAILED/REJECTED/INPUT_REQUIRED/AUTH_REQUIRED"
            )

        response = result.message
        if response is not None:
            if response.role is not A2ARole.ROLE_AGENT:
                raise A2AProtocolError("handler response message must use ROLE_AGENT")
            if response.task_id not in {None, working.id}:
                raise A2AProtocolError("handler response taskId mismatch")
            if response.context_id not in {None, working.context_id}:
                raise A2AProtocolError("handler response contextId mismatch")
            response = A2AMessage(
                message_id=response.message_id,
                role=response.role,
                parts=response.parts,
                context_id=working.context_id,
                task_id=working.id,
                metadata=response.metadata,
                extensions=response.extensions,
                reference_task_ids=response.reference_task_ids,
            )
        history = working.history + (() if response is None else (response,))
        completed = A2ATask(
            id=working.id,
            context_id=working.context_id,
            status=A2ATaskStatus(result.state, response),
            artifacts=working.artifacts + tuple(result.artifacts),
            history=history,
            metadata={**working.metadata, **result.metadata},
        )
        self.store.put(completed)
        return completed

    @staticmethod
    def _history_limited(task: A2ATask, length: int | None) -> A2ATask:
        if length is None:
            return task
        history = () if length == 0 else task.history[-length:]
        return A2ATask(
            id=task.id,
            context_id=task.context_id,
            status=task.status,
            artifacts=task.artifacts,
            history=history,
            metadata=task.metadata,
        )

    def get_task(self, task_id: str, *, history_length: int | None = None) -> A2ATask:
        if history_length is not None and history_length < 0:
            raise A2AProtocolError("history_length cannot be negative")
        return self._history_limited(self.store.get(task_id), history_length)

    def list_tasks(
        self,
        *,
        context_id: str | None = None,
        states: set[A2ATaskState] | None = None,
    ) -> tuple[A2ATask, ...]:
        return self.store.list(context_id=context_id, states=states)

    def cancel_task(self, task_id: str) -> A2ATask:
        task = self.store.get(task_id)
        if task.status.state in TERMINAL_A2A_STATES:
            raise A2AProtocolError(
                f"cannot cancel terminal task: {task.status.state.value}"
            )
        message = self._agent_message(
            "Task canceled.", task_id=task.id, context_id=task.context_id
        )
        cancelled = self._with_status(
            task,
            A2ATaskState.TASK_STATE_CANCELED,
            message=message,
            history=task.history + (message,),
        )
        self.store.put(cancelled)
        return cancelled

    def handle_operation(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Teaching dispatcher using normative A2A v1 RPC operation names."""

        if method == "GetAgentCard":
            return self.card.to_dict()
        if method == "SendMessage":
            message_raw = params.get("message")
            if not isinstance(message_raw, dict):
                raise A2AProtocolError("SendMessage.message must be an object")
            config_raw = params.get("configuration") or {}
            if not isinstance(config_raw, dict):
                raise A2AProtocolError("SendMessage.configuration must be an object")
            request = A2ASendMessageRequest(
                message=A2AMessage.from_dict(message_raw),
                configuration=A2ASendMessageConfiguration(
                    accepted_output_modes=tuple(
                        str(item) for item in config_raw.get("acceptedOutputModes", [])
                    ),
                    history_length=(
                        None
                        if config_raw.get("historyLength") is None
                        else int(config_raw["historyLength"])
                    ),
                    return_immediately=bool(config_raw.get("returnImmediately", False)),
                ),
                tenant=None if params.get("tenant") is None else str(params["tenant"]),
                metadata=dict(params.get("metadata") or {}),
            )
            return self.send_message(request).to_dict()
        if method == "GetTask":
            task_id = str(params.get("id", ""))
            if not task_id:
                raise A2AProtocolError("GetTask.id is required")
            history_length = params.get("historyLength")
            return self.get_task(
                task_id,
                history_length=None if history_length is None else int(history_length),
            ).to_dict()
        if method == "ListTasks":
            raw_states = params.get("states")
            states = None
            if raw_states is not None:
                if not isinstance(raw_states, list):
                    raise A2AProtocolError("ListTasks.states must be an array")
                states = {A2ATaskState(str(value)) for value in raw_states}
            return {
                "tasks": [
                    task.to_dict()
                    for task in self.list_tasks(
                        context_id=(
                            None
                            if params.get("contextId") is None
                            else str(params["contextId"])
                        ),
                        states=states,
                    )
                ]
            }
        if method == "CancelTask":
            task_id = str(params.get("id", ""))
            if not task_id:
                raise A2AProtocolError("CancelTask.id is required")
            return self.cancel_task(task_id).to_dict()
        raise A2AProtocolError(f"unsupported educational A2A operation: {method}")

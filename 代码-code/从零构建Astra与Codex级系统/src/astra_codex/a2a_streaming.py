"""A2A v1 streaming response objects and deterministic send-stream generator.

Primary source:
https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

The official ``StreamResponse`` oneof can carry ``task``, ``message``,
``status_update`` or ``artifact_update``. This module models those exact four
payload families and provides a small generator used by the reference SSE
transport. It does not claim that a synchronous handler becomes magically
incremental: the reference generator emits the durable submitted Task first,
then final artifact/status deltas after ``process_task`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .a2a import (
    A2AArtifact,
    A2AMessage,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATask,
    A2ATaskStatus,
)


@dataclass(frozen=True, slots=True)
class A2ATaskStatusUpdateEvent:
    task_id: str
    context_id: str
    status: A2ATaskStatus
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "contextId": self.context_id,
            "status": self.status.to_dict(),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2ATaskStatusUpdateEvent":
        return cls(
            task_id=str(payload["taskId"]),
            context_id=str(payload["contextId"]),
            status=A2ATaskStatus.from_dict(dict(payload["status"])),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class A2ATaskArtifactUpdateEvent:
    task_id: str
    context_id: str
    artifact: A2AArtifact
    append: bool = False
    last_chunk: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "contextId": self.context_id,
            "artifact": self.artifact.to_dict(),
            "append": self.append,
            "lastChunk": self.last_chunk,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2ATaskArtifactUpdateEvent":
        return cls(
            task_id=str(payload["taskId"]),
            context_id=str(payload["contextId"]),
            artifact=A2AArtifact.from_dict(dict(payload["artifact"])),
            append=bool(payload.get("append", False)),
            last_chunk=bool(payload.get("lastChunk", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class A2AStreamResponse:
    task: A2ATask | None = None
    message: A2AMessage | None = None
    status_update: A2ATaskStatusUpdateEvent | None = None
    artifact_update: A2ATaskArtifactUpdateEvent | None = None

    def __post_init__(self) -> None:
        populated = sum(
            value is not None
            for value in (
                self.task,
                self.message,
                self.status_update,
                self.artifact_update,
            )
        )
        if populated != 1:
            raise ValueError("A2A StreamResponse requires exactly one payload")

    def to_dict(self) -> dict[str, Any]:
        if self.task is not None:
            return {"task": self.task.to_dict()}
        if self.message is not None:
            return {"message": self.message.to_dict()}
        if self.status_update is not None:
            return {"statusUpdate": self.status_update.to_dict()}
        assert self.artifact_update is not None
        return {"artifactUpdate": self.artifact_update.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "A2AStreamResponse":
        if not isinstance(payload, dict):
            raise ValueError("A2A StreamResponse must be an object")
        keys = [
            key
            for key in ("task", "message", "statusUpdate", "artifactUpdate")
            if key in payload
        ]
        if len(keys) != 1:
            raise ValueError("A2A StreamResponse must contain exactly one payload key")
        key = keys[0]
        raw = payload[key]
        if not isinstance(raw, dict):
            raise ValueError(f"A2A StreamResponse.{key} must be an object")
        if key == "task":
            return cls(task=A2ATask.from_dict(raw))
        if key == "message":
            return cls(message=A2AMessage.from_dict(raw))
        if key == "statusUpdate":
            return cls(status_update=A2ATaskStatusUpdateEvent.from_dict(raw))
        return cls(artifact_update=A2ATaskArtifactUpdateEvent.from_dict(raw))


def parse_send_message_request(payload: dict[str, Any]) -> A2ASendMessageRequest:
    message_raw = payload.get("message")
    if not isinstance(message_raw, dict):
        raise ValueError("SendMessage.message must be an object")
    config_raw = payload.get("configuration") or {}
    if not isinstance(config_raw, dict):
        raise ValueError("SendMessage.configuration must be an object")
    return A2ASendMessageRequest(
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
            # Streaming owns the response channel. We deliberately create the
            # Task first and then process it so the first frame can be flushed.
            return_immediately=True,
        ),
        tenant=None if payload.get("tenant") is None else str(payload["tenant"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def stream_send_message(
    service: A2AService,
    payload: dict[str, Any],
) -> Iterator[A2AStreamResponse]:
    """Yield a source-aligned reference stream for one SendStreamingMessage.

    The first response is the durable SUBMITTED Task. The handler then runs.
    Newly created artifacts are emitted as ``artifactUpdate`` events, followed
    by a final ``statusUpdate``. A truly asynchronous production implementation
    would publish WORKING/intermediate deltas from the executor itself; this
    deterministic reference layer intentionally does not fabricate deltas the
    underlying handler never exposed.
    """

    request = parse_send_message_request(payload)
    submitted = service.send_message(request)
    yield A2AStreamResponse(task=submitted)

    before_artifacts = {artifact.artifact_id for artifact in submitted.artifacts}
    final = service.process_task(submitted.id)
    for artifact in final.artifacts:
        if artifact.artifact_id in before_artifacts:
            continue
        yield A2AStreamResponse(
            artifact_update=A2ATaskArtifactUpdateEvent(
                task_id=final.id,
                context_id=final.context_id,
                artifact=artifact,
                append=False,
                last_chunk=True,
            )
        )
    yield A2AStreamResponse(
        status_update=A2ATaskStatusUpdateEvent(
            task_id=final.id,
            context_id=final.context_id,
            status=final.status,
        )
    )

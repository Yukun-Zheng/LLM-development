"""Bridge the A2A task model into this project's DurableAgentRuntime.

This adapter is intentionally explicit: A2A describes a remote task lifecycle;
DurableAgentRuntime owns local Thread/WorkItem/Turn execution. The bridge maps
identity and evidence between the two instead of pretending they are the same
state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .a2a import (
    A2AArtifact,
    A2AExecutionResult,
    A2AMessage,
    A2APart,
    A2ARole,
    A2ATask,
    A2ATaskState,
)
from .durable import ThreadStatus
from .runtime import DurableAgentRuntime


def _text_from_message(message: A2AMessage) -> str:
    texts: list[str] = []
    unsupported: list[str] = []
    for part in message.parts:
        if part.text is not None:
            texts.append(part.text)
        elif part.data is not None:
            # Structured data is preserved textually for the teaching bridge so
            # the model does not silently lose it. Raw/url inputs require a
            # dedicated fetch/media policy and are rejected below.
            import json

            texts.append(json.dumps(part.data, ensure_ascii=False, sort_keys=True))
        elif part.raw is not None:
            unsupported.append("raw")
        elif part.url is not None:
            unsupported.append("url")
    if unsupported:
        kinds = ", ".join(sorted(set(unsupported)))
        raise ValueError(
            f"A2A runtime bridge does not yet resolve {kinds} input Parts; "
            "use text/data or add an explicit media fetch policy"
        )
    if not texts:
        raise ValueError("A2A message contains no bridgeable text/data content")
    return "\n".join(texts)


def _artifact_projection(runtime: DurableAgentRuntime, thread_id: str, task: A2ATask) -> tuple[A2AArtifact, ...]:
    existing = {artifact.artifact_id for artifact in task.artifacts}
    projected: list[A2AArtifact] = []
    for record in runtime.artifact_store.list_thread(thread_id):
        if record.artifact_id in existing:
            continue
        projected.append(
            A2AArtifact(
                artifact_id=record.artifact_id,
                name=record.kind,
                description="Artifact produced by the local durable runtime.",
                parts=(
                    A2APart(
                        data={
                            "artifactId": record.artifact_id,
                            "kind": record.kind,
                            "sha256": record.sha256,
                            "sizeBytes": record.size_bytes,
                            "metadata": record.metadata,
                        },
                        media_type="application/json",
                    ),
                ),
                metadata={
                    "runtimeThreadId": thread_id,
                    "sha256": record.sha256,
                },
            )
        )
    return tuple(projected)


@dataclass(slots=True)
class DurableRuntimeA2AHandler:
    """A2AService handler backed by ``DurableAgentRuntime``.

    A deterministic local runtime thread id is derived from the A2A task id on
    the first turn and then persisted in A2A task metadata. Continuations reuse
    the same runtime thread. Each input message becomes one durable WorkItem.

    This is a synchronous bridge. ``returnImmediately`` belongs to A2AService:
    when that flag is true, the A2A task remains SUBMITTED until a runner later
    calls ``process_task``.
    """

    runtime: DurableAgentRuntime
    worker_id: str = "a2a-local-worker"
    lease_seconds: float = 300.0

    def _runtime_thread_id(self, task: A2ATask) -> str:
        existing = task.metadata.get("runtimeThreadId")
        if existing is not None:
            if not isinstance(existing, str) or not existing:
                raise ValueError("A2A task runtimeThreadId metadata is invalid")
            self.runtime.thread_store.project(existing)
            return existing

        candidate = f"a2a_{task.id}"
        try:
            self.runtime.thread_store.project(candidate)
        except KeyError:
            self.runtime.create_thread(candidate)
        return candidate

    def __call__(self, task: A2ATask, message: A2AMessage) -> A2AExecutionResult:
        runtime_thread_id = self._runtime_thread_id(task)
        try:
            text = _text_from_message(message)
        except ValueError as exc:
            reply = A2AMessage(
                message_id=f"reply_{message.message_id}",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text=str(exc)),),
                task_id=task.id,
                context_id=task.context_id,
            )
            return A2AExecutionResult(
                state=A2ATaskState.TASK_STATE_REJECTED,
                message=reply,
                metadata={"runtimeThreadId": runtime_thread_id},
            )

        work_item_id = f"a2a_{task.id}_{message.message_id}"
        try:
            self.runtime.submit(
                runtime_thread_id,
                text,
                item_id=work_item_id,
            )
            execution = self.runtime.run_one(
                self.worker_id,
                lease_seconds=self.lease_seconds,
                allowed_thread_ids={runtime_thread_id},
            )
        except Exception as exc:
            reply = A2AMessage(
                message_id=f"reply_{message.message_id}",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text=f"runtime execution failed: {type(exc).__name__}: {exc}"),),
                task_id=task.id,
                context_id=task.context_id,
            )
            return A2AExecutionResult(
                state=A2ATaskState.TASK_STATE_FAILED,
                message=reply,
                metadata={
                    "runtimeThreadId": runtime_thread_id,
                    "runtimeWorkItemId": work_item_id,
                },
            )

        if execution is None:
            reply = A2AMessage(
                message_id=f"reply_{message.message_id}",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text="runtime did not claim the submitted work item"),),
                task_id=task.id,
                context_id=task.context_id,
            )
            return A2AExecutionResult(
                state=A2ATaskState.TASK_STATE_FAILED,
                message=reply,
                metadata={
                    "runtimeThreadId": runtime_thread_id,
                    "runtimeWorkItemId": work_item_id,
                },
            )

        completed = execution.status.startswith("completed")
        final_text = execution.final_answer or execution.status
        reply = A2AMessage(
            message_id=f"reply_{message.message_id}",
            role=A2ARole.ROLE_AGENT,
            parts=(A2APart(text=final_text),),
            task_id=task.id,
            context_id=task.context_id,
        )
        projection = self.runtime.thread_store.project(runtime_thread_id)
        if projection.status is ThreadStatus.FAILED:
            completed = False

        return A2AExecutionResult(
            state=(
                A2ATaskState.TASK_STATE_COMPLETED
                if completed
                else A2ATaskState.TASK_STATE_FAILED
            ),
            message=reply,
            artifacts=_artifact_projection(self.runtime, runtime_thread_id, task),
            metadata={
                "runtimeThreadId": runtime_thread_id,
                "runtimeWorkItemId": execution.work_item_id,
                "runtimeTurnId": execution.turn_id,
                "runtimeStatus": execution.status,
            },
        )

from __future__ import annotations

import threading

from astra_codex.a2a import (
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2AExecutionResult,
    A2AMessage,
    A2APart,
    A2ARole,
    A2ASendMessageRequest,
    A2AService,
    A2ATaskState,
    A2ATaskStore,
)
from astra_codex.a2a_sse import A2AStreamingHTTPClient, LocalA2AStreamingHTTPServer


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Streaming Agent",
        description="A2A streaming reference fixture.",
        supported_interfaces=(
            A2AAgentInterface("http://127.0.0.1:1", "HTTP+JSON", "1.0"),
        ),
        version="0.1.0",
        default_input_modes=("text/plain",),
        default_output_modes=("text/plain",),
        skills=(
            A2AAgentSkill(
                id="stream.task",
                name="Stream task",
                description="Emit A2A stream responses.",
                tags=("stream",),
            ),
        ),
        streaming=True,
    )


def _request(message_id: str = "msg_stream") -> A2ASendMessageRequest:
    return A2ASendMessageRequest(
        A2AMessage(
            message_id=message_id,
            role=A2ARole.ROLE_USER,
            parts=(A2APart(text="stream this task"),),
        )
    )


def test_first_submitted_task_is_flushed_before_slow_handler_completes(tmp_path) -> None:
    handler_started = threading.Event()
    release_handler = threading.Event()

    def handler(task, message):  # type: ignore[no-untyped-def]
        del message
        handler_started.set()
        assert release_handler.wait(timeout=2.0)
        return A2AExecutionResult(
            message=A2AMessage(
                message_id="msg_final",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text="stream completed"),),
                task_id=task.id,
                context_id=task.context_id,
            )
        )

    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store, handler=handler)
        with LocalA2AStreamingHTTPServer(service) as server:
            responses = A2AStreamingHTTPClient(server.base_url).send_streaming_message(
                _request()
            )
            first = next(responses)
            assert first.task is not None
            assert first.task.status.state is A2ATaskState.TASK_STATE_SUBMITTED
            assert handler_started.wait(timeout=1.0)

            release_handler.set()
            remaining = list(responses)
            assert len(remaining) == 1
            assert remaining[0].status_update is not None
            assert (
                remaining[0].status_update.status.state
                is A2ATaskState.TASK_STATE_COMPLETED
            )
            assert remaining[0].status_update.status.message is not None
            assert (
                remaining[0].status_update.status.message.parts[0].text
                == "stream completed"
            )


def test_stream_emits_artifact_delta_before_final_status(tmp_path) -> None:
    def handler(task, message):  # type: ignore[no-untyped-def]
        del message
        artifact = A2AArtifact(
            artifact_id="artifact_patch",
            name="patch",
            parts=(
                A2APart(
                    data={"diff": "+fixed"},
                    media_type="application/json",
                ),
            ),
        )
        reply = A2AMessage(
            message_id="msg_with_artifact",
            role=A2ARole.ROLE_AGENT,
            parts=(A2APart(text="artifact ready"),),
            task_id=task.id,
            context_id=task.context_id,
        )
        return A2AExecutionResult(message=reply, artifacts=(artifact,))

    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store, handler=handler)
        with LocalA2AStreamingHTTPServer(service) as server:
            responses = list(
                A2AStreamingHTTPClient(server.base_url).send_streaming_message(
                    _request("msg_artifact_stream")
                )
            )

            assert len(responses) == 3
            assert responses[0].task is not None
            update = responses[1].artifact_update
            assert update is not None
            assert update.artifact.artifact_id == "artifact_patch"
            assert update.append is False
            assert update.last_chunk is True
            assert responses[2].status_update is not None
            assert (
                responses[2].status_update.status.state
                is A2ATaskState.TASK_STATE_COMPLETED
            )

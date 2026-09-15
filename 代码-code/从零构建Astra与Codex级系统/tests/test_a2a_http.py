from __future__ import annotations

import pytest

from astra_codex.a2a import (
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AExecutionResult,
    A2AMessage,
    A2APart,
    A2ARole,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATaskState,
    A2ATaskStore,
)
from astra_codex.a2a_http import A2AHTTPClient, A2AHTTPError, LocalA2AHTTPServer


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Repository Agent",
        description="A2A HTTP teaching fixture.",
        supported_interfaces=(
            A2AAgentInterface(
                "http://127.0.0.1:1",
                "HTTP+JSON",
                "1.0",
            ),
        ),
        version="0.1.0",
        default_input_modes=("text/plain",),
        default_output_modes=("text/plain",),
        skills=(
            A2AAgentSkill(
                id="repo.inspect",
                name="Inspect repository",
                description="Inspect a repository.",
                tags=("code",),
            ),
        ),
    )


def _message(text: str, message_id: str) -> A2AMessage:
    return A2AMessage(
        message_id=message_id,
        role=A2ARole.ROLE_USER,
        parts=(A2APart(text=text),),
    )


def test_well_known_card_and_sync_send_over_real_http(tmp_path) -> None:
    def handler(task, message):  # type: ignore[no-untyped-def]
        text = message.parts[0].text
        return A2AExecutionResult(
            message=A2AMessage(
                message_id="msg_reply",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text=f"handled: {text}"),),
                task_id=task.id,
                context_id=task.context_id,
            )
        )

    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store, handler=handler)
        with LocalA2AHTTPServer(service) as server:
            client = A2AHTTPClient(server.base_url)
            card = client.get_agent_card()
            assert card.name == "Repository Agent"
            assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"
            assert card.supported_interfaces[0].protocol_version == "1.0"

            completed = client.send_message(
                A2ASendMessageRequest(_message("inspect", "msg_http_sync"))
            )
            assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert completed.status.message is not None
            assert completed.status.message.parts[0].text == "handled: inspect"
            assert completed.history[-1].role is A2ARole.ROLE_AGENT


def test_return_immediately_get_list_history_and_cancel_over_http(tmp_path) -> None:
    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store)
        with LocalA2AHTTPServer(service) as server:
            client = A2AHTTPClient(server.base_url)
            submitted = client.send_message(
                A2ASendMessageRequest(
                    _message("long task", "msg_http_async"),
                    A2ASendMessageConfiguration(return_immediately=True),
                )
            )
            assert submitted.status.state is A2ATaskState.TASK_STATE_SUBMITTED

            # The server thread owns a separate SQLite connection, but the task
            # remains durable and visible from both clients and the original store.
            persisted = store.get(submitted.id)
            assert persisted.context_id == submitted.context_id

            no_history = client.get_task(submitted.id, history_length=0)
            assert no_history.history == ()

            listed = client.list_tasks(context_id=submitted.context_id)
            assert [task.id for task in listed] == [submitted.id]
            submitted_only = client.list_tasks(
                states={A2ATaskState.TASK_STATE_SUBMITTED}
            )
            assert [task.id for task in submitted_only] == [submitted.id]

            cancelled = client.cancel_task(submitted.id)
            assert cancelled.status.state is A2ATaskState.TASK_STATE_CANCELED
            assert client.get_task(submitted.id).status.state is A2ATaskState.TASK_STATE_CANCELED


def test_http_binding_rejects_unimplemented_pagination_instead_of_faking_it(tmp_path) -> None:
    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store)
        with LocalA2AHTTPServer(service) as server:
            client = A2AHTTPClient(server.base_url)
            with pytest.raises(A2AHTTPError) as exc_info:
                client._request("GET", "/tasks?pageSize=10")
            assert exc_info.value.status == 501
            assert "pagination" in exc_info.value.message

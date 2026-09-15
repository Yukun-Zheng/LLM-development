from __future__ import annotations

import pytest

from astra_codex.a2a import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2AExecutionResult,
    A2AMessage,
    A2APart,
    A2AProtocolError,
    A2ARole,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATaskState,
    A2ATaskStore,
)


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Repository Agent",
        description="A teaching A2A agent that inspects repository tasks.",
        supported_interfaces=(
            A2AAgentInterface(
                "http://127.0.0.1:8000/a2a",
                "HTTP+JSON",
                A2A_PROTOCOL_VERSION,
            ),
        ),
        version="0.1.0",
        default_input_modes=("text/plain",),
        default_output_modes=("text/plain", "application/json"),
        skills=(
            A2AAgentSkill(
                id="repo.inspect",
                name="Repository inspection",
                description="Inspect a repository and return evidence.",
                tags=("code", "repository"),
            ),
        ),
    )


def _user_message(text: str, *, message_id: str = "msg_user", **kwargs) -> A2AMessage:
    return A2AMessage(
        message_id=message_id,
        role=A2ARole.ROLE_USER,
        parts=(A2APart(text=text),),
        **kwargs,
    )


def test_v1_task_state_spellings_match_normative_enum() -> None:
    assert [state.value for state in A2ATaskState] == [
        "TASK_STATE_UNSPECIFIED",
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_AUTH_REQUIRED",
    ]


def test_v1_part_json_uses_oneof_member_not_legacy_kind() -> None:
    part = A2APart(text="hello", media_type="text/plain")
    payload = part.to_dict()
    assert payload == {"text": "hello", "mediaType": "text/plain"}
    assert "kind" not in payload
    assert A2APart.from_dict(payload) == part

    raw = A2APart(raw=b"abc", filename="x.bin")
    restored = A2APart.from_dict(raw.to_dict())
    assert restored.raw == b"abc"

    with pytest.raises(A2AProtocolError, match="exactly one"):
        A2APart(text="x", data={"also": "set"})


def test_agent_card_places_protocol_version_on_supported_interface() -> None:
    payload = _card().to_dict()
    assert "protocolVersion" not in payload
    assert payload["supportedInterfaces"][0] == {
        "url": "http://127.0.0.1:8000/a2a",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    }
    assert payload["skills"][0]["id"] == "repo.inspect"


def test_return_immediately_persists_submitted_task_then_manual_processing(tmp_path) -> None:
    path = tmp_path / "a2a.sqlite"

    def handler(task, message):  # type: ignore[no-untyped-def]
        response = A2AMessage(
            message_id="msg_agent",
            role=A2ARole.ROLE_AGENT,
            parts=(A2APart(text="inspection complete"),),
        )
        artifact = A2AArtifact(
            artifact_id="artifact_report",
            name="report",
            parts=(A2APart(data={"files": 7}, media_type="application/json"),),
        )
        return A2AExecutionResult(
            state=A2ATaskState.TASK_STATE_COMPLETED,
            message=response,
            artifacts=(artifact,),
            metadata={"handler": "fixture"},
        )

    with A2ATaskStore(path) as store:
        service = A2AService(_card(), store, handler=handler)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _user_message("inspect repository"),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )
        assert submitted.status.state is A2ATaskState.TASK_STATE_SUBMITTED
        assert submitted.history[0].task_id == submitted.id
        assert submitted.history[0].context_id == submitted.context_id
        task_id = submitted.id

    with A2ATaskStore(path) as reopened:
        persisted = reopened.get(task_id)
        assert persisted.status.state is A2ATaskState.TASK_STATE_SUBMITTED
        service = A2AService(_card(), reopened, handler=handler)
        completed = service.process_task(task_id)
        assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
        assert completed.status.message is not None
        assert completed.status.message.task_id == task_id
        assert completed.artifacts[0].artifact_id == "artifact_report"
        assert completed.metadata["handler"] == "fixture"
        assert len(completed.history) == 2


def test_send_message_sync_get_list_and_history_limit(tmp_path) -> None:
    def handler(task, message):  # type: ignore[no-untyped-def]
        del message
        return A2AExecutionResult(
            message=A2AMessage(
                message_id="msg_reply",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text="done"),),
                task_id=task.id,
                context_id=task.context_id,
            )
        )

    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store, handler=handler)
        result = service.send_message(A2ASendMessageRequest(_user_message("do it")))
        assert result.status.state is A2ATaskState.TASK_STATE_COMPLETED

        fetched = service.get_task(result.id, history_length=1)
        assert len(fetched.history) == 1
        assert fetched.history[0].role is A2ARole.ROLE_AGENT

        listed = service.list_tasks(context_id=result.context_id)
        assert [task.id for task in listed] == [result.id]
        completed_only = service.list_tasks(
            states={A2ATaskState.TASK_STATE_COMPLETED}
        )
        assert [task.id for task in completed_only] == [result.id]


def test_continuation_requires_matching_context_and_nonterminal_task(tmp_path) -> None:
    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _user_message("first"),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )

        wrong_context = _user_message(
            "continue",
            message_id="msg_continue",
            task_id=submitted.id,
            context_id="ctx_wrong",
        )
        with pytest.raises(A2AProtocolError, match="contextId"):
            service.send_message(
                A2ASendMessageRequest(
                    wrong_context,
                    A2ASendMessageConfiguration(return_immediately=True),
                )
            )

        completed = service.process_task(submitted.id)
        assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
        with pytest.raises(A2AProtocolError, match="terminal task"):
            service.send_message(
                A2ASendMessageRequest(
                    _user_message(
                        "too late",
                        message_id="msg_late",
                        task_id=submitted.id,
                        context_id=submitted.context_id,
                    ),
                    A2ASendMessageConfiguration(return_immediately=True),
                )
            )


def test_cancel_task_only_before_terminal_completion(tmp_path) -> None:
    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _user_message("long task"),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )
        cancelled = service.cancel_task(submitted.id)
        assert cancelled.status.state is A2ATaskState.TASK_STATE_CANCELED
        assert cancelled.status.message is not None
        assert cancelled.status.message.role is A2ARole.ROLE_AGENT
        with pytest.raises(A2AProtocolError, match="terminal task"):
            service.cancel_task(submitted.id)


def test_operation_dispatch_uses_v1_rpc_names_and_exact_wire_shapes(tmp_path) -> None:
    with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
        service = A2AService(_card(), store)
        card = service.handle_operation("GetAgentCard", {})
        assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"

        sent = service.handle_operation(
            "SendMessage",
            {
                "message": _user_message("inspect", message_id="msg_wire").to_dict(),
                "configuration": {"returnImmediately": True, "historyLength": 0},
            },
        )
        assert sent["status"]["state"] == "TASK_STATE_SUBMITTED"
        assert "history" not in sent
        task_id = sent["id"]

        fetched = service.handle_operation("GetTask", {"id": task_id})
        assert fetched["id"] == task_id
        assert fetched["history"][0]["parts"][0] == {"text": "inspect"}

        listed = service.handle_operation(
            "ListTasks", {"states": ["TASK_STATE_SUBMITTED"]}
        )
        assert [task["id"] for task in listed["tasks"]] == [task_id]

        cancelled = service.handle_operation("CancelTask", {"id": task_id})
        assert cancelled["status"]["state"] == "TASK_STATE_CANCELED"

        with pytest.raises(A2AProtocolError, match="unsupported"):
            service.handle_operation("message/send", {})

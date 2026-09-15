from __future__ import annotations

import threading

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
    A2ATaskState,
    A2ATaskStore,
)
from astra_codex.a2a_http import A2AHTTPError
from astra_codex.a2a_subscription import (
    A2ASubscriptionHTTPClient,
    A2ATaskUpdateJournal,
    JournaledA2AService,
    LocalA2ASubscriptionHTTPServer,
)


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Subscription Agent",
        description="Durable A2A SubscribeToTask fixture.",
        supported_interfaces=(
            A2AAgentInterface("http://127.0.0.1:1", "HTTP+JSON", "1.0"),
        ),
        version="0.1.0",
        default_input_modes=("text/plain",),
        default_output_modes=("text/plain",),
        skills=(
            A2AAgentSkill(
                id="subscribe.task",
                name="Subscribe task",
                description="Exercise durable task updates.",
                tags=("subscribe",),
            ),
        ),
        streaming=True,
    )


def _message() -> A2AMessage:
    return A2AMessage(
        message_id="msg_subscribe",
        role=A2ARole.ROLE_USER,
        parts=(A2APart(text="finish after subscription begins"),),
    )


def test_subscribe_replays_submitted_then_live_terminal_update(tmp_path) -> None:
    task_db = tmp_path / "tasks.sqlite"
    update_db = tmp_path / "updates.sqlite"
    handler_started = threading.Event()

    def handler(task, message):  # type: ignore[no-untyped-def]
        del message
        handler_started.set()
        return A2AExecutionResult(
            message=A2AMessage(
                message_id="msg_subscribe_done",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text="subscription complete"),),
                task_id=task.id,
                context_id=task.context_id,
            )
        )

    with A2ATaskStore(task_db) as store, A2ATaskUpdateJournal(update_db) as journal:
        service = JournaledA2AService(_card(), store, journal, handler=handler)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _message(),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )
        assert submitted.status.state is A2ATaskState.TASK_STATE_SUBMITTED
        assert journal.latest_id(submitted.id) > 0

        with LocalA2ASubscriptionHTTPServer(task_db, update_db) as server:
            updates = A2ASubscriptionHTTPClient(server.base_url).subscribe(submitted.id)
            first_id, first = next(updates)
            assert first.task is not None
            assert first.task.status.state is A2ATaskState.TASK_STATE_SUBMITTED
            assert first_id > 0

            completed = service.process_task(submitted.id)
            assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert handler_started.is_set()

            final_id, final = next(updates)
            assert final_id > first_id
            assert final.task is not None
            assert final.task.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert final.task.status.message is not None
            assert final.task.status.message.parts[0].text == "subscription complete"
            with pytest.raises(StopIteration):
                next(updates)


def test_subscription_last_event_id_replays_only_updates_after_disconnect(tmp_path) -> None:
    task_db = tmp_path / "tasks.sqlite"
    update_db = tmp_path / "updates.sqlite"

    def handler(task, message):  # type: ignore[no-untyped-def]
        del message
        return A2AExecutionResult(
            message=A2AMessage(
                message_id="msg_reconnect_done",
                role=A2ARole.ROLE_AGENT,
                parts=(A2APart(text="completed while disconnected"),),
                task_id=task.id,
                context_id=task.context_id,
            )
        )

    with A2ATaskStore(task_db) as store, A2ATaskUpdateJournal(update_db) as journal:
        service = JournaledA2AService(_card(), store, journal, handler=handler)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _message(),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )
        first_id = journal.latest_id(submitted.id)
        assert first_id > 0

        # Simulate the subscriber disappearing after observing SUBMITTED.
        completed = service.process_task(submitted.id)
        assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
        terminal_id = journal.latest_id(submitted.id)
        assert terminal_id > first_id

        with LocalA2ASubscriptionHTTPServer(task_db, update_db) as server:
            resumed = A2ASubscriptionHTTPClient(server.base_url).subscribe(
                submitted.id,
                last_event_id=first_id,
            )
            replayed_id, replayed = next(resumed)
            assert replayed_id == terminal_id
            assert replayed.task is not None
            assert replayed.task.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert replayed.task.status.message is not None
            assert (
                replayed.task.status.message.parts[0].text
                == "completed while disconnected"
            )
            with pytest.raises(StopIteration):
                next(resumed)


def test_journal_deduplicates_identical_consecutive_task_snapshots(tmp_path) -> None:
    task_db = tmp_path / "tasks.sqlite"
    update_db = tmp_path / "updates.sqlite"
    with A2ATaskStore(task_db) as store, A2ATaskUpdateJournal(update_db) as journal:
        service = JournaledA2AService(_card(), store, journal)
        submitted = service.send_message(
            A2ASendMessageRequest(
                _message(),
                A2ASendMessageConfiguration(return_immediately=True),
            )
        )
        first_id = journal.latest_id(submitted.id)
        duplicate_id = journal.append(submitted)
        assert duplicate_id == first_id
        assert len(journal.read(submitted.id)) == 1


def test_subscribe_to_already_terminal_task_returns_unsupported_reference_error(tmp_path) -> None:
    task_db = tmp_path / "tasks.sqlite"
    update_db = tmp_path / "updates.sqlite"
    with A2ATaskStore(task_db) as store, A2ATaskUpdateJournal(update_db) as journal:
        service = JournaledA2AService(_card(), store, journal)
        terminal = service.send_message(A2ASendMessageRequest(_message()))
        assert terminal.status.state is A2ATaskState.TASK_STATE_COMPLETED

        with LocalA2ASubscriptionHTTPServer(task_db, update_db) as server:
            with pytest.raises(A2AHTTPError) as exc_info:
                A2ASubscriptionHTTPClient(server.base_url).subscribe(terminal.id)
            assert exc_info.value.status == 409
            assert "terminal" in exc_info.value.message

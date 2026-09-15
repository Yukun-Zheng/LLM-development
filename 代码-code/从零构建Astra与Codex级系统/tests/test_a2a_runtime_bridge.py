from __future__ import annotations

from astra_codex.a2a import (
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AMessage,
    A2APart,
    A2ARole,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATaskState,
    A2ATaskStore,
)
from astra_codex.a2a_runtime_bridge import DurableRuntimeA2AHandler
from astra_codex.agent import ScriptedBackend
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_queue import WorkStatus
from astra_codex.tools import ToolRegistry


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Durable Repository Agent",
        description="A2A-to-DurableAgentRuntime teaching bridge.",
        supported_interfaces=(
            A2AAgentInterface("http://127.0.0.1:1", "HTTP+JSON", "1.0"),
        ),
        version="0.1.0",
        default_input_modes=("text/plain", "application/json"),
        default_output_modes=("text/plain", "application/json"),
        skills=(
            A2AAgentSkill(
                id="runtime.execute",
                name="Execute durable task",
                description="Execute one task in the local durable runtime.",
                tags=("durable", "agent"),
            ),
        ),
    )


def _message(part: A2APart, message_id: str = "msg_bridge") -> A2AMessage:
    return A2AMessage(
        message_id=message_id,
        role=A2ARole.ROLE_USER,
        parts=(part,),
    )


def test_a2a_task_executes_real_durable_thread_work_and_turn(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "runtime",
        ScriptedBackend(["durable answer"]),
        ToolRegistry([]),
    ) as runtime:
        handler = DurableRuntimeA2AHandler(runtime, worker_id="worker_a2a")
        with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
            service = A2AService(_card(), store, handler=handler)
            task = service.send_message(
                A2ASendMessageRequest(_message(A2APart(text="do durable work")))
            )

            assert task.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert task.status.message is not None
            assert task.status.message.parts[0].text == "durable answer"

            runtime_thread_id = task.metadata["runtimeThreadId"]
            assert runtime_thread_id == f"a2a_{task.id}"
            projection = runtime.thread_store.project(runtime_thread_id)
            assert projection.submissions == ("do durable work",)
            assert projection.active_turn_id is None

            work_item_id = task.metadata["runtimeWorkItemId"]
            work = runtime.work_queue.get(work_item_id)
            assert work.status is WorkStatus.COMPLETED
            assert task.metadata["runtimeTurnId"] is not None
            assert task.metadata["runtimeStatus"] == "completed"


def test_return_immediately_defers_runtime_execution_until_process_task(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "runtime",
        ScriptedBackend(["processed later"]),
        ToolRegistry([]),
    ) as runtime:
        handler = DurableRuntimeA2AHandler(runtime)
        with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
            service = A2AService(_card(), store, handler=handler)
            submitted = service.send_message(
                A2ASendMessageRequest(
                    _message(A2APart(text="defer me"), "msg_deferred"),
                    A2ASendMessageConfiguration(return_immediately=True),
                )
            )
            assert submitted.status.state is A2ATaskState.TASK_STATE_SUBMITTED
            assert "runtimeThreadId" not in submitted.metadata

            completed = service.process_task(submitted.id)
            assert completed.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert completed.status.message is not None
            assert completed.status.message.parts[0].text == "processed later"
            assert completed.metadata["runtimeThreadId"] == f"a2a_{submitted.id}"


def test_bridge_rejects_raw_media_without_silently_fetching_or_decoding_it(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "runtime",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        handler = DurableRuntimeA2AHandler(runtime)
        with A2ATaskStore(tmp_path / "a2a.sqlite") as store:
            service = A2AService(_card(), store, handler=handler)
            task = service.send_message(
                A2ASendMessageRequest(_message(A2APart(raw=b"binary"), "msg_raw"))
            )

            assert task.status.state is A2ATaskState.TASK_STATE_REJECTED
            assert task.status.message is not None
            assert "media fetch policy" in task.status.message.parts[0].text
            runtime_thread_id = task.metadata["runtimeThreadId"]
            assert runtime.thread_store.project(runtime_thread_id).submissions == ()

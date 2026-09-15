from __future__ import annotations

from astra_codex.a2a import (
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AMessage,
    A2APart,
    A2ARole,
    A2ASendMessageRequest,
    A2ATaskState,
)
from astra_codex.a2a_http import A2AHTTPClient
from astra_codex.a2a_runtime_http import LocalA2ADurableRuntimeHTTPServer
from astra_codex.agent import ScriptedBackend
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_queue import WorkStatus
from astra_codex.tools import ToolRegistry


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Remote Durable Agent",
        description="HTTP+JSON gateway into DurableAgentRuntime.",
        supported_interfaces=(
            A2AAgentInterface("http://127.0.0.1:1", "HTTP+JSON", "1.0"),
        ),
        version="0.1.0",
        default_input_modes=("text/plain",),
        default_output_modes=("text/plain",),
        skills=(
            A2AAgentSkill(
                id="durable.execute",
                name="Durable execute",
                description="Execute one durable agent turn.",
                tags=("a2a", "durable"),
            ),
        ),
    )


def test_real_http_a2a_message_enters_durable_thread_work_and_turn(tmp_path) -> None:
    state_dir = tmp_path / "runtime"
    with DurableAgentRuntime(
        state_dir,
        ScriptedBackend(["remote durable answer"]),
        ToolRegistry([]),
    ) as runtime:
        with LocalA2ADurableRuntimeHTTPServer(
            _card(),
            tmp_path / "a2a.sqlite",
            runtime,
            worker_id="remote_worker",
        ) as server:
            client = A2AHTTPClient(server.base_url)
            task = client.send_message(
                A2ASendMessageRequest(
                    A2AMessage(
                        message_id="msg_remote_runtime",
                        role=A2ARole.ROLE_USER,
                        parts=(A2APart(text="execute over the wire"),),
                    )
                )
            )

            assert task.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert task.status.message is not None
            assert task.status.message.parts[0].text == "remote durable answer"
            assert task.metadata["runtimeStatus"] == "completed"

            runtime_thread_id = task.metadata["runtimeThreadId"]
            work_item_id = task.metadata["runtimeWorkItemId"]

        # The HTTP server used its own thread-owned SQLite handles. After it is
        # closed, the original runtime can replay the same committed state.
        projection = runtime.thread_store.project(runtime_thread_id)
        assert projection.submissions == ("execute over the wire",)
        assert projection.active_turn_id is None
        assert runtime.work_queue.get(work_item_id).status is WorkStatus.COMPLETED


def test_remote_gateway_persists_a2a_task_across_server_restart(tmp_path) -> None:
    state_dir = tmp_path / "runtime"
    task_store = tmp_path / "a2a.sqlite"
    backend = ScriptedBackend(["persistent answer"])

    with DurableAgentRuntime(state_dir, backend, ToolRegistry([])) as runtime:
        with LocalA2ADurableRuntimeHTTPServer(_card(), task_store, runtime) as server:
            client = A2AHTTPClient(server.base_url)
            created = client.send_message(
                A2ASendMessageRequest(
                    A2AMessage(
                        message_id="msg_restart",
                        role=A2ARole.ROLE_USER,
                        parts=(A2APart(text="persist remote task"),),
                    )
                )
            )
            task_id = created.id

        with LocalA2ADurableRuntimeHTTPServer(_card(), task_store, runtime) as server:
            restored = A2AHTTPClient(server.base_url).get_task(task_id)
            assert restored.id == task_id
            assert restored.status.state is A2ATaskState.TASK_STATE_COMPLETED
            assert restored.metadata["runtimeThreadId"] == f"a2a_{task_id}"

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from astra_codex.agent import Message
from astra_codex.artifacts import ArtifactStore
from astra_codex.codex_harness import CodexHarness
from astra_codex.runtime_control import ControlledBackend, LeaseHeartbeat
from astra_codex.runtime_queue import DurableWorkQueue
from astra_codex.steering import DurableSteeringQueue, SteeringStatus
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry, ToolResult


@dataclass
class RecordingBackend:
    outputs: list[str]
    calls: list[list[Message]] = field(default_factory=list)
    index: int = 0

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        del tools
        self.calls.append(list(messages))
        output = self.outputs[self.index]
        self.index += 1
        return output


class SteeringTool:
    spec = ToolSpec(
        name="steer_now",
        description="Simulate a user steering message arriving while a tool runs.",
        parameters={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def __init__(self, queue: DurableSteeringQueue, thread_id: str) -> None:
        self.queue = queue
        self.thread_id = thread_id
        self.steering_id: str | None = None

    def run(self, arguments: dict[str, object]) -> ToolResult:
        del arguments
        self.steering_id = self.queue.submit(
            self.thread_id,
            "stop editing code; inspect the documentation first",
            steering_id="steer_mid_turn",
            now=2.0,
        )
        return ToolResult(True, "steering submitted")


def test_lease_heartbeat_prevents_premature_reclaim(tmp_path) -> None:
    with DurableWorkQueue(tmp_path / "work.sqlite") as queue:
        item_id = queue.enqueue("thr", "turn", {"content": "work"}, now=0.0)
        claimed = queue.claim("worker_a", lease_seconds=5.0, now=0.0)
        assert claimed is not None
        assert claimed.item_id == item_id

        heartbeat = LeaseHeartbeat(queue, item_id, "worker_a", lease_seconds=10.0)
        assert heartbeat.renew(now=4.0) == 14.0
        assert queue.get(item_id).lease_until == 14.0

        # Original lease would have expired at t=5, but the renewed lease keeps
        # ownership with worker_a until t=14.
        assert queue.claim("worker_b", now=6.0) is None

        reclaimed = queue.claim("worker_b", now=15.0)
        assert reclaimed is not None
        assert reclaimed.item_id == item_id
        assert reclaimed.lease_owner == "worker_b"


def test_live_steering_is_injected_into_next_model_step(tmp_path) -> None:
    with DurableSteeringQueue(tmp_path / "steering.sqlite") as steering:
        thread_id = "thr_live"
        raw_backend = RecordingBackend(
            [
                '{"tool":"steer_now","arguments":{}}',
                "done after reading the steering update",
            ]
        )
        tool = SteeringTool(steering, thread_id)
        controlled = ControlledBackend(
            raw_backend,
            steering_queue=steering,
            thread_id=thread_id,
        )
        result = CodexHarness(controlled, ToolRegistry([tool])).run_turn("work on the task")

        assert result.final_answer == "done after reading the steering update"
        assert len(raw_backend.calls) == 2
        second_call = raw_backend.calls[1]
        assert any(
            message.role == "user"
            and "inspect the documentation first" in message.content
            for message in second_call
        )
        assert tool.steering_id == "steer_mid_turn"
        stored = steering.get("steer_mid_turn")
        assert stored.status is SteeringStatus.CONSUMED
        assert steering.pending(thread_id) == ()


def test_controlled_backend_runs_heartbeat_before_sampling() -> None:
    calls: list[str] = []
    raw_backend = RecordingBackend(["done"])
    controlled = ControlledBackend(
        raw_backend,
        heartbeat=lambda: calls.append("beat"),
    )

    result = CodexHarness(controlled, ToolRegistry([])).run_turn("finish")

    assert result.final_answer == "done"
    assert calls == ["beat"]


def test_artifact_store_snapshots_bytes_with_checksum_and_thread_index(tmp_path) -> None:
    root = tmp_path / "artifacts"
    with ArtifactStore(root) as store:
        first = store.put_bytes(
            "thr_a",
            b"same bytes",
            kind="patch",
            metadata={"work_item_id": "work_1"},
            artifact_id="art_1",
            now=1.0,
        )
        second = store.put_bytes(
            "thr_a",
            b"same bytes",
            kind="patch",
            artifact_id="art_2",
            now=2.0,
        )

        first_record = store.get(first)
        second_record = store.get(second)
        assert first_record.path == second_record.path
        assert first_record.sha256 == second_record.sha256
        assert first_record.metadata["work_item_id"] == "work_1"
        assert store.read_bytes(first) == b"same bytes"
        assert [record.artifact_id for record in store.list_thread("thr_a")] == [
            "art_1",
            "art_2",
        ]


def test_artifact_store_detects_corruption(tmp_path) -> None:
    with ArtifactStore(tmp_path / "artifacts") as store:
        artifact_id = store.put_bytes("thr", b"trusted", kind="result")
        record = store.get(artifact_id)
        Path(record.path).write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            store.read_bytes(artifact_id)

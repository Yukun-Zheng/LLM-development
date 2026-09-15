from __future__ import annotations

import time
from dataclasses import dataclass

from astra_codex.agent import Message
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_control import BackgroundLeaseHeartbeat
from astra_codex.runtime_queue import DurableWorkQueue
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry


def test_background_heartbeat_keeps_lease_alive_without_foreground_sampling(tmp_path) -> None:
    path = tmp_path / "work.sqlite"
    with DurableWorkQueue(path) as queue:
        item_id = queue.enqueue("thr", "turn", {"content": "slow"})
        claimed = queue.claim("worker_a", lease_seconds=0.12)
        assert claimed is not None

        with BackgroundLeaseHeartbeat(
            str(path),
            item_id,
            "worker_a",
            lease_seconds=0.12,
            interval_seconds=0.03,
        ) as heartbeat:
            time.sleep(0.18)
            with DurableWorkQueue(path) as competitor:
                assert competitor.claim("worker_b") is None

        assert heartbeat.renewals >= 2
        assert heartbeat.last_error is None


@dataclass
class BlockingProbeBackend:
    queue_path: str
    competitor_claim: object | None = None

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        del messages, tools
        # Longer than the original lease. Without a background renewal thread,
        # worker_b could legally reclaim this work while model.generate blocks.
        time.sleep(0.18)
        with DurableWorkQueue(self.queue_path) as competitor:
            self.competitor_claim = competitor.claim("worker_b")
        return "done after slow model call"


def test_integrated_runtime_background_heartbeat_covers_one_long_model_call(tmp_path) -> None:
    state = tmp_path / "state"
    backend = BlockingProbeBackend(str(state / "work.sqlite"))

    with DurableAgentRuntime(state, backend, ToolRegistry([])) as runtime:
        thread_id = runtime.create_thread("thr_slow")
        runtime.submit(thread_id, "slow turn", item_id="work_slow")
        result = runtime.run_one("worker_a", lease_seconds=0.12)

        assert result is not None
        assert result.status == "completed"
        assert result.final_answer == "done after slow model call"
        assert backend.competitor_claim is None

        summaries = runtime.event_stream.read(
            thread_id=thread_id,
            topics={"work.background_heartbeat_summary"},
        )
        assert len(summaries) == 1
        assert summaries[0].payload["renewals"] >= 2
        assert summaries[0].payload["error"] is None

from __future__ import annotations

import time
from dataclasses import dataclass

from astra_codex.agent import Message
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_control import BackgroundLeaseHeartbeat
from astra_codex.runtime_queue import DurableWorkQueue
from astra_codex.structured import ToolSpec
from astra_codex.tools import ToolRegistry


LEASE_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 0.05
BLOCKING_SECONDS = 0.65


def test_background_heartbeat_keeps_lease_alive_without_foreground_sampling(tmp_path) -> None:
    path = tmp_path / "work.sqlite"
    with DurableWorkQueue(path) as queue:
        item_id = queue.enqueue("thr", "turn", {"content": "slow"})
        claimed = queue.claim("worker_a", lease_seconds=LEASE_SECONDS)
        assert claimed is not None

        with BackgroundLeaseHeartbeat(
            str(path),
            item_id,
            "worker_a",
            lease_seconds=LEASE_SECONDS,
            interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        ) as heartbeat:
            # The protected section exceeds the original lease. The margin is
            # intentionally much larger than a scheduler quantum so this tests
            # lease semantics rather than CI timing luck.
            time.sleep(BLOCKING_SECONDS)
            with DurableWorkQueue(path) as competitor:
                assert competitor.claim("worker_b") is None

        # start() performs one synchronous renewal and waits for one successful
        # renewal by the background thread before returning.
        assert heartbeat.renewals >= 3
        assert heartbeat.last_error is None


@dataclass
class BlockingProbeBackend:
    queue_path: str
    competitor_claim: object | None = None

    def generate(self, messages: list[Message], tools: list[ToolSpec]) -> str:
        del messages, tools
        # Longer than the original lease. Without a background renewal thread,
        # worker_b could legally reclaim this work while model.generate blocks.
        time.sleep(BLOCKING_SECONDS)
        with DurableWorkQueue(self.queue_path) as competitor:
            self.competitor_claim = competitor.claim("worker_b")
        return "done after slow model call"


def test_integrated_runtime_background_heartbeat_covers_one_long_model_call(tmp_path) -> None:
    state = tmp_path / "state"
    backend = BlockingProbeBackend(str(state / "work.sqlite"))

    with DurableAgentRuntime(state, backend, ToolRegistry([])) as runtime:
        thread_id = runtime.create_thread("thr_slow")
        runtime.submit(thread_id, "slow turn", item_id="work_slow")
        result = runtime.run_one("worker_a", lease_seconds=LEASE_SECONDS)

        assert result is not None
        assert result.status == "completed"
        assert result.final_answer == "done after slow model call"
        assert backend.competitor_claim is None

        summaries = runtime.event_stream.read(
            thread_id=thread_id,
            topics={"work.background_heartbeat_summary"},
        )
        assert len(summaries) == 1
        assert summaries[0].payload["renewals"] >= 3
        assert summaries[0].payload["error"] is None

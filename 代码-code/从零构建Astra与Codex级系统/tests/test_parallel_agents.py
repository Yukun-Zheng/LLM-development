from __future__ import annotations

import threading
import time
from collections import defaultdict

from astra_codex.agent_graph import AgentStatus, MessageStatus, PersistentAgentGraph
from astra_codex.parallel_agents import (
    ParallelAgentCoordinator,
    ParallelTask,
    ParallelTaskResult,
)


def _graph_with_workers(tmp_path, count: int = 2):
    graph = PersistentAgentGraph(tmp_path / "agents.sqlite")
    root = graph.create_agent("thr_parallel", "coordinator", agent_id="coordinator")
    workers = [
        graph.create_agent(
            "thr_parallel",
            "worker",
            agent_id=f"worker_{index}",
            parent_agent_id=root,
        )
        for index in range(count)
    ]
    return graph, root, workers


def test_two_worker_tasks_really_overlap_and_mailbox_results_persist(tmp_path) -> None:
    graph, root, workers = _graph_with_workers(tmp_path, 2)
    barrier = threading.Barrier(2, timeout=2.0)
    entered: list[str] = []
    lock = threading.Lock()

    def worker(agent, message):  # type: ignore[no-untyped-def]
        with lock:
            entered.append(agent.agent_id)
        barrier.wait()
        task_id = str(message.payload["taskId"])
        return ParallelTaskResult(
            task_id,
            agent.agent_id,
            True,
            f"finished {task_id}",
            {"worker": agent.agent_id},
        )

    try:
        coordinator = ParallelAgentCoordinator(graph, root, workers)
        results = coordinator.run(
            [
                ParallelTask("a", {"goal": "A"}),
                ParallelTask("b", {"goal": "B"}),
            ],
            worker,
        )

        assert {result.agent_id for result in results} == set(workers)
        assert [result.task_id for result in results] == ["a", "b"]
        assert all(result.ok for result in results)
        assert len(entered) == 2

        for task_id in ("a", "b"):
            message = graph.get_message(f"msg_task_{task_id}")
            assert message.status is MessageStatus.ACKED
            assert message.result is not None
            assert message.result["ok"] is True
        assert all(graph.get_agent(worker_id).status is AgentStatus.ACTIVE for worker_id in workers)
    finally:
        graph.close()


def test_scheduler_never_runs_two_tasks_on_same_agent_concurrently(tmp_path) -> None:
    graph, root, workers = _graph_with_workers(tmp_path, 2)
    active = defaultdict(int)
    peak = defaultdict(int)
    lock = threading.Lock()

    def worker(agent, message):  # type: ignore[no-untyped-def]
        with lock:
            active[agent.agent_id] += 1
            peak[agent.agent_id] = max(peak[agent.agent_id], active[agent.agent_id])
        time.sleep(0.03)
        with lock:
            active[agent.agent_id] -= 1
        task_id = str(message.payload["taskId"])
        return ParallelTaskResult(task_id, agent.agent_id, True, "done")

    try:
        coordinator = ParallelAgentCoordinator(graph, root, workers)
        results = coordinator.run(
            [ParallelTask(f"task_{index}", {"index": index}) for index in range(6)],
            worker,
        )
        assert len(results) == 6
        assert set(peak) == set(workers)
        assert all(value == 1 for value in peak.values())
    finally:
        graph.close()


def test_worker_exception_marks_only_that_agent_failed_and_other_worker_continues(tmp_path) -> None:
    graph, root, workers = _graph_with_workers(tmp_path, 2)
    failing_worker = workers[0]

    def worker(agent, message):  # type: ignore[no-untyped-def]
        task_id = str(message.payload["taskId"])
        if agent.agent_id == failing_worker:
            raise RuntimeError("simulated worker crash")
        time.sleep(0.02)
        return ParallelTaskResult(task_id, agent.agent_id, True, "recovered pool capacity")

    try:
        coordinator = ParallelAgentCoordinator(graph, root, workers)
        results = coordinator.run(
            [
                ParallelTask("first", {}),
                ParallelTask("second", {}),
                ParallelTask("third", {}),
            ],
            worker,
        )

        assert len(results) == 3
        failed = next(result for result in results if result.task_id == "first")
        assert failed.ok is False
        assert "simulated worker crash" in failed.summary
        assert graph.get_agent(failing_worker).status is AgentStatus.FAILED

        survivors = [result for result in results if result.task_id in {"second", "third"}]
        assert all(result.ok for result in survivors)
        assert {result.agent_id for result in survivors} == {workers[1]}
        assert graph.get_agent(workers[1]).status is AgentStatus.ACTIVE
    finally:
        graph.close()


def test_wrong_task_or_agent_identity_is_converted_to_failed_result(tmp_path) -> None:
    graph, root, workers = _graph_with_workers(tmp_path, 1)

    def wrong_task(agent, message):  # type: ignore[no-untyped-def]
        del message
        return ParallelTaskResult("different", agent.agent_id, True, "wrong")

    try:
        coordinator = ParallelAgentCoordinator(graph, root, workers)
        result = coordinator.run([ParallelTask("expected", {})], wrong_task)[0]
        assert result.ok is False
        assert result.task_id == "expected"
        assert result.evidence["returned_task_id"] == "different"
        message = graph.get_message("msg_task_expected")
        assert message.status is MessageStatus.ACKED
        assert message.result is not None and message.result["ok"] is False
    finally:
        graph.close()

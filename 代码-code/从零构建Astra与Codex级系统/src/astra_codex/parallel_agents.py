from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .agent_graph import (
    AgentMessage,
    AgentNode,
    AgentStatus,
    PersistentAgentGraph,
    TERMINAL_AGENT_STATUSES,
)


@dataclass(frozen=True, slots=True)
class ParallelTask:
    task_id: str
    payload: dict[str, Any]
    kind: str = "task.assign"


@dataclass(frozen=True, slots=True)
class ParallelTaskResult:
    task_id: str
    agent_id: str
    ok: bool
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None


ParallelWorker = Callable[[AgentNode, AgentMessage], ParallelTaskResult]


@dataclass(frozen=True, slots=True)
class _RunningTask:
    task: ParallelTask
    agent: AgentNode
    message: AgentMessage
    started_at: float


class ParallelAgentCoordinator:
    """Reference one-task-per-agent parallel scheduler.

    Durable AgentGraph/Mailbox operations stay on the coordinator thread. Worker
    functions execute concurrently but receive immutable snapshots; they do not
    share the SQLite connection. This avoids confusing Python-thread safety with
    distributed-agent semantics while still giving us real wall-clock overlap to
    measure.

    The scheduler is deliberately simple:

    * one active task per worker agent;
    * FIFO task queue;
    * mailbox assignment persisted before worker execution;
    * successful/failed worker results are ACKed as message results;
    * an uncaught worker exception marks that agent FAILED and the remaining
      queue can continue on still-active workers.

    It is not yet a process pool, remote actor runtime or worktree merger.
    """

    def __init__(
        self,
        graph: PersistentAgentGraph,
        coordinator_agent_id: str,
        worker_agent_ids: Iterable[str],
        *,
        max_workers: int | None = None,
        message_lease_seconds: float = 3600.0,
    ) -> None:
        self.graph = graph
        self.coordinator = graph.get_agent(coordinator_agent_id)
        worker_ids = tuple(worker_agent_ids)
        if not worker_ids:
            raise ValueError("at least one worker agent is required")
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("duplicate worker agent ids")
        workers = tuple(graph.get_agent(agent_id) for agent_id in worker_ids)
        if any(worker.thread_id != self.coordinator.thread_id for worker in workers):
            raise ValueError("coordinator and workers must share one thread")
        if any(worker.agent_id == self.coordinator.agent_id for worker in workers):
            raise ValueError("coordinator cannot also be a worker")
        if message_lease_seconds <= 0:
            raise ValueError("message_lease_seconds must be positive")
        resolved_max = len(workers) if max_workers is None else max_workers
        if resolved_max <= 0:
            raise ValueError("max_workers must be positive")
        self.workers = workers
        self.max_workers = min(resolved_max, len(workers))
        self.message_lease_seconds = message_lease_seconds

    @staticmethod
    def _normalize_result(
        running: _RunningTask,
        result: ParallelTaskResult,
        *,
        finished_at: float,
    ) -> ParallelTaskResult:
        if result.task_id != running.task.task_id:
            return ParallelTaskResult(
                task_id=running.task.task_id,
                agent_id=running.agent.agent_id,
                ok=False,
                summary="worker returned a result for the wrong task id",
                evidence={"returned_task_id": result.task_id},
                started_at=running.started_at,
                finished_at=finished_at,
            )
        if result.agent_id != running.agent.agent_id:
            return ParallelTaskResult(
                task_id=running.task.task_id,
                agent_id=running.agent.agent_id,
                ok=False,
                summary="worker returned a result for the wrong agent id",
                evidence={"returned_agent_id": result.agent_id},
                started_at=running.started_at,
                finished_at=finished_at,
            )
        return ParallelTaskResult(
            task_id=result.task_id,
            agent_id=result.agent_id,
            ok=result.ok,
            summary=result.summary,
            evidence=dict(result.evidence),
            started_at=running.started_at
            if result.started_at is None
            else result.started_at,
            finished_at=finished_at
            if result.finished_at is None
            else result.finished_at,
        )

    def run(
        self,
        tasks: Iterable[ParallelTask],
        worker: ParallelWorker,
    ) -> list[ParallelTaskResult]:
        pending = deque(tasks)
        task_ids = [task.task_id for task in pending]
        if not task_ids:
            return []
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate parallel task ids")
        if any(not task_id for task_id in task_ids):
            raise ValueError("task_id cannot be empty")

        available = deque(
            node.agent_id
            for node in self.workers
            if node.status not in TERMINAL_AGENT_STATUSES
        )
        if not available:
            raise RuntimeError("no active worker agents are available")

        results: list[ParallelTaskResult] = []
        running: dict[Future[ParallelTaskResult], _RunningTask] = {}

        def launch(
            executor: ThreadPoolExecutor,
            task: ParallelTask,
            agent_id: str,
        ) -> None:
            agent = self.graph.get_agent(agent_id)
            if agent.status in TERMINAL_AGENT_STATUSES:
                raise RuntimeError(f"worker agent is terminal: {agent_id}")
            message_id = f"msg_task_{task.task_id}"
            self.graph.send(
                self.coordinator.agent_id,
                agent_id,
                task.kind,
                {"taskId": task.task_id, **task.payload},
                message_id=message_id,
            )
            claim_owner = f"parallel-coordinator:{agent_id}"
            message = self.graph.claim_message(
                agent_id,
                claim_owner,
                lease_seconds=self.message_lease_seconds,
                kinds={task.kind},
            )
            if message is None or message.message_id != message_id:
                raise RuntimeError(
                    f"failed to claim freshly assigned message {message_id!r}"
                )
            self.graph.set_status(agent_id, AgentStatus.BUSY, reason=task.task_id)
            started_at = time.perf_counter()
            future = executor.submit(worker, self.graph.get_agent(agent_id), message)
            running[future] = _RunningTask(task, agent, message, started_at)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending or running:
                while pending and available and len(running) < self.max_workers:
                    task = pending.popleft()
                    agent_id = available.popleft()
                    launch(executor, task, agent_id)

                if not running:
                    if pending:
                        raise RuntimeError(
                            "parallel task queue still has work but no active workers remain"
                        )
                    break

                done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in done:
                    current = running.pop(future)
                    claim_owner = f"parallel-coordinator:{current.agent.agent_id}"
                    finished_at = time.perf_counter()
                    try:
                        raw = future.result()
                    except Exception as exc:
                        result = ParallelTaskResult(
                            task_id=current.task.task_id,
                            agent_id=current.agent.agent_id,
                            ok=False,
                            summary=f"worker raised {type(exc).__name__}: {exc}",
                            evidence={"exception": type(exc).__name__},
                            started_at=current.started_at,
                            finished_at=finished_at,
                        )
                        self.graph.ack_message(
                            current.message.message_id,
                            claim_owner,
                            result={
                                "taskId": result.task_id,
                                "ok": False,
                                "summary": result.summary,
                                "evidence": result.evidence,
                            },
                        )
                        self.graph.set_status(
                            current.agent.agent_id,
                            AgentStatus.FAILED,
                            reason=result.summary,
                        )
                    else:
                        result = self._normalize_result(
                            current,
                            raw,
                            finished_at=finished_at,
                        )
                        self.graph.ack_message(
                            current.message.message_id,
                            claim_owner,
                            result={
                                "taskId": result.task_id,
                                "ok": result.ok,
                                "summary": result.summary,
                                "evidence": result.evidence,
                            },
                        )
                        self.graph.set_status(
                            current.agent.agent_id,
                            AgentStatus.ACTIVE,
                            reason=f"finished {current.task.task_id}",
                        )
                        available.append(current.agent.agent_id)
                    results.append(result)

        order = {task_id: index for index, task_id in enumerate(task_ids)}
        results.sort(key=lambda result: order[result.task_id])
        return results

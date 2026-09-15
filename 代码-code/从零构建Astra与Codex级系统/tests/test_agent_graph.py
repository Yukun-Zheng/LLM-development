from __future__ import annotations

import pytest

from astra_codex.agent_graph import (
    AgentStatus,
    MessageStatus,
    PersistentAgentGraph,
)


def test_agent_graph_persists_hierarchy_status_and_lineage(tmp_path) -> None:
    path = tmp_path / "agents.sqlite"
    with PersistentAgentGraph(path) as graph:
        root = graph.create_agent(
            "thr_1",
            "coordinator",
            agent_id="agent_root",
            metadata={"model": "coordinator-model"},
            now=1.0,
        )
        worker = graph.create_agent(
            "thr_1",
            "worker",
            agent_id="agent_worker",
            parent_agent_id=root,
            metadata={"worktree": "worker-a"},
            now=2.0,
        )
        reviewer = graph.create_agent(
            "thr_1",
            "reviewer",
            agent_id="agent_reviewer",
            parent_agent_id=root,
            now=3.0,
        )
        graph.set_status(worker, AgentStatus.BUSY, reason="task assigned", now=4.0)

        assert [node.agent_id for node in graph.children(root)] == [
            worker,
            reviewer,
        ]
        assert [node.agent_id for node in graph.lineage(worker)] == [root, worker]
        assert graph.get_agent(worker).status is AgentStatus.BUSY
        assert graph.events(worker)[-1].payload["to"] == "busy"

    with PersistentAgentGraph(path) as reopened:
        assert reopened.get_agent("agent_root").metadata["model"] == "coordinator-model"
        assert reopened.get_agent("agent_worker").metadata["worktree"] == "worker-a"
        assert reopened.get_agent("agent_worker").status is AgentStatus.BUSY
        assert [node.agent_id for node in reopened.agents_for_thread("thr_1")] == [
            "agent_root",
            "agent_worker",
            "agent_reviewer",
        ]


def test_agent_graph_enforces_thread_and_terminal_boundaries(tmp_path) -> None:
    with PersistentAgentGraph(tmp_path / "agents.sqlite") as graph:
        root = graph.create_agent("thr_a", "coordinator", agent_id="root")
        graph.create_agent("thr_b", "coordinator", agent_id="other")

        with pytest.raises(ValueError, match="same thread"):
            graph.create_agent(
                "thr_b",
                "worker",
                agent_id="bad_child",
                parent_agent_id=root,
            )

        graph.set_status(root, AgentStatus.COMPLETED)
        with pytest.raises(RuntimeError, match="invalid agent transition"):
            graph.set_status(root, AgentStatus.ACTIVE)
        with pytest.raises(RuntimeError, match="terminal"):
            graph.create_agent(
                "thr_a",
                "worker",
                agent_id="late_child",
                parent_agent_id=root,
            )


def test_mailbox_lease_reclaim_ack_and_restart_persistence(tmp_path) -> None:
    path = tmp_path / "agents.sqlite"
    with PersistentAgentGraph(path) as graph:
        coordinator = graph.create_agent(
            "thr_mail",
            "coordinator",
            agent_id="coordinator",
        )
        worker = graph.create_agent(
            "thr_mail",
            "worker",
            agent_id="worker",
            parent_agent_id=coordinator,
        )
        message_id = graph.send(
            coordinator,
            worker,
            "task.assign",
            {"taskId": "task_1", "goal": "inspect subsystem"},
            message_id="msg_task",
            now=1.0,
        )

        first = graph.claim_message(
            worker,
            "process_a",
            lease_seconds=5.0,
            now=2.0,
        )
        assert first is not None
        assert first.message_id == message_id
        assert first.status is MessageStatus.LEASED
        assert graph.claim_message(worker, "process_b", now=3.0) is None

        reclaimed = graph.claim_message(worker, "process_b", now=8.0)
        assert reclaimed is not None
        assert reclaimed.message_id == message_id
        assert reclaimed.lease_owner == "process_b"

        with pytest.raises(PermissionError, match="lease belongs"):
            graph.ack_message(message_id, "process_a", now=9.0)

        graph.ack_message(
            message_id,
            "process_b",
            result={"summary": "done", "artifactId": "art_patch"},
            now=10.0,
        )
        acked = graph.get_message(message_id)
        assert acked.status is MessageStatus.ACKED
        assert acked.result == {"summary": "done", "artifactId": "art_patch"}

    with PersistentAgentGraph(path) as reopened:
        acked = reopened.get_message("msg_task")
        assert acked.status is MessageStatus.ACKED
        assert reopened.claim_message("worker", "process_c", now=20.0) is None
        kinds = [event.event_type for event in reopened.events("worker")]
        assert "mailbox.received" in kinds
        assert "mailbox.claimed" in kinds
        assert "mailbox.acked" in kinds


def test_mailbox_filters_kind_and_release_returns_message_to_pending(tmp_path) -> None:
    with PersistentAgentGraph(tmp_path / "agents.sqlite") as graph:
        root = graph.create_agent("thr", "coordinator", agent_id="root")
        worker = graph.create_agent(
            "thr",
            "worker",
            agent_id="worker",
            parent_agent_id=root,
        )
        graph.send(root, worker, "chat", {"text": "hello"}, message_id="msg_chat", now=1.0)
        graph.send(
            root,
            worker,
            "task.assign",
            {"taskId": "t1"},
            message_id="msg_task",
            now=2.0,
        )

        task = graph.claim_message(
            worker,
            "worker_process",
            kinds={"task.assign"},
            now=3.0,
        )
        assert task is not None and task.message_id == "msg_task"
        graph.release_message("msg_task", "worker_process", now=4.0)
        assert graph.get_message("msg_task").status is MessageStatus.PENDING

        task_again = graph.claim_message(
            worker,
            "worker_process_2",
            kinds={"task.assign"},
            now=5.0,
        )
        assert task_again is not None and task_again.message_id == "msg_task"

        pending = graph.inbox(worker, statuses={MessageStatus.PENDING})
        assert [message.message_id for message in pending] == ["msg_chat"]


def test_cancel_subtree_cancels_descendants_before_parent_and_blocks_new_mail(tmp_path) -> None:
    with PersistentAgentGraph(tmp_path / "agents.sqlite") as graph:
        root = graph.create_agent("thr", "coordinator", agent_id="root")
        child = graph.create_agent(
            "thr",
            "worker",
            agent_id="child",
            parent_agent_id=root,
        )
        grandchild = graph.create_agent(
            "thr",
            "reviewer",
            agent_id="grandchild",
            parent_agent_id=child,
        )
        sibling = graph.create_agent(
            "thr",
            "worker",
            agent_id="sibling",
            parent_agent_id=root,
        )

        cancelled = graph.cancel_subtree(child, reason="coordinator aborted branch", now=10.0)
        assert cancelled == (grandchild, child)
        assert graph.get_agent(child).status is AgentStatus.CANCELLED
        assert graph.get_agent(grandchild).status is AgentStatus.CANCELLED
        assert graph.get_agent(root).status is AgentStatus.ACTIVE
        assert graph.get_agent(sibling).status is AgentStatus.ACTIVE

        with pytest.raises(RuntimeError, match="terminal recipient"):
            graph.send(root, child, "chat", {"text": "too late"})
        with pytest.raises(RuntimeError, match="terminal sender"):
            graph.send(child, root, "chat", {"text": "too late"})


def test_cross_thread_mail_is_rejected(tmp_path) -> None:
    with PersistentAgentGraph(tmp_path / "agents.sqlite") as graph:
        left = graph.create_agent("thr_left", "worker", agent_id="left")
        right = graph.create_agent("thr_right", "worker", agent_id="right")
        with pytest.raises(ValueError, match="cross-thread"):
            graph.send(left, right, "chat", {"text": "no"})

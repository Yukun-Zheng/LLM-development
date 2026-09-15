from __future__ import annotations

import pytest

from astra_codex.agent import ScriptedBackend
from astra_codex.app_server import AgentAppClient, AgentAppServer, AppServerError, InProcessAppTransport
from astra_codex.control_auth import AuthenticationError, BearerTokenAuthorizer, Principal
from astra_codex.http_app_server import HTTPAppTransport, LocalHTTPAppServer
from astra_codex.runtime import DurableAgentRuntime
from astra_codex.runtime_queue import WorkStatus
from astra_codex.tools import ToolRegistry


def _authorizer() -> BearerTokenAuthorizer:
    authorizer = BearerTokenAuthorizer()
    authorizer.register(
        "admin-secret",
        Principal("admin"),
    )
    authorizer.register(
        "thread-a-reader",
        Principal(
            "reader-a",
            allowed_methods=frozenset({"server/discover", "thread/get", "event/poll"}),
            thread_ids=frozenset({"thr_a"}),
        ),
    )
    authorizer.register(
        "thread-a-worker",
        Principal(
            "worker-a",
            allowed_methods=frozenset({"runtime/runOne"}),
            thread_ids=frozenset({"thr_a"}),
        ),
    )
    return authorizer


def test_authenticated_app_server_rejects_missing_and_invalid_tokens(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state", ScriptedBackend([]), ToolRegistry([])
    ) as runtime:
        server = AgentAppServer(runtime, authorizer=_authorizer())

        missing = AgentAppClient(InProcessAppTransport(server))
        with pytest.raises(AppServerError) as exc_info:
            missing.call("server/discover")
        assert exc_info.value.code == -32001

        invalid = AgentAppClient(
            InProcessAppTransport(server, bearer_token="wrong-secret")
        )
        with pytest.raises(AppServerError) as exc_info:
            invalid.call("server/discover")
        assert exc_info.value.code == -32001


def test_bearer_expiry_is_enforced_at_authentication_time() -> None:
    authorizer = BearerTokenAuthorizer()
    principal = Principal("short-lived")
    authorizer.register(
        "ephemeral",
        principal,
        now=100.0,
        expires_at=110.0,
    )

    assert authorizer.authenticate("ephemeral", now=109.999) == principal
    with pytest.raises(AuthenticationError, match="expired"):
        authorizer.authenticate("ephemeral", now=110.0)


def test_revoked_bearer_cannot_authenticate_and_revoke_is_idempotent() -> None:
    authorizer = BearerTokenAuthorizer()
    authorizer.register("revocable", Principal("operator"), now=1.0)
    authorizer.revoke("revocable", now=5.0)
    authorizer.revoke("revocable", now=6.0)

    credential = authorizer.credential("revocable")
    assert credential.revoked_at == 5.0
    with pytest.raises(AuthenticationError, match="revoked"):
        authorizer.authenticate("revocable", now=6.0)


def test_rotation_preserves_scope_revokes_old_and_activates_new() -> None:
    authorizer = BearerTokenAuthorizer()
    principal = Principal(
        "rotating-worker",
        allowed_methods=frozenset({"runtime/runOne"}),
        thread_ids=frozenset({"thr_a"}),
    )
    authorizer.register("old-secret", principal, now=1.0, expires_at=50.0)
    authorizer.rotate(
        "old-secret",
        "new-secret",
        now=10.0,
        expires_at=100.0,
    )

    with pytest.raises(AuthenticationError, match="revoked"):
        authorizer.authenticate("old-secret", now=11.0)
    assert authorizer.authenticate("new-secret", now=11.0) == principal
    assert authorizer.credential("old-secret").revoked_at == 10.0
    assert authorizer.credential("new-secret").expires_at == 100.0


def test_failed_rotation_does_not_revoke_old_credential() -> None:
    authorizer = BearerTokenAuthorizer()
    principal = Principal("primary")
    authorizer.register("old", principal, now=1.0)
    authorizer.register("already-used", Principal("other"), now=1.0)

    with pytest.raises(ValueError, match="already registered"):
        authorizer.rotate("old", "already-used", now=2.0)

    assert authorizer.authenticate("old", now=3.0) == principal
    assert authorizer.credential("old").revoked_at is None


def test_thread_scope_blocks_cross_thread_reads_and_unscoped_event_feed(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state", ScriptedBackend([]), ToolRegistry([])
    ) as runtime:
        runtime.create_thread("thr_a")
        runtime.create_thread("thr_b")
        server = AgentAppServer(runtime, authorizer=_authorizer())
        reader = AgentAppClient(
            InProcessAppTransport(server, bearer_token="thread-a-reader")
        )

        assert reader.call("thread/get", {"threadId": "thr_a"})["threadId"] == "thr_a"
        assert reader.call("event/poll", {"threadId": "thr_a"})["events"]

        with pytest.raises(AppServerError) as exc_info:
            reader.call("thread/get", {"threadId": "thr_b"})
        assert exc_info.value.code == -32003

        with pytest.raises(AppServerError) as exc_info:
            reader.call("event/poll")
        assert exc_info.value.code == -32003

        with pytest.raises(AppServerError) as exc_info:
            reader.call("thread/submit", {"threadId": "thr_a", "content": "no write"})
        assert exc_info.value.code == -32003


def test_scoped_worker_requires_thread_and_queue_claim_cannot_leak_other_thread(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["thread a done"]),
        ToolRegistry([]),
    ) as runtime:
        runtime.create_thread("thr_a")
        runtime.create_thread("thr_b")

        # Deliberately enqueue B first. A global FIFO claim would pick B. The
        # scoped worker must skip it inside the SQL claim and lease A instead.
        work_b = runtime.submit("thr_b", "older b task", item_id="work_b", now=1.0)
        work_a = runtime.submit("thr_a", "allowed a task", item_id="work_a", now=2.0)

        server = AgentAppServer(runtime, authorizer=_authorizer())
        scoped_worker = AgentAppClient(
            InProcessAppTransport(server, bearer_token="thread-a-worker")
        )

        with pytest.raises(AppServerError) as exc_info:
            scoped_worker.call("runtime/runOne", {"workerId": "worker"})
        assert exc_info.value.code == -32003

        with pytest.raises(AppServerError) as exc_info:
            scoped_worker.call(
                "runtime/runOne",
                {"workerId": "worker", "threadId": "thr_b"},
            )
        assert exc_info.value.code == -32003

        result = scoped_worker.call(
            "runtime/runOne",
            {"workerId": "worker", "threadId": "thr_a"},
        )
        assert result["workItemId"] == work_a
        assert result["threadId"] == "thr_a"
        assert result["status"] == "completed"
        assert result["finalAnswer"] == "thread a done"

        assert runtime.work_queue.get(work_a).status is WorkStatus.COMPLETED
        assert runtime.work_queue.get(work_b).status is WorkStatus.PENDING


def test_admin_principal_can_use_global_worker_endpoint(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend(["done"]),
        ToolRegistry([]),
    ) as runtime:
        server = AgentAppServer(runtime, authorizer=_authorizer())
        admin = AgentAppClient(
            InProcessAppTransport(server, bearer_token="admin-secret")
        )
        admin.call("thread/create", {"threadId": "thr_admin"})
        admin.call(
            "thread/submit",
            {"threadId": "thr_admin", "content": "finish", "itemId": "work_admin"},
        )
        result = admin.call("runtime/runOne", {"workerId": "admin-worker"})
        assert result["status"] == "completed"
        assert result["finalAnswer"] == "done"


def test_http_transport_carries_bearer_token_and_enforces_policy(tmp_path) -> None:
    with DurableAgentRuntime(
        tmp_path / "state",
        ScriptedBackend([]),
        ToolRegistry([]),
    ) as runtime:
        runtime.create_thread("thr_a")
        prototype = AgentAppServer(runtime, authorizer=_authorizer())
        with LocalHTTPAppServer(prototype) as http_server:
            anonymous = AgentAppClient(HTTPAppTransport(http_server.rpc_url))
            with pytest.raises(AppServerError) as exc_info:
                anonymous.call("server/discover")
            assert exc_info.value.code == -32001

            reader = AgentAppClient(
                HTTPAppTransport(
                    http_server.rpc_url,
                    bearer_token="thread-a-reader",
                )
            )
            projection = reader.call("thread/get", {"threadId": "thr_a"})
            assert projection["threadId"] == "thr_a"

            with pytest.raises(AppServerError) as exc_info:
                reader.call("thread/get", {"threadId": "thr_b"})
            assert exc_info.value.code == -32003

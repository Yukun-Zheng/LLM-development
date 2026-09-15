from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .control_auth import (
    AuthenticationError,
    AuthorizationError,
    BearerTokenAuthorizer,
)
from .durable import ThreadProjection
from .event_stream import RuntimeEvent
from .runtime import DurableAgentRuntime, RuntimeExecutionRecord


class AppServerError(ValueError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class AppRequest:
    request_id: str | int
    method: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AppResponse:
    request_id: str | int | None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": self.request_id}
        if self.error is None:
            payload["result"] = self.result
        else:
            payload["error"] = self.error
        return payload


class AppTransport(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def _projection_payload(projection: ThreadProjection) -> dict[str, Any]:
    return {
        "threadId": projection.thread_id,
        "status": projection.status.value,
        "activeTurnId": projection.active_turn_id,
        "lastEventId": projection.last_event_id,
        "submissions": list(projection.submissions),
        "lastCheckpoint": projection.last_checkpoint,
        "failureReason": projection.failure_reason,
        "cancellationReason": projection.cancellation_reason,
        "parentThreadId": projection.parent_thread_id,
        "parentEventId": projection.parent_event_id,
    }


def _execution_payload(record: RuntimeExecutionRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "workItemId": record.work_item_id,
        "threadId": record.thread_id,
        "turnId": record.turn_id,
        "status": record.status,
        "finalAnswer": record.final_answer,
        "modelSteps": record.model_steps,
    }


def _runtime_event_payload(event: RuntimeEvent) -> dict[str, Any]:
    return {
        "eventId": event.event_id,
        "topic": event.topic,
        "payload": event.payload,
        "threadId": event.thread_id,
        "turnId": event.turn_id,
        "createdAt": event.created_at,
    }


class AgentAppServer:
    """JSON-RPC control plane over ``DurableAgentRuntime``.

    The App Server can run unauthenticated for in-process teaching tests, or be
    given ``BearerTokenAuthorizer``. When authorization is enabled, every
    request is authenticated before dispatch and checked against explicit method
    and thread scopes.

    The current event feed is durable cursor polling. Network transport, TLS and
    push subscriptions are separate layers; enabling bearer auth does not make a
    plaintext network channel safe for internet exposure.
    """

    def __init__(
        self,
        runtime: DurableAgentRuntime,
        *,
        authorizer: BearerTokenAuthorizer | None = None,
    ) -> None:
        self.runtime = runtime
        self.authorizer = authorizer

    def handle(
        self,
        payload: dict[str, Any],
        *,
        bearer_token: str | None = None,
    ) -> dict[str, Any]:
        request_id = payload.get("id") if isinstance(payload, dict) else None
        try:
            request = self._parse(payload)
            if self.authorizer is not None:
                principal = self.authorizer.authenticate(bearer_token)
                self.authorizer.authorize(principal, request.method, request.params)
            result = self._dispatch(request.method, request.params)
            return AppResponse(request.request_id, result=result).to_dict()
        except AuthenticationError as exc:
            return AppResponse(
                request_id,
                error={"code": -32001, "message": f"Unauthenticated: {exc}"},
            ).to_dict()
        except AuthorizationError as exc:
            return AppResponse(
                request_id,
                error={"code": -32003, "message": f"Forbidden: {exc}"},
            ).to_dict()
        except AppServerError as exc:
            error: dict[str, Any] = {"code": exc.code, "message": exc.message}
            if exc.data is not None:
                error["data"] = exc.data
            return AppResponse(request_id, error=error).to_dict()
        except (KeyError, RuntimeError, ValueError, PermissionError) as exc:
            return AppResponse(
                request_id,
                error={
                    "code": -32000,
                    "message": f"Runtime error: {type(exc).__name__}: {exc}",
                },
            ).to_dict()
        except Exception as exc:
            return AppResponse(
                request_id,
                error={
                    "code": -32603,
                    "message": f"Internal error: {type(exc).__name__}: {exc}",
                },
            ).to_dict()

    def _parse(self, payload: dict[str, Any]) -> AppRequest:
        if not isinstance(payload, dict):
            raise AppServerError(-32600, "Invalid Request")
        if payload.get("jsonrpc") != "2.0":
            raise AppServerError(-32600, "jsonrpc must be '2.0'")
        request_id = payload.get("id")
        if not isinstance(request_id, (str, int)):
            raise AppServerError(-32600, "id must be a string or integer")
        method = payload.get("method")
        if not isinstance(method, str):
            raise AppServerError(-32600, "method must be a string")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise AppServerError(-32602, "params must be an object")
        return AppRequest(request_id, method, params)

    @staticmethod
    def _require_str(params: dict[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value:
            raise AppServerError(-32602, f"{key} must be a non-empty string")
        return value

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "server/discover":
            return {
                "serverInfo": {"name": "astra-codex-app-server", "version": "0.3.0"},
                "methods": [
                    "thread/create",
                    "thread/get",
                    "thread/submit",
                    "thread/steer",
                    "thread/pause",
                    "thread/resume",
                    "thread/cancel",
                    "thread/fork",
                    "runtime/runOne",
                    "artifact/list",
                    "event/poll",
                ],
                "eventFeed": {
                    "mode": "cursor-poll",
                    "cursor": "eventId",
                    "replayable": True,
                },
                "authentication": {
                    "required": self.authorizer is not None,
                    "scheme": "bearer" if self.authorizer is not None else None,
                },
            }

        if method == "thread/create":
            thread_id = params.get("threadId")
            if thread_id is not None and not isinstance(thread_id, str):
                raise AppServerError(-32602, "threadId must be a string")
            created = self.runtime.create_thread(thread_id)
            return _projection_payload(self.runtime.thread_store.project(created))

        if method == "thread/get":
            thread_id = self._require_str(params, "threadId")
            return _projection_payload(self.runtime.thread_store.project(thread_id))

        if method == "thread/submit":
            thread_id = self._require_str(params, "threadId")
            content = self._require_str(params, "content")
            item_id = params.get("itemId")
            if item_id is not None and not isinstance(item_id, str):
                raise AppServerError(-32602, "itemId must be a string")
            created_item = self.runtime.submit(thread_id, content, item_id=item_id)
            return {"workItemId": created_item}

        if method == "thread/steer":
            thread_id = self._require_str(params, "threadId")
            content = self._require_str(params, "content")
            steering_id = params.get("steeringId")
            if steering_id is not None and not isinstance(steering_id, str):
                raise AppServerError(-32602, "steeringId must be a string")
            created = self.runtime.steer(
                thread_id,
                content,
                steering_id=steering_id,
            )
            return {"steeringId": created}

        if method == "thread/pause":
            thread_id = self._require_str(params, "threadId")
            reason = params.get("reason", "")
            if not isinstance(reason, str):
                raise AppServerError(-32602, "reason must be a string")
            self.runtime.thread_store.pause(thread_id, reason)
            self.runtime._emit(
                "thread.paused", {"reason": reason}, thread_id=thread_id
            )
            return _projection_payload(self.runtime.thread_store.project(thread_id))

        if method == "thread/resume":
            thread_id = self._require_str(params, "threadId")
            self.runtime.thread_store.resume(thread_id)
            self.runtime._emit("thread.resumed", {}, thread_id=thread_id)
            return _projection_payload(self.runtime.thread_store.project(thread_id))

        if method == "thread/cancel":
            thread_id = self._require_str(params, "threadId")
            reason = params.get("reason", "")
            if not isinstance(reason, str):
                raise AppServerError(-32602, "reason must be a string")
            self.runtime.thread_store.cancel_thread(thread_id, reason)
            self.runtime._emit(
                "thread.cancelled", {"reason": reason}, thread_id=thread_id
            )
            return _projection_payload(self.runtime.thread_store.project(thread_id))

        if method == "thread/fork":
            thread_id = self._require_str(params, "threadId")
            new_thread_id = params.get("newThreadId")
            if new_thread_id is not None and not isinstance(new_thread_id, str):
                raise AppServerError(-32602, "newThreadId must be a string")
            through = params.get("throughEventId")
            if through is not None and not isinstance(through, int):
                raise AppServerError(-32602, "throughEventId must be an integer")
            child = self.runtime.thread_store.fork_thread(
                thread_id,
                new_thread_id=new_thread_id,
                through_event_id=through,
            )
            self.runtime._emit(
                "thread.forked",
                {"parentThreadId": thread_id, "throughEventId": through},
                thread_id=child,
            )
            return _projection_payload(self.runtime.thread_store.project(child))

        if method == "runtime/runOne":
            worker_id = self._require_str(params, "workerId")
            lease_seconds = params.get("leaseSeconds", 300.0)
            if not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
                raise AppServerError(-32602, "leaseSeconds must be positive")
            return _execution_payload(
                self.runtime.run_one(worker_id, lease_seconds=float(lease_seconds))
            )

        if method == "artifact/list":
            thread_id = self._require_str(params, "threadId")
            return {
                "artifacts": [
                    {
                        "artifactId": item.artifact_id,
                        "kind": item.kind,
                        "sha256": item.sha256,
                        "sizeBytes": item.size_bytes,
                        "metadata": item.metadata,
                        "createdAt": item.created_at,
                    }
                    for item in self.runtime.artifact_store.list_thread(thread_id)
                ]
            }

        if method == "event/poll":
            after = params.get("afterEventId", 0)
            limit = params.get("limit", 100)
            thread_id = params.get("threadId")
            topics = params.get("topics")
            if not isinstance(after, int) or after < 0:
                raise AppServerError(-32602, "afterEventId must be a non-negative integer")
            if not isinstance(limit, int) or limit <= 0 or limit > 1000:
                raise AppServerError(-32602, "limit must be an integer between 1 and 1000")
            if thread_id is not None and not isinstance(thread_id, str):
                raise AppServerError(-32602, "threadId must be a string")
            if topics is not None and (
                not isinstance(topics, list)
                or not all(isinstance(topic, str) and topic for topic in topics)
            ):
                raise AppServerError(-32602, "topics must be a list of non-empty strings")

            events = self.runtime.event_stream.read(
                after_id=after,
                limit=limit,
                thread_id=thread_id,
                topics=topics,
            )
            next_after = events[-1].event_id if events else after
            return {
                "events": [_runtime_event_payload(event) for event in events],
                "nextAfterEventId": next_after,
                "highWatermark": self.runtime.event_stream.high_watermark,
            }

        raise AppServerError(-32601, f"Method not found: {method}")


class InProcessAppTransport:
    def __init__(
        self,
        server: AgentAppServer,
        *,
        bearer_token: str | None = None,
    ) -> None:
        self.server = server
        self.bearer_token = bearer_token

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.server.handle(payload, bearer_token=self.bearer_token)


class AgentAppClient:
    def __init__(self, transport: AppTransport) -> None:
        self.transport = transport
        self._next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        response = self.transport.request(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        if response.get("error") is not None:
            error = response["error"]
            raise AppServerError(
                int(error.get("code", -32603)),
                str(error.get("message", "unknown error")),
                error.get("data"),
            )
        return response.get("result")

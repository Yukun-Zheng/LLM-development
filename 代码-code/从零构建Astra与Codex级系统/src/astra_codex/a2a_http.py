"""Small source-first A2A v1 HTTP+JSON binding.

Normative source pinned for this teaching implementation:
https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

Implemented unprefixed v1 routes:

* GET  /.well-known/agent-card.json
* POST /message:send
* GET  /tasks/{id}
* GET  /tasks
* POST /tasks/{id}:cancel

This module deliberately does not claim complete A2A conformance. Streaming,
push notifications, authenticated extended cards, tenant-prefixed bindings,
security schemes, pagination and protocol conformance certification remain
separate layers.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .a2a import (
    A2AAgentCard,
    A2AProtocolError,
    A2ASendMessageConfiguration,
    A2ASendMessageRequest,
    A2AService,
    A2ATask,
    A2ATaskState,
    A2ATaskStore,
)

A2A_AGENT_CARD_PATH = "/.well-known/agent-card.json"


class A2AHTTPError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _configuration_payload(config: A2ASendMessageConfiguration) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config.accepted_output_modes:
        payload["acceptedOutputModes"] = list(config.accepted_output_modes)
    if config.history_length is not None:
        payload["historyLength"] = config.history_length
    if config.return_immediately:
        payload["returnImmediately"] = True
    return payload


def _send_request_payload(request: A2ASendMessageRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": request.message.to_dict()}
    configuration = _configuration_payload(request.configuration)
    if configuration:
        payload["configuration"] = configuration
    if request.metadata:
        payload["metadata"] = request.metadata
    if request.tenant is not None:
        payload["tenant"] = request.tenant
    return payload


class LocalA2AHTTPServer:
    """Serialized loopback reference server for the A2A HTTP+JSON subset.

    ``A2ATaskStore`` uses SQLite's normal thread-affine connections. The server
    thread therefore opens its own store handle pointing at the same durable DB
    instead of sharing the caller's connection across threads.
    """

    def __init__(
        self,
        service: A2AService,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.prototype = service
        self._thread_service: A2AService | None = None
        self._ready = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                try:
                    result = outer._handle_http("GET", self.path, None)
                    self._write_json(200, result)
                except A2AHTTPError as exc:
                    self._write_json(exc.status, {"error": {"message": exc.message}})

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body: dict[str, Any] = {}
                    if length:
                        raw = self.rfile.read(length)
                        decoded = json.loads(raw.decode("utf-8"))
                        if not isinstance(decoded, dict):
                            raise A2AHTTPError(400, "request body must be a JSON object")
                        body = decoded
                    result = outer._handle_http("POST", self.path, body)
                    self._write_json(200, result)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._write_json(400, {"error": {"message": f"invalid JSON: {exc}"}})
                except A2AHTTPError as exc:
                    self._write_json(exc.status, {"error": {"message": exc.message}})

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = HTTPServer((host, port), Handler)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def _require_service(self) -> A2AService:
        if self._thread_service is None:
            raise A2AHTTPError(503, "A2A service is not ready")
        return self._thread_service

    @staticmethod
    def _single_query(query: dict[str, list[str]], name: str) -> str | None:
        values = query.get(name)
        if not values:
            return None
        if len(values) != 1:
            raise A2AHTTPError(400, f"query parameter {name} must occur once")
        return values[0]

    def _handle_http(
        self,
        method: str,
        target: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        service = self._require_service()
        parsed = urlparse(target)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query, keep_blank_values=True)

        try:
            if method == "GET" and path == A2A_AGENT_CARD_PATH:
                return service.card.to_dict()

            if method == "POST" and path == "/message:send":
                assert body is not None
                return service.handle_operation("SendMessage", body)

            if method == "GET" and path == "/tasks":
                params: dict[str, Any] = {}
                context_id = self._single_query(query, "contextId")
                if context_id is not None:
                    params["contextId"] = context_id
                states = query.get("state", [])
                if states:
                    params["states"] = states
                # Pagination is intentionally not fabricated. Reject it until
                # the store has a real opaque page-token implementation.
                if "pageSize" in query or "pageToken" in query:
                    raise A2AHTTPError(
                        501,
                        "reference ListTasks does not yet implement A2A pagination",
                    )
                return service.handle_operation("ListTasks", params)

            if path.startswith("/tasks/"):
                suffix = path[len("/tasks/") :]
                if method == "POST" and suffix.endswith(":cancel"):
                    task_id = suffix[: -len(":cancel")]
                    if not task_id:
                        raise A2AHTTPError(400, "task id is required")
                    return service.handle_operation("CancelTask", {"id": task_id})

                if method == "GET" and suffix and ":" not in suffix:
                    params = {"id": suffix}
                    history_length = self._single_query(query, "historyLength")
                    if history_length is not None:
                        try:
                            params["historyLength"] = int(history_length)
                        except ValueError as exc:
                            raise A2AHTTPError(
                                400, "historyLength must be an integer"
                            ) from exc
                    return service.handle_operation("GetTask", params)

            raise A2AHTTPError(404, f"unsupported A2A route: {method} {path}")
        except KeyError as exc:
            raise A2AHTTPError(404, str(exc)) from exc
        except (A2AProtocolError, ValueError, TypeError) as exc:
            raise A2AHTTPError(400, str(exc)) from exc

    def _serve(self) -> None:
        try:
            with A2ATaskStore(self.prototype.store.path) as store:
                self._thread_service = A2AService(
                    self.prototype.card,
                    store,
                    handler=self.prototype.handler,
                )
                self._ready.set()
                self.server.serve_forever()
        finally:
            self._thread_service = None
            self._ready.set()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("A2A HTTP server is already running")
        self._ready.clear()
        self.thread = threading.Thread(
            target=self._serve,
            name="astra-codex-a2a-http",
            daemon=True,
        )
        self.thread.start()
        if not self._ready.wait(timeout=5.0) or self._thread_service is None:
            raise RuntimeError("A2A HTTP server failed to initialize")

    def close(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=5.0)
        self.server.server_close()
        self.thread = None
        self._thread_service = None

    def __enter__(self) -> "LocalA2AHTTPServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class A2AHTTPClient:
    """Synchronous client for the implemented A2A HTTP+JSON reference subset."""

    def __init__(self, base_url: str, *, timeout_s: float = 10.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - explicit loopback/reference client
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
                message = str(body.get("error", {}).get("message", exc.reason))
            except Exception:
                message = str(exc.reason)
            raise A2AHTTPError(exc.code, message) from exc
        if not isinstance(decoded, dict):
            raise A2AHTTPError(502, "A2A server returned a non-object JSON body")
        return decoded

    def get_agent_card(self) -> A2AAgentCard:
        # AgentCard.from_dict is intentionally not implemented yet; return the
        # wire object through a thin reconstruction of fields this project owns.
        payload = self._request("GET", A2A_AGENT_CARD_PATH)
        interfaces = payload.get("supportedInterfaces", [])
        skills = payload.get("skills", [])
        from .a2a import A2AAgentInterface, A2AAgentSkill

        return A2AAgentCard(
            name=str(payload["name"]),
            description=str(payload["description"]),
            supported_interfaces=tuple(
                A2AAgentInterface(
                    url=str(item["url"]),
                    protocol_binding=str(item["protocolBinding"]),
                    protocol_version=str(item["protocolVersion"]),
                    tenant=None if item.get("tenant") is None else str(item["tenant"]),
                )
                for item in interfaces
            ),
            version=str(payload["version"]),
            default_input_modes=tuple(str(x) for x in payload["defaultInputModes"]),
            default_output_modes=tuple(str(x) for x in payload["defaultOutputModes"]),
            skills=tuple(
                A2AAgentSkill(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    description=str(item["description"]),
                    tags=tuple(str(x) for x in item.get("tags", [])),
                    examples=tuple(str(x) for x in item.get("examples", [])),
                    input_modes=tuple(str(x) for x in item.get("inputModes", [])),
                    output_modes=tuple(str(x) for x in item.get("outputModes", [])),
                )
                for item in skills
            ),
            streaming=bool(payload.get("capabilities", {}).get("streaming", False)),
            push_notifications=bool(
                payload.get("capabilities", {}).get("pushNotifications", False)
            ),
            extended_agent_card=bool(
                payload.get("capabilities", {}).get("extendedAgentCard", False)
            ),
            documentation_url=(
                None
                if payload.get("documentationUrl") is None
                else str(payload["documentationUrl"])
            ),
        )

    def send_message(self, request: A2ASendMessageRequest) -> A2ATask:
        return A2ATask.from_dict(
            self._request("POST", "/message:send", _send_request_payload(request))
        )

    def get_task(
        self,
        task_id: str,
        *,
        history_length: int | None = None,
    ) -> A2ATask:
        query = ""
        if history_length is not None:
            query = "?" + urlencode({"historyLength": history_length})
        return A2ATask.from_dict(
            self._request("GET", f"/tasks/{quote(task_id, safe='')}{query}")
        )

    def list_tasks(
        self,
        *,
        context_id: str | None = None,
        states: set[A2ATaskState] | None = None,
    ) -> tuple[A2ATask, ...]:
        params: list[tuple[str, str]] = []
        if context_id is not None:
            params.append(("contextId", context_id))
        if states:
            params.extend(("state", state.value) for state in sorted(states, key=lambda x: x.value))
        suffix = "?" + urlencode(params) if params else ""
        payload = self._request("GET", "/tasks" + suffix)
        raw_tasks = payload.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise A2AHTTPError(502, "ListTasks response.tasks must be an array")
        return tuple(A2ATask.from_dict(item) for item in raw_tasks)

    def cancel_task(self, task_id: str) -> A2ATask:
        return A2ATask.from_dict(
            self._request("POST", f"/tasks/{quote(task_id, safe='')}:cancel", {})
        )

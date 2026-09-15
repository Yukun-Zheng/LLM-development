from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.request import Request, urlopen

from .app_server import AgentAppServer, AppTransport
from .runtime import DurableAgentRuntime


def _bearer_token(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


class LocalHTTPAppServer:
    """Small stdlib HTTP transport for the educational App Server.

    The server binds to loopback by default and exposes one POST endpoint:
    ``/rpc``. Requests are deliberately serialized through ``HTTPServer`` so the
    reference control plane does not pretend to provide production concurrency.

    SQLite connections are thread-affine by default. Rather than disabling that
    safety check globally, the HTTP server opens its *own* DurableAgentRuntime
    handle inside the server thread, pointing at the same durable state files.
    The prototype App Server's authorizer is preserved, so bearer identity is
    checked in the same policy layer for in-process and HTTP transports.

    There is intentionally no TLS, CORS policy or internet-facing hardening.
    Bearer authentication over plaintext HTTP is only acceptable here because
    the reference server binds to loopback; remote deployment needs TLS and a
    stronger identity/session design.
    """

    def __init__(
        self,
        app_server: AgentAppServer,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.prototype = app_server
        self._thread_app_server: AgentAppServer | None = None
        self._ready = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path != "/rpc":
                    self._write_json(404, {"error": "not found"})
                    return
                if outer._thread_app_server is None:
                    self._write_json(503, {"error": "app server not ready"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0:
                        raise ValueError("empty request body")
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("JSON-RPC request must be an object")
                    response = outer._thread_app_server.handle(
                        payload,
                        bearer_token=_bearer_token(self.headers.get("Authorization")),
                    )
                    self._write_json(200, response)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    self._write_json(
                        400,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": str(exc)},
                        },
                    )

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.server = HTTPServer((host, port), Handler)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def rpc_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/rpc"

    def _serve(self) -> None:
        prototype_runtime = self.prototype.runtime
        registry = prototype_runtime.journaled_tools.registry
        retryable = prototype_runtime.journaled_tools.retryable_in_doubt_tools
        try:
            with DurableAgentRuntime(
                prototype_runtime.state_dir,
                prototype_runtime.backend,
                registry,
                max_model_steps=prototype_runtime.max_model_steps,
                retryable_in_doubt_tools=retryable,
            ) as runtime_handle:
                self._thread_app_server = AgentAppServer(
                    runtime_handle,
                    authorizer=self.prototype.authorizer,
                )
                self._ready.set()
                self.server.serve_forever()
        finally:
            self._thread_app_server = None
            self._ready.set()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("HTTP App Server is already running")
        self._ready.clear()
        self.thread = threading.Thread(
            target=self._serve,
            name="astra-codex-app-server",
            daemon=True,
        )
        self.thread.start()
        if not self._ready.wait(timeout=5.0) or self._thread_app_server is None:
            raise RuntimeError("HTTP App Server failed to initialize its runtime handle")

    def close(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=5.0)
        self.server.server_close()
        self.thread = None
        self._thread_app_server = None

    def __enter__(self) -> "LocalHTTPAppServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HTTPAppTransport(AppTransport):
    """Synchronous JSON-RPC-over-HTTP client transport."""

    def __init__(
        self,
        rpc_url: str,
        *,
        timeout_s: float = 10.0,
        bearer_token: str | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.rpc_url = rpc_url
        self.timeout_s = timeout_s
        self.bearer_token = bearer_token

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            self.rpc_url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - explicit local educational transport
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("App Server returned a non-object JSON response")
        return decoded

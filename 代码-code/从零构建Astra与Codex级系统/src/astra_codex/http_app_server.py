from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

from .app_server import AgentAppServer, AppTransport


class LocalHTTPAppServer:
    """Small stdlib HTTP transport for the educational App Server.

    The server binds to loopback by default and exposes one POST endpoint:
    ``/rpc``. It intentionally has no TLS, authentication, CORS policy or
    internet-facing hardening. Those must be added before any non-local use.
    """

    def __init__(
        self,
        app_server: AgentAppServer,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.app_server = app_server
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                if self.path != "/rpc":
                    self._write_json(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0:
                        raise ValueError("empty request body")
                    raw = self.rfile.read(length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("JSON-RPC request must be an object")
                    response = outer.app_server.handle(payload)
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

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    @property
    def rpc_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/rpc"

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("HTTP App Server is already running")
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="astra-codex-app-server",
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None

    def __enter__(self) -> "LocalHTTPAppServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class HTTPAppTransport(AppTransport):
    """Synchronous JSON-RPC-over-HTTP client transport."""

    def __init__(self, rpc_url: str, *, timeout_s: float = 10.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.rpc_url = rpc_url
        self.timeout_s = timeout_s

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.rpc_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - explicit local educational transport
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("App Server returned a non-object JSON response")
        return decoded

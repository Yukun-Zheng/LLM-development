"""Reference SSE transport for A2A v1 ``SendStreamingMessage``.

Pinned source:
https://github.com/a2aproject/A2A/blob/6d6640c29b102f7a8d23784901351b5d2454fe71/specification/a2a.proto

The normative HTTP route is ``POST /message:stream`` and the RPC returns a
stream of ``StreamResponse`` values.  Server-Sent Events are used as the HTTP
stream framing in this educational binding.  ``SubscribeToTask`` is kept out of
this module until the Task store has a real asynchronous update source.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .a2a import A2AService, A2ATaskStore
from .a2a_http import A2AHTTPError, _send_request_payload
from .a2a_streaming import A2AStreamResponse, stream_send_message


def _encode_sse(response: A2AStreamResponse) -> bytes:
    payload = json.dumps(
        response.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode("utf-8")


def _decode_sse_lines(lines: Iterator[bytes]) -> Iterator[A2AStreamResponse]:
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                if not isinstance(payload, dict):
                    raise A2AHTTPError(502, "SSE data must decode to a JSON object")
                yield A2AStreamResponse.from_dict(payload)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if data_lines:
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise A2AHTTPError(502, "SSE data must decode to a JSON object")
        yield A2AStreamResponse.from_dict(payload)


class LocalA2AStreamingHTTPServer:
    """Loopback reference server for ``POST /message:stream``.

    The server thread owns its SQLite Task-store connection. Each yielded
    ``StreamResponse`` is flushed immediately, so the initial SUBMITTED Task can
    reach the client before the synchronous task handler finishes.
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

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                if self.path != "/message:stream":
                    self._write_json(404, {"error": {"message": "not found"}})
                    return
                if outer._thread_service is None:
                    self._write_json(503, {"error": {"message": "A2A stream service not ready"}})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0:
                        raise A2AHTTPError(400, "empty request body")
                    decoded = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(decoded, dict):
                        raise A2AHTTPError(400, "request body must be a JSON object")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for response in stream_send_message(outer._thread_service, decoded):
                        self.wfile.write(_encode_sse(response))
                        self.wfile.flush()
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._write_json(400, {"error": {"message": f"invalid JSON: {exc}"}})
                except (A2AHTTPError, ValueError) as exc:
                    # If headers were already sent this cannot be converted into
                    # a new HTTP status. In the deterministic reference path all
                    # request validation happens before the first stream frame.
                    if not self.wfile.closed:
                        try:
                            self._write_json(
                                exc.status if isinstance(exc, A2AHTTPError) else 400,
                                {"error": {"message": str(exc)}},
                            )
                        except OSError:
                            pass

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
    def base_url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

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
            raise RuntimeError("A2A streaming HTTP server is already running")
        self._ready.clear()
        self.thread = threading.Thread(
            target=self._serve,
            name="astra-codex-a2a-streaming-http",
            daemon=True,
        )
        self.thread.start()
        if not self._ready.wait(timeout=5.0) or self._thread_service is None:
            raise RuntimeError("A2A streaming HTTP server failed to initialize")

    def close(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=5.0)
        self.server.server_close()
        self.thread = None
        self._thread_service = None

    def __enter__(self) -> "LocalA2AStreamingHTTPServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class A2AStreamingHTTPClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send_streaming_message(self, request) -> Iterator[A2AStreamResponse]:  # type: ignore[no-untyped-def]
        body = json.dumps(_send_request_payload(request), ensure_ascii=False).encode("utf-8")
        http_request = Request(
            self.base_url + "/message:stream",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            response = urlopen(http_request, timeout=self.timeout_s)  # noqa: S310 - explicit loopback/reference client
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                message = str(payload.get("error", {}).get("message", exc.reason))
            except Exception:
                message = str(exc.reason)
            raise A2AHTTPError(exc.code, message) from exc
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            response.close()
            raise A2AHTTPError(502, f"expected text/event-stream, got {content_type!r}")

        def iterator() -> Iterator[A2AStreamResponse]:
            try:
                yield from _decode_sse_lines(iter(response))
            finally:
                response.close()

        return iterator()

"""Thread-owned A2A HTTP gateway backed by DurableAgentRuntime.

The ordinary ``LocalA2AHTTPServer`` preserves an arbitrary handler. A handler
that closes over a DurableAgentRuntime constructed on another thread would
violate SQLite thread affinity. This specialized gateway mirrors the project's
App-Server design: the server thread opens its own runtime handles over the same
durable state directory, then installs ``DurableRuntimeA2AHandler`` there.
"""

from __future__ import annotations

from pathlib import Path

from .a2a import A2AAgentCard, A2AService, A2ATaskStore
from .a2a_http import LocalA2AHTTPServer
from .a2a_runtime_bridge import DurableRuntimeA2AHandler
from .runtime import DurableAgentRuntime


class LocalA2ADurableRuntimeHTTPServer(LocalA2AHTTPServer):
    """Real localhost A2A HTTP → durable Agent-OS reference gateway.

    This remains a single-process teaching server. The important property is
    ownership: SQLite-backed A2A and Agent-OS state is opened in the same server
    thread that services HTTP requests. No SQLite connection object crosses a
    thread boundary.
    """

    def __init__(
        self,
        card: A2AAgentCard,
        task_store_path: str | Path,
        runtime: DurableAgentRuntime,
        *,
        worker_id: str = "a2a-http-worker",
        lease_seconds: float = 300.0,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.task_store_path = Path(task_store_path).resolve()
        self.task_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_prototype = runtime
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self._bootstrap_store = A2ATaskStore(self.task_store_path)
        super().__init__(
            A2AService(card, self._bootstrap_store),
            host=host,
            port=port,
        )

    def _serve(self) -> None:
        prototype = self.runtime_prototype
        registry = prototype.journaled_tools.registry
        retryable = prototype.journaled_tools.retryable_in_doubt_tools
        try:
            with A2ATaskStore(self.task_store_path) as store:
                with DurableAgentRuntime(
                    prototype.state_dir,
                    prototype.backend,
                    registry,
                    max_model_steps=prototype.max_model_steps,
                    retryable_in_doubt_tools=retryable,
                ) as runtime_handle:
                    handler = DurableRuntimeA2AHandler(
                        runtime_handle,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                    self._thread_service = A2AService(
                        self.prototype.card,
                        store,
                        handler=handler,
                    )
                    self._ready.set()
                    self.server.serve_forever()
        finally:
            self._thread_service = None
            self._ready.set()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._bootstrap_store.close()

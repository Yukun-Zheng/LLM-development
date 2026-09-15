from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated control-plane identity and explicit capability scope.

    ``allowed_methods=None`` means all App-Server methods. ``thread_ids=None``
    means all threads. A finite ``thread_ids`` set is intentionally restrictive:
    methods that cannot be scoped to a target thread (for example the current
    global ``runtime/runOne`` worker claim) are rejected for such principals.
    """

    subject: str
    allowed_methods: frozenset[str] | None = None
    thread_ids: frozenset[str] | None = None


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


class BearerTokenAuthorizer:
    """In-memory bearer-token authenticator plus method/thread authorization.

    Tokens are stored only as SHA-256 digests. This is still a teaching/local
    control-plane mechanism, not a full identity system: there is no token
    expiry, rotation protocol, OIDC, mTLS, audit service or distributed policy
    engine yet.
    """

    def __init__(self) -> None:
        self._principals: dict[str, Principal] = {}

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register(self, token: str, principal: Principal) -> None:
        if not token:
            raise ValueError("bearer token cannot be empty")
        if not principal.subject:
            raise ValueError("principal subject cannot be empty")
        digest = self._digest(token)
        if digest in self._principals:
            raise ValueError("bearer token is already registered")
        self._principals[digest] = principal

    def authenticate(self, token: str | None) -> Principal:
        if token is None or not token:
            raise AuthenticationError("missing bearer token")
        principal = self._principals.get(self._digest(token))
        if principal is None:
            raise AuthenticationError("invalid bearer token")
        return principal

    def authorize(
        self,
        principal: Principal,
        method: str,
        params: dict[str, Any],
    ) -> None:
        methods = principal.allowed_methods
        if methods is not None and method not in methods:
            raise AuthorizationError(
                f"principal {principal.subject!r} cannot call method {method!r}"
            )

        thread_scope = principal.thread_ids
        if thread_scope is None:
            return

        # server/discover contains no tenant data and is safe after successful
        # authentication, provided the method itself is allowed above.
        if method == "server/discover":
            return

        # The current worker endpoint claims the next global work item and does
        # not accept a thread selector. It cannot safely be used by a scoped
        # principal until the queue supports an authorized thread filter.
        if method == "runtime/runOne":
            raise AuthorizationError(
                "thread-scoped principals cannot call global runtime/runOne"
            )

        if method == "thread/create":
            requested = params.get("threadId")
            if not isinstance(requested, str) or not requested:
                raise AuthorizationError(
                    "thread-scoped principals must create an explicit authorized threadId"
                )
            self._require_thread(principal, requested)
            return

        thread_id = params.get("threadId")
        if not isinstance(thread_id, str) or not thread_id:
            raise AuthorizationError(
                f"thread-scoped principal requires threadId for method {method!r}"
            )
        self._require_thread(principal, thread_id)

        if method == "thread/fork":
            child = params.get("newThreadId")
            if not isinstance(child, str) or not child:
                raise AuthorizationError(
                    "thread-scoped principals must supply an explicit authorized newThreadId"
                )
            self._require_thread(principal, child)

    @staticmethod
    def _require_thread(principal: Principal, thread_id: str) -> None:
        assert principal.thread_ids is not None
        if thread_id not in principal.thread_ids:
            raise AuthorizationError(
                f"principal {principal.subject!r} cannot access thread {thread_id!r}"
            )

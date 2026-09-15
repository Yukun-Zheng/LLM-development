from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated control-plane identity and explicit capability scope.

    ``allowed_methods=None`` means all App-Server methods. ``thread_ids=None``
    means all threads. A finite ``thread_ids`` set requires every resource-bearing
    request to identify a thread that belongs to the principal's scope.
    """

    subject: str
    allowed_methods: frozenset[str] | None = None
    thread_ids: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class BearerCredential:
    """Stored credential metadata; the original bearer secret is never retained."""

    principal: Principal
    issued_at: float
    expires_at: float | None = None
    revoked_at: float | None = None

    def active(self, now: float) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or now < self.expires_at


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


class BearerTokenAuthorizer:
    """In-memory bearer authentication + lifecycle + scoped authorization.

    Bearer secrets are represented only by SHA-256 digests in the lookup table.
    The reference lifecycle supports explicit expiry, revocation and atomic-ish
    rotation at the authorizer layer. This is still not a production identity
    service: credential metadata is process-local, there is no OIDC/mTLS,
    distributed revocation propagation, secure persistence or external audit
    service yet.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, BearerCredential] = {}

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register(
        self,
        token: str,
        principal: Principal,
        *,
        expires_at: float | None = None,
        now: float | None = None,
    ) -> None:
        if not token:
            raise ValueError("bearer token cannot be empty")
        if not principal.subject:
            raise ValueError("principal subject cannot be empty")
        issued_at = time.time() if now is None else now
        if expires_at is not None and expires_at <= issued_at:
            raise ValueError("expires_at must be later than issued_at")
        digest = self._digest(token)
        if digest in self._credentials:
            raise ValueError("bearer token is already registered")
        self._credentials[digest] = BearerCredential(
            principal=principal,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def authenticate(
        self,
        token: str | None,
        *,
        now: float | None = None,
    ) -> Principal:
        if token is None or not token:
            raise AuthenticationError("missing bearer token")
        credential = self._credentials.get(self._digest(token))
        if credential is None:
            raise AuthenticationError("invalid bearer token")
        timestamp = time.time() if now is None else now
        if credential.revoked_at is not None:
            raise AuthenticationError("revoked bearer token")
        if credential.expires_at is not None and timestamp >= credential.expires_at:
            raise AuthenticationError("expired bearer token")
        return credential.principal

    def revoke(self, token: str, *, now: float | None = None) -> None:
        digest = self._digest(token)
        credential = self._credentials.get(digest)
        if credential is None:
            raise KeyError("unknown bearer token")
        if credential.revoked_at is not None:
            return
        timestamp = time.time() if now is None else now
        self._credentials[digest] = replace(credential, revoked_at=timestamp)

    def rotate(
        self,
        old_token: str,
        new_token: str,
        *,
        expires_at: float | None = None,
        now: float | None = None,
    ) -> None:
        """Replace one active credential while preserving its Principal scope."""

        timestamp = time.time() if now is None else now
        principal = self.authenticate(old_token, now=timestamp)
        new_digest = self._digest(new_token)
        if new_digest in self._credentials:
            raise ValueError("new bearer token is already registered")
        self.register(
            new_token,
            principal,
            expires_at=expires_at,
            now=timestamp,
        )
        self.revoke(old_token, now=timestamp)

    def credential(self, token: str) -> BearerCredential:
        credential = self._credentials.get(self._digest(token))
        if credential is None:
            raise KeyError("unknown bearer token")
        return credential

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

        if method == "server/discover":
            return

        if method == "runtime/runOne":
            requested = params.get("threadId")
            if not isinstance(requested, str) or not requested:
                raise AuthorizationError(
                    "thread-scoped principals must supply threadId to runtime/runOne"
                )
            self._require_thread(principal, requested)
            return

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

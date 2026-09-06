"""One error shape for everything the gateway returns.

The rule: a client sees a stable, documented envelope; the operator sees the
detail in the logs, correlated by ``request_id``. Nothing that describes the
gateway's internals - upstream hostnames, provider error bodies, exception
text, tracebacks - crosses that line.

That is not decoration. Upstream error bodies routinely carry the provider
account id, the internal endpoint, and occasionally a fragment of the prompt;
a stack trace carries the filesystem layout and library versions. Passing them
through is how a gateway becomes a reconnaissance endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrorCode:
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ALL_PROVIDERS_FAILED = "all_providers_failed"
    UPSTREAM_RATE_LIMITED = "upstream_rate_limited"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    INVALID_REQUEST = "invalid_request"
    UNAUTHENTICATED = "unauthenticated"
    INTERNAL_ERROR = "internal_error"


#: Messages are constants, not f-strings built from upstream data. A message
#: that interpolates an exception is one refactor away from leaking one.
_MESSAGES = {
    ErrorCode.RATE_LIMIT_EXCEEDED: "Token rate limit exceeded for this API key.",
    ErrorCode.ALL_PROVIDERS_FAILED: "No upstream model provider could serve this request.",
    ErrorCode.UPSTREAM_RATE_LIMITED: "Every upstream model provider is rate limited.",
    ErrorCode.UPSTREAM_TIMEOUT: "The upstream model provider did not respond in time.",
    ErrorCode.INVALID_REQUEST: "The request payload was invalid.",
    ErrorCode.UNAUTHENTICATED: "A valid API key is required.",
    ErrorCode.INTERNAL_ERROR: "The gateway encountered an internal error.",
}

_STATUS = {
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.ALL_PROVIDERS_FAILED: 502,
    # Capacity exhaustion upstream is retryable. Reporting it as a 502 tells a
    # client its request failed permanently, so it retries immediately into an
    # upstream that is still full.
    ErrorCode.UPSTREAM_RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_TIMEOUT: 504,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.INTERNAL_ERROR: 500,
}

#: Only these keys may appear under ``error.details``. An allow-list, because a
#: deny-list of "things that leak" is a list nobody keeps up to date.
_ALLOWED_DETAIL_KEYS = frozenset(
    {
        "limit_tokens_per_minute",
        "requested_tokens",
        "tokens_used_in_window",
        "retry_after_seconds",
        "providers_attempted",
        "timeout_ms",
        "field",
    }
)


@dataclass
class GatewayError(Exception):
    code: str
    request_id: str
    details: dict[str, Any] | None = None
    #: Full, unredacted context for the log. Never serialised to the client.
    internal_note: str = ""

    @property
    def status_code(self) -> int:
        return _STATUS.get(self.code, 500)

    @property
    def message(self) -> str:
        return _MESSAGES.get(self.code, _MESSAGES[ErrorCode.INTERNAL_ERROR])

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "request_id": self.request_id,
        }
        if self.details:
            # ``is not None`` as well as the allow-list: the filter keys on
            # name only, so a key set to None survived and the public envelope
            # published ``"timeout_ms": null`` on every non-timeout failure.
            filtered = {
                k: v
                for k, v in self.details.items()
                if k in _ALLOWED_DETAIL_KEYS and v is not None
            }
            if filtered:
                error["details"] = filtered
        return {"error": error}

    def headers(self) -> dict[str, str]:
        headers = {"X-Request-Id": self.request_id}
        retry_after = (self.details or {}).get("retry_after_seconds")
        if retry_after is not None:
            # Integer seconds, per RFC 9110; round up so a client that obeys it
            # cannot come back a fraction of a second too early.
            headers["Retry-After"] = str(max(1, int(retry_after + 0.999)))
        return headers

    def __str__(self) -> str:  # pragma: no cover - for logs only
        return f"{self.code}[{self.request_id}] {self.internal_note or self.message}"

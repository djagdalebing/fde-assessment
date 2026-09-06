"""Bearer-token authentication for the gateway.

The brief asks the gateway to "read an incoming ``Bearer <token>`` HTTP header
and extract the user's role", so that is all this does: a token registry that
resolves a token to a role.

In a real deployment the registry is a call to the identity provider, or a
cache in front of one - never a literal in the source tree. ``TokenVerifier``
is the seam where that swap happens.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass


class AuthError(Exception):
    """The caller could not be authenticated.

    The message is intentionally coarse. Telling an attacker whether a token
    was unknown or merely malformed is free reconnaissance; the detail goes to
    the log, not the response.
    """


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str

    def __str__(self) -> str:
        return f"{self.subject}({self.role})"


#: Demo registry.
DEFAULT_TOKENS: dict[str, Principal] = {
    "admin-token-abc123": Principal("alice@example.com", "admin"),
    "viewer-token-def456": Principal("bob@example.com", "viewer"),
}


class TokenVerifier:
    """Resolves a raw ``Authorization`` header value to a ``Principal``."""

    def __init__(self, tokens: dict[str, Principal] | None = None) -> None:
        self._tokens = dict(DEFAULT_TOKENS if tokens is None else tokens)

    def principal_from_header(self, header_value: str | None) -> Principal:
        if not header_value:
            raise AuthError("missing Authorization header")

        # RFC 7235 makes the scheme case-insensitive; be liberal there and
        # strict about everything that follows.
        scheme, _, token = header_value.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError("Authorization header must be 'Bearer <token>'")

        # Compare against every known token in constant time. A plain dict
        # lookup leaks length and prefix information through timing.
        #
        # Compared verbatim, not stripped. ``token.strip()`` meant
        # "Bearer admin-token-abc123   " and "Bearer  admin-token-abc123"
        # both authenticated - not an escalation, since the real secret is
        # still required, but it widens the accepted credential surface for
        # no benefit and makes the constant-time compare a little less honest.
        candidate = token
        if not candidate.isascii():
            # ``hmac.compare_digest`` refuses non-ASCII ``str`` and would raise
            # a TypeError straight out of the request handler.
            raise AuthError("non-ASCII byte in Authorization header")

        found: Principal | None = None
        for known, principal in self._tokens.items():
            if hmac.compare_digest(known, candidate):
                found = principal
        if found is None:
            raise AuthError("unknown token")
        return found

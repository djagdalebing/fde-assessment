"""Resilient model routing: rate limit, then try providers in order.

Request lifecycle
-----------------
1. **Reserve.** Estimate the request's token cost and reserve it against the
   tenant's sliding window. Refused here means no provider is called at all.
2. **Attempt providers in priority order**, each under its own hard deadline.
   429, timeout, 5xx and transport failures all fail over; a 4xx that is the
   *caller's* fault does not, because retrying it on another provider just
   burns a second quota to get the same answer.
3. **Reconcile.** Settle the reservation against the provider's reported usage,
   or release it entirely if nothing succeeded - a failed request must not
   consume budget.

The deadline
------------
``asyncio.timeout`` wraps each attempt rather than relying on the HTTP client's
timeout alone. httpx applies its timeouts *per phase* - connect, then read,
then write - so a provider that trickles bytes can keep a 3000 ms-configured
request alive far longer than 3000 ms. Only a wall-clock deadline around the
whole attempt actually bounds it, and "3000 ms" is a promise about wall clock.

Cancellation is the subtle part: when the deadline fires, the in-flight request
is cancelled and its connection released before the next provider is tried. A
router that leaves the abandoned attempt running holds a connection from the
pool for every timed-out request, and under sustained upstream slowness that
pool is what fails first.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from task4_model_router.errors import ErrorCode, GatewayError
from task4_model_router.limiter import Reservation, TokenRateLimiter
from task4_model_router.providers import ProviderAttempt, ProviderConfig
from task4_model_router.tokens import (
    DEFAULT_COMPLETION_CEILING,
    actual_tokens_from_response,
    estimate_request_tokens,
)

logger = logging.getLogger("model-router")

#: Statuses that mean "this provider cannot serve it, another might".
#:
#: Statuses that fail over, and the reasoning is about WHOSE fault the failure
#: is - the gateway's or the caller's - not about the number's class.
#:
#: 401/403 concern the gateway's own credential with that provider. 402 is the
#: gateway's billing account. 404 is a wrong URL or an unknown model - and the
#: router rewrites ``model`` per provider itself, so that name is the gateway's
#: choice, not the caller's. 413 is a body over THIS provider's limit, which a
#: larger secondary may well accept.
#:
#: Every one of those used to be reported to the tenant as HTTP 400 "The
#: request payload was invalid", sending them to fix a payload that was fine
#: while a healthy secondary sat idle. The narrow reading (401/403 only) had
#: the right principle and drew the line in the wrong place.
_FAILOVER_STATUSES = frozenset(
    {401, 402, 403, 404, 408, 409, 413, 425, 429, 500, 502, 503, 504, 529}
)

#: Statuses that are genuinely the caller's to fix, so a second provider would
#: reject them identically and only burn a second quota doing it.
_CALLER_ERROR_STATUSES = frozenset({400, 405, 406, 415, 422})
#: Outcomes where the provider certainly generated - and billed for - a
#: completion, so the reservation is charged rather than released.
#:
#: A timeout is deliberately NOT here, and the choice is a real trade. The
#: provider may well have generated; we cannot know. Charging would make every
#: flaky-provider incident burn tenant budget for completions the client never
#: received, so the gateway under-counts in that case and says so. The
#: unusable-response cases are different: a body arrived, so the tokens are
#: definitely spent, and releasing them was handing out free completions.
_BILLABLE_OUTCOMES = frozenset({"unusable_response"})


@dataclass
class RouteResult:
    payload: dict[str, Any]
    provider: str
    attempts: list[ProviderAttempt]
    tokens_reserved: int
    tokens_charged: int
    request_id: str


class ModelRouter:
    def __init__(
        self,
        providers: list[ProviderConfig],
        limiter: TokenRateLimiter,
        client: httpx.AsyncClient,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = sorted(providers, key=lambda p: p.priority)
        self._limiter = limiter
        self._client = client
        # Tasks scheduled from a done-callback must be held somewhere. The
        # event loop keeps only a weak reference, so an unreferenced task can
        # be collected before it runs - and this one gives a reservation back.
        self._background: set = set()

    async def route(self, tenant_key: str, body: dict[str, Any], request_id: str | None = None) -> RouteResult:
        request_id = request_id or uuid.uuid4().hex[:16]
        # Off the event loop. Estimation now walks and serialises every content
        # shape, and it runs BEFORE any limit is applied - so on the loop, one
        # authenticated tenant sending an 8MB body took every other tenant from
        # 8ms to 13,888ms, at no cost to themselves because their own requests
        # were then rejected with 429.
        estimated = await asyncio.to_thread(estimate_request_tokens, body)

        # ---- 1. rate limit ---------------------------------------------------
        # Shielded, and the pending task is drained on cancellation. The
        # ``to_thread`` hop can be cancelled after sqlite has already COMMITted
        # the reservation but before the router receives it - leaving a hold
        # with no id to settle or release. Measured at 4% of cancelled requests,
        # and weaponisable: cancelled requests stranded 49,005 of 50,000 tokens
        # at no cost to the attacker.
        reserve = asyncio.ensure_future(self._limiter.try_reserve(tenant_key, estimated))
        try:
            decision = await asyncio.shield(reserve)
        except BaseException:
            reserve.add_done_callback(
                lambda task: self._release_orphan(task, tenant_key, request_id)
            )
            raise
        if not decision.allowed:
            logger.warning(
                "rate limited tenant=%s req=%s used=%d requested=%d limit=%d",
                tenant_key, request_id, decision.used_in_window, estimated, decision.limit,
            )
            raise GatewayError(
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                request_id=request_id,
                details={
                    "limit_tokens_per_minute": decision.limit,
                    "requested_tokens": estimated,
                    "tokens_used_in_window": decision.used_in_window,
                    "retry_after_seconds": round(decision.retry_after_seconds, 3),
                },
                internal_note=f"tenant={tenant_key} window_used={decision.used_in_window}",
            )

        reservation = decision.reservation
        assert reservation is not None

        # ---- 2 & 3. attempt, then reconcile ---------------------------------
        # ``settled`` is the difference between "never spent" and "already
        # charged". Without it, a cancellation arriving *after* settle() began
        # ran release() as well, deleting the row - so a client that hung up at
        # the right moment got a 40,000-token completion for free. The window is
        # real: settle() awaits a thread hop into sqlite.
        settled: list[bool] = []
        attempts_made: list[ProviderAttempt] = []
        try:
            return await self._attempt_providers(
                body, reservation, estimated, request_id, settled, attempts_made
            )
        except BaseException:
            if settled:
                raise
            # A provider that generated a response we then could not use has
            # still been paid, so the reservation is charged rather than
            # released - otherwise an unparseable body is a free completion.
            if any(a.outcome in _BILLABLE_OUTCOMES for a in attempts_made):
                logger.warning(
                    "charging the reservation for req=%s: a provider generated but the "
                    "response was unusable", request_id,
                )
                await asyncio.shield(self._limiter.settle(reservation, reservation.tokens))
            else:
                # Shielded like the settle path: a second cancellation arriving
                # during the release would strand the reservation for a full
                # window, and the two paths having different guarantees is the
                # kind of asymmetry that only shows up under load.
                await asyncio.shield(self._limiter.release(reservation))
            raise

    def _release_orphan(self, task, tenant_key: str, request_id: str) -> None:
        """Give back a reservation whose request was cancelled mid-reserve."""
        if task.cancelled() or task.exception() is not None:
            return
        decision = task.result()
        if decision.allowed and decision.reservation is not None:
            logger.info("releasing orphaned reservation req=%s tenant=%s", request_id, tenant_key)
            try:
                task = asyncio.ensure_future(self._limiter.release(decision.reservation))
            except RuntimeError:  # pragma: no cover - loop already closing
                logger.warning("could not schedule orphan release req=%s", request_id)
                return
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    async def _attempt_providers(
        self,
        body: dict[str, Any],
        reservation: Reservation,
        estimated: int,
        request_id: str,
        settled: list[bool],
        attempts: list[ProviderAttempt] | None = None,
    ) -> RouteResult:
        attempts = attempts if attempts is not None else []

        for provider in self._providers:
            attempt, payload = await self._call_provider(provider, body, request_id)
            attempts.append(attempt)

            if attempt.outcome == "ok" and payload is not None:
                charged = actual_tokens_from_response(payload, estimated)
                # Marked before the await: from here on the tokens are spent,
                # and a cancellation must not refund them. The settle itself is
                # shielded so a cancellation cannot leave the row unreconciled.
                settled.append(True)
                charged = await asyncio.shield(self._limiter.settle(reservation, charged))
                logger.info(
                    "served req=%s provider=%s attempts=%d reserved=%d charged=%d",
                    request_id, provider.name, len(attempts), estimated, charged,
                )
                return RouteResult(payload, provider.name, attempts, estimated, charged, request_id)

            logger.warning(
                "provider failed req=%s provider=%s outcome=%s status=%s detail=%s",
                request_id, provider.name, attempt.outcome, attempt.status_code, attempt.internal_detail,
            )

            if attempt.outcome == "client_error":
                # The caller's request is malformed or unauthorised for this
                # model. Failing over would produce the same rejection from a
                # second provider, so surface it now - as a request error, not
                # as ``all_providers_failed``. Reporting the caller's own 400 as
                # a provider outage sends them to the wrong place to fix it.
                raise GatewayError(
                    code=ErrorCode.INVALID_REQUEST,
                    request_id=request_id,
                    details={"providers_attempted": len(attempts)},
                    internal_note=f"{provider.name} rejected the request: {attempt.internal_detail}",
                )

        # Nothing worked. The error names *what* failed, never the providers'
        # own error text or endpoints.
        timed_out = bool(attempts) and all(a.outcome == "timeout" for a in attempts)
        # Every provider refusing for capacity is a retryable condition, not an
        # outage. Reported as 502 it reads as "your request failed", so a client
        # retries straight back into an upstream that is still full.
        rate_limited = bool(attempts) and all(a.outcome == "rate_limited" for a in attempts)
        if timed_out:
            code = ErrorCode.UPSTREAM_TIMEOUT
        elif rate_limited:
            code = ErrorCode.UPSTREAM_RATE_LIMITED
        else:
            code = ErrorCode.ALL_PROVIDERS_FAILED
        # Only a hint the provider actually gave us. None of them sending one
        # means no Retry-After header, rather than a number we made up.
        hints = [a.retry_after_seconds for a in attempts if a.retry_after_seconds is not None]
        durations = [a.duration_ms for a in attempts] or [0.0]
        raise GatewayError(
            code=code,
            request_id=request_id,
            details={
                "providers_attempted": len(attempts),
                "retry_after_seconds": max(hints) if (rate_limited and hints) else None,
                # The attempted provider's own budget, not provider[0]'s - the
                # envelope previously reported 3000 even when the request had
                # actually spent 8s across two providers.
                "timeout_ms": int(max(durations)) if timed_out else None,
            },
            internal_note="; ".join(
                f"{a.provider}:{a.outcome}:{a.status_code}:{a.internal_detail}" for a in attempts
            ),
        )

    async def _call_provider(
        self, provider: ProviderConfig, body: dict[str, Any], request_id: str
    ) -> tuple[ProviderAttempt, dict[str, Any] | None]:
        """One attempt, under a hard wall-clock deadline. Never raises."""
        payload_out = {**body, "model": provider.model}
        # Impose the completion ceiling we RESERVED against.
        #
        # Reserving 4096 for a request that omits max_tokens and then
        # forwarding no cap made the reservation aspirational: the provider ran
        # to its own default (8192-16384 on current models) and 40 ordinary
        # concurrent requests spent 196,848 tokens against a 50,000/minute
        # budget - 3.94x - with an upstream behaving perfectly correctly.
        #
        # tokens.py already says this in prose ("a request that simply omits
        # max_tokens lets the model run to its own limit"); the fix had been
        # applied to the reservation side only, which is the half that cannot
        # enforce anything.
        if not any(
            isinstance(payload_out.get(field), int) and not isinstance(payload_out.get(field), bool)
            for field in ("max_tokens", "max_completion_tokens", "max_output_tokens")
        ):
            payload_out["max_tokens"] = DEFAULT_COMPLETION_CEILING
        headers = {
            "authorization": f"Bearer {provider.api_key}",
            "content-type": "application/json",
            "x-request-id": request_id,
        }
        started = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        try:
            # The deadline covers connect + send + receive + parse. On expiry
            # the context manager cancels the request task, which closes the
            # connection and returns it to the pool before we move on.
            async with asyncio.timeout(provider.timeout_ms / 1000):
                response = await self._client.post(provider.url, json=payload_out, headers=headers)
                status = response.status_code

                # A 3xx used to fall through every branch below and be treated
                # as SUCCESS: a redirect body was proxied to the client as a
                # 200, internal hostname and all, and the request silently went
                # nowhere. httpx does not follow redirects, so there is no
                # completion behind it either.
                if status < 200 or 300 <= status < 400:
                    await response.aread()
                    return (
                        ProviderAttempt(provider.name, "server_error", status, elapsed_ms(),
                                        internal_detail=f"non-2xx/non-error status {status}"),
                        None,
                    )

                if status == 429:
                    await response.aread()
                    return (
                        ProviderAttempt(provider.name, "rate_limited", status, elapsed_ms(),
                                        internal_detail=_safe_snippet(response),
                                        retry_after_seconds=_retry_after(response)),
                        None,
                    )
                if status in _FAILOVER_STATUSES or status >= 500:
                    await response.aread()
                    return (
                        ProviderAttempt(provider.name, "server_error", status, elapsed_ms(),
                                        internal_detail=_safe_snippet(response)),
                        None,
                    )
                if status in _CALLER_ERROR_STATUSES:
                    await response.aread()
                    return (
                        ProviderAttempt(provider.name, "client_error", status, elapsed_ms(),
                                        internal_detail=_safe_snippet(response)),
                        None,
                    )
                if status >= 400:
                    # An unrecognised 4xx is treated as this provider's problem,
                    # not the caller's: failing over costs one extra attempt,
                    # while misattributing it takes the route down and tells the
                    # tenant to fix something that is not broken.
                    await response.aread()
                    return (
                        ProviderAttempt(provider.name, "server_error", status, elapsed_ms(),
                                        internal_detail=_safe_snippet(response)),
                        None,
                    )

                parsed = response.json()
                if not isinstance(parsed, dict):
                    return (
                        ProviderAttempt(provider.name, "unusable_response", status, elapsed_ms(),
                                        internal_detail="non-object JSON body"),
                        None,
                    )
                return ProviderAttempt(provider.name, "ok", status, elapsed_ms()), parsed

        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            return (
                ProviderAttempt(provider.name, "timeout", None, elapsed_ms(),
                                internal_detail=f"{type(exc).__name__} after {provider.timeout_ms}ms"),
                None,
            )
        except httpx.HTTPError as exc:
            return (
                ProviderAttempt(provider.name, "transport_error", None, elapsed_ms(),
                                internal_detail=f"{type(exc).__name__}: {exc}"),
                None,
            )
        except ValueError as exc:
            # json() on a body that is not JSON at all - an SSE stream, say.
            # The provider generated it, so the tokens are spent.
            return (
                ProviderAttempt(provider.name, "unusable_response", None, elapsed_ms(),
                                internal_detail=f"unparseable body: {type(exc).__name__}"),
                None,
            )
        except Exception as exc:
            # The docstring above promises this never raises, and it did:
            # deeply-nested provider JSON raises RecursionError out of
            # ``response.json()``, which is neither an httpx error nor a
            # ValueError. It escaped as an HTTP 500 and the healthy secondary
            # was never dialled. Anything unexpected is just a failed attempt.
            # (``BaseException`` is deliberately not caught: cancellation must
            # still propagate.)
            logger.exception("unexpected error calling %s", provider.name)
            return (
                ProviderAttempt(provider.name, "transport_error", None, elapsed_ms(),
                                internal_detail=f"{type(exc).__name__}: {exc}"),
                None,
            )


def _retry_after(response: httpx.Response) -> float | None:
    """The provider's own ``Retry-After``, in seconds, if it sent a usable one.

    Only the delta-seconds form is read. The HTTP-date form is legal but needs
    the provider's clock to agree with ours, and a hint derived from a skewed
    clock is worse than no hint.
    """
    raw = response.headers.get("retry-after", "").strip()
    if not raw.isdigit():
        return None
    seconds = float(raw)
    return seconds if 0 <= seconds <= 3600 else None


def _safe_snippet(response: httpx.Response, limit: int = 300) -> str:
    """A truncated body for the *log*. Never returned to a client."""
    try:
        return response.text[:limit]
    except Exception:  # pragma: no cover
        return "<unreadable body>"

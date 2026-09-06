"""Task 4: sliding-window limiting, failover mechanics, and error sanitisation.

The failover tests use ``httpx.MockTransport`` so a 429 or a hung provider is
produced deterministically rather than by timing luck, and the SQLite limiter
runs against a real file in ``tmp_path`` - not ``:memory:`` - so persistence
and eviction are genuinely exercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import statistics
import sys
import time
import uuid

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task4_model_router.app import create_app  # noqa: E402
from task4_model_router.errors import ErrorCode, GatewayError  # noqa: E402
from task4_model_router.limiter import TokenRateLimiter  # noqa: E402
from task4_model_router.providers import ProviderConfig  # noqa: E402
from task4_model_router.router import ModelRouter  # noqa: E402
from task4_model_router.tokens import (  # noqa: E402
    MAX_REPORTED_TOKENS,
    estimate_request_tokens,
)

PRIMARY_URL = "http://primary.test/v1/chat/completions"
SECONDARY_URL = "http://secondary.test/v1/chat/completions"


def providers(primary_timeout_ms=3000, secondary_timeout_ms=5000):
    return [
        ProviderConfig("primary", PRIMARY_URL, "primary-model-large", "sk-p", primary_timeout_ms, 0),
        ProviderConfig("secondary", SECONDARY_URL, "secondary-model-small", "sk-s", secondary_timeout_ms, 1),
    ]


def ok_body(provider: str, total_tokens: int = 100):
    return {
        "id": "chatcmpl-1",
        "model": f"{provider}-model",
        "provider": provider,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": total_tokens - 40, "total_tokens": total_tokens},
    }


class RecordingTransport(httpx.MockTransport):
    """A MockTransport that records which provider URLs were actually called."""

    def __init__(self, handler):
        self.calls: list[str] = []

        async def wrapper(request):
            self.calls.append(str(request.url))
            return await handler(request)

        super().__init__(wrapper)


def build_router(handler, tmp_path, limit=50_000, primary_timeout_ms=3000):
    transport = RecordingTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=30.0)
    limiter = TokenRateLimiter(tmp_path / "router.db", default_limit=limit)
    router = ModelRouter(providers(primary_timeout_ms), limiter, client)
    return router, limiter, client, transport


# --------------------------------------------------------------------------- #
# Sliding-window limiter
# --------------------------------------------------------------------------- #
async def test_reservations_accumulate_and_then_refuse(tmp_path):
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=1000)
    for _ in range(4):
        assert (await limiter.try_reserve("acme", 250)).allowed
    refused = await limiter.try_reserve("acme", 1)
    assert not refused.allowed
    assert refused.used_in_window == 1000
    assert refused.limit == 1000


async def test_tenants_are_isolated(tmp_path):
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=100)
    assert (await limiter.try_reserve("acme", 100)).allowed
    assert not (await limiter.try_reserve("acme", 1)).allowed
    assert (await limiter.try_reserve("globex", 100)).allowed, "one tenant exhausted another's budget"


async def test_window_slides_and_old_rows_are_evicted(tmp_path):
    """A 1-second window makes the sliding behaviour observable in a test."""
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=100, window_seconds=1)
    assert (await limiter.try_reserve("acme", 100)).allowed
    assert not (await limiter.try_reserve("acme", 1)).allowed

    await asyncio.sleep(1.1)

    assert (await limiter.try_reserve("acme", 100)).allowed, "the window did not slide"
    # The expired row is gone, not merely ignored - the table stays proportional
    # to the active window rather than to lifetime traffic.
    assert await limiter.row_count() == 1


async def test_a_fixed_bucket_would_allow_double_spend_but_this_does_not(tmp_path):
    """The specific burst a fixed one-minute bucket lets through.

    Spend the full budget, wait until just over half the window, and try again.
    A calendar-minute bucket would have reset; a sliding window has not.
    """
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=100, window_seconds=2)
    assert (await limiter.try_reserve("acme", 100)).allowed
    await asyncio.sleep(1.2)
    assert not (await limiter.try_reserve("acme", 100)).allowed, "double-spend across the window boundary"


async def test_settle_reconciles_the_reservation_downwards(tmp_path):
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=1000)
    decision = await limiter.try_reserve("acme", 600)
    assert await limiter.usage("acme") == 600
    await limiter.settle(decision.reservation, 120)
    assert await limiter.usage("acme") == 120, "the reservation was not reconciled to actual usage"
    # The freed headroom is immediately reusable.
    assert (await limiter.try_reserve("acme", 800)).allowed


async def test_release_returns_the_whole_reservation(tmp_path):
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=1000)
    decision = await limiter.try_reserve("acme", 900)
    await limiter.release(decision.reservation)
    assert await limiter.usage("acme") == 0
    assert (await limiter.try_reserve("acme", 1000)).allowed


async def test_concurrent_reservations_never_exceed_the_limit(tmp_path):
    """The check-then-act race, run 200 times at once.

    Without ``BEGIN IMMEDIATE`` around the read and the insert, several of
    these read the same usage figure and all decide there is room.
    """
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=10_000)
    decisions = await asyncio.gather(*[limiter.try_reserve("acme", 100) for _ in range(200)])
    granted = [d for d in decisions if d.allowed]
    assert len(granted) == 100, f"{len(granted)} reservations granted; expected exactly 100"
    assert await limiter.usage("acme") == 10_000
    assert await limiter.usage("acme") <= 10_000


async def test_state_survives_a_restart(tmp_path):
    """The reason the store is on disk.

    An in-memory counter hands a crash-looping gateway a fresh budget on every
    restart, which is unbounded spend against the provider bill.
    """
    path = tmp_path / "persist.db"
    first = TokenRateLimiter(path, default_limit=1000)
    assert (await first.try_reserve("acme", 900)).allowed

    second = TokenRateLimiter(path, default_limit=1000)  # simulates a restart
    assert await second.usage("acme") == 900
    assert not (await second.try_reserve("acme", 200)).allowed


async def test_retry_after_reflects_when_headroom_actually_returns(tmp_path):
    """A flat 60s is correct but punishing; report the real wait."""
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=100, window_seconds=10)
    await limiter.try_reserve("acme", 60)
    await asyncio.sleep(0.4)
    await limiter.try_reserve("acme", 40)

    refused = await limiter.try_reserve("acme", 50)
    assert not refused.allowed
    # The first 60-token row expires ~10s after it was created, and that alone
    # frees enough. So the wait is just under 10s, not the full window from now.
    assert 9.0 < refused.retry_after_seconds <= 10.0


async def test_request_larger_than_the_whole_budget_is_refused(tmp_path):
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=100)
    decision = await limiter.try_reserve("acme", 5_000)
    assert not decision.allowed
    assert decision.retry_after_seconds > 0


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #


def test_estimate_grows_with_prompt_size():
    short = estimate_request_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    long = estimate_request_tokens({"messages": [{"role": "user", "content": "word " * 2000}], "max_tokens": 1})
    # The completion floor is a constant in both, so compare the prompt part.
    assert long - short > 1500, "prompt size is not reflected in the reservation"


def test_estimate_handles_multimodal_and_malformed_content():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"}, {"type": "image_url"}]},
            {"role": "user", "content": None},
            "not-a-dict",
        ],
        "max_tokens": 5,
    }
    assert estimate_request_tokens(body) > 5


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #
async def test_primary_success_does_not_touch_secondary(tmp_path):
    async def handler(request):
        assert "secondary" not in str(request.url)
        return httpx.Response(200, json=ok_body("primary"))

    router, limiter, client, transport = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert result.provider == "primary"
    assert transport.calls == [PRIMARY_URL]


async def test_primary_429_fails_over_to_secondary(tmp_path):
    async def handler(request):
        if "primary" in str(request.url):
            return httpx.Response(429, json={"error": {"message": "org-8813 over limit"}})
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, transport = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()

    assert result.provider == "secondary"
    assert result.payload["provider"] == "secondary"
    assert transport.calls == [PRIMARY_URL, SECONDARY_URL]
    assert [a.outcome for a in result.attempts] == ["rate_limited", "ok"]


async def test_primary_timeout_fails_over_within_the_deadline(tmp_path):
    """The 3000 ms promise, measured.

    The primary sleeps for 10 s. The router must abandon it at its configured
    deadline and have the secondary's answer well before the primary would
    have replied.
    """
    async def handler(request):
        if "primary" in str(request.url):
            await asyncio.sleep(10)
            return httpx.Response(200, json=ok_body("primary"))
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, transport = build_router(handler, tmp_path, primary_timeout_ms=300)
    started = time.perf_counter()
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    elapsed = time.perf_counter() - started
    await client.aclose()

    assert result.provider == "secondary"
    assert [a.outcome for a in result.attempts] == ["timeout", "ok"]
    assert 0.3 <= elapsed < 1.5, f"failover took {elapsed:.2f}s; the deadline did not fire cleanly"


async def test_deadline_is_wall_clock_not_per_phase(tmp_path):
    """A provider that trickles bytes must still be cut off on time.

    httpx applies its timeouts per phase, so a slow drip resets the read timer
    indefinitely. Only the router's wall-clock deadline bounds this.
    """
    async def handler(request):
        if "primary" in str(request.url):
            async def drip():
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    yield b" "
                yield json.dumps(ok_body("primary")).encode()

            return httpx.Response(200, headers={"content-type": "application/json"}, content=drip())
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, transport = build_router(handler, tmp_path, primary_timeout_ms=400)
    started = time.perf_counter()
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    elapsed = time.perf_counter() - started
    await client.aclose()

    assert result.provider == "secondary"
    assert elapsed < 2.0, f"a trickling provider held the request for {elapsed:.2f}s"


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
async def test_server_errors_fail_over(tmp_path, status):
    async def handler(request):
        if "primary" in str(request.url):
            return httpx.Response(status, json={"error": {"message": "node p-77 crashed"}})
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, _ = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert result.provider == "secondary"


async def test_transport_errors_fail_over(tmp_path):
    async def handler(request):
        if "primary" in str(request.url):
            raise httpx.ConnectError("connection refused to 10.0.4.9:8080", request=request)
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, _ = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert result.provider == "secondary"
    assert result.attempts[0].outcome == "transport_error"


@pytest.mark.parametrize("status", [400, 422, 415])
async def test_caller_errors_do_not_fail_over(tmp_path, status):
    """Retrying the caller's own mistake on a second provider just burns quota."""
    async def handler(request):
        return httpx.Response(status, json={"error": {"message": "bad request"}})

    router, limiter, client, transport = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert transport.calls == [PRIMARY_URL], "a client error was retried on the secondary"
    # Surfaced as the caller's problem, not as a provider outage - reporting a
    # 400 as ``all_providers_failed`` sends them to the wrong place to fix it.
    assert excinfo.value.code == ErrorCode.INVALID_REQUEST
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("status", [401, 403])
async def test_provider_credential_failures_fail_over(tmp_path, status):
    """401/403 from a provider is about the GATEWAY's credential, not the
    caller's request.

    Classed as a client error, an expired or rotated key on the primary took
    the whole route down while a healthy secondary sat idle - and told the
    tenant their payload was bad, which sends them somewhere they cannot fix
    it.
    """
    async def handler(request):
        if PRIMARY_URL in str(request.url):
            return httpx.Response(status, json={"error": {"message": "invalid api key"}})
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, transport = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert result.provider == "secondary", "a provider credential failure was not failed over"
    assert result.attempts[0].outcome == "server_error"


async def test_every_provider_rate_limited_is_retryable_not_an_outage(tmp_path):
    """All-429 is capacity exhaustion, which clears. Reported as 502 the client
    reads it as a permanent failure and retries straight back into a full
    upstream."""
    async def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"Retry-After": "7"})

    router, limiter, client, _ = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    error = excinfo.value
    assert error.code == ErrorCode.UPSTREAM_RATE_LIMITED
    assert error.status_code == 429
    # The provider's own hint, propagated - never one we invented.
    assert error.to_payload()["error"]["details"]["retry_after_seconds"] == 7.0
    assert error.headers()["Retry-After"] == "7"


async def test_no_retry_after_is_invented_when_the_provider_sent_none(tmp_path):
    async def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    router, limiter, client, _ = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert excinfo.value.code == ErrorCode.UPSTREAM_RATE_LIMITED
    assert "retry_after_seconds" not in excinfo.value.to_payload()["error"].get("details", {})
    assert "Retry-After" not in excinfo.value.headers()


async def test_the_envelope_does_not_publish_null_details(tmp_path):
    """``timeout_ms`` was set to None on every non-timeout failure and the
    allow-list filter keyed on name only, so the public envelope carried
    ``"timeout_ms": null``."""
    async def handler(request):
        return httpx.Response(503, json={"error": {"message": "down"}})

    router, limiter, client, _ = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    details = excinfo.value.to_payload()["error"].get("details", {})
    assert all(v is not None for v in details.values()), f"null in the public envelope: {details}"


async def test_all_providers_down_raises_a_sanitised_error(tmp_path):
    async def handler(request):
        return httpx.Response(503, json={"error": {"message": "node p-77 down", "stack": "Traceback..."}})

    router, limiter, client, _ = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()

    error = excinfo.value
    assert error.code == ErrorCode.ALL_PROVIDERS_FAILED
    assert error.status_code == 502
    payload = json.dumps(error.to_payload())
    assert "p-77" not in payload and "Traceback" not in payload
    # The detail is preserved for the operator, just not for the client.
    assert "p-77" in error.internal_note


async def test_all_providers_timing_out_reports_a_timeout(tmp_path):
    async def handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=ok_body("primary"))

    router, limiter, client, _ = build_router(handler, tmp_path, primary_timeout_ms=200)
    router._providers[1] = ProviderConfig("secondary", SECONDARY_URL, "m", "k", 200, 1)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert excinfo.value.code == ErrorCode.UPSTREAM_TIMEOUT
    assert excinfo.value.status_code == 504


# --------------------------------------------------------------------------- #
# Limiter x router interaction
# --------------------------------------------------------------------------- #
async def test_rate_limited_request_never_calls_a_provider(tmp_path):
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary"))

    router, limiter, client, transport = build_router(handler, tmp_path, limit=100)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000})
    await client.aclose()
    assert excinfo.value.code == ErrorCode.RATE_LIMIT_EXCEEDED
    assert transport.calls == [], "a rate-limited request still hit a provider"


async def test_failed_request_does_not_consume_budget(tmp_path):
    async def handler(request):
        return httpx.Response(503, json={"error": {"message": "down"}})

    router, limiter, client, _ = build_router(handler, tmp_path)
    with pytest.raises(GatewayError):
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000})
    await client.aclose()
    assert await limiter.usage("acme") == 0, "a failed request charged the tenant"


async def test_successful_request_charges_reported_usage_not_the_estimate(tmp_path):
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=77))

    router, limiter, client, _ = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4000})
    await client.aclose()
    assert result.tokens_reserved > 4000
    assert result.tokens_charged == 77
    assert await limiter.usage("acme") == 77


async def test_failover_charges_the_provider_that_actually_answered(tmp_path):
    async def handler(request):
        if "primary" in str(request.url):
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=ok_body("secondary", total_tokens=55))

    router, limiter, client, _ = build_router(handler, tmp_path)
    await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2000})
    await client.aclose()
    assert await limiter.usage("acme") == 55, "failover double-charged or charged the estimate"


async def test_client_disconnect_releases_the_reservation(tmp_path):
    """Cancellation must not leave a hold behind, or budget leaks on every hangup."""
    async def handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=ok_body("primary"))

    router, limiter, client, _ = build_router(handler, tmp_path)
    task = asyncio.create_task(
        router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3000})
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    assert await limiter.usage("acme") == 0, "a cancelled request leaked its reservation"


# --------------------------------------------------------------------------- #
# The limit must actually bound spend
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": {"text": "PROMPT"}}], "max_tokens": 10},
        {"messages": [{"role": "user", "content": ["PROMPT"]}], "max_tokens": 10},
        {"messages": [{"role": "user", "content": [{"type": "x", "value": "PROMPT"}]}], "max_tokens": 10},
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{"schema": "PROMPT"}], "max_tokens": 10},
        {"messages": [{"role": "user", "content": "hi"}], "system": "PROMPT", "max_tokens": 10},
    ],
)
def test_the_estimator_cannot_be_evaded_by_content_shape(body):
    """A big prompt must cost a big reservation whatever shape it arrives in.

    The previous version counted only ``str`` content and returned zero for
    everything else, so a 60,001-token prompt sent as a dict - or as a list of
    bare strings, or as parts keyed anything but "text" - was estimated at 104.
    The limit was gameable by 577x with a perfectly well-formed request.
    """
    filled = json.loads(json.dumps(body).replace("PROMPT", "word " * 20000))
    estimate = estimate_request_tokens(filled)
    assert estimate > 15_000, f"a ~20k-token prompt was estimated at {estimate}"


def test_n_multiplies_the_completion_reservation():
    single = estimate_request_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000})
    twenty = estimate_request_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000, "n": 20})
    assert twenty >= single * 19, "n was ignored - a straight 20x under-reservation"


def test_max_completion_tokens_is_honoured():
    """The current OpenAI field name; honouring only ``max_tokens`` meant a
    modern client silently fell back to the 512 default."""
    body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 4000}
    assert estimate_request_tokens(body) >= 4000


async def test_an_over_limit_tenant_does_not_slow_other_tenants(tmp_path):
    """The failure mode used to be inverted.

    Computing the retry-after hint inside the exclusive write transaction made
    the *denial* path ~10x more expensive than the allow path while holding the
    global write lock, so one tenant hitting its limit took an unrelated tenant
    from 7ms to 7,200ms and produced 500s out of the limiter itself.
    """
    limiter = TokenRateLimiter(tmp_path / "l.db", default_limit=50_000)
    connection = sqlite3.connect(tmp_path / "l.db")
    now = int(time.time() * 1000)
    connection.executemany(
        "INSERT INTO token_usage (id, tenant_key, created_ms, tokens, settled) VALUES (?, ?, ?, 1, 1)",
        [(uuid.uuid4().hex, "acme", now) for _ in range(20_000)],
    )
    connection.commit()
    connection.close()

    async def probe(count):
        latencies = []
        for _ in range(15):
            start = time.perf_counter()
            await limiter.try_reserve("globex", 10)
            latencies.append((time.perf_counter() - start) * 1000)
        return latencies

    baseline = statistics.median(await probe(15))

    errors = []

    async def flood():
        for _ in range(150):
            try:
                await limiter.try_reserve("acme", 100)
            except Exception as exc:  # noqa: BLE001 - the point is that there are none
                errors.append(type(exc).__name__)

    tasks = [asyncio.create_task(flood()) for _ in range(4)]
    under_load = statistics.median(await probe(15))
    await asyncio.gather(*tasks)

    assert not errors, f"the limiter raised under contention: {errors[:3]}"
    assert under_load < baseline * 20 + 50, (
        f"an over-limit tenant degraded another from {baseline:.1f}ms to {under_load:.1f}ms"
    )


async def test_implausible_reported_usage_does_not_crash_the_request(tmp_path):
    """``total_tokens >= 2**63`` raised OverflowError out of the sqlite driver,
    turning a completion the tenant had already paid for into a 500."""
    async def absurd(request):
        return httpx.Response(200, json={
            "id": "x", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 2**63 + 5},
        })

    router, limiter, client, _ = build_router(absurd, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    await client.aclose()
    # Clamped to a plausible ceiling rather than crashing, and the tenant is
    # charged that rather than nothing.
    assert 0 < result.tokens_charged <= MAX_REPORTED_TOKENS
    assert await limiter.usage("acme") == result.tokens_charged


@pytest.mark.parametrize("bad_usage", [{"total_tokens": -5}, {"total_tokens": 12.5}, {"total_tokens": True}, {}])
async def test_malformed_usage_falls_back_to_the_reservation(tmp_path, bad_usage):
    async def odd(request):
        return httpx.Response(200, json={
            "id": "x", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": bad_usage,
        })

    router, limiter, client, _ = build_router(odd, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    await client.aclose()
    assert result.tokens_charged == result.tokens_reserved


async def test_settled_spend_is_not_refunded_by_a_late_cancellation(tmp_path):
    """A cancellation arriving after settle() began also ran release(), which
    deleted the row - so a client that hung up at the right moment got its
    completion for free."""
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=200))

    router, limiter, client, _ = build_router(handler, tmp_path)
    original_settle = limiter.settle

    async def slow_settle(reservation, actual):
        await asyncio.sleep(0.2)  # widen the real to_thread window
        await original_settle(reservation, actual)

    limiter.settle = slow_settle
    task = asyncio.create_task(
        router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 500})
    )
    await asyncio.sleep(0.1)  # inside settle
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)  # let the shielded settle finish
    await client.aclose()

    assert await limiter.usage("acme") > 0, "settled spend was refunded by a cancellation"


# --------------------------------------------------------------------------- #
# Upstream responses the router must not trust
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_redirects_are_not_treated_as_success(tmp_path, status):
    """A 3xx fell through every branch and was treated as SUCCESS: the body was
    proxied to the client as a 200, internal hostname and all, and the request
    silently went nowhere (httpx does not follow redirects)."""
    async def redirect(request):
        return httpx.Response(status, json={"id": "redirected", "usage": {"total_tokens": 5},
                                            "secret": "internal-host-10.0.3.17"})

    router, limiter, client, _ = build_router(redirect, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert "10.0.3.17" not in json.dumps(excinfo.value.to_payload())


async def test_a_genuine_idle_period_still_expires_rows(tmp_path):
    """The step detection must not turn into "never expire anything"."""
    limiter = TokenRateLimiter(tmp_path / "idle.db", default_limit=100, window_seconds=1)
    assert (await limiter.try_reserve("acme", 100)).allowed
    await asyncio.sleep(1.1)
    assert (await limiter.try_reserve("acme", 100)).allowed, "the window stopped sliding"


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
@pytest.fixture
async def api(tmp_path):
    # ``total_tokens`` is what the provider *reports*; the limiter settles the
    # reservation down to it, so it is what actually accrues against the window.
    state = {"primary_status": 200, "secondary_status": 200, "total_tokens": 100}

    def respond(name, status):
        if status == 429:
            return httpx.Response(429, json={"error": {"message": "org-8813 over limit"}})
        if status >= 500:
            return httpx.Response(status, json={"error": {"stack": "Traceback ...", "node": "p-77"}})
        return httpx.Response(200, json=ok_body(name, total_tokens=state["total_tokens"]))

    async def handler(request):
        if "primary" in str(request.url):
            return respond("primary", state["primary_status"])
        return respond("secondary", state["secondary_status"])

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
    app = create_app(
        db_path=str(tmp_path / "api.db"),
        providers=providers(),
        client=upstream,
        default_limit=50_000,
        api_keys={"sk-tenant-acme-001": "acme", "sk-tenant-globex-002": "globex"},
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        client.state = state
        yield client
    await upstream.aclose()


AUTH = {"Authorization": "Bearer sk-tenant-acme-001"}


async def test_end_to_end_success(api):
    response = await api.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gateway"]["served_by"] == "primary"
    assert body["gateway"]["tokens_charged"] > 0
    assert response.headers["x-request-id"] == body["gateway"]["request_id"]


async def test_end_to_end_failover_is_visible_to_the_caller(api):
    api.state["primary_status"] = 429
    response = await api.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}, headers=AUTH
    )
    body = response.json()
    assert body["gateway"]["served_by"] == "secondary"
    assert [a["outcome"] for a in body["gateway"]["attempts"]] == ["rate_limited", "ok"]


async def test_end_to_end_rate_limit_envelope(api):
    """Drive traffic until the budget is exhausted, then check the refusal.

    Written as "loop until refused" rather than a fixed request count so it
    asserts the behaviour, not a brittle sum of estimator constants.
    """
    api.state["total_tokens"] = 12_000
    big = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 12_000}

    response = None
    for _ in range(10):
        response = await api.post("/v1/chat/completions", json=big, headers=AUTH)
        if response.status_code == 429:
            break
    else:
        pytest.fail("the limiter never refused a request")

    assert response.status_code == 429
    error = response.json()["error"]
    assert error["code"] == ErrorCode.RATE_LIMIT_EXCEEDED
    assert error["message"] == "Token rate limit exceeded for this API key."
    assert error["details"]["limit_tokens_per_minute"] == 50_000
    assert error["details"]["requested_tokens"] >= 12_000
    assert error["details"]["tokens_used_in_window"] > 0
    assert "Retry-After" in response.headers
    assert int(response.headers["Retry-After"]) >= 1


async def test_upstream_failure_envelope_leaks_nothing(api):
    api.state["primary_status"] = 503
    api.state["secondary_status"] = 503
    response = await api.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]}, headers=AUTH
    )
    assert response.status_code in (502, 504)
    text = response.text
    assert "Traceback" not in text
    assert "primary.test" not in text and "secondary.test" not in text
    assert set(response.json()["error"]) <= {"code", "message", "request_id", "details"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}])
async def test_unauthenticated_requests_are_refused(api, headers):
    response = await api.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}, headers=headers
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == ErrorCode.UNAUTHENTICATED


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": "hi"}, []])
async def test_invalid_bodies_are_refused(api, body):
    response = await api.post("/v1/chat/completions", json=body, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST


async def test_malformed_json_is_refused(api):
    response = await api.post(
        "/v1/chat/completions", content="{oops", headers={**AUTH, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "oops" not in response.text


async def test_tenants_do_not_share_a_budget_over_http(api):
    api.state["total_tokens"] = 12_000
    big = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 12_000}
    other = {"Authorization": "Bearer sk-tenant-globex-002"}

    for _ in range(10):
        if (await api.post("/v1/chat/completions", json=big, headers=AUTH)).status_code == 429:
            break
    else:
        pytest.fail("the limiter never refused acme")

    # globex has spent nothing, so its budget must be untouched.
    assert (await api.post("/v1/chat/completions", json=big, headers=other)).status_code == 200


# --------------------------------------------------------------------------- #
# Error envelope unit tests
# --------------------------------------------------------------------------- #
def test_details_are_allow_listed():
    error = GatewayError(
        ErrorCode.RATE_LIMIT_EXCEEDED,
        "req-1",
        {"limit_tokens_per_minute": 100, "upstream_host": "10.0.0.5", "stack": "Traceback"},
    )
    details = error.to_payload()["error"]["details"]
    assert details == {"limit_tokens_per_minute": 100}


def test_internal_note_is_never_serialised():
    error = GatewayError(ErrorCode.INTERNAL_ERROR, "req-2", internal_note="db password=hunter2")
    assert "hunter2" not in json.dumps(error.to_payload())


def test_retry_after_header_rounds_up():
    error = GatewayError(ErrorCode.RATE_LIMIT_EXCEEDED, "r", {"retry_after_seconds": 0.2})
    assert error.headers()["Retry-After"] == "1"
    error = GatewayError(ErrorCode.RATE_LIMIT_EXCEEDED, "r", {"retry_after_seconds": 12.4})
    assert error.headers()["Retry-After"] == "13"


# --------------------------------------------------------------------------- #
# Round-two findings
# --------------------------------------------------------------------------- #
def test_the_completion_ceiling_is_a_ceiling_not_a_guess():
    """``max_tokens`` absent means the model runs to *its* limit, not to 512."""
    body = {"messages": [{"role": "user", "content": "summarise this"}]}
    assert estimate_request_tokens(body) >= 4096


async def test_cancelling_during_the_reserve_does_not_strand_the_budget(tmp_path):
    """The ``to_thread`` hop can be cancelled after sqlite COMMITs the
    reservation but before the router receives it, leaving a hold with no id to
    settle or release - 4% of cancelled requests, and weaponisable to strand a
    tenant's whole budget at no cost."""
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=50))

    router, limiter, client, _ = build_router(handler, tmp_path, limit=50_000)
    original = limiter.try_reserve

    async def slow_reserve(tenant, tokens):
        await asyncio.sleep(0.15)
        return await original(tenant, tokens)

    limiter.try_reserve = slow_reserve
    for _ in range(20):
        task = asyncio.create_task(
            router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000})
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(0.5)   # let the shielded reserves finish and release
    await client.aclose()

    assert await limiter.usage("acme") == 0, "cancelled requests stranded budget"


async def test_estimation_does_not_block_the_event_loop(tmp_path):
    """Estimation now walks and serialises every content shape, and runs BEFORE
    any limit - so on the loop one tenant's 8MB body took every other tenant
    from 8ms to 13,888ms, for free, because their own request was then 429'd."""
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=50))

    router, limiter, client, _ = build_router(handler, tmp_path, limit=10_000_000)
    huge = {"messages": [{"role": "user", "content": "a" * (2 * 1024 * 1024)}], "max_tokens": 10}
    small = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}

    baseline = time.perf_counter()
    await router.route("globex", dict(small))
    baseline = time.perf_counter() - baseline

    heavy = asyncio.create_task(router.route("acme", huge))
    await asyncio.sleep(0)
    started = time.perf_counter()
    await router.route("globex", dict(small))
    under_load = time.perf_counter() - started
    with contextlib.suppress(Exception):
        await heavy
    await client.aclose()

    assert under_load < baseline + 0.5, (
        f"an unrelated tenant went from {baseline * 1000:.0f}ms to {under_load * 1000:.0f}ms"
    )


async def test_concurrent_requests_cannot_exceed_the_limit(tmp_path):
    """The check-then-act race, which is the one a rate limiter must not lose.

    Issued sequentially, any limiter looks correct. Issued all at once, one
    that reads "current usage" and then inserts lets every request see the same
    room and proceed. The reservation is taken inside the same transaction as
    the check, so admission is bounded even when 2,000 requests arrive
    together.
    """
    async def honest(request):
        # Bills within the ``max_tokens`` it was given, so the reservation is a
        # genuine ceiling on this request and the bound is the limiter's alone.
        return httpx.Response(200, json=ok_body("primary", total_tokens=100))

    router, limiter, client, _ = build_router(honest, tmp_path, limit=50_000)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    results = await asyncio.gather(
        *[router.route("acme", dict(body)) for _ in range(2000)], return_exceptions=True
    )
    await client.aclose()

    served = sum(1 for r in results if not isinstance(r, Exception))
    real_spend = served * 100
    assert real_spend <= 50_000, f"{served} concurrent requests spent {real_spend:,} against 50,000"
    assert await limiter.usage("acme") == real_spend, "the ledger disagrees with real spend"
    refused = [r for r in results if isinstance(r, GatewayError)]
    assert refused, "nothing was refused, so the limit was never reached"
    assert all(r.code == ErrorCode.RATE_LIMIT_EXCEEDED for r in refused)


def test_the_estimator_walks_the_whole_body():
    """``_call_provider`` forwards ``{**body}``, so a field allow-list was the
    wrong shape: 405 KB parked in ``context_documents`` estimated at 21."""
    small = estimate_request_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})
    with_payload = estimate_request_tokens({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "context_documents": ["word " * 20000],
    })
    assert with_payload - small > 15_000, "text outside the known fields was not counted"


def test_the_estimate_is_never_below_what_goes_on_the_wire():
    """The floor that closes the evasion class.

    The structural walk was evaded four separate times - text in an unwalked
    field, text nested past the depth guard, text in a dict KEY, and text under
    a skip-listed name - each fixed as an instance while the next shape stayed
    open. ``_call_provider`` forwards ``{**body}``, so the estimate can never be
    below the serialised size of what it forwards, whatever shape the text
    arrives in.

    This deliberately reverses an earlier contract: configuration fields used to
    be exempt so a long ``model`` name was not billed as prompt. Over-counting a
    config field wastes a little of a tenant's own budget; under-counting one
    handed out 300x the limit.
    """
    import json as _json

    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    payload = "C" * 200_000

    shapes = {
        "plain content": {"messages": [{"role": "user", "content": payload}], "max_tokens": 1},
        "unwalked field": {**base, "context_documents": [payload]},
        "dict key": {**base, "ctx": {payload: 1}},
        "skip-listed name": {**base, "temperature": payload},
        "another skip-listed": {**base, "seed": payload},
        "nested past the guard": {**base, "ctx": _nest_dicts(payload, 25)},
    }
    for label, body in shapes.items():
        estimate = estimate_request_tokens(body)
        floor = len(_json.dumps(body)) // 8  # deliberately loose
        assert estimate > floor, f"{label}: {estimate} tokens for {len(_json.dumps(body))} bytes"


@pytest.mark.parametrize(
    "body",
    [
        {"metadata": {"note": "word " * 100_000}},
        {"user": "word " * 100_000},
        {"stop": ["word " * 100_000]},
        {"logit_bias": {"k": "word " * 100_000}},
        {"messages": [{"role": "user", "content": "hi", "metadata": {"n": "word " * 100_000}}]},
    ],
)
def test_a_field_the_gateway_forwards_cannot_hide_from_the_estimator(body):
    """The skip-list was applied at every depth and included keys that carry
    arbitrary caller content. ``_call_provider`` forwards ``{**body}``, so 500 KB
    parked in ``metadata`` was billed at 16 tokens and arrived upstream in full.

    The rule the estimator states: anything the gateway forwards, it must count.
    """
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    plain = estimate_request_tokens(base)
    hidden = estimate_request_tokens({**base, **body})
    assert hidden - plain > 50_000, f"{list(body)[0]} carried 500 KB uncounted"


async def test_a_generated_but_unusable_response_is_still_charged(tmp_path):
    """Three paths released the whole reservation after a provider had already
    generated: a streaming body, an oversized response, and a non-object JSON
    body. Free completions, invisible in the ledger - round one's silent
    overspend, reintroduced somewhere else.
    """
    async def sse_body(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"data: {}\n\n")

    router, limiter, client, _ = build_router(sse_body, tmp_path, limit=500_000)
    with pytest.raises(GatewayError):
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100})
    await client.aclose()
    assert await limiter.usage("acme") > 0, "a generated completion was not charged"


async def test_a_rate_limited_provider_is_not_charged(tmp_path):
    """The counterpart: a 429 means nothing was generated, so nothing is owed."""
    async def refuses(request):
        return httpx.Response(429, json={"error": {"message": "busy"}})

    router, limiter, client, _ = build_router(refuses, tmp_path, limit=500_000)
    with pytest.raises(GatewayError):
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100})
    await client.aclose()
    assert await limiter.usage("acme") == 0, "a refused request was charged"


async def test_the_success_path_does_not_relay_provider_internals(tmp_path):
    """Sanitisation was complete on the error path and entirely absent on the
    success path: a 200 carrying a stack trace, an internal endpoint and a
    provider API key was returned to the client unchanged."""
    async def leaky_success(request):
        payload = ok_body("primary", total_tokens=50)
        payload["error"] = {"stack": 'Traceback: File "/srv/provider/app.py"'}
        payload["endpoint"] = "http://10.0.0.7:8443/internal"
        payload["api_key"] = "sk-provider-REAL"
        return httpx.Response(200, json=payload)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(leaky_success), timeout=30.0)
    app = create_app(db_path=str(tmp_path / "s.db"), providers=providers(), client=upstream,
                     default_limit=500_000, api_keys={"sk-tenant-acme-001": "acme"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post("/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": "hi"}]},
                                 headers={"Authorization": "Bearer sk-tenant-acme-001"})
    await upstream.aclose()
    assert response.status_code == 200
    for secret in ("Traceback", "/srv/provider", "10.0.0.7", "sk-provider-REAL"):
        assert secret not in response.text, f"{secret!r} reached the client on the success path"


async def test_streaming_requests_are_refused_not_silently_failed(tmp_path):
    """`stream:true` produced a 502 "no provider could serve this" AFTER both
    providers had generated and billed."""
    calls = []

    async def counting(request):
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: {}\n\n")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(counting), timeout=30.0)
    app = create_app(db_path=str(tmp_path / "st.db"), providers=providers(), client=upstream,
                     default_limit=500_000, api_keys={"sk-tenant-acme-001": "acme"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post("/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                                 headers={"Authorization": "Bearer sk-tenant-acme-001"})
    await upstream.aclose()
    assert response.status_code == 400
    assert calls == [], "providers were dialled for a request that could never be served"


async def test_an_oversized_body_is_refused_before_tokenisation(tmp_path):
    """Estimation walks and serialises the whole body and runs BEFORE the rate
    limit, so an unbounded body is server work an attacker gets for free: the
    429 that follows records ``tokens_used_in_window: 0``."""
    from task4_model_router import app as app_module

    async def handler(request):
        return httpx.Response(200, json=ok_body("primary"))

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    application = create_app(
        db_path=str(tmp_path / "router.db"), providers=providers(), client=client
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://gw"
    ) as gateway_client:
        oversized = "A" * (app_module.MAX_REQUEST_BYTES + 1024)
        response = await gateway_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": oversized}]},
            headers={"Authorization": "Bearer sk-tenant-acme-001"},
        )
    await client.aclose()
    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST


async def test_an_oversized_chunked_body_is_refused_as_it_arrives(tmp_path):
    """A chunked upload declares no Content-Length, so the cap has to hold as
    the bytes arrive rather than after they are all in memory."""
    from task4_model_router import app as app_module

    async def handler(request):
        return httpx.Response(200, json=ok_body("primary"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = create_app(
        db_path=str(tmp_path / "router.db"), providers=providers(), client=client
    )

    async def chunked():
        yield b'{"messages":[{"role":"user","content":"'
        for _ in range((app_module.MAX_REQUEST_BYTES // 65536) + 4):
            yield b"A" * 65536
        yield b'"}]}'

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://gw"
    ) as gateway_client:
        response = await gateway_client.post(
            "/v1/chat/completions",
            content=chunked(),
            headers={"Authorization": "Bearer sk-tenant-acme-001"},
        )
    await client.aclose()
    assert response.status_code == 400, f"a chunked oversized body was accepted: {response.status_code}"
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST


async def test_method_errors_use_the_standard_envelope(tmp_path):
    """Starlette's default 405 body has no ``code`` and no ``request_id``,
    contradicting the one-error-shape promise the module makes."""
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = create_app(
        db_path=str(tmp_path / "router.db"), providers=providers(), client=client
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://gw"
    ) as gateway_client:
        response = await gateway_client.get("/v1/chat/completions")
    await client.aclose()
    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == ErrorCode.INVALID_REQUEST
    assert body["error"]["request_id"]
    assert "detail" not in body


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda text: {"ctx": text}, id="flat"),
        pytest.param(
            lambda text: {"ctx": _nest_dicts(text, 25)}, id="nested-dicts-25"
        ),
        pytest.param(
            lambda text: {"ctx": _nest_lists(text, 25)}, id="nested-lists-25"
        ),
    ],
)
def test_nesting_depth_cannot_hide_text_from_the_estimator(wrap):
    """The depth guard used to ``return 0``.

    The evasion test beside this one varies the key *name* and never the
    nesting *depth*, so it could not see this: 200 KB wrapped in 25 dicts
    estimated at 16 tokens, and because the limit is checked before any
    provider call, ten concurrent requests all passed at 16 and sent 1,038,488
    tokens against a 50,000/minute budget - a 20.8x breach of the headline
    requirement. Past the depth limit the remainder is measured in bulk.
    """
    text = "word " * 40_000  # 200 KB
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    flat = estimate_request_tokens({**base, "ctx": text})
    hidden = estimate_request_tokens({**base, **wrap(text)})
    assert flat > 20_000, f"the corpus is too small to prove anything: {flat}"
    # Measured against the honest count, not a magic threshold: burying the
    # same text deeper must not make it meaningfully cheaper.
    assert hidden > flat * 0.5, f"deep text counted at {hidden} vs flat {flat}"


def _nest_dicts(value, depth):
    for _ in range(depth):
        value = {"x": value}
    return value


def _nest_lists(value, depth):
    for _ in range(depth):
        value = [value]
    return value


def test_numbers_are_counted_because_they_are_forwarded_and_billed():
    """300,000 floats is 2.9 MB on the wire, forwarded verbatim, and was
    estimated at 16 tokens - numbers were skipped as "not text"."""
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    numbers = [round(index * 0.000001, 6) for index in range(300_000)]
    estimate = estimate_request_tokens({**base, "embeddings": numbers})
    # ~2.9 MB on the wire; anything near the 16-token baseline means the
    # payload was skipped rather than counted.
    assert estimate > 100_000, f"2.9MB of numbers estimated at {estimate} tokens"


async def test_a_lying_provider_cannot_write_an_arbitrary_charge(tmp_path):
    """A flat 10,000,000 ceiling was 200x a 50,000 budget, so one malformed
    ``usage`` block reported a number the budget cannot express back to the
    client as ``tokens_charged``."""
    async def liar(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=9_223_372_036_854_775_807))

    router, limiter, client, _ = build_router(liar, tmp_path, limit=50_000)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    used = await limiter.usage("acme")
    await client.aclose()

    assert result.tokens_charged <= 100_000, f"charged {result.tokens_charged:,}"
    assert used == result.tokens_charged, "the ledger and the receipt disagree"


async def test_eviction_survives_a_refusal(tmp_path):
    """Eviction ran inside the reserve transaction, so a refusal rolled it back
    - and a tenant sitting at its limit is exactly when the table is largest."""
    limiter = TokenRateLimiter(str(tmp_path / "router.db"), default_limit=1_000)
    stale = limiter._now_ms() - 120_000
    connection = limiter._connect()
    try:
        for index in range(50):
            connection.execute(
                "INSERT INTO token_usage (id, tenant_key, created_ms, tokens, settled) "
                "VALUES (?, 'acme', ?, 10, 1)",
                (f"stale-{index}", stale),
            )
    finally:
        connection.close()
    assert await limiter.row_count() == 50

    # Fill the window, then make a request that must be refused.
    decision = await limiter.try_reserve("acme", 1_000)
    assert decision.allowed
    refused = await limiter.try_reserve("acme", 1_000)
    assert not refused.allowed

    assert await limiter.row_count() == 1, "the refusal rolled back its own eviction"


async def test_a_settle_whose_row_was_evicted_still_records_the_charge(tmp_path):
    """A request that outlives the window has its reservation row evicted
    first, so the settle UPDATE matched nothing - the charge vanished while the
    client was still told what it had been charged."""
    limiter = TokenRateLimiter(str(tmp_path / "router.db"), default_limit=50_000)
    decision = await limiter.try_reserve("acme", 500)
    assert decision.allowed
    reservation = decision.reservation

    # Evict it, exactly as the window would.
    connection = limiter._connect()
    try:
        connection.execute("DELETE FROM token_usage WHERE id = ?", (reservation.id,))
    finally:
        connection.close()
    assert await limiter.usage("acme") == 0

    charged = await limiter.settle(reservation, 320)
    assert charged == 320
    assert await limiter.usage("acme") == 320, "the settled charge was lost"


async def test_the_budget_is_keyed_on_the_api_key(tmp_path):
    """The brief says "per tenant API key". Two keys issued to one tenant
    therefore get a budget each, not a shared one."""
    async def handler(request):
        return httpx.Response(200, json=ok_body("primary", total_tokens=400))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = create_app(
        db_path=str(tmp_path / "router.db"),
        providers=providers(),
        client=client,
        # 410 is chosen so one settled response (400) plus the next
        # reservation (~16) exceeds it, and a single request does not.
        default_limit=410,
        api_keys={"key-one": "acme", "key-two": "acme"},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://gw"
    ) as gateway:
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        first = await gateway.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-one"}
        )
        # Same tenant, different key: must not be refused on the first key's spend.
        second = await gateway.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-two"}
        )
        # ...but the first key's own budget is now gone.
        third = await gateway.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer key-one"}
        )
    await client.aclose()
    assert first.status_code == 200
    assert second.status_code == 200, "a second key shared the first key's budget"
    assert third.status_code == 429, "the first key's own budget was not enforced"


def test_the_structural_walk_counts_dict_keys():
    """Isolated from the serialised floor, which would otherwise mask this.

    ``_walk_text`` iterated ``value.items()`` and recursed only into the value,
    so 3.9 MB of text in a KEY measured 8 tokens against 487,507 for the same
    text one position to the right. The floor now backstops it either way, but
    the walk should be right on its own - the floor is the safety net, not the
    mechanism.
    """
    from task4_model_router.tokens import _walk_text

    payload = "A" * 100_000
    in_value = _walk_text({"ctx": {"k": payload}})
    in_key = _walk_text({"ctx": {payload: 1}})
    # Measured against the same text one position to the right, not a magic
    # number - the tokeniser packs repeated characters, so an absolute
    # threshold here is a guess that fails for the wrong reason.
    assert in_value > 1_000, f"the fixture is too small to prove anything: {in_value}"
    assert in_key >= in_value * 0.9, f"text in a key measured {in_key} vs {in_value} in a value"


@pytest.mark.parametrize("status", [401, 402, 403, 404, 413])
async def test_provider_side_4xx_fails_over_instead_of_blaming_the_caller(tmp_path, status):
    """Whose fault the failure is, not which class the number is in.

    402 is the gateway's billing account with that provider. 404 is a wrong URL
    or an unknown model - and the router rewrites ``model`` per provider, so
    that name is the gateway's choice. 413 is a body over THIS provider's limit,
    which a larger secondary may accept. Each was reported to the tenant as
    HTTP 400 "The request payload was invalid" with the secondary never dialled.
    """
    async def handler(request):
        if PRIMARY_URL in str(request.url):
            return httpx.Response(status, json={"error": {"message": "provider side"}})
        return httpx.Response(200, json=ok_body("secondary"))

    router, limiter, client, _ = build_router(handler, tmp_path)
    result = await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert result.provider == "secondary", f"{status} did not fail over"
    assert [a.outcome for a in result.attempts] == ["server_error", "ok"]


@pytest.mark.parametrize("status", [400, 415, 422])
async def test_a_genuine_caller_error_still_does_not_burn_a_second_quota(tmp_path, status):
    """The other half of the rule: a payload a second provider would reject
    identically must not be retried on a second quota."""
    async def handler(request):
        return httpx.Response(status, json={"error": {"message": "your payload"}})

    router, limiter, client, transport = build_router(handler, tmp_path)
    with pytest.raises(GatewayError) as excinfo:
        await router.route("acme", {"messages": [{"role": "user", "content": "hi"}]})
    await client.aclose()
    assert transport.calls == [PRIMARY_URL], "a caller error was retried on the secondary"
    assert excinfo.value.code == ErrorCode.INVALID_REQUEST


async def test_the_reserved_ceiling_is_imposed_on_the_provider(tmp_path):
    """The reservation has to be enforceable, not aspirational.

    The router reserved DEFAULT_COMPLETION_CEILING for a request that omits
    ``max_tokens`` and then forwarded no cap, so the provider ran to its own
    default (8192-16384 on current models). 40 ordinary concurrent requests
    spent 196,848 tokens against a 50,000/minute budget - 3.94x - against an
    upstream behaving perfectly correctly.
    """
    from task4_model_router.tokens import DEFAULT_COMPLETION_CEILING

    sent = []

    async def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body("primary", total_tokens=10))

    router, limiter, client, _ = build_router(handler, tmp_path)
    await router.route("acme", {"messages": [{"role": "user", "content": "Summarise."}]})
    await router.route(
        "acme", {"messages": [{"role": "user", "content": "Summarise."}], "max_tokens": 50}
    )
    await client.aclose()

    assert sent[0]["max_tokens"] == DEFAULT_COMPLETION_CEILING, (
        "no cap was imposed, so the reservation cannot bind the provider"
    )
    assert sent[1]["max_tokens"] == 50, "the caller's own max_tokens was overwritten"


@pytest.mark.parametrize("field", ["max_completion_tokens", "max_output_tokens"])
async def test_a_callers_alternative_ceiling_field_is_respected(tmp_path, field):
    """Newer field names bound the completion just as well; overwriting them
    with our default would silently change what the caller asked for."""
    sent = []

    async def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body("primary", total_tokens=10))

    router, limiter, client, _ = build_router(handler, tmp_path)
    await router.route(
        "acme", {"messages": [{"role": "user", "content": "hi"}], field: 64}
    )
    await client.aclose()
    assert sent[0][field] == 64
    assert "max_tokens" not in sent[0], f"{field} was set but max_tokens was added anyway"


async def test_the_limit_holds_when_the_caller_omits_max_tokens(tmp_path):
    """End to end: a provider that honours whatever cap it is given must not be
    able to spend past the budget when the caller sets none."""
    async def handler(request):
        body = json.loads(request.content)
        # An honest provider: generates exactly up to the cap it was given.
        return httpx.Response(200, json=ok_body("primary", total_tokens=body["max_tokens"]))

    router, limiter, client, _ = build_router(handler, tmp_path, limit=50_000)
    results = await asyncio.gather(
        *[router.route("acme", {"messages": [{"role": "user", "content": "Summarise."}]})
          for _ in range(60)],
        return_exceptions=True,
    )
    await client.aclose()
    served = [r for r in results if not isinstance(r, Exception)]
    spend = sum(r.tokens_charged for r in served)
    assert spend <= 50_000, f"{len(served)} requests spent {spend:,} against 50,000"
    assert await limiter.usage("acme") == spend

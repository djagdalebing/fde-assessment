"""Upstream model providers and the mock used to exercise the router."""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class ProviderConfig:
    name: str
    url: str
    model: str
    api_key: str = ""
    #: Hard deadline for one attempt against this provider, in milliseconds.
    timeout_ms: int = 3000
    #: Providers are tried in ascending priority order.
    priority: int = 0


@dataclass
class ProviderAttempt:
    """What happened on one attempt. Used for logs and the error envelope."""

    provider: str
    outcome: str  # "ok" | "rate_limited" | "timeout" | "server_error" | "transport_error"
    status_code: int | None = None
    duration_ms: float = 0.0
    #: Operator-only. Never serialised into a client response.
    internal_detail: str = ""
    #: Seconds from the provider's own ``Retry-After`` on a 429, when it sent
    #: one. Never fabricated - a backoff hint we invented would be worse than
    #: none at all.
    retry_after_seconds: float | None = None


def default_providers() -> list[ProviderConfig]:
    return [
        ProviderConfig(
            name="primary",
            url=os.environ.get("PRIMARY_PROVIDER_URL", "http://127.0.0.1:9004/v1/chat/completions"),
            model=os.environ.get("PRIMARY_MODEL", "primary-model-large"),
            api_key=os.environ.get("PRIMARY_API_KEY", "sk-primary"),
            timeout_ms=int(os.environ.get("PRIMARY_TIMEOUT_MS", "3000")),
            priority=0,
        ),
        ProviderConfig(
            name="secondary",
            url=os.environ.get("SECONDARY_PROVIDER_URL", "http://127.0.0.1:9005/v1/chat/completions"),
            model=os.environ.get("SECONDARY_MODEL", "secondary-model-small"),
            api_key=os.environ.get("SECONDARY_API_KEY", "sk-secondary"),
            timeout_ms=int(os.environ.get("SECONDARY_TIMEOUT_MS", "5000")),
            priority=1,
        ),
    ]


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #
@dataclass
class MockBehaviour:
    """Knobs the tests use to make a provider misbehave on demand."""

    status: int = 200
    delay_seconds: float = 0.0
    name: str = "mock"
    fail_times: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)


def make_mock_provider(behaviour: MockBehaviour) -> FastAPI:
    app = FastAPI(title=f"mock-provider-{behaviour.name}")

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        behaviour.calls.append({"model": body.get("model"), "at": time.time()})

        if behaviour.delay_seconds:
            await asyncio.sleep(behaviour.delay_seconds)

        status = behaviour.status
        if behaviour.fail_times > 0:
            behaviour.fail_times -= 1
            status = behaviour.status if behaviour.status >= 400 else 500

        if status == 429:
            # A realistic upstream 429: chatty, and carrying account internals
            # the gateway must not pass on.
            return JSONResponse(
                status_code=429,
                headers={"retry-after": "20"},
                content={
                    "error": {
                        "message": (
                            "Rate limit reached for org-8813 on requests per min. "
                            "Contact ops@provider-internal.example"
                        ),
                        "type": "requests",
                        "internal_trace": "File \"/srv/provider/app/limits.py\", line 214, in check",
                    }
                },
            )
        if status >= 500:
            return JSONResponse(
                status_code=status,
                content={"error": {
                    "message": "upstream node p-77 crashed",
                    "stack": "Traceback (most recent call last): ...",
                }},
            )

        completion_tokens = random.randint(20, 60)
        return JSONResponse(
            {
                "id": f"chatcmpl-{behaviour.name}",
                "object": "chat.completion",
                "model": body.get("model", "mock"),
                "provider": behaviour.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"Reply from {behaviour.name}."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 30 + completion_tokens,
                },
            }
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "provider": behaviour.name}

    return app

"""HTTP surface for the model router."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from task4_model_router.errors import ErrorCode, GatewayError
from task4_model_router.limiter import DEFAULT_LIMIT_TOKENS_PER_MINUTE, TokenRateLimiter
from task4_model_router.providers import ProviderConfig, default_providers
from task4_model_router.router import ModelRouter

logger = logging.getLogger("model-router.app")

DB_PATH = os.environ.get("ROUTER_DB_PATH", "./data/router.db")
DEFAULT_LIMIT = int(os.environ.get("ROUTER_TOKENS_PER_MINUTE", DEFAULT_LIMIT_TOKENS_PER_MINUTE))
#: Cap on a request body, applied AS IT ARRIVES and before any tokenisation.
#:
#: Estimation walks and serialises the whole body and it runs BEFORE the rate
#: limit, so an unbounded body is server work an attacker gets for free - the
#: request is rejected 429 afterwards with ``tokens_used_in_window: 0``, which
#: costs them nothing and can be repeated.
MAX_REQUEST_BYTES = int(os.environ.get("ROUTER_MAX_REQUEST_BYTES", str(4 * 1024 * 1024)))

#: Response fields relayed to the client. An allow-list, because the success
#: path was verbatim passthrough: a 200 carrying `error.stack` with a filesystem
#: path, an internal endpoint and a provider API key was returned unchanged.
#: Sanitisation was complete on the error path and absent on the success path.
_ALLOWED_RESPONSE_FIELDS = frozenset({
    "id", "object", "created", "model", "choices", "usage",
    "system_fingerprint", "service_tier", "provider",
})

#: Demo tenant registry. A real deployment resolves the key against an IdP or
#: a tenants table; the shape - key in, tenant id out - is the same.
API_KEYS: dict[str, str] = {
    "sk-tenant-acme-001": "acme",
    "sk-tenant-globex-002": "globex",
}


def create_app(
    db_path: str | None = None,
    providers: list[ProviderConfig] | None = None,
    client: httpx.AsyncClient | None = None,
    default_limit: int | None = None,
    api_keys: dict[str, str] | None = None,
) -> FastAPI:
    resolved_keys = API_KEYS if api_keys is None else api_keys
    default_limit = DEFAULT_LIMIT if default_limit is None else default_limit

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = getattr(app.state, "client", None) is None
        if owned:
            app.state.client = httpx.AsyncClient(
                # Deliberately looser than the per-attempt deadline: the
                # router's asyncio deadline is the real bound, and a client
                # timeout that fired first would mask it.
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            )
        app.state.router = ModelRouter(
            providers or default_providers(), app.state.limiter, app.state.client
        )

        try:
            yield
        finally:
            if owned:
                await app.state.client.aclose()

    app = FastAPI(title="llm-model-router", lifespan=lifespan)
    app.state.client = client
    app.state.limiter = TokenRateLimiter(db_path or DB_PATH, default_limit=default_limit)
    # Built here as well as in lifespan so the app works when mounted over
    # ASGI in tests, which do not run the lifespan.
    if client is not None:
        app.state.router = ModelRouter(providers or default_providers(), app.state.limiter, client)

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        logger.warning("gateway error: %s", exc)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers=exc.headers())

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Route 404/405 and friends through the same envelope.

        Starlette answers these with its own ``{"detail": ...}`` shape, which
        has no ``code`` and no ``request_id`` - so ``GET /v1/chat/completions``
        contradicted the one-error-shape promise this module makes.
        """
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
        code = ErrorCode.UNAUTHENTICATED if exc.status_code == 401 else ErrorCode.INVALID_REQUEST
        error = GatewayError(code, request_id, internal_note=f"http {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code, content=error.to_payload(), headers=error.headers()
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """The backstop.

        Without this, FastAPI's default handler returns a 500 whose body may
        carry the traceback when the server runs with debug on - and that is
        exactly the leak the brief calls out. Everything unexpected collapses
        to one opaque envelope, with the detail going to the log under a
        request id the operator can grep for.
        """
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
        logger.exception("unhandled error req=%s", request_id)
        error = GatewayError(ErrorCode.INTERNAL_ERROR, request_id)
        return JSONResponse(status_code=500, content=error.to_payload(), headers=error.headers())

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> JSONResponse:
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        limit_key = _limit_key_for(request, resolved_keys, request_id)

        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            raise GatewayError(ErrorCode.INVALID_REQUEST, request_id, {"field": "body"},
                               internal_note=f"declared body {declared} bytes")

        chunks: list[bytes] = []
        received = 0
        try:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_REQUEST_BYTES:
                    raise GatewayError(ErrorCode.INVALID_REQUEST, request_id, {"field": "body"},
                                       internal_note=f"body exceeded {MAX_REQUEST_BYTES} bytes")
                chunks.append(chunk)
        except ClientDisconnect:
            logger.info("client disconnected mid-body req=%s", request_id)
            return JSONResponse(status_code=499, content=None)

        try:
            # Parsing a multi-megabyte body blocks the loop, and the care taken
            # to move estimation off it is wasted if the parse stays on it.
            body = await asyncio.to_thread(json.loads, b"".join(chunks))
        except Exception as exc:
            raise GatewayError(
                ErrorCode.INVALID_REQUEST, request_id, {"field": "body"},
                internal_note=f"json decode: {exc}",
            ) from exc

        if not isinstance(body, dict):
            raise GatewayError(ErrorCode.INVALID_REQUEST, request_id, {"field": "body"})
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError(ErrorCode.INVALID_REQUEST, request_id, {"field": "messages"})
        if body.get("stream"):
            # Refused explicitly. Passing it through produced a 502 "no provider
            # could serve this" *after* both providers had generated and billed
            # - the worst of both: the client gets an outage message and the
            # tenant pays for two completions.
            raise GatewayError(ErrorCode.INVALID_REQUEST, request_id, {"field": "stream"},
                               internal_note="streaming responses are not supported by this router")

        result = await app.state.router.route(limit_key, body, request_id)

        payload = {k: v for k, v in result.payload.items() if k in _ALLOWED_RESPONSE_FIELDS}
        # Report the usage the gateway actually charged. Forwarding the
        # provider's own block let a response claim 9,223,372,036,854,775,813
        # tokens while the ledger recorded 15 - two different answers to the
        # same question, one of them in the customer's hands.
        upstream_usage = payload.get("usage")
        payload["usage"] = {
            **(upstream_usage if isinstance(upstream_usage, dict) else {}),
            "total_tokens": result.tokens_charged,
        }
        # Routing metadata the caller legitimately needs: which model actually
        # answered, and what it cost them against their budget.
        payload["gateway"] = {
            "request_id": result.request_id,
            "served_by": result.provider,
            "attempts": [{"provider": a.provider, "outcome": a.outcome} for a in result.attempts],
            "tokens_charged": result.tokens_charged,
        }
        return JSONResponse(payload, headers={"X-Request-Id": result.request_id})

    return app


def _limit_key_for(request: Request, api_keys: dict[str, str], request_id: str) -> str:
    """The rate-limit key for this caller.

    The brief says "per tenant API key", so the budget is keyed on the API key
    itself rather than on the tenant it resolves to. The distinction only shows
    up when one tenant holds several keys: keyed on the tenant they share one
    50,000/minute budget, keyed on the key each gets its own. The brief's
    wording points at the latter, and it is also the reading that makes a key a
    unit of capacity you can hand out and revoke.
    """
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise GatewayError(ErrorCode.UNAUTHENTICATED, request_id, internal_note="missing/blank bearer")
    key = token.strip()
    tenant = api_keys.get(key)
    if tenant is None:
        raise GatewayError(ErrorCode.UNAUTHENTICATED, request_id, internal_note="unknown api key")
    # The tenant is what gets logged; the KEY is what gets the budget.
    return f"{tenant}:{key}"


def __getattr__(name: str):
    """Build the app lazily, on first attribute access.

    ``app = create_app()`` at module scope created the sqlite file as an
    *import* side effect, so every uvicorn worker raced to create and
    WAL-convert the same fresh database before serving anything - and three of
    four died with "database is locked". Building on first access moves that
    into the worker's own startup, where the retry in ``_connect`` covers it.
    """
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

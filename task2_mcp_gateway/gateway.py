"""MCP security gateway: an authenticating, policy-enforcing JSON-RPC proxy.

    client ──Bearer──▶ gateway ──service token──▶ downstream MCP server
                         │
                         └─ refuses admin_* for non-admins, in-process

Request handling, in order:

1. **Authenticate.** Read the ``Bearer <token>`` header and resolve it to a
   role. No valid credential, no proxying. Refusals are both HTTP 401 and a
   JSON-RPC error body, so an MCP client sees something it can parse either way.
2. **Parse the JSON-RPC envelope**, recovering the request id wherever possible
   so the error can be correlated to the call that caused it.
3. **Authorize.** ``tools/list`` is forwarded transparently. ``tools/call`` is
   checked against the tool policy: ``admin_*`` requires the admin role.
4. **Forward what survives**, with the caller's token replaced by the gateway's
   own downstream credential.

The load-bearing property is in step 3/4: a denied call returns ``-32001``
*without* the downstream ever being contacted, so the block is a real refusal
and not merely a filtered response.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from task2_mcp_gateway.auth import AuthError, Principal, TokenVerifier
from task2_mcp_gateway.errors import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHENTICATED,
    UPSTREAM_UNAVAILABLE,
    error_response,
    unauthorized_tool_call,
)
from task2_mcp_gateway.policy import Policy

logger = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9002/mcp")
DOWNSTREAM_SERVICE_TOKEN = os.environ.get("DOWNSTREAM_SERVICE_TOKEN", "downstream-service-token")
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("GATEWAY_UPSTREAM_TIMEOUT", "30"))
#: Cap on a request body, applied AS IT ARRIVES.
#:
#: The gateway buffers the raw bytes, builds a parsed object, then re-serialises
#: it to forward - roughly 3x the body in memory before anything is checked. A
#: security gateway is exactly where that ceiling belongs, and checking the
#: length after buffering would allocate the memory the check exists to prevent.
MAX_BODY_BYTES = int(os.environ.get("GATEWAY_MAX_BODY_BYTES", str(4 * 1024 * 1024)))
#: Cap on the downstream's response. The inbound cap protects the gateway from
#: its clients; this one protects it from the server it proxies to, which a
#: security gateway has no more reason to trust with its memory.
MAX_RESPONSE_BYTES = int(os.environ.get("GATEWAY_MAX_RESPONSE_BYTES", str(4 * 1024 * 1024)))


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _for_log(value: str, limit: int = 120) -> str:
    """Make a caller-controlled value safe to write to the audit log.

    A tool name carrying a newline let a viewer append a syntactically perfect
    forged audit record naming a different subject and role. Emitted headers
    were already sanitised for exactly this reason; the log sink - which in a
    security gateway is the evidence trail - was not.
    """
    return "".join(ch if ch.isprintable() else " " for ch in value)[:limit]


def _ascii_header(value: str, limit: int = 200) -> str:
    """Make a value safe to put in an outgoing header.

    Every header the gateway *emits* has to be ASCII and printable. httpx
    raises on non-ASCII header values, and an unfiltered CR or LF is header
    splitting - so both are neutralised here rather than trusted to the
    inbound parser.
    """
    encoded = value.encode("ascii", "replace").decode("ascii")
    return "".join(ch if ch.isprintable() else " " for ch in encoded)[:limit]


def request_reference(request: Request) -> str:
    """A correlation id: the client's, if it is usable, else a fresh one."""
    supplied = request.headers.get("x-request-id")
    if supplied and supplied.isascii() and supplied.isprintable() and 0 < len(supplied) <= 200:
        return supplied
    return uuid.uuid4().hex[:12]


def sole_authorization_header(request: Request) -> str | None:
    """The one ``Authorization`` header, or a refusal if there is more than one.

    Starlette's ``headers.get`` returns the first of a repeated header. A
    fronting proxy that takes the *last* would then authorise a different
    principal than the one this gateway checked - the classic confused deputy.
    Ambiguity is refused rather than resolved.
    """
    values = request.headers.getlist("authorization")
    if len(values) > 1:
        raise AuthError("multiple Authorization headers")
    return values[0] if values else None


def extract_id(message: Any) -> Any:
    """Best-effort id recovery so an error can be correlated to its request.

    ``bool`` is excluded even though it is an ``int`` subclass: JSON-RPC ids are
    strings or numbers, and echoing ``true`` back as an id is not valid. A
    ``float`` id *is* a JSON number and is echoed - dropping it to ``null``
    left the client unable to correlate the refusal with its own request.
    """
    if isinstance(message, dict):
        candidate = message.get("id")
        if isinstance(candidate, bool):
            return None
        if isinstance(candidate, (str, int, float)) or candidate is None:
            return candidate
    return None


def _is_valid_id(candidate: Any) -> bool:
    """JSON-RPC 2.0: an id is a String, a Number, or Null. ``bool`` is not a
    Number here even though Python says it is an ``int``."""
    if candidate is None:
        return True
    if isinstance(candidate, bool):
        return False
    return isinstance(candidate, (str, int, float))


def validate_envelope(message: Any) -> dict[str, Any] | None:
    """Return a JSON-RPC error for a structurally invalid message, else ``None``."""
    if not isinstance(message, dict):
        return error_response(None, INVALID_REQUEST, "Invalid Request: message must be a JSON object")
    if message.get("jsonrpc") != "2.0":
        return error_response(extract_id(message), INVALID_REQUEST, "Invalid Request: jsonrpc must be '2.0'")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        return error_response(extract_id(message), INVALID_REQUEST, "Invalid Request: 'method' must be a string")
    if "id" in message and not _is_valid_id(message["id"]):
        # JSON-RPC 2.0 allows a String, Number or Null id and nothing else.
        # Accepting ``true``/``{}``/``[]`` here meant ``extract_id`` mapped them
        # to None, the request was FORWARDED, the privileged tool executed, and
        # the correct downstream reply was then rejected as "malformed" - so a
        # side-effectful call ran and the client was told the upstream broke.
        return error_response(
            None, INVALID_REQUEST, "Invalid Request: 'id' must be a string or a number"
        )
    if "id" in message and message["id"] is None:
        # MCP forbids a null request id, and JSON-RPC treats a message carrying
        # an ``id`` member as a Request whatever its value - so this is neither
        # a notification nor answerable. Relaying it let the downstream's
        # ``null`` body through as the response, which is not a Response object.
        return error_response(None, INVALID_REQUEST, "Invalid Request: 'id' must not be null")
    params = message.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        return error_response(extract_id(message), INVALID_REQUEST, "Invalid Request: 'params' must be object or array")
    return None


def _normalise(value: str) -> str:
    """Casefold and strip, for a fail-closed second look at a name."""
    return value.strip().casefold()


def screen_message(message: dict[str, Any], principal: Principal, policy: Policy) -> dict[str, Any] | None:
    """Decide whether the message may be forwarded.

    Returns ``None`` to forward, or the JSON-RPC error to return in its place.
    Every method other than ``tools/call`` - ``tools/list`` included - is
    forwarded transparently, which is what the brief asks for.

    The gate is checked against the raw name *and* a casefolded, stripped one.
    Matching the exact literal only meant ``"Tools/Call"`` and
    ``" admin_reset_key"`` were forwarded with no policy check at all. Neither
    escalates against a downstream that dispatches byte-exactly - and this one
    does - but "check one exact string, forward everything else" makes the
    gateway's correctness depend on the downstream's parser matching its own,
    which is the assumption re-serialising the body exists to remove. Adding a
    normalised second look can only ever refuse more, never less.
    """
    method = message["method"]
    if method != "tools/call":
        if _normalise(method) == "tools/call":
            logger.warning(
                "refused near-miss method %r from %s", _for_log(method), principal.subject
            )
            return error_response(
                extract_id(message),
                INVALID_REQUEST,
                "Invalid Request: unrecognised spelling of 'tools/call'",
            )
        return None

    params = message.get("params")
    if not isinstance(params, dict):
        return error_response(
            extract_id(message), INVALID_PARAMS, "Invalid params: tools/call requires a params object"
        )
    tool_name = params.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return error_response(
            extract_id(message), INVALID_PARAMS, "Invalid params: tools/call requires params.name"
        )

    decision = policy.check_tool_call(tool_name, principal.role)
    normalised = policy.check_tool_call(_normalise(tool_name), principal.role)
    if decision.allowed and normalised.allowed:
        return None
    if decision.allowed:
        decision = normalised

    logger.warning(
        "denied tools/call name=%s subject=%s role=%s required=%s",
        _for_log(tool_name),
        principal.subject,
        principal.role,
        decision.required_role,
    )
    return unauthorized_tool_call(extract_id(message), tool_name, principal.role, decision.required_role)


def downstream_headers(
    principal: Principal, request_ref: str, incoming: Any = None
) -> dict[str, str]:
    """Headers for the upstream call.

    Nothing is copied from the client request. The gateway authenticates to the
    downstream as *itself*, and asserts the caller's identity in headers the
    downstream can trust precisely because only the gateway holds the service
    token.
    """
    incoming = incoming or {}
    forwarded = {
        "content-type": "application/json",
        # Both types, as MCP Streamable HTTP requires. Sending only
        # application/json made a spec-conformant server answer 406 and the
        # gateway could not complete a single request against one.
        "accept": "application/json, text/event-stream",
        # The gateway re-serialises the body it forwards and reads the reply
        # itself; asking for compression buys nothing and is one more thing to
        # get wrong on the way back out.
        "accept-encoding": "identity",
        "authorization": f"Bearer {DOWNSTREAM_SERVICE_TOKEN}",
        "x-mcp-user": _ascii_header(principal.subject),
        "x-mcp-role": _ascii_header(principal.role),
        "x-request-id": _ascii_header(request_ref),
    }
    # The MCP session headers are the client's, not ours to invent - but they
    # do have to survive the hop, or session management breaks in both
    # directions. Everything else the client sent is still dropped.
    for name in ("mcp-session-id", "mcp-protocol-version"):
        value = incoming.get(name)
        if value and value.isascii():
            forwarded[name] = _ascii_header(value)
    return forwarded


def create_app(
    downstream_url: str | None = None,
    verifier: TokenVerifier | None = None,
    policy: Policy | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    resolved_url = downstream_url or DOWNSTREAM_URL
    resolved_verifier = verifier or TokenVerifier()
    resolved_policy = policy or Policy()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One pooled client for the process. Building an AsyncClient per request
        # throws away connection reuse and shows up immediately as added latency.
        if getattr(app.state, "client", None) is not None:
            yield
            return
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_SECONDS, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    # ``redirect_slashes=False``: a POST to "/mcp/" answered with a 307, and a
    # redirect carries the caller's Authorization header to wherever it points.
    # A security proxy should not be issuing those.
    app = FastAPI(title="mcp-security-gateway", lifespan=lifespan, redirect_slashes=False)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """404/405 and friends, in a frame an MCP client can parse.

        Starlette answers these with its own ``{"detail": ...}``, which is not
        JSON-RPC at all - so the one-shape promise this module makes held for
        every path except the ones a misconfigured client actually hits.
        """
        request_ref = request_reference(request)
        return JSONResponse(
            status_code=exc.status_code,
            headers={"X-Request-Id": request_ref},
            content=error_response(None, INVALID_REQUEST, f"Invalid Request: {exc.status_code}"),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Backstop so nothing escapes as a bare HTTP 500 with no JSON-RPC body."""
        request_ref = request_reference(request)
        logger.exception("unhandled gateway error ref=%s", request_ref)
        return JSONResponse(
            status_code=500,
            headers={"X-Request-Id": request_ref},
            content=error_response(None, INTERNAL_ERROR, "Internal server error"),
        )

    # An injected client is installed eagerly so the app is usable without a
    # lifespan run - which is exactly how the tests mount it over ASGI.
    app.state.client = client
    app.state.downstream_url = resolved_url
    app.state.verifier = resolved_verifier
    app.state.policy = resolved_policy

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/mcp")
    async def proxy(request: Request) -> Response:
        request_ref = request_reference(request)

        # ---- 1. authenticate ------------------------------------------------
        try:
            principal = resolved_verifier.principal_from_header(sole_authorization_header(request))
        except AuthError as exc:
            logger.warning("auth failed ref=%s: %s", request_ref, exc)
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer", "X-Request-Id": request_ref},
                content=error_response(None, UNAUTHENTICATED, "Unauthorized"),
            )

        # ---- 2. parse -------------------------------------------------------
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            logger.warning("oversized body ref=%s declared=%s", request_ref, declared)
            return _too_large(request_ref)

        chunks: list[bytes] = []
        received = 0
        try:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_BODY_BYTES:
                    logger.warning("oversized body ref=%s bytes>%d", request_ref, MAX_BODY_BYTES)
                    return _too_large(request_ref)
                chunks.append(chunk)
        except ClientDisconnect:
            # The client went away mid-upload. Nothing to answer and nothing
            # wrong; letting it reach the catch-all logs a stack trace per
            # aborted request, which is a cheap log-flood for any valid token.
            logger.info("client disconnected mid-body ref=%s", request_ref)
            return Response(status_code=499)
        raw = b"".join(chunks)
        try:
            # ``parse_constant`` rejects NaN/Infinity/-Infinity. Python's json
            # accepts them by default and re-emits them, producing a body that
            # is not valid JSON for any other parser downstream.
            message = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError):
            # Not just JSONDecodeError: a non-UTF-8 body raises UnicodeDecodeError
            # and deeply nested JSON raises RecursionError.
            return JSONResponse(
                status_code=400,
                headers={"X-Request-Id": request_ref},
                content=error_response(None, PARSE_ERROR, "Parse error"),
            )

        envelope_error = validate_envelope(message)
        if envelope_error is not None:
            return JSONResponse(
                status_code=400, headers={"X-Request-Id": request_ref}, content=envelope_error
            )

        # ---- 3. authorize ---------------------------------------------------
        refusal = screen_message(message, principal, resolved_policy)
        if refusal is not None:
            # Returned here, before any upstream call: the downstream server
            # never learns the request existed.
            if "id" not in message:
                # JSON-RPC 2.0 §4.1: a server MUST NOT reply to a notification.
                # The call is still refused - it simply is not forwarded - but
                # answering it with ``"id": null`` was a protocol violation, and
                # a client that correlates on id cannot use the response anyway.
                logger.warning("denied notification ref=%s", request_ref)
                return Response(status_code=202, headers={"X-Request-Id": request_ref})
            return JSONResponse(
                status_code=200, headers={"X-Request-Id": request_ref}, content=refusal
            )

        # ---- 4. forward -----------------------------------------------------
        client_ = app.state.client

        # Peek at the reply's content type before deciding how to read it. An
        # SSE channel must be relayed chunk by chunk - draining it first made a
        # 20-second notification stream arrive as one socket read after 20
        # seconds of silence, which is not a stream at all, and MAX_RESPONSE_BYTES
        # would eventually kill a long-lived one outright.
        try:
            streamed = await _open_stream(
                client_,
                app.state.downstream_url,
                json.dumps(message).encode(),
                downstream_headers(principal, request_ref, request.headers),
            )
        except httpx.TimeoutException as exc:
            logger.error("upstream timed out ref=%s: %s", request_ref, exc)
            return JSONResponse(
                status_code=504, headers={"X-Request-Id": request_ref},
                content=error_response(extract_id(message), UPSTREAM_UNAVAILABLE,
                                       "Upstream server timed out"),
            )
        except httpx.HTTPError as exc:
            logger.error("upstream unreachable ref=%s: %s", request_ref, exc)
            return JSONResponse(
                status_code=502, headers={"X-Request-Id": request_ref},
                content=error_response(extract_id(message), UPSTREAM_UNAVAILABLE,
                                       "Upstream server unavailable"),
            )

        context, response = streamed

        # Status FIRST, content type second. Testing the content type first
        # meant an SSE-framed failure skipped every check below it: a
        # downstream 500 was relayed verbatim, status and body, so a stack
        # trace and a Postgres DSN went straight to the client - past the
        # sanitisation this module documents and the JSON path performs.
        if response.status_code >= 300:
            await _drain(context, response)
            logger.error(
                "upstream error ref=%s status=%s: %.300s",
                request_ref, response.status_code, response.text[:300],
            )
            return JSONResponse(
                status_code=502,
                headers={"X-Request-Id": request_ref},
                content=error_response(
                    extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server error"
                ),
            )

        if "id" not in message:
            # A notification gets no response body whatever the downstream
            # chose to send back - a rule the JSON path applied and the SSE
            # path returned before ever reaching.
            await _drain(context, response)
            return Response(status_code=202, headers={"X-Request-Id": request_ref})

        if "text/event-stream" in response.headers.get("content-type", ""):
            relayed = {"X-Request-Id": request_ref}
            session_id = response.headers.get("mcp-session-id")
            if session_id and session_id.isascii():
                relayed["Mcp-Session-Id"] = _ascii_header(session_id)
            return StreamingResponse(
                _relay_sse(context, response, request_ref),
                status_code=response.status_code,
                media_type="text/event-stream",
                headers=relayed,
            )
        try:
            upstream = await _read_capped(context, response)
            if upstream is None:
                logger.error("oversized downstream response ref=%s", request_ref)
                return JSONResponse(
                    status_code=502,
                    headers={"X-Request-Id": request_ref},
                    content=error_response(
                        extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server error"
                    ),
                )
        except httpx.TimeoutException as exc:
            logger.error("upstream timed out ref=%s: %s", request_ref, exc)
            return JSONResponse(
                status_code=504,
                headers={"X-Request-Id": request_ref},
                content=error_response(
                    extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server timed out"
                ),
            )
        except httpx.HTTPError as exc:
            logger.error("upstream unreachable ref=%s: %s", request_ref, exc)
            return JSONResponse(
                status_code=502,
                headers={"X-Request-Id": request_ref},
                content=error_response(
                    extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server unavailable"
                ),
            )

        return _build_response(upstream, message, request_ref)

    return app


async def _open_stream(client, url: str, content: bytes, headers: dict):
    """Open the upstream and return ``(context, response)`` with headers read.

    The body is untouched, so the caller can decide between relaying an event
    stream incrementally and buffering a JSON reply under the size cap.
    """
    context = client.stream("POST", url, content=content, headers=headers)
    response = await context.__aenter__()
    return context, response


async def _drain(context, response) -> None:
    """Read and close a reply we are not going to relay."""
    with contextlib.suppress(Exception):
        await response.aread()
    with contextlib.suppress(Exception):
        await context.__aexit__(None, None, None)


async def _relay_sse(context, response, request_ref: str):
    """Pump upstream SSE bytes to the client as they arrive."""
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.HTTPError as exc:
        logger.error("sse relay interrupted ref=%s: %s", request_ref, exc)
    finally:
        with contextlib.suppress(Exception):
            await context.__aexit__(None, None, None)


async def _read_capped(context, response):
    """Read an already-open reply, stopping once the cap is exceeded.

    Streamed and checked as the bytes arrive. An earlier version awaited a
    buffering ``post()`` and then measured ``len(upstream.content)`` - the cap
    fired only after the allocation it exists to prevent, so it reported the
    problem rather than bounding it.

    Takes the open response rather than issuing its own: the caller has already
    sent the request in order to see the content type, and sending a second one
    executed every tool twice.

    Returns ``None`` when the cap is hit.
    """
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                await context.__aexit__(None, None, None)
                return None
        await context.__aexit__(None, None, None)
    except httpx.HTTPError:
        with contextlib.suppress(Exception):
            await context.__aexit__(None, None, None)
        raise
    # Hand back a fully-read response so the rest of the path is unchanged.
    #
    # ``aiter_bytes`` yields bytes httpx has ALREADY content-decoded, so the
    # original ``content-encoding`` must not travel with them: the constructor
    # would decode a second time and raise DecodingError. That turned every
    # request against any gzip-compressing downstream into a 502 - and the
    # gateway asks for gzip itself - which is a total outage, not an edge case.
    headers = httpx.Headers(
        [
            (name, value)
            for name, value in response.headers.raw
            if name.lower() not in (b"content-encoding", b"content-length")
        ]
    )
    return httpx.Response(
        status_code=response.status_code,
        headers=headers,
        content=bytes(body),
        request=response.request,
    )


def _too_large(request_ref: str) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        headers={"X-Request-Id": request_ref},
        content=error_response(None, INVALID_REQUEST, "Invalid Request: payload too large"),
    )


def _build_response(upstream: httpx.Response, message: dict[str, Any], request_ref: str) -> Response:
    headers = {"X-Request-Id": request_ref}

    if "id" not in message or (upstream.status_code == 202 and not upstream.content):
        # A notification gets no response body, whatever the downstream chose
        # to send back. Relaying the downstream's ``null`` body verbatim gave
        # the client a 202 carrying JSON ``null``.
        return Response(status_code=202, headers=headers)

    # A 3xx is not a success. It used to fall through to ``upstream.json()``
    # and was rescued only by the ValueError handler, so a redirect that
    # happened to carry a JSON body would have been relayed as a result -
    # complete with whatever its Location header pointed at.
    if 300 <= upstream.status_code < 400:
        logger.error("upstream redirect ref=%s status=%s", request_ref, upstream.status_code)
        return JSONResponse(
            status_code=502,
            headers=headers,
            content=error_response(extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server error"),
        )

    # A non-2xx upstream body is the gateway's problem, not the client's. These
    # routinely carry a stack trace, an internal hostname or a database DSN, so
    # only the fact of the failure crosses the boundary.
    if upstream.status_code >= 400:
        logger.error("upstream error ref=%s status=%s: %.300s", request_ref, upstream.status_code, upstream.text)
        return JSONResponse(
            status_code=502,
            headers=headers,
            content=error_response(extract_id(message), UPSTREAM_UNAVAILABLE, "Upstream server error"),
        )

    try:
        payload = upstream.json()
    except ValueError:
        logger.error("upstream returned non-JSON ref=%s status=%s", request_ref, upstream.status_code)
        return JSONResponse(
            status_code=502,
            headers=headers,
            content=error_response(extract_id(message), INTERNAL_ERROR, "Upstream returned a malformed response"),
        )

    invalid = _invalid_response(payload, extract_id(message))
    if invalid is not None:
        logger.error("upstream sent a non-conforming response ref=%s: %s", request_ref, invalid)
        return JSONResponse(
            status_code=502,
            headers=headers,
            content=error_response(extract_id(message), INTERNAL_ERROR, "Upstream returned a malformed response"),
        )

    session_id = upstream.headers.get("mcp-session-id")
    if session_id and session_id.isascii():
        headers["Mcp-Session-Id"] = _ascii_header(session_id)

    # The upstream's 2xx status is preserved, but Content-Type is the gateway's
    # to set: an upstream that answered text/html with a JSON body would
    # otherwise have that type relayed.
    return JSONResponse(status_code=upstream.status_code, content=payload, headers=headers)


def _invalid_response(payload: Any, expected_id: Any) -> str | None:
    """Why this is not a valid JSON-RPC Response to the request we sent, if so.

    The gateway validated inbound to the byte and accepted the downstream's
    reply on faith, which is the wrong way round for a security proxy: a
    mismatched id, a bare ``{"foo": "bar"}``, and a body carrying both
    ``result`` and ``error`` - which JSON-RPC forbids - were all relayed to the
    client verbatim.
    """
    if not isinstance(payload, dict):
        return "not a JSON object"
    if payload.get("jsonrpc") != "2.0":
        return "missing or wrong 'jsonrpc' member"
    has_result = "result" in payload
    has_error = "error" in payload
    if has_result == has_error:
        return "must carry exactly one of 'result' or 'error'"
    if has_error and not isinstance(payload.get("error"), dict):
        return "'error' must be an object"
    if payload.get("id") != expected_id:
        # Correlation is the client's only way to match a reply to its call.
        return f"id {payload.get('id')!r} does not match the request"
    return None


app = create_app()

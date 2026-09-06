"""LLM gateway with an inline streaming PII guardrail.

    client ◀── redacted SSE ── gateway ── SSE ──▶ provider

The gateway never holds the full response. Per stream it retains one
``StreamingRedactor`` (bounded hold-back buffer) and one ``SSEDecoder``
(bounded to the current event), so memory is O(1) in response length. Bytes
move client-ward as soon as they are provably safe: there is no batching timer
and no wait for the upstream to finish.

Two failure modes are handled explicitly, because both leak:

* **The stream dies mid-response.** Whatever is in the hold-back buffer is
  *not* flushed to the client - it was withheld precisely because it might be
  the front half of a credit card. It is dropped.
* **The upstream errors before any body.** The request is opened before a
  status line is committed, so the provider's status is mirrored rather than
  flattened to 200 with an in-band error a client cannot act on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from task3_streaming_guardrail.redactor import RedactionStats, StreamingRedactor
from task3_streaming_guardrail.sse import SSEDecoder, encode_sse

logger = logging.getLogger("llm-gateway")

UPSTREAM_URL = os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:9003/v1/chat/completions")
UPSTREAM_API_KEY = os.environ.get("LLM_UPSTREAM_API_KEY", "sk-mock-upstream-key")
# Generous: a slow model legitimately takes minutes. The read timeout is what
# actually matters for a hung stream, and httpx applies it per read.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
# Must be >= redactor.MAX_MATCH_LENGTH; StreamingRedactor enforces that, since
# a smaller window would emit the front half of a long value in the clear.
MAX_HOLDBACK = int(os.environ.get("GUARDRAIL_MAX_HOLDBACK", "1024"))
#: Above this many characters in one chunk, redaction moves to a worker thread.
#:
#: Redaction is synchronous CPU inside the async generator. On a normal delta
#: (a few tokens) that is microseconds and a thread hop would cost more than it
#: saves - TTFT is the thing being protected. On a big one it is not: a
#: 128,000-character delta took a co-tenant's /healthz from 0.56ms to 730ms,
#: because nothing yielded to the loop for the whole scan. The threshold keeps
#: the fast path fast and stops one large chunk stalling every other
#: connection on the worker.
OFFLOAD_ABOVE_CHARS = int(os.environ.get("GUARDRAIL_OFFLOAD_ABOVE", "4096"))


class RedactorPool:
    """One ``StreamingRedactor`` per independent text stream in a response.

    A single redactor cannot be shared across fields. The deltas for
    ``choices[0].content``, ``choices[1].content`` and a tool call's
    ``arguments`` are interleaved on the wire but are *separate* strings, and
    feeding them into one buffer would splice them together - inventing matches
    across a boundary that does not exist, and corrupting the hold-back for
    each. Keyed by path, each stream gets its own state.
    """

    def __init__(self) -> None:
        self._redactors: dict[tuple, StreamingRedactor] = {}

    def feed(self, key: tuple, text: str) -> str:
        redactor = self._redactors.get(key)
        if redactor is None:
            redactor = StreamingRedactor(max_holdback=MAX_HOLDBACK)
            self._redactors[key] = redactor
        return redactor.feed(text)

    def redact_once(self, text: str) -> str:
        """Redact a value that arrives complete, with no hold-back carried."""
        redactor = StreamingRedactor(max_holdback=MAX_HOLDBACK)
        return redactor.feed(text) + redactor.flush()

    def flush(self) -> list[tuple[tuple, str]]:
        """Residue per stream, in the order the streams were first seen."""
        out = []
        for key, redactor in self._redactors.items():
            remaining = redactor.flush()
            if remaining:
                out.append((key, remaining))
        return out

    def flush_choice(self, index) -> list[tuple[tuple, str]]:
        """Residue for one choice only.

        Called when that choice's ``finish_reason`` arrives. Held-back text
        used to be emitted after the finish chunk, so a client that treats
        ``finish_reason`` as end-of-message rendered the text *without* its
        redaction marker and dropped the tail - content loss, and the one
        ordering a conforming client is entitled to rely on.
        """
        out = []
        for key, redactor in self._redactors.items():
            if key and key[0] == index:
                remaining = redactor.flush()
                if remaining:
                    out.append((key, remaining))
        return out

    @property
    def stats(self) -> RedactionStats:
        """Combined counts across every stream in the response."""
        total = RedactionStats()
        for redactor in self._redactors.values():
            for name, count in redactor.stats.counts.items():
                total.counts[name] = total.counts.get(name, 0) + count
            total.characters_in += redactor.stats.characters_in
            total.characters_out += redactor.stats.characters_out
        return total

    @property
    def pending(self) -> int:
        return sum(r.pending for r in self._redactors.values())


#: Keys whose string values are structural rather than model-authored, and so
#: are left alone.
#:
#: A DENY-list, because the allow-list it replaces was fail-open: only
#: ``content``/``reasoning_content``/``refusal`` and tool-call arguments were
#: redacted, so every other shape a provider emits went through verbatim -
#: legacy ``choices[].text``, a content-parts array, the non-streaming
#: ``message`` container, and Anthropic's ``delta.text`` all leaked an address
#: and a card in testing. Anything not named here is treated as model output
#: and redacted; a new provider shape is then caught by default instead of
#: waiting to be enumerated.
_STRUCTURAL_KEYS = frozenset({
    "id", "object", "created", "model", "role", "type", "index",
    "finish_reason", "system_fingerprint", "service_tier", "fingerprint",
    "name", "event", "object_type", "stop_reason",
})

#: Field names whose values arrive INCREMENTALLY and so need continuing
#: hold-back state. Everything else arrives whole and is redacted atomically.
#:
#: Matched on the LEAF name, not on any ancestor. Testing whether "delta"
#: appeared anywhere in the path was too loose: an annotation's URL sits at
#: ``delta.annotations[].url_citation.url`` and arrives complete, so hold-back
#: truncated it to "https://example.com/" and delivered the rest in two later
#: chunks. Only these leaves stream.
_STREAMING_LEAVES = frozenset({
    "content", "text", "refusal", "reasoning_content", "arguments",
})


def redact_payload(payload: dict[str, Any], pool: RedactorPool) -> tuple[dict[str, Any], bool]:
    """Return the chunk with every model-authored string redacted.

    Walks the whole object rather than reaching for known field names. Each
    string gets its own redactor, keyed by the path it sits at, so independent
    streams keep independent hold-back state - a choice's ``content`` and a
    tool call's ``arguments`` must not share a buffer or they splice.

    The second element says whether anything textual was carried at all, so the
    caller can drop a chunk that is now entirely empty rather than emitting a
    no-op delta.
    """
    carried: list[bool] = []
    redacted = _redact_node(payload, pool, (), carried, choice_index=None)
    return redacted, bool(carried)


def _redact_node(node: Any, pool: RedactorPool, path: tuple, carried: list, choice_index):
    """Recursively redact model-authored strings, keyed by their path."""
    if isinstance(node, dict):
        out = {}
        # A choice's DECLARED index keys its streams. Every chunk carries a
        # single choice at position 0, so keying by position collapsed all n
        # completions onto one redactor and spliced them together.
        declared = node.get("index")
        local_index = (
            declared
            if isinstance(declared, int) and not isinstance(declared, bool)
            else choice_index
        )
        for key, value in node.items():
            if key in _STRUCTURAL_KEYS or not isinstance(key, str):
                out[key] = value
                continue
            out[key] = _redact_node(value, pool, path + (key,), carried, local_index)
        return out
    if isinstance(node, list):
        return [
            _redact_node(item, pool, path + (position,), carried, choice_index)
            for position, item in enumerate(node)
        ]
    if isinstance(node, str) and node:
        carried.append(True)
        if _is_streaming_path(path):
            return pool.feed((choice_index,) + path, node)
        # Single-shot: redact it whole and flush, so nothing is held back from
        # a value that is already complete.
        return pool.redact_once(node)
    return node


def _is_streaming_path(path: tuple) -> bool:
    """Whether a value at this path is delivered incrementally.

    Only the delta-carrying fields stream. A ``request_id``, a citation URL or
    a usage note arrives complete in each chunk, and running hold-back over one
    emptied it from every chunk and re-emitted the pieces concatenated later.
    """
    leaf = next((step for step in reversed(path) if isinstance(step, str)), None)
    return leaf in _STREAMING_LEAVES


def _payload_text_length(payload: Any) -> int:
    """Roughly how much model text this chunk carries, for the offload decision."""
    if isinstance(payload, dict):
        return sum(
            _payload_text_length(v) for k, v in payload.items() if k not in _STRUCTURAL_KEYS
        )
    if isinstance(payload, list):
        return sum(_payload_text_length(v) for v in payload)
    return len(payload) if isinstance(payload, str) else 0


def _chunk_has_content(payload: dict[str, Any]) -> bool:
    """True if the redacted chunk still carries something worth sending."""
    for choice in payload.get("choices", []) or []:
        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
            return True
    if _payload_text_length(payload) > 0:
        return True
    # Scaffolding with no text of its own - a role announcement, an opening
    # tool_calls frame - is still worth forwarding.
    return _has_non_text_structure(payload)


def _has_non_text_structure(node: Any) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _STRUCTURAL_KEYS and key not in ("index", "type"):
                return True
            if _has_non_text_structure(value):
                return True
        return False
    if isinstance(node, list):
        return any(_has_non_text_structure(v) for v in node)
    return isinstance(node, (int, float, bool)) and not isinstance(node, bool)


def _is_chunk_key(key: tuple) -> bool:
    """Whether this stream's residue belongs in a chat-completion chunk.

    ``__raw__`` and ``__bare__`` streams come from events that were never
    OpenAI chunks - a keep-alive, a provider extension, a bare JSON string.
    Reshaping their residue into a chunk emitted ``"index": "__raw__"``, a
    string where the schema requires an integer.
    """
    return bool(key) and not (isinstance(key[0], str) and key[0].startswith("__"))


def _nest_at_path(path: tuple, value: Any) -> Any:
    """Rebuild the nested shape a redacted string was found at."""
    node = value
    for step in reversed(path):
        if isinstance(step, int):
            node = [{} for _ in range(step)] + [node]
        else:
            node = {step: node}
    return node


def _residue_chunk(template: dict[str, Any] | None, key: tuple, text: str) -> dict[str, Any]:
    """A chunk carrying one stream's flushed residue, shaped like its siblings.

    The residue is re-nested at the exact path the text came from, so a legacy
    ``choices[].text``, a content-parts array and a chat ``delta.content`` each
    come back in the shape their own stream used rather than being forced into
    one assumed layout.
    """
    base = template or {}
    choice_index, *path = key
    body = _nest_at_path(tuple(path), text) if path else {"choices": [{"delta": {"content": text}}]}
    if not isinstance(body, dict):
        body = {"choices": [{"delta": {"content": text}}]}
    chunk = {
        "id": base.get("id", "chatcmpl-guardrail"),
        "object": base.get("object", "chat.completion.chunk"),
        "created": base.get("created", int(time.time())),
        "model": base.get("model", "unknown"),
        **body,
    }
    # Stamp the declared choice index back on, so a client can correlate.
    for choice in chunk.get("choices", []) or []:
        if isinstance(choice, dict) and "index" not in choice:
            choice["index"] = choice_index if isinstance(choice_index, int) else 0
        if isinstance(choice, dict):
            choice.setdefault("finish_reason", None)
    return chunk


def _residue_events(residues: list[tuple[tuple, str]], template: dict[str, Any] | None) -> list[bytes]:
    """Encode each stream's residue in the shape that stream was carried in."""
    out = []
    for key, text in residues:
        if _is_chunk_key(key):
            out.append(encode_sse(json.dumps(_residue_chunk(template, key, text))))
        elif key and key[0] == "__bare__":
            out.append(encode_sse(json.dumps(text)))
        else:
            out.append(encode_sse(text))
    return out


def create_app(upstream_url: str | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    resolved_upstream = upstream_url or UPSTREAM_URL

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "client", None) is not None:
            yield
            return
        app.state.client = httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="llm-gateway-guardrail", lifespan=lifespan)
    app.state.client = client
    app.state.upstream_url = resolved_upstream
    #: Last stream's stats, for the tests and for a /metrics-style endpoint.
    app.state.last_stats = None

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            # A malformed body is the caller's mistake, not a server fault. It
            # used to escape as HTTP 500 with a full traceback through
            # Starlette - the wrong status, and a stack trace on the wire.
            logger.warning("malformed request body")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Request body is not valid JSON",
                                   "type": "invalid_request_error"}},
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Request body must be a JSON object",
                                   "type": "invalid_request_error"}},
            )
        # Always streamed, whatever the caller asked for. The task requires
        # that the full response is never accumulated before forwarding, and
        # assembling a single JSON completion means doing exactly that. A
        # client wanting one body can join the deltas itself; the gateway will
        # not hold them.
        body = {**body, "stream": True}
        upstream_headers = {
            "authorization": f"Bearer {UPSTREAM_API_KEY}",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

        # The upstream request is opened HERE, before a status line is
        # committed to, so a provider's 401/429/500 is mirrored as that status
        # rather than delivered as an in-band error under a blanket 200. A
        # client cannot tell "retry with backoff" from "your key is dead" if
        # every failure arrives as 200.
        stream = request.app.state.client.stream(
            "POST", resolved_upstream, json=body, headers=upstream_headers
        )
        try:
            upstream = await stream.__aenter__()
        except httpx.HTTPError as exc:
            logger.error("upstream unreachable: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "Upstream provider unavailable", "type": "upstream_error"}},
            )

        if upstream.status_code >= 400:
            await upstream.aread()
            status = upstream.status_code
            logger.error("upstream status %s: %.300s", status, upstream.text)
            await stream.__aexit__(None, None, None)
            # The body is the gateway's problem, not the client's - provider
            # error bodies carry account ids, internal endpoints and prompt
            # fragments. Only the status crosses the boundary.
            return JSONResponse(
                status_code=status,
                content={"error": {
                    "message": "Upstream provider returned an error",
                    "type": "upstream_error",
                    "code": status,
                }},
            )

        return StreamingResponse(
            _stream(request, stream, upstream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    return app


async def _stream(request: Request, stream, upstream) -> AsyncIterator[bytes]:
    """The hot path.

    One pass per upstream byte chunk: decode SSE incrementally, run each text
    delta through the redactor, re-encode and yield. Nothing is accumulated -
    every ``yield`` leaves this function immediately for the socket.

    ``stream`` is the already-entered upstream context manager; the caller
    opened it so the status could be mirrored before headers were committed,
    and this generator owns closing it.
    """
    app = request.app
    pool = RedactorPool()
    decoder = SSEDecoder()
    last_chunk: dict[str, Any] | None = None
    completed = False

    try:
        async for raw in upstream.aiter_bytes():
            for event in decoder.feed(raw):
                async for out in _handle_event(event, pool, last_chunk):
                    yield out
                last_chunk = _remember(event, last_chunk)
                if event.is_done:
                    completed = True
            if completed:
                break

        for event in decoder.flush():
            async for out in _handle_event(event, pool, last_chunk):
                yield out
            if event.is_done:
                completed = True

        if not completed:
            # Upstream ended without [DONE]. Flush the guardrail so a value
            # sitting at the very end of the response is still redacted, then
            # terminate the stream ourselves.
            for encoded in _residue_events(pool.flush(), last_chunk):
                yield encoded
            yield encode_sse("[DONE]")

    except ValueError as exc:
        # The SSE decoder refuses an event that never terminates. That has to
        # become an in-band error: raising through the handler aborted the
        # response with no [DONE] and dumped a traceback through Starlette.
        logger.error("malformed upstream stream: %s", exc)
        yield encode_sse(json.dumps({"error": {
            "message": "Upstream sent a malformed stream", "type": "upstream_error",
        }}))
        yield encode_sse("[DONE]")
    except httpx.HTTPError as exc:
        # The buffer is intentionally *not* flushed here. It was withheld
        # because it might be the first half of a credit card; a dead
        # connection is not a reason to hand it over.
        logger.error("stream failed after %d chars: %s", pool.stats.characters_out, exc)
        yield encode_sse(json.dumps({"error": {
            "message": "Upstream stream interrupted",
            "type": "upstream_error",
        }}))
        yield encode_sse("[DONE]")
    finally:
        app.state.last_stats = pool.stats
        if pool.stats.total:
            logger.info("redacted %s", pool.stats.counts)
        # Always release the socket, whether the stream finished, the client
        # left, or something raised.
        with contextlib.suppress(Exception):
            await stream.__aexit__(None, None, None)


def _finished_choice_indexes(payload: dict[str, Any]) -> list[int]:
    """Declared indexes of any choice this chunk marks as finished."""
    finished = []
    for position, choice in enumerate(payload.get("choices", []) or []):
        if not isinstance(choice, dict) or choice.get("finish_reason") is None:
            continue
        declared = choice.get("index")
        finished.append(
            declared if isinstance(declared, int) and not isinstance(declared, bool) else position
        )
    return finished


def _remember(event, last_chunk):
    if event.is_done:
        return last_chunk
    try:
        payload = json.loads(event.data)
    except json.JSONDecodeError:
        return last_chunk
    return payload if isinstance(payload, dict) else last_chunk


async def _handle_event(event, pool: RedactorPool, last_chunk) -> AsyncIterator[bytes]:
    """Translate one upstream event into zero or more client-bound events."""
    if event.is_done:
        # Terminal event: everything still held back is now final.
        for encoded in _residue_events(pool.flush(), last_chunk):
            yield encoded
        yield encode_sse("[DONE]")
        return

    try:
        payload = json.loads(event.data)
    except json.JSONDecodeError:
        # Not JSON - a keep-alive or a provider extension. Redact it as plain
        # text rather than forwarding blind: an event body is still something
        # the model may have influenced.
        yield encode_sse(pool.feed(("__raw__",), event.data), event.event, event.id, event.retry)
        return

    if isinstance(payload, str):
        # Some providers stream a bare JSON string. It is model text.
        yield encode_sse(
            json.dumps(pool.feed(("__bare__",), payload)), event.event, event.id, event.retry
        )
        return

    if not isinstance(payload, dict):
        yield encode_sse(json.dumps(payload), event.event, event.id, event.retry)
        return

    # A choice that has just finished gets its residue FIRST, so a client that
    # stops at ``finish_reason`` has already seen the redacted tail.
    for index in _finished_choice_indexes(payload):
        for encoded in _residue_events(pool.flush_choice(index), payload):
            yield encoded

    if _payload_text_length(payload) > OFFLOAD_ABOVE_CHARS:
        # One redactor belongs to one stream and this generator is the only
        # thing feeding it, so moving the call to a thread does not introduce
        # concurrent access to the pool - the await here is the whole point.
        redacted, carried_text = await asyncio.to_thread(redact_payload, payload, pool)
    else:
        redacted, carried_text = redact_payload(payload, pool)
    if not carried_text:
        # Nothing textual in this chunk - role announcements, usage records,
        # provider extensions. Forward unchanged.
        yield encode_sse(json.dumps(redacted), event.event, event.id, event.retry)
        return

    if _chunk_has_content(redacted):
        yield encode_sse(json.dumps(redacted), event.event, event.id, event.retry)
    # Otherwise every field was absorbed into a hold-back buffer and the chunk
    # now carries nothing. Emitting an empty delta would be pure overhead.


app = create_app()

"""Customer-support MCP server (stdio transport).

Run it:

    python -m task1_mcp_server.server

Design notes
------------
**stdout is the wire.** Nothing but JSON-RPC frames may reach fd 1. Three
layers enforce that:

1. ``configure_logging`` sends the root logger to stderr and never installs a
   stdout handler.
2. ``_StdoutToStderr`` replaces ``sys.stdout`` for the life of the process, so
   a stray ``print()`` - in this module or in any library it imports - lands on
   stderr with a loud prefix instead of corrupting the stream.
3. The SDK's ``stdio_server()`` additionally points fd 1 at stderr while
   serving, so even a subprocess writing to the raw descriptor misses the wire.

**Two kinds of failure, two different wire shapes.** This is the distinction
the MCP spec draws, and it is what a client needs in order to behave sensibly:

* *Protocol errors* - malformed arguments, unknown tool - are JSON-RPC error
  responses (``-32602``). The call never happened; retrying it unchanged is
  pointless, and the agent must fix the request.
* *Tool execution errors* - unknown customer, refund over balance - are
  successful JSON-RPC responses carrying ``isError: true``. The call happened
  and the model is meant to see the failure text and reason about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from collections import Counter, deque
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ValidationError

from task1_mcp_server.schemas import (
    CUSTOMER_RECORD_SCHEMA,
    REFUND_RECEIPT_SCHEMA,
    GetCustomerRecordInput,
    TriggerRefundInput,
    json_schema_for,
)
from task1_mcp_server.store import (
    CustomerNotFoundError,
    CustomerStore,
    RefundExceedsBalanceError,
    RefundTooSmallError,
)
from task1_mcp_server.wire import StdoutToStderr, claim_stdio_wire  # noqa: F401

SERVER_NAME = "customer-support"
SERVER_VERSION = "1.0.0"

logger = logging.getLogger(SERVER_NAME)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def configure_logging(level: str | None = None) -> None:
    """Send all logging to stderr. Never add a stdout handler here."""
    resolved = (level or os.environ.get("MCP_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, resolved, logging.INFO))


# --------------------------------------------------------------------------- #
# Tool definitions
# --------------------------------------------------------------------------- #
TOOLS: list[types.Tool] = [
    types.Tool(
        name="get_customer_record",
        description=(
            "Look up a customer's billing record by id. Read-only. "
            "Returns name, email, plan, status and refundable balance."
        ),
        input_schema=json_schema_for(GetCustomerRecordInput),
        output_schema=CUSTOMER_RECORD_SCHEMA,
    ),
    types.Tool(
        name="trigger_refund",
        description=(
            "Issue a refund against a customer's refundable balance. "
            "This has a real financial side effect and is not idempotent."
        ),
        input_schema=json_schema_for(TriggerRefundInput),
        output_schema=REFUND_RECEIPT_SCHEMA,
    ),
]

_TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_customer_record": GetCustomerRecordInput,
    "trigger_refund": TriggerRefundInput,
}


# --------------------------------------------------------------------------- #
# Validation -> JSON-RPC error mapping
# --------------------------------------------------------------------------- #
def _format_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Flatten a pydantic ValidationError into client-actionable entries.

    ``input`` is echoed back only for short scalars: echoing an arbitrarily
    large rejected payload into an error message is how you turn a validation
    failure into an amplification vector.
    """
    entries: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        entry: dict[str, Any] = {"field": loc, "code": err["type"], "message": err["msg"]}
        received = err.get("input")
        if isinstance(received, (str, int, float, bool)) or received is None:
            text = repr(received)
            entry["received"] = text if len(text) <= 80 else text[:77] + "..."
        else:
            entry["received_type"] = type(received).__name__
        entries.append(entry)
    return entries


def validate_arguments(tool_name: str, arguments: dict[str, Any] | None) -> BaseModel:
    """Validate raw arguments, or raise ``MCPError`` with a JSON-RPC code.

    Both failure modes here are ``-32602 Invalid params``:
      * unknown tool name - per the MCP tools spec, an unknown tool is a
        protocol-level invalid-params error, not a tool result;
      * arguments that do not satisfy the tool's schema.
    """
    model = _TOOL_INPUT_MODELS.get(tool_name)
    if model is None:
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"Unknown tool: {tool_name}",
            data={
                "errors": [{
                    "field": "name",
                    "code": "unknown_tool",
                    "message": f"No tool named {tool_name!r}",
                }],
                "available_tools": sorted(_TOOL_INPUT_MODELS),
            },
        )

    # ``arguments`` is optional on the wire. Treat a missing object as ``{}`` so
    # the schema - not an AttributeError - reports the missing required fields.
    # ``arguments`` shape is screened in the pump, upstream of here, so this
    # function only ever sees a dict or None. There used to be a second check
    # here emitting a THIRD ``-32602`` shape (``{"received_type": ...}``, no
    # ``errors`` key); unreachable, but it broke the one-code-one-shape
    # contract the moment anything called this directly.
    raw = arguments if arguments is not None else {}

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"Invalid params for tool '{tool_name}'",
            data={"tool": tool_name, "errors": _format_validation_errors(exc)},
        ) from exc


def _ok(payload: dict[str, Any]) -> types.CallToolResult:
    """A successful tool result: human-readable text plus structured content."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        structured_content=payload,
    )


def _tool_error(message: str, **extra: Any) -> types.CallToolResult:
    """A *tool execution* failure - a successful JSON-RPC response, isError=true.

    Deliberately carries no ``structuredContent``. That field describes the
    tool's *success* shape, so emitting an error payload through it hands a
    client something that does not match what the tool said it returns. The
    failure detail goes in ``content``, which is where MCP puts it.
    """
    payload = {"error": message, **extra}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        is_error=True,
    )


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
#: Smallest drain wait the server will accept. Draining is not a delay to be
#: tuned away - it now waits on the set of outstanding requests, so it costs
#: nothing when there is no work in flight.
_MIN_DRAIN_MS = 250.0
_MAX_DRAIN_MS = 60_000.0


def _drain_grace_ms() -> float:
    """Parse ``MCP_DRAIN_GRACE_MS`` as a drain *timeout*, not an on/off switch.

    It used to be a fixed post-EOF sleep that could be set to zero, and zero
    meant a request already in flight was abandoned: 200 piped refunds executed
    200 refunds and returned 112 receipts, leaving 88 that moved money and told
    the client nothing. On a tool that issues refunds, "disable the drain" is
    not a setting anyone should be able to choose, so the value is clamped to a
    floor.

    Malformed values are rejected rather than propagated: ``abc`` raised an
    unhandled ValueError at startup, ``1e400`` became ``inf`` and the server
    never exited after end-of-input, and ``nan`` and negatives degraded to no
    wait at all.
    """
    raw = os.environ.get("MCP_DRAIN_GRACE_MS", "250")
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring MCP_DRAIN_GRACE_MS=%r: not a number", raw)
        return _MIN_DRAIN_MS
    if value != value or value < 0:  # NaN or negative
        logger.warning("ignoring MCP_DRAIN_GRACE_MS=%r: out of range", raw)
        return _MIN_DRAIN_MS
    if value < _MIN_DRAIN_MS:
        logger.warning(
            "MCP_DRAIN_GRACE_MS=%r is below the %.0fms floor; a shorter drain abandons "
            "in-flight requests that have already had their effect", raw, _MIN_DRAIN_MS
        )
        return _MIN_DRAIN_MS
    return min(value, _MAX_DRAIN_MS)


def _sanitised(handler):
    """Map any unexpected handler exception to a bare ``-32603``.

    Without this the SDK surfaced a raw ``RuntimeError`` as
    ``{"code": 0, "message": "<the exception text>"}`` - not a valid JSON-RPC
    code, and carrying whatever the exception happened to say. A stack of file
    paths and internal hostnames reached the client from the one service here
    that moves money. The detail belongs in the log, correlated by tool name.
    """

    async def guarded(ctx, params):
        try:
            return await handler(ctx, params)
        except MCPError:
            raise  # already a deliberate, sanitised protocol error
        except Exception:
            logger.exception("unhandled error in %s", getattr(handler, "__name__", "handler"))
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal server error") from None

    guarded.__name__ = getattr(handler, "__name__", "guarded")
    return guarded


def build_server(store: CustomerStore | None = None) -> Server:
    store = store or CustomerStore()

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        logger.debug("tools/list")
        return types.ListToolsResult(tools=TOOLS)

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        name = params.name
        # Raises MCPError(-32602) for unknown tool or schema violation. Letting
        # it propagate is what produces a JSON-RPC error frame rather than a
        # result - the call is rejected before any side effect can run.
        args = validate_arguments(name, params.arguments)
        logger.info("tools/call %s", name)

        if isinstance(args, GetCustomerRecordInput):
            try:
                return _ok(store.get(args.customer_id))
            except CustomerNotFoundError:
                return _tool_error(f"No customer found with id {args.customer_id}", customer_id=args.customer_id)

        if isinstance(args, TriggerRefundInput):
            try:
                receipt = store.refund(args.customer_id, args.amount, args.reason)
            except CustomerNotFoundError:
                return _tool_error(f"No customer found with id {args.customer_id}", customer_id=args.customer_id)
            except (RefundExceedsBalanceError, RefundTooSmallError) as exc:
                return _tool_error(str(exc), customer_id=args.customer_id, requested_amount=args.amount)
            logger.info("refund %s issued for %s", receipt["refund_id"], args.customer_id)
            return _ok(receipt)

        # Unreachable: every model in _TOOL_INPUT_MODELS is handled above.
        raise MCPError(code=types.INTERNAL_ERROR, message=f"No handler wired for tool '{name}'")

    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="Customer Support",
        instructions=(
            "Look up customer billing records and issue refunds. "
            "Customer ids are formatted CUST-XXXXX (five digits)."
        ),
        on_list_tools=_sanitised(on_list_tools),
        on_call_tool=_sanitised(on_call_tool),
    )
    return server


class LineRecordingInput:
    """Wraps the wire's input and keeps each raw line for the pump to consult.

    The SDK's stdio transport parses each line and, on failure, forwards the
    *exception* into the read stream - the offending text is gone by then. But
    JSON-RPC requires a ``-32700``/``-32600`` response, and building one needs
    the original line: to tell a parse failure from a structurally invalid
    request, and to recover the ``id`` to echo.

    Lines go into a FIFO rather than a "last line" slot. The transport emits
    exactly one stream item per line, so the pump popping one line per item is
    an exact pairing - whereas a single slot races, because the reader can
    advance to the next line between the pump receiving an item and inspecting
    the slot.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._iterator = None
        self.lines: deque[str] = deque()

    def __aiter__(self) -> LineRecordingInput:
        # ``anyio``'s AsyncFile implements __aiter__ as an async generator, so
        # the iterator has to be taken from it rather than assumed to be self.
        self._iterator = self._wrapped.__aiter__()
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            self._iterator = self._wrapped.__aiter__()
        line = await self._iterator.__anext__()
        self.lines.append(line)
        return line

    def take(self) -> str:
        return self.lines.popleft() if self.lines else ""


def error_for_unparsed_line(line: str) -> types.JSONRPCError:
    """Build the JSON-RPC error owed to a line the transport could not accept.

    * Not JSON at all -> ``-32700 Parse error`` with a null id, since no id
      can be recovered.
    * Valid JSON that is not a valid request -> ``-32600 Invalid Request``,
      echoing the id when the payload carries a usable one.

    Batch arrays land in the second case deliberately: batching was removed
    from MCP in protocol revision 2025-06-18, so refusing is correct - but
    refusing *silently*, which is what happened before, leaves the client
    waiting forever for a response that is never coming.
    """
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError):
        return types.JSONRPCError(
            jsonrpc="2.0",
            id=None,
            error=types.ErrorData(
                code=types.PARSE_ERROR,
                message="Parse error",
                data={"errors": [{
                    "field": "<message>", "code": "parse_error",
                    "message": "the line was not valid JSON",
                }]},
            ),
        )

    request_id = None
    detail = "Invalid Request"
    if isinstance(payload, list):
        detail = "Invalid Request: JSON-RPC batching is not supported"
    elif isinstance(payload, dict):
        candidate = payload.get("id")
        # bool is an int subclass; a boolean id is not a valid id.
        #
        # A float is deliberately NOT echoed here, and this is where Task 1 and
        # Task 2 legitimately differ. MCP narrows the JSON-RPC id to a string
        # or an integer, so a float id is invalid *at this layer* and there is
        # no conformant id to echo - null is the honest answer. Task 2 proxies
        # plain JSON-RPC 2.0, where any number is a valid id, so it echoes one.
        # Same-looking code, two different specifications.
        if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
            request_id = candidate

    return types.JSONRPCError(
        jsonrpc="2.0",
        id=request_id,
        error=types.ErrorData(
            code=types.INVALID_REQUEST,
            message=detail,
            # Same ``data.errors`` shape as every other rejection: one error
            # code meaning one response shape is the whole point.
            data={"errors": [{"field": "<envelope>", "code": "invalid_request", "message": detail}]},
        ),
    )


def tools_call_envelope_error(params: Any) -> dict[str, Any] | None:
    """Reject a malformed ``tools/call`` envelope with the same shape as a
    schema failure, or return ``None`` if it is well formed.

    The SDK surface-validates spec methods *before* dispatching to a handler,
    and its rejection carries no detail: a missing ``name`` produced ``-32602``
    with ``"data": ""``, while a rejection from the schema layer carried a full
    per-field breakdown. Two shapes for one error code is a contract an agent
    cannot program against, so the envelope is screened here - upstream of the
    SDK's check - and every ``tools/call`` rejection now looks the same.
    """
    if params is None:
        # Only *absent* params reach this. A non-dict ``params`` - an array,
        # say - is rejected by the SDK's own model at parse time and arrives as
        # a -32600 envelope error, so the branch that used to test for it here
        # could never run. Removed rather than left as reassuring dead code.
        return {
            "errors": [{
                "field": "params",
                "code": "missing",
                "message": "tools/call requires a params object",
            }],
        }

    name = params.get("name")
    if not isinstance(name, str) or not name:
        return {
            "errors": [{
                "field": "name",
                "code": "missing" if name is None else "string_type",
                "message": "params.name must be a non-empty string naming a tool",
            }],
            "available_tools": sorted(_TOOL_INPUT_MODELS),
        }

    arguments = params.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        return {
            "tool": name,
            "errors": [{
                "field": "arguments",
                "code": "dict_type",
                "message": "params.arguments must be a JSON object",
                "received_type": type(arguments).__name__,
            }],
        }
    return None


async def _pump_send(stream: Any, item: Any) -> None:
    """Send a pump-generated frame without crediting the drain."""
    sender = getattr(stream, "untracked_send", None)
    if sender is not None:
        await sender(item)
    else:  # pragma: no cover - only when the stream is not wrapped
        await stream.send(item)


def _is_request_masquerading_as_notification(message, line: str) -> bool:
    """True when the raw line carried an ``id`` the message model dropped."""
    inner = getattr(message, "message", message)
    if not isinstance(inner, types.JSONRPCNotification):
        return False
    try:
        payload = json.loads(line)
    except Exception:
        return False
    # MCP forbids a null request id, so an explicit ``"id": null`` is an
    # invalid request rather than a notification - and answering nothing left
    # the client waiting forever, which is the exact bug class this closes.
    return isinstance(payload, dict) and "id" in payload


class DrainTracker:
    """Tracks requests that have been dispatched but not yet answered.

    A fixed sleep after end-of-input is not a drain, it is a tuned race - and
    it lost: with ``MCP_DRAIN_GRACE_MS=0``, 200 piped refund requests executed
    200 refunds and returned 112 receipts. Eighty-eight refunds moved money and
    told the client nothing, whose only recourse is to retry and refund twice.

    Waiting on the actual set of outstanding request ids removes the guess.
    """

    def __init__(self) -> None:
        # A COUNTER, not a set. JSON-RPC says a client must not reuse an
        # outstanding id, but this server accepts and executes those requests -
        # so it takes the money either way. With a set, N concurrent requests
        # sharing an id collapsed to one entry and the first answer emptied it:
        # 100 refunds executed, 56 receipts returned. That is precisely the
        # failure the drain was written to eliminate, reintroduced one layer
        # down by the fix's own data structure.
        self._pending: Counter[Any] = Counter()
        self._idle = anyio.Event()
        self._idle.set()

    def dispatched(self, request_id: Any) -> None:
        self._pending[request_id] += 1
        self._idle = anyio.Event()

    def answered(self, request_id: Any) -> None:
        if self._pending.get(request_id):
            self._pending[request_id] -= 1
            if self._pending[request_id] <= 0:
                del self._pending[request_id]
        if not self._pending:
            self._idle.set()

    def is_outstanding(self, request_id: Any) -> bool:
        return bool(self._pending.get(request_id))

    async def wait_idle(self, timeout: float) -> bool:
        """True if everything drained, False if the timeout won."""
        if not self._pending:
            return True
        with anyio.move_on_after(timeout) as scope:
            await self._idle.wait()
        return not scope.cancel_called


#: ``code`` for an entry the SDK produced rather than the schema layer. Named
#: so a client can tell "the server refused this before dispatch" from a
#: specific field failure like ``value_error`` or ``extra_forbidden``.
_SURFACE_ERROR_CODE = "request_rejected"


def _normalise_error_detail(message: Any) -> None:
    """Give every error frame the same ``data`` shape, whoever produced it.

    The SDK surface-validates spec methods before dispatch, and its own
    rejections carry ``data`` as a bare string - ``""`` for an invalid-params
    failure, the method name for method-not-found - while every rejection from
    the schema layer carries ``data.errors`` with per-field detail. One error
    code meaning two response shapes is a contract an agent cannot program
    against: ``error["data"]["errors"]`` raises TypeError on half of them.

    Rewritten in place on the way out, so there is one shape regardless of
    which layer refused.

    Only the two shapes the SDK actually emits are touched: ``None`` and a bare
    string. Anything already structured is left exactly as it is - a previous
    version of this transform tested ``not isinstance(data, dict)`` and so
    destroyed a deliberate *array* payload, which JSON-RPC explicitly permits
    ("a Primitive or Structured value"), replacing two field hints with a
    wrong one. There is a test for that, and it caught this rewrite.
    """
    error = getattr(message, "error", None)
    if error is None:
        return
    detail = getattr(error, "data", None)
    if detail is not None and not isinstance(detail, str):
        return
    # Same keys as a schema-layer entry, ``code`` included. Producing entries
    # without it left ``error["data"]["errors"][0]["code"]`` raising KeyError
    # on exactly the paths this function exists to make uniform - the shape was
    # normalised, the entry was not.
    entry: dict[str, Any] = {
        "field": None,
        "code": _SURFACE_ERROR_CODE,
        "message": getattr(error, "message", "") or "",
    }
    if isinstance(detail, str) and detail:
        entry["detail"] = detail[:200]
    try:
        error.data = {"errors": [entry]}
    except (AttributeError, ValueError):  # pragma: no cover - frozen model
        pass


class TrackingWriteStream:
    """Wraps the transport's write side to observe outgoing responses.

    Its only job is to tell the drain tracker when a request has been answered,
    so shutdown can wait for work that is still in flight instead of guessing.
    """

    def __init__(self, inner, tracker: DrainTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    async def send(self, item: Any) -> None:
        message = getattr(item, "message", item)
        _normalise_error_detail(getattr(message, "root", message))
        await self._inner.send(item)
        request_id = getattr(message, "id", None)
        if request_id is not None:
            self._tracker.answered(request_id)

    async def untracked_send(self, item: Any) -> None:
        """Write a frame WITHOUT crediting the drain.

        The pump answers malformed frames itself, without the server ever
        seeing them. Sending those through the tracked path credited the drain
        for work that was never dispatched - and when the malformed frame
        reused an in-flight id, it cancelled a real refund's drain slot. Which
        writer produced a frame is not recoverable from the frame, so the two
        writers use two methods.
        """
        await self._inner.send(item)

    # Dunder methods are looked up on the type, not the instance, so
    # __getattr__ does not forward them - the transport uses this as an async
    # context manager and got a TypeError.
    async def __aenter__(self) -> TrackingWriteStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        return await self._inner.__aexit__(*exc)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextlib.asynccontextmanager
async def protocol_error_pump(read_stream, write_stream, recorder, seconds: float, tracker=None) -> Any:
    """Sit between the transport and the server, doing two jobs.

    **Answer malformed frames.** The transport hands unparseable input to the
    server as an exception, which the run loop logs at DEBUG and drops - so at
    the default log level a malformed request produced no response and no
    output at all, and the client hung. Those are intercepted here and turned
    into the ``-32700``/``-32600`` frames the spec requires.

    **Delay end-of-input by ``seconds``.** When stdin reaches EOF the run loop
    shuts down and cancels any request still being handled, so its response
    never reaches the wire. A long-lived client never notices, because it holds
    stdin open. But ``... < requests.jsonl``, which is how anyone first pokes at
    a stdio server, feeds every line at once and hits EOF immediately, losing
    the last reply. (An unmodified SDK server loses more than one.) Set
    ``MCP_DRAIN_GRACE_MS`` sets that wait in milliseconds. It is clamped to
    [250, 60000] and is NOT an on/off switch - 0 is raised to the floor, with a
    warning, because a drain of zero loses exactly the replies this exists to
    keep. The docstring used to say 0 disabled it, which it never did.
    """
    send, receive = anyio.create_memory_object_stream(max_buffer_size=64)

    async def pump() -> None:
        async with send:
            async for message in read_stream:
                line = recorder.take() if recorder is not None else ""

                # A blank or whitespace-only line is framing, not a request.
                # Answering it with -32700 meant a trailing newline in a
                # requests file produced an unsolicited parse error.
                if isinstance(message, Exception) and not line.strip():
                    continue

                if isinstance(message, Exception):
                    frame = error_for_unparsed_line(line)
                    logger.warning(
                        "rejected malformed frame: code=%s (%d bytes)",
                        frame.error.code,
                        len(line),
                    )
                    await _pump_send(write_stream, SessionMessage(frame))
                    continue

                # A request whose ``id`` is not a string or integer - a float,
                # an object, a bool - does not fail validation. It falls
                # through the message union to the *notification* variant,
                # which has no id, so the server answers nothing and the client
                # waits forever. Catch it by comparing against the raw line.
                inner = getattr(message, "message", message)
                if (
                    isinstance(inner, types.JSONRPCRequest)
                    and inner.method == "tools/call"
                ):
                    envelope_error = tools_call_envelope_error(inner.params)
                    if envelope_error is not None:
                        logger.warning("rejected malformed tools/call envelope")
                        await _pump_send(
                            write_stream,
                            SessionMessage(
                                types.JSONRPCError(
                                    jsonrpc="2.0",
                                    id=inner.id,
                                    error=types.ErrorData(
                                        code=types.INVALID_PARAMS,
                                        message="Invalid params for tools/call",
                                        data=envelope_error,
                                    ),
                                )
                            )
                        )
                        continue

                if _is_request_masquerading_as_notification(message, line):
                    logger.warning("rejected frame with an unusable id")
                    await _pump_send(
                        write_stream,
                        SessionMessage(
                            types.JSONRPCError(
                                jsonrpc="2.0",
                                id=None,
                                error=types.ErrorData(
                                    code=types.INVALID_REQUEST,
                                    message="Invalid Request: 'id' must be a string or integer",
                                    data={"errors": [{
                                        "field": "id",
                                        "code": "id_type",
                                        "message": "a request id must be a string or an integer",
                                    }]},
                                ),
                            )
                        )
                    )
                    continue

                inner = getattr(message, "message", message)
                if tracker is not None and isinstance(inner, types.JSONRPCRequest):
                    tracker.dispatched(inner.id)
                await send.send(message)

            # Input is exhausted. Wait for the requests already dispatched to be
            # answered, rather than sleeping a fixed interval and hoping.
            if tracker is not None and seconds > 0:
                if not await tracker.wait_idle(seconds):
                    logger.warning(
                        "shutting down with %d request(s) unanswered", sum(tracker._pending.values())
                    )
            elif seconds > 0:
                await anyio.sleep(seconds)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(pump)
        try:
            yield receive
        finally:
            task_group.cancel_scope.cancel()


async def serve(wire_in, wire_out, store: CustomerStore | None = None) -> None:
    """Run the server over an already-claimed wire.

    The claim happens in ``__main__`` before this module is imported, so the
    descriptors arrive here as plain file objects. Both are handed to the
    transport explicitly; fd 0 and fd 1 are already pointed elsewhere.
    """
    server = build_server(store)
    grace_ms = _drain_grace_ms()
    logger.info("%s v%s starting on stdio", SERVER_NAME, SERVER_VERSION)

    recorder = LineRecordingInput(anyio.wrap_file(wire_in))
    tracker = DrainTracker()
    async with stdio_server(stdin=recorder, stdout=anyio.wrap_file(wire_out)) as (read_stream, write_stream):
        tracked = TrackingWriteStream(write_stream, tracker)
        async with protocol_error_pump(read_stream, tracked, recorder, grace_ms / 1000, tracker) as inbound:
            await server.run(inbound, tracked, server.create_initialization_options())


async def main() -> None:
    """Claim the wire and serve. Prefer ``python -m task1_mcp_server``.

    That entry point claims the wire *before* importing this module, so it also
    covers anything a dependency prints while being imported. This function
    exists for programmatic use, where the imports have already happened.
    """
    configure_logging()
    with claim_stdio_wire() as (wire_in, wire_out):
        await serve(wire_in, wire_out)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        logger.info("shutting down")

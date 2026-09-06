"""Task 1: strict validation, JSON-RPC error mapping, and stdout isolation.

The end-to-end tests spawn the real server as a subprocess and speak raw
JSON-RPC over its stdin/stdout, so they exercise the actual transport rather
than an in-process shortcut.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.shared.message import SessionMessage  # noqa: E402
from task1_mcp_server.server import DrainTracker  # noqa: E402
from task1_mcp_server.store import CustomerStore  # noqa: E402
from tests.stdio_client import StdioServerProcess  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_NOISY_SERVER = r"""
            import asyncio, ctypes, os, subprocess, sys
            from task1_mcp_server.wire import claim_stdio_wire

            with claim_stdio_wire() as (wire_in, wire_out):
                import anyio
                import mcp_types as types
                from mcp.server.stdio import stdio_server
                from task1_mcp_server import server as base

                def build_noisy():
                    srv = base.build_server()
                    original = srv.get_request_handler("tools/call").handler

                    async def noisy_handler(ctx, params):
                        print("chatty library banner")
                        os.write(1, b"raw fd 1 write from a child\n")
                        sys.stdout.write("another stray line\n")
                        sys.__stdout__.write("via sys dunder stdout\n")
                        sys.__stdout__.flush()
                        subprocess.run(["/bin/echo", "from a subprocess"], check=False)
                        ctypes.CDLL(None).write(1, b"raw libc write\n", 15)
                        return await original(ctx, params)

                    srv.add_request_handler("tools/call", types.CallToolRequestParams, noisy_handler)
                    return srv

                async def main():
                    base.configure_logging()
                    srv = build_noisy()
                    recorder = base.LineRecordingInput(anyio.wrap_file(wire_in))
                    async with stdio_server(stdin=recorder, stdout=anyio.wrap_file(wire_out)) as (r, w):
                        async with base.protocol_error_pump(r, w, recorder, 0.25) as inbound:
                            await srv.run(inbound, w, srv.create_initialization_options())

                asyncio.run(main())
"""
INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601


@pytest.fixture
def server():
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, MCP_LOG_LEVEL="DEBUG")
    with StdioServerProcess([sys.executable, "-m", "task1_mcp_server"], cwd=REPO_ROOT, env=env) as proc:
        proc.initialize()
        yield proc


def call(proc: StdioServerProcess, name: str, arguments: dict | None) -> dict:
    params: dict = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return proc.request("tools/call", params)


# --------------------------------------------------------------------------- #
# Protocol flow
# --------------------------------------------------------------------------- #
def test_initialize_and_list_tools(server):
    response = server.request("tools/list")
    assert "error" not in response
    tools = {t["name"]: t for t in response["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    refund_schema = tools["trigger_refund"]["inputSchema"]
    assert refund_schema["additionalProperties"] is False
    assert refund_schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund_schema["properties"]["reason"]["minLength"] == 10
    assert refund_schema["properties"]["customer_id"]["pattern"] == r"^CUST-[0-9]{5}$"


def test_unknown_method_is_method_not_found(server):
    response = server.request("tools/nonexistent")
    assert response["error"]["code"] == METHOD_NOT_FOUND


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
def test_get_customer_record_success(server):
    response = call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    assert "error" not in response
    result = response["result"]
    assert not result.get("isError")
    assert result["structuredContent"]["name"] == "Ada Lovelace"


def test_trigger_refund_success_and_balance_moves(server):
    response = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 100.0, "reason": "Duplicate charge on invoice 88213"},
    )
    receipt = response["result"]["structuredContent"]
    assert receipt["status"] == "accepted"
    assert receipt["refund_id"].startswith("RFND-")
    assert receipt["remaining_refundable_balance"] == 400.0

    after = call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    assert after["result"]["structuredContent"]["refundable_balance"] == 400.0


def test_integer_amount_is_accepted_as_float(server):
    """int -> float is the one coercion strict mode should still allow."""
    response = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-01337", "amount": 25, "reason": "Service credit for outage"},
    )
    assert "error" not in response
    assert response["result"]["structuredContent"]["amount"] == 25.0


# --------------------------------------------------------------------------- #
# Protocol errors: malformed input must be -32602, never a tool result
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arguments, expect_field",
    [
        ({"customer_id": "CUST-1"}, "customer_id"),               # too short
        ({"customer_id": "cust-00042"}, "customer_id"),            # lowercase prefix
        ({"customer_id": "CUST-ABCDE"}, "customer_id"),            # non-digits
        ({"customer_id": "CUST-000420"}, "customer_id"),           # too long
        ({"customer_id": " CUST-00042"}, "customer_id"),           # leading space
        ({"customer_id": "CUST-00042 "}, "customer_id"),           # trailing space
        ({"customer_id": "CUST-00042\n"}, "customer_id"),          # newline injection
        ({"customer_id": 42}, "customer_id"),                      # wrong type
        ({"customer_id": None}, "customer_id"),                    # null
        ({}, "customer_id"),                                       # missing
        ({"customer_id": "CUST-00042", "extra": "x"}, "extra"),    # extra key forbidden
    ],
)
def test_get_customer_record_rejects_malformed_input(server, arguments, expect_field):
    response = call(server, "get_customer_record", arguments)
    assert "result" not in response, f"malformed input leaked through as a result: {response}"
    error = response["error"]
    assert error["code"] == INVALID_PARAMS
    fields = {e["field"] for e in error["data"]["errors"]}
    assert expect_field in fields


@pytest.mark.parametrize(
    "arguments, expect_field",
    [
        ({"customer_id": "CUST-00042", "amount": 0, "reason": "Duplicate charge seen"}, "amount"),
        ({"customer_id": "CUST-00042", "amount": -5.0, "reason": "Duplicate charge seen"}, "amount"),
        ({"customer_id": "CUST-00042", "amount": "100.00", "reason": "Duplicate charge seen"}, "amount"),
        ({"customer_id": "CUST-00042", "amount": 10.0, "reason": "too short"}, "reason"),
        ({"customer_id": "CUST-00042", "amount": 10.0, "reason": ""}, "reason"),
        ({"customer_id": "CUST-00042", "amount": 10.0}, "reason"),
        ({"customer_id": "CUST-00042", "reason": "Duplicate charge seen"}, "amount"),
        ({"customer_id": "BAD", "amount": 10.0, "reason": "Duplicate charge seen"}, "customer_id"),
    ],
)
def test_trigger_refund_rejects_malformed_input(server, arguments, expect_field):
    response = call(server, "trigger_refund", arguments)
    assert "result" not in response, f"malformed refund leaked through as a result: {response}"
    error = response["error"]
    assert error["code"] == INVALID_PARAMS
    fields = {e["field"] for e in error["data"]["errors"]}
    assert expect_field in fields


def test_nan_and_infinity_amounts_are_rejected(server):
    """Many JSON encoders emit bare NaN/Infinity. They are never a valid refund."""
    for literal in ("NaN", "Infinity", "-Infinity"):
        server._next_id += 1
        request_id = server._next_id
        raw = (
            '{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":"trigger_refund",'
            '"arguments":{"customer_id":"CUST-00042","amount":%s,"reason":"Duplicate charge seen"}}}'
            % (request_id, literal)
        )
        server._proc.stdin.write(raw + "\n")
        server._proc.stdin.flush()
        while True:
            message = json.loads(server._read_line())
            if message.get("id") == request_id:
                break
        assert "result" not in message, f"{literal} was accepted as an amount"
        assert message["error"]["code"] == INVALID_PARAMS


@pytest.mark.parametrize(
    "params, expect_field",
    [
        ({"arguments": {}}, "name"),                                       # no name
        ({"name": 5}, "name"),                                             # non-string name
        ({"name": ""}, "name"),                                            # empty name
        ({"name": "get_customer_record", "arguments": [1, 2]}, "arguments"),
        ({"name": "get_customer_record", "arguments": "x"}, "arguments"),
        (None, "params"),                                                  # no params at all
    ],
)
def test_every_tools_call_rejection_has_the_same_shape(server, params, expect_field):
    """One error code must mean one response shape.

    The SDK surface-validates spec methods before dispatching to a handler, and
    its rejection carried no detail - a missing ``name`` produced ``-32602``
    with ``"data": ""``, while a rejection from the schema layer carried a full
    per-field breakdown. An agent cannot program against two shapes for one
    code, so the envelope is screened upstream of the SDK's check.
    """
    payload = {"jsonrpc": "2.0", "id": 99, "method": "tools/call"}
    if params is not None:
        payload["params"] = params
    response = server.request("tools/call", params) if params is not None else server.request("tools/call")

    error = response["error"]
    assert error["code"] == INVALID_PARAMS
    assert isinstance(error["data"], dict), f"no structured detail: {error!r}"
    fields = {e["field"] for e in error["data"]["errors"]}
    assert expect_field in fields


def test_unknown_tool_is_invalid_params(server):
    response = call(server, "admin_drop_database", {"customer_id": "CUST-00042"})
    assert response["error"]["code"] == INVALID_PARAMS
    assert "Unknown tool" in response["error"]["message"]


def test_missing_arguments_object_reports_missing_fields(server):
    response = call(server, "trigger_refund", None)
    error = response["error"]
    assert error["code"] == INVALID_PARAMS
    fields = {e["field"] for e in error["data"]["errors"]}
    assert {"customer_id", "amount", "reason"} <= fields


def test_validation_failure_has_no_side_effect(server):
    """A rejected refund must not move money."""
    before = call(server, "get_customer_record", {"customer_id": "CUST-01337"})
    balance_before = before["result"]["structuredContent"]["refundable_balance"]
    rejected = call(server, "trigger_refund", {"customer_id": "CUST-01337", "amount": 5.0, "reason": "short"})
    assert rejected["error"]["code"] == INVALID_PARAMS
    after = call(server, "get_customer_record", {"customer_id": "CUST-01337"})
    assert after["result"]["structuredContent"]["refundable_balance"] == balance_before


# --------------------------------------------------------------------------- #
# Tool execution errors: valid request, failed operation -> isError result
# --------------------------------------------------------------------------- #


def test_refund_over_balance_is_tool_error(server):
    response = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00007", "amount": 250.0, "reason": "Refund request from support ticket 41"},
    )
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "structuredContent" not in response["result"]
    assert "exceeds refundable balance" in response["result"]["content"][0]["text"]


def test_an_enormous_finite_amount_is_refused_not_a_crash(server):
    """``amount * 100`` overflows to infinity above ~1.8e306 and ``round(inf)``
    raises OverflowError, so a large but perfectly finite amount escaped the
    schema and surfaced as -32603 "Internal server error" from the one tool
    that moves money. It is a refusal, not a server fault."""
    response = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 1e307, "reason": "Duplicate charge seen"},
    )
    assert "error" not in response, f"a finite amount became a protocol error: {response}"
    assert response["result"]["isError"] is True
    assert "exceeds refundable balance" in response["result"]["content"][0]["text"]


def test_the_balance_is_untouched_after_an_enormous_amount(server):
    call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 1e307, "reason": "Duplicate charge seen"},
    )
    record = call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    assert record["result"]["structuredContent"]["refundable_balance"] == 500.0
    assert record["result"]["structuredContent"]["refund_count"] == 0


@pytest.mark.parametrize("amount", [0.001, 0.004, 1e-320])
def test_a_refund_that_rounds_to_zero_cents_is_refused(server, amount):
    """A positive sub-cent amount used to return a receipt saying
    ``"status": "accepted"`` with ``"amount": 0.0`` - while the balance was
    untouched and the refund count still went up. Accepting a refund that
    moves no money is a lie to whatever reads the receipt."""
    response = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00042", "amount": amount, "reason": "Rounding correction here"},
    )
    assert "error" not in response
    assert response["result"]["isError"] is True, f"sub-cent refund was accepted: {response}"
    assert "rounds to zero cents" in response["result"]["content"][0]["text"]

    record = call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    assert record["result"]["structuredContent"]["refundable_balance"] == 500.0
    assert record["result"]["structuredContent"]["refund_count"] == 0, "a no-op refund was recorded"


# --------------------------------------------------------------------------- #
# Malformed frames must be answered, not silently dropped
# --------------------------------------------------------------------------- #
def raw_roundtrip(lines: list[str]) -> list[dict]:
    """Send raw lines through a fresh server and collect every response."""
    handshake = [
        json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2026-07-28", "capabilities": {},
                       "clientInfo": {"name": "cli", "version": "1"}},
        }),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "task1_mcp_server"],
        input="\n".join(handshake + lines) + "\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT),
        timeout=30,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_unparseable_line_gets_a_parse_error():
    """Previously produced no response at all - and, at the default log level,
    no output either. The client simply hung."""
    responses = raw_roundtrip(["this is not json at all"])
    errors = [r for r in responses if "error" in r]
    assert errors, "a malformed frame produced no response"
    assert errors[0]["error"]["code"] == -32700
    assert errors[0]["id"] is None


@pytest.mark.parametrize(
    "line, expect_id",
    [
        ('{"id":7,"method":"tools/list"}', 7),                       # no jsonrpc member
        ('{"jsonrpc":"1.0","id":8,"method":"tools/list"}', 8),        # wrong version
        ('{"jsonrpc":"2.0","id":"abc"}', "abc"),                      # no method
        ('[{"jsonrpc":"2.0","id":9,"method":"tools/list"}]', None),   # batch: removed in MCP
        ('42', None),
        ('null', None),
    ],
)
def test_structurally_invalid_frames_get_invalid_request(line, expect_id):
    responses = raw_roundtrip([line])
    errors = [r for r in responses if "error" in r]
    assert errors, f"no response for {line}"
    assert errors[0]["error"]["code"] == -32600
    assert errors[0]["id"] == expect_id, "the id should be echoed when one can be recovered"


@pytest.mark.parametrize("bad_id", ["9.5", "{\"a\":1}", "true", "[1]"])
def test_requests_with_an_unusable_id_are_answered(bad_id):
    """An id that is not a string or integer used to fall through the message
    union to the *notification* variant - no validation error, no response, and
    a client waiting forever."""
    responses = raw_roundtrip(['{"jsonrpc":"2.0","id":%s,"method":"tools/list"}' % bad_id])
    errors = [r for r in responses if "error" in r]
    assert errors, f"id {bad_id} produced no response"
    assert errors[0]["error"]["code"] == -32600
    assert "id" in errors[0]["error"]["message"]


def test_a_malformed_frame_does_not_break_the_session():
    """The connection must survive: later requests still get answered."""
    responses = raw_roundtrip([
        "garbage",
        '{"jsonrpc":"2.0","id":50,"method":"tools/list"}',
    ])
    by_id = {r.get("id"): r for r in responses}
    assert by_id[None]["error"]["code"] == -32700
    assert "result" in by_id[50]


def test_legitimate_notifications_still_get_no_response():
    responses = raw_roundtrip([
        '{"jsonrpc":"2.0","method":"notifications/progress",'
        '"params":{"progressToken":"t","progress":1}}'
    ])
    assert [r["id"] for r in responses] == [1], "a notification was answered"


def test_unhandled_handler_error_is_a_sanitised_internal_error(tmp_path):
    """An unhandled exception used to reach the wire as ``{"code": 0,
    "message": "<the exception text>"}`` - not a valid JSON-RPC code, and
    carrying file paths and hostnames out of the process."""
    secret = "SECRET-INTERNAL /var/secrets/db.pem at 10.0.3.7"
    boom = tmp_path / "boom_server.py"
    boom.write_text(textwrap.dedent(f'''
        import asyncio, sys
        from task1_mcp_server.wire import claim_stdio_wire

        with claim_stdio_wire() as (wire_in, wire_out):
            import anyio, mcp_types as types
            from mcp.server.stdio import stdio_server
            from task1_mcp_server import server as base

            async def main():
                base.configure_logging()
                srv = base.build_server()
                async def boom(ctx, params):
                    raise RuntimeError({secret!r})
                srv.add_request_handler("tools/call", types.CallToolRequestParams, base._sanitised(boom))
                recorder = base.LineRecordingInput(anyio.wrap_file(wire_in))
                async with stdio_server(stdin=recorder, stdout=anyio.wrap_file(wire_out)) as (r, w):
                    async with base.protocol_error_pump(r, w, recorder, 0.25) as inbound:
                        await srv.run(inbound, w, srv.create_initialization_options())

            asyncio.run(main())
    '''))
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    with StdioServerProcess([sys.executable, str(boom)], cwd=REPO_ROOT, env=env) as proc:
        proc.initialize()
        response = call(proc, "get_customer_record", {"customer_id": "CUST-00042"})

    assert response["error"]["code"] == -32603
    assert response["error"]["message"] == "Internal server error"
    body = json.dumps(response)
    assert secret not in body
    assert "/var/secrets" not in body and "10.0.3.7" not in body and "Traceback" not in body
    # The detail is still available to the operator.
    assert secret in "".join(proc.stderr_text)


# --------------------------------------------------------------------------- #
# Validation edge cases found by adversarial review
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "customer_id",
    ["CUST-\u0660\u0661\u0662\u0663\u0664",   # Arabic-Indic digits
     "CUST-\u0966\u0967\u0968\u0969\u096a",   # Devanagari digits
     "CUST-\uff10\uff11\uff12\uff13\uff14"],  # fullwidth digits
)
def test_non_ascii_digits_are_rejected(server, customer_id):
    r"""``\d`` is Unicode-aware in Python but ASCII-only in the ECMA-262 regex
    published as the tool's JSON Schema. The two must not disagree."""
    response = call(server, "get_customer_record", {"customer_id": customer_id})
    assert "result" not in response, f"{customer_id!r} passed validation"
    assert response["error"]["code"] == INVALID_PARAMS




def test_one_cent_is_still_a_valid_refund(server):
    response = call(server, "trigger_refund", {
        "customer_id": "CUST-00042", "amount": 0.01, "reason": "Rounding correction applied"})
    assert response["result"]["structuredContent"]["amount"] == 0.01


def test_zero_balance_customer_cannot_be_refunded_at_all(server):
    """CUST-00007 has a zero balance; no amount should ever succeed."""
    for amount in (0.01, 0.5, 100.0):
        response = call(server, "trigger_refund", {
            "customer_id": "CUST-00007", "amount": amount, "reason": "Probing the balance floor"})
        assert "error" in response or response["result"]["isError"] is True
    after = call(server, "get_customer_record", {"customer_id": "CUST-00007"})
    assert after["result"]["structuredContent"]["refundable_balance"] == 0.0


def test_a_refund_that_empties_a_balance_does_not_produce_negative_zero():
    """The previous assertion was unreachable by construction: every refund in
    that test is refused, so the balance is never written and the check could
    not fail."""
    store = CustomerStore()
    balance = store.get("CUST-01337")["refundable_balance"]
    receipt = store.refund("CUST-01337", balance, "Emptying the balance exactly")
    assert receipt["remaining_refundable_balance"] == 0.0
    assert str(receipt["remaining_refundable_balance"]) != "-0.0"
    assert str(store.get("CUST-01337")["refundable_balance"]) != "-0.0"








# --------------------------------------------------------------------------- #
# STDIO isolation
# --------------------------------------------------------------------------- #
def test_every_stdout_line_is_valid_jsonrpc(server):
    """Drive a full session, then assert stdout contains nothing but frames."""
    server.request("tools/list")
    call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    call(server, "get_customer_record", {"customer_id": "NOPE"})
    call(server, "trigger_refund", {"customer_id": "CUST-00042", "amount": 1.0, "reason": "Goodwill credit issued"})
    call(server, "get_customer_record", {"customer_id": "CUST-99999"})

    assert server.stdout_lines, "server wrote nothing to stdout"
    for line in server.stdout_lines:
        message = json.loads(line)  # raises if a log line ever slipped onto stdout
        assert message.get("jsonrpc") == "2.0"
        assert "result" in message or "error" in message or "method" in message


def test_debug_logging_goes_to_stderr(server):
    server.request("tools/list")
    call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    stderr = "".join(server.stderr_text)
    assert "starting on stdio" in stderr, "expected startup log on stderr"
    assert "tools/call" in stderr


def test_stray_print_in_a_handler_cannot_corrupt_stdout(tmp_path):
    """The strongest form of the isolation claim.

    A variant server whose tool handler fires six different stdout attacks -
    including two that bypass the ``sys.stdout`` swap entirely (``sys.__stdout__``
    and a raw ``libc write(2)`` through ctypes) and one that bypasses Python
    altogether (a subprocess inheriting fd 1) - still yields a stdout stream
    that parses cleanly, because fd 1 itself has been pointed at stderr.
    """
    noisy = tmp_path / "noisy_server.py"
    noisy.write_text(textwrap.dedent(_NOISY_SERVER))
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    with StdioServerProcess([sys.executable, str(noisy)], cwd=REPO_ROOT, env=env) as proc:
        proc.initialize()
        response = call(proc, "get_customer_record", {"customer_id": "CUST-00042"})
        assert response["result"]["structuredContent"]["name"] == "Ada Lovelace"
        for line in proc.stdout_lines:
            json.loads(line)
        stderr = "".join(proc.stderr_text)
        for noise in (
            "chatty library banner",
            "raw fd 1 write from a child",
            "another stray line",
            "via sys dunder stdout",
            "from a subprocess",
            "raw libc write",
        ):
            assert noise in stderr, f"{noise!r} did not land on stderr"


def test_batched_stdin_gets_every_response(tmp_path):
    """Feed a file of requests and read the answers - the scripting path.

    Every line arrives at once and stdin hits EOF immediately, so without a
    drain grace the run loop tears down while the last handler is still in
    flight and its response never reaches the wire.
    """
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2026-07-28", "capabilities": {},
                    "clientInfo": {"name": "cli", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00042"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "trigger_refund",
                    "arguments": {"customer_id": "CUST-00042", "amount": 5.0,
                                  "reason": "Duplicate charge on invoice 88213"}}},
    ]
    infile = tmp_path / "requests.jsonl"
    infile.write_text("\n".join(json.dumps(r) for r in requests) + "\n")

    completed = subprocess.run(
        [sys.executable, "-m", "task1_mcp_server"],
        stdin=open(infile),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT),
        timeout=30,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in responses] == [1, 2, 3, 4], (
        f"expected a response per request, got {[r.get('id') for r in responses]}"
    )
    assert responses[3]["result"]["structuredContent"]["status"] == "accepted"


def find_stdout_hygiene_violations(package_dir, allowed_scopes=("claim_stdio_wire", "StdoutToStderr")):
    """The shipped hygiene check, as a function both tests call.

    It used to be inlined in one test and *re-implemented* in the meta-test
    written to prove it worked - so the meta-test validated a copy.

    It also only recognised two of the six realistic ways to reach the wire.
    ``from sys import stdout``, an aliased ``print``, ``sys.__stdout__``,
    ``os.write(1, ...)`` and ``getattr(sys, "stdout")`` all passed - including
    two forms the project's own sabotage test uses as attack vectors. Each is
    now a rule.
    """
    offenders = []
    for filename in sorted(os.listdir(package_dir)):
        if not filename.endswith(".py"):
            continue
        source = open(os.path.join(package_dir, filename)).read()
        tree = ast.parse(source, filename)

        sanctioned_lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in allowed_scopes
            ):
                sanctioned_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

        # Names bound to something that writes to the real stdout.
        aliased: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "sys":
                for alias in node.names:
                    if alias.name in ("stdout", "__stdout__"):
                        aliased.add(alias.asname or alias.name)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "print":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliased.add(target.id)
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "print"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliased.add(target.id)

        def flag(node, what):
            if node.lineno not in sanctioned_lines:
                offenders.append(f"{filename}:{node.lineno} {what}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and (func.id == "print" or func.id in aliased):
                    flag(node, f"calls {func.id}()")
                if isinstance(func, ast.Attribute) and func.attr == "print":
                    flag(node, "calls a print attribute")
                # os.write(1, ...) - straight at the descriptor.
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "write"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 1
                ):
                    flag(node, "writes to fd 1 directly")
                # getattr(sys, "stdout")
                if (
                    isinstance(func, ast.Name)
                    and func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "sys"
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in ("stdout", "__stdout__")
                ):
                    flag(node, "reaches stdout via getattr")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("stdout", "__stdout__")
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                flag(node, f"touches sys.{node.attr} outside the wire guard")
            if isinstance(node, ast.Name) and node.id in aliased and isinstance(node.ctx, ast.Load):
                flag(node, f"uses {node.id}, which is bound to stdout")
    return offenders


def test_no_module_writes_to_stdout_except_the_wire_guard():
    """Static guard against the regression the runtime tests protect against."""
    violations = find_stdout_hygiene_violations(os.path.join(REPO_ROOT, "task1_mcp_server"))
    assert not violations, f"stdout hygiene violations: {violations}"


def test_the_hygiene_check_actually_fails_on_a_violation(tmp_path):
    """Meta-test: prove the SHIPPED guard can fail, by calling it.

    An earlier version re-implemented the detection inline, so it proved
    nothing about the check that actually runs. A check that cannot fail is
    worse than no check, because it is believed - and a meta-test that checks a
    copy of the check is the same mistake one level up.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "clean.py").write_text("VALUE = 1\n")
    forms = {
        "a_attr.py": "import sys\nsys.stdout.write('leak')\n",
        "b_print.py": "print('leak')\n",
        "c_fromimport.py": "from sys import stdout\nstdout.write('leak')\n",
        "d_aliased.py": "import builtins\n_p = builtins.print\n_p('leak')\n",
        "e_dunder.py": "import sys\nsys.__stdout__.write('leak')\n",
        "f_fd.py": "import os\nos.write(1, b'leak')\n",
        "g_getattr.py": "import sys\ngetattr(sys, 'stdout').write('leak')\n",
    }
    for name, body in forms.items():
        (package / name).write_text(body)

    violations = find_stdout_hygiene_violations(str(package))
    for name in forms:
        assert any(name in v for v in violations), f"{name} was not detected"
    assert not any("clean.py" in v for v in violations)


def test_the_hygiene_check_is_not_defeated_by_widening_the_allowlist(tmp_path):
    """Widening ``allowed_scopes`` by one word used to let a genuine leak inside
    the money-moving method through both the guard and its meta-test."""
    package = tmp_path / "pkg2"
    package.mkdir()
    (package / "store.py").write_text(
        "import sys\n\n\nclass S:\n    def refund(self):\n        sys.stdout.write('LEAKED')\n"
    )
    widened = find_stdout_hygiene_violations(str(package), allowed_scopes=("refund",))
    assert not widened, "precondition: a widened allowlist hides it"
    # The shipped allowlist must not hide it.
    assert find_stdout_hygiene_violations(str(package)), "the default allowlist hid a real leak"


# --------------------------------------------------------------------------- #
# Round-two findings
# --------------------------------------------------------------------------- #
def run_with_env(lines, env_overrides=None, timeout=60):
    """Pipe raw lines through the server and return (responses, stderr)."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, **(env_overrides or {}))
    completed = subprocess.run(
        [sys.executable, "-m", "task1_mcp_server"],
        input="\n".join(lines) + "\n",
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, timeout=timeout,
    )
    return (
        [json.loads(line) for line in completed.stdout.splitlines() if line.strip()],
        completed.stderr,
    )


HANDSHAKE = [
    json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2026-07-28", "capabilities": {},
                           "clientInfo": {"name": "cli", "version": "1"}}}),
    json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
]


@pytest.mark.parametrize("grace", ["0", "250", "abc", "1e400", "nan", "-5", "5000"])
def test_every_refund_that_executes_returns_a_receipt(grace):
    """The worst of the round-two findings, on the tool that moves money.

    The drain was a fixed post-EOF sleep, so ``MCP_DRAIN_GRACE_MS=0`` executed
    200 refunds and returned 112 receipts: 88 moved money and told the client
    nothing, whose only recourse is to retry and refund twice. Malformed values
    were worse - ``abc`` crashed at startup and ``1e400`` hung forever. The
    drain now waits on the set of outstanding requests, and the setting is a
    timeout with a floor rather than an on/off switch.
    """
    refunds = [
        json.dumps({"jsonrpc": "2.0", "id": index, "method": "tools/call",
                    "params": {"name": "trigger_refund", "arguments": {
                        "customer_id": "CUST-00042", "amount": 0.01,
                        "reason": "Duplicate charge on invoice"}}})
        for index in range(2, 102)
    ]
    responses, stderr = run_with_env(HANDSHAKE + refunds, {"MCP_DRAIN_GRACE_MS": grace})
    executed = stderr.count("refund RFND")
    answered = sum(1 for r in responses if r.get("id") != 1)
    assert executed > 0, "the scenario did not actually issue refunds"
    assert answered == executed, (
        f"{executed} refunds executed but only {answered} receipts returned"
    )


@pytest.mark.parametrize(
    "lines, expect_code",
    [
        ([json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})], -32602),   # pre-init
        (HANDSHAKE + [json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize",
                                  "params": {"protocolVersion": 5}})], -32602),
        (HANDSHAKE + [json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/list",
                                  "params": {"cursor": 123}})], -32602),
        (HANDSHAKE + [json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                  "params": {"name": "admin_drop", "arguments": {}}})], -32602),
        (HANDSHAKE + [json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                  "params": {"name": "get_customer_record",
                                             "arguments": {"customer_id": "BAD"}}})], -32602),
        (HANDSHAKE + ['{"jsonrpc":"2.0","id":9,"method":"tools/call","params":[1,2]}'], -32600),
        (HANDSHAKE + ["this is not json"], -32700),
    ],
)
def test_every_rejection_carries_the_right_code_and_one_shape(lines, expect_code):
    """One error code must mean one response shape - and so must all of them.

    The SDK surface-validates spec methods before dispatch and its own
    rejections carried ``data`` as a bare string (``""`` for invalid params,
    the method name for method-not-found), while the schema layer carried
    ``data.errors``. An agent doing ``error["data"]["errors"]`` got a
    TypeError on half of them, so the shape is normalised on the way out.

    Previously this assertion was gated behind ``if "tools/call" in ...``,
    which meant it never ran on the paths that had the wrong shape.
    """
    responses, _ = run_with_env(lines)
    errors = [r for r in responses if "error" in r]
    assert errors, "no error response at all"
    error = errors[-1]["error"]
    assert error["code"] == expect_code
    assert isinstance(error.get("data"), dict), f"data was {error.get('data')!r}"
    assert "errors" in error["data"], "no structured per-field detail"
    assert isinstance(error["data"]["errors"], list) and error["data"]["errors"]


def test_method_not_found_also_carries_the_structured_shape():
    responses, _ = run_with_env(
        HANDSHAKE + [json.dumps({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})]
    )
    error = next(r["error"] for r in responses if "error" in r)
    assert error["code"] == METHOD_NOT_FOUND
    assert isinstance(error["data"], dict), f"data was {error['data']!r}"
    assert error["data"]["errors"][0]["detail"] == "resources/list"


def test_a_null_id_request_is_answered():
    """MCP forbids a null request id, so it is an invalid request - but it was
    treated as a notification and answered with nothing, hanging the client."""
    responses, _ = run_with_env(
        HANDSHAKE + ['{"jsonrpc":"2.0","id":null,"method":"tools/list"}']
    )
    assert any(r.get("error", {}).get("code") == -32600 for r in responses)


def test_blank_lines_do_not_produce_spurious_parse_errors():
    """A trailing newline in a requests file yielded an unsolicited -32700."""
    responses, _ = run_with_env(HANDSHAKE + ["", "   ", "\t", json.dumps(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/list"})])
    codes = [r["error"]["code"] for r in responses if "error" in r]
    assert -32700 not in codes, f"blank framing lines produced parse errors: {codes}"
    assert any(r.get("id") == 5 and "result" in r for r in responses)








@pytest.mark.parametrize(
    "reason",
    ["Duplicate charge on invoice 88213", "重複請求のための返金処理",
     "refund \U0001f600 duplicate charge", "abcdefghij"],
)
def test_legitimate_reasons_are_still_accepted(server, reason):
    """The tightening must not start rejecting real audit text."""
    response = call(server, "trigger_refund", {
        "customer_id": "CUST-00042", "amount": 1.0, "reason": reason})
    assert "error" not in response, f"{reason!r} was wrongly rejected"




# --------------------------------------------------------------------------- #
# Round-three findings
# --------------------------------------------------------------------------- #
async def test_the_drain_counts_requests_not_distinct_ids():
    """A ``set`` collapsed duplicate ids, so the first answer released the drain.

    JSON-RPC says a client must not reuse an outstanding id, but this server
    accepts and *executes* those requests - it takes the money either way - and
    then lost 45 of 100 receipts. Exactly the failure the drain was written to
    eliminate, reintroduced one layer down by the fix's own data structure.
    """
    tracker = DrainTracker()
    for _ in range(3):
        tracker.dispatched(7)
    tracker.answered(7)
    assert not await tracker.wait_idle(0.1), "the drain released with two requests outstanding"
    tracker.answered(7)
    tracker.answered(7)
    assert await tracker.wait_idle(0.1)


def test_every_refund_returns_a_receipt_even_when_ids_repeat():
    refunds = [
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "trigger_refund", "arguments": {
                        "customer_id": "CUST-00042", "amount": 0.01,
                        "reason": "Duplicate charge on invoice"}}})
        for _ in range(60)
    ]
    responses, stderr = run_with_env(HANDSHAKE + refunds)
    executed = stderr.count("refund RFND")
    answered = sum(1 for r in responses if r.get("id") == 7)
    assert executed > 0
    assert answered == executed, f"{executed} refunds executed, {answered} receipts returned"


async def test_a_malformed_frame_does_not_cancel_an_unrelated_request():
    """``TrackingWriteStream`` credited EVERY outgoing frame carrying an id,
    including pump-written errors for ids never dispatched - so a malformed
    frame reusing an in-flight id released the drain early.

    Driven through the stream, not the tracker: asserting on ``DrainTracker``
    alone left the crediting rule - which lives in the stream - untested, and
    the mutant that credits every frame passed.
    """
    import mcp_types as types_module

    from task1_mcp_server.server import TrackingWriteStream

    class Sink:
        async def send(self, item):
            return None

    tracker = DrainTracker()
    tracker.dispatched(42)          # a real refund is in flight
    stream = TrackingWriteStream(Sink(), tracker)

    # A pump-written error frame reusing that id, for a request the server was
    # never asked to handle. It must not count as the answer.
    await stream.untracked_send(SessionMessage(types_module.JSONRPCError(
        jsonrpc="2.0", id=42,
        error=types_module.ErrorData(code=types_module.INVALID_REQUEST, message="Invalid Request"),
    )))
    assert tracker.is_outstanding(42), "a pump-written frame was credited as the answer"
    assert not await tracker.wait_idle(0.1), "the drain released with a request outstanding"


async def test_pump_generated_frames_do_not_credit_the_drain():
    """The mechanism, tested where it lives.

    The end-to-end version below is a useful guard but cannot fail while the
    handlers are fast enough to finish inside the drain floor - so it passes
    even when the pump credits the drain. This one asserts the routing rule
    itself: two writers, two methods.
    """
    from task1_mcp_server.server import TrackingWriteStream, _pump_send
    import mcp_types as types_module

    class Sink:
        async def send(self, item):
            return None

    tracker = DrainTracker()
    tracker.dispatched(42)
    stream = TrackingWriteStream(Sink(), tracker)
    frame = SessionMessage(types_module.JSONRPCError(
        jsonrpc="2.0", id=42,
        error=types_module.ErrorData(code=types_module.INVALID_REQUEST, message="Invalid Request"),
    ))

    await _pump_send(stream, frame)
    assert tracker.is_outstanding(42), "a pump-generated frame consumed the drain slot"

    # The server's own answer does credit it.
    await stream.send(frame)
    assert not tracker.is_outstanding(42)


def test_malformed_frames_do_not_consume_another_request_s_drain_slot():
    """The same property end to end, through the real pump.

    Interleaving refunds and malformed frames that reuse the same id: if the
    pump's own error frames credit the drain, they answer the refunds' slots
    and the drain releases while money-moving work is still in flight.
    """
    lines = []
    for _ in range(40):
        lines.append(json.dumps({
            "jsonrpc": "2.0", "id": 42, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00042", "amount": 0.01,
                "reason": "Duplicate charge on invoice"}}}))
        # Malformed: screened by the pump, answered without reaching the server.
        lines.append('{"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":5}}')

    responses, stderr = run_with_env(HANDSHAKE + lines)
    executed = stderr.count("refund RFND")
    receipts = sum(
        1 for r in responses
        if r.get("id") == 42 and "result" in r and not r["result"].get("isError")
    )
    assert executed > 0
    assert receipts == executed, (
        f"{executed} refunds executed but {receipts} receipts returned - "
        "pump-generated frames consumed the drain slots"
    )


def test_a_deliberate_non_dict_error_payload_is_not_rewritten():
    """The transport rewrite was a blanket ``not isinstance(data, dict)``
    transform: a deliberate ``-32602`` whose ``data`` is an array - legal per
    JSON-RPC, "a Primitive or Structured value" - had both field hints
    destroyed and a wrong ``field: params`` stamped on it."""
    import mcp_types as types_module

    from task1_mcp_server.server import DrainTracker as _DT
    from task1_mcp_server.server import TrackingWriteStream

    sent = []

    class Recorder:
        async def send(self, item):
            sent.append(item)

    hints = [{"field": "amount", "hint": "must be <= remaining balance"},
             {"field": "reason", "hint": "cite the ticket id"}]
    stream = TrackingWriteStream(Recorder(), _DT())
    frame = SessionMessage(types_module.JSONRPCError(
        jsonrpc="2.0", id=1,
        error=types_module.ErrorData(code=types_module.INVALID_PARAMS, message="Invalid params", data=hints),
    ))
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(stream.send(frame))
    assert sent[0].message.error.data == hints, "a deliberate structured payload was rewritten"


@pytest.mark.parametrize("bad_id", ["9.5", '{"a":1}', "true", "[1]", "null"])
def test_an_unusable_id_error_also_carries_structured_detail(bad_id):
    """The round-two claim was that -32700, -32600 and -32602 ALL carry
    ``data.errors``. This path built its ErrorData with no ``data`` at all, and
    the transport rewrite only normalised -32602."""
    responses, _ = run_with_env(HANDSHAKE + ['{"jsonrpc":"2.0","id":%s,"method":"tools/list"}' % bad_id])
    errors = [r for r in responses if "error" in r]
    assert errors, f"id {bad_id} produced no response"
    error = errors[-1]["error"]
    assert isinstance(error.get("data"), dict), f"data was {error.get('data')!r}"
    assert "errors" in error["data"]




@pytest.mark.parametrize("reason", ["Bad charge", "Dup charge", "a b c d e f", "refund now"])
def test_the_brief_s_ten_character_minimum_is_honoured(server, reason):
    """The brief says "reason string with minimum length of 10". Requiring ten
    *visible* characters was stricter than specified and rejected exactly the
    reasons a caller would write, while the published schema advertised
    ``minLength: 10`` and accepted them."""
    assert len(reason) >= 10
    response = call(server, "trigger_refund", {
        "customer_id": "CUST-00042", "amount": 1.0, "reason": reason})
    assert "error" not in response, f"{reason!r} was rejected despite meeting the stated minimum"




def test_the_stdout_buffer_can_be_wrapped_without_killing_the_process():
    """``StdoutToStderr.buffer`` handed out the live ``sys.stderr.buffer``, so
    the standard force-UTF-8 idiom closed stderr and aborted the interpreter."""
    completed = subprocess.run(
        [sys.executable, "-c",
         "import sys, io, gc\n"
         "sys.path.insert(0, %r)\n" % REPO_ROOT +
         "from task1_mcp_server.wire import StdoutToStderr\n"
         "g = StdoutToStderr(sys.stderr)\n"
         "w = io.TextIOWrapper(g.buffer, encoding='utf-8')\n"
         "del w\n"
         "gc.collect()\n"
         "g.reconfigure(encoding='utf-8')\n"
         "assert g.writable() and not g.readable()\n"
         "sys.stderr.write('alive\\n')\n"],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "alive" in completed.stderr
    assert "lost sys.stderr" not in completed.stderr


def test_a_closed_descriptor_exits_cleanly_without_a_traceback():
    completed = subprocess.run(
        [sys.executable, "-c",
         "import os, runpy\nos.close(1)\nrunpy.run_module('task1_mcp_server', run_name='__main__')"],
        capture_output=True, env=dict(os.environ, PYTHONPATH=REPO_ROOT), timeout=30,
    )
    assert completed.returncode == 2
    assert b"Traceback" not in completed.stderr
    assert b"not open" in completed.stderr


def test_the_drain_timeout_is_clamped():
    """``1e400`` became ``inf`` and the server never exited. The clamp that
    stops that was unpinned - the test that names the hazard passed because the
    drain emptied first."""
    from task1_mcp_server.server import _MAX_DRAIN_MS, _drain_grace_ms

    for value in ("1e400", "inf", "999999999"):
        os.environ["MCP_DRAIN_GRACE_MS"] = value
        try:
            assert _drain_grace_ms() <= _MAX_DRAIN_MS, f"{value} was not clamped"
        finally:
            del os.environ["MCP_DRAIN_GRACE_MS"]


def test_the_sys_stdout_guard_is_installed():
    """One of the three advertised isolation layers could be deleted outright
    with the suite still green."""
    completed = subprocess.run(
        [sys.executable, "-c",
         "import sys\nsys.path.insert(0, %r)\n" % REPO_ROOT +
         "from task1_mcp_server.wire import claim_stdio_wire, StdoutToStderr\n"
         "with claim_stdio_wire() as (_i, _o):\n"
         "    kind = type(sys.stdout).__name__\n"
         "sys.stderr.write(kind + '\\n')\n"],
        capture_output=True, text=True, timeout=30,
    )
    assert "StdoutToStderr" in completed.stderr, (
        f"sys.stdout was not replaced inside the claim: {completed.stderr!r}"
    )


def test_the_line_fifo_is_a_queue_not_a_slot():
    """A "last line" slot races: the reader can advance before the pump reads
    it. The docstring explains exactly that, and nothing tested it."""
    from task1_mcp_server.server import LineRecordingInput

    recorder = LineRecordingInput(None)
    recorder.lines.extend(["first\n", "second\n", "third\n"])
    assert recorder.take() == "first\n", "lines are not consumed in arrival order"
    assert recorder.take() == "second\n"
    assert recorder.take() == "third\n"
    assert recorder.take() == "", "an exhausted queue must not reuse the last line"


def test_published_descriptions_carry_no_markup(server):
    """Every description in a tool schema is read by a model, not by Sphinx.

    The model docstrings are published verbatim, so RST markup written for a
    human reader ("``get_customer_record``") arrived in the model's context as
    literal backticks.
    """
    response = server.request("tools/list")
    described = []
    for tool in response["result"]["tools"]:
        described.append(tool.get("description", ""))
        schema = tool["inputSchema"]
        described.append(schema.get("description", ""))
        for prop in schema.get("properties", {}).values():
            described.append(prop.get("description", ""))

    for text in described:
        assert "``" not in text, f"RST markup published to the model: {text!r}"
        assert ":param" not in text and ":return" not in text, f"docstring markup: {text!r}"


def test_tools_publish_an_output_schema(server):
    """MCP: a server returning ``structuredContent`` SHOULD publish an
    ``outputSchema`` describing it, or a client is handed structured data it
    has no way to validate."""
    tools = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}
    for name, tool in tools.items():
        assert "outputSchema" in tool, f"{name} returns structuredContent with no outputSchema"
        assert tool["outputSchema"]["type"] == "object"
        assert tool["outputSchema"]["required"]


def test_success_results_validate_against_the_published_output_schema(server):
    """The published schema has to describe what the tool actually returns."""
    jsonschema = pytest.importorskip("jsonschema")
    tools = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}

    record = call(server, "get_customer_record", {"customer_id": "CUST-00042"})
    jsonschema.validate(
        record["result"]["structuredContent"], tools["get_customer_record"]["outputSchema"]
    )

    receipt = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 10.0, "reason": "Duplicate charge seen"},
    )
    jsonschema.validate(
        receipt["result"]["structuredContent"], tools["trigger_refund"]["outputSchema"]
    )


def test_error_results_carry_no_structured_content(server):
    """An ``outputSchema`` describes the SUCCESS shape. Routing an error payload
    through it would hand a client structured data that fails validation against
    the very schema the tool published for it."""
    for arguments in (
        {"customer_id": "CUST-99999"},
    ):
        response = call(server, "get_customer_record", arguments)
        assert response["result"]["isError"] is True
        assert "structuredContent" not in response["result"]

    over_balance = call(
        server,
        "trigger_refund",
        {"customer_id": "CUST-00007", "amount": 250.0, "reason": "Duplicate charge seen"},
    )
    assert over_balance["result"]["isError"] is True
    assert "structuredContent" not in over_balance["result"]


@pytest.mark.parametrize(
    "reason, label",
    [
        (" " * 10, "ten spaces"),
        ("\n" * 10, "ten newlines"),
        ("\t" * 12, "twelve tabs"),
        ("\x00" * 10, "ten NUL bytes"),
        ("​" * 12, "twelve zero-width spaces"),
    ],
)
def test_a_reason_with_nothing_readable_in_it_is_rejected(server, reason, label):
    """``min_length`` counts code points, so each of these satisfied it - and
    each was accepted, moved money, and echoed the whitespace back in the
    receipt. The field exists for an audit trail; none of these is one."""
    response = call(
        server, "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 10.0, "reason": reason},
    )
    assert "result" not in response, f"{label} was accepted as a reason: {response}"
    assert response["error"]["code"] == INVALID_PARAMS
    assert "reason" in {e["field"] for e in response["error"]["data"]["errors"]}


def test_an_unbounded_reason_is_rejected(server):
    """The reason is echoed back in the receipt, so an unbounded one is
    amplification: 1,000,000 characters in, 1,000,000 characters out."""
    response = call(
        server, "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 10.0, "reason": "x" * 1_000_000},
    )
    assert "result" not in response
    assert response["error"]["code"] == INVALID_PARAMS


def test_a_rejected_enormous_amount_does_not_echo_309_digits(server):
    """``:.2f`` on 1e308 writes out every digit, so the refusal message ran to
    437 characters. Input echoes are truncated everywhere else."""
    response = call(
        server, "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 1e308, "reason": "Duplicate charge seen"},
    )
    text = response["result"]["content"][0]["text"]
    assert response["result"]["isError"] is True
    assert len(text) < 200, f"refusal message was {len(text)} characters"
    assert "000000000" not in text


def test_the_drain_grace_env_var_is_a_timeout_not_a_switch():
    """The docstring claimed ``MCP_DRAIN_GRACE_MS=0`` disabled the drain. It
    never did - 0 is clamped up to the floor, because a drain of zero loses
    exactly the replies the drain exists to keep."""
    from task1_mcp_server.server import _MIN_DRAIN_MS, _drain_grace_ms

    for value, expected in [("0", _MIN_DRAIN_MS), ("-5", _MIN_DRAIN_MS), ("100", _MIN_DRAIN_MS)]:
        with mock.patch.dict(os.environ, {"MCP_DRAIN_GRACE_MS": value}):
            assert _drain_grace_ms() == expected
    with mock.patch.dict(os.environ, {"MCP_DRAIN_GRACE_MS": "900"}):
        assert _drain_grace_ms() == 900.0

    source = (pathlib.Path(__file__).parent.parent / "task1_mcp_server" / "server.py").read_text()
    assert "MCP_DRAIN_GRACE_MS=0`` to disable" not in source, "the stale claim is back"


def test_a_broken_pipe_exits_quietly_without_a_traceback(tmp_path):
    """anyio wraps transport failures in a ``BaseExceptionGroup``, which is not
    an ``OSError`` - so a client closing the pipe mid-session escaped as a
    46-line traceback carrying absolute paths, and exited 1."""
    requests = tmp_path / "bulk.jsonl"
    requests.write_text(
        "\n".join(HANDSHAKE + [
            json.dumps({"jsonrpc": "2.0", "id": n, "method": "tools/list"}) for n in range(200)
        ]) + "\n"
    )
    completed = subprocess.run(
        f"{sys.executable} -m task1_mcp_server < {requests} | head -2",
        shell=True, capture_output=True, text=True, timeout=60,
        env=dict(os.environ, PYTHONPATH=REPO_ROOT),
    )
    assert "Traceback" not in completed.stderr, completed.stderr[:800]
    assert "BaseExceptionGroup" not in completed.stderr
    assert "unhandled errors in a TaskGroup" not in completed.stderr


def test_a_float_id_is_refused_because_mcp_narrows_the_json_rpc_id():
    """Task 1 and Task 2 differ here on purpose, and the reason is the spec.

    MCP restricts a request id to a string or an integer, so a float id is
    invalid at this layer and there is no conformant id to echo back. Task 2
    proxies plain JSON-RPC 2.0, where any number is a legal id, and echoes it.
    The frame is still ANSWERED - the failure mode that matters is a client
    left waiting forever.
    """
    responses, _ = run_with_env(
        HANDSHAKE + ['{"jsonrpc":"2.0","id":1.5,"method":"tools/list"}']
    )
    errors = [r for r in responses if "error" in r]
    assert errors, "a float id produced no response at all"
    assert errors[-1]["error"]["code"] == -32600
    assert errors[-1]["id"] is None
    assert isinstance(errors[-1]["error"]["data"], dict)


@pytest.mark.parametrize(
    "reason, accepted, label",
    [
        ("Dup charge", True, "ten characters including a space"),
        ("a b c d e f", True, "spaces count toward the length"),
        ("Duplicate charge seen", True, "ordinary reason"),
        ("a" + "\x00" * 9, False, "one visible character padded with NULs"),
        ("\x00" * 10, False, "ten NUL bytes"),
        (" " * 10, False, "ten spaces"),
        ("too short", False, "nine characters"),
    ],
)
def test_the_reason_rule_counts_printable_characters(server, reason, accepted, label):
    """Non-printable characters do not count toward the minimum, and something
    must be visible - but spaces do count, because the brief asks for a length
    of ten and "Dup charge" is ten."""
    response = call(
        server, "trigger_refund",
        {"customer_id": "CUST-00042", "amount": 1.0, "reason": reason},
    )
    if accepted:
        assert "error" not in response, f"{label} was rejected: {response}"
        assert response["result"]["isError"] is False
    else:
        assert "result" not in response, f"{label} was accepted: {response}"
        assert response["error"]["code"] == INVALID_PARAMS


def test_ctrl_c_stops_the_server_instead_of_being_ignored(tmp_path):
    """The SDK reads stdin on a worker thread parked in ``readline``. A
    KeyboardInterrupt goes to the main thread, so the loop waited on a thread
    blocked in a read that never returned - the process was still alive after
    10 seconds. It exits promptly if the signal lands during startup, which is
    why the hang looked intermittent.
    """
    import signal as _signal

    process = subprocess.Popen(
        [sys.executable, "-m", "task1_mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=dict(os.environ, PYTHONPATH=REPO_ROOT),
    )
    for line in HANDSHAKE:
        process.stdin.write(line + "\n")
        process.stdin.flush()
    time.sleep(1.0)  # let the reader thread settle into its blocking read
    process.send_signal(_signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise AssertionError("SIGINT was ignored; the process had to be killed")
    assert "Traceback" not in process.stderr.read()


def test_sigterm_still_exits_cleanly(tmp_path):
    """Regression: an earlier attempt at the SIGINT fix installed a handler for
    SIGTERM too and broke the one signal that already worked."""

    process = subprocess.Popen(
        [sys.executable, "-m", "task1_mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=dict(os.environ, PYTHONPATH=REPO_ROOT),
    )
    for line in HANDSHAKE:
        process.stdin.write(line + "\n")
        process.stdin.flush()
    time.sleep(1.0)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise AssertionError("SIGTERM was ignored")
    assert "Traceback" not in process.stderr.read()


def test_the_published_reason_pattern_matches_what_the_server_enforces(server):
    """A client validating against the tool's own schema must get the same
    answer as the server. Advertising only ``minLength: 10`` while enforcing a
    printable-character count meant "Dup\\tcharge" passed locally and was
    refused on the wire."""
    import re as _re

    schema = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}
    pattern = schema["trigger_refund"]["inputSchema"]["properties"]["reason"].get("pattern")
    assert pattern, "the reason rule is not published at all"
    compiled = _re.compile(pattern)

    for reason in ("Duplicate charge seen", "Dup charge", "Dup\tcharge", "a" + "\x00" * 9, " " * 10):
        schema_accepts = compiled.match(reason) is not None
        response = call(
            server, "trigger_refund",
            {"customer_id": "CUST-00042", "amount": 1.0, "reason": reason},
        )
        server_accepts = "error" not in response
        assert schema_accepts == server_accepts, (
            f"{reason!r}: schema says {schema_accepts}, server says {server_accepts}"
        )


@pytest.mark.parametrize("amount", [19.185, 46.505, 0.125, 3.755, 12.345])
def test_the_receipt_amount_is_what_actually_left_the_balance(amount):
    """One quantisation, both published numbers derived from it.

    The receipt rounded the amount while the balance rounded the difference,
    independently - so a refund of 19.185 returned a receipt saying 19.18 while
    19.19 actually left the balance. On the one tool that moves money, the
    receipt handed back has to be what the ledger did.
    """
    from task1_mcp_server.store import CustomerStore

    store = CustomerStore()
    before = store.get("CUST-00042")["refundable_balance"]
    receipt = store.refund("CUST-00042", amount, "Duplicate charge")
    after = store.get("CUST-00042")["refundable_balance"]

    moved = round(before - after, 2)
    assert receipt["amount"] == moved, (
        f"receipt says {receipt['amount']} but {moved} left the balance"
    )
    assert receipt["remaining_refundable_balance"] == after


def test_receipt_and_ledger_agree_across_a_sweep():
    """Half-cent amounts are where independent rounding diverges, in both
    directions - 1,564 of 200,000 amounts disagreed.

    Each iteration seeds its own customer: ``CustomerStore()`` shares the module
    seed by reference, so a sweep against the default data drains one balance.
    """
    from task1_mcp_server.store import Customer, CustomerStore, RefundTooSmallError

    mismatches = []
    for step in range(4000):
        amount = round(0.005 + step * 0.0025, 4)
        customer = Customer(
            customer_id="CUST-00042", name="T", email="t@example.com",
            plan="pro", status="active", refundable_balance=1000.0,
        )
        store = CustomerStore(customers=[customer])
        before = store.get("CUST-00042")["refundable_balance"]
        try:
            receipt = store.refund("CUST-00042", amount, "Duplicate charge")
        except RefundTooSmallError:
            # Rounds to zero cents and is refused by design; nothing to compare.
            continue
        after = store.get("CUST-00042")["refundable_balance"]
        if receipt["amount"] != round(before - after, 2):
            mismatches.append(amount)
    assert not mismatches, f"{len(mismatches)} amounts disagree, e.g. {mismatches[:5]}"

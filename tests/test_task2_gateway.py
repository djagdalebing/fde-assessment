"""Task 2: authentication, method-level policy, and proxy behaviour.

The gateway is mounted over ASGI and pointed at the mock downstream, also
mounted over ASGI, so a whole request/response round trip runs in-process with
no ports involved. The mock records every call it receives, which is what lets
these tests assert the far stronger claim: a blocked ``admin_*`` call is not
merely answered with an error, it never reaches the downstream at all.
"""

from __future__ import annotations

import os
import sys

import json
import json as _json
import time
import socket

import httpx
import pytest
from fastapi import FastAPI as _FastAPI
from fastapi import Request as _Request
from fastapi.responses import Response as _Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task2_mcp_gateway import mock_downstream  # noqa: E402
from task2_mcp_gateway.auth import AuthError, TokenVerifier  # noqa: E402
from task2_mcp_gateway.gateway import (  # noqa: E402
    _ascii_header,
    create_app,
    request_reference,
)
from task2_mcp_gateway.policy import Policy  # noqa: E402
from tests.live_server import live_server  # noqa: E402

ADMIN = "Bearer admin-token-abc123"
VIEWER = "Bearer viewer-token-def456"
UNAUTHORIZED_TOOL_CALL = -32001


@pytest.fixture
def downstream_log():
    mock_downstream.CALL_LOG.clear()
    yield mock_downstream.CALL_LOG
    mock_downstream.CALL_LOG.clear()


@pytest.fixture
async def gateway():
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_downstream.app),
        base_url="http://downstream",
    )
    app = create_app(
        downstream_url="http://downstream/mcp",
        verifier=TokenVerifier(),
        policy=Policy(),
        client=downstream_client,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        yield client
    await downstream_client.aclose()


async def rpc(client, payload, token=VIEWER, request_id=1):
    headers = {"Authorization": token} if token else {}
    return await client.post("/mcp", json=payload, headers=headers)


def call_payload(name, arguments=None, request_id=1):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "header",
    [None, "", "admin-token-abc123", "Basic admin-token-abc123", "Bearer", "Bearer ", "Bearer wrong-token"],
)
async def test_bad_credentials_are_rejected(gateway, downstream_log, header):
    response = await rpc(gateway, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token=header)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["message"] == "Unauthorized"
    assert downstream_log == [], "an unauthenticated request reached the downstream server"


async def test_bearer_scheme_is_case_insensitive(gateway):
    response = await rpc(gateway, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, token="bearer admin-token-abc123")
    assert response.status_code == 200
    assert "result" in response.json()


# --------------------------------------------------------------------------- #
# tools/list is forwarded transparently
# --------------------------------------------------------------------------- #
async def test_tools_list_is_forwarded_transparently_for_viewer(gateway, downstream_log):
    response = await rpc(gateway, {"jsonrpc": "2.0", "id": 7, "method": "tools/list"}, token=VIEWER)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 7
    names = [t["name"] for t in body["result"]["tools"]]
    assert names == [t["name"] for t in mock_downstream.TOOLS]
    assert "admin_reset_key" in names
    assert [c["method"] for c in downstream_log] == ["tools/list"]


async def test_tools_list_forwarded_for_admin(gateway):
    response = await rpc(gateway, {"jsonrpc": "2.0", "id": 7, "method": "tools/list"}, token=ADMIN)
    assert len(response.json()["result"]["tools"]) == len(mock_downstream.TOOLS)


# --------------------------------------------------------------------------- #
# tools/call authorization - the core requirement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_tenant", "admin_"])
async def test_viewer_admin_tool_call_is_blocked_before_downstream(gateway, downstream_log, tool):
    response = await rpc(gateway, call_payload(tool, {"tenant_id": "acme"}), token=VIEWER)

    assert response.status_code == 200, "the refusal is a JSON-RPC error, not an HTTP error"
    body = response.json()
    assert body["id"] == 1
    assert "result" not in body
    assert body["error"]["code"] == UNAUTHORIZED_TOOL_CALL
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert body["error"]["data"]["required_role"] == "admin"
    assert body["error"]["data"]["actual_role"] == "viewer"

    assert downstream_log == [], "the blocked call still reached the downstream server"


@pytest.mark.parametrize("tool", ["admin_reset_key", "admin_delete_tenant"])
async def test_admin_admin_tool_call_is_forwarded(gateway, downstream_log, tool):
    response = await rpc(gateway, call_payload(tool, {"tenant_id": "acme"}), token=ADMIN)
    body = response.json()
    assert "error" not in body
    assert body["result"]["structuredContent"]["executed"] == tool
    assert [c["tool"] for c in downstream_log] == [tool]


@pytest.mark.parametrize("tool", ["get_customer_record", "search_orders"])
async def test_non_admin_tools_are_forwarded_for_viewer(gateway, downstream_log, tool):
    args = {"customer_id": "CUST-00042"} if tool == "get_customer_record" else {"query": "widget"}
    response = await rpc(gateway, call_payload(tool, args), token=VIEWER)
    assert "error" not in response.json()
    assert [c["tool"] for c in downstream_log] == [tool]


@pytest.mark.parametrize("tool", ["get_admin_reset_key", "x_admin_reset_key", "administrator_tools"])
async def test_names_that_merely_contain_admin_are_not_privileged(gateway, downstream_log, tool):
    """Canonicalisation must not turn the prefix rule into a substring search."""
    response = await rpc(gateway, call_payload(tool, {"tenant_id": "acme"}), token=VIEWER)
    assert response.json()["error"]["code"] == -32602, "expected the downstream's unknown-tool error"
    assert [c["tool"] for c in downstream_log] == [tool]


async def test_an_admin_reaches_an_admin_tool(gateway, downstream_log):
    """The rule must refuse a viewer without also blocking the admin it exists
    to let through."""
    response = await rpc(gateway, call_payload("admin_reset_key", {"tenant_id": "acme"}), token=ADMIN)
    assert "error" not in response.json()
    assert [c["tool"] for c in downstream_log] == ["admin_reset_key"]


# --------------------------------------------------------------------------- #
# Envelope handling
# --------------------------------------------------------------------------- #
async def test_malformed_json_is_parse_error(gateway):
    response = await gateway.post(
        "/mcp", content="{not json", headers={"Authorization": VIEWER, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "payload",
    [
        {"id": 1, "method": "tools/list"},                          # missing jsonrpc
        {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},         # wrong version
        {"jsonrpc": "2.0", "id": 1},                                 # missing method
        {"jsonrpc": "2.0", "id": 1, "method": 5},                    # non-string method
        "just a string",
        42,
    ],
)
async def test_invalid_envelope_is_invalid_request(gateway, downstream_log, payload):
    response = await rpc(gateway, payload, token=VIEWER)
    assert response.json()["error"]["code"] == -32600
    assert downstream_log == []


@pytest.mark.parametrize(
    "params",
    [None, [], {"arguments": {}}, {"name": ""}, {"name": 5}],
)
async def test_tools_call_without_a_valid_name_is_invalid_params(gateway, downstream_log, params):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    if params is not None:
        payload["params"] = params
    response = await rpc(gateway, payload, token=VIEWER)
    assert response.json()["error"]["code"] == -32602
    assert downstream_log == [], "a tools/call with no usable name reached the downstream"


# --------------------------------------------------------------------------- #
# Inputs that used to escape as a bare HTTP 500
# --------------------------------------------------------------------------- #
def test_non_ascii_authorization_is_an_auth_error_not_a_crash():
    """``hmac.compare_digest`` raises TypeError on non-ASCII ``str``.

    Only ``AuthError`` was caught, so this escaped as a plain-text HTTP 500
    with no JSON-RPC body - reachable with no valid credential at all, and
    trivially distinguishable from every other auth failure, which defeats the
    uniform-failure property the gateway claims.
    """
    verifier = TokenVerifier()
    for header in ["Bearer \u00e9\u00e8\u00ea", "Bearer tok\u00e9n", "B\u00e9arer x"]:
        with pytest.raises(AuthError):
            verifier.principal_from_header(header)


def test_non_ascii_authorization_over_a_real_socket_is_a_401():
    """httpx refuses to send a non-ASCII header, so this needs a raw socket -
    which is exactly how the attack arrives in the first place.
    """
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_downstream.app), base_url="http://downstream"
    )
    app = create_app(downstream_url="http://downstream/mcp", client=downstream_client)
    mock_downstream.CALL_LOG.clear()

    with live_server(app) as base_url:
        host, port = base_url.rsplit(":", 1)
        raw = (
            b"POST /mcp HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer \xc3\xa9\xc3\xa8\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n\r\n{}"
        )
        with socket.create_connection(("127.0.0.1", int(port)), timeout=10) as sock:
            sock.sendall(raw)
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
    response = b"".join(chunks).decode("utf-8", "replace")

    assert " 401 " in response.split("\r\n")[0], f"expected 401, got {response.splitlines()[0]!r}"
    assert "Internal Server Error" not in response
    assert "Unauthorized" in response
    assert mock_downstream.CALL_LOG == []


async def test_non_utf8_body_is_a_parse_error(gateway):
    response = await gateway.post(
        "/mcp",
        content=b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"\xff\xfe"}}',
        headers={"Authorization": VIEWER, "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


async def test_deeply_nested_json_is_a_parse_error(gateway):
    payload = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":'
        + b"[" * 60000 + b"]" * 60000 + b"}}"
    )
    response = await gateway.post(
        "/mcp", content=payload, headers={"Authorization": VIEWER, "Content-Type": "application/json"}
    )
    assert response.status_code in (400, 413)
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------- #
# Header handling
# --------------------------------------------------------------------------- #


async def test_upstream_error_body_is_not_relayed():
    """An upstream 5xx with a JSON body used to pass through verbatim - one
    test upstream leaked a Postgres DSN and password this way."""
    async def leaky_error(request):
        return httpx.Response(500, json={
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32603, "message":
                      "psycopg2 OperationalError: host=db-primary.internal port=5432 "
                      "user=mcp_svc password=hunter2"},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(leaky_error))
    app = create_app(downstream_url="http://downstream/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers={"Authorization": ADMIN}
        )
    await client.aclose()
    assert response.status_code == 502
    for secret in ("hunter2", "db-primary.internal", "psycopg2", "5432"):
        assert secret not in response.text, f"{secret!r} leaked to the client"
    assert response.json()["error"]["code"] == -32002


def test_non_ascii_in_any_header_does_not_crash():
    """The round-one fix guarded ``Authorization`` alone; every other client
    header was copied into the upstream request, where httpx ASCII-encodes it
    and raised - so a UTF-8 ``User-Agent`` produced a hard 500 on every request.

    Driven over a raw socket, because httpx refuses to *send* a non-ASCII
    header - which is exactly why the only way this reaches a server is the way
    an attacker sends it.
    """
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_downstream.app), base_url="http://downstream"
    )
    app = create_app(downstream_url="http://downstream/mcp", client=downstream_client)
    body = b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

    statuses = {}
    with live_server(app) as base_url:
        port = int(base_url.rsplit(":", 1)[1])
        for header in (b"User-Agent", b"Accept", b"Mcp-Session-Id", b"X-Request-Id", b"X-Foo"):
            raw = (
                b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Authorization: Bearer viewer-token-def456\r\n"
                + header + b": caf\xc3\xa9-client\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
                sock.sendall(raw)
                chunks = []
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
            statuses[header.decode()] = b"".join(chunks).split(b"\r\n")[0].decode("utf-8", "replace")

    for header, status_line in statuses.items():
        assert " 200 " in status_line, f"{header} with a UTF-8 value -> {status_line!r}"


# --------------------------------------------------------------------------- #
# Upstream failure handling
# --------------------------------------------------------------------------- #
async def test_upstream_timeout_is_sanitised(downstream_log):
    async def always_timeout(request):
        raise httpx.ConnectTimeout("connect to 10.0.3.7:9002 timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(always_timeout))
    app = create_app(downstream_url="http://internal-mcp.svc.cluster.local:9002/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw_client:
        response = await gw_client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"}, headers={"Authorization": ADMIN}
        )
    await client.aclose()
    assert response.status_code == 504
    body = response.json()
    assert body["id"] == 9
    assert body["error"]["code"] == -32002
    text = response.text
    assert "10.0.3.7" not in text and "cluster.local" not in text and "Traceback" not in text


async def test_upstream_connection_error_is_sanitised():
    async def always_fail(request):
        raise httpx.ConnectError("[Errno 111] Connection refused to 10.0.3.7:9002", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(always_fail))
    app = create_app(downstream_url="http://internal-mcp.svc:9002/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw_client:
        response = await gw_client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"}, headers={"Authorization": ADMIN}
        )
    await client.aclose()
    assert response.status_code == 502
    assert response.json()["error"]["code"] == -32002
    assert "Errno 111" not in response.text and "10.0.3.7" not in response.text


async def test_upstream_non_json_response_is_sanitised():
    async def html_error(request):
        return httpx.Response(500, text="<html><body>nginx internal error at /srv/app/main.py:88</body></html>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(html_error))
    app = create_app(downstream_url="http://downstream/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw_client:
        response = await gw_client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"}, headers={"Authorization": ADMIN}
        )
    await client.aclose()
    assert response.status_code == 502
    assert "nginx" not in response.text and "main.py" not in response.text


async def test_duplicate_authorization_headers_are_refused(downstream_log):
    """Starlette returns the first of a repeated header. A fronting proxy that
    took the *last* would authorise a different principal than the one this
    gateway checked - a confused deputy. Ambiguity is refused, not resolved."""
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_downstream.app), base_url="http://downstream"
    )
    app = create_app(downstream_url="http://downstream/mcp", client=downstream_client)
    with live_server(app) as base_url:
        port = int(base_url.rsplit(":", 1)[1])
        raw = (
            b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Authorization: Bearer viewer-token-def456\r\n"
            b"Authorization: Bearer admin-token-abc123\r\n"
            b"Content-Type: application/json\r\nContent-Length: 76\r\nConnection: close\r\n\r\n"
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_x"}}  '
        )
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(raw)
            chunks = []
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
    await downstream_client.aclose()
    response = b"".join(chunks).decode("utf-8", "replace")
    assert " 401 " in response.split("\r\n")[0], f"expected 401, got {response.splitlines()[0]!r}"
    assert downstream_log == []


# --------------------------------------------------------------------------- #
# Policy unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tool_name, role, allowed",
    [
        ("admin_reset_key", "admin", True),
        ("admin_reset_key", "viewer", False),
        ("admin_", "viewer", False),
        ("admin_delete_tenant", "viewer", False),
        ("get_customer_record", "viewer", True),
        ("get_customer_record", "admin", True),
        # ``admin_`` is a PREFIX rule, not a substring one.
        ("get_admin_reset_key", "viewer", True),
        ("x_admin_reset_key", "viewer", True),
        ("administrator_tools", "viewer", True),
    ],
)
def test_the_admin_rule_is_a_prefix_rule(tool_name, role, allowed):
    decision = Policy().check_tool_call(tool_name, role)
    assert decision.allowed is allowed
    if not allowed:
        assert decision.required_role == "admin"


def test_an_unknown_role_is_refused_an_admin_tool():
    """Anything that is not the required role is refused - the check is an
    equality against the rule, not a deny-list of known-bad roles."""
    for role in ("", "Admin", "ADMIN", "superadmin", "root", "admin "):
        assert Policy().check_tool_call("admin_reset_key", role).allowed is False


# --------------------------------------------------------------------------- #
# Round-two findings
# --------------------------------------------------------------------------- #


async def test_nan_and_infinity_are_not_accepted_as_json(gateway, downstream_log):
    """Python's json accepts and re-emits them; no other parser does, so the
    gateway would forward a body that is not valid JSON."""
    for literal in (b"NaN", b"Infinity", b"-Infinity"):
        response = await gateway.post(
            "/mcp",
            content=b'{"jsonrpc":"2.0","id":' + literal + b',"method":"tools/list"}',
            headers={"Authorization": VIEWER, "Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == -32700
    assert downstream_log == []


# --------------------------------------------------------------------------- #
# Round-three findings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tool", ["get_customer_record", "search_orders", "admin-reset-key", "list_admins", "a.b.c"]
)
async def test_wellformed_names_are_not_over_blocked(gateway, tool):
    """The identifier check must not make legitimate tools unreachable."""
    response = await rpc(gateway, call_payload(tool, {"query": "x"}), token=VIEWER)
    error = response.json().get("error")
    assert error is None or error["code"] != -32602 or "not permitted" not in error["message"], (
        f"{tool!r} was wrongly rejected as a malformed name"
    )


@pytest.mark.parametrize("injected", ["a\rb", "a\nb", "a\x00b", "a\tb", "a\x7fb"])
def test_emitted_header_values_strip_control_characters(injected):
    """ASCII is not enough - control characters are ASCII. Header splitting was
    blocked only by the inbound parser rejecting them first."""
    cleaned = _ascii_header(injected)
    assert cleaned.isprintable(), f"{injected!r} -> {cleaned!r}"
    assert "\r" not in cleaned and "\n" not in cleaned and "\x00" not in cleaned


def test_the_correlation_id_rejects_control_characters():
    class FakeRequest:
        def __init__(self, value):
            self.headers = {"x-request-id": value}

    assert request_reference(FakeRequest("clean-id-123")) == "clean-id-123"
    for bad in ("a\rb", "a\nb", "a\x00b"):
        assert request_reference(FakeRequest(bad)) != bad, f"{bad!r} was echoed verbatim"


async def test_a_client_disconnect_mid_upload_is_not_an_error():
    """``request.stream()`` raises ClientDisconnect, which reached the catch-all
    and logged a stack trace per aborted request - a cheap log-flood."""
    downstream_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_downstream.app), base_url="http://downstream")
    app = create_app(downstream_url="http://downstream/mcp", client=downstream_client)

    with live_server(app) as base_url:
        port = int(base_url.rsplit(":", 1)[1])
        for _ in range(5):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(
                    b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    b"Authorization: Bearer viewer-token-def456\r\n"
                    b"Content-Type: application/json\r\nContent-Length: 100000\r\n\r\n"
                    + b"x" * 100
                )
                sock.close()   # abort mid-body
        # The server must still be serving.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            body = b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
            sock.sendall(
                b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Authorization: Bearer viewer-token-def456\r\n"
                b"Content-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
            )
            received = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                received += chunk
    await downstream_client.aclose()
    assert b" 200 " in received.split(b"\r\n")[0]


# --- coverage for controls that mutation testing showed were untested ------- #


async def test_the_gateway_forces_its_own_content_type():
    """An upstream answering text/html with a JSON body would otherwise have
    that type relayed to the client."""
    async def html_typed(request):
        return httpx.Response(
            200,
            content=json.dumps({"jsonrpc": "2.0", "id": 1,
                                "result": {"x": "<script>alert(1)</script>"}}).encode(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(html_typed))
    app = create_app(downstream_url="http://downstream/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                 headers={"Authorization": ADMIN})
    await client.aclose()
    assert response.headers["content-type"].startswith("application/json")


async def test_an_unexpected_error_becomes_a_json_rpc_frame_not_a_bare_500():
    """The catch-all backstop: every path it names is handled upstream of it,
    so it had no coverage at all."""
    async def explode(request):
        raise ZeroDivisionError("boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(explode))
    app = create_app(downstream_url="http://downstream/mcp", client=client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw"
    ) as gw:
        response = await gw.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                 headers={"Authorization": ADMIN})
    await client.aclose()
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == -32603
    assert "boom" not in response.text and "Traceback" not in response.text
    assert "X-Request-Id" in response.headers or "x-request-id" in response.headers


def test_emitted_header_values_are_length_capped():
    assert len(_ascii_header("x" * 5000)) <= 200




# --------------------------------------------------------------------------- #
# Wire-format conformance
# --------------------------------------------------------------------------- #
async def test_authentication_failure_is_not_the_authorization_code(gateway, downstream_log):
    """-32001 is the code the brief assigns to the tool-authorization decision.
    Reusing it for "I do not know who you are" left a client unable to tell an
    invalid token from an insufficient role by code alone."""
    for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic xyz"}):
        response = await gateway.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == -32003, "auth reused the authz code"
    assert downstream_log == []


async def test_a_denied_notification_gets_no_response_body(gateway, downstream_log):
    """JSON-RPC 2.0 4.1: a server MUST NOT reply to a notification. The call is
    still refused - it simply is not forwarded - but answering it with
    ``"id": null`` was a protocol violation."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/call",
              "params": {"name": "admin_reset_key", "arguments": {}}},
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 202
    assert response.content == b"", f"a notification was answered: {response.content!r}"
    assert downstream_log == [], "a denied notification reached the downstream"


async def test_a_forwarded_notification_gets_no_response_body(gateway):
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "tools/list"}, headers={"Authorization": VIEWER}
    )
    assert response.status_code == 202
    assert response.content == b"", f"got a body for a notification: {response.content!r}"


async def test_a_float_id_is_echoed_not_dropped(gateway):
    """A float is a JSON number and so a legal id. Replacing it with ``null``
    left the caller unable to correlate the refusal with its own request."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1.5, "method": "tools/call",
              "params": {"name": "admin_reset_key", "arguments": {}}},
        headers={"Authorization": VIEWER},
    )
    body = response.json()
    assert body["id"] == 1.5, f"id was not echoed: {body}"
    assert body["error"]["code"] == -32001


async def test_a_boolean_id_is_still_not_echoed(gateway):
    """``true`` is not a JSON-RPC id even though Python says it is an int."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": True, "method": "tools/call",
              "params": {"name": "admin_reset_key", "arguments": {}}},
        headers={"Authorization": VIEWER},
    )
    assert response.json()["id"] is None


async def test_an_oversized_body_is_refused_before_it_is_buffered(gateway, downstream_log):
    """The gateway buffers the raw bytes, parses them, then re-serialises to
    forward - roughly 3x the body in memory before anything is checked."""
    from task2_mcp_gateway import gateway as gateway_module

    oversized = "A" * (gateway_module.MAX_BODY_BYTES + 1024)
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "search_orders", "arguments": {"query": oversized}}},
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == -32600
    assert downstream_log == [], "an oversized body was forwarded"


async def test_an_oversized_chunked_body_is_refused_as_it_arrives(gateway, downstream_log):
    """The declared-length check is the easy half. A chunked upload declares no
    length at all, so the cap has to hold as the bytes arrive - checking after
    buffering would allocate exactly the memory the cap exists to prevent."""
    from task2_mcp_gateway import gateway as gateway_module

    async def chunked():
        # No Content-Length: httpx uses chunked transfer for a stream body.
        yield b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{"q":"'
        for _ in range((gateway_module.MAX_BODY_BYTES // 65536) + 4):
            yield b"A" * 65536
        yield b'"}}}'

    response = await gateway.post("/mcp", content=chunked(), headers={"Authorization": VIEWER})
    assert response.status_code == 413, f"a chunked oversized body was accepted: {response.status_code}"
    assert downstream_log == []


async def test_http_level_errors_still_speak_json_rpc(gateway):
    """Starlette answers 405 with its own ``{"detail": ...}``, which is not a
    frame an MCP client can parse."""
    response = await gateway.get("/mcp", headers={"Authorization": VIEWER})
    assert response.status_code == 405
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["error"]["code"] == -32600
    assert "detail" not in body


async def test_a_trailing_slash_does_not_redirect_the_bearer_token(gateway):
    """A 307 carries the caller's Authorization header to wherever it points.
    A security proxy should not be issuing those."""
    response = await gateway.post(
        "/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": VIEWER},
    )
    assert response.status_code != 307, "the gateway redirected a credentialed request"
    assert response.status_code == 404
    assert response.json()["jsonrpc"] == "2.0"


async def test_a_null_id_is_an_invalid_request_not_a_notification(gateway, downstream_log):
    """JSON-RPC treats a message carrying an ``id`` member as a Request whatever
    its value, and MCP forbids a null one. Relaying it let the downstream's
    ``null`` body through as the response, which is not a Response object."""
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "id": None, "method": "tools/list"},
        headers={"Authorization": VIEWER},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert body != None  # noqa: E711 - the point is that the body is a frame
    assert downstream_log == []


@pytest.mark.parametrize("method", ["Tools/Call", "TOOLS/CALL", " tools/call", "tools/call "])
async def test_a_near_miss_spelling_of_tools_call_is_refused(gateway, downstream_log, method):
    """"Check one exact string, forward everything else" makes the gateway's
    correctness depend on the downstream's parser matching its own."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method,
              "params": {"name": "admin_reset_key", "arguments": {}}},
        headers={"Authorization": VIEWER},
    )
    assert response.json()["error"]["code"] == -32600
    assert downstream_log == [], f"{method!r} was forwarded without a policy check"


@pytest.mark.parametrize("name", ["ADMIN_reset_key", " admin_reset_key", "Admin_Reset_Key"])
async def test_a_near_miss_spelling_of_an_admin_tool_is_refused(gateway, downstream_log, name):
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": {}}},
        headers={"Authorization": VIEWER},
    )
    assert response.json()["error"]["code"] == -32001
    assert downstream_log == [], f"{name!r} was forwarded without a policy check"


@pytest.mark.parametrize(
    "downstream_payload, why",
    [
        ({"jsonrpc": "2.0", "id": 999, "result": {}}, "id does not match the request"),
        ({"foo": "bar"}, "not a JSON-RPC response at all"),
        ({"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": -1, "message": "x"}},
         "both result and error, which JSON-RPC forbids"),
        ([{"jsonrpc": "2.0", "id": 1, "result": {}}], "a bare array"),
    ],
)
async def test_a_non_conforming_downstream_response_is_not_relayed(downstream_payload, why):
    """The gateway validated inbound to the byte and took the downstream's reply
    on faith - the wrong way round for a security proxy."""
    async def handler(request):
        return httpx.Response(200, json=downstream_payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER},
        )
    await client.aclose()
    assert response.status_code == 502, f"relayed a response that was {why}"
    assert response.json()["error"]["code"] == -32603


async def test_a_downstream_redirect_is_not_treated_as_a_result():
    """A 3xx fell through to ``upstream.json()`` and was rescued only by the
    ValueError handler, so a redirect carrying a JSON body would be relayed."""
    async def handler(request):
        return httpx.Response(
            301, json={"jsonrpc": "2.0", "id": 1, "result": {"moved": True}},
            headers={"location": "http://internal.svc/elsewhere"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER},
        )
    await client.aclose()
    assert response.status_code == 502
    assert "internal.svc" not in response.text


@pytest.mark.parametrize("header", ["Bearer admin-token-abc123   ", "Bearer  admin-token-abc123"])
async def test_a_padded_token_does_not_authenticate(gateway, header):
    """Stripping the token meant padding a real secret still authenticated."""
    response = await gateway.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Authorization": header},
    )
    assert response.status_code == 401


async def test_an_oversized_downstream_response_is_not_relayed():
    """The inbound cap protects the gateway from its clients; this one protects
    it from the server it proxies to, which it has no more reason to trust with
    its memory."""
    from task2_mcp_gateway import gateway as gateway_module

    async def handler(request):
        padding = "A" * (gateway_module.MAX_RESPONSE_BYTES + 1024)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": padding}]}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER},
        )
    await client.aclose()
    assert response.status_code == 502
    assert len(response.content) < 4096, "an oversized downstream body was relayed"


async def test_a_tool_name_cannot_forge_an_audit_record(gateway, caplog):
    """A newline in a caller-controlled tool name let a viewer append a
    syntactically perfect forged audit line naming a different subject and
    role. Emitted headers were already sanitised for this; the log sink - the
    evidence trail in a security gateway - was not."""
    import logging

    evil = (
        "admin_x\ndenied tools/call name=harmless "
        "subject=attacker@evil role=admin required=admin"
    )
    with caplog.at_level(logging.WARNING, logger="mcp-gateway"):
        response = await gateway.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": evil, "arguments": {}}},
            headers={"Authorization": VIEWER},
        )
    assert response.json()["error"]["code"] == -32001
    for record in caplog.records:
        assert "\n" not in record.getMessage(), f"newline reached the log: {record.getMessage()!r}"
        assert "attacker@evil role=admin" not in record.getMessage().split("subject=")[-1]


async def test_an_oversized_downstream_reply_is_capped_as_it_arrives():
    """The cap awaited a buffering ``post()`` and then measured the body, so it
    fired after the allocation it exists to prevent."""
    from task2_mcp_gateway import gateway as gateway_module

    chunk = b"A" * 65536
    total = gateway_module.MAX_RESPONSE_BYTES + 10 * len(chunk)

    async def handler(request):
        async def generate():
            sent = 0
            while sent < total:
                sent += len(chunk)
                yield chunk

        return httpx.Response(200, content=generate(), headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER},
        )
    await client.aclose()
    assert response.status_code == 502
    assert len(response.content) < 4096


def _streamable_http_server(record: dict):
    """A downstream that behaves like a spec-conformant MCP server."""
    application = _FastAPI()

    @application.post("/mcp")
    async def endpoint(request: _Request):
        record["accept"] = request.headers.get("accept", "")
        record["session"] = request.headers.get("mcp-session-id")
        record["protocol"] = request.headers.get("mcp-protocol-version")
        if "text/event-stream" not in record["accept"]:
            # What a conformant server does when the client will not take SSE.
            return _Response("Not Acceptable", status_code=406)
        return _Response(
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n',
            media_type="text/event-stream",
            headers={"Mcp-Session-Id": "sess-42"},
        )

    return application


async def test_the_gateway_can_proxy_a_streamable_http_mcp_server():
    """Two independent blockers made this impossible: ``Accept`` was hardcoded
    to ``application/json`` so a conformant server answered 406, and an SSE
    reply failed ``upstream.json()`` and became a 502."""
    record: dict = {}
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_streamable_http_server(record)), base_url="http://down"
    )
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER, "Mcp-Session-Id": "sess-42",
                     "MCP-Protocol-Version": "2025-06-18"},
        )
    await client.aclose()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"result"' in response.text
    # Session identity survives the hop in both directions.
    assert record["session"] == "sess-42"
    assert record["protocol"] == "2025-06-18"
    assert response.headers.get("mcp-session-id") == "sess-42"
    assert "text/event-stream" in record["accept"]


async def test_the_caller_token_is_still_not_forwarded_on_the_mcp_path(downstream_log):
    """Relaying session headers must not become relaying everything."""
    record: dict = {}
    server = _streamable_http_server(record)

    seen: dict = {}

    @server.middleware("http")
    async def capture(request: _Request, call_next):
        seen["authorization"] = request.headers.get("authorization")
        seen["role"] = request.headers.get("x-mcp-role")
        return await call_next(request)

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://down")
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": VIEWER, "X-Mcp-Role": "admin"},
        )
    await client.aclose()
    assert seen["authorization"] == "Bearer downstream-service-token"
    assert seen["role"] == "viewer", "a client-supplied role was believed"


async def test_a_compressing_downstream_is_not_a_total_outage():
    """The streamed-cap rewrite rebuilt the reply carrying the original
    ``content-encoding`` onto bytes httpx had already decoded, so it decoded a
    second time and raised - every request against any gzip-compressing
    downstream returned 502.

    The downstream here compresses regardless of ``Accept-Encoding``. That is
    deliberate: the gateway now also asks for ``identity``, and with a
    politely-negotiating downstream nothing would be compressed and this test
    would pass against the broken code. Two defences, and this one tests the
    inner one.
    """
    import gzip as _gzip

    body = _json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "x"} for _ in range(50)]}}
    ).encode()

    async def handler(request):
        return httpx.Response(
            200,
            content=_gzip.compress(body),
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(downstream_url="http://down/mcp", client=client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as gw:
        response = await gw.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Authorization": ADMIN},
        )
    await client.aclose()
    assert response.status_code == 200, response.text
    assert response.json()["result"]["tools"]


@pytest.mark.parametrize("bad_id", [True, False, {"a": 1}, [1], [], {}])
async def test_a_non_scalar_id_never_reaches_the_downstream(gateway, downstream_log, bad_id):
    """JSON-RPC allows a String, Number or Null id. Accepting anything else
    meant the id was mapped to None, the request was FORWARDED, the privileged
    tool executed, and the correct reply was then rejected as malformed - so a
    side-effectful call ran and the client was told the upstream broke."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": bad_id, "method": "tools/call",
              "params": {"name": "admin_reset_key", "arguments": {"tenant_id": "acme"}}},
        headers={"Authorization": ADMIN},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600
    assert downstream_log == [], f"id={bad_id!r} executed the tool: {downstream_log}"


async def test_a_forwarded_call_is_sent_exactly_once(gateway, downstream_log):
    """Peeking at the reply's content type must not mean issuing the request
    twice - doing so rotated the key twice for one client call."""
    response = await gateway.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "admin_reset_key", "arguments": {"tenant_id": "acme"}}},
        headers={"Authorization": ADMIN},
    )
    assert "error" not in response.json()
    assert [c["tool"] for c in downstream_log] == ["admin_reset_key"]


async def test_an_sse_reply_is_relayed_incrementally():
    """Draining the stream before relaying made a notification channel arrive
    as one socket read after the upstream finished - not a stream at all."""
    import asyncio as _asyncio

    from fastapi.responses import StreamingResponse as _Streaming

    provider = _FastAPI()

    @provider.post("/mcp")
    async def endpoint(request: _Request):
        async def generate():
            for index in range(4):
                yield (
                    f'event: message\ndata: {{"jsonrpc":"2.0","id":{index},"result":{{}}}}\n\n'
                ).encode()
                await _asyncio.sleep(0.3)

        return _Streaming(generate(), media_type="text/event-stream")

    @provider.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    with live_server(provider) as upstream:
        app = create_app(downstream_url=f"{upstream}/mcp")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                arrivals = []
                start = time.perf_counter()
                async with client.stream(
                    "POST", f"{gateway_url}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Authorization": VIEWER},
                ) as response:
                    assert response.headers["content-type"].startswith("text/event-stream")
                    async for _ in response.aiter_bytes():
                        arrivals.append(time.perf_counter() - start)

    assert len(arrivals) >= 3, f"buffered into {len(arrivals)} read(s): {arrivals}"
    assert arrivals[-1] - arrivals[0] > 0.4, f"all arrived at once: {arrivals}"


def _sse_downstream(status: int = 200, secret: str = ""):
    """A downstream that answers with an event stream."""
    from fastapi.responses import StreamingResponse as _Streaming

    application = _FastAPI()

    @application.post("/mcp")
    async def endpoint(request: _Request):
        if status >= 300:
            return _Response(
                f"data: TRACEBACK db={secret}\n\n", status_code=status,
                media_type="text/event-stream",
            )

        async def generate():
            yield b'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'

        return _Streaming(generate(), media_type="text/event-stream")

    @application.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return application


async def test_a_non_2xx_sse_reply_is_sanitised_like_any_other():
    """Testing the content type before the status let an SSE-framed failure
    skip every check below it - a downstream 500 was relayed verbatim, status
    and body, so a stack trace and a DSN reached the client past the
    sanitisation this module documents and the JSON path performs."""
    secret = "postgres://user:pw@internal-host/db"
    with live_server(_sse_downstream(status=500, secret=secret)) as upstream:
        app = create_app(downstream_url=f"{upstream}/mcp")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{gateway_url}/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Authorization": ADMIN},
                )
    assert response.status_code == 502
    assert secret not in response.text, f"DSN reached the client: {response.text!r}"
    assert response.json()["error"]["code"] == -32002


async def test_a_notification_gets_no_body_on_the_sse_path_either():
    """The JSON path applied this rule; the SSE path returned before reaching
    it, so a notification came back with a full event-stream body."""
    with live_server(_sse_downstream()) as upstream:
        app = create_app(downstream_url=f"{upstream}/mcp")
        with live_server(app) as gateway_url:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{gateway_url}/mcp",
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers={"Authorization": ADMIN},
                )
    assert response.status_code == 202
    assert response.content == b"", f"a notification was answered: {response.content!r}"

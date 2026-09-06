"""A mock MCP server speaking JSON-RPC over HTTP.

Stands in for the real tool server behind the gateway. It performs **no**
authorization of its own - that is the point of the exercise: the gateway is
the only thing between a viewer and ``admin_reset_key``, so a test that gets a
successful reset back through the gateway is a genuine policy failure.

It does record who called it, so tests can assert the gateway propagates
identity and never forwards the caller's own bearer token upstream.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DOWNSTREAM_SERVICE_TOKEN = os.environ.get("DOWNSTREAM_SERVICE_TOKEN", "downstream-service-token")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_customer_record",
        "description": "Look up a customer billing record.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string", "pattern": r"^CUST-\d{5}$"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_orders",
        "description": "Search orders by free text.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate a tenant's API key. Privileged.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "admin_delete_tenant",
        "description": "Permanently delete a tenant. Privileged.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
            "additionalProperties": False,
        },
    },
]

app = FastAPI(title="mock-downstream-mcp")

#: Every request the downstream actually saw. Tests assert on this to prove
#: blocked calls never reached it.
CALL_LOG: list[dict[str, Any]] = []


def _result(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _text_result(request_id: Any, text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if structured is not None:
        payload["structuredContent"] = structured
    return _result(request_id, payload)


def handle_message(message: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    CALL_LOG.append(
        {
            "method": method,
            "tool": params.get("name") if method == "tools/call" else None,
            "forwarded_user": headers.get("x-mcp-user"),
            "forwarded_role": headers.get("x-mcp-role"),
            "authorization": headers.get("authorization"),
        }
    )

    # A notification carries no id and gets no response.
    if request_id is None:
        return None

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion", "2026-07-28"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-downstream", "version": "1.0.0"},
            },
        )
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "get_customer_record":
            return _text_result(
                request_id,
                f"record for {arguments.get('customer_id')}",
                {"customer_id": arguments.get("customer_id"), "plan": "enterprise"},
            )
        if name == "search_orders":
            return _text_result(request_id, f"0 orders matching {arguments.get('query')!r}", {"orders": []})
        if name in ("admin_reset_key", "admin_delete_tenant"):
            # Reaching this line via a viewer's request means the gateway failed.
            return _text_result(
                request_id,
                f"{name} executed for tenant {arguments.get('tenant_id')}",
                {"executed": name, "tenant_id": arguments.get("tenant_id")},
            )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": f"Unknown tool: {name}"},
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "Method not found", "data": method},
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    headers = {k.lower(): v for k, v in request.headers.items()}

    response = handle_message(body, headers)
    if response is None:
        return JSONResponse(content=None, status_code=202)
    return JSONResponse(content=response)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

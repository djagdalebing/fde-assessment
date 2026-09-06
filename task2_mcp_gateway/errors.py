"""JSON-RPC error construction for the gateway.

The gateway speaks JSON-RPC even when it refuses to proxy, because the caller
is an MCP client and a bare HTTP 403 with an HTML body is not something an
agent can act on. Transport-level problems (no credential at all) still get the
right HTTP status *and* a JSON-RPC body, so both layers agree.
"""

from __future__ import annotations

from typing import Any

# Standard JSON-RPC 2.0 codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Implementation-defined server errors (-32000..-32099 is reserved for these).
#
# ``-32001`` is the code the brief assigns to the *authorization* decision:
# "you may not call this tool". Authentication - "I do not know who you are" -
# is a different failure with a different remedy, and reusing -32001 for it
# left a client unable to tell "my token is invalid" from "my role is
# insufficient" by code alone.
UNAUTHORIZED_TOOL_CALL = -32001
UPSTREAM_UNAVAILABLE = -32002
UNAUTHENTICATED = -32003


def error_response(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC error object.

    ``request_id`` is echoed verbatim - including ``None``, which is what the
    spec requires when the request was unparseable enough that no id could be
    recovered.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def unauthorized_tool_call(request_id: Any, tool_name: str, role: str, required: str) -> dict[str, Any]:
    return error_response(
        request_id,
        UNAUTHORIZED_TOOL_CALL,
        "Unauthorized Tool Call",
        {
            "tool": tool_name,
            "required_role": required,
            "actual_role": role,
            "reason": f"Tool '{tool_name}' requires the '{required}' role; caller holds '{role}'.",
        },
    )

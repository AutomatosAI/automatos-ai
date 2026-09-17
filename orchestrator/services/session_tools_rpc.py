"""PRD-245 W1 — the MCP wire a ticket session talks to us over (decision D1).

Claude Code (and Codex) speak the Model Context Protocol: JSON-RPC 2.0 over a
single HTTP endpoint. This module is that protocol and nothing else — auth,
scope and execution live in ``api/session_tools.py`` and
``services/session_tools.py`` — so the wire is unit-testable without a client.

**Why hand-written rather than the SDK.** The official ``mcp`` package requires
``uvicorn>=0.31.1`` (and a new httpx major); this backend pins
``uvicorn==0.24.0`` and boots on it. Taking the SDK would mean upgrading the ASGI
server under a mature app for four JSON shapes — and this repo has already lost a
day to a starlette/fastapi pin interaction (``starlette-coldboot-include-router``).
The protocol a tools-only server must answer is small and stable; we implement it
and test it. Recorded as the PRD's named fallback, not an oversight.

What a client needs answered, in the order Claude Code 2.1.267 sends it:

* ``server/discover`` — the v2 runtime's capability PROBE, sent before anything
  else. Method-not-found over HTTP 200 is the correct answer: it classifies us
  as a legacy server and the real handshake follows;
* ``initialize`` — version + capabilities + who we are;
* ``notifications/initialized`` (and any other notification) — no reply at all;
* ``tools/list`` — the fixed list (``services/session_tools.py``);
* ``tools/call`` — run one, and report a FAILURE AS A RESULT (``isError``), not
  as a JSON-RPC error: a tool that refused is something the model must read and
  act on, while a JSON-RPC error is a broken request;
* ``ping`` — liveness;
* anything else — method not found, which a client tolerates for a capability we
  never advertised.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from services import session_tools
from services.session_tools import SessionContext, SessionToolRefused

logger = logging.getLogger(__name__)

SERVER_NAME = "automatos"
SERVER_TITLE = "Automatos"
# The dated protocol revisions we answer. A client asking for one of these gets
# it echoed; anything else gets our newest, which is what the spec asks of a
# server that cannot speak the requested version.
#
# These are exactly the pre-2026 revisions Claude Code 2.1.267 accepts back:
# echoing anything outside its list — ``2026-07-28`` above all, which its own
# capability PROBE names — fails the handshake with "Server's protocol version
# is not supported". 2.1.267 asks for 2025-11-25.
SUPPORTED_PROTOCOL_VERSIONS: Tuple[str, ...] = (
    "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07",
)
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
# The v2-runtime capability probe a 2.1.267 client sends BEFORE ``initialize``.
# Answering it with method-not-found (HTTP 200, JSON-RPC -32601) is what makes
# the client classify us as a legacy server and carry on with the handshake;
# hanging, closing or a 4xx here loses the connection before it starts.
DISCOVERY_PROBE_METHOD = "server/discover"

# JSON-RPC 2.0
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

MAX_RESULT_CHARS = session_tools.MAX_TOOL_RESULT_CHARS


def _error(rpc_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": err}


def _result(rpc_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _text_content(text: str, *, is_error: bool = False) -> Dict[str, Any]:
    body = text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n…(truncated)"
    return {"content": [{"type": "text", "text": body}], "isError": bool(is_error)}


def render_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """One executor result as MCP tool output. A refusal or failure is content
    with ``isError``, so the model reads WHY and can choose differently."""
    ok = bool(result.get("success"))
    if ok:
        payload = {k: v for k, v in result.items() if k not in ("success", "tool")}
        body = payload.get("result", payload) if isinstance(payload, dict) else payload
        return _text_content(json.dumps(body, indent=2, default=str) if not isinstance(body, str) else body)
    message = str(result.get("error") or "the tool did not succeed and gave no reason")
    hint = result.get("hint") or result.get("detail")
    return _text_content(f"{message}\n{hint}" if hint else message, is_error=True)


def protocol_version(requested: Any) -> str:
    asked = str(requested or "").strip()
    return asked if asked in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION


def initialize_result(requested: Any, *, server_version: str) -> Dict[str, Any]:
    return {
        "protocolVersion": protocol_version(requested),
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "title": SERVER_TITLE, "version": server_version},
        "instructions": (
            "Automatos, the manager that gave you this ticket. These tools reach the board, "
            "your reports and the workspace's knowledge. Scope is fixed to your own ticket."
        ),
    }


def tools_list_result() -> Dict[str, Any]:
    return {"tools": [dict(t) for t in session_tools.definitions()]}


async def handle_message(
    message: Any,
    ctx: SessionContext,
    *,
    server_version: str,
    call: Callable[[session_tools.SessionTool, Any, SessionContext], Awaitable[Dict[str, Any]]],
    on_call: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message → the reply, or ``None`` for a notification.

    ``call`` runs a tool with ALREADY-SCOPED parameters (injected, so the wire
    is testable without the executor); the scope — and the refusal when a call
    asks for something a session may not have — is applied HERE, so no caller
    can pass round it. ``on_call`` is asked before each tool call and returns a
    REFUSAL REASON when the ticket has spent its allowance, else ``None``.
    """
    if not isinstance(message, dict):
        return _error(None, INVALID_REQUEST, "a JSON-RPC message must be an object")
    rpc_id = message.get("id")
    method = str(message.get("method") or "")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    is_notification = "id" not in message

    if method.startswith("notifications/"):
        return None
    if is_notification:
        return None

    if method == "initialize":
        return _result(rpc_id, initialize_result(params.get("protocolVersion"), server_version=server_version))
    if method == "ping":
        return _result(rpc_id, {})
    if method == "tools/list":
        return _result(rpc_id, tools_list_result())
    if method == "tools/call":
        return await _handle_tools_call(rpc_id, params, ctx, call=call, on_call=on_call)
    if method in ("prompts/list", "resources/list", "resources/templates/list"):
        # Capabilities we never advertised. A client that asks anyway (its
        # discovery step lists prompts and resources alongside tools) gets an
        # empty list rather than an error: cheaper than being reported broken.
        return _result(rpc_id, {"prompts": []} if method == "prompts/list" else {"resources": []})
    # Everything else — including the ``server/discover`` probe, which MUST get
    # this and not a 4xx — is method-not-found over HTTP 200.
    return _error(rpc_id, METHOD_NOT_FOUND, f"unknown method {method!r}")


async def _handle_tools_call(
    rpc_id: Any,
    params: Dict[str, Any],
    ctx: SessionContext,
    *,
    call: Callable[[session_tools.SessionTool, Any, SessionContext], Awaitable[Dict[str, Any]]],
    on_call: Optional[Callable[[str], Optional[str]]],
) -> Dict[str, Any]:
    name = str(params.get("name") or "")
    tool = session_tools.get_tool(name)
    if tool is None:
        offered = ", ".join(session_tools.tool_names())
        return _result(rpc_id, _text_content(
            f"{name!r} is not a tool this session has. Available: {offered}.", is_error=True))
    if on_call is not None:
        refusal = on_call(tool.name)
        if refusal:
            return _result(rpc_id, _text_content(refusal, is_error=True))
    try:
        scoped = session_tools.resolve_parameters(tool, params.get("arguments"), ctx)
        result = await call(tool, scoped, ctx)
    except SessionToolRefused as refused:
        return _result(rpc_id, _text_content(str(refused), is_error=True))
    except Exception:  # noqa: BLE001 — a broken tool is output, never a dead session
        logger.error("[session-tools] %s failed for ticket #%s", tool.name, ctx.task_id, exc_info=True)
        return _result(rpc_id, _text_content(
            f"{tool.name} could not run — the operator can see why in the backend log.", is_error=True))
    return _result(rpc_id, render_result(result))


async def handle_payload(
    payload: Any,
    ctx: SessionContext,
    *,
    server_version: str,
    call: Callable[[session_tools.SessionTool, Any, SessionContext], Awaitable[Dict[str, Any]]],
    on_call: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[Any]:
    """A whole request body: one message, or a batch. ``None`` = nothing to
    answer (every message was a notification), which the route sends as 202."""
    if isinstance(payload, list):
        if not payload:
            return _error(None, INVALID_REQUEST, "an empty batch is not a request")
        replies: List[Dict[str, Any]] = []
        for message in payload:
            reply = await handle_message(message, ctx, server_version=server_version, call=call, on_call=on_call)
            if reply is not None:
                replies.append(reply)
        return replies or None
    return await handle_message(payload, ctx, server_version=server_version, call=call, on_call=on_call)

"""PRD-245 W1 — ``/api/v1/session-tools/mcp``: the loopback MCP endpoint a ticket
session calls Automatos through (decision D1).

One audience, one guard: the SESSION. It authenticates with the per-ticket token
minted at claim (``Authorization: Bearer`` / ``X-Session-Token``), which resolves
to exactly one running ticket, its agent and its workspace — never to a user. A
call names a tool; the ticket names the scope.

Three things this route owns, and nothing else:

* **auth** — the token → ``SessionContext``; a token whose ticket has ended
  resolves to nothing (401);
* **the allowance** — a bound on how many tool calls one ticket may make, so a
  looping session cannot hammer the board (``SESSION_TOOLS_MAX_CALLS_PER_TICKET``);
* **transport** — JSON in, JSON out; a notification is a 202 with no body.

The protocol is ``services/session_tools_rpc.py``; execution is
``services/session_tools.py`` (through the API agents' own executor, so the
platform's policy gate, registry validation and tool telemetry all apply).

Behind ``CLI_RUNTIME_ENABLED`` (404 when off) like every other session-mode
route: the boot gate refuses that flag outside the local edition, so this can
never be reachable in the hosted one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from services import cli_host_service as svc
from services import session_tools, session_tools_rpc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/session-tools", tags=["session-tools"])

SESSION_TOKEN_HEADER = "X-Session-Token"
BEARER_PREFIX = "bearer "
CALLS_KEY = "platform_calls"
# What this server calls itself to a client. The TOOL LIST is the contract that
# matters; this version moves when the wave changes what a session can do.
SERVER_VERSION = "1.0"


def _require_cli_runtime() -> None:
    if not bool(getattr(config, "CLI_RUNTIME_ENABLED", False)):
        raise HTTPException(
            status_code=404,
            detail="Session mode (CLI runtime) is not enabled on this instance "
                   "(CLI_RUNTIME_ENABLED=true, local edition only).",
        )


def bearer_token(request: Request) -> str:
    """The session's token from either header. No OAuth, no discovery: this is a
    loopback endpoint with a per-ticket secret."""
    header = request.headers.get("Authorization") or ""
    if header.lower().startswith(BEARER_PREFIX):
        return header[len(BEARER_PREFIX):].strip()
    return (request.headers.get(SESSION_TOKEN_HEADER) or "").strip()


async def require_session(
    request: Request, db: Session = Depends(get_db)
) -> tuple[Any, session_tools.SessionContext]:
    """The running ticket this token belongs to, as the call's whole identity."""
    _require_cli_runtime()
    resolved = svc.resolve_session_token(db, bearer_token(request))
    if resolved is None:
        # 401 with NO ``WWW-Authenticate``: that header is exactly what starts a
        # client's OAuth discovery, and this endpoint has none — the ticket's
        # bearer token is the whole auth story. With a configured Authorization
        # header, a client reports the server as failed and attempts no browser
        # flow, which is what we want the operator to see.
        raise HTTPException(status_code=401, detail="invalid or expired session token")
    task, agent = resolved
    ctx = session_tools.SessionContext(
        task_id=int(task.id),
        agent_id=int(task.assigned_agent_id) if task.assigned_agent_id else None,
        agent_name=getattr(agent, "name", None),
        workspace_id=task.workspace_id,
    )
    return task, ctx


def call_allowance(db: Session, task: Any) -> Optional[str]:
    """Count this call against the ticket's allowance; a string is the refusal
    the model reads.

    The counter is written with ``jsonb_set`` — ONE key, in one statement —
    rather than by writing back a whole ``runtime_ref`` read at the start of the
    request. The host flushes its events onto the same row while the session is
    calling tools, and a whole-document write would silently drop whatever it
    had just recorded. One of the things it records is ``pending_permissions``,
    which is what decides whether a held command sends the ticket to review, so
    a lost update there would turn "the operator never answered" into a ticket
    that reads as finished.

    Fail-open on a bookkeeping error, but still COUNT: the cap is the only bound
    on this endpoint, and a database hiccup must not quietly remove it for the
    rest of the run.
    """
    cap = int(getattr(config, "SESSION_TOOLS_MAX_CALLS_PER_TICKET", 0) or 0)
    used = _count_call(db, task)
    if cap and used > cap:
        return (
            f"This ticket has used its {cap} Automatos tool calls. Work with what you have and "
            "finish your turn; say in your result that you hit the limit."
        )
    return None


def _count_call(db: Session, task: Any) -> int:
    """This call's number. Persisted where it can be; counted regardless."""
    from sqlalchemy import text as sql_text

    in_memory = int((task.runtime_ref or {}).get(CALLS_KEY) or 0) + 1
    try:
        row = db.execute(
            sql_text(
                """
                UPDATE board_tasks
                   SET runtime_ref = jsonb_set(
                           COALESCE(runtime_ref, '{}'::jsonb),
                           :key,
                           to_jsonb(COALESCE((runtime_ref ->> :field)::int, 0) + 1),
                           true)
                 WHERE id = :task_id
             RETURNING (runtime_ref ->> :field)::int
                """
            ),
            {"key": "{" + CALLS_KEY + "}", "field": CALLS_KEY, "task_id": int(task.id)},
        ).first()
        db.commit()
        if row and row[0] is not None:
            # keep the in-session object in step without writing the whole document
            task.runtime_ref = {**(task.runtime_ref or {}), CALLS_KEY: int(row[0])}
            return int(row[0])
    except Exception:  # noqa: BLE001 — a counter must never be why a ticket cannot work
        logger.debug("[session-tools] call counter not persisted for ticket #%s",
                     getattr(task, "id", "?"), exc_info=True)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
    task.runtime_ref = {**(task.runtime_ref or {}), CALLS_KEY: in_memory}
    return in_memory


# Only POST is defined on ``/mcp`` ON PURPOSE. A client opens a GET for a
# server-initiated SSE stream right after the handshake and accepts 405 for a
# server that has none — which is what FastAPI answers for an undefined method
# on a defined path. The same goes for the shutdown DELETE. Adding a GET here
# would mean owning a real event stream.
@router.post("/mcp")
async def session_tools_mcp(
    request: Request,
    session: tuple = Depends(require_session),
    db: Session = Depends(get_db),
) -> Response:
    """The MCP endpoint. JSON-RPC in, JSON-RPC out; 202 for a notification."""
    task, ctx = session
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 — a malformed body is a protocol error, not a 500
        return _json({"jsonrpc": "2.0", "id": None,
                      "error": {"code": session_tools_rpc.PARSE_ERROR, "message": "invalid JSON"}})

    async def _call(tool, scoped_params, context):
        # The wire scoped these (services/session_tools_rpc.py); we only run them.
        return await session_tools.call_tool(db, tool, scoped_params, context)

    reply = await session_tools_rpc.handle_payload(
        payload, ctx,
        server_version=SERVER_VERSION,
        call=_call,
        on_call=lambda _name: call_allowance(db, task),
    )
    if reply is None:
        return Response(status_code=202)
    return _json(reply)


def _json(body: Any) -> Response:
    from fastapi.responses import JSONResponse

    # No session id: this server is stateless, so a client has nothing to echo.
    return JSONResponse(content=body, headers={"MCP-Protocol-Version": session_tools_rpc.LATEST_PROTOCOL_VERSION})


@router.get("/manifest")
async def session_tools_manifest(
    session: tuple = Depends(require_session),
) -> Dict[str, Any]:
    """What this session may call, for a human reading the ticket. The same list
    the endpoint advertises — never a second definition."""
    _task, ctx = session
    return {
        "task_id": ctx.task_id,
        "agent": ctx.agent_name,
        "tools": [dict(t) for t in session_tools.definitions()],
    }

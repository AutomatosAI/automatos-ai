"""PRD-234 S1a — the CLI host routes (``/api/v1/cli-hosts``).

Two audiences, two guards:

* the **operator** (the local instance's single user) issues pairing codes and
  lists hosts — behind the workspace-admin guard, like every other admin surface;
* the **host** pairs once with a code, then authenticates every call with the
  ``X-CLI-Host-Token`` header (``require_cli_host``) and is confined to the
  workspace its row belongs to.

Every route is behind ``CLI_RUNTIME_ENABLED`` (404 when off): session mode is a
local-edition feature, and the boot gate in ``config.validate_auth_edition``
refuses the flag in saas — so these handlers can never do anything there.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.cli_hosts import CliHost, CliHostStatus
from services import cli_host_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cli-hosts", tags=["cli-hosts"])

HOST_TOKEN_HEADER = "X-CLI-Host-Token"


def _require_cli_runtime() -> None:
    if not bool(getattr(config, "CLI_RUNTIME_ENABLED", False)):
        raise HTTPException(
            status_code=404,
            detail="Session mode (CLI runtime) is not enabled on this instance "
                   "(CLI_RUNTIME_ENABLED=true, local edition only).",
        )


async def _require_operator(
    ctx: RequestContext = Depends(require_workspace_admin),
) -> RequestContext:
    """Operator surface: workspace admin (the local edition's single operator)."""
    _require_cli_runtime()
    return ctx


async def require_cli_host(
    host_id: UUID, request: Request, db: Session = Depends(get_db)
) -> CliHost:
    """Host surface: the paired host named in the path, proven by its token."""
    _require_cli_runtime()
    token = request.headers.get(HOST_TOKEN_HEADER) or ""
    host = svc.resolve_host_by_token(db, token)
    if host is None or str(host.id) != str(host_id):
        raise HTTPException(status_code=401, detail="invalid or missing CLI host token")
    return host


# ── bodies ───────────────────────────────────────────────────────────────────

class PairingCodeCreate(BaseModel):
    name: Optional[str] = Field(None, max_length=120)


class PairRequest(BaseModel):
    code: str = Field(..., min_length=4, max_length=32)
    name: Optional[str] = Field(None, max_length=120)
    capabilities: Optional[Dict[str, Any]] = None


class RunningSession(BaseModel):
    task_id: int
    session_id: Optional[str] = None
    attempt: Optional[int] = None


class HeartbeatRequest(BaseModel):
    capabilities: Optional[Dict[str, Any]] = None
    running: List[RunningSession] = Field(default_factory=list)


class ClaimRequest(BaseModel):
    limit: int = Field(1, ge=1, le=svc.MAX_CLAIM_LIMIT)


class EventsRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(default_factory=list)


class TerminalRequest(BaseModel):
    """PRD-239 S7: where the Canvas terminal should open — a ticket's real
    directory, or an agent's working directory; neither = the host's default."""
    task_id: Optional[int] = None
    cwd: Optional[str] = Field(None, max_length=1024)


class SessionRequest(BaseModel):
    """PRD-239 S7 v2: open (or resume) the Runtime Canvas session with a session
    agent in a conversation — one ticket per chat and agent."""
    agent_id: int
    chat_id: str = Field(..., min_length=1, max_length=128)


class ResultRequest(BaseModel):
    attempt: Optional[int] = None
    status: str = Field("success", pattern="^(success|error|cancelled)$")
    result_text: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    files_touched: List[str] = Field(default_factory=list)
    permission_denials: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None
    exit_reason: Optional[str] = None
    transcript_path: Optional[str] = None


# ── operator surface ─────────────────────────────────────────────────────────

@router.get("")
async def list_cli_hosts(
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    return {"hosts": svc.list_hosts(db, ctx.workspace_id)}


@router.get("/health")
async def cli_host_health(
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    """PRD-235 W3: is a Claude Code host online for this workspace, since when
    was one last seen, and how many CLI tickets are waiting. The board banner
    reads it; the ticket line says the same thing per ticket."""
    return svc.host_health(db, ctx.workspace_id)


@router.get("/workspace-check")
async def workspace_check(
    path: str = Query(..., min_length=1, max_length=1024, description="An agent's working_directory as typed"),
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    """PRD-239 S6: what a cli agent's working directory would mean before it is
    saved — valid, browsable in the Canvas (as which root), and inside the
    paired host's allowed directories. Read-only; nothing is written."""
    return svc.workspace_check(db, ctx.workspace_id, path)


@router.post("/{host_id}/terminal")
async def open_terminal(
    host_id: UUID,
    body: TerminalRequest,
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    """PRD-239 S7: mint a single-use grant for the operator's own shell on the
    paired host (served on the host's loopback for the browser on that machine).
    Nothing runs here; the host opens the shell when the browser connects."""
    host = (
        db.query(CliHost)
        .filter(
            CliHost.id == host_id,
            CliHost.workspace_id == ctx.workspace_id,
            CliHost.status == CliHostStatus.PAIRED.value,
        )
        .first()
    )
    if host is None:
        raise HTTPException(status_code=404, detail="no paired CLI host with that id in this workspace")
    try:
        return svc.mint_terminal_grant(db, host, cwd=body.cwd, task_id=body.task_id)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/sessions")
async def open_session(
    body: SessionRequest,
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    """PRD-239 S7 v2: the Runtime Canvas. Picking a session agent in a chat (or
    reopening it) gets ONE ticket per chat + agent whose Claude Code session the
    Canvas terminal starts or resumes on the operator's paired host. Nothing is
    dispatched: the human drives the session."""
    from core.models.core import Agent
    from services.board_consent import actor_from_user_id
    from services.cli_ticket_lane import is_cli_agent, open_session_ticket

    agent = (
        db.query(Agent)
        .filter(Agent.id == body.agent_id, Agent.workspace_id == ctx.workspace_id)
        .first()
    )
    if agent is None:
        raise HTTPException(status_code=404, detail="no agent with that id in this workspace")
    if not is_cli_agent(db, agent.id):
        raise HTTPException(status_code=422, detail=f"{agent.name} is not a session agent (runtime: cli)")
    host = svc.newest_online_host(db, ctx.workspace_id)
    if host is None:
        raise HTTPException(status_code=409, detail="no CLI host is online — start it with `make cli-host` and try again")
    task, created = open_session_ticket(
        db, workspace_id=ctx.workspace_id, agent=agent, chat_id=body.chat_id, host=host,
        actor=actor_from_user_id(getattr(ctx.user, "id", None)),
    )
    ref = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
    return {
        "task_id": task.id,
        "host_id": str(host.id),
        "created": created,
        "status": task.status,
        "agent_name": agent.name,
        "cwd": ref.get("cwd"),
        "explorer_root": ref.get("explorer_root"),
    }


@router.post("/pairing-codes")
async def create_pairing_code(
    body: PairingCodeCreate,
    ctx: RequestContext = Depends(_require_operator),
    db: Session = Depends(get_db),
):
    """Issue a one-time pairing code. Shown ONCE; expires in ten minutes."""
    host, code, expires = svc.create_pairing_code(db, ctx.workspace_id, body.name)
    return {
        "host_id": str(host.id),
        "code": code,
        "expires_at": expires.isoformat(),
        "pair_command": f"make cli-host PAIR={code}",
    }


# ── host surface ─────────────────────────────────────────────────────────────

@router.post("/pair")
async def pair(body: PairRequest, db: Session = Depends(get_db)):
    """Exchange a pairing code for a host token (returned exactly once)."""
    _require_cli_runtime()
    paired = svc.pair_host(db, body.code, body.name, body.capabilities)
    if paired is None:
        raise HTTPException(status_code=401, detail="invalid or expired pairing code")
    host, token = paired
    return {
        "host_id": str(host.id),
        "workspace_id": str(host.workspace_id),
        "token": token,
        "token_header": HOST_TOKEN_HEADER,
    }


@router.post("/{host_id}/heartbeat")
async def heartbeat(
    body: HeartbeatRequest,
    host: CliHost = Depends(require_cli_host),
    db: Session = Depends(get_db),
):
    running = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in body.running]
    out = svc.record_heartbeat(db, host, body.capabilities, running)
    out.update(svc.contract_fields())  # PRD-235 W3: the host restarts itself when this moves
    return out


@router.post("/{host_id}/claim")
async def claim(
    body: ClaimRequest,
    host: CliHost = Depends(require_cli_host),
    db: Session = Depends(get_db),
):
    return svc.claim_for_host(db, host, body.limit)


@router.post("/{host_id}/tasks/{task_id}/events")
async def events(
    task_id: int,
    body: EventsRequest,
    host: CliHost = Depends(require_cli_host),
    db: Session = Depends(get_db),
):
    try:
        return svc.record_events(db, host, task_id, body.events)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/{host_id}/tasks/{task_id}/result")
async def result(
    task_id: int,
    body: ResultRequest,
    host: CliHost = Depends(require_cli_host),
    db: Session = Depends(get_db),
):
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    try:
        return await svc.apply_result(db, host, task_id, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

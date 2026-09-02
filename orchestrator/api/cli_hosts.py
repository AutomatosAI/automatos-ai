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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from core.models.cli_hosts import CliHost
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
    return svc.record_heartbeat(db, host, body.capabilities, running)


@router.post("/{host_id}/claim")
async def claim(
    body: ClaimRequest,
    host: CliHost = Depends(require_cli_host),
    db: Session = Depends(get_db),
):
    return {"tasks": svc.claim_for_host(db, host, body.limit)}


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

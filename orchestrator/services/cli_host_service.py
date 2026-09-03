"""PRD-234 S1a — the CLI host contract: pairing, claim, events, results.

The board is the queue. A paired host claims the ``assigned`` tickets of
``runtime: cli`` agents through the SAME ``FOR UPDATE SKIP LOCKED`` claim the
dispatcher uses (with a runtime filter), renews the lease with every event
batch, and posts one idempotent result per attempt that lands through the
board's single completion writer — so a session result is indistinguishable
from an API run's on the board, in Reports and in notifications.

Security posture (PRD-234 §Design 6): a host proves itself with a token it
received exactly once, in exchange for a one-time pairing code the operator
read from the UI. Only the SHA-256 of either secret is ever stored. Tokens are
compared by digest lookup + constant-time compare. Everything here is
workspace-scoped through the host row; a host never sees another workspace.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from core.cli_runtime import (
    CONFIG_ALLOWED_TOOLS_KEY,
    CONFIG_MODEL_KEY,
    CONFIG_PROVIDER_KEY,
    CONFIG_WORKING_DIRECTORY_KEY,
    RUNTIME_CLI,
)
from core.models.cli_hosts import CliHost, CliHostStatus
from core.models.core import Agent, BoardTask
from services.board_dispatcher import claim_tasks, renew_lease
from services.board_events import notify_board_event

logger = logging.getLogger(__name__)

PAIRING_CODE_TTL_SECONDS = 600
HOST_TOKEN_BYTES = 32
MAX_CLAIM_LIMIT = 50
# No 0/O/1/I — a code is read off a screen and typed once.
_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def hash_secret(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def normalize_pairing_code(code: Optional[str]) -> str:
    return (code or "").strip().upper().replace(" ", "")


def _new_pairing_code() -> str:
    raw = "".join(secrets.choice(_PAIRING_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


# ── pairing ─────────────────────────────────────────────────────────────────

def create_pairing_code(
    db: Session, workspace_id: Any, name: Optional[str] = None
) -> Tuple[CliHost, str, datetime]:
    """Issue a one-time pairing code (returned in clear ONCE) as a pending host row."""
    code = _new_pairing_code()
    expires = _now() + timedelta(seconds=PAIRING_CODE_TTL_SECONDS)
    host = CliHost(
        workspace_id=workspace_id,
        name=(name or "cli-host")[:120],
        status=CliHostStatus.PENDING.value,
        pairing_code_hash=hash_secret(normalize_pairing_code(code)),
        pairing_expires_at=expires,
    )
    db.add(host)
    db.commit()
    db.refresh(host)
    return host, code, expires


def pair_host(
    db: Session,
    code: str,
    name: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[CliHost, str]]:
    """Exchange a valid, unexpired pairing code for a host token (returned ONCE).

    Returns ``None`` for an unknown, used or expired code — the caller answers
    401 and says nothing more (the code is the only secret at this point).
    """
    norm = normalize_pairing_code(code)
    if not norm:
        return None
    host = (
        db.query(CliHost)
        .filter(
            CliHost.pairing_code_hash == hash_secret(norm),
            CliHost.status == CliHostStatus.PENDING.value,
        )
        .first()
    )
    if host is None:
        return None
    expires = host.pairing_expires_at
    if expires is not None:
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < _now():
            return None
    token = secrets.token_urlsafe(HOST_TOKEN_BYTES)
    host.token_hash = hash_secret(token)
    host.pairing_code_hash = None
    host.pairing_expires_at = None
    host.status = CliHostStatus.PAIRED.value
    host.paired_at = _now()
    host.last_seen_at = host.paired_at
    if name:
        host.name = name[:120]
    if capabilities is not None:
        host.capabilities = dict(capabilities)
    db.commit()
    db.refresh(host)
    logger.info("[cli-host] paired host %s (%s) in workspace %s", host.id, host.name, host.workspace_id)
    return host, token


def resolve_host_by_token(db: Session, token: Optional[str]) -> Optional[CliHost]:
    """The PAIRED host holding this token, or ``None``. Digest lookup + constant-time compare."""
    if not token:
        return None
    digest = hash_secret(token)
    host = (
        db.query(CliHost)
        .filter(CliHost.token_hash == digest, CliHost.status == CliHostStatus.PAIRED.value)
        .first()
    )
    if host is None or not hmac.compare_digest(host.token_hash or "", digest):
        return None
    return host


def revoke_host(db: Session, host: CliHost) -> None:
    host.status = CliHostStatus.REVOKED.value
    host.token_hash = None
    host.revoked_at = _now()
    db.commit()


def list_hosts(db: Session, workspace_id: Any) -> List[Dict[str, Any]]:
    rows = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id)
        .order_by(CliHost.created_at.desc())
        .all()
    )
    return [h.to_dict() for h in rows]


# ── heartbeat + reconciliation ───────────────────────────────────────────────

def record_heartbeat(
    db: Session,
    host: CliHost,
    capabilities: Optional[Dict[str, Any]],
    running: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Update presence + capabilities and reconcile the sessions the host says it runs.

    * a session on an ``in_progress`` ticket this host owns → lease kept alive;
    * a session on an ``assigned`` ticket this host owns (the sweeper requeued it
      while the host was away) → re-attached, never re-dispatched (PRD-234 §B6);
    * anything else (terminal, cancelled, not ours) → reported ``stale`` so the
      host stops that process.
    """
    lease_seconds = config.BOARD_DISPATCH_LEASE_SECONDS
    host.last_seen_at = _now()
    if capabilities is not None:
        host.capabilities = dict(capabilities)

    reattached: List[int] = []
    stale: List[int] = []
    for item in running or []:
        task_id = item.get("task_id")
        session_id = item.get("session_id")
        task = (
            db.query(BoardTask)
            .filter(BoardTask.id == task_id, BoardTask.workspace_id == host.workspace_id)
            .first()
        )
        if task is None:
            stale.append(task_id)
            continue
        ref = dict(task.runtime_ref or {})
        ours = ref.get("host_id") == str(host.id) and (
            session_id is None or ref.get("session_id") == session_id
        )
        if task.status == "in_progress" and ours:
            task.lease_until = _now() + timedelta(seconds=lease_seconds)
            continue
        if task.status == "assigned" and ours:
            task.status = "in_progress"
            task.lease_until = _now() + timedelta(seconds=lease_seconds)
            ref["reattached_at"] = _iso(_now())
            task.runtime_ref = ref  # rebuild, never mutate in place (JSONB)
            reattached.append(task.id)
            notify_board_event(
                db, workspace_id=host.workspace_id, task_id=task.id,
                status="in_progress", event="task_claimed",
            )
            continue
        stale.append(task.id)
    db.commit()
    return {"reattached": reattached, "stale": stale, "server_time": _iso(_now())}


# ── claim ────────────────────────────────────────────────────────────────────

def _blocked_pending_approval(db: Session, task: BoardTask) -> bool:
    """The board's PRD-181 approval gate, applied at host claim exactly as at API
    launch. Imported lazily: ``api.board_tasks`` imports this package's siblings."""
    from api.board_tasks import _board_task_blocked_pending_approval

    return _board_task_blocked_pending_approval(
        db, task.id, task.assigned_agent_id, str(task.workspace_id)
    )


def _ticket_prompt(task: BoardTask) -> str:
    prompt = task.raw_prompt or task.description or task.title or ""
    if task.review_feedback:
        # Same redo fold-in as the dispatcher (Q44); consumed for this attempt only.
        prompt = (
            f"{prompt}\n\n## Reviewer feedback on your previous attempt\n"
            f"{task.review_feedback}\n\nAddress this feedback in your redo."
        )
        task.review_feedback = None
    return prompt


def claim_for_host(db: Session, host: CliHost, limit: int = 1) -> Dict[str, Any]:
    """Claim up to ``limit`` ``cli`` tickets of this host's workspace for it.

    The claim is the dispatcher's exactly-once statement with ``runtime='cli'``
    and the workspace filter. Each claimed ticket gets a pre-assigned session id
    in ``runtime_ref`` so the host can start ``claude --session-id <id>`` and the
    transcript path is known up front.

    Returns ``{"tasks": [...], "parked": [...]}``: ``parked`` names the tickets the
    board's approval gate held back at claim time (status ``blocked`` with the
    grant in the reason) so the host can SAY so instead of polling in silence —
    the operator approves them in the Command Centre and they come back.
    """
    from uuid import uuid4

    limit = max(1, min(int(limit or 1), MAX_CLAIM_LIMIT))
    claimed = claim_tasks(
        db,
        worker_id=f"cli-host:{host.id}",
        limit=limit,
        lease_seconds=config.BOARD_DISPATCH_LEASE_SECONDS,
        max_slots_per_agent=None,
        runtime=RUNTIME_CLI,
        workspace_id=host.workspace_id,
    )
    out: List[Dict[str, Any]] = []
    parked: List[Dict[str, Any]] = []
    for task in claimed:
        if _blocked_pending_approval(db, task):
            db.refresh(task)
            parked.append({"task_id": task.id, "title": task.title, "reason": task.blocked_reason})
            continue  # parked ``blocked`` by the gate; the answered-resume loop returns it
        agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
        cfg = (getattr(agent, "configuration", None) if agent else None) or {}
        session_id = str(uuid4())
        ref = {
            "runtime": RUNTIME_CLI,
            "provider": cfg.get(CONFIG_PROVIDER_KEY),
            "model": cfg.get(CONFIG_MODEL_KEY),
            "host_id": str(host.id),
            "session_id": session_id,
            "attempt": int(task.attempts or 0),
            "claimed_at": _iso(_now()),
            "cwd": cfg.get(CONFIG_WORKING_DIRECTORY_KEY),
        }
        task.runtime_ref = ref
        out.append(
            {
                "task_id": task.id,
                "workspace_id": str(task.workspace_id),
                "agent_id": task.assigned_agent_id,
                "agent_name": getattr(agent, "name", None),
                "title": task.title,
                "prompt": _ticket_prompt(task),
                "review_mode": task.review_mode or "auto",
                "attachment_ids": task.attachment_ids or [],
                "provider": ref["provider"],
                "model": ref["model"],
                "allowed_tools": cfg.get(CONFIG_ALLOWED_TOOLS_KEY),
                "cwd": ref["cwd"],
                "session_id": session_id,
                "attempt": ref["attempt"],
                "lease_seconds": config.BOARD_DISPATCH_LEASE_SECONDS,
            }
        )
    db.commit()
    return {"tasks": out, "parked": parked}


# ── events + results ─────────────────────────────────────────────────────────

def _owned_task(db: Session, host: CliHost, task_id: int) -> BoardTask:
    task = (
        db.query(BoardTask)
        .filter(BoardTask.id == task_id, BoardTask.workspace_id == host.workspace_id)
        .first()
    )
    if task is None:
        raise LookupError(f"task {task_id} not found in this host's workspace")
    ref = task.runtime_ref or {}
    if ref.get("host_id") != str(host.id):
        raise PermissionError(f"task {task_id} is not claimed by this host")
    return task


def record_events(
    db: Session, host: CliHost, task_id: int, events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Absorb a batch of hook events: renew the lease, keep a compact live summary
    in ``runtime_ref`` (live tool, transcript path, counts), and hand back control
    (``cancel``) the host must act on. Events are not persisted individually here
    — S2 maps them to board events and the fleet."""
    task = _owned_task(db, host, task_id)
    renewed = renew_lease(db, task_id, lease_seconds=config.BOARD_DISPATCH_LEASE_SECONDS)
    ref = dict(task.runtime_ref or {})
    events = events or []
    ref["events_seen"] = int(ref.get("events_seen") or 0) + len(events)
    ref["last_event_at"] = _iso(_now())
    for ev in events:
        if not isinstance(ev, dict):
            continue
        name = ev.get("event") or ev.get("hook_event_name")
        if name:
            ref["last_event"] = name
        if name == "PreToolUse" and ev.get("tool_name"):
            ref["live_tool"] = ev["tool_name"]
            # PRD-234 S2: the ticket's live log — tool + what it was about, bounded.
            entry: Dict[str, Any] = {"at": _iso(_now()), "tool": str(ev["tool_name"])[:60]}
            if ev.get("subject"):
                entry["subject"] = str(ev["subject"])[:200]
            ref["recent_tools"] = (list(ref.get("recent_tools") or []) + [entry])[-RECENT_TOOLS_KEPT:]
        elif name in ("PostToolUse", "Stop", "SessionEnd"):
            ref.pop("live_tool", None)
        if ev.get("session_id"):
            ref["cli_session_id"] = ev["session_id"]
        if ev.get("transcript_path"):
            ref["transcript_path"] = ev["transcript_path"]
    task.runtime_ref = ref
    db.commit()
    control: List[str] = []
    if task.status == "cancelled" or ref.get("cancel_requested_at"):
        control.append("cancel")
    return {"status": task.status, "lease_renewed": bool(renewed), "control": control}


def _tokens_used(usage: Dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    try:
        return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return 0


RECENT_TOOLS_KEPT = 30

# register() refuses 'report' — ReportService owns that type; a session's .md is a document.
_DELIVERABLE_TYPE_OVERRIDES = {"report": "document"}


PROJECTS_PREFIX = "projects"


def _clean_relative(rel: str) -> Optional[str]:
    rel = rel.strip("/")
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")):
        return None
    return rel


def workspace_relative_path(host_path: str, workspace_id: str, projects_dir: Optional[str] = None) -> Optional[str]:
    """A session's file path on the host → the worker's view of it, or ``None``.

    * ``…/<AUTOMATOS_WORKSPACE_DIR>/<workspace_id>/sessions/68/hello.py`` →
      ``sessions/68/hello.py`` — the workspace-id segment is the anchor both
      sides share (the worker's layout is ``<root>/<workspace_id>/<relative>``).
    * ``<LOCAL_PROJECTS_DIR>/repo/app.py`` → ``projects/repo/app.py`` — the owner's
      projects folder is mounted read-only into the worker under ``projects/``.

    The host's absolute path means nothing inside this container. ``None`` when
    the file is elsewhere — it then stays a reference in ``runtime_ref.files_touched``.
    """
    path = str(host_path)
    marker = f"/{workspace_id}/"
    idx = path.find(marker)
    if idx >= 0:
        return _clean_relative(path[idx + len(marker):])
    root = (projects_dir or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        rel = _clean_relative(path[len(root):])
        return f"{PROJECTS_PREFIX}/{rel}" if rel else None
    return None


def _register_session_deliverables(
    db: Session, task: BoardTask, files: Iterable[str], *, agent_id: Optional[int],
    agent_name: Optional[str], session_id: Optional[str],
) -> List[Dict[str, Any]]:
    """PRD-234 S2: every file a session wrote under the workspace volume becomes a
    deliverable of the ticket (``source_type='task'``), through the same
    ``DeliverableService.register`` mission promotion uses (#611). Metadata only:
    the bytes already sit where the worker serves them. Fail-soft per file."""
    from services.deliverable_service import (
        AGENT_REGISTERABLE_ARTIFACT_TYPES, DeliverableService, _infer_artifact_type,
    )
    workspace_id = str(task.workspace_id)
    volume = Path(config.WORKSPACE_VOLUME_PATH) / workspace_id
    service = DeliverableService(db, workspace_id)
    registered: List[Dict[str, Any]] = []
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    for host_path in files:
        rel = workspace_relative_path(str(host_path), workspace_id, projects_dir)
        if rel is None:
            continue
        inferred = _infer_artifact_type(rel)
        if inferred not in AGENT_REGISTERABLE_ARTIFACT_TYPES:
            continue
        if rel.startswith(PROJECTS_PREFIX + "/"):
            # The projects folder is mounted into the worker, not here: register
            # without a size; the worker serves the bytes behind preview_url.
            size = None
        else:
            full = volume / rel
            try:
                size = full.stat().st_size if full.is_file() else None
            except OSError:
                size = None
            if size is None:
                continue  # not visible from this container → reference only
        artifact_type = _DELIVERABLE_TYPE_OVERRIDES.get(inferred, inferred)
        try:
            res = service.register(
                file_path=rel, source_type="task", source_id=str(task.id),
                agent_id=agent_id, agent_name=agent_name, artifact_type=artifact_type,
                file_size_bytes=size,
                summary=f"Written by a Claude Code session for ticket #{task.id}",
                extra={"task_id": task.id, "session_id": session_id, "host_path": str(host_path),
                       "runtime": RUNTIME_CLI},
            )
        except Exception as exc:  # noqa: BLE001 — one bad file must not lose the result
            logger.warning("[CliHost] deliverable registration failed for %s: %s", rel, exc)
            continue
        if res.get("success"):
            registered.append({"id": res.get("deliverable_id"), "file_path": rel,
                               "title": rel.rsplit("/", 1)[-1], "artifact_type": artifact_type})
    return registered


MAX_DENIALS_KEPT = 20


def _denial_summary(denial: Any) -> Dict[str, Any]:
    """One denial as the ticket shows it: tool, stage, reason, and the command or
    path it was about (never the whole tool input)."""
    if not isinstance(denial, dict):
        return {"tool": "?", "reason": str(denial)[:300]}
    raw_input = denial.get("input") if isinstance(denial.get("input"), dict) else {}
    subject = raw_input.get("command") or raw_input.get("file_path") or raw_input.get("path")
    out: Dict[str, Any] = {
        "tool": str(denial.get("tool") or "?")[:60],
        "stage": str(denial.get("stage") or "")[:40],
        "reason": str(denial.get("reason") or "")[:300],
    }
    if subject:
        out["subject"] = str(subject)[:300]
    return out


async def apply_result(
    db: Session, host: CliHost, task_id: int, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Land a session's terminal result through the board's ONE completion writer.

    Idempotent per ``(task, attempt)``: a duplicate POST, a stale attempt, or a
    task that already left ``in_progress`` (cancelled, requeued, finished) is a
    no-op that says so. Any permission denial during the turn forces ``review``
    — "couldn't run the tests" must never read as ``done`` (PRD-234 §C1).
    """
    from api.board_tasks import finalize_board_task_run

    task = _owned_task(db, host, task_id)
    ref = dict(task.runtime_ref or {})
    attempt = payload.get("attempt")
    if attempt is not None and ref.get("attempt") is not None and int(attempt) != int(ref["attempt"]):
        return {"applied": False, "reason": "stale attempt", "status": task.status}
    if task.status != "in_progress":
        return {"applied": False, "reason": f"task is {task.status}", "status": task.status}

    status = str(payload.get("status") or "success").lower()
    denials = payload.get("permission_denials") or []
    usage = payload.get("usage") or {}
    files = payload.get("files_touched") or []
    exec_result: Dict[str, Any] = {
        "status": "error" if status == "error" else ("cancelled" if status == "cancelled" else "success"),
        "result": payload.get("result_text") or "",
        "error": payload.get("error"),
        "tokens_used": _tokens_used(usage) if isinstance(usage, dict) else 0,
        "usage": usage,
        "runtime": RUNTIME_CLI,
        "billing_source": "subscription",
        "session_id": ref.get("session_id"),
        "files_touched": files,
        "permission_denials": denials,
    }
    ref.update(
        {
            "finished_at": _iso(_now()),
            "exit_reason": payload.get("exit_reason") or status,
            "files_touched": files,
            "usage": usage,
            "denials": len(denials),
            # The reasons, not just the count: a ticket in review must say WHY
            # ("'python3 hello.py' is outside this ticket's Bash allowlist").
            "permission_denials": [_denial_summary(d) for d in denials[:MAX_DENIALS_KEPT]],
        }
    )
    if payload.get("transcript_path"):
        ref["transcript_path"] = payload["transcript_path"]

    # PRD-234 S2: files under the workspace volume → the ticket's deliverables;
    # the session facts ride exec_result so the task report can show them.
    agent_row = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first() if task.assigned_agent_id else None
    deliverables = _register_session_deliverables(
        db, task, files, agent_id=task.assigned_agent_id,
        agent_name=getattr(agent_row, "name", None), session_id=ref.get("session_id"),
    )
    ref["deliverables"] = deliverables
    exec_result["deliverables"] = deliverables
    exec_result["session"] = {
        "session_id": ref.get("session_id"),
        "host_id": str(host.id),
        "provider": ref.get("provider"),
        "model": (usage.get("model") if isinstance(usage, dict) else None) or ref.get("model"),
        "cwd": ref.get("cwd"),
        "exit_reason": ref.get("exit_reason"),
        "transcript_path": ref.get("transcript_path"),
        "recent_tools": list(ref.get("recent_tools") or []),
        "permission_denials": list(ref.get("permission_denials") or []),
    }
    task.runtime_ref = ref
    db.commit()

    terminal = await finalize_board_task_run(
        db,
        task_id=task.id,
        workspace_id=str(task.workspace_id),
        agent_id=task.assigned_agent_id,
        exec_result=exec_result,
        review_mode=task.review_mode or "auto",
        force_review=bool(denials),
    )
    return {"applied": terminal is not None, "status": terminal or task.status}

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
    CONFIG_WORKTREE_KEY,
    CONFIG_MODEL_KEY,
    CONFIG_PROVIDER_KEY,
    CONFIG_WORKING_DIRECTORY_KEY,
    RUNTIME_CLI,
)
from core.llm.usage_context import LANE_BOARD_TASK, LANE_SESSION
from core.models.cli_hosts import CliHost, CliHostStatus
from core.models.core import Agent, BoardTask
from services.board_dispatcher import claim_tasks, renew_lease
from services.board_events import notify_board_event
from services.cli_ticket_lane import SESSION_MODE_TERMINAL

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


# ── PRD-235 W3: the host restarts itself when this contract moves ────────────
# The host is a process on the operator's machine loaded from a checkout; the
# app is rebuilt from a branch. Every heartbeat answer carries a fingerprint of
# the host↔backend contract (these modules) and the host version this backend
# was built for. A host that sees the fingerprint change drains and exits; its
# service manager brings it back on the new code. Bump EXPECTED_CLI_HOST_VERSION
# whenever the wire contract changes so a stale checkout is told, not surprised.
EXPECTED_CLI_HOST_VERSION = "0.6.0"  # 2026-09-09: a no-folder ticket runs in <deliverables root>/sessions/<ticket>; --default-root (#722). 0.5.0: results and TerminalClosed carry the turn's token usage

_CONTRACT_MODULES = ("api/cli_hosts.py", "services/cli_host_service.py", "core/cli_runtime.py")


def _compute_host_contract() -> str:
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for rel in _CONTRACT_MODULES:
        try:
            h.update(rel.encode())
            h.update((root / rel).read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:16]


HOST_CONTRACT = _compute_host_contract()


def contract_fields() -> Dict[str, Any]:
    """Appended to every heartbeat answer (and the health view)."""
    return {"host_contract": HOST_CONTRACT, "expected_host_version": EXPECTED_CLI_HOST_VERSION}


def host_health(db: Session, workspace_id: Any) -> Dict[str, Any]:
    """The executor view the board needs: is any paired host online, since when
    was it last seen, and how many Claude Code tickets are waiting for it."""
    from core.cli_runtime import RUNTIME_CLI
    from core.models.core import Agent

    hosts = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id, CliHost.status == CliHostStatus.PAIRED.value)
        .all()
    )
    online = [h for h in hosts if h.is_online()]
    last_seen = max((h.last_seen_at for h in hosts if h.last_seen_at), default=None)
    cli_agent_ids = [
        a.id for a in db.query(Agent).filter(Agent.workspace_id == workspace_id).all()
        if (a.configuration or {}).get("runtime") == RUNTIME_CLI
    ]
    waiting = 0
    if cli_agent_ids:
        waiting = (
            db.query(BoardTask)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.assigned_agent_id.in_(cli_agent_ids),
                BoardTask.status == "assigned",
            )
            .count()
        )
    return {
        "online": bool(online),
        "paired_hosts": len(hosts),
        "online_hosts": [h.to_dict() for h in online],
        "last_seen_at": last_seen.isoformat() if last_seen else None,
        "cli_agents": len(cli_agent_ids),
        "waiting_tickets": waiting,
        **contract_fields(),
    }


def host_allow_dirs(db: Session, workspace_id: Any) -> List[str]:
    """The directories the workspace's paired hosts announced they may run in
    (``capabilities.allow_dirs``, PRD-239 S6). Empty when no host reported any."""
    hosts = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id, CliHost.status == CliHostStatus.PAIRED.value)
        .all()
    )
    roots: List[str] = []
    for host in hosts:
        caps = host.capabilities if isinstance(host.capabilities, dict) else {}
        for raw in caps.get("allow_dirs") or []:
            if isinstance(raw, str) and raw and raw not in roots:
                roots.append(raw)
    return sorted(roots)


def _inside(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/")


def workspace_check(db: Session, workspace_id: Any, path: str) -> Dict[str, Any]:
    """PRD-239 S6: what a cli agent's ``working_directory`` would mean, before it
    is saved — valid or not, browsable in the Canvas as which root, and inside
    the paired host's allowed directories or not (``None`` when no host has
    announced them)."""
    from core.cli_runtime import validate_working_directory

    path = (path or "").strip()
    errors = validate_working_directory(path)
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    root = None if errors else browsable_root(path, str(workspace_id), projects_dir)
    roots = host_allow_dirs(db, workspace_id)
    allowed: Optional[bool] = None
    if roots and not errors:
        allowed = any(_inside(path, r) for r in roots)
    return {
        "path": path,
        "valid": not errors,
        "errors": errors,
        "explorer_root": root,
        "browsable": root is not None,
        "allowed": allowed,
        "allowed_roots": roots,
        "projects_dir": projects_dir,
    }


# ── PRD-239 S7: terminal grants (the Canvas terminal) ─────────────────────────
# The operator asks for a terminal; the backend mints a single-use, short-lived
# grant and hands the browser the host's loopback URL. The host learns the grant
# on its next heartbeat. Grants live in Redis (shared across workers) with an
# in-process fallback for a single-worker local stack without Redis.

TERMINAL_GRANT_TTL_SECONDS = 120
_TERMINAL_GRANTS: Dict[str, List[Dict[str, Any]]] = {}


def _terminal_grants_key(host_id: Any) -> str:
    return f"cli-host:terminal-grants:{host_id}"


def _redis():
    try:
        from core.redis.client import get_redis_client

        return get_redis_client()
    except Exception:  # noqa: BLE001 — no Redis is a supported local shape
        return None


def push_terminal_grant(host_id: Any, grant: Dict[str, Any]) -> None:
    import json as _json

    client = _redis()
    if client is not None:
        try:
            key = _terminal_grants_key(host_id)
            client.rpush(key, _json.dumps(grant))
            client.expire(key, TERMINAL_GRANT_TTL_SECONDS)
            return
        except Exception:  # noqa: BLE001 — fall back to the process store
            logger.debug("[cli-host] terminal grant not stored in Redis — using the process store", exc_info=True)
    _TERMINAL_GRANTS.setdefault(str(host_id), []).append(grant)


def pop_terminal_grants(host_id: Any) -> List[Dict[str, Any]]:
    """Every grant minted for this host since its last heartbeat, unexpired."""
    import json as _json
    import time as _time

    grants: List[Dict[str, Any]] = []
    client = _redis()
    if client is not None:
        try:
            key = _terminal_grants_key(host_id)
            raw = client.lrange(key, 0, -1)
            client.delete(key)
            for item in raw or []:
                text = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
                parsed = _json.loads(text)
                if isinstance(parsed, dict):
                    grants.append(parsed)
        except Exception:  # noqa: BLE001
            logger.debug("[cli-host] terminal grants not read from Redis", exc_info=True)
    grants += _TERMINAL_GRANTS.pop(str(host_id), [])
    now = _time.time()
    fresh = []
    for grant in grants:
        try:
            if float(grant.get("expires_at") or 0) > now:
                fresh.append(grant)
        except (TypeError, ValueError):
            continue
    return fresh


def mint_terminal_grant(
    db: Session, host: CliHost, *, cwd: Optional[str] = None, task_id: Optional[int] = None, shell: bool = False,
) -> Dict[str, Any]:
    """A grant for one terminal on ``host``: in a ticket's real directory
    (``task_id``), in an agent's working directory (``cwd``, checked against
    the host's allowed directories), or in the host's default folder."""
    import time as _time

    from core.cli_runtime import validate_working_directory

    caps = host.capabilities if isinstance(host.capabilities, dict) else {}
    port = caps.get("terminal_port")
    if not port:
        raise LookupError("this host does not serve a terminal — update the host to 0.3.0+ and restart it")
    resolved_cwd: Optional[str] = None
    launch: Optional[Dict[str, Any]] = None
    if task_id is not None:
        task = (
            db.query(BoardTask)
            .filter(BoardTask.id == int(task_id), BoardTask.workspace_id == host.workspace_id)
            .first()
        )
        if task is None:
            raise LookupError(f"task {task_id} not found in this workspace")
        ref = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
        resolved_cwd = str(ref["cwd"]) if ref.get("cwd") else None
        # ``shell`` = the operator wants a plain shell in the ticket's folder (an
        # extra terminal tab beside the session), not the session itself.
        launch = None if shell else _terminal_launch_for(db, task, ref, host)
    elif cwd:
        errors = validate_working_directory(cwd)
        if errors:
            raise ValueError(errors[0])
        clean = cwd.strip()
        roots = host_allow_dirs(db, host.workspace_id)
        if roots and not any(_inside(clean, r) for r in roots):
            raise PermissionError(f"{clean} is outside the directories this host may run in ({', '.join(roots)})")
        resolved_cwd = clean
    token = secrets.token_urlsafe(24)
    expires_at = _time.time() + TERMINAL_GRANT_TTL_SECONDS
    grant = {
        "token": token,
        "cwd": resolved_cwd,
        "task_id": str(task_id) if task_id is not None else None,
        "expires_at": expires_at,
        # PRD-239 S7 v2: what the host runs in the PTY (None = the login shell).
        "launch": launch,
    }
    push_terminal_grant(host.id, grant)
    db.commit()
    return {
        "launch": _launch_summary(launch),
        "token": token,
        "port": int(port),
        "ws_url": f"ws://127.0.0.1:{int(port)}/terminal?token={token}",
        "cwd": resolved_cwd,
        "task_id": grant["task_id"],
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
    }


def list_hosts(db: Session, workspace_id: Any) -> List[Dict[str, Any]]:
    rows = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id)
        .order_by(CliHost.created_at.desc())
        .all()
    )
    return [h.to_dict() for h in rows]


# ── heartbeat + reconciliation ───────────────────────────────────────────────

# ── Settings → Session mode (PRD-239 S6c) ───────────────────────────────────
SESSION_MODE_SETTINGS_KEY = "session_mode"
DEFAULT_FOLDER_PROJECTS = "projects"   # tickets for agents without a folder run in LOCAL_PROJECTS_DIR
DEFAULT_FOLDER_SESSIONS = "sessions"   # … in a fresh ./workspaces/<ws>/sessions/<ticket>
DEFAULT_FOLDER_CHOICES = (DEFAULT_FOLDER_PROJECTS, DEFAULT_FOLDER_SESSIONS)


def _workspace_row(db: Session, workspace_id: Any):
    from core.models.workspaces import Workspace

    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


def session_mode_settings(db: Session, workspace_id: Any) -> Dict[str, Any]:
    """What the operator sees and sets on Settings → Session mode: where tickets
    run when their agent names no folder, plus the projects folder as the stack
    was started with (a Docker mount — set in .env, read here) and how it is
    mounted. The default is the projects folder when one is configured — most
    tickets are "fix a bug in a repo" or "start a new repo" — else a fresh
    sessions folder per ticket."""
    ws = _workspace_row(db, workspace_id)
    stored = ((getattr(ws, "settings", None) or {}).get(SESSION_MODE_SETTINGS_KEY) or {}) if ws is not None else {}
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    choice = stored.get("default_folder")
    if choice not in DEFAULT_FOLDER_CHOICES:
        choice = DEFAULT_FOLDER_PROJECTS if projects_dir else DEFAULT_FOLDER_SESSIONS
    return {
        "default_folder": choice,
        "default_folder_explicit": stored.get("default_folder") in DEFAULT_FOLDER_CHOICES,
        "local_projects_dir": projects_dir,
        "projects_mount": (getattr(config, "LOCAL_PROJECTS_MOUNT", "") or None),
        # The deliverables root on the host (AUTOMATOS_WORKSPACE_DIR as `make up`
        # exported it) — beside the projects folder in Settings → Session mode.
        "workspace_dir": configured_workspace_dir(),
        "host_allowed_roots": host_allow_dirs(db, workspace_id),
    }


def save_session_mode_settings(db: Session, workspace_id: Any, *, default_folder: str) -> Dict[str, Any]:
    if default_folder not in DEFAULT_FOLDER_CHOICES:
        raise ValueError(f"default_folder must be one of {list(DEFAULT_FOLDER_CHOICES)}")
    ws = _workspace_row(db, workspace_id)
    if ws is None:
        raise LookupError("workspace not found")
    from sqlalchemy.orm.attributes import flag_modified

    current = dict(getattr(ws, "settings", None) or {})
    section = dict(current.get(SESSION_MODE_SETTINGS_KEY) or {})
    ws.settings = {**current, SESSION_MODE_SETTINGS_KEY: {**section, "default_folder": default_folder}}  # rebuild, never mutate (JSONB)
    flag_modified(ws, "settings")
    db.commit()
    return session_mode_settings(db, workspace_id)


def default_session_folder(db: Session, workspace_id: Any) -> Optional[str]:
    """The folder a ticket runs in when its agent names none: the projects
    folder when the workspace says so and one is configured, else ``None`` —
    the host then uses its per-ticket ``sessions/<ticket>`` folder."""
    try:
        settings = session_mode_settings(db, workspace_id)
    except Exception:  # noqa: BLE001 — a settings problem must never block a claim
        logger.warning("[cli-host] session-mode settings unreadable for workspace %s", workspace_id, exc_info=True)
        return None
    if settings["default_folder"] == DEFAULT_FOLDER_PROJECTS and settings["local_projects_dir"]:
        return settings["local_projects_dir"]
    return None


def _host_version_at_least(raw: Any, minimum: Tuple[int, ...]) -> bool:
    try:
        parts = tuple(int(x) for x in str(raw or "").split("."))
    except ValueError:
        return False
    return bool(parts) and parts >= minimum


def _terminal_launch_for(db: Session, task: BoardTask, ref: Dict[str, Any], host: CliHost) -> Optional[Dict[str, Any]]:
    """PRD-239 S7 v2: what the Canvas terminal runs for this ticket — the agent's
    own Claude Code session under the ticket's session id (the host resumes it
    when the transcript exists on that machine, else starts it under that id so
    the next open resumes it). A ticket without a cli session → a plain shell."""
    if ref.get("runtime") != RUNTIME_CLI and ref.get("mode") != SESSION_MODE_TERMINAL:
        return None
    session_id = ref.get("cli_session_id") or ref.get("session_id")
    if not session_id:
        return None
    caps = host.capabilities if isinstance(host.capabilities, dict) else {}
    if not _host_version_at_least(caps.get("host_version"), (0, 4, 0)):
        raise LookupError("this host cannot launch a session in the terminal — update the host to 0.4.0+ and restart it")
    agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first() if task.assigned_agent_id else None
    if ref.get("mode") == SESSION_MODE_TERMINAL and ref.get("host_id") != str(host.id):
        # The session moves with the operator: the host that opens it owns its events.
        task.runtime_ref = {**ref, "host_id": str(host.id)}
    return {
        "kind": "claude",
        "session_id": str(session_id),
        "system_prompt": _session_system_prompt(agent),
        "model": ref.get("model"),
        "agent_name": getattr(agent, "name", None),
    }


def _launch_summary(launch: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the browser may know about the launch (never the prompt)."""
    if not launch:
        return None
    return {"kind": launch.get("kind"), "session_id": launch.get("session_id"), "agent_name": launch.get("agent_name")}


def newest_online_host(db: Session, workspace_id: Any) -> Optional[CliHost]:
    """The paired host that heartbeated most recently — where a Runtime Canvas
    session opens — or None when no host is online."""
    from core.models.cli_hosts import CliHostStatus

    hosts = db.query(CliHost).filter(
        CliHost.workspace_id == workspace_id, CliHost.status == CliHostStatus.PAIRED.value,
    ).all()
    online = [h for h in hosts if h.is_online()]
    if not online:
        return None
    return max(online, key=lambda h: h.last_seen_at or datetime.min.replace(tzinfo=timezone.utc))


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
    return {
        "reattached": reattached,
        "stale": stale,
        "server_time": _iso(_now()),
        # PRD-239 S7: the terminal grants the operator asked for since the last beat.
        "terminal_grants": pop_terminal_grants(host.id),
    }


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
        from services.cli_ticket_lane import NO_HOST_REASON
        if task.blocked_reason == NO_HOST_REASON:
            task.blocked_reason = None  # a host is here now
        agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
        cfg = (getattr(agent, "configuration", None) if agent else None) or {}
        prior = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
        resume_session_id = _resume_session_for(prior, host)
        session_id = str(uuid4())
        ref = {
            "runtime": RUNTIME_CLI,
            "provider": cfg.get(CONFIG_PROVIDER_KEY),
            "model": cfg.get(CONFIG_MODEL_KEY),
            "host_id": str(host.id),
            "session_id": session_id,
            "attempt": int(task.attempts or 0),
            "claimed_at": _iso(_now()),
            # PRD-239 S6c: an agent without a folder runs where the workspace says
            # (the projects folder by default), else the host's sessions/<ticket>.
            "cwd": cfg.get(CONFIG_WORKING_DIRECTORY_KEY) or default_session_folder(db, task.workspace_id),
        }
        if resume_session_id:
            ref["resume_session_id"] = resume_session_id
        ref["explorer_root"] = explorer_root_for(
            task.id, ref["cwd"], task.workspace_id, getattr(config, "LOCAL_PROJECTS_DIR", "") or None,
        )
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
                # PRD-239: worktree per ticket is the agent's choice (default on).
                "worktree": cfg.get(CONFIG_WORKTREE_KEY, True) is not False,
                "session_id": session_id,
                "attempt": ref["attempt"],
                "lease_seconds": config.BOARD_DISPATCH_LEASE_SECONDS,
                # PRD-239 S1: the agent's soul (description, persona, skills),
                # stable per agent — the host appends it to the session prompt.
                "system_prompt": _session_system_prompt(agent),
                # PRD-239: continue the session a lane asked to resume, on the
                # host that ran it (``claude --resume``); None starts a fresh one.
                "resume_session_id": resume_session_id,
            }
        )
    db.commit()
    return {"tasks": out, "parked": parked}


def _session_system_prompt(agent: Optional[Agent]) -> str:
    """Never lets a rendering problem block a claim — the host falls back to
    name + rules when this is empty."""
    if agent is None:
        return ""
    try:
        from services.cli_session_prompt import session_system_prompt

        return session_system_prompt(agent)
    except Exception:  # noqa: BLE001
        logger.warning("[cli-host] session prompt rendering failed for agent %s", getattr(agent, "id", "?"), exc_info=True)
        return ""


def _resume_session_for(prior: Dict[str, Any], host: CliHost) -> Optional[str]:
    """The session id the ticket asked to resume — only when THIS host ran it
    (a Claude Code transcript lives on one machine)."""
    session_id = prior.get("resume_session_id") if isinstance(prior, dict) else None
    if not session_id:
        return None
    if str(prior.get("resume_host_id") or "") != str(host.id):
        return None
    return str(session_id)


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


def _record_session_cwd(ref: Dict[str, Any], task: BoardTask, cwd: str) -> None:
    """The directory the session actually runs in, plus the explorer root that
    follows from it. One writer for SessionStart and the result (PRD-239)."""
    ref["cwd"] = cwd
    ref["explorer_root"] = explorer_root_for(
        task.id, cwd, task.workspace_id, getattr(config, "LOCAL_PROJECTS_DIR", "") or None,
    )


TERMINAL_EVENTS = ("TerminalOpened", "TerminalClosed")


def _terminal_event_name(ev: Any) -> Optional[str]:
    name = (ev.get("hook_event_name") or ev.get("event")) if isinstance(ev, dict) else None
    return name if name in TERMINAL_EVENTS else None


def _record_terminal_events(db: Session, host: CliHost, task_id: int, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PRD-239 S7 v2: the Runtime Canvas attached to / left a ticket. No lease is
    ever set (the sweeper and the claim loop must never touch a session the
    human drives); an interactive session ticket is ``in_progress`` while a
    terminal is attached and ``done`` otherwise. A host-run ticket reopened in
    the terminal keeps its status and only records where the session runs."""
    task = _owned_task(db, host, task_id)
    ref = dict(task.runtime_ref or {})
    interactive = ref.get("mode") == SESSION_MODE_TERMINAL
    status_before = task.status
    for ev in events:
        name = _terminal_event_name(ev)
        if ev.get("session_id"):
            ref["cli_session_id"] = str(ev["session_id"])
        if ev.get("cwd"):
            _record_session_cwd(ref, task, str(ev["cwd"]))
        ref["last_event"] = name
        ref["last_event_at"] = _iso(_now())
        if name == "TerminalOpened":
            ref["terminal_attached_at"] = _iso(_now())
            ref["terminal_resumed"] = bool(ev.get("resumed"))
            ref.pop("terminal_closed_at", None)
            if interactive:
                task.status = "in_progress"
                task.lease_until = None
                task.completed_at = None
        elif name == "TerminalClosed":
            closed_usage = ev.get("usage")
            book_session_usage(
                task, ref, closed_usage if isinstance(closed_usage, dict) else None,
                status="success" if not ev.get("exit_code") else "error",
                request_type=LANE_SESSION,
                execution_id=f"session:{ev.get('session_id') or ref.get('cli_session_id') or task.id}",
            )
            ref["terminal_closed_at"] = _iso(_now())
            ref.pop("terminal_attached_at", None)
            if interactive:
                task.status = "done"
                task.completed_at = _now()
                task.lease_until = None
    task.runtime_ref = ref  # rebuild, never mutate in place (JSONB)
    db.commit()
    if task.status != status_before:
        notify_board_event(
            db, workspace_id=host.workspace_id, task_id=task.id, status=task.status,
            event="task_claimed" if task.status == "in_progress" else "task_completed",
        )
    return {"status": task.status, "lease_renewed": False, "control": {}, "decisions": []}


def record_events(
    db: Session, host: CliHost, task_id: int, events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Absorb a batch of hook events: renew the lease, keep a compact live summary
    in ``runtime_ref`` (live tool, transcript path, counts), and hand back control
    (``cancel``) the host must act on. Events are not persisted individually here
    — S2 maps them to board events and the fleet."""
    events = events or []
    if events and all(_terminal_event_name(ev) for ev in events):
        return _record_terminal_events(db, host, task_id, events)
    task = _owned_task(db, host, task_id)
    renewed = renew_lease(db, task_id, lease_seconds=config.BOARD_DISPATCH_LEASE_SECONDS)
    ref = dict(task.runtime_ref or {})
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
        # PRD-235 W2: the session's effective working directory (SessionStart carries
        # it) — the absolute host path editor deeplinks need; the explorer root follows.
        # PRD-239: it always wins over the configured directory — a git repo runs in
        # a --worktree, and that is where the transcript and the edits live.
        if name == "SessionStart" and ev.get("cwd"):
            _record_session_cwd(ref, task, str(ev["cwd"]))
        if name == "PermissionRequest":
            note_pending_permission(ref, ev)
    task.runtime_ref = ref
    db.commit()
    # PRD-235 W2 S3: the same events light up the Code Canvas panel.
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    canvas: List[Dict[str, Any]] = []
    for ev in events:
        if isinstance(ev, dict):
            canvas.extend(canvas_events_for(task, ref, ev, projects_dir))
    publish_canvas_events(task.workspace_id, canvas)
    control: List[str] = []
    if task.status == "cancelled" or ref.get("cancel_requested_at"):
        control.append("cancel")
    decisions = take_undelivered_decisions(ref)
    if decisions:
        task.runtime_ref = dict(ref)
        db.commit()
    return {"status": task.status, "lease_renewed": bool(renewed), "control": control, "decisions": decisions}


def _session_duration_ms(ref: Dict[str, Any]) -> Optional[int]:
    started = ref.get("claimed_at") or ref.get("terminal_attached_at")
    if not started:
        return None
    try:
        began = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        return max(0, int((_now() - began).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


def book_session_usage(
    task: BoardTask,
    ref: Dict[str, Any],
    usage: Optional[Dict[str, Any]],
    *,
    status: str,
    request_type: str,
    execution_id: str,
    error: Optional[str] = None,
) -> int:
    """A session's tokens reach ``llm_usage`` like any API call (2026-09-09):
    provider ``claude_code``, tier ``subscription``, $0 — so the Analytics page
    shows what the user's own Claude Code plan did beside what the API routes
    cost. Silent when the host reported no usage at all (no transcript)."""
    if not isinstance(usage, dict) or not usage:
        return 0
    from core.llm.usage_tracker import UsageTracker

    return UsageTracker.track_session(
        task.workspace_id,
        cli_provider=str(ref.get("provider") or ""),
        usage=usage,
        agent_id=task.assigned_agent_id,
        execution_id=execution_id,
        request_type=request_type,
        status=status,
        latency_ms=_session_duration_ms(ref),
        error_message=error,
        fallback_model=ref.get("model"),
    )


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
# The explorer root for the deliverables folder itself — the Canvas's own root
# token (frontend `WORKSPACE_ROOT`), so a session rooted there browses everything.
WORKSPACE_ROOT_FOLDER = "."


def configured_workspace_dir() -> Optional[str]:
    """The deliverables root on the host, when the stack was started with an
    ABSOLUTE ``AUTOMATOS_WORKSPACE_DIR`` (what ``make up`` exports). Compose mounts
    that folder as the local workspace's root, so it is the second anchor for
    mapping a session's host paths onto the worker's view. A relative value (a
    plain ``docker compose up`` with the default) means nothing to this process."""
    raw = (getattr(config, "AUTOMATOS_WORKSPACE_DIR", "") or "").strip().rstrip("/")
    return raw if raw.startswith("/") else None


def _clean_relative(rel: str) -> Optional[str]:
    rel = rel.strip("/")
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")):
        return None
    return rel


def workspace_relative_path(
    host_path: str,
    workspace_id: str,
    projects_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
) -> Optional[str]:
    """A session's file path on the host → the worker's view of it, or ``None``.

    * ``…/<workspace_id>/sessions/68/hello.py`` → ``sessions/68/hello.py`` — the
      workspace-id segment is an anchor both sides share (the nested layout:
      ``<volume>/<workspace_id>/<relative>``).
    * ``<AUTOMATOS_WORKSPACE_DIR>/sessions/68/hello.py`` → ``sessions/68/hello.py``
      — the local edition mounts that folder AS the workspace root (no
      workspace-id folder on the host), so the folder itself is the anchor.
    * ``<LOCAL_PROJECTS_DIR>/repo/app.py`` → ``projects/repo/app.py`` — the owner's
      projects folder is mounted read-only into the worker under ``projects/``.

    When the deliverables root sits inside the projects folder (or the other way
    round) the LONGER matching root wins, so ``~/Development/deliverables/reports/x.md``
    is ``reports/x.md``, not ``projects/deliverables/reports/x.md``. ``workspace_dir``
    defaults to the configured ``AUTOMATOS_WORKSPACE_DIR`` when absolute.

    The host's absolute path means nothing inside this container. ``None`` when
    the file is elsewhere — it then stays a reference in ``runtime_ref.files_touched``.
    """
    path = str(host_path)
    marker = f"/{workspace_id}/"
    idx = path.find(marker)
    if idx >= 0:
        return _clean_relative(path[idx + len(marker):])
    if workspace_dir is None:
        workspace_dir = configured_workspace_dir()
    anchors = [
        (root.rstrip("/"), prefix)
        for root, prefix in ((workspace_dir, ""), (projects_dir, PROJECTS_PREFIX))
        if root and root.rstrip("/")
    ]
    for root, prefix in sorted(anchors, key=lambda a: len(a[0]), reverse=True):
        if path == root or path.startswith(root + "/"):
            rel = _clean_relative(path[len(root):])
            if not rel:
                return None
            return f"{prefix}/{rel}" if prefix else rel
    return None


CANVAS_CHANNEL = "workspace:ws:{workspace_id}:canvas:events"
CANVAS_SCHEMA_VERSION = 1
_EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")


def _canvas_envelope(workspace_id: Any, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace-worker's canvas event shape (``canvas_events._envelope``) — the
    Code Canvas panel renders these unchanged, whichever engine produced them."""
    return {
        "schema_version": CANVAS_SCHEMA_VERSION,
        "event_type": event_type,
        "workspace_id": str(workspace_id),
        "data": data,
        "timestamp": _iso(_now()),
    }


PENDING_PERMISSIONS_KEPT = 20


def note_pending_permission(ref: Dict[str, Any], ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A session's permission question (PRD-235 W2 S3) → remembered on the ticket
    until the operator answers. Returns the stored entry, or None for a malformed event."""
    request_id = ev.get("request_id")
    if not request_id:
        return None
    entry = {
        "request_id": str(request_id),
        "tool": str(ev.get("tool_name") or "?")[:60],
        "subject": (str(ev["subject"])[:300] if ev.get("subject") else None),
        "reason": str(ev.get("reason") or "")[:300],
        "at": _iso(_now()),
    }
    pending = [p for p in (ref.get("pending_permissions") or []) if p.get("request_id") != entry["request_id"]]
    pending.append(entry)
    ref["pending_permissions"] = pending[-PENDING_PERMISSIONS_KEPT:]
    return entry


def record_permission_decision(ref: Dict[str, Any], request_id: str, approved: bool, actor: str) -> bool:
    """The operator's answer. False when the question is unknown (already answered or expired)."""
    pending = ref.get("pending_permissions") or []
    if not any(p.get("request_id") == str(request_id) for p in pending):
        return False
    ref["pending_permissions"] = [p for p in pending if p.get("request_id") != str(request_id)]
    decisions = dict(ref.get("permission_decisions") or {})
    decisions[str(request_id)] = {"approved": bool(approved), "by": actor, "at": _iso(_now()), "delivered": False}
    ref["permission_decisions"] = decisions
    return True


def take_undelivered_decisions(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Answers the host has not received yet; marks them delivered."""
    decisions = dict(ref.get("permission_decisions") or {})
    out = []
    for rid, d in decisions.items():
        if isinstance(d, dict) and not d.get("delivered"):
            out.append({"request_id": rid, "approved": bool(d.get("approved"))})
            decisions[rid] = {**d, "delivered": True}
    if out:
        ref["permission_decisions"] = decisions
    return out


def canvas_events_for(task: Any, ref: Dict[str, Any], ev: Dict[str, Any], projects_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """PRD-235 W2 S3: one hook event from the CLI host → the canvas events the Code
    Canvas panel already understands (tool call / result, file edit, session status).
    Every event carries ``task_id`` and ``session_id`` so a ticket-rooted Canvas can
    keep only its own session. Pure; never raises on odd input."""
    name = str(ev.get("event") or ev.get("hook_event_name") or "")
    base = {"source": "cli", "task_id": task.id, "session_id": ref.get("session_id"), "at": ev.get("at")}
    ws = task.workspace_id
    out: List[Dict[str, Any]] = []
    tool = ev.get("tool_name")
    subject = ev.get("subject")
    if name == "SessionStart":
        out.append(_canvas_envelope(ws, "canvas.session.status", {**base, "status": "running"}))
    elif name == "PreToolUse" and tool:
        out.append(_canvas_envelope(ws, "canvas.tool.call", {**base, "tool_name": str(tool), "input": ({"subject": str(subject)} if subject else {})}))
    elif name == "PostToolUse" and tool:
        out.append(_canvas_envelope(ws, "canvas.tool.result", {**base, "tool_name": str(tool)}))
        if tool in _EDIT_TOOLS and subject:
            rel = workspace_relative_path(str(subject), str(ws), projects_dir)
            out.append(_canvas_envelope(ws, "canvas.file.edit", {**base, "tool_name": str(tool), "path": rel or str(subject)}))
    elif name in ("Stop", "SessionEnd"):
        out.append(_canvas_envelope(ws, "canvas.session.status", {**base, "status": "stopped"}))
    elif name == "PermissionRequest" and ev.get("request_id"):
        # The panel renders this as an approval card (a bare permission card for a
        # command; the DiffCard shape needs old/new content the host does not send).
        data = {**base, "request_id": str(ev["request_id"]), "tool_name": str(tool or "?"),
                "reason": str(ev.get("reason") or "")[:300]}
        if subject:
            data["command" if tool == "Bash" else "path"] = str(subject)[:300]
        out.append(_canvas_envelope(ws, "canvas.permission.request", data))
    return out


def publish_canvas_events(workspace_id: Any, events: List[Dict[str, Any]]) -> int:
    """Fan the events out on the workspace's canvas channel (Redis pub/sub); fail-soft
    — the board still has ``runtime_ref``; only the live panel goes quiet."""
    if not events:
        return 0
    try:
        from core.redis.client import get_redis_client
        client = get_redis_client()
        if client is None:
            return 0
        channel = CANVAS_CHANNEL.format(workspace_id=str(workspace_id))
        return sum(1 for e in events if client.publish(channel, e))
    except Exception:  # noqa: BLE001
        logger.debug("[CliHost] canvas publish skipped", exc_info=True)
        return 0


def explorer_root_for(task_id: int, cwd: Optional[str], workspace_id: Any, projects_dir: Optional[str]) -> Optional[str]:
    """PRD-235 W2: where the Deliverables explorer (and the chat's Code mode) should
    open for this session — the worker-relative folder. A session with no working
    directory runs in ``sessions/<ticket>`` by the host's own rule; one inside the
    workspace volume or the projects folder maps through ``workspace_relative_path``;
    anywhere else is not browsable from the platform (``None``)."""
    if not cwd:
        return f"sessions/{task_id}"
    return browsable_root(str(cwd), str(workspace_id), projects_dir)


def browsable_root(folder: str, workspace_id: str, projects_dir: Optional[str]) -> Optional[str]:
    """The explorer root for a FOLDER on the host: the projects folder itself is
    ``projects`` (PRD-239 S6b — an agent rooted there browses all of it; a FILE
    path is never the root, so ``workspace_relative_path`` keeps saying None),
    anything else maps like a file path."""
    root = (projects_dir or "").rstrip("/")
    if root and folder.rstrip("/") == root:
        return PROJECTS_PREFIX
    workspace_dir = configured_workspace_dir()
    if workspace_dir and folder.rstrip("/") == workspace_dir:
        return WORKSPACE_ROOT_FOLDER
    return workspace_relative_path(folder, workspace_id, projects_dir)


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


def decide_session_permission(db: Session, task: BoardTask, request_id: str, approved: bool, actor: str) -> Dict[str, Any]:
    """PRD-235 W2 S3: the operator's answer to a session's permission question,
    recorded on the ticket and picked up by the host on its next event flush; the
    Canvas hears the outcome as a status line."""
    ref = dict(task.runtime_ref or {})
    if ref.get("runtime") != RUNTIME_CLI:
        raise LookupError("this task is not a Claude Code session")
    if not record_permission_decision(ref, request_id, approved, actor):
        raise LookupError(f"no pending permission question {request_id}")
    task.runtime_ref = ref
    db.commit()
    publish_canvas_events(task.workspace_id, [
        _canvas_envelope(task.workspace_id, "canvas.session.status", {
            "source": "cli", "task_id": task.id, "session_id": ref.get("session_id"),
            "status": "running", "decision": {"request_id": str(request_id), "approved": bool(approved)},
        }),
    ])
    return {"task_id": task.id, "request_id": str(request_id), "approved": bool(approved), "pending": len(ref.get("pending_permissions") or [])}


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
    # PRD-239: the directory the session really ran in (a git repo gets a
    # --worktree) wins over the configured one — it is where `claude --resume`
    # finds the transcript and where the editor links should open.
    if payload.get("effective_cwd"):
        _record_session_cwd(ref, task, str(payload["effective_cwd"]))
    # PRD-235 W2 S3: a question nobody answered before the session ended is stale —
    # its denial is already on the record (permission_denials); drop it from the queue.
    if ref.get("pending_permissions"):
        ref["expired_permissions"] = (ref.get("expired_permissions") or []) + ref["pending_permissions"]
        ref["pending_permissions"] = []

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
    book_session_usage(
        task, ref, usage,
        status=exec_result["status"],
        request_type=LANE_BOARD_TASK,
        execution_id=f"board_task:{task.id}",
        error=payload.get("error"),
    )
    # PRD-235 W2 S3: the final message and the end of the turn reach the Canvas too.
    final_text = payload.get("result_text") or payload.get("error") or ""
    base = {"source": "cli", "task_id": task.id, "session_id": ref.get("session_id")}
    publish_canvas_events(task.workspace_id, [
        _canvas_envelope(task.workspace_id, "canvas.assistant.text", {**base, "text": str(final_text)[:4000]}),
        _canvas_envelope(task.workspace_id, "canvas.turn.complete", {**base, "status": status}),
        _canvas_envelope(task.workspace_id, "canvas.session.status", {**base, "status": "stopped" if status != "error" else "failed"}),
    ])

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

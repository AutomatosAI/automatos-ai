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

import json
import hashlib
import re
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from config import config
from services.session_tools import definitions as session_tool_definitions
from services.session_tools import tool_names as session_tool_names
from core.cli_runtime import (
    CLI_PRESETS, CONFIG_ALLOWED_TOOLS_KEY, CONFIG_MODEL_KEY, CONFIG_PROVIDER_KEY, CONFIG_WORKING_DIRECTORY_KEY, CONFIG_WORKTREE_KEY, PROVIDER_CLAUDE, RUNTIME_CLI, registry_public,
)
from core.llm.usage_context import LANE_BOARD_TASK, LANE_SESSION
from core.models.approval_grants import SUBJECT_BOARD_TASK
from core.models.cli_hosts import CliHost, CliHostStatus
from core.models.core import Agent, BoardTask
from services.board_dispatcher import RUN_ID_KEY, claim_tasks, renew_lease
from services.board_events import notify_board_event
from services.cli_ticket_lane import SESSION_MODE_TERMINAL
from services.session_denials import classify_denial, forces_review
from services.session_report import APPROVAL_NOT_ON_RECORD

logger = logging.getLogger(__name__)

PAIRING_CODE_TTL_SECONDS = 600
HOST_TOKEN_BYTES = 32
MAX_CLAIM_LIMIT = 50
# No 0/O/1/I — a code is read off a screen and typed once.
_PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
# PRD-245 S1.1 — the credential a ticket session calls Automatos with. Minted at
# claim, only its HASH is kept (on the ticket, no migration), handed to the host
# once in the claim payload, and dead the moment the ticket leaves in_progress.
# Nothing durable: a token in a session transcript stops working when its ticket
# ends, which is the whole point of scoping it to one ticket.
SESSION_TOKEN_BYTES = 32
SESSION_TOKEN_HASH_KEY = "session_token_sha256"
# F131 (night 4, B47): the claim offered the session its Automatos tools, and the
# session's MCP client reached them (stamped at its `initialize`). A ticket that
# was offered them and never connected ran without any of its platform tools.
SESSION_TOOLS_OFFERED_KEY = "session_tools_offered"
SESSION_CONNECTED_KEY = "mcp_connected_at"
SESSION_TOOLS_PATH = "/api/v1/session-tools/mcp"
# What of an ask we keep ON the ticket (the grant row is the record; this is the
# fold-in for the next session's prompt, and it rides a JSONB column).
# How many questions ONE ticket may raise across its whole life. Each one is a
# card, a bell and a Telegram message addressed to the operator.
MAX_ASKS_PER_TICKET = 6
MAX_ASK_QUESTION_KEPT = 1000
# F037 (night 1): the owner's answer was cut here at 2,000 characters, mid-word,
# with nothing said to anyone — three of 28 answers lost ~1,050 characters of
# instructions. An answer runs to pages before it is cut now, and when it is,
# the text the session reads says so and where the rest is.
MAX_ASK_ANSWER_KEPT = 16_000
CUT_NOTE = "\n\n[{what} cut here at {kept:,} characters; {more:,} more are on question #{grant_id}.]"


def _kept(text: Any, limit: int, *, what: str, grant_id: Any) -> str:
    """``text`` as kept on the ticket: whole, or cut with a note saying so."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + CUT_NOTE.format(what=what, kept=limit, more=len(text) - limit, grant_id=grant_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    """A stored datetime as UTC-aware. SQLite (and some drivers) hand back naive
    values for a timezone-aware column; comparing those to ``_now()`` raises."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def mint_session_token(ref: Dict[str, Any]) -> str:
    """A fresh per-ticket token; the ref keeps only its hash (PRD-245 S1.1)."""
    token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    ref[SESSION_TOKEN_HASH_KEY] = hash_secret(token)
    return token


def clear_session_token(ref: Dict[str, Any]) -> None:
    ref.pop(SESSION_TOKEN_HASH_KEY, None)


def revoke_session_token(db: Session, task: Any) -> bool:
    """Kill this ticket's session credential on the row, now. True iff one was there.

    Called from ``apply_result``'s early return for a ticket that left
    ``in_progress`` (a stale attempt's token is already replaced, F211); cancel does the same
    thing inline (``clear_session_token``). The sweeper's requeue does NOT — it
    nulls the lease and leaves the hash, which is safe only because the lookup
    below requires a LIVE LEASE as well as ``in_progress``. ``in_progress`` alone is not enough to keep a token safe: a ticket
    that stops being ``in_progress`` can become ``in_progress`` again without a
    new claim (a board drag, a status PATCH, a heartbeat re-attach), and the
    plaintext is still in the session's transcript and its ``mcp.json``. The row
    is the only place the credential can be destroyed.
    """
    ref = dict(getattr(task, "runtime_ref", None) or {})
    if SESSION_TOKEN_HASH_KEY not in ref:
        return False
    clear_session_token(ref)
    task.runtime_ref = ref
    return True


def resolve_session_token(db: Session, token: Optional[str]) -> Optional[Tuple[BoardTask, Optional[Agent]]]:
    """The ticket and agent a session token belongs to, or ``None``.

    The ticket must still be ``in_progress`` AND hold a live lease. Status alone
    is not a session: a ticket can return to ``in_progress`` without a claim — a
    board drag, a status PATCH, a heartbeat re-attach — and that must not revive
    a credential whose plaintext is sitting in an old transcript. Only a claim
    sets a lease, and the host renews it on every event flush, so a running
    session always has one and nothing else does.
    """
    if not token or not str(token).strip():
        return None
    digest = hash_secret(str(token).strip())
    try:
        task = (
            db.query(BoardTask)
            .filter(
                BoardTask.status == "in_progress",
                BoardTask.lease_until.isnot(None),
                BoardTask.lease_until > _now(),
                BoardTask.runtime_ref[SESSION_TOKEN_HASH_KEY].astext == digest,
            )
            .first()
        )
    except Exception:  # noqa: BLE001 — a backend without JSONB text indexing, or a test double
        logger.debug("[cli-host] session-token lookup by JSONB failed; scanning in_progress tickets", exc_info=True)
        task = _scan_for_session_token(db, digest)
    if task is None:
        return None
    agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first() if task.assigned_agent_id else None
    return task, agent


def _scan_for_session_token(db: Session, digest: str) -> Optional[BoardTask]:
    """Fallback lookup: the local edition has a handful of running tickets."""
    try:
        rows = db.query(BoardTask).filter(BoardTask.status == "in_progress").all()
    except Exception:  # noqa: BLE001
        return None
    now = _now()
    for task in rows or []:
        ref = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
        if not secrets.compare_digest(str(ref.get(SESSION_TOKEN_HASH_KEY) or ""), digest):
            continue
        # The same live-lease rule the indexed query applies — a fallback that
        # answered where the query would not is a way around the rule.
        lease = getattr(task, "lease_until", None)
        if lease is None or _aware(lease) <= now:
            return None
        return task
    return None


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
EXPECTED_CLI_HOST_VERSION = "0.8.0"  # 2026-09-17: the claim carries the ticket's Automatos tools (``session_tools``, ``session_tools_path``, ``session_token``) — a host that predates them writes no MCP config and the session sees no platform tools, silently (PRD-245 W1). 0.7.0: the CLI is a parameter — capabilities carry every CLI under ``clis`` with served/reason, ``providers`` = the served ids (CLI adapter design). 0.6.0: a no-folder ticket runs in <deliverables root>/sessions/<ticket>

_CONTRACT_MODULES = ("api/cli_hosts.py", "services/cli_host_service.py", "core/cli_runtime.py", "core/cli_presets.py")


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
        # CLI adapter design §8.2/§8.3: which CLIs a ticket can be claimed for now.
        "providers_online": sorted({p for h in online for p in (served_providers_of(h) or [])}),
        # …and the CLIs the registry knows at all, so the picker renders from here, not a hardcoded list.
        "registry": registry_public(),
        "last_seen_at": last_seen.isoformat() if last_seen else None,
        "cli_agents": len(cli_agent_ids),
        "waiting_tickets": waiting,
        **contract_fields(),
    }


def served_providers_of(host: Any) -> Optional[List[str]]:
    """The CLIs a host announced it can run (``capabilities.providers``, the list
    the host builds from what is installed and logged in). ``None`` when the host
    never said — no filter is applied, as before the field existed; ``[]`` when it
    said it has none — it claims nothing."""
    caps = getattr(host, "capabilities", None)
    if not isinstance(caps, dict) or "providers" not in caps:
        return None
    raw = caps.get("providers")
    if not isinstance(raw, list):
        return []
    return [p for p in raw if isinstance(p, str) and p]


def serving_providers(db: Session, workspace_id: Any) -> List[str]:
    """The union of CLIs the workspace's ONLINE hosts serve — what a session ticket
    can be claimed for right now. The lane's blocked line and the picker read it."""
    hosts = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id, CliHost.status == CliHostStatus.PAIRED.value)
        .all()
    )
    out: List[str] = []
    for host in hosts:
        if not host.is_online():
            continue
        for provider in served_providers_of(host) or []:
            if provider not in out:
                out.append(provider)
    return sorted(out)


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
DEFAULT_FOLDER_SESSIONS = "sessions"   # … in a fresh ./workspaces/<ws>/sessions/<ticket> — the default (F042)
DEFAULT_FOLDER_CHOICES = (DEFAULT_FOLDER_PROJECTS, DEFAULT_FOLDER_SESSIONS)


def _workspace_row(db: Session, workspace_id: Any):
    from core.models.workspaces import Workspace

    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


def session_mode_settings(db: Session, workspace_id: Any) -> Dict[str, Any]:
    """What the operator sees and sets on Settings → Session mode: where tickets
    run when their agent names no folder, plus the projects folder as the stack
    was started with (a Docker mount — set in .env, read here) and how it is
    mounted. The default is a fresh sessions folder per ticket; the projects
    folder only when the operator chooses it. F042 (night 1): a folder-less OPS
    ticket started at the top of ~/Development — the folder that holds the
    Automatos checkout — walked in and sourced the platform's .env. A session
    started in a folder of its own reaches no repository it was not given."""
    ws = _workspace_row(db, workspace_id)
    stored = ((getattr(ws, "settings", None) or {}).get(SESSION_MODE_SETTINGS_KEY) or {}) if ws is not None else {}
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    choice = stored.get("default_folder")
    if choice not in DEFAULT_FOLDER_CHOICES:
        choice = DEFAULT_FOLDER_SESSIONS
    return {
        "default_folder": choice,
        "default_folder_explicit": stored.get("default_folder") in DEFAULT_FOLDER_CHOICES,
        "local_projects_dir": projects_dir,
        "projects_mount": (getattr(config, "LOCAL_PROJECTS_MOUNT", "") or None),
        # The deliverables root on the host (AUTOMATOS_WORKSPACE_DIR as `make up`
        # exported it) — beside the projects folder in Settings → Session mode.
        "workspace_dir": configured_workspace_dir(),
        # PRD-245 S1.5: what a ticket session of this workspace can call, so the
        # agent form can say it instead of the operator finding out from a report.
        "session_tools": [
            {"name": d["name"], "description": d["description"]} for d in session_tool_definitions()
        ],
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
    folder when the operator chose it and one is configured, else ``None`` —
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
        # The grant names the CLI; the host's adapter for it spells the command (design §7).
        "kind": ref.get("provider") or PROVIDER_CLAUDE,
        "session_id": str(session_id),
        # The terminal has no hooks (no policy gate): the soul only, never the
        # ticket session's tools block (PRD-245 S0.6).
        "system_prompt": _session_system_prompt(agent, ticket_session=False),
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


def _answers_fold_in(task: BoardTask) -> str:
    """PRD-245 W2 — the questions this ticket asked and what came back, for the
    prompt of the session that picks the work up. Read on the claim; the asks stay
    on the ticket as its record."""
    ref = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
    # Not the ones a previous resume already showed (``folded_at``) — a two-turn
    # ticket must not re-read yesterday's answer as if it were new.
    answered = [a for a in session_asks(ref)
                if a.get("answered_at") and a.get("answer") and not a.get("folded_at")]
    if not answered:
        return ""
    lines = ["## Answers to your questions",
             "You asked, and the operator answered. Carry on from where you stopped."]
    for ask in answered:
        lines += ["", f"**You asked:** {ask.get('question') or '(the question is on the ticket)'}",
                  f"**Answer:** {ask['answer']}"]
    return "\n".join(lines)


# What a mission's shared memory is worth carrying into a session prompt. Past a
# handful of points it stops being context and starts being a document.
FIELD_MEMORY_MAX_POINTS = 6
FIELD_MEMORY_MAX_VALUE_CHARS = 400
# A claim waits this long for the field and no longer.
FIELD_MEMORY_TIMEOUT_SECONDS = 5.0


def _field_memory_block(db: Session, task: BoardTask) -> str:
    """What the other agents on this mission already found, as a prompt section.

    Night 1 (2026-09-18): an API agent on a mission gets the field memory
    section rendered into its context; a session agent got nothing, so the two
    runtimes on one mission were not working from the same knowledge.

    The field is an async Qdrant read and the claim is sync (its callers, tests
    included, are sync), so the read runs on its own loop in one worker thread,
    bounded by a timeout. Returns "" for a standalone ticket and for EVERY
    failure — a ticket must never fail to claim because memory is slow or down.
    """
    run_id = getattr(task, "orchestration_run_id", None)
    if not run_id:
        return ""
    try:
        row = db.execute(
            sa_text("SELECT config FROM orchestration_runs WHERE id = :run_id"),
            {"run_id": str(run_id)},
        ).fetchone()
        blob = (row.config if row else None) or {}
        field_id = blob.get("field_id") if isinstance(blob, dict) else None
        if not field_id:
            return ""
        points = _read_field_points(
            str(field_id),
            task.title or task.description or "",
            int(task.assigned_agent_id or 0),
        )
    except Exception:  # noqa: BLE001
        logger.warning("[cli-host] field memory unavailable for ticket %s", task.id, exc_info=True)
        return ""

    if not points:
        return ""

    lines = ["## Field memory — what the other agents on this mission have found"]
    for point in points[:FIELD_MEMORY_MAX_POINTS]:
        key = str(point.get("key") or "note")
        value = str(point.get("value") or "")[:FIELD_MEMORY_MAX_VALUE_CHARS]
        lines.append(f"- **{key}**: {value}")
    lines.append(
        "\nAdd to it with `record_memory` when you find something the other agents "
        "would otherwise re-derive."
    )
    return "\n".join(lines)


def _read_field_points(field_id: str, query: str, agent_id: int) -> List[Dict[str, Any]]:
    """One field query, on its own loop, in its own thread, with a deadline."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    async def _query() -> List[Dict[str, Any]]:
        from modules.context.factory import get_shared_context

        field = get_shared_context()
        if not field:
            return []
        return await field.query(
            context_id=field_id, query=query, agent_id=agent_id,
            top_k=FIELD_MEMORY_MAX_POINTS,
        ) or []

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_query())).result(
            timeout=FIELD_MEMORY_TIMEOUT_SECONDS,
        )


def _ticket_prompt(task: BoardTask, field_memory: str = "") -> str:
    from services.ticket_owner_ask import ticket_answers_block
    from services.ticket_redo import redo_block

    prompt = task.raw_prompt or task.description or task.title or ""
    answers = _answers_fold_in(task)
    if answers:
        prompt = f"{prompt}\n\n{answers}"
    # F183: the owner's answers from Questions (a parked ticket re-queued by one).
    owners = ticket_answers_block(getattr(task, "planning_data", None))
    if owners:
        prompt = f"{prompt}\n\n{owners}"
    if field_memory:
        prompt = f"{prompt}\n\n{field_memory}"
    # Same redo fold-in as the dispatcher (Q44 + F198); consumed for this attempt.
    redo = redo_block(task)
    if redo:
        prompt = f"{prompt}\n\n{redo}"
        task.review_feedback = None
    return prompt


def _claim_attempt(task: BoardTask, prior: Dict[str, Any]) -> int:
    """F209: the number of this claim, which the host echoes with its result. It
    never repeats on a ticket. Run Now resets ``attempts`` to 0 (and a usage-limit
    pause refunds one), so a re-claim was numbered like the session it replaced,
    and that session's late result passed apply_result's stale-attempt check and
    finished the ticket under the new run. ``attempts`` stays the retry budget."""
    before = prior.get("attempt")
    return max(int(task.attempts or 0), before + 1 if isinstance(before, int) else 0)


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
    # CLI adapter design §8.2: the claim is filtered by the CLIs this host serves.
    # A host that announced no CLI at all takes nothing — refusing a ticket after
    # the claim would bounce it between hosts of different CLIs every poll.
    providers = served_providers_of(host)
    if providers is not None and not providers:
        logger.info("[cli-host] host %s announces no CLI it can run — claiming nothing", host.id)
        return {"tasks": [], "parked": []}
    claimed = claim_tasks(
        db,
        worker_id=f"cli-host:{host.id}",
        limit=limit,
        lease_seconds=config.BOARD_DISPATCH_LEASE_SECONDS,
        max_slots_per_agent=None,
        runtime=RUNTIME_CLI,
        workspace_id=host.workspace_id,
        providers=providers,
    )
    out: List[Dict[str, Any]] = []
    parked: List[Dict[str, Any]] = []
    for task in claimed:
        if _blocked_pending_approval(db, task):
            db.refresh(task)
            parked.append({"task_id": task.id, "title": task.title, "reason": task.blocked_reason})
            continue  # parked ``blocked`` by the gate; the answered-resume loop returns it
        from services.cli_ticket_lane import NO_HOST_REASON, is_no_cli_host_reason
        if task.blocked_reason == NO_HOST_REASON or is_no_cli_host_reason(task.blocked_reason):
            task.blocked_reason = None  # a host that runs this CLI is here now
        agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first()
        cfg = (getattr(agent, "configuration", None) if agent else None) or {}
        prior = task.runtime_ref if isinstance(task.runtime_ref, dict) else {}
        resume_session_id = _resume_session_for(prior, host)
        session_id = str(uuid4())
        provider = cfg.get(CONFIG_PROVIDER_KEY) or PROVIDER_CLAUDE
        ref = {
            "runtime": RUNTIME_CLI,
            "provider": provider,
            "provider_label": CLI_PRESETS[provider].label if provider in CLI_PRESETS else provider,
            "model": cfg.get(CONFIG_MODEL_KEY),
            "host_id": str(host.id),
            "session_id": session_id,
            "attempt": _claim_attempt(task, prior),
            RUN_ID_KEY: prior.get(RUN_ID_KEY),  # F209: the claim's run, stamped by claim_tasks
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
        # PRD-245 W2: the asks this ticket already made are its record — the
        # answer to a resumed session is folded into the prompt from them, and
        # MAX_ASKS_PER_TICKET counts them across the ticket's life. The claim
        # builds a fresh ``ref``, so they have to be carried, and the prompt has
        # to be rendered AFTER they are on the row. Building it before (the bug)
        # read the new empty ref: no answer ever reached the resumed session and
        # the ceiling reset to zero every claim.
        ref[SESSION_ASKS_KEY] = session_asks(prior)
        # F094: the notes on the ticket (the operator's, the session's, the
        # mission's verdict) are its record too; a claim that resumes the same
        # run keeps them. A mission step's next run starts with none.
        prior_notes = prior.get(SESSION_NOTES_KEY)
        if isinstance(prior_notes, list) and prior_notes:
            ref[SESSION_NOTES_KEY] = prior_notes
        # PRD-245 S1.1: the session's own credential for the Automatos tools.
        # Handed over ONCE, in this payload; only its hash stays on the ticket.
        session_token = mint_session_token(ref)
        ref[SESSION_TOOLS_OFFERED_KEY] = True
        task.runtime_ref = ref
        prompt = _ticket_prompt(task, _field_memory_block(db, task))  # reads the carried asks
        # Mark the answers just folded in, so a LATER resume of the same ticket
        # does not render them again.
        if ref.get(SESSION_ASKS_KEY):
            ref[SESSION_ASKS_KEY] = _mark_answers_folded(ref[SESSION_ASKS_KEY])
            task.runtime_ref = ref
        out.append(
            {
                "task_id": task.id,
                "workspace_id": str(task.workspace_id),
                "agent_id": task.assigned_agent_id,
                "agent_name": getattr(agent, "name", None),
                "title": task.title,
                "prompt": prompt,
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
                # PRD-245 W1: the Automatos tools this session may call, and how
                # to reach them. The host writes them into the session's own MCP
                # config and allows exactly these names at the gate.
                "session_tools": list(session_tool_names()),
                # The PATH, not a URL: the host joins it to the backend address
                # it was started with. A container cannot know the address the
                # session on the operator's machine must dial.
                "session_tools_path": SESSION_TOOLS_PATH,
                "session_token": session_token,
            }
        )
    db.commit()
    return {"tasks": out, "parked": parked}


def _session_system_prompt(agent: Optional[Agent], *, ticket_session: bool = True) -> str:
    """Never lets a rendering problem block a claim — the host falls back to
    name + rules when this is empty. ``ticket_session=False`` for the Canvas
    terminal, which runs without the policy gate the tools block describes."""
    if agent is None:
        return ""
    try:
        from services.cli_session_prompt import session_system_prompt

        return session_system_prompt(agent, ticket_session=ticket_session)
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
    if task.status != status_before:
        # F119: before the commit that carries it — the host's events route
        # returns without another, and the request's rollback dropped it.
        notify_board_event(
            db, workspace_id=host.workspace_id, task_id=task.id, status=task.status,
            event="task_claimed" if task.status == "in_progress" else "task_completed",
        )
    db.commit()
    return {"status": task.status, "lease_renewed": False, "control": {}, "decisions": []}


def _absorb_hook_event(ref: Dict[str, Any], task: BoardTask, ev: Dict[str, Any]) -> None:
    """One hook event into the ticket's live summary (``ref`` is the caller's
    working copy of ``runtime_ref``; the caller writes it back once)."""
    name = ev.get("event") or ev.get("hook_event_name")
    if name:
        ref["last_event"] = name
    if name == "PreToolUse" and ev.get("tool_name"):
        decided = tool_decision(ev, ref.get("permission_decisions"))
        # PRD-234 S2: the ticket's live log — tool + what it was about, bounded.
        # F167: and what the host decided, and why.
        entry: Dict[str, Any] = {"tool": str(ev["tool_name"])[:60], **decided,
                                 **({"event_id": _event_id(ev)} if _event_id(ev) else {})}
        if ev.get("subject"):
            entry["subject"] = str(ev["subject"])[:200]
        if not _already_kept(ref, entry):
            if tool_call_ran(decided):
                ref["live_tool"] = ev["tool_name"]
            ref["recent_tools"] = (list(ref.get("recent_tools") or [])
                                   + [{"at": _iso(_now()), **entry}])[-RECENT_TOOLS_KEPT:]
            if decided:
                ref["tool_decisions"] = tally_tool_decision(ref.get("tool_decisions"), decided)
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


async def record_events(
    db: Session, host: CliHost, task_id: int, events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Absorb a batch of hook events: renew the lease, keep a compact live summary
    in ``runtime_ref`` (live tool, transcript path, counts), raise a question for
    every command the host is holding (PRD-245 S0.4), and hand back control
    (``cancel``) and the operator's answers the host must act on. Events are not
    persisted individually here — S2 maps them to board events and the fleet."""
    events = [ev for ev in (events or []) if isinstance(ev, dict)]
    if events and all(_terminal_event_name(ev) for ev in events):
        return _record_terminal_events(db, host, task_id, events)
    task = _owned_task(db, host, task_id)
    renewed = renew_lease(db, task_id, lease_seconds=config.BOARD_DISPATCH_LEASE_SECONDS)
    ref = dict(task.runtime_ref or {})
    ref["events_seen"] = int(ref.get("events_seen") or 0) + len(events)
    ref["last_event_at"] = _iso(_now())
    for ev in events:
        _absorb_hook_event(ref, task, ev)
    task.runtime_ref = ref
    db.commit()
    ref = await raise_session_holds(db, task, ref)
    # PRD-235 W2 S3: the same events light up the Code Canvas panel.
    projects_dir = getattr(config, "LOCAL_PROJECTS_DIR", "") or None
    canvas: List[Dict[str, Any]] = []
    for ev in events:
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


# ── PRD-245 S0.4: a held command is a question (Questions tab, bell, Telegram) ──
# The host holds a shell command the policy could not judge and asks. Before
# this wave the question was a card on the ticket's Canvas over live SSE and
# nowhere else (nineteen holds, zero answered, 2026-09-17). Now every hold is
# ONE PRD-225 question row on the ticket — the tab lists it with the ticket's
# cascade, the bell rings, the Telegram bridge delivers it and correlates the
# reply — while the ticket keeps RUNNING (the host is waiting, not parked).

SESSION_HOLD_MARKER = "cli_permission"       # ``ApprovalGrant.details[<marker>] = {request_id, task_id}``
# PRD-245 W2 — a question the SESSION asked through ``ask_human`` (not a held
# command). The ticket keeps its asks so the turn's end knows to park instead of
# finishing, and so the answer can be folded into the prompt of the session that
# picks the work back up.
SESSION_ASK_MARKER = "cli_ask"               # ``ApprovalGrant.details[<marker>] = {task_id}``
SESSION_ASKS_KEY = "session_asks"            # ``runtime_ref[<key>] = [{grant_id, question, answer?, …}]``
SESSION_NOTES_KEY = "session_notes"          # ``runtime_ref[<key>] = [{note, at, by}]``, appended only
PARKED_FOR_ANSWER_REASON = "Waiting on your answer to the agent's question (ask #{grant_id})"
SESSION_HOLD_OPTION_ALLOW = "allow"
SESSION_HOLD_OPTION_DENY = "deny"
SESSION_HOLD_OPTIONS = (SESSION_HOLD_OPTION_ALLOW, SESSION_HOLD_OPTION_DENY)
# The host's default ``--ask-timeout`` (services/cli-host … config.py); the row's
# ``expires_at`` mirrors it. The row is CLOSED by the session's result in any case
# (``apply_result``) — the host does not report its actual timeout on the event.
SESSION_HOLD_TTL_SECONDS = 3600


def session_hold_question(task_id: Any, entry: Dict[str, Any]) -> str:
    """The question the operator sees, wherever it reaches them.

    Night 1 (2026-09-18): the card was a raw, truncated shell command plus the
    gate's own wording — the operator had to reverse-engineer what the agent was
    trying to do before deciding. So: what it wants, in a plain sentence, first;
    the exact command behind a disclosure; the gate's reason last. The answer
    words are spelled out because a Telegram reply sees no buttons and anything
    but ``allow`` is read as deny (fail closed).
    """
    subject = str(entry.get("subject") or entry.get("tool") or "?")
    intent = str(entry.get("intent") or entry.get("description") or "").strip()

    lines = [f"**Allow this command in ticket #{task_id}?**", ""]
    lines += [intent or _plain_intent(subject), ""]
    # The full command, never truncated, but folded away — the summary line is
    # what most decisions are made on.
    lines += ["<details><summary>The exact command</summary>", "", "```sh", subject, "```", "", "</details>"]
    if entry.get("reason"):
        lines += ["", f"_Held because: {entry['reason']}_"]
    lines += ["", f"Answer `{SESSION_HOLD_OPTION_ALLOW}` or `{SESSION_HOLD_OPTION_DENY}`."]
    return "\n".join(lines)


# What the common read-only verbs actually do, so a card can say it in English
# rather than showing a shell fragment and hoping.
_VERB_INTENTS: Dict[str, str] = {
    "comm": "compare two sorted files line by line",
    "diff": "compare two files",
    "cat": "read a file",
    "head": "read the start of a file",
    "tail": "read the end of a file",
    "grep": "search files for a pattern",
    "rg": "search files for a pattern",
    "find": "look for files",
    "ls": "list a directory",
    "wc": "count lines or words in a file",
    "sort": "sort lines",
    "uniq": "collapse repeated lines",
    "sed": "transform text",
    "awk": "extract fields from text",
    "curl": "fetch a URL",
    "git": "run a git command",
    "npm": "run an npm command",
    "python": "run a Python command",
    "python3": "run a Python command",
}


_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATOR_RE = re.compile(r"&&|\|\||[;|]")


def _command_verbs(command: str) -> List[str]:
    """The programs a command line actually runs, in order, without repeats.

    Skips blank lines, comment lines and leading ``VAR=value`` assignments —
    none of which is a program. F056 (night 2, grant 355): this took the first
    WORD of ``WORK=/…/sessions/359`` and printed "The agent wants to run 359";
    the program three lines down was ``grep``.
    """
    verbs: List[str] = []
    for line in command.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for segment in _SEPARATOR_RE.split(line):
            words = segment.split()
            while words and _ASSIGNMENT_RE.match(words[0]):
                words = words[1:]
            if not words or words[0].startswith("#"):
                continue
            verb = words[0].rsplit("/", 1)[-1]      # /usr/bin/grep -> grep
            if verb and verb not in verbs:
                verbs.append(verb)
    return verbs


def _plain_intent(command: str) -> str:
    """One sentence describing what the held command would do."""
    verbs = _command_verbs(command)
    if not verbs:
        if _ASSIGNMENT_RE.match(command.strip()):
            return "The agent wants to set a shell variable (it runs no program)."
        return "The agent wants to run a command."
    first = verbs[0]
    what = _VERB_INTENTS.get(first)
    lead = f"**{what}** (`{first}`)" if what else f"run **{first}**"
    rest = verbs[1:]
    if not rest:
        return f"The agent wants to {lead}."
    tail = ", ".join(f"`{v}`" for v in rest[:4]) + (f" and {len(rest) - 4} more" if len(rest) > 4 else "")
    return f"The agent wants to {lead}, then {tail}."


def is_allow_answer(answer: Any) -> bool:
    """Only an answer that starts with ``allow`` allows; everything else denies."""
    return str(answer or "").strip().lower().startswith(SESSION_HOLD_OPTION_ALLOW)


def session_ask_marker(grant: Any) -> Optional[Dict[str, Any]]:
    """``{task_id}`` when this question row is a session's own ask, else None."""
    details = getattr(grant, "details", None)
    marker = details.get(SESSION_ASK_MARKER) if isinstance(details, dict) else None
    if not isinstance(marker, dict) or marker.get("task_id") is None:
        return None
    return marker


def session_asks(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in (ref.get(SESSION_ASKS_KEY) or []) if isinstance(a, dict)]


def open_session_asks(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The asks nobody has answered yet — why a finished turn parks."""
    return [a for a in session_asks(ref) if not a.get("answered_at")]


def record_session_ask(ref: Dict[str, Any], *, grant_id: Any, question: str) -> Dict[str, Any]:
    """Remember an ask the session just made. Returns the rebuilt ref."""
    entry = {"grant_id": int(grant_id),
             "question": _kept(question, MAX_ASK_QUESTION_KEPT, what="The question is", grant_id=grant_id),
             "asked_at": _iso(_now())}
    return {**ref, SESSION_ASKS_KEY: [*session_asks(ref), entry]}


def _mark_answers_folded(asks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp ``folded_at`` on every answered ask, so ``_answers_fold_in`` shows
    it once. Idempotent — an already-folded ask keeps its first stamp."""
    now = _iso(_now())
    out = []
    for ask in asks:
        if isinstance(ask, dict) and ask.get("answered_at") and ask.get("answer") and not ask.get("folded_at"):
            out.append({**ask, "folded_at": now})
        else:
            out.append(ask)
    return out


def record_session_answer(ref: Dict[str, Any], *, grant_id: Any, answer: str) -> Dict[str, Any]:
    """Write the operator's answer onto the ask it belongs to. Returns the
    rebuilt ref; the same ref when the ask is unknown or already answered."""
    updated: List[Dict[str, Any]] = []
    touched = False
    for ask in session_asks(ref):
        if not touched and int(ask.get("grant_id") or 0) == int(grant_id) and not ask.get("answered_at"):
            kept = _kept(answer, MAX_ASK_ANSWER_KEPT, what="The owner's answer is", grant_id=grant_id)
            if len(str(answer)) > MAX_ASK_ANSWER_KEPT:
                logger.warning("[cli-host] the answer to ask #%s is %s characters — %s kept on the ticket, "
                               "the rest stays on the question", grant_id, len(str(answer)), MAX_ASK_ANSWER_KEPT)
            updated.append({**ask, "answer": kept, "answered_at": _iso(_now())})
            touched = True
            continue
        updated.append(ask)
    return {**ref, SESSION_ASKS_KEY: updated} if touched else ref


async def raise_session_ask(
    db: Session,
    *,
    task_id: Any,
    workspace_id: Any,
    agent_id: Optional[int],
    agent_name: Optional[str],
    question: str,
    options: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """PRD-245 W2 — one question from a session.

    Filed through PRD-225's SHARED internals (the same function
    ``platform_ask_human`` dispatches to), so it reaches the Questions tab, the
    bell and Telegram with nothing new — but UNPARKED: parking a ticket whose
    session is still mid-turn would make its own result undeliverable
    (``apply_result`` writes only a run that is still ``in_progress``). The
    turn's end parks it (``_park_for_answer``).

    Returns an executor-shaped result, so the session reads it as ordinary tool
    output."""
    from modules.tools.discovery.handlers_asks import stage_question

    task = (
        db.query(BoardTask)
        .filter(BoardTask.id == int(task_id), BoardTask.workspace_id == workspace_id)
        .first()
    )
    if task is None:
        return {"success": False, "error": f"ticket #{task_id} is not in this workspace"}

    # One open question at a time, and a hard ceiling per ticket. Every ask
    # raises a card in the Questions tab, rings the bell and sends a Telegram
    # message with text the session chose — so "ask politely once" cannot be a
    # prompt instruction alone. A session whose prompt has been steered would
    # otherwise reach the operator as many times as its tool allowance allows.
    ref = dict(task.runtime_ref or {})
    still_open = open_session_asks(ref)
    if still_open:
        return {"success": False,
                "error": "you already have a question waiting for an answer on this ticket: "
                         f"{str(still_open[-1].get('question') or '')[:120]!r}. Finish what you can "
                         "without it and end your turn — the answer resumes you."}
    if len(session_asks(ref)) >= MAX_ASKS_PER_TICKET:
        return {"success": False,
                "error": f"this ticket has asked its {MAX_ASKS_PER_TICKET} questions. Say what you "
                         "still need in your final message and end your turn."}

    try:
        staged = await stage_question(
            db, workspace_id,
            subject_type=SUBJECT_BOARD_TASK, subject_id=str(int(task_id)),
            question=question,
            options=list(options or []) or None,
            asked_by_agent_id=agent_id, agent_name=agent_name,
            park=None,                                   # the turn's end parks it
            details={SESSION_ASK_MARKER: {"task_id": int(task_id)}},
        )
    except Exception as exc:  # noqa: BLE001 — the session reads the reason and carries on
        logger.error("[cli-host] ticket #%s could not file its question", task_id, exc_info=True)
        return {"success": False,
                "error": f"the question could not be filed ({type(exc).__name__}); "
                         "put it in your final message instead"}
    ask_id = staged.get("ask_id") if isinstance(staged, dict) else None
    if ask_id is None:
        return {"success": False,
                "error": "the question could not be filed; put it in your final message instead"}
    task.runtime_ref = record_session_ask(dict(task.runtime_ref or {}), grant_id=ask_id, question=question)
    db.commit()
    logger.info("[cli-host] ticket #%s asked the operator (ask #%s)", task_id, ask_id)
    return {
        "success": True,
        "result": {
            "ask_id": int(ask_id),
            "message": (
                f"Asked the operator (question #{ask_id}). It is on their Questions tab and their phone. "
                "Your ticket parks on it when your turn ends and picks up here — with the answer — once "
                "they reply. Finish everything that does not depend on the answer now, then end your "
                "turn. Do not wait and do not ask again."
            ),
        },
    }


def answer_session_ask(db: Session, grant: Any) -> bool:
    """PRD-245 W2 — PRD-225's answer path reached a session's own question.

    The answer is written onto the ticket (so the session that picks the work up
    reads it in its prompt) and the ticket is re-queued: ``blocked`` →
    ``assigned`` for the host to claim and RESUME the same Claude Code session.
    True when the work actually moves.

    A ticket still ``in_progress`` (the operator answered mid-turn) is only
    recorded here: its own turn end does the re-queue, because a running session
    must never be claimed twice."""
    marker = session_ask_marker(grant)
    if marker is None:
        return False
    task = (
        db.query(BoardTask)
        .filter(BoardTask.id == int(marker["task_id"]), BoardTask.workspace_id == grant.workspace_id)
        .first()
    )
    if task is None:
        logger.warning("[cli-host] ask #%s names ticket #%s, which is not in workspace %s",
                       grant.id, marker["task_id"], grant.workspace_id)
        return False
    ref = record_session_answer(dict(task.runtime_ref or {}), grant_id=grant.id,
                                answer=str(getattr(grant, "answer_text", "") or ""))
    if task.status == "in_progress":
        task.runtime_ref = ref
        db.commit()
        logger.info("[cli-host] ask #%s answered while ticket #%s still runs — its turn end picks it up",
                    grant.id, task.id)
        return False
    if task.status != "blocked":
        task.runtime_ref = ref
        db.commit()
        logger.info("[cli-host] ask #%s answered, but ticket #%s is %s — nothing to resume",
                    grant.id, task.id, task.status)
        return False
    # F036: the answer is recorded on the ticket either way, but it only
    # resumes a ticket that is parked FOR it. Night 1's ticket 136 was stopped
    # by a person at 18:18:06 and re-claimed at 18:19:17 — one second after its
    # dead session's question was answered.
    from services.operator_stop import operator_stop

    if operator_stop(task):
        task.runtime_ref = ref
        db.commit()
        logger.info("[cli-host] ask #%s answered, but ticket #%s was stopped by a person — not resumed",
                    grant.id, task.id)
        return False
    if requeue_exhausted(task):
        task.runtime_ref = ref
        park_exhausted(db, task, "it has been resumed on answers too many times")
        return False
    task.runtime_ref = _mark_resumable(ref)
    task.status = "assigned"
    task.blocked_at = None
    task.blocked_reason = None
    db.commit()
    _notify_status(db, task)
    _notify_available(db, task)
    logger.info("[cli-host] ticket #%s resumes on the answer to ask #%s", task.id, grant.id)
    return True


def session_hold_marker(grant: Any) -> Optional[Dict[str, Any]]:
    """``{request_id, task_id}`` when this question row is a session hold, else None."""
    details = getattr(grant, "details", None)
    marker = details.get(SESSION_HOLD_MARKER) if isinstance(details, dict) else None
    if not isinstance(marker, dict) or not marker.get("request_id") or marker.get("task_id") is None:
        return None
    return marker


def pending_permission_entry(ref: Dict[str, Any], request_id: Any) -> Optional[Dict[str, Any]]:
    for entry in ref.get("pending_permissions") or []:
        if isinstance(entry, dict) and entry.get("request_id") == str(request_id):
            return entry
    return None


async def raise_session_holds(db: Session, task: BoardTask, ref: Dict[str, Any]) -> Dict[str, Any]:
    """Every pending permission without a question row yet gets one; the grant
    id is kept on the entry, so a re-flushed event creates nothing twice. Returns
    the (rebuilt) ``runtime_ref`` the caller continues with — the one it came in
    with when filing fails: the event flush (lease, decisions) must still reach
    the host, and the Canvas card stands alone until the next flush."""
    pending = [p for p in (ref.get("pending_permissions") or []) if isinstance(p, dict)]
    if not any(p.get("grant_id") is None for p in pending):
        return ref
    try:
        return await _raise_session_holds(db, task, ref, pending)
    except Exception:  # noqa: BLE001 — never break the flush over a question row
        logger.error("[cli-host] hold questions for ticket #%s not filed this flush", task.id, exc_info=True)
        db.rollback()
        return ref


async def _raise_session_holds(
    db: Session, task: BoardTask, ref: Dict[str, Any], pending: List[Dict[str, Any]],
) -> Dict[str, Any]:
    known = _open_hold_question_ids_by_request(db, task)  # rows an earlier flush committed before failing
    agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first() if task.assigned_agent_id else None
    agent_name = getattr(agent, "name", None)
    updated: List[Dict[str, Any]] = []
    for entry in pending:
        if entry.get("grant_id") is not None:
            updated.append(entry)
            continue
        grant_id = known.get(str(entry.get("request_id")))
        if grant_id is None:
            grant_id = await _stage_hold_question(db, task, agent_name, entry)
        updated.append({**entry, "grant_id": grant_id} if grant_id is not None else entry)
    # Staging a question can spend seconds (bell + Telegram), and Claude Code
    # issues tool calls in parallel — the session's own ``ask_human`` or the
    # operator's answer to another hold can commit onto THIS row meanwhile.
    # ``record_events`` already committed the events, so re-reading loses nothing
    # of ours and picks up theirs; this write then owns ``pending_permissions``
    # ONLY, instead of stamping a whole document read before the wait over it.
    db.refresh(task)
    fresh = dict(task.runtime_ref or {})
    fresh["pending_permissions"] = updated
    task.runtime_ref = fresh
    db.commit()
    return fresh


async def _stage_hold_question(
    db: Session, task: BoardTask, agent_name: Optional[str], entry: Dict[str, Any],
) -> Optional[int]:
    """One hold → one question row through the shared PRD-225 internals (the
    function ``platform_ask_human`` dispatches to), unparked. A failure here is
    logged and leaves the Canvas card as the only surface — the host's event
    flush (lease, decisions) must still be answered."""
    from modules.tools.discovery.handlers_asks import stage_question

    request_id = str(entry.get("request_id"))
    try:
        res = await stage_question(
            db, task.workspace_id,
            subject_type=SUBJECT_BOARD_TASK, subject_id=str(task.id),
            question=session_hold_question(task.id, entry), options=list(SESSION_HOLD_OPTIONS),
            ttl_seconds=SESSION_HOLD_TTL_SECONDS,
            asked_by_agent_id=task.assigned_agent_id, agent_name=agent_name,
            details={SESSION_HOLD_MARKER: {"request_id": request_id, "task_id": task.id}},
        )
    except Exception:  # noqa: BLE001 — the events flush must reach the host regardless
        logger.error("[cli-host] no question row for hold %s on ticket #%s — the Canvas card stands alone",
                     request_id, task.id, exc_info=True)
        db.rollback()
        return None
    ask_id = res.get("ask_id") if isinstance(res, dict) else None
    if ask_id is None:
        logger.error("[cli-host] hold %s on ticket #%s: the ask internals returned no id (%s)", request_id, task.id, res)
        return None
    logger.info("[cli-host] hold %s on ticket #%s is question #%s", request_id, task.id, ask_id)
    return int(ask_id)


def answer_session_hold(db: Session, grant: Any) -> bool:
    """PRD-225's answer path reached a session hold: the operator's ``allow`` /
    ``deny`` becomes the ticket's decision for the host's next event flush. The
    ticket is never re-queued (it is running). False when the hold is no longer
    open — answered from the Canvas card first, or the session already ended —
    so the confirmation says nothing resumed."""
    marker = session_hold_marker(grant)
    if marker is None:
        return False
    task = (
        db.query(BoardTask)
        .filter(BoardTask.id == int(marker["task_id"]), BoardTask.workspace_id == grant.workspace_id)
        .first()
    )
    if task is None:
        logger.warning("[cli-host] question #%s names ticket #%s, which is not in workspace %s",
                       grant.id, marker["task_id"], grant.workspace_id)
        return False
    approved = is_allow_answer(grant.answer_text)
    try:
        decide_session_permission(db, task, str(marker["request_id"]), approved, grant.answered_by or "user:unknown")
    except LookupError as exc:
        logger.info("[cli-host] question #%s: hold %s on ticket #%s is already decided or gone — %s",
                    grant.id, marker["request_id"], task.id, exc)
        return False
    return True


def _close_hold_question(db: Session, grant_id: Any, approved: bool, actor: str) -> bool:
    """The Canvas card was answered first: the Questions row follows (the same
    pending→granted statement the answer route uses; a no-op when the row was
    the one that carried the answer)."""
    if grant_id is None:
        return False
    from core.services.approval_grants import answer_pending_grant

    answer = SESSION_HOLD_OPTION_ALLOW if approved else SESSION_HOLD_OPTION_DENY
    return answer_pending_grant(db, int(grant_id), answer_text=answer, answered_by=actor)


def _open_hold_question_rows(db: Session, task: BoardTask) -> List[Any]:
    """Every still-pending hold question on this ticket — the marker rows, not
    the (capped) pending list, so no row outlives its session."""
    from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION

    rows = (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == task.workspace_id,
            ApprovalGrant.subject_type == SUBJECT_BOARD_TASK,
            ApprovalGrant.subject_id == str(task.id),
            ApprovalGrant.kind == KIND_QUESTION,
            ApprovalGrant.status == GrantStatus.PENDING.value,
        )
        .all()
    )
    return [g for g in rows if session_hold_marker(g) is not None]


def _open_hold_question_ids(db: Session, task: BoardTask) -> List[int]:
    return [int(g.id) for g in _open_hold_question_rows(db, task)]


def _open_hold_question_ids_by_request(db: Session, task: BoardTask) -> Dict[str, int]:
    """``request_id`` → question id for the open hold rows of this ticket, so a
    row committed by a flush that then failed is reused, never duplicated."""
    return {str(session_hold_marker(g)["request_id"]): int(g.id) for g in _open_hold_question_rows(db, task)}


def _expire_hold_questions(db: Session, task: BoardTask, entries: List[Dict[str, Any]], host: CliHost) -> int:
    """The session ended with holds unanswered: their question rows close as
    ``expired`` (nobody can answer them any more); an answer that won a moment
    earlier is left alone."""
    from core.services.approval_grants import expire_pending_grants

    ids = {int(e["grant_id"]) for e in entries if isinstance(e, dict) and e.get("grant_id") is not None}
    ids |= set(_open_hold_question_ids(db, task))
    if not ids:
        return 0
    closed = expire_pending_grants(db, task.workspace_id, ids, revoked_by=f"cli-host:{host.id}")
    logger.info("[cli-host] ticket #%s ended with %s unanswered hold(s); %s question row(s) expired", task.id, len(ids), closed)
    return closed


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

# F167: what the host decided for a tool call (host ≥ this change). A ticket said
# a command "needed your approval, and it went through" when nobody was asked:
# the board knew which tools ran, never what the host decided or why.
TOOL_DECISIONS = ("allow", "ask", "deny")
TOOL_ANSWERS = ("approved", "denied", "no answer")   # a hold's outcome, as the host reports it
TOOL_DECISION_TALLY = (*TOOL_DECISIONS, "approved", "unrecorded")
TOOL_DECISION_REASON_CHARS = 300
EVENT_ID_CHARS = 64


def _event_id(ev: Dict[str, Any]) -> Optional[str]:
    """The host's id for one reported tool call, or None (an older host, or junk)."""
    value = ev.get("event_id")
    return value if isinstance(value, str) and 0 < len(value) <= EVENT_ID_CHARS else None


def _already_kept(ref: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """The ticket already keeps this exact entry: the host re-posts a batch whose
    response it lost, and the same call must count once. A re-post is the SAME
    event, so everything but the time must match — id, tool, command, decision,
    reason and answer. An id reused for anything else is recorded: no call hides
    behind an earlier one's id. An entry with no id counts, as before."""
    return bool(entry.get("event_id")) and any(
        isinstance(t, dict) and {k: v for k, v in t.items() if k != "at"} == entry
        for t in ref.get("recent_tools") or [])


def tool_decision(ev: Dict[str, Any], decisions: Any = None) -> Dict[str, Any]:
    """``{decision, reason?, answer?}`` from a PreToolUse event — the known words
    only — or ``{}`` from a host that does not report its decisions.

    The host learns an approval only from this backend (``record_permission_decision``,
    then the events answer), so a real one is always on record here: for that
    request, that tool and that command. An approval that is not is shown as such,
    never as the operator's."""
    decision = ev.get("decision")
    if decision not in TOOL_DECISIONS:
        return {}
    out: Dict[str, Any] = {"decision": decision}
    reason = ev.get("reason")
    if isinstance(reason, str) and reason.strip():
        out["reason"] = reason.strip()[:TOOL_DECISION_REASON_CHARS]
    answer = ev.get("answer")
    if decision == "ask" and answer in TOOL_ANSWERS:
        out["answer"] = answer if answer != "approved" or approval_on_record(decisions, ev) \
            else APPROVAL_NOT_ON_RECORD
    return out


def approval_on_record(decisions: Any, ev: Dict[str, Any]) -> bool:
    """The operator approved THIS question: the request the event names, for the
    same tool and command (a request id replayed for another command is not)."""
    request_id = ev.get("request_id")
    record = decisions.get(str(request_id)) if isinstance(decisions, dict) and request_id else None
    if not isinstance(record, dict) or record.get("approved") is not True:
        return False
    subject = str(ev["subject"])[:PERMISSION_SUBJECT_CHARS] if ev.get("subject") else None
    return record.get("tool") == str(ev.get("tool_name") or "?")[:60] and record.get("subject") == subject


def tool_call_ran(decided: Dict[str, Any]) -> bool:
    """An allow, or a hold the host says was approved. With no decision reported
    (an older host), the call is taken as run, as it always was."""
    return (not decided or decided["decision"] == "allow"
            or decided.get("answer") in ("approved", APPROVAL_NOT_ON_RECORD))


def tally_tool_decision(tally: Any, decided: Dict[str, Any]) -> Dict[str, int]:
    """The ticket's count of every decision — ``recent_tools`` keeps only the last
    few calls. ``approved`` counts the holds the operator approved; ``unrecorded``
    the approvals a host reported that are not on record."""
    counts = {k: v for k, v in (tally.items() if isinstance(tally, dict) else ())
              if k in TOOL_DECISION_TALLY and isinstance(v, int) and not isinstance(v, bool)}
    answer = decided.get("answer")
    keys = [decided["decision"], *(["approved"] if answer == "approved" else []),
            *(["unrecorded"] if answer == APPROVAL_NOT_ON_RECORD else [])]
    return {**counts, **{k: counts.get(k, 0) + 1 for k in keys}}

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
PERMISSION_SUBJECT_CHARS = 300


def note_pending_permission(ref: Dict[str, Any], ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A session's permission question (PRD-235 W2 S3) → remembered on the ticket
    until the operator answers. Returns the stored entry, or None for a malformed
    event or one the operator already answered (the host re-sends a batch its
    POST lost; an answered question must not come back). A re-sent open question
    keeps the question row it already has (``grant_id``, PRD-245 S0.4)."""
    request_id = ev.get("request_id")
    if not request_id:
        return None
    if str(request_id) in (ref.get("permission_decisions") or {}):
        return None
    previous = pending_permission_entry(ref, request_id)
    entry = {
        "request_id": str(request_id),
        "tool": str(ev.get("tool_name") or "?")[:60],
        "subject": (str(ev["subject"])[:PERMISSION_SUBJECT_CHARS] if ev.get("subject") else None),
        "reason": str(ev.get("reason") or "")[:300],
        "at": _iso(_now()),
    }
    if previous is not None and previous.get("grant_id") is not None:
        entry["grant_id"] = previous["grant_id"]
    pending = [p for p in (ref.get("pending_permissions") or []) if p.get("request_id") != entry["request_id"]]
    pending.append(entry)
    ref["pending_permissions"] = pending[-PENDING_PERMISSIONS_KEPT:]
    return entry


def record_permission_decision(ref: Dict[str, Any], request_id: str, approved: bool, actor: str) -> bool:
    """The operator's answer. False when the question is unknown (already answered or expired).
    F167: it keeps what was asked (tool, command), so an approval the host reports
    can be matched to the question it answered."""
    pending = ref.get("pending_permissions") or []
    asked = next((p for p in pending if p.get("request_id") == str(request_id)), None)
    if asked is None:
        return False
    ref["pending_permissions"] = [p for p in pending if p.get("request_id") != str(request_id)]
    decisions = dict(ref.get("permission_decisions") or {})
    decisions[str(request_id)] = {"approved": bool(approved), "by": actor, "at": _iso(_now()), "delivered": False,
                                  "tool": asked.get("tool"), "subject": asked.get("subject")}
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
    """One denial as the ticket shows it: tool, stage, reason, the command or
    path it was about (never the whole tool input) and what it MEANS — its
    ``kind`` (PRD-245 S0.3, D6): only a hold puts the ticket in review."""
    if not isinstance(denial, dict):
        return {"tool": "?", "reason": str(denial)[:300], "kind": classify_denial(None, denial)}
    raw_input = denial.get("input") if isinstance(denial.get("input"), dict) else {}
    subject = raw_input.get("command") or raw_input.get("file_path") or raw_input.get("path")
    out: Dict[str, Any] = {
        "tool": str(denial.get("tool") or "?")[:60],
        "stage": str(denial.get("stage") or "")[:40],
        "reason": str(denial.get("reason") or "")[:300],
        "kind": classify_denial(denial.get("stage"), denial.get("reason")),
    }
    if subject:
        out["subject"] = str(subject)[:300]
    return out


def decide_session_permission(db: Session, task: BoardTask, request_id: str, approved: bool, actor: str) -> Dict[str, Any]:
    """PRD-235 W2 S3: the operator's answer to a session's permission question,
    recorded on the ticket and picked up by the host on its next event flush; the
    Canvas hears the outcome as a status line. PRD-245 S0.4: the hold's question
    row (Questions tab / Telegram) closes with the same answer, whichever surface
    answered first — the second answer finds no pending question and says so."""
    ref = dict(task.runtime_ref or {})
    if ref.get("runtime") != RUNTIME_CLI:
        raise LookupError("this task is not a Claude Code session")
    entry = pending_permission_entry(ref, request_id)
    if entry is None or not record_permission_decision(ref, request_id, approved, actor):
        raise LookupError(f"no pending permission question {request_id}")
    task.runtime_ref = ref
    _close_hold_question(db, entry.get("grant_id"), approved, actor)
    db.commit()
    publish_canvas_events(task.workspace_id, [
        _canvas_envelope(task.workspace_id, "canvas.session.status", {
            "source": "cli", "task_id": task.id, "session_id": ref.get("session_id"),
            "status": "running", "decision": {"request_id": str(request_id), "approved": bool(approved)},
        }),
    ])
    return {"task_id": task.id, "request_id": str(request_id), "approved": bool(approved), "pending": len(ref.get("pending_permissions") or [])}


def _shadow_session_end(
    task: Any, ref: Dict[str, Any], payload: Dict[str, Any], exec_result: Dict[str, Any],
    files: Any, denials: Any,
) -> None:
    """PRD-248 S5 (shadow only): the decision engine reads the session's final
    message and says whether the work is complete, whether nothing was done, and
    whether the owner is needed — logged beside the status the board is about to
    apply. Lazy, off by default, fail-open; the result is never touched."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine, judgements

        engine = get_decision_engine()
        if engine.dials().session_end_mode == MODE_OFF:
            return
        engine.shadow(
            judgements.shadow_session_end(
                engine,
                workspace_id=getattr(task, "workspace_id", None),
                task_id=getattr(task, "id", None),
                attempt=payload.get("attempt", ref.get("attempt")),
                title=getattr(task, "title", "") or "",
                description=getattr(task, "description", "") or "",
                final_text=str(payload.get("result_text") or payload.get("error") or ""),
                exit_reason=str(payload.get("exit_reason") or exec_result.get("status") or ""),
                files_touched=len(files) if isinstance(files, (list, tuple)) else 0,
                denials=len(denials) if isinstance(denials, (list, tuple)) else 0,
                platform_status=str(exec_result.get("status") or ""),
            ),
            purpose=judgements.PURPOSE_SESSION_END,
        )
    except Exception:  # noqa: BLE001 — never into a result
        logger.debug("[decision] session-end shadow skipped", exc_info=True)


async def apply_result(
    db: Session, host: CliHost, task_id: int, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Land a session's terminal result through the board's ONE completion writer.

    Idempotent per ``(task, attempt)``: a duplicate POST, a stale attempt, or a
    task that already left ``in_progress`` (cancelled, requeued, finished) is a
    no-op that says so. A HELD command the operator did not allow forces
    ``review`` — "couldn't run the tests" must never read as ``done`` (PRD-234
    §C1); a refused read outside the directory, a tool a session never has or a
    denied TUI prompt is recorded and does not (PRD-245 S0.3, D6).
    """
    from api.board_tasks import finalize_board_task_run

    task = _owned_task(db, host, task_id)
    ref = dict(task.runtime_ref or {})
    attempt = payload.get("attempt")
    # F211: a stale attempt's credential is already dead. Every claim builds a
    # fresh ref and mints its own token, so the hash on the row is the NEWER
    # claim's; revoking it here cut the live session off its platform tools.
    if attempt is not None and ref.get("attempt") is not None and int(attempt) != int(ref["attempt"]):
        return {"applied": False, "reason": "stale attempt", "status": task.status}
    # Whatever else is true, this host's run of this ticket is over, so its
    # credential dies here — BEFORE the early return. A result that arrives for
    # a ticket someone already moved used to leave the hash on the row with the
    # plaintext still in the transcript and in ``mcp.json``; the next flip back
    # to ``in_progress`` revived it.
    if task.status != "in_progress":
        if revoke_session_token(db, task):
            db.commit()
        return {"applied": False, "reason": f"task is {task.status}", "status": task.status}

    status = str(payload.get("status") or "success").lower()
    if status == "usage_limit":
        return _release_for_usage_limit(db, task, ref, payload)
    if status == "host_stopped":
        return _release_for_host_stop(db, task, ref, payload, host)
    denials = payload.get("permission_denials") or []
    denial_summaries = [_denial_summary(d) for d in denials]
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
    _shadow_session_end(task, ref, payload, exec_result, files, denials)
    ref.update(
        {
            "finished_at": _iso(_now()),
            "exit_reason": payload.get("exit_reason") or status,
            "files_touched": files,
            "usage": usage,
            "denials": len(denials),
            # The reasons, not just the count: a ticket in review must say WHY
            # ("'python3 hello.py' is outside this ticket's Bash allowlist"), and
            # each one's kind — only a hold is a reason for review.
            "permission_denials": denial_summaries[:MAX_DENIALS_KEPT],
        }
    )
    if payload.get("transcript_path"):
        ref["transcript_path"] = payload["transcript_path"]
    # PRD-245 S1.1: the session is over — its credential stops working. (The
    # lookup also requires ``in_progress``, so this is belt and braces.)
    clear_session_token(ref)
    # PRD-239: the directory the session really ran in (a git repo gets a
    # --worktree) wins over the configured one — it is where `claude --resume`
    # finds the transcript and where the editor links should open.
    if payload.get("effective_cwd"):
        _record_session_cwd(ref, task, str(payload["effective_cwd"]))
    # PRD-235 W2 S3: a question nobody answered before the session ended is stale —
    # its denial is already on the record (permission_denials); drop it from the
    # queue. PRD-245 S0.4: its question row (Questions tab / Telegram) expires too.
    stale = list(ref.get("pending_permissions") or [])
    if stale:
        ref["expired_permissions"] = (ref.get("expired_permissions") or []) + stale
        ref["pending_permissions"] = []
    _expire_hold_questions(db, task, stale, host)

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
        "tool_decisions": dict(ref.get("tool_decisions") or {}),
        "permission_denials": list(ref.get("permission_denials") or []),
    }
    # A concurrent ``answer_session_ask`` (the operator answered while the turn
    # was still running) commits ``session_asks`` between this function's top
    # read and this write. Fold that answer in before the whole-document write,
    # or the park below reads a stale ledger and blocks the ticket on a question
    # already answered — a ticket that then never resumes.
    ref = _merge_fresh_session_asks(db, task, ref)
    # F094: likewise a note appended meanwhile (the mission's verdict on a step
    # whose session outran the wait, a progress note) is kept.
    ref = _merge_fresh_session_notes(db, task, ref)
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

    # PRD-245 W2: the session asked the operator something. Its turn is over, but
    # its WORK is not: the ticket parks on the question instead of finishing, and
    # the answer re-queues it to resume this same Claude session. The result text,
    # the usage and the deliverables of this turn are already recorded above.
    parked = _park_for_answer(db, task, ref)
    if parked is not None:
        return {"applied": True, "status": parked}

    terminal = await finalize_board_task_run(
        db,
        task_id=task.id,
        workspace_id=str(task.workspace_id),
        agent_id=task.assigned_agent_id,
        exec_result=exec_result,
        review_mode=task.review_mode or "auto",
        # F209: finalize only while the ticket is still on the run this result was
        # written under; a redispatch and re-claim since the commit above make it
        # another run's (finalize re-reads the row under its lock).
        run_id=ref.get(RUN_ID_KEY),
        # PRD-245 S0.3 (D6): review only when a held command went unanswered or
        # was denied (an unclassifiable refusal counts as one — fail closed) —
        # or when the turn produced nothing at all (night 1, finding 14): no
        # result text, no files, no report is not a finished piece of work, and
        # closing it ``done`` is how sessions that could not file a report came
        # to look successful.
        force_review=forces_review(denial_summaries) or _produced_nothing(exec_result),
    )
    return {"applied": terminal is not None, "status": terminal or task.status}


def _release_for_usage_limit(db: Session, task: BoardTask, ref: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """F083: the CLI's plan window closed mid-turn. That is a pause, not a failed
    attempt: the ticket goes back to the queue with the claim's attempt refunded
    (a limit must never use up the two a ticket gets), its credential dies, and
    it says why and when it resumes. The host stops claiming for that CLI until
    then, so the ticket is not handed straight back to a closed window."""
    reason = str(payload.get("error") or "paused: usage limit")
    return _release_to_queue(db, task, ref, payload, exit_reason="usage_limit", reason=reason,
                             record=("paused", {"reason": reason, "resets_at": payload.get("resets_at")}))


def _release_for_host_stop(db: Session, task: BoardTask, ref: Dict[str, Any], payload: Dict[str, Any],
                           host: CliHost) -> Dict[str, Any]:
    """F015 (night 1): the CLI host stopped while this ticket ran — its service
    restarted, the host was reinstalled, the machine went down. Night 1 wrote
    those as ``cancelled`` with no one and no reason on them: tickets the owner
    never stopped. It is neither the owner's stop nor a failed attempt: the
    ticket goes back to the queue with the claim's attempt refunded, saying
    which host stopped and why, and the next claim picks it up."""
    reason = str(payload.get("error") or "the CLI host stopped")
    by = f"cli-host:{host.id}"
    logger.info("[cli-host] task %s back in the queue — %s (%s)", task.id, reason, by)
    return _release_to_queue(db, task, ref, payload, exit_reason="host_stopped", reason=reason,
                             record=("released", {"reason": reason, "by": by}))


def _release_to_queue(db: Session, task: BoardTask, ref: Dict[str, Any], payload: Dict[str, Any], *,
                      exit_reason: str, reason: str, record: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A claimed ticket back to ``assigned``: the claim's attempt refunded, the
    session's credential dead, the turn's tokens still booked, and ``record``
    (key, facts) on the ticket saying why."""
    clear_session_token(ref)
    now = _iso(_now())
    key, facts = record
    ref.update({"exit_reason": exit_reason, "finished_at": now, key: {**facts, "at": now}})
    task.runtime_ref = ref
    task.status = "assigned"
    task.lease_until = None
    task.attempts = max(0, int(task.attempts or 0) - 1)
    db.commit()
    # The turn's tokens before the release are still real spend (booked as the
    # error they used to be booked as).
    book_session_usage(
        task, ref, payload.get("usage") or {},
        status="error", request_type=LANE_BOARD_TASK, execution_id=f"board_task:{task.id}", error=reason,
    )
    return {"applied": True, "status": "assigned", "released": True, "reason": reason}


def _produced_nothing(exec_result: Dict[str, Any]) -> bool:
    """True when a successful-looking turn left no trace a human could read.

    A session that answered in text has ``result``; one that did work has files.
    Neither means the session opened, decided there was nothing to do, and ended
    — which belongs in ``review`` with the reason, not in ``done``.
    """
    if exec_result.get("status") != "success":
        return False       # error / cancelled already have their own endings
    return not (exec_result.get("result") or "").strip() and not exec_result.get("deliverables")


def _merge_fresh_session_asks(db: Session, task: BoardTask, ref: Dict[str, Any]) -> Dict[str, Any]:
    """``ref`` with any answer a concurrent request wrote to this row's
    ``session_asks`` folded in. Read-only re-select; returns ``ref`` unchanged
    on any error or when nothing new is there."""
    from sqlalchemy import text as sql_text

    try:
        row = db.execute(
            sql_text("SELECT runtime_ref FROM board_tasks WHERE id = :id"), {"id": int(task.id)}
        ).first()
    except Exception:  # noqa: BLE001 — a merge must never fail the result
        return ref
    fresh = (row[0] if row and isinstance(row[0], dict) else {}) or {}
    by_grant = {int(a.get("grant_id") or 0): a for a in fresh.get(SESSION_ASKS_KEY, []) if isinstance(a, dict)}
    if not by_grant:
        return ref
    merged: List[Dict[str, Any]] = []
    changed = False
    for ask in session_asks(ref):
        other = by_grant.get(int(ask.get("grant_id") or 0))
        if other and other.get("answered_at") and not ask.get("answered_at"):
            merged.append({**ask, "answer": other.get("answer"), "answered_at": other.get("answered_at")})
            changed = True
        else:
            merged.append(ask)
    return {**ref, SESSION_ASKS_KEY: merged} if changed else ref


def _merge_fresh_session_notes(db: Session, task: BoardTask, ref: Dict[str, Any]) -> Dict[str, Any]:
    """``ref`` with the notes appended to this row since ``ref`` was read. Notes
    are only ever appended, so a longer list on the row is this one plus the
    new ones. Read-only re-select; ``ref`` unchanged on any error."""
    from sqlalchemy import text as sql_text

    try:
        row = db.execute(
            sql_text("SELECT runtime_ref FROM board_tasks WHERE id = :id"), {"id": int(task.id)}
        ).first()
    except Exception:  # noqa: BLE001 — a merge must never fail the result
        return ref
    fresh = (row[0] if row and isinstance(row[0], dict) else {}) or {}
    theirs = fresh.get(SESSION_NOTES_KEY)
    mine = ref.get(SESSION_NOTES_KEY)
    if isinstance(theirs, list) and len(theirs) > (len(mine) if isinstance(mine, list) else 0):
        return {**ref, SESSION_NOTES_KEY: theirs}
    return ref


def append_session_note(db: Session, *, task_id: Any, workspace_id: Any, note: str, by: str) -> Dict[str, Any]:
    """Append one note to a ticket's ``session_notes`` in the caller's
    transaction; returns the entry. ONE key, in one statement (``jsonb_set``
    append) — never a whole-document write, which would clobber the host's
    concurrent event flush."""
    from sqlalchemy import text as sql_text

    entry = {"note": str(note)[:MAX_ASK_QUESTION_KEPT], "at": _iso(_now()), "by": by}
    db.execute(
        sql_text(
            """
            UPDATE board_tasks
               SET runtime_ref = jsonb_set(
                       COALESCE(runtime_ref, CAST('{}' AS jsonb)),
                       CAST(:path AS text[]),
                       COALESCE(runtime_ref -> :key, CAST('[]' AS jsonb)) || CAST(:entry AS jsonb),
                       true)
             WHERE id = :task_id AND workspace_id = :ws
            """
        ),
        {"path": "{%s}" % SESSION_NOTES_KEY, "key": SESSION_NOTES_KEY,
         "entry": json.dumps([entry]), "task_id": int(task_id), "ws": str(workspace_id)},
    )
    return entry


def record_session_note(db: Session, *, task_id: Any, workspace_id: Any,
                        agent_name: Optional[str], note: str) -> Dict[str, Any]:
    """Append a progress note to a running ticket, for the operator to read.

    ONE key, in one statement (``append_session_note``). The note also goes to
    the ticket's Code Canvas so the operator sees it live. Returns an
    executor-shaped result the session reads as ordinary tool output.
    """
    try:
        entry = append_session_note(db, task_id=task_id, workspace_id=workspace_id, note=note,
                                    by=agent_name or "the session")
        db.commit()
    except Exception as exc:  # noqa: BLE001 — the session reads the reason
        logger.warning("[cli-host] progress note not recorded for ticket #%s", task_id, exc_info=True)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return {"success": False, "error": f"the note could not be saved ({type(exc).__name__})"}
    publish_note_line(workspace_id, task_id, entry["note"])
    return {"success": True, "result": {"recorded": True, "note": entry["note"]}}


def publish_note_line(workspace_id: Any, task_id: Any, note: str) -> None:
    """A ticket's new note, live in its Code Canvas. Best-effort: the note is
    already on the ticket."""
    try:
        publish_canvas_events(workspace_id, [
            _canvas_envelope(workspace_id, "canvas.session.status", {
                "source": "cli", "task_id": int(task_id), "status": "running", "note": note,
            }),
        ])
    except Exception:  # noqa: BLE001 — the note is saved; the live line is best-effort
        logger.debug("[cli-host] note canvas line not published for ticket #%s", task_id, exc_info=True)


def requeue_exhausted(task: BoardTask) -> bool:
    """True once this ticket has been attempted more times than anyone should.

    The lease sweeper has always had ``BOARD_DISPATCH_MAX_ATTEMPTS``; the paths
    that put a ticket back on the board for a NON-lease reason — an answered
    ask, a resumed session — had no ceiling at all, which is how night 1
    re-dispatched one ticket 534 times. This is the backstop they share.
    """
    # getattr: a ticket that has never been claimed has recorded no attempts —
    # and the existing PRD-245 suites build tickets without the column. Reading
    # it directly (4109206c8) broke six of their tests.
    return int(getattr(task, "attempts", 0) or 0) >= int(config.BOARD_DISPATCH_HARD_ATTEMPT_CAP)


def park_exhausted(db: Session, task: BoardTask, why: str) -> str:
    """Send a ticket that has run out of attempts to a human, not round again."""
    task.status = "review"
    task.lease_until = None
    task.blocked_at = None
    task.blocked_reason = None
    task.completed_at = _now()
    task.review_feedback = (
        f"Stopped after {getattr(task, 'attempts', 0) or 0} attempts — {why}. "
        "Nothing was re-queued; this needs a person."
    )
    _notify_status(db, task)  # F119: before the commit that carries it
    db.commit()
    logger.warning(
        "[cli-host] ticket #%s hit the hard attempt cap (%s) — parked in review, not re-queued",
        task.id, config.BOARD_DISPATCH_HARD_ATTEMPT_CAP,
    )
    return task.status


def _park_for_answer(db: Session, task: BoardTask, ref: Dict[str, Any]) -> Optional[str]:
    """PRD-245 W2 — the turn ended with a question open, or with one answered
    while it ran.

    Returns the status written (``blocked`` or ``assigned``), or ``None`` when the
    session asked nothing and the normal completion writer should run.

    * an OPEN ask → ``blocked`` on the question; the answer re-queues it;
    * an ANSWERED ask (the operator was quick) → back to ``assigned``, because
      the work can carry on now.

    Either way the ticket keeps its session id, so the host RESUMES the same
    Claude Code session instead of starting a fresh one with no memory of the
    work so far."""
    asks = session_asks(ref)
    if not asks:
        return None
    open_asks = [a for a in asks if not a.get("answered_at")]
    # An ANSWERED ask is a reason to resume exactly ONCE — until its answer has
    # been folded into a prompt (``_answers_fold_in`` marks ``folded_at`` at the
    # claim that showed it). Without this check the ticket re-queued at the end
    # of EVERY later turn, because its ask ledger still held an answered entry:
    # night 1 re-dispatched three tickets 188 times between them, each launching
    # a fresh session that found the work already done and ended ("110th
    # dispatch, ticket unchanged… No action taken"). The dispatch cap does not
    # cover this path — it guards lease-expiry requeues.
    resumable = [a for a in asks if a.get("answered_at") and a.get("answer") and not a.get("folded_at")]
    if not open_asks and not resumable:
        return None
    if resumable and not open_asks and requeue_exhausted(task):
        return park_exhausted(db, task, "an answered question kept sending it back")
    ref = _mark_resumable(ref)
    if open_asks:
        task.status = "blocked"
        task.blocked_at = _now()
        task.blocked_reason = PARKED_FOR_ANSWER_REASON.format(grant_id=open_asks[0].get("grant_id"))
    else:
        task.status = "assigned"
        task.blocked_at = None
        task.blocked_reason = None
    task.lease_until = None            # not a running session any more
    task.runtime_ref = ref
    # F119: before the commit that carries them — apply_result's route returns
    # without another commit, and the request's rollback dropped them.
    _notify_status(db, task)
    if not open_asks:
        _notify_available(db, task)
    db.commit()
    logger.info("[cli-host] ticket #%s parks on %s ask(s) → %s",
                task.id, len(open_asks) or "answered", task.status)
    return task.status


def _mark_resumable(ref: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the next claim to CONTINUE this Claude Code session (``claude --resume``)
    on the host that ran it — a transcript lives on one machine."""
    session_id = ref.get("cli_session_id") or ref.get("session_id")
    if not session_id:
        return ref
    return {**ref, "resume_session_id": str(session_id), "resume_host_id": ref.get("host_id")}


def _notify_status(db: Session, task: BoardTask) -> None:
    try:
        notify_board_event(db, workspace_id=str(task.workspace_id), task_id=task.id,
                           status=task.status, event="task_updated")
    except Exception:  # noqa: BLE001
        logger.debug("[cli-host] board notify skipped for ticket #%s", task.id, exc_info=True)


def _notify_available(db: Session, task: BoardTask) -> None:
    try:
        from services.board_dispatcher import notify_task_available

        notify_task_available(db, workspace_id=str(task.workspace_id), task_id=task.id)
    except Exception:  # noqa: BLE001
        logger.debug("[cli-host] dispatch notify skipped for ticket #%s", task.id, exc_info=True)

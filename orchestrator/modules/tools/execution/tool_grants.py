"""PRD-193 (P2-12) — tool-call approval grants: the ask finally has a "yes".

One shared seam between the confirmation gate and the PRD-181 grant substrate:

* ``tool_call_subject_id`` / ``params_hash`` — the deterministic call key. The
  hash covers the canonical **model-provided** params only (server-injected
  ``_``-prefixed plumbing like ``_agent_id`` is stripped), so the ask, the
  human's approval, and the retried call all land on the SAME subject.
* ``issue_tool_grant``   — at the ask, create (or reuse) a PENDING
  ``ApprovalGrant`` scoped ``tool_call`` and announce it to the workspace's
  humans on non-chat lanes (chat sees the S3 card live — no double-notify).
* ``attach_ask_grant``   — enrich the executor's ask return with ``grant_id``,
  ``risk_class`` and the AI-Act oversight fields the S3 card renders.
* ``consume_tool_grant`` — immediately before the confirmation return, resolve
  an authorising grant for the same subject: GRANTED + unexpired + params-hash
  equality ⇒ the gate opens. Destructive grants are single-use — retired on use
  via ``revoke_grant(..., revoked_by="system:consumed")`` (locked decision:
  destructive ⇒ single-use exact-params; write-class ⇒ TTL window per call-key
  on the existing 24h ``APPROVAL_GRANT_TTL_SECONDS``).

Fail-safe posture (the whole point of this gate): every error on the issue or
consult side lands on the ask — never an exception, never an execution. This
module therefore never raises into the executor.

Subject-first by design: PRD-192's policy-plane ``ask`` verdicts route through
this same seam later (workspace + action + canonical params — no per-surface
fork). Reuses PRD-181's model/service wholesale: no new table, no migration,
no new config key.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Set
from uuid import UUID

logger = logging.getLogger(__name__)

# Actor ref recorded when a single-use (destructive) grant is retired on use.
GRANT_CONSUMED_BY = "system:consumed"

# Strong refs to fire-and-forget notification tasks (the board_approval /
# api/webhooks.py background-task idiom) so the loop cannot GC them mid-flight.
_NOTIFY_TASKS: Set["asyncio.Task"] = set()

# caller_context keys snapshotted into ``details`` so the S4 resume re-dispatch
# reproduces the original call's identity/telemetry posture — nothing more.
_CALLER_SNAPSHOT_KEYS = (
    "user_id",
    "system_role",
    "workspace_role",
    "conversation_id",
    "turn_id",
)


# ---------------------------------------------------------------------------
# The deterministic call key (pure)
# ---------------------------------------------------------------------------

def canonical_params(params: Any) -> Dict[str, Any]:
    """The model-provided view of a call's params.

    Server-injected plumbing (``_agent_id``, ``_agent_name``, ``_created_by``,
    …) is stripped so the ask and the retry produce the same key regardless of
    which lane injected what.
    """
    if not isinstance(params, dict):
        return {}
    return {k: v for k, v in params.items() if not str(k).startswith("_")}


def params_hash(params: Any) -> str:
    """Stable hash of the canonical params (sorted-key JSON, sha256)."""
    canon = canonical_params(params)
    try:
        raw = json.dumps(canon, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - json.default=str makes this rare
        raw = str(sorted(canon.items(), key=lambda kv: str(kv[0])))
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()


def tool_call_subject_id(workspace_id: UUID | str, action: str, params: Any) -> str:
    """Deterministic ``subject_id`` for a tool call: workspace + action + params.

    Human-readable prefix (the action) + digest — fits the 255-char column and
    keeps the P2-15 approvals queue debuggable.
    """
    digest = hashlib.sha256(
        "|".join([str(workspace_id), str(action or ""), params_hash(params)]).encode(
            "utf-8", "ignore"
        )
    ).hexdigest()
    return f"{action}:{digest[:32]}"


def _risk_class_for(action: str, permission_level: Optional[str]) -> str:
    """Pure risk classification (policy_document.classify_action), fail-safe."""
    try:
        from modules.policy.policy_document import classify_action

        return classify_action(action, permission_level=permission_level)
    except Exception:  # pragma: no cover - classifier is pure; import guard only
        logger.warning("[tool_grants] risk classification failed for %s", action, exc_info=True)
        return "destructive" if permission_level == "destructive" else "internal_write"


def _lane_for(caller_context: Optional[Dict[str, Any]]) -> str:
    """Coarse lane marker for the details snapshot + notification routing.

    Chat threads ``conversation_id`` (consumers/chatbot build_tool_caller_context);
    board runs thread ``board_task_id``; everything else is the agent lane
    (heartbeat / scheduled / mission).
    """
    ctx = caller_context or {}
    if ctx.get("conversation_id"):
        return "chat"
    if ctx.get("board_task_id") is not None:
        return "board"
    return "agent"


# ---------------------------------------------------------------------------
# Issue at the ask (S1)
# ---------------------------------------------------------------------------

def issue_tool_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    action: str,
    params: Any,
    permission_level: Optional[str] = None,
    description: Optional[str] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Create (or reuse) the PENDING ``tool_call`` grant for this exact call.

    Returns the grant, or ``None`` on any failure — the caller returns the ask
    either way (the ask is the floor). Never raises. Caller owns the txn
    (mirrors the PRD-181 service: stage + flush, never commit).
    """
    try:
        from core.models.approval_grants import SUBJECT_TOOL_CALL
        from core.services.approval_grants import create_grant, find_pending_grant

        subject_id = tool_call_subject_id(workspace_id, action, params)

        existing = find_pending_grant(
            db, workspace_id, subject_type=SUBJECT_TOOL_CALL, subject_id=subject_id
        )
        if existing is not None:
            # Idempotent ask: the pending row (already announced on creation)
            # is the actionable thing — no grant spam, no re-notify.
            return existing

        ctx = caller_context or {}
        agent_id: Optional[int] = None
        raw_agent = params.get("_agent_id") if isinstance(params, dict) else None
        try:
            agent_id = int(raw_agent) if raw_agent is not None else None
        except (TypeError, ValueError):
            agent_id = None

        risk_class = _risk_class_for(action, permission_level)
        lane = _lane_for(caller_context)

        grant = create_grant(
            db,
            workspace_id,
            subject_type=SUBJECT_TOOL_CALL,
            subject_id=subject_id,
            tool_name=action,
            risk_tier=risk_class,
            agent_id=agent_id,
            reason=(
                f"Confirmation required before running {action}: "
                f"{(description or 'gated platform action')[:200]}"
            ),
        )

        caller_snapshot = {
            k: ctx.get(k) for k in _CALLER_SNAPSHOT_KEYS if ctx.get(k) is not None
        }
        details: Dict[str, Any] = {
            "action": action,
            "params": canonical_params(params),
            "params_hash": params_hash(params),
            "lane": lane,
            "action_description": (description or "")[:300],
        }
        if caller_snapshot:
            details["caller_context"] = caller_snapshot
        if ctx.get("conversation_id"):
            details["conversation_id"] = ctx.get("conversation_id")
        if ctx.get("turn_id"):
            details["turn_id"] = ctx.get("turn_id")
        if ctx.get("board_task_id") is not None:
            details["board_task_id"] = ctx.get("board_task_id")
        grant.details = details

        # PRD-193 S5: an ask that fires with nobody watching must not be
        # silent — announce the fresh pending grant on non-chat lanes (chat
        # renders the S3 card live in the conversation; don't double-notify).
        if lane != "chat":
            _notify_approval_pending(grant, workspace_id)

        return grant
    except Exception:
        logger.warning(
            "[tool_grants] grant issuance failed for %s (ask still returned)",
            action, exc_info=True,
        )
        return None


def enrich_ask_with_grant(ask: Dict[str, Any], grant: Any, *, risk_class: str) -> Dict[str, Any]:
    """Return the ask WITH the grant + AI-Act oversight fields attached (pure).

    Field names mirror the PRD-163/181 mission card payload (``risk_class`` /
    ``risk_tier`` / ``oversight_rationale`` / ``requires_approval``) so the S3
    ``tool_approval`` card renders through the same presentation.
    """
    oversight: Dict[str, Any]
    try:
        from modules.policy.ai_act import OversightTier, oversight_for_risk

        mapping = oversight_for_risk(risk_class)
        oversight = mapping.to_dict()
        if mapping.tier != OversightTier.HUMAN_IN_THE_LOOP:
            # A gated ask awaiting approval is human-in-the-loop by definition
            # (the write-class gated actions classify on-the-loop) — floor the
            # tier so the card never implies "no human oversight", the same
            # posture as the mission card's ``_mission_oversight``. The true
            # risk_class is kept; the rationale states the gated-ask reality.
            oversight["tier"] = OversightTier.HUMAN_IN_THE_LOOP.value
            oversight["rationale"] = (
                "This action is confirmation-gated: a human must approve it "
                "before it runs."
            )
            oversight["requires_approval"] = True
    except Exception:  # pragma: no cover - pure import; guard only
        oversight = {
            "risk_class": risk_class or "unknown",
            "tier": "human_in_the_loop",
            "rationale": "This action requires human approval before it runs.",
            "requires_approval": True,
        }
    return {
        **ask,
        "grant_id": getattr(grant, "id", None),
        "risk_class": oversight.get("risk_class", risk_class),
        "risk_tier": oversight.get("tier"),
        "oversight_rationale": oversight.get("rationale"),
        "requires_approval": True,
    }


def attach_ask_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    action: str,
    params: Any,
    ask: Dict[str, Any],
    permission_level: Optional[str] = None,
    description: Optional[str] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Issue the grant and return the enriched ask; on ANY failure return the
    ask unchanged. The single fail-safe entry the confirmation gate calls."""
    try:
        grant = issue_tool_grant(
            db,
            workspace_id,
            action=action,
            params=params,
            permission_level=permission_level,
            description=description,
            caller_context=caller_context,
        )
        if grant is None:
            return ask
        return enrich_ask_with_grant(
            ask, grant, risk_class=_risk_class_for(action, permission_level)
        )
    except Exception:  # pragma: no cover - issue/enrich already guard; belt & braces
        logger.warning("[tool_grants] ask enrichment failed for %s", action, exc_info=True)
        return ask


# ---------------------------------------------------------------------------
# Consume at the yes (S2)
# ---------------------------------------------------------------------------

def consume_tool_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    action: str,
    params: Any,
    permission_level: Optional[str] = None,
) -> Optional[Any]:
    """Return the authorising grant for this exact call, or ``None``.

    Authorising = GRANTED + unexpired (``is_authorising``) + ``params_hash``
    equality (the grant authorises *the* call, not the tool). Destructive
    grants are single-use: retired here, on use, via ``revoke_grant`` with
    ``revoked_by="system:consumed"`` — reuse of the existing lifecycle, no new
    status, no schema change (locked decision 1). Write-class grants stay
    GRANTED for their TTL window per call-key.

    Never raises; any error ⇒ ``None`` ⇒ the ask stands. Fail-closed is the
    whole point — this gate fronts deletes / member removal / system settings.
    """
    try:
        from core.models.approval_grants import SUBJECT_TOOL_CALL
        from core.services.approval_grants import (
            find_active_grant,
            is_authorising,
            revoke_grant,
        )
        from modules.policy.policy_document import RISK_DESTRUCTIVE

        subject_id = tool_call_subject_id(workspace_id, action, params)
        grant = find_active_grant(
            db, workspace_id, subject_type=SUBJECT_TOOL_CALL, subject_id=subject_id
        )
        if grant is None or not is_authorising(grant):
            return None

        details = grant.details if isinstance(grant.details, dict) else {}
        if details.get("params_hash") != params_hash(params):
            # Params drifted from what the human approved — that is a new ask.
            return None

        stored_risk = grant.risk_tier or _risk_class_for(action, permission_level)
        if stored_risk == RISK_DESTRUCTIVE:
            # Single-use: the yes covered exactly one execution.
            revoke_grant(grant, revoked_by=GRANT_CONSUMED_BY)
        return grant
    except Exception:
        logger.warning(
            "[tool_grants] grant consult failed for %s — failing closed to the ask",
            action, exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Non-chat-lane notification (S5) — fire-and-forget, never blocks the ask
# ---------------------------------------------------------------------------

def _notify_approval_pending(grant: Any, workspace_id: UUID | str) -> None:
    """Schedule the ``approval_pending`` notification without blocking the gate.

    Mirrors ``services/board_approval``: dispatched off-transaction on the
    running loop with its own session; no loop (pure unit tests, sync callers)
    ⇒ skip silently. A dispatch fault never reaches the executor.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        task = loop.create_task(
            _dispatch_approval_pending(
                str(workspace_id),
                grant_id=getattr(grant, "id", None),
                tool_name=getattr(grant, "tool_name", None),
                risk_tier=getattr(grant, "risk_tier", None),
                reason=getattr(grant, "reason", None),
            )
        )
        _NOTIFY_TASKS.add(task)
        task.add_done_callback(_NOTIFY_TASKS.discard)
    except Exception:  # pragma: no cover - create_task on a live loop
        logger.warning("[tool_grants] approval_pending scheduling failed", exc_info=True)


async def _dispatch_approval_pending(
    workspace_id: str,
    *,
    grant_id: Any,
    tool_name: Optional[str],
    risk_tier: Optional[str],
    reason: Optional[str],
) -> None:
    """Dispatch ``approval_pending`` through the canonical notification seam.

    Owns its session: by the time the loop runs this, the creating caller's
    transaction is finished (and its session may be closed). The grant id rides
    ``link_id`` so the P2-15 approvals queue has a deep-link target.
    """
    from core.database.database import SessionLocal
    from core.services.notification_dispatcher import NotificationDispatcher

    db = SessionLocal()
    try:
        message = reason or f"A gated action ({tool_name or 'platform action'}) is awaiting approval."
        if risk_tier:
            message = f"{message} [risk: {risk_tier}]"
        await NotificationDispatcher(db, workspace_id).dispatch(
            event_type="approval_pending",
            title=f"Approval needed: {tool_name or 'platform action'}",
            message=message,
            link_type="approval_grant",
            link_id=str(grant_id) if grant_id is not None else None,
            status="action_required",
        )
    except Exception:
        logger.warning(
            "[tool_grants] approval_pending dispatch failed for grant %s",
            grant_id, exc_info=True,
        )
    finally:
        try:
            db.close()
        except Exception:  # pragma: no cover
            pass


__all__ = [
    "GRANT_CONSUMED_BY",
    "attach_ask_grant",
    "canonical_params",
    "consume_tool_grant",
    "enrich_ask_with_grant",
    "issue_tool_grant",
    "params_hash",
    "tool_call_subject_id",
]

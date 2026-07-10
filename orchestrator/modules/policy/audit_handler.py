"""Policy plane — the audit handler that attaches to the bus (PRD-181 S1).

This is the seam ``bus.py`` was built for ("audit + compaction policy can
attach later", bus.py:18). It makes the policy bus the **single write point**
for the per-tenant audit log: on every ``PRE_TOOL_USE`` fire, one ``AuditLog``
row records the tool, the tenant, the actor, the verdict (allow | ask | deny),
its reason, the risk tier, and — for a block — the policy error code.

Recording *every* verdict, including the allows, is deliberate: EU-AI-Act Art.12
is about traceable record-keeping of the system's automatic actions, and the
rest of Wave 11 (approval cards, oversight rationale, export) reads this table.
The handler is the only thing that writes policy verdicts to audit, so there is
no double-logging.

Contract with the bus (bus.py):
- a handler returns ``Optional[Verdict]``; audit is a **side-effect**, never a
  policy opinion, so it always returns ``None``.
- a handler that raises is treated as no-opinion, but we still catch here so a
  DB fault never even reaches the bus's warning path — the audit write must
  never wedge or slow the tool loop, and never turn a fault into a surprise.

Kept import-light: the DB session and the ``AuditService`` are only imported
inside the write, so this module loads stdlib-only for the unit tests.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from modules.policy.bus import EventContext
from modules.policy.types import Event, Verdict

logger = logging.getLogger(__name__)

# Actor types recorded in `audit_logs.actor_type` (mirrors AuditLog).
ACTOR_USER = "user"
ACTOR_AGENT = "agent"
ACTOR_SYSTEM = "system"


def _resolve_actor(ctx: EventContext) -> tuple[Optional[int], str]:
    """Resolve ``(user_id, actor_type)`` for this call.

    - A human caller (``caller_context.user_id``) ⇒ ``(id, 'user')``.
    - No human, but a real agent id ⇒ ``(None, 'agent')``.
    - Neither ⇒ ``(None, 'system')`` (heartbeat / scheduled / factory).
    """
    cc = ctx.caller_context or {}
    raw_user = cc.get("user_id")
    if raw_user is not None:
        try:
            return int(raw_user), ACTOR_USER
        except (TypeError, ValueError):
            # A non-int principal (e.g. a Clerk string id) — record as a user
            # actor but keep the raw value in details, not the FK column.
            return None, ACTOR_USER
    if ctx.agent_id:  # 0 / None ⇒ not a real agent
        return None, ACTOR_AGENT
    return None, ACTOR_SYSTEM


def audit_policy_verdict(
    db: Any, event: Event, ctx: EventContext
) -> Optional[Any]:
    """Write one ``AuditLog`` row for the verdict carried on ``ctx.data``.

    Returns the created row (for tests) or ``None`` when there is nothing to
    record (no verdict on the context) or the write failed. Never raises.
    """
    verdict: Optional[Verdict] = ctx.data.get("verdict")
    if verdict is None:
        # A non-policy event (or a fire that carried no verdict) — nothing to
        # record. Not an error: the bus fires other events too.
        return None

    try:
        from core.workspaces.audit import AuditService

        user_id, actor_type = _resolve_actor(ctx)
        decision = verdict.decision.value  # "allow" | "ask" | "deny" | "defer"

        cc = ctx.caller_context or {}
        details = {
            "verdict": decision,
            "reason": verdict.reason,
            "risk": ctx.data.get("risk"),
            # PRD-192 S1/S2: which stage of the mode dial produced this row
            # (off | shadow | destructive | on) — the shadow report aggregates
            # on it and stage advancement is judged against it.
            "mode": ctx.data.get("mode"),
            "actor_type": actor_type,
            "agent_id": ctx.agent_id,
            "event": event.value,
            "trace_id": ctx.data.get("trace_id"),
            "system_role": cc.get("system_role"),
            "workspace_role": cc.get("workspace_role"),
            # A non-int / string principal (Clerk id) is preserved here rather
            # than in the integer FK column.
            "raw_user_id": cc.get("user_id") if user_id is None and actor_type == ACTOR_USER else None,
            "error_code": verdict.error.code if verdict.error is not None else None,
        }

        return AuditService(db).log(
            workspace_id=str(ctx.workspace_id) if ctx.workspace_id is not None else None,
            user_id=user_id,
            actor_type=actor_type,
            action=f"policy:{decision}",
            resource_type="tool",
            resource_id=None,
            resource_name=ctx.tool_name,
            details=details,
        )
    except Exception:
        # A fault in audit must never wedge or slow tool execution.
        logger.warning(
            "[policy.audit] failed to record verdict for tool=%s ws=%s",
            ctx.tool_name, ctx.workspace_id, exc_info=True,
        )
        return None


_AUDIT_HANDLER_REGISTERED = False


def register_audit_handler(session_factory: Optional[Callable[[], Any]] = None) -> bool:
    """Attach the audit handler to the process-wide policy bus, once.

    Idempotent: safe to call on every startup / worker boot. Returns ``True`` if
    it registered on this call, ``False`` if already registered. ``session_factory``
    defaults to the app's ``SessionLocal`` (imported lazily so this module stays
    stdlib-only for tests).
    """
    global _AUDIT_HANDLER_REGISTERED
    if _AUDIT_HANDLER_REGISTERED:
        return False

    if session_factory is None:
        from core.database.database import SessionLocal as session_factory  # type: ignore

    from modules.policy.bus import get_policy_bus

    get_policy_bus().register(Event.PRE_TOOL_USE, make_audit_handler(session_factory))
    _AUDIT_HANDLER_REGISTERED = True
    logger.info("[policy.audit] audit handler attached to the policy bus (Art.12 record-keeping)")
    return True


def make_audit_handler(
    session_factory: Callable[[], Any],
):
    """Build a bus handler that records verdicts using a fresh DB session.

    ``session_factory`` returns a live sync session (e.g. ``SessionLocal``).
    The handler always returns ``None`` (audit is never a policy opinion) and
    never raises into the bus.
    """

    def _handler(event: Event, ctx: EventContext) -> Optional[Verdict]:
        db = None
        try:
            db = session_factory()
            audit_policy_verdict(db, event, ctx)
        except Exception:
            logger.warning("[policy.audit] handler failed", exc_info=True)
        finally:
            # Only close sessions we own and that expose close().
            if db is not None and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:  # pragma: no cover
                    pass
        return None  # audit is a side-effect, never a verdict

    return _handler

"""PRD-181 S2 — approval-grant lifecycle (create / grant / revoke / expire).

The service layer over :class:`core.models.approval_grants.ApprovalGrant`. Kept
small and side-effect-explicit: the caller owns the transaction (these stage +
flush; they never commit), mirroring ``approval_policy`` / ``policy_document``.

Authorisation semantics (``is_authorising``): a grant authorises its subject iff
it is ``GRANTED`` **and** not past ``expires_at``. A ``PENDING`` grant blocks (the
subject waits); ``REVOKED`` / ``EXPIRED`` / ``DENIED`` never authorise. Expiry is
evaluated at read time so a missed sweep can never leave a stale authorisation
live — the clock, not a background job, is the source of truth.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_APPROVAL

logger = logging.getLogger(__name__)

# Default grant TTL — a pending approval that no human touches lapses so it can't
# authorise indefinitely once granted. Sourced from config (no hardcoded values);
# falls back to 24h if config is unavailable (e.g. stdlib-only test import).
def _default_ttl_seconds() -> int:
    try:
        from config import config

        return int(config.APPROVAL_GRANT_TTL_SECONDS)
    except Exception:
        return 24 * 3600


DEFAULT_TTL_SECONDS = _default_ttl_seconds()


def _now(now: Optional[datetime] = None) -> datetime:
    return now or datetime.now(timezone.utc)


def create_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    subject_type: str,
    subject_id: str,
    tool_name: Optional[str] = None,
    risk_tier: Optional[str] = None,
    agent_id: Optional[int] = None,
    reason: Optional[str] = None,
    estimated_cost_usd: Optional[float] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: Optional[datetime] = None,
    # PRD-225: question-kind asks. Defaulting ``kind`` to 'approval' and the ask
    # fields to None keeps every existing approval caller byte-for-byte the same.
    kind: str = KIND_APPROVAL,
    question_md: Optional[str] = None,
    options: Optional[list] = None,
    channel_refs: Optional[dict] = None,
    asked_by_agent_id: Optional[int] = None,
    details: Optional[dict] = None,
) -> ApprovalGrant:
    """Stage a new PENDING, expiring grant for a subject. Caller owns the txn.

    PRD-225: pass ``kind='question'`` with ``question_md`` to stage a free-text
    ask (``pending`` = open, ``granted`` = answered, ``denied`` = dismissed).
    PRD-245: ``details`` seeds the row's JSONB (a session hold's marker); left
    out, the column keeps its default exactly as before.
    """
    ts = _now(now)
    grant = ApprovalGrant(
        workspace_id=workspace_id,
        subject_type=subject_type,
        subject_id=str(subject_id),
        tool_name=tool_name,
        risk_tier=risk_tier,
        agent_id=agent_id,
        status=GrantStatus.PENDING.value,
        reason=reason,
        estimated_cost_usd=(f"{float(estimated_cost_usd):.6f}" if estimated_cost_usd is not None else None),
        requested_at=ts,
        expires_at=ts + timedelta(seconds=max(1, int(ttl_seconds))),
        kind=kind,
        question_md=question_md,
        options=options,
        channel_refs=channel_refs,
        asked_by_agent_id=asked_by_agent_id,
    )
    if details:
        grant.details = dict(details)
    db.add(grant)
    try:
        db.flush()
    except Exception:  # pragma: no cover - flush is a no-op in some fakes
        logger.debug("[approval_grants] flush skipped", exc_info=True)
    _shadow_hold(grant, workspace_id)
    return grant


def _shadow_hold(grant: ApprovalGrant, workspace_id: Any) -> None:
    """PRD-248 S5 (shadow only): the decision engine scores what is being asked
    for — blast radius, intent, whether a non-technical owner could judge it —
    and logs it with the grant id so the human's eventual answer can be joined.
    Lazy, off by default, fail-open; it never grants, denies, or delays."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine, judgements

        engine = get_decision_engine()
        if engine.dials().hold_risk_mode == MODE_OFF:
            return
        engine.shadow(
            judgements.shadow_hold(
                engine,
                workspace_id=workspace_id,
                grant_id=getattr(grant, "id", None),
                kind=str(getattr(grant, "kind", "") or ""),
                subject_type=str(getattr(grant, "subject_type", "") or ""),
                subject_id=getattr(grant, "subject_id", None),
                tool_name=getattr(grant, "tool_name", None),
                risk_tier=getattr(grant, "risk_tier", None),
                question_md=getattr(grant, "question_md", None) or getattr(grant, "reason", None),
                options=getattr(grant, "options", None),
                reason=getattr(grant, "reason", None),
                agent_id=getattr(grant, "agent_id", None) or getattr(grant, "asked_by_agent_id", None),
            ),
            purpose=judgements.PURPOSE_HOLD,
        )
    except Exception:  # noqa: BLE001 — never into a grant
        logger.debug("[decision] hold shadow skipped", exc_info=True)


def find_active_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    subject_type: str,
    subject_id: str,
    now: Optional[datetime] = None,
) -> Optional[ApprovalGrant]:
    """Return the most recent GRANTED-and-unexpired grant for a subject, or None.

    A PENDING grant is *not* active (it blocks); only a live GRANTED one authorises.
    """
    ts = _now(now)
    try:
        rows = (
            db.query(ApprovalGrant)
            .filter(
                ApprovalGrant.workspace_id == workspace_id,
                ApprovalGrant.subject_type == subject_type,
                ApprovalGrant.subject_id == str(subject_id),
                ApprovalGrant.status == GrantStatus.GRANTED.value,
            )
            .order_by(ApprovalGrant.granted_at.desc())
            .all()
        )
    except Exception:
        logger.warning("[approval_grants] active-grant read failed", exc_info=True)
        return None
    for g in rows:
        if is_authorising(g, now=ts):
            return g
    return None


def find_pending_grant(
    db: Any,
    workspace_id: UUID | str,
    *,
    subject_type: str,
    subject_id: str,
) -> Optional[ApprovalGrant]:
    """Return an existing PENDING grant for a subject (idempotency guard), or None."""
    try:
        return (
            db.query(ApprovalGrant)
            .filter(
                ApprovalGrant.workspace_id == workspace_id,
                ApprovalGrant.subject_type == subject_type,
                ApprovalGrant.subject_id == str(subject_id),
                ApprovalGrant.status == GrantStatus.PENDING.value,
            )
            .order_by(ApprovalGrant.requested_at.desc())
            .first()
        )
    except Exception:
        logger.warning("[approval_grants] pending-grant read failed", exc_info=True)
        return None


def grant_grant(grant: ApprovalGrant, *, granted_by: str, now: Optional[datetime] = None) -> ApprovalGrant:
    """Approve a PENDING grant (a human said yes). Mutates the row in place."""
    grant.status = GrantStatus.GRANTED.value
    grant.granted_at = _now(now)
    grant.granted_by = granted_by
    return grant


def deny_grant(grant: ApprovalGrant, *, revoked_by: str, now: Optional[datetime] = None) -> ApprovalGrant:
    """A human refused. The subject fails rather than retrying."""
    grant.status = GrantStatus.DENIED.value
    grant.revoked_at = _now(now)
    grant.revoked_by = revoked_by
    return grant


def revoke_grant(grant: ApprovalGrant, *, revoked_by: str, now: Optional[datetime] = None) -> ApprovalGrant:
    """Retract a grant before expiry. A GRANTED grant stops authorising at once."""
    grant.status = GrantStatus.REVOKED.value
    grant.revoked_at = _now(now)
    grant.revoked_by = revoked_by
    return grant


def answer_pending_grant(
    db: Any, grant_id: int, *, answer_text: str, answered_by: str, now: Optional[datetime] = None,
) -> bool:
    """PRD-225: flip ONE question ``pending → granted`` with its answer, atomically.

    ``UPDATE … WHERE id = :id AND status = 'pending'`` is the compare-and-swap
    that lets two concurrent answers apply exactly once (P225-RVW-14): the loser
    matches 0 rows and this returns ``False``. ``synchronize_session=False`` —
    the caller syncs its in-memory row when it won. Caller owns the txn.
    """
    flipped = (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.id == grant_id,
            ApprovalGrant.status == GrantStatus.PENDING.value,
        )
        .update(
            {
                ApprovalGrant.status: GrantStatus.GRANTED.value,
                ApprovalGrant.answer_text: answer_text,
                ApprovalGrant.answered_by: answered_by,
                ApprovalGrant.answered_at: _now(now),
            },
            synchronize_session=False,
        )
    )
    return bool(flipped)


def expire_pending_grants(
    db: Any, workspace_id: UUID | str, grant_ids: Any, *, revoked_by: str, now: Optional[datetime] = None,
) -> int:
    """PRD-245 S0.4: close PENDING rows nobody can answer any more — the session
    that held the command has ended (``EXPIRED``, the lazily-set status the
    model always allowed). One filtered UPDATE guarded on ``pending`` so an
    answer that won the race a moment earlier is never overwritten. Returns
    the number of rows closed. Caller owns the txn.
    """
    ids = sorted({int(g) for g in (grant_ids or []) if g is not None})
    if not ids:
        return 0
    closed = (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == workspace_id,
            ApprovalGrant.id.in_(ids),
            ApprovalGrant.status == GrantStatus.PENDING.value,
        )
        .update(
            {
                ApprovalGrant.status: GrantStatus.EXPIRED.value,
                ApprovalGrant.revoked_at: _now(now),
                ApprovalGrant.revoked_by: revoked_by,
            },
            synchronize_session=False,
        )
    )
    return int(closed or 0)


def is_authorising(grant: ApprovalGrant, *, now: Optional[datetime] = None) -> bool:
    """True iff this grant currently authorises its subject.

    GRANTED and not past ``expires_at``. Expiry is evaluated here (read time) so
    a missed sweep can never leave a stale authorisation live.
    """
    if grant.status != GrantStatus.GRANTED.value:
        return False
    if grant.expires_at is not None:
        exp = grant.expires_at
        # tolerate naive datetimes from a fake/session by assuming UTC
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if _now(now) >= exp:
            return False
    return True

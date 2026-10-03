"""PRD-252 R5 — one "Needs you" number: what waits for the owner, counted once.

Five counters had five definitions. The Board tab badge counted every open
ticket; ATTENTION counted a grant-blocked ticket twice (the ticket and its
grant); the Needs you widget left out failures; Auto's pill counted questions
and "decisions", read from a super-admin-only endpoint. Needs you is now:

* tickets in Review that are the owner's to judge. A mission step in Review
  waits for its mission's own check. A mission's card in Review is the mission
  waiting for its plan's approval, so it is counted once, as an approval;
* open questions;
* pending approvals: approval grants, and missions waiting for their plan's
  approval;
* tickets that failed in the selected period (a failed mission step is its
  mission's to handle).

Questions and approval grants are answered by workspace admins only (the
grants API), so for anyone else they are neither counted nor listed: each
viewer's number is the number of rows they can open. ``needs_you`` serves the
count and the rows; ATTENTION and every badge read the same count.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.orm import Session

PERIOD_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
DEFAULT_PERIOD = "1d"
# Rows listed per kind: every one, in practice (F225: 25 listed 35 rows for 43). A
# kind with more says how many more; the count is always exact.
ROWS_PER_KIND = 200
KINDS = ("review", "question", "approval", "failed")
# A mission's own card on the board (orchestration_board_bridge).
MISSION_CARD = "orchestration"

# Each kind's count, scoped to the workspace. A pending grant stays open until it
# is answered, exactly as the Questions tab lists it.
_COUNTS = text("""
    SELECT
      (SELECT COUNT(*) FROM board_tasks
        WHERE workspace_id = CAST(:ws AS uuid) AND status = 'review'
          AND source_type NOT IN ('orchestration_task', 'orchestration')) AS review,
      (SELECT COUNT(*) FROM approval_grants
        WHERE workspace_id = CAST(:ws AS uuid) AND status = 'pending' AND kind = 'question'
          AND CAST(:asks AS boolean)) AS question,
      (SELECT COUNT(*) FROM approval_grants
        WHERE workspace_id = CAST(:ws AS uuid) AND status = 'pending' AND COALESCE(kind, 'approval') <> 'question'
          AND CAST(:asks AS boolean))
      + (SELECT COUNT(*) FROM orchestration_runs
        WHERE workspace_id = CAST(:ws AS uuid) AND state = 'awaiting_approval') AS approval,
      (SELECT COUNT(*) FROM board_tasks
        WHERE workspace_id = CAST(:ws AS uuid) AND status = 'failed' AND source_type <> 'orchestration_task'
          AND COALESCE(completed_at, updated_at) >= :since) AS failed
""")

_REVIEW_ROWS = text("""
    SELECT bt.id, bt.title, bt.source_type, bt.orchestration_run_id, a.name AS agent_name,
           COALESCE(bt.completed_at, bt.updated_at) AS at
      FROM board_tasks bt LEFT JOIN agents a ON a.id = bt.assigned_agent_id AND a.workspace_id = bt.workspace_id
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'review'
       AND bt.source_type NOT IN ('orchestration_task', 'orchestration')
  ORDER BY at DESC NULLS LAST LIMIT :limit
""")
# A failed mission card opens its mission: its run id rides along.
_FAILED_ROWS = text("""
    SELECT bt.id, bt.title, bt.source_type, bt.orchestration_run_id, a.name AS agent_name,
           COALESCE(bt.completed_at, bt.updated_at) AS at
      FROM board_tasks bt LEFT JOIN agents a ON a.id = bt.assigned_agent_id AND a.workspace_id = bt.workspace_id
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'failed' AND bt.source_type <> 'orchestration_task'
       AND COALESCE(bt.completed_at, bt.updated_at) >= :since
  ORDER BY at DESC NULLS LAST LIMIT :limit
""")
_MISSION_ROWS = text("""
    SELECT id, goal, updated_at FROM orchestration_runs
     WHERE workspace_id = CAST(:ws AS uuid) AND state = 'awaiting_approval'
  ORDER BY updated_at DESC NULLS LAST LIMIT :limit
""")


def normal_period(period: str) -> str:
    """``period`` when it is one the counters know, else the default (a day)."""
    return period if period in PERIOD_DAYS else DEFAULT_PERIOD


def period_start(period: str) -> datetime:
    """The start of ``period`` ('1d', '7d', '30d', '90d'; anything else is a day)."""
    return datetime.now(timezone.utc) - timedelta(days=PERIOD_DAYS[normal_period(period)])


def needs_you_counts(db: Session, workspace_id: Any, since: datetime, *, may_answer: bool = True) -> Dict[str, int]:
    """Each kind's count and their ``total``: the one Needs-you number.
    ``may_answer`` False (not a workspace admin) leaves questions and approval
    grants out, since that viewer can neither open nor answer them."""
    row = db.execute(_COUNTS, {"ws": str(workspace_id), "since": since, "asks": bool(may_answer)}).first()
    counts = {kind: int(getattr(row, kind, 0) or 0) for kind in KINDS} if row else dict.fromkeys(KINDS, 0)
    return {**counts, "total": sum(counts.values())}


def needs_you(db: Session, workspace_id: Any, period: str = DEFAULT_PERIOD, *, may_answer: bool = True) -> Dict[str, Any]:
    """The number and the rows behind it, newest first in each kind."""
    period = normal_period(period)
    since = period_start(period)
    params = {"ws": str(workspace_id), "since": since, "limit": ROWS_PER_KIND}
    questions = _pending_grants(db, workspace_id, questions=True) if may_answer else []
    approvals = _pending_grants(db, workspace_id, questions=False) if may_answer else []
    rows: Dict[str, List[Dict[str, Any]]] = {
        "review": [_ticket_row(r) for r in db.execute(_REVIEW_ROWS, params)],
        "question": _grant_rows(db, workspace_id, questions),
        "approval": _approval_rows(db, workspace_id, approvals, params),
        "failed": [_ticket_row(r) for r in db.execute(_FAILED_ROWS, params)],
    }
    counts = needs_you_counts(db, workspace_id, since, may_answer=may_answer)
    return {"period": period, "total": counts.pop("total"), "counts": counts, "rows": rows}


def _pending_grants(db: Session, workspace_id: Any, *, questions: bool) -> List[Any]:
    """The newest open questions (``questions``) or pending approval grants."""
    from sqlalchemy import func

    from core.models.approval_grants import KIND_APPROVAL, KIND_QUESTION, ApprovalGrant, GrantStatus

    kind = func.coalesce(ApprovalGrant.kind, KIND_APPROVAL)
    return (
        db.query(ApprovalGrant)
        .filter(ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.status == GrantStatus.PENDING.value,
                kind == KIND_QUESTION if questions else kind != KIND_QUESTION)
        .order_by(ApprovalGrant.requested_at.desc())
        .limit(ROWS_PER_KIND)
        .all()
    )


def _grant_rows(db: Session, workspace_id: Any, grants: List[Any]) -> List[Dict[str, Any]]:
    """A question or approval grant as a row: the ticket it opens in, and who asked (F091-E1)."""
    from core.models.approval_grants import KIND_QUESTION
    from services.grant_owners import grant_owners

    owners = grant_owners(db, workspace_id, grants)
    rows = []
    for g in grants:
        owner = owners.get(g.id) or {}
        rows.append({
            "source": "grant",
            "id": str(g.id),
            "title": g.question_md if g.kind == KIND_QUESTION else (g.reason or g.tool_name or "Approval"),
            "ticket_id": (owner.get("ticket") or {}).get("id"),
            "agent_name": (owner.get("agent") or {}).get("name"),
            "at": _iso(g.requested_at),
        })
    return rows


def _approval_rows(db: Session, workspace_id: Any, grants: List[Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Approval grants and missions waiting for their plan's approval, newest first."""
    missions = [
        {"source": "mission", "id": str(r.id), "title": r.goal, "ticket_id": None, "agent_name": None,
         "at": _iso(r.updated_at)}
        for r in db.execute(_MISSION_ROWS, params)
    ]
    merged = _grant_rows(db, workspace_id, grants) + missions
    return sorted(merged, key=lambda row: row["at"] or "", reverse=True)[:ROWS_PER_KIND]


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _ticket_row(r: Any) -> Dict[str, Any]:
    """A ticket opens in the board's viewer; only a mission's own card opens its
    mission. A session-run mission step also carries the run id, and opens itself."""
    mission = str(r.orchestration_run_id) if r.source_type == MISSION_CARD and r.orchestration_run_id else None
    return {"ticket_id": r.id, "title": r.title, "agent_name": r.agent_name, "mission_id": mission, "at": _iso(r.at)}

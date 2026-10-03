"""PRD-252 R5 — one "Needs you" number: what waits for the owner, counted once.

Five counters had five definitions. The Board tab badge counted every open
ticket; ATTENTION counted a grant-blocked ticket twice (the ticket and its
grant); the Needs you widget left out failures; Auto's pill counted questions
and "decisions", read from a super-admin-only endpoint. Needs you is now:

* tickets in Review that are the owner's to judge. A mission step in Review
  waits for its mission's own check. A mission's card in Review is the mission
  waiting for its plan's approval, so it is counted once, as an approval;
* open questions;
* pending approvals that can still be given: approval grants that have not
  lapsed, and missions waiting for their plan's approval (each with its card's
  number);
* stuck tickets: ones nothing will move until the owner does (F246);
* failed tickets, until the owner deals with them (a failed mission step is its
  mission's to handle).

F246 (night 7): Needs you missed whatever was not a plain Review or failed
card. A failure dropped out after a day though nobody had dealt with it;
approvals stayed after they lapsed or after their card was cancelled; mission
plans had no card number; and stuck cards were never counted.

Questions and approval grants are answered by workspace admins only (the
grants API), so for anyone else they are neither counted nor listed: each
viewer's number is the number of rows they can open. ``needs_you`` serves the
count and the rows; ATTENTION and every badge read the same count.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

# Rows listed per kind: every one, in practice (F225: 25 listed 35 rows for 43). A
# kind with more says how many more; the count is always exact.
ROWS_PER_KIND = 200
KINDS = ("review", "question", "approval", "stuck", "failed")
# A mission's own card on the board (orchestration_board_bridge).
MISSION_CARD = "orchestration"
# A step card still waiting for its mission to run it.
OPEN_STEP_STATUSES = ["inbox", "assigned", "in_progress", "review", "blocked"]
# Why a stuck ticket is stuck: the `why` _STUCK_ROWS gives each one.
STUCK_NO_HOST = "no_host"                # a CLI agent's ticket, and no host that runs its CLI is online
STUCK_NO_AGENT = "no_agent"              # Assigned to nobody
STUCK_NOT_PICKED_UP = "not_picked_up"    # a playbook's card in Assigned: the board never runs one
STUCK_MISSION_ENDED = "mission_ended"    # a step whose mission ended without it

# Review, failed and mission-approval counts, scoped to the workspace. A failed
# ticket counts until it leaves Failed (F246: a '1d' window let #0003, #0004,
# #0050 and #0052 drop out though nobody had dealt with them).
_COUNTS = text("""
    SELECT
      (SELECT COUNT(*) FROM board_tasks bt
        WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'review'
          AND (bt.source_type NOT IN ('orchestration_task', 'orchestration')
               OR (bt.source_type = 'orchestration_task' AND bt.review_mode = 'human' AND NOT EXISTS (
                   SELECT 1 FROM orchestration_tasks ot JOIN orchestration_runs r ON r.id = ot.run_id
                    WHERE ot.id = bt.orchestration_task_id AND r.state = ANY(:ended))))) AS review,
      (SELECT COUNT(*) FROM orchestration_runs
        WHERE workspace_id = CAST(:ws AS uuid) AND state = 'awaiting_approval') AS mission_approval,
      (SELECT COUNT(*) FROM board_tasks
        WHERE workspace_id = CAST(:ws AS uuid) AND status = 'failed' AND source_type <> 'orchestration_task') AS failed
""")

# Open questions, or approvals that can still be given (``:questions`` picks the
# kind), newest first; ``of_all`` is the exact count whatever ``:limit`` lists.
# A question stays open past its expiry, as the Questions tab lists it. An
# approval past its expiry can no longer authorise anything (is_authorising), so
# it is not waiting for anyone (F246: #999 and #1000, lapsed six days). Neither
# counts once the ticket it is about is cancelled or closed (#1138 for #0093,
# #0144, #0166): the ticket a grant belongs to is the one grant_owners names.
_ASKS = text("""
    SELECT g.id, COUNT(*) OVER () AS of_all
      FROM approval_grants g
     WHERE g.workspace_id = CAST(:ws AS uuid) AND g.status = 'pending'
       AND (COALESCE(g.kind, 'approval') = 'question') = CAST(:questions AS boolean)
       AND (COALESCE(g.kind, 'approval') = 'question' OR g.expires_at IS NULL OR g.expires_at > :now)
       AND NOT EXISTS (
         SELECT 1 FROM board_tasks gone
          WHERE gone.workspace_id = g.workspace_id AND gone.status IN ('cancelled', 'closed')
            AND ((g.subject_type = 'board_task' AND CAST(gone.id AS text) = g.subject_id)
              OR CAST(gone.id AS text) = g.details->>'board_task_id'
              OR (g.subject_type = 'tool_call' AND CAST(gone.orchestration_task_id AS text) = g.subject_id)))
  ORDER BY g.requested_at DESC NULLS LAST LIMIT :limit
""")

# A mission step in Review is its mission's to check, unless the owner asked to
# check it (F242: review_mode human, a step held for the owner while its mission
# waits). A step whose mission ended can no longer be let through: it is stuck.
_REVIEW_ROWS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.source_type, bt.parent_task_id, bt.orchestration_run_id,
           a.name AS agent_name, COALESCE(bt.completed_at, bt.updated_at) AS at
      FROM board_tasks bt LEFT JOIN agents a ON a.id = bt.assigned_agent_id AND a.workspace_id = bt.workspace_id
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'review'
       AND (bt.source_type NOT IN ('orchestration_task', 'orchestration')
            OR (bt.source_type = 'orchestration_task' AND bt.review_mode = 'human' AND NOT EXISTS (
                SELECT 1 FROM orchestration_tasks ot JOIN orchestration_runs r ON r.id = ot.run_id
                 WHERE ot.id = bt.orchestration_task_id AND r.state = ANY(:ended))))
  ORDER BY at DESC NULLS LAST LIMIT :limit
""")
# A failed mission card opens its mission: its run id rides along.
_FAILED_ROWS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.source_type, bt.orchestration_run_id, a.name AS agent_name,
           COALESCE(bt.completed_at, bt.updated_at) AS at
      FROM board_tasks bt LEFT JOIN agents a ON a.id = bt.assigned_agent_id AND a.workspace_id = bt.workspace_id
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'failed' AND bt.source_type <> 'orchestration_task'
  ORDER BY at DESC NULLS LAST LIMIT :limit
""")
# F246: a mission waiting for its plan's approval carries its card (#0031, #0126, #0176).
_MISSION_ROWS = text("""
    SELECT r.id, r.goal, r.updated_at, card.id AS card_id, card.workspace_seq
      FROM orchestration_runs r
      LEFT JOIN LATERAL (
        SELECT bt.id, bt.workspace_seq FROM board_tasks bt
         WHERE bt.workspace_id = r.workspace_id AND bt.orchestration_run_id = r.id AND bt.source_type = 'orchestration'
      ORDER BY bt.id LIMIT 1) card ON true
     WHERE r.workspace_id = CAST(:ws AS uuid) AND r.state = 'awaiting_approval'
  ORDER BY r.updated_at DESC NULLS LAST LIMIT :limit
""")
# F246: tickets nothing will move until the owner does, newest first; ``of_all``
# is the exact count. Assigned to nobody (#0067); a playbook's own card in
# Assigned, which the board never runs (#0070, #0149: rejected or re-briefed); a
# CLI agent's ticket while no host that runs its CLI is online (#0016, #0161,
# #0177: the line the board writes on it); a mission step still open after its
# mission ended (#0119.3, #0176.9-.12). A playbook step's session ticket
# ('recipe:<run>:<step>') is claimed by a CLI host, so only a no-host line stalls it.
_STUCK_ROWS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.source_type, bt.parent_task_id, bt.orchestration_run_id,
           a.name AS agent_name, bt.updated_at AS at, COUNT(*) OVER () AS of_all,
           CASE WHEN bt.source_type = 'orchestration_task' THEN :why_mission_ended
                WHEN bt.blocked_reason = :no_host_line OR starts_with(bt.blocked_reason, :no_cli_host_prefix)
                  THEN :why_no_host
                WHEN bt.assigned_agent_id IS NULL THEN :why_no_agent
                ELSE :why_not_picked_up END AS why
      FROM board_tasks bt
      LEFT JOIN agents a ON a.id = bt.assigned_agent_id AND a.workspace_id = bt.workspace_id
      LEFT JOIN orchestration_tasks ot ON ot.id = bt.orchestration_task_id
      LEFT JOIN orchestration_runs r ON r.id = COALESCE(bt.orchestration_run_id, ot.run_id)
                                    AND r.workspace_id = bt.workspace_id
     WHERE bt.workspace_id = CAST(:ws AS uuid)
       AND ((bt.status = 'assigned' AND bt.source_type NOT IN ('orchestration_task', 'orchestration')
             AND (bt.assigned_agent_id IS NULL
                  OR (bt.source_type = 'recipe' AND COALESCE(bt.source_id, '') NOT LIKE 'recipe:%')
                  OR bt.blocked_reason = :no_host_line OR starts_with(bt.blocked_reason, :no_cli_host_prefix)))
         OR (bt.source_type = 'orchestration_task' AND bt.status = ANY(:open_step) AND r.state = ANY(:ended)))
  ORDER BY at DESC NULLS LAST LIMIT :limit
""")


def needs_you_counts(db: Session, workspace_id: Any, *, may_answer: bool = True) -> Dict[str, int]:
    """Each kind's count and their ``total``: the one Needs-you number.
    ``may_answer`` False (not a workspace admin) leaves questions and approval
    grants out, since that viewer can neither open nor answer them."""
    return _counted(db, workspace_id, may_answer=may_answer, limit=1)[0]


def needs_you(db: Session, workspace_id: Any, *, may_answer: bool = True) -> Dict[str, Any]:
    """The number and the rows behind it, newest first in each kind."""
    counts, listed = _counted(db, workspace_id, may_answer=may_answer, limit=ROWS_PER_KIND)
    params = {"ws": str(workspace_id), "limit": ROWS_PER_KIND, "ended": _ended_states()}
    rows: Dict[str, List[Dict[str, Any]]] = {
        "review": _numbered_rows(db, workspace_id, db.execute(_REVIEW_ROWS, params).all()),
        "question": _grant_rows(db, workspace_id, _grants_by_id(db, workspace_id, listed["question"])),
        "approval": _approval_rows(db, workspace_id, _grants_by_id(db, workspace_id, listed["approval"]), params),
        "stuck": _stuck_rows(db, workspace_id, listed["stuck"]),
        "failed": [_ticket_row(r) for r in db.execute(_FAILED_ROWS, params)],
    }
    return {"total": counts.pop("total"), "counts": counts, "rows": rows}


def _counted(
    db: Session, workspace_id: Any, *, may_answer: bool, limit: int,
) -> Tuple[Dict[str, int], Dict[str, List[Any]]]:
    """Every kind's exact count with ``total``, and the newest ``limit`` of the
    questions, approval grants and stuck tickets (one read gives both)."""
    row = db.execute(_COUNTS, {"ws": str(workspace_id), "ended": _ended_states()}).first()
    questions = _asks(db, workspace_id, questions=True, limit=limit) if may_answer else (0, [])
    approvals = _asks(db, workspace_id, questions=False, limit=limit) if may_answer else (0, [])
    stuck = _stuck(db, workspace_id, limit=limit)
    counts = {
        "review": _count(row, "review"),
        "question": questions[0],
        "approval": approvals[0] + _count(row, "mission_approval"),
        "stuck": stuck[0],
        "failed": _count(row, "failed"),
    }
    listed = {"question": questions[1], "approval": approvals[1], "stuck": stuck[1]}
    return {**counts, "total": sum(counts.values())}, listed


def _count(row: Any, name: str) -> int:
    return int(getattr(row, name, 0) or 0) if row is not None else 0


def _asks(db: Session, workspace_id: Any, *, questions: bool, limit: int) -> Tuple[int, List[int]]:
    """How many open questions (``questions``) or approvals that can still be
    given there are, and the ids of the newest ``limit``."""
    found = db.execute(_ASKS, {"ws": str(workspace_id), "questions": questions,
                               "now": datetime.now(timezone.utc), "limit": limit}).all()
    return (int(found[0].of_all) if found else 0), [r.id for r in found]


def _stuck(db: Session, workspace_id: Any, *, limit: int) -> Tuple[int, List[Any]]:
    """How many tickets are stuck, and the newest ``limit`` of them."""
    from services.cli_ticket_lane import NO_CLI_HOST_PREFIX, NO_HOST_REASON

    found = db.execute(_STUCK_ROWS, {
        "ws": str(workspace_id), "limit": limit, "open_step": OPEN_STEP_STATUSES, "ended": _ended_states(),
        "no_host_line": NO_HOST_REASON, "no_cli_host_prefix": NO_CLI_HOST_PREFIX,
        "why_no_host": STUCK_NO_HOST, "why_no_agent": STUCK_NO_AGENT,
        "why_not_picked_up": STUCK_NOT_PICKED_UP, "why_mission_ended": STUCK_MISSION_ENDED,
    }).all()
    return (int(found[0].of_all) if found else 0), found


def _ended_states() -> List[str]:
    """A mission's ended states: it runs none of its steps again."""
    from core.models.orchestration_enums import TERMINAL_RUN_STATES

    return sorted(state.value for state in TERMINAL_RUN_STATES)


def _grants_by_id(db: Session, workspace_id: Any, ids: List[int]) -> List[Any]:
    """The grants ``ids`` names, in that order."""
    from core.models.approval_grants import ApprovalGrant

    if not ids:
        return []
    found = {g.id: g for g in db.query(ApprovalGrant).filter(
        ApprovalGrant.workspace_id == workspace_id, ApprovalGrant.id.in_(ids)).all()}
    return [found[i] for i in ids if i in found]


def _grant_rows(db: Session, workspace_id: Any, grants: List[Any]) -> List[Dict[str, Any]]:
    """A question or approval grant as a row: the ticket it opens in (and its
    number, PRD-252 R4), and who asked (F091-E1)."""
    from core.models.approval_grants import KIND_QUESTION
    from services.grant_owners import grant_owners

    owners = grant_owners(db, workspace_id, grants)
    tickets = {g.id: (owners.get(g.id) or {}).get("ticket") or {} for g in grants}
    numbers = _numbers_of(db, workspace_id, {t["id"] for t in tickets.values() if t.get("id")})
    rows = []
    for g in grants:
        ticket_id = tickets[g.id].get("id")
        rows.append({
            "source": "grant",
            "id": str(g.id),
            "title": g.question_md if g.kind == KIND_QUESTION else (g.reason or g.tool_name or "Approval"),
            "ticket_id": ticket_id,
            "ticket_number": numbers.get(ticket_id),
            "agent_name": ((owners.get(g.id) or {}).get("agent") or {}).get("name"),
            "at": _iso(g.requested_at),
        })
    return rows


def _numbers_of(db: Session, workspace_id: Any, ticket_ids: set) -> Dict[int, Optional[str]]:
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    if not ticket_ids:
        return {}
    tickets = db.query(BoardTask).filter(BoardTask.id.in_(ticket_ids), BoardTask.workspace_id == workspace_id).all()
    return ticket_numbers(db, workspace_id, tickets)


def _approval_rows(db: Session, workspace_id: Any, grants: List[Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Approval grants and missions waiting for their plan's approval, newest
    first. A mission's row opens its plan and names its card by number (F246)."""
    from services.ticket_numbers import format_number

    missions = [
        {"source": "mission", "id": str(r.id), "title": r.goal, "ticket_id": r.card_id,
         "ticket_number": format_number(r.workspace_seq), "agent_name": None, "at": _iso(r.updated_at)}
        for r in db.execute(_MISSION_ROWS, params)
    ]
    merged = _grant_rows(db, workspace_id, grants) + missions
    return sorted(merged, key=lambda row: row["at"] or "", reverse=True)[:ROWS_PER_KIND]


def _stuck_rows(db: Session, workspace_id: Any, rows: List[Any]) -> List[Dict[str, Any]]:
    """A stuck ticket opens itself, named by its number (a mission step's is its
    card's: #0176.9), with why it is stuck."""
    return [{**row, "why": r.why} for row, r in zip(_numbered_rows(db, workspace_id, rows), rows)]


def _numbered_rows(db: Session, workspace_id: Any, rows: List[Any]) -> List[Dict[str, Any]]:
    """Ticket rows named by their numbers, a mission step's being its card's (#0139.2)."""
    from services.ticket_numbers import ticket_numbers

    numbers = ticket_numbers(db, workspace_id, rows)
    return [{**_ticket_row(r), "number": numbers.get(r.id)} for r in rows]


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _ticket_row(r: Any) -> Dict[str, Any]:
    """A ticket opens in the board's viewer; only a mission's own card opens its
    mission. A session-run mission step also carries the run id, and opens itself."""
    from services.ticket_numbers import format_number

    mission = str(r.orchestration_run_id) if r.source_type == MISSION_CARD and r.orchestration_run_id else None
    return {"ticket_id": r.id, "number": format_number(r.workspace_seq), "title": r.title,
            "agent_name": r.agent_name, "mission_id": mission, "at": _iso(r.at)}

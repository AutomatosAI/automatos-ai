"""PRD-268 US-C03b (§6.2.3-6.2.5, FR-6) — ``platform_suggest_next_step``: a check-in records one suggestion.

The one write a check-in makes. It inserts an ``open`` ``proactive_suggestions`` row for
the workspace owner (K11), the person a suggestion is addressed to, and changes nothing
else: Start and Hide (US-C06a) are the person's.

It refuses, with a sentence the model reads, a call that:

* is not a check-in's: the run's facts (``_check_in``: its run id, its successful reads,
  its agent) reach it only from the server, through ``check_in_lane.carries_the_check_in``;
* is not a suggestion: ``suggestion_fields.read_suggestion`` reads every argument;
* cites a read the run did not make (provenance, ``modules.proactive.provenance_ok``);
* passes a §6.9 cap: per run, new per person a day, open per agent, by the workspace's own
  caps over the config defaults (``services.proactive.limits``); caps that are wrong refuse
  every suggestion;
* repeats an open suggestion, or one hidden in the last 14 days ("Already suggested: …");
* proposes a ticket for an agent that is not in the workspace.

Past ``max_open_per_person`` the person's oldest open suggestions expire to make room, in
the insert's savepoint, so a lost insert expires nothing. The open-row unique index
(workspace, person, subject key) settles a race between two check-ins: the insert runs in
a savepoint, and the loser is told it was already suggested. Any other integrity error is
raised, and the executor reports the action failed.

On success the handler commits the session it was given, as the other write handlers do.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.proactive import SUGGESTION_EXPIRED, SUGGESTION_HIDDEN, SUGGESTION_OPEN, ProactiveSuggestion
from modules.proactive import CapRefusal, Candidate, Caps, find_duplicate, oldest_to_expire, provenance_ok
from modules.proactive.dedupe import HIDDEN_WINDOW
from modules.tools.discovery.check_in_lane import CHECK_IN_PARAM
from modules.tools.discovery.suggestion_fields import PAYLOAD_AGENT, Suggestion, read_suggestion
from services.proactive.limits import room_refusal, workspace_caps
from services.proactive.runs import suggestions_made

logger = logging.getLogger(__name__)

__all__ = ["suggest_next_step"]

# Where a suggestion came from (§6.2.4): Auto's workspace check-in, or an agent's
# check-in, which is its heartbeat converted (source_type 'heartbeat').
SOURCE_CHECK_IN = "check_in"
SOURCE_AGENT_CHECK_IN = "heartbeat"
# Open suggestions expire after 7 days (§6.2.4).
OPEN_LIFETIME = timedelta(days=7)

NOT_IN_A_CHECK_IN = "platform_suggest_next_step records what a check-in found, and runs only in a check-in."
NO_PROVENANCE = ("Every evidence ref must be a read this check-in made that succeeded: cite its type and id "
                 "as the read gave them.")
NO_ADDRESSEE = "This workspace has no owner to suggest to, so nothing was recorded."
CAPPED = "No more suggestions now: the {cap} cap is {limit}. Record nothing more."
ALREADY_SUGGESTED = "Already suggested: {title}"
NO_SUCH_AGENT = "There is no agent {agent_id} in this workspace to give the ticket to."
CAPS_WRONG = "This workspace's suggestion caps are set wrong, so nothing can be recorded. Record nothing more."

NOT_IN_A_CHECK_IN_CODE = "not_in_a_check_in"
NOT_A_SUGGESTION_CODE = "not_a_suggestion"
NO_PROVENANCE_CODE = "no_provenance"
NO_ADDRESSEE_CODE = "no_addressee"
CAPPED_CODE = "suggestion_cap"
ALREADY_SUGGESTED_CODE = "already_suggested"
NO_SUCH_AGENT_CODE = "no_such_agent"
CAPS_WRONG_CODE = "caps_wrong"


@dataclass(frozen=True)
class _Run:
    """The check-in a call belongs to, as the server handed it over."""

    run_id: UUID
    reads: Tuple[Any, ...]
    agent_id: Optional[int]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _refusal(error: str, code: str) -> Dict[str, Any]:
    return {"success": False, "error": error, "error_code": code}


def _run_of(check_in: Any) -> Optional[_Run]:
    """The run in ``_check_in``, or None when the call carries no well-formed one."""
    if not isinstance(check_in, dict):
        return None
    agent_id = check_in.get("agent_id")
    try:
        run_id = UUID(str(check_in.get("run_id")))
        agent = int(agent_id) if agent_id is not None and not isinstance(agent_id, bool) else None
    except (TypeError, ValueError):
        return None
    if agent_id is not None and agent is None:
        return None
    return _Run(run_id=run_id, reads=tuple(check_in.get("reads") or ()), agent_id=agent)


def _addressee(db: Session, workspace_id: UUID) -> Optional[int]:
    """The workspace owner's ``users.id`` (K11), or None."""
    from core.models.workspaces import Workspace
    from core.workspaces.owner import workspace_owner

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    owner = workspace_owner(db, workspace)
    return int(owner.id) if owner is not None else None


def _caps(db: Session, workspace_id: UUID) -> Optional[Caps]:
    """The workspace's caps; None when they are wrong (logged)."""
    try:
        return workspace_caps(db, workspace_id)
    except ValueError:
        logger.exception("[proactive] the caps of workspace %s are wrong: no suggestion recorded", workspace_id)
        return None


def _cap_refusal(db: Session, workspace_id: UUID, user_id: int, run: _Run, caps: Caps,
                 now: datetime) -> Optional[CapRefusal]:
    """The §6.9 cap one more suggestion would pass, or None."""
    return room_refusal(db, workspace_id, user_id, run.agent_id, caps, now,
                        made_this_run=suggestions_made(db, workspace_id, run.run_id))


def _make_room(db: Session, workspace_id: UUID, user_id: int, caps: Caps, now: datetime) -> int:
    """Expire the person's oldest open suggestions so that one more fits under
    ``max_open_per_person``; how many expired."""
    open_rows = db.query(ProactiveSuggestion.id, ProactiveSuggestion.created_at).filter(
        ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id,
        ProactiveSuggestion.status == SUGGESTION_OPEN).all()
    expired = [row.id for row in oldest_to_expire(caps, open_rows)]
    if expired:
        db.query(ProactiveSuggestion).filter(
            ProactiveSuggestion.id.in_(expired), ProactiveSuggestion.status == SUGGESTION_OPEN,
        ).update({ProactiveSuggestion.status: SUGGESTION_EXPIRED, ProactiveSuggestion.expires_at: now},
                 synchronize_session=False)
    return len(expired)


def _duplicate(db: Session, workspace_id: UUID, user_id: int, suggestion: Suggestion,
               now: datetime) -> Optional[ProactiveSuggestion]:
    """The person's open suggestion, or one hidden in the last 14 days, that this one repeats."""
    mine = (ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id)
    hidden_at = func.coalesce(ProactiveSuggestion.decided_at, ProactiveSuggestion.last_seen_at,
                              ProactiveSuggestion.created_at)
    open_rows = db.query(ProactiveSuggestion).filter(*mine, ProactiveSuggestion.status == SUGGESTION_OPEN).all()
    hidden_rows = db.query(ProactiveSuggestion).filter(
        *mine, ProactiveSuggestion.status == SUGGESTION_HIDDEN, hidden_at >= now - HIDDEN_WINDOW).all()
    candidate = Candidate(title=suggestion.title, subject_key=suggestion.subject_key)
    return find_duplicate(candidate, open_rows, hidden_rows, now)


def _held_open(db: Session, workspace_id: UUID, user_id: int, subject_key: str) -> bool:
    """Whether the person already has an open suggestion under ``subject_key``."""
    return db.query(ProactiveSuggestion.id).filter(
        ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id,
        ProactiveSuggestion.subject_key == subject_key, ProactiveSuggestion.status == SUGGESTION_OPEN,
    ).first() is not None


def _unknown_agent(db: Session, workspace_id: UUID, suggestion: Suggestion) -> Optional[int]:
    """The proposed ticket's agent when it is not in this workspace, else None."""
    from core.models.core import Agent

    agent_id = suggestion.proposed_action["payload"].get(PAYLOAD_AGENT)
    if agent_id is None:
        return None
    found = db.query(Agent.id).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    return None if found else agent_id


def _insert(db: Session, workspace_id: UUID, user_id: int, run: _Run, suggestion: Suggestion, caps: Caps,
            now: datetime) -> Optional[ProactiveSuggestion]:
    """The new open row, the oldest past the open cap expired, committed; None when a
    concurrent check-in holds its open subject key."""
    row = ProactiveSuggestion(
        workspace_id=workspace_id, user_id=user_id,
        source_type=SOURCE_CHECK_IN if run.agent_id is None else SOURCE_AGENT_CHECK_IN,
        source_agent_id=run.agent_id, run_id=run.run_id,
        title=suggestion.title, detail=suggestion.detail, evidence=[dict(ref) for ref in suggestion.evidence],
        proposed_action={**suggestion.proposed_action, "payload": dict(suggestion.proposed_action["payload"])},
        subject_key=suggestion.subject_key, severity=suggestion.severity, status=SUGGESTION_OPEN,
        occurrences=1, last_seen_at=now, created_at=now, expires_at=now + OPEN_LIFETIME,
    )
    try:
        with db.begin_nested():
            expired = _make_room(db, workspace_id, user_id, caps, now)
            db.add(row)
            db.flush()
    except IntegrityError:
        if not _held_open(db, workspace_id, user_id, suggestion.subject_key):
            raise
        logger.info("[proactive] run %s: subject already open in workspace %s", run.run_id, workspace_id)
        return None
    db.commit()
    if expired:
        logger.info("[proactive] run %s expired %d open suggestion(s) for room in %s", run.run_id, expired,
                    workspace_id)
    return row


def _record(db: Session, workspace_id: UUID, run: _Run, suggestion: Suggestion) -> Dict[str, Any]:
    """Every check that reads the database, then the insert."""
    now = _now()
    user_id = _addressee(db, workspace_id)
    if user_id is None:
        return _refusal(NO_ADDRESSEE, NO_ADDRESSEE_CODE)
    caps = _caps(db, workspace_id)
    if caps is None:
        return _refusal(CAPS_WRONG, CAPS_WRONG_CODE)
    cap = _cap_refusal(db, workspace_id, user_id, run, caps, now)
    if cap is not None:
        return _refusal(CAPPED.format(cap=cap.cap, limit=cap.limit), CAPPED_CODE)
    duplicate = _duplicate(db, workspace_id, user_id, suggestion, now)
    if duplicate is not None:
        return _refusal(ALREADY_SUGGESTED.format(title=duplicate.title), ALREADY_SUGGESTED_CODE)
    unknown = _unknown_agent(db, workspace_id, suggestion)
    if unknown is not None:
        return _refusal(NO_SUCH_AGENT.format(agent_id=unknown), NO_SUCH_AGENT_CODE)
    row = _insert(db, workspace_id, user_id, run, suggestion, caps, now)
    if row is None:
        return _refusal(ALREADY_SUGGESTED.format(title=suggestion.title), ALREADY_SUGGESTED_CODE)
    logger.info("[proactive] run %s suggested %s in workspace %s", run.run_id, row.id, workspace_id)
    return {"success": True, "data": {"suggestion_id": str(row.id), "title": row.title,
                                      "subject_key": row.subject_key}}


async def suggest_next_step(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Record one suggestion of a check-in's for the workspace owner, or say why not."""
    run = _run_of((params or {}).get(CHECK_IN_PARAM))
    if run is None:
        return _refusal(NOT_IN_A_CHECK_IN, NOT_IN_A_CHECK_IN_CODE)
    try:
        suggestion = read_suggestion(params)
    except ValueError as exc:
        return _refusal(str(exc), NOT_A_SUGGESTION_CODE)
    if not provenance_ok(suggestion.evidence, run.reads):
        return _refusal(NO_PROVENANCE, NO_PROVENANCE_CODE)
    return _record(db, UUID(str(workspace_id)), run, suggestion)

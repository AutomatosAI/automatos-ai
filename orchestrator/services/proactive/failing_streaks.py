"""PRD-268 §6.2.2, FR-5 — failing streaks: one card per streak, a slower check-in, and the reset.

An agent's check-in that fails is an ``error`` run whose ``error_class`` names the failure
(``services.proactive.failures``), and it makes no further model call. Its streak (the run,
and the agent's newest runs before it that failed with the same class, skips passed over,
``modules.proactive.streaks``) is counted from ``proactive_runs``; no column holds it.

* **The card.** The first failure of an (agent, error class) streak records one suggestion
  for the workspace owner (K11), at no model cost: ``subject_key``
  ``agent:<id>:error:<class>``, ``source_type`` ``streak``, its title and detail from
  ``services.proactive.streak_text``, the streak's runs as its evidence, and Start a ticket
  for Auto. Each later failure updates the open card (``occurrences``, ``last_seen_at``, its
  detail and evidence) and makes none. A streak whose card the person hid or started gets
  no other, and neither does one whose card they hid in the last 14 days (§6.2.4).
* **The slower check-in.** From the third failure in a row, ``held_back`` skips the agent's
  check-ins until ``next_interval`` (doubling up to 24 hours, or the workspace's
  ``max_interval_minutes`` when that is shorter) has passed since its last run, less the
  scheduler's few minutes of grace. Each skip is a ``skipped_cap`` run
  (``failing_streak``) with no model call, and the card says how often the agent checks in now.
* **The reset.** The first run that works ends every streak of the agent: its open cards
  expire, their detail saying the check-ins work again, and with no failure left to count
  the interval is the heartbeat's own again.

Auto's workspace check-in has no agent to key a streak by; its first failure still rings
the bell (``services.proactive.heartbeat_bell``). Bookkeeping that fails is logged and never
fails the check-in, whose run row is written first.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import and_, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.proactive import (
    RUN_ERROR,
    RUN_SKIPPED_CAP,
    SUGGESTION_EXPIRED,
    SUGGESTION_HIDDEN,
    SUGGESTION_OPEN,
    ProactiveRun,
    ProactiveSuggestion,
)
from modules.proactive.dedupe import HIDDEN_WINDOW, agent_error_key, agent_error_prefix
from modules.proactive.provenance import MAX_EVIDENCE
from modules.proactive.streaks import (
    DOUBLING_FROM,
    MAX_STREAK_INTERVAL_MINUTES,
    consecutive_failures,
    is_due,
    next_interval,
)
from modules.proactive.window import SUCCESSFUL_OUTCOMES
from modules.tools.discovery.handlers_proactive import OPEN_LIFETIME
from modules.tools.discovery.suggestion_fields import BOARD_TASK, PAYLOAD_AGENT
from services.proactive.gates import Refusal
from services.proactive.limits import workspace_caps
from services.proactive.runs import KIND_AGENT, SCHEDULER_GRACE, finished_runs, owner_id, ran_at
from services.proactive.streak_text import streak_detail, streak_title, works_again_detail

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_INTERVAL_MINUTES",
    "FAILING_STREAK",
    "SOURCE_STREAK",
    "base_minutes",
    "ceiling_minutes",
    "held_back",
    "holds",
    "note_run",
]

# What held a check-in back, recorded as its skipped_cap run's error_class.
FAILING_STREAK = "failing_streak"
SOURCE_STREAK = "streak"
# An agent that has stopped working is for today, not for later.
STREAK_SEVERITY = "today"
# An agent heartbeat's interval when its config states none (HeartbeatService.schedule_agent_heartbeat).
DEFAULT_INTERVAL_MINUTES = 60
# The finished runs a streak is counted from: the interval reaches its 24 hours long before.
STREAK_READ_LIMIT = 50
RUN_REF = "run"
# Today's name for an agent whose row is gone (_dispatch_heartbeat_notification).
UNNAMED_AGENT = "agent-{agent_id}"


@dataclass(frozen=True)
class _Failure:
    """One failed check-in of an agent, with what its card needs."""

    workspace_id: UUID
    user_id: int
    agent_id: int
    name: str
    error_class: str
    base_minutes: int
    now: datetime
    # The longest a streak holds the agent back: the workspace's max_interval_minutes, at most 24 hours.
    ceiling_minutes: int = MAX_STREAK_INTERVAL_MINUTES

    @property
    def key(self) -> str:
        """The streak's subject key: ``agent:<id>:error:<class>``."""
        return agent_error_key(self.agent_id, self.error_class)


def base_minutes(hb_config: Optional[Dict[str, Any]]) -> int:
    """The heartbeat's own interval, as the scheduler reads it; a value that is not a
    positive whole number is the scheduler's default."""
    value = (hb_config or {}).get("interval_minutes", DEFAULT_INTERVAL_MINUTES)
    if isinstance(value, bool):
        return DEFAULT_INTERVAL_MINUTES
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_MINUTES
    return minutes if minutes > 0 else DEFAULT_INTERVAL_MINUTES


def _streak(runs: Sequence[Any], error_class: Optional[str]) -> Tuple[Any, ...]:
    """The newest runs that failed in a row with ``error_class``, newest first."""
    return tuple(runs[:consecutive_failures(runs, error_class or "")])


def _agent_runs(db: Session, workspace_id: UUID, agent_id: int) -> Tuple[ProactiveRun, ...]:
    return finished_runs(db, workspace_id, KIND_AGENT, agent_id, STREAK_READ_LIMIT)


def holds(runs: Sequence[Any], base: int, now: datetime, ceiling: int = MAX_STREAK_INTERVAL_MINUTES) -> bool:
    """Whether an agent's newest finished runs (``outcome``, ``error_class``,
    ``window_until``, ``created_at``; newest first) hold its next check-in back: a streak of
    three or more failures whose ``next_interval`` (up to ``ceiling`` minutes) has not
    passed since its last run, less ``SCHEDULER_GRACE``."""
    if not runs or runs[0].outcome != RUN_ERROR:
        return False
    count = len(_streak(runs, runs[0].error_class))
    return count >= DOUBLING_FROM and not is_due(ran_at(runs[0]), base, count, now + SCHEDULER_GRACE, ceiling)


def ceiling_minutes(db: Session, workspace_id: UUID) -> int:
    """The longest a failing streak holds the workspace's agents back: its
    ``max_interval_minutes``, at most the PRD's 24 hours. Caps that cannot be read keep
    the 24 hours, the fewest runs (logged)."""
    try:
        with db.begin_nested():
            return min(workspace_caps(db, workspace_id).max_interval_minutes, MAX_STREAK_INTERVAL_MINUTES)
    except Exception:  # noqa: BLE001 — the longest hold makes the fewest runs
        logger.exception("[proactive] the caps of %s could not be read: streaks hold for 24 hours", workspace_id)
        return MAX_STREAK_INTERVAL_MINUTES


def held_back(db: Session, workspace_id: UUID, agent_id: int, hb_config: Optional[Dict[str, Any]],
              now: datetime) -> Optional[Refusal]:
    """``skipped_cap`` (``failing_streak``) while the agent's failing streak holds its
    check-in back, else None to go ahead."""
    runs = _agent_runs(db, workspace_id, agent_id)
    if holds(runs, base_minutes(hb_config), now, ceiling_minutes(db, workspace_id)):
        return Refusal(RUN_SKIPPED_CAP, FAILING_STREAK)
    return None


def _agent_name(db: Session, workspace_id: UUID, agent_id: int) -> Optional[str]:
    """The agent's name in its workspace; None when the agent is not there."""
    from core.models.core import Agent

    row = db.query(Agent.name).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    if row is None:
        return None
    return row[0] or UNNAMED_AGENT.format(agent_id=agent_id)


def _evidence(streak: Sequence[ProactiveRun]) -> List[Dict[str, str]]:
    """The streak's newest runs, as the card's evidence refs."""
    return [{"type": RUN_REF, "id": str(run.id)} for run in streak[:MAX_EVIDENCE]]


def _fix_it(db: Session, workspace_id: UUID) -> Dict[str, Any]:
    """What Start does: a ticket for the workspace's Auto (K12), unassigned when it has none."""
    from services.harness_grants import resolve_auto_agent

    auto = resolve_auto_agent(db, workspace_id)
    return {"kind": BOARD_TASK, "payload": {PAYLOAD_AGENT: int(auto["id"])} if auto else {}}


def _open_card(db: Session, failure: _Failure) -> Optional[ProactiveSuggestion]:
    return db.query(ProactiveSuggestion).filter(
        ProactiveSuggestion.workspace_id == failure.workspace_id, ProactiveSuggestion.user_id == failure.user_id,
        ProactiveSuggestion.subject_key == failure.key, ProactiveSuggestion.status == SUGGESTION_OPEN,
    ).first()


def _made_already(db: Session, failure: _Failure, streak: Sequence[ProactiveRun]) -> bool:
    """Whether this streak already had its card (the person hid or started it), or the
    person hid a card about it in the last 14 days."""
    hidden_at = func.coalesce(ProactiveSuggestion.decided_at, ProactiveSuggestion.last_seen_at,
                              ProactiveSuggestion.created_at)
    return db.query(ProactiveSuggestion.id).filter(
        ProactiveSuggestion.workspace_id == failure.workspace_id, ProactiveSuggestion.user_id == failure.user_id,
        ProactiveSuggestion.subject_key == failure.key,
        or_(ProactiveSuggestion.run_id.in_([run.id for run in streak]),
            and_(ProactiveSuggestion.status == SUGGESTION_HIDDEN, hidden_at >= failure.now - HIDDEN_WINDOW)),
    ).first() is not None


def _seen_again(card: ProactiveSuggestion, failure: _Failure, streak: Sequence[ProactiveRun]) -> None:
    """A later failure: the open card counts it (``occurrences``, every failure it has seen),
    stays open while the agent fails, and its detail says how long the streak is, read from
    the runs, and how often the agent checks in now."""
    occurrences = int(card.occurrences or 0) + 1
    count, since = len(streak), ran_at(streak[-1])
    if count >= STREAK_READ_LIMIT:  # the streak reaches past the runs read: its card, made at its start, knows
        count, since = max(count, occurrences), min(since, card.created_at)
    card.occurrences = occurrences
    card.last_seen_at = failure.now
    card.expires_at = failure.now + OPEN_LIFETIME
    card.detail = streak_detail(failure.error_class, count, since, failure.base_minutes,
                                next_interval(failure.base_minutes, len(streak), failure.ceiling_minutes))
    card.evidence = _evidence(streak)


def _add_card(db: Session, failure: _Failure, streak: Sequence[ProactiveRun]) -> None:
    """The streak's one card, open for the owner; a concurrent writer's open card wins."""
    count, first = len(streak), streak[-1]
    row = ProactiveSuggestion(
        workspace_id=failure.workspace_id, user_id=failure.user_id, source_type=SOURCE_STREAK,
        source_agent_id=failure.agent_id, run_id=first.id, title=streak_title(failure.name, failure.error_class),
        detail=streak_detail(failure.error_class, count, ran_at(first), failure.base_minutes,
                             next_interval(failure.base_minutes, count, failure.ceiling_minutes)),
        evidence=_evidence(streak), proposed_action=_fix_it(db, failure.workspace_id), subject_key=failure.key,
        severity=STREAK_SEVERITY, status=SUGGESTION_OPEN, occurrences=count, last_seen_at=failure.now,
        created_at=failure.now, expires_at=failure.now + OPEN_LIFETIME,
    )
    try:
        with db.begin_nested():
            db.add(row)
            db.flush()
    except IntegrityError:
        if _open_card(db, failure) is None:
            raise
        logger.info("[proactive] the streak %s in %s is already open", failure.key, failure.workspace_id)


def _note_failure(db: Session, run: Any, workspace_id: UUID, agent_id: int, base: int, now: datetime) -> None:
    """Record a failed run's streak: its card made, or the open one counted."""
    name, user_id = _agent_name(db, workspace_id, agent_id), owner_id(db, workspace_id)
    streak = _streak(_agent_runs(db, workspace_id, agent_id), run.error_class)
    if name is None or user_id is None or not streak:
        return
    failure = _Failure(workspace_id=workspace_id, user_id=user_id, agent_id=agent_id, name=name,
                       error_class=str(run.error_class), base_minutes=base, now=now,
                       ceiling_minutes=ceiling_minutes(db, workspace_id))
    card = _open_card(db, failure)
    if card is not None:
        _seen_again(card, failure, streak)
    elif not _made_already(db, failure, streak):
        _add_card(db, failure, streak)
    db.commit()
    logger.info("[proactive] agent %s in %s: failure %d of streak %s", agent_id, workspace_id, len(streak),
                failure.key)


def _note_success(db: Session, workspace_id: UUID, agent_id: int, now: datetime) -> None:
    """A run that worked ends the agent's streaks: each open card expires, saying so."""
    prefix = agent_error_prefix(agent_id)
    cards = db.query(ProactiveSuggestion).filter(
        ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.status == SUGGESTION_OPEN,
        ProactiveSuggestion.subject_key.startswith(prefix, autoescape=True),
    ).all()
    if not cards:
        return
    name = _agent_name(db, workspace_id, agent_id) or UNNAMED_AGENT.format(agent_id=agent_id)
    for card in cards:
        detail = works_again_detail(name, card.subject_key[len(prefix):], int(card.occurrences or 1))
        db.query(ProactiveSuggestion).filter(
            ProactiveSuggestion.id == card.id, ProactiveSuggestion.status == SUGGESTION_OPEN,
        ).update({ProactiveSuggestion.status: SUGGESTION_EXPIRED, ProactiveSuggestion.detail: detail,
                  ProactiveSuggestion.expires_at: now}, synchronize_session=False)
    db.commit()
    logger.info("[proactive] agent %s in %s works again: %d streak card(s) expired", agent_id, workspace_id,
                len(cards))


def note_run(db: Session, run: Any, workspace_id: UUID, agent_id: int, hb_config: Optional[Dict[str, Any]],
             now: datetime) -> bool:
    """After an agent's check-in (a runner ``CheckInResult``): a failure records its streak,
    a run that worked ends them; a skip does neither. True when that completed; a failure
    is logged and never reaches the check-in."""
    try:
        if run.outcome == RUN_ERROR:
            _note_failure(db, run, workspace_id, agent_id, base_minutes(hb_config), now)
        elif run.outcome in SUCCESSFUL_OUTCOMES:
            _note_success(db, workspace_id, agent_id, now)
    except Exception:  # noqa: BLE001 — logged; the run row is written, and the check-in stands
        db.rollback()
        logger.exception("[proactive] agent %s's streak in %s could not be recorded", agent_id, workspace_id)
        return False
    return True

"""PRD-268 §6.2.4, §6.9, FR-14 — a check-in's run row, and what the runner reads back from the runs.

Every check-in writes one ``proactive_runs`` row, whatever happened: refused at its start
(``skipped_*``, no model call), ``silent``, ``suggested`` or ``error``. Its ``kind`` says
which check-in it was: Auto's workspace check-in (``workspace_check_in``, no agent) or an
agent's (``agent_check_in``).

``error_class`` holds what stopped a run: for an ``error`` the error's class (the streak's,
US-C05, or ``cost_ceiling``); for a skip, the check that refused it (``switched_off``,
``no_addressee``, ``paused``, ``trial_only``, ``spend_ceiling`` or the cap's name).

The runner reads back where a check-in's window starts (its last successful run of the same
kind, ``modules.proactive.window``), the check-ins that reached the model in the last day
(the §6.9 cap counts those, a silent run without a model call does not), and the person's
open and recently hidden titles its prompt lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.proactive import (
    ERROR_CLASS_LENGTH,
    MODEL_LENGTH,
    SUGGESTION_HIDDEN,
    SUGGESTION_OPEN,
    ProactiveRun,
    ProactiveSuggestion,
)
from modules.proactive.dedupe import HIDDEN_WINDOW
from modules.proactive.window import SUCCESSFUL_OUTCOMES, last_success_at

__all__ = [
    "KIND_AGENT",
    "KIND_WORKSPACE",
    "MAX_TITLES",
    "MODEL_RUN_WINDOW",
    "RunRecord",
    "known_titles",
    "last_success",
    "model_runs_today",
    "suggestions_made",
    "write_run",
]

KIND_WORKSPACE = "workspace_check_in"
KIND_AGENT = "agent_check_in"
# "Check-in runs that reach the model, per workspace per day" (§6.9) counts the last 24 hours,
# as the suggestion tool counts a person's new suggestions per day.
MODEL_RUN_WINDOW = timedelta(hours=24)
# The open and recently hidden titles a prompt lists, newest first: the open-per-person cap is 20.
MAX_TITLES = 20


@dataclass(frozen=True)
class RunRecord:
    """One check-in run, as its ``proactive_runs`` row records it."""

    run_id: UUID
    workspace_id: UUID
    kind: str
    outcome: str
    agent_id: Optional[int] = None
    window_since: Optional[datetime] = None
    window_until: Optional[datetime] = None
    signals: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    suggestions: int = 0
    model: Optional[str] = None
    tokens: int = 0
    cost_usd: float = 0.0
    error_class: Optional[str] = None


def _cut(value: Optional[str], length: int) -> Optional[str]:
    return value[:length] if value else None


def write_run(db: Session, record: RunRecord) -> ProactiveRun:
    """Insert the run's row and commit."""
    row = ProactiveRun(
        id=record.run_id, workspace_id=record.workspace_id, agent_id=record.agent_id, kind=record.kind,
        window_since=record.window_since, window_until=record.window_until, outcome=record.outcome,
        signals=[dict(signal) for signal in record.signals], suggestions=int(record.suggestions),
        model=_cut(record.model, MODEL_LENGTH), tokens=int(record.tokens),
        cost_usd=round(float(record.cost_usd), 6), error_class=_cut(record.error_class, ERROR_CLASS_LENGTH),
    )
    db.add(row)
    db.commit()
    return row


def _same_check_in(workspace_id: UUID, kind: str, agent_id: Optional[int]) -> Tuple[Any, ...]:
    agent = ProactiveRun.agent_id.is_(None) if agent_id is None else ProactiveRun.agent_id == agent_id
    return ProactiveRun.workspace_id == workspace_id, ProactiveRun.kind == kind, agent


def last_success(db: Session, workspace_id: UUID, kind: str, agent_id: Optional[int]) -> Optional[datetime]:
    """What this check-in's latest successful run read up to, or None when none succeeded."""
    latest = db.query(ProactiveRun).filter(
        *_same_check_in(workspace_id, kind, agent_id), ProactiveRun.outcome.in_(tuple(SUCCESSFUL_OUTCOMES)),
    ).order_by(ProactiveRun.created_at.desc()).limit(1).all()
    return last_success_at(latest)


def model_runs_today(db: Session, workspace_id: UUID, now: datetime) -> int:
    """The workspace's check-ins that reached the model in the last 24 hours: each run that
    names a model, whatever its outcome."""
    return int(db.query(func.count(ProactiveRun.id)).filter(
        ProactiveRun.workspace_id == workspace_id, ProactiveRun.model.isnot(None),
        ProactiveRun.created_at >= now - MODEL_RUN_WINDOW,
    ).scalar() or 0)


def suggestions_made(db: Session, workspace_id: UUID, run_id: UUID) -> int:
    """How many suggestions this run recorded."""
    return int(db.query(func.count(ProactiveSuggestion.id)).filter(
        ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.run_id == run_id,
    ).scalar() or 0)


def _titles(db: Session, *criteria: Any) -> Tuple[str, ...]:
    rows = db.query(ProactiveSuggestion.title).filter(*criteria).order_by(
        ProactiveSuggestion.created_at.desc()).limit(MAX_TITLES).all()
    return tuple(row[0] for row in rows)


def known_titles(db: Session, workspace_id: UUID, user_id: int, now: datetime) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """The person's open titles and the titles they hid in the last 14 days, newest first,
    at most ``MAX_TITLES`` each: a check-in is told never to repeat them."""
    mine = (ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id)
    hidden_at = func.coalesce(ProactiveSuggestion.decided_at, ProactiveSuggestion.last_seen_at,
                              ProactiveSuggestion.created_at)
    open_titles = _titles(db, *mine, ProactiveSuggestion.status == SUGGESTION_OPEN)
    hidden_titles = _titles(db, *mine, ProactiveSuggestion.status == SUGGESTION_HIDDEN,
                            hidden_at >= now - HIDDEN_WINDOW)
    return open_titles, hidden_titles

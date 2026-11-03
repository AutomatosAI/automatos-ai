"""PRD-268 §6.7 (US-C13c) — suggestion precision, live: what people did with the suggestions they were given.

There is no gold set: the numbers are read from the configured database, each workspace's, or one
workspace's (``python -m evals.proactive precision --workspace <id>``). Nothing is written: the read is
one SELECT in a READ ONLY transaction, rolled back, and no model is called.

A suggestion is decided once it is started or hidden (§6.7: started ÷ (started + hidden)); an open or
expired one is not. A started suggestion passed review when the ticket its Start filed
(``started_ref.board_task_id``, in the same workspace) is ``done`` in ``board_tasks`` and still waits
for a person's review (``review_mode`` human, as Start files it): done on such a ticket is a person
passing it in Review. Nothing else counts: a started suggestion whose Start filed no ticket (a chat
suggestion), whose ticket is gone, or whose ticket is not done did not pass review.

Each workspace is scored on its own (``score_precision``: started / decided and started and passed
review / decided against their bars, and each source agent under the slow share after the minimum
decisions); the run passes when every workspace read passes, and a run with no decided suggestion
does not. The numbers are the totals over the workspaces read; the audit lines give each workspace's
own numbers and its slow source agents, by id.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.models.proactive import SUGGESTION_HIDDEN, SUGGESTION_STARTED
from evals.proactive.runtime import EvalCannotRun
from evals.proactive.scoring import Decision, Score, score_precision
from services.proactive.start_ticket import REVIEW_BY_A_PERSON

__all__ = ["NAME", "evaluate", "open_session", "read_decisions", "workspace_scores"]

NAME = "precision"
# The board's done status (board_tasks.status: inbox, assigned, in_progress, review, done).
TICKET_DONE = "done"
STARTED_TICKET_KEY = "board_task_id"
READ_ONLY = "SET TRANSACTION READ ONLY"
# Every decided suggestion, with whether the ticket its Start filed is done under a person's review; one
# workspace when :workspace is given. Bound parameters only; the ticket's id is compared as text, so a
# started_ref that holds no integer matches no ticket instead of failing the cast.
_DECISIONS_SQL = text(
    "SELECT CAST(s.workspace_id AS text) AS workspace_id, s.source_agent_id, s.status, "
    "  COALESCE(bt.status = :done AND bt.review_mode = :reviewed, false) AS passed_review "
    "FROM proactive_suggestions s "
    "LEFT JOIN board_tasks bt ON bt.workspace_id = s.workspace_id "
    "  AND CAST(bt.id AS text) = s.started_ref ->> :ticket_key "
    "WHERE s.status IN (:started, :hidden) "
    "  AND (CAST(:workspace AS uuid) IS NULL OR s.workspace_id = CAST(:workspace AS uuid)) "
    "ORDER BY 1"
)
NOT_A_WORKSPACE = "--workspace is {value!r}, which is not a workspace id."
WORKSPACE_LINE = ("workspace {workspace}: {verdict}, decided {decided}, started {started} ({started_share}), "
                  "started and passed review {passed_review} ({passed_review_share})")
SLOW_AGENT_LINE = "   slow source agent {agent}: {share} of its {decided} decisions started"
NO_SHARE = "n/a"
DIGITS = 3

SessionFactory = Callable[[], Session]


def open_session() -> Session:
    """A session on the configured database."""
    from core.database.database import SessionLocal

    return SessionLocal()


def read_decisions(db: Session, workspace_id: Optional[str]) -> Dict[str, List[Decision]]:
    """Every decided suggestion as a ``Decision``, by workspace (one when ``workspace_id`` is given)."""
    rows = db.execute(_DECISIONS_SQL, {
        "done": TICKET_DONE, "reviewed": REVIEW_BY_A_PERSON, "ticket_key": STARTED_TICKET_KEY,
        "started": SUGGESTION_STARTED, "hidden": SUGGESTION_HIDDEN, "workspace": workspace_id,
    }).fetchall()
    found: Dict[str, List[Decision]] = defaultdict(list)
    for workspace, agent, status, passed in rows:
        started = status == SUGGESTION_STARTED
        found[str(workspace)].append(Decision(source_agent=agent, started=started, passed_review=started and passed))
    return dict(found)


def _read(factory: SessionFactory, workspace_id: Optional[str]) -> Dict[str, List[Decision]]:
    """``read_decisions`` in a READ ONLY transaction of its own, rolled back and closed after."""
    db = factory()
    try:
        db.execute(text(READ_ONLY))
        return read_decisions(db, workspace_id)
    finally:
        db.rollback()
        db.close()


def workspace_scores(decisions: Dict[str, List[Decision]]) -> Dict[str, Score]:
    """Each workspace's own score, by workspace id."""
    return {workspace: score_precision(rows) for workspace, rows in sorted(decisions.items())}


def _shown(value: Optional[float]) -> str:
    return NO_SHARE if value is None else f"{value:.{DIGITS}f}"


def _lines(workspace: str, score: Score) -> Tuple[str, ...]:
    n = score.numbers
    head = WORKSPACE_LINE.format(
        workspace=workspace, verdict=score.verdict, decided=n["decided"], started=n["started"],
        started_share=_shown(n["started_share"]), passed_review=n["passed_review"],
        passed_review_share=_shown(n["passed_review_share"]))
    slow = tuple(SLOW_AGENT_LINE.format(agent=agent, share=_shown(share), decided=decided)
                 for agent, decided, share in n["slow_agents"])
    return (head, *slow)


def _combined(scores: Dict[str, Score], everything: Iterable[Decision]) -> Score:
    """The totals over every workspace read, passed when each workspace passed (and there is one)."""
    total = score_precision(list(everything))
    numbers = {"workspaces": len(scores), "workspaces_under_bar": sum(1 for s in scores.values() if not s.passed),
               **total.numbers}
    audit = tuple(line for workspace, score in scores.items() for line in _lines(workspace, score))
    passed = bool(scores) and all(score.passed for score in scores.values())
    return replace(total, numbers=MappingProxyType(numbers), passed=passed, audit=audit)


def _workspace(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(UUID(str(value)))
    except ValueError as exc:
        raise EvalCannotRun(NOT_A_WORKSPACE.format(value=value)) from exc


async def evaluate(workspace_id: Optional[str] = None, session_factory: Optional[SessionFactory] = None) -> Score:
    """Read every decided suggestion (one workspace's, when given) and score each workspace's precision."""
    workspace = _workspace(workspace_id)
    decisions = await asyncio.to_thread(_read, session_factory or open_session, workspace)
    scores = workspace_scores(decisions)
    return _combined(scores, (d for rows in decisions.values() for d in rows))

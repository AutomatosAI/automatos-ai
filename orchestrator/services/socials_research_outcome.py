"""How a plan's research run ended, for the plan tick (19 Nov 2026).

On 18 Nov the plan's first Content bank research run added no topics, ended ``completed`` and
told nobody; two minutes later the tick made the whole week from titles with no facts. Two
rules close that:

* :func:`in_flight`: while the plan's latest research run has not finished (for at most
  ``RESEARCH_WAIT``), the tick makes none of the plan's posts: they are made from what research
  finds, never before it.
* :func:`tell_if_empty`: a finished run that gave the bank nothing (no topic added, none given
  facts) is told to the owner once, in the run's own words, so a silent "completed" never
  stands for a researched week.

Both read the run by the plan's ``research.last_run_id`` (``services/socials_plan_research.launch``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.models.core import RecipeExecution
from core.models.socials import SocialCampaign, SocialTopic
from modules.socials import plan_notify, plan_store

logger = logging.getLogger(__name__)

# The engine's ending statuses: any other (pending, running, queued, paused) is still going.
FINISHED = frozenset({"completed", "failed", "cancelled"})
# A run still unfinished after this long no longer holds the plan's posts back.
RESEARCH_WAIT = timedelta(minutes=30)
CHECKED = "checked_run_id"
EMPTY_EVENT = ("social_plan_research_unavailable", "Research added nothing: ", "error")
NO_REASON = "the run gave no topics and no facts. Research again, or add topics with their facts."
REASON_MAX_CHARS = 240


def _latest_run(db: Any, plan: SocialCampaign) -> Optional[RecipeExecution]:
    run_id = (plan.research or {}).get("last_run_id")
    if not run_id:
        return None
    return db.query(RecipeExecution).filter(
        RecipeExecution.execution_id == run_id, RecipeExecution.workspace_id == plan.workspace_id,
    ).first()


def _aware(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _launched_at(plan: SocialCampaign, run: RecipeExecution) -> Optional[datetime]:
    """When the run was asked for: the plan's ``last_run_at`` (an aware moment the launch wrote), else
    the run's ``started_at`` (a column without a zone, read as UTC)."""
    raw = (plan.research or {}).get("last_run_at")
    try:
        asked = _aware(datetime.fromisoformat(raw)) if raw else None
    except ValueError:
        asked = None
    return asked or _aware(getattr(run, "started_at", None))


def in_flight(db: Any, plan: SocialCampaign, now: datetime) -> bool:
    """The plan's latest research run has not finished, and was asked for within ``RESEARCH_WAIT``."""
    run = _latest_run(db, plan)
    if run is None or (run.status or "").lower() in FINISHED:
        return False
    launched = _launched_at(plan, run)
    return launched is not None and now - launched < RESEARCH_WAIT


def gave_the_bank(db: Any, plan: SocialCampaign, since: datetime) -> int:
    """How many of the plan's topics hold facts and were added or changed since ``since`` (read in
    Python: a bank holds at most ``topics.MAX_BANK``, and the clocks compare the same on every engine)."""
    rows = db.query(SocialTopic).filter(SocialTopic.campaign_id == plan.id).all()
    return sum(1 for topic in rows if topic.facts and _touched_since(topic, since))


def _touched_since(topic: SocialTopic, since: datetime) -> bool:
    moments = (_aware(topic.created_at), _aware(getattr(topic, "updated_at", None)))
    return any(moment is not None and moment >= since for moment in moments)


def reason_of(run: RecipeExecution) -> str:
    """The run's own first line, as the owner's notice says it."""
    output = run.output_data if isinstance(run.output_data, dict) else {}
    text = str(output.get("final_output") or run.error_message or "").strip()
    first = text.split("\n", 1)[0].strip() if text else ""
    return first[:REASON_MAX_CHARS] if first else NO_REASON


def tell_if_empty(db: Any, plan: SocialCampaign) -> bool:
    """A finished run that gave the bank nothing, told once (the plan remembers the run). Commits."""
    run = _latest_run(db, plan)
    research = dict(plan.research or {})
    if run is None or (run.status or "").lower() not in FINISHED or research.get(CHECKED) == run.execution_id:
        return False
    launched = _launched_at(plan, run)
    empty = launched is not None and gave_the_bank(db, plan, launched) == 0
    plan.research = {**research, CHECKED: run.execution_id}
    db.commit()
    if empty:
        logger.warning("[Socials] plan %s: research %s gave the bank nothing", plan.id, run.execution_id)
        plan_notify.notify_plan(plan.workspace_id, plan.id, EMPTY_EVENT, f"{plan.name}: {reason_of(run)}")
    return empty


def check_finished(session_factory: Any) -> int:
    """The plan tick's pass: each active plan's finished research run checked once; how many were empty."""
    db = session_factory()
    told = 0
    try:
        for plan in plan_store.active_plans(db):
            try:
                told += int(tell_if_empty(db, plan))
            except Exception:  # noqa: BLE001 — one plan never stops the pass; logged
                logger.exception("[Socials] plan %s: its research outcome could not be read", plan.id)
                db.rollback()
    finally:
        db.close()
    return told


__all__ = ["RESEARCH_WAIT", "check_finished", "gave_the_bank", "in_flight", "reason_of", "tell_if_empty"]

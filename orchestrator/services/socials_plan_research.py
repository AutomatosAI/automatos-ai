"""PRD-251B Wave 2 (B8; US-B204): research fills a plan's content bank.

The seeded playbook **Content bank research** (the Socials package, run by the Social
Media Director) reads the plan (``platform_get_social_plan``), then the workspace's
knowledge, Deliverables and brand-kit website, and its GitHub through Composio when it
is connected, and writes topics through the one draft-only tool
``platform_add_social_topics``. It runs:

* weekly, at the plan's ``research.day`` and ``research.time`` (default Monday 06:00, in
  the plan's timezone), when the plan tick (``socials_plan_maker.run_tick``) finds it
  due: from a week before the plan starts until it ends;
* on **Research again** (``POST /api/socials/plans/{id}/research``).

The workspace runs its own installed copy of the playbook. A plan's save installs it in a
workspace that never had it, and Research again puts a missing one back first
(PRD-251C US-C101, US-C102: ``services/socials_research_setup.py``). The weekly run never
installs: without a copy it tells the owner why, once a day, in the content bank's words. A
trial workspace on the hosted edition gets no weekly run (no background burn, PRD-222);
Research again still works. The run is a ``RecipeExecution`` launched through the playbook
engine, as a cron playbook is (``playbook_scheduler._fire_playbook``).
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import anyio

from config import config
from core.models.core import RecipeExecution, WorkflowTemplate
from core.models.socials import SocialCampaign
from core.models.workspaces import Workspace
from core.seeds.seed_socials_package import RESEARCH_PLAYBOOK_TEMPLATE_ID
from modules.socials import plan_notify, plan_store, plans, service
from modules.socials.settings import socials_off_reason

logger = logging.getLogger(__name__)

WEEKLY_TRIGGER = "socials_plan_research"
EXECUTION_TYPE = "socials_plan_research"
RESEARCH_LEAD = timedelta(days=7)  # research starts this long before the plan's first day
NO_PLAYBOOK = "Research needs this workspace's Content bank research playbook: Research again puts it back."
UNAVAILABLE_EVENT = ("social_plan_research_unavailable", "Research could not run: ", "error")


class ResearchUnavailable(service.SocialsError):
    """The workspace has no installed Content bank research playbook (409)."""


def installed_playbook(db: Any, workspace_id: UUID) -> Optional[WorkflowTemplate]:
    """The workspace's own copy of the Content bank research playbook, if installed."""
    source = db.query(WorkflowTemplate.id).filter(WorkflowTemplate.template_id == RESEARCH_PLAYBOOK_TEMPLATE_ID).first()
    if source is None:
        return None
    return (
        db.query(WorkflowTemplate)
        .filter(WorkflowTemplate.workspace_id == workspace_id, WorkflowTemplate.cloned_from_id == source.id)
        .order_by(WorkflowTemplate.id.desc())
        .first()
    )


def _parse(value: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def last_weekly_moment(plan: SocialCampaign, now: datetime) -> datetime:
    """The latest weekly research moment at or before ``now``, in UTC."""
    settings = plans.validate_research(plan.research)
    zone = plans.zone_of(plan)
    local = now.astimezone(zone)
    back = (local.weekday() - plans.WEEKDAYS.index(settings["day"])) % 7
    moment = plans.local_to_utc(local.date() - timedelta(days=back), settings["time"], zone)
    return moment if moment <= now else moment - timedelta(days=7)


def research_due(plan: SocialCampaign, now: datetime) -> bool:
    """Weekly research is due: switched on, the plan active and within its research window,
    and no run since the latest weekly moment."""
    research = plan.research or {}
    if plan.status != plans.ACTIVE or not plans.validate_research(research)["enabled"]:
        return False
    if plan.starts_on is None or plan.ends_on is None:
        return False
    today = now.astimezone(plans.zone_of(plan)).date()
    if not plan.starts_on - RESEARCH_LEAD <= today <= plan.ends_on:
        return False
    last = _parse(research.get("last_run_at"))
    return last is None or last < last_weekly_moment(plan, now)


def launch(db: Any, plan: SocialCampaign, *, triggered_by: str, now: datetime) -> str:
    """Start the plan's research run (committed first, as the engine expects); its execution id.

    Runs on a worker thread (a plain-``def`` route, the plan tick): the engine's task
    starts on the event loop from there.
    """
    from services.playbook_engine import get_playbook_engine

    playbook = installed_playbook(db, plan.workspace_id)
    if playbook is None:
        raise ResearchUnavailable(NO_PLAYBOOK)
    execution_id = f"research-{uuid4().hex[:12]}"
    inputs = {"plan_id": str(plan.id), "plan_name": plan.name}
    db.add(RecipeExecution(
        execution_id=execution_id, recipe_id=playbook.id, workspace_id=plan.workspace_id, status="pending",
        input_data=inputs, triggered_by=triggered_by,
        execution_metadata={"execution_type": EXECUTION_TYPE, "total_steps": len(playbook.steps or []), "plan_id": str(plan.id)},
    ))
    plan.research = {**(plan.research or {}), "last_run_at": now.isoformat(), "last_run_id": execution_id}
    db.commit()
    anyio.from_thread.run_sync(functools.partial(
        get_playbook_engine().launch, recipe_execution_id=execution_id, recipe_id=playbook.id,
        workspace_id=UUID(str(plan.workspace_id)), input_data=inputs,
    ))
    logger.info("[Socials] plan %s: research %s started (%s)", plan.id, execution_id, triggered_by)
    return execution_id


def background_allowed(workspace: Optional[Workspace]) -> bool:
    """Socials on, and no trial workspace's background burn on the hosted edition (PRD-222):
    the weekly research run, and the results' reads (PRD-251C US-C402)."""
    if workspace is None or socials_off_reason(workspace) is not None:
        return False
    if (config.AUTH_EDITION or "").strip().lower() == "local":
        return True
    from services.trial_ledger import is_trial_active_workspace

    return not is_trial_active_workspace(workspace)


def _tell_unavailable(db: Any, plan: SocialCampaign, now: datetime) -> None:
    """The weekly run without a playbook (C4): it never installs one; it says why, once a day."""
    from services.socials_research_setup import research_note

    if not plan_notify.once_today(plan, UNAVAILABLE_EVENT[0], now.astimezone(plans.zone_of(plan)).date()):
        return
    note = research_note(db, plan.workspace_id) or NO_PLAYBOOK
    db.commit()
    plan_notify.notify_plan(plan.workspace_id, plan.id, UNAVAILABLE_EVENT, f"{plan.name}: {note}")


def launch_due(now: datetime) -> int:
    """The plan tick's research pass: each plan whose weekly research is due, started. How many."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    started = 0
    try:
        for plan in plan_store.active_plans(db):
            if not research_due(plan, now) or not background_allowed(db.get(Workspace, plan.workspace_id)):
                continue
            try:
                launch(db, plan, triggered_by=WEEKLY_TRIGGER, now=now)
                started += 1
            except ResearchUnavailable:
                db.rollback()
                _tell_unavailable(db, plan, now)
            except Exception:  # noqa: BLE001 — one plan never stops the pass; logged
                logger.exception("[Socials] plan %s: the weekly research could not start", plan.id)
                db.rollback()
    finally:
        db.close()
    return started

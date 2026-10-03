"""PRD-251C Wave 1 (C4; US-C101): research comes with the first plan.

A plan's research runs the workspace's own copy of the marketplace playbook **Content bank
research** (``services/socials_plan_research.py``). A workspace that never installed the
Socials package had no copy, so every Research again answered 409 (the owner's test, 3 Oct).

Saving a plan (``POST`` or ``PUT /api/socials/plans``) in a workspace that never had research
installs that playbook, with the agent its step runs (the Social Media Director), through the
package installer (``package_installer.install_playbook``): that playbook only, not the rest
of the Socials package. A copy already there (the whole package installed before) is left as
it is. "Never had" is ``workspace.settings['socials'].research_installed_at``, set on that
first install: new workspaces get research with their first plan, workspaces from before
PRD-251C with their next plan save, and a playbook the owner deleted on purpose is not put
back by a save.

The install runs after the plan's commit, inline on the route's worker thread: the installer
is async, so it runs on the event loop through ``anyio.from_thread`` (never ``asyncio.run``
in a threadpool thread). Not fire-and-forget: Plan with Auto starts research right after
the save, and must find the copy. A failure is logged and leaves the plan saved; the
content bank then says research is not set up (``research_note``).

Two installs in one workspace never both clone: each takes the workspace's research lock,
a transaction-scoped advisory lock that is never waited for (a blocking wait from sync
SQLAlchemy on the event loop would freeze it, F105). The one that does not get it installs
nothing.
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

import anyio
from sqlalchemy import text

from core.models.workspaces import Workspace
from core.seeds.seed_socials_package import RESEARCH_PLAYBOOK_TEMPLATE_ID
from modules.socials.settings import WORKSPACE_SOCIALS_SETTINGS_KEY
from services import package_installer, socials_plan_research
from services.agent_quota import AgentLimitReached

logger = logging.getLogger(__name__)

KEY_RESEARCH_INSTALLED_AT = "research_installed_at"
# The research lock's key space in the database ('socr'), apart from every other advisory lock.
RESEARCH_LOCK_NAMESPACE = 0x736F6372
_TRY_LOCK = text("SELECT pg_try_advisory_xact_lock(:namespace, hashtext(:key))")

# What an install did.
INSTALLED = "installed"  # the playbook was cloned into the workspace now
PRESENT = "present"  # the workspace had a copy already: left as it is
SKIPPED = "skipped"  # the workspace had research before (the flag): a save puts nothing back
BUSY = "busy"  # another install in this workspace holds the lock

# What the content bank says while research cannot run.
NOT_SET_UP = "Research is not set up in this workspace yet."
REMOVED = "Research is off: its playbook was removed."


def research_installed_at(settings: Optional[Dict[str, Any]]) -> Optional[str]:
    """When the workspace first had research (ISO), or None: never."""
    socials = (settings or {}).get(WORKSPACE_SOCIALS_SETTINGS_KEY)
    value = socials.get(KEY_RESEARCH_INSTALLED_AT) if isinstance(socials, dict) else None
    return str(value) if value else None


def with_research_installed_at(settings: Optional[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
    """A new settings object with the flag set at ``now``; the one given is not changed."""
    current = dict(settings or {})
    socials = current.get(WORKSPACE_SOCIALS_SETTINGS_KEY)
    socials = dict(socials) if isinstance(socials, dict) else {}
    return {**current, WORKSPACE_SOCIALS_SETTINGS_KEY: {**socials, KEY_RESEARCH_INSTALLED_AT: now.isoformat()}}


def _lock_is_ours(db: Any, workspace_id: UUID) -> bool:
    """Take the workspace's research lock until this transaction ends, without waiting."""
    if db.get_bind().dialect.name != "postgresql":
        return True  # SQLite in the unit tests: one process, nothing to race
    params = {"namespace": RESEARCH_LOCK_NAMESPACE, "key": str(workspace_id)}
    return bool(db.execute(_TRY_LOCK, params).scalar())


async def _install_locked(db: Any, workspace_id: UUID, now: datetime, first_only: bool) -> str:
    if not _lock_is_ours(db, workspace_id):
        return BUSY
    workspace = db.query(Workspace).populate_existing().filter(Workspace.id == workspace_id).one_or_none()
    if workspace is None or (first_only and research_installed_at(workspace.settings)):
        return SKIPPED
    outcome = PRESENT
    if socials_plan_research.installed_playbook(db, workspace_id) is None:
        await package_installer.install_playbook(db, workspace_id, RESEARCH_PLAYBOOK_TEMPLATE_ID)
        outcome = INSTALLED
    if not research_installed_at(workspace.settings):
        workspace.settings = with_research_installed_at(workspace.settings, now)
    return outcome


async def install(db: Any, workspace_id: UUID, *, now: datetime, first_only: bool) -> str:
    """Install the research playbook when the workspace has no copy, and record that it had
    research; commits. With ``first_only``, a workspace that had research before gets
    nothing (``SKIPPED``). Runs on the event loop. Raises what the installer raises
    (``PackageInstallError``: no marketplace playbook; ``AgentLimitReached``: the plan is full
    of agents), with nothing installed."""
    try:
        outcome = await _install_locked(db, workspace_id, now, first_only)
    except Exception:
        db.rollback()
        raise
    db.commit()  # the lock goes with the transaction
    return outcome


def after_plan_save(db: Any, workspace_id: UUID, now: datetime) -> None:
    """US-C101, on a plan route's worker thread once the plan is committed: a workspace that
    never had research gets it. Never raises: a failure is logged, and the plan stays saved."""
    workspace = db.get(Workspace, workspace_id)
    if workspace is None or research_installed_at(workspace.settings):
        return
    try:
        outcome = anyio.from_thread.run(functools.partial(install, db, workspace_id, now=now, first_only=True))
    except (package_installer.PackageInstallError, AgentLimitReached) as exc:  # no seed, or the plan is full
        logger.warning("[Socials] workspace %s: research was not set up with a plan's save: %s", workspace_id, exc)
        return
    except Exception:  # noqa: BLE001 — research's install never fails a plan's save; logged
        logger.exception("[Socials] workspace %s: research could not be set up with a plan's save", workspace_id)
        return
    logger.info("[Socials] workspace %s: research set up with a plan's save (%s)", workspace_id, outcome)


def research_note(db: Any, workspace_id: UUID) -> Optional[str]:
    """What the content bank says while research cannot run in the workspace; None when it can."""
    if socials_plan_research.installed_playbook(db, workspace_id) is not None:
        return None
    workspace = db.get(Workspace, workspace_id)
    return REMOVED if workspace is not None and research_installed_at(workspace.settings) else NOT_SET_UP

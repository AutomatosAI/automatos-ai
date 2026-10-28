"""PRD-268 §6.9, FR-13 — the caps a workspace's check-ins are held to, and what they count.

* ``default_caps``: the §6.9 caps as ``config`` sets them (``PROACTIVE_MAX_MODEL_RUNS_PER_DAY``
  and the rest), for every workspace that set none of its own.
* ``caps_from``: a workspace's own caps, ``settings.proactive.caps`` (its owners' and
  admins', ``services.proactive.settings``), each over its default. A workspace can only
  tighten the platform's caps: a stored cap looser than config's (above it, or a shorter
  minimum interval) is held to config's, and ``looser_than_platform`` names the caps a
  change would loosen, which the settings route refuses. A value the caps refuse
  (``modules.proactive.Caps``) is a ``ValueError``: nothing is checked against caps that
  are wrong.
* ``workspace_caps``: read from the workspace row. A read that fails raises, and so do caps
  that are wrong: the run start refuses the check-in (fail closed, ``services.proactive.gates``)
  and the suggestion tool records nothing.
* ``room_refusal``: the suggestion cap one more suggestion would pass, counted from
  ``proactive_suggestions`` (``new_today``, ``open_for_agent``), for ``platform_suggest_next_step``.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import config
from core.models.proactive import SUGGESTION_OPEN, ProactiveSuggestion
from modules.proactive import CapRefusal, Caps, suggestion_refusal
from services.proactive.switches import WORKSPACE_KEY

__all__ = [
    "CAPS_KEY",
    "EDITABLE_CAPS",
    "NEW_PER_DAY_WINDOW",
    "caps_as_dict",
    "caps_from",
    "default_caps",
    "looser_than_platform",
    "new_today",
    "open_for_agent",
    "room_refusal",
    "workspace_caps",
]

# A workspace's own caps: workspaces.settings[WORKSPACE_KEY][CAPS_KEY][<cap>].
CAPS_KEY = "caps"
# The caps a workspace may set. The suggestion push's (max_pushes_per_person_per_day) is W2's.
EDITABLE_CAPS = (
    "max_model_runs_per_day",
    "max_suggestions_per_run",
    "max_new_per_person_per_day",
    "max_open_per_person",
    "max_open_per_agent",
    "min_interval_minutes",
    "max_interval_minutes",
)
# The one cap a workspace tightens by raising it; every other cap it tightens by lowering.
RAISED_TO_TIGHTEN = frozenset({"min_interval_minutes"})
# "New suggestions per person per day" (§6.9) counts the last 24 hours.
NEW_PER_DAY_WINDOW = timedelta(hours=24)


def default_caps() -> Caps:
    """The §6.9 caps as config sets them. Values the caps refuse raise ``ValueError``."""
    return replace(
        Caps(),
        max_model_runs_per_day=config.PROACTIVE_MAX_MODEL_RUNS_PER_DAY,
        max_suggestions_per_run=config.PROACTIVE_MAX_SUGGESTIONS_PER_RUN,
        max_new_per_person_per_day=config.PROACTIVE_MAX_NEW_PER_PERSON_PER_DAY,
        max_open_per_person=config.PROACTIVE_MAX_OPEN_PER_PERSON,
        max_open_per_agent=config.PROACTIVE_MAX_OPEN_PER_AGENT,
        min_interval_minutes=config.PROACTIVE_MIN_INTERVAL_MINUTES,
        max_interval_minutes=config.PROACTIVE_MAX_INTERVAL_MINUTES,
    )


def _stated_caps(settings: Any) -> Mapping[str, Any]:
    """``settings.proactive.caps`` as stored, or nothing; anything but a mapping is refused."""
    proactive = settings.get(WORKSPACE_KEY) if isinstance(settings, dict) else None
    stated = proactive.get(CAPS_KEY) if isinstance(proactive, dict) else None
    if stated is None:
        return {}
    if not isinstance(stated, dict):
        raise ValueError(f"settings.{WORKSPACE_KEY}.{CAPS_KEY} is not a set of caps")
    return stated


def _looser(name: str, value: int, platform: Caps) -> bool:
    limit = getattr(platform, name)
    return value < limit if name in RAISED_TO_TIGHTEN else value > limit


def looser_than_platform(changes: Mapping[str, Optional[int]], platform: Caps) -> Tuple[str, ...]:
    """The caps in ``changes`` set looser than the platform's (None, back to the default, is not)."""
    return tuple(name for name in EDITABLE_CAPS
                 if isinstance(changes.get(name), int) and _looser(name, changes[name], platform))


def _held_to(own: Caps, platform: Caps) -> Caps:
    """``own`` with every cap it set looser than the platform's held to the platform's."""
    return replace(own, **{name: getattr(platform, name) for name in EDITABLE_CAPS
                           if _looser(name, getattr(own, name), platform)})


def caps_from(settings: Any) -> Caps:
    """The caps a workspace's ``settings`` state, each cap it leaves out at its default and
    none looser than the platform's. A cap the caps refuse raises ``ValueError``; names that
    are not caps are ignored."""
    platform = default_caps()
    stated = _stated_caps(settings)
    own = replace(platform, **{name: stated[name] for name in EDITABLE_CAPS if name in stated})
    return _held_to(own, platform)


def caps_as_dict(caps: Caps) -> Dict[str, int]:
    """The caps a workspace may set, by name, as ``GET /api/proactive/settings`` shows them."""
    return {name: getattr(caps, name) for name in EDITABLE_CAPS}


def workspace_caps(db: Session, workspace_id: UUID) -> Caps:
    """The workspace's caps. A read that fails raises, as do caps that are wrong
    (``ValueError``); a workspace that is not there has the defaults."""
    from core.models.workspaces import Workspace

    row = db.query(Workspace.settings).filter(Workspace.id == workspace_id).first()
    return caps_from(row[0] if row is not None else None)


def _count(db: Session, *criteria: Any) -> int:
    return int(db.query(func.count(ProactiveSuggestion.id)).filter(*criteria).scalar() or 0)


def new_today(db: Session, workspace_id: UUID, user_id: int, now: datetime) -> int:
    """The suggestions made for the person in the workspace in the last 24 hours, whatever their state."""
    return _count(db, ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id,
                  ProactiveSuggestion.created_at >= now - NEW_PER_DAY_WINDOW)


def open_for_agent(db: Session, workspace_id: UUID, agent_id: int) -> int:
    """The agent's suggestions in the workspace that are still open."""
    return _count(db, ProactiveSuggestion.workspace_id == workspace_id,
                  ProactiveSuggestion.source_agent_id == agent_id, ProactiveSuggestion.status == SUGGESTION_OPEN)


def room_refusal(db: Session, workspace_id: UUID, user_id: int, agent_id: Optional[int], caps: Caps,
                 now: datetime, made_this_run: int = 0) -> Optional[CapRefusal]:
    """The cap one more suggestion for the person would pass (``modules.proactive.suggestion_refusal``),
    with what the database counts: ``agent_id`` is the source agent, None for Auto's check-in."""
    return suggestion_refusal(
        caps, made_this_run=made_this_run, new_today_for_person=new_today(db, workspace_id, user_id, now),
        open_for_agent=None if agent_id is None else open_for_agent(db, workspace_id, agent_id))

"""Which memory profile a distillation uses, and what it adds to the prompt (Wave 4, §5.6).

A profile (``core.models.memory_profiles``) says what counts as a durable fact in one domain.
An agent's ``configuration.memory_profile`` picks it, else its workspace's
``settings.memory_profile``, else ``general``. ``general`` is today's distillation prompt,
unchanged; any other profile adds its focus, what to keep and skip, and worked examples
before the prompt's writing rules. A workspace's own profile wins over a seeded one with the
same slug. A slug that names no profile is logged and distils as ``general``.

Off unless ``MEMORY_PROFILES_ENABLED``: then every distillation is ``general``, as before.
The fact types stay the platform's (``MEMORY_FACT_TYPES``): a profile changes what is kept,
not how it is classified.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

PROFILE_KEY = "memory_profile"
GENERAL = "general"
SLUG_MAX = 64
# The distillation prompt's first writing rule: the profile goes in just before it.
WRITING_RULES_ANCHOR = "Write each `fact` as a standalone"
PROFILE_HEADING = "What counts as a durable fact here"


@dataclass(frozen=True)
class Profile:
    """What a profile adds to the distillation prompt."""

    slug: str
    name: str
    focus: str
    keep: Tuple[str, ...] = ()
    skip: Tuple[str, ...] = ()
    examples: Tuple[Mapping[str, Any], ...] = ()


def slug_errors(configuration: Any) -> List[str]:
    """Errors (empty when valid) for an agent's ``configuration.memory_profile``: absent, or
    a profile's slug. Whether the slug names a profile is checked when a distillation uses it."""
    raw = configuration.get(PROFILE_KEY) if isinstance(configuration, Mapping) else None
    if raw is None or (isinstance(raw, str) and raw.strip() and len(raw.strip()) <= SLUG_MAX):
        return []
    return [f"configuration.{PROFILE_KEY} must be a profile's slug, such as {GENERAL!r}"]


def chosen_slug(agent_configuration: Any, workspace_settings: Any) -> str:
    """The agent's profile, else the workspace's, else ``general``."""
    for source in (agent_configuration, workspace_settings):
        value = source.get(PROFILE_KEY) if isinstance(source, Mapping) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return GENERAL


def profile_for(db: Any, workspace_id: Any, agent_id: Optional[int]) -> Optional[Profile]:
    """The profile a distillation for ``agent_id`` in ``workspace_id`` uses; None for
    ``general`` (today's prompt) and whenever the dial is off."""
    from config import config

    if not getattr(config, "MEMORY_PROFILES_ENABLED", False):
        return None
    slug = chosen_slug(_agent_configuration(db, agent_id), _workspace_settings(db, workspace_id))
    if slug == GENERAL:
        return None
    row = _profile_row(db, workspace_id, slug)
    if row is None:
        logger.warning("[memory] memory profile %r names no profile in workspace %s; distilling as general",
                       slug, workspace_id)
        return None
    return Profile(slug=row.slug, name=row.name, focus=row.focus or "", keep=tuple(row.keep or ()),
                   skip=tuple(row.skip or ()), examples=tuple(row.examples or ()))


def _agent_configuration(db: Any, agent_id: Optional[int]) -> Any:
    from core.models import Agent

    if agent_id is None:
        return None
    row = db.query(Agent.configuration).filter(Agent.id == agent_id).first()
    return row[0] if row else None


def _workspace_settings(db: Any, workspace_id: Any) -> Any:
    from core.models import Workspace

    if not workspace_id:
        return None
    row = db.query(Workspace.settings).filter(Workspace.id == str(workspace_id)).first()
    return row[0] if row else None


def _profile_row(db: Any, workspace_id: Any, slug: str) -> Any:
    """The workspace's own profile with ``slug``, else the seeded one."""
    from core.models.memory_profiles import MemoryProfile

    rows = (db.query(MemoryProfile)
            .filter(MemoryProfile.slug == slug,
                    (MemoryProfile.workspace_id == str(workspace_id)) | MemoryProfile.workspace_id.is_(None))
            .all())
    return next((r for r in rows if r.workspace_id is not None), rows[0] if rows else None)


async def distillation_profile(workspace_id: Any, agent_id: Optional[int]) -> Optional[Profile]:
    """``profile_for`` on its own session, off the event loop. A read that fails is logged and
    the distillation is ``general``: a profile never stops a turn's memory."""
    from config import config

    if not getattr(config, "MEMORY_PROFILES_ENABLED", False):
        return None
    return await asyncio.to_thread(_profile_on_own_session, workspace_id, agent_id)


def _profile_on_own_session(workspace_id: Any, agent_id: Optional[int]) -> Optional[Profile]:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return profile_for(db, workspace_id, agent_id)
    except Exception:  # noqa: BLE001 — logged; the distillation goes on as general
        logger.exception("[memory] could not read the memory profile for workspace %s; distilling as general",
                         workspace_id)
        return None
    finally:
        db.close()


def profile_block(profile: Profile) -> str:
    """The profile as the distillation prompt reads it."""
    lines = [f"{PROFILE_HEADING} ({profile.name} profile): {profile.focus}".rstrip()]
    if profile.keep:
        lines += ["Keep:", *(f"- {item}" for item in profile.keep)]
    if profile.skip:
        lines += ["Skip:", *(f"- {item}" for item in profile.skip)]
    if profile.examples:
        lines += ["Examples, as you would return them:",
                  *(json.dumps({"fact": e.get("fact"), "type": e.get("type")}, ensure_ascii=False)
                    for e in profile.examples)]
    return "\n".join(lines)


def with_profile(prompt: str, profile: Optional[Profile]) -> str:
    """``prompt`` with the profile before its writing rules; unchanged for ``general`` (None)."""
    if profile is None:
        return prompt
    block = profile_block(profile)
    if WRITING_RULES_ANCHOR in prompt:
        return prompt.replace(WRITING_RULES_ANCHOR, f"{block}\n\n{WRITING_RULES_ANCHOR}", 1)
    logger.warning("[memory] the distillation prompt has no writing rules to place the profile before; appended")
    return f"{prompt}\n\n{block}"

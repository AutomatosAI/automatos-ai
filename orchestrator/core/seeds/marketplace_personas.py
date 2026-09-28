"""
A seeded marketplace agent's persona is its seed's (PRD-251 P251W1-RVW-1)
=========================================================================

Every install copies a marketplace agent's persona onto the installing workspace's
clone: ``clone_agent_to_workspace`` (a package, a Playbook's cascade, a single agent)
and vertical provisioning's ``_seed_roster``. The marketplace row is global, so a
persona changed on it would run in every workspace that installs the agent
afterwards, with that workspace's tools, data and connections.

No route writes it: ``PUT /api/agents/{agent_id}/persona`` answers 404 for any agent
outside the caller's workspace. The seeders put the seed's persona back at every
boot all the same, so a row changed before that fix, or by hand, is restored before
the next install copies it.
"""
from __future__ import annotations

import logging
from typing import List, Mapping

from sqlalchemy.orm import Session

from core.models.core import Agent

logger = logging.getLogger(__name__)

MARKETPLACE = "marketplace"
# How much of a replaced prompt the restore logs: enough to tell a planted prompt from
# an older seed's text after the row itself no longer holds it.
LOGGED_PROMPT_CHARS = 200


def restore_seeded_personas(db: Session, prompts: Mapping[str, str]) -> List[str]:
    """Put the seed's persona back on each marketplace agent named in ``prompts``
    (slug to the seed's persona prompt) whose persona differs from it. Returns the
    slugs restored; the caller commits."""
    rows = db.query(Agent).filter(Agent.owner_type == MARKETPLACE, Agent.slug.in_(list(prompts))).all()
    return [row.slug for row in rows if _restore(row, prompts[row.slug])]


def _restore(agent: Agent, prompt: str) -> bool:
    """A seeded persona is a custom prompt with no persona row: put it back where it differs.
    What it replaces is logged (the prompt as a repr, so it cannot forge log lines)."""
    if agent.persona_id is None and agent.custom_persona_prompt == prompt and agent.use_custom_persona is True:
        return False
    logger.warning(
        "Marketplace agent %s (id=%s): its persona differed from its seed; the seed's is restored. "
        "Replaced persona_id=%s use_custom_persona=%s custom_persona_prompt=%r (first %d characters)",
        agent.slug, agent.id, agent.persona_id, agent.use_custom_persona,
        (agent.custom_persona_prompt or "")[:LOGGED_PROMPT_CHARS], LOGGED_PROMPT_CHARS,
    )
    agent.persona_id = None
    agent.custom_persona_prompt = prompt
    agent.use_custom_persona = True
    return True

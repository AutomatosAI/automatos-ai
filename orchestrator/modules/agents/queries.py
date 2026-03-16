"""
Agent Query Utilities
=====================

Centralized agent loading with eager relationship loading.
Prevents N+1 queries when ContextService sections access agent.skills,
agent.persona, etc.

Usage:
    from modules.agents.queries import get_agent_with_context

    agent = get_agent_with_context(db, agent_id)
    # agent.skills and agent.persona are pre-loaded
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from core.models import Agent

logger = logging.getLogger(__name__)


def get_agent_with_context(db: Session, agent_id: int) -> Optional[Agent]:
    """Load an agent with all associations needed for ContextService.

    Eagerly loads:
    - skills (used by SkillsSection)
    - persona (used by IdentitySection)

    Use this for any path that feeds into ContextService.build_context().
    For simple status checks or admin endpoints, a bare query is fine.
    """
    return (
        db.query(Agent)
        .options(
            joinedload(Agent.skills),
            joinedload(Agent.persona),
        )
        .filter_by(id=agent_id)
        .first()
    )

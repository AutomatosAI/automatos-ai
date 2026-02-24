"""
PRD-64: Semantic Agent Indexer for Tier 2.5 Routing.

Embeds each agent's capabilities (description, tags, tools, persona, skills,
plugins) into a vector and matches incoming queries by cosine similarity.

Reuses the centralized EmbeddingManager (qwen3-embedding-8b, 2048-dim).
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from core.models.core import Agent

logger = logging.getLogger(__name__)

# --- Thresholds ---
SIMILARITY_DIRECT_ROUTE = 0.75   # Route directly if top match exceeds this
SIMILARITY_CANDIDATE_MIN = 0.45  # Include in Tier 3 candidate list above this
MAX_LLM_CANDIDATES = 5           # Max agents passed to Tier 3


# ------------------------------------------------------------------
# Text construction
# ------------------------------------------------------------------

def build_agent_semantic_text(agent: Agent, db: Session) -> str:
    """Concatenate all agent capability data into a single text block for embedding.

    Pulls from: name, description, tags, marketplace_category, agent_type,
    persona, skills, connected apps/tools, and assigned plugins.
    """
    parts: List[str] = []

    # Core identity
    parts.append(f"Agent: {agent.name}")
    if agent.description:
        parts.append(f"Description: {agent.description}")
    if agent.agent_type:
        parts.append(f"Type: {agent.agent_type}")
    if agent.marketplace_category:
        parts.append(f"Category: {agent.marketplace_category}")

    # Tags
    tags = agent.tags or []
    if tags:
        parts.append(f"Tags: {', '.join(str(t) for t in tags)}")

    # Persona
    try:
        if agent.persona_id and agent.persona:
            persona = agent.persona
            parts.append(f"Persona: {persona.name}")
            if persona.description:
                parts.append(f"Persona description: {persona.description}")
    except Exception:
        logger.debug("[semantic] Could not load persona for agent %d", agent.id, exc_info=True)

    # Custom persona prompt
    if agent.use_custom_persona and agent.custom_persona_prompt:
        parts.append(f"Custom persona: {agent.custom_persona_prompt[:300]}")

    # Skills
    try:
        if agent.skills:
            skill_parts = []
            for skill in agent.skills:
                desc = f": {skill.description}" if skill.description else ""
                skill_parts.append(f"{skill.name}{desc}")
            parts.append(f"Skills: {'; '.join(skill_parts)}")
    except Exception:
        logger.debug("[semantic] Could not load skills for agent %d", agent.id, exc_info=True)

    # Connected apps (via AgentAppAssignment)
    try:
        from core.models.composio_cache import AgentAppAssignment, ComposioAppCache

        app_assignments = (
            db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent.id,
                AgentAppAssignment.is_active.is_(True),
            )
            .all()
        )
        if app_assignments:
            app_texts = []
            for assignment in app_assignments:
                app_name = assignment.app_name
                # Try to get description from cache
                cached = (
                    db.query(ComposioAppCache)
                    .filter(ComposioAppCache.app_name == app_name)
                    .first()
                )
                desc = ""
                if cached and cached.description:
                    desc = f": {cached.description[:120]}"
                app_texts.append(f"{app_name}{desc}")
            parts.append(f"Connected apps: {'; '.join(app_texts)}")
    except Exception:
        logger.debug("[semantic] Could not load apps for agent %d", agent.id, exc_info=True)

    # Assigned plugins
    try:
        if agent.assigned_plugins:
            from core.models.marketplace_plugins import MarketplacePlugin

            plugin_texts = []
            for ap in agent.assigned_plugins:
                plugin = db.query(MarketplacePlugin).get(ap.plugin_id)
                if plugin:
                    desc = f": {plugin.description[:120]}" if plugin.description else ""
                    plugin_texts.append(f"{plugin.name}{desc}")
            if plugin_texts:
                parts.append(f"Plugins: {'; '.join(plugin_texts)}")
    except Exception:
        logger.debug("[semantic] Could not load plugins for agent %d", agent.id, exc_info=True)

    return "\n".join(parts)


def compute_text_hash(text: str) -> str:
    """SHA-256 hex digest for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------

async def embed_agent(agent: Agent, db: Session, *, force: bool = False) -> bool:
    """Generate and store semantic embedding for a single agent.

    Returns True if the embedding was (re)generated, False if skipped.
    """
    text = build_agent_semantic_text(agent, db)
    text_hash = compute_text_hash(text)

    if not force and agent.semantic_text_hash == text_hash and agent.semantic_embedding:
        logger.debug("[semantic] Agent %d text unchanged, skipping embed", agent.id)
        return False

    from core.llm.embedding_manager import get_embedding_manager

    embedding_mgr = get_embedding_manager()
    try:
        vector = await embedding_mgr.generate_embedding(text)
    except Exception:
        logger.exception("[semantic] Failed to embed agent %d", agent.id)
        return False

    agent.semantic_embedding = vector
    agent.semantic_text_hash = text_hash
    db.add(agent)
    db.commit()

    logger.info("[semantic] Embedded agent %d (%s) — %d dims", agent.id, agent.name, len(vector))
    return True


async def embed_workspace_agents(
    workspace_id, db: Session, *, force: bool = False
) -> int:
    """Embed all active agents in a workspace. Returns count of (re)embedded agents."""
    agents = (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id, Agent.status == "active")
        .all()
    )

    count = 0
    for agent in agents:
        try:
            if await embed_agent(agent, db, force=force):
                count += 1
        except Exception:
            logger.exception("[semantic] Error embedding agent %d", agent.id)

    logger.info(
        "[semantic] Workspace %s: embedded %d/%d agents (force=%s)",
        workspace_id, count, len(agents), force,
    )
    return count


# ------------------------------------------------------------------
# Similarity search
# ------------------------------------------------------------------

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def find_similar_agents(
    query: str,
    workspace_id,
    db: Session,
    *,
    min_score: float = SIMILARITY_CANDIDATE_MIN,
) -> List[Tuple[Agent, float]]:
    """Embed the query and return agents ranked by cosine similarity.

    Only returns agents whose score >= *min_score*, sorted descending.
    """
    # Get agents that have embeddings
    agents = (
        db.query(Agent)
        .filter(
            Agent.workspace_id == workspace_id,
            Agent.status == "active",
            Agent.semantic_embedding.isnot(None),
        )
        .all()
    )

    if not agents:
        return []

    from core.llm.embedding_manager import get_embedding_manager

    embedding_mgr = get_embedding_manager()
    try:
        query_vec = await embedding_mgr.generate_embedding(query)
    except Exception:
        logger.exception("[semantic] Failed to embed query")
        return []

    scored: List[Tuple[Agent, float]] = []
    for agent in agents:
        score = cosine_similarity(query_vec, agent.semantic_embedding)
        if score >= min_score:
            scored.append((agent, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

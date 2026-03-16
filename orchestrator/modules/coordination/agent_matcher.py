"""
Agent Matcher — PRD-82A
========================

Deterministic scoring to select the best roster agent for a mission task.

Scoring weights (from PRD-102 Section 6.2, rebalanced for 82A):
  - skill_match:    0.40  — agent's skill/description matches task's agent_role
  - tool_coverage:  0.25  — fraction of task's required tools the agent has
  - model_fit:      0.15  — agent's model context + capability for the task
  - availability:   0.10  — agent has no running tasks in current missions
  - history:        0.10  — placeholder (0.5) until wired in 82B

Threshold: 0.4 minimum score to be considered a match.
Returns the single best-scoring agent, or None.

Source: PRD-82A Section 12 (US-010), PRD-102 Section 6.2
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.models.composio_cache import AgentAppAssignment
from core.models.core import Agent
from core.models.orchestration import OrchestrationTask
from core.models.orchestration_enums import TaskState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# Rebalanced: skill_match is the primary differentiator for roster selection.
# ---------------------------------------------------------------------------

WEIGHT_SKILL_MATCH: float = 0.40
WEIGHT_TOOL_COVERAGE: float = 0.25
WEIGHT_MODEL_FIT: float = 0.15
WEIGHT_AVAILABILITY: float = 0.10
WEIGHT_HISTORY: float = 0.10

MATCH_THRESHOLD: float = 0.4

# Known large-context models (128k+) — prefer these for later-sequence tasks
# that carry upstream outputs in their prompt.
_LARGE_CONTEXT_MODELS = frozenset({
    "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash",
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "claude-sonnet-4", "claude-opus-4", "claude-haiku-4",
    "claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus",
    "deepseek-chat", "deepseek-r1",
    "qwen-2.5",
})

# Role synonyms — maps common role keywords to broader categories
# so "research analyst" matches agents tagged/described as "researcher"
_ROLE_SYNONYMS: Dict[str, List[str]] = {
    "research": ["researcher", "research", "investigator", "analyst", "search"],
    "analyst": ["analyst", "analysis", "analytics", "intelligence", "assess"],
    "writer": ["writer", "writing", "content", "author", "copywriter", "blog"],
    "reviewer": ["reviewer", "review", "editor", "proofreader", "quality"],
    "coder": ["coder", "developer", "engineer", "programmer", "coding"],
    "designer": ["designer", "design", "ui", "ux", "creative"],
    "summarizer": ["summarizer", "summary", "synthesis", "consolidate", "report"],
    "search": ["search", "web", "browse", "lookup", "find"],
    "document": ["document", "report", "scribe", "draft", "write"],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    """Immutable result of agent matching for a single agent."""

    agent_id: int
    agent_name: str
    total_score: float
    tool_coverage: float
    skill_match: float
    model_fit: float
    availability: float
    history: float


# ---------------------------------------------------------------------------
# AgentMatcher
# ---------------------------------------------------------------------------


class AgentMatcher:
    """
    Deterministic agent scoring and selection for mission tasks.

    Stateless — all data comes from arguments or DB queries.
    """

    @staticmethod
    def match(
        db: Session,
        task: OrchestrationTask,
        agents: Sequence[Agent],
        task_spec: Optional[Dict[str, Any]] = None,
    ) -> Optional[MatchResult]:
        """
        Find the best roster agent for *task*.

        Args:
            db: SQLAlchemy session (for availability + tool queries).
            task: The OrchestrationTask to match against.
            agents: Candidate roster agents (pre-filtered to workspace).
            task_spec: Optional dict overrides with keys:
                - required_tools (list[str])
                - agent_role (str)
                - preferred_model (str)

        Returns:
            MatchResult for the highest-scoring agent, or None if no agent
            meets the threshold.
        """
        if not agents:
            logger.warning(
                "AgentMatcher.match: no candidate agents for task %s", task.id
            )
            return None

        spec = task_spec or {}
        required_tools: List[str] = spec.get(
            "required_tools", []
        )
        agent_role: Optional[str] = spec.get("agent_role") or task.agent_role
        preferred_model: Optional[str] = spec.get("preferred_model")

        # Determine if task has upstream context (later tasks need larger models)
        has_upstream = bool(
            isinstance(task.input_context, dict)
            and task.input_context.get("upstream_outputs")
        )

        # Pre-fetch tool assignments for all candidate agents in one query
        agent_ids = [a.id for a in agents]
        tool_map = _build_tool_map(db, agent_ids)

        # Pre-fetch busy agent IDs (agents with ASSIGNED or RUNNING tasks)
        busy_agent_ids = _get_busy_agent_ids(db, agent_ids)

        best: Optional[MatchResult] = None
        all_scores: List[MatchResult] = []

        for agent in agents:
            if agent.status != "active":
                continue

            scores = _score_agent(
                agent=agent,
                required_tools=required_tools,
                agent_role=agent_role,
                preferred_model=preferred_model,
                agent_tools=tool_map.get(agent.id, set()),
                is_busy=agent.id in busy_agent_ids,
                has_upstream=has_upstream,
            )
            all_scores.append(scores)

            if scores.total_score >= MATCH_THRESHOLD and (
                best is None or scores.total_score > best.total_score
            ):
                best = scores

        # Debug: log all candidates ranked by score
        if all_scores:
            ranked = sorted(all_scores, key=lambda s: s.total_score, reverse=True)
            top_5 = ranked[:5]
            candidates_str = ", ".join(
                f"{s.agent_name}(id={s.agent_id} skill={s.skill_match:.2f} "
                f"tool={s.tool_coverage:.2f} model={s.model_fit:.2f} "
                f"total={s.total_score:.3f})"
                for s in top_5
            )
            logger.info(
                "AgentMatcher candidates for task %s (role=%s): %s",
                task.id,
                agent_role,
                candidates_str,
            )

        if best is not None:
            logger.info(
                "AgentMatcher: matched agent %s (id=%d, score=%.3f) for task %s",
                best.agent_name,
                best.agent_id,
                best.total_score,
                task.id,
            )
        else:
            logger.warning(
                "AgentMatcher: no agent met threshold %.2f for task %s (role=%s, tools=%s)",
                MATCH_THRESHOLD,
                task.id,
                agent_role,
                required_tools,
            )

        return best


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tool_map(
    db: Session, agent_ids: Sequence[int]
) -> Dict[int, set]:
    """
    Return {agent_id: set(app_name_lower)} for all active tool assignments.
    """
    if not agent_ids:
        return {}

    rows = (
        db.query(
            AgentAppAssignment.agent_id,
            AgentAppAssignment.app_name,
        )
        .filter(
            and_(
                AgentAppAssignment.agent_id.in_(agent_ids),
                AgentAppAssignment.is_active == True,  # noqa: E712
            )
        )
        .all()
    )

    tool_map: Dict[int, set] = {}
    for agent_id, app_name in rows:
        tool_map.setdefault(agent_id, set()).add(app_name.lower() if app_name else "")
    return tool_map


def _get_busy_agent_ids(
    db: Session, agent_ids: Sequence[int]
) -> frozenset:
    """
    Return frozenset of agent IDs that currently have ASSIGNED or RUNNING tasks.
    """
    if not agent_ids:
        return frozenset()

    rows = (
        db.query(OrchestrationTask.assigned_agent_id)
        .filter(
            and_(
                OrchestrationTask.assigned_agent_id.in_(agent_ids),
                OrchestrationTask.state.in_([
                    TaskState.ASSIGNED.value,
                    TaskState.RUNNING.value,
                ]),
            )
        )
        .distinct()
        .all()
    )

    return frozenset(row[0] for row in rows)


def _score_agent(
    *,
    agent: Agent,
    required_tools: List[str],
    agent_role: Optional[str],
    preferred_model: Optional[str],
    agent_tools: set,
    is_busy: bool,
    has_upstream: bool = False,
) -> MatchResult:
    """
    Compute weighted score for a single agent.

    Returns an immutable MatchResult.
    """
    # --- tool_coverage (0.25) ---
    if required_tools:
        required_lower = {t.lower() for t in required_tools}
        matched = required_lower & agent_tools
        tool_score = len(matched) / len(required_lower)
    else:
        # No tool requirement — neutral score (not a freebie)
        tool_score = 0.5

    # --- skill_match (0.40) — primary differentiator ---
    if agent_role:
        role_lower = agent_role.lower()
        skill_score = _compute_skill_match(agent, role_lower)
    else:
        skill_score = 0.5

    # --- model_fit (0.15) ---
    model_score = _compute_model_fit(
        agent=agent,
        preferred_model=preferred_model,
        has_upstream=has_upstream,
    )

    # --- availability (0.10) ---
    availability_score = 0.5 if is_busy else 1.0

    # --- history (0.10) — placeholder until 82B ---
    history_score = 0.5

    total = (
        WEIGHT_SKILL_MATCH * skill_score
        + WEIGHT_TOOL_COVERAGE * tool_score
        + WEIGHT_MODEL_FIT * model_score
        + WEIGHT_AVAILABILITY * availability_score
        + WEIGHT_HISTORY * history_score
    )

    return MatchResult(
        agent_id=agent.id,
        agent_name=agent.name,
        total_score=round(total, 4),
        tool_coverage=round(tool_score, 4),
        skill_match=round(skill_score, 4),
        model_fit=round(model_score, 4),
        availability=round(availability_score, 4),
        history=round(history_score, 4),
    )


def _compute_skill_match(agent: Agent, role_lower: str) -> float:
    """
    Check if agent matches the requested role.

    Uses multi-token matching: "research analyst" checks both "research"
    and "analyst" against the agent's name, description, skills, and tags.
    Also expands via _ROLE_SYNONYMS for broader matching.

    Scoring hierarchy:
      1.0  — agent name or skill name matches the role exactly
      0.85 — multiple role tokens match (e.g., 2+ tokens found)
      0.75 — role appears as substring in name, description, or skill names
      0.6  — a single role token or synonym matches description/skills
      0.5  — role keyword found in agent tags
      0.0  — no match
    """
    agent_name_lower = (agent.name or "").lower()
    agent_desc_lower = (agent.description or "").lower()

    # Exact name match
    if role_lower == agent_name_lower:
        return 1.0

    # Check skill names
    skill_names: List[str] = []
    try:
        if agent.skills:
            skill_names = [(s.name or "").lower() for s in agent.skills]
    except Exception:
        pass  # Detached session — skills not loaded

    if role_lower in skill_names:
        return 1.0

    # Full substring match in name or description
    if role_lower in agent_name_lower or role_lower in agent_desc_lower:
        return 0.75

    # Full substring match in any skill name
    if any(role_lower in sn for sn in skill_names):
        return 0.75

    # --- Token-level matching ---
    # Split role into tokens: "research analyst" → ["research", "analyst"]
    role_tokens = [t for t in re.split(r'[\s_\-/]+', role_lower) if len(t) > 2]

    # Expand tokens with synonyms
    expanded_tokens = set(role_tokens)
    for token in role_tokens:
        for category, synonyms in _ROLE_SYNONYMS.items():
            if token in synonyms or token == category:
                expanded_tokens.update(synonyms)

    # Build agent text corpus for token matching
    corpus = f"{agent_name_lower} {agent_desc_lower} {' '.join(skill_names)}"

    # Count how many expanded tokens match
    matching_tokens = sum(1 for t in expanded_tokens if t in corpus)

    if matching_tokens >= 3:
        return 0.85
    if matching_tokens >= 2:
        return 0.75
    if matching_tokens >= 1:
        return 0.6

    # Tag match (weakest signal)
    tags = agent.tags or []
    if isinstance(tags, list):
        tag_lower = [str(t).lower() for t in tags]
        tag_corpus = " ".join(tag_lower)
        if any(t in tag_corpus for t in expanded_tokens):
            return 0.5

    return 0.0


def _compute_model_fit(
    *,
    agent: Agent,
    preferred_model: Optional[str],
    has_upstream: bool,
) -> float:
    """
    Score model suitability.

    When a preferred_model is specified, exact match = 1.0, else 0.3.
    When no preference but task has upstream outputs, prefer agents
    with large-context models (128k+).
    """
    agent_model = _get_agent_model(agent)
    agent_model_lower = agent_model.lower()

    if preferred_model:
        return 1.0 if preferred_model.lower() in agent_model_lower else 0.3

    if has_upstream:
        # Prefer large-context models for tasks carrying upstream outputs
        is_large = any(m in agent_model_lower for m in _LARGE_CONTEXT_MODELS)
        return 0.9 if is_large else 0.3

    return 0.5


def _get_agent_model(agent: Agent) -> str:
    """Extract the model ID string from agent's model_config."""
    config = agent.model_config
    if isinstance(config, dict):
        return config.get("model_id", "")
    return ""

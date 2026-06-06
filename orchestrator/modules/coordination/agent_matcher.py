"""
Agent Matcher — PRD-82A
========================

Deterministic scoring to select the best roster agent for a mission task.

Scoring weights (from PRD-102 Section 6.2, rebalanced for 82A):
  - skill_match:    0.40  — agent's skill/description matches task's agent_role
  - tool_coverage:  0.25  — fraction of task's required tools the agent has
  - model_fit:      0.15  — agent's model context + capability for the task
  - availability:   0.10  — agent has no running tasks in current missions
  - history:        0.10  — avg verification score from past tasks (82B US-003)

Threshold: 0.4 minimum score to be considered a match.
Returns the single best-scoring agent, or None.

Source: PRD-82A Section 12 (US-010), PRD-102 Section 6.2
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from config import Config
from core.models.composio_cache import AgentAppAssignment
from core.models.core import Agent
from core.models.orchestration import OrchestrationEvent, OrchestrationTask
from core.models.orchestration_enums import EventType, TaskState

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
    "admin": ["admin", "operations", "ops", "configure", "setup", "workspace"],
}

# Canonical capability vocabulary the mission planner assigns from. Tasks bind
# to one of these CAPABILITIES (not a specific agent name); the matcher then
# scores every active agent for the capability and dispatches the best fit.
CANONICAL_ROLES: frozenset = frozenset(_ROLE_SYNONYMS)


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

        # Pre-fetch history-based scores (PRD-82B US-003)
        history_map = _build_history_map(
            db,
            agent_ids,
            agent_role=agent_role,
            lookback_days=Config.COORDINATOR_HISTORY_LOOKBACK_DAYS,
            min_datapoints=Config.COORDINATOR_HISTORY_MIN_DATAPOINTS,
        )

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
                history_score=history_map.get(agent.id, 0.5),
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
                f"hist={s.history:.2f} total={s.total_score:.3f})"
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


def _build_history_map(
    db: Session,
    agent_ids: Sequence[int],
    agent_role: Optional[str] = None,
    lookback_days: int = 30,
    min_datapoints: int = 3,
) -> Dict[int, float]:
    """
    Batch-query verification scores for candidate agents.

    Returns {agent_id: avg_score} for agents with enough data points.
    Agents below min_datapoints get 0.5 (neutral).

    Scores are extracted from TASK_VERIFICATION_COMPLETED event payloads
    on verified tasks assigned to each agent.
    """
    if not agent_ids:
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Build base filter: verified tasks assigned to our candidate agents,
    # updated within the lookback window
    task_filters = [
        OrchestrationTask.assigned_agent_id.in_(agent_ids),
        OrchestrationTask.state == TaskState.VERIFIED.value,
        OrchestrationTask.updated_at >= cutoff,
    ]
    if agent_role:
        task_filters.append(OrchestrationTask.agent_role == agent_role)

    # Get task IDs and their assigned agents in one query
    task_rows = (
        db.query(
            OrchestrationTask.id,
            OrchestrationTask.assigned_agent_id,
        )
        .filter(and_(*task_filters))
        .all()
    )

    if not task_rows:
        return {}

    task_id_to_agent: Dict[str, int] = {
        str(row[0]): row[1] for row in task_rows
    }
    task_ids = [row[0] for row in task_rows]

    # Fetch verification events for those tasks
    events = (
        db.query(
            OrchestrationEvent.task_id,
            OrchestrationEvent.payload,
        )
        .filter(
            and_(
                OrchestrationEvent.task_id.in_(task_ids),
                OrchestrationEvent.event_type == EventType.TASK_VERIFICATION_COMPLETED.value,
            )
        )
        .all()
    )

    # Aggregate scores per agent
    agent_scores: Dict[int, List[float]] = {}
    for task_id, payload in events:
        if not payload or not isinstance(payload, dict):
            continue
        scores = payload.get("scores", {})
        if not scores:
            continue
        # Average across score dimensions (relevance, completeness, etc.)
        dimension_values = [v for v in scores.values() if isinstance(v, (int, float))]
        if not dimension_values:
            continue
        avg_score = sum(dimension_values) / len(dimension_values)
        agent_id = task_id_to_agent.get(str(task_id))
        if agent_id is not None:
            agent_scores.setdefault(agent_id, []).append(avg_score)

    # Build result map: only include agents with enough data
    result: Dict[int, float] = {}
    for agent_id, scores_list in agent_scores.items():
        if len(scores_list) >= min_datapoints:
            result[agent_id] = round(
                sum(scores_list) / len(scores_list), 4
            )

    return result


def _score_agent(
    *,
    agent: Agent,
    required_tools: List[str],
    agent_role: Optional[str],
    preferred_model: Optional[str],
    agent_tools: set,
    is_busy: bool,
    has_upstream: bool = False,
    history_score: float = 0.5,
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

    # --- history (0.10) — wired in 82B US-003 ---
    # history_score is passed in from pre-computed history_map

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

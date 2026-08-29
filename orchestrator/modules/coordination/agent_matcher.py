"""
Agent Matcher — PRD-82A, extended by PRD-164 S2 (Q21 semantic blend)
=====================================================================

Deterministic scoring to select the best roster agent for a mission task.

Legacy scoring weights (from PRD-102 Section 6.2, rebalanced for 82A) sum
to 1.0 and are preserved verbatim so behavior is unchanged when semantic
signals are unavailable:
  - skill_match:    0.40  — agent's skill/description matches task's agent_role
  - tool_coverage:  0.25  — fraction of task's required tools the agent has
  - model_fit:      0.15  — agent's model context + capability for the task
  - availability:   0.10  — agent has no running tasks in current missions
  - history:        0.10  — avg verification score from past tasks (82B US-003)

PRD-164 S2 (Q21) adds two ADDITIVE components, blended by renormalizing the
weighted mean over the components actually present:
  - semantic:       0.35  — cosine similarity between the task text and the
                            agent's capability card (PRD-64
                            ``agents.semantic_embedding``, JSONB + python
                            cosine; one embedding call per dispatch)
  - field_signal:   0.15  — live field signal (PRD-166): agents that recently
                            contributed resonant knowledge to the workspace
                            field rank higher for related tasks

Every ranked agent carries a human-readable ``reason`` string, persisted on
the task row and surfaced to the PRD-163 approval card. Explicit agent
overrides (PRD-163 S4 — an approval-edited ``agent_role`` that names a roster
agent exactly) ALWAYS win, regardless of score and threshold.

Threshold: 0.4 minimum score to be considered a match (overrides bypass it).

Source: PRD-82A Section 12 (US-010), PRD-102 Section 6.2, PRD-164 S2 (Q21)
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
from core.models.orchestration_enums import BUSY_TASK_STATES, EventType, TaskState
from modules.coordination.match_signals import (
    SemanticSignals,
    compute_semantic_signals_sync,
    compute_signals_for_tasks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights — the legacy five sum to 1.0; the Q21 components are
# additive and renormalized, so absent signals reproduce legacy scores
# exactly (missing embeddings never change dispatch behavior).
# ---------------------------------------------------------------------------

WEIGHT_SKILL_MATCH: float = 0.40
WEIGHT_TOOL_COVERAGE: float = 0.25
WEIGHT_MODEL_FIT: float = 0.15
WEIGHT_AVAILABILITY: float = 0.10
WEIGHT_HISTORY: float = 0.10

# PRD-164 S2 (Q21): capability-card embedding + live field signal.
WEIGHT_SEMANTIC: float = 0.35
WEIGHT_FIELD_SIGNAL: float = 0.15

MATCH_THRESHOLD: float = 0.4

# How many ranked agents the persisted match annotation keeps.
_ANNOTATION_RANKED_LIMIT: int = 3

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
# An agent_role that instead names a roster agent EXACTLY is treated as an
# explicit override (PRD-163 S4 approval edit) and always wins.
CANONICAL_ROLES: frozenset = frozenset(_ROLE_SYNONYMS)


# ---------------------------------------------------------------------------
# Result dataclasses
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
    # PRD-164 S2 — Q21 blend components (None = signal absent from the blend)
    semantic: Optional[float] = None
    field_signal: Optional[float] = None
    # PRD-164 S2 — human-readable reason + explicit-override flag (PRD-163 S4)
    reason: str = ""
    is_override: bool = False


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
        semantic: Optional[SemanticSignals] = None,
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
            semantic: Optional pre-computed Q21 signals (PRD-164 S2) — see
                :meth:`compute_semantic_signals_sync`. None = lexical-only,
                identical to pre-164 behavior.

        Returns:
            MatchResult for the highest-scoring agent, or None if no agent
            meets the threshold. An explicit agent override (PRD-163 S4)
            is returned regardless of score and threshold.
        """
        ranked = AgentMatcher.rank(
            db=db, task=task, agents=agents, task_spec=task_spec, semantic=semantic,
        )
        if not ranked:
            return None

        spec = task_spec or {}
        agent_role = spec.get("agent_role") or task.agent_role

        # Debug: log all candidates ranked by score
        top_5 = ranked[:5]
        candidates_str = ", ".join(
            f"{s.agent_name}(id={s.agent_id} skill={s.skill_match:.2f} "
            f"tool={s.tool_coverage:.2f} model={s.model_fit:.2f} "
            f"hist={s.history:.2f} total={s.total_score:.3f}"
            f"{' OVERRIDE' if s.is_override else ''})"
            for s in top_5
        )
        logger.info(
            "AgentMatcher candidates for task %s (role=%s): %s",
            task.id,
            agent_role,
            candidates_str,
        )

        best = ranked[0]
        if not best.is_override and best.total_score < MATCH_THRESHOLD:
            logger.warning(
                "AgentMatcher: no agent met threshold %.2f for task %s (role=%s, tools=%s)",
                MATCH_THRESHOLD,
                task.id,
                agent_role,
                spec.get("required_tools", []),
            )
            return None

        logger.info(
            "AgentMatcher: matched agent %s (id=%d, score=%.3f, override=%s) for task %s — %s",
            best.agent_name,
            best.agent_id,
            best.total_score,
            best.is_override,
            task.id,
            best.reason,
        )
        return best

    @staticmethod
    def rank(
        db: Session,
        task: OrchestrationTask,
        agents: Sequence[Agent],
        task_spec: Optional[Dict[str, Any]] = None,
        semantic: Optional[SemanticSignals] = None,
    ) -> List[MatchResult]:
        """PRD-164 S2: score every active candidate and return them ranked
        (override first, then total score desc, agent id asc), EACH with a
        human-readable reason string. Prefetches the DB-backed maps and
        delegates to the pure :meth:`_rank_with_context`.
        """
        if not agents:
            logger.warning(
                "AgentMatcher.rank: no candidate agents for task %s", task.id
            )
            return []

        spec = task_spec or {}
        required_tools: List[str] = spec.get("required_tools", [])
        agent_role: Optional[str] = spec.get("agent_role") or task.agent_role
        preferred_model: Optional[str] = spec.get("preferred_model")

        # Determine if task carries upstream context (later tasks need larger
        # models). PRD-164 S4 (Q22): dispatch context is the budgeted field
        # digest pinned by _prepare_task — present on re-matches after a first
        # dispatch, exactly when the raw stuffing used to be.
        has_upstream = bool(
            isinstance(task.input_context, dict)
            and task.input_context.get("field_digest")
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

        return AgentMatcher._rank_with_context(
            agents=agents,
            agent_role=agent_role,
            required_tools=required_tools,
            preferred_model=preferred_model,
            has_upstream=has_upstream,
            tool_map=tool_map,
            busy_agent_ids=busy_agent_ids,
            history_map=history_map,
            semantic=semantic,
        )

    @staticmethod
    def _rank_with_context(
        *,
        agents: Sequence[Agent],
        agent_role: Optional[str],
        required_tools: List[str],
        preferred_model: Optional[str],
        has_upstream: bool,
        tool_map: Dict[int, set],
        busy_agent_ids: frozenset,
        history_map: Dict[int, float],
        semantic: Optional[SemanticSignals] = None,
    ) -> List[MatchResult]:
        """Pure ranking core (no DB) — unit-tested by the golden matrix.

        Explicit overrides (PRD-163 S4): when ``agent_role`` names an active
        candidate agent exactly (name or slug, case-insensitive), that agent
        is ranked first regardless of its blended score.
        """
        override_agent_id = _find_override_agent_id(agent_role, agents)

        similarity_map = semantic.similarity_by_agent if semantic else {}
        field_map = semantic.field_by_agent if semantic else {}

        results: List[MatchResult] = []
        for agent in agents:
            if agent.status != "active":
                continue

            # Component present iff the map is non-empty; a card-less agent in
            # a carded roster scores neutral 0.5 (no card is not a bad card).
            semantic_score: Optional[float] = None
            if similarity_map:
                semantic_score = similarity_map.get(agent.id, 0.5)
            field_score: Optional[float] = None
            if field_map:
                field_score = field_map.get(agent.id, 0.0)

            results.append(_score_agent(
                agent=agent,
                required_tools=required_tools,
                agent_role=agent_role,
                preferred_model=preferred_model,
                agent_tools=tool_map.get(agent.id, set()),
                is_busy=agent.id in busy_agent_ids,
                has_upstream=has_upstream,
                history_score=history_map.get(agent.id, 0.5),
                semantic_score=semantic_score,
                field_score=field_score,
                is_override=(agent.id == override_agent_id),
            ))

        # Override first, then score desc; agent_id asc keeps ties stable.
        results.sort(key=lambda r: (not r.is_override, -r.total_score, r.agent_id))
        return results

    # ------------------------------------------------------------------
    # PRD-164 S2 — Q21 semantic signal computation (one embedding call per
    # dispatch), implemented in modules.coordination.match_signals and
    # exposed here so dispatcher/coordinator keep a single matcher seam.
    # ------------------------------------------------------------------

    @staticmethod
    async def compute_signals_for_tasks(
        tasks: Sequence[OrchestrationTask],
        agents: Sequence[Agent],
        workspace_id: Optional[UUID],
    ) -> Dict[Any, SemanticSignals]:
        """See :func:`modules.coordination.match_signals.compute_signals_for_tasks`."""
        return await compute_signals_for_tasks(tasks, agents, workspace_id)

    @staticmethod
    def compute_semantic_signals_sync(
        *,
        task: OrchestrationTask,
        agents: Sequence[Agent],
        workspace_id: Optional[UUID],
    ) -> Optional[SemanticSignals]:
        """See :func:`modules.coordination.match_signals.compute_semantic_signals_sync`."""
        return compute_semantic_signals_sync(
            task=task, agents=agents, workspace_id=workspace_id,
        )


# ---------------------------------------------------------------------------
# PRD-164 S2 — pure helpers (annotation, override, signals)
# ---------------------------------------------------------------------------


def build_match_annotation(ranked: Sequence[MatchResult]) -> Dict[str, Any]:
    """The persisted match annotation: stored on the task row
    (``input_context['agent_match']``) and mirrored into the ``run.plan``
    snapshot so the PRD-163 approval card can show WHY each agent was picked.
    """
    top = ranked[0]
    return {
        "agent_id": top.agent_id,
        "agent_name": top.agent_name,
        "score": top.total_score,
        "reason": top.reason,
        "is_override": top.is_override,
        "ranked": [
            {
                "agent_id": r.agent_id,
                "agent_name": r.agent_name,
                "score": r.total_score,
                "reason": r.reason,
            }
            for r in ranked[:_ANNOTATION_RANKED_LIMIT]
        ],
    }


def _find_override_agent_id(
    agent_role: Optional[str], agents: Sequence[Agent]
) -> Optional[int]:
    """PRD-163 S4 explicit override: an ``agent_role`` that names an ACTIVE
    candidate agent exactly (name or slug, case-insensitive). The planner only
    emits capability roles (CANONICAL_ROLES — see
    test_planner_capability_routing), so a name here is deliberate human
    intent from the approval-edit path and must always win.
    """
    role = (agent_role or "").strip().lower()
    if not role:
        return None

    matches = [
        a for a in agents
        if a.status == "active" and (
            (a.name or "").strip().lower() == role
            or (getattr(a, "slug", None) or "").strip().lower() == role
        )
    ]
    if not matches:
        return None
    # Deterministic when duplicated names exist: lowest id wins.
    return min(matches, key=lambda a: a.id).id


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
                OrchestrationTask.state.in_(BUSY_TASK_STATES),
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
    semantic_score: Optional[float] = None,
    field_score: Optional[float] = None,
    is_override: bool = False,
) -> MatchResult:
    """
    Compute weighted score for a single agent (PRD-164 S2: Q21 blend with
    renormalization over the components present, plus a reason string).

    Returns an immutable MatchResult.
    """
    # --- tool_coverage (0.25) ---
    matched_tool_count = 0
    if required_tools:
        required_lower = {t.lower() for t in required_tools}
        matched = required_lower & agent_tools
        matched_tool_count = len(matched)
        tool_score = matched_tool_count / len(required_lower)
    else:
        # No tool requirement — neutral score (not a freebie)
        tool_score = 0.5

    # --- skill_match (0.40) — primary lexical differentiator ---
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
    weight_sum = 1.0

    # --- PRD-164 S2 (Q21): additive semantic + field components ---
    if semantic_score is not None:
        total += WEIGHT_SEMANTIC * semantic_score
        weight_sum += WEIGHT_SEMANTIC
    if field_score is not None:
        total += WEIGHT_FIELD_SIGNAL * field_score
        weight_sum += WEIGHT_FIELD_SIGNAL
    total /= weight_sum

    reason = _compose_reason(
        agent_name=agent.name,
        agent_role=agent_role,
        required_tools=required_tools,
        matched_tool_count=matched_tool_count,
        skill_score=skill_score,
        semantic_score=semantic_score,
        field_score=field_score,
        history_score=history_score,
        availability_score=availability_score,
        is_override=is_override,
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
        semantic=round(semantic_score, 4) if semantic_score is not None else None,
        field_signal=round(field_score, 4) if field_score is not None else None,
        reason=reason,
        is_override=is_override,
    )


def _compose_reason(
    *,
    agent_name: str,
    agent_role: Optional[str],
    required_tools: List[str],
    matched_tool_count: int,
    skill_score: float,
    semantic_score: Optional[float],
    field_score: Optional[float],
    history_score: float,
    availability_score: float,
    is_override: bool,
) -> str:
    """Deterministic, human-readable reason for the approval card and the
    TASK_ASSIGNED audit trail."""
    if is_override:
        return (
            f"Explicitly assigned: the plan names agent '{agent_name}' for this "
            f"task (PRD-163 approval override) — selection bypasses scoring."
        )

    clauses: List[str] = []
    if agent_role:
        if skill_score >= 0.75:
            clauses.append(f"strong role match for '{agent_role}'")
        elif skill_score >= 0.5:
            clauses.append(f"partial role match for '{agent_role}'")
        else:
            clauses.append(f"no direct role match for '{agent_role}'")

    if semantic_score is not None:
        if semantic_score >= 0.75:
            clauses.append(
                f"capability profile closely matches the task "
                f"(similarity {semantic_score:.2f})"
            )
        elif semantic_score >= 0.45:
            clauses.append(
                f"capability profile relates to the task "
                f"(similarity {semantic_score:.2f})"
            )
        else:
            clauses.append(
                f"weak capability-profile similarity ({semantic_score:.2f})"
            )

    if required_tools:
        clauses.append(
            f"covers {matched_tool_count}/{len(required_tools)} required tools"
        )

    if history_score > 0.7:
        clauses.append(f"strong verified-task history ({history_score:.2f})")
    elif history_score < 0.3:
        clauses.append(f"weak verified-task history ({history_score:.2f})")

    if field_score is not None and field_score > 0:
        clauses.append("recently contributed relevant knowledge to the mission field")

    if availability_score < 1.0:
        clauses.append("currently busy with another task")

    if not clauses:
        clauses.append("neutral fit on all signals")

    text = "; ".join(clauses)
    return text[0].upper() + text[1:]


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
    When no preference but the task carries upstream dispatch context
    (the Q22 field digest), prefer agents with large-context models (128k+).
    """
    agent_model = _get_agent_model(agent)
    agent_model_lower = agent_model.lower()

    if preferred_model:
        return 1.0 if preferred_model.lower() in agent_model_lower else 0.3

    if has_upstream:
        # Prefer large-context models for tasks carrying upstream context
        is_large = any(m in agent_model_lower for m in _LARGE_CONTEXT_MODELS)
        return 0.9 if is_large else 0.3

    return 0.5


def _get_agent_model(agent: Agent) -> str:
    """Extract the model ID string from agent's model_config."""
    config = agent.model_config
    if isinstance(config, dict):
        return config.get("model_id", "")
    return ""

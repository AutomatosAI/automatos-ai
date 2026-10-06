"""PRD-248 S5 / tuning (6 Oct): the ticket_assign judgement, beside the matcher's ranking.

Shadow only. When ``AgentMatcher.rank`` has ranked the roster for a mission step, the
decision engine scores each candidate's fit for the step, reading the owner's own
words for it (the mission's goal) beside the role the planner wrote, and the row
records whether it agreed. Lazy, off by default, fail-open: the ranking is never
touched.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from core.models.orchestration import OrchestrationRun

logger = logging.getLogger(__name__)


def shadow_assignment(
    task: Any,
    agents: Sequence[Any],
    ranked: Sequence[Any],  # agent_matcher.MatchResult (its agent_name)
    agent_role: Optional[str],
    required_tools: Sequence[str],
    db: Any = None,
) -> None:
    """PRD-248 S5 (shadow only): the decision engine picks from the same roster
    beside this ranking and logs whether it agreed. It reads the mission's goal
    (the owner's own words) beside the step and the role the planner wrote.
    Lazy, off by default, fail-open — the returned ranking is never touched."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine, judgements

        engine = get_decision_engine()
        if not ranked or engine.dials().ticket_assign_mode == MODE_OFF:
            return
        candidates = [
            (
                getattr(a, "name", "") or "",
                getattr(a, "description", "") or getattr(a, "job_title", "") or "",
            )
            for a in agents
        ]
        engine.shadow(
            judgements.shadow_assignment(
                engine,
                workspace_id=getattr(agents[0], "workspace_id", None) if agents else None,
                task_id=getattr(task, "id", None),
                title=getattr(task, "title", "") or "",
                description=getattr(task, "description", "") or "",
                role=agent_role,
                required_tools=list(required_tools or []),
                candidates=candidates,
                platform_ranked=[r.agent_name for r in ranked],
                mission_brief=_mission_goal(db, task),
            ),
            purpose=judgements.PURPOSE_ASSIGN,
        )
    except Exception:  # noqa: BLE001 — never into a mission
        logger.debug("[decision] assignment shadow skipped", exc_info=True)


def _mission_goal(db: Any, task: Any) -> Optional[str]:
    """The goal of the mission a task belongs to (the owner's words), or None."""
    run_id = getattr(task, "run_id", None)
    if db is None or run_id is None:
        return None
    goal = db.query(OrchestrationRun.goal).filter(OrchestrationRun.id == run_id).scalar()
    return str(goal) if goal else None

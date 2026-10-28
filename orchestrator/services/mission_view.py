"""A Mission as the Missions API serves it (moved out of api/missions.py, which serves it as
``_run_to_response``).

Its run's columns, the shape of its plan (parallel groups, synthesis steps, a complexity
tier from how many steps run at once), F282's ``check_each_step`` under one name, and
(PRD-268 US-C17b, §6.11.1) where it stands, what happens next and what Try again does, in
the words every row uses (``services/proactive/display_state.py``).
"""
from __future__ import annotations

from typing import Any, Dict

from core.models.orchestration import OrchestrationRun
from modules.coordination.owner_checks import checks_each_step
from modules.proactive.locale import DEFAULT_LOCALE
from services.proactive.display_state import mission_status

__all__ = ["mission_dict"]

# How many steps run at once makes a Mission moderate, or complex.
MODERATE_AT = 2
COMPLEX_AT = 3
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_MAX_RETRIES = 3


def mission_dict(run: OrchestrationRun, locale: str = DEFAULT_LOCALE) -> Dict[str, Any]:
    """A run as a dict matching MissionResponse, with its ``display_state``, ``next_step``
    (in ``locale``) and ``retry``."""
    return {**_run_columns(run), **_plan_shape(run), **mission_status(run.state, locale)}


def _run_columns(run: OrchestrationRun) -> Dict[str, Any]:
    """The run's own columns, defaults filled in."""
    return {
        "id": str(run.id),
        "workspace_id": str(run.workspace_id),
        "goal": run.goal,
        "state": run.state,
        "state_type": run.state_type,
        "plan": run.plan,
        "config": run.config,
        # F282: one normalised name for the owner's "check each step with me",
        # whatever spelling it was written under (modules/coordination/owner_checks.py).
        "check_each_step": checks_each_step(run.config),
        "output_summary": run.output_summary,
        "token_budget_estimate": run.token_budget_estimate,
        "tokens_used": run.tokens_used or 0,
        "max_retries": run.max_retries or DEFAULT_MAX_RETRIES,
        "max_concurrent": run.max_concurrent or DEFAULT_MAX_CONCURRENT,
        "replan_count": run.replan_count or 0,
        "created_by": run.created_by,
        "stop_reason": getattr(run, "stop_reason", None),
        "stop_detail": getattr(run, "stop_detail", None),
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _plan_shape(run: OrchestrationRun) -> Dict[str, Any]:
    """The parallel groups and synthesis steps in the plan's JSONB, and the complexity
    tier its ``max_concurrent`` gives."""
    plan_tasks = (run.plan or {}).get("tasks", [])
    return {
        "complexity_tier": _complexity_tier(run.max_concurrent or DEFAULT_MAX_CONCURRENT),
        "parallel_groups": sorted({t.get("parallel_group") for t in plan_tasks if t.get("parallel_group")}),
        "has_synthesis_tasks": any(t.get("task_type") == "synthesis" for t in plan_tasks),
    }


def _complexity_tier(max_concurrent: int) -> str:
    if max_concurrent >= COMPLEX_AT:
        return "complex"
    if max_concurrent >= MODERATE_AT:
        return "moderate"
    return "simple"

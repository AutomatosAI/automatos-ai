"""
Blueprint Validator — check agents against governance blueprints and mission budgets.

Used by:
- Platform tools: platform_validate_agent, platform_check_budget
- API layer: agents.py (warnings on create/update)
- Coordinator: pre-dispatch validation
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def get_default_blueprint(db: Session, workspace_id: UUID) -> Optional[Any]:
    """Get the default blueprint for a workspace, or None."""
    from core.models.blueprints import AgentBlueprint

    return (
        db.query(AgentBlueprint)
        .filter(
            AgentBlueprint.workspace_id == workspace_id,
            AgentBlueprint.is_default == True,  # noqa: E712
        )
        .first()
    )


def get_blueprint_by_id(db: Session, workspace_id: UUID, blueprint_id: UUID) -> Optional[Any]:
    """Get a specific blueprint."""
    from core.models.blueprints import AgentBlueprint

    return (
        db.query(AgentBlueprint)
        .filter(
            AgentBlueprint.id == blueprint_id,
            AgentBlueprint.workspace_id == workspace_id,
        )
        .first()
    )


def validate_agent(
    db: Session,
    workspace_id: UUID,
    agent_id: int,
    blueprint_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """
    Validate an agent against a blueprint. Uses default blueprint if none specified.

    Returns: {
        "pass": bool,
        "warnings": [...],
        "failures": [...],
        "agent_name": str,
        "blueprint_name": str | None,
    }
    """
    from core.models import Agent

    agent = (
        db.query(Agent)
        .filter(Agent.id == agent_id, Agent.workspace_id == workspace_id)
        .first()
    )
    if not agent:
        return {"pass": False, "failures": ["Agent not found"], "warnings": [], "agent_name": "unknown", "blueprint_name": None}

    blueprint = (
        get_blueprint_by_id(db, workspace_id, blueprint_id)
        if blueprint_id
        else get_default_blueprint(db, workspace_id)
    )

    if not blueprint:
        return {"pass": True, "failures": [], "warnings": ["No blueprint configured — skipping validation"], "agent_name": agent.name, "blueprint_name": None}

    rules = blueprint.rules or {}
    warnings: List[str] = []
    failures: List[str] = []

    # Rule: min_tools
    min_tools = rules.get("min_tools")
    if min_tools is not None:
        # AgentAppAssignment lives in core.models.composio_cache (schema v2);
        # core.models.tool_assignments only defines WorkspaceToolConfig. The
        # old import raised ImportError on the FIRST min_tools evaluation --
        # no test exercised this rule until PRD-204 S8's strict-spawn test.
        from core.models.composio_cache import AgentAppAssignment

        tool_count = (
            db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent_id)
            .count()
        )
        if tool_count < min_tools:
            failures.append(f"Agent has {tool_count} tools, minimum is {min_tools}")

    # Rule: require_system_prompt
    if rules.get("require_system_prompt"):
        if not agent.system_prompt or len(agent.system_prompt.strip()) < 10:
            failures.append("Agent has no system prompt (or it's too short)")

    # Rule: required_tags
    required_tags = rules.get("required_tags", [])
    if required_tags:
        agent_tags = set(agent.tags or [])
        missing = [t for t in required_tags if t not in agent_tags]
        if missing:
            warnings.append(f"Missing recommended tags: {', '.join(missing)}")

    # Rule: allowed_models
    allowed_models = rules.get("allowed_models")
    if allowed_models and agent.model:
        if agent.model not in allowed_models:
            warnings.append(f"Agent model '{agent.model}' not in allowed list: {', '.join(allowed_models)}")

    passed = len(failures) == 0

    # enforce_mode: "strict" blocks dispatch, "advisory" (default) only warns
    enforce_mode = rules.get("enforce_mode", "advisory")

    return {
        "pass": passed,
        "failures": failures,
        "warnings": warnings,
        "agent_name": agent.name,
        "blueprint_name": blueprint.name,
        "enforce_mode": enforce_mode,
    }


def check_authority(
    db: Session,
    workspace_id: UUID,
    agent_id: int,
    blueprint_id: Optional[UUID] = None,
) -> tuple[bool, List[str]]:
    """
    Pre-dispatch authority check. Returns (allowed, violations).

    If the blueprint's enforce_mode is "strict", failures block dispatch.
    If "advisory" (default), always allowed (failures logged as warnings).
    """
    result = validate_agent(db, workspace_id, agent_id, blueprint_id)

    if result.get("enforce_mode") == "strict" and not result["pass"]:
        return False, result["failures"]

    return True, result.get("warnings", [])


def check_mission_budget(
    db: Session,
    workspace_id: UUID,
    run_id: UUID,
) -> Dict[str, Any]:
    """
    Check a mission's budget status.

    Returns: {
        "status": "ok" | "warning" | "exceeded",
        "budget_config": {...} | None,
        "budget_spent": {...},
        "remaining_cost": float | None,
        "remaining_tokens": int | None,
        "alert": str | None,
    }
    """
    from core.models.orchestration import OrchestrationRun

    run = (
        db.query(OrchestrationRun)
        .filter(
            OrchestrationRun.id == run_id,
            OrchestrationRun.workspace_id == workspace_id,
        )
        .first()
    )

    if not run:
        return {"status": "error", "alert": "Mission not found"}

    config = run.budget_config or {}
    spent = run.budget_spent or {}

    if not config:
        return {
            "status": "ok",
            "budget_config": None,
            "budget_spent": spent,
            "remaining_cost": None,
            "remaining_tokens": None,
            "alert": None,
        }

    max_cost = config.get("max_cost")
    max_tokens = config.get("max_tokens")
    alert_pct = config.get("alert_at_pct", 80)

    spent_cost = spent.get("cost", 0)
    spent_tokens = spent.get("tokens", 0)

    remaining_cost = (max_cost - spent_cost) if max_cost else None
    remaining_tokens = (max_tokens - spent_tokens) if max_tokens else None

    status = "ok"
    alert = None

    # Check cost
    if max_cost and max_cost > 0:
        pct_used = (spent_cost / max_cost) * 100
        if pct_used >= 100:
            status = "exceeded"
            alert = f"Budget exceeded: ${spent_cost:.4f} / ${max_cost:.4f}"
        elif pct_used >= alert_pct:
            status = "warning"
            alert = f"Budget {pct_used:.0f}% used: ${spent_cost:.4f} / ${max_cost:.4f}"

    # Check tokens
    if max_tokens and max_tokens > 0 and status != "exceeded":
        pct_used = (spent_tokens / max_tokens) * 100
        if pct_used >= 100:
            status = "exceeded"
            alert = f"Token budget exceeded: {spent_tokens:,} / {max_tokens:,}"
        elif pct_used >= alert_pct and status == "ok":
            status = "warning"
            alert = f"Token budget {pct_used:.0f}% used: {spent_tokens:,} / {max_tokens:,}"

    return {
        "status": status,
        "budget_config": config,
        "budget_spent": spent,
        "remaining_cost": round(remaining_cost, 4) if remaining_cost is not None else None,
        "remaining_tokens": remaining_tokens,
        "alert": alert,
    }

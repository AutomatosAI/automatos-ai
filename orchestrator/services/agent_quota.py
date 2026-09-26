"""F200 — one agent-limit check on every path that makes a workspace agent.

Night 6: the Install button refused a package ("over_quota": 8 agents on a basic
plan of 5), yet Auto had just made three agents one by one through
platform_install_marketplace_agent and the owner made a fourth by hand (#330,
POST /api/agents); neither was checked. Only the package install knew the plan.

Every create path now asks here first (platform_create_agent, the marketplace
clone behind package, single-agent and marketplace-page installs, POST
/api/agents and its bulk form) and creates nothing when the agents would not
fit: the owner is told the plan's limit and the count before it is crossed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text

DEFAULT_PLAN = "basic"
# The HTTP status a create past the plan's limit gets (the plan must change first).
AGENT_LIMIT_STATUS = 402
# ...and one that meets another create in the same workspace (retry shortly).
AGENT_LIMIT_BUSY_STATUS = 409
BUSY_MESSAGE = ("Another agent is being added to this workspace right now. Nothing was created; "
                "try again in a moment.")
_COUNT_LOCK_KEY = "agent-limit:{workspace_id}"


class AgentLimitReached(Exception):
    """A create that would take the workspace past its plan's agent limit."""

    def __init__(self, refusal: Dict[str, Any]):
        super().__init__(refusal["message"])
        self.refusal = refusal


def workspace_agent_count(db: Any, workspace_id: Any) -> int:
    """The workspace's own agents: what the plan's ``max_agents`` counts."""
    from core.models.core import Agent

    return (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id, Agent.owner_type == "workspace")
        .count()
    )


def plan_agent_limit(workspace: Any) -> Tuple[str, int]:
    """The workspace's plan and its ``max_agents`` (0 = unlimited)."""
    from services.plan_tiers import get_tier

    plan = (getattr(workspace, "plan", None) or DEFAULT_PLAN) if workspace is not None else DEFAULT_PLAN
    return plan, int((get_tier(plan) or {}).get("max_agents", 0) or 0)


def _count_is_ours(db: Any, workspace_id: Any) -> bool:
    """Take the workspace's agent-count lock until this transaction ends, without
    waiting (F200 review: two creates at the limit both passed the count). It is
    re-entrant, so a package's several clones in one transaction pass. It never
    blocks: several create paths keep their transaction open across awaits, and a
    blocking wait from sync SQLAlchemy would freeze the event loop (F105)."""
    held = db.execute(text("SELECT pg_try_advisory_xact_lock(hashtext(:key))"),
                      {"key": _COUNT_LOCK_KEY.format(workspace_id=workspace_id)}).scalar()
    return bool(held)


def agent_limit_refusal(db: Any, workspace_id: Any, adding: int = 1) -> Optional[Dict[str, Any]]:
    """None when ``adding`` more agents fit the workspace's plan; otherwise what to
    tell the owner, before anything is created."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    plan, max_agents = plan_agent_limit(workspace)
    if max_agents <= 0:
        return None
    if not _count_is_ours(db, workspace_id):
        return {"success": False, "busy": True, "http_status": AGENT_LIMIT_BUSY_STATUS, "message": BUSY_MESSAGE}
    current = workspace_agent_count(db, workspace_id)
    if current + adding <= max_agents:
        return None
    more = "another agent" if adding == 1 else f"{adding} more agents"
    return {
        "success": False,
        "over_quota": True,
        "message": (f"Your {plan} plan includes {max_agents} agents and this workspace has {current}, "
                    f"so {more} would go over it. Nothing was created: upgrade the plan, or remove "
                    "an agent first."),
        "plan": plan,
        "current_agents": current,
        "max_agents": max_agents,
        "http_status": AGENT_LIMIT_STATUS,
    }


def require_agent_room(db: Any, workspace_id: Any, adding: int = 1) -> None:
    """Raise AgentLimitReached (carrying the refusal) when the agents would not fit."""
    refusal = agent_limit_refusal(db, workspace_id, adding)
    if refusal:
        raise AgentLimitReached(refusal)

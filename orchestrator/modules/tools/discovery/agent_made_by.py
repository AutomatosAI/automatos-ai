"""PRD-256 FX-012 (A430): the agent tools say who made each agent, when, and how it runs.

Night 12: asked "which agents did you create for me last night", Auto could not tell from its
tools. platform_list_agents gave each agent's id, runtime, tags and created_at, never who made
it; platform_get_agent gave neither who made it nor its runtime. Both now carry ``created_by``
in words (a chat's platform_create_agent, the Agents page, the agent factory) beside
``created_at`` and ``tags``, and platform_get_agent its ``runtime`` ('cli' or 'api'). They are
read in one query over the agents the answer already holds, in the caller's workspace; a public
widget turn's answer (which names no id) is left as it is.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

# Agent.created_by as each path that makes an agent writes it, in words for the model.
MADE_BY = {
    "platform": "platform: made with platform_create_agent, by Auto or an agent",
    "api": "api: made on the Agents page",
    "agent_factory": "agent_factory: made by the agent factory",
}
NOT_RECORDED = "not recorded"


def made_by(raw: Any) -> str:
    """``Agent.created_by`` in words: who made the agent."""
    if not raw:
        return NOT_RECORDED
    return MADE_BY.get(str(raw), str(raw))


def _made(db: Any, workspace_id: Any, ids: List[int]) -> Dict[int, Tuple[str, str]]:
    """Each agent's (created_by in words, runtime), for ``ids`` in this workspace."""
    from core.cli_runtime import runtime_kind_of
    from core.models import Agent

    rows = (db.query(Agent.id, Agent.created_by, Agent.configuration)
            .filter(Agent.workspace_id == workspace_id, Agent.id.in_(ids)).all())
    return {row.id: (made_by(row.created_by),
                     runtime_kind_of(row.configuration if isinstance(row.configuration, dict) else {}))
            for row in rows}


def _read(db: Any, workspace_id: Any, ids: List[int]) -> Dict[int, Tuple[str, str]]:
    """``_made``, or nothing when it can't be read: the answer stands either way."""
    if not ids:
        return {}
    try:
        return _made(db, workspace_id, ids)
    except Exception:  # noqa: BLE001 — the courtesy fields never fail the listing
        logger.exception("[agents] could not read who made agents %s in workspace %s", ids, workspace_id)
        return {}


def _with_made(agent: Any, made: Dict[int, Tuple[str, str]]) -> Any:
    if not isinstance(agent, dict) or agent.get("id") not in made:
        return agent
    created_by, runtime = made[agent["id"]]
    return {**agent, "created_by": created_by, "runtime": agent.get("runtime") or runtime}


def says_who_made_it(handler: Handler) -> Handler:
    """Wrap platform_list_agents or platform_get_agent: each agent says who made it and how it runs."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        result = await handler(db, workspace_id, params)
        if not isinstance(result, dict) or result.get("success") is not True:
            return result
        if isinstance(result.get("agents"), list):
            agents = result["agents"]
            made = _read(db, workspace_id, [a["id"] for a in agents if isinstance(a, dict) and "id" in a])
            return {**result, "agents": [_with_made(agent, made) for agent in agents]}
        agent = result.get("agent")
        if not isinstance(agent, dict) or "id" not in agent:
            return result
        return {**result, "agent": _with_made(agent, _read(db, workspace_id, [agent["id"]]))}
    return wrapped


def lists_the_team(handler: Handler) -> Handler:
    """Wrap platform_list_agents: who made each agent (here), then whether it can run now
    (F244), the listing made to fit after both (F362)."""
    from services.agent_availability import says_who_can_run

    return says_who_can_run(says_who_made_it(handler))


__all__ = ["MADE_BY", "NOT_RECORDED", "lists_the_team", "made_by", "says_who_made_it"]

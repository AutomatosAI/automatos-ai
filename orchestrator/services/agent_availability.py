"""F244 (night 7): an agent that can't run says so before it is given a card.

All eight agents said "active" while every run failed for the AI credit (06:53-07:30Z).
#0177, given to the Mac helper, waited for a CLI host that was not there. What stops
an agent is read from the database, so every worker answers the same:
- an agent that calls the AI provider (an API agent) can't run while the workspace's
  AI credit is out. That means a run failed for credit after the last model call that
  went through;
- a CLI session agent can't run while no CLI host that runs its CLI is online, unless it
  ran inside the recent-run window (N15-6: a run shows a host serves it, whatever a lapsed
  heartbeat or its stored flag says; services/agent_last_run.py).

"active" keeps its meaning (the agent is switched on). Why it can't run *now* is
said beside it, and Auto can suggest one that can.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.agent_roster_fit import every_agent_fits

logger = logging.getLogger(__name__)

CREDIT_OUT = ("Can't run now: your AI credit ran out. Top up the AI provider account, or give the card to a "
              "CLI session agent.")
NO_HOST = "Can't run now: no CLI host is online to run it. Start the CLI host on the computer it is paired with."
NO_CLI_FOR_IT = "Can't run now: {why}"
# A run's credit failure, in the words it is stored in (core.llm.credit), old and new.
_CREDIT_FAILURES = ("Your AI credit ran out%", "The AI provider's account ran out of credit%")
_CREDIT_OUT_SQL = text(
    "SELECT EXISTS (SELECT 1 FROM board_tasks bt WHERE bt.workspace_id = CAST(:ws AS uuid) "
    "AND bt.status = 'failed' AND (bt.error_message LIKE :mine OR bt.error_message LIKE :before) "
    "AND bt.completed_at > COALESCE((SELECT max(u.created_at) FROM llm_usage u "
    "WHERE u.workspace_id = CAST(:ws AS uuid) AND u.total_tokens > 0), '-infinity'))"
)


def ai_credit_out(db: Session, workspace_id: Any) -> bool:
    """A run failed for credit after the workspace's last model call that went through."""
    mine, before = _CREDIT_FAILURES
    return bool(db.execute(_CREDIT_OUT_SQL, {"ws": str(workspace_id), "mine": mine, "before": before}).scalar())


def why_unavailable(db: Session, workspace_id: Any, agents: Iterable[Any]) -> Dict[int, str]:
    """Each agent that can't run now, with why. An agent that can is left out. The
    credit and the hosts are read once, and only if an agent needs them."""
    from core.cli_runtime import RUNTIME_CLI, runtime_kind_of

    found: Dict[str, Any] = {}
    out: Dict[int, str] = {}
    for agent in agents:
        config = _configuration(agent)
        if runtime_kind_of(config) == RUNTIME_CLI:
            reason = _no_host(db, workspace_id, config, found)
        else:
            reason = CREDIT_OUT if _once(found, "credit_out", lambda: ai_credit_out(db, workspace_id)) else None
        if reason:
            out[_agent_id(agent)] = reason
    return _unless_it_ran(db, workspace_id, out)


def _unless_it_ran(db: Session, workspace_id: Any, reasons: Dict[int, str]) -> Dict[int, str]:
    """``reasons`` without the no-host ones of agents that ran inside the window (N15-6): a run
    shows a host serves them, whatever a lapsed heartbeat says. The credit is live and stays."""
    from services.agent_last_run import last_runs

    host_bound = [agent_id for agent_id, why in reasons.items() if why != CREDIT_OUT]
    ran = last_runs(db, workspace_id, host_bound) if host_bound else {}
    return {agent_id: why for agent_id, why in reasons.items() if agent_id not in ran}


def _no_host(db: Session, workspace_id: Any, config: Mapping[str, Any], found: Dict[str, Any]) -> Any:
    from services.cli_ticket_lane import host_online, no_cli_host_reason_for

    if not _once(found, "host_online", lambda: host_online(db, workspace_id)):
        return NO_HOST
    cli = str(config.get("provider") or "claude")
    missing = _once(found, f"cli:{cli}", lambda: no_cli_host_reason_for(db, workspace_id, cli))
    return NO_CLI_FOR_IT.format(why=missing) if missing else None


def _once(found: Dict[str, Any], key: str, read: Callable[[], Any]) -> Any:
    """``read()`` the first time ``key`` is asked for in this listing; the answer after that."""
    if key not in found:
        found[key] = read()
    return found[key]


def _configuration(agent: Any) -> Mapping[str, Any]:
    config = agent.get("configuration") if isinstance(agent, dict) else getattr(agent, "configuration", None)
    return config if isinstance(config, dict) else {}


def _agent_id(agent: Any) -> Any:
    return agent.get("id") if isinstance(agent, dict) else getattr(agent, "id", None)


def says_who_can_run(handler: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Wrap platform_list_agents or platform_get_agent: each agent says whether it can run now,
    why not, and its last run (N15-6), so Auto suggests one that can, and every agent reaches the
    model however long the team (F362). Never on a public widget turn."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        result = await handler(db, workspace_id, params)
        if not isinstance(result, dict) or not result.get("success") or widget_turn():
            return result
        agents, agent = result.get("agents"), result.get("agent")
        if isinstance(agents, list) and agents:
            return every_agent_fits({**result, "agents": _can_run(db, workspace_id, agents)})
        if isinstance(agent, dict) and "id" in agent:
            return {**result, "agent": _can_run(db, workspace_id, [agent])[0]}
        return result
    return wrapped


def _can_run(db: Session, workspace_id: Any, agents: List[Any]) -> List[Any]:
    """Each listed agent with ``can_run``, ``why`` when it can't, and its last run (N15-6)."""
    from services.agent_last_run import read_last_runs, with_last_run

    listed = [a for a in agents if isinstance(a, dict)]
    reasons = _reasons(db, workspace_id, [
        {"id": a.get("id"), "configuration": {"runtime": a.get("runtime"), "provider": a.get("provider")}}
        for a in listed])
    ran = read_last_runs(db, workspace_id, [a.get("id") for a in listed])

    def one(agent: Any) -> Any:
        if not isinstance(agent, dict):
            return agent
        why = {"why": reasons[agent["id"]]} if agent.get("id") in reasons else {}
        return with_last_run({**agent, "can_run": agent.get("id") not in reasons, **why}, ran)
    return [one(a) for a in agents]


def with_unavailable(endpoint: Callable[..., Awaitable[List[Any]]]) -> Callable[..., Awaitable[List[Any]]]:
    """Wrap the Agents page's list (GET /api/agents/): an agent that can't run now
    carries ``unavailable``, the sentence the page shows beside it."""
    @functools.wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> List[Any]:
        responses = await endpoint(*args, **kwargs)
        ctx, db = kwargs.get("ctx"), kwargs.get("db")
        if not responses or ctx is None or db is None:
            return responses
        reasons = _reasons(db, ctx.workspace_id, responses)
        return [r.model_copy(update={"unavailable": reasons[r.id]}) if r.id in reasons else r for r in responses]
    return wrapped


def _reasons(db: Session, workspace_id: Any, agents: Iterable[Any]) -> Dict[int, str]:
    """``why_unavailable``, or nothing when it can't be read: the list stands either way."""
    try:
        return why_unavailable(db, workspace_id, agents)
    except Exception:
        logger.exception("[agents] could not tell which agents can run now in workspace %s", workspace_id)
        return {}


__all__ = ["CREDIT_OUT", "NO_HOST", "ai_credit_out", "says_who_can_run", "why_unavailable", "with_unavailable"]

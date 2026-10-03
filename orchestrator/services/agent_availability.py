"""F244 (night 7): an agent that can't run says so before it is given a card.

All eight agents said "active" while every run failed for the AI credit (06:53-07:30Z).
#0177, given to the Mac helper, waited for a CLI host that was not there. What stops
an agent is read from the database, so every worker answers the same:
- an agent that calls the AI provider (an API agent) can't run while the workspace's
  AI credit is out. That means a run failed for credit after the last model call that
  went through;
- a CLI session agent can't run while no CLI host that runs its CLI is online.

"active" keeps its meaning (the agent is switched on). Why it can't run *now* is
said beside it, and Auto can suggest one that can.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping

from sqlalchemy import text
from sqlalchemy.orm import Session

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
    return out


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
    """Wrap platform_list_agents: each agent says whether it can run now, and why not,
    so Auto suggests one that can. Never on a public widget turn."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        result = await handler(db, workspace_id, params)
        agents = result.get("agents") if isinstance(result, dict) and result.get("success") else None
        if not agents or widget_turn():
            return result
        reasons = _reasons(db, workspace_id, [
            {"id": a.get("id"), "configuration": {"runtime": a.get("runtime"), "provider": a.get("provider")}}
            for a in agents])
        listed = [{**a, "can_run": a.get("id") not in reasons, **({"why": reasons[a["id"]]} if a.get("id") in reasons
                                                                   else {})} for a in agents]
        return {**result, "agents": listed}
    return wrapped


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

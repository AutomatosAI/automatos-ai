"""F362 (night 10c): platform_list_agents names every agent, however long the team.

Night 10c, chat 3c8c7a3e 17:17:23: "we don't currently have an agent with the specific job
title 'Brand Designer'" while #347 Brand Designer was active. The workspace had 49 agents.
The listing gave each one about 900 characters (prompt flags, counts, its folder, the reason
it can't run), and the tool formatter cuts a platform result at 20,000 characters, so Auto
read the first 22 agents by id and never the newest, the designer among them.

So a listing that would be cut is made to fit before it reaches the model: every agent with
what routing needs (its id, name, job title, team, status, runtime and whether it can run
now), and, should even that be too long, one short line per agent. It says it was shortened and
where an agent's full setup is (platform_get_agent). A listing that fits is left as it is.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

ESSENTIALS = ("id", "name", "job_title", "team", "status", "runtime", "provider", "can_run", "why")
# The formatter's "Tool: <name>\nStatus: success\n\n" header, with room to spare.
HEADER_ALLOWANCE = 120
SHORTENED = ("All {count} agents are listed, each with what routing needs; the full listing was too long to "
             "show. platform_get_agent gives one agent's full setup.")
LINE_SEP = " | "   # ASCII: the formatter's json.dumps escapes anything else
CAN_RUN, CANNOT_RUN = "can run", "can't run now"


def _formatter_cut() -> int:
    """Where the tool formatter cuts a platform result for the model: ``format_for_llm``'s default
    ``max_chars``, which its outer wrapper sets."""
    from modules.tools.formatting.generated_document_summary import DEFAULT_MAX_CHARS

    return DEFAULT_MAX_CHARS


def _shown_chars(result: Dict[str, Any]) -> int:
    """The length of ``result`` as the formatter shows it to the model."""
    shown = {k: v for k, v in result.items() if k != "success"}
    return len(json.dumps(shown, default=str, indent=2))


def _essentials(agent: Any) -> Any:
    if not isinstance(agent, dict):
        return agent
    return {key: agent[key] for key in ESSENTIALS if agent.get(key) not in (None, "", [])}


def _line(agent: Any) -> Any:
    """One agent on one line: "Brand Designer (id 347) | Brand Designer | Socials | active | cli | can run".
    Why one can't run is left to platform_get_agent, so the lines stay short. Never "#347": a '#' names a
    ticket (PRD-252)."""
    if not isinstance(agent, dict):
        return agent
    head = f"{agent.get('name')} (id {agent.get('id')})"
    parts = [str(agent[key]) for key in ("job_title", "team", "status", "runtime") if agent.get(key)]
    can = CAN_RUN if agent.get("can_run", True) else CANNOT_RUN
    return LINE_SEP.join([head, *parts, can])


def every_agent_fits(result: Dict[str, Any]) -> Dict[str, Any]:
    """``result`` (a platform_list_agents answer) shortened, when the formatter would cut it, so that
    every agent still reaches the model; unchanged when it fits. A new dict."""
    agents: List[Any] = result.get("agents") if isinstance(result.get("agents"), list) else []
    budget = _formatter_cut() - HEADER_ALLOWANCE
    if not agents or _shown_chars(result) <= budget:
        return result
    note = SHORTENED.format(count=len(agents))
    trimmed = {**result, "note": note, "agents": [_essentials(a) for a in agents]}
    if _shown_chars(trimmed) <= budget:
        return trimmed
    return {**result, "note": note, "agents": [_line(a) for a in agents]}


__all__ = ["every_agent_fits"]

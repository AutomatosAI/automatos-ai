"""What the owner said about a mission they ask Auto for (night 8: F282, F287).

- F282: "check each step" was written eight ways, and sometimes not at all: #0250
  ("Check each step with me before it counts as done") got no setting, and Auto said
  "Yes, it will"; #0454 and #0458 (the sentence that worked on #0428 and #0446) got
  none, and ran start to finish. 10 of 29 missions where the owner asked checked with
  them. When the owner's own words ask for it, the mission checks each step, whatever
  the call carried.
- F287: the planner swapped the agents the owner named (#0323, #0324, #0352, #0356,
  #0394, #0400). Auto passed them as {"Analyst": "Analyst"}, [{"agent_name": …}] or
  not at all, and staffing that names no agent's work pins nobody. The owner's own
  sentence names the work: "the Analyst works out the margin…, and the Operations
  Manager drafts the email…". Each agent named that way is pinned to its work.

The owner's words are their latest messages in the chat the call came from
(``_origin_chat_id``, set by the platform executor, never by the model).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

# The owner asking for every step to wait for them, as night 8 said it.
ASKS_FOR_CHECKS = re.compile(
    r"\b(?:check\w*|see|show(?: me)?|approv\w*|review\w*|ok|okay)\b[^.!?\n]{0,40}\b(?:each|every|any)\s+step\b"
    r"|\b(?:each|every)\s+step\b[^.!?\n]{0,40}\b(?:wait\w*|stop\w*|paus\w*|check\w*|approv\w*|ok\b|okay)"
    r"|\b(?:stop|pause|wait)\w*\b[^.!?\n]{0,40}\b(?:after|before|on|at|for)\s+(?:each|every)\s+(?:step|one)\b"
    r"|\bdon'?t\s+(?:mark|move on|go on|continue|start)[^.!?\n]{0,60}\buntil\b"
    r"|\bstep[- ]by[- ]step\b", re.I)
LETS_IT_RUN = re.compile(r"\blet it run\b|\bno need to check\b|\bdon'?t (?:check|stop|wait)\b", re.I)
# A clause that gives a named agent its work: "works out", "writes", "drafts", "to work out".
_WORK_START = re.compile(r"^\W*(?:then\s+|also\s+|first\s+)?(?:to\s+[a-z]+|[a-z]+s)\b", re.I)
_NOT_WORK = frozenset({"is", "was", "has", "keeps", "uses", "says", "seems", "likes", "its", "this", "his"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n")
# What joins one agent's work to the next agent's name: "…the margin, and the Content Creator".
_JOINING_TAIL = re.compile(r"(?:[\s,;]*\b(?:and|then|while|with|the|also)\b)+[\s,;]*$", re.I)
SHORT_PREFIXES = ("shopify ",)
MAX_DUTY_CHARS = 500


def owners_words(db: Session, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """The owner's latest messages in the chat the call came from, newest first; []
    outside a chat."""
    from modules.tools.discovery.handlers_board_task_review import owner_words
    from modules.tools.discovery.handlers_watches import _origin_chat_id

    return [words for words in owner_words(db, workspace_id, _origin_chat_id(params or {})) if words]


def asks_for_checks(said: Sequence[str]) -> bool:
    """Whether the owner's latest words ask for every step to wait for their check."""
    latest = said[0] if said else ""
    return bool(ASKS_FOR_CHECKS.search(latest)) and not LETS_IT_RUN.search(latest)


def staffing_from_words(db: Session, workspace_id: Any, said: Sequence[str]) -> List[Dict[str, str]]:
    """[{agent, does}] for each agent the owner's words give work to, from the latest
    message that does; [] when none does."""
    roster = _roster(db, workspace_id)
    for text in said:
        staffing = _named_with_work(text or "", roster)
        if staffing:
            return staffing
    return []


def staffing_from_steps(steps: Any, roster_names: Sequence[str]) -> List[Dict[str, str]]:
    """[{agent, does}] from steps that name an agent and say what it does."""
    from modules.tools.discovery.mission_asks import STEP_AGENT_KEYS, STEP_WORK_KEYS

    found: List[Dict[str, str]] = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        agent = next((str(step[k]).strip() for k in STEP_AGENT_KEYS if step.get(k)), "")
        does = next((str(step[k]).strip() for k in STEP_WORK_KEYS if step.get(k)), "")
        if agent and does and agent.lower() in {name.lower() for name in roster_names}:
            found.append({"agent": agent, "does": does[:MAX_DUTY_CHARS]})
    return _merged(found)


def roster_names(db: Session, workspace_id: Any) -> List[str]:
    return [agent.name for agent in _roster(db, workspace_id)]


def _roster(db: Session, workspace_id: Any) -> List[Any]:
    """The workspace's active agents whose name only one of them has (so a pin by
    name is never a guess)."""
    from core.models.core import Agent

    agents = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.status == "active").all()
    names = [str(agent.name or "").strip().lower() for agent in agents]
    return [agent for agent, name in zip(agents, names) if name and names.count(name) == 1]


def _named_with_work(text: str, roster: List[Any]) -> List[Dict[str, str]]:
    matches = _mentions(text, roster)
    found: List[Dict[str, str]] = []
    for n, (_start, end, agent) in enumerate(matches):
        stop = matches[n + 1][0] if n + 1 < len(matches) else len(text)
        clause = _JOINING_TAIL.sub("", _SENTENCE_END.split(text[end:stop], maxsplit=1)[0]).strip(" ,;:")
        if _gives_work(clause):
            found.append({"agent": agent.name, "does": clause[:MAX_DUTY_CHARS]})
    return _merged(found)


def _mentions(text: str, roster: List[Any]) -> List[Tuple[int, int, Any]]:
    """Where each agent is named, longest name first, with names inside longer ones
    dropped ("Analyst" inside "Business Analyst")."""
    found: List[Tuple[int, int, Any]] = []
    for agent in roster:
        for variant in _variants(str(agent.name)):
            found.extend((m.start(), m.end(), agent)
                         for m in re.finditer(rf"\b{re.escape(variant)}\b", text, re.I))
    found.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    kept: List[Tuple[int, int, Any]] = []
    for match in found:
        if not any(match[0] >= k[0] and match[1] <= k[1] for k in kept):
            kept.append(match)
    return kept


def _variants(name: str) -> List[str]:
    """"Shopify Operations Manager" is also "Operations Manager" and "Ops Manager"."""
    names = [name]
    for prefix in SHORT_PREFIXES:
        if name.lower().startswith(prefix):
            names.append(name[len(prefix):])
    return names + [n.replace("Operations", "Ops") for n in names if "Operations" in n]


def _gives_work(clause: str) -> bool:
    match = _WORK_START.match(clause)
    return bool(match) and match.group(0).strip(" ,;:").split()[-1].lower() not in _NOT_WORK


def _merged(found: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """One entry per agent, its pieces of work joined."""
    by_agent: Dict[str, List[str]] = {}
    for entry in found:
        by_agent.setdefault(entry["agent"], []).append(entry["does"])
    return [{"agent": agent, "does": "; ".join(works)[:MAX_DUTY_CHARS]} for agent, works in by_agent.items()]


def staffing_names_the_work(staffing: Any) -> bool:
    """Staffing the coordinator can pin: a list of {agent, does}."""
    return isinstance(staffing, list) and bool(staffing) and all(
        isinstance(e, dict) and e.get("agent") and e.get("does") for e in staffing)


def chosen_staffing(db: Session, workspace_id: Any, params: Dict[str, Any],
                    said: Sequence[str]) -> Optional[List[Dict[str, str]]]:
    """The staffing to pin: the call's own when it names each agent's work, else
    the call's steps that do, else the owner's words; None when nothing names work."""
    if staffing_names_the_work(params.get("staffing")):
        return params["staffing"]
    from_steps = staffing_from_steps(params.get("steps"), roster_names(db, workspace_id))
    return from_steps or staffing_from_words(db, workspace_id, said) or None


__all__ = ["asks_for_checks", "chosen_staffing", "owners_words", "staffing_from_words",
           "staffing_names_the_work"]

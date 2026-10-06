"""PRD-248 — Auto's classifier as typed questions (pure).

The Tier-3 rubric in ``build_assessment_prompt`` becomes typed questions the
decision engine answers in one call: complexity, routing lane, two yes/no gates,
one yes/no per tool domain and, when a roster exists, one Score per agent for
the agent the user points to (PRD-248 tuning, 6 Oct: split from two big
Choices). The option descriptions ARE the rubric's definitions, shortened — a
System One model reads words, not intent, and wordier instructions measured
WORSE calibration in the early-access tests, so these stay terse.

Nothing here does I/O. ``AutoBrain`` owns the call and the mapping into a
``ComplexityAssessment``; this module owns the questions, the verdict rule and
the field-by-field comparison the shadow log records.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from core.llm.decisions.questions import (
    CHOICE_MAX_OPTIONS,
    Choice,
    DecisionResult,
    Noul,
    Question,
    Score,
)

PURPOSE = "classifier"
MESSAGE_MAX_CHARS = 4000
PREVIEW_CHARS = 160
NONE_AGENT = "none"
NO_DOMAIN = "none"
# PRD-248 tuning (6 Oct): the roster (9 to 31 agents) and the tool domains were one big
# Choice each; the vendor's figures are 40% right for a 12-way choice and 91% when it is
# split. Each agent now gets its own Score and each domain its own yes/no, keyed by
# these prefixes so they never collide with the other questions.
AGENT_KEY = "agent:"
DOMAIN_KEY = "domain:"
# How the user's message points to one agent, low to high. Worded as what is there,
# never as what is missing (the vendor lists negation and indirection as failure modes).
TARGET_LEVELS = (
    "The message is about other work or other people.",
    "The message's work suits this agent's role.",
    "The message names this agent's role.",
    "The message names this agent.",
)
# Named by role or by name: at least level 2, more than "suits its role" (1).
TARGET_PICK_FLOOR = 1.5
# The Tier-3 routing roster's lines (``roster_lines``).
ROSTER_DESCRIPTION_CHARS = 80
ROSTER_SESSION_LANE = " [Claude Code session]"
ROSTER_REST = "- Also on the team: "

COMPLEXITY_CRITERIA: Dict[str, str] = {
    "atom": (
        "Greetings, chitchat, opinions, simple factual questions, jokes, "
        "acknowledgements. No tools needed. The most common category."
    ),
    "molecule": (
        "Needs one to three tools or actions: send an email, search docs, list "
        "agents, check emails and calendar. A straightforward fetch-and-display, "
        "even with several tools, is molecule."
    ),
    "cell": (
        "Needs tools plus memory or earlier context: reply to that email we "
        "discussed, update the report from last week."
    ),
    "organ": (
        "Needs four or more agents with different specialisations coordinating "
        "on a multi-phase plan."
    ),
    "organism": (
        "An enterprise multi-step pipeline with verification and feedback loops. "
        "Very rare."
    ),
}

ACTION_CRITERIA: Dict[str, str] = {
    "respond": (
        "Auto answers this turn itself: greetings, chitchat, questions, and "
        "anything addressed to Auto about the platform (create or list agents, "
        "close tickets, manage the workspace)."
    ),
    "delegate": (
        "A specialist agent answers THIS conversation inline and it is over; the "
        "user wants an answer in the chat now."
    ),
    "assign": (
        "A named or single agent does work off-thread on the board to a "
        "deliverable, not an inline answer: the user names an agent or role, "
        "hands off work that outlives this turn, or wants no conversational answer."
    ),
    "mission": (
        "A multi-agent project: several specialisations coordinating in phases. "
        "Only for genuinely multi-agent work; one named agent doing one piece of "
        "work is assign."
    ),
}

DOMAIN_CRITERIA: Dict[str, str] = {
    "platform": (
        "Creating, listing or managing agents, skills, plugins, playbooks, tasks "
        "or workspace resources; anything about the platform itself."
    ),
    "email": "Reading, sending or replying to email.",
    "calendar": "Meetings, scheduling, availability.",
    "documents": "Searching, reading or writing documents and reports.",
    "code": "Repositories, code, GitHub, deployments.",
    "database": "Data queries, analytics, tables, numbers.",
    "web": "Looking something up on the internet.",
    NO_DOMAIN: "No tool domain: chitchat, opinions, general knowledge.",
}

NEEDS_MEMORY = Noul(
    "Does answering this require remembering earlier conversations or the "
    "user's stored preferences?"
)
NEEDS_MULTI_AGENT = Noul("Would this need several different agents working together?")


def agent_role(agent: Any) -> Optional[str]:
    """An agent's role as the roster says it. An Agent row has no ``role``: its role is its
    ``job_title`` (F362: the roster read ``role`` and so never said "Brand Designer")."""
    role = getattr(agent, "job_title", None) or getattr(agent, "role", None)
    return str(role) if role else None


def roster_entries(agents: Iterable[Any]) -> List[Dict[str, Any]]:
    """The active roster as the classifier sees it: each agent's name and role only. The
    full descriptions drew the model away from the complexity and action questions."""
    entries: List[Dict[str, Any]] = []
    for agent in agents:
        name = (getattr(agent, "name", None) or getattr(agent, "slug", None) or "").strip()
        if not name:
            continue
        entry: Dict[str, Any] = {"name": name}
        role = agent_role(agent)
        if role:
            entry["role"] = role
        entries.append(entry)
    return entries


def _labelled(agent: Any) -> str:
    name = getattr(agent, "name", None) or getattr(agent, "slug", None) or "agent"
    role = agent_role(agent)
    return f"{name} ({role})" if role else str(name)


def _described(agent: Any) -> str:
    """One roster line: name, role, its lane and the start of its description. PRD-234: a Claude
    Code session agent runs on the user's machine under their own login, so it says so."""
    desc = (getattr(agent, "description", None) or "").strip()
    cfg = getattr(agent, "configuration", None) or {}
    lane = ROSTER_SESSION_LANE if isinstance(cfg, dict) and cfg.get("runtime") == "cli" else ""
    return f"- {_labelled(agent)}{lane}" + (f": {desc[:ROSTER_DESCRIPTION_CHARS]}" if desc else "")


def _by_id(agent: Any) -> int:
    agent_id = getattr(agent, "id", None)
    return agent_id if isinstance(agent_id, int) else 0


def roster_lines(agents: Iterable[Any], described: int) -> List[str]:
    """The Tier-3 routing roster: the first ``described`` agents (oldest first) each on a line of
    its own, and every one after them named on one last line, so no agent is left out (F362: a
    capped roster never named #347, the workspace's 49th agent)."""
    ordered = sorted(agents, key=_by_id)
    lines = [_described(agent) for agent in ordered[:described]]
    rest = [_labelled(agent) for agent in ordered[described:]]
    return lines + ([ROSTER_REST + ", ".join(rest)] if rest else [])


def build_state(
    message: str, conversation_length: int, roster: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {
        "message": (message or "")[:MESSAGE_MAX_CHARS],
        "conversation_turn": int(conversation_length or 0),
        "agents": [dict(entry) for entry in roster],
    }


def agent_options(names: Iterable[str]) -> List[str]:
    """Distinct roster names, first spelling wins, capped so ``none`` fits."""
    seen: Dict[str, str] = {}
    for name in names:
        key = (name or "").strip().lower()
        if key and key != NONE_AGENT and key not in seen:
            seen[key] = name.strip()
    return list(seen.values())[: CHOICE_MAX_OPTIONS - 1]


RosterItem = Union[str, Mapping[str, Any]]


def _roster_roles(roster: Iterable[RosterItem]) -> Dict[str, str]:
    """Distinct names (as ``agent_options``) with each one's role ("" for none)."""
    items = [{"name": item} if isinstance(item, str) else dict(item) for item in roster]
    roles = {str(item.get("name") or "").strip(): str(item.get("role") or "") for item in items}
    return {name: roles.get(name, "") for name in agent_options(roles)}


def target_questions(roster: Iterable[RosterItem]) -> Dict[str, Question]:
    """One Score per agent: how the user's message points to it."""
    return {
        f"{AGENT_KEY}{name}": Score(
            f"How does the user's message point to the agent {name}?" + (f" Its role: {role}" if role else ""),
            list(TARGET_LEVELS),
        )
        for name, role in _roster_roles(roster).items()
    }


def domain_questions() -> Dict[str, Question]:
    """One yes/no per tool domain; answering may need more than one."""
    return {
        f"{DOMAIN_KEY}{domain}": Noul(f"Answering this message takes {domain} tools: {description}")
        for domain, description in DOMAIN_CRITERIA.items()
        if domain != NO_DOMAIN
    }


def build_questions(roster: Iterable[RosterItem]) -> Dict[str, Question]:
    """``roster``: the entries ``roster_entries`` makes (or bare names)."""
    questions: Dict[str, Question] = {
        "complexity": Choice(
            "How much machinery does answering this message take?", COMPLEXITY_CRITERIA
        ),
        "action": Choice("Where should the work happen?", ACTION_CRITERIA),
        "needs_memory": NEEDS_MEMORY,
        "needs_multi_agent": NEEDS_MULTI_AGENT,
    }
    questions.update(domain_questions())
    questions.update(target_questions(roster))
    return questions


def target_pick(result: DecisionResult) -> Optional[str]:
    """The agent the message points to most, by name or by role; ``none`` when it points
    to no agent that far; None when no agent was answered."""
    points = {
        key[len(AGENT_KEY):]: float(answer.score)
        for key, answer in result.answers.items()
        if key.startswith(AGENT_KEY) and answer.score is not None
    }
    if not points:
        return None
    best = max(points, key=lambda name: points[name])
    return best if points[best] >= TARGET_PICK_FLOOR else NONE_AGENT


def domain_hints(result: DecisionResult) -> Optional[List[str]]:
    """The domains answered yes, in the catalogue's order; None when none was answered."""
    answered = {
        key[len(DOMAIN_KEY):]: answer.yes
        for key, answer in result.answers.items()
        if key.startswith(DOMAIN_KEY) and answer.yes is not None
    }
    if not answered:
        return None
    return [domain for domain in DOMAIN_CRITERIA if answered.get(domain)]


def verdict_from_result(
    result: DecisionResult, *, min_confidence: float
) -> Optional[Dict[str, Any]]:
    """The engine's answers as a classifier verdict, or None when the two
    decisions that steer the turn (complexity, action) are missing, unknown, or
    below the confidence floor."""
    comp = result.get("complexity")
    act = result.get("action")
    if comp is None or act is None:
        return None
    if comp.choice not in COMPLEXITY_CRITERIA or act.choice not in ACTION_CRITERIA:
        return None
    confidence = min(comp.certainty, act.certainty)
    if confidence < float(min_confidence):
        return None

    hints = domain_hints(result) or []
    memory = result.get("needs_memory")
    multi = result.get("needs_multi_agent")
    pick = target_pick(result)
    target_name = pick if pick and pick != NONE_AGENT else None
    return {
        "complexity": comp.choice,
        "action": act.choice,
        "confidence": round(confidence, 4),
        "needs_memory": bool(memory.yes) if memory is not None and memory.yes is not None else False,
        "needs_multi_agent": bool(multi.yes) if multi is not None and multi.yes is not None else False,
        "tool_hints": hints,
        "target_agent_name": target_name,
    }


def compare(verdict: Mapping[str, Any], result: DecisionResult) -> Dict[str, Optional[bool]]:
    """Field-by-field agreement between a tier's verdict (``to_dict()`` shape)
    and the engine's answers. None where the engine gave no usable answer.
    ``tool_domain`` is indicative only: Tier 3 writes free-text hints
    ("github") where the engine picks from a fixed domain list ("code")."""
    out: Dict[str, Optional[bool]] = {}
    for key in ("complexity", "action"):
        answer = result.get(key)
        out[key] = (answer.choice == verdict.get(key)) if answer is not None and answer.choice else None
    for key in ("needs_memory", "needs_multi_agent"):
        answer = result.get(key)
        out[key] = (
            (answer.yes == bool(verdict.get(key)))
            if answer is not None and answer.yes is not None
            else None
        )
    engine_hints = domain_hints(result)
    if engine_hints is not None:
        hints = [str(h).lower() for h in (verdict.get("tool_hints") or [])]
        out["tool_domain"] = bool(set(engine_hints) & set(hints)) if hints else not engine_hints
    else:
        out["tool_domain"] = None
    pick = target_pick(result)
    if pick is not None:
        name = (verdict.get("target_agent_name") or "").strip().lower()
        out["target_agent"] = (pick.lower() == name) if name else (pick == NONE_AGENT)
    else:
        out["target_agent"] = None
    return out


def message_digest(message: str) -> str:
    return hashlib.sha256((message or "").lower().strip().encode()).hexdigest()[:16]

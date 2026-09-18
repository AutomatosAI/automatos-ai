"""PRD-248 — Auto's classifier as typed questions (pure).

The Tier-3 rubric in ``build_assessment_prompt`` becomes five or six typed
questions the decision engine answers in one call: complexity, routing lane,
tool domain, two yes/no gates and, when a roster exists, the named target
agent. The option descriptions ARE the rubric's definitions, shortened — a
System One model reads words, not intent, and wordier instructions measured
WORSE calibration in the early-access tests, so these stay terse.

Nothing here does I/O. ``AutoBrain`` owns the call and the mapping into a
``ComplexityAssessment``; this module owns the questions, the verdict rule and
the field-by-field comparison the shadow log records.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.llm.decisions.questions import (
    CHOICE_MAX_OPTIONS,
    Choice,
    DecisionResult,
    Noul,
    Question,
)

PURPOSE = "classifier"
MESSAGE_MAX_CHARS = 4000
PREVIEW_CHARS = 160
ROSTER_DESC_CHARS = 80
NONE_AGENT = "none"
NO_DOMAIN = "none"

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


def roster_entries(agents: Iterable[Any]) -> List[Dict[str, Any]]:
    """The active roster as the classifier sees it: name, role, a short
    description, and whether it is a Claude Code session agent."""
    entries: List[Dict[str, Any]] = []
    for agent in agents:
        name = (getattr(agent, "name", None) or getattr(agent, "slug", None) or "").strip()
        if not name:
            continue
        entry: Dict[str, Any] = {"name": name}
        role = getattr(agent, "role", None)
        if role:
            entry["role"] = str(role)
        desc = (getattr(agent, "description", None) or "").strip()
        if desc:
            entry["description"] = desc[:ROSTER_DESC_CHARS]
        cfg = getattr(agent, "configuration", None) or {}
        if isinstance(cfg, dict) and cfg.get("runtime") == "cli":
            entry["kind"] = "Claude Code session"
        entries.append(entry)
    return entries


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


def build_questions(agent_names: Iterable[str]) -> Dict[str, Question]:
    questions: Dict[str, Question] = {
        "complexity": Choice(
            "How much machinery does answering this message take?", COMPLEXITY_CRITERIA
        ),
        "action": Choice("Where should the work happen?", ACTION_CRITERIA),
        "tool_domain": Choice(
            "Which tool domain, if any, would answering need?", DOMAIN_CRITERIA
        ),
        "needs_memory": NEEDS_MEMORY,
        "needs_multi_agent": NEEDS_MULTI_AGENT,
    }
    names = agent_options(agent_names)
    if names:
        criteria: Dict[str, Optional[str]] = {name: None for name in names}
        criteria[NONE_AGENT] = "The user names no particular agent or role."
        questions["target_agent"] = Choice(
            "Which agent does the user name, or clearly mean, to hand this work to? "
            "Pick none unless the user's own words name that agent or its role.",
            criteria,
        )
    return questions


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

    domain = result.get("tool_domain")
    hints: List[str] = []
    if domain is not None and domain.choice in DOMAIN_CRITERIA and domain.choice != NO_DOMAIN:
        hints = [domain.choice]
    memory = result.get("needs_memory")
    multi = result.get("needs_multi_agent")
    target = result.get("target_agent")
    target_name = (
        target.choice if target is not None and target.choice and target.choice != NONE_AGENT else None
    )
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
    domain = result.get("tool_domain")
    if domain is not None and domain.choice:
        hints = [str(h).lower() for h in (verdict.get("tool_hints") or [])]
        out["tool_domain"] = (domain.choice in hints) if hints else (domain.choice == NO_DOMAIN)
    else:
        out["tool_domain"] = None
    target = result.get("target_agent")
    if target is not None and target.choice:
        name = (verdict.get("target_agent_name") or "").strip().lower()
        out["target_agent"] = (target.choice.lower() == name) if name else (target.choice == NONE_AGENT)
    else:
        out["target_agent"] = None
    return out


def message_digest(message: str) -> str:
    return hashlib.sha256((message or "").lower().strip().encode()).hexdigest()[:16]

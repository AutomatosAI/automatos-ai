"""PRD-256 FX-014 (M3, F387): the Tier-3 verdict is read field by field, and a bad field loses only itself.

Night 12: the classifier wrote the lane word in the complexity field (``"complexity": "assign"``);
``Complexity("assign")`` raised inside ``AutoBrain._llm_classify`` and the whole verdict (its action,
its target agent, its tool hints) was dropped for MOLECULE/RESPOND, while the log said "falling back
to ATOM". 33 hand-offs ("Get OPS to…", "Ask RESEARCHER…") never reached the ASSIGN lane.

Now the action and the target agent are read first; a complexity that is no level is the lane's own
(MOLECULE for respond and assign, ORGAN for a mission) and the rest of the verdict is kept; a lane
word sent only in the complexity field is read as the action. Only a reply with no JSON object in it
loses the lane, and its log says what it keeps: the tools.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from consumers.chatbot.auto import Action, Complexity, ComplexityAssessment

logger = logging.getLogger(__name__)

TIER3_CONFIDENCE = 0.85
LOST_LANE_CONFIDENCE = 0.50
DEFAULT_REASONING = "LLM classified"
LOST_LANE_REASONING = "LLM classification failed — defaulting to MOLECULE (tools available)"
LOST_THE_LANE = "[AutoBrain] Tier 3 verdict unreadable — kept tools, lost the lane: %s"
# The deprecated lane word (PRD-125) is read as the mission it became.
ACTION_ALIASES = {"workflow": Action.MISSION.value}
# The complexity a lane implies when the classifier's own is no level.
LANE_COMPLEXITY = {Action.MISSION: Complexity.ORGAN}
DEFAULT_COMPLEXITY = Complexity.MOLECULE
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

Match = Callable[[str], Tuple[Optional[int], Optional[str]]]


class VerdictUnreadable(ValueError):
    """The classifier's reply holds no JSON object to read."""


@dataclass(frozen=True)
class Verdict:
    """The Tier-3 verdict as read: every field the reply gave, each checked on its own."""

    action: Action
    target_agent: str
    complexity: Complexity
    tool_hints: List[str] = field(default_factory=list)
    needs_memory: bool = False
    needs_multi_agent: bool = False
    reasoning: str = DEFAULT_REASONING


def _word(value: Any) -> str:
    return str(value or "").strip().lower()


def read_action(data: Dict[str, Any]) -> Action:
    """The lane: ``action``, else a lane word the classifier put in ``complexity``; RESPOND when neither is one."""
    for said in (_word(data.get("action")), _word(data.get("complexity"))):
        said = ACTION_ALIASES.get(said, said)
        if said in Action._value2member_map_:
            return Action(said)
    if data.get("action"):
        logger.warning("[AutoBrain] Tier 3 action %r is no lane — RESPOND, the rest of the verdict kept",
                       data.get("action"))
    return Action.RESPOND


def read_complexity(data: Dict[str, Any], action: Action) -> Complexity:
    """The level the classifier gave, or the lane's own when it gave none of the five (night 12: "assign")."""
    said = _word(data.get("complexity"))
    if said in Complexity._value2member_map_:
        return Complexity(said)
    fallback = LANE_COMPLEXITY.get(action, DEFAULT_COMPLEXITY)
    logger.warning("[AutoBrain] Tier 3 complexity %r is no level — %s for the %s lane, the verdict kept",
                   data.get("complexity"), fallback.value, action.value)
    return fallback


def _hints(value: Any) -> List[str]:
    return [str(hint) for hint in value if str(hint).strip()] if isinstance(value, list) else []


def read_verdict(content: str) -> Verdict:
    """The verdict in the classifier's reply. Raises ``VerdictUnreadable`` only when no JSON object is in it."""
    found = _JSON_OBJECT.search(content or "")
    if not found:
        raise VerdictUnreadable(f"no JSON object in the reply: {(content or '')[:80]!r}")
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError as exc:
        raise VerdictUnreadable(f"the JSON does not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise VerdictUnreadable("the JSON is not an object")
    action = read_action(data)
    return Verdict(
        action=action,
        target_agent=str(data.get("target_agent") or "").strip(),
        complexity=read_complexity(data, action),
        tool_hints=_hints(data.get("tool_hints")),
        needs_memory=bool(data.get("needs_memory", False)),
        needs_multi_agent=bool(data.get("needs_multi_agent", False)),
        reasoning=str(data.get("reasoning") or DEFAULT_REASONING),
    )


def to_assessment(verdict: Verdict, match: Match) -> ComplexityAssessment:
    """The assessment Auto acts on. On the ASSIGN lane ``match`` resolves the named agent against the
    roster the classifier saw; a name that resolves to no one is kept for the ask (PRD-224 US-004)."""
    target_id: Optional[int] = None
    target_name: Optional[str] = None
    if verdict.action == Action.ASSIGN:
        target_id, target_name = match(verdict.target_agent)
        if target_name is None and verdict.target_agent:
            target_name = verdict.target_agent
    return ComplexityAssessment(
        complexity=verdict.complexity, action=verdict.action, reasoning=verdict.reasoning,
        confidence=TIER3_CONFIDENCE, target_agent_id=target_id, target_agent_name=target_name,
        needs_memory=verdict.needs_memory, tool_hints=list(verdict.tool_hints),
        needs_multi_agent=verdict.needs_multi_agent,
    )


def lost_the_lane() -> ComplexityAssessment:
    """No verdict to read: MOLECULE/RESPOND, so the tools stay (a wrong ATOM would strip them); the
    caller logs ``LOST_THE_LANE``. A wrong MOLECULE only adds tool schemas; the model still decides."""
    return ComplexityAssessment(
        complexity=Complexity.MOLECULE, action=Action.RESPOND, reasoning=LOST_LANE_REASONING,
        confidence=LOST_LANE_CONFIDENCE, needs_memory=False, tool_hints=[], needs_multi_agent=False,
    )


__all__ = ["DEFAULT_COMPLEXITY", "LANE_COMPLEXITY", "LOST_THE_LANE", "Verdict", "VerdictUnreadable",
           "lost_the_lane", "read_action", "read_complexity", "read_verdict", "to_assessment"]

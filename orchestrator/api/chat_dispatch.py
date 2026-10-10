"""Who answers a chat turn (PRD-256 US-010, Decision D2).

Auto answers every turn of the owner's chat. The one exception is an agent the owner
chose in the UI (``request.agentId``). AutoBrain's verdict picks the lane Auto works in,
never another agent:

- ASSIGN: Auto files the ticket for the named agent (or hands on the named card) and
  reports the number. A role with no roster match asks once.
- MISSION: Auto answers, and the turn carries the mission suggestion card.
- RESPOND (a platform hint included): Auto answers with its tools.

The DELEGATE lane, which sent the turn to the Universal Router and let a specialist
answer in its own persona, is gone: AutoBrain turns a DELEGATE verdict into Auto's answer
or a named agent's ticket before it reaches here (``consumers.chatbot.handoffs``).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from consumers.chatbot.auto import Action, AutoBrain, ComplexityAssessment, apply_assign_bias, is_deferred_phrasing
from consumers.chatbot.clash_card import clashed_card, said_before
from consumers.chatbot.handoffs import PLATFORM_HINT, needs_no_apps, with_card_directive

logger = logging.getLogger(__name__)

HEADER_CONFIDENCE_FORMAT = "{:.2f}"


@dataclass(frozen=True)
class TurnLane:
    """The agent that answers this turn and how its turn runs."""

    agent_id: int
    assessment: Optional[ComplexityAssessment] = None
    suggest_mission: bool = False
    session_agent: bool = False

    @property
    def skip_composio(self) -> bool:
        """Whether the turn may skip the workspace's connected apps (Composio)."""
        return needs_no_apps(self.assessment)


def chosen_agent_lane(db: Session, agent_id: int) -> TurnLane:
    """The owner chose this agent in the UI: it answers. A session agent (runtime: cli)
    talks in the Runtime Canvas terminal instead (PRD-239)."""
    from services.cli_ticket_lane import is_cli_agent

    logger.info("[chat] Direct mode: agent_id=%s", agent_id)
    return TurnLane(agent_id=agent_id, session_agent=is_cli_agent(db, agent_id))


def _assign_lane(auto_agent_id: int, assessment: ComplexityAssessment, message_text: str, before: str) -> TurnLane:
    """PRD-224 US-004: Auto files the board ticket for the named agent, then confirms in
    one line. Checked before the platform hint so a "platform" hint can't collapse it. A directive
    the lane already wrote (FX-014: several agents carry the name, so Auto asks which) is kept.
    P256-FIX-RVW-33: the id given after a card's clash hands on the card the owner said ``before``."""
    asks_which = assessment.context_directive
    deferred = apply_assign_bias(assessment, message_text)
    card = clashed_card(assessment, message_text, before)
    deferred = deferred or bool(card and is_deferred_phrasing(before))
    assessment = with_card_directive(assessment, message_text, deferred=deferred, card=card)
    if asks_which:
        assessment = dataclasses.replace(assessment, context_directive=asks_which)
    logger.info(
        "[Auto] ASSIGN lane — agent=%r resolved=%s deferred=%s: agent_id=%s",
        assessment.target_agent_name, assessment.target_agent_id is not None, deferred, auto_agent_id,
    )
    return TurnLane(agent_id=auto_agent_id, assessment=assessment)


def lane_for(auto_agent_id: int, assessment: ComplexityAssessment, message_text: str, before: str = "") -> TurnLane:
    """The lane for AutoBrain's verdict. Auto answers on every lane. ``before``: the owner's
    message before this one (an answer to "which one?" hands on the card that message named)."""
    if assessment.action == Action.ASSIGN:
        return _assign_lane(auto_agent_id, assessment, message_text, before)
    platform = PLATFORM_HINT in (assessment.tool_hints or [])
    if assessment.action == Action.MISSION and not platform:
        logger.info("[Auto] MISSION suggested (complexity=%s)", assessment.complexity.value)
        return TurnLane(agent_id=auto_agent_id, assessment=assessment, suggest_mission=True)
    if platform and assessment.action != Action.RESPOND:
        # Platform management is Auto's core job: it handles it with its platform tools.
        assessment = dataclasses.replace(assessment, action=Action.RESPOND)
    logger.info(
        "[Auto] Auto answers (complexity=%s, hints=%s): agent_id=%s",
        assessment.complexity.value, assessment.tool_hints, auto_agent_id,
    )
    return TurnLane(agent_id=auto_agent_id, assessment=assessment)


async def auto_lane(
    db: Session, workspace_id: Any, *, auto_agent_id: int, message_text: str, history_length: int,
    before: str = "",
) -> TurnLane:
    """No agent chosen: AutoBrain classifies the message and Auto answers on its lane."""
    assessment = await AutoBrain(db, str(workspace_id)).assess(message_text, history_length)
    logger.info(
        "[Auto] Complexity=%s action=%s tool_hints=%s reasoning=%s",
        assessment.complexity.value, assessment.action.value, assessment.tool_hints, assessment.reasoning,
    )
    return lane_for(auto_agent_id, assessment, message_text, before)


def response_headers(assessment: Optional[ComplexityAssessment]) -> Dict[str, str]:
    """The streaming response's headers, with AutoBrain's verdict when there is one."""
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "x-vercel-ai-data-stream": "v1",
    }
    if assessment is None:
        return headers
    headers.update({
        "x-auto-complexity": assessment.complexity.value,
        "x-auto-action": assessment.action.value,
        "x-auto-confidence": HEADER_CONFIDENCE_FORMAT.format(assessment.confidence),
        "x-auto-needs-memory": str(assessment.needs_memory).lower(),
    })
    if assessment.tool_hints:
        headers["x-auto-tool-hints"] = ",".join(assessment.tool_hints)
    return headers


__all__ = ["TurnLane", "auto_lane", "chosen_agent_lane", "lane_for", "response_headers", "said_before"]

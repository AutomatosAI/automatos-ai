"""Spoken turns skip the classifier's model call (voice latency C-1).

A turn sent with ``spokenMode`` on ``POST /api/chat`` and no ``agentId`` is classified by
AutoBrain before the response headers go out. Its last tier is a model call. While
``SPOKEN_MODE_SKIP_CLASSIFIER`` is true a spoken turn runs the tiers that cost no model call
and stops there: the hand-off table, the onboarding pin, the cache, the regex fast paths and
the lane's own rewrites (a named teammate's ticket, "answer it here") all still run. The
decision engine, as a shadow or as Tier 2.5, and the Tier 3 completion do not.

A turn no tier matched gets ``UNCLASSIFIED``: MOLECULE/RESPOND with one marker in its
reasoning. ``api.chat_dispatch.auto_lane`` reads the marker and gives the turn no assessment,
the shape Auto Live and ``agentId`` turns already run. The marker is read by containment,
because the lane's rewrites append to ``reasoning``. An unclassified verdict is never cached:
it would answer typed turns from Tier 1 for a day.

What this gives up on spoken turns, on purpose: Tier 3's light ATOM path for a general
question, its memory skip and tool hints, new mission cards, and a hand-off by role or by a
name the roster match does not catch. That is why the switch is off until measured.

While the switch is false ``auto_lane`` builds ``AutoBrain``, as before.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from config import config
from consumers.chatbot.auto import AutoBrain, ComplexityAssessment
from consumers.chatbot.verdict_parser import lost_the_lane

UNCLASSIFIED = "Spoken turn: no tier matched and no model was asked"


def skips_the_classifier(spoken: bool) -> bool:
    """Whether this turn runs only the tiers that cost no model call: a spoken turn while
    ``SPOKEN_MODE_SKIP_CLASSIFIER`` is true."""
    return bool(spoken) and bool(config.SPOKEN_MODE_SKIP_CLASSIFIER)


def is_unclassified(assessment: Any) -> bool:
    """Whether ``assessment`` is the verdict of a spoken turn no tier matched, as returned or
    after the lane rewrote it (the rewrites append to its reasoning)."""
    reasoning = getattr(assessment, "reasoning", None)
    return isinstance(reasoning, str) and UNCLASSIFIED in reasoning


class SpokenAutoBrain(AutoBrain):
    """AutoBrain for a turn that is heard: the tiers that cost no model call."""

    def _start_shadow(self, message: str, conversation_length: int) -> None:
        """No shadow call beside the tiers: it is a model call too."""
        return None

    async def _decision_classify(self, message: str, conversation_length: int) -> Optional[ComplexityAssessment]:
        """No Tier 2.5: the decision engine is not asked."""
        return None

    async def _llm_classify(self, message: str, conversation_length: int) -> ComplexityAssessment:
        """No Tier 3: the turn is unclassified (MOLECULE/RESPOND, the tools kept)."""
        return dataclasses.replace(lost_the_lane(), reasoning=UNCLASSIFIED)

    def _cache_store(self, msg_lower: str, assessment: ComplexityAssessment) -> None:
        """Every verdict but the unclassified one is cached, as before."""
        if not is_unclassified(assessment):
            super()._cache_store(msg_lower, assessment)


__all__ = ["SpokenAutoBrain", "UNCLASSIFIED", "is_unclassified", "skips_the_classifier"]

"""
Theme voice: can Auto Live start for this workspace now?
========================================================

``GET /api/voice/live-status`` answers ``available`` and, when Live can't start, the
``reason``: the first of the mint's gates (``modules/voice/mint.py``) that would refuse,
in the mint's order, read without refusing. The caller is not a gate here (the status read
is authenticated already); the gates are the plan (SaaS only, D34), the credits (PRD-265 D-6:
a call is paid by the minute from credits; ``live_billing.has_credits``), the platform switch,
the three Retell credentials, the workspace's own toggle and the month's minutes.

A reason is a stable code, never a sentence: the web app says it in the member's words
(``frontend/lib/voice/live/refusal.ts``), as the Automatos Go app keys on the mint's coded
refusal. The plan's code and the credits' code are the mint's own, so one code means one thing on
both routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.voice.mint import VOICE_LIVE_PLAN_REQUIRED
from services.credits.routing import CREDITS_EXHAUSTED_CODE

LIVE_REASON_PLAN = VOICE_LIVE_PLAN_REQUIRED
LIVE_REASON_NO_CREDITS = CREDITS_EXHAUSTED_CODE
LIVE_REASON_PLATFORM_OFF = "voice_live_platform_off"
LIVE_REASON_NOT_ARMED = "voice_live_not_armed"
LIVE_REASON_WORKSPACE_OFF = "voice_live_workspace_off"
LIVE_REASON_CAP_REACHED = "voice_live_cap_reached"


@dataclass(frozen=True)
class LiveGates:
    """Each of the mint's gates for one workspace, passed (True) or not."""

    plan_allows: bool
    credits_allow: bool
    platform_enabled: bool
    armed: bool
    workspace_enabled: bool
    cap_allows: bool


def live_unavailable_reason(gates: LiveGates) -> Optional[str]:
    """The code of the first gate that would refuse a mint, in the mint's order; None when
    every gate passes and Auto Live can start."""
    ordered = (
        (gates.plan_allows, LIVE_REASON_PLAN),
        (gates.credits_allow, LIVE_REASON_NO_CREDITS),
        (gates.platform_enabled, LIVE_REASON_PLATFORM_OFF),
        (gates.armed, LIVE_REASON_NOT_ARMED),
        (gates.workspace_enabled, LIVE_REASON_WORKSPACE_OFF),
        (gates.cap_allows, LIVE_REASON_CAP_REACHED),
    )
    return next((reason for passed, reason in ordered if not passed), None)

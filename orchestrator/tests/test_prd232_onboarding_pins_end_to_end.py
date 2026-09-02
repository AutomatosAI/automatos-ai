"""PRD-232 × PRD-222 — the onboarding spine's pins survive END-TO-END under the
new surface composition (both halves).

Agreed with the onboarding pod on 2026-08-29 before the 232 integration:
``_apply_onboarding_prior`` (#647) unions six onboarding actions into the
narrowed dispatcher enum; US-001 keeps the ``platform_execute`` dispatcher in
the surface through every ``SmartToolRouter.route()`` branch; US-014 changes
first-class attachment (promotion-as-prior) but deliberately keeps promoted
names in the enum. The interaction is invisible at merge time and only shows
up as onboarding losing its tools in prod — so it is a TEST, not a note:

  half 1 — onboarding ACTIVE: the six actions ride the dispatcher's enum through
           route() and reach the final surface, on a low-signal turn ("Yes please.")
  half 2 — onboarding COMPLETED / SKIPPED: the six are ABSENT from that enum —
           the pin stayed CONDITIONAL (invariant 1, the one most likely to be lost).

Reuses the two existing suites' fixtures verbatim so a drift in either mechanism
fails here first.
"""
from __future__ import annotations

import pytest

from modules.tools.discovery.actions_onboarding import ONBOARDING_PRIOR_ACTIONS
from modules.tools.tool_router import _apply_onboarding_prior
from tests.test_prd222_onboarding_tool_prior import _WS, _FakeSession, WS_ID, _NARROWED
from tests.test_prd232_us001_dispatcher_survives_route import (  # noqa: F401 (fixture import)
    Intent,
    SmartToolRouter,
    _FakeClassifier,
    _dispatcher_from,
    _dispatcher_tool,
    _intent_result,
    _realistic_surface,
    routing_env,
)

LOW_SIGNAL_TURN = "Yes please."


def _surface_with_enum(enum):
    """The realistic surface, but with the dispatcher carrying THIS enum —
    exactly what tool_router hands smart_tool_router on a real turn."""
    tools = [t for t in _realistic_surface() if t.get("function", {}).get("name") != "platform_execute"]
    return tools + [_dispatcher_tool(enum)]


def _enum_of(tool):
    return set(tool["function"]["parameters"]["properties"]["action"]["enum"])


async def _route_low_signal(routing_env, enum):
    routing_env["install_registry"]()

    async def rank(query, agent_id=None, top_k=15, **kw):
        # A low-signal turn ranks nothing useful — the exact shape of the live failure.
        return [("platform_list_agents", 0.4, ["platform_list_agents"])]

    routing_env["install_rank_chains"](rank)
    r = SmartToolRouter()
    r.classifier = _FakeClassifier(_intent_result(primary=Intent.DATA_QUERY))
    return await r.route(query=LOW_SIGNAL_TURN, available_tools=_surface_with_enum(enum), agent_id=7)


@pytest.mark.asyncio
async def test_active_onboarding_pins_reach_the_final_surface(routing_env):
    """Half 1: mid-onboarding, all six spine actions are in the dispatcher enum
    that survives route() — the model can actually call them on 'Yes please.'"""
    narrowing = _apply_onboarding_prior(_NARROWED, _FakeSession(_WS("teach")), WS_ID, False, False)
    enum = narrowing[0]
    assert set(ONBOARDING_PRIOR_ACTIONS) <= set(enum), "prior did not fold the spine into the enum"

    result = await _route_low_signal(routing_env, enum)
    dispatcher = _dispatcher_from(result)
    assert dispatcher is not None, "dispatcher stripped on a low-signal turn (C1 regression)"
    assert set(ONBOARDING_PRIOR_ACTIONS) <= _enum_of(dispatcher), (
        "spine actions fell out of the enum between tool_router and route()"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["completed", "skipped"])
async def test_finished_onboarding_pins_are_absent(routing_env, stage):
    """Half 2 (invariant 1 — CONDITIONAL, not permanent): once onboarding is
    completed/skipped the six actions are NOT forced into the enum; the
    dispatcher still survives, carrying only what ranking/narrowing produced."""
    narrowing = _apply_onboarding_prior(_NARROWED, _FakeSession(_WS(stage)), WS_ID, False, False)
    assert narrowing == _NARROWED, "terminal stage must leave narrowing untouched"

    result = await _route_low_signal(routing_env, narrowing[0])
    dispatcher = _dispatcher_from(result)
    assert dispatcher is not None
    assert not (set(ONBOARDING_PRIOR_ACTIONS) & _enum_of(dispatcher)), (
        "onboarding pins leaked into a finished workspace's enum — the pin went permanent"
    )

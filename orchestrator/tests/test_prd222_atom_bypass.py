"""PRD-222 — the onboarding spine must never ride the ATOM bypass.

The spine lives ONLY in the full ContextService path: the OnboardingSection
(mode CHATBOT) plus the platform tools that advance the stage
(``platform_update_onboarding``, the site scan, the package search). The ATOM
lane bypasses ContextService AND strips those tools — and every classifier
tier is onboarding-blind: the Tier-3 rubric reads a business intro ("We're
Harbourline Coffee Roasters, we sell…") as a greeting ("when in doubt, choose
atom"), and Tier 1 then caches that verdict per exact message text for 24h.
Live result (2026-08-29): a mid-onboarding workspace greeted its intro like
small talk and the flow never engaged.

The fix is Tier 0 in ``AutoBrain.assess``: while the workspace's onboarding
stage is non-terminal, the turn is pinned to MOLECULE / RESPOND with the
"platform" hint — before the cache, never cached, fail-soft on any load error.

These are PURE tests: ``AutoBrain`` over a fake session; cache and LLM tiers
are stubbed to explode if consulted.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.auto import Action, AutoBrain, Complexity
from services import onboarding_state

WS_ID = "11111111-1111-1111-1111-111111111111"

INTRO = (
    "We're Harbourline Coffee Roasters — a small-batch specialty coffee "
    "roaster in Bristol. We sell whole bean and ground coffee direct to "
    "consumers on Shopify, run a subscription club, and wholesale to about "
    "30 cafés around the South West."
)

ACTIVE_STAGES = [
    s for s in onboarding_state.STAGE_ORDER
    if s not in onboarding_state.TERMINAL_STAGES
]


# --------------------------------------------------------------------------- #
# Fakes — a Workspace row carrying an onboarding doc + a session that returns
# it from ``.query(Workspace).filter(...).first()`` (all the pin reads).
# --------------------------------------------------------------------------- #


class _WS:
    def __init__(self, stage):
        self.onboarding = (
            None if stage is None else {"stage": stage, "stages": {}, "segment": {}}
        )


class _FakeQuery:
    def __init__(self, ws):
        self._ws = ws

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._ws


class _FakeSession:
    def __init__(self, ws):
        self._ws = ws

    def query(self, _model):
        return _FakeQuery(self._ws)


class _BoomSession:
    def query(self, _model):
        raise RuntimeError("db down")


def _brain(session, monkeypatch=None, *, cache=None):
    brain = AutoBrain(session, WS_ID)
    brain._redis = None  # Tier 1 off unless a test stubs the lookup itself
    if monkeypatch is not None:
        monkeypatch.setattr(
            brain,
            "_cache_lookup",
            cache if cache is not None else _explode("cache consulted"),
        )
        monkeypatch.setattr(brain, "_cache_store", _explode("pin was cached"))
        monkeypatch.setattr(brain, "_llm_classify", _explode("LLM tier consulted"))
    return brain


def _explode(why):
    def _boom(*_a, **_k):
        raise AssertionError(why)

    return _boom


# --------------------------------------------------------------------------- #
# The pin — every non-terminal stage forces the full path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ACTIVE_STAGES)
async def test_active_onboarding_pins_full_path_before_cache_and_llm(
    monkeypatch, stage
):
    # Cache, cache-store and LLM stubs all raise — the pin must return before
    # ANY other tier runs and must never be cached (the stage moves).
    brain = _brain(_FakeSession(_WS(stage)), monkeypatch)
    result = await brain.assess(INTRO)
    assert result.complexity == Complexity.MOLECULE
    assert result.action == Action.RESPOND
    assert "platform" in result.tool_hints


@pytest.mark.asyncio
async def test_poisoned_cached_atom_cannot_strip_the_spine(monkeypatch):
    # The live failure: yesterday's run cached atom for this exact intro text.
    # The pin sits ABOVE Tier 1, so the poisoned entry is never even read.
    def _cached_atom(_msg):
        raise AssertionError("Tier 1 ran — the pin must precede the cache")

    brain = _brain(_FakeSession(_WS("questions")), monkeypatch, cache=_cached_atom)
    result = await brain.assess(INTRO)
    assert result.complexity == Complexity.MOLECULE


@pytest.mark.asyncio
async def test_doc_less_workspace_reads_active_and_pins(monkeypatch):
    # A NULL onboarding column reads as not_started (new-workspace posture —
    # veterans were backfilled to 'skipped' by prd222_veteran_skip_backfill).
    brain = _brain(_FakeSession(_WS(None)), monkeypatch)
    result = await brain.assess(INTRO)
    assert result.complexity == Complexity.MOLECULE


# --------------------------------------------------------------------------- #
# No-op set — terminal stages and failures classify exactly as before
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", sorted(onboarding_state.TERMINAL_STAGES))
async def test_terminal_stage_classifies_normally(stage):
    brain = _brain(_FakeSession(_WS(stage)))
    result = await brain.assess("hello")
    assert result.complexity == Complexity.ATOM  # Tier-2 greeting, untouched


@pytest.mark.asyncio
async def test_missing_workspace_classifies_normally():
    brain = _brain(_FakeSession(None))
    result = await brain.assess("hello")
    assert result.complexity == Complexity.ATOM


@pytest.mark.asyncio
async def test_workspace_load_error_fails_soft_to_normal_classification():
    brain = _brain(_BoomSession())
    result = await brain.assess("hello")
    assert result.complexity == Complexity.ATOM

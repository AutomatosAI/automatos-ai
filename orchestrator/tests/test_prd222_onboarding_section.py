"""PRD-222 W1S2 / US-009 — OnboardingSection v2 (the stage-aware spine).

Pure tests: the section is exercised against a MagicMock session whose
``query(Workspace).filter(...).first()`` yields a fake workspace at a chosen
stage — no DB. Covers the stage-based trigger (replacing the old agent_count
heuristic), per-stage content, the trust-string prohibition, the
platform_update_onboarding + direct-vs-mission instructions, honest-degrade of
the scan offer, and the rendered-token budget.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.context.sections.onboarding as onb_mod
from core.context_guard import count_tokens
from modules.context.sections.base import SectionContext
from modules.context.sections.onboarding import OnboardingSection

_SOURCE = Path(onb_mod.__file__).read_text()

_ACTIVE_STAGES = ("not_started", "questions", "teach", "proposal", "building", "boom", "powerup")
_TERMINAL_STAGES = ("completed", "skipped")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


class _FakeWorkspace:
    def __init__(self, onboarding):
        self.onboarding = onboarding


def _doc(stage, *, comfort=None, trial=None):
    seg = {"comfort": comfort} if comfort else {}
    doc = {"stage": stage, "stages": {}, "segment": seg}
    if trial is not None:
        doc["trial"] = trial
    return doc


def _ctx(*, workspace=None, messages=None, workspace_id="ws-1"):
    """SectionContext with a MagicMock session returning ``workspace``."""
    db = None
    if workspace is not None or workspace_id is not None:
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = workspace
    return SectionContext(
        agent=None,
        workspace_id=workspace_id,
        db_session=db,
        messages=messages or [],
    )


def _render(ctx):
    return asyncio.run(OnboardingSection().render(ctx))


def _render_stage(stage, *, comfort=None, trial=None, messages=None):
    ws = _FakeWorkspace(_doc(stage, comfort=comfort, trial=trial))
    return _render(_ctx(workspace=ws, messages=messages))


# --------------------------------------------------------------------------- #
# AC1 — trigger is stage-based; manual phrases still work; heuristic gone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", _TERMINAL_STAGES)
def test_terminal_stage_renders_empty(stage):
    assert _render_stage(stage) == ""


@pytest.mark.parametrize("stage", _ACTIVE_STAGES)
def test_active_stage_renders_nonempty(stage):
    out = _render_stage(stage)
    assert out.strip()
    assert "Onboarding" in out


def test_no_db_and_no_trigger_renders_empty():
    # No workspace loadable and no manual phrase → the section stays silent.
    ctx = SectionContext(agent=None, workspace_id=None, db_session=None, messages=[])
    assert _render(ctx) == ""


@pytest.mark.parametrize("phrase", sorted(onb_mod._TRIGGER_PHRASES))
def test_manual_trigger_phrase_reactivates_completed_workspace(phrase):
    # A completed workspace is silent — UNLESS the user re-triggers by phrase.
    msgs = [{"role": "user", "content": f"Hi Auto, {phrase} please"}]
    out = _render_stage("completed", messages=msgs)
    assert out.strip()
    assert "already" in out.lower()  # the re-trigger note
    assert "three questions" in out.lower()  # restarts from the questions


def test_completed_without_phrase_stays_empty_even_with_messages():
    msgs = [{"role": "user", "content": "what's the weather"}]
    assert _render_stage("completed", messages=msgs) == ""


def test_agent_count_heuristic_removed_from_source():
    # AC1: the agent_count == 0 heuristic is gone from this section.
    assert "_check_empty_workspace" not in _SOURCE
    assert "Agent.status" not in _SOURCE
    assert "import Agent" not in _SOURCE
    assert ".count()" not in _SOURCE


def test_trigger_is_stage_driven_via_state_service():
    # Trigger now consults onboarding_state (the server-side machine), not agents.
    assert "onboarding_state" in _SOURCE
    assert "TERMINAL_STAGES" in _SOURCE


# --------------------------------------------------------------------------- #
# AC2 — per-stage content; OpenRouter copy; approval; trust-string prohibition
# --------------------------------------------------------------------------- #

_STAGE_MARKERS = {
    "not_started": "first message",
    "questions": "three questions",
    "teach": "teach auto their business",
    "proposal": "approval gate",
    "building": "narrate every step",
    "boom": "payoff",
    "powerup": "connect a key",
}


@pytest.mark.parametrize("stage,marker", list(_STAGE_MARKERS.items()))
def test_each_stage_yields_its_own_variant(stage, marker):
    out = _render_stage(stage).lower()
    assert marker in out
    # Only ONE stage block is rendered — no other stage's unique marker leaks in.
    others = [m for s, m in _STAGE_MARKERS.items() if s not in (stage, "not_started", "questions")]
    for other in others:
        if other != marker:
            assert other not in out


def test_powerup_contains_openrouter_copy():
    out = _render_stage("powerup")
    low = out.lower()
    assert "openrouter" in low
    assert "400+" in out
    assert "pay-as-you-go" in low
    assert "openai" in low and "anthropic" in low  # other providers collapsed beneath


def test_proposal_contains_explicit_approval_instruction():
    out = _render_stage("proposal").lower()
    assert "approval gate" in out
    assert "before they say yes" in out or "explicit yes" in out
    assert "nothing is built" in out


def test_trust_strings_only_appear_in_prohibition_lines():
    # AC2: skip_verification / auto_approve appear NOWHERE except a prohibition.
    for token in ("skip_verification", "auto_approve"):
        lines = [ln for ln in _SOURCE.splitlines() if token in ln]
        assert lines, f"{token} should appear (in a prohibition line)"
        for ln in lines:
            assert "never" in ln.lower(), f"{token} outside a prohibition line: {ln!r}"


def test_rendered_trust_strings_only_in_prohibition():
    # The prohibition line IS rendered in every stage (it's a common rule); the
    # tokens must never appear in a line that tells Auto to SET them.
    for stage in _ACTIVE_STAGES:
        for ln in _render_stage(stage).splitlines():
            if "skip_verification" in ln or "auto_approve" in ln:
                assert "never" in ln.lower(), f"[{stage}] non-prohibition: {ln!r}"


# --------------------------------------------------------------------------- #
# AC3 — records advances via the tool; names the direct-vs-mission threshold
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", _ACTIVE_STAGES)
def test_every_active_stage_instructs_the_update_tool(stage):
    assert "platform_update_onboarding" in _render_stage(stage)


def test_names_direct_vs_mission_threshold():
    out = _render_stage("proposal").lower()
    assert "3 or fewer agents" in out
    assert "2 or fewer playbooks" in out
    assert "direct tool calls" in out
    assert "mission" in out
    assert "awaiting_approval" in out


def test_comfort_level_echoed_and_adaptation_instructed():
    out = _render_stage("questions", comfort="brand new")
    assert "brand new" in out  # the actual comfort surfaced in the header
    assert "comfort" in out.lower()  # the adapt-your-register rule


# --------------------------------------------------------------------------- #
# Honest-degrade — scan offer suppressed when Firecrawl is unconfigured
# --------------------------------------------------------------------------- #


def test_teach_offers_scan_when_firecrawl_configured():
    caps = {"firecrawl_configured": True, "llm_key_valid": False,
            "composio_configured": False, "redis_configured": False}
    with patch("services.capability_report.onboarding_capabilities", return_value=caps):
        out = _render_stage("teach")
    assert "platform_scan_business_site" in out
    assert "NOT available" not in out


def test_teach_suppresses_scan_when_firecrawl_unconfigured():
    caps = {"firecrawl_configured": False, "llm_key_valid": False,
            "composio_configured": False, "redis_configured": False}
    with patch("services.capability_report.onboarding_capabilities", return_value=caps):
        out = _render_stage("teach")
    assert "NOT available" in out
    assert "document upload" in out.lower()


# --------------------------------------------------------------------------- #
# Trial balance surfaced in the power-up copy
# --------------------------------------------------------------------------- #


def test_powerup_shows_concrete_trial_balance_when_known():
    trial = {"granted_usd": 5.0, "spent_usd": 1.6, "state": "active"}
    out = _render_stage("powerup", trial=trial)
    assert "$3.40 of $5.00" in out


def test_powerup_trial_line_falls_back_when_no_trial():
    out = _render_stage("powerup")
    assert "trial credit" in out.lower()


# --------------------------------------------------------------------------- #
# AC4 — largest rendered variant within the registered token budget
# --------------------------------------------------------------------------- #


def test_largest_variant_within_budget():
    sec = OnboardingSection()
    trial = {"granted_usd": 5.0, "spent_usd": 1.6, "state": "active"}
    worst = 0
    for stage in _ACTIVE_STAGES:
        out = _render_stage(stage, comfort="brand new", trial=trial)
        worst = max(worst, count_tokens(out))
    assert sec.max_tokens is not None
    assert worst <= sec.max_tokens, f"largest variant {worst} > cap {sec.max_tokens}"
    # Also assert we did not need to raise the cap above the original 800.
    assert sec.max_tokens == 800


# --------------------------------------------------------------------------- #
# PRD-230 US-002 — capability doctrine v2 ("Auto knows its own shop").
# The 7 reflexes render on EVERY active stage so Auto never improvises the CSV
# workaround that motivated this PRD.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", _ACTIVE_STAGES)
def test_capability_doctrine_renders_on_every_active_stage(stage):
    low = _render_stage(stage).lower()
    # (1) Composio + connect-card routing
    assert "composio" in low and "connect card" in low
    # (3) scan-on-URL reflex
    assert "platform_scan_business_site" in low
    # (4) marketplace-first staffing
    assert "marketplace-first" in low


def test_doctrine_point_1_composio_connect_card_routing():
    out = _render_stage("teach")
    assert "Composio" in out
    assert "connect card" in out.lower()
    # never apologise / route to the card instead of improvising
    assert "never apologise" in out.lower() or "route to the card" in out.lower()


def test_doctrine_point_2_shopify_two_step_sync_truth():
    out = _render_stage("proposal")
    low = out.lower()
    assert "two-step" in low
    assert "settings → widget sdk" in low  # the Site appears here
    assert "knowledge graph" in low        # sync unlocks it (canonical term)
    assert "sync" in low


def test_doctrine_point_3_scan_on_url_immediately():
    out = _render_stage("questions")
    assert "platform_scan_business_site" in out
    assert "url" in out.lower()
    assert "firecrawl" in out.lower()  # honest-degrade caveat named


def test_doctrine_point_4_marketplace_first_staffing():
    out = _render_stage("building").lower()
    assert "marketplace-first" in out
    assert "prebuilt" in out
    assert "before building custom" in out


def test_doctrine_point_5_honest_widget_no_csv_line():
    out = _render_stage("teach")
    assert "no CSVs" in out
    assert "we sync directly" in out
    assert "widgets and agents" in out


def test_doctrine_point_6_basic_plan_comms_early():
    out = _render_stage("questions").lower()
    assert "you're on basic while we set up" in out
    assert "pick your plan together" in out


def test_doctrine_point_7_exact_stage_vocabulary_listed():
    # Auto must never invent a stage — the section names the EXACT vocabulary.
    from services import onboarding_state

    out = _render_stage("proposal")
    for stage in onboarding_state.ALL_STAGES:  # the real enum, no drift
        assert f"`{stage}`" in out, f"stage {stage!r} missing from the doctrine list"
    assert "never invent" in out.lower()


def test_doctrine_present_in_source_for_grep_gate():
    # The acceptance gate greps the section file directly.
    for needle in (
        "connect card", "Composio", "Widget SDK", "Knowledge Graph",
        "platform_scan_business_site", "marketplace-first", "no CSVs",
        "you're on Basic while we set up", "not_started", "skipped",
    ):
        assert needle in _SOURCE, f"doctrine needle {needle!r} missing from section source"

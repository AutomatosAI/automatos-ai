"""F362 (night 10c): a brand ask goes to the Brand designer, on a ticket, the first time.

Chat 3c8c7a3e: "Make the orange an accent only" took four messages: Auto called
platform_propose_brand_kit itself three times and was refused for having no ticket, offered to
"ask a human", picked the WRITER, then said there was no Brand Designer (#347 was active).
"More space between sections" reached platform_update_system_setting, and "Warmer, please"
became a three-task mission with a RESEARCHER. PRD-255 US-014's note never fired for those
words, and a note is only a prompt: the turn's lane was the classifier's to choose.

Now the words are read (a colour given a role, space between sections, a style word that opens
a conversation), AutoBrain pins the turn to the ASSIGN lane with the designer resolved before
any tier, and a proposal made with no ticket tells Auto to file it for the designer by name.
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot import brand_to_the_designer as brand
from consumers.chatbot.auto import Action, AutoBrain, Complexity, apply_assign_bias
from consumers.chatbot.named_template_note import read_note

WS = UUID("6c1f0e2a-3b4d-4e5f-8a9b-0c1d2e3f4a5b")
DESIGNER = NS(id=347, name="Brand Designer", status="active")
NIGHT_10C = ["Make the orange an accent only", "More space between sections",
             "Not you — my brand. The documents feel a bit cold. Warmer, please."]


@pytest.fixture
def designer(monkeypatch):
    from core.seeds import seed_brand_designer

    found = {"agent": DESIGNER}
    monkeypatch.setattr(seed_brand_designer, "find_brand_designer", lambda db, workspace_id: found["agent"])
    return found


@pytest.mark.parametrize("said", NIGHT_10C)
def test_the_nights_brand_asks_are_read_as_brand_work(said):
    assert brand.asks_for_brand_work(said) is True


def test_a_style_word_on_its_own_is_brand_work_only_when_it_opens_the_conversation():
    assert brand.asks_for_brand_work("Warmer, please", opening=True) is True
    assert brand.asks_for_brand_work("A bit more space, please.", opening=True) is True
    assert brand.asks_for_brand_work("Warmer, please") is False          # after a draft: its tone, perhaps
    assert brand.asks_for_brand_work("It's warmer today.", opening=True) is False


class _ChatDb:
    def begin_nested(self):
        return contextlib.nullcontext()


def test_an_opening_style_word_gets_the_designer_note_and_a_later_one_does_not(designer, monkeypatch):
    from modules.documents import template_service

    class _Templates:
        def __init__(self, db):
            pass

        def list_templates(self, workspace_id):
            return []

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    opening = read_note(_ChatDb(), WS, ["Warmer, please"])
    later = read_note(_ChatDb(), WS, ["Warmer, please", "Draft a thank-you email to Rosa."])

    assert opening is not None and 'assigned_agent_name "Brand Designer"' in opening
    assert later is None


def test_the_note_says_a_style_ask_is_one_ticket_never_a_mission_or_a_setting(designer):
    note = brand.designer_note(object(), WS, "More space between sections", seed=lambda ws: None)

    assert "never a mission (platform_create_mission)" in note
    assert "never a platform setting (platform_update_system_setting)" in note


def _brain(monkeypatch, onboarding=False):
    brain = AutoBrain(object(), str(WS))
    brain._redis = None
    monkeypatch.setattr(brain, "_onboarding_active", lambda: onboarding)

    def no_tier(*_a, **_k):
        raise AssertionError("a brand ask is decided before the cache and the tiers")

    for tier in ("_cache_lookup", "_run_fast_heuristics", "_decision_classify", "_llm_classify"):
        monkeypatch.setattr(brain, tier, no_tier)
    return brain


@pytest.mark.parametrize("said", NIGHT_10C)
def test_autobrain_pins_a_brand_ask_to_the_designers_ticket(designer, monkeypatch, said):
    verdict = asyncio.run(_brain(monkeypatch).assess(said, 3))

    assert verdict.action == Action.ASSIGN and verdict.complexity == Complexity.MOLECULE
    assert (verdict.target_agent_id, verdict.target_agent_name) == (347, "Brand Designer")
    apply_assign_bias(verdict, said)
    assert "assigned_agent_name=\"Brand Designer\"" in verdict.context_directive
    assert "platform_create_task" in verdict.tool_hints


def test_an_opening_warmer_please_is_pinned_and_a_later_one_is_left_to_the_tiers(designer, monkeypatch):
    opening = asyncio.run(_brain(monkeypatch).assess("Warmer, please", 1))
    assert opening.target_agent_name == "Brand Designer"

    with pytest.raises(AssertionError, match="decided before the cache"):
        asyncio.run(_brain(monkeypatch).assess("Warmer, please", 4))


@pytest.mark.parametrize("found", [None, NS(id=347, name="Brand Designer", status="paused")])
def test_without_an_active_designer_the_tiers_decide(designer, monkeypatch, found):
    designer["agent"] = found

    with pytest.raises(AssertionError, match="decided before the cache"):
        asyncio.run(_brain(monkeypatch).assess(NIGHT_10C[0], 3))


def test_mid_onboarding_the_onboarding_pin_still_comes_first(designer, monkeypatch):
    verdict = asyncio.run(_brain(monkeypatch, onboarding=True).assess(NIGHT_10C[0], 3))

    assert verdict.action == Action.RESPOND and "Onboarding active" in verdict.reasoning


def test_a_proposal_made_with_no_ticket_tells_auto_to_file_it_for_the_designer(designer):
    from modules.tools.discovery import handlers_brand_proposals as bp

    answer = asyncio.run(bp.propose_brand_kit(object(), WS, {"brand_kit": {"accent_use": "sparing"}}))

    assert answer["success"] is False and "has none" in answer["error"]
    assert 'platform_create_task (assigned_agent_name "Brand Designer"' in answer["error"]
    assert "ask a human" not in answer["error"]

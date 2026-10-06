"""PRD-255 US-014: Auto routes a brand ask to the Brand designer (thesis: Auto delegates).

"Help me design my brand", "improve the kit", "make our templates", "less orange": the
turn gets a note to file a ticket for the workspace's Brand designer, by name, and start
it; Auto designs nothing itself and never changes the kit on such an ask. A question
about the kit, a "do it yourself", and customer paperwork (F337(c)) get no such note. A
workspace with no designer yet gets one seeded; one the owner removed is not brought back.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot import brand_to_the_designer as brand
from consumers.chatbot.named_template_note import fills_the_named_template, read_note
from consumers.chatbot.paperwork_to_the_team import team_note
from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT

WS = UUID("4d9b3a19-6c0e-4f2a-9b1c-0d1e2f3a4b50")
DESIGNER = NS(id=12, name="Brand Designer")
PASSAGES = "passages from brand-notes.md"
PRICE_LIST = ("Can you make me a wholesale price list for cafés, as a spreadsheet? Every coffee we sell, price per "
              "kg, minimum order, carriage and payment terms.")


@pytest.mark.parametrize("said", [
    "Help me design my brand.",
    "Can you improve the kit? It looks dated.",
    "Make our templates match the new logo.",
    "Redesign our brand kit from the logo.",
    "Less orange, please.",
    "Could you make the brand warmer?",
    "We need more space on the pages.",
    "Make our invoice template look like us.",
    "Sort out our colours on everything.",
    "More orange in the header, less in the body.",
])
def test_a_brand_ask_is_read(said):
    assert brand.asks_for_brand_work(said) is True


@pytest.mark.parametrize("said", [
    "What colour is our accent?",                                     # a question about the kit
    "Did the designer finish the brand kit?",
    "How would you improve our brand?",
    "Design my brand yourself, I want it now.",                       # kept with Auto
    "Improve the kit, but don't bother the team.",
    "Make me an invoice for Salt Kitchen: 8 kg Harbour Blend.",       # customer paperwork (F337(c))
    "Design a flyer in our brand colours for the Harvest Club.",
    PRICE_LIST,
    "Make the Lantern Kitchen invoice on my Harbourline Invoice template.",   # a named template (F351)
    "We need more space in the calendar next week.",                  # not the look
    "I need more space in my documents folder.",
    "It's warmer today.",
    "Less red tape for our suppliers, please.",                       # not a colour
    "More gold stock for the shop.",
    "Fix the font size in the table.",                                # not the owner's brand
    "Build a colour palette for my garden.",
    "Make something with my template.",                               # fills one (F351)
    "",
])
def test_anything_else_is_not(said):
    assert brand.asks_for_brand_work(said) is False


def _found(monkeypatch, designer):
    from core.seeds import seed_brand_designer

    seen = []

    def find(db, workspace_id):
        seen.append(workspace_id)
        return designer

    monkeypatch.setattr(seed_brand_designer, "find_brand_designer", find)
    return seen


def _never_seed(workspace_id):
    raise AssertionError("the workspace already has its designer: nothing is seeded")


def test_the_note_files_the_ticket_for_the_designer_by_name_and_keeps_the_kit_from_auto(monkeypatch):
    looked = _found(monkeypatch, DESIGNER)
    note = brand.designer_note(object(), WS, "Help me design my brand.", seed=_never_seed)

    assert looked == [WS]
    assert note.startswith("The owner asked for brand work")
    assert "don't design it yourself" in note
    assert "don't call platform_update_brand_kit, platform_propose_brand_kit or the template tools" in note
    assert 'platform_create_task with assigned_agent_name "Brand Designer"' in note
    assert "platform_update_task_status to 'in_progress'" in note
    assert "carries the owner's words exactly" in note and "invents none" in note
    for step in ("read the logo", "approves, with the Brand Board drawn from the proposal and not yet saved",
                 "save it only after the owner approves", "an invoice, a letter, a proposal and three social cards",
                 "report back with the board and the set"):
        assert step in note
    assert DISPATCH_CONTRACT_FRAGMENT in note


def test_a_workspace_without_a_designer_gets_one_seeded_and_named(monkeypatch):
    _found(monkeypatch, None)
    seeded = []

    def seed(workspace_id):
        seeded.append(workspace_id)
        return "Brand Designer"

    note = brand.designer_note(object(), WS, "Improve the kit.", seed=seed)
    assert seeded == [WS] and 'assigned_agent_name "Brand Designer"' in note


def test_a_designer_the_owner_removed_is_not_brought_back(monkeypatch):
    _found(monkeypatch, None)
    note = brand.designer_note(object(), WS, "Less orange.", seed=lambda ws: None)

    assert note == brand.REMOVED_NOTE
    assert "platform_create_task" not in note and "Don't change the brand kit" in note


def test_a_seed_that_fails_is_logged_and_auto_still_keeps_its_hands_off_the_kit(monkeypatch, caplog):
    _found(monkeypatch, None)

    def broken(workspace_id):
        raise RuntimeError("database gone")

    with caplog.at_level(logging.ERROR):
        note = brand.designer_note(object(), WS, "Design my brand.", seed=broken)
    assert note == brand.UNAVAILABLE_NOTE and "could not be found or seeded" in caplog.text


def test_no_note_and_no_lookup_for_anything_else(monkeypatch):
    looked = _found(monkeypatch, DESIGNER)
    assert brand.designer_note(object(), WS, "What colour is our accent?", seed=_never_seed) is None
    assert looked == []


# ── the turn ───────────────────────────────────────────────────────────────

class _ChatDb:
    """The chat's session, faked: a savepoint, and templates that must not be read for a brand ask."""

    def __init__(self, templates=None):
        self.templates = templates

    def begin_nested(self):
        return contextlib.nullcontext()


@pytest.fixture
def templates(monkeypatch):
    from modules.documents import template_service

    read = []

    class _Templates:
        def __init__(self, db):
            self.db = db

        def list_templates(self, workspace_id):
            read.append(workspace_id)
            return []

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    return read


def test_a_brand_ask_gets_the_designer_note_before_any_template_is_read(monkeypatch, templates):
    _found(monkeypatch, DESIGNER)
    note = read_note(_ChatDb(), WS, ["Make our templates match the new logo."])

    assert 'assigned_agent_name "Brand Designer"' in note and templates == []


def test_paperwork_still_goes_to_the_team(monkeypatch, templates):
    _found(monkeypatch, DESIGNER)
    assert read_note(_ChatDb(), WS, [PRICE_LIST]) == team_note(PRICE_LIST)


def _turn(said, widget_mode=False):
    chat = NS(db=_ChatDb(), workspace_id=str(WS), widget_mode=widget_mode)
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": said}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": PASSAGES})
        yield "searched"

    async def run():
        return [f async for f in fills_the_named_template(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run()) == ["searched"]
    return messages


def test_the_owners_brand_ask_reaches_the_model_as_the_designer_note(monkeypatch, templates):
    _found(monkeypatch, DESIGNER)
    messages = _turn("Help me design my brand.")

    assert messages[-2]["content"] == PASSAGES
    assert messages[-1]["role"] == "system" and messages[-1]["content"].startswith("The owner asked for brand work")


def test_a_widget_visitors_brand_ask_gets_no_note(monkeypatch, templates):
    looked = _found(monkeypatch, DESIGNER)
    messages = _turn("Help me design my brand.", widget_mode=True)

    assert messages[-1]["content"] == PASSAGES and looked == []

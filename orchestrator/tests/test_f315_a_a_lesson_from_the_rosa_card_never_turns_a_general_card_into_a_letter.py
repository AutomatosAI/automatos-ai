"""F315 (night 9): a lesson from one card is a rule, never another card's content.

#1871, the Support Agent's "Draft reply to Rosa at Lantern Kitchen — delivery on 10 kg",
was sent back: "You've signed it as Lantern Kitchen - that's Rosa's café, not us. We're
Harbourline. Start 'Hi Rosa,' and sign off 'Gerard, Harbourline Coffee Roasters' as in
the brand voice doc." The Support Agent's general question #1849, "Delivery charge on
10 kg (second opinion)", re-ran with that correction among its standing notes (F249's
lessons) and came back as a letter beginning "Hi Rosa".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

ROSA = ("You've signed it as Lantern Kitchen - that's Rosa's café, not us. We're Harbourline. Start 'Hi Rosa,' and "
        "sign off 'Gerard, Harbourline Coffee Roasters' as in the brand voice doc.")
SOURCES = "Say which of my documents each figure came from."
GENERAL = "A café wants 10 kg of coffee next week. What do we charge them for delivery, and is it ever free?"


@pytest.fixture
def support(db_session, seed_workspace):
    """The Support Agent: the Rosa card sent back, a general card sent back, and #1849 to run."""
    from core.models import Agent
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    agent = Agent(name="Support Agent", agent_type="custom", description="", status="active", configuration={},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()

    def card(title, brief, status="done", notes=()):
        made = BoardTask(workspace_id=ws, title=title, description=brief, raw_prompt=brief, status=status,
                         assigned_agent_id=agent.id, planning_data={"owner_corrections": [
                             {"note": note, "by": "user:o", "at": f"2026-10-04T12:5{n}:00+00:00"}
                             for n, note in enumerate(notes)]})
        db_session.add(made)
        db_session.flush()
        return made

    card("Draft reply to Rosa at Lantern Kitchen — delivery on 10 kg",
         "Rosa at Lantern Kitchen asked what delivery costs on a 10 kg order. Draft me a reply.", notes=[ROSA])
    card("Delivery charge on a 10 kg café order", GENERAL, notes=[SOURCES])
    general = card("Delivery charge on 10 kg (second opinion)", GENERAL, status="assigned")
    return NS(db=db_session, ws=ws, agent=agent, card=card, general=general)


def test_a_general_card_never_gets_the_rosa_cards_correction(support):
    from services.ticket_redo import KIND_STAYS, STANDING_HEADING, redo_block

    told = redo_block(support.general)

    assert STANDING_HEADING in told and f"- {SOURCES}" in told        # a rule still reaches the next card
    assert "Rosa" not in told and "Lantern Kitchen" not in told        # night 9: "Hi Rosa" on #1849
    assert KIND_STAYS in told


def test_another_card_about_rosa_still_gets_it(support):
    from services.ticket_redo import redo_block

    follow_up = support.card("Follow-up to Rosa at Lantern Kitchen about her first order",
                             "Rosa has ordered. Thank her.", status="assigned")

    assert f"- {ROSA}" in redo_block(follow_up)


def test_a_rule_with_no_card_of_its_own_is_carried_as_before():
    from services.lesson_scope import is_that_cards, particulars

    rosa_card = ("Draft reply to Rosa at Lantern Kitchen — delivery on 10 kg", "Rosa at Lantern Kitchen asked.")
    assert particulars(*rosa_card) == {"Rosa", "Lantern", "Kitchen"}
    assert is_that_cards(ROSA, rosa_card, GENERAL)
    assert not is_that_cards("Leave off 'Perfect!' at the start.", ("Reply to Theo Frost", ""), GENERAL)
    assert not is_that_cards(ROSA, rosa_card, "Reply to Rosa about her delivery")


# ── The draft guides: another card's lesson never makes a question a customer draft ──

@pytest.fixture
def guides(monkeypatch):
    import consumers.chatbot.knowledge_prefetch as knowledge_prefetch
    import modules.tools.tool_router as tool_router
    from config import config

    searched = []

    class _Router:
        async def execute_and_format(self, tool_name, tool_args, **kwargs):
            searched.append(tool_args["query"])
            return {"raw_result": {"results": []}, "frontend_data": None}

    monkeypatch.setattr(knowledge_prefetch, "documents_in", lambda db, ws: 3)
    monkeypatch.setattr(tool_router, "get_tool_router", lambda: _Router())
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", True, raising=False)
    monkeypatch.setattr(type(config), "KNOWLEDGE_PREFETCH_PASSAGES", 5, raising=False)
    monkeypatch.setattr(type(config), "KNOWLEDGE_PREFETCH_MIN_SCORE", 0.5, raising=False)
    return searched


def test_a_general_cards_lessons_never_make_it_a_customer_draft(guides):
    from services.draft_guides import guides_for_draft
    from services.ticket_redo import STANDING_HEADING

    lesson = "When you reply to a café, start with Hi and their first name, and sign off Gerard."
    prompt = f"{GENERAL}\n\n{STANDING_HEADING}\nFollow each one.\n- {lesson}"

    assert asyncio.run(guides_for_draft(None, "dacae30f-7840-40c1-8d03-25c3910affd0", 339, prompt)) == prompt
    assert guides == []                                               # the brief is a question, not a draft

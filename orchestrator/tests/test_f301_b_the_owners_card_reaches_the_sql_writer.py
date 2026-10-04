"""F301 (night 9, query side, B6) — the owner's card reaches the SQL writer.

"Were any Harvest Club boxes late going out in September?" (#1882). The Inventory
Watchdog asked the database tool for "all subscription orders that shipped late in
September 2026" (audit row 289) and reported "9 Harvest Club boxes". Nine boxes were
late across every plan; the Harvest Club had 1 of 68 (order 5207, plan_code CLUB). #1859
went the same way through the Business Analyst (audit row 249). The agent's restatement
dropped the plan, and on a card nothing carried the owner's own words to the writer —
the chat lane has done that since F077 (A).

The words of a card the owner wrote now go to the SQL writer as the owner's question,
beside the agent's restatement; cards the platform wrote, and other workspaces' cards,
never do.
"""
from __future__ import annotations

import pytest

from tests import helpers_shop_database as shop

LATE_BOXES = "Were any Harvest Club boxes late going out in September? How many, and where did that come from?"
RESTATED = "Show me all subscription orders that shipped late in September 2026"


@pytest.fixture
def cards(db_session, seed_workspace):
    from core.models.core import BoardTask

    ws = seed_workspace(shop.WORKSPACE)
    other_ws = seed_workspace()
    made = {
        "owner": BoardTask(workspace_id=ws, title="Late club boxes in September (Inventory)",
                           description=LATE_BOXES, created_by_type="user", status="in_progress"),
        "mission_step": BoardTask(workspace_id=ws, title="Identify late boxes",
                                  description="OBJECTIVE: list late boxes.", created_by_type="system"),
        "elsewhere": BoardTask(workspace_id=other_ws, title="Another shop's card",
                               description="How many Taster boxes?", created_by_type="user"),
    }
    db_session.add_all(made.values())
    db_session.flush()
    return {name: card.id for name, card in made.items()}


def _owner_question(monkeypatch, db_session, card_id):
    service = shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT 1", "data": [], "row_count": 0},
                            facts=shop.SHOP_FACTS)
    shop.ask_like_a_board_agent(RESTATED, caller_context={"board_task_id": card_id}, db=db_session)
    return service.asked[-1]["owner_question"]


def test_the_owners_card_goes_to_the_writer_beside_the_agents_restatement(monkeypatch, db_session, cards):
    asked = _owner_question(monkeypatch, db_session, cards["owner"])

    assert asked == f"Late club boxes in September (Inventory)\n{LATE_BOXES}"


def test_a_card_the_platform_wrote_is_not_the_owners_words(monkeypatch, db_session, cards):
    assert _owner_question(monkeypatch, db_session, cards["mission_step"]) is None


def test_another_workspaces_card_never_reaches_the_writer(monkeypatch, db_session, cards):
    assert _owner_question(monkeypatch, db_session, cards["elsewhere"]) is None


def test_the_chat_turns_own_words_still_come_first(monkeypatch, db_session, cards):
    service = shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT 1", "data": [], "row_count": 0},
                            facts=shop.SHOP_FACTS)
    shop.ask_like_a_board_agent(RESTATED, caller_context={"board_task_id": cards["owner"],
                                                          "user_query": "only the club, please"}, db=db_session)

    assert service.asked[-1]["owner_question"] == "only the club, please"


def test_the_writer_is_told_the_owners_words_win():
    """What the writer does with them is F077 (A)'s prompt, unchanged."""
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService

    prompt = NaturalLanguageToSQLService(llm_provider=None)._build_prompt(
        question=RESTATED, schema_metadata=shop.SHOP_SCHEMA, semantic_layer=None, dialect="postgresql",
        examples=None, owner_question=LATE_BOXES)

    assert f'QUESTION (the person\'s own words; answer this):\n"""\n{LATE_BOXES}\n"""' in prompt
    assert f"RESTATED BY THE ASSISTANT: {RESTATED}" in prompt

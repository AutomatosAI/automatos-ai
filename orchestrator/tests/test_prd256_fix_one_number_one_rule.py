"""P256-FIX-N15-9 (F429, night 15): one number per card, and a superseded standing rule retired.

Night 15: one card went by three numbers (the board's #2894, Auto's '1607', a card's '2603'), and
platform_get_task #2895 was refused. platform_ask_human took only a row id and answered 'Parked
board_task 2603 and asked the human (ask #1607)'; the click line said 'task_id 2603' beside the
number; the feed and the schedule handed the model a ticket's row id beside or without its number;
platform_my_actions listed any '#N' a card's words held (an agent's id, a card's own) as a card.
And the memory store kept the owner's 12 kg cafe minimum after they replaced it with 10 kg, so the
Memory page listed both while the rules said 10.

- A card is named by its board number and taken by it: platform_get_task and platform_ask_human
  read '#N' (and bare N) as the card the board numbers N, even when another card's row id is N, and
  what the model reads names it by number alone.
- The owner's new standing rule retires the rule it replaces (named in ``replaces``, or the same
  words with a number changed): the old row stays in the store, retired, and no read returns it;
  the Memory page, Auto's browse, recall and the rules agree. Saying it again brings it back.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

MOVED_ASIDE = 500_000
NEVER_A_NUMBER = 9_999_999


# ── One number per card ─────────────────────────────────────────────────────────────

@pytest.fixture
def board(db_session, seed_workspace):
    """Two cards: 'Cafe order' numbered with the other card's row id (night 15: #2894 and 2603)."""
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    other = BoardTask(workspace_id=ws, title="Old supplier note", status="in_progress", description="Old.")
    cafe = BoardTask(workspace_id=ws, title="Cafe order", status="in_progress", description="10 kg minimum.")
    db_session.add_all([other, cafe])
    db_session.flush()
    renumber = text("UPDATE board_tasks SET workspace_seq = :seq WHERE id = :id")
    db_session.execute(renumber, {"seq": other.id + MOVED_ASIDE, "id": other.id})
    db_session.execute(renumber, {"seq": other.id, "id": cafe.id})
    db_session.expire_all()
    return NS(db=db_session, ws=ws, cafe=cafe, other=other, number=f"#{other.id:04d}")


@pytest.fixture
def quiet_asks(monkeypatch):
    """No bell and no Telegram for a question."""
    from modules.tools.discovery import handlers_asks

    async def _nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(handlers_asks, "_dispatch_question_pending", _nothing)
    monkeypatch.setattr(handlers_asks, "_capture_question_telegram", _nothing)


def test_platform_get_task_takes_the_board_number_when_another_cards_row_id_is_that_number(board):
    from modules.tools.discovery.handlers_board_tasks import get_board_task
    from modules.tools.formatting.tickets_by_number import by_number_only

    for said in (board.number, str(board.other.id)):
        answer = asyncio.run(get_board_task(board.db, board.ws, {"task_id": said}))
        assert answer["success"] is True and answer["task"]["title"] == "Cafe order", said
        seen = by_number_only(answer)
        assert seen["task"]["number"] == board.number and "id" not in seen["task"]


def test_a_question_parks_the_card_the_board_numbers_and_names_it_by_that_number(board, quiet_asks):
    from modules.tools.discovery.handlers_asks import ask_human

    answer = asyncio.run(ask_human(board.db, board.ws, {"subject_type": "board_task", "subject_id": board.number,
                                                         "question": "10 kg or 12 kg?"}))
    board.db.refresh(board.cafe)
    board.db.refresh(board.other)

    assert answer["success"] is True and answer["parked"] is True
    assert board.cafe.status == "blocked" and board.other.status == "in_progress"
    assert answer["subject_id"] == board.number
    assert answer["message"].startswith(f"Parked ticket {board.number} and asked the owner")
    assert str(board.cafe.id) not in answer["message"] and f"#{answer['ask_id']}" not in answer["message"]


def test_bare_digits_are_the_board_number_and_a_number_no_card_has_is_refused(board, quiet_asks):
    from modules.tools.discovery.handlers_asks import ask_human

    bare = asyncio.run(ask_human(board.db, board.ws, {"subject_type": "board_task", "subject_id": str(board.other.id),
                                                       "question": "Which box?"}))
    board.db.refresh(board.cafe)
    assert bare["success"] is True and board.cafe.status == "blocked" and bare["subject_id"] == board.number

    missing = asyncio.run(ask_human(board.db, board.ws, {"subject_type": "board_task",
                                                          "subject_id": f"#{NEVER_A_NUMBER}", "question": "?"}))
    assert missing == {"success": False, "error": f"No ticket #{NEVER_A_NUMBER} in this workspace. "
                                                  "platform_list_tasks shows each ticket's number."}


def test_a_question_about_other_work_is_left_as_it_was():
    from modules.tools.discovery.ask_ticket_numbers import asks_by_ticket_number

    seen = []

    @asks_by_ticket_number
    async def handler(db, workspace_id, params):
        seen.append(params)
        return {"success": False, "parked": False, "error": "only board_task"}

    params = {"subject_type": "tool_call", "subject_id": "call-1", "question": "?"}
    assert asyncio.run(handler(MagicMock(), uuid4(), params))["error"] == "only board_task"
    assert seen == [params]


def test_the_model_reads_a_feed_or_schedule_ticket_by_its_number_alone():
    from modules.tools.formatting.tickets_by_number import by_number_only

    feed = {"items": [{"id": "task-2603", "source_id": "2603", "number": "#2894", "name": "Cafe order"},
                      {"id": "run-77", "source_id": "77", "name": "A playbook run"}]}
    seen = by_number_only(feed)["items"]
    assert seen[0] == {"number": "#2894", "name": "Cafe order"}
    assert seen[1] == feed["items"][1]                                     # no number: its ids stay
    assert by_number_only({"board_task_id": 2603, "number": "#2894"}) == {"number": "#2894"}
    assert by_number_only({"board_task_id": 2603}) == {"board_task_id": 2603}


def test_a_ticket_named_by_board_task_id_carries_its_number(board):
    from modules.tools.discovery.board_task_numbers import with_board_task_numbers
    from modules.tools.formatting.tickets_by_number import by_number_only

    schedule = {"success": True, "scheduled": [{"board_task_id": board.cafe.id, "name": "Cafe order"},
                                               {"id": "heartbeat-3", "name": "Morning check"}]}
    numbered = with_board_task_numbers(board.db, board.ws, schedule)
    assert numbered["scheduled"][0] == {"board_task_id": board.cafe.id, "name": "Cafe order", "number": board.number}
    assert numbered["scheduled"][1] == schedule["scheduled"][1]
    assert by_number_only(numbered)["scheduled"][0] == {"name": "Cafe order", "number": board.number}
    filed = with_board_task_numbers(board.db, board.ws, {"success": True, "board_task_id": board.cafe.id})
    assert filed["number"] == board.number
    assert with_board_task_numbers(board.db, uuid4(), schedule) == schedule      # another workspace's: no number


def test_the_click_line_names_a_numbered_card_by_its_number_alone():
    from services.click_memory import outcome_of
    from services.click_results import click_line, said_ids

    ran = {"success": True, "ids": {"task_id": "2603", "number": "#2894", "title": "Cafe order"}}
    line = click_line(NS(question_md="Approve (move to Done) #2894 'Cafe order':", tool_name="x"), ran)

    assert line == "Your click ran: approve (move to Done) #2894 'Cafe order' → number #2894, title Cafe order"
    assert "2603" not in outcome_of(ran)
    unnumbered = {"success": True, "ids": {"mission_id": "3f2", "task_id": "2603"}}
    assert said_ids(unnumbered) == "mission_id 3f2, task_id 2603"           # no number: the id is all there is


def test_my_actions_lists_only_the_board_numbers_a_card_shows(board):
    from modules.tools.discovery.my_actions_clicks import board_numbers_said

    card = f"Send the cafe order on {board.number} for agent #{NEVER_A_NUMBER} (send card #{NEVER_A_NUMBER + 1})"
    assert board_numbers_said(board.db, board.ws, card) == [board.number]
    assert board_numbers_said(board.db, uuid4(), card) == []                 # another workspace has no such card


# ── One cafe minimum: the replaced rule is retired ──────────────────────────────────

WS = UUID(int=1)
OWNER, MEMBER = 7, 8
TWELVE = "The cafe minimum order is 12 kg."
TEN = "The cafe minimum order is 10 kg."
MONDAY = "Club dispatch day is Monday."
TUESDAY = "Club dispatch day is Tuesday."


def _value(payload, key):
    for part in key.split("."):
        payload = payload.get(part) if isinstance(payload, dict) else None
    return payload


def _matches(payload, condition):
    value, match = _value(payload, condition.key), condition.match
    return value in match.any if hasattr(match, "any") else value == match.value


class _Qdrant:
    """Qdrant's scroll (must and should, paged), query_points (every point the filter matches,
    scored alike), upsert and set_payload."""

    def __init__(self) -> None:
        self.points: list = []

    async def upsert(self, collection_name, points):
        self.points.extend(NS(id=p.id, payload=p.payload) for p in points)

    async def set_payload(self, collection_name, payload, points):
        for point in self.points:
            if point.id in points:
                point.payload = {**point.payload, **payload}

    def _hits(self, flt):
        return [p for p in self.points if all(_matches(p.payload, c) for c in flt.must or [])
                and (not flt.should or any(_matches(p.payload, c) for c in flt.should))]

    async def scroll(self, collection_name, scroll_filter, limit, offset=None, **_kwargs):
        hits = self._hits(scroll_filter)
        start = offset or 0
        return hits[start:start + limit], (start + limit if start + limit < len(hits) else None)

    async def query_points(self, collection_name, query, query_filter, limit, **_kwargs):
        return NS(points=[NS(id=p.id, payload=p.payload, score=0.9) for p in self._hits(query_filter)][:limit])


@pytest.fixture
def qdrant(monkeypatch):
    import modules.memory.unified_memory_service as ums
    from config import config
    from modules.memory.durable_store import DurableMemoryStore

    client = _Qdrant()
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client = client
    store._embedder = NS(generate_embedding=AsyncMock(return_value=[0.0]))
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    client.service, client.store = service, store
    return client


@pytest.fixture
def owner_says(monkeypatch):
    """What the person's messages in the chat say, as the store reads them."""
    import modules.tools.discovery.said_instructions as said

    messages: list = []
    monkeypatch.setattr(said, "chat_owner_turns", lambda db, workspace_id, params: list(messages))
    return messages


def _store(content, *, by=OWNER, **extra):
    from modules.tools.discovery.handlers_workspace import store_memory

    return asyncio.run(store_memory(MagicMock(), WS, {"content": content, "_driving_user_id": by, **extra}))


def _rules(viewer=OWNER):
    from modules.context.sections.memory import owners_rules

    return list(asyncio.run(owners_rules(WS, viewer_subject_id=f"user:{viewer}", widget_mode=False)).texts)


def _row(client, content):
    return next(p for p in client.points if p.payload["content"] == content)


def _page(client, monkeypatch, viewer=OWNER):
    """GET /api/v1/memory/browse over the real store (no agents, no L2 rows)."""
    from api import memory_stats

    client.service.list_short_term_by_type = AsyncMock(return_value=[])
    monkeypatch.setattr(memory_stats, "_get_memory_service", lambda: client.service)
    monkeypatch.setattr(memory_stats, "_get_agent_ids", lambda workspace_id, db: [])
    page = asyncio.run(memory_stats.browse_memories(query=None, limit=20, content_type=None, tier=None,
                                                    ctx=NS(workspace_id=WS, user=NS(id=viewer)), db=MagicMock()))
    return [m["content"] for m in page["memories"]]


def _autos_browse(viewer=OWNER):
    from modules.tools.discovery.handlers_search import browse_memories

    answer = asyncio.run(browse_memories(MagicMock(), WS, {"_driving_user_id": viewer}))
    return [m["content"] for m in answer["memories"]]


def test_twelve_kg_replaced_by_ten_is_retired_and_every_list_agrees(qdrant, owner_says, monkeypatch):
    owner_says.append(f"Remember: {TWELVE}")
    assert _store(TWELVE)["standing_rule"] is True
    owner_says.append(f"Change of plan. {TEN}")
    answer = _store(TEN)

    assert answer["success"] is True and answer["standing_rule"] is True and answer["retired"] == [TWELVE]
    assert _rules() == [TEN]
    assert TEN in _page(qdrant, monkeypatch) and TWELVE not in _page(qdrant, monkeypatch)
    assert _autos_browse() == [TEN]
    monkeypatch.setattr("core.observability.substrate_metrics.record_substrate_search_nowait", lambda **kw: None)
    recalled = asyncio.run(qdrant.service._durable.search("cafe minimum", f"mem:{WS}", limit=5))
    assert [m["memory"] for m in recalled] == [TEN]
    old = _row(qdrant, TWELVE).payload["metadata"]                      # kept, retired: nothing deleted
    assert old["replaced_by"] == TEN and old["retired_at"] and old["stated_by"] == "owner"


def test_a_rule_named_in_replaces_is_retired_whatever_its_words(qdrant, owner_says):
    owner_says.extend([MONDAY, TUESDAY])
    _store(MONDAY)
    assert "retired" not in _store(TUESDAY)                             # a different word: two rules, unless named
    assert sorted(_rules()) == sorted([MONDAY, TUESDAY])
    owner_says.append("Club dispatch day is Wednesday.")
    answer = _store("Club dispatch day is Wednesday.", replaces=f"- {TUESDAY}")
    assert answer["retired"] == [TUESDAY] and sorted(_rules()) == sorted([MONDAY, "Club dispatch day is Wednesday."])


def test_a_rule_named_by_its_id_is_retired(qdrant, owner_says):
    owner_says.extend([MONDAY, TUESDAY])
    _store(MONDAY)
    answer = _store(TUESDAY, replaces=_row(qdrant, MONDAY).id)
    assert answer["retired"] == [MONDAY] and _rules() == [TUESDAY]


def test_saying_a_retired_rule_again_brings_it_back_and_retires_the_one_that_replaced_it(qdrant, owner_says):
    owner_says.extend([TWELVE, TEN])
    _store(TWELVE)
    _store(TEN)
    answer = _store(TWELVE)

    assert answer["retired"] == [TEN] and _rules() == [TWELVE]
    assert len(qdrant.points) == 2                                        # the same row, back: no copy
    assert "retired_at" not in _row(qdrant, TWELVE).payload["metadata"]


def test_an_old_12_kg_memory_that_was_never_a_rule_is_retired_by_the_10_kg_rule(qdrant, owner_says, monkeypatch):
    _store(TWELVE)                                                        # night 14's row: owned, no stated_by
    assert TWELVE in _page(qdrant, monkeypatch) and _rules() == []
    owner_says.append(TEN)
    assert _store(TEN)["retired"] == [TWELVE]
    assert _page(qdrant, monkeypatch) == [TEN] and _rules() == [TEN]


def test_replaces_never_hides_a_memory_that_is_not_a_standing_rule(qdrant, owner_says):
    note = "Quay pays on thirty day terms, from the invoice."
    _store(note)                                                          # Auto's wording: owned, no rule
    owner_says.append(TEN)
    answer = _store(TEN, replaces=note)
    assert "retired" not in answer and "retired_at" not in _row(qdrant, note).payload["metadata"]


# P256-FIX-N17-4 (night 17, F447): 'forget the other posting rules … replace the lot'.
WEDNESDAY = "Club dispatch day is Wednesday."
THE_LOT = "Forget the other dispatch rules. The only one that counts: club dispatch day is Wednesday."
THE_LOT_THEIRS = "The only one that counts: club dispatch day is Wednesday."


@pytest.mark.parametrize("as_text", [False, True])
def test_replace_the_lot_retires_every_rule_named_once_the_new_one_is_theirs(qdrant, owner_says, as_text):
    owner_says.extend([MONDAY, TUESDAY])
    _store(MONDAY)
    _store(TUESDAY)
    owner_says.append(THE_LOT)
    replaces = json.dumps([MONDAY, TUESDAY]) if as_text else [MONDAY, TUESDAY]
    first = _store(WEDNESDAY, replaces=replaces)                         # their lead-in left out: no rule
    assert first["standing_rule"] is False and "retired" not in first and first["store_as"] == THE_LOT_THEIRS
    assert sorted(_rules()) == sorted([MONDAY, TUESDAY])
    assert not any(p.payload["metadata"].get("retired_at") for p in qdrant.points)

    again = _store(first["store_as"], replaces=replaces)
    assert again["standing_rule"] is True and sorted(again["retired"]) == sorted([MONDAY, TUESDAY])
    assert _rules() == [THE_LOT_THEIRS]


def test_replace_the_lot_by_ids_and_an_ordinary_memory_in_the_list_is_kept(qdrant, owner_says):
    note = "Quay pays on thirty day terms, from the invoice."
    _store(note)                                                          # Auto's wording: owned, no rule
    owner_says.extend([MONDAY, TUESDAY])
    _store(MONDAY)
    _store(TUESDAY)
    owner_says.append(THE_LOT)
    named = [_row(qdrant, MONDAY).id, _row(qdrant, TUESDAY).id, note]
    answer = _store(THE_LOT_THEIRS, replaces=named)
    assert sorted(answer["retired"]) == sorted([MONDAY, TUESDAY]) and _rules() == [THE_LOT_THEIRS]
    assert "retired_at" not in _row(qdrant, note).payload["metadata"]


def test_replaces_is_read_as_text_or_a_list_by_the_schema_and_the_handler():
    from modules.tools.discovery.actions_memory_store import _STORE_MEMORY_PARAMETERS
    from modules.tools.discovery.rule_replaces import _named
    from modules.tools.execution.schema_reads import _takes_a_container

    replaces = _STORE_MEMORY_PARAMETERS["properties"]["replaces"]
    assert replaces["type"] == ["string", "array"] and replaces["items"] == {"type": "string"}
    assert _takes_a_container(replaces) is False                            # the executor leaves text as text
    assert _named(MONDAY) == _named([MONDAY]) == _named(json.dumps([MONDAY]))
    assert _named(None) == _named([]) == _named(True) == (frozenset(), frozenset())
    assert _named(["abc-1", " "])[0] == frozenset({"abc-1"})


def test_the_delete_and_store_descriptions_say_how_to_replace_a_rule():
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    assert "list each in replaces" in registry.get("platform_store_memory").description
    delete = registry.get("platform_delete_memory").description
    assert "Not for replacing a standing rule" in delete and "standing_rule true" in delete


def test_another_write_of_a_retired_rule_leaves_it_retired(qdrant, owner_says):
    owner_says.append(TWELVE)
    _store(TWELVE)
    owner_says[:] = [TEN]
    _store(TEN)
    again = _store(TWELVE)                                                # not in the owner's words this time
    assert again["success"] is True and again["standing_rule"] is False
    assert _row(qdrant, TWELVE).payload["metadata"]["retired_at"] and _rules() == [TEN]


def test_what_auto_words_itself_retires_nothing(qdrant, owner_says):
    owner_says.append(TWELVE)
    _store(TWELVE)
    answer = _store("Cafe minimum is now 10 kg per order, the owner said.", replaces=TWELVE)
    assert answer["standing_rule"] is False and "retired" not in answer
    assert _rules() == [TWELVE]


def test_another_members_rule_is_never_retired(qdrant, owner_says):
    owner_says.extend([TWELVE, TEN])
    _store(TWELVE, by=MEMBER)
    answer = _store(TEN, replaces=TWELVE)
    assert "retired" not in answer
    assert _rules(MEMBER) == [TWELVE] and _rules() == [TEN]


def test_a_retire_that_fails_keeps_the_new_rule_and_says_both_are_stored(qdrant, owner_says, monkeypatch, caplog):
    import modules.memory.retired as retired
    from modules.tools.discovery.rule_replaces import NOT_RETIRED_NOTE

    async def _down(*args, **kwargs):
        raise RuntimeError("qdrant down")

    owner_says.extend([TWELVE, TEN])
    _store(TWELVE)
    monkeypatch.setattr(retired, "retire", _down)
    answer = _store(TEN)
    assert answer["success"] is True and answer["standing_rule"] is True
    assert answer["replaced_rule_kept"] == NOT_RETIRED_NOTE
    assert "was not retired" in caplog.text


@pytest.mark.parametrize("old, new, same", [
    (TWELVE, TEN, True),
    ("Order 12kg of Konga for the cafe", "Order 10kg of Konga for the cafe", True),
    ("Price the box at £12 for members", "Price the box at £14 for members", True),
    (TWELVE, TWELVE, False),                                              # the same rule
    ("Week 1", "Week 2", False),                                          # too few words to be one statement
    ("Post 3 times on Monday", "Post 2 times on Friday", False),          # a word changed too
    ("Price the box at £12 for members", "Price the box at 12% for members", False),   # another unit
    ("Ship 12 bags to Quay", "Ship 12 bags to Quay today", False),        # words added
    ("Table 1 is reserved for VIP guests", "Table 2 is reserved for VIP guests", False),   # a number that names
    ("The Q1 target is 40 bags", "The Q2 target is 40 bags", False),
    ("We post at 10am on Tuesdays", "We post at 11am on Tuesdays", True),
])
def test_same_but_its_numbers(old, new, same):
    from modules.tools.discovery.rule_replaces import same_but_its_numbers

    assert same_but_its_numbers(old, new) is same

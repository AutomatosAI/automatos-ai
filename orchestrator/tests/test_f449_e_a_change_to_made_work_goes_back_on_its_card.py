"""P256-FIX-N17-10 (F449, night 17): a change to work already made goes back on its card.

Ask 170, "make the guji quote 8kg fortnightly instead and tell cellar", filed #1754 "Read the existing
Guji quote from the Knowledge Base" while #1751, the quote, waited in Review; ask 219, "Make the
Tidewater quote shorter…", filed #1757 "Find the existing Tidewater quote in the knowledge base". Both
ran in new folders, could not open #1751 and left two quotes that disagree. A new card about made work
the owner described for a change is refused now, with the card that holds it and the call that sends it
back with their words (``follows_the_owner._a_copy``, ``described_work``). The route works when taken:
ask 140 sent #1746 back.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery.described_work import DescribedCard, described

GUJI = "make the guji quote 8kg fortnightly instead and tell cellar"
TIDEWATER = "Make the Tidewater quote shorter, it runs to two pages."
SEND_BACK_1751 = 'platform_update_task_status {task_id: "#1751", status: "assigned"'


def _card(ref, title, status="review", agent="QUOTER", file_path="quotes/guji-quote-v2.pdf", day=14):
    task = NS(id=3000 + int(ref.strip("#")) % 1000, title=title, status=status, source_type="user", description="",
              created_at=datetime(2026, 10, day, 21, 5, tzinfo=timezone.utc))
    return DescribedCard(ref=ref, task=task, agent=agent, file_path=file_path)


GUJI_CARD = _card("#1751", "Guji quote for Lantern Kitchen")


@pytest.fixture
def turn(monkeypatch):
    """The owner's words this turn, naming no card by number, and the cards that hold the work."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import OwnerTurn

    looked_up = []

    def _set(latest, cards=(GUJI_CARD,), named=()):
        def _cards(db, ws, words, limit=3):
            looked_up.append(list(words))
            return list(cards)
        monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: OwnerTurn(latest=latest, earlier="",
                                                                               cards=tuple(named)))
        monkeypatch.setattr(guard, "described_cards", _cards)
        return looked_up
    monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: ())
    return _set


def _new_card(title, description="Find it and change it as the owner asked."):
    from modules.tools.discovery.follows_the_owner import refusal_for

    return refusal_for(None, uuid4(), "platform_create_task", {"title": title, "description": description,
                                                              "assigned_agent_name": "QUOTER"},
                       {"conversation_id": str(uuid4())})


def test_ask_170_goes_back_on_the_quotes_card(turn):
    looked_up = turn(GUJI)
    refusal = _new_card("Read the existing Guji quote from the Knowledge Base")

    assert looked_up == [["guji", "quote"]]
    assert refusal.startswith("#1751 ('Guji quote for Lantern Kitchen', review) is the work the owner means")
    assert SEND_BACK_1751 in refusal and "make the guji quote 8kg fortnightly instead" in refusal
    assert (refusal.card, refusal.owner_reason) == ("#1751", "#1751 is already on your board, so no new card was made")


def test_ask_219_likewise(turn):
    turn(TIDEWATER, cards=(_card("#1751", "Tidewater quote", file_path="quotes/tidewater.pdf"),))
    refusal = _new_card("Find the existing Tidewater quote in the knowledge base")

    assert "#1751" in refusal and SEND_BACK_1751 in refusal and refusal.card == "#1751"


def test_two_cards_holding_it_are_listed_newest_first_and_none_is_sent_back(turn):
    newer, older = _card("#1757", "Guji quote (short)", day=15), GUJI_CARD
    turn(GUJI, cards=(newer, older))
    refusal = _new_card("Update the Guji quote")

    assert refusal.index("#1757 'Guji quote (short)', QUOTER, review, 15 Oct") < refusal.index("#1751 'Guji quote")
    assert "act on the one they mean, or ask them; don't make a new card" in refusal.lower()
    assert "platform_update_task_status" not in refusal and refusal.card is None


def test_work_still_running_is_waited_for(turn):
    turn(GUJI, cards=(_card("#1751", "Guji quote", status="in_progress"),))
    refusal = _new_card("Guji quote, 8kg fortnightly")

    assert "It is running: wait for its result, then send it back with the owner's words." in refusal
    assert "platform_update_task_status" not in refusal


@pytest.mark.parametrize("latest, title", [
    ("draft the newsletter intro", "Newsletter intro"),                                      # new work
    ("Fix the van booking and draft the newsletter intro", "Draft the newsletter intro"),    # the edit is the van's
], ids=["new-work", "edit-in-another-clause"])
def test_new_work_is_filed_whatever_the_board_holds(turn, latest, title):
    looked_up = turn(latest, cards=(_card("#1690", "October newsletter intro"),))
    assert _new_card(title) is None and looked_up == []


def test_a_card_about_other_work_beside_an_edit_is_filed(turn):
    looked_up = turn("Make the Tidewater quote shorter and draft a Quay invoice")
    assert _new_card("Draft the Quay invoice", "Invoice Quay for 12kg of Guji.") is None and looked_up == []


def test_a_new_card_asked_for_names_the_work_and_its_file(turn):
    turn("Make a new card to make the Guji quote shorter")
    refused = _new_card("Shorter Guji quote", "Write a shorter Guji quote.")
    named = _new_card("Shorter Guji quote", "Start from #1751 and its file quotes/guji-quote-v2.pdf; cut it to a page.")

    assert "#1751" in refused and "quotes/guji-quote-v2.pdf" in refused and refused.card == "#1751"
    assert named is None


def test_no_card_holding_it_files_the_card(turn):
    """No card holds the work (described_cards leaves a cancelled one out): the new card is filed."""
    looked_up = turn(GUJI, cards=())
    assert _new_card("Read the existing Guji quote") is None and looked_up == [["guji", "quote"]]


def test_a_card_named_by_its_number_is_never_looked_up_by_its_words(turn):
    from modules.tools.discovery.owner_turn import NamedCard

    source = NamedCard(ref="#1746", seq=1746, step=None, task=NS(id=3746, title="Tidewater quote", status="done",
                                                                  source_type="user", description=""))
    looked_up = turn("Use the Tidewater quote for the Quay letter (see #1746)", named=(source,))
    assert _new_card("Quay letter", "Write the Quay letter from the Tidewater quote.") is None and looked_up == []


def test_the_words_that_describe_the_work():
    assert described(GUJI) == ["guji", "quote"]
    assert described(TIDEWATER) == ["tidewater", "quote"]
    assert described("tweak the Hale & Co invoice") == ["hale", "co", "invoice"]
    assert described("draft the newsletter intro") == [] and described("change the quote") == []


def test_the_prompts_say_a_change_goes_back_on_its_card():
    from consumers.chatbot.auto import build_assign_directive
    from consumers.chatbot.handoffs import CHANGE_GOES_BACK, TEAM_NOTE

    assert CHANGE_GOES_BACK == ("When the owner asks for a change to work already made, find its card with "
                                "platform_list_tasks {search: the work's name} and send it back with their words; "
                                "don't file a new card.")
    assert CHANGE_GOES_BACK in TEAM_NOTE.format(article="a", kind="quote")
    for resolved in (True, False):
        directive = build_assign_directive(target_agent_name="QUOTER" if resolved else None, resolved=resolved,
                                           deferred=False)
        assert CHANGE_GOES_BACK in directive


# ── described_cards on the board ───────────────────────────────────────────────

def _board_card(db, ws, title, status="review", seq=None, made=None):
    from core.models.core import BoardTask

    card = BoardTask(workspace_id=ws, title=title, status=status, priority="medium", workspace_seq=seq,
                     created_at=made)
    db.add(card)
    db.flush()
    return card


def _file(db, ws, card, path, made):
    db.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path, created_at) "
        "VALUES (CAST(:ws AS uuid), 'task', :card, 'document', :title, :path, :made)"),
        {"ws": str(ws), "card": str(card.id), "title": path.rsplit("/", 1)[-1], "path": path, "made": made})
    db.flush()


@pytest.fixture
def board(db_session, seed_workspace):
    from tests import test_f161_a_later_step_reads_an_earlier_steps_file as f161

    db_session.execute(text(f161.OUTPUTS_VIEW))
    return NS(db=db_session, ws=UUID(seed_workspace()), other=UUID(seed_workspace()))


def _at(day):
    return datetime(2026, 10, day, 20, 0, tzinfo=timezone.utc)


def test_the_cards_holding_the_work_newest_first_with_their_newest_file(board):
    from modules.tools.discovery.described_work import described_cards

    titled = _board_card(board.db, board.ws, "Guji quote for Lantern Kitchen", seq=1751, made=_at(14))
    _file(board.db, board.ws, titled, "quotes/guji-v1.pdf", _at(14))
    _file(board.db, board.ws, titled, "quotes/guji-v2.pdf", _at(15))
    by_file = _board_card(board.db, board.ws, "Lantern Kitchen paperwork", seq=1752, made=_at(15))
    _file(board.db, board.ws, by_file, "quotes/guji-quote-lantern.pdf", _at(15))
    _board_card(board.db, board.ws, "Guji quote, first try", status="cancelled", seq=1700, made=_at(16))
    _board_card(board.db, board.other, "Guji quote", seq=1751, made=_at(16))           # another workspace

    found = described_cards(board.db, board.ws, ["guji", "quote"])

    assert [(card.ref, card.file_path) for card in found] == [("#1752", "quotes/guji-quote-lantern.pdf"),
                                                              ("#1751", "quotes/guji-v2.pdf")]


def test_at_most_three_cards_are_read(board):
    from modules.tools.discovery.described_work import described_cards

    for day in range(10, 15):
        _board_card(board.db, board.ws, f"Tidewater quote v{day}", seq=1800 + day, made=_at(day))

    found = described_cards(board.db, board.ws, ["tidewater", "quote"])
    assert [card.ref for card in found] == ["#1814", "#1813", "#1812"]
    assert all(card.file_path is None for card in found)


def test_no_words_read_nothing(board):
    from modules.tools.discovery.described_work import described_cards

    assert described_cards(board.db, board.ws, []) == []

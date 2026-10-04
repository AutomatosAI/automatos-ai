"""F241 and F265 (night 7b): a new card, its brief, and a note made in chat.

- "Please give #0192 to the Shopify Support Agent" made a NEW card, #0194 "Handle task
  #0192", and #0192 stayed in the Inbox.
- Auto's brief for #0182 told the Analyst "Use `platform_update_task_status` to mark the
  task as 'review' when complete". The agent's own move was refused, and its first
  answer said a table was made where there was none (F265).
- "Approve #0198 with this note: Right, 204 bags a sack" left the owner's note on #0198
  credited to "an agent".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

NIGHT_7B_BRIEF = (
    "**OBJECTIVE** — Calculate the margin per 250g bag for the three October coffees.\n"
    "**OUTPUT** — A table showing the margin for each coffee. The working for each calculation must be shown.\n"
    "**TOOLS** — Use `platform_read_document` to access any relevant pricing or cost documents. "
    "Use `platform_update_task_status` to mark the task as 'review' when complete.\n"
    "**BOUNDARIES** —\n* Roast loss: 15%.\n* The task is done when all three coffees are worked out."
)


@pytest.fixture
def shop(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_board_tasks as handlers
    from core.models.core import BoardTask

    for quiet in ("_notify_board_safe", "_notify_dispatch_safe", "_consent_for_chat_filed"):
        monkeypatch.setattr(handlers, quiet, lambda *a, **k: None)
    ws = UUID(seed_workspace())
    card = BoardTask(workspace_id=ws, title="Reply to Ruth at Quayside: split bag", status="inbox",
                     description="Draft the reply.")
    db_session.add(card)
    db_session.flush()
    return NS(db=db_session, ws=ws, card=card, number=f"#{card.workspace_seq:04d}", handlers=handlers)


def test_a_new_card_that_is_another_cards_number_is_refused(shop):
    from modules.tools.discovery.new_card_checks import copy_refusal

    for title in (f"Handle task {shop.number}", shop.number, f"Process card {shop.number}",
                  f"Please do ticket {shop.number}"):
        refusal = copy_refusal(shop.db, shop.ws, title)
        assert refusal and "already on the board, so no new card was made" in refusal, title
        assert f'platform_assign_task with task_id "{shop.number}"' in refusal


def test_a_new_card_that_only_mentions_another_is_made(shop):
    from modules.tools.discovery.new_card_checks import copy_refusal

    for title in (f"Follow up on {shop.number} once Ruth replies", "Handle task #9999",
                  "Price list for The Lantern Room"):
        assert copy_refusal(shop.db, shop.ws, title) is None, title


def test_the_create_tool_files_no_copy(shop):
    out = asyncio.run(shop.handlers.create_board_task(shop.db, shop.ws, {
        "title": f"Handle task {shop.number}", "description": f"OBJECTIVE: Process task {shop.number}.",
        "assigned_agent_name": "Shopify Support Agent", "status": "assigned"}))
    from core.models.core import BoardTask

    assert out["success"] is False and "already on the board" in out["error"]
    assert shop.db.query(BoardTask).filter(BoardTask.workspace_id == shop.ws).count() == 1


def test_a_brief_loses_the_sentence_that_moves_its_card():
    from modules.tools.discovery.new_card_checks import without_status_orders

    brief, taken = without_status_orders(NIGHT_7B_BRIEF)

    assert taken is True
    assert "platform_update_task_status" not in brief and "mark the task" not in brief
    assert "Use `platform_read_document` to access any relevant pricing or cost documents." in brief
    assert "**OBJECTIVE**" in brief and "Roast loss: 15%." in brief


def test_a_brief_with_no_status_order_is_left_as_it_is():
    from modules.tools.discovery.new_card_checks import without_status_orders

    plain = "OBJECTIVE: Reply to Priya. OUTPUT: The email, starting at To:. Put it on this card."
    assert without_status_orders(plain) == (plain, False)
    assert without_status_orders(None) == (None, False)


def test_the_filed_card_carries_the_brief_without_the_order_and_says_so(shop):
    from core.models.core import BoardTask

    out = asyncio.run(shop.handlers.create_board_task(shop.db, shop.ws, {
        "title": "Margin per 250 g bag for the October coffees", "description": NIGHT_7B_BRIEF}))

    assert out["success"] is True and "board moves the card" in out["brief_note"]
    filed = shop.db.query(BoardTask).filter(BoardTask.id == out["task_id"]).one()
    assert "platform_update_task_status" not in filed.description


def test_autos_assign_directive_says_the_board_moves_the_card():
    from consumers.chatbot.auto import BOARD_MOVES_THE_CARD, build_assign_directive

    for resolved in (True, False):
        directive = build_assign_directive(target_agent_name="Analyst" if resolved else None,
                                           resolved=resolved, deferred=False)
        assert BOARD_MOVES_THE_CARD in directive


def _notes(shop):
    shop.db.refresh(shop.card)
    return (shop.card.runtime_ref or {}).get("session_notes") or []


@pytest.fixture
def notes_in_this_session(shop, monkeypatch):
    """The ticket notes are written in a session of their own (ticket_changes._append);
    here, in the test's own, whose rows another session can't see before teardown."""
    from contextlib import contextmanager

    import core.database.database as database

    @contextmanager
    def this_session():
        yield shop.db

    monkeypatch.setattr(database, "get_db_session", this_session)


def test_a_note_auto_writes_in_a_chat_the_owner_drives_is_the_owners(shop):
    out = asyncio.run(shop.handlers.update_board_task(shop.db, shop.ws, {
        "task_id": shop.number, "note": "Right, 204 bags a sack. Thanks.", "_user_id": "user_owner"}))

    assert out["success"] is True and out["updated"]["note"] == "added"
    assert [(n["note"], n["by"]) for n in _notes(shop)] == [("Right, 204 bags a sack. Thanks.", "you")]


def test_a_note_with_another_change_is_the_owners_too(shop):
    out = asyncio.run(shop.handlers.update_board_task(shop.db, shop.ws, {
        "task_id": shop.number, "priority": "high", "note": "Ruth needs it today.", "_user_id": "user_owner"}))

    assert out["updated"] == {"priority": "high", "note": "added"}
    assert _notes(shop)[-1]["by"] == "you"


def test_a_note_from_an_agents_own_lane_is_still_an_agents(shop):
    asyncio.run(shop.handlers.update_board_task(shop.db, shop.ws, {"task_id": shop.number, "note": "Supplier replied."}))
    assert _notes(shop)[-1]["by"] == "an agent"


def test_notes_and_brief_reach_their_declared_names():
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import map_optional_aliases

    action = get_action_registry().get("platform_update_task")
    mapped = map_optional_aliases("platform_update_task", action,
                                  {"task_id": "#0199", "brief": "About 80 words.", "notes": "Plain, please."}, "t")
    assert mapped == {"task_id": "#0199", "description": "About 80 words.", "note": "Plain, please."}


# ── The re-brief: "Update #0199 with that brief and send it back" ────────────────

@pytest.fixture
def in_review(shop, monkeypatch):
    """#0199-like: a Content Creator's card in Review with its first draft."""
    import api.board_tasks as board
    from sqlalchemy import text

    monkeypatch.setattr(board, "notify_task_available", lambda *a, **k: None)
    agent = shop.db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace') "
        "RETURNING id"), {"w": str(shop.ws)}).scalar()
    shop.card.assigned_agent_id = agent
    shop.card.status = "review"
    shop.card.description = "Write the blog intro on resting espresso."
    shop.card.result = '"At Harbourline Coffee Roasters, we believe great espresso starts…"'
    shop.db.flush()
    return shop


NEW_BRIEF = ("About 80 words, plain, first person plural: fresh coffee gives off gas and five days' rest makes "
             "shots steadier. Mention Sam our head roaster. No line before the paragraph, no quote marks.")


def test_a_new_brief_with_send_back_goes_back_to_its_agent_on_the_same_card(in_review, notes_in_this_session):
    out = asyncio.run(in_review.handlers.update_board_task(in_review.db, in_review.ws, {
        "task_id": in_review.number, "description": NEW_BRIEF, "send_back": True, "_user_id": "user_owner"}))

    assert out["success"] is True and out["updated"]["send_back"] is True
    card = in_review.card
    in_review.db.refresh(card)
    assert card.status == "assigned" and card.description == NEW_BRIEF and card.raw_prompt == NEW_BRIEF
    assert card.planning_data["previous_briefs"][-1]["description"] == "Write the blog intro on resting espresso."
    assert "we believe great espresso" in card.planning_data["previous_runs"][-1]["result"]   # the draft is kept
    assert "new brief" in card.planning_data["owner_corrections"][-1]["note"]
    assert any("Gave this a new brief and sent it back" in n["note"] for n in _notes(in_review))


def test_send_back_without_a_brief_says_where_the_brief_goes(in_review):
    out = asyncio.run(in_review.handlers.update_board_task(in_review.db, in_review.ws, {
        "task_id": in_review.number, "send_back": True}))
    assert out["success"] is False and "the brief goes in description" in out["error"]
    in_review.db.refresh(in_review.card)
    assert in_review.card.status == "review"


def test_a_running_card_is_not_rebriefed_under_its_run(in_review):
    in_review.card.status = "in_progress"
    in_review.db.flush()
    out = asyncio.run(in_review.handlers.update_board_task(in_review.db, in_review.ws, {
        "task_id": in_review.number, "description": NEW_BRIEF, "send_back": True}))
    assert out["success"] is False and "re-brief it once it stops" in out["error"]


def test_a_plain_edit_of_the_brief_loses_its_status_order_too(in_review):
    asyncio.run(in_review.handlers.update_board_task(in_review.db, in_review.ws, {
        "task_id": in_review.number, "description": NIGHT_7B_BRIEF}))
    in_review.db.refresh(in_review.card)
    assert "platform_update_task_status" not in in_review.card.description
    assert in_review.card.status == "review"                           # an edit never moves the card


def test_a_status_sent_to_the_edit_tool_is_told_about_send_back():
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import undeclared_params_refusal

    action = get_action_registry().get("platform_update_task")
    refused = undeclared_params_refusal("platform_update_task", action,
                                        {"task_id": 199, "status": "pending", "description": NEW_BRIEF}, "t")
    assert "send_back: true" in refused


def _cancel_with(note_written):
    """A status handler that cancels the card, writing its own note when told to (F273's cancel)."""
    from modules.tools.discovery.ticket_changes import STATUS, guarded_and_recorded

    @guarded_and_recorded(STATUS)
    async def cancel(db, workspace_id, params):
        from core.models.core import BoardTask
        from services.cli_host_service import append_session_note

        db.query(BoardTask).filter(BoardTask.id == params["task_id"]).update({"status": "cancelled"})
        if note_written:
            append_session_note(db, task_id=params["task_id"], workspace_id=workspace_id,
                                note="Cancelled the playbook run, in chat.", by="Auto")
        db.commit()
        return {"success": True, "task_id": params["task_id"], "status": "cancelled"}
    return cancel


def test_a_cancel_that_notes_itself_gets_no_second_note(shop, notes_in_this_session):
    asyncio.run(_cancel_with(True)(shop.db, shop.ws, {"task_id": shop.card.id, "status": "cancelled"}))
    assert [n["note"] for n in _notes(shop)] == ["Cancelled the playbook run, in chat."]


def test_a_cancel_that_writes_no_note_is_noted_as_a_move(shop, notes_in_this_session):
    asyncio.run(_cancel_with(False)(shop.db, shop.ws, {"task_id": shop.card.id, "status": "cancelled"}))
    assert [(n["note"], n["by"]) for n in _notes(shop)] == [("Moved this from Inbox to Cancelled.", "An agent")]

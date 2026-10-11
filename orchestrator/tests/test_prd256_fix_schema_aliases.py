"""PRD-256 FX-013 (night 12, M2, F390): strict schemas accept what the model reliably sends.

- 31 ``platform_query_data`` calls were refused for sending ``query`` instead of ``question``
  (A255, A319, A457, A475, A529): ``query`` is read as ``question``, and
  ``platform_query_database`` runs ``platform_query_data``.
- 10 social posts were refused because ``variables`` / ``copy`` came as JSON text ("Input should
  be a valid dictionary"): a promoted tool's object or array parameter sent as JSON text is read
  as what it holds, through one helper (``params_text.json_container``) at the executor.
- 6 ``store_memory`` calls were refused on ``source_type`` until Auto asked the owner to pick one
  (A536): it is optional, platform_verified on a turn a person drives, else claude_reports.
- ``platform_update_task`` with a new description and a status on a worked card was refused while
  the owner's correction was urgent (A440): the description re-briefs, the status is dropped and
  the receipt says "status ignored: a re-brief sends the card back by itself".
"""
from __future__ import annotations

import asyncio
import json

import pytest

from tests import test_f241_e_a_new_card_never_copies_one_and_notes_are_the_owners as f241

shop, in_review, notes_in_this_session = f241.shop, f241.in_review, f241.notes_in_this_session

QUESTION = "How many active Harvest Club subscriptions do we have?"
OWNER = 7


def _action(name):
    from modules.tools.discovery import get_action_registry

    return get_action_registry().get(name)


def _executed(tool_name, parameters):
    """What ``execute_tool`` runs after the executor's decoder (``decodes_nested_params``)."""
    from modules.tools.execution.params_text import decodes_nested_params

    seen = {}

    async def execute(self, name, params, *args, **kwargs):
        seen.update(name=name, params=params)
        return {"success": True}

    asyncio.run(decodes_nested_params(execute)(None, tool_name, parameters))
    return seen["name"], seen["params"]


# ── platform_query_data: query is the question ───────────────────────────────

def test_query_is_read_as_the_question_and_passes_the_strict_checks():
    from modules.tools.execution.direct_contract import missing_on_a_direct_call
    from modules.tools.execution.unified_executor import VIA_DIRECT_CALL, undeclared_params_refusal

    name, params = _executed("platform_query_data", {"query": QUESTION})

    assert (name, params) == ("platform_query_data", {"question": QUESTION})
    action = _action(name)
    assert undeclared_params_refusal(name, action, params, "t", VIA_DIRECT_CALL) is None   # night 12: refused
    assert missing_on_a_direct_call(name, action, params) is None


def test_query_through_the_dispatcher_is_read_as_the_question():
    name, call = _executed("platform_execute", {"action": "platform_query_data", "params": {"query": QUESTION}})
    assert name == "platform_execute" and call == {"action": "platform_query_data", "params": {"question": QUESTION}}


def test_platform_query_database_runs_platform_query_data():
    assert _executed("platform_query_database", {"query": QUESTION}) == ("platform_query_data", {"question": QUESTION})
    _, call = _executed("platform_execute", {"action": "platform_query_database", "params": {"query": QUESTION}})
    assert call == {"action": "platform_query_data", "params": {"question": QUESTION}}


def test_a_question_sent_is_never_replaced_by_query():
    from modules.tools.execution.schema_reads import keys_read

    assert keys_read("platform_query_data", {"question": QUESTION, "query": "other"})["question"] == QUESTION
    assert keys_read("platform_query_graph", {"query": QUESTION}) == {"query": QUESTION}   # only where it is named


def test_the_schema_says_both_names():
    description = _action("platform_query_data").parameters["properties"]["question"]["description"]
    assert "'query'" in description and _action("platform_query_data").parameters["required"] == ["question"]


def test_a_genuinely_missing_question_is_refused_naming_the_key():
    from modules.tools.discovery.handlers_scheduling import MISSING_QUESTION, query_data
    from modules.tools.execution.direct_contract import missing_on_a_direct_call

    out = asyncio.run(query_data(None, None, {"database_id": "sales_db"}))
    assert out == {"success": False, "error": MISSING_QUESTION}
    assert "question" in out["error"] and "{\"question\":" in out["error"]
    refusal = missing_on_a_direct_call("platform_query_data", _action("platform_query_data"), {})
    assert "['question']" in refusal and 'platform_query_data({"question": "<string>"})' in refusal


# ── JSON text for an object or array ────────────────────────────────────────

POST = {"variables": {"headline": "Guji is back"}, "copy": {"base": "Our Guji is back for October."},
        "media": {"9:16": ["d-1"]}, "sources": [{"claim": "headline", "kind": "url", "ref": "https://x.test"}],
        "footage": {"prompt": "Steam over a cup"}}


def test_a_social_posts_objects_sent_as_json_text_are_the_objects():
    sent = {"title": "Guji is back", **{key: json.dumps(value) for key, value in POST.items()}}
    name, params = _executed("platform_create_social_post", sent)
    assert name == "platform_create_social_post" and params == {"title": "Guji is back", **POST}
    assert all(isinstance(value, str) for key, value in sent.items())       # the caller's call is not changed


def test_twice_encoded_text_and_the_dispatchers_params_are_read_too():
    twice = json.dumps(json.dumps(POST["variables"]))
    _, params = _executed("platform_create_social_post", {"variables": twice})
    assert params == {"variables": POST["variables"]}
    _, call = _executed("platform_execute", {"action": "platform_create_social_post",
                                             "params": json.dumps({"copy": json.dumps(POST["copy"])})})
    assert call["params"] == {"copy": POST["copy"]}


def test_text_that_holds_no_object_stays_text_for_the_strict_check():
    from modules.tools.execution.schema_reads import containers_decoded

    action = _action("platform_create_social_post")
    sent = {"title": '{"not": "a container field"}', "variables": "headline: Guji", "copy": '"just text"'}
    assert containers_decoded(action, sent) == sent


def test_every_promoted_tools_object_and_array_parameters_are_read():
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.schema_reads import _takes_a_container, containers_decoded

    checked = 0
    for action in (a for a in get_action_registry().get_all() if a.promoted):
        props = (action.parameters or {}).get("properties") or {}
        for key, prop in props.items():
            if _takes_a_container(prop):
                value = [1, 2] if "array" in json.dumps(prop.get("type")) else {"k": "v"}
                assert containers_decoded(action, {key: json.dumps(value)}) == {key: value}, (action.name, key)
                checked += 1
    assert checked >= 5


def test_an_action_that_is_not_promoted_keeps_its_own_reading():
    from types import SimpleNamespace as NS

    from modules.tools.execution.schema_reads import containers_decoded

    action = NS(name="platform_x", promoted=False, parameters={"properties": {"data": {"type": "object"}}})
    assert containers_decoded(action, {"data": '{"a": 1}'}) == {"data": '{"a": 1}'}
    either = NS(name="platform_y", promoted=True, parameters={"properties": {"kit": {"type": ["object", "string"]}}})
    assert containers_decoded(either, {"kit": '{"a": 1}'}) == {"kit": '{"a": 1}'}   # text is one of its types


# ── store_memory: source_type has a default ─────────────────────────────────

def test_source_type_is_optional_and_set_by_who_drives_the_turn():
    from modules.tools.discovery.memory_source import source_type_read

    assert source_type_read({"content": "Deploy day is Thursday", "_driving_user_id": OWNER})[0][
        "source_type"] == "platform_verified"
    assert source_type_read({"content": "Deploy day is Thursday"})[0]["source_type"] == "claude_reports"
    assert source_type_read({"content": "x", "source_type": "inference"}) == (
        {"content": "x", "source_type": "inference"}, "")


def test_a_source_type_outside_the_four_is_stored_as_the_default_never_refused():
    from modules.tools.discovery.memory_source import defaults_the_source_type

    seen = {}

    async def store(db, workspace_id, params):
        seen.update(params)
        return {"success": True, "message": "Stored in memory"}

    out = asyncio.run(defaults_the_source_type(store)(None, None, {"content": "x", "source_type": "owner_said",
                                                                    "_driving_user_id": OWNER}))
    assert seen["source_type"] == "platform_verified" and out["success"] is True
    assert "owner_said" in out["note"] and "stored as platform_verified" in out["note"]


def test_store_memory_carries_the_driver_and_its_schema_says_the_default():
    from modules.tools.discovery.handlers_workspace import store_memory
    from modules.tools.discovery.platform_executor import _DRIVER_AWARE_ACTIONS

    action = _action("platform_store_memory")
    prop = action.parameters["properties"]["source_type"]
    assert "source_type" not in action.parameters["required"]
    assert "platform_verified" in prop["description"] and "claude_reports" in prop["description"]
    assert "Never ask the user" in prop["description"] and "never ask the user" in action.description
    assert "platform_store_memory" in _DRIVER_AWARE_ACTIONS        # _driving_user_id is server-injected
    assert store_memory.__wrapped__ is not None                    # the handler is the wrapped one


# ── a re-brief with a status ────────────────────────────────────────────────

def test_a_rebrief_with_a_status_drops_the_status_and_says_so():
    from modules.tools.discovery.ticket_edit_moves import STATUS_IGNORED, rebriefed_without_its_status

    seen = {}

    async def edit(db, workspace_id, params):
        seen.update(params)
        return {"success": True, "task_id": 5, "message": "#0005 has the new brief."}

    out = asyncio.run(rebriefed_without_its_status(edit, None, None, {"task_id": 5, "description": "Redo",
                                                                     "status": "in_progress", "note": "Urgent"}))
    assert seen == {"task_id": 5, "description": "Redo", "note": "Urgent"}
    assert out["success"] is True and out["status_ignored"] is True and out["sent_back"] is True
    assert out["message"] == f"#0005 has the new brief. {STATUS_IGNORED}"


def test_the_receipt_says_the_status_was_ignored_and_no_move_is_recorded():
    from consumers.chatbot.receipts import receipt
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

    params = {"task_id": 5, "description": "Redo with the September sheet", "status": "in_progress"}
    result = {"success": True, "task_id": 5, "sent_back": True, "status_ignored": True}

    effect = receipt("platform_update_task", params, result)["effect"]
    assert "status ignored: a re-brief sends the card back by itself" in effect and "started" not in effect
    tracker = ToolExecutionTracker()
    tracker.record_outcome("platform_update_task", params, result)
    assert "platform_update_task_status:in_progress" not in tracker.succeeded
    assert "send_back" in tracker.succeeded


def test_a_move_with_no_rebrief_is_still_recorded_as_the_move():
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

    tracker = ToolExecutionTracker()
    tracker.record_outcome("platform_update_task", {"task_id": 5, "status": "done"}, {"success": True})
    assert "platform_update_task_status:done" in tracker.succeeded


def test_night_12s_urgent_correction_re_briefs_the_card(in_review, notes_in_this_session):
    """A440: a new brief and a status on a card in Review: re-briefed, back with its agent."""
    out = asyncio.run(in_review.handlers.update_board_task(in_review.db, in_review.ws, {
        "task_id": in_review.number, "description": f241.NEW_BRIEF, "status": "in_progress",
        "_user_id": "user_owner"}))

    assert out["success"] is True and out["status_ignored"] is True
    in_review.db.refresh(in_review.card)
    assert (in_review.card.status, in_review.card.description) == ("assigned", f241.NEW_BRIEF)


@pytest.mark.parametrize("status", ["done", "cancelled"])
def test_a_card_nobody_has_worked_on_still_moves(shop, status, monkeypatch):
    """No re-brief, no drop: an unworked card's description is a plain edit, then the move."""
    from modules.tools.discovery import handlers_board_task_done
    from modules.tools.discovery.ticket_edit_moves import edited_then_moved

    moved = {}

    async def move(db, workspace_id, params):
        moved.update(params)
        return {"success": True, "task_id": shop.card.id, "status": params["status"]}

    async def edit(db, workspace_id, params):
        return {"success": True, "task_id": shop.card.id, "updated": {"description": "changed"}}

    monkeypatch.setattr(handlers_board_task_done, "update_board_task_status", move)
    out = asyncio.run(edited_then_moved(edit, shop.db, shop.ws, {"task_id": shop.card.id, "description": "New",
                                                                 "status": status}))
    assert out["success"] is True and "status_ignored" not in out and moved["status"] == status

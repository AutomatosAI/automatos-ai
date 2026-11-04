"""F105 (wave 2): a platform action's own gates read the database off the event loop.

Wave 1 moved ``PlatformActionExecutor.clear`` to the threadpool. The rest of a call still
read the session on the loop: ``answers_the_question_first`` and ``follows_the_owner`` read
the owner's words in a person's chat; ``asks_the_owner_first`` read the board and the grants
for an owner-only call; and ``_run_cleared`` ran the hierarchy check (``can_actor_modify``),
the team scope's running-mission lookup, the person a mission action is made for, the
caller's admin role for ``platform_list_tools`` and a failed handler's rollback.

Each now runs in FastAPI's threadpool while the call waits for it; the handler still runs on
the loop. ``_run_cleared`` was split verbatim (every string, call, condition, returned dict
and assignment of the old body is in the new helpers): what each gate decides, and in what
order, is unchanged.

DB-free: each read is patched to record the thread it runs on.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from modules.tools.discovery import follows_the_owner as fto
from modules.tools.discovery import owner_only, question_turns
from modules.tools.discovery import platform_executor as pe

WS = uuid4()
WRITE = NS(name="platform_update_agent", permission_level="write", requires_confirmation=False)
READ = NS(name="platform_query_graph", permission_level="read", requires_confirmation=False)


class _Db:
    """The executor's session: a running mission for the team scope, and a rollback."""

    def __init__(self, seen):
        self.seen = seen

    def query(self, *_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        self.seen.append(("running mission read", threading.get_ident()))
        return NS(config={"agent_id": 12})

    def rollback(self):
        self.seen.append(("rolled back", threading.get_ident()))


@pytest.fixture
def seen(monkeypatch):
    seen = []
    monkeypatch.setattr("core.security.rate_limiter.check_rate_limit", AsyncMock(return_value=None))
    monkeypatch.setattr(owner_only, "as_the_owner_said", _recorded(seen, "owner's words read", lambda *a: a[3]))
    return seen


def _recorded(seen, name, answer):
    def step(*args, **kwargs):
        seen.append((name, threading.get_ident()))
        return answer(*args, **kwargs)
    return step


def _handler(seen, fails=False):
    async def handler(db, workspace_id, params):
        seen.append(("handler", threading.get_ident()))
        if fails:
            raise RuntimeError("the handler broke")
        return {"success": True, "params": params}
    return handler


def _run(seen, action, params, action_def, caller_context=None, handler=None):
    """(the loop's thread, the answer) of ``_run_cleared`` for a call past ``clear``."""
    cleared = pe.Cleared(action_def, False, None, False)

    async def run():
        executor = pe.PlatformActionExecutor(_Db(seen), WS)
        answer = await executor._run_cleared(action, params, caller_context, cleared, handler or _handler(seen))
        return threading.get_ident(), answer

    return asyncio.run(run())


def _off_the_loop(seen, loop, *names):
    where = dict(seen)
    return all(where[name] != loop for name in names)


def test_the_hierarchy_refusal_is_read_off_the_loop_and_unchanged(seen, monkeypatch):
    decision = NS(allowed=False, reason="not_in_chain", escalation_target="auto", message=None, actor_name="Scout")
    monkeypatch.setattr(pe, "can_actor_modify", _recorded(seen, "hierarchy read", lambda db, **kw: decision))

    loop, answer = _run(seen, "platform_update_agent", {"agent_id": 9, "_agent_id": 4}, WRITE)

    assert answer == {
        "success": False, "permission_denied": True, "reason": "not_in_chain", "escalation_target": "auto",
        "error": "Action 'platform_update_agent' denied — not_in_chain. Route this through the auto for arbitration.",
    }
    assert [name for name, _ in seen] == ["owner's words read", "hierarchy read"]
    assert _off_the_loop(seen, loop, "owner's words read", "hierarchy read")


def test_every_target_of_a_bulk_call_is_checked_and_one_refusal_denies_it(seen, monkeypatch):
    asked = []

    def decide(db, **kwargs):
        asked.append((kwargs["target_id"], threading.get_ident()))
        return NS(allowed=kwargs["target_id"] == 1, reason="not_your_card", escalation_target=None,
                  message="Ask the card's owner.", actor_name="Scout")

    monkeypatch.setattr(pe, "can_actor_modify", decide)
    action = NS(name="platform_update_task_status", permission_level="write", requires_confirmation=False)

    loop, answer = _run(seen, "platform_update_task_status", {"task_ids": [1, 2], "_agent_id": 4}, action)

    assert answer["error"] == "Action 'platform_update_task_status' denied (not_your_card). Ask the card's owner."
    assert [target for target, _ in asked] == [1, 2] and all(thread != loop for _, thread in asked)
    assert "handler" not in dict(seen)


def test_a_hierarchy_check_that_fails_still_denies(seen, monkeypatch):
    def broken(db, **kwargs):
        raise RuntimeError("the pool is exhausted")

    monkeypatch.setattr(pe, "can_actor_modify", broken)

    _, answer = _run(seen, "platform_update_agent", {"agent_id": 9}, WRITE)

    assert (answer["permission_denied"], answer["reason"], answer["escalation_target"]) == (
        True, "permission_check_failed", "auto")


def test_the_server_params_are_read_off_the_loop_and_the_handler_runs_on_it(seen, monkeypatch):
    loop, answer = _run(seen, "platform_query_graph", {"query": "suppliers", "_turn_id": "spoofed"}, READ,
                        caller_context={"turn_id": "turn-105"})

    assert answer["params"] == {"query": "suppliers", "_agent_id": 12, "_turn_id": "turn-105"}
    assert [name for name, _ in seen] == ["owner's words read", "running mission read", "handler"]
    assert _off_the_loop(seen, loop, "running mission read") and dict(seen)["handler"] == loop


def test_a_mission_action_finds_its_person_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(pe, "_person_for_mission", _recorded(seen, "person read", lambda db, ctx: "owner@cafe.test"))
    action = NS(name="platform_create_mission", permission_level="write", requires_confirmation=False)

    loop, answer = _run(seen, "platform_create_mission", {"goal": "Restock", "_created_by": "spoofed"}, action)

    assert answer["params"] == {"goal": "Restock", "_created_by": "owner@cafe.test"}
    assert _off_the_loop(seen, loop, "person read")


def test_the_tool_list_reads_the_callers_role_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(pe.PlatformActionExecutor, "_caller_is_admin",
                        _recorded(seen, "admin role read", lambda self, ctx, full: True))
    action = NS(name="platform_list_tools", permission_level="read", requires_confirmation=False)

    loop, answer = _run(seen, "platform_list_tools", {"_caller_is_admin": False}, action)

    assert answer["params"] == {"_caller_is_super_admin": False, "_caller_is_admin": True}
    assert _off_the_loop(seen, loop, "admin role read")


def test_a_failed_handler_is_rolled_back_off_the_loop_with_the_same_answer(seen):
    loop, answer = _run(seen, "platform_query_graph", {"_agent_id": 3}, READ, handler=_handler(seen, fails=True))

    assert answer == {"success": False, "error": "Action 'platform_query_graph' failed"}
    assert dict(seen)["handler"] == loop and _off_the_loop(seen, loop, "rolled back")


def _execute(seen, monkeypatch, action, caller_context):
    """``execute`` with its gates and handler stubbed: (the loop's thread, the answer)."""
    monkeypatch.setattr(pe.PlatformActionExecutor, "clear", lambda self, *a: pe.Cleared(None, False, None, False))
    monkeypatch.setattr(pe.PlatformActionExecutor, "_run_cleared",
                        lambda self, *a: _handler(seen)(None, WS, {}))
    monkeypatch.setattr("modules.tools.execution.tool_grants.give_back_unused", lambda db, grant_id, result: False)

    async def run():
        answer = await pe.PlatformActionExecutor(_Db(seen), WS).execute(action, {}, caller_context=caller_context)
        return threading.get_ident(), answer

    return asyncio.run(run())


def test_a_question_turn_and_the_owners_words_are_read_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(question_turns, "refusal_on_a_question",
                        _recorded(seen, "question read", lambda *a: None))
    monkeypatch.setattr(fto, "refusal_for", _recorded(seen, "words read", lambda *a: "the owner asked for a post"))

    loop, answer = _execute(seen, monkeypatch, "platform_create_agent", {"conversation_id": "chat-1"})

    assert answer == fto.refused("the owner asked for a post")
    assert [name for name, _ in seen] == ["question read", "words read"]
    assert _off_the_loop(seen, loop, "question read", "words read")


def test_a_how_to_question_still_refuses_the_change(seen, monkeypatch):
    monkeypatch.setattr(question_turns, "refusal_on_a_question",
                        _recorded(seen, "question read", lambda *a: "Answer it in words."))

    loop, answer = _execute(seen, monkeypatch, "platform_create_agent", {"conversation_id": "chat-1"})

    assert answer == {"success": False, "error": "Answer it in words."}
    assert _off_the_loop(seen, loop, "question read")


OWNER = {"driving_user_id": "5", "conversation_id": "chat-1"}


def _owner_only(seen, monkeypatch, grant):
    monkeypatch.setattr(owner_only, "is_owner_only", _recorded(seen, "board read", lambda *a, **k: True))
    monkeypatch.setattr(owner_only, "_the_click", _recorded(seen, "click read", lambda *a: grant))
    monkeypatch.setattr(owner_only, "after_the_click",
                        _recorded(seen, "click kept", lambda db, g, result: {**result, "approved_via_grant_id": g.id}))
    monkeypatch.setattr("modules.tools.discovery.mission_create_checks.refused_before_the_mission_card",
                        lambda *a: None)
    monkeypatch.setattr(owner_only, "platform_ask", _recorded(seen, "ask raised", lambda *a: {"requires_confirmation": True}))
    monkeypatch.setattr(pe, "can_actor_modify", lambda db, **kw: NS(allowed=True))


def test_an_owner_only_call_runs_on_the_click_its_reads_off_the_loop(seen, monkeypatch):
    _owner_only(seen, monkeypatch, NS(id=77, granted_by="user:5"))

    loop, answer = _run(seen, "platform_update_agent", {"agent_id": 9}, WRITE, caller_context=OWNER)

    assert answer["approved_via_grant_id"] == 77
    assert answer["params"][owner_only.CLICKED_BY] == "user:5"
    assert [name for name, _ in seen] == ["owner's words read", "board read", "click read", "handler", "click kept"]
    assert _off_the_loop(seen, loop, "owner's words read", "board read", "click read", "click kept")
    assert dict(seen)["handler"] == loop


def test_an_owner_only_call_without_a_click_is_asked_off_the_loop(seen, monkeypatch):
    _owner_only(seen, monkeypatch, None)

    loop, answer = _run(seen, "platform_update_agent", {"agent_id": 9}, WRITE, caller_context=OWNER)

    assert answer == {"requires_confirmation": True}
    assert "handler" not in dict(seen) and _off_the_loop(seen, loop, "click read", "ask raised")

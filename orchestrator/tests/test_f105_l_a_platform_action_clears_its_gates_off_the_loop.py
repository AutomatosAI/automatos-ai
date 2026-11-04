"""F105 (4 Nov): a platform action's synchronous database work runs off the event loop.

The loop watchdog caught a 2.7 s stall inside ``_execute_platform_action``. Every
platform action read the calling agent's name (exec_platform) and cleared its gates
(``PlatformActionExecutor.clear``: the autonomy dial, the admin and owner checks, the
grant it consults or issues) on the session, synchronously, on the event loop; a yes that
did nothing was given back the same way (``give_back_unused``). Each now runs in a worker
thread while the action waits for it, so the session is used by one thread at a time. The
handler still runs on the loop. What each step decides is unchanged.

DB-free: the executor's steps are patched to record where they run.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from uuid import uuid4

from modules.tools.discovery import platform_executor as pe
from modules.tools.execution import exec_platform, tool_grants

WS = uuid4()
ACTION = "platform_list_agents"
NOTHING_DONE = {"success": False, "error": "nothing to do"}


def _patch_the_steps(monkeypatch, seen, cleared):
    def clear(self, action_name, params, caller_context):
        seen.append(("clear", threading.get_ident()))
        return cleared

    async def run_cleared(self, action_name, params, caller_context, cleared, handler):
        seen.append(("handler", threading.get_ident()))
        return dict(NOTHING_DONE)

    def give_back(db, grant_id, result):
        seen.append(("give back", threading.get_ident()))
        return grant_id is not None and result.get("success") is False

    monkeypatch.setattr(pe.PlatformActionExecutor, "clear", clear)
    monkeypatch.setattr(pe.PlatformActionExecutor, "_run_cleared", run_cleared)
    monkeypatch.setattr(tool_grants, "give_back_unused", give_back)


def _execute():
    async def run():
        result = await pe.PlatformActionExecutor(None, WS).execute(ACTION, {}, caller_context=None)
        return threading.get_ident(), result

    return asyncio.run(run())


def test_the_gates_and_the_give_back_run_off_the_loop_and_the_handler_on_it(monkeypatch):
    seen = []
    _patch_the_steps(monkeypatch, seen, pe.Cleared(None, False, 41, False))

    loop_thread, result = _execute()

    assert result == NOTHING_DONE
    assert [step for step, _ in seen] == ["clear", "handler", "give back"]
    where = dict(seen)
    assert where["clear"] != loop_thread and where["give back"] != loop_thread
    assert where["handler"] == loop_thread


def test_a_refusal_at_the_gates_still_stops_the_call(monkeypatch):
    seen = []
    refusal = {"success": False, "permission_denied": True, "error": "requires workspace admin or owner role."}
    _patch_the_steps(monkeypatch, seen, refusal)

    _, result = _execute()

    assert result == refusal
    assert [step for step, _ in seen] == ["clear"]


class _Agents:
    """The executor's session: one agent row, read on whichever thread asks."""

    def __init__(self, seen, name="Brand Designer", fails=False):
        self._seen, self._name, self._fails = seen, name, fails

    def query(self, *_):
        return self

    def filter(self, *_):
        return self

    def first(self):
        self._seen.append(("agent read", threading.get_ident()))
        if self._fails:
            raise RuntimeError("the pool is exhausted")
        return NS(name=self._name)


def _call_with_agent(monkeypatch, db):
    handed = []

    async def execute(self, action_name, params, caller_context=None):
        handed.append(params)
        return {"success": True}

    monkeypatch.setattr(pe.PlatformActionExecutor, "execute", execute)

    async def run():
        await exec_platform.execute_platform_action(
            NS(db=db), ACTION, {"_agent_id": 1, "_agent_name": "Auto", "limit": 5}, workspace_id=WS, agent_id=7,
        )
        return threading.get_ident()

    return asyncio.run(run()), handed


def test_the_calling_agents_name_is_read_off_the_loop(monkeypatch):
    seen = []

    loop_thread, handed = _call_with_agent(monkeypatch, _Agents(seen))

    assert handed == [{"limit": 5, "_agent_id": 7, "_agent_name": "Brand Designer"}]
    assert [step for step, _ in seen] == ["agent read"] and seen[0][1] != loop_thread


def test_a_failed_name_lookup_still_runs_the_action_without_a_name(monkeypatch):
    seen = []

    _, handed = _call_with_agent(monkeypatch, _Agents(seen, fails=True))

    assert handed == [{"limit": 5, "_agent_id": 7}]

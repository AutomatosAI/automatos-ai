"""F155 — work a widget turn starts keeps the turn's restrictions.

A mission's tasks run later, on the coordinator tick, outside the widget turn
and so outside the restrictions it carries (the key's scopes, its team lock,
never admin, never autonomous). A mission started on a widget turn now carries
its origin on its config, server-set (core.security.surface.stamp_origin): the
surface, the key's scopes and its team lock. Its approval and each task run
under it again (origin_surface), and a task assigned to a Claude Code session,
which those restrictions cannot reach, is refused. A widget turn cannot set the
mission's cost ceiling.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

WIDGET_ORIGIN = {"origin_surface": "widget", "origin_scopes": ["chat", "documents:read"],
                 "origin_team": "franchise-a"}


def _widget_turn():
    from core.security.surface import WIDGET, turn_surface

    return turn_surface(WIDGET, ("chat", "documents:read"), "Franchise-A")


def test_the_origin_is_the_turns_and_never_the_callers():
    from core.security.surface import stamp_origin

    forged = {"origin_surface": "dashboard", "origin_scopes": ["admin"], "origin_team": None, "goal": "g"}
    with _widget_turn():
        assert stamp_origin(forged) == {"goal": "g", **WIDGET_ORIGIN}
    assert stamp_origin({**WIDGET_ORIGIN, "goal": "g"}) == {"goal": "g"}


def test_a_widget_turns_mission_records_its_origin_and_sets_no_cost_ceiling(monkeypatch):
    from modules.tools.discovery import handlers_missions as missions

    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    run = NS(id=uuid4(), goal="g", state="awaiting_approval", plan={"tasks": []}, config={})

    def create(in_widget_turn):
        coordinator = NS(create_mission=AsyncMock(return_value=run))
        params = {"goal": "g", "config": {"cost_ceiling": 1_000_000, "async_planning": True}}
        with patch("services.coordinator_service.CoordinatorService", return_value=coordinator):
            if in_widget_turn:
                with _widget_turn():
                    asyncio.run(missions.create_mission(MagicMock(), uuid4(), params))
            else:
                asyncio.run(missions.create_mission(MagicMock(), uuid4(), params))
        return coordinator.create_mission.call_args.kwargs["config"]

    widget = create(True)
    assert {key: widget.get(key) for key in WIDGET_ORIGIN} == WIDGET_ORIGIN
    assert "cost_ceiling" not in widget
    assert create(False)["cost_ceiling"] == 1_000_000


def _observed_turn():
    from core.security.surface import widget_scopes, widget_team, widget_turn

    return widget_turn(), widget_scopes(), widget_team()


def test_a_widget_born_missions_task_runs_under_its_keys_restrictions():
    from services.coordinator_service import CoordinatorService

    seen = []

    async def _execute(**kwargs):
        seen.append(_observed_turn())
        return {"status": "success"}

    factory = NS(execute_with_prompt=_execute)
    task = NS(id=uuid4())
    service = CoordinatorService()
    asyncio.run(service._run_agent_io(factory, 7, "do it", task, [], origin=WIDGET_ORIGIN))
    asyncio.run(service._run_agent_io(factory, 7, "do it", task, [], origin={}))
    assert seen == [(True, frozenset({"chat", "documents:read"}), "franchise-a"), (False, frozenset(), None)]


def test_a_widget_born_missions_task_never_runs_on_a_claude_code_session():
    from services.coordinator_service import WIDGET_SESSION_REFUSAL, CoordinatorService

    service = CoordinatorService()
    service._run_cli_ticket = AsyncMock(return_value={"status": "success"})
    prepared = {"cli_agent": True, "task": NS(id=uuid4()), "prompt": "p", "agent_id": 7, "workspace_id": uuid4(),
                "run_id": uuid4(), "mode_caps": {}}
    assert asyncio.run(service._task_io({**prepared, "origin": WIDGET_ORIGIN})) == {
        "status": "error", "error": WIDGET_SESSION_REFUSAL}
    service._run_cli_ticket.assert_not_awaited()
    assert asyncio.run(service._task_io({**prepared, "origin": {}})) == {"status": "success"}

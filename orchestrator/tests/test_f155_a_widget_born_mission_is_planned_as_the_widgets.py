"""F155 — a widget-born mission is planned under its widget key's restrictions.

Only the approval of a mission planned on the tick ran under its widget
origin. The planner itself, and a later replan, did not. The planning context
pack took no team, so its knowledge search (whose chunks seed the tasks'
titles and descriptions) read every team's documents. It also recalled the
workspace's mission history and field memory from the owner's missions. Both
the plan and the replan now run under the origin, and the pack follows the
key: its team lock scopes the knowledge search, and no history or field
memory is recalled on a widget turn.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

WIDGET_ORIGIN = {"origin_surface": "widget", "origin_scopes": ["chat", "documents:read"],
                 "origin_team": "franchise-a"}
WIDGET_TURN = (True, frozenset({"chat", "documents:read"}), "franchise-a")
NO_TURN = (False, frozenset(), None)


def _observed_turn():
    from core.security.surface import widget_scopes, widget_team, widget_turn

    return widget_turn(), widget_scopes(), widget_team()


def test_a_widget_born_mission_is_planned_under_its_origin(db_session, seed_workspace, monkeypatch):
    import services.coordinator_service as coordinator
    from modules.coordination.agent_matcher import AgentMatcher
    from modules.coordination.planner import MissionPlanner

    seen = []

    async def _decompose(**kwargs):
        seen.append(_observed_turn())
        return NS(tasks=[], token_estimate=1000)

    monkeypatch.setattr(MissionPlanner, "decompose", _decompose)
    monkeypatch.setattr(AgentMatcher, "compute_signals_for_tasks", AsyncMock(return_value={}))
    monkeypatch.setattr("services.daily_spend_guard.refuse_new_work", lambda *a, **k: None)
    for name in ("create_mission_board_task", "emit_event", "transition_run"):
        monkeypatch.setattr(coordinator, name, lambda *a, **k: None)
    monkeypatch.setattr(coordinator, "_dispatch_mission_event", AsyncMock())
    service = coordinator.CoordinatorService()
    service._persist_decomposition = lambda *a, **k: {}
    service._annotate_match_previews = lambda *a, **k: None
    service._queue_initial_tasks = lambda *a, **k: None
    service._create_mission_field = AsyncMock()
    ws = UUID(seed_workspace())
    for config in ({"async_planning": True, **WIDGET_ORIGIN}, {"async_planning": True}):
        asyncio.run(service._run_planning(db_session, NS(id=uuid4(), workspace_id=ws, goal="g", config=config,
                                                         plan=None)))
    assert seen == [WIDGET_TURN, NO_TURN]


def test_a_widget_born_mission_is_replanned_under_its_origin(db_session, seed_workspace, monkeypatch):
    import services.coordinator_service as coordinator
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import RunState
    from modules.coordination.planner import MissionPlanner, PlanValidationError

    seen = []

    async def _replan(**kwargs):
        seen.append(_observed_turn())
        raise PlanValidationError("stop here")

    monkeypatch.setattr(MissionPlanner, "replan", _replan)
    monkeypatch.setattr(coordinator, "transition_run", lambda *a, **k: None)
    ws = UUID(seed_workspace())
    for config in (dict(WIDGET_ORIGIN), {}):
        run = OrchestrationRun(workspace_id=ws, goal="Draft the reply", state=RunState.FAILED.value,
                               created_by="user_test", config=config)
        db_session.add(run)
        db_session.flush()
        with pytest.raises(PlanValidationError):
            asyncio.run(coordinator.CoordinatorService().replan_mission(db_session, run.id, "user_test"))
    assert seen == [WIDGET_TURN, NO_TURN]


def _pack(monkeypatch, mock_db, in_widget_turn):
    """Build the MissionPlanner's pack; return the team its knowledge search
    used and which history sources it recalled."""
    from core.security.surface import WIDGET, turn_surface
    from modules.context.sections.field_memory import FieldMemorySection
    from modules.context.service import ContextService

    searched, recalled = [], []

    async def _retrieve(**kwargs):
        searched.append(kwargs.get("team"))

    async def _history(**kwargs):
        recalled.append("history")
        return []

    async def _field(workspace_id, query):
        recalled.append("field")
        return []

    monkeypatch.setattr("modules.rag.service.get_rag_service", lambda: NS(retrieve=_retrieve))
    monkeypatch.setattr("modules.memory.unified_memory_service.get_unified_memory_service",
                        lambda: NS(search_short_term=_history, search_long_term=_history))
    monkeypatch.setattr(FieldMemorySection, "_query_workspace_field", staticmethod(_field))

    def build():
        asyncio.run(ContextService(mock_db).build_planning_context(goal="Plan the franchise launch",
                                                                   workspace_id=str(uuid4()), include_roster=False))

    if in_widget_turn:
        with turn_surface(WIDGET, ("chat", "documents:read"), "Franchise-A"):
            build()
    else:
        build()
    return searched, sorted(set(recalled))


def test_the_planning_pack_follows_the_widget_key(monkeypatch, mock_db):
    assert _pack(monkeypatch, mock_db, in_widget_turn=True) == (["franchise-a"], [])
    assert _pack(monkeypatch, mock_db, in_widget_turn=False) == ([None], ["field", "history"])

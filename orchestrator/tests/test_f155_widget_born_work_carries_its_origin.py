"""F155 — work a widget turn starts keeps the turn's restrictions.

A mission's tasks and a playbook's steps run later, outside the widget turn
(the coordinator tick, the playbook engine's task) and so outside the
restrictions it carries (the key's scopes, its team lock, never admin, never
autonomous). A mission or playbook run started on a widget turn now carries its
origin, server-set (core.security.surface.stamp_origin): the surface, the key's
scopes and its team lock, on the mission's config or the run's metadata. The
mission's approval, each of its tasks and each of the run's steps run under it
again (origin_surface), and so do a retried or rerun playbook run. Work assigned
to a Claude Code session, which those restrictions cannot reach, is refused; a
playbook step is offered only what the key's scopes allow and none of the
owner's connected apps. A widget turn cannot set the mission's cost ceiling.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock
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


def test_a_widget_turns_mission_records_its_origin_and_sets_no_cost_ceiling(monkeypatch, mock_db):
    from modules.tools.discovery import handlers_missions as missions
    from services.coordinator_service import CoordinatorService

    monkeypatch.setattr(missions, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    created, create = [], CoordinatorService.create_mission

    async def _recorded(self, **kwargs):
        created.append(await create(self, **kwargs))
        return created[-1]

    monkeypatch.setattr(CoordinatorService, "create_mission", _recorded)
    params = {"goal": "g", "config": {"cost_ceiling": 1_000_000, "async_planning": True}}
    with _widget_turn():
        asyncio.run(missions.create_mission(mock_db, uuid4(), params))
    asyncio.run(missions.create_mission(mock_db, uuid4(), params))
    widget, owner = (run.config for run in created)
    assert {key: widget.get(key) for key in WIDGET_ORIGIN} == WIDGET_ORIGIN
    assert "cost_ceiling" not in widget
    assert owner["cost_ceiling"] == 1_000_000 and "origin_surface" not in owner


FORGED_ORIGIN = {"origin_surface": "widget", "origin_scopes": ["chat"], "origin_team": "franchise-b"}


def test_every_creation_path_drops_a_callers_origin_and_stamps_a_widget_turns(mock_db, db_session, seed_workspace):
    """REST (POST /api/missions, /import-plan) reaches the coordinator without the
    tool handler: a forged origin_team would re-scope the mission's document reads
    to another team."""
    from uuid import UUID

    from services.coordinator_service import CoordinatorService

    coordinator = CoordinatorService()
    run = asyncio.run(coordinator.create_mission(db=mock_db, workspace_id=uuid4(), goal="g", created_by="user_test",
                                                 config={"async_planning": True, **FORGED_ORIGIN}))
    assert run.config == {"async_planning": True}
    imported = coordinator.import_plan(db=db_session, workspace_id=UUID(seed_workspace()), goal="g",
                                       plan={"tasks": [{"title": "Draft the letter"}]}, created_by="user_test",
                                       config=dict(FORGED_ORIGIN))
    assert imported.config == {"imported_plan": True}

    owners = {"async_planning": True, "cost_ceiling": 1_000_000, "auto_approve": True}
    with _widget_turn():
        run = asyncio.run(coordinator.create_mission(db=mock_db, workspace_id=uuid4(), goal="g",
                                                     created_by="widget", config={**owners, **FORGED_ORIGIN}))
    assert run.config == {"async_planning": True, **WIDGET_ORIGIN}


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


WIDGET_TURN = (True, frozenset({"chat", "documents:read"}), "franchise-a")
NO_TURN = (False, frozenset(), None)


def test_a_widget_turns_playbook_run_records_its_origin(monkeypatch):
    from modules.tools.discovery import handlers_playbooks as playbooks

    added = []
    db = MagicMock()
    db.add.side_effect = added.append
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = NS(id=79, name="Visitor reply")
    monkeypatch.setattr("services.concurrency_guard.check_concurrency", AsyncMock(return_value=NS(allowed=True)))
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    monkeypatch.setattr("services.playbook_engine.get_playbook_engine", lambda: NS(launch=MagicMock()))

    with _widget_turn():
        asyncio.run(playbooks.execute_playbook(db, uuid4(), {"playbook_id": 79}))
    asyncio.run(playbooks.execute_playbook(db, uuid4(), {"playbook_id": 79}))
    assert [row.execution_metadata for row in added] == [WIDGET_ORIGIN, None]


def _run_steps(monkeypatch, execution_metadata):
    from tests.helpers_playbook_run import done, run_playbook

    steps, after = [], []
    run_playbook(monkeypatch, outcomes=[done("Listed."), done("Sent.")], step_seconds=1, exec_config={},
                 execution_metadata=execution_metadata, on_step=lambda _kwargs: steps.append(_observed_turn()),
                 after_run=lambda: after.append(_observed_turn()))
    return steps, after


def test_a_widget_born_playbook_runs_steps_under_its_keys_restrictions(monkeypatch):
    assert _run_steps(monkeypatch, {"execution_type": "recipe_direct", **WIDGET_ORIGIN}) == (
        [WIDGET_TURN, WIDGET_TURN], [NO_TURN])  # and the mark ends with the run
    assert _run_steps(monkeypatch, {}) == ([NO_TURN, NO_TURN], [NO_TURN])


def test_a_widget_born_playbook_step_never_runs_on_a_claude_code_session(monkeypatch):
    from api import recipe_executor as rex
    from services import cli_ticket_lane as lane

    ticket = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(lane, "is_cli_agent", lambda db, agent_id: True)
    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", ticket)
    with _widget_turn():
        refused = asyncio.run(rex._execute_step(MagicMock(), NS(id=7), "Draft the reply.", uuid4(), max_iterations=1))
    assert (refused["status"], refused["error"]) == ("error", rex.WIDGET_SESSION_REFUSAL)
    ticket.assert_not_awaited()
    asyncio.run(rex._execute_step(MagicMock(), NS(id=7), "Draft the reply.", uuid4(), max_iterations=1))
    ticket.assert_awaited_once()


def _schema(name, actions=None):
    parameters = {"type": "object", "properties": {"action": {"type": "string", "enum": actions}}} if actions else {}
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


OFFERED = [
    _schema("search_knowledge"),
    _schema("composio_execute"),
    _schema("workspace_write_file"),
    _schema("platform_execute", ["platform_list_documents", "platform_create_agent", "platform_execute_playbook"]),
]


def _offered_to_a_step(monkeypatch):
    """What the step's model is offered, and which Composio lookups ran."""
    from api import recipe_executor as rex

    offered, composio = [], []

    class _Context:
        def __init__(self, db):
            pass

        async def build_context(self, **kwargs):
            return NS(system_prompt="You are the club secretary.", tools=list(OFFERED))

    class _Composio:
        def __init__(self, db):
            pass

        def get_tools_for_step(self, **kwargs):
            composio.append("tools")

        def build_hints(self, **kwargs):
            composio.append("hints")
            return NS(hint_lines=[], strategy_used="none", matched_actions=[])

    async def _answer(messages, tools):
        offered.append(tools)
        return NS(content="Done.", tool_calls=None, usage={"total_tokens": 5})

    runtime = NS(llm_manager=NS(generate_response=_answer))
    factory = NS(activate_agent=AsyncMock(return_value=runtime))
    monkeypatch.setattr("services.cli_ticket_lane.is_cli_agent", lambda db, agent_id: False)
    monkeypatch.setattr("modules.context.ContextService", _Context)
    monkeypatch.setattr("modules.tools.services.composio_tool_service.ComposioToolService", _Composio)
    monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", _Composio)
    monkeypatch.setattr("modules.agents.factory.agent_factory.AgentFactory", lambda db_session: factory)
    monkeypatch.setattr("modules.tools.tool_router.get_tool_router", lambda: MagicMock())
    asyncio.run(rex._execute_step(MagicMock(), NS(id=7), "Draft the reply.", uuid4(), max_iterations=1))
    return offered[0], composio


def test_a_widget_born_playbook_step_is_offered_only_its_keys_tools(monkeypatch):
    with _widget_turn():
        tools, composio = _offered_to_a_step(monkeypatch)
    assert [schema["function"]["name"] for schema in tools] == ["search_knowledge", "platform_execute"]
    assert tools[1]["function"]["parameters"]["properties"]["action"]["enum"] == ["platform_list_documents"]
    assert composio == []  # the owner's connected apps are never looked up
    tools, composio = _offered_to_a_step(monkeypatch)
    assert tools == OFFERED and composio == ["tools", "hints"]


def test_a_rerun_of_a_widget_born_playbook_run_is_widget_born():
    from core.security.surface import widget_born
    from services.watch_rerun import create_rerun_execution

    recipe = NS(id=79, steps=[{}, {}])

    def rerun(metadata):
        original = NS(execution_id="exec-120", workspace_id=uuid4(), input_data={"visitor": "hi"}, attempt_count=1,
                      execution_metadata=metadata)
        return create_rerun_execution(MagicMock(), recipe, original).execution_metadata

    widget = rerun({"execution_type": "recipe_direct", **WIDGET_ORIGIN})
    assert {key: widget.get(key) for key in WIDGET_ORIGIN} == WIDGET_ORIGIN
    assert not widget_born(rerun({"execution_type": "recipe_direct"}))


def test_a_retry_of_a_widget_born_playbook_run_is_widget_born(monkeypatch):
    from services import task_reconciler as reconciler

    monkeypatch.setattr(reconciler, "_step_tickets", lambda db, execution_id: [])
    monkeypatch.setattr("services.board_task_bridge.complete_recipe_board_task", lambda *a, **k: None)
    row = NS(execution_id="exec-120", recipe_id=79, workspace_id=uuid4(), input_data={}, attempt_count=1,
             execution_metadata={"execution_config": {"max_retries": 3}, **WIDGET_ORIGIN})
    db = MagicMock()
    asyncio.run(reconciler.TaskReconciler()._handle_stalled(row, db, reason="running", timeout=300))
    inserted = [call for call in db.execute.call_args_list if "INSERT INTO recipe_executions" in str(call.args[0])]
    retry = json.loads(inserted[0].args[1]["meta"])
    assert {key: retry.get(key) for key in WIDGET_ORIGIN} == WIDGET_ORIGIN

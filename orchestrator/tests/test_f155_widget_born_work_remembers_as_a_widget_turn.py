"""F155 — work a widget turn starts recalls and stores memory as a widget turn does.

A widget turn recalls only its agent's own memories and stores none (F154).
The work it starts ran later without those limits:
- a mission's tasks and a playbook's steps built their context with the
  workspace's whole memory;
- a playbook run recalled the playbook's learnings from the owner's runs,
  and afterwards stored its own and taught the playbook;
- a mission kept a field in the workspace's shared field memory (which
  planners and heartbeat agents recall workspace-wide), and stored its
  summary and its failed tasks in the memory later missions plan from.
Now any context built under the widget mark is a widget context, and
widget-born work recalls, learns and stores none of that.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

WIDGET_ORIGIN = {"origin_surface": "widget", "origin_scopes": ["chat", "documents:read"],
                 "origin_team": "franchise-a"}


def _widget_turn():
    from core.security.surface import WIDGET, turn_surface

    return turn_surface(WIDGET, ("chat", "documents:read"), "Franchise-A")


def test_a_context_built_for_widget_born_work_is_a_widget_context(monkeypatch, mock_db):
    from modules.context.modes import ContextMode
    from modules.context.sections.memory import MemorySection
    from modules.context.sections.tools import ToolsSection
    from modules.context.service import ContextService

    seen = []

    async def _render(self, ctx):
        seen.append(ctx.widget_mode)
        return ""

    monkeypatch.setattr(MemorySection, "render", _render)
    monkeypatch.setattr(ToolsSection, "load_tools", AsyncMock(return_value=([], "auto")))
    agent = NS(id=42, name="Scribe", agent_type="assistant", description="Drafts replies", use_custom_persona=False,
               custom_persona_prompt=None, persona=None, skills=[], team=None)

    def build():
        asyncio.run(ContextService(mock_db).build_context(mode=ContextMode.TASK_EXECUTION, agent=agent,
                                                          workspace_id=str(uuid4()), task_description="Draft it"))

    with _widget_turn():
        build()
    build()
    assert seen == [True, False]


def test_a_widget_born_playbook_run_recalls_learns_and_stores_nothing(monkeypatch):
    from tests import helpers_playbook_run as helper

    calls = []

    async def _recall(self, **kwargs):
        calls.append("recall")

    async def _store(self, *args, **kwargs):
        calls.append("store")

    class _Learning:
        def __init__(self, db=None):
            pass

        def analyze_execution(self, execution_id):
            calls.append("learn")
            return {}

    monkeypatch.setattr(helper._Memory, "retrieve_relevant_memories", _recall)
    monkeypatch.setattr(helper._Memory, "store_execution_memory", _store)
    monkeypatch.setattr("core.services.playbook_learning_service.PlaybookLearningService", _Learning)

    def run(metadata):
        calls.clear()
        helper.run_playbook(monkeypatch, outcomes=[helper.done("Listed."), helper.done("Sent.")], step_seconds=1,
                            exec_config={"auto_learning": True}, execution_metadata=metadata)
        return list(calls)

    assert run({"execution_type": "recipe_direct", **WIDGET_ORIGIN}) == []
    assert run({}) == ["recall", "learn", "store"]


class _MissionMemory:
    stored: list = []

    def __init__(self, db=None):
        pass

    async def store_mission_summary(self, **kwargs):
        _MissionMemory.stored.append(("summary", kwargs["run_id"]))

    async def store_task_failure(self, task):
        _MissionMemory.stored.append(("failure", task.id))


def test_a_widget_born_missions_summary_is_not_remembered(monkeypatch):
    import services.coordinator_service as coordinator

    monkeypatch.setattr("core.services.mission_memory_service.MissionMemoryService", _MissionMemory)
    _MissionMemory.stored = []
    for run_id, config in (("widget-run", dict(WIDGET_ORIGIN)), ("owner-run", {})):
        db = MagicMock()
        db.get.return_value = NS(config=config)
        asyncio.run(coordinator._store_mission_memory_safe(db, run_id, outcome="completed"))
    assert _MissionMemory.stored == [("summary", "owner-run")]


def test_a_widget_born_missions_failed_task_is_not_remembered(monkeypatch, db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import RunState, TaskState
    from services.coordinator_service import CoordinatorService

    monkeypatch.setattr("core.services.mission_memory_service.MissionMemoryService", _MissionMemory)
    _MissionMemory.stored = []
    failed = {"status": "error", "error": "The agent produced nothing.", "execution": {"tokens_used": 0}}
    tasks = {}
    for name, config in (("widget", dict(WIDGET_ORIGIN)), ("owner", {})):
        run = OrchestrationRun(workspace_id=UUID(seed_workspace()), goal="Draft the reply",
                               state=RunState.RUNNING.value, created_by="user_test", tokens_used=0, config=config)
        db_session.add(run)
        db_session.flush()
        tasks[name] = MagicMock(id=uuid4(), state=TaskState.FAILED.value, title="Draft the reply")
        with patch("services.coordinator_service.MissionDispatcher.record_task_completion"), \
                patch("services.coordinator_service._dispatch_mission_event", new=AsyncMock()), \
                patch("services.coordinator_service._narrate_mission"), \
                patch.object(CoordinatorService, "_inject_task_output_into_field", new=AsyncMock()), \
                patch.object(db_session, "refresh"):
            asyncio.run(CoordinatorService()._record_task_result(db_session, run, tasks[name], 7, failed))
    assert _MissionMemory.stored == [("failure", tasks["owner"].id)]


def test_a_widget_born_mission_keeps_no_field_in_the_workspaces_field_memory():
    from services.coordinator_service import CoordinatorService

    service = CoordinatorService()
    service._get_field = MagicMock()
    assert asyncio.run(service._create_mission_field(MagicMock(), NS(config=dict(WIDGET_ORIGIN)))) is None
    service._get_field.assert_not_called()

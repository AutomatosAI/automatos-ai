"""PRD-204 S9 -- Auto's watch tools + auto-create.

Locks the tool-schema/handler contract (required[] lists ONLY fields the
handler cannot default -- the blueprint-rules-wall regression class),
exercises the four handlers workspace-scoped against a real DB, and covers
the launch-handler auto-create gate (watch_auto_create default ON, OFF
respected, idempotent, run_and_report policy, criteria = request text).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from core.database.database import get_database_url
from core.models.core import RecipeExecution, WorkflowTemplate
from core.models.orchestration import OrchestrationRun
from core.models.watches import Watch
from core.models.watch_enums import WatchPolicy, WatchStatus
from modules.tools.discovery.action_registry import ActionRegistry
from modules.tools.discovery.actions_watches import register_watch_actions
from modules.tools.discovery.handlers_watches import (
    auto_create_watch,
    cancel_watch,
    create_watch,
    get_watch,
    list_watches,
)

FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)

WATCH_TOOLS = (
    "platform_create_watch",
    "platform_list_watches",
    "platform_get_watch",
    "platform_cancel_watch",
)


# ---------------------------------------------------------------------------
# Schema contract (no DB)
# ---------------------------------------------------------------------------


def _registry():
    registry = ActionRegistry()
    register_watch_actions(registry)
    return registry


def test_all_four_tools_register():
    registry = _registry()
    for name in WATCH_TOOLS:
        assert registry.get(name) is not None, name


def test_required_lists_only_undefaultable_fields():
    """The onboarding-wall guard: a required field the handler defaults
    dead-ends the LLM. Only identity fields may be required."""
    registry = _registry()

    assert registry.get("platform_create_watch").parameters["required"] == [
        "target_type",
        "target_id",
    ]
    assert registry.get("platform_list_watches").parameters["required"] == []
    assert registry.get("platform_get_watch").parameters["required"] == ["watch_id"]
    assert registry.get("platform_cancel_watch").parameters["required"] == ["watch_id"]

    # Every required key must exist in properties (schema self-consistency).
    for name in WATCH_TOOLS:
        params = registry.get(name).parameters
        for req in params["required"]:
            assert req in params["properties"], f"{name}.{req}"


def test_permission_levels():
    registry = _registry()
    assert registry.get("platform_create_watch").permission_level == "write"
    assert registry.get("platform_cancel_watch").permission_level == "write"
    assert registry.get("platform_list_watches").permission_level == "read"
    assert registry.get("platform_get_watch").permission_level == "read"
    for name in WATCH_TOOLS:
        action = registry.get(name)
        assert action.requires_confirmation is False
        assert action.admin_only is False


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM orchestration_runs LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watch tools suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-watch-tools"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session.sweep()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


def _seed_run(s, ws_id, goal="ship the quarterly report"):
    run = OrchestrationRun(
        workspace_id=ws_id,
        goal=goal,
        state="running",
        state_type="active",
        created_by="user_test",
    )
    s.add(run)
    s.commit()
    return run


def _seed_execution(s, ws_id):
    recipe = WorkflowTemplate(
        template_id=f"prd204-wt-{uuid.uuid4().hex[:10]}",
        name="tools test playbook",
        description="prd204 watch tools",
        workspace_id=ws_id,
        template_definition={"steps": []},
        steps=[{"step_id": "s1", "order": 1}],
        created_by="user_test",
    )
    s.add(recipe)
    s.commit()
    execution = RecipeExecution(
        execution_id=f"exec-{uuid.uuid4().hex[:12]}",
        recipe_id=recipe.id,
        workspace_id=ws_id,
        status="running",
        input_data={},
    )
    s.add(execution)
    s.commit()
    return recipe, execution


def _call(handler, s, ws_id, params):
    return asyncio.run(handler(s, ws_id, params))


# ---------------------------------------------------------------------------
# create: defaults, validation, workspace scoping, duplicate friendliness
# ---------------------------------------------------------------------------


def test_create_with_only_required_fields_defaults_everything(workspace, new_session):
    """Required-only call must succeed -- proof every optional field has a
    handler default (the contract the schema test locks)."""
    s = new_session()
    run = _seed_run(s, workspace)

    result = _call(create_watch, s, workspace,
                   {"target_type": "mission", "target_id": str(run.id)})
    s.commit()

    assert result["success"] is True, result
    watch = result["watch"]
    assert watch["policy"] == WatchPolicy.RUN_AND_REPORT.value
    assert watch["quality_threshold"] == pytest.approx(0.8)
    assert watch["action_budget"] == 2
    assert watch["deadline_at"] is None
    # Title/criteria derived from the mission's goal (the request text).
    assert "quarterly report" in watch["title"]
    assert watch["success_criteria"] == "ship the quarterly report"
    s.close()


def test_create_validates_target_type_and_existence(workspace, new_session):
    s = new_session()
    bad_type = _call(create_watch, s, workspace,
                     {"target_type": "banana", "target_id": "1"})
    assert bad_type["success"] is False
    assert "target_type" in bad_type["error"]

    missing = _call(create_watch, s, workspace,
                    {"target_type": "mission", "target_id": str(uuid.uuid4())})
    assert missing["success"] is False
    assert "found" in missing["error"]

    no_id = _call(create_watch, s, workspace, {"target_type": "mission"})
    assert no_id["success"] is False

    bad_policy = _call(create_watch, s, workspace,
                       {"target_type": "mission", "target_id": str(uuid.uuid4()),
                        "policy": "yolo"})
    assert bad_policy["success"] is False
    s.close()


def test_create_is_workspace_scoped(workspace, new_session):
    """A mission in ANOTHER workspace is invisible here."""
    s = new_session()
    other_ws = str(uuid.uuid4())
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n)"),
        {"id": other_ws, "n": "prd204-other"},
    )
    s.commit()
    foreign_run = _seed_run(s, other_ws)

    result = _call(create_watch, s, workspace,
                   {"target_type": "mission", "target_id": str(foreign_run.id)})
    assert result["success"] is False

    # cleanup the second workspace
    s.execute(text("DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)"),
              {"w": other_ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": other_ws})
    s.commit()
    s.close()


def test_duplicate_create_returns_existing_not_dead_end(workspace, new_session):
    s = new_session()
    run = _seed_run(s, workspace)
    params = {"target_type": "mission", "target_id": str(run.id)}

    first = _call(create_watch, s, workspace, params)
    s.commit()
    second = _call(create_watch, s, workspace, params)

    assert second["success"] is True
    assert second["existing"] is True
    assert second["watch"]["id"] == first["watch"]["id"]
    s.close()


def test_create_on_playbook_execution_and_options(workspace, new_session):
    s = new_session()
    _, execution = _seed_execution(s, workspace)

    result = _call(
        create_watch, s, workspace,
        {
            "target_type": "playbook_execution",
            "target_id": execution.execution_id,
            "policy": "score_and_improve",
            "quality_threshold": 0.9,
            "deadline_hours": 4,
            "action_budget": 1,
            "success_criteria": "a clean summary with citations",
        },
    )
    s.commit()
    assert result["success"] is True, result
    watch = result["watch"]
    assert watch["policy"] == "score_and_improve"
    assert watch["quality_threshold"] == pytest.approx(0.9)
    assert watch["action_budget"] == 1
    assert watch["deadline_at"] is not None
    assert watch["success_criteria"] == "a clean summary with citations"
    s.close()


# ---------------------------------------------------------------------------
# list / get / cancel round-trip
# ---------------------------------------------------------------------------


def test_list_get_cancel_round_trip(workspace, new_session):
    s = new_session()
    run = _seed_run(s, workspace)
    created = _call(create_watch, s, workspace,
                    {"target_type": "mission", "target_id": str(run.id)})
    s.commit()
    watch_id = created["watch"]["id"]

    listed = _call(list_watches, s, workspace, {})
    assert listed["success"] is True
    assert [w["id"] for w in listed["watches"]] == [watch_id]

    detail = _call(get_watch, s, workspace, {"watch_id": watch_id})
    assert detail["success"] is True
    assert detail["watch"]["id"] == watch_id
    assert detail["watch"]["lineage"][0]["target_id"] == str(run.id)
    assert [e["event_type"] for e in detail["recent_events"]] == ["created"]

    cancelled = _call(cancel_watch, s, workspace,
                      {"watch_id": watch_id, "reason": "user changed their mind"})
    s.commit()
    assert cancelled["success"] is True
    assert cancelled["watch"]["status"] == WatchStatus.CANCELLED.value

    # Live-only list no longer shows it; include_closed does.
    assert _call(list_watches, s, workspace, {})["watches"] == []
    closed = _call(list_watches, s, workspace, {"include_closed": True})
    assert [w["id"] for w in closed["watches"]] == [watch_id]

    # Cancelling again is a clean error, not a crash.
    again = _call(cancel_watch, s, workspace, {"watch_id": watch_id})
    assert again["success"] is False
    assert "closed" in again["error"]
    s.close()


def test_get_and_cancel_are_workspace_scoped(workspace, new_session):
    s = new_session()
    run = _seed_run(s, workspace)
    created = _call(create_watch, s, workspace,
                    {"target_type": "mission", "target_id": str(run.id)})
    s.commit()
    watch_id = created["watch"]["id"]

    foreign_ws = str(uuid.uuid4())
    assert _call(get_watch, s, foreign_ws, {"watch_id": watch_id})["success"] is False
    assert _call(cancel_watch, s, foreign_ws, {"watch_id": watch_id})["success"] is False
    s.close()


# ---------------------------------------------------------------------------
# Auto-create (Q1)
# ---------------------------------------------------------------------------


def test_auto_create_default_on_run_and_report(workspace, new_session):
    s = new_session()
    run = _seed_run(s, workspace, goal="research competitor pricing")

    watch = auto_create_watch(
        s, workspace,
        target_type="mission",
        target_id=str(run.id),
        title="Watch: research competitor pricing",
        success_criteria="research competitor pricing",
        created_by="user_abc",
    )
    s.commit()

    assert watch is not None
    assert watch.policy == WatchPolicy.RUN_AND_REPORT.value
    assert watch.success_criteria == "research competitor pricing"
    assert watch.created_by == "user_abc"

    # Idempotent: a second auto-create on the same live target is a no-op.
    assert auto_create_watch(
        s, workspace,
        target_type="mission", target_id=str(run.id),
        title="dup", success_criteria="dup",
    ) is None
    assert (
        s.query(Watch).filter(Watch.workspace_id == workspace).count() == 1
    )
    s.close()


def test_auto_create_respects_workspace_setting_off(workspace, new_session):
    s = new_session()
    run = _seed_run(s, workspace)
    s.execute(
        text(
            "UPDATE workspaces SET settings = CAST(:cfg AS jsonb) "
            "WHERE id = CAST(:w AS uuid)"
        ),
        {"cfg": '{"watch_auto_create": false}', "w": workspace},
    )
    s.commit()

    assert auto_create_watch(
        s, workspace,
        target_type="mission", target_id=str(run.id),
        title="t", success_criteria="c",
    ) is None
    assert s.query(Watch).filter(Watch.workspace_id == workspace).count() == 0
    s.close()


def test_auto_create_never_raises(workspace, new_session, monkeypatch):
    """A broken watcher must not break a launch (fail-soft contract)."""
    from services import watch_service

    s = new_session()
    run = _seed_run(s, workspace)
    monkeypatch.setattr(
        watch_service.WatchService,
        "create_watch",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    result = auto_create_watch(
        s, workspace,
        target_type="mission", target_id=str(run.id),
        title="t", success_criteria="c",
    )
    assert result is None  # swallowed, logged
    s.close()


def test_execute_playbook_tool_auto_creates_watch(workspace, new_session, monkeypatch):
    """The playbook-execute tool wires auto-create for real (engine stubbed)."""
    import services.concurrency_guard as cg
    import services.playbook_engine as pe
    from types import SimpleNamespace

    from modules.tools.discovery.handlers_playbooks import execute_playbook

    async def _allow(workspace_id, db):
        return SimpleNamespace(allowed=True, reason=None, limits={})

    launched = []
    monkeypatch.setattr(cg, "check_concurrency", _allow)
    monkeypatch.setattr(
        pe, "get_playbook_engine",
        lambda: SimpleNamespace(launch=lambda **kw: launched.append(kw)),
    )

    s = new_session()
    recipe, _ = _seed_execution(s, workspace)

    result = asyncio.run(
        execute_playbook(
            s, uuid.UUID(workspace),
            {"playbook_id": recipe.id, "input_data": {"topic": "pricing"},
             "_created_by": "user_abc"},
        )
    )
    assert result["success"] is True, result
    assert launched  # engine seam fired

    watch = (
        s.query(Watch)
        .filter(
            Watch.workspace_id == workspace,
            Watch.target_type == "playbook_execution",
            Watch.target_id == result["execution_id"],
        )
        .one()
    )
    assert watch.policy == WatchPolicy.RUN_AND_REPORT.value
    assert recipe.name[:20] in watch.title
    assert "pricing" in watch.success_criteria  # request text seeded
    assert watch.created_by == "user_abc"
    s.close()


def test_create_mission_tool_auto_creates_watch(workspace, new_session, monkeypatch):
    """The mission tool wires auto-create for real (coordinator stubbed --
    the planner is an LLM)."""
    from services.coordinator_service import CoordinatorService
    from modules.tools.discovery.handlers_missions import create_mission

    s = new_session()
    seeded_run = _seed_run(s, workspace, goal="draft the launch narrative")
    seeded_run.state = "awaiting_approval"
    seeded_run.state_type = "blocked"
    seeded_run.plan = {"tasks": []}
    s.commit()

    async def _fake_create_mission(self, db, workspace_id, goal, created_by, config=None):
        return seeded_run

    monkeypatch.setattr(CoordinatorService, "create_mission", _fake_create_mission)

    result = asyncio.run(
        create_mission(
            s, uuid.UUID(workspace),
            {"goal": "draft the launch narrative", "_created_by": "user_abc"},
        )
    )
    assert result["success"] is True, result

    watch = (
        s.query(Watch)
        .filter(
            Watch.workspace_id == workspace,
            Watch.target_type == "mission",
            Watch.target_id == str(seeded_run.id),
        )
        .one()
    )
    assert watch.policy == WatchPolicy.RUN_AND_REPORT.value
    assert watch.success_criteria == "draft the launch narrative"
    s.close()

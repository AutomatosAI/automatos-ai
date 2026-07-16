"""PRD-204 S7 -- rerun + tweak + grant wiring.

Covers: the copy idiom (inputs + retry_of lineage + attempt_count), the
per-execution step-override merge (shared playbook definition NEVER
mutated), the approval gate (always_ask parks a SUBJECT_PLAYBOOK_RUN grant;
full_auto under-ceiling launches; over-ceiling estimate parks), grant ->
resume launches + the watch follows, deny -> no launch + needs_attention.

The playbook engine launch is ALWAYS stubbed (no local runs -- CI is the
gate); notification dispatch is captured at the dispatcher seam (the
``notifications`` table is migration-only and absent from the test schema).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url
from core.models import WorkflowTemplate
from core.models.core import LLMUsage, RecipeExecution
from core.models.approval_grants import (
    ApprovalGrant,
    GrantStatus,
    SUBJECT_PLAYBOOK_RUN,
)
from core.models.watches import WatchEvent
from core.models.watch_enums import WatchEventType, WatchStatus
from core.services.approval_grants import grant_grant
from services.watch_service import WatchService

FROZEN_NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Pure: executor-side override merge
# ---------------------------------------------------------------------------


def test_apply_step_overrides_new_dicts_only_prompt_honoured():
    from api.recipe_executor import _apply_step_overrides

    steps = [
        {"step_id": "s1", "order": 1, "prompt_template": "original one", "agent_id": 7},
        {"step_id": "s2", "order": 2, "prompt_template": "original two", "agent_id": 8},
    ]
    overrides = {
        "s1": {"prompt_template": "tweaked one", "agent_id": 99, "order": 42},
        "missing": {"prompt_template": "never lands"},
        "s2": "not-a-dict",
    }
    merged = _apply_step_overrides(steps, overrides)

    assert merged[0]["prompt_template"] == "tweaked one"
    # Only prompt_template is honoured -- agent/order untouched.
    assert merged[0]["agent_id"] == 7
    assert merged[0]["order"] == 1
    assert merged[1]["prompt_template"] == "original two"
    # The source list is never mutated (new dicts).
    assert steps[0]["prompt_template"] == "original one"
    assert merged[0] is not steps[0]

    # No overrides -> plain copies.
    plain = _apply_step_overrides(steps, None)
    assert plain[0] == steps[0] and plain[0] is not steps[0]


def test_validate_step_overrides_boundary():
    from services.watch_rerun import validate_step_overrides

    class _Recipe:
        steps = [{"step_id": "s1", "order": 1}, {"step_id": "s2", "order": 2}]

    ok, err = validate_step_overrides(_Recipe(), {"s1": {"prompt_template": "new"}})
    assert err is None
    assert ok == {"s1": {"prompt_template": "new"}}

    assert validate_step_overrides(_Recipe(), None) == (None, None)
    assert validate_step_overrides(_Recipe(), {}) == (None, None)

    _, err = validate_step_overrides(_Recipe(), {"nope": {"prompt_template": "x"}})
    assert "Unknown step_id" in err
    _, err = validate_step_overrides(_Recipe(), {"s1": "just a string"})
    assert "must be an object" in err
    _, err = validate_step_overrides(_Recipe(), {"s1": {"prompt_template": "  "}})
    assert "non-empty string" in err
    _, err = validate_step_overrides(_Recipe(), ["not", "a", "dict"])
    assert "must be an object" in err


# ---------------------------------------------------------------------------
# DB fixtures (stage-1 pattern; skip cleanly without Postgres)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM recipe_executions LIMIT 1"))
            c.execute(text("SELECT 1 FROM approval_grants LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"rerun suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(new_session):
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": "prd204-rerun"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM llm_usage WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM recipe_executions WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


@pytest.fixture
def stub_launch(monkeypatch):
    """No local playbook runs: capture engine launches instead."""
    import services.watch_rerun as wr

    launched = []
    monkeypatch.setattr(wr, "launch_execution", lambda execution: launched.append(execution.execution_id))
    return launched


@pytest.fixture
def capture_notifications(monkeypatch):
    """Capture NotificationDispatcher.dispatch at the seam (never SELECT
    notifications -- the table is migration-only)."""
    from core.services.notification_dispatcher import NotificationDispatcher

    sent = []

    async def _capture(self, event_type, title, message=None, **kwargs):
        sent.append({"event_type": event_type, "title": title,
                     "message": message, **kwargs})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _capture)
    return sent


def _seed_recipe_and_execution(s, ws_id, *, cost_rows=()):
    recipe = WorkflowTemplate(
        template_id=f"prd204-rr-{uuid.uuid4().hex[:10]}",
        name="rerun test playbook",
        description="prd204 rerun",
        workspace_id=ws_id,
        template_definition={"steps": []},
        steps=[
            {"step_id": "s1", "order": 1, "prompt_template": "do the thing"},
            {"step_id": "s2", "order": 2, "prompt_template": "summarise it"},
        ],
        created_by="user_test",
    )
    s.add(recipe)
    s.commit()

    original = RecipeExecution(
        execution_id=f"exec-{uuid.uuid4().hex[:12]}",
        recipe_id=recipe.id,
        workspace_id=ws_id,
        status="completed",
        input_data={"topic": "quarterly numbers"},
        attempt_count=1,
        triggered_by="user@example.com",
    )
    s.add(original)
    for cost in cost_rows:
        s.add(
            LLMUsage(
                workspace_id=ws_id,
                model_id="openai/gpt-4o-mini",
                provider="openai",
                tier="direct",
                execution_id=original.execution_id,
                request_type="recipe",
                input_tokens=100,
                output_tokens=100,
                total_tokens=200,
                input_cost=cost / 2,
                output_cost=cost / 2,
                total_cost=cost,
                status="success",
            )
        )
    s.commit()
    return recipe, original


def _set_policy(s, ws_id, settings):
    s.execute(
        text("UPDATE workspaces SET settings = CAST(:cfg AS jsonb) WHERE id = CAST(:w AS uuid)"),
        {"cfg": __import__("json").dumps(settings), "w": ws_id},
    )
    s.commit()


def _make_watch(s, ws_id, original):
    watch = WatchService.create_watch(
        s,
        workspace_id=ws_id,
        watch_type="playbook_execution",
        target_type="playbook_execution",
        target_id=original.execution_id,
        title="Watch: rerun test",
        success_criteria="deliver the summary",
        policy="score_and_improve",
        now=FROZEN_NOW,
    )
    s.commit()
    return watch


# ---------------------------------------------------------------------------
# Copy idiom
# ---------------------------------------------------------------------------


def test_create_rerun_execution_copies_inputs_and_lineage(workspace, new_session):
    from services.watch_rerun import create_rerun_execution

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    overrides = {"s1": {"prompt_template": "do the thing, but cite sources"}}

    rerun = create_rerun_execution(
        s, recipe, original, step_overrides=overrides, triggered_by="watch_rerun"
    )
    s.commit()

    assert rerun.execution_id.startswith("rerun-")
    assert rerun.retry_of == original.execution_id
    assert rerun.attempt_count == 2
    assert rerun.triggered_by == "watch_rerun"
    assert rerun.input_data == {"topic": "quarterly numbers"}
    assert rerun.input_data is not original.input_data  # a copy, never shared
    assert rerun.execution_metadata["rerun_of"] == original.execution_id
    assert rerun.execution_metadata["step_overrides"] == overrides

    # The shared definition is untouched -- the tweak lives ONLY on this
    # execution's metadata.
    s.refresh(recipe)
    assert recipe.steps[0]["prompt_template"] == "do the thing"
    s.close()


def test_estimate_rerun_cost_sums_original_llm_usage(workspace, new_session):
    from services.watch_rerun import estimate_rerun_cost_usd

    s = new_session()
    _, original = _seed_recipe_and_execution(s, workspace, cost_rows=(1.25, 0.75))
    estimate = estimate_rerun_cost_usd(s, workspace, original.execution_id)
    assert estimate == pytest.approx(2.0)
    # No usage rows -> 0.0, never raises.
    assert estimate_rerun_cost_usd(s, workspace, "exec-none") == 0.0
    s.close()


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------


def test_always_ask_parks_grant_watch_and_notifies(
    workspace, new_session, stub_launch, capture_notifications
):
    from services.watch_rerun import request_rerun

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    watch = _make_watch(s, workspace, original)
    # Default policy is always_ask (no settings needed).

    outcome = asyncio.run(
        request_rerun(
            s,
            workspace_id=workspace,
            recipe=recipe,
            original=original,
            step_overrides={"s1": {"prompt_template": "tweak"}},
            triggered_by="watch_rerun",
            watch=watch,
        )
    )
    s.commit()

    assert outcome.launched is False
    assert stub_launch == []  # nothing ran
    assert outcome.grant_id is not None

    grant = s.query(ApprovalGrant).get(outcome.grant_id)
    assert grant.subject_type == SUBJECT_PLAYBOOK_RUN
    assert grant.subject_id == original.execution_id
    assert grant.status == GrantStatus.PENDING.value
    assert grant.details["watch_action"] == "rerun"
    assert grant.details["watch_id"] == str(watch.id)
    assert grant.details["rerun_of"] == original.execution_id
    assert grant.details["spec"]["input_data"] == {"topic": "quarterly numbers"}
    assert grant.details["spec"]["step_overrides"] == {"s1": {"prompt_template": "tweak"}}

    s.refresh(watch)
    assert watch.status == WatchStatus.AWAITING_APPROVAL.value

    pending = [n for n in capture_notifications if n["event_type"] == "approval_pending"]
    assert len(pending) == 1
    assert str(grant.id) in str(pending[0].get("link_id"))

    # Idempotent: a second ask reuses the pending grant.
    second = asyncio.run(
        request_rerun(
            s, workspace_id=workspace, recipe=recipe, original=original,
            triggered_by="watch_rerun",
        )
    )
    assert second.grant_id == grant.id
    assert (
        s.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == workspace,
            ApprovalGrant.subject_id == original.execution_id,
        )
        .count()
        == 1
    )
    s.close()


def test_full_auto_under_ceiling_launches_and_watch_follows(
    workspace, new_session, stub_launch, capture_notifications
):
    from services.watch_rerun import request_rerun

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    watch = _make_watch(s, workspace, original)
    _set_policy(
        s,
        workspace,
        {
            "approval_policy": {"policy": "full_auto"},
            "autonomy": {"level": "full"},  # Section 12.3 gate ON
        },
    )

    outcome = asyncio.run(
        request_rerun(
            s,
            workspace_id=workspace,
            recipe=recipe,
            original=original,
            step_overrides={"s2": {"prompt_template": "summarise with citations"}},
            triggered_by="watch_rerun",
            watch=watch,
        )
    )

    assert outcome.launched is True
    assert stub_launch == [outcome.execution_id]

    rerun = (
        s.query(RecipeExecution)
        .filter(RecipeExecution.execution_id == outcome.execution_id)
        .first()
    )
    assert rerun is not None
    assert rerun.retry_of == original.execution_id

    s.refresh(watch)
    # One watch, one verdict: the watch now supervises the rerun.
    assert watch.target_id == outcome.execution_id
    assert watch.status == WatchStatus.WATCHING.value
    follow_events = (
        s.query(WatchEvent)
        .filter_by(watch_id=watch.id, event_type=WatchEventType.FOLLOW.value)
        .all()
    )
    assert len(follow_events) == 1
    # Overrides recorded on the watch event for before/after comparison.
    assert follow_events[0].snapshot["step_overrides"] == {
        "s2": {"prompt_template": "summarise with citations"}
    }
    s.close()


def test_over_ceiling_estimate_parks_under_auto_below_budget(
    workspace, new_session, stub_launch, capture_notifications
):
    from services.watch_rerun import request_rerun

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace, cost_rows=(5.0,))
    _set_policy(
        s,
        workspace,
        {"approval_policy": {"policy": "auto_below_budget", "approval_dollar_ceiling": 1.0}},
    )

    outcome = asyncio.run(
        request_rerun(s, workspace_id=workspace, recipe=recipe, original=original)
    )
    s.commit()
    assert outcome.launched is False  # $5 estimate > $1 ceiling -> ask
    assert outcome.grant_id is not None
    assert stub_launch == []

    # Raise the ceiling above the estimate -> auto path.
    s.query(ApprovalGrant).filter(ApprovalGrant.id == outcome.grant_id).delete()
    _set_policy(
        s,
        workspace,
        {"approval_policy": {"policy": "auto_below_budget", "approval_dollar_ceiling": 10.0}},
    )
    second = asyncio.run(
        request_rerun(s, workspace_id=workspace, recipe=recipe, original=original)
    )
    assert second.launched is True
    assert stub_launch == [second.execution_id]
    s.close()


# ---------------------------------------------------------------------------
# Grant resume / deny
# ---------------------------------------------------------------------------


def _park_rerun_grant(s, workspace, recipe, original, watch):
    from services.watch_rerun import request_rerun

    outcome = asyncio.run(
        request_rerun(
            s,
            workspace_id=workspace,
            recipe=recipe,
            original=original,
            step_overrides={"s1": {"prompt_template": "tweaked"}},
            triggered_by="watch_rerun",
            watch=watch,
        )
    )
    s.commit()
    return s.query(ApprovalGrant).get(outcome.grant_id)


def test_grant_resume_launches_stored_spec_and_watch_follows(
    workspace, new_session, stub_launch, capture_notifications
):
    from services.watch_rerun import resume_playbook_run_grant

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    watch = _make_watch(s, workspace, original)
    grant = _park_rerun_grant(s, workspace, recipe, original, watch)

    grant_grant(grant, granted_by="user:42")
    asyncio.run(resume_playbook_run_grant(s, grant))
    s.commit()

    result = grant.details["executed_result"]
    assert result["success"] is True
    new_execution_id = result["execution_id"]
    assert stub_launch == [new_execution_id]

    rerun = (
        s.query(RecipeExecution)
        .filter(RecipeExecution.execution_id == new_execution_id)
        .first()
    )
    assert rerun.retry_of == original.execution_id
    assert rerun.execution_metadata["step_overrides"] == {"s1": {"prompt_template": "tweaked"}}
    assert rerun.triggered_by == "watch_rerun"

    s.refresh(watch)
    assert watch.status == WatchStatus.WATCHING.value  # resumed from the park
    assert watch.target_id == new_execution_id
    s.close()


def test_grant_deny_no_launch_watch_needs_attention(
    workspace, new_session, stub_launch, capture_notifications
):
    from core.services.approval_grants import deny_grant
    from services.watch_rerun import fail_playbook_run_grant

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    watch = _make_watch(s, workspace, original)
    grant = _park_rerun_grant(s, workspace, recipe, original, watch)

    deny_grant(grant, revoked_by="user:42")
    fail_playbook_run_grant(s, grant)
    s.commit()

    assert stub_launch == []
    assert grant.details["executed_result"]["success"] is False
    # No rerun row was ever created.
    reruns = (
        s.query(RecipeExecution)
        .filter(RecipeExecution.retry_of == original.execution_id)
        .count()
    )
    assert reruns == 0

    s.refresh(watch)
    assert watch.status == WatchStatus.NEEDS_ATTENTION.value
    s.close()


def test_resume_with_missing_target_parks_watch(
    workspace, new_session, stub_launch, capture_notifications
):
    """Self-healing: a granted rerun whose recipe/execution vanished parks
    the watch instead of crashing the grant endpoint."""
    from services.watch_rerun import resume_playbook_run_grant

    s = new_session()
    recipe, original = _seed_recipe_and_execution(s, workspace)
    watch = _make_watch(s, workspace, original)
    grant = _park_rerun_grant(s, workspace, recipe, original, watch)

    # Vaporise the original execution.
    s.query(RecipeExecution).filter(
        RecipeExecution.execution_id == original.execution_id
    ).delete()
    s.commit()

    grant_grant(grant, granted_by="user:42")
    asyncio.run(resume_playbook_run_grant(s, grant))
    s.commit()

    assert grant.details["executed_result"]["success"] is False
    assert stub_launch == []
    s.refresh(watch)
    assert watch.status == WatchStatus.NEEDS_ATTENTION.value
    s.close()

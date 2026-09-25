"""F166 — the local owner approves a mission from chat.

The executor attributed a mission action to the person chatting only through
their Clerk id (``caller_context['user_id']``). The local edition's operator has
no Clerk id. The chat threads their internal ``users.id`` as ``driving_user_id``
(PRD-234 D16), and the executor never read it. So under the owner's always-ask
policy F036 refused "go ahead" and "approve mission <id>": "no person is behind
this request". Reject, pause, resume, cancel, replan and plan edits lost their
person the same way, and were recorded as the agent.

With no Clerk id, the executor now records the person's email, read from
``driving_user_id``. That is what the local REST API records too
(``created_by = ctx.user.id``). It is never the bare ``users.id``, because a
digit string in ``created_by`` is already an agent id wherever no person drove
the action. The readers that turn ``created_by`` back into a person
(notifications, narration, watch notices) take the Clerk id, then the email,
and never a number. A widget turn, a board ticket and a workflow name nobody, so
they are still refused. So is an unreadable user. A ``_created_by`` or
``driving_user_id`` in the tool's arguments is ignored.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from core.security.surface import WIDGET, turn_surface

NO_PERSON = "no person is behind this request"
WIDGET_CHAT = {"user_query": "go ahead", "conversation_id": "w-1", "turn_id": "t-1"}
BOARD_TICKET = {"field_context": {"field_id": "f-85"}, "board_task_id": 85}
WORKFLOW = {"playbook_execution_id": "exec-1", "playbook_step": 2}

# action → (the coordinator method it calls, extra tool args)
LIFECYCLE = {
    "platform_approve_mission": ("approve_plan", {}),
    "platform_reject_mission": ("reject_plan", {"reason": "not now"}),
    "platform_pause_mission": ("pause_mission", {}),
    "platform_resume_mission": ("resume_mission", {}),
    "platform_cancel_mission": ("cancel_mission", {}),
    "platform_replan_mission": ("replan_mission", {"notes": "use the café's own supplier"}),
    "platform_update_mission_plan": ("update_mission_plan", {"task_edits": [{"task_id": "t1", "agent_id": 3}]}),
}


def _user(db, clerk=None):
    email = f"owner-{uuid4().hex[:8]}@cafe.test"
    user_id = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                         {"e": email, "u": email.split("@")[0], "c": clerk}).scalar()
    return NS(id=user_id, email=email)


def _local_chat(user_id):
    """What the local chat threads: no Clerk id, the internal id of the person typing."""
    return {"user_query": "go ahead", "conversation_id": "a7af93bd", "turn_id": "t-1", "driving_user_id": str(user_id)}


@pytest.fixture
def mission(db_session, seed_workspace, monkeypatch):
    """A workspace whose owner approves every mission, its owner, and one of its missions."""
    from modules.tools.discovery import handlers_missions as hm

    monkeypatch.setattr("core.services.approval_policy.load_approval_policy",
                        lambda db, ws: {"policy": "always_ask", "approval_dollar_ceiling": None,
                                        "auto_proceed_after_seconds": None})
    monkeypatch.setattr(hm, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    run = NS(id=uuid4(), goal="Reorder the café's oat milk", state="running", config={}, token_budget_estimate=0,
             plan={"tasks": [{"title": "Check stock"}]})
    monkeypatch.setattr(hm, "_resolve_run", lambda db, ws, params: (run, None))
    coordinator = MagicMock()
    for method, _ in LIFECYCLE.values():
        setattr(coordinator, method, MagicMock(return_value=run))
    coordinator.replan_mission = AsyncMock(return_value=run)
    coordinator.create_mission = AsyncMock(return_value=NS(**{**vars(run), "state": "awaiting_approval"}))
    monkeypatch.setattr("services.coordinator_service.CoordinatorService", lambda: coordinator)
    return NS(db=db_session, ws=UUID(seed_workspace()), run=run, coordinator=coordinator, owner=_user(db_session))


def _execute(mission, action, params, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(mission.db, mission.ws)
    args = params if action == "platform_create_mission" else {"mission_id": str(mission.run.id), **params}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, args, caller_context))


def _actor(mission, action):
    """The person the coordinator recorded the action for."""
    method = getattr(mission.coordinator, LIFECYCLE[action][0])
    return method.call_args.args[2]


def test_a_local_owner_approves_a_mission_from_chat(mission):
    reply = _execute(mission, "platform_approve_mission", {}, _local_chat(mission.owner.id))
    assert reply["success"] is True, reply
    assert _actor(mission, "platform_approve_mission") == mission.owner.email


@pytest.mark.parametrize("action", sorted(LIFECYCLE))
def test_every_mission_action_from_a_local_chat_is_made_for_its_person(mission, action):
    reply = _execute(mission, action, LIFECYCLE[action][1], _local_chat(mission.owner.id))
    assert reply["success"] is True, reply
    assert _actor(mission, action) == mission.owner.email


def test_a_mission_created_from_a_local_chat_is_created_by_its_person(mission):
    reply = _execute(mission, "platform_create_mission", {"goal": "Reorder the café's oat milk"},
                     _local_chat(mission.owner.id))
    assert reply["success"] is True, reply
    assert mission.coordinator.create_mission.call_args.kwargs["created_by"] == mission.owner.email


def test_on_saas_the_clerk_id_still_names_the_person(mission):
    _execute(mission, "platform_approve_mission", {}, {**_local_chat(mission.owner.id), "user_id": "user_2abc"})
    assert _actor(mission, "platform_approve_mission") == "user_2abc"


def test_a_driving_user_nobody_can_read_is_nobody(mission):
    """No users row: the call names nobody, so F036 still leaves the approval to the owner."""
    reply = _execute(mission, "platform_approve_mission", {}, _local_chat(2_000_000_000))
    assert reply["success"] is False and NO_PERSON in reply["error"]
    mission.coordinator.approve_plan.assert_not_called()


@pytest.mark.parametrize("lane", [None, BOARD_TICKET, WORKFLOW], ids=["no-context", "board-ticket", "workflow"])
def test_a_lane_no_person_drives_still_cannot_approve(mission, lane):
    reply = _execute(mission, "platform_approve_mission", {}, lane)
    assert reply["success"] is False and NO_PERSON in reply["error"]
    mission.coordinator.approve_plan.assert_not_called()


def test_a_widget_turn_still_cannot_approve(mission):
    with turn_surface(WIDGET, ("chat", "missions:read"), None):
        reply = _execute(mission, "platform_approve_mission", {}, WIDGET_CHAT)
    assert reply["success"] is False
    mission.coordinator.approve_plan.assert_not_called()


def test_a_widget_turn_is_made_for_nobody_whatever_its_context_names(mission):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    seen = []

    async def _handler(db, ws, params):
        seen.append(params)
        return {"success": True}

    executor = PlatformActionExecutor(mission.db, mission.ws)
    executor._handlers["platform_approve_mission"] = _handler
    context = {**_local_chat(mission.owner.id), "user_id": "user_2abc"}
    with turn_surface(WIDGET, ("chat",), None), \
            patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        asyncio.run(executor.execute("platform_approve_mission", {"mission_id": str(mission.run.id)}, context))
    assert seen and "_created_by" not in seen[0]


SPOOFS = {"_created_by": "user_owner", "driving_user_id": "1", "user_id": "user_owner"}


@pytest.mark.parametrize("lane", [BOARD_TICKET, WORKFLOW], ids=["board-ticket", "workflow"])
def test_a_person_named_in_the_tool_args_is_ignored(mission, lane):
    reply = _execute(mission, "platform_approve_mission", dict(SPOOFS), lane)
    assert reply["success"] is False and NO_PERSON in reply["error"]
    mission.coordinator.approve_plan.assert_not_called()


def test_a_person_named_in_the_tool_args_does_not_replace_the_one_chatting(mission):
    _execute(mission, "platform_approve_mission", dict(SPOOFS), _local_chat(mission.owner.id))
    assert _actor(mission, "platform_approve_mission") == mission.owner.email


def test_actor_reads_the_recorded_person():
    from modules.tools.discovery.handlers_missions import _actor

    assert _actor({"_created_by": "owner@cafe.test", "_agent_id": 58}) == "owner@cafe.test"
    assert _actor({"_agent_id": 58}) == "58"


# ── the readers that turn created_by back into a person ─────────────────────

def test_a_recorded_person_is_a_clerk_id_or_an_email_never_a_number(db_session):
    from core.auth.actor import resolve_recorded_person

    saas, local = _user(db_session, clerk=f"user_{uuid4().hex[:10]}"), _user(db_session)
    clerk = db_session.execute(text("SELECT clerk_user_id FROM users WHERE id = :i"), {"i": saas.id}).scalar()
    assert resolve_recorded_person(db_session, clerk) == saas.id
    assert resolve_recorded_person(db_session, local.email) == local.id
    # an agent id — even one that equals a real users.id — is nobody
    assert resolve_recorded_person(db_session, str(local.id)) is None
    assert resolve_recorded_person(db_session, None) is None and resolve_recorded_person(db_session, "agent") is None


def test_notifications_narration_and_watch_notices_find_the_local_owner(db_session, seed_workspace):
    from services.chat_messenger import _resolve_user_int_id
    from services.coordinator_service import _dispatch_mission_event
    from services.watch_notifications import dispatch_watch_notification

    owner, ws = _user(db_session), seed_workspace()
    assert _resolve_user_int_id(db_session, owner.email) == owner.id  # narration's owner
    for created_by, expected in ((owner.email, owner.id), (str(owner.id), None)):  # an agent id stays unresolved
        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock()
        with patch("core.services.notification_dispatcher.NotificationDispatcher", return_value=dispatcher):
            asyncio.run(_dispatch_mission_event(db_session, NS(id=uuid4(), workspace_id=ws, created_by=created_by),
                                                "mission_completed", "Done", None))
            asyncio.run(dispatch_watch_notification(db_session, NS(id=1, workspace_id=ws, created_by=created_by),
                                                    event_type="watch_verdict", title="Verdict", message=None))
        assert [c.kwargs["user_id"] for c in dispatcher.dispatch.await_args_list] == [expected, expected]

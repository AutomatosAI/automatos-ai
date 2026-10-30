"""PRD-268 US-C11c (§6.3, FR-9, FR-14, K7, K9, K10) — Auto's follow-up tools, against Postgres.

``platform_schedule_follow_up``, ``platform_list_follow_ups`` and ``platform_cancel_follow_up``
keep Auto's "I'll check again at 15:10". Proven here:

* six schedules at once, from six sessions, with a pending cap of five, make exactly five
  rows: the counts and the insert run under the workspace's advisory lock;
* the person and the chat come from the server-built caller context only; a ``_follow_up``,
  chat id or user id the model sends is ignored;
* outside a person's own chat turn, from an agent other than the workspace's Auto, in shadow
  or off, and while the person is paused, each call is refused and nothing is written;
* a follow-up about a Mission, a playbook run or a ticket, or whose reason names a ticket
  by its board number, is told to use a watch;
* the list shows only the caller's own; cancelling another person's is not found; schedule
  and cancel each write one audit row, which never carries the reason;
* the CHECK_IN surface offers none of the four chat-only tools, and the lane refuses them.

``agent_scheduled_tasks`` has no model, so CI's create_all database has none: the module
makes it, in the shape the migrations give it, when it is missing, and drops it after. The
rows commit (six sessions must see each other's), and each test sweeps its own.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

SCHEDULE, LIST, CANCEL = "platform_schedule_follow_up", "platform_list_follow_ups", "platform_cancel_follow_up"
REASON = "Check whether the Lisbon card moved; Ana said the landlord may pull out"
SCHEDULED_TASKS = """
    CREATE TABLE IF NOT EXISTS agent_scheduled_tasks (
        id SERIAL PRIMARY KEY, workspace_id UUID NOT NULL, created_by_agent_id INTEGER,
        target_agent_id INTEGER, task_type VARCHAR(20) NOT NULL DEFAULT 'one_shot',
        description TEXT NOT NULL, schedule VARCHAR(100) NOT NULL, max_runs INTEGER,
        run_count INTEGER NOT NULL DEFAULT 0, status VARCHAR(20) NOT NULL DEFAULT 'active',
        last_run_at TIMESTAMPTZ, next_run_at TIMESTAMPTZ, last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_by_user_id VARCHAR(255), deliver_as VARCHAR(20) NOT NULL DEFAULT 'chat', payload JSONB,
        origin_chat_id UUID, for_user_id INTEGER,
        CONSTRAINT ck_scheduled_task_deliver_as CHECK (deliver_as IN ('chat', 'board_task', 'follow_up')))
"""


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind FROM chats LIMIT 1"))
            c.execute(text("SELECT paused FROM proactive_person_settings LIMIT 1"))
            c.execute(text("SELECT actor_type FROM audit_logs LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the follow-up tool tests need Postgres with the PRD-205 and PRD-268 tables: {exc}")
    with eng.begin() as c:
        built = not c.execute(text("SELECT to_regclass('agent_scheduled_tasks') IS NOT NULL")).scalar()
        if built:
            c.execute(text(SCHEDULED_TASKS))
    yield eng
    if built:
        with eng.begin() as c:
            c.execute(text("SET LOCAL lock_timeout = '5s'"))
            c.execute(text("DROP TABLE IF EXISTS agent_scheduled_tasks"))
    eng.dispose()


def _person(s, tag):
    clerk = f"user_prd268w3fu_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return clerk, int(pk)


def _workspace(s, owner):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) VALUES (CAST(:id AS uuid), "
                   "'prd268w3 follow-ups', :o, CAST(:settings AS jsonb))"),
              {"id": ws, "o": owner, "settings": json.dumps({"proactive": {"enabled": True}})})
    return ws


def _agent(s, ws, name, slug=None, system=False):
    return int(s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent, configuration, slug) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', :s, CAST('{}' AS json), :slug) RETURNING id"),
        {"n": name, "w": ws, "s": system, "slug": slug}).scalar())


def _chat(s, ws, user):
    chat = str(uuid.uuid4())
    s.execute(text("INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind) "
                   "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), 'Lisbon', 'private', 'user')"),
              {"id": chat, "u": user, "w": ws})
    return chat


def _sweep(new_session, workspaces, clerks):
    s = new_session.sweep()
    for table in ("audit_logs", "agent_scheduled_tasks", "proactive_person_settings", "chats", "workspace_members",
                  "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": clerks})
    s.commit()


@pytest.fixture
def team(engine, new_session, monkeypatch):
    """Ana owns the workspace and chats with Auto; Ben is an editor there with a chat of his own;
    Kim belongs to another workspace. VECTOR is an ordinary agent; the platform mode is live."""
    from core.llm import manager
    from services.scheduled_task_service import ScheduledTaskService

    s = new_session()
    (ana_clerk, ana), (ben_clerk, ben), (kim_clerk, kim) = _person(s, "ana"), _person(s, "ben"), _person(s, "kim")
    ws, other_ws = _workspace(s, ana), _workspace(s, kim)
    s.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                   "VALUES (CAST(:w AS uuid), :u, 'editor', true)"), {"w": ws, "u": ben})
    team = NS(ws=ws, other_ws=other_ws, ana=ana, ben=ben, kim=kim, new_session=new_session,
              auto=_agent(s, ws, "Auto", f"auto-{ws}", system=True), vector=_agent(s, ws, "VECTOR"),
              other_auto=_agent(s, other_ws, "Auto", f"auto-{other_ws}", system=True),
              chat=_chat(s, ws, ana), ben_chat=_chat(s, ws, ben), mode={"value": "live"}, registered=[],
              unscheduled=[])
    s.commit()
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: team.mode["value"] if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(ScheduledTaskService, "_register_with_scheduler",
                        lambda self, *args: team.registered.append(args))
    monkeypatch.setattr(ScheduledTaskService, "_reschedule", lambda self, *args: team.unscheduled.append(args))
    yield team
    _sweep(new_session, [ws, other_ws], [ana_clerk, ben_clerk, kim_clerk])


def _context(team, user=None, chat=None):
    """The chat's server-built caller context for ``user``'s turn in ``chat`` (Ana's, by default)."""
    return {"driving_user_id": str(user or team.ana), "conversation_id": chat or team.chat, "user_id": "user_clerk"}


def _params(team, *, user=None, chat=None, agent=None, **params):
    """A tool call's params as the executor hands them to the handler: the model's, the agent
    exec_platform minted, and the follow-up's server-side facts."""
    from modules.tools.discovery.follow_up_context import follow_up_params

    sent = {"_agent_id": agent or team.auto, **params}
    return follow_up_params(SCHEDULE, sent, _context(team, user, chat))


def _for(team, action, *, agent=None, **params):
    """A list or cancel call's params, as the executor hands them over in Ana's turn."""
    from modules.tools.discovery.follow_up_context import follow_up_params

    return follow_up_params(action, {"_agent_id": agent or team.auto, **params}, _context(team))


def _call(team, action, params):
    from modules.tools.discovery import handlers_follow_ups

    handler = {SCHEDULE: handlers_follow_ups.schedule_follow_up, LIST: handlers_follow_ups.list_follow_ups,
               CANCEL: handlers_follow_ups.cancel_follow_up}[action]
    return asyncio.run(handler(team.new_session(), team.ws, params))


def _schedule(team, minutes=30, **over):
    return _call(team, SCHEDULE, _params(team, **{"reason": REASON, "in_minutes": minutes, **over}))


def _execute(team, action, params, context):
    """One call through ``PlatformActionExecutor.execute``, as a chat turn makes it."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(team.new_session(), UUID(team.ws))
    executor._full_autonomy = lambda: False
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params, context))


def _rows(team):
    s = team.new_session()
    return s.execute(text(
        "SELECT id, deliver_as, task_type, created_by_agent_id, target_agent_id, for_user_id, origin_chat_id, "
        "description, status, created_by_user_id, payload, next_run_at, schedule FROM agent_scheduled_tasks "
        "WHERE workspace_id = CAST(:w AS uuid) ORDER BY id"), {"w": team.ws}).fetchall()


def _audits(team):
    s = team.new_session()
    return s.execute(text("SELECT action, user_id, actor_type, resource_type, resource_id, resource_name, details "
                          "FROM audit_logs WHERE workspace_id = CAST(:w AS uuid) AND resource_type = 'follow_up' "
                          "ORDER BY id"), {"w": team.ws}).fetchall()


def _refused(reply, code):
    return reply.get("success") is False and reply.get("error_code") == code


# --------------------------------------------------------------------------- the lock


def test_six_schedules_at_once_with_a_pending_cap_of_five_make_exactly_five(team):
    from modules.proactive.follow_up_caps import CapBroken, FollowUpCaps
    from services.proactive.follow_ups import FollowUpRequest, schedule_follow_up

    now = datetime.now(timezone.utc)
    request = FollowUpRequest(workspace_id=UUID(team.ws), user_id=team.ana, chat_id=UUID(team.chat),
                              agent_id=team.auto, at=now + timedelta(hours=1), reason=REASON)
    sessions = [team.new_session() for _ in range(6)]
    start, results, errors = threading.Barrier(6), [], []

    def schedule(session):
        start.wait()
        try:
            results.append(schedule_follow_up(session, request, FollowUpCaps(max_pending=5), now))
        except Exception as error:  # noqa: BLE001 — reported by the assertion below
            errors.append(error)

    threads = [threading.Thread(target=schedule, args=(session,)) for session in sessions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads) and errors == []
    assert len(_rows(team)) == 5
    assert sorted(type(result).__name__ for result in results) == ["CapBroken"] + ["int"] * 5
    assert [result for result in results if isinstance(result, CapBroken)] == [CapBroken("max_pending", 5)]
    assert len(team.registered) == 5 and len(_audits(team)) == 5


# --------------------------------------------------------------------------- the row, from the server only


def test_a_follow_up_is_one_row_of_auto_for_the_person_in_their_chat(team):
    reply = _schedule(team, about={"kind": "other", "id": "lisbon-lease"})

    assert reply["success"] is True and isinstance(reply["follow_up_id"], int)
    (row,) = _rows(team)
    assert (row.id, row.deliver_as, row.task_type, row.status) == (reply["follow_up_id"], "follow_up", "one_shot",
                                                                    "active")
    assert (row.created_by_agent_id, row.target_agent_id, row.for_user_id) == (team.auto, team.auto, team.ana)
    assert (str(row.origin_chat_id), row.description, row.created_by_user_id) == (team.chat, REASON, None)
    assert row.payload == {"about": {"kind": "other", "id": "lisbon-lease"}}
    assert row.schedule == reply["at"] and row.next_run_at == datetime.fromisoformat(reply["at"])
    assert team.registered == [(row.id, "one_shot", row.schedule, team.auto)]


def test_the_person_and_the_chat_come_from_the_caller_context_and_the_models_are_ignored(team):
    forged = {"reason": REASON, "in_minutes": 30, "_agent_id": team.auto, "chat_id": team.ben_chat,
              "user_id": team.ben, "for_user_id": team.ben, "origin_chat_id": team.ben_chat,
              "_follow_up": {"user_id": team.ben, "chat_id": team.ben_chat}}

    reply = _execute(team, SCHEDULE, forged, _context(team))

    assert reply["success"] is True, reply
    (row,) = _rows(team)
    assert (row.for_user_id, str(row.origin_chat_id)) == (team.ana, team.chat)


def test_with_no_caller_context_a_forged_follow_up_reaches_nobody(team):
    forged = {"reason": REASON, "in_minutes": 30, "_agent_id": team.auto,
              "_follow_up": {"user_id": team.ana, "chat_id": team.chat}}

    reply = _execute(team, SCHEDULE, forged, None)

    assert _refused(reply, "no_person") and _rows(team) == []


def test_a_check_ins_call_is_made_for_nobody_even_past_the_lane(team):
    """The second wall (K10): a follow-up's own turn runs in the check-in lane, and a call from it
    that reached the handler some other way than the executor would still schedule nothing."""
    from modules.tools.discovery.follow_up_context import follow_up_params

    lane = {**_context(team), "lane": "check_in"}
    params = follow_up_params(SCHEDULE, {"reason": REASON, "in_minutes": 30, "_agent_id": team.auto}, lane)

    assert params["_follow_up"]["user_id"] is None
    assert _refused(_call(team, SCHEDULE, params), "no_person") and _rows(team) == []


def test_the_follow_up_params_strip_a_forged_one_from_every_call():
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.follow_up_context import FOLLOW_UP_PARAM, follow_up_params

    forged = {"reason": "x", FOLLOW_UP_PARAM: {"user_id": 9, "chat_id": "theirs"}}
    context = {"driving_user_id": "7", "conversation_id": "c-7", "locale": "pt-BR"}

    assert follow_up_params(SCHEDULE, forged, context) == {
        "reason": "x", FOLLOW_UP_PARAM: {"user_id": 7, "chat_id": "c-7", "locale": "pt-BR"}}
    assert follow_up_params(LIST, json.dumps(forged), {}) == {
        "reason": "x", FOLLOW_UP_PARAM: {"user_id": None, "chat_id": None, "locale": None}}
    assert follow_up_params("platform_list_agents", forged, context) == {"reason": "x"}
    with turn_surface(WIDGET):
        assert follow_up_params(CANCEL, forged, context)[FOLLOW_UP_PARAM]["user_id"] is None


# --------------------------------------------------------------------------- who may schedule (K10)


@pytest.mark.parametrize("who, code", [
    ("no person", "no_person"), ("no conversation", "no_conversation"), ("another's chat", "no_conversation"),
    ("not in the workspace", "no_person"), ("another agent", "not_auto"), ("another workspace's Auto", "not_auto"),
])
def test_outside_a_persons_own_chat_turn_or_from_another_agent_nothing_is_written(team, who, code):
    from modules.tools.discovery.follow_up_context import follow_up_params

    sent = {"reason": REASON, "in_minutes": 30, "_agent_id": team.auto}
    params = {
        "no person": lambda: follow_up_params(SCHEDULE, sent, {"conversation_id": team.chat}),
        "no conversation": lambda: follow_up_params(SCHEDULE, sent, {"driving_user_id": str(team.ana)}),
        "another's chat": lambda: _params(team, chat=team.ben_chat, reason=REASON, in_minutes=30),
        "not in the workspace": lambda: _params(team, user=team.kim, reason=REASON, in_minutes=30),
        "another agent": lambda: _params(team, agent=team.vector, reason=REASON, in_minutes=30),
        "another workspace's Auto": lambda: _params(team, agent=team.other_auto, reason=REASON, in_minutes=30),
    }[who]()

    assert _refused(_call(team, SCHEDULE, params), code)
    assert _rows(team) == [] and _audits(team) == [] and team.registered == []


@pytest.mark.parametrize("mode", ["shadow", "off"])
def test_in_shadow_or_off_nothing_is_promised(team, mode):
    team.mode["value"] = mode

    assert _refused(_schedule(team), "not_live")
    assert _rows(team) == [] and _audits(team) == []


def test_a_workspace_with_its_switch_off_is_not_live(team):
    s = team.new_session()
    s.execute(text("UPDATE workspaces SET settings = CAST('{}' AS jsonb) WHERE id = CAST(:w AS uuid)"), {"w": team.ws})
    s.commit()

    assert _refused(_schedule(team), "not_live") and _rows(team) == []


def test_while_the_person_is_paused_nothing_is_scheduled(team):
    s = team.new_session()
    s.execute(text("INSERT INTO proactive_person_settings (workspace_id, user_id, paused) "
                   "VALUES (CAST(:w AS uuid), :u, true)"), {"w": team.ws, "u": team.ana})
    s.commit()

    assert _refused(_schedule(team), "paused")
    assert _rows(team) == [] and _audits(team) == []


# --------------------------------------------------------------------------- use a watch, and the arguments


@pytest.mark.parametrize("over", [
    {"about": {"kind": "mission", "id": "7f1c"}}, {"about": {"kind": "playbook_run", "id": "41"}},
    {"about": {"kind": "ticket", "id": "0042"}}, {"about": {"kind": "other", "id": "#0042"}},
    {"reason": "See whether #0042 got its review"},
    {"reason": "Check ticket 42 again after lunch"}, {"reason": "Look at step #0051.3 once the run ends"},
])
def test_a_mission_a_playbook_run_or_a_ticket_is_told_to_use_a_watch(team, over):
    from services.proactive.follow_up_strings import say

    reply = _schedule(team, **over)

    assert _refused(reply, "use_a_watch") and reply["error"] == say("en", "use_a_watch")
    assert _rows(team) == []


def test_a_reason_with_times_and_counts_is_no_ticket():
    from modules.tools.discovery.handlers_follow_ups import names_a_ticket

    assert not names_a_ticket("Check at 15:10 whether the 3 quotes came in; #1st priority is the landlord")


@pytest.mark.parametrize("over, code", [
    ({"reason": "  "}, "no_reason"), ({"reason": "x" * 501}, "long_reason"),
    ({"about": {"kind": "invoice"}}, "bad_about"), ({"in_minutes": None}, "no_time"),
    ({"at": "2026-10-30T15:10:00Z"}, "both_times"), ({"in_minutes": None, "at": "2026-10-30T15:10:00"}, "naive_time"),
    ({"in_minutes": "soon"}, "not_a_time"), ({"in_minutes": 9}, "follow_up_cap"),
    ({"in_minutes": 7 * 24 * 60 + 1}, "follow_up_cap"),
])
def test_an_argument_it_cannot_keep_is_refused_with_nothing_written(team, over, code):
    assert _refused(_schedule(team, **over), code)
    assert _rows(team) == []


def test_a_nul_in_the_reason_is_dropped_before_it_is_kept(team):
    assert _schedule(team, reason="Check the lease\x00 again")["success"] is True
    assert [row.description for row in _rows(team)] == ["Check the lease again"]


def test_a_cap_says_its_name_and_value(team):
    reply = _schedule(team, minutes=9)

    assert (reply["cap"], reply["limit"]) == ("min_minutes", 10)
    assert "min_minutes" in reply["error"] and "10" in reply["error"]


def test_the_sixth_pending_follow_up_is_refused_by_its_cap(team):
    for _ in range(5):
        assert _schedule(team)["success"] is True

    reply = _schedule(team)

    assert _refused(reply, "follow_up_cap") and (reply["cap"], reply["limit"]) == ("max_pending", 5)
    assert len(_rows(team)) == 5


# --------------------------------------------------------------------------- list and cancel, the person's own


def _seed(team, user, chat, status="active", minutes=60):
    s = team.new_session()
    at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    row = s.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, task_type, "
        "description, schedule, next_run_at, origin_chat_id, deliver_as, for_user_id, status) VALUES "
        "(CAST(:w AS uuid), :a, :a, 'one_shot', :r, :at_text, :at, CAST(:c AS uuid), 'follow_up', :u, :s) "
        "RETURNING id"), {"w": team.ws, "a": team.auto, "r": f"{user}'s reason", "at_text": at.isoformat(),
                          "at": at, "c": chat, "u": user, "s": status}).scalar()
    s.commit()
    return int(row)


def test_the_list_shows_only_the_callers_own(team):
    mine, fired = _seed(team, team.ana, team.chat), _seed(team, team.ana, team.chat, status="completed")
    _seed(team, team.ben, team.ben_chat)

    pending = _call(team, LIST, _for(team, LIST))
    every = _call(team, LIST, _for(team, LIST, status="all"))

    assert pending["success"] is True and [item["id"] for item in pending["follow_ups"]] == [mine]
    assert {item["id"]: item["status"] for item in every["follow_ups"]} == {mine: "pending", fired: "fired"}
    assert pending["follow_ups"][0]["chat_id"] == team.chat and pending["note"]


def test_the_service_lists_one_chats_follow_ups_when_it_names_the_chat(team):
    """The filter the follow-up routes read (US-C11e): one chat of the person's, and no one else's."""
    from services.proactive.follow_ups import list_follow_ups

    s = team.new_session()
    second = _chat(s, team.ws, team.ana)
    s.commit()
    here, there = _seed(team, team.ana, team.chat), _seed(team, team.ana, second)
    _seed(team, team.ben, team.chat)

    listed = list_follow_ups(team.new_session(), UUID(team.ws), team.ana, chat_id=UUID(second))

    assert [item.id for item in listed] == [there] and here != there
    assert [item.id for item in list_follow_ups(team.new_session(), UUID(team.ws), team.ana)] == [here, there]


def test_cancelling_another_persons_is_not_found_and_changes_nothing(team):
    bens = _seed(team, team.ben, team.ben_chat)

    reply = _call(team, CANCEL, _for(team, CANCEL, follow_up_id=bens))

    assert _refused(reply, "not_found")
    assert [row.status for row in _rows(team)] == ["active"] and _audits(team) == [] and team.unscheduled == []


def test_cancelling_ones_own_cancels_it_once(team):
    mine = _seed(team, team.ana, team.chat)

    first = _call(team, CANCEL, _for(team, CANCEL, follow_up_id=mine))
    again = _call(team, CANCEL, _for(team, CANCEL, follow_up_id=str(mine)))

    assert first["success"] is True and first["status"] == "cancelled"
    assert _refused(again, "already_cancelled")
    assert [row.status for row in _rows(team)] == ["cancelled"] and team.unscheduled == [(mine, "cancelled")]


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_fired_one_is_not_cancelled_and_lists_as_fired(team, status):
    fired = _seed(team, team.ana, team.chat, status=status)

    assert _refused(_call(team, CANCEL, _for(team, CANCEL, follow_up_id=fired)), "already_fired")
    listed = _call(team, LIST, _for(team, LIST, status="all"))["follow_ups"]
    assert [(item["id"], item["status"]) for item in listed] == [(fired, "fired")]


@pytest.mark.parametrize("follow_up_id", [None, "", "soon", "\u00b2", 0, -3, True, "1" * 19])
def test_a_follow_up_id_that_is_no_number_is_refused_before_any_read(team, follow_up_id):
    assert _refused(_call(team, CANCEL, _for(team, CANCEL, follow_up_id=follow_up_id)), "bad_id")


def test_every_follow_up_lists_the_pending_ones_first(team):
    old = _seed(team, team.ana, team.chat, status="cancelled", minutes=10)
    later, sooner = _seed(team, team.ana, team.chat, minutes=120), _seed(team, team.ana, team.chat, minutes=60)

    listed = _call(team, LIST, _for(team, LIST, status="all"))["follow_ups"]

    assert [item["id"] for item in listed] == [sooner, later, old]


@pytest.mark.parametrize("action", [LIST, CANCEL])
def test_list_and_cancel_are_refused_for_nobody_and_for_another_agent(team, action):
    mine = _seed(team, team.ana, team.chat)
    nobody = _call(team, action, {"follow_up_id": mine})
    other = _call(team, action, _for(team, action, agent=team.vector, follow_up_id=mine))

    assert _refused(nobody, "no_person") and _refused(other, "not_auto")
    assert [row.status for row in _rows(team)] == ["active"]


# --------------------------------------------------------------------------- the audit rows


def test_schedule_and_cancel_each_write_one_audit_row_without_the_reason(team):
    scheduled = _schedule(team)
    _call(team, CANCEL, _for(team, CANCEL, follow_up_id=scheduled["follow_up_id"]))

    audits = _audits(team)
    assert [(a.action, a.user_id, a.actor_type, a.resource_id) for a in audits] == [
        ("proactive:follow_up_scheduled", team.ana, "agent", str(scheduled["follow_up_id"])),
        ("proactive:follow_up_cancelled", team.ana, "agent", str(scheduled["follow_up_id"])),
    ]
    assert audits[0].details == {"at": scheduled["at"], "agent_id": team.auto, "chat_id": team.chat}
    assert all(REASON not in json.dumps(a.details) and a.resource_name is None for a in audits)


# --------------------------------------------------------------------------- registered, and chat only


def test_the_three_tools_are_registered_handled_and_on_the_hierarchy_gates_allow_list():
    import importlib.util
    from pathlib import Path

    from modules.tools.discovery import handlers_follow_ups
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    registry = get_action_registry()
    levels = {name: (registry.get(name).permission_level, registry.get(name).category, registry.get(name).promoted,
                     registry.get(name).examples) for name in (SCHEDULE, LIST, CANCEL)}
    assert levels == {SCHEDULE: ("write", "proactive", True, []), LIST: ("read", "proactive", False, []),
                      CANCEL: ("write", "proactive", False, [])}
    assert registry.get(SCHEDULE).parameters["required"] == ["reason"]
    assert (PLATFORM_HANDLERS[SCHEDULE], PLATFORM_HANDLERS[LIST], PLATFORM_HANDLERS[CANCEL]) == (
        handlers_follow_ups.schedule_follow_up, handlers_follow_ups.list_follow_ups,
        handlers_follow_ups.cancel_follow_up)
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_hierarchy_gate.py"
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate_prd268w3", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert {SCHEDULE, CANCEL} <= gate.ALLOW_LIST and LIST not in gate.ALLOW_LIST


def test_the_check_in_surface_offers_none_of_the_chat_only_tools_and_the_lane_refuses_them():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.check_in_lane import (
        CHAT_ONLY_ACTIONS, check_in_refusal, may_clear_in_a_check_in, offered_in_a_check_in,
    )
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    assert CHAT_ONLY_ACTIONS == {SCHEDULE, LIST, CANCEL, "platform_offer_replies"}
    registry = get_action_registry()
    for name in CHAT_ONLY_ACTIONS:
        action = registry.get(name)
        assert action is None or not offered_in_a_check_in(action), name
        assert not may_clear_in_a_check_in(name), name
    executor = PlatformActionExecutor(None, uuid.uuid4())
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True})
    executor._handlers = {**executor._handlers, **{name: handler for name in CHAT_ONLY_ACTIONS}}
    for name in CHAT_ONLY_ACTIONS:
        context = {"lane": "check_in", "driving_user_id": "7", "conversation_id": "c-7"}
        assert asyncio.run(executor.execute(name, {"reason": REASON}, context)) == check_in_refusal()
    handler.assert_not_called()

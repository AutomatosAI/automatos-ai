"""PRD-268 US-C07a (§6.9, FR-13, K7) — switches and caps: the platform mode, the workspace
switch, a person's pause, and the §6.9 caps, a workspace's own over the config defaults.

On Postgres, inside a rolled-back transaction (the runner, the suggestion tool and the routes
commit inside it, by SAVEPOINT). Proven here:

* **the switches** (``services.proactive.switches``): a workspace's mode is the platform's
  where it switched check-ins on, else off; no platform row is ``PROACTIVE_MODE_DEFAULT``;
  a settings read that fails skips the run (fail closed): the platform mode and the
  workspace switch (``skipped_off``), the owner's pause (``skipped_paused``) and the caps
  (``skipped_cap``, ``caps_unreadable``), each a run row and no model call;
* **the caps** (``services.proactive.limits``): config's defaults are the §6.9 numbers and
  every workspace's that set none; a workspace's own replace them one by one and can only
  tighten them (a looser stored cap is held to the platform's); each refuses with its
  outcome, as a ``skipped_cap`` run row at the start (the model runs per day, the minimum
  interval, a suggestion cap at 0) or in the suggestion tool (per run, new per person per
  day, open per agent, at the workspace's own values), where a run at a counting cap still
  goes ahead so a failing streak can end; past the open-per-person cap the person's oldest
  open suggestions expire, and an insert that loses its race expires nothing; the workspace's
  maximum interval is the longest a failing streak holds a check-in back; caps stored wrong
  refuse every check-in and every suggestion;
* **the routes**: owners and admins read and change ``settings.proactive`` (editors, viewers
  and strangers 403), in both editions; the rest of ``settings`` is kept; a cap sent as null
  is back at its default; a cap looser than the platform's, or caps that do not fit, are 422
  and nothing is saved; each change is audited once committed; any member reads and changes
  their own pause (a stranger 403, no person behind the caller 403), upserted and audited;
  plain ``def``s, in the route manifest, the cap settings in the config surface.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

ORCH = Path(__file__).resolve().parents[1]
MANIFEST = ORCH / "reports" / "route-manifest.json"
CONFIG_SURFACE = ORCH / "reports" / "config-surface.json"
SETTINGS = "/api/proactive/settings"
ME = "/api/proactive/me"
MODEL = "claude-haiku-5-5"
TITLE = "Ticket #0846 has been blocked for two days"
TITLE_KEY = "title:ticket #0846 has been blocked for two days"
TICKET_READ = {"type": "ticket", "id": "0846"}
DEFAULT_CAPS = {"max_model_runs_per_day": 48, "max_suggestions_per_run": 3, "max_new_per_person_per_day": 10,
                "max_open_per_person": 20, "max_open_per_agent": 3, "min_interval_minutes": 30,
                "max_interval_minutes": 1440}
DEFAULTS = {"enabled": True, "caps": DEFAULT_CAPS, "platform_caps": DEFAULT_CAPS}
CONFIG_NAMES = {name: f"PROACTIVE_{name.upper()}" for name in DEFAULT_CAPS}


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the runner, the tool and the routes commit, and none may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _person(db, tag: str) -> NS:
    clerk = f"user_prd268sw_{tag}_{uuid.uuid4().hex[:10]}"
    pk = db.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                    {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(pk=int(pk), clerk=clerk)


def _workspace(db, owner: NS, settings=None) -> uuid.UUID:
    ws = uuid.uuid4()
    db.execute(text(
        "INSERT INTO workspaces (id, name, slug, plan, owner_id, settings) VALUES (CAST(:id AS uuid), "
        "'prd268 switches', :slug, 'pro', :o, CAST(:settings AS jsonb))"),
        {"id": str(ws), "slug": f"prd268-switches-{ws.hex[:10]}", "o": owner.pk,
         "settings": json.dumps(settings or {})})
    return ws


def _member(db, ws: uuid.UUID, role: str) -> NS:
    person = _person(db, role)
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"), {"ws": str(ws), "uid": person.pk, "role": role})
    return person


def _agent(db, ws: uuid.UUID) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('VECTOR', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar())


@pytest.fixture
def team(db_session) -> NS:
    """A workspace with check-ins switched on, its owner, an admin, an editor, a viewer and an
    agent; another workspace and its owner."""
    owner = _person(db_session, "owner")
    ws = _workspace(db_session, owner, {"proactive": {"enabled": True}, "theme": "dusk"})
    other_owner = _person(db_session, "other")
    team = NS(db=db_session, ws=ws, owner=owner, admin=_member(db_session, ws, "admin"),
              editor=_member(db_session, ws, "editor"), viewer=_member(db_session, ws, "viewer"),
              agent=_agent(db_session, ws), other_owner=other_owner, other_ws=_workspace(db_session, other_owner))
    db_session.commit()
    return team


def _set_settings(team, settings: dict) -> None:
    team.db.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:w AS uuid)"),
                    {"s": json.dumps(settings), "w": str(team.ws)})
    team.db.commit()


def _caps(team, **caps) -> None:
    _set_settings(team, {"proactive": {"enabled": True, "caps": caps}})


def _stored(team) -> dict:
    team.db.expire_all()
    return team.db.execute(text("SELECT settings FROM workspaces WHERE id = CAST(:w AS uuid)"),
                           {"w": str(team.ws)}).scalar()


def _run_row(team, *, outcome: str, minutes_ago: float, model=None, agent=None, error_class=None) -> None:
    at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    team.db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, agent_id, kind, outcome, model, error_class, "
        "window_until, created_at) VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :agent, :kind, :outcome, "
        ":model, :error, :at, :at)"),
        {"id": str(uuid.uuid4()), "ws": str(team.ws), "agent": agent, "outcome": outcome, "model": model,
         "error": error_class, "at": at, "kind": "workspace_check_in" if agent is None else "agent_check_in"})
    team.db.commit()


def _suggestion(team, key: str, *, status: str = "open", agent=None, user=None, ws=None,
                hours_ago: float = 0) -> uuid.UUID:
    sid, at = uuid.uuid4(), datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    team.db.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, source_agent_id, title, "
        "proposed_action, subject_key, status, created_at, last_seen_at) VALUES (CAST(:id AS uuid), "
        "CAST(:ws AS uuid), :user, 'check_in', :agent, :title, CAST(:action AS jsonb), :key, :status, :at, :at)"),
        {"id": str(sid), "ws": str(ws or team.ws), "user": user or team.owner.pk, "agent": agent, "title": key,
         "action": json.dumps({"kind": "chat", "payload": {}}), "key": key, "status": status, "at": at})
    team.db.commit()
    return sid


def _suggestion_row(team, sid: uuid.UUID):
    from core.models.proactive import ProactiveSuggestion

    team.db.expire_all()
    return team.db.get(ProactiveSuggestion, sid)


def _audit(team, resource_type: str) -> list:
    return [tuple(row) for row in team.db.execute(text(
        "SELECT action, user_id, details FROM audit_logs WHERE workspace_id = CAST(:w AS uuid) "
        "AND resource_type = :rt ORDER BY id"), {"w": str(team.ws), "rt": resource_type}).fetchall()]


# ── The check-in's start: the runner, refused before the model ─────────────

@pytest.fixture
def checking(team, monkeypatch):
    """The runner on the test's session, in the local edition (credits pass), the platform
    mode shadow unless a test sets it; a check-in that got past its start and reached the
    model would be recorded as an error."""
    from config import config
    from core.llm import manager
    from services.proactive import runner

    rig = NS(mode="shadow", conversed=[])

    def setting(category, key):
        if rig.mode is RuntimeError:
            raise RuntimeError("system_settings unreadable")
        return rig.mode if (category, key) == ("proactive", "mode") else None

    async def converse(db, plan, now):
        rig.conversed.append(plan)
        raise AssertionError("the check-in reached the model")

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "SPEND_CEILING_USD", 0)
    monkeypatch.setattr(manager, "read_system_setting", setting)
    monkeypatch.setattr(runner, "_session", lambda: nullcontext(team.db))
    monkeypatch.setattr(runner, "_converse", converse)
    monkeypatch.setattr(runner, "read_signals", lambda db, workspace_id, since, until: ())
    monkeypatch.setattr(runner, "heartbeat_llm", lambda workspace_id, agent_id: NS(config=NS(model=MODEL)))
    return rig


def _workspace_check_in(team):
    from services.proactive import runner

    return asyncio.run(runner.run_workspace_check_in(team.ws))


def _agent_check_in(team):
    from services.proactive import runner

    return asyncio.run(runner.run_agent_check_in(team.agent, team.ws, {"enabled": True, "interval_minutes": 60}))


def _refused(team, checking, result, outcome: str, reason: str) -> None:
    """Refused at its start: its run row says why, and nothing reached the model."""
    from core.models.proactive import ProactiveRun

    assert (result.outcome, result.error_class, result.suggestions) == (outcome, reason, 0)
    team.db.expire_all()
    run = team.db.get(ProactiveRun, result.run_id)
    assert (run.outcome, run.error_class, run.model, run.tokens) == (outcome, reason, None, 0)
    assert checking.conversed == []


def _goes_ahead(team, agent_id=None) -> bool:
    from services.proactive.gates import start_refusal

    return start_refusal(team.db, team.ws, team.owner.pk, datetime.now(timezone.utc), "a test", agent_id) is None


def _failing_reads(db, monkeypatch, entity) -> None:
    """The session's reads of ``entity`` fail as a lost connection would; every other read runs."""
    real = db.query

    def query(*entities, **kwargs):
        if any(candidate is entity for candidate in entities):
            raise OperationalError("SELECT", {}, ConnectionError("server closed the connection unexpectedly"))
        return real(*entities, **kwargs)

    monkeypatch.setattr(db, "query", query)


# ── The switches (FR-13, K7) ────────────────────────────────────────────────

@pytest.mark.parametrize("platform,enabled,mode", [
    ("shadow", True, "shadow"), ("live", True, "live"), ("live", False, "off"), ("off", True, "off"),
    ("sideways", True, "off"),
])
def test_a_workspaces_mode_is_the_platforms_where_it_switched_check_ins_on(team, checking, platform, enabled, mode):
    from services.proactive.switches import effective_mode

    checking.mode = platform
    _set_settings(team, {"proactive": {"enabled": enabled}})
    assert effective_mode(team.db, team.ws) == mode


def test_with_no_platform_row_the_mode_is_the_config_default(team, checking, monkeypatch):
    from config import config
    from services.proactive.switches import effective_mode

    checking.mode = None
    assert config.PROACTIVE_MODE_DEFAULT == "off" and effective_mode(team.db, team.ws) == "off"
    monkeypatch.setattr(config, "PROACTIVE_MODE_DEFAULT", "shadow")
    assert effective_mode(team.db, team.ws) == "shadow"


@pytest.mark.parametrize("settings", [{}, {"proactive": {}}, {"proactive": {"enabled": "true"}}, {"proactive": True}])
def test_the_workspace_switch_is_off_unless_it_is_true(team, checking, settings):
    from services.proactive.switches import effective_mode

    _set_settings(team, settings)
    assert effective_mode(team.db, team.ws) == "off"
    _refused(team, checking, _workspace_check_in(team), "skipped_off", "switched_off")


def _unreadable_caps(db, workspace_id):
    raise OperationalError("SELECT settings", {}, ConnectionError("server closed the connection"))


def test_a_settings_read_error_skips_the_run_fail_closed(team, checking, monkeypatch):
    from core.models.proactive import ProactivePersonSettings
    from core.models.workspaces import Workspace
    from services.proactive import gates

    checking.mode = RuntimeError
    _refused(team, checking, _agent_check_in(team), "skipped_off", "switched_off")
    checking.mode = "shadow"
    with monkeypatch.context() as patched:
        _failing_reads(team.db, patched, Workspace.settings)
        _refused(team, checking, _agent_check_in(team), "skipped_off", "switched_off")
    with monkeypatch.context() as patched:
        _failing_reads(team.db, patched, ProactivePersonSettings.paused)
        _refused(team, checking, _agent_check_in(team), "skipped_paused", "paused")
    monkeypatch.setattr(gates, "workspace_caps", _unreadable_caps)
    _refused(team, checking, _agent_check_in(team), "skipped_cap", "caps_unreadable")


def test_a_paused_owner_is_skipped_and_resuming_lets_the_check_in_start(team, checking):
    assert _as(team, team.owner).put(ME, json={"paused": True}).json() == {"paused": True}
    _refused(team, checking, _workspace_check_in(team), "skipped_paused", "paused")

    assert _as(team, team.owner).put(ME, json={"paused": False}).json() == {"paused": False}
    assert _goes_ahead(team)


def test_another_members_pause_does_not_hold_the_owners_check_ins(team, checking):
    from services.proactive.switches import person_paused

    _as(team, team.viewer).put(ME, json={"paused": True})

    assert person_paused(team.db, team.ws, team.viewer.pk) and not person_paused(team.db, team.ws, team.owner.pk)
    assert not person_paused(team.db, team.other_ws, team.viewer.pk)
    assert _goes_ahead(team)


# ── The caps (§6.9) ─────────────────────────────────────────────────────────

def test_the_config_defaults_are_the_prd_caps_and_every_workspaces_that_set_none(team, monkeypatch):
    from config import config
    from modules.proactive import Caps
    from services.proactive.limits import EDITABLE_CAPS, caps_as_dict, default_caps, workspace_caps

    assert set(EDITABLE_CAPS) == set(DEFAULT_CAPS)
    assert {name: getattr(config, CONFIG_NAMES[name]) for name in DEFAULT_CAPS} == DEFAULT_CAPS
    assert caps_as_dict(workspace_caps(team.db, team.ws)) == DEFAULT_CAPS

    monkeypatch.setattr(config, "PROACTIVE_MAX_MODEL_RUNS_PER_DAY", 6)
    monkeypatch.setattr(config, "PROACTIVE_MIN_INTERVAL_MINUTES", 45)
    assert (default_caps().max_model_runs_per_day, workspace_caps(team.db, team.ws).min_interval_minutes) == (6, 45)
    assert Caps().max_model_runs_per_day == 48  # the pure rules keep the PRD's numbers


def test_a_workspaces_own_caps_replace_the_defaults_one_by_one(team):
    from services.proactive.limits import caps_as_dict, workspace_caps

    _caps(team, max_model_runs_per_day=5, max_open_per_agent=1, not_a_cap=9)
    assert caps_as_dict(workspace_caps(team.db, team.ws)) == {
        **DEFAULT_CAPS, "max_model_runs_per_day": 5, "max_open_per_agent": 1}


def test_a_workspace_can_only_tighten_the_platforms_caps(team, monkeypatch):
    from config import config
    from services.proactive.limits import caps_as_dict, workspace_caps

    _caps(team, max_model_runs_per_day=1000, min_interval_minutes=5, max_interval_minutes=600, max_open_per_agent=2)
    assert caps_as_dict(workspace_caps(team.db, team.ws)) == {
        **DEFAULT_CAPS, "max_interval_minutes": 600, "max_open_per_agent": 2}

    monkeypatch.setattr(config, "PROACTIVE_MAX_OPEN_PER_AGENT", 1)  # the platform tightened since
    assert workspace_caps(team.db, team.ws).max_open_per_agent == 1


def test_a_workspaces_own_model_run_cap_refuses_with_skipped_cap(team, checking):
    _caps(team, max_model_runs_per_day=2)
    _run_row(team, outcome="silent", model=MODEL, minutes_ago=300)
    assert _goes_ahead(team)
    _run_row(team, outcome="suggested", model=MODEL, minutes_ago=200)

    _refused(team, checking, _workspace_check_in(team), "skipped_cap", "max_model_runs_per_day")


@pytest.mark.parametrize("minutes_ago,held", [(1, True), (20, True), (26, False), (60, False)])
def test_a_check_in_sooner_than_the_minimum_interval_after_its_last_is_refused(team, checking, minutes_ago, held):
    _run_row(team, outcome="error", agent=team.agent, minutes_ago=minutes_ago)

    if held:
        _refused(team, checking, _agent_check_in(team), "skipped_cap", "min_interval_minutes")
    else:  # within the scheduler's five minutes of grace, the fire is the one due
        assert _goes_ahead(team, team.agent)


def test_skipped_runs_never_move_the_interval_on_and_each_check_in_keeps_its_own(team, checking):
    _run_row(team, outcome="silent", agent=team.agent, minutes_ago=45)
    _run_row(team, outcome="skipped_cap", agent=team.agent, minutes_ago=1)
    assert _goes_ahead(team, team.agent)

    _run_row(team, outcome="suggested", agent=team.agent, minutes_ago=2)
    assert _goes_ahead(team)  # Auto's workspace check-in is its own check-in
    _refused(team, checking, _agent_check_in(team), "skipped_cap", "min_interval_minutes")


def test_a_workspaces_own_minimum_interval_is_the_one_held(team, checking):
    _run_row(team, outcome="silent", minutes_ago=90)
    assert _goes_ahead(team)

    _caps(team, min_interval_minutes=120)
    _refused(team, checking, _workspace_check_in(team), "skipped_cap", "min_interval_minutes")


def test_a_check_in_at_a_counting_cap_still_runs_and_the_tool_refuses_its_suggestion(team, checking):
    _caps(team, max_new_per_person_per_day=4)
    _suggestion(team, f"agent:{team.agent}:error:http_401", agent=team.agent)  # its failing streak's card
    for n in range(2):
        _suggestion(team, f"title:agent finding {n}", agent=team.agent)

    assert _goes_ahead(team, team.agent)  # the read-only run goes ahead: a run that works ends the streak
    reply = _suggest(team, TITLE, _check_in(agent=team.agent))
    assert reply["error_code"] == "suggestion_cap" and "the max_open_per_agent cap is 3" in reply["error"]

    _suggestion(team, "title:yesterday", hours_ago=30)
    _suggestion(team, "title:hidden today", status="hidden")
    assert _goes_ahead(team)
    reply = _suggest(team, TITLE, _check_in())
    assert reply["error_code"] == "suggestion_cap" and "the max_new_per_person_per_day cap is 4" in reply["error"]


@pytest.mark.parametrize("cap", ["max_suggestions_per_run", "max_open_per_person", "max_model_runs_per_day",
                                 "max_new_per_person_per_day"])
def test_a_cap_of_zero_refuses_every_check_in(team, checking, cap):
    _caps(team, **{cap: 0})
    _refused(team, checking, _workspace_check_in(team), "skipped_cap", cap)


def test_an_open_per_agent_cap_of_zero_refuses_the_agents_check_ins_and_not_autos(team, checking):
    _caps(team, max_open_per_agent=0)

    _refused(team, checking, _agent_check_in(team), "skipped_cap", "max_open_per_agent")
    assert _goes_ahead(team)
    assert _suggest(team, TITLE, _check_in())["success"] is True  # Auto's suggestions have no source agent


def test_the_workspaces_maximum_interval_is_the_longest_a_failing_streak_holds_back(team, monkeypatch):
    from services.proactive import failing_streaks

    for minutes_ago in (200, 140, 88):  # the third failure in a row: the hour doubles to two
        _run_row(team, outcome="error", agent=team.agent, error_class="http_401", minutes_ago=minutes_ago)
    hb, now = {"interval_minutes": 60}, datetime.now(timezone.utc)

    assert failing_streaks.held_back(team.db, team.ws, team.agent, hb, now).reason == "failing_streak"
    _caps(team, max_interval_minutes=90)
    assert failing_streaks.ceiling_minutes(team.db, team.ws) == 90
    assert failing_streaks.held_back(team.db, team.ws, team.agent, hb, now) is None

    monkeypatch.setattr(failing_streaks, "workspace_caps", _unreadable_caps)
    assert failing_streaks.ceiling_minutes(team.db, team.ws) == 24 * 60  # the longest hold: the fewest runs


@pytest.mark.parametrize("wrong", [
    {"min_interval_minutes": 120, "max_interval_minutes": 60}, {"max_model_runs_per_day": -1},
    {"max_open_per_agent": "3"}, {"min_interval_minutes": 0}, [48],
])
def test_caps_stored_wrong_refuse_every_check_in(team, checking, wrong):
    _set_settings(team, {"proactive": {"enabled": True, "caps": wrong}})
    _refused(team, checking, _workspace_check_in(team), "skipped_cap", "caps_unreadable")


# ── The caps in the suggestion tool ─────────────────────────────────────────

def _check_in(run=None, agent=None) -> dict:
    return {"run_id": str(run or uuid.uuid4()), "reads": [TICKET_READ], "agent_id": agent}


def _suggest(team, title: str, check_in: dict) -> dict:
    from modules.tools.discovery.check_in_lane import CHECK_IN_PARAM
    from modules.tools.discovery.handlers_proactive import suggest_next_step

    params = {"title": title, "detail": "VECTOR waits on the owner's answer.", "evidence": [TICKET_READ],
              "proposed_action": {"kind": "chat"}, CHECK_IN_PARAM: check_in}
    return asyncio.run(suggest_next_step(team.db, team.ws, params))


def test_the_suggestion_tool_holds_to_the_workspaces_own_caps(team):
    _caps(team, max_suggestions_per_run=1, max_open_per_agent=1)
    run = uuid.uuid4()

    assert _suggest(team, "Ticket #0846 is stuck", _check_in(run))["success"] is True
    reply = _suggest(team, "Ticket #0847 is stuck", _check_in(run))
    assert reply["error_code"] == "suggestion_cap" and "the max_suggestions_per_run cap is 1" in reply["error"]

    assert _suggest(team, "Ticket #0848 is stuck", _check_in(agent=team.agent))["success"] is True
    reply = _suggest(team, "Ticket #0849 is stuck", _check_in(agent=team.agent))
    assert reply["error_code"] == "suggestion_cap" and "the max_open_per_agent cap is 1" in reply["error"]


def test_caps_stored_wrong_record_no_suggestion(team):
    from modules.tools.discovery.handlers_proactive import CAPS_WRONG

    _caps(team, max_open_per_person=-2)

    reply = _suggest(team, TITLE, _check_in())
    assert (reply["success"], reply["error_code"], reply["error"]) == (False, "caps_wrong", CAPS_WRONG)
    assert team.db.execute(text("SELECT count(*) FROM proactive_suggestions WHERE workspace_id = CAST(:w AS uuid)"),
                           {"w": str(team.ws)}).scalar() == 0


def test_past_the_open_cap_the_persons_oldest_open_suggestions_expire(team):
    _caps(team, max_open_per_person=3)
    oldest = _suggestion(team, "title:oldest", hours_ago=50)
    kept = [_suggestion(team, "title:middle", hours_ago=20), _suggestion(team, "title:newest", hours_ago=1)]
    hidden = _suggestion(team, "title:hidden long ago", status="hidden", hours_ago=70)
    admins = _suggestion(team, "title:the admin's", user=team.admin.pk, hours_ago=90)
    elsewhere = _suggestion(team, "title:elsewhere", user=team.other_owner.pk, ws=team.other_ws, hours_ago=90)

    reply = _suggest(team, TITLE, _check_in())

    assert reply["success"] is True
    expired = _suggestion_row(team, oldest)
    assert expired.status == "expired" and expired.decided_at is None
    assert datetime.now(timezone.utc) - expired.expires_at < timedelta(minutes=1)
    assert [_suggestion_row(team, sid).status for sid in kept] == ["open", "open"]
    assert _suggestion_row(team, uuid.UUID(reply["data"]["suggestion_id"])).status == "open"
    assert [_suggestion_row(team, sid).status for sid in (hidden, admins, elsewhere)] == ["hidden", "open", "open"]


def test_an_insert_that_loses_its_race_expires_nothing(team, monkeypatch):
    from modules.tools.discovery import handlers_proactive

    _caps(team, max_open_per_person=2)
    oldest = _suggestion(team, "title:oldest", hours_ago=50)
    _suggestion(team, TITLE_KEY, hours_ago=1)  # a concurrent check-in's, inserted after the dedupe read
    monkeypatch.setattr(handlers_proactive, "_duplicate", lambda db, workspace_id, user_id, suggestion, now: None)

    reply = _suggest(team, TITLE, _check_in())

    assert reply["error_code"] == "already_suggested"
    assert _suggestion_row(team, oldest).status == "open"


# ── The routes ──────────────────────────────────────────────────────────────

@pytest.fixture
def saas(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


def _ctx(person, ws, system_role: str = "user"):
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role=system_role)
    return RequestContext(workspace_id=ws, user=user, auth_type="clerk")


def _client(db, ctx) -> TestClient:
    import api.proactive as proactive
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(proactive.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _as(team, person, system_role: str = "user") -> TestClient:
    return _client(team.db, _ctx(person, team.ws, system_role))


def test_owners_and_admins_read_the_settings_each_cap_at_its_default(team, saas):
    for person in (team.owner, team.admin):
        reply = _as(team, person).get(SETTINGS)
        assert reply.status_code == 200 and reply.json() == DEFAULTS


def test_the_local_operator_reads_and_changes_the_settings_and_its_own_pause(team, monkeypatch):
    from config import config
    from core.auth.dependencies import RequestContext, UserContext

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    operator = UserContext(id="operator@local.test", email="operator@local.test", system_role="super_admin",
                           raw_claims={"source": "local_operator", "user_id": team.owner.pk})
    client = _client(team.db, RequestContext(workspace_id=team.ws, user=operator, auth_type="anonymous"))

    assert client.get(SETTINGS).json() == DEFAULTS
    assert client.put(SETTINGS, json={"caps": {"max_open_per_agent": 2}}).json()["caps"]["max_open_per_agent"] == 2
    assert client.put(ME, json={"paused": True}).json() == {"paused": True}
    assert [(action, user_id) for action, user_id, _ in _audit(team, "workspace")] == [
        ("proactive:settings_changed", team.owner.pk)]
    assert [(action, user_id) for action, user_id, _ in _audit(team, "proactive_person_settings")] == [
        ("proactive:paused", team.owner.pk)]


@pytest.mark.parametrize("role", ["editor", "viewer", "stranger"])
def test_editors_viewers_and_strangers_neither_read_nor_change_the_settings(team, saas, role):
    person = team.other_owner if role == "stranger" else getattr(team, role)

    assert _as(team, person).get(SETTINGS).status_code == 403
    assert _as(team, person).put(SETTINGS, json={"enabled": False}).status_code == 403
    assert _stored(team)["proactive"] == {"enabled": True} and _audit(team, "workspace") == []


def test_a_change_writes_settings_proactive_keeps_the_rest_and_is_audited(team, saas):
    caps = {"max_model_runs_per_day": 12, "min_interval_minutes": 60}

    reply = _as(team, team.admin).put(SETTINGS, json={"enabled": False, "caps": caps})

    assert reply.status_code == 200
    assert reply.json() == {"enabled": False, "caps": {**DEFAULT_CAPS, **caps}, "platform_caps": DEFAULT_CAPS}
    assert _stored(team) == {"theme": "dusk", "proactive": {"enabled": False, "caps": caps}}
    [(action, user_id, details)] = _audit(team, "workspace")
    assert (action, user_id) == ("proactive:settings_changed", team.admin.pk)
    assert details["changed"] == {"enabled": False, "caps": caps}
    assert details["settings"] == reply.json()


def test_a_cap_sent_as_null_is_back_at_its_default_and_what_is_left_out_stays(team, saas):
    _caps(team, max_model_runs_per_day=12, max_open_per_agent=1)

    reply = _as(team, team.owner).put(SETTINGS, json={"caps": {"max_model_runs_per_day": None}})

    assert reply.json() == {**DEFAULTS, "caps": {**DEFAULT_CAPS, "max_open_per_agent": 1}}
    assert _stored(team) == {"proactive": {"enabled": True, "caps": {"max_open_per_agent": 1}}}


@pytest.mark.parametrize("body", [
    {"caps": {"min_interval_minutes": 120, "max_interval_minutes": 60}},
    {"caps": {"max_model_runs_per_day": 49}},
    {"caps": {"min_interval_minutes": 29}},
    {"caps": {"max_interval_minutes": 1441}},
    {"caps": {"max_model_runs_per_day": -1}},
    {"caps": {"max_model_runs_per_day": "48"}},
    {"caps": {"min_interval_minutes": 0}},
    {"caps": {"max_pushes_per_person_per_day": 5}},
    {"enabled": "yes"},
    {"mode": "live"},
])
def test_caps_that_do_not_fit_are_refused_and_nothing_is_saved(team, saas, body):
    reply = _as(team, team.owner).put(SETTINGS, json=body)

    assert reply.status_code == 422
    assert _stored(team)["proactive"] == {"enabled": True} and _audit(team, "workspace") == []


@pytest.mark.parametrize("caps,code", [
    ({"min_interval_minutes": 120, "max_interval_minutes": 60}, "caps_wrong"),
    ({"max_open_per_person": 21}, "caps_looser"),
])
def test_a_refused_change_is_said_in_the_callers_language(team, saas, caps, code):
    from services.proactive.decision_text import STRINGS

    reply = _as(team, team.owner).put(SETTINGS, json={"caps": caps}, headers={"Accept-Language": "pt-BR,pt;q=0.9"})

    assert reply.json() == {"code": code, "detail": STRINGS["pt-BR"][code]}
    assert set(STRINGS["en"]) == set(STRINGS["pt-BR"])


def test_caps_stored_wrong_are_shown_as_such_and_saving_them_again_fixes_them(team, saas):
    _caps(team, max_open_per_agent=-3)

    shown = _as(team, team.owner).get(SETTINGS)
    assert shown.status_code == 409 and shown.json()["code"] == "caps_stored_wrong"

    fixed = _as(team, team.owner).put(SETTINGS, json={"caps": {"max_open_per_agent": None}})
    assert fixed.status_code == 200 and fixed.json()["caps"] == DEFAULT_CAPS


@pytest.mark.parametrize("role", ["owner", "admin", "editor", "viewer"])
def test_any_member_pauses_and_resumes_their_own_suggestions(team, saas, role):
    person = getattr(team, role)
    client = _as(team, person)

    assert client.get(ME).json() == {"paused": False}
    assert client.put(ME, json={"paused": True}).json() == {"paused": True}
    assert client.get(ME).json() == {"paused": True}
    assert client.put(ME, json={"paused": False}).json() == {"paused": False}

    rows = team.db.execute(text("SELECT user_id, paused FROM proactive_person_settings "
                                "WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(team.ws)}).fetchall()
    assert [tuple(row) for row in rows] == [(person.pk, False)]  # one row, upserted
    audit = _audit(team, "proactive_person_settings")
    assert [(action, user_id) for action, user_id, _ in audit] == [
        ("proactive:paused", person.pk), ("proactive:resumed", person.pk)]


def test_a_stranger_or_a_caller_with_no_person_cannot_pause(team, saas):
    assert _as(team, team.other_owner).get(ME).status_code == 403
    assert _as(team, team.other_owner).put(ME, json={"paused": True}).status_code == 403

    nobody = NS(clerk=f"user_prd268sw_ghost_{uuid.uuid4().hex[:10]}")  # a platform admin with no users row
    refused = _as(team, nobody, system_role="super_admin").put(ME, json={"paused": True})
    assert refused.status_code == 403 and refused.json()["code"] == "no_person"
    assert _as(team, team.owner).put(ME, json={"paused": "yes"}).status_code == 422
    assert team.db.execute(text("SELECT count(*) FROM proactive_person_settings "
                                "WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(team.ws)}).scalar() == 0


def test_the_routes_are_plain_defs_mounted_and_in_the_manifest():
    import api.proactive as proactive
    from router_manifest import MANIFEST_ROUTERS, RouterSpec
    from services.proactive.limits import EDITABLE_CAPS

    for handler in (proactive.get_settings, proactive.put_settings, proactive.get_my_settings,
                    proactive.put_my_settings):
        assert not inspect.iscoroutinefunction(handler), handler.__name__
    assert RouterSpec("api.proactive") in MANIFEST_ROUTERS
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    served = {(method, route.path) for route in proactive.router.routes for method in route.methods}
    for method, path in (("GET", SETTINGS), ("PUT", SETTINGS), ("GET", ME), ("PUT", ME)):
        assert {"method": method, "path": path} in manifest["routes"] and (method, path) in served
    assert manifest["route_count"] == len(manifest["routes"])
    assert set(proactive.CapsChange.model_fields) == set(EDITABLE_CAPS)


def test_the_cap_settings_are_in_the_config_surface():
    names = json.loads(CONFIG_SURFACE.read_text(encoding="utf-8"))["settings"]

    assert set(CONFIG_NAMES.values()) <= set(names) and names == sorted(names)

"""PRD-268 US-C07b (§6.9, FR-11) — per-person quiet hours.

Proven here:

* **the window** (``services.proactive.quiet.is_person_quiet``): a window whose start is later
  than its end runs across midnight, quiet from its start up to (not at) its end; a daytime
  window is quiet only inside it; a window that starts where it ends is never quiet; the
  window is read on the person's wall clock, so across a change of clocks (Europe/Dublin,
  spring and autumn 2026) the same UTC hour is quiet one day and not the next; with no zone of
  their own it is read in the workspace's quiet-hours zone, else UTC, and a zone zoneinfo
  does not know is passed over; a naive moment is UTC;
* **the workspace's window on top**: it is quiet when either window is; a person with no
  window of their own is quiet only in the workspace's; a workspace window that is off adds
  nothing;
* **the route** (Postgres, a rolled-back transaction): any member sets, reads and clears their
  own quiet hours and timezone through ``PUT``/``GET /api/proactive/me``, in both editions;
  what a ``PUT`` leaves out stays (one upserted row); each person keeps their own; a quiet
  hours change is audited once committed; hours that do not fit are 422 and nothing is
  saved; the hours stored are the ones ``is_person_quiet`` reads, with the workspace window
  loaded by ``load_auto_reporting_settings``.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, time, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

ME = "/api/proactive/me"
DUBLIN = "Europe/Dublin"
SAO_PAULO = "America/Sao_Paulo"
NO_QUIET_HOURS = {"quiet_start": None, "quiet_end": None, "timezone": None}


def _at(stamp: str) -> datetime:
    """A UTC moment from ``YYYY-MM-DDTHH:MM``."""
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)


def _person(start=None, end=None, zone=None):
    from services.proactive.settings import PersonSettings

    return PersonSettings(quiet_start=start, quiet_end=end, timezone=zone)


def _workspace_window(start="22:00", end="08:00", zone=DUBLIN, enabled=True) -> dict:
    return {"quiet_hours": {"enabled": enabled, "start": start, "end": end, "timezone": zone}}


def _quiet(person, workspace, stamp: str) -> bool:
    from services.proactive.quiet import is_person_quiet

    return is_person_quiet(person, workspace, _at(stamp))


# ── The window ──────────────────────────────────────────────────────────────

NIGHT = _person(time(22, 0), time(7, 0), "UTC")


@pytest.mark.parametrize("hhmm,quiet", [
    ("21:59", False), ("22:00", True), ("23:30", True), ("00:00", True), ("03:00", True),
    ("06:59", True), ("07:00", False), ("12:00", False),
])
def test_a_window_across_midnight_is_quiet_from_its_start_up_to_its_end(hhmm, quiet):
    assert _quiet(NIGHT, {}, f"2026-01-15T{hhmm}") is quiet


@pytest.mark.parametrize("hhmm,quiet", [("12:59", False), ("13:00", True), ("13:59", True), ("14:00", False),
                                        ("23:00", False)])
def test_a_daytime_window_is_quiet_only_inside_it(hhmm, quiet):
    assert _quiet(_person(time(13, 0), time(14, 0), "UTC"), {}, f"2026-01-15T{hhmm}") is quiet


def test_a_window_that_starts_where_it_ends_is_never_quiet():
    same = _person(time(9, 0), time(9, 0), "UTC")

    assert not any(_quiet(same, {}, f"2026-01-15T{hh:02d}:00") for hh in range(24))


@pytest.mark.parametrize("stamp,quiet", [
    # Spring forward, Sunday 29 March 2026, 01:00 UTC: 06:30 UTC is 06:30 GMT the day before, 07:30 IST after.
    ("2026-03-28T06:30", True),
    ("2026-03-29T00:30", True),   # 00:30 GMT, the night of the change
    ("2026-03-29T05:59", True),   # 06:59 IST
    ("2026-03-29T06:00", False),  # 07:00 IST: the window ended an hour earlier in UTC
    ("2026-03-29T06:30", False),
    # Fall back, Sunday 25 October 2026, 01:00 UTC: 21:30 UTC is 22:30 IST the day before, 21:30 GMT after.
    ("2026-10-24T21:30", True),
    ("2026-10-25T21:30", False),
    ("2026-10-25T06:59", True),   # 06:59 GMT
    ("2026-10-25T07:00", False),
])
def test_the_window_follows_the_persons_wall_clock_across_a_change_of_clocks(stamp, quiet):
    assert _quiet(_person(time(22, 0), time(7, 0), DUBLIN), {}, stamp) is quiet


def test_with_no_zone_of_their_own_the_window_is_read_in_the_workspaces_else_utc():
    from services.proactive.quiet import person_timezone

    own = _person(time(22, 0), time(7, 0))
    workspace_zone = _workspace_window(zone=SAO_PAULO, enabled=False)

    assert person_timezone(own, workspace_zone) == SAO_PAULO
    assert _quiet(own, workspace_zone, "2026-06-01T09:00")  # 06:00 in São Paulo
    assert person_timezone(own, {}) == "UTC"
    assert not _quiet(own, {}, "2026-06-01T09:00")  # 09:00 UTC
    assert person_timezone(_person(time(22, 0), time(7, 0), DUBLIN), workspace_zone) == DUBLIN


def test_a_zone_zoneinfo_does_not_know_is_passed_over():
    from services.proactive.quiet import person_timezone

    assert person_timezone(_person(zone="Mars/Olympus"), _workspace_window(zone=SAO_PAULO)) == SAO_PAULO
    assert person_timezone(_person(zone="Mars/Olympus"), _workspace_window(zone="Moon/Base")) == "UTC"


def test_a_naive_moment_is_read_as_utc():
    from services.proactive.quiet import is_person_quiet

    assert is_person_quiet(NIGHT, {}, datetime(2026, 1, 15, 23, 0))
    assert not is_person_quiet(NIGHT, {}, datetime(2026, 1, 15, 12, 0))


# ── The workspace's window on top ───────────────────────────────────────────

LUNCH = _person(time(13, 0), time(14, 0), "UTC")


@pytest.mark.parametrize("stamp,quiet", [
    ("2026-01-15T13:30", True),   # the person's window
    ("2026-01-15T23:00", True),   # the workspace's: 23:00 GMT
    ("2026-01-15T07:59", True),   # still the workspace's
    ("2026-01-15T08:00", False),  # neither
    ("2026-07-15T07:30", False),  # 08:30 IST: the workspace's window is on Dublin's clock
])
def test_the_workspace_window_applies_on_top_of_the_persons(stamp, quiet):
    assert _quiet(LUNCH, _workspace_window(), stamp) is quiet


@pytest.mark.parametrize("person", [None, _person(), _person(zone=DUBLIN), _person(time(22, 0), None, "UTC")])
def test_a_person_with_no_window_of_their_own_is_quiet_only_in_the_workspaces(person):
    assert _quiet(person, _workspace_window(), "2026-01-15T23:00")
    assert not _quiet(person, _workspace_window(), "2026-01-15T13:30")
    assert not _quiet(person, {}, "2026-01-15T23:00")


def test_a_workspace_window_that_is_off_adds_nothing():
    assert not _quiet(LUNCH, _workspace_window(enabled=False), "2026-01-15T23:00")
    assert _quiet(LUNCH, _workspace_window(enabled=False), "2026-01-15T13:30")


# ── The route ───────────────────────────────────────────────────────────────

@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the route commits, and may not take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _user(db, tag: str) -> NS:
    clerk = f"user_prd268qh_{tag}_{uuid.uuid4().hex[:10]}"
    pk = db.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                    {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(pk=int(pk), clerk=clerk)


def _member(db, ws: uuid.UUID, role: str) -> NS:
    person = _user(db, role)
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"), {"ws": str(ws), "uid": person.pk, "role": role})
    return person


@pytest.fixture
def team(db_session) -> NS:
    """A workspace whose own quiet hours are 22:00 to 08:00 in Dublin, its owner and a viewer."""
    owner = _user(db_session, "owner")
    ws = uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO workspaces (id, name, slug, plan, owner_id, settings) VALUES (CAST(:id AS uuid), "
        "'prd268 quiet hours', :slug, 'pro', :o, CAST(:settings AS jsonb))"),
        {"id": str(ws), "slug": f"prd268-quiet-{ws.hex[:10]}", "o": owner.pk,
         "settings": json.dumps({"auto_reporting": _workspace_window()})})
    team = NS(db=db_session, ws=ws, owner=owner, viewer=_member(db_session, ws, "viewer"))
    db_session.commit()
    return team


@pytest.fixture
def saas(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


def _client(db, ctx) -> TestClient:
    import api.proactive as proactive
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(proactive.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _as(team, person) -> TestClient:
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role="user")
    return _client(team.db, RequestContext(workspace_id=team.ws, user=user, auth_type="clerk"))


def _rows(team) -> list:
    team.db.expire_all()
    return [tuple(row) for row in team.db.execute(text(
        "SELECT user_id, paused, quiet_start, quiet_end, timezone FROM proactive_person_settings "
        "WHERE workspace_id = CAST(:w AS uuid) ORDER BY user_id"), {"w": str(team.ws)}).fetchall()]


def _audit(team) -> list:
    return [tuple(row) for row in team.db.execute(text(
        "SELECT action, user_id, details FROM audit_logs WHERE workspace_id = CAST(:w AS uuid) "
        "AND resource_type = 'proactive_person_settings' ORDER BY id"), {"w": str(team.ws)}).fetchall()]


NIGHT_BODY = {"quiet_start": "22:00", "quiet_end": "07:00", "timezone": DUBLIN}


@pytest.mark.parametrize("role", ["owner", "viewer"])
def test_any_member_sets_and_reads_back_their_quiet_hours_and_timezone(team, saas, role):
    person = getattr(team, role)
    client = _as(team, person)

    assert client.get(ME).json() == {"paused": False, **NO_QUIET_HOURS}
    reply = client.put(ME, json=NIGHT_BODY)

    assert reply.status_code == 200 and reply.json() == {"paused": False, **NIGHT_BODY}
    assert client.get(ME).json() == {"paused": False, **NIGHT_BODY}
    assert _rows(team) == [(person.pk, False, time(22, 0), time(7, 0), DUBLIN)]
    [(action, user_id, details)] = _audit(team)
    assert (action, user_id) == ("proactive:quiet_hours_changed", person.pk)
    assert details["changed"] == NIGHT_BODY and details["quiet_hours"] == NIGHT_BODY


def test_what_a_put_leaves_out_stays_and_nulls_clear_the_hours(team, saas):
    client = _as(team, team.owner)

    client.put(ME, json={"paused": True})
    assert client.put(ME, json=NIGHT_BODY).json() == {"paused": True, **NIGHT_BODY}
    assert client.put(ME, json={"paused": False}).json() == {"paused": False, **NIGHT_BODY}
    cleared = client.put(ME, json={"quiet_start": None, "quiet_end": None})
    assert cleared.json() == {"paused": False, **NO_QUIET_HOURS, "timezone": DUBLIN}
    assert client.put(ME, json={"timezone": None}).json() == {"paused": False, **NO_QUIET_HOURS}

    assert _rows(team) == [(team.owner.pk, False, None, None, None)]  # one row, upserted
    assert [action for action, _, _ in _audit(team)] == [
        "proactive:paused", "proactive:quiet_hours_changed", "proactive:resumed",
        "proactive:quiet_hours_changed", "proactive:quiet_hours_changed"]


def test_each_person_keeps_their_own_quiet_hours(team, saas):
    _as(team, team.owner).put(ME, json=NIGHT_BODY)

    assert _as(team, team.viewer).get(ME).json() == {"paused": False, **NO_QUIET_HOURS}
    assert _rows(team) == [(team.owner.pk, False, time(22, 0), time(7, 0), DUBLIN)]


@pytest.mark.parametrize("body", [
    {},
    {"quiet_start": "22:00"},
    {"quiet_end": "07:00", "timezone": DUBLIN},
    {"quiet_start": "22:00", "quiet_end": None},
    {"quiet_start": "22:00", "quiet_end": "22:00"},
    {"quiet_start": "24:00", "quiet_end": "07:00"},
    {"quiet_start": "7:00", "quiet_end": "08:00"},
    {"quiet_start": "22:00:00", "quiet_end": "07:00"},
    {"quiet_start": 22, "quiet_end": 7},
    {"timezone": "Mars/Olympus"},
    {"timezone": ""},
    {"timezone": "x" * 65},
    {"paused": None},
    {"quiet": {"start": "22:00", "end": "07:00"}},
])
def test_quiet_hours_that_do_not_fit_are_refused_and_nothing_is_saved(team, saas, body):
    reply = _as(team, team.owner).put(ME, json=body)

    assert reply.status_code == 422
    assert _rows(team) == [] and _audit(team) == []


def test_the_local_operator_sets_its_own_quiet_hours(team, monkeypatch):
    from config import config
    from core.auth.dependencies import RequestContext, UserContext

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    operator = UserContext(id="operator@local.test", email="operator@local.test", system_role="super_admin",
                           raw_claims={"source": "local_operator", "user_id": team.owner.pk})
    client = _client(team.db, RequestContext(workspace_id=team.ws, user=operator, auth_type="anonymous"))

    assert client.put(ME, json=NIGHT_BODY).json() == {"paused": False, **NIGHT_BODY}
    assert _rows(team) == [(team.owner.pk, False, time(22, 0), time(7, 0), DUBLIN)]


def test_the_stored_hours_are_the_ones_is_person_quiet_reads_with_the_workspace_window(team, saas):
    from core.models.proactive import ProactivePersonSettings
    from core.services.auto_reporting import load_auto_reporting_settings
    from services.proactive.quiet import is_person_quiet

    _as(team, team.owner).put(ME, json={"quiet_start": "13:00", "quiet_end": "14:00", "timezone": SAO_PAULO})
    team.db.expire_all()
    row = team.db.get(ProactivePersonSettings, (team.ws, team.owner.pk))
    workspace = load_auto_reporting_settings(team.db, team.ws)

    assert is_person_quiet(row, workspace, _at("2026-06-01T16:30"))      # 13:30 in São Paulo
    assert is_person_quiet(row, workspace, _at("2026-06-01T22:30"))      # 23:30 IST: the workspace's
    assert not is_person_quiet(row, workspace, _at("2026-06-01T12:00"))  # 09:00 in São Paulo, 13:00 IST

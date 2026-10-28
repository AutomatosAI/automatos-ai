"""PRD-268 US-C17a — every needs-you row says where it stands, what happens next and what Try again does.

§6.11.1: the server computes the state once (``services/proactive/display_state.py``) and writes
the next step from one strings table in English and Portuguese (Brazil)
(``services/proactive/strings.py``), chosen by the request's ``Accept-Language`` (K13). ``GET
/api/activity/needs-you`` adds ``display_state``, ``next_step`` (at most 90 characters) and
``retry`` to each row; ``needs_you()`` itself and its counts are unchanged.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

import services.needs_you as needs_you_module
from core.models.approval_grants import GrantStatus
from core.models.orchestration_enums import RunState
from modules.proactive.locale import LOCALES
from services.needs_you import KINDS, needs_you
from services.proactive import display_state as ds
from services.proactive.strings import APPROVAL_PLAN, FAILED_MISSION, NEXT_STEP_CHARS, STRINGS, STUCK_REASONS

NOW = datetime.now(timezone.utc)
ADDED = ("display_state", "next_step", "retry")
# Every STUCK_* constant services/needs_you.py defines, read by name: a new one without a sentence fails.
STUCK_CONSTANTS = {name: value for name, value in vars(needs_you_module).items()
                   if name.startswith("STUCK_") and isinstance(value, str)}
KIND_STATES = {"review": "needs_you", "question": "needs_you", "approval": "needs_you", "stuck": "needs_you",
               "failed": "failed"}
BOARD_STATES = {"inbox": None, "assigned": "working", "in_progress": "working", "review": "working",
                "blocked": None, "done": "done", "closed": "done", "failed": "failed", "cancelled": None}
RUN_STATES = {"pending": None, "planning": "working", "awaiting_approval": None, "running": "working",
              "paused": None, "replanning": "working", "verifying": "working", "awaiting_human": None,
              "completed": "done", "failed": "failed", "cancelled": None}
GRANT_STATES = {"pending": None, "granted": "done", "revoked": None, "expired": None, "denied": None}


# --------------------------------------------------------------------------- the strings table


def test_every_stuck_constant_is_a_reason_the_table_has():
    assert STUCK_CONSTANTS, "services/needs_you.py names why a ticket is stuck in STUCK_* constants"
    assert set(STUCK_CONSTANTS.values()) == set(STUCK_REASONS)


@pytest.mark.parametrize("locale", LOCALES)
def test_both_languages_have_every_kind_and_every_stuck_reason(locale):
    table = STRINGS[locale]

    assert set(KINDS) <= set(table) and set(STUCK_CONSTANTS.values()) <= set(table)
    assert set(table) == set(STRINGS["en"])                   # no key in one language only
    assert all(0 < len(sentence) <= NEXT_STEP_CHARS for sentence in table.values())


def test_the_table_is_english_and_portuguese_each_its_own_copy():
    assert set(STRINGS) == set(LOCALES) == {"en", "pt-BR"}
    assert all(STRINGS["pt-BR"][key] != STRINGS["en"][key] for key in STRINGS["en"])


# --------------------------------------------------------------------------- the state, the next step, retry


@pytest.mark.parametrize("kind", KINDS)
def test_every_needs_you_kind_maps(kind):
    assert set(KIND_STATES) == set(KINDS)
    for source in (ds.BOARD_TASK, ds.MISSION, ds.GRANT):
        assert ds.display_state(source, None, kind) == KIND_STATES[kind]
    for locale in LOCALES:
        assert ds.next_step(locale, kind) == STRINGS[locale][kind]


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("why", sorted(STUCK_CONSTANTS.values()))
def test_every_stuck_reason_has_its_own_next_step(why, locale):
    said = ds.next_step(locale, "stuck", why=why)

    assert said == STRINGS[locale][why] and said != STRINGS[locale]["stuck"]
    assert ds.next_step(locale, "stuck", why=why, source=ds.MISSION) == said   # a Mission's pause says its reason
    assert ds.next_step(locale, "stuck", why="a reason nobody wrote") == STRINGS[locale]["stuck"]


@pytest.mark.parametrize("locale", LOCALES)
def test_a_missions_own_approval_and_failure_say_what_is_done_to_the_mission(locale):
    assert ds.next_step(locale, "approval", source=ds.MISSION) == STRINGS[locale][APPROVAL_PLAN]
    assert ds.next_step(locale, "failed", source=ds.MISSION) == STRINGS[locale][FAILED_MISSION]
    assert ds.next_step(locale, "approval", source=ds.GRANT) == STRINGS[locale]["approval"]


@pytest.mark.parametrize("source, table, statuses", [
    (ds.BOARD_TASK, BOARD_STATES, None),
    (ds.MISSION, RUN_STATES, {state.value for state in RunState}),
    (ds.GRANT, GRANT_STATES, {status.value for status in GrantStatus}),
])
def test_every_board_run_and_grant_status_maps(source, table, statuses):
    from api.board_tasks import VALID_STATUSES

    assert set(table) == (statuses if statuses is not None else VALID_STATUSES)
    assert {status: ds.display_state(source, status) for status in table} == table


def test_an_enum_member_reads_as_its_value():
    assert ds.display_state(ds.MISSION, RunState.RUNNING) == "working"
    assert ds.display_state(ds.GRANT, GrantStatus.GRANTED) == "done"


def test_unknown_gives_none():
    assert ds.display_state(ds.BOARD_TASK, "archived") is None
    assert ds.display_state("playbook_run", "completed") is None
    assert ds.display_state(ds.BOARD_TASK, None) is None and ds.display_state(ds.MISSION, 3) is None
    assert ds.display_state(ds.BOARD_TASK, "done", "a kind nobody wrote") == "done"     # its status still says
    assert ds.next_step("en", "a kind nobody wrote") is None
    assert ds.retry(ds.BOARD_TASK, "archived") is None
    assert "ready" in ds.DISPLAY_STATES and "ready" not in {*BOARD_STATES.values(), *RUN_STATES.values()}


def test_try_again_is_for_a_failed_row_only():
    assert set(ds.RETRIES) == {"rerun_mission", "replan_mission", "retry_ticket"}
    assert ds.retry(ds.BOARD_TASK, "failed") == ds.retry(ds.BOARD_TASK, None, "failed") == "retry_ticket"
    assert ds.retry(ds.MISSION, RunState.FAILED) == ds.retry(ds.MISSION, None, "failed") == "replan_mission"
    for source, status in [(ds.BOARD_TASK, "done"), (ds.BOARD_TASK, "review"), (ds.MISSION, "completed"),
                           (ds.MISSION, "cancelled"), (ds.GRANT, "denied"), (ds.GRANT, "pending")]:
        assert ds.retry(source, status) is None
    assert ds.retry(ds.BOARD_TASK, "failed", "stuck") is None      # a needs-you kind decides


# --------------------------------------------------------------------------- the rows, without a database

SAMPLE = {
    "total": 8, "counts": {"review": 1, "question": 1, "approval": 2, "stuck": 2, "failed": 2},
    "rows": {
        "review": [{"ticket_id": 1, "number": "#0001", "title": "Price list", "mission_id": None, "opens": "ticket"}],
        "question": [{"source": "grant", "id": "7", "title": "Which café opens first?", "ticket_id": 2}],
        "approval": [{"source": "grant", "id": "8", "title": "Price list", "ticket_id": 1},
                     {"source": "mission", "id": "r-1", "title": "Launch the autumn blend", "ticket_id": None}],
        "stuck": [{"ticket_id": 3, "why": "no_agent", "mission_id": None, "opens": "ticket"},
                  {"ticket_id": 4, "why": "over_budget", "mission_id": "r-2", "opens": "mission"}],
        "failed": [{"ticket_id": 5, "mission_id": None, "opens": "ticket"},
                   {"ticket_id": 6, "mission_id": "r-3", "opens": "mission"}],
    },
}


def test_each_row_gains_its_three_fields_and_nothing_else_changes():
    before = copy.deepcopy(SAMPLE)
    out = ds.needs_you_with_status(SAMPLE, "en")

    assert SAMPLE == before                                         # nothing it was given is changed
    assert (out["total"], out["counts"]) == (SAMPLE["total"], SAMPLE["counts"])
    for kind, rows in out["rows"].items():
        assert [{k: v for k, v in row.items() if k not in ADDED} for row in rows] == SAMPLE["rows"][kind]
    said = {kind: [(row["display_state"], row["next_step"], row["retry"]) for row in rows]
            for kind, rows in out["rows"].items()}
    en = STRINGS["en"]
    assert said == {
        "review": [("needs_you", en["review"], None)],
        "question": [("needs_you", en["question"], None)],
        "approval": [("needs_you", en["approval"], None), ("needs_you", en[APPROVAL_PLAN], None)],
        "stuck": [("needs_you", en["no_agent"], None), ("needs_you", en["over_budget"], None)],
        "failed": [("failed", en["failed"], "retry_ticket"), ("failed", en[FAILED_MISSION], "replan_mission")],
    }


def test_a_language_without_a_table_reads_english():
    out = ds.needs_you_with_status(SAMPLE, "fr")
    assert out["rows"]["review"][0]["next_step"] == STRINGS["en"]["review"]


# --------------------------------------------------------------------------- the route, on Postgres


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the needs-you status tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _run(s, ws, goal, state, state_type, stop_reason=None):
    return str(s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type, stop_reason) "
        "VALUES (CAST(:w AS uuid), :g, 'owner', :st, :sty, :why) RETURNING id"),
        {"w": ws, "g": goal, "st": state, "sty": state_type, "why": stop_reason}).scalar())


def _ticket(s, ws, title, status, *, source="user", done=None, run=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, completed_at, orchestration_run_id) "
        "VALUES (CAST(:w AS uuid), :t, :s, :src, :done, CAST(:run AS uuid)) RETURNING id"),
        {"w": ws, "t": title, "s": status, "src": source, "done": done, "run": run}).scalar()


def _grant(s, ws, kind, subject_id, *, question=None, reason=None):
    s.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, status, kind, question_md, reason, "
        "details, requested_at) VALUES (CAST(:w AS uuid), 'board_task', :sid, 'pending', :k, :q, :r, "
        "CAST(:d AS jsonb), :at)"),
        {"w": ws, "sid": str(subject_id), "k": kind, "q": question, "r": reason, "d": json.dumps({}), "at": NOW})


@pytest.fixture
def floor(engine, new_session):
    """One café's floor with a row of every kind: two stuck (no agent; a Mission at its budget), two failed."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd268-needs-you-status')"),
              {"id": ws})
    _run(s, ws, "Launch the autumn blend", "awaiting_approval", "blocked")
    paused = _run(s, ws, "Spring menu", "paused", "blocked", "budget_exhausted")
    lost = _run(s, ws, "Winter menu", "failed", "terminal")
    hour_ago = NOW - timedelta(hours=1)
    t = NS(
        review=_ticket(s, ws, "Price list for the cafés", "review", done=hour_ago),
        blocked=_ticket(s, ws, "Rota for October", "blocked"),
        unassigned=_ticket(s, ws, "Order the cups", "assigned"),
        over_budget=_ticket(s, ws, "Spring menu", "blocked", source="orchestration", run=paused),
        failed=_ticket(s, ws, "Weekly numbers", "failed", done=hour_ago),
        failed_card=_ticket(s, ws, "Winter menu", "failed", source="orchestration", done=hour_ago, run=lost),
        ws=ws,
    )
    _grant(s, ws, "question", t.blocked, question="Which café opens first on Sundays?")
    _grant(s, ws, "approval", t.review, reason="Send the price list to 40 cafés")
    s.commit()
    yield t
    s = new_session.sweep()
    for table in ("approval_grants", "board_tasks", "orchestration_runs"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


@pytest.fixture
def client(floor, new_session, monkeypatch):
    """The activity router as an owner of the floor's workspace."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.activity as activity
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    monkeypatch.setattr(activity, "may_answer_workspace_questions", lambda db, ctx: True)
    app = FastAPI()
    app.include_router(activity.router)
    db = new_session()
    app.dependency_overrides[get_request_context_hybrid] = lambda: NS(workspace_id=UUID(floor.ws))
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _needs_you(client, language=None):
    response = client.get("/api/activity/needs-you", headers={"Accept-Language": language} if language else {})
    assert response.status_code == 200, response.text
    return response.json()


def _said(out):
    """Each row's three fields by its ticket (a Mission's plan row by ``mission``)."""
    return {(kind, row.get("ticket_id") or "mission"): (row["display_state"], row["next_step"], row["retry"])
            for kind, rows in out["rows"].items() for row in rows}


def test_needs_you_itself_and_its_counts_are_unchanged(floor, new_session):
    plain = needs_you(new_session(), UUID(floor.ws))

    assert plain["counts"] == {"review": 1, "question": 1, "approval": 2, "stuck": 2, "failed": 2}
    assert plain["total"] == 8
    assert not any(field in row for rows in plain["rows"].values() for row in rows for field in ADDED)


def test_the_route_adds_each_rows_status_in_english_by_default(floor, client, new_session):
    out = _needs_you(client)
    plain = needs_you(new_session(), UUID(floor.ws))

    assert (out["total"], out["counts"]) == (plain["total"], plain["counts"]) and out["total"] == 8
    assert {kind: len(rows) for kind, rows in out["rows"].items()} == plain["counts"]
    en = STRINGS["en"]
    assert _said(out) == {
        ("review", floor.review): ("needs_you", en["review"], None),
        ("question", floor.blocked): ("needs_you", en["question"], None),
        ("approval", floor.review): ("needs_you", en["approval"], None),     # the grant on the price list
        ("approval", "mission"): ("needs_you", en[APPROVAL_PLAN], None),
        ("stuck", floor.unassigned): ("needs_you", en["no_agent"], None),
        ("stuck", floor.over_budget): ("needs_you", en["over_budget"], None),
        ("failed", floor.failed): ("failed", en["failed"], "retry_ticket"),
        ("failed", floor.failed_card): ("failed", en[FAILED_MISSION], "replan_mission"),
    }
    assert all(len(said[1]) <= NEXT_STEP_CHARS for said in _said(out).values())


def test_accept_language_pt_br_returns_portuguese(floor, client):
    english, portuguese = _needs_you(client, "en-GB"), _needs_you(client, "pt-BR,pt;q=0.9,en;q=0.8")

    assert (portuguese["total"], portuguese["counts"]) == (english["total"], english["counts"])
    pt = {key: (state, STRINGS["pt-BR"][_key_of(sentence)], retry)
          for key, (state, sentence, retry) in _said(english).items()}
    assert _said(portuguese) == pt
    sentences = {said[1] for said in _said(portuguese).values()}
    assert sentences <= set(STRINGS["pt-BR"].values()) and not sentences & set(STRINGS["en"].values())


def _key_of(sentence):
    """The strings-table key of an English sentence."""
    return next(key for key, said in STRINGS["en"].items() if said == sentence)

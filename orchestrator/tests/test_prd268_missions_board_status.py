"""PRD-268 US-C17b — the Missions list and the board's ticket reads use the same status words.

§6.11.1: a Mission's row (``GET /api/missions``, built by ``services/mission_view.py``, which
``api/missions.py`` still calls ``_run_to_response``) and a board ticket
(``services/board_task_view.py board_dict``) gain ``display_state``, ``next_step`` and
``retry`` from ``services/proactive/display_state.py``, from their own status. A Mission waiting
for its plan's approval is a needs-you approval row; a ticket in Review is working only while
its Mission checks it. Unknown gives None.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from core.models.orchestration_enums import RunState
from core.services import ticket_reasons as tr
from services.proactive import display_state as ds
from services.proactive.strings import APPROVAL_PLAN, FAILED_MISSION, NEXT_STEP_CHARS, STRINGS

ADDED = ("display_state", "next_step", "retry")
NOTHING = (None, None, None)
EN, PT = STRINGS["en"], STRINGS["pt-BR"]
# Every run state, what its row says (state, next step in English, retry).
RUNS = {
    "pending": NOTHING,
    "planning": ("working", None, None),
    "awaiting_approval": ("needs_you", EN[APPROVAL_PLAN], None),
    "running": ("working", None, None),
    "paused": NOTHING,
    "replanning": ("working", None, None),
    "verifying": ("working", None, None),
    "awaiting_human": NOTHING,
    "completed": ("done", None, None),
    "failed": ("failed", EN[FAILED_MISSION], "replan_mission"),
    "cancelled": NOTHING,
}
# Every board status, for a plain ticket (in Review: one its Mission checks).
BOARD = {
    "inbox": NOTHING,
    "assigned": ("working", None, None),
    "in_progress": ("working", None, None),
    "review": ("working", None, None),
    "blocked": NOTHING,
    "done": ("done", None, None),
    "closed": ("done", None, None),
    "failed": ("failed", EN["failed"], "retry_ticket"),
    "cancelled": NOTHING,
}
# The keys a Mission's row carried before this story: they are all still there, unchanged.
MISSION_KEYS = {
    "id", "workspace_id", "goal", "state", "state_type", "plan", "config", "check_each_step", "output_summary",
    "token_budget_estimate", "tokens_used", "max_retries", "max_concurrent", "replan_count", "complexity_tier",
    "parallel_groups", "has_synthesis_tasks", "created_by", "stop_reason", "stop_detail", "started_at",
    "completed_at", "created_at", "updated_at",
}


def _said(fields):
    return tuple(fields[key] for key in ADDED)


def _ticket(status, source="user", run_id=None):
    return NS(status=status, source_type=source, orchestration_run_id=run_id)


# --------------------------------------------------------------------------- Missions


def test_every_run_state_maps():
    assert set(RUNS) == {state.value for state in RunState}
    assert {state: _said(ds.mission_status(state)) for state in RUNS} == RUNS


@pytest.mark.parametrize("state, said", [
    (RunState.RUNNING, ("working", None, None)),
    (RunState.PLANNING, ("working", None, None)),
    (RunState.COMPLETED, ("done", None, None)),
    (RunState.FAILED, ("failed", EN[FAILED_MISSION], "replan_mission")),
])
def test_running_planning_completed_and_failed_runs_map(state, said):
    assert _said(ds.mission_status(state)) == _said(ds.mission_status(state.value)) == said


def test_a_mission_waiting_for_its_plans_approval_needs_you_as_its_needs_you_row_says():
    row = ds.row_status("approval", {"source": "mission", "id": "r-1"}, "en")      # US-C17a's row for it

    assert ds.mission_status(RunState.AWAITING_APPROVAL) == row == {
        "display_state": "needs_you", "next_step": EN[APPROVAL_PLAN], "retry": None}


def test_a_missions_next_step_is_in_the_language_asked_for():
    assert _said(ds.mission_status("failed", "pt-BR")) == ("failed", PT[FAILED_MISSION], "replan_mission")
    assert _said(ds.mission_status("awaiting_approval", "pt-BR")) == ("needs_you", PT[APPROVAL_PLAN], None)
    assert ds.mission_status("failed", "fr")["next_step"] == EN[FAILED_MISSION]


def test_the_missions_list_rows_gain_the_three_fields_and_keep_every_other():
    import api.missions as missions
    from services.mission_view import mission_dict

    run = NS(id=uuid.uuid4(), workspace_id=uuid.uuid4(), goal="Launch the autumn blend", state="failed",
             state_type="terminal", config={"auto_approve_steps": True}, output_summary=None,
             token_budget_estimate=None, tokens_used=None, max_retries=None, max_concurrent=3, replan_count=None,
             created_by="owner", stop_reason=None, stop_detail=None, started_at=None, completed_at=None,
             created_at=None, updated_at=None,
             plan={"tasks": [{"parallel_group": "b"}, {"parallel_group": "a", "task_type": "synthesis"}, {}]})

    row = mission_dict(run)

    assert missions._run_to_response is mission_dict                # every Missions route serves the same row
    assert set(row) == MISSION_KEYS | set(ADDED)
    assert _said(row) == ("failed", EN[FAILED_MISSION], "replan_mission")
    assert (row["parallel_groups"], row["has_synthesis_tasks"], row["complexity_tier"]) == (["a", "b"], True, "complex")
    assert (row["tokens_used"], row["max_retries"], row["replan_count"], row["check_each_step"]) == (0, 3, 0, False)
    assert mission_dict(NS(**{**vars(run), "max_concurrent": 2}))["complexity_tier"] == "moderate"
    simple = mission_dict(NS(**{**vars(run), "max_concurrent": None, "plan": None}))
    assert (simple["complexity_tier"], simple["max_concurrent"], simple["parallel_groups"]) == ("simple", 1, [])


# --------------------------------------------------------------------------- the board


def test_every_board_status_maps():
    from api.board_tasks import VALID_STATUSES

    assert set(BOARD) == VALID_STATUSES
    assert {status: _said(ds.ticket_status(_ticket(status), tr.MISSION_CHECKING if status == "review" else None))
            for status in BOARD} == BOARD


@pytest.mark.parametrize("code", [tr.ASKED, tr.MISSION_PLAN, tr.APPROVAL_ACTION, tr.MOVED_BY_YOU, tr.UNEXPLAINED, None])
def test_a_ticket_in_review_is_working_only_while_its_mission_checks_it(code):
    assert _said(ds.ticket_status(_ticket("review", "orchestration_task"), tr.MISSION_CHECKING))[0] == "working"
    assert _said(ds.ticket_status(_ticket("review"), code)) == NOTHING          # the owner's: needs_you() says


def test_a_failed_ticket_card_and_step_each_say_what_try_again_does():
    run_id = uuid.uuid4()

    assert _said(ds.ticket_status(_ticket("failed"), None)) == ("failed", EN["failed"], "retry_ticket")
    assert _said(ds.ticket_status(_ticket("failed", "orchestration", run_id), None)) == (
        "failed", EN[FAILED_MISSION], "replan_mission")                         # the Mission's own card
    assert _said(ds.ticket_status(_ticket("failed", "orchestration"), None))[2] == "retry_ticket"   # no Mission
    assert _said(ds.ticket_status(_ticket("failed", "orchestration_task", run_id), None)) == ("failed", None, None)
    assert _said(ds.ticket_status(_ticket("failed"), None, "pt-BR")) == ("failed", PT["failed"], "retry_ticket")


def test_a_failed_ticket_says_what_its_needs_you_row_says():
    card = ds.ticket_status(_ticket("failed", "orchestration", uuid.uuid4()), None)
    plain = ds.ticket_status(_ticket("failed"), None)

    assert card == ds.row_status("failed", {"ticket_id": 6, "opens": "mission"}, "en")
    assert plain == ds.row_status("failed", {"ticket_id": 5, "opens": "ticket"}, "en")


def test_unknown_gives_none():
    for state in ("archived", None, 3, ""):
        assert _said(ds.mission_status(state)) == NOTHING
    for status in ("archived", None, ""):
        assert _said(ds.ticket_status(_ticket(status), None)) == NOTHING
    assert _said(ds.ticket_status(NS(), None)) == NOTHING                    # a ticket with no status at all


def test_every_next_step_fits_its_line():
    for locale in ("en", "pt-BR"):
        said = [ds.mission_status(state, locale)["next_step"] for state in RUNS]
        said += [ds.ticket_status(_ticket(status, source, uuid.uuid4()), None, locale)["next_step"]
                 for status in BOARD for source in ("user", "orchestration")]
        assert all(len(sentence) <= NEXT_STEP_CHARS for sentence in said if sentence)


def test_the_board_serves_the_three_fields_on_each_ticket():
    from core.models.core import BoardTask
    from services.board_task_view import board_dict

    def task(status, source="user", mode="auto"):
        return BoardTask(id=7, status=status, source_type=source, review_mode=mode, tags=[], attempts=0)

    assert _said(board_dict(task("failed"))) == ("failed", EN["failed"], "retry_ticket")
    assert _said(board_dict(task("review", "orchestration_task"))) == ("working", None, None)
    assert _said(board_dict(task("review", "orchestration_task", "human"))) == NOTHING     # the owner checks it
    served = board_dict(task("in_progress"), "#0007")
    assert _said(served) == ("working", None, None) and served["number"] == "#0007"


# --------------------------------------------------------------------------- the reads, on Postgres


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Missions and board status tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _run(s, ws, goal, state, state_type):
    return str(s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), :g, 'owner', :st, :sty) RETURNING id"),
        {"w": ws, "g": goal, "st": state, "sty": state_type}).scalar())


def _file(s, ws, title, status, *, source="user", mode="auto", run=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, review_mode, orchestration_run_id) "
        "VALUES (CAST(:w AS uuid), :t, :s, :src, :mode, CAST(:run AS uuid)) RETURNING id"),
        {"w": ws, "t": title, "s": status, "src": source, "mode": mode, "run": run}).scalar()


@pytest.fixture
def cafe(engine, new_session):
    """A café's Missions in five states, and tickets on its board."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd268-missions-board-status')"),
              {"id": ws})
    runs = {state: _run(s, ws, f"Menu: {state}", state, state_type) for state, state_type in (
        ("running", "active"), ("planning", "active"), ("completed", "terminal"), ("failed", "terminal"),
        ("awaiting_approval", "blocked"))}
    tickets = {
        "rota": _file(s, ws, "Rota for October", "in_progress"),
        "price_list": _file(s, ws, "Price list", "failed"),
        "winter_menu": _file(s, ws, "Winter menu", "failed", source="orchestration", run=runs["failed"]),
        "step": _file(s, ws, "Check the menu", "review", source="orchestration_task"),
        "cups": _file(s, ws, "Order the cups", "inbox"),
    }
    s.commit()
    yield NS(ws=ws, runs=runs, tickets=tickets)
    s = new_session.sweep()
    for table in ("board_tasks", "orchestration_runs"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_the_missions_list_route_says_where_each_mission_stands(cafe, new_session):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.missions as missions
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db
    from services.needs_you import needs_you

    app = FastAPI()
    app.include_router(missions.router)
    db = new_session()
    app.dependency_overrides[get_request_context_hybrid] = lambda: NS(workspace_id=UUID(cafe.ws))
    app.dependency_overrides[get_db] = lambda: db

    response = TestClient(app).get("/api/missions", params={"limit": 20})

    assert response.status_code == 200, response.text
    said = {row["state"]: _said(row) for row in response.json()["missions"]}
    assert said == {state: RUNS[state] for state in cafe.runs}
    waiting = [row["id"] for row in needs_you(new_session(), UUID(cafe.ws))["rows"]["approval"]
               if row["source"] == "mission"]
    assert waiting == [cafe.runs["awaiting_approval"]]                       # the row needs_you() lists


def test_the_boards_reads_say_where_each_ticket_stands(cafe, new_session):
    from core.models.core import BoardTask
    from services.board_task_view import enrich_with_agents
    from services.needs_you import needs_you

    db = new_session()
    tasks = db.query(BoardTask).filter(BoardTask.workspace_id == UUID(cafe.ws)).all()

    served = {row["id"]: _said(row) for row in enrich_with_agents(tasks, db, UUID(cafe.ws))}

    assert served == {
        cafe.tickets["rota"]: ("working", None, None),
        cafe.tickets["price_list"]: ("failed", EN["failed"], "retry_ticket"),
        cafe.tickets["winter_menu"]: ("failed", EN[FAILED_MISSION], "replan_mission"),
        cafe.tickets["step"]: ("working", None, None),                       # its Mission checks it
        cafe.tickets["cups"]: NOTHING,
    }
    failed = needs_you(new_session(), UUID(cafe.ws))["rows"]["failed"]
    assert {row["ticket_id"] for row in failed} == {cafe.tickets["price_list"], cafe.tickets["winter_menu"]}
    for row in failed:                                                       # the board says what Needs you says
        assert served[row["ticket_id"]] == _said(ds.row_status("failed", row, "en"))

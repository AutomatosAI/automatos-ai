"""A filed ticket and its operator consent become visible in one commit.

Ticket 343 (a scheduled ticket, local edition, "Ask before any Mission or task runs"):
created 09:00:00.67 as ``assigned`` and committed; its consent grant #288 was requested
at 09:00:01.42 and granted at 09:00:02.32; the dispatcher, woken by something else, had
claimed the ticket at 09:00:01.66, found no grant, and parked it behind a new request
(#290). ``_file_board_task`` committed the ticket first and recorded the consent after.

``record_operator_consent(commit=False)`` stages the grant inside a savepoint of the
caller's transaction and commits nothing, so the code that files a ticket commits the
ticket and its grant together. Proven here for the filing paths changed to do so: the
schedule, the board's create route, Auto's assign and status tools, a mission step's
card and a suggestion's Start. Elsewhere, beside their neighbours: Auto's heartbeat review
ticket (test_session_heartbeat_reports_when_it_ends), a Claude Code lane's ticket
(test_prd234_lane_consent) and a card Auto files (test_board_task_handlers). Fake sessions
record the order of commits and the consent; the long tool handlers are read for their
call order, as test_grant_approve_commit_ordering reads the approvals route.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.services.approval_grants as grants  # noqa: E402
from services import board_consent  # noqa: E402

WS = uuid4()
SAVEPOINT = "savepoint"
RELEASED = "released savepoint"
ROLLED_BACK_TO_SAVEPOINT = "rolled back to savepoint"
COMMIT = "commit"
ROLLBACK = "rollback"


class _Savepoint:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append(SAVEPOINT)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append(ROLLED_BACK_TO_SAVEPOINT if exc_type else RELEASED)
        return False


class _SessionLog:
    """A session that only records what was asked of its transaction."""

    def __init__(self):
        self.log = []

    def begin_nested(self):
        return _Savepoint(self.log)

    def commit(self):
        self.log.append(COMMIT)

    def rollback(self):
        self.log.append(ROLLBACK)


def _no_grant_yet(monkeypatch):
    created = []
    monkeypatch.setattr(grants, "find_active_grant", lambda db, ws, **kw: None)
    monkeypatch.setattr(grants, "find_pending_grant", lambda db, ws, **kw: None)

    def _create(db, ws, **kw):
        grant = SimpleNamespace(id=288, status="pending", **kw)
        created.append(grant)
        return grant

    def _grant(grant, *, granted_by, now=None):
        grant.status = "granted"
        grant.granted_by = granted_by
        return grant

    monkeypatch.setattr(grants, "create_grant", _create)
    monkeypatch.setattr(grants, "grant_grant", _grant)
    return created


# ── record_operator_consent(commit=False) ──────────────────────────────────

def test_a_staged_consent_is_written_in_a_savepoint_and_commits_nothing(monkeypatch):
    created = _no_grant_yet(monkeypatch)
    db = _SessionLog()
    out = board_consent.record_operator_consent(
        db, workspace_id="ws", task_id=343, agent_id=15, actor="user:2",
        why=board_consent.WHY_SCHEDULED_AND_ASSIGNED, commit=False,
    )
    assert out == board_consent.CREATED
    assert created[0].subject_id == "343" and created[0].status == "granted"
    assert db.log == [SAVEPOINT, RELEASED]  # the caller's commit carries it


def test_a_staged_consent_that_fails_rolls_back_its_savepoint_alone(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(grants, "find_active_grant", _boom)
    db = _SessionLog()
    out = board_consent.record_operator_consent(
        db, workspace_id="ws", task_id=343, agent_id=15, actor="user:2", why="x", commit=False,
    )
    # Never the caller's rollback: the ticket it is filing still commits, and the gate asks.
    assert out == board_consent.ERROR and db.log == [SAVEPOINT, ROLLED_BACK_TO_SAVEPOINT]


def test_the_committing_consent_still_commits_its_own_grant(monkeypatch):
    _no_grant_yet(monkeypatch)
    db = _SessionLog()
    assert board_consent.record_operator_consent(
        db, workspace_id="ws", task_id=72, agent_id=15, actor="user:2", why=board_consent.WHY_RUN_NOW,
    ) == board_consent.CREATED
    assert db.log == [COMMIT]


def test_the_ticket_consents_pass_the_staging_on(monkeypatch):
    seen = []
    monkeypatch.setattr(board_consent, "creation_is_consent", lambda: True)
    monkeypatch.setattr(board_consent, "record_operator_consent", lambda db, **kw: seen.append(kw) or "created")
    task = SimpleNamespace(id=343, status="assigned", assigned_agent_id=15)
    board_consent.consent_for_created_ticket(object(), workspace_id="ws", task=task, actor="user:1",
                                             why=board_consent.WHY_SCHEDULED_AND_ASSIGNED, commit=False)
    board_consent.consent_for_lane_ticket(object(), workspace_id="ws", task=task, source_type="heartbeat",
                                          commit=False)
    board_consent.consent_for_lane_ticket(object(), workspace_id="ws", task=task, source_type="heartbeat")
    assert [kw["commit"] for kw in seen] == [False, False, True]


# ── the board's create route (api.board_task_create.file_ticket) ───────────

class _FilingSession:
    def __init__(self, log):
        self.log = log

    def add(self, obj):
        self.obj = obj

    def flush(self):
        self.obj.id = 343
        self.log.append("flush")

    def commit(self):
        self.log.append(COMMIT)

    def refresh(self, obj):
        pass


def _columns(agent_id):
    return {
        "title": "Weekly numbers", "description": None, "raw_prompt": None,
        "status": "assigned" if agent_id else "inbox", "priority": "medium", "review_mode": "auto",
        "assigned_agent_id": agent_id, "parent_task_id": None, "source_type": "user", "source_id": None,
        "tags": [], "planning_data": None, "attachment_ids": [],
    }


def _board_filing(monkeypatch):
    import api.board_task_create as create

    log = []
    monkeypatch.setattr(create, "consent_for_created_ticket",
                        lambda db, **kw: log.append(("consent", kw["task"].id, kw.get("commit"))) or "created")
    monkeypatch.setattr(create, "notify_board_event", lambda db, **kw: log.append("board event"))
    monkeypatch.setattr(create, "notify_task_available", lambda db, **kw: log.append("wake"))
    monkeypatch.setattr(create, "_note_no_host_for_cli", lambda db, task: None)
    ctx = SimpleNamespace(workspace_id=WS, auth_type="clerk",
                          user=SimpleNamespace(id="u-1", clerk_user_id="clerk_owner", email=None))
    return create, ctx, log


def test_a_ticket_filed_on_the_board_commits_with_its_consent(monkeypatch):
    create, ctx, log = _board_filing(monkeypatch)
    create.file_ticket(_FilingSession(log), ctx, _columns(agent_id=15))
    assert log == ["flush", ("consent", 343, False), "board event", COMMIT, "wake", COMMIT]


def test_a_ticket_filed_in_the_inbox_records_no_consent_and_wakes_nothing(monkeypatch):
    create, ctx, log = _board_filing(monkeypatch)
    create.file_ticket(_FilingSession(log), ctx, _columns(agent_id=None))
    assert log == ["flush", "board event", COMMIT]


# ── Auto's chat tools (handlers_board_tasks) ───────────────────────────────

_HANDLERS = _ORCH / "modules" / "tools" / "discovery" / "handlers_board_tasks.py"
_LANE = _ORCH / "services" / "cli_ticket_lane.py"


def _calls(path: Path, function: str, names: set) -> list:
    """The calls named in ``names`` inside ``function`` of ``path``, in source order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function)
    found = []
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if name in names:
            found.append((call.lineno, name, call))
    return sorted(found, key=lambda item: item[0])


def test_auto_s_assign_commits_the_assignment_with_its_consent():
    order = [name for _, name, _ in _calls(_HANDLERS, "assign_board_task", {"commit", "_consent_for_chat_filed"})]
    assert order == ["_consent_for_chat_filed", "commit"], order


def test_auto_s_move_back_to_assigned_commits_with_its_consent():
    calls = _calls(_HANDLERS, "update_board_task_status", {"commit", "_consent_for_chat_filed", "apply_explicit_status"})
    order = [name for _, name, _ in calls]
    moved = order.index("apply_explicit_status")
    # The status change made by apply_explicit_status is committed only with the consent.
    assert order[moved + 1:moved + 3] == ["_consent_for_chat_filed", "commit"], order


def test_auto_s_consent_is_staged_for_the_caller_s_commit(monkeypatch):
    from modules.tools.discovery.handlers_board_tasks import _consent_for_chat_filed

    seen = []
    monkeypatch.setattr(board_consent, "consent_for_created_ticket", lambda db, **kw: seen.append(kw) or "created")
    task = SimpleNamespace(id=343, status="assigned", assigned_agent_id=15)
    _consent_for_chat_filed(object(), WS, task, {"_user_id": 7})
    assert seen[0]["commit"] is False and seen[0]["actor"] == "user:7"


# ── a mission step's card (cli_ticket_lane.claim_step_card) ────────────────

def test_a_step_card_moves_to_assigned_in_the_commit_that_carries_its_consent():
    calls = _calls(_LANE, "claim_step_card", {"commit", "consent_for_lane_ticket"})
    assert [name for _, name, _ in calls] == ["consent_for_lane_ticket", "commit", "commit"]
    consent = calls[0][2]
    staged = [kw.value.value for kw in consent.keywords if kw.arg == "commit"]
    assert staged == [False]


# ── a suggestion's Start (services.proactive.start_ticket) ─────────────────

def test_a_started_suggestion_s_ticket_is_approved_before_the_start_commits(monkeypatch):
    from services.proactive import decisions, start_ticket

    seen = []
    monkeypatch.setattr(board_consent, "consent_for_created_ticket", lambda db, **kw: seen.append(kw) or "created")
    task = SimpleNamespace(id=846, workspace_id=WS, status="assigned", assigned_agent_id=3)
    start_ticket.approve(object(), task, "user:clicker")
    assert seen[0]["commit"] is False and seen[0]["actor"] == "user:clicker"
    begun = [name for _, name, _ in _calls(Path(decisions.__file__), "_begin", {"file_ticket", "approve"})]
    assert begun == ["file_ticket", "approve"]


def test_the_wake_after_the_start_s_commit_records_no_consent(monkeypatch):
    import services.board_dispatcher as dispatcher
    from services.proactive import start_ticket

    woken = []
    monkeypatch.setattr(board_consent, "consent_for_created_ticket",
                        lambda db, **kw: pytest.fail("the consent came after the Start's commit"))
    monkeypatch.setattr(dispatcher, "notify_task_available", lambda db, **kw: woken.append(kw["task_id"]))
    db = _SessionLog()
    start_ticket.wake(db, SimpleNamespace(id=846, workspace_id=WS, status="assigned", assigned_agent_id=3))
    assert woken == [846] and db.log == [COMMIT]


# ── a card in Review given to another agent (handlers_board_task_assign) ────

class _GivenBack:
    """One card, a filter that finds it, ``_decide``'s compare-and-set, and a logged commit."""

    def __init__(self, card, log):
        self.card, self.log = card, log

    def query(self, model):
        return self

    def filter(self, *conds):
        return self

    def first(self):
        return self.card

    def update(self, values, synchronize_session=False):
        return 1

    def commit(self):
        self.log.append(COMMIT)


def test_a_card_given_to_another_agent_goes_back_with_its_consent_in_one_commit(monkeypatch):
    """F309's give-back moved the card to Assigned, woke the dispatcher and committed, and only
    then recorded the owner's consent: the card was claimable before its grant existed."""
    import api.board_tasks as bt
    from modules.tools.discovery import handlers_board_task_assign as give
    from modules.tools.discovery import handlers_board_tasks as handlers

    log = []
    card = SimpleNamespace(id=1859, workspace_id=WS, status="review", assigned_agent_id=12, source_type="user",
                           workspace_seq=1859)
    monkeypatch.setattr(bt, "keep_previous_run", lambda task, **kw: None)
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: log.append("wake"))
    monkeypatch.setattr(handlers, "_consent_for_chat_filed",
                        lambda db, ws, task, params: log.append(("consent", task.status)))
    out = give._run_again(_GivenBack(card, log), WS, {"_user_id": "owner"}, (1859, 11),
                          {"success": True, "assigned_agent": "Shopify Business Analyst"})
    assert out["runs_again"] is True and card.status == "assigned"
    assert log == [("consent", "assigned"), "wake", COMMIT]  # one commit: the move, its consent, the wake


# ── on the database: the savepoint the staged consent is written in ────────

def test_a_staged_consent_commits_with_its_ticket_and_a_failed_one_leaves_the_ticket(
        db_session, seed_workspace, monkeypatch):
    """The transactional session of the DB-backed suites (conftest ``db_session``, rolled back
    after the test). A consent staged with ``commit=False`` and one commit leave the ticket and
    its granted grant, which the gate finds; a staging that fails after its grant row was
    written rolls back its savepoint alone: the ticket commits, its grant does not."""
    from sqlalchemy import text

    from core.models.approval_grants import ApprovalGrant, SUBJECT_BOARD_TASK
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    agent = db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar()

    def _filed(title, fail=False):
        task = BoardTask(workspace_id=ws, title=title, priority="medium", source_type="scheduled_task",
                         status="assigned", assigned_agent_id=agent, attempts=0)
        db_session.add(task)
        db_session.flush()
        if fail:
            def _boom(grant, **kw):
                raise RuntimeError("the grant could not be approved")
            monkeypatch.setattr(grants, "grant_grant", _boom)   # after create_grant flushed its row
        out = board_consent.record_operator_consent(
            db_session, workspace_id=ws, task_id=task.id, agent_id=agent, actor="user:owner",
            why=board_consent.WHY_SCHEDULED_AND_ASSIGNED, commit=False)
        db_session.commit()
        return task.id, out

    def _grants(task_id):
        return db_session.query(ApprovalGrant).filter(
            ApprovalGrant.workspace_id == ws, ApprovalGrant.subject_type == SUBJECT_BOARD_TASK,
            ApprovalGrant.subject_id == str(task_id)).all()

    kept, out = _filed("Weekly stock count")
    assert out == board_consent.CREATED and db_session.get(BoardTask, kept) is not None
    assert [g.status for g in _grants(kept)] == ["granted"]
    assert grants.find_active_grant(db_session, ws, subject_type=SUBJECT_BOARD_TASK, subject_id=str(kept))

    lost, out = _filed("Monthly stock count", fail=True)
    assert out == board_consent.ERROR and db_session.get(BoardTask, lost) is not None
    assert _grants(lost) == []

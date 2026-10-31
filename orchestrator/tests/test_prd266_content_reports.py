"""PRD-266 US-016a: filing a content report (``services/content_reports``).

A report of an Auto message or a Deliverable is ONE ``audit_logs`` row
(``content_reported``, ``{reason, note, excerpt}``) and ONE in-app notice to the
workspace's owner, naming the target kind and the reason only. What is not the
reporter's to report (another workspace's message, a user message, another
member's private chat, a deleted or unknown Deliverable, a malformed id) is not
found and writes nothing. A second report of the same target by the same person
answers the first report and writes and sends nothing.

Real Postgres (CI) for the chats, messages, users and audit rows, in a session
whose commits and rollbacks stay inside a savepoint (so ``AuditService.log``'s
commit and the notice's rollback never take the seed). The notice dispatcher,
the rate limiter and the Deliverable view (``v_workspace_outputs``, which
``init_test_db`` does not build) are fakes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from sqlalchemy import text  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import core.workspaces.owner as owner_mod  # noqa: E402
from core.models.core import Chat, Message  # noqa: E402
from core.services.notification_dispatcher import VALID_EVENT_TYPES  # noqa: E402
from core.workspaces.audit import AuditLog, AuditService  # noqa: E402
from services.content_reports import AUDIT_ACTION, EXCERPT_MAX_CHARS  # noqa: E402
from services.content_reports import service, targets  # noqa: E402

AUTO_REPLY = "Here is the plan Auto drafted for the launch."
NOTE = "This names a real customer."


# ---------------------------------------------------------------------------
# Fixtures: a savepoint session, people, chats, and the fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session(test_engine):
    """A session whose commit releases a savepoint and whose rollback returns to one."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _user(db, name: str) -> int:
    tag = uuid.uuid4().hex[:10]
    return db.execute(
        text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
        {"e": f"{name}-{tag}@reports.example.test", "u": f"{name}-{tag}", "c": f"user_{tag}"},
    ).scalar()


def _owner_member(db, workspace_id: str, user_pk: int) -> None:
    db.execute(
        text(
            "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"
        ),
        {"ws": workspace_id, "user": user_pk},
    )


def _chat_message(db, workspace_id: str, chat_owner: int, *, text_: str = AUTO_REPLY,
                  role: str = "assistant", visibility: str = "private") -> str:
    chat = Chat(id=uuid.uuid4(), user_id=chat_owner, workspace_id=uuid.UUID(workspace_id),
                title="Launch", visibility=visibility, kind="user")
    db.add(chat)
    db.flush()
    message = Message(id=uuid.uuid4(), chat_id=chat.id, workspace_id=uuid.UUID(workspace_id),
                      role=role, parts=[{"type": "text", "text": text_}])
    db.add(message)
    db.flush()
    return str(message.id)


@pytest.fixture
def floor(db_session, seed_workspace):
    """One workspace with an owner and a reporter; a second workspace; all committed."""
    db = db_session
    ws, other_ws = seed_workspace(), seed_workspace()
    owner, reporter, other = _user(db, "owner"), _user(db, "reporter"), _user(db, "other")
    _owner_member(db, ws, owner)
    db.commit()
    return SimpleNamespace(db=db, ws=ws, other_ws=other_ws, owner=owner, reporter=reporter, other=other)


def _ctx(user_pk, workspace_id):
    user = None if user_pk is None else SimpleNamespace(id=user_pk, system_role="user")
    return SimpleNamespace(user=user, workspace_id=uuid.UUID(workspace_id))


@pytest.fixture
def notices(monkeypatch):
    """Every dispatch the service makes, as ``(workspace_id, kwargs)``."""
    sent = []

    class _Dispatcher:
        def __init__(self, db, workspace_id):
            self.workspace_id = str(workspace_id)

        async def dispatch(self, **kwargs):
            sent.append((self.workspace_id, kwargs))
            return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(service, "NotificationDispatcher", _Dispatcher)
    return sent


@pytest.fixture
def limiter(monkeypatch):
    """Records each rate-limit check; never refuses."""
    calls = []

    async def _check(workspace_id, operation, **kwargs):
        calls.append((workspace_id, operation, kwargs))

    monkeypatch.setattr(service.rate_limiter, "check_rate_limit", _check)
    return calls


@pytest.fixture
def deliverables(monkeypatch):
    """The live Deliverables ``get_deliverable`` finds, keyed ``(workspace_id, id)``."""
    live: dict = {}

    class _Deliverables:
        def __init__(self, db, workspace_id):
            self.workspace_id = str(workspace_id)

        async def get_deliverable(self, deliverable_id, include_content=False):
            found = live.get((self.workspace_id, deliverable_id))
            if found is None:
                return {"success": False, "error": "Deliverable not found"}
            return {"success": True, "deliverable": found}

    monkeypatch.setattr(targets, "DeliverableService", _Deliverables)
    return live


def _file(floor, target_type, target_id, *, reporter=None, reason="harmful", note=NOTE):
    ctx = _ctx(floor.reporter if reporter is None else reporter, floor.ws)
    return asyncio.run(service.file_report(
        floor.db, ctx, target_type=target_type, target_id=target_id, reason=reason, note=note,
    ))


def _report_rows(floor):
    return (
        floor.db.query(AuditLog)
        .filter(AuditLog.action == AUDIT_ACTION,
                AuditLog.workspace_id.in_([uuid.UUID(floor.ws), uuid.UUID(floor.other_ws)]))
        .all()
    )


# ---------------------------------------------------------------------------
# A message, a Deliverable
# ---------------------------------------------------------------------------


def test_the_event_type_is_known_to_the_dispatcher():
    assert AUDIT_ACTION == "content_reported"
    assert AUDIT_ACTION in VALID_EVENT_TYPES


def test_a_message_report_writes_one_audit_row_and_one_notice_to_the_owner(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)

    filed = _file(floor, "message", message_id)

    assert filed.duplicate is False and filed.report_id > 0
    rows = _report_rows(floor)
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.user_id, row.actor_type) == (filed.report_id, floor.reporter, "user")
    assert (row.resource_type, row.resource_id) == ("message", message_id)
    assert row.details == {"reason": "harmful", "note": NOTE, "excerpt": AUTO_REPLY}
    assert len(notices) == 1
    workspace_id, notice = notices[0]
    assert workspace_id == floor.ws
    assert notice["user_id"] == floor.owner
    assert notice["event_type"] == "content_reported" and notice["title"] == "Content reported"
    assert (notice["link_type"], notice["link_id"]) == ("content_report", str(filed.report_id))
    assert notice["status"] == "warning"


def test_the_reporter_is_rate_limited_on_their_own_bucket(floor, notices, limiter):
    _file(floor, "message", _chat_message(floor.db, floor.ws, floor.reporter))

    max_requests, window_seconds = service.CONTENT_REPORT_RATE_LIMIT
    assert limiter == [(floor.ws, "content_report", {
        "max_requests": max_requests, "window_seconds": window_seconds, "subject_id": str(floor.reporter),
    })]


def test_over_the_limit_nothing_is_written_or_sent(floor, notices, monkeypatch):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)

    async def _refuse(*_args, **_kwargs):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    monkeypatch.setattr(service.rate_limiter, "check_rate_limit", _refuse)
    with pytest.raises(HTTPException) as refused:
        _file(floor, "message", message_id)
    assert refused.value.status_code == 429
    assert _report_rows(floor) == [] and notices == []


def test_a_public_chat_of_another_member_may_be_reported(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.other, visibility="public")

    assert _file(floor, "message", message_id).duplicate is False
    assert len(_report_rows(floor)) == 1


def test_a_deliverable_report_keeps_its_title_and_summary(floor, notices, limiter, deliverables):
    deliverable_id = str(uuid.uuid4())
    deliverables[(floor.ws, deliverable_id)] = {"title": "Q3 plan", "summary": "Budget and hires."}

    filed = _file(floor, "deliverable", deliverable_id, reason="inaccurate", note=None)

    assert filed.excerpt == "Q3 plan — Budget and hires."
    (row,) = _report_rows(floor)
    assert (row.resource_type, row.resource_id) == ("deliverable", deliverable_id)
    assert row.details == {"reason": "inaccurate", "note": None, "excerpt": "Q3 plan — Budget and hires."}
    assert [n["user_id"] for _, n in notices] == [floor.owner]


# ---------------------------------------------------------------------------
# Not the reporter's to report: not found, nothing written
# ---------------------------------------------------------------------------


def _another_workspaces_message(floor, deliverables):
    return "message", _chat_message(floor.db, floor.other_ws, floor.reporter)


def _a_user_message(floor, deliverables):
    return "message", _chat_message(floor.db, floor.ws, floor.reporter, role="user")


def _another_members_private_chat(floor, deliverables):
    return "message", _chat_message(floor.db, floor.ws, floor.other)


def _a_deleted_or_unknown_deliverable(floor, deliverables):
    return "deliverable", str(uuid.uuid4())


def _another_workspaces_deliverable(floor, deliverables):
    deliverable_id = str(uuid.uuid4())
    deliverables[(floor.other_ws, deliverable_id)] = {"title": "Theirs", "summary": None}
    return "deliverable", deliverable_id


def _a_malformed_message_id(floor, deliverables):
    return "message", "not-a-uuid"


def _a_malformed_deliverable_id(floor, deliverables):
    return "deliverable", "1; DROP TABLE audit_logs"


@pytest.mark.parametrize("target", [
    _another_workspaces_message, _a_user_message, _another_members_private_chat,
    _a_deleted_or_unknown_deliverable, _another_workspaces_deliverable,
    _a_malformed_message_id, _a_malformed_deliverable_id,
])
def test_a_target_that_is_not_the_reporters_is_not_found(floor, notices, limiter, deliverables, target):
    target_type, target_id = target(floor, deliverables)

    with pytest.raises(service.ReportTargetNotFound):
        _file(floor, target_type, target_id)
    assert _report_rows(floor) == [] and notices == []


def test_a_caller_with_no_person_cannot_report(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)
    ctx = _ctx(None, floor.ws)

    with pytest.raises(service.ReportNeedsAPerson):
        asyncio.run(service.file_report(floor.db, ctx, target_type="message", target_id=message_id,
                                        reason="other", note=None))
    assert _report_rows(floor) == [] and notices == [] and limiter == []


# ---------------------------------------------------------------------------
# Duplicates, no owner, a failed write, a failed notice
# ---------------------------------------------------------------------------


def test_a_second_report_of_the_same_target_writes_and_sends_nothing(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)
    first = _file(floor, "message", message_id)

    again = _file(floor, "message", message_id.upper(), reason="offensive")

    assert again.duplicate is True and again.report_id == first.report_id
    assert len(_report_rows(floor)) == 1
    assert len(notices) == 1


def test_another_member_reporting_the_same_public_message_is_a_new_report(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter, visibility="public")
    _file(floor, "message", message_id)

    second = _file(floor, "message", message_id, reporter=floor.other)

    assert second.duplicate is False
    assert len(_report_rows(floor)) == 2 and len(notices) == 2


def test_no_owner_means_no_notice_and_the_report_stands(floor, notices, limiter, monkeypatch, caplog):
    floor.db.execute(text("DELETE FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": floor.ws})
    floor.db.commit()
    # The local edition falls back to the operator by email: name one nobody has.
    monkeypatch.setattr(owner_mod.config, "LOCAL_OPERATOR_EMAIL", f"nobody-{uuid.uuid4().hex}@reports.example.test")
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)

    with caplog.at_level(logging.WARNING, logger=service.logger.name):
        filed = _file(floor, "message", message_id)

    assert notices == []
    assert [r.id for r in _report_rows(floor)] == [filed.report_id]
    assert any("no owner" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_an_audit_write_that_fails_records_nothing_and_notifies_nobody(floor, notices, limiter, monkeypatch):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)
    monkeypatch.setattr(AuditService, "log", lambda self, **kwargs: None)

    with pytest.raises(service.ReportNotRecorded):
        _file(floor, "message", message_id)
    assert notices == []


def test_a_notice_that_fails_is_logged_and_the_report_stands(floor, limiter, monkeypatch, caplog):
    class _Broken:
        def __init__(self, db, workspace_id):
            pass

        async def dispatch(self, **kwargs):
            raise RuntimeError("notifications table unavailable")

    monkeypatch.setattr(service, "NotificationDispatcher", _Broken)
    message_id = _chat_message(floor.db, floor.ws, floor.reporter)
    floor.db.commit()

    with caplog.at_level(logging.ERROR, logger=service.logger.name):
        filed = _file(floor, "message", message_id)

    assert filed.duplicate is False
    assert [r.id for r in _report_rows(floor)] == [filed.report_id]
    assert any(r.exc_info and "notice failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# What the report keeps, and what the owner's notice never carries
# ---------------------------------------------------------------------------


def test_the_excerpt_of_a_long_message_is_at_most_200_characters(floor, notices, limiter):
    message_id = _chat_message(floor.db, floor.ws, floor.reporter, text_="word " * 1000)

    filed = _file(floor, "message", message_id)

    assert len("word " * 1000) == 5000
    assert len(filed.excerpt) <= EXCERPT_MAX_CHARS == 200
    (row,) = _report_rows(floor)
    assert row.details["excerpt"] == filed.excerpt


def test_the_notice_carries_neither_the_excerpt_nor_the_note(floor, notices, limiter):
    secret_reply = "The access code for the vault is 4417-ALPHA."
    secret_note = "Reported because Jordan's phone number is in it."
    message_id = _chat_message(floor.db, floor.ws, floor.reporter, text_=secret_reply)

    _file(floor, "message", message_id, reason="privacy", note=secret_note)

    (_, notice), = notices
    shown = f"{notice['title']} {notice['message']}"
    assert "4417" not in shown and "vault" not in shown
    assert "Jordan" not in shown and "phone" not in shown
    assert "An Auto message" in notice["message"] and "privacy" in notice["message"]


def test_the_notice_message_names_the_kind_and_the_reason_only():
    assert service.notice_message("deliverable", "offensive") == (
        "A Deliverable was reported as offensive. Open Governance to review it."
    )

"""PRD-266 US-011b (§4.4, K6, K7, K8, K13, K15) — the fan-out: the right phones for each push type.

``services/push/recipients.py`` and ``services/push/fanout.py``, on Postgres with a fake
``PushSender``. Proven here:

* Who: a question reaches the owner's, an admin's and an editor's phones and never a
  viewer's (D16); a review reaches the roles holding ``missions:update``; ``auto_message``
  reaches the named person only (nobody without one); a member made inactive (or a viewer),
  or a deactivated account, after the event receives nothing; the local operator owns every
  workspace.
* Which phones: never another workspace's, a revoked, push-off, token-less or web one; clerk
  rows on SaaS and paired rows on a local install (§8); nothing from a local install until
  the operator agreed (K13); a person's push row beats the workspace default, both ways (K7).
* What: a signed v2 payload per phone (its own push key), the question and its options or,
  redacted, the generic line; a grant from another workspace, or one already answered, is
  never read.
* After: no transaction is held while the sender works; accepted tickets are kept and a
  message with no ticket counts as refused; DeviceNotRegistered turns that phone off with the
  reason and a ``push_failed`` event and keeps its token (K15), but never a phone that
  registered a new token during the send; a sender that raises never raises out of
  ``deliver``; nothing logs a token, a push key, a question, an option or a body, and a
  database error is logged by its class, never its bound parameters (a push token).
* ``hand_over``: False and nothing handed over when no phone in the workspace can be pushed,
  or when its check fails (logged; the caller's transaction stays usable); otherwise
  ``deliver`` runs on push's own threads under an event loop (never ``core.best_effort``'s,
  which write usage and billing rows), inline without one, with no recipient work on the
  caller's session.

CI's test database has no ``notification_preferences`` (no model; ``init_test_db`` does not
build it), so the fixture runs the real ``prd128_notifications`` revision's DDL inside the
test's transaction, which is rolled back with it.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import threading
import uuid
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import bindparam, text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from config import config  # noqa: E402
from core.models.approval_grants import KIND_QUESTION, SUBJECT_TOOL_CALL, ApprovalGrant  # noqa: E402
from core.models.core import BoardTask  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.mobile_push import MobilePushTicket  # noqa: E402
from services.push import consent, fanout, recipients  # noqa: E402
from services.push.fanout import PushEvent  # noqa: E402
from services.push.payload import PUSH_TYPE_EVENTS, QUESTION_LINE, verify_signature  # noqa: E402
from services.push.sender import DEVICE_NOT_REGISTERED, TICKET_ERROR, TICKET_OK, PushTicket  # noqa: E402
from services.ticket_numbers import format_number  # noqa: E402

PREFERENCES_REVISION = ORCH / "alembic" / "versions" / "prd128_notifications.py"
T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
QUESTION_MD = "Ship the **Lisbon** deck?"
QUESTION_TEXT = "Ship the Lisbon deck?"
OPTIONS = ["Yes, ship it", "Hold it"]
CONSENT_KEYS = (consent.KEY_PUSH_VIA_EXPO, consent.KEY_ANSWERED_AT)


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeSender:
    """Accepts every message (a fresh ticket id) unless ``errors`` names its token's error code."""

    def __init__(self, errors=None, raises=None, during=None):
        self.sent = []
        self.errors = dict(errors or {})
        self.raises = raises
        self.during = during

    def send(self, messages, /):
        if self.during is not None:
            self.during()
        if self.raises is not None:
            raise self.raises
        self.sent.extend(messages)
        return [
            PushTicket(to=m.to, status=TICKET_ERROR, error=self.errors[m.to]) if m.to in self.errors
            else PushTicket(to=m.to, status=TICKET_OK, ticket_id=str(uuid.uuid4()))
            for m in messages
        ]

    def receipts(self, ticket_ids, /):
        return {}

    def to(self) -> set:
        return {message.to for message in self.sent}


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    ``deliver`` commits, and rolls back when its sender raises; here either reaches the last
    savepoint only, and every seed commits (``_settle``), so the rollback never takes the seed.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    db.commit()


def _revision_statements(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    statements: list = []
    revision.op = SimpleNamespace(execute=statements.append)
    revision.upgrade()
    return statements


@pytest.fixture(autouse=True)
def preferences_table(db_session):
    """``notification_preferences`` as production has it (prd128's DDL; IF NOT EXISTS)."""
    for statement in _revision_statements(PREFERENCES_REVISION):
        db_session.execute(text(statement))
    _settle(db_session)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(fanout, "_now", lambda: T0)


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    return config


@pytest.fixture
def local(monkeypatch, db_session):
    """A local install with no stored answer; ``agree(True|False)`` records the operator's."""
    from core.database import database

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    db_session.execute(
        text("DELETE FROM system_settings WHERE category = :category AND key IN :keys")
        .bindparams(bindparam("keys", expanding=True)),
        {"category": consent.CONSENT_CATEGORY, "keys": list(CONSENT_KEYS)},
    )
    _settle(db_session)
    monkeypatch.setattr(database, "SessionLocal", lambda: _BorrowedSession(db_session))

    def agree(agreed: bool) -> None:
        consent.set_push_consent(db_session, agreed)
        _settle(db_session)

    return agree


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
    _settle(db_session)
    return workspace_id


# --------------------------------------------------------------------------- seeds


def _user(db, email=None) -> int:
    tag = uuid.uuid4().hex[:12]
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-fanout-{tag}", "e": email or f"prd266-fanout-{tag}@t.test"},
    ).scalar()
    _settle(db)
    return pk


def _member(db, workspace_id, role, *, active=True) -> int:
    user_pk = _user(db)
    db.execute(text(
        "INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
        "VALUES (CAST(:ws AS uuid), :uid, :role, :active)"),
        {"ws": workspace_id, "uid": user_pk, "role": role, "active": active})
    _settle(db)
    return user_pk


def _owner(db, workspace_id) -> int:
    user_pk = _user(db)
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": user_pk, "ws": workspace_id})
    _settle(db)
    return user_pk


def _phone(db, workspace_id, user_pk, *, kind="clerk", platform="ios", status="active",
           push_enabled=True, token=True) -> MobileDevice:
    tag = uuid.uuid4().hex
    device = MobileDevice(
        workspace_id=uuid.UUID(str(workspace_id)), user_id=user_pk, kind=kind, status=status,
        install_id=f"prd266-fanout-{tag[:12]}", platform=platform, locale="en-IE",
        expo_push_token=f"ExponentPushToken[{tag}]" if token else None,
        push_enabled=push_enabled, push_key=f"prd266-fanout-key-{tag}", events=[],
    )
    db.add(device)
    _settle(db)
    return device


def _question(db, workspace_id) -> int:
    grant = ApprovalGrant(workspace_id=uuid.UUID(str(workspace_id)), subject_type=SUBJECT_TOOL_CALL,
                          subject_id=f"call-{uuid.uuid4().hex[:8]}", kind=KIND_QUESTION,
                          question_md=QUESTION_MD, options=list(OPTIONS))
    db.add(grant)
    _settle(db)
    return grant.id


def _push_row(db, workspace_id, user_pk, event_type, enabled, destination="push") -> None:
    db.execute(text(
        "INSERT INTO notification_preferences (workspace_id, user_id, event_type, destination, enabled) "
        "VALUES (CAST(:ws AS uuid), :uid, :event_type, :destination, :enabled)"),
        {"ws": workspace_id, "uid": user_pk, "event_type": event_type,
         "destination": destination, "enabled": enabled})
    _settle(db)


def _question_event(workspace_id, grant_id, **overrides) -> PushEvent:
    fields = {"workspace_id": workspace_id, "event_type": "question_pending", "notification_id": "n-1",
              "link_type": "question", "link_id": str(grant_id), **overrides}
    return PushEvent(**fields)


def _deliver(db, event, sender=None):
    sender = sender if sender is not None else _FakeSender()
    counts = fanout.deliver(event, sender=sender, session_factory=lambda: _BorrowedSession(db))
    return sender, counts


def _tokens(*devices) -> set:
    return {device.expo_push_token for device in devices}


# --------------------------------------------------------------------------- who


def test_a_question_reaches_the_owner_admin_and_editor_but_never_a_viewer(saas, ws, db_session):
    phones = {role: _phone(db_session, ws, _member(db_session, ws, role)) for role in ("admin", "editor", "viewer")}
    phones["owner"] = _phone(db_session, ws, _owner(db_session, ws))
    keys = {phone.expo_push_token: phone.push_key for phone in phones.values()}
    grant_id = _question(db_session, ws)

    sender, counts = _deliver(db_session, _question_event(ws, grant_id))

    assert sender.to() == _tokens(phones["owner"], phones["admin"], phones["editor"])
    assert (counts["recipients"], counts["devices"], counts["sent"]) == (3, 3, 3)
    for message in sender.sent:
        assert message.body == QUESTION_TEXT
        assert message.category_id == "automatos.question.2"
        assert message.data["question"] == QUESTION_TEXT and message.data["options"] == OPTIONS
        assert message.data["grant_id"] == str(grant_id) and message.data["workspace_id"] == ws
        assert message.data["notification_id"] == "n-1" and message.data["type"] == "needs_answer"
        assert verify_signature(keys[message.to], message.data, T0) is True


@pytest.mark.parametrize("event_type, link_type", [("approval_pending", "approval_grant"), ("mission_plan_ready", "mission")])
def test_every_needs_answer_event_skips_a_viewer_and_a_named_person_must_be_allowed(saas, ws, db_session,
                                                                                   event_type, link_type):
    editor = _member(db_session, ws, "editor")
    viewer = _member(db_session, ws, "viewer")
    editor_phone, viewer_phone = _phone(db_session, ws, editor), _phone(db_session, ws, viewer)
    event = PushEvent(workspace_id=ws, event_type=event_type, link_type=link_type, link_id="7")

    assert _deliver(db_session, event)[0].to() == _tokens(editor_phone)
    named = PushEvent(workspace_id=ws, event_type=event_type, link_type=link_type, link_id="7", user_id=editor)
    assert _deliver(db_session, named)[0].to() == _tokens(editor_phone)
    to_viewer = PushEvent(workspace_id=ws, event_type=event_type, link_type=link_type, link_id="7", user_id=viewer)
    sender, counts = _deliver(db_session, to_viewer)
    assert sender.sent == [] and counts["recipients"] == 0
    assert viewer_phone.push_enabled is True


def test_a_review_reaches_the_roles_holding_missions_update(saas, ws, db_session):
    phones = {role: _phone(db_session, ws, _member(db_session, ws, role))
              for role in ("admin", "editor", "viewer", "member")}
    phones["owner"] = _phone(db_session, ws, _owner(db_session, ws))
    task = BoardTask(workspace_id=uuid.UUID(ws), title="Lisbon deck")  # the fan-out reads its number only
    db_session.add(task)
    _settle(db_session)
    number = format_number(task.workspace_seq)

    event = PushEvent(workspace_id=ws, event_type="review_pending", link_type="task", link_id=str(task.id))
    sender, _counts = _deliver(db_session, event)

    assert sender.to() == _tokens(phones["owner"], phones["admin"], phones["editor"])
    assert number and all(message.data["ticket_number"] == number for message in sender.sent)
    assert all(message.body == f"A ticket is ready for review · {number}" for message in sender.sent)
    assert all(message.data["type"] == "review" and "grant_id" not in message.data for message in sender.sent)


def test_auto_message_reaches_the_named_person_only_and_nobody_without_one(saas, ws, db_session):
    writer, other = _member(db_session, ws, "viewer"), _member(db_session, ws, "editor")
    writer_phone = _phone(db_session, ws, writer)
    _phone(db_session, ws, other)
    chat_id = str(uuid.uuid4())

    named = PushEvent(workspace_id=ws, event_type="auto_message", link_type="chat", link_id=chat_id, user_id=writer)
    sender, counts = _deliver(db_session, named)
    assert sender.to() == _tokens(writer_phone) and counts["recipients"] == 1
    assert sender.sent[0].data["link_id"] == chat_id and sender.sent[0].body == "Auto messaged you"

    nobody = PushEvent(workspace_id=ws, event_type="auto_message", link_type="chat", link_id=chat_id)
    sender, counts = _deliver(db_session, nobody)
    assert sender.sent == [] and counts == dict.fromkeys(fanout.COUNT_KEYS, 0)


def test_a_member_made_inactive_or_a_viewer_after_the_event_receives_nothing(saas, ws, db_session):
    leaving, demoted, disabled, staying = (_member(db_session, ws, "editor") for _ in range(4))
    phones = {pk: _phone(db_session, ws, pk) for pk in (leaving, demoted, disabled, staying)}
    event = _question_event(ws, _question(db_session, ws))

    db_session.execute(text("UPDATE workspace_members SET is_active = false WHERE user_id = :uid"), {"uid": leaving})
    db_session.execute(text("UPDATE workspace_members SET role = 'viewer' WHERE user_id = :uid"), {"uid": demoted})
    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :uid"), {"uid": disabled})
    _settle(db_session)

    assert _deliver(db_session, event)[0].to() == _tokens(phones[staying])
    named = _question_event(ws, event.link_id, user_id=leaving)
    assert _deliver(db_session, named)[0].sent == []


def test_the_local_operator_owns_every_workspace(local, ws, db_session, monkeypatch, seed_workspace):
    operator_email = f"prd266-operator-{uuid.uuid4().hex[:8]}@t.test"
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", operator_email)
    operator = _user(db_session, operator_email)
    other_ws = seed_workspace()
    _settle(db_session)
    phones = [_phone(db_session, workspace, operator, kind="paired") for workspace in (ws, other_ws)]
    local(True)

    for phone, workspace in zip(phones, (ws, other_ws)):
        assert _deliver(db_session, _question_event(workspace, "1"))[0].to() == _tokens(phone)
    assert recipients.recipients(db_session, ws, "needs_answer") == [operator]

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    assert recipients.recipients(db_session, ws, "needs_answer") == []


# --------------------------------------------------------------------------- which phones


def test_another_workspaces_phones_and_grant_are_never_used(saas, ws, db_session, seed_workspace):
    other_ws = seed_workspace()
    _settle(db_session)
    person = _member(db_session, ws, "editor")
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :uid, 'editor', true)"), {"ws": other_ws, "uid": person})
    _settle(db_session)
    here, there = _phone(db_session, ws, person), _phone(db_session, other_ws, person)
    foreign_grant = _question(db_session, other_ws)

    sender, _counts = _deliver(db_session, _question_event(ws, foreign_grant))

    assert sender.to() == _tokens(here) and there.expo_push_token not in sender.to()
    data = sender.sent[0].data
    assert "question" not in data and "options" not in data and "grant_id" not in data
    assert sender.sent[0].body == QUESTION_LINE


def test_a_revoked_push_off_tokenless_or_web_phone_never_receives(saas, ws, db_session):
    person = _member(db_session, ws, "admin")
    good = _phone(db_session, ws, person)
    _phone(db_session, ws, person, status="revoked")
    _phone(db_session, ws, person, status="pending")
    _phone(db_session, ws, person, push_enabled=False)
    _phone(db_session, ws, person, token=False)
    _phone(db_session, ws, person, platform="web")

    sender, counts = _deliver(db_session, _question_event(ws, _question(db_session, ws)))

    assert sender.to() == _tokens(good) and counts["devices"] == 1


def test_saas_pushes_clerk_rows_only_and_a_local_install_paired_rows_only(local, ws, db_session, monkeypatch):
    person = _owner(db_session, ws)
    clerk, paired = _phone(db_session, ws, person, kind="clerk"), _phone(db_session, ws, person, kind="paired")
    local(True)
    event = _question_event(ws, _question(db_session, ws))

    assert _deliver(db_session, event)[0].to() == _tokens(paired)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    assert _deliver(db_session, event)[0].to() == _tokens(clerk)


def test_a_local_install_sends_nothing_without_consent_and_sends_with_it(local, ws, db_session):
    phone = _phone(db_session, ws, _owner(db_session, ws), kind="paired")
    event = _question_event(ws, _question(db_session, ws))

    assert _deliver(db_session, event)[0].sent == []
    assert fanout.hand_over(db_session, event) is False
    local(False)
    assert _deliver(db_session, event)[0].sent == []
    local(True)
    assert _deliver(db_session, event)[0].to() == _tokens(phone)


def test_a_persons_push_row_off_beats_the_workspace_default_on(saas, ws, db_session):
    quiet, loud = _member(db_session, ws, "editor"), _member(db_session, ws, "editor")
    quiet_phone, loud_phone = _phone(db_session, ws, quiet), _phone(db_session, ws, loud)
    _push_row(db_session, ws, None, "question_pending", True)
    _push_row(db_session, ws, quiet, "question_pending", False)
    _push_row(db_session, ws, loud, "question_pending", False, destination="in_app")  # not a push switch
    _push_row(db_session, ws, loud, "approval_pending", False)  # another event's switch

    assert _deliver(db_session, _question_event(ws, "1"))[0].to() == _tokens(loud_phone)
    assert quiet_phone.push_enabled is True


def test_a_persons_push_row_on_beats_the_workspace_default_off(saas, ws, db_session):
    keen, other = _member(db_session, ws, "editor"), _member(db_session, ws, "editor")
    keen_phone = _phone(db_session, ws, keen)
    _phone(db_session, ws, other)
    _push_row(db_session, ws, None, "question_pending", False)
    _push_row(db_session, ws, keen, "question_pending", True)

    assert _deliver(db_session, _question_event(ws, "1"))[0].to() == _tokens(keen_phone)


# --------------------------------------------------------------------------- what


def test_redaction_drops_the_question_and_its_options(saas, ws, db_session):
    db_session.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:ws AS uuid)"),
                       {"s": '{"push": {"redact_questions": true}}', "ws": ws})
    _settle(db_session)
    phone = _phone(db_session, ws, _owner(db_session, ws))
    grant_id = _question(db_session, ws)

    sender, _counts = _deliver(db_session, _question_event(ws, grant_id))

    (message,) = sender.sent
    assert "question" not in message.data and "options" not in message.data
    assert message.data["grant_id"] == str(grant_id)
    assert message.body == QUESTION_LINE and message.category_id == "automatos.open"
    assert verify_signature(phone.push_key, message.data, T0) is True


def test_an_answered_question_is_pushed_generic(saas, ws, db_session):
    _phone(db_session, ws, _owner(db_session, ws))
    grant_id = _question(db_session, ws)
    db_session.execute(text("UPDATE approval_grants SET status = 'granted' WHERE id = :id"), {"id": grant_id})
    _settle(db_session)

    (message,) = _deliver(db_session, _question_event(ws, grant_id))[0].sent

    assert "question" not in message.data and "grant_id" not in message.data
    assert message.body == QUESTION_LINE and message.category_id == "automatos.open"


# --------------------------------------------------------------------------- after


def _tickets(db, workspace_id) -> list:
    return db.query(MobilePushTicket).filter(MobilePushTicket.workspace_id == uuid.UUID(workspace_id)).all()


def test_accepted_tickets_are_kept_and_a_refused_one_is_not(saas, ws, db_session):
    person = _owner(db_session, ws)
    taken, refused = _phone(db_session, ws, person), _phone(db_session, ws, person)
    sender = _FakeSender(errors={refused.expo_push_token: "MessageTooBig"})

    _sender, counts = _deliver(db_session, _question_event(ws, "1"), sender)

    rows = _tickets(db_session, ws)
    assert [(row.device_id, row.push_type) for row in rows] == [(taken.id, "needs_answer")]
    assert (counts["sent"], counts["refused"], counts["disabled"]) == (1, 1, 0)
    db_session.refresh(refused)
    assert refused.push_enabled is True and refused.push_disabled_reason is None


def test_device_not_registered_turns_that_phone_off_and_keeps_its_token(saas, ws, db_session):
    person = _owner(db_session, ws)
    gone, kept = _phone(db_session, ws, person), _phone(db_session, ws, person)
    gone_token = gone.expo_push_token
    sender = _FakeSender(errors={gone_token: DEVICE_NOT_REGISTERED})

    _sender, counts = _deliver(db_session, _question_event(ws, "1"), sender)

    db_session.refresh(gone)
    db_session.refresh(kept)
    assert counts["disabled"] == 1
    assert (gone.push_enabled, gone.push_disabled_reason) == (False, "DeviceNotRegistered")
    assert gone.expo_push_token == gone_token
    assert gone.events[-1] == {"event": "push_failed", "at": T0.isoformat()}
    assert (kept.push_enabled, kept.push_disabled_reason, kept.events) == (True, None, [])
    assert [row.device_id for row in _tickets(db_session, ws)] == [kept.id]
    assert _deliver(db_session, _question_event(ws, "1"))[0].to() == _tokens(kept)


def test_no_transaction_is_held_while_expo_answers_and_a_new_token_stays_on(saas, ws, db_session):
    phone = _phone(db_session, ws, _owner(db_session, ws))
    device_id, old_token = phone.id, phone.expo_push_token
    new_token = f"ExponentPushToken[{uuid.uuid4().hex}]"
    held = []

    def reregistered_during_the_send():
        held.append(db_session.in_transaction())
        db_session.execute(text("UPDATE mobile_devices SET expo_push_token = :t WHERE id = :id"),
                           {"t": new_token, "id": device_id})
        _settle(db_session)

    sender = _FakeSender(errors={old_token: DEVICE_NOT_REGISTERED}, during=reregistered_during_the_send)
    _sender, counts = _deliver(db_session, _question_event(ws, "1"), sender)

    assert held == [False]
    db_session.refresh(phone)
    assert counts["disabled"] == 0
    assert (phone.push_enabled, phone.push_disabled_reason, phone.expo_push_token) == (True, None, new_token)


def test_a_message_the_sender_gave_no_ticket_for_counts_as_refused(saas, ws, db_session, caplog):
    person = _owner(db_session, ws)
    _phone(db_session, ws, person)
    _phone(db_session, ws, person)

    class _Short(_FakeSender):
        def send(self, messages, /):
            return super().send(messages)[:1]

    with caplog.at_level(logging.ERROR, logger=fanout.__name__):
        _sender, counts = _deliver(db_session, _question_event(ws, "1"), _Short())

    assert (counts["devices"], counts["sent"], counts["refused"]) == (2, 1, 1)
    assert "1 tickets for 2 messages" in caplog.text
    assert len(_tickets(db_session, ws)) == 1


def test_a_sender_that_raises_never_raises_out_of_deliver(saas, ws, db_session, caplog):
    phone = _phone(db_session, ws, _owner(db_session, ws))

    with caplog.at_level(logging.ERROR, logger=fanout.__name__):
        _sender, counts = _deliver(db_session, _question_event(ws, "1"), _FakeSender(raises=RuntimeError("down")))

    assert counts == dict.fromkeys(fanout.COUNT_KEYS, 0)
    assert "Push fan-out failed" in caplog.text
    assert _tickets(db_session, ws) == []
    db_session.refresh(phone)
    assert phone.push_enabled is True


def test_a_database_error_is_logged_by_its_class_never_its_bound_token(saas, ws, db_session, monkeypatch,
                                                                       caplog):
    from sqlalchemy.exc import OperationalError

    token = _phone(db_session, ws, _owner(db_session, ws)).expo_push_token

    def lost_connection(db, sent, now):
        raise OperationalError("SELECT mobile_devices.id FROM mobile_devices "
                               "WHERE mobile_devices.expo_push_token = %(token)s FOR UPDATE",
                               {"token": sent.token}, Exception("server closed the connection unexpectedly"))

    monkeypatch.setattr(fanout, "_turn_off", lost_connection)
    with caplog.at_level(logging.DEBUG, logger=fanout.__name__):
        _sender, counts = _deliver(db_session, _question_event(ws, "1"),
                                   _FakeSender(errors={token: DEVICE_NOT_REGISTERED}))

    assert counts == dict.fromkeys(fanout.COUNT_KEYS, 0)
    assert "Push fan-out failed" in caplog.text and "OperationalError" in caplog.text
    assert token not in caplog.text and "server closed" not in caplog.text


def test_nothing_is_logged_but_ids_and_counts(saas, ws, db_session, caplog):
    person = _owner(db_session, ws)
    gone, kept = _phone(db_session, ws, person), _phone(db_session, ws, person)
    secrets = [gone.expo_push_token, kept.expo_push_token, gone.push_key, kept.push_key, QUESTION_TEXT, *OPTIONS]

    grant_id = _question(db_session, ws)

    with caplog.at_level(logging.DEBUG, logger="services"):
        sender, _counts = _deliver(db_session, _question_event(ws, grant_id),
                                   _FakeSender(errors={gone.expo_push_token: DEVICE_NOT_REGISTERED}))
        _deliver(db_session, _question_event(ws, grant_id), _FakeSender(raises=RuntimeError("expo down")))

    assert sender.sent and "Push fan-out" in caplog.text
    assert not [secret for secret in secrets if secret in caplog.text]
    assert not [message.body for message in sender.sent if message.body in caplog.text]


# --------------------------------------------------------------------------- hand_over


class _Recorder:
    def __init__(self):
        self.calls = []
        self.ran = threading.Event()

    def __call__(self, event, **kwargs):
        self.calls.append((event, threading.current_thread().name, kwargs))
        self.ran.set()
        return dict.fromkeys(fanout.COUNT_KEYS, 0)


def test_hand_over_returns_false_and_hands_nothing_over_when_no_phone_can_be_pushed(saas, ws, db_session,
                                                                                    monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(fanout, "deliver", recorder)
    person = _owner(db_session, ws)
    event = _question_event(ws, "1")

    assert fanout.hand_over(db_session, event) is False  # no phone at all
    for kwargs in ({"status": "revoked"}, {"push_enabled": False}, {"token": False}, {"platform": "web"},
                   {"kind": "paired"}):
        _phone(db_session, ws, person, **kwargs)
    assert fanout.hand_over(db_session, event) is False
    _phone(db_session, ws, person)
    assert fanout.hand_over(db_session, PushEvent(workspace_id=ws, event_type="task_complete")) is False
    assert recorder.calls == []


def test_a_hand_over_whose_check_fails_is_false_logged_and_leaves_the_caller_usable(saas, ws, db_session,
                                                                                    monkeypatch, caplog):
    recorder = _Recorder()
    monkeypatch.setattr(fanout, "deliver", recorder)
    _phone(db_session, ws, _owner(db_session, ws))

    with caplog.at_level(logging.ERROR, logger=fanout.__name__):
        assert fanout.hand_over(db_session, PushEvent(workspace_id="not-a-uuid", event_type="question_pending")) is False
        monkeypatch.setattr(fanout, "has_eligible_device",
                            lambda db, _ws: db.execute(text("SELECT * FROM no_such_table_prd266")))
        assert fanout.hand_over(db_session, _question_event(ws, "1")) is False

    assert caplog.text.count("Push hand-over failed") == 2
    assert db_session.execute(text("SELECT 1")).scalar() == 1  # the savepoint kept the caller's transaction
    assert recorder.calls == []


def test_hand_over_runs_deliver_on_the_push_threads_without_recipient_work(saas, ws, db_session, monkeypatch):
    from core import best_effort

    recorder = _Recorder()
    monkeypatch.setattr(fanout, "deliver", recorder)
    monkeypatch.setattr(fanout, "recipients", lambda *a, **k: pytest.fail("hand_over reads no recipients"))
    monkeypatch.setattr(best_effort, "_submit",
                        lambda *a, **k: pytest.fail("push never takes the best-effort threads"))
    _phone(db_session, ws, _member(db_session, ws, "viewer"))
    event = _question_event(ws, "1")

    async def on_the_loop():
        return fanout.hand_over(db_session, event)

    assert asyncio.run(on_the_loop()) is True
    assert recorder.ran.wait(timeout=5) is True
    assert fanout.hand_over(db_session, event) is True  # no loop: inline
    (on_loop, thread_name, _kw), (inline, inline_thread, _kw2) = recorder.calls
    assert on_loop is event and thread_name.startswith(f"{fanout.PUSH_THREAD_PREFIX}_")
    assert inline is event and inline_thread == threading.current_thread().name
    assert fanout._push_executor._max_workers == fanout.PUSH_DELIVERY_THREADS == 2


# --------------------------------------------------------------------------- the rules


def test_the_rules_cover_exactly_the_push_types_and_reuse_the_answerer_roles():
    from core.auth import workspace_admin

    assert set(recipients.ROLE_RULES) == set(PUSH_TYPE_EVENTS)
    assert recipients.WORKSPACE_ANSWERER_ROLES is workspace_admin.WORKSPACE_ANSWERER_ROLES
    assert [role for role in ("owner", "admin", "editor", "viewer") if recipients.ROLE_RULES["needs_answer"](role)] \
        == ["owner", "admin", "editor"]
    assert [role for role in ("owner", "admin", "editor", "viewer") if recipients.ROLE_RULES["review"](role)] \
        == ["owner", "admin", "editor"]
    source = (ORCH / "services" / "push" / "recipients.py").read_text()
    assert '"editor"' not in source and "'editor'" not in source  # one list: W1's
    with pytest.raises(FrozenInstanceError):
        PushEvent(workspace_id="w", event_type="question_pending").user_id = 1  # type: ignore[misc]

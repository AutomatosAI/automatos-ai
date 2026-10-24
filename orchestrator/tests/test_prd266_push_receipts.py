"""PRD-266 US-010e (§4.4, §4.8, K15, D21) — the receipts job: Expo's receipts turn off phones that left.

``services/push/receipts.py`` on Postgres with a fake ``PushSender``, at a fixed clock. Proven here:

* A ``DeviceNotRegistered`` receipt turns off only that phone, with the reason and one
  ``push_failed`` event however many of its tickets say so, and keeps its token (K15); a phone
  revoked since keeps ``device_revoked``. An ok receipt deletes its ticket, and so does any
  other error, whose code (never a token) is logged.
* The window: a ticket younger than 15 minutes is not asked about; a 25-hour-old one is
  deleted unread; one with no receipt yet stays; ids go in batches of at most 1,000, the
  reads ended (no transaction held) while the sender answers, every ticket asked once per run.
* A sender that raises leaves the tickets and never raises out; a local install whose operator
  has not agreed asks Expo nothing (D21).
* ``register`` adds one interval job with no args (a Redis job store pickles them) and is
  False without a running scheduler; ``start_schedule_reconcile`` registers it on the leader.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text

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
from core.models.mobile_devices import MobileDevice  # noqa: E402
from core.models.mobile_push import MobilePushTicket  # noqa: E402
from services import schedule_reconcile  # noqa: E402
from services.push import consent, receipts  # noqa: E402
from services.push.sender import DEVICE_NOT_REGISTERED, TICKET_ERROR, TICKET_OK, PushReceipt  # noqa: E402

T0 = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
RIPE = timedelta(minutes=20)


class _BorrowedSession:
    """The test's session, lent to code that opens its own: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


class _FakeSender:
    """Answers ``receipts`` from ``answers`` (ticket id → receipt); records every batch asked."""

    def __init__(self, answers=None, raises=None, during=None):
        self.answers = dict(answers or {})
        self.raises = raises
        self.during = during
        self.asked = []

    def send(self, messages, /):
        raise AssertionError("the receipts job never sends")

    def receipts(self, ticket_ids, /):
        self.asked.append(list(ticket_ids))
        if self.during is not None:
            self.during()
        if self.raises is not None:
            raise self.raises
        return {ticket_id: self.answers[ticket_id] for ticket_id in ticket_ids if ticket_id in self.answers}

    def every_id_asked(self) -> list:
        return [ticket_id for batch in self.asked for ticket_id in batch]


def _ok(ticket_id) -> PushReceipt:
    return PushReceipt(ticket_id=ticket_id, status=TICKET_OK)


def _error(ticket_id, code) -> PushReceipt:
    return PushReceipt(ticket_id=ticket_id, status=TICKET_ERROR, error=code)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    ``run_receipts`` commits once per batch and rolls back when its sender raises; here either
    reaches the last savepoint only, and every seed commits (``_settle``), so a rollback never
    takes the seed.
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


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    return config


@pytest.fixture
def ws(seed_workspace, db_session) -> str:
    workspace_id = seed_workspace()
    _settle(db_session)
    return workspace_id


def _user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-receipts-{tag}", "e": f"prd266-receipts-{tag}@t.test"},
    ).scalar()
    _settle(db)
    return pk


def _phone(db, workspace_id, *, status="active", push_enabled=True, token=True, reason=None) -> MobileDevice:
    tag = uuid.uuid4().hex
    device = MobileDevice(
        workspace_id=uuid.UUID(str(workspace_id)), user_id=_user(db), kind="clerk", status=status,
        install_id=f"prd266-receipts-{tag[:12]}", platform="ios", locale="en-IE",
        expo_push_token=f"ExponentPushToken[{tag}]" if token else None,
        push_enabled=push_enabled, push_disabled_reason=reason, push_key=f"prd266-receipts-key-{tag}", events=[],
    )
    db.add(device)
    _settle(db)
    return device


def _ticket(db, device, age=RIPE, ticket_id=None) -> str:
    ticket_id = ticket_id or str(uuid.uuid4())
    db.add(MobilePushTicket(ticket_id=ticket_id, device_id=device.id, workspace_id=device.workspace_id,
                            push_type="needs_answer", created_at=T0 - age))
    _settle(db)
    return ticket_id


def _run(db, sender):
    return receipts.run_receipts(sender=sender, session_factory=lambda: _BorrowedSession(db), now=T0)


def _kept(db) -> set:
    db.expire_all()
    return {row.ticket_id for row in db.query(MobilePushTicket.ticket_id).all()}


# --------------------------------------------------------------------------- what a receipt does


def test_device_not_registered_turns_off_only_that_phone_with_the_reason_and_one_event(saas, ws, db_session):
    gone, kept = _phone(db_session, ws), _phone(db_session, ws)
    gone_token = gone.expo_push_token
    gone_tickets = [_ticket(db_session, gone), _ticket(db_session, gone)]
    kept_ticket = _ticket(db_session, kept)
    answers = {**{t: _error(t, DEVICE_NOT_REGISTERED) for t in gone_tickets}, kept_ticket: _ok(kept_ticket)}

    counts = _run(db_session, _FakeSender(answers))

    db_session.refresh(gone)
    db_session.refresh(kept)
    assert (gone.push_enabled, gone.push_disabled_reason) == (False, "DeviceNotRegistered")
    assert gone.expo_push_token == gone_token
    assert gone.events == [{"event": "push_failed", "at": T0.isoformat()}]
    assert (kept.push_enabled, kept.push_disabled_reason, kept.events) == (True, None, [])
    assert _kept(db_session) == set()
    assert (counts["asked"], counts["delivered"], counts["failed"], counts["disabled"]) == (3, 1, 2, 1)


def test_a_phone_revoked_or_already_off_keeps_its_reason(saas, ws, db_session):
    revoked = _phone(db_session, ws, status="revoked", push_enabled=False, token=False, reason="device_revoked")
    switched_off = _phone(db_session, ws, push_enabled=False)
    tickets = [_ticket(db_session, revoked), _ticket(db_session, switched_off)]

    counts = _run(db_session, _FakeSender({t: _error(t, DEVICE_NOT_REGISTERED) for t in tickets}))

    db_session.refresh(revoked)
    db_session.refresh(switched_off)
    assert (revoked.push_disabled_reason, revoked.events) == ("device_revoked", [])
    assert (switched_off.push_disabled_reason, switched_off.events) == (None, [])
    assert counts["disabled"] == 0 and _kept(db_session) == set()


def test_an_ok_receipt_deletes_its_ticket_and_another_error_is_logged_by_code_and_deleted(saas, ws, db_session,
                                                                                         caplog):
    phone = _phone(db_session, ws)
    delivered, too_fast = _ticket(db_session, phone), _ticket(db_session, phone)
    sender = _FakeSender({delivered: _ok(delivered), too_fast: _error(too_fast, "MessageRateExceeded")})

    with caplog.at_level(logging.DEBUG, logger="services"):
        counts = _run(db_session, sender)

    db_session.refresh(phone)
    assert _kept(db_session) == set()
    assert (phone.push_enabled, phone.push_disabled_reason, phone.events) == (True, None, [])
    assert (counts["delivered"], counts["failed"], counts["disabled"]) == (1, 1, 0)
    assert "MessageRateExceeded=1" in caplog.text
    assert phone.expo_push_token not in caplog.text and phone.push_key not in caplog.text


# --------------------------------------------------------------------------- which tickets


def test_a_ticket_younger_than_fifteen_minutes_is_not_read(saas, ws, db_session):
    phone = _phone(db_session, ws)
    fresh = _ticket(db_session, phone, age=timedelta(minutes=14))
    sender = _FakeSender({fresh: _ok(fresh)})

    counts = _run(db_session, sender)

    assert sender.asked == [] and counts["asked"] == 0
    assert _kept(db_session) == {fresh}
    assert receipts.PUSH_RECEIPT_MIN_AGE_SECONDS == 900


def test_a_ticket_older_than_a_day_is_deleted_unread(saas, ws, db_session):
    phone = _phone(db_session, ws)
    stale = _ticket(db_session, phone, age=timedelta(hours=25))
    ripe = _ticket(db_session, phone)
    sender = _FakeSender({stale: _error(stale, DEVICE_NOT_REGISTERED), ripe: _ok(ripe)})

    counts = _run(db_session, sender)

    db_session.refresh(phone)
    assert sender.every_id_asked() == [ripe]
    assert _kept(db_session) == set() and counts["expired"] == 1
    assert phone.push_enabled is True
    assert receipts.PUSH_RECEIPT_MAX_AGE_SECONDS == 86_400


def test_a_ticket_with_no_receipt_yet_stays(saas, ws, db_session):
    phone = _phone(db_session, ws)
    waiting, answered = _ticket(db_session, phone), _ticket(db_session, phone)

    counts = _run(db_session, _FakeSender({answered: _ok(answered)}))

    assert _kept(db_session) == {waiting}
    assert (counts["asked"], counts["delivered"], counts["waiting"]) == (2, 1, 1)


def test_ids_go_in_batches_and_no_transaction_is_held_while_the_sender_answers(saas, ws, db_session, monkeypatch):
    monkeypatch.setattr(receipts, "RECEIPTS_BATCH", 2)
    phone = _phone(db_session, ws)
    same_moment = [_ticket(db_session, phone) for _ in range(5)]  # one fan-out: one created_at
    held = []
    sender = _FakeSender({same_moment[0]: _ok(same_moment[0])},
                         during=lambda: held.append(db_session.in_transaction()))

    counts = _run(db_session, sender)

    assert [len(batch) for batch in sender.asked] == [2, 2, 1]
    assert sorted(sender.every_id_asked()) == sorted(same_moment)
    assert held == [False, False, False]
    assert _kept(db_session) == set(same_moment[1:]) and counts["waiting"] == 4


def test_the_batch_size_is_expos_limit():
    assert receipts.RECEIPTS_BATCH == 1000


# --------------------------------------------------------------------------- failures and consent


def test_a_sender_that_raises_leaves_the_tickets_and_does_not_raise(saas, ws, db_session, caplog):
    phone = _phone(db_session, ws)
    tickets = {_ticket(db_session, phone), _ticket(db_session, phone)}
    stale = _ticket(db_session, phone, age=timedelta(hours=25))

    with caplog.at_level(logging.ERROR, logger=receipts.__name__):
        counts = _run(db_session, _FakeSender(raises=RuntimeError("expo down")))

    assert _kept(db_session) == tickets and stale not in _kept(db_session)
    assert counts == {**dict.fromkeys(receipts.COUNT_KEYS, 0), "expired": 1}
    assert "Push receipts: the run failed" in caplog.text
    db_session.refresh(phone)
    assert phone.push_enabled is True


def test_a_local_install_whose_operator_has_not_agreed_asks_expo_nothing(ws, db_session, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(consent, "expo_push_allowed", lambda: False)
    phone = _phone(db_session, ws)
    ripe = _ticket(db_session, phone)
    _ticket(db_session, phone, age=timedelta(hours=25))
    sender = _FakeSender({ripe: _ok(ripe)})

    counts = _run(db_session, sender)

    assert sender.asked == [] and _kept(db_session) == {ripe}
    assert counts["expired"] == 1 and counts["asked"] == 0


# --------------------------------------------------------------------------- the scheduler


class _FakeScheduler:
    """The slice of APScheduler ``register`` touches."""

    def __init__(self, running=True):
        self.running = running
        self.jobs = []

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})

    def ids(self) -> list:
        return [job["id"] for job in self.jobs]


def test_register_adds_one_job_with_no_args_and_is_false_without_a_running_scheduler():
    scheduler = _FakeScheduler()

    assert receipts.register(scheduler) is True

    (job,) = scheduler.jobs
    assert job["func"] is receipts.run_receipts and job["id"] == receipts.RECEIPTS_JOB_ID
    assert job["trigger"].interval == timedelta(seconds=config.PUSH_RECEIPTS_INTERVAL_SECONDS)
    assert "args" not in job and "kwargs" not in job
    assert (job["replace_existing"], job["max_instances"], job["coalesce"]) == (True, 1, True)
    assert config.PUSH_RECEIPTS_INTERVAL_SECONDS >= 60

    stopped = _FakeScheduler(running=False)
    assert receipts.register(None) is False and receipts.register(stopped) is False
    assert stopped.jobs == []


def test_start_schedule_reconcile_registers_it_on_the_leader(monkeypatch):
    scheduler = _FakeScheduler()
    monkeypatch.setattr(schedule_reconcile, "run_reconcile_once", lambda s, db=None: {})

    assert asyncio.run(schedule_reconcile.start_schedule_reconcile(scheduler)) is True

    assert scheduler.ids().count(receipts.RECEIPTS_JOB_ID) == 1
    (job,) = [job for job in scheduler.jobs if job["id"] == receipts.RECEIPTS_JOB_ID]
    assert job["func"] is receipts.run_receipts and "args" not in job

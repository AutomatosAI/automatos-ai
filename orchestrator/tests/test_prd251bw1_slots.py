"""PRD-251B Wave 1, US-B105 — planned slots (B11): ``PUT /slot``, approve into the slot, a
passed slot ends missed.

On the S0.3b API harness (SQLite, the real gate) with the fake scheduler. Pinned:

* before approval the slot is set alone, stored in UTC with its zone: status, hash and
  approval untouched; ``null`` clears it; an unknown zone or a bad timestamp is 422; an
  archived post is 409;
* approving a post with a future slot schedules it into the slot (one job registered);
  with a past or no slot it stays approved and unscheduled;
* a slot change on an approved or scheduled post goes through the one schedule path
  (the job moves, the approval stands); ``null`` is refused there;
* the leader's reconcile pass ends an unapproved post whose slot passed the grace
  ``missed``, once, with the missed notification; a slot inside the grace, and an
  approved post with a passed slot, are untouched;
* a missed post without an approval restarts as a draft at a new slot and can be
  submitted again; one whose approval stands is rescheduled;
* on the CI Postgres, two passes racing on one post leave one transition.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from config import config  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import publish_lifecycle, schedule_jobs, service  # noqa: E402
from tests.test_prd251_api import WS_A, _approved, _create, _post, api  # noqa: E402,F401
from tests.test_prd251w3_schedule import FakeScheduler, leader  # noqa: E402,F401

LONDON = ZoneInfo("Europe/London")


def _put_slot(api, post_id, planned_for, tz=None):
    body = {"planned_for": planned_for.isoformat() if isinstance(planned_for, datetime) else planned_for}
    if tz is not None:
        body["timezone"] = tz
    return api.client.put(f"/api/socials/posts/{post_id}/slot", json=body)


def _future(days=3):
    return datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=days)


def _past(seconds):
    return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=seconds)


def _row(api, post_id):
    api.session.expire_all()
    return service.get_post(api.session, WS_A, uuid.UUID(post_id))


# ---------------------------------------------------------------------------
# Before approval: the slot alone
# ---------------------------------------------------------------------------


def test_before_approval_the_slot_is_set_alone_in_utc_with_its_zone(api):
    post = _create(api)
    resp = _put_slot(api, post["id"], datetime(2026, 10, 14, 12, 0, tzinfo=LONDON), "Europe/London")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["planned_for"].startswith("2026-10-14T11:00")  # BST → UTC
    assert body["timezone"] == "Europe/London"
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("draft", post["content_hash"], None)
    assert body["review_log"] == post["review_log"]  # a slot is not a review event

    assert _put_slot(api, post["id"], None).json()["planned_for"] is None
    assert _put_slot(api, post["id"], _future(), "Mars/Olympus").status_code == 422
    assert api.client.put(f"/api/socials/posts/{post['id']}/slot", json={"planned_for": "yesterday"}).status_code == 422

    submitted = _post(api, post["id"], "submit")
    assert submitted.status_code == 200
    assert _put_slot(api, post["id"], _future()).json()["status"] == "needs_approval"
    rejected = _post(api, post["id"], "reject", {"reason": "Off brand"})
    assert rejected.status_code == 200 and rejected.json()["status"] == "archived"
    assert _put_slot(api, post["id"], _future()).status_code == 409


# ---------------------------------------------------------------------------
# Approve into the slot
# ---------------------------------------------------------------------------


def test_approving_with_a_future_slot_schedules_the_post_into_it(api, leader):
    post = _create(api)
    slot = _future(5)
    _put_slot(api, post["id"], slot, "UTC")
    submitted = _post(api, post["id"], "submit").json()
    approved = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"]})
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["status"] == "scheduled" and body["scheduled_for"] == body["planned_for"]
    assert body["approved_hash"] == body["content_hash"]
    assert [entry["action"] for entry in body["review_log"][-2:]] == ["approve", "schedule"]
    assert leader.jobs[f"social-publish-{post['id']}"].trigger.run_date == slot

    late = _create(api, title="Late")
    _put_slot(api, late["id"], _past(3600))
    submitted = _post(api, late["id"], "submit").json()
    body = _post(api, late["id"], "approve", {"content_hash": submitted["content_hash"]}).json()
    assert body["status"] == "approved" and body["scheduled_for"] is None
    assert f"social-publish-{late['id']}" not in leader.jobs


def test_a_slot_change_on_an_approved_or_scheduled_post_goes_through_schedule(api, leader):
    post = _create(api)
    first = _future(5)
    _put_slot(api, post["id"], first, "UTC")
    submitted = _post(api, post["id"], "submit").json()
    scheduled = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"]}).json()
    assert scheduled["status"] == "scheduled"

    later = _future(9)
    moved = _put_slot(api, post["id"], later, "Europe/Lisbon")
    assert moved.status_code == 200, moved.text
    body = moved.json()
    assert body["status"] == "scheduled" and body["timezone"] == "Europe/Lisbon"
    assert body["scheduled_for"] == body["planned_for"] and body["scheduled_for"] != scheduled["scheduled_for"]
    assert body["approved_hash"] == scheduled["approved_hash"] == body["content_hash"]
    assert leader.jobs[f"social-publish-{post['id']}"].trigger.run_date == later
    assert _put_slot(api, post["id"], None).status_code == 422  # an approved post keeps a slot

    plain = _approved(api, title="No slot yet")
    assert plain["status"] == "approved"
    assert _put_slot(api, plain["id"], _future(2)).json()["status"] == "scheduled"


# ---------------------------------------------------------------------------
# The pass: a passed slot ends missed, once
# ---------------------------------------------------------------------------


@pytest.fixture
def told(monkeypatch):
    calls = []
    monkeypatch.setattr(schedule_jobs.notify, "notify_publish_outcome", lambda ws, post_id, title, status: calls.append((ws, str(post_id), title, status)))
    return calls


def _passed_post(api, title="Passed", submit=True):
    post = _create(api, title=title)
    _put_slot(api, post["id"], _past(config.SOCIALS_MISFIRE_GRACE_SECONDS + 120))
    if submit:
        assert _post(api, post["id"], "submit").status_code == 200
    return post


def test_the_pass_ends_an_unapproved_passed_slot_missed_once_and_leaves_the_rest(api, told):
    passed = _passed_post(api)
    within = _create(api, title="Within the grace")
    _put_slot(api, within["id"], _past(10))
    approved_late = _create(api, title="Approved late")
    _put_slot(api, approved_late["id"], _past(config.SOCIALS_MISFIRE_GRACE_SECONDS + 120))
    submitted = _post(api, approved_late["id"], "submit").json()
    assert _post(api, approved_late["id"], "approve", {"content_hash": submitted["content_hash"]}).json()["status"] == "approved"

    result = schedule_jobs.reconcile(FakeScheduler(), api.session)
    assert result["passed"] == 1
    row = _row(api, passed["id"])
    assert row.status == "missed" and row.review_log[-1]["action"] == "slot_passed"
    assert "passed with no approval" in row.review_log[-1]["comment"]
    assert told == [(WS_A, passed["id"], "Passed", "missed")]
    assert _row(api, within["id"]).status == "draft" and _row(api, approved_late["id"]).status == "approved"

    assert schedule_jobs.reconcile(FakeScheduler(), api.session)["passed"] == 0  # idempotent
    assert len(told) == 1


def test_a_missed_post_restarts_as_a_draft_at_a_new_slot_or_is_rescheduled_when_its_approval_stands(api, leader, told):
    passed = _passed_post(api)
    schedule_jobs.reconcile(FakeScheduler(), api.session)
    assert _row(api, passed["id"]).status == "missed"
    slot = _future(4)
    restarted = _put_slot(api, passed["id"], slot, "UTC")
    assert restarted.status_code == 200, restarted.text
    assert restarted.json()["status"] == "draft" and restarted.json()["planned_for"].startswith(slot.isoformat()[:16])
    assert restarted.json()["review_log"][-1]["action"] == "reslot"
    assert _post(api, passed["id"], "submit").json()["status"] == "needs_approval"
    assert _put_slot(api, passed["id"], None).status_code == 200  # a draft again: the slot may be cleared

    # A missed post whose approval stands (Wave 3's miss) is rescheduled instead.
    approved = _approved(api, title="Was scheduled")
    assert _put_slot(api, approved["id"], _future(2)).json()["status"] == "scheduled"
    row = _row(api, approved["id"])
    publish_lifecycle.miss(row, "scheduler", "late")
    api.session.commit()
    assert _row(api, approved["id"]).status == "missed"
    again = _put_slot(api, approved["id"], _future(6), "UTC")
    assert again.status_code == 200 and again.json()["status"] == "scheduled"
    assert again.json()["approved_hash"] == again.json()["content_hash"]


# ---------------------------------------------------------------------------
# @integration: two passes racing on the CI Postgres leave one transition
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_two_passes_racing_on_one_post_leave_one_transition_on_postgres(monkeypatch):
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT planned_for FROM social_posts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI"):
            raise
        pytest.skip(f"the Postgres check needs the test database: {exc}")
    factory = sessionmaker(bind=engine)
    told, lock = [], threading.Lock()

    def record(ws, post_id, title, status):
        with lock:
            told.append((str(post_id), status))

    monkeypatch.setattr(schedule_jobs.notify, "notify_publish_outcome", record)
    workspace_id = uuid.uuid4()
    with factory() as db:
        db.add(Workspace(id=workspace_id, name="b105-race", plan="basic", plan_limits={}, settings={"socials": {"enabled": True}}))
        db.commit()
        post = service.create_draft(db, workspace_id=workspace_id, created_by="a", title="Race", copy={"base": "Race."})
        db.flush()
        service.submit(post, "a")
        service.set_planned_for(post, datetime.now(timezone.utc) - timedelta(seconds=config.SOCIALS_MISFIRE_GRACE_SECONDS + 300), "UTC")
        db.commit()
        post_id = post.id
    try:
        barrier, results = threading.Barrier(2), []

        def one_pass():
            with factory() as db:
                barrier.wait()
                results.append(schedule_jobs.pass_planned_slots(db, datetime.now(timezone.utc)))

        threads = [threading.Thread(target=one_pass) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with factory() as db:
            row = service.get_post(db, workspace_id, post_id)
            assert row.status == "missed"
            assert [e["action"] for e in row.review_log].count("slot_passed") == 1
        assert sum(results) == 1 and told == [(str(post_id), "missed")]
    finally:
        with factory() as db:
            db.execute(sa.text("DELETE FROM social_posts WHERE workspace_id = :ws"), {"ws": workspace_id})
            db.execute(sa.text("DELETE FROM workspaces WHERE id = :ws"), {"ws": workspace_id})
            db.commit()
        engine.dispose()

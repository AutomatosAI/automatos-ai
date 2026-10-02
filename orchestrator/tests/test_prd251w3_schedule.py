"""PRD-251 Wave 3, US-306 (S3.1a, D10) — scheduling: one-shot jobs, reconcile, missed slots.

With a fake scheduler (it records jobs the way APScheduler holds them) and the
executor mocked (the US-301 harness):

* scheduling a post registers ``social-publish-<post_id>``: the module-level
  ``fire_scheduled_post`` with plain string args (the RedisJobStore pickles it), at the
  slot, firing however late it runs; unscheduling, an edit that voids the approval
  and publish now remove it; a reschedule (``POST /schedule`` on a scheduled post)
  moves it and keeps the approval;
* the leader's reconcile pass registers a missing job, moves one whose slot changed
  and removes an orphan;
* a fire within ``SOCIALS_MISFIRE_GRACE_SECONDS`` publishes; later than that, with
  Socials off, or with an approval that no longer matches, the post ends ``missed``,
  the workspace is told, and nothing is published; a missed post can be rescheduled
  or published now; a job fired twice publishes once (on the CI Postgres too).
"""
from __future__ import annotations

import asyncio
import os
import pickle
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from apscheduler.jobstores.base import JobLookupError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251w3_publisher as harness  # noqa: E402
from config import config  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import notify, schedule_jobs, service  # noqa: E402
from modules.socials import settings as socials_settings  # noqa: E402
from modules.socials.publishing import run_publish  # noqa: E402
from tests.test_prd251w3_publisher import LINKEDIN, WS, FakeExecutor, _approved_post, _post, _runtime, _target  # noqa: E402

api = api_harness.api
env = harness.env
NOW = datetime(2026, 11, 9, 9, 0, tzinfo=timezone.utc)


class FakeScheduler:
    """Holds jobs as APScheduler does: by id, each with its function, args and trigger."""

    running = True

    def __init__(self):
        self.jobs = {}

    def add_job(self, func, trigger, *, args, id, replace_existing, max_instances, misfire_grace_time):
        assert replace_existing and max_instances == 1 and misfire_grace_time is None
        self.jobs[id] = SimpleNamespace(id=id, func=func, args=list(args), trigger=trigger)

    def remove_job(self, job_id):
        if job_id not in self.jobs:
            raise JobLookupError(job_id)
        del self.jobs[job_id]

    def get_jobs(self):
        return list(self.jobs.values())


@pytest.fixture
def leader(monkeypatch):
    scheduler = FakeScheduler()
    monkeypatch.setattr(schedule_jobs, "_leader", lambda: scheduler)
    return scheduler


def _slot(days=30):
    return datetime.now(timezone.utc) + timedelta(days=days)


# ---------------------------------------------------------------------------
# The job follows the post: schedule, reschedule, unschedule, a voiding edit, publish now
# ---------------------------------------------------------------------------


def _schedule(api, post_id, slot, tz="Europe/Lisbon"):
    return api_harness._post(api, post_id, "schedule", {"scheduled_for": slot.isoformat(), "timezone": tz})


def test_scheduling_registers_a_picklable_one_shot_job_at_the_slot(api, leader):
    approved = api_harness._approved(api)
    slot = _slot()

    assert _schedule(api, approved["id"], slot).status_code == 200

    job = leader.jobs[f"social-publish-{approved['id']}"]
    assert job.func is schedule_jobs.fire_scheduled_post and job.args == [approved["id"], str(api_harness.WS_A)]
    assert job.trigger.run_date == slot
    assert pickle.loads(pickle.dumps((job.func, job.args))) == (job.func, job.args)  # the RedisJobStore's way


def test_a_reschedule_moves_the_job_and_keeps_the_approval(api, leader):
    approved = api_harness._approved(api)
    _schedule(api, approved["id"], _slot(30))
    later = _slot(40)

    moved = _schedule(api, approved["id"], later, tz="America/New_York")

    assert moved.status_code == 200
    body = moved.json()
    assert body["status"] == "scheduled" and body["timezone"] == "America/New_York"
    assert body["approved_hash"] == approved["approved_hash"] == body["content_hash"]
    assert body["review_log"][-1]["action"] == "schedule"
    assert leader.jobs[f"social-publish-{approved['id']}"].trigger.run_date == later


def test_unschedule_and_a_voiding_edit_remove_the_job(api, leader):
    one, two = api_harness._approved(api), api_harness._approved(api)
    _schedule(api, one["id"], _slot())
    _schedule(api, two["id"], _slot())

    assert api_harness._post(api, one["id"], "unschedule").status_code == 200
    edited = api.client.patch(f"/api/socials/posts/{two['id']}", json={"copy": {"base": "Changed after approval"}})

    assert edited.status_code == 200 and edited.json()["status"] == "needs_approval"
    assert leader.jobs == {}
    # The edited post is rejected from review: still no job.
    assert api_harness._post(api, two["id"], "reject", {"reason": "Off brand"}).json()["status"] == "archived"
    assert leader.jobs == {}


def test_publish_now_on_a_scheduled_post_removes_its_job(api, leader):
    approved = api_harness._approved(api)
    row = api.session.get(harness.SocialPost, uuid.UUID(approved["id"]))
    service.update_post(row, "user-author", {"targets": [api_harness._LINKEDIN_TEXT]})
    api.session.commit()
    approved = api_harness._post(api, approved["id"], "approve", {"content_hash": row.content_hash}).json()
    _schedule(api, approved["id"], _slot())
    assert leader.jobs

    assert api_harness._post(api, approved["id"], "publish-now").status_code == 202
    assert leader.jobs == {} and len(api.launched) == 1


def test_without_the_scheduler_on_this_worker_a_write_registers_nothing(api, monkeypatch):
    monkeypatch.setattr(schedule_jobs, "_leader", lambda: None)
    approved = api_harness._approved(api)
    assert _schedule(api, approved["id"], _slot()).status_code == 200  # the reconcile pass registers it


# ---------------------------------------------------------------------------
# The reconcile pass
# ---------------------------------------------------------------------------


def _scheduled_post(env, slot, *targets):
    post_id = _approved_post(env, *(targets or (_target("linkedin", "text"),)))
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        post.status, post.scheduled_for, post.timezone = service.SCHEDULED, slot, "Europe/Lisbon"
        db.commit()
    return post_id


def test_reconcile_registers_a_missing_job_moves_a_changed_one_and_removes_an_orphan(env):
    scheduler = FakeScheduler()
    missing = _scheduled_post(env, _slot(3))
    moved = _scheduled_post(env, _slot(4))
    schedule_jobs.register(scheduler, moved, WS, _slot(9))  # the job still has the old slot
    orphan = uuid.uuid4()
    schedule_jobs.register(scheduler, orphan, WS, _slot(5))  # its post is no longer scheduled
    unrelated = SimpleNamespace(id="scheduled_task_7", trigger=None)
    scheduler.jobs[unrelated.id] = unrelated

    with env.factory() as db:
        result = schedule_jobs.reconcile(scheduler, db)
        slots = {row.id: row.scheduled_for for row in db.query(harness.SocialPost)}

    assert result == {"added": 1, "moved": 1, "removed": 1, "ended": 0, "passed": 0}
    assert set(scheduler.jobs) == {f"social-publish-{missing}", f"social-publish-{moved}", "scheduled_task_7"}
    assert scheduler.jobs[f"social-publish-{moved}"].trigger.run_date == schedule_jobs._utc(slots[moved])
    with env.factory() as db:
        assert schedule_jobs.reconcile(scheduler, db) == {"added": 0, "moved": 0, "removed": 0, "ended": 0, "passed": 0}  # idempotent


def test_reconcile_does_nothing_off_the_leader(env):
    with env.factory() as db:
        assert schedule_jobs.reconcile(None, db)["skipped"] is True


# ---------------------------------------------------------------------------
# The fire: publish within the grace, missed after it
# ---------------------------------------------------------------------------


@pytest.fixture
def fire(env, monkeypatch):
    """The fire, with the workspace's Socials switch on and the publish on the fake executor."""
    engine = env.factory.kw["bind"]
    copies = sa.MetaData()
    api_harness._sqlite_copy(Workspace.__table__, copies)
    copies.create_all(engine)
    with env.factory() as db:
        db.add(Workspace(id=WS, name="Harvest", plan="basic", plan_limits={}, settings={"socials": {"enabled": True}},
                         onboarding={}, created_at=NOW, updated_at=NOW))
        db.commit()
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    executor = FakeExecutor(LINKEDIN)

    async def publish(job, *, session_factory):
        return await run_publish(job, executor=executor, session_factory=session_factory, runtime=_runtime(env))

    monkeypatch.setattr(schedule_jobs, "run_publish", publish)

    def run(post_id, now):
        return asyncio.run(schedule_jobs.fire_scheduled_post(str(post_id), str(WS), session_factory=env.factory, clock=lambda: now))

    return SimpleNamespace(run=run, executor=executor)


def test_a_fire_within_the_grace_publishes(env, fire, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MISFIRE_GRACE_SECONDS", 1800)
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)

    assert fire.run(post_id, slot + timedelta(minutes=29)) == "published"

    assert fire.executor.actions == ["LINKEDIN_GET_MY_INFO", "LINKEDIN_CREATE_LINKED_IN_POST"]
    post = _post(env, post_id)
    assert post.status == "published"
    assert [e["action"] for e in post.review_log[-2:]] == ["publish", "published"]
    assert post.review_log[-2]["by"] == "scheduler"


def test_a_slot_missed_beyond_the_grace_ends_missed_notifies_and_publishes_nothing(env, fire, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MISFIRE_GRACE_SECONDS", 1800)
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)

    assert fire.run(post_id, slot + timedelta(minutes=45)) == "missed"

    assert fire.executor.calls == []
    post = _post(env, post_id)
    assert post.status == "missed" and post.review_log[-1]["action"] == "missed"
    assert "45 minutes" in post.review_log[-1]["comment"] and "nothing was published" in post.review_log[-1]["comment"]
    assert post.approved_hash == post.content_hash  # the approval stands
    assert [(n["event_type"], n["link_id"]) for n in env.notices] == [("social_post_missed", str(post_id))]


def test_a_slot_with_socials_off_is_missed(env, fire, monkeypatch):
    with env.factory() as db:
        db.get(Workspace, WS).settings = {"socials": {"enabled": False}}
        db.commit()
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)
    assert fire.run(post_id, slot) == "missed" and fire.executor.calls == []
    assert "Socials was off" in _post(env, post_id).review_log[-1]["comment"]


def test_a_slot_whose_approval_no_longer_matches_is_missed_not_published(env, fire):
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)
    with env.factory() as db:
        service.get_post(db, WS, post_id).approved_hash = "f" * 64
        db.commit()
    assert fire.run(post_id, slot) == "missed" and fire.executor.calls == []
    assert "approval no longer matched" in _post(env, post_id).review_log[-1]["comment"]


def test_an_early_fire_or_an_unscheduled_post_does_nothing(env, fire):
    slot = _slot(2)
    post_id = _scheduled_post(env, slot)
    assert fire.run(post_id, slot - timedelta(hours=1)) is None  # the slot moved later
    with env.factory() as db:
        service.unschedule(service.get_post(db, WS, post_id), "user-author")
        db.commit()
    assert fire.run(post_id, slot) is None
    assert fire.executor.calls == [] and _post(env, post_id).status == "approved"


def test_a_job_fired_twice_publishes_once(env, fire):
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)
    assert fire.run(post_id, slot) == "published"
    assert fire.run(post_id, slot) is None  # the post is no longer scheduled
    assert fire.executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1


def test_a_reschedule_that_lands_while_the_job_fires_wins(env, fire, monkeypatch):
    """The fire read the post at its old slot; a person moved it before the fire
    claimed it. Nothing publishes at the old slot: the post stays scheduled at the new one."""
    slot = _slot(1)
    later = slot + timedelta(days=2)
    post_id = _scheduled_post(env, slot)

    read = service.get_post

    def read_then_reschedule(db, workspace_id, pid):
        post = read(db, workspace_id, pid)  # the fire has read the old slot
        monkeypatch.setattr(service, "get_post", read)
        with env.factory() as other:
            service.schedule(read(other, WS, post_id), "user-author", later, "Europe/Lisbon")
            other.commit()
        return post

    monkeypatch.setattr(service, "get_post", read_then_reschedule)
    assert fire.run(post_id, slot) is None

    assert fire.executor.calls == []
    post = _post(env, post_id)
    assert post.status == "scheduled" and post.approved_hash == post.content_hash
    assert datetime.fromisoformat(post.scheduled_for).replace(tzinfo=timezone.utc) == later

def test_a_missed_post_can_be_rescheduled_or_published_now(env, fire, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MISFIRE_GRACE_SECONDS", 60)
    slot = _slot(1)
    rescheduled, published = _scheduled_post(env, slot), _scheduled_post(env, slot)
    assert fire.run(rescheduled, slot + timedelta(hours=2)) == "missed"
    assert fire.run(published, slot + timedelta(hours=2)) == "missed"

    with env.factory() as db:
        post = service.get_post(db, WS, rescheduled)
        service.schedule(post, "user-author", _slot(3), "Europe/Lisbon")
        db.commit()
    assert _post(env, rescheduled).status == "scheduled"
    assert harness._publish(env, published, FakeExecutor(LINKEDIN)) == "published"


def test_a_missed_post_with_a_changed_content_is_not_published(env, fire, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MISFIRE_GRACE_SECONDS", 60)
    slot = _slot(1)
    post_id = _scheduled_post(env, slot)
    assert fire.run(post_id, slot + timedelta(hours=2)) == "missed"
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        service.update_post(post, "user-author", {"copy": {"base": "Edited while missed"}})
        db.commit()
        assert post.status == "needs_approval"
    with pytest.raises(service.NotPublishable):
        harness._claim(env, post_id)


def test_the_notification_event_is_valid():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert notify.MISSED_EVENT == "social_post_missed" and "social_post_missed" in VALID_EVENT_TYPES


# ---------------------------------------------------------------------------
# @integration: a double fire on the CI Postgres publishes once
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_double_fire_on_postgres_publishes_once(monkeypatch):
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM social_posts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI"):
            raise
        pytest.skip(f"the Postgres check needs the test database: {exc}")
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(harness.publish_records.media_urls, "resolve_post_media", lambda db, post: [])
    monkeypatch.setattr(notify, "_dispatcher", lambda db, ws: SimpleNamespace(dispatch=lambda **k: asyncio.sleep(0)))
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    executor = FakeExecutor(LINKEDIN)

    async def publish(job, *, session_factory):
        return await run_publish(job, executor=executor, session_factory=session_factory)

    monkeypatch.setattr(schedule_jobs, "run_publish", publish)
    workspace_id = uuid.uuid4()
    with factory() as db:
        db.add(Workspace(id=workspace_id, name="w3-double-fire", plan="basic", plan_limits={}, settings={"socials": {"enabled": True}}))
        db.commit()
        post = service.create_draft(db, workspace_id=workspace_id, created_by="a", title="Twice", copy={"base": "Twice."})
        db.flush()
        service.update_post(post, "a", {"targets": [_target("linkedin", "text")]})
        service.submit(post, "a")
        service.approve(post, "r", content_hash=post.content_hash)
        service.schedule(post, "r", _slot(1), "UTC")
        db.commit()
        post.scheduled_for = datetime.now(timezone.utc) - timedelta(seconds=30)  # due now
        db.commit()
        post_id = post.id
    try:
        results = []

        def fire():
            results.append(asyncio.run(schedule_jobs.fire_scheduled_post(str(post_id), str(workspace_id), session_factory=factory)))

        threads = [threading.Thread(target=fire) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert sorted(results, key=str) == [None, "published"]
        assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1
    finally:
        with factory() as db:
            db.execute(sa.text("DELETE FROM social_posts WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.commit()
        engine.dispose()

"""PRD-261 C-05h — the scheduler of the local edition's revocation-feed pull.

``ThemeRevocationScheduler.start`` runs against a fake scheduler that records ``add_job``. Pins
(plan §2.6 C-05h): the pull is registered every ``THEME_REVOCATIONS_PULL_MINUTES`` with
``max_instances=1`` and its first run 60 s after start; never in the hosted edition or with an
empty ``THEME_REVOCATIONS_URL``; the job is a module-level function a RedisJobStore can hold by
reference; it pulls in its own session and never raises.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from apscheduler.triggers.interval import IntervalTrigger  # noqa: E402
from apscheduler.util import obj_to_ref, ref_to_obj  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import services.theme_revocation_scheduler as scheduler_mod  # noqa: E402
from config import config  # noqa: E402
from services.themes import revocation_pull  # noqa: E402
from services.themes.revocation_pull import PullOutcome  # noqa: E402

FEED_URL = "https://feed.example.test/api/themes/revocations"


class _FakeScheduler:
    """Records every ``add_job`` call."""

    def __init__(self):
        self.jobs = []

    def add_job(self, func, trigger, **kwargs):
        self.jobs.append(SimpleNamespace(func=func, trigger=trigger, **kwargs))


def _start():
    fake = _FakeScheduler()
    started = scheduler_mod.ThemeRevocationScheduler().start(fake)
    return started, fake.jobs


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, FEED_URL, raising=False)


def test_the_pull_runs_every_pull_minutes_one_at_a_time_first_a_minute_after_start(local, monkeypatch):
    monkeypatch.setattr(config, scheduler_mod.PULL_MINUTES_SETTING, 15, raising=False)
    before = datetime.now(timezone.utc)
    started, jobs = _start()
    after = datetime.now(timezone.utc)
    assert started is True
    [job] = jobs
    assert job.func is scheduler_mod.run_scheduled_pull
    assert isinstance(job.trigger, IntervalTrigger) and job.trigger.interval == timedelta(minutes=15)
    assert (job.id, job.max_instances, job.replace_existing) == (scheduler_mod.JOB_ID, 1, True)
    delay = timedelta(seconds=scheduler_mod.FIRST_RUN_DELAY_SECONDS)
    assert scheduler_mod.FIRST_RUN_DELAY_SECONDS == 60
    assert before + delay <= job.next_run_time <= after + delay


def test_the_interval_is_an_hour_unless_set(monkeypatch):
    assert scheduler_mod.DEFAULT_PULL_MINUTES == 60
    monkeypatch.setattr(config, scheduler_mod.PULL_MINUTES_SETTING, 0, raising=False)
    assert scheduler_mod.pull_minutes() == 60
    monkeypatch.setattr(config, scheduler_mod.PULL_MINUTES_SETTING, 5, raising=False)
    assert scheduler_mod.pull_minutes() == 5


@pytest.mark.parametrize("edition,url", [("saas", FEED_URL), ("local", ""), ("local", "   ")])
def test_never_in_the_hosted_edition_or_with_no_feed_url(monkeypatch, edition, url):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, url, raising=False)
    assert _start() == (False, [])


def test_the_job_is_a_module_function_a_job_store_can_hold():
    job = scheduler_mod.run_scheduled_pull
    assert (job.__module__, job.__qualname__) == ("services.theme_revocation_scheduler", "run_scheduled_pull")
    assert ref_to_obj(obj_to_ref(job)) is job


def test_the_job_pulls_in_its_own_session_and_never_raises(monkeypatch):
    sessions = []

    def pulled(db):
        sessions.append(db)
        return PullOutcome(ok=True, rows=2)

    monkeypatch.setattr(scheduler_mod, "pull_feed", pulled)
    assert asyncio.run(scheduler_mod.run_scheduled_pull()) is None
    assert len(sessions) == 1 and isinstance(sessions[0], Session)
    monkeypatch.setattr(scheduler_mod, "pull_feed", lambda db: PullOutcome(ok=False, error="The feed answered 500."))
    assert asyncio.run(scheduler_mod.run_scheduled_pull()) is None

    def broken():
        raise RuntimeError("no database")

    monkeypatch.setattr(scheduler_mod, "_pull_in_session", broken)
    assert asyncio.run(scheduler_mod.run_scheduled_pull()) is None

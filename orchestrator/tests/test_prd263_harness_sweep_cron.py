"""PRD-263 US-012 — the sweep runs on a weekday daytime cron with an explicit UTC timezone.

The sweep was fixed at Sunday 02:00 (``_CRON_EXPRESSION``, no timezone passed), inside
the night window where F408, F409 and F416 happened, and "Sunday 02:00 UTC" was
hard-coded in its log lines and in the ``next_scheduled_run`` Auto reads. It now fires
on ``config.HARNESS_SWEEP_CRON`` (default Monday 10:00), built in UTC by
``schedule_util.scheduler_cron_trigger`` (never ``CronTrigger.from_crontab``, which reads
a numeric weekday a day late: F240), and ``next_scheduled_run`` is that trigger's next
fire time in ISO form, so any worker can answer.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[1]
SUNDAY_NOON = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
MONDAY_TEN = datetime(2026, 10, 19, 10, 0, tzinfo=timezone.utc)
FIRES = 10
DAYTIME_HOURS = range(8, 18)
WEEKEND = (5, 6)  # datetime.weekday(): Saturday, Sunday


def _shipped_default() -> str:
    src = (ORCH / "config.py").read_text()
    (default,) = re.findall(r'HARNESS_SWEEP_CRON: str = os\.getenv\("HARNESS_SWEEP_CRON", "([^"]+)"\)', src)
    return default


@pytest.fixture
def default_cron(monkeypatch):
    """The shipped default, whatever this environment sets."""
    from config import config

    monkeypatch.setattr(config, "HARNESS_SWEEP_CRON", _shipped_default())


def _fires(trigger, count, start=SUNDAY_NOON):
    fired, now = [], start
    for _ in range(count):
        at = trigger.get_next_fire_time(fired[-1] if fired else None, now)
        fired.append(at)
        now = at + timedelta(seconds=1)
    return fired


def test_the_default_fires_monday_ten_utc_on_weekdays_in_daytime(default_cron):
    from services.harness_runs import sweep_trigger

    trigger = sweep_trigger()
    assert trigger.get_next_fire_time(None, SUNDAY_NOON) == MONDAY_TEN
    fired = _fires(trigger, FIRES)
    assert all(at.weekday() not in WEEKEND for at in fired), fired
    assert all(at.astimezone(timezone.utc).hour in DAYTIME_HOURS for at in fired), fired
    assert trigger.timezone.utcoffset(SUNDAY_NOON) == timedelta(0)
    assert trigger.timezone.tzname(SUNDAY_NOON) == "UTC"


def test_the_sweep_job_is_registered_on_that_trigger(default_cron):
    from services.harness_service import HarnessService

    class _Scheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, func, **kwargs):
            self.jobs.append((func, kwargs))

    svc = HarnessService()
    svc._scheduler = _Scheduler()
    asyncio.run(svc._register_workspace_jobs())

    ((func, kwargs),) = svc._scheduler.jobs
    assert func == svc._harness_sweep and kwargs["id"] == "harness_sweep"
    assert not {"day_of_week", "hour", "minute"} & set(kwargs)  # no fixed Sunday 02:00 fields
    assert kwargs["trigger"].get_next_fire_time(None, SUNDAY_NOON) == MONDAY_TEN


def test_a_configured_cron_is_honoured_with_its_weekday_by_name(monkeypatch):
    from config import config
    from services.harness_runs import next_sweep_run, sweep_trigger

    monkeypatch.setattr(config, "HARNESS_SWEEP_CRON", "30 9 * * 2-4")  # cron's 2 is Tuesday (F240)
    assert sweep_trigger().get_next_fire_time(None, SUNDAY_NOON) == datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc)
    assert next_sweep_run(SUNDAY_NOON) == "2026-10-20T09:30:00+00:00"


def test_next_scheduled_run_is_the_triggers_next_fire_in_iso_utc(default_cron, monkeypatch):
    from config import config
    from services.harness_runs import next_sweep_run

    assert next_sweep_run(SUNDAY_NOON) == "2026-10-19T10:00:00+00:00"
    monkeypatch.setattr(config, "HARNESS_SWEEP_CRON", "not a cron")
    assert next_sweep_run(SUNDAY_NOON) is None


def test_the_fixed_sunday_schedule_is_gone_and_the_setting_is_listed():
    src = (ORCH / "services" / "harness_service.py").read_text()
    for gone in ("Sunday", "_CRON_EXPRESSION", "from_crontab"):
        assert gone not in src, gone
    settings = json.loads((ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert "HARNESS_SWEEP_CRON" in settings and settings == sorted(settings)

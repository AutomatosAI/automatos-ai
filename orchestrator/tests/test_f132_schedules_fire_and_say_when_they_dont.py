"""F132 (night 4, the persona's fix-first #4) — playbook schedules fire, in the
zone they were saved in, and a run that does not start says so.

Night 4: 2 of 5 one-off schedules fired (B17, B44 ×2, B60, B63, B90) and the
rest left no run, no bell and no log. Auto's platform_schedule_playbook built a
fresh, never-started PlaybookSchedulerService, called it with the wrong
arguments, and swallowed the TypeError, so the schedule reached no scheduler;
it fired only when a restart happened to re-read it in time (the "~35 minutes"
was the restart, not a horizon). update_playbook synced nothing. The trigger was
built without a zone, so '40 7 23 9 *' saved by a UK owner fired at 07:40 UTC
(B35). APScheduler's 1 s misfire grace dropped a late fire silently, and a fire
skipped at the run limit left only a log line.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

pytest.importorskip("apscheduler")
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: E402
from apscheduler.schedulers.background import BackgroundScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger as RealCronTrigger  # noqa: E402

import services.playbook_scheduler as sched_mod  # noqa: E402
from config import config  # noqa: E402
from core.models.core import WorkflowTemplate  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.tools.discovery import handlers_playbooks  # noqa: E402

WS = UUID("00000000-0000-0000-0000-0000000000c1")
UK = {"orchestrator": {"heartbeat": {"timezone": "Europe/London"}}}


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def get(self, _id):
        return self.first()

    def count(self):
        return 0                      # no run in flight (F134's note reads this)


class _Db:
    def __init__(self, playbooks=(), settings=None):
        self.rows = {WorkflowTemplate: list(playbooks),
                     Workspace: [SimpleNamespace(id=WS, settings=settings or {})]}
        self.deleted = []

    def query(self, model):
        return _Query(self.rows.get(model, []))

    def add(self, obj):
        pass

    def delete(self, obj):
        self.deleted.append(obj)

    def flush(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _playbook(schedule_config=None, pid=86):
    return SimpleNamespace(id=pid, name="Monthly numbers", workspace_id=WS, schedule_config=schedule_config,
                           steps=[{"order": 1, "agent_id": 7, "prompt_template": "Numbers."}])


def _cron(expression="20 18 23 9 *", zone="Europe/London", enabled=True):
    return {"type": "cron", "cron_expression": expression, "timezone": zone, "enabled": enabled}


@pytest.fixture(autouse=True)
def _real_trigger(monkeypatch):
    # test_playbook_scheduler swaps a fake CronTrigger into the module for the whole process.
    monkeypatch.setattr(sched_mod, "CronTrigger", RealCronTrigger)


def _paused_scheduler():
    """A started scheduler that fires nothing: add, replace and remove act as they do live."""
    scheduler = BackgroundScheduler()
    scheduler.start(paused=True)
    return scheduler


@pytest.fixture
def leader(monkeypatch):
    """The worker that holds the scheduler lock: the singleton hosts a scheduler."""
    service = sched_mod.PlaybookSchedulerService()
    service._scheduler = _paused_scheduler()
    monkeypatch.setattr(sched_mod, "_playbook_scheduler", service)
    yield service
    service._scheduler.shutdown(wait=False)


def _schedule(db, **params):
    return asyncio.run(handlers_playbooks.schedule_playbook(db, WS, {"playbook_id": 86, **params}))


# ── Auto's schedule reaches the scheduler, in its zone ──────────────────────

def test_autos_schedule_reaches_the_running_scheduler_in_the_workspaces_zone(leader):
    playbook = _playbook()
    reply = _schedule(_Db([playbook], settings=UK), cron_expression="20 18 23 9 *")
    assert reply["success"] is True
    assert playbook.schedule_config == _cron()
    job = leader._scheduler.get_job("playbook_cron_86")
    assert job is not None and str(job.trigger.timezone) == "Europe/London"
    assert "20 18 23 9 * in Europe/London. Active now." in reply["message"]


def test_a_workspace_without_a_zone_saves_utc_by_name(leader):
    playbook = _playbook()
    _schedule(_Db([playbook]), cron_expression="0 9 * * 1")
    assert playbook.schedule_config["timezone"] == "UTC"


@pytest.mark.parametrize("params, said", [
    ({"cron_expression": "61 18 23 9 *"}, "scheduling failed: invalid cron '61 18 23 9 *'"),
    ({"cron_expression": "20 18 23 9 *", "timezone": "Europe/Atlantis"},
     "scheduling failed: unknown timezone 'Europe/Atlantis'"),
])
def test_a_cron_or_zone_the_scheduler_cannot_use_is_refused_before_anything_is_saved(leader, params, said):
    playbook = _playbook()
    reply = _schedule(_Db([playbook]), **params)
    assert reply["success"] is False and reply["error"].startswith(said)
    assert playbook.schedule_config is None
    assert leader._scheduler.get_jobs() == []


def test_a_scheduler_that_refuses_is_reported_never_scheduled(monkeypatch, leader):
    def _refuse(playbook):
        raise RuntimeError("job store unavailable")

    monkeypatch.setattr(sched_mod, "sync_playbook_schedule", _refuse, raising=False)
    reply = _schedule(_Db([_playbook()]), cron_expression="20 18 23 9 *")
    assert reply["success"] is False
    assert reply["error"] == "scheduling failed: job store unavailable"


def test_an_update_that_changes_the_schedule_syncs_it_and_names_the_zone(leader):
    playbook = _playbook()
    reply = asyncio.run(handlers_playbooks.update_playbook(
        _Db([playbook], settings=UK), WS,
        {"playbook_id": 86, "schedule_config": {"type": "cron", "cron_expression": "40 7 * * 1"}}))
    assert reply["success"] is True
    assert playbook.schedule_config == {"type": "cron", "cron_expression": "40 7 * * 1", "timezone": "Europe/London"}
    assert str(leader._scheduler.get_job("playbook_cron_86").trigger.timezone) == "Europe/London"
    assert "Schedule: 40 7 * * 1 in Europe/London. Active now." in reply["message"]


# ── the trigger fires in the saved zone (B35) ───────────────────────────────

def test_the_trigger_fires_in_the_saved_zone_not_the_servers(leader):
    leader.schedule_playbook(_playbook(_cron("40 7 23 9 *"), pid=35))
    trigger = leader._scheduler.get_job("playbook_cron_35").trigger
    fire = trigger.get_next_fire_time(None, datetime(2026, 9, 22, 12, tzinfo=timezone.utc))
    assert fire.astimezone(timezone.utc) == datetime(2026, 9, 23, 6, 40, tzinfo=timezone.utc)  # 07:40 BST


# ── a worker without the scheduler defers to the leader's tick ──────────────

def test_a_schedule_saved_where_no_scheduler_runs_is_registered_by_the_leaders_tick(monkeypatch):
    monkeypatch.setattr(sched_mod, "_playbook_scheduler", sched_mod.PlaybookSchedulerService())
    playbook = _playbook()
    reply = _schedule(_Db([playbook], settings=UK), cron_expression="20 18 23 9 *")
    assert reply["success"] is True and "the scheduler picks it up within 60 s" in reply["message"]

    on_the_leader = sched_mod.PlaybookSchedulerService()
    on_the_leader._scheduler = _paused_scheduler()
    try:
        assert on_the_leader.reconcile_with_db(_Db([playbook])) == {"added": 1, "changed": 0, "removed": 0}
        assert str(on_the_leader._scheduler.get_job("playbook_cron_86").trigger.timezone) == "Europe/London"
    finally:
        on_the_leader._scheduler.shutdown(wait=False)


def test_the_tick_is_idempotent_and_follows_a_change(leader):
    playbook = _playbook(_cron(zone="UTC"))
    db = _Db([playbook])
    assert leader.reconcile_with_db(db) == {"added": 1, "changed": 0, "removed": 0}
    assert leader.reconcile_with_db(db) == {"added": 0, "changed": 0, "removed": 0}
    playbook.schedule_config = _cron(zone="Europe/London")
    assert leader.reconcile_with_db(db) == {"added": 0, "changed": 1, "removed": 0}
    playbook.schedule_config = _cron(enabled=False)
    assert leader.reconcile_with_db(db) == {"added": 0, "changed": 0, "removed": 1}


# ── a run that does not start says so ───────────────────────────────────────

def test_jobs_carry_the_configured_misfire_grace(leader):
    leader.schedule_playbook(_playbook(_cron()))
    job = leader._scheduler.get_job("playbook_cron_86")
    assert job.misfire_grace_time == config.PLAYBOOK_SCHEDULE_MISFIRE_GRACE_SECONDS == 300


def test_a_missed_fire_is_logged_and_rings_the_owner(monkeypatch, caplog, leader):
    rung = []

    async def _ring(playbook_id, reason):
        rung.append((playbook_id, reason))

    monkeypatch.setattr(leader, "_notify_skipped_by_id", _ring)
    missed = SimpleNamespace(job_id="playbook_cron_86",
                             scheduled_run_time=datetime(2026, 9, 23, 17, 20, tzinfo=timezone.utc))

    async def _scheduler_reports_it():
        leader._on_job_missed(missed)
        await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger="services.playbook_scheduler"):
        asyncio.run(_scheduler_reports_it())
    assert "scheduled run skipped: playbook 86, it was due at 17:20 UTC" in caplog.text
    assert rung and rung[0][0] == 86 and "late" in rung[0][1]


def test_the_listener_is_registered_when_the_service_starts(monkeypatch):
    service = sched_mod.PlaybookSchedulerService()
    monkeypatch.setattr(service, "_load_cron_playbooks", AsyncMock())
    scheduler = AsyncIOScheduler()
    asyncio.run(service.start(scheduler=scheduler))
    assert any(callback == service._on_job_missed for callback, _mask in scheduler._listeners)


def test_a_fire_skipped_at_the_run_limit_rings_the_owner(monkeypatch):
    import services.concurrency_guard as guard
    import services.playbook_breaker as breaker
    import core.database.database as dbmod

    playbook = _playbook(_cron())
    db = _Db([playbook])
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: db)
    monkeypatch.setattr(breaker, "breaker_is_open", lambda _db, _pid: False)
    monkeypatch.setattr(guard, "check_concurrency",
                        AsyncMock(return_value=SimpleNamespace(allowed=False, reason="3 of 3 runs going")))
    service = sched_mod.PlaybookSchedulerService()
    rung = []

    async def _ring(_db, pb, reason):
        rung.append((pb.id, reason))

    monkeypatch.setattr(service, "_notify_schedule_skipped", _ring, raising=False)
    asyncio.run(service._fire_playbook(86, str(WS)))
    assert rung == [(86, "the workspace was at its run limit (3 of 3 runs going)")]
    assert len(db.deleted) == 1                                     # the pending row is rolled back, as before


def test_boot_names_the_schedules_that_now_fire_in_their_saved_zone(monkeypatch, caplog, leader):
    import core.database.database as dbmod

    db = _Db([_playbook(_cron(zone="UTC"), pid=80), _playbook(_cron(zone="Europe/London"), pid=86)])
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: db)
    with caplog.at_level(logging.WARNING, logger="services.playbook_scheduler"):
        asyncio.run(leader._load_cron_playbooks())
    assert "1 schedules now fire in their saved zone, not the server's UTC: ids [86]" in caplog.text


# ── the UI route saves the zone by name and refuses what cannot fire ────────

def test_the_route_saves_the_zone_by_name_and_refuses_a_bad_cron():
    from fastapi import HTTPException

    from api.workflow_recipes import _explicit_schedule

    recipe = SimpleNamespace(schedule_config={"type": "cron", "cron_expression": "40 7 23 9 *"})
    _explicit_schedule(recipe, _Db(settings=UK), WS)
    assert recipe.schedule_config["timezone"] == "Europe/London"

    bad = SimpleNamespace(schedule_config={"type": "cron", "cron_expression": "40 25 * * *"})
    with pytest.raises(HTTPException) as refused:
        _explicit_schedule(bad, _Db(), WS)
    assert refused.value.status_code == 400 and "invalid cron '40 25 * * *'" in refused.value.detail

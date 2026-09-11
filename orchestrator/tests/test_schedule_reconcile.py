"""Leader reconcile tick — a schedule change made on any worker reaches the
scheduler worker within a minute, not at the next restart.

Production runs several uvicorn workers; exactly one holds the scheduler lock
and hosts APScheduler. A scheduled task created / paused / resumed / cancelled,
or a heartbeat toggled / re-configured, is written by whichever worker served
the request and only touched THAT worker's (absent) scheduler. The tick diffs
the DB against the registered jobs for both sources.

Fake-session tests (raw-DDL ``agent_scheduled_tasks``; no DB in CI).
"""
from __future__ import annotations

import asyncio
import os
import sys
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

from services import schedule_reconcile as sr  # noqa: E402
from services.heartbeat_service import HeartbeatService  # noqa: E402
from services.scheduled_task_service import JOB_ID_PREFIX, ScheduledTaskService  # noqa: E402


class _Job:
    def __init__(self, job_id):
        self.id = job_id


class _FakeScheduler:
    """The slice of APScheduler both reconcilers touch."""

    def __init__(self, jobs=(), running=True):
        self.running = running
        self._jobs = {job_id: _Job(job_id) for job_id in jobs}
        self.added: list = []
        self.removed: list = []

    def get_jobs(self):
        return list(self._jobs.values())

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def add_job(self, func, trigger=None, id=None, **kw):  # noqa: A002 — APScheduler's name
        self._jobs[id] = _Job(id)
        self.added.append(id)
        return self._jobs[id]

    def remove_job(self, job_id):
        del self._jobs[job_id]
        self.removed.append(job_id)

    def ids(self):
        return set(self._jobs)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeTaskDB:
    def __init__(self, active_rows):
        self.rows = active_rows
        self.execute_calls = 0
        self.closed = False

    def execute(self, stmt, params=None):
        self.execute_calls += 1
        return _Result(self.rows)

    def close(self):
        self.closed = True


class _FakeAgentDB:
    def __init__(self, agents):
        self._agents = agents

    def query(self, *cols):
        return SimpleNamespace(all=lambda: list(self._agents))


def _task_row(task_id, task_type="recurring", schedule="0 9 * * 1-5", agent=7):
    return SimpleNamespace(id=task_id, task_type=task_type, schedule=schedule, target_agent_id=agent)


def _agent(agent_id, hb, ws="ws-1"):
    cfg = {"heartbeat": hb} if hb is not None else {}
    return SimpleNamespace(id=agent_id, workspace_id=ws, configuration=cfg)


@pytest.fixture
def unified(monkeypatch):
    """_register_with_scheduler resolves the unified scheduler itself — point it
    at the same fake the reconcile diffs against."""
    fake = _FakeScheduler()
    import services.scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "get_unified_scheduler", lambda: SimpleNamespace(apscheduler=fake))
    return fake


# ── scheduled tasks ────────────────────────────────────────────────────────


class TestScheduledTaskReconcile:
    def test_adds_missing_removes_stale_leaves_present(self, unified):
        for job_id in (f"{JOB_ID_PREFIX}2", f"{JOB_ID_PREFIX}3", "agent_hb_9"):
            unified.add_job(None, id=job_id)
        unified.added.clear()
        db = _FakeTaskDB([_task_row(1), _task_row(2)])

        out = ScheduledTaskService(db, workspace_id=None).reconcile_with_scheduler(unified)

        assert out == {"added": 1, "removed": 1, "skipped": False}
        assert unified.added == [f"{JOB_ID_PREFIX}1"]  # task 2 present: NOT re-added
        assert unified.removed == [f"{JOB_ID_PREFIX}3"]  # cancelled elsewhere
        assert unified.ids() == {f"{JOB_ID_PREFIX}1", f"{JOB_ID_PREFIX}2", "agent_hb_9"}

    def test_second_pass_is_a_noop(self, unified):
        db = _FakeTaskDB([_task_row(1)])
        svc = ScheduledTaskService(db, workspace_id=None)
        svc.reconcile_with_scheduler(unified)
        assert svc.reconcile_with_scheduler(unified) == {"added": 0, "removed": 0, "skipped": False}

    def test_skips_off_the_leader_without_touching_the_db(self):
        db = _FakeTaskDB([_task_row(1)])
        svc = ScheduledTaskService(db, workspace_id=None)
        assert svc.reconcile_with_scheduler(_FakeScheduler(running=False))["skipped"] is True
        assert svc.reconcile_with_scheduler(None)["skipped"] is True
        assert db.execute_calls == 0

    def test_non_leader_register_points_at_the_tick_not_a_restart(self, monkeypatch, caplog):
        import services.scheduler as sched_mod

        monkeypatch.setattr(sched_mod, "get_unified_scheduler", lambda: SimpleNamespace(apscheduler=None))
        svc = ScheduledTaskService(_FakeTaskDB([]), workspace_id=None)
        with caplog.at_level("INFO"):
            svc._register_with_scheduler(5, "recurring", "0 9 * * *", 7)
        assert "reconcile tick" in caplog.text
        assert "restart" not in caplog.text


# ── heartbeats ─────────────────────────────────────────────────────────────


class TestHeartbeatReconcile:
    @staticmethod
    def _service(fake):
        svc = HeartbeatService()
        svc._scheduler = fake
        return svc

    def test_diff_adds_changes_removes(self):
        fake = _FakeScheduler(jobs=("agent_hb_2", "agent_hb_3", "agent_hb_4", f"{JOB_ID_PREFIX}1"))
        svc = self._service(fake)
        unchanged = {"enabled": True, "interval_minutes": 30}
        svc._hb_signatures[2] = svc.heartbeat_signature("ws-1", unchanged)
        svc._hb_signatures[3] = svc.heartbeat_signature("ws-1", {"enabled": True, "interval_minutes": 15})
        db = _FakeAgentDB([
            _agent(1, {"enabled": True, "interval_minutes": 60}),   # toggled on elsewhere: no job yet
            _agent(2, unchanged),                                    # present, same config
            _agent(3, {"enabled": True, "interval_minutes": 60}),   # re-configured 15m → 60m
            _agent(4, {"enabled": False, "interval_minutes": 60}),  # toggled off elsewhere
            _agent(5, {"interval_minutes": 60}),                     # no flag: never fires
            _agent(6, None),
        ])

        out = svc.reconcile_agent_heartbeats(db)

        assert out == {"added": 1, "removed": 1, "changed": 1}
        assert fake.added == ["agent_hb_1", "agent_hb_3"]
        assert fake.removed == ["agent_hb_3", "agent_hb_4"]  # 3 = replace, 4 = gone
        assert fake.ids() == {"agent_hb_1", "agent_hb_2", "agent_hb_3", f"{JOB_ID_PREFIX}1"}
        assert 4 not in svc._hb_signatures
        # a second pass with the same DB changes nothing
        assert svc.reconcile_agent_heartbeats(db) == {"added": 0, "removed": 0, "changed": 0}

    def test_no_scheduler_on_this_worker_is_a_noop(self):
        svc = HeartbeatService()
        db = _FakeAgentDB([_agent(1, {"enabled": True, "interval_minutes": 60})])
        assert svc.reconcile_agent_heartbeats(db) == {"added": 0, "removed": 0, "changed": 0}

    def test_boot_loader_populates_signatures_via_schedule(self):
        """schedule_agent_heartbeat records the signature, so jobs the boot
        loader adds are recognised as current by the first tick."""
        fake = _FakeScheduler()
        svc = self._service(fake)
        hb = {"enabled": True, "interval_minutes": 30}
        svc.schedule_agent_heartbeat(8, "ws-1", hb)
        assert svc._hb_signatures[8] == svc.heartbeat_signature("ws-1", hb)
        assert svc.reconcile_agent_heartbeats(_FakeAgentDB([_agent(8, hb)])) == {"added": 0, "removed": 0, "changed": 0}


# ── the tick ───────────────────────────────────────────────────────────────


class TestTick:
    def test_run_reconcile_once_opens_and_closes_its_own_session(self, monkeypatch):
        db = _FakeTaskDB([])
        import core.database.database as dbmod
        import services.heartbeat_service as hbmod

        monkeypatch.setattr(dbmod, "SessionLocal", lambda: db)
        monkeypatch.setattr(
            sr, "ScheduledTaskService",
            lambda _db, workspace_id=None: SimpleNamespace(
                reconcile_with_scheduler=lambda s: {"added": 1, "removed": 0, "skipped": False}),
        )
        monkeypatch.setattr(
            hbmod, "get_heartbeat_service",
            lambda: SimpleNamespace(reconcile_agent_heartbeats=lambda _db: {"added": 0, "removed": 2, "changed": 0}),
        )
        out = sr.run_reconcile_once(_FakeScheduler())
        assert out == {
            "tasks": {"added": 1, "removed": 0, "skipped": False},
            "heartbeats": {"added": 0, "removed": 2, "changed": 0},
        }
        assert db.closed is True

    def test_start_registers_the_interval_job_and_runs_a_first_pass(self, monkeypatch):
        fake = _FakeScheduler()
        passes: list = []
        monkeypatch.setattr(sr, "run_reconcile_once", lambda s, db=None: passes.append(s) or {})
        assert asyncio.run(sr.start_schedule_reconcile(fake)) is True
        assert fake.added == [sr.RECONCILE_JOB_ID]
        assert passes == [fake]

    def test_start_is_a_noop_off_the_leader(self):
        assert asyncio.run(sr.start_schedule_reconcile(None)) is False
        fake = _FakeScheduler(running=False)
        assert asyncio.run(sr.start_schedule_reconcile(fake)) is False
        assert fake.added == []

    def test_a_failed_pass_does_not_kill_the_job(self, monkeypatch):
        def boom(s, db=None):
            raise RuntimeError("db down")

        monkeypatch.setattr(sr, "run_reconcile_once", boom)
        sr._tick(_FakeScheduler())  # no raise

    def test_boot_starts_the_tick_after_loading_tasks(self):
        src = (_ORCH / "main.py").read_text()
        load = src.index("load_active_tasks_to_scheduler()")
        tick = src.index("start_schedule_reconcile(shared_sched)")
        coordinator = src.index("PRD-82A: Coordinator tick")
        assert load < tick < coordinator

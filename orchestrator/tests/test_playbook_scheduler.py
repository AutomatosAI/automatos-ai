"""Unit tests for PlaybookSchedulerService.

Tests the scheduler in isolation with mocked APScheduler, DB, and executor.

apscheduler is imported at the TOP of services.playbook_scheduler, so we must
satisfy that import before importing the service. apscheduler is not installed
in the local venv (it IS in CI), so we install minimal stubs ONLY where absent,
import the service, then restore sys.modules — the stubs never reach the
collection of sibling test modules. (PRD-142 W2-S2b.)
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeCronTrigger:
    """Minimal CronTrigger stub that validates basic cron syntax."""
    @classmethod
    def from_crontab(cls, expression: str):
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Wrong number of fields; got {len(parts)}, expected 5")
        for p in parts:
            if p == "*" or p.replace("-", "").replace("/", "").replace(",", "").isdigit():
                continue
            raise ValueError(f"Invalid cron field: {p}")
        return MagicMock(name="CronTrigger")


# ---------------------------------------------------------------------------
# apscheduler stubs: snapshot -> install-if-absent -> import -> restore.
# ---------------------------------------------------------------------------
_APS_KEYS = (
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.asyncio",
    "apscheduler.jobstores",
    "apscheduler.jobstores.memory",
    "apscheduler.triggers",
    "apscheduler.triggers.cron",
)
_APS_SNAPSHOT = {k: sys.modules.get(k) for k in _APS_KEYS}


def _install_apscheduler_stubs():
    _pkg = MagicMock()
    stubs = {
        "apscheduler": _pkg,
        "apscheduler.schedulers": _pkg,
        "apscheduler.schedulers.asyncio": MagicMock(AsyncIOScheduler=MagicMock),
        "apscheduler.jobstores": _pkg,
        "apscheduler.jobstores.memory": MagicMock(MemoryJobStore=MagicMock),
        "apscheduler.triggers": _pkg,
        "apscheduler.triggers.cron": MagicMock(CronTrigger=_FakeCronTrigger),
    }
    for name, mod in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = mod


def _restore_apscheduler_stubs():
    for k, prior in _APS_SNAPSHOT.items():
        if prior is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = prior


_install_apscheduler_stubs()
from services.playbook_scheduler import PlaybookSchedulerService, get_playbook_scheduler
import services.playbook_scheduler as sched_mod

# Replace the (real-or-stub) CronTrigger with our fake so cron validation is
# deterministic regardless of whether apscheduler is installed.
sched_mod.CronTrigger = _FakeCronTrigger
_restore_apscheduler_stubs()


# ---------------------------------------------------------------------------
# Helpers: mock modules for the lazy imports inside _load_cron_playbooks /
# _fire_playbook.
# ---------------------------------------------------------------------------

def _make_db_mock(query_results=None, first_result=None):
    """Create a mock DB session + a module mock for core.database.database."""
    mock_db = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = query_results or []
    mock_query.first.return_value = first_result
    mock_db.query.return_value = mock_query

    db_module = types.ModuleType("core.database.database")
    db_module.SessionLocal = MagicMock(return_value=mock_db)
    db_module.get_db = MagicMock()
    return mock_db, db_module


def _make_playbook_model_mock():
    """Create a WorkflowTemplate mock with SQLAlchemy-like column attributes."""
    model = MagicMock()
    model.schedule_config = MagicMock()
    model.workspace_id = MagicMock()
    model.steps = MagicMock()
    return model


def _lazy_import_patches(db_module, extra_modules=None):
    """sys.modules dict patching the lazy imports in _fire_playbook /
    _load_cron_playbooks. ``WorkflowTemplate`` (aliased WorkflowPlaybook),
    ``RecipeExecution``, ``api.recipe_executor`` and ``launch_recipe_task`` keep
    their source names — the scheduler module was renamed but those symbols were
    not."""
    playbook_model = _make_playbook_model_mock()
    mock_core_models = MagicMock()
    mock_core_models.WorkflowTemplate = playbook_model

    modules = {
        "core.database.database": db_module,
        "core.models": mock_core_models,
        "core.models.core": MagicMock(RecipeExecution=MagicMock()),
        "api.recipe_executor": MagicMock(launch_recipe_task=MagicMock()),
        "services.concurrency_guard": MagicMock(
            check_concurrency=AsyncMock(return_value=MagicMock(allowed=True, reason=""))
        ),
        "sqlalchemy": sys.modules.get("sqlalchemy", MagicMock()),
    }
    if extra_modules:
        modules.update(extra_modules)
    return modules


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestSchedulerLifecycle:
    """Tests for start() and stop() methods."""

    @pytest.mark.asyncio
    async def test_start_standalone_creates_own_scheduler(self):
        """start() without shared scheduler creates AsyncIOScheduler."""
        svc = PlaybookSchedulerService()

        mock_sched_instance = MagicMock()
        mock_load = AsyncMock()
        svc._load_cron_playbooks = mock_load

        with patch.object(sched_mod, "AsyncIOScheduler", return_value=mock_sched_instance), \
             patch.object(sched_mod, "MemoryJobStore", return_value=MagicMock()):
            await svc.start()

        assert svc._scheduler is mock_sched_instance
        assert svc._owns_scheduler is True
        mock_sched_instance.start.assert_called_once()
        mock_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_with_shared_scheduler(self):
        """start(scheduler=X) uses shared scheduler, does not create its own."""
        svc = PlaybookSchedulerService()

        shared_sched = MagicMock()
        mock_load = AsyncMock()
        svc._load_cron_playbooks = mock_load

        await svc.start(scheduler=shared_sched)

        assert svc._scheduler is shared_sched
        assert svc._owns_scheduler is False
        # Should NOT call start on the shared scheduler (UnifiedScheduler owns that)
        shared_sched.start.assert_not_called()
        mock_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_owned_scheduler(self):
        """stop() calls scheduler.shutdown() only when we own it."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched
        svc._owns_scheduler = True

        await svc.stop()

        mock_sched.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_stop_shared_scheduler_no_shutdown(self):
        """stop() does NOT shutdown the shared scheduler."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched
        svc._owns_scheduler = False

        await svc.stop()

        mock_sched.shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_no_scheduler_is_noop(self):
        """stop() with no scheduler does not error."""
        svc = PlaybookSchedulerService()
        svc._scheduler = None
        await svc.stop()  # should not raise


# ===========================================================================
# Schedule / Unschedule
# ===========================================================================

class TestScheduleUnschedule:
    """Tests for schedule_playbook() and unschedule_playbook()."""

    def test_schedule_playbook_valid_cron(self, mock_playbook):
        """schedule_playbook() with valid cron adds a job via add_job."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        svc.schedule_playbook(mock_playbook)

        mock_sched.add_job.assert_called_once()
        call_kwargs = mock_sched.add_job.call_args
        assert call_kwargs.kwargs["id"] == "playbook_cron_42"
        assert call_kwargs.kwargs["replace_existing"] is True

    def test_schedule_playbook_replaces_existing(self, mock_playbook):
        """If job already exists, removes it then adds new one."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()  # job exists
        svc._scheduler = mock_sched

        svc.schedule_playbook(mock_playbook)

        mock_sched.remove_job.assert_called_once_with("playbook_cron_42")
        mock_sched.add_job.assert_called_once()

    def test_schedule_playbook_invalid_cron(self, mock_playbook):
        """Invalid cron expression logs error, no job added."""
        mock_playbook.schedule_config = {"type": "cron", "cron_expression": "not valid cron"}

        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        svc.schedule_playbook(mock_playbook)

        mock_sched.add_job.assert_not_called()

    def test_schedule_playbook_no_expression(self, mock_playbook):
        """Missing cron_expression early-returns without adding job."""
        mock_playbook.schedule_config = {"type": "cron"}

        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        svc.schedule_playbook(mock_playbook)

        mock_sched.add_job.assert_not_called()

    def test_unschedule_playbook(self):
        """unschedule_playbook() removes job by playbook ID."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()  # job exists
        svc._scheduler = mock_sched

        svc.unschedule_playbook(1)

        mock_sched.remove_job.assert_called_once_with("playbook_cron_1")

    def test_unschedule_playbook_not_found(self):
        """unschedule_playbook() with no existing job does not error."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        svc.unschedule_playbook(999)

        mock_sched.remove_job.assert_not_called()


# ===========================================================================
# Load from DB
# ===========================================================================

class TestLoadCronPlaybooks:
    """Tests for _load_cron_playbooks()."""

    @pytest.mark.asyncio
    async def test_load_cron_playbooks(self, mock_playbook, mock_playbook_manual):
        """_load_cron_playbooks queries DB and schedules only cron-type playbooks."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(query_results=[mock_playbook, mock_playbook_manual])

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._load_cron_playbooks()

        # Only the cron playbook should be scheduled (manual is skipped)
        assert mock_sched.add_job.call_count == 1


# ===========================================================================
# Fire
# ===========================================================================

class TestFirePlaybook:
    """Tests for _fire_playbook()."""

    @pytest.mark.asyncio
    async def test_fire_playbook_success(self, mock_playbook):
        """Re-fetches playbook, creates RecipeExecution, calls launch_recipe_task."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(first_result=mock_playbook)
        mock_launch = MagicMock()
        mock_execution_cls = MagicMock()

        extra = {
            "core.models.core": MagicMock(RecipeExecution=mock_execution_cls),
            "api.recipe_executor": MagicMock(launch_recipe_task=mock_launch),
        }

        with patch.dict(sys.modules, _lazy_import_patches(db_module, extra)):
            await svc._fire_playbook(mock_playbook.id, str(mock_playbook.workspace_id))

        # Created + committed exactly one execution (concurrency allowed -> no rollback)
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        # Launched the playbook task
        mock_launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_fire_playbook_concurrency_blocked(self, mock_playbook):
        """Concurrency limit reached -> rolls back execution, does not launch."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(first_result=mock_playbook)
        mock_launch = MagicMock()

        extra = {
            "api.recipe_executor": MagicMock(launch_recipe_task=mock_launch),
            "services.concurrency_guard": MagicMock(
                check_concurrency=AsyncMock(return_value=MagicMock(allowed=False, reason="at capacity"))
            ),
        }

        with patch.dict(sys.modules, _lazy_import_patches(db_module, extra)):
            await svc._fire_playbook(mock_playbook.id, str(mock_playbook.workspace_id))

        # Pending execution added then deleted (commit for both add and rollback)
        mock_db.add.assert_called_once()
        mock_db.delete.assert_called_once()
        # Never launched
        mock_launch.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_playbook_deleted(self, mock_playbook):
        """Playbook no longer in DB -> unschedules, no execution."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(first_result=None)

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._fire_playbook(mock_playbook.id, str(mock_playbook.workspace_id))

        # Should unschedule
        mock_sched.remove_job.assert_called_once_with(f"playbook_cron_{mock_playbook.id}")
        # Should NOT create an execution
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_playbook_no_longer_cron(self, mock_playbook):
        """Playbook changed to manual -> unschedules, no execution."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()
        svc._scheduler = mock_sched

        # Playbook is now manual
        mock_playbook.schedule_config = {"type": "manual"}

        mock_db, db_module = _make_db_mock(first_result=mock_playbook)

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._fire_playbook(mock_playbook.id, str(mock_playbook.workspace_id))

        # Should unschedule
        mock_sched.remove_job.assert_called_once_with(f"playbook_cron_{mock_playbook.id}")
        # Should NOT create an execution
        mock_db.add.assert_not_called()


# ===========================================================================
# Status
# ===========================================================================

class TestGetStatus:
    """Tests for get_status()."""

    def test_get_status_with_jobs(self):
        """get_status() returns active flag and job list."""
        svc = PlaybookSchedulerService()
        mock_sched = MagicMock()
        mock_sched.running = True

        mock_job = MagicMock()
        mock_job.id = "playbook_cron_42"
        mock_job.next_run_time.isoformat.return_value = "2026-03-02T09:00:00"
        mock_job.trigger = MagicMock(__str__=lambda self: "cron[hour='9']")
        mock_sched.get_jobs.return_value = [mock_job]

        svc._scheduler = mock_sched

        status = svc.get_status()
        assert status["active"] is True
        assert len(status["jobs"]) == 1
        assert status["jobs"][0]["id"] == "playbook_cron_42"

    def test_get_status_no_scheduler(self):
        """get_status() with no scheduler returns inactive."""
        svc = PlaybookSchedulerService()
        svc._scheduler = None

        status = svc.get_status()
        assert status["active"] is False
        assert status["jobs"] == []


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    """Tests for get_playbook_scheduler() singleton."""

    def test_singleton_returns_same_instance(self):
        """get_playbook_scheduler() returns same instance on repeated calls."""
        # Reset singleton
        sched_mod._playbook_scheduler = None

        svc1 = get_playbook_scheduler()
        svc2 = get_playbook_scheduler()
        assert svc1 is svc2

        # Clean up
        sched_mod._playbook_scheduler = None

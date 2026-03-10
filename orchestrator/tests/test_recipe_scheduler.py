"""Unit tests for RecipeSchedulerService.

Tests the scheduler in isolation with mocked APScheduler, DB, and executor.
Follows async test pattern from test_plugin_runtime_integration.py.
"""
import sys
import types
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub apscheduler modules before importing recipe_scheduler
# ---------------------------------------------------------------------------
_mock_apscheduler = MagicMock()


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


sys.modules.setdefault("apscheduler", _mock_apscheduler)
sys.modules.setdefault("apscheduler.schedulers", _mock_apscheduler)
sys.modules.setdefault("apscheduler.schedulers.asyncio", MagicMock(AsyncIOScheduler=MagicMock))
sys.modules.setdefault("apscheduler.jobstores", _mock_apscheduler)
sys.modules.setdefault("apscheduler.jobstores.memory", MagicMock(MemoryJobStore=MagicMock))
sys.modules.setdefault("apscheduler.triggers", _mock_apscheduler)
sys.modules.setdefault("apscheduler.triggers.cron", MagicMock(CronTrigger=_FakeCronTrigger))

# Now we can import
from services.recipe_scheduler import RecipeSchedulerService, get_recipe_scheduler
import services.recipe_scheduler as sched_mod

# Replace the real CronTrigger with our fake so schedule_recipe validation works
sched_mod.CronTrigger = _FakeCronTrigger


# ---------------------------------------------------------------------------
# Helper: mock modules for lazy imports inside _load_cron_recipes / _fire_recipe
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


def _make_recipe_model_mock():
    """Create a WorkflowTemplate mock with SQLAlchemy-like column attributes."""
    model = MagicMock()
    # SQLAlchemy column descriptors used in .filter()
    model.schedule_config = MagicMock()
    model.workspace_id = MagicMock()
    model.steps = MagicMock()
    return model


def _lazy_import_patches(db_module, extra_modules=None):
    """Build a sys.modules dict for patching lazy imports in _fire_recipe / _load_cron_recipes."""
    recipe_model = _make_recipe_model_mock()
    mock_core_models = MagicMock()
    mock_core_models.WorkflowTemplate = recipe_model

    mock_exec_module = MagicMock()
    mock_exec_module.execute_recipe_direct = AsyncMock()

    modules = {
        "core.database.database": db_module,
        "core.models": mock_core_models,
        "core.models.core": MagicMock(RecipeExecution=MagicMock()),
        "api.recipe_executor": mock_exec_module,
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
        svc = RecipeSchedulerService()

        mock_sched_instance = MagicMock()
        mock_load = AsyncMock()
        svc._load_cron_recipes = mock_load

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
        svc = RecipeSchedulerService()

        shared_sched = MagicMock()
        mock_load = AsyncMock()
        svc._load_cron_recipes = mock_load

        await svc.start(scheduler=shared_sched)

        assert svc._scheduler is shared_sched
        assert svc._owns_scheduler is False
        # Should NOT have called start on the shared scheduler (UnifiedScheduler handles that)
        shared_sched.start.assert_not_called()
        mock_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_owned_scheduler(self):
        """stop() calls scheduler.shutdown() only when we own it."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched
        svc._owns_scheduler = True

        await svc.stop()

        mock_sched.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_stop_shared_scheduler_no_shutdown(self):
        """stop() does NOT shutdown the shared scheduler."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched
        svc._owns_scheduler = False

        await svc.stop()

        mock_sched.shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_no_scheduler_is_noop(self):
        """stop() with no scheduler does not error."""
        svc = RecipeSchedulerService()
        svc._scheduler = None
        await svc.stop()  # should not raise


# ===========================================================================
# Schedule / Unschedule
# ===========================================================================

class TestScheduleUnschedule:
    """Tests for schedule_recipe() and unschedule_recipe()."""

    def test_schedule_recipe_valid_cron(self, mock_recipe):
        """schedule_recipe() with valid cron adds a job via add_job."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        svc.schedule_recipe(mock_recipe)

        mock_sched.add_job.assert_called_once()
        call_kwargs = mock_sched.add_job.call_args
        assert call_kwargs.kwargs["id"] == "recipe_cron_42"
        assert call_kwargs.kwargs["replace_existing"] is True

    def test_schedule_recipe_replaces_existing(self, mock_recipe):
        """If job already exists, removes it then adds new one."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()  # job exists
        svc._scheduler = mock_sched

        svc.schedule_recipe(mock_recipe)

        mock_sched.remove_job.assert_called_once_with("recipe_cron_42")
        mock_sched.add_job.assert_called_once()

    def test_schedule_recipe_invalid_cron(self, mock_recipe):
        """Invalid cron expression logs error, no job added."""
        mock_recipe.schedule_config = {"type": "cron", "cron_expression": "not valid cron"}

        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        svc.schedule_recipe(mock_recipe)

        mock_sched.add_job.assert_not_called()

    def test_schedule_recipe_no_expression(self, mock_recipe):
        """Missing cron_expression early-returns without adding job."""
        mock_recipe.schedule_config = {"type": "cron"}

        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        svc.schedule_recipe(mock_recipe)

        mock_sched.add_job.assert_not_called()

    def test_unschedule_recipe(self):
        """unschedule_recipe() removes job by recipe ID."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()  # job exists
        svc._scheduler = mock_sched

        svc.unschedule_recipe(1)

        mock_sched.remove_job.assert_called_once_with("recipe_cron_1")

    def test_unschedule_recipe_not_found(self):
        """unschedule_recipe() with no existing job does not error."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        svc.unschedule_recipe(999)

        mock_sched.remove_job.assert_not_called()


# ===========================================================================
# Load from DB
# ===========================================================================

class TestLoadCronRecipes:
    """Tests for _load_cron_recipes()."""

    @pytest.mark.asyncio
    async def test_load_cron_recipes(self, mock_recipe, mock_recipe_manual):
        """_load_cron_recipes queries DB and schedules only cron-type recipes."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = None
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(query_results=[mock_recipe, mock_recipe_manual])

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._load_cron_recipes()

        # Only the cron recipe should be scheduled (manual is skipped)
        assert mock_sched.add_job.call_count == 1


# ===========================================================================
# Fire
# ===========================================================================

class TestFireRecipe:
    """Tests for _fire_recipe()."""

    @pytest.mark.asyncio
    async def test_fire_recipe_success(self, mock_recipe):
        """Re-fetches recipe, creates RecipeExecution, calls launch_recipe_task."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(first_result=mock_recipe)
        mock_launch = MagicMock()
        mock_execution_cls = MagicMock()

        extra = {
            "core.models.core": MagicMock(RecipeExecution=mock_execution_cls),
            "api.recipe_executor": MagicMock(launch_recipe_task=mock_launch),
        }

        with patch.dict(sys.modules, _lazy_import_patches(db_module, extra)):
            await svc._fire_recipe(mock_recipe.id, str(mock_recipe.workspace_id))

        # Should have added a RecipeExecution to the DB
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        # Should have launched the recipe task
        mock_launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_fire_recipe_deleted(self, mock_recipe):
        """Recipe no longer in DB -> unschedules, no execution."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()
        svc._scheduler = mock_sched

        mock_db, db_module = _make_db_mock(first_result=None)

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._fire_recipe(mock_recipe.id, str(mock_recipe.workspace_id))

        # Should unschedule
        mock_sched.remove_job.assert_called_once_with(f"recipe_cron_{mock_recipe.id}")
        # Should NOT create an execution
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_recipe_no_longer_cron(self, mock_recipe):
        """Recipe changed to manual -> unschedules, no execution."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.get_job.return_value = MagicMock()
        svc._scheduler = mock_sched

        # Recipe is now manual
        mock_recipe.schedule_config = {"type": "manual"}

        mock_db, db_module = _make_db_mock(first_result=mock_recipe)

        with patch.dict(sys.modules, _lazy_import_patches(db_module)):
            await svc._fire_recipe(mock_recipe.id, str(mock_recipe.workspace_id))

        # Should unschedule
        mock_sched.remove_job.assert_called_once_with(f"recipe_cron_{mock_recipe.id}")
        # Should NOT create an execution
        mock_db.add.assert_not_called()


# ===========================================================================
# Status
# ===========================================================================

class TestGetStatus:
    """Tests for get_status()."""

    def test_get_status_with_jobs(self):
        """get_status() returns active flag and job list."""
        svc = RecipeSchedulerService()
        mock_sched = MagicMock()
        mock_sched.running = True

        mock_job = MagicMock()
        mock_job.id = "recipe_cron_42"
        mock_job.next_run_time.isoformat.return_value = "2026-03-02T09:00:00"
        mock_job.trigger = MagicMock(__str__=lambda self: "cron[hour='9']")
        mock_sched.get_jobs.return_value = [mock_job]

        svc._scheduler = mock_sched

        status = svc.get_status()
        assert status["active"] is True
        assert len(status["jobs"]) == 1
        assert status["jobs"][0]["id"] == "recipe_cron_42"

    def test_get_status_no_scheduler(self):
        """get_status() with no scheduler returns inactive."""
        svc = RecipeSchedulerService()
        svc._scheduler = None

        status = svc.get_status()
        assert status["active"] is False
        assert status["jobs"] == []


# ===========================================================================
# Singleton
# ===========================================================================

class TestSingleton:
    """Tests for get_recipe_scheduler() singleton."""

    def test_singleton_returns_same_instance(self):
        """get_recipe_scheduler() returns same instance on repeated calls."""
        # Reset singleton
        sched_mod._recipe_scheduler = None

        svc1 = get_recipe_scheduler()
        svc2 = get_recipe_scheduler()
        assert svc1 is svc2

        # Clean up
        sched_mod._recipe_scheduler = None

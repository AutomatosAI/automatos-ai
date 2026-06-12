"""Unit tests for UnifiedScheduler + HeartbeatService cron conversion."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub apscheduler modules before importing
# ---------------------------------------------------------------------------
_mock_apscheduler = MagicMock()
sys.modules.setdefault("apscheduler", _mock_apscheduler)
sys.modules.setdefault("apscheduler.schedulers", _mock_apscheduler)
sys.modules.setdefault("apscheduler.schedulers.asyncio", MagicMock(AsyncIOScheduler=MagicMock))
sys.modules.setdefault("apscheduler.jobstores", _mock_apscheduler)
sys.modules.setdefault("apscheduler.jobstores.memory", MagicMock(MemoryJobStore=MagicMock))
sys.modules.setdefault("apscheduler.triggers", _mock_apscheduler)
sys.modules.setdefault("apscheduler.triggers.cron", MagicMock(CronTrigger=MagicMock))

from services.scheduler import UnifiedScheduler, get_unified_scheduler
import services.scheduler as sched_mod


# ===========================================================================
# UnifiedScheduler
# ===========================================================================

class TestUnifiedScheduler:

    def test_start_creates_and_starts_scheduler(self):
        svc = UnifiedScheduler()
        mock_sched = MagicMock()

        with patch.object(sched_mod, "AsyncIOScheduler", return_value=mock_sched), \
             patch.object(sched_mod, "MemoryJobStore", return_value=MagicMock()):
            svc.start()

        assert svc.apscheduler is mock_sched
        mock_sched.start.assert_called_once()

    def test_start_idempotent_when_running(self):
        svc = UnifiedScheduler()
        mock_sched = MagicMock()
        mock_sched.running = True
        svc._scheduler = mock_sched

        svc.start()  # should skip

        # AsyncIOScheduler constructor should NOT be called again
        mock_sched.start.assert_not_called()

    def test_stop_shuts_down_scheduler(self):
        svc = UnifiedScheduler()
        mock_sched = MagicMock()
        svc._scheduler = mock_sched

        svc.stop()

        mock_sched.shutdown.assert_called_once_with(wait=False)
        assert svc._scheduler is None

    def test_stop_noop_when_none(self):
        svc = UnifiedScheduler()
        svc._scheduler = None
        svc.stop()  # should not raise

    def test_singleton_returns_same_instance(self):
        sched_mod._unified_scheduler = None

        s1 = get_unified_scheduler()
        s2 = get_unified_scheduler()
        assert s1 is s2

        sched_mod._unified_scheduler = None


# ===========================================================================
# HeartbeatService: _interval_to_cron_trigger
# ===========================================================================

class TestIntervalToCronTrigger:
    """Test the cron conversion helper that replaces interval triggers."""

    def _make_real_trigger(self):
        """Import with real CronTrigger for validation."""
        # We need the real CronTrigger to test the conversion
        from apscheduler.triggers.cron import CronTrigger
        return CronTrigger

    # PRD-162: _interval_to_cron_trigger now builds from the shared
    # schedule_util.interval_to_cron() via CronTrigger.from_crontab(), so the
    # cron-field math lives in ONE place. These assert the cron STRING handed to
    # from_crontab — behaviour-equivalent to the old explicit minute/hour kwargs.

    def test_60min_is_top_of_hour(self):
        """60 minutes → fires at minute 0 every hour."""
        from services.heartbeat_service import HeartbeatService

        with patch("services.heartbeat_service.CronTrigger") as MockCron:
            MockCron.from_crontab.side_effect = lambda expr: expr
            result = HeartbeatService._interval_to_cron_trigger(60)

        assert result == "0 * * * *"

    def test_30min_fires_twice_per_hour(self):
        """30 minutes → fires at :00 and :30."""
        from services.heartbeat_service import HeartbeatService

        with patch("services.heartbeat_service.CronTrigger") as MockCron:
            MockCron.from_crontab.side_effect = lambda expr: expr
            result = HeartbeatService._interval_to_cron_trigger(30)

        assert result == "0,30 * * * *"

    def test_15min_fires_four_times_per_hour(self):
        """15 minutes → fires at :00, :15, :30, :45."""
        from services.heartbeat_service import HeartbeatService

        with patch("services.heartbeat_service.CronTrigger") as MockCron:
            MockCron.from_crontab.side_effect = lambda expr: expr
            result = HeartbeatService._interval_to_cron_trigger(15)

        assert result == "0,15,30,45 * * * *"

    def test_120min_fires_every_2_hours(self):
        """120 minutes → fires at minute 0, every 2nd hour."""
        from services.heartbeat_service import HeartbeatService

        with patch("services.heartbeat_service.CronTrigger") as MockCron:
            MockCron.from_crontab.side_effect = lambda expr: expr
            result = HeartbeatService._interval_to_cron_trigger(120)

        assert result == "0 */2 * * *"

    def test_zero_defaults_to_60(self):
        """0 minutes → treated as 60 (top of hour)."""
        from services.heartbeat_service import HeartbeatService

        with patch("services.heartbeat_service.CronTrigger") as MockCron:
            MockCron.from_crontab.side_effect = lambda expr: expr
            result = HeartbeatService._interval_to_cron_trigger(0)

        assert result == "0 * * * *"

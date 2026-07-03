"""PRD-177 S3 (F018): metadata sync scheduler + fail-CLOSED destructive gate.

Two gaps:

1. ``ComposioActionMetadata`` sync has no scheduler entry — the classification
   table only fills on app-enable / manual trigger, so on a cold table the
   destructive-action gate has nothing to check.
2. When the metadata table is empty, ``check_action_eligibility`` returned
   ``True`` (fail-OPEN) for EVERY action — including destructive ones. A missing
   classification must not silently permit a destructive Composio action.

Fix: register the sync alongside the nightly edge recompute (same scheduler),
and flip the empty-table path to fail-CLOSED for destructive intent when the
new config flag ``COMPOSIO_DESTRUCTIVE_FAIL_CLOSED`` is on (default True), while
still permitting clearly non-destructive intents so a cold start isn't bricked.

Pure unit tests — no DB, no Composio, no network. The eligibility check runs
against a fake session; the scheduler test uses a fake APScheduler.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


# ---------------------------------------------------------------------------
# Fake SQLAlchemy session that reports an EMPTY metadata table
# ---------------------------------------------------------------------------

class _EmptyResult:
    def scalar_one_or_none(self):
        return None

    def first(self):
        return None


class _EmptyMetadataSession:
    """execute(select(...)) always yields nothing — the table is empty."""

    def execute(self, *a, **k):
        return _EmptyResult()


# ---------------------------------------------------------------------------
# S3a: fail-CLOSED destructive gate on empty metadata
# ---------------------------------------------------------------------------

def _load_capability_filter():
    """Import ActionCapabilityFilter without triggering modules/tools/__init__."""
    from modules.tools.services.action_capability_filter import (
        get_action_capability_filter,
    )

    return get_action_capability_filter


def test_destructive_gate_fail_closed():
    """With an EMPTY metadata table, a destructive intent is DENIED (not allowed).

    Before F018 this returned ``(True, 'Metadata not yet synced (allowing)')``.
    """
    get_filter = _load_capability_filter()
    svc = get_filter(_EmptyMetadataSession())

    eligible, reason = svc.check_action_eligibility(
        action_id="SLACK_DELETE_MESSAGE",
        intent="delete that message from the channel",
        allow_destructive=False,
    )
    assert eligible is False, (
        "destructive action on empty metadata must fail CLOSED, not allow"
    )
    assert "sync" in reason.lower() or "confirm" in reason.lower() or "verif" in reason.lower()


def test_non_destructive_intent_still_allowed_on_empty_metadata():
    """Cold-start isn't bricked: a clearly non-destructive intent still passes on
    an empty table, so the platform works before the first sync."""
    get_filter = _load_capability_filter()
    svc = get_filter(_EmptyMetadataSession())

    eligible, _reason = svc.check_action_eligibility(
        action_id="SLACK_SEND_MESSAGE",
        intent="send a friendly message to the team",
        allow_destructive=False,
    )
    assert eligible is True


def test_explicit_allow_destructive_overrides_on_empty_metadata():
    """When the caller has already confirmed (allow_destructive=True), the gate
    permits even a destructive-looking intent on an empty table."""
    get_filter = _load_capability_filter()
    svc = get_filter(_EmptyMetadataSession())

    eligible, _reason = svc.check_action_eligibility(
        action_id="SLACK_DELETE_MESSAGE",
        intent="delete that message",
        allow_destructive=True,
    )
    assert eligible is True


# ---------------------------------------------------------------------------
# S3b: metadata sync is registered on the scheduler
# ---------------------------------------------------------------------------

class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def add_job(self, func, trigger, **kwargs):
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})


def _load_sync_scheduler():
    """Load the composio sync scheduler module directly."""
    path = Path(_orchestrator_root) / "services" / "composio_sync_scheduler.py"
    spec = importlib.util.spec_from_file_location("composio_sync_scheduler_prd177", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_metadata_sync_scheduled():
    """A composio metadata sync job is registered on the scheduler (the same
    APScheduler that runs the nightly edge recompute)."""
    try:
        mod = _load_sync_scheduler()
    except FileNotFoundError:
        pytest.fail("services/composio_sync_scheduler.py must exist (F018)")

    scheduler = _FakeScheduler()
    svc = mod.get_composio_sync_scheduler()
    await svc.start(scheduler)

    assert scheduler.jobs, "a composio sync job must be registered on the scheduler"
    job = scheduler.jobs[0]
    assert job["trigger"] == "cron"
    assert job.get("id"), "the sync job needs a stable id (replace_existing safe)"
    assert callable(job["func"])

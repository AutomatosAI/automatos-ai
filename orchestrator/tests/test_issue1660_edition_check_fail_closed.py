"""#1660: the edition check fails closed when it never ran, and backs off while it is failing.

* **Never ran** (``NOT_RUN``: a boot that stopped before ``confirm_edition``): the device lane and the pairing
  routes do not serve a device token unchecked; the first request runs the check, and is refused (503) when
  it cannot run.
* **Failing** (``FAILED``): the check runs again at most once every ``RETRY_INTERVAL_SECONDS`` per process; a
  request in between is answered 503 at once, without a session; one after the interval retries it.
* **One at a time**: requests run in the threadpool; while one runs the check, the others are refused, not
  queued behind it.
"""
from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import api.mobile_devices as devices_api  # noqa: E402
from config import config  # noqa: E402
from core.auth import device  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from services.mobile import edition_change  # noqa: E402

TOKEN = "amd_" + "x" * 43
REQUEST = SimpleNamespace(headers={})


class _Checks:
    """Stands in for ``_run_check``: counts the runs, and fails while ``failing``."""

    def __init__(self, failing=False):
        self.runs = 0
        self.failing = failing

    def __call__(self):
        self.runs += 1
        if self.failing:
            raise RuntimeError("database unreachable")


@pytest.fixture
def install(monkeypatch):
    """An enterprise install that pairs, whose boot never reached the check, on a clock the test moves."""
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "ENTERPRISE_DEVICE_PAIRING", "true")
    monkeypatch.setattr(edition_change, "_status", edition_change.NOT_RUN)
    clock = SimpleNamespace(now=5000.0)
    monkeypatch.setattr(edition_change, "_seconds", lambda: clock.now)
    checks = _Checks()
    monkeypatch.setattr(edition_change, "_run_check", checks)
    return SimpleNamespace(clock=clock, checks=checks)


def _refused(call) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (503, edition_change.NOT_CONFIRMED)


@pytest.mark.asyncio
async def test_a_check_that_never_ran_is_run_by_the_first_device_request(install, monkeypatch):
    found = RequestContext(workspace_id="ws", user=UserContext(id="person"), auth_type=device.DEVICE_AUTH_TYPE)
    monkeypatch.setattr(device, "_resolve_in_own_session", lambda request, token: found)
    monkeypatch.setattr(device, "_hybrid", lambda: SimpleNamespace(_enrich_log_context=lambda ctx: None))

    assert edition_change.needs_confirming() is True
    assert await device.resolve_device_context(REQUEST, TOKEN) is found
    assert install.checks.runs == 1 and edition_change.needs_confirming() is False


@pytest.mark.asyncio
async def test_a_check_that_never_ran_and_cannot_run_refuses_the_device_lane_and_pairing(install, monkeypatch):
    install.checks.failing = True
    monkeypatch.setattr(device, "_resolve_in_own_session", lambda request, token: pytest.fail("no lookup"))

    with pytest.raises(HTTPException) as refused:
        await device.resolve_device_context(REQUEST, TOKEN)
    assert refused.value.status_code == 503 and install.checks.runs == 1

    install.clock.now += edition_change.RETRY_INTERVAL_SECONDS
    _refused(devices_api._require_device_pairing)
    assert install.checks.runs == 2


def test_a_failed_check_is_not_run_again_within_the_interval_and_is_after_it(install):
    install.checks.failing = True
    assert edition_change.confirm_edition() is False and install.checks.runs == 1

    install.clock.now += edition_change.RETRY_INTERVAL_SECONDS - 1
    _refused(edition_change.ensure_confirmed)
    _refused(devices_api._require_device_pairing)
    assert install.checks.runs == 1

    install.clock.now += 1
    install.checks.failing = False
    edition_change.ensure_confirmed()
    assert install.checks.runs == 2 and edition_change.needs_confirming() is False


def test_a_retry_that_fails_again_starts_a_new_interval(install):
    install.checks.failing = True
    _refused(edition_change.ensure_confirmed)
    install.clock.now += edition_change.RETRY_INTERVAL_SECONDS
    _refused(edition_change.ensure_confirmed)
    install.clock.now += edition_change.RETRY_INTERVAL_SECONDS - 1
    _refused(edition_change.ensure_confirmed)
    assert install.checks.runs == 2


def test_while_one_request_runs_the_check_the_others_are_refused_at_once(install, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_check():
        install.checks.runs += 1
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(edition_change, "_run_check", slow_check)
    first = threading.Thread(target=edition_change.ensure_confirmed)
    first.start()
    try:
        assert started.wait(5)
        _refused(edition_change.ensure_confirmed)
    finally:
        release.set()
        first.join(5)
    assert install.checks.runs == 1 and edition_change.needs_confirming() is False

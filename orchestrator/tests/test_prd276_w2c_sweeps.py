"""PRD-276 Wave 2c (first batch): four platform sweeps are scheduled job kinds.

The run-callback sweep, the end-user retention sweep, Expo's push receipts and the mobile digest
each had their own APScheduler job, run on the scheduler process. Each is now a platform schedule
that only queues, and a job kind whose handler runs the same function on a worker. Each catches up
``latest``: one run after downtime does what the missed runs would have. Every worker loads them.
"""
from __future__ import annotations

import asyncio

import pytest

from core.jobs.registry import CATCH_UP_LATEST, kind_of

SWEEPS = [
    ("services.run_callbacks", "SWEEP_KIND", "run_sweep", "queue_stale_callbacks"),
    ("services.end_user_retention", "RETENTION_KIND", "run_sweep", "sweep_end_user_retention"),
    ("services.push.receipts", "RECEIPTS_KIND", "run_receipts_job", "run_receipts"),
    ("services.mobile.digest", "DIGEST_KIND", "run_digests_job", "run_digests"),
]


@pytest.mark.parametrize("module_name, kind_name, handler_name, work_name", SWEEPS)
def test_each_sweep_is_a_latest_kind_whose_handler_runs_its_work(monkeypatch, module_name, kind_name,
                                                                 handler_name, work_name):
    import importlib

    module = importlib.import_module(module_name)
    ran = []
    work = getattr(module, work_name)

    if asyncio.iscoroutinefunction(work):
        async def fake(*args, **kwargs):
            ran.append(work_name)
    else:
        def fake(*args, **kwargs):
            ran.append(work_name)

    monkeypatch.setattr(module, work_name, fake)
    spec = kind_of(getattr(module, kind_name))
    assert spec.catch_up == CATCH_UP_LATEST and spec.concurrency == 1
    asyncio.run(getattr(module, handler_name)(object()))
    assert ran == [work_name]


def test_every_worker_loads_the_sweeps():
    from core.jobs import worker

    assert {module for module, *_ in SWEEPS} <= set(worker.HANDLER_MODULES)

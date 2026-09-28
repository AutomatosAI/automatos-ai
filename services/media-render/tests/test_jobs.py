"""Render jobs (US-102): immutable snapshots, final states, expiry and the work dir."""

from __future__ import annotations

import pytest

from helpers import bundle
from media_render.bundle import parse_bundle
from media_render.jobs import DONE, PREPARING, QUEUED, JobStore, delete_files, job_json, reset_work_dir


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_a_job_is_replaced_never_changed(settings, tmp_path):
    clock = Clock()
    store = JobStore(tmp_path, clock)
    job = store.create(parse_bundle(bundle(reference="post-9"), settings, {}))
    assert job.status == PREPARING and job.dir == tmp_path / job.id and len(job.id) == 32
    queued = store.update(job.id, status=QUEUED)
    assert job.status == PREPARING and queued.status == QUEUED and store.get(job.id) is queued
    assert store.active_count() == 1
    with pytest.raises(ValueError):
        store.finish(job.id, QUEUED)
    clock.now += 5
    done = store.finish(job.id, DONE)
    assert done.finished_at == clock.now and store.active_count() == 0
    body = job_json(done, queue_position=None)
    assert body["reference"] == "post-9" and body["status"] == "done" and body["finished_at"].endswith("Z")
    assert "queue_position" not in body


def test_the_active_count_is_the_services_or_one_workspaces(settings, tmp_path):
    """P251W1-RVW-4: admission counts one workspace's unfinished jobs apart from the rest."""
    store = JobStore(tmp_path, Clock())
    ours = [store.create(parse_bundle(bundle("ws-a"), settings, {})) for _ in range(3)]
    store.create(parse_bundle(bundle("ws-b"), settings, {}))
    store.update(ours[1].id, status=QUEUED)
    store.finish(ours[2].id, DONE)
    assert store.active_count() == 3
    assert store.active_count("ws-a") == 2 and store.active_count("ws-b") == 1
    assert store.active_count("ws-c") == 0


def test_finished_jobs_expire_after_the_ttl(settings, tmp_path):
    clock = Clock()
    store = JobStore(tmp_path, clock)
    parsed = parse_bundle(bundle(), settings, {})
    finished, running = store.create(parsed), store.create(parsed)
    finished.dir.mkdir()
    store.finish(finished.id, DONE)
    clock.now += 59
    assert store.expire(60) == []
    clock.now += 2
    expired = store.expire(60)
    assert [job.id for job in expired] == [finished.id]
    delete_files(expired)
    assert not finished.dir.exists()
    assert store.get(finished.id) is None and store.get(running.id) is not None


def test_a_new_process_clears_only_job_directories(tmp_path):
    stale = tmp_path / ("a" * 32)
    (stale / "project").mkdir(parents=True)
    kept = tmp_path / "not-a-job"
    kept.mkdir()
    reset_work_dir(tmp_path)
    assert not stale.exists() and kept.exists()

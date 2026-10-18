"""PRD-262 US-006: what one authorised request may spend is capped.

- `metrics` names known templates only, at most one per template (11): /assess
  and /score start one thread and one FutureAGI evaluation per entry;
- `teacher_model` is allow-listed; `dataset` holds at most 10 examples (what the
  platform sends); `scoring_template` names a known template;
- a bounded number of /optimize jobs run at once.
"""
from __future__ import annotations

import threading

import pytest

ROUTES_WITH_METRICS = {
    "/assess": {"prompt_content": "You are helpful."},
    "/score": {"input_text": "q", "output_text": "a"},
}


def _optimize_body(**overrides):
    body = {
        "prompt_content": "You are helpful.",
        "dataset": [{"input": "q", "output": "a"}],
        "teacher_model": "gpt-4o-mini",
    }
    return {**body, **overrides}


@pytest.mark.parametrize("path", sorted(ROUTES_WITH_METRICS))
def test_more_metrics_than_templates_is_a_422(client, authed, worker_main, fake_scoring, path):
    names = list(worker_main.TEMPLATE_CONFIG)
    assert len(names) == 11
    body = {**ROUTES_WITH_METRICS[path], "metrics": names + [names[0]]}
    response = client.post(path, headers=authed, json=body)
    assert response.status_code == 422, response.text
    assert fake_scoring == []


@pytest.mark.parametrize("path", sorted(ROUTES_WITH_METRICS))
def test_an_unknown_metric_is_a_422(client, authed, fake_scoring, path):
    body = {**ROUTES_WITH_METRICS[path], "metrics": ["is_helpful", "spend_my_quota"]}
    response = client.post(path, headers=authed, json=body)
    assert response.status_code == 422, response.text
    assert fake_scoring == []


@pytest.mark.parametrize("path", sorted(ROUTES_WITH_METRICS))
def test_every_template_once_is_accepted(client, authed, worker_main, fake_scoring, path):
    names = list(worker_main.TEMPLATE_CONFIG)
    response = client.post(path, headers=authed, json={**ROUTES_WITH_METRICS[path], "metrics": names})
    assert response.status_code == 200, response.text
    assert sorted(fake_scoring) == sorted(names)


@pytest.mark.parametrize("overrides", [
    {"teacher_model": "gpt-4o"},
    {"teacher_model": "o1-pro"},
    {"dataset": [{"input": "q", "output": "a"}] * 11},
    {"dataset": []},
    {"scoring_template": "spend_my_quota"},
    {"num_rounds": 21},
], ids=["teacher-gpt-4o", "teacher-o1-pro", "dataset-11", "dataset-empty", "unknown-template", "rounds-21"])
def test_optimize_refuses_inputs_outside_the_caps(client, authed, overrides):
    response = client.post("/optimize", headers=authed, json=_optimize_body(**overrides))
    assert response.status_code == 422, response.text


@pytest.fixture()
def blocked_jobs(worker_main, monkeypatch):
    """Optimisation jobs that run until the test lets them finish; no OpenAI call."""
    for name in ("FUTUREAGI_API_KEY", "FUTUREAGI_SECRET_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "ci-not-a-real-key")
    release = threading.Event()
    started: list[str] = []

    def job(job_id, req):
        started.append(job_id)
        release.wait(timeout=30)

    monkeypatch.setattr(worker_main, "_run_optimize_job", job)
    yield started
    release.set()
    _wait_until_every_slot_is_free(worker_main)


def _wait_until_every_slot_is_free(worker_main):
    taken = []
    try:
        for _ in range(worker_main.MAX_CONCURRENT_JOBS):
            assert worker_main._JOB_SLOTS.acquire(timeout=10), "an optimisation slot was never freed"
            taken.append(1)
    finally:
        for _ in taken:
            worker_main._JOB_SLOTS.release()


def test_concurrent_optimisation_jobs_are_bounded(client, authed, worker_main, blocked_jobs):
    limit = worker_main.MAX_CONCURRENT_JOBS
    assert limit >= 1
    for _ in range(limit):
        response = client.post("/optimize", headers=authed, json=_optimize_body())
        assert response.status_code == 200, response.text
    refused = client.post("/optimize", headers=authed, json=_optimize_body())
    assert refused.status_code == 429, refused.text


class _StoreThatRefusesInserts(dict):
    def __setitem__(self, key, value):
        raise RuntimeError("the job store failed after the slot was taken")


@pytest.mark.parametrize("broken", ["cleanup", "job-store"])
def test_a_failure_before_the_job_starts_gives_its_slot_back(client, authed, worker_main, monkeypatch, broken):
    for name in ("FUTUREAGI_API_KEY", "FUTUREAGI_SECRET_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "ci-not-a-real-key")

    def explode():
        raise RuntimeError("the cleanup failed after the slot was taken")

    if broken == "cleanup":
        monkeypatch.setattr(worker_main, "_cleanup_old_jobs", explode)
    else:
        monkeypatch.setattr(worker_main, "_optimize_jobs", _StoreThatRefusesInserts())
    for _ in range(worker_main.MAX_CONCURRENT_JOBS + 1):
        with pytest.raises(RuntimeError):
            client.post("/optimize", headers=authed, json=_optimize_body())
    _wait_until_every_slot_is_free(worker_main)


def test_a_finished_job_frees_its_slot(client, authed, worker_main, monkeypatch):
    for name in ("FUTUREAGI_API_KEY", "FUTUREAGI_SECRET_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "ci-not-a-real-key")

    finished: list[str] = []

    def quick_job(job_id, req):
        # The real job records its own failure on the job and returns (main._run_optimize_job).
        finished.append(job_id)

    monkeypatch.setattr(worker_main, "_run_optimize_job", quick_job)
    for _ in range(worker_main.MAX_CONCURRENT_JOBS + 2):
        response = client.post("/optimize", headers=authed, json=_optimize_body())
        assert response.status_code == 200, response.text
        _wait_until_every_slot_is_free(worker_main)
    assert len(finished) == worker_main.MAX_CONCURRENT_JOBS + 2

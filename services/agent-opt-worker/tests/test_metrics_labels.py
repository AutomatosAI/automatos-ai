"""PRD-262 US-006: the worker's Prometheus labels are route templates.

A job id per poll, a scanner's paths or a forged Host header must not add
series: unmatched paths share one "unmatched" label.
"""
from __future__ import annotations

from prometheus_client import REGISTRY

SAMPLE = "agent_opt_worker_requests_total"


def _labels(name: str) -> set[str]:
    return {
        sample.labels[name]
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == SAMPLE
    }


def test_requests_are_labelled_by_route_template(client, authed):
    for job_id in ("job-a", "job-b", "job-c"):
        assert client.get(f"/optimize/{job_id}", headers=authed).status_code == 404
    for path in ("/wp-login.php", "/.env", "/optimize/x/y", "/admin/../etc/passwd"):
        assert client.get(path).status_code == 404
    client.get("/health", headers={"Host": "evil.example/forged"})
    client.request("PROPFIND", "/health")

    endpoints = _labels("endpoint")
    assert {"/optimize/{job_id}", "unmatched", "/health"} <= endpoints
    raw = {e for e in endpoints if e != "unmatched" and "{" not in e and e not in {"/health", "/assess", "/safety", "/score", "/optimize"}}
    assert not raw, f"raw paths leaked into the endpoint label: {raw}"
    assert _labels("method") <= {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "OTHER"}
    assert "OTHER" in _labels("method")

"""Performance baselines — latency SLO assertions for critical endpoints.

These tests validate that key endpoints respond within acceptable time
thresholds. Failures indicate performance regression, not bugs.

SLO tiers:
  - FAST (< 500ms):  Health, list endpoints, static config
  - MEDIUM (< 2s):   Detail lookups, search, analytics
  - SLOW (< 10s):    Execution endpoints, streaming, orchestrator runs
"""

import time

import pytest


# ── SLO thresholds (seconds) ────────────────────────────────────────

FAST = 0.5      # 500ms
MEDIUM = 2.0    # 2s
SLOW = 10.0     # 10s


def _timed_get(client, url, params=None, threshold=MEDIUM):
    """GET with latency measurement."""
    start = time.monotonic()
    r = client.get(url, params=params or {})
    elapsed = time.monotonic() - start
    return r, elapsed, threshold


def _timed_post(client, url, json=None, threshold=MEDIUM):
    """POST with latency measurement."""
    start = time.monotonic()
    r = client.post(url, json=json or {})
    elapsed = time.monotonic() - start
    return r, elapsed, threshold


def _assert_slo(r, elapsed, threshold, label):
    """Assert response succeeded within SLO."""
    assert r.status_code == 200, f"{label}: expected 200, got {r.status_code}"
    assert elapsed < threshold, (
        f"{label}: responded in {elapsed:.2f}s, SLO is {threshold}s"
    )


# ── FAST tier (< 500ms) ─────────────────────────────────────────────

def test_health_latency(client):
    """Health endpoint must respond in < 500ms."""
    r, elapsed, threshold = _timed_get(client, "/health", threshold=FAST)
    _assert_slo(r, elapsed, threshold, "GET /health")


def test_agent_list_latency(client):
    """Agent list must respond in < 500ms."""
    r, elapsed, threshold = _timed_get(client, "/api/agents/", threshold=FAST)
    _assert_slo(r, elapsed, threshold, "GET /api/agents/")


def test_workspace_current_latency(client):
    """Current workspace must respond in < 500ms."""
    r, elapsed, threshold = _timed_get(client, "/api/workspaces/current", threshold=FAST)
    _assert_slo(r, elapsed, threshold, "GET /api/workspaces/current")


def test_model_list_latency(client):
    """Model list must respond in < 500ms."""
    r, elapsed, threshold = _timed_get(client, "/api/models/", threshold=FAST)
    _assert_slo(r, elapsed, threshold, "GET /api/models/")


def test_key_list_latency(client):
    """Key list must respond in < 500ms."""
    r, elapsed, threshold = _timed_get(client, "/api/keys", threshold=FAST)
    _assert_slo(r, elapsed, threshold, "GET /api/keys")


# ── MEDIUM tier (< 2s) ──────────────────────────────────────────────

def test_chat_history_latency(client):
    """Chat history must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/chat/history", params={"limit": 10}, threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/chat/history")


def test_memory_stats_latency(client):
    """Memory stats must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/v1/memory/stats/real", threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/v1/memory/stats/real")


def test_analytics_dashboard_latency(client):
    """Analytics dashboard must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/analytics/dashboard", threshold=MEDIUM
    )
    # Accept 200 or 404 (if not configured)
    if r.status_code == 200:
        assert elapsed < threshold, (
            f"GET /api/analytics/dashboard: {elapsed:.2f}s > {threshold}s SLO"
        )


def test_document_list_latency(client):
    """Document list must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/documents/", threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/documents/")


def test_tools_marketplace_latency(client):
    """Tools marketplace must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/tools/marketplace", params={"limit": 10}, threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/tools/marketplace")


def test_mission_list_latency(client):
    """Mission list must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/missions", params={"limit": 10}, threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/missions")


def test_heartbeat_status_latency(client):
    """Heartbeat status must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/heartbeat/status", threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/heartbeat/status")


def test_recipe_list_latency(client):
    """Recipe list must respond in < 2s."""
    r, elapsed, threshold = _timed_get(
        client, "/api/workflow-recipes", params={"limit": 10}, threshold=MEDIUM
    )
    _assert_slo(r, elapsed, threshold, "GET /api/workflow-recipes")


# ── SLOW tier (< 10s) ───────────────────────────────────────────────

def test_heartbeat_orchestrator_run_latency(client):
    """Orchestrator run must complete in < 10s."""
    r, elapsed, threshold = _timed_post(
        client, "/api/heartbeat/orchestrator/run", threshold=SLOW
    )
    _assert_slo(r, elapsed, threshold, "POST /api/heartbeat/orchestrator/run")


def test_model_recommend_latency(client):
    """Model recommendation must respond in < 10s."""
    r, elapsed, threshold = _timed_post(
        client, "/api/models/recommend",
        json={"max_cost": 1.0, "min_context": 4000},
        threshold=SLOW,
    )
    _assert_slo(r, elapsed, threshold, "POST /api/models/recommend")

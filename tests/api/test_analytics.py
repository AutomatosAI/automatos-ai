"""Analytics, memory, evaluation, settings, and system metrics smoke tests."""


def test_dashboard_summary(client):
    r = client.get("/analytics/dashboard/summary")
    assert r.status_code == 200


def test_memory_stats(client):
    r = client.get("/api/v1/memory/stats/real")
    assert r.status_code == 200


def test_agent_performance(client):
    """Endpoint removed in PRD-125 Phase 3 — verify it returns 404/405."""
    r = client.get("/analytics/agents/performance")
    assert r.status_code in (404, 405, 410)


def test_system_settings(client):
    r = client.get("/api/system-settings/")
    assert r.status_code == 200


def test_system_metrics(client):
    r = client.get("/api/system/metrics")
    assert r.status_code == 200

"""Analytics error handling — validates graceful failures for edge cases."""


def test_analytics_dashboard_invalid_period(client):
    """GET /api/analytics/dashboard/overview with invalid period should not 500."""
    r = client.get("/api/analytics/dashboard/overview", params={"period": "INVALID"})
    assert r.status_code != 500, (
        f"Dashboard with invalid period returned 500: {r.text[:300]}"
    )


def test_llm_usage_invalid_date_range(client):
    """GET /api/analytics/llm/usage with invalid dates should not 500."""
    r = client.get("/api/analytics/llm/usage", params={
        "start_date": "not-a-date",
        "end_date": "also-not-a-date",
    })
    assert r.status_code != 500, (
        f"LLM usage with bad dates returned 500: {r.text[:300]}"
    )


def test_agent_analytics_invalid_agent(client):
    """GET /api/analytics/agents/{id} for non-existent agent."""
    r = client.get("/api/analytics/agents/999999")
    assert r.status_code != 500, (
        f"Analytics for non-existent agent returned 500: {r.text[:300]}"
    )


def test_llm_costs_negative_limit(client):
    """GET /api/analytics/llm/costs with negative limit should not 500."""
    r = client.get("/api/analytics/llm/costs", params={"limit": -1})
    assert r.status_code != 500, (
        f"LLM costs with negative limit returned 500: {r.text[:300]}"
    )

"""Journey 14: LLM analytics — usage, costs, projections."""


def test_llm_summary(client):
    r = client.get("/api/analytics/llm/summary", params={"period": "7d"})
    assert r.status_code == 200
    assert "total_requests" in r.json()


def test_llm_usage(client):
    r = client.get("/api/analytics/llm/usage", params={"period": "7d", "group_by": "model"})
    assert r.status_code == 200


def test_llm_costs(client):
    r = client.get("/api/analytics/llm/costs", params={"period": "7d", "breakdown": "model"})
    assert r.status_code == 200


def test_llm_recommendations(client):
    r = client.get("/api/analytics/llm/recommendations")
    assert r.status_code == 200


def test_llm_projections(client):
    r = client.get("/api/analytics/llm/projections", params={"period": "30d"})
    assert r.status_code == 200

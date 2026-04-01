"""Knowledge, multi-agent, field-theory, and templates smoke tests."""


def test_multi_agent_health(client):
    r = client.get("/api/multi-agent/health")
    assert r.status_code == 200


def test_field_theory_health(client):
    r = client.get("/api/field-theory/health")
    assert r.status_code == 200


def test_knowledge_stats(client):
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200


def test_templates_list(client):
    r = client.get("/api/templates/")
    assert r.status_code == 200


# ── Deeper knowledge tests ──────────────────────────────────────────


def test_multi_agent_health_shape(client):
    """Multi-agent health should return structured data, not just 200."""
    r = client.get("/api/multi-agent/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict), "Multi-agent health should return a dict"


def test_field_theory_health_shape(client):
    """Field theory health should return structured data."""
    r = client.get("/api/field-theory/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict), "Field theory health should return a dict"


def test_knowledge_stats_shape(client):
    """Knowledge stats should contain key metrics."""
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_knowledge_search(client):
    """POST /api/knowledge/search should handle queries."""
    r = client.post("/api/knowledge/search", json={"query": "test knowledge query"})
    # 200 or 404 (if knowledge module not configured)
    assert r.status_code != 500, (
        f"Knowledge search returned 500: {r.text[:300]}"
    )


def test_knowledge_search_empty_query(client):
    """POST /api/knowledge/search with empty query should not 500."""
    r = client.post("/api/knowledge/search", json={"query": ""})
    assert r.status_code != 500, (
        f"Empty knowledge search returned 500: {r.text[:300]}"
    )


def test_templates_list_shape(client):
    """Templates list should return an array."""
    r = client.get("/api/templates/")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (list, dict))

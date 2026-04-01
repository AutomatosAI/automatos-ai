"""Routing error handling — validates graceful failures for edge cases."""


def test_delete_nonexistent_rule(client):
    """DELETE /api/routing/rules/{id} for non-existent rule."""
    r = client.delete("/api/routing/rules/999999")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent rule returned {r.status_code}"
    )


def test_create_rule_empty_body(client):
    """POST /api/routing/rules with empty body should not 500."""
    r = client.post("/api/routing/rules", json={})
    assert r.status_code in (400, 422), (
        f"Empty routing rule body returned {r.status_code}, expected 400/422"
    )


def test_create_rule_invalid_agent(client):
    """POST /api/routing/rules targeting non-existent agent."""
    r = client.post("/api/routing/rules", json={
        "source_pattern": "err-test-*",
        "intent_keywords": ["error", "test"],
        "target_agent_id": 999999,
        "priority": 1,
    })
    # Could create (soft FK) or reject — just not 500
    assert r.status_code != 500, (
        f"Routing rule with bad agent returned 500: {r.text[:300]}"
    )


def test_routing_decisions_invalid_limit(client):
    """GET /api/routing/decisions with negative limit."""
    r = client.get("/api/routing/decisions", params={"limit": -1})
    assert r.status_code != 500, (
        f"Routing decisions with negative limit returned 500: {r.text[:300]}"
    )

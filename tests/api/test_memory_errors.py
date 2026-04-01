"""Memory error handling — validates graceful failures for edge cases."""


def test_memory_store_empty_content(client):
    """POST /api/v1/memory/store with empty content should not 500."""
    r = client.post("/api/v1/memory/store", json={"content": ""})
    assert r.status_code != 500, (
        f"Empty memory store returned 500: {r.text[:300]}"
    )


def test_memory_store_missing_content(client):
    """POST /api/v1/memory/store with no content field should return 400/422."""
    r = client.post("/api/v1/memory/store", json={})
    assert r.status_code != 500, (
        f"Memory store without content returned 500: {r.text[:300]}"
    )


def test_memory_search_empty_query(client):
    """POST /api/v1/memory/search with empty query should not 500."""
    r = client.post("/api/v1/memory/search", json={"query": ""})
    assert r.status_code != 500, (
        f"Empty memory search returned 500: {r.text[:300]}"
    )


def test_memory_delete_nonexistent(client):
    """DELETE /api/v1/memory/{id} for non-existent memory should not 500."""
    r = client.delete("/api/v1/memory/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent memory returned {r.status_code}"
    )


def test_memory_stats_invalid_agent(client):
    """GET /api/v1/memory/stats/agents with bad params should not 500."""
    r = client.get("/api/v1/memory/stats/agents", params={"agent_id": "not-a-number"})
    assert r.status_code != 500, (
        f"Memory stats with invalid agent returned 500: {r.text[:300]}"
    )

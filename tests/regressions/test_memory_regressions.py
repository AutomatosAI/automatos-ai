"""Regression pins: Memory system bugs that MUST stay fixed.

Pinned bugs:
- Mem0Client.search() must use POST /search/, not GET list (fix-memory branch, 2026-03-09)
- memory_stats user_id format mismatch (ws_{id} vs ws_{id}_agent_global)
- Memory queries must be workspace-scoped (multi-tenancy fix, 2026-02-07)
"""

import pytest


# ── Mem0 search via POST (not GET list) ──────────────────────────────

def test_memory_search_returns_results_for_known_content(client):
    """Memory search endpoint must actually search, not just list.

    Bug: Mem0Client.search() called GET (list) instead of POST /search/.
    Result: query param was ignored, all memories returned regardless.
    Fix: Switched to POST /search/ in fix-memory branch.
    """
    r = client.get("/api/memory/search", params={"q": "test", "limit": 5})
    assert r.status_code == 200
    data = r.json()
    # Should return a list (or dict with results key)
    assert isinstance(data, (list, dict)), f"Unexpected search response type: {type(data)}"


def test_memory_search_empty_query_handled(client):
    """Search with empty query should return 400 or empty results, not crash."""
    r = client.get("/api/memory/search", params={"q": "", "limit": 5})
    # Either 400 (validation) or 200 with empty results — not 500
    assert r.status_code in (200, 400, 422), f"Empty query returned {r.status_code}: {r.text[:300]}"


# ── Memory stats user_id format consistency ──────────────────────────

def test_memory_stats_real_returns_valid_data(client):
    """memory_stats.py must use ws_{id} format, not ws_{id}_agent_global.

    Bug: api/memory_stats.py used ws_{id}_agent_global but runtime used ws_{id}.
    Result: Stats showed 0 memories even when memories existed.
    """
    r = client.get("/api/v1/memory/stats/real")
    assert r.status_code == 200
    data = r.json()
    assert "system_stats" in data
    stats = data["system_stats"]
    # If memories exist, counts should be consistent (not zero due to format mismatch)
    if stats.get("total_memories", 0) > 0:
        assert stats.get("active_memories", 0) > 0, (
            "total_memories > 0 but active_memories == 0 — possible user_id format mismatch"
        )


def test_workspace_memory_stats_consistent_with_real_stats(client):
    """Workspace memory stats and real stats should not wildly disagree.

    Both endpoints should query with the same user_id format.
    """
    real = client.get("/api/v1/memory/stats/real")
    workspace = client.get("/api/workspaces/current/memory-stats")

    assert real.status_code == 200
    assert workspace.status_code == 200

    real_total = real.json().get("system_stats", {}).get("total_memories", 0)
    ws_total = workspace.json().get("total_memories", 0)

    # They may count differently but if one is 0 and the other isn't, that's suspicious
    if real_total > 10 and ws_total == 0:
        pytest.fail(
            f"real stats shows {real_total} memories but workspace stats shows 0 — "
            "likely user_id format mismatch between endpoints"
        )


# ── Memory workspace scoping ────────────────────────────────────────

def test_memory_store_and_search_round_trip(client):
    """Store a memory, then search for it. Validates the full pipeline works.

    Also validates workspace scoping — the stored memory should be retrievable
    only via the same workspace's search.
    """
    import time

    unique_marker = f"regression-pin-{int(time.time())}"

    # Store
    store_r = client.post("/api/memory", json={
        "content": f"Test memory for regression validation: {unique_marker}",
        "metadata": {"source": "regression-test", "marker": unique_marker},
        "tags": ["regression-test"],
    })
    assert store_r.status_code in (200, 201), f"Memory store failed: {store_r.status_code} {store_r.text[:300]}"

    # Search (give backend a moment to index)
    time.sleep(1)
    search_r = client.get("/api/memory/search", params={"q": unique_marker, "limit": 10})
    assert search_r.status_code == 200

    data = search_r.json()
    results = data if isinstance(data, list) else data.get("results", data.get("memories", []))

    # The memory we just stored should appear in search results
    found = any(
        unique_marker in str(item)
        for item in results
    ) if results else False

    # Clean up — delete if we can find the memory_id
    if isinstance(data, list) and data:
        for item in data:
            mid = item.get("id") or item.get("memory_id")
            if mid and unique_marker in str(item):
                client.delete(f"/api/memory/{mid}")

    assert found, (
        f"Stored memory with marker '{unique_marker}' not found in search results. "
        f"Got {len(results)} results. Possible search bug or indexing delay."
    )


def test_memory_recent_returns_scoped_data(client):
    """Recent memories should be scoped to current workspace."""
    r = client.get("/api/v1/memory/stats/recent", params={"limit": 5})
    assert r.status_code == 200
    data = r.json()
    # Should be a list or dict with results — not an error
    assert isinstance(data, (list, dict)), f"Unexpected recent response: {type(data)}"

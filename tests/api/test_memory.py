"""Journey 05: Memory persistence — stats and breakdown."""


def test_memory_stats_real(client):
    r = client.get("/api/v1/memory/stats/real")
    assert r.status_code == 200
    assert "system_stats" in r.json()


def test_memory_stats_agents(client):
    r = client.get("/api/v1/memory/stats/agents")
    assert r.status_code == 200


def test_memory_stats_recent(client):
    r = client.get("/api/v1/memory/stats/recent", params={"limit": 5})
    assert r.status_code == 200


def test_workspace_memory_stats(client):
    r = client.get("/api/workspaces/current/memory-stats")
    assert r.status_code == 200
    assert "total_memories" in r.json()

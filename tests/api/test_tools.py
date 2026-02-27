"""Journey 06: Tool execution — marketplace, connected, search, refresh."""

from .helpers import pick, SEARCH_TERMS


def test_tools_marketplace(client):
    r = client.get("/api/tools/marketplace", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "apps" in data or "items" in data


def test_tools_stats(client):
    r = client.get("/api/tools/stats")
    assert r.status_code == 200
    assert "total_apps" in r.json()


def test_tools_connected(client):
    r = client.get("/api/tools/connected")
    assert r.status_code == 200


def test_tools_search(client):
    r = client.get("/api/tools/marketplace", params={"search": pick(SEARCH_TERMS), "limit": 5})
    assert r.status_code == 200


def test_tools_refresh(client):
    r = client.post("/api/tools/refresh-connections")
    assert r.status_code == 200


def test_list_credentials(client):
    r = client.get("/api/credentials/")
    assert r.status_code == 200


def test_marketplace_items(client):
    r = client.get("/api/marketplace/items")
    assert r.status_code == 200

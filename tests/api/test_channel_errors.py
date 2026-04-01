"""Channel error handling — validates graceful failures for edge cases."""


def test_get_nonexistent_channel(client):
    """GET /api/channels/{id} should handle non-existent channel."""
    r = client.get("/api/channels/999999")
    assert r.status_code in (404, 400), (
        f"Non-existent channel returned {r.status_code}"
    )


def test_create_channel_empty_body(client):
    """POST /api/channels with empty body should not 500."""
    r = client.post("/api/channels", json={})
    assert r.status_code in (400, 422), (
        f"Empty channel body returned {r.status_code}, expected 400/422"
    )


def test_create_channel_invalid_platform(client):
    """POST /api/channels with invalid platform type."""
    r = client.post("/api/channels", json={
        "platform": "INVALID_PLATFORM_XYZ",
        "config": {"name": "bad-channel"},
    })
    assert r.status_code != 500, (
        f"Invalid platform returned 500: {r.text[:300]}"
    )


def test_update_nonexistent_channel(client):
    """PUT /api/channels/{id} for non-existent channel."""
    r = client.put("/api/channels/999999", json={"config": {"name": "ghost"}})
    assert r.status_code in (404, 400), (
        f"Update non-existent channel returned {r.status_code}"
    )


def test_delete_nonexistent_channel(client):
    """DELETE /api/channels/{id} for non-existent channel."""
    r = client.delete("/api/channels/999999")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent channel returned {r.status_code}"
    )

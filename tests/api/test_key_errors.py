"""BYOK key error handling — validates graceful failures for edge cases."""


def test_add_key_empty_body(client):
    """POST /api/keys with empty body should not 500."""
    r = client.post("/api/keys", json={})
    assert r.status_code in (400, 422), (
        f"Empty key body returned {r.status_code}, expected 400/422"
    )


def test_add_key_missing_provider(client):
    """POST /api/keys without provider should fail gracefully."""
    r = client.post("/api/keys", json={
        "api_key": "sk-test-000",
        "display_name": "Missing provider",
    })
    assert r.status_code != 500, (
        f"Key without provider returned 500: {r.text[:300]}"
    )


def test_add_key_invalid_provider(client):
    """POST /api/keys with invalid provider name."""
    r = client.post("/api/keys", json={
        "provider": "FAKE_PROVIDER_XYZ",
        "api_key": "sk-test-000",
        "display_name": "Invalid provider",
    })
    # May accept unknown provider or reject — not 500
    assert r.status_code != 500, (
        f"Invalid provider returned 500: {r.text[:300]}"
    )


def test_delete_nonexistent_key(client):
    """DELETE /api/keys/{id} for non-existent key."""
    r = client.delete("/api/keys/999999")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent key returned {r.status_code}"
    )


def test_test_nonexistent_key(client):
    """POST /api/keys/{id}/test for non-existent key."""
    r = client.post("/api/keys/999999/test")
    assert r.status_code in (404, 400), (
        f"Test non-existent key returned {r.status_code}"
    )

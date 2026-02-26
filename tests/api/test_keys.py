"""Journey 12: BYOK API keys — CRUD + validation."""

import pytest


@pytest.fixture(scope="module")
def key_state():
    return {"key_id": None}


def test_list_keys(client):
    r = client.get("/api/keys")
    assert r.status_code == 200


def test_add_key(client, key_state, created_key_ids):
    r = client.post(
        "/api/keys",
        json={
            "provider": "openai",
            "api_key": "sk-test-invalid-nightly-run-key-000000",
            "display_name": "API Test Key (nightly)",
        },
    )
    assert r.status_code in (200, 201)
    data = r.json()
    assert "id" in data
    key_state["key_id"] = data["id"]
    created_key_ids.append(data["id"])


def test_test_key(client, key_state):
    if not key_state["key_id"]:
        pytest.skip("No key created")
    r = client.post(f"/api/keys/{key_state['key_id']}/test")
    assert r.status_code == 200
    # Key is invalid, so valid should be False
    data = r.json()
    assert "valid" in data


def test_platform_key_status(client):
    r = client.get("/api/keys/platform-status")
    assert r.status_code == 200
    assert "platform_keys" in r.json()


def test_byok_preferences(client):
    r = client.get("/api/workspaces/current/byok-preferences")
    assert r.status_code == 200


def test_delete_key(client, key_state, created_key_ids):
    if not key_state["key_id"]:
        pytest.skip("No key created")
    kid = key_state["key_id"]
    r = client.delete(f"/api/keys/{kid}")
    assert r.status_code in (200, 204)
    if kid in created_key_ids:
        created_key_ids.remove(kid)

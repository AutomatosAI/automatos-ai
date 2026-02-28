"""Journey 08: Channel lifecycle — CRUD + analytics."""

import pytest
from .helpers import pick, uid, CHANNEL_NAMES


@pytest.fixture(scope="module")
def channel_state():
    return {"channel_id": None}


def test_list_channels(client):
    r = client.get("/api/channels")
    assert r.status_code == 200


def test_create_channel(client, channel_state, created_channel_ids):
    name = f"{pick(CHANNEL_NAMES)}-{uid('ch')}"
    r = client.post(
        "/api/channels",
        json={"platform": "webhook", "config": {"name": name}},
    )
    assert r.status_code in (200, 201)
    data = r.json()
    assert "id" in data
    channel_state["channel_id"] = data["id"]
    created_channel_ids.append(data["id"])


def test_update_channel(client, channel_state):
    if not channel_state["channel_id"]:
        pytest.skip("No channel created")
    r = client.put(
        f"/api/channels/{channel_state['channel_id']}",
        json={"config": {"name": "updated-test-channel"}},
    )
    assert r.status_code == 200


def test_channel_analytics(client):
    r = client.get("/api/channels/analytics")
    assert r.status_code == 200


def test_delete_channel(client, channel_state, created_channel_ids):
    if not channel_state["channel_id"]:
        pytest.skip("No channel created")
    cid = channel_state["channel_id"]
    r = client.delete(f"/api/channels/{cid}")
    assert r.status_code == 200
    if cid in created_channel_ids:
        created_channel_ids.remove(cid)

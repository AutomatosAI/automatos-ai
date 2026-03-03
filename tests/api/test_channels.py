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


def test_channel_analytics_source_query(client):
    """Channel analytics must not silently swallow SQL errors.

    Bug: orchestrator/api/channels.py line 326 queries
        SELECT source_channel FROM routing_decisions
    but routing_decisions only has a 'source' column (line 96 of
    orchestrator/core/models/routing.py). The column 'source_channel'
    exists on routing_rules (line 115), not routing_decisions.

    The exception is caught at line 356 and returns empty data, masking
    the bug. This test verifies the response shape is valid and that
    today_by_source is a dict (not None or missing).

    Fix: channels.py lines 326+329+346 — change source_channel to source.
    """
    r = client.get("/api/channels/analytics")
    assert r.status_code == 200
    data = r.json()
    assert "today_by_source" in data, (
        "Response missing 'today_by_source' key. "
        "Bug: channels.py:326 queries non-existent column 'source_channel' "
        "on routing_decisions table. Should be 'source'."
    )
    assert isinstance(data["today_by_source"], dict), (
        f"today_by_source should be dict, got {type(data['today_by_source']).__name__}. "
        f"Bug: SQL error caught silently at channels.py:356."
    )
    assert "channels" in data, "Response missing 'channels' key"


def test_delete_channel(client, channel_state, created_channel_ids):
    if not channel_state["channel_id"]:
        pytest.skip("No channel created")
    cid = channel_state["channel_id"]
    r = client.delete(f"/api/channels/{cid}")
    assert r.status_code == 200
    if cid in created_channel_ids:
        created_channel_ids.remove(cid)

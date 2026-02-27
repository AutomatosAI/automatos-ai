"""Journey 11: Webhook triggers — verification, send, orchestrator settings."""

import re

import pytest
from .helpers import pick, WEBHOOK_PAYLOADS


@pytest.fixture(scope="module")
def webhook_key(client):
    """Extract webhook key from workspace config."""
    r = client.get("/api/workspaces/current")
    if r.status_code != 200:
        return None
    ws = r.json()
    url = ws.get("webhook_url", "") or ""
    match = re.search(r"/ws/([^/]+)", url)
    if match:
        return match.group(1)
    # Fallback to slug or id
    return ws.get("slug") or ws.get("id")


def test_webhook_verify(client, webhook_key):
    if not webhook_key:
        pytest.skip("No webhook key available")
    r = client.get(f"/api/webhooks/ws/{webhook_key}")
    assert r.status_code in (200, 400)


def test_webhook_send(client, webhook_key):
    if not webhook_key:
        pytest.skip("No webhook key available")
    r = client.post(f"/api/webhooks/ws/{webhook_key}", json=pick(WEBHOOK_PAYLOADS))
    assert r.status_code in (200, 202)


def test_orchestrator_settings(client):
    r = client.get("/api/workspaces/current/orchestrator")
    assert r.status_code == 200


def test_update_orchestrator(client):
    """PUT orchestrator — preserves existing personality_mode."""
    r = client.get("/api/workspaces/current/orchestrator")
    if r.status_code != 200:
        pytest.skip("Cannot read orchestrator settings")
    current = r.json()
    mode = current.get("personality_mode", "professional")
    r2 = client.put(
        "/api/workspaces/current/orchestrator",
        json={"personality_mode": mode},
    )
    assert r2.status_code == 200

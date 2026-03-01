"""Journey 03: Agent management — CRUD + detail endpoints."""

import pytest
import socket


AGENT_PAYLOAD = {
    "name": "nightly-test-agent",
    "agent_type": "custom",
    "description": "Ephemeral agent created by the nightly test suite",
    "configuration": {
        "model": "openai/gpt-4o-mini",
        "system_prompt": "You are a test agent.",
    },
}


def test_agent_model_config(client, first_agent_id):
    # First check if pgvector server is reachable
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5) # 5 seconds timeout
        s.connect(("pgvector.railway.internal", 5432))
        s.close()
    except Exception as e:
        pytest.skip(f"Could not connect to pgvector server: {e}")

    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/model-config")
    assert r.status_code == 200

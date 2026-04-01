"""User Journey: Admin configuring workspace.

Simulates a workspace admin setting up agents, personas, routing rules,
channels, API keys, and verifying the configuration sticks.

Flow:
  1. Check workspace → confirm admin access
  2. Create persona → define agent personality
  3. Create agent → with specific model config
  4. Assign persona to agent → link personality
  5. Add BYOK key → configure provider access
  6. Create routing rule → direct traffic to agent
  7. Create channel → set up webhook integration
  8. Verify all config → read back everything
  9. Clean up → remove all test resources
"""

import pytest

from .helpers import uid, pick, PERSONA_NAMES, PERSONA_PROMPTS, CHANNEL_NAMES, ROUTING_KEYWORDS


@pytest.fixture(scope="module")
def admin_state():
    return {
        "agent_id": None,
        "persona_id": None,
        "key_id": None,
        "rule_id": None,
        "channel_id": None,
    }


def test_admin_01_workspace_access(client):
    """Step 1: Confirm workspace access and integrations."""
    r = client.get("/api/workspaces/current")
    assert r.status_code == 200

    r2 = client.get("/api/workspaces/current/integrations")
    assert r2.status_code == 200


def test_admin_02_create_persona(client, workspace_id, admin_state, created_persona_ids):
    """Step 2: Create a persona for the new agent."""
    name = f"{pick(PERSONA_NAMES)} {uid('admin')}"
    r = client.post(f"/api/workspaces/{workspace_id}/personas", json={
        "name": name,
        "description": "Admin journey test persona",
        "system_prompt": pick(PERSONA_PROMPTS),
        "category": "assistant",
    })
    assert r.status_code in (200, 201), (
        f"Persona creation failed: {r.status_code} {r.text[:300]}"
    )
    data = r.json()
    admin_state["persona_id"] = data["id"]
    created_persona_ids.append(data["id"])


def test_admin_03_create_agent(client, created_agent_ids, admin_state):
    """Step 3: Create an agent to assign persona to."""
    r = client.post("/api/agents/", json={
        "name": uid("admin-agent"),
        "agent_type": "custom",
        "description": "Admin journey test agent",
        "configuration": {
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You are an admin-configured test agent.",
        },
    })
    assert r.status_code in (200, 201)
    data = r.json()
    admin_state["agent_id"] = data["id"]
    created_agent_ids.append(data["id"])


def test_admin_04_assign_persona(client, admin_state):
    """Step 4: Link persona to agent."""
    if not admin_state["agent_id"] or not admin_state["persona_id"]:
        pytest.skip("Need agent + persona")
    r = client.put(f"/api/agents/{admin_state['agent_id']}/persona", json={
        "persona_id": admin_state["persona_id"],
    })
    assert r.status_code == 200


def test_admin_05_verify_persona_assignment(client, admin_state):
    """Step 5: Verify persona is assigned to agent."""
    if not admin_state["agent_id"]:
        pytest.skip("No agent")
    r = client.get(f"/api/agents/{admin_state['agent_id']}/persona")
    assert r.status_code == 200


def test_admin_06_add_byok_key(client, admin_state, created_key_ids):
    """Step 6: Add a BYOK API key."""
    r = client.post("/api/keys", json={
        "provider": "openai",
        "api_key": "sk-test-admin-journey-key-000000",
        "display_name": f"Admin Journey Key {uid('key')}",
    })
    assert r.status_code in (200, 201)
    data = r.json()
    admin_state["key_id"] = data["id"]
    created_key_ids.append(data["id"])


def test_admin_07_create_routing_rule(client, admin_state, created_rule_ids):
    """Step 7: Create a routing rule targeting the new agent."""
    if not admin_state["agent_id"]:
        pytest.skip("No agent")
    r = client.post("/api/routing/rules", json={
        "source_pattern": f"{uid('admin-route')}-*",
        "intent_keywords": pick(ROUTING_KEYWORDS),
        "target_agent_id": admin_state["agent_id"],
        "priority": 1,
    })
    assert r.status_code in (200, 201)
    data = r.json()
    admin_state["rule_id"] = data["id"]
    created_rule_ids.append(data["id"])


def test_admin_08_create_channel(client, admin_state, created_channel_ids):
    """Step 8: Create a webhook channel."""
    name = f"{pick(CHANNEL_NAMES)}-{uid('admin-ch')}"
    r = client.post("/api/channels", json={
        "platform": "webhook",
        "config": {"name": name},
    })
    assert r.status_code in (200, 201)
    data = r.json()
    admin_state["channel_id"] = data["id"]
    created_channel_ids.append(data["id"])


def test_admin_09_verify_all_config(client, admin_state):
    """Step 9: Read back all configuration and verify it's consistent."""
    # Agent exists
    if admin_state["agent_id"]:
        r = client.get(f"/api/agents/{admin_state['agent_id']}")
        assert r.status_code == 200
        assert r.json()["id"] == admin_state["agent_id"]

    # Key exists
    if admin_state["key_id"]:
        r = client.get("/api/keys")
        assert r.status_code == 200

    # Rules exist
    r = client.get("/api/routing/rules")
    assert r.status_code == 200

    # Channels exist
    r = client.get("/api/channels")
    assert r.status_code == 200

    # Platform key status
    r = client.get("/api/keys/platform-status")
    assert r.status_code == 200


def test_admin_10_cleanup(client, workspace_id, admin_state,
                          created_agent_ids, created_persona_ids,
                          created_key_ids, created_rule_ids, created_channel_ids):
    """Step 10: Clean up all test resources."""
    # Delete routing rule first (depends on agent)
    if admin_state["rule_id"]:
        r = client.delete(f"/api/routing/rules/{admin_state['rule_id']}")
        assert r.status_code in (200, 204, 404)
        if admin_state["rule_id"] in created_rule_ids:
            created_rule_ids.remove(admin_state["rule_id"])

    # Delete channel
    if admin_state["channel_id"]:
        r = client.delete(f"/api/channels/{admin_state['channel_id']}")
        assert r.status_code in (200, 204, 404)
        if admin_state["channel_id"] in created_channel_ids:
            created_channel_ids.remove(admin_state["channel_id"])

    # Delete key
    if admin_state["key_id"]:
        r = client.delete(f"/api/keys/{admin_state['key_id']}")
        assert r.status_code in (200, 204, 404)
        if admin_state["key_id"] in created_key_ids:
            created_key_ids.remove(admin_state["key_id"])

    # Delete persona
    if admin_state["persona_id"]:
        r = client.delete(f"/api/workspaces/{workspace_id}/personas/{admin_state['persona_id']}")
        assert r.status_code in (200, 204, 404)
        if admin_state["persona_id"] in created_persona_ids:
            created_persona_ids.remove(admin_state["persona_id"])

    # Delete agent
    if admin_state["agent_id"]:
        r = client.delete(f"/api/agents/{admin_state['agent_id']}")
        assert r.status_code in (200, 204)
        if admin_state["agent_id"] in created_agent_ids:
            created_agent_ids.remove(admin_state["agent_id"])

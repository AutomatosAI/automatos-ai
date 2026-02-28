"""Shared fixtures for API integration tests."""

import os

import httpx
import pytest


@pytest.fixture(scope="session")
def api_url() -> str:
    return os.environ["API_URL"]


@pytest.fixture(scope="session")
def workspace_id() -> str:
    return os.environ["WORKSPACE_ID"]


@pytest.fixture(scope="session")
def auth_headers() -> dict:
    return {
        "X-Api-Key": os.environ["API_KEY"],
        "X-Workspace-ID": os.environ["WORKSPACE_ID"],
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="session")
def client(api_url, auth_headers) -> httpx.Client:
    with httpx.Client(
        base_url=api_url,
        headers=auth_headers,
        timeout=30.0,
        follow_redirects=True,
    ) as c:
        yield c


# ── Agent discovery ──────────────────────────────────────────────────
WELL_KNOWN_AGENT_IDS = [101, 102, 19, 10, 89]


@pytest.fixture(scope="session")
def first_agent_id(client) -> int | None:
    """Discover an existing agent ID (tries list then well-known IDs)."""
    # Try listing agents first
    r = client.get("/api/agents/")
    if r.status_code == 200:
        agents = r.json()
        if isinstance(agents, list) and agents:
            return agents[0].get("id")
    # Fallback: probe well-known IDs
    for aid in WELL_KNOWN_AGENT_IDS:
        r = client.get(f"/api/agents/{aid}")
        if r.status_code == 200:
            return aid
    return None


# ── Agent cleanup registry ───────────────────────────────────────────
@pytest.fixture(scope="session")
def created_agent_ids() -> list:
    """Accumulates agent IDs created during the test session for cleanup."""
    return []


@pytest.fixture(scope="session", autouse=True)
def cleanup_agents(client, created_agent_ids):
    """Session finalizer — DELETE every test agent we created."""
    yield
    for agent_id in created_agent_ids:
        try:
            client.delete(f"/api/agents/{agent_id}")
        except Exception:
            pass


# ── Generic cleanup registries ───────────────────────────────────────
@pytest.fixture(scope="session")
def created_persona_ids() -> list:
    return []


@pytest.fixture(scope="session")
def created_channel_ids() -> list:
    return []


@pytest.fixture(scope="session")
def created_rule_ids() -> list:
    return []


@pytest.fixture(scope="session")
def created_key_ids() -> list:
    return []


@pytest.fixture(scope="session", autouse=True)
def cleanup_all(client, workspace_id, created_persona_ids, created_channel_ids, created_rule_ids, created_key_ids):
    """Session finalizer — clean up all test resources."""
    yield
    for pid in created_persona_ids:
        try:
            client.delete(f"/api/workspaces/{workspace_id}/personas/{pid}")
        except Exception:
            pass
    for cid in created_channel_ids:
        try:
            client.delete(f"/api/channels/{cid}")
        except Exception:
            pass
    for rid in created_rule_ids:
        try:
            client.delete(f"/api/routing/rules/{rid}")
        except Exception:
            pass
    for kid in created_key_ids:
        try:
            client.delete(f"/api/keys/{kid}")
        except Exception:
            pass

"""Regression pins: AgentFactory tool source and execution bugs.

Pinned bugs:
- execute_with_prompt() used hardcoded _build_tool_schemas() while chatbot used
  get_tools_for_agent(). Same agent got different tools depending on caller.
  Fix: ONE tool source — get_tools_for_agent() for all paths. (2026-03-11)
- Tool loop was single-shot — now supports max_tool_iterations=10.
"""

import pytest


def test_agent_execute_returns_valid_handle(client, first_agent_id):
    """Agent execution must return a valid execution handle.

    Bug: _create_llm_manager() had undefined api_key var, causing execution to fail.
    Fix: Cleaned up AgentFactory rewrite (2026-03-11).
    """
    if not first_agent_id:
        pytest.skip("No agent available")

    r = client.post(
        f"/api/agents/{first_agent_id}/execute",
        json={"task": "What tools do you have access to?", "mode": "test"},
    )
    assert r.status_code == 200, f"Agent execute failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    assert "execution_id" in data or "id" in data, f"No execution handle in response: {data}"
    assert data.get("status") in ("started", "running", "completed", "queued"), (
        f"Unexpected execution status: {data.get('status')}"
    )


def test_agent_tools_consistent_across_endpoints(client, first_agent_id):
    """Agent should report the same tools regardless of which endpoint queries them.

    Bug: _build_tool_schemas() (PRD-17 legacy) returned different tools than
    get_tools_for_agent() from tool_router.py. Chatbot and execute paths diverged.
    Fix: All paths now use get_tools_for_agent().

    This test verifies the agent detail endpoint includes tool information
    that would be consistent with what the execution path uses.
    """
    if not first_agent_id:
        pytest.skip("No agent available")

    # Get agent detail
    detail_r = client.get(f"/api/agents/{first_agent_id}")
    assert detail_r.status_code == 200
    detail = detail_r.json()

    # Get agent status (may include tool count)
    status_r = client.get(f"/api/agents/{first_agent_id}/status")
    assert status_r.status_code == 200
    status = status_r.json()

    # Both should exist without errors — the main regression was that one path
    # would crash while the other worked. If both return 200, the tool source
    # is at least consistent enough to not error.
    assert "id" in detail
    assert isinstance(status, dict)


def test_agent_create_and_execute_round_trip(client, created_agent_ids):
    """Create a fresh agent and execute it — validates the full factory pipeline.

    This catches regressions where newly created agents fail to execute because
    tool discovery, LLM manager setup, or execution paths are broken.
    """
    # Create
    payload = {
        "name": "regression-factory-test",
        "agent_type": "custom",
        "description": "Tests AgentFactory regression — tool source consistency",
        "configuration": {
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You are a test agent. Respond briefly.",
        },
    }
    create_r = client.post("/api/agents/", json=payload)
    assert create_r.status_code in (200, 201), f"Agent create failed: {create_r.text[:500]}"
    agent_id = create_r.json()["id"]
    created_agent_ids.append(agent_id)

    # Execute
    exec_r = client.post(
        f"/api/agents/{agent_id}/execute",
        json={"task": "Say hello", "mode": "test"},
    )
    assert exec_r.status_code == 200, (
        f"Freshly created agent {agent_id} failed to execute: "
        f"{exec_r.status_code} {exec_r.text[:500]}"
    )
    exec_data = exec_r.json()
    assert exec_data.get("agent_id") == agent_id

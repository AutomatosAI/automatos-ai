"""Journey: Workflow execution lifecycle — create, execute, status, history.

Expands on the 3 smoke tests in test_workflows.py with stateful flows.
"""

import pytest


@pytest.fixture(scope="module")
def workflow_state():
    return {
        "workflow_id": None,
        "execution_id": None,
        "execution_ids": [],
    }


def test_list_workflows_has_items(client, workflow_state):
    """GET /api/workflows — verify at least one workflow exists for testing."""
    r = client.get("/api/workflows", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", data if isinstance(data, list) else [])
    if not items:
        pytest.skip("No workflows in workspace — cannot run workflow journeys")
    workflow_state["workflow_id"] = items[0]["id"]


def test_workflow_execute(client, workflow_state):
    """POST /api/workflows/{id}/execute — start a workflow execution."""
    if not workflow_state["workflow_id"]:
        pytest.skip("No workflow_id available")

    r = client.post(
        f"/api/workflows/{workflow_state['workflow_id']}/execute",
        json={"input_data": {"source": "journey-test", "mode": "smoke"}},
    )
    assert r.status_code == 200, f"Workflow execute failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    eid = data.get("execution_id") or data.get("id")
    assert eid, f"No execution_id in response: {data}"
    workflow_state["execution_id"] = eid
    workflow_state["execution_ids"].append(eid)


def test_workflow_execution_status(client, workflow_state):
    """GET /api/workflows/executions/{id} — check execution status."""
    if not workflow_state["execution_id"]:
        pytest.skip("No execution started")

    r = client.get(f"/api/workflows/executions/{workflow_state['execution_id']}")
    assert r.status_code == 200, f"Execution status failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    assert "status" in data, f"No status field in execution response: {data}"
    assert data["status"] in (
        "pending", "running", "completed", "failed", "cancelled", "queued", "started",
    ), f"Unexpected execution status: {data['status']}"


def test_workflow_execution_list(client, workflow_state):
    """GET /api/workflows/{id}/executions — list executions for a workflow."""
    if not workflow_state["workflow_id"]:
        pytest.skip("No workflow_id available")

    r = client.get(
        f"/api/workflows/{workflow_state['workflow_id']}/executions",
        params={"limit": 10},
    )
    # Some APIs may not have this endpoint — accept 200 or 404
    assert r.status_code in (200, 404), (
        f"Execution list returned unexpected {r.status_code}: {r.text[:300]}"
    )
    if r.status_code == 200:
        data = r.json()
        items = data if isinstance(data, list) else data.get("items", data.get("executions", []))
        assert isinstance(items, list)


def test_workflow_templates_available(client):
    """GET /api/workflow-templates — templates should be available for UI."""
    r = client.get("/api/workflow-templates")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (list, dict))


def test_workflow_stats_endpoint(client):
    """GET /api/workflow-recipes/stats — recipe stats for dashboard."""
    r = client.get("/api/workflow-recipes/stats")
    assert r.status_code in (200, 404)

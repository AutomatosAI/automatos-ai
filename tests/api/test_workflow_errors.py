"""Workflow error handling — validates graceful failures for edge cases."""


def test_get_nonexistent_workflow(client):
    """GET /api/workflows/{id} for non-existent workflow."""
    r = client.get("/api/workflows/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent workflow returned {r.status_code}"
    )


def test_execute_workflow_empty_body(client):
    """POST /api/workflows/execute with empty body should not 500."""
    r = client.post("/api/workflows/execute", json={})
    assert r.status_code in (400, 422), (
        f"Empty workflow execute returned {r.status_code}, expected 400/422"
    )


def test_workflow_execution_status_nonexistent(client):
    """GET /api/workflows/executions/{id} for non-existent execution."""
    r = client.get("/api/workflows/executions/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent execution returned {r.status_code}"
    )


def test_get_nonexistent_recipe(client):
    """GET /api/workflow-recipes/{id} for non-existent recipe."""
    r = client.get("/api/workflow-recipes/nonexistent-recipe-id-000")
    assert r.status_code in (404, 400), (
        f"Non-existent recipe returned {r.status_code}"
    )


def test_create_recipe_empty_body(client):
    """POST /api/workflow-recipes with empty body should not 500."""
    r = client.post("/api/workflow-recipes", json={})
    assert r.status_code in (400, 422), (
        f"Empty recipe body returned {r.status_code}, expected 400/422"
    )


def test_create_recipe_missing_steps(client):
    """POST /api/workflow-recipes without steps should fail gracefully."""
    r = client.post("/api/workflow-recipes", json={
        "name": "No steps recipe",
        "template_definition": {"version": "1.0"},
    })
    assert r.status_code != 500, (
        f"Recipe without steps returned 500: {r.text[:300]}"
    )

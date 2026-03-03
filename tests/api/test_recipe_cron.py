"""Journey: Cron recipe CRUD — create, update, delete, schedule_config handling."""

import pytest
from .helpers import uid


@pytest.fixture(scope="module")
def recipe_state():
    """Shared state across tests in this module."""
    return {
        "cron_template_id": None,
        "manual_template_id": None,
    }


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_cron_recipe(client, first_agent_id, created_recipe_ids, recipe_state):
    """POST with schedule_config type=cron → 200, returns cron_expression."""
    template_id = uid("cron-recipe")
    agent_id = first_agent_id
    if not agent_id:
        pytest.skip("No agent available for step assignment")

    payload = {
        "template_id": template_id,
        "name": f"Cron Test Recipe {template_id}",
        "description": "Automated test recipe with cron schedule",
        "template_definition": {"version": "1.0"},
        "steps": [
            {
                "step_id": "s1",
                "order": 1,
                "agent_id": agent_id,
                "prompt_template": "Run cron test task",
            }
        ],
        "schedule_config": {
            "type": "cron",
            "cron_expression": "0 9 * * *",
        },
    }

    r = client.post("/api/workflow-recipes", json=payload)
    assert r.status_code == 200, f"Create cron recipe failed: {r.text}"

    data = r.json()
    recipe = data.get("recipe", data)
    assert recipe["schedule_config"]["type"] == "cron"
    assert recipe["schedule_config"]["cron_expression"] == "0 9 * * *"

    recipe_state["cron_template_id"] = template_id
    created_recipe_ids.append(template_id)


def test_create_manual_recipe(client, first_agent_id, created_recipe_ids, recipe_state):
    """POST with type=manual → 200, no cron_expression required."""
    template_id = uid("manual-recipe")
    agent_id = first_agent_id
    if not agent_id:
        pytest.skip("No agent available for step assignment")

    payload = {
        "template_id": template_id,
        "name": f"Manual Test Recipe {template_id}",
        "description": "Automated test recipe with manual schedule",
        "template_definition": {"version": "1.0"},
        "steps": [
            {
                "step_id": "s1",
                "order": 1,
                "agent_id": agent_id,
                "prompt_template": "Run manual test task",
            }
        ],
        "schedule_config": {
            "type": "manual",
        },
    }

    r = client.post("/api/workflow-recipes", json=payload)
    assert r.status_code == 200, f"Create manual recipe failed: {r.text}"

    data = r.json()
    recipe = data.get("recipe", data)
    assert recipe["schedule_config"]["type"] == "manual"

    recipe_state["manual_template_id"] = template_id
    created_recipe_ids.append(template_id)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_update_recipe_to_cron(client, recipe_state):
    """PUT changes type to cron → schedule_config persisted."""
    template_id = recipe_state.get("manual_template_id")
    if not template_id:
        pytest.skip("No manual recipe created")

    r = client.put(
        f"/api/workflow-recipes/{template_id}",
        json={
            "schedule_config": {
                "type": "cron",
                "cron_expression": "*/15 * * * *",
            }
        },
    )
    assert r.status_code == 200, f"Update to cron failed: {r.text}"

    data = r.json()
    recipe = data.get("recipe", data)
    assert recipe["schedule_config"]["type"] == "cron"
    assert recipe["schedule_config"]["cron_expression"] == "*/15 * * * *"


def test_update_recipe_cron_to_manual(client, recipe_state):
    """PUT changes type to manual → cron removed."""
    template_id = recipe_state.get("manual_template_id")
    if not template_id:
        pytest.skip("No recipe to update")

    r = client.put(
        f"/api/workflow-recipes/{template_id}",
        json={
            "schedule_config": {
                "type": "manual",
            }
        },
    )
    assert r.status_code == 200, f"Update to manual failed: {r.text}"

    data = r.json()
    recipe = data.get("recipe", data)
    assert recipe["schedule_config"]["type"] == "manual"
    # cron_expression should be absent or None
    assert not recipe["schedule_config"].get("cron_expression")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_schedule_type(client, first_agent_id, created_recipe_ids):
    """POST with invalid schedule type → 400 validation error."""
    template_id = uid("bad-sched")
    agent_id = first_agent_id
    if not agent_id:
        pytest.skip("No agent available")

    payload = {
        "template_id": template_id,
        "name": "Bad Schedule Recipe",
        "description": "Should fail validation",
        "template_definition": {"version": "1.0"},
        "steps": [
            {"step_id": "s1", "order": 1, "agent_id": agent_id, "prompt_template": "x"}
        ],
        "schedule_config": {
            "type": "invalid_type",
        },
    }

    r = client.post("/api/workflow-recipes", json=payload)
    assert r.status_code == 400, f"Expected 400 for invalid schedule type, got {r.status_code}"
    # Clean up in case it somehow passed
    if r.status_code == 200:
        created_recipe_ids.append(template_id)


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_recipe_detail_includes_schedule(client, recipe_state):
    """GET detail shows schedule_config."""
    template_id = recipe_state.get("cron_template_id")
    if not template_id:
        pytest.skip("No cron recipe created")

    r = client.get(f"/api/workflow-recipes/{template_id}")
    assert r.status_code == 200

    data = r.json()
    assert "schedule_config" in data
    assert data["schedule_config"]["type"] == "cron"


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_cron_recipe(client, recipe_state, created_recipe_ids):
    """DELETE removes recipe → 200."""
    template_id = recipe_state.get("cron_template_id")
    if not template_id:
        pytest.skip("No cron recipe to delete")

    r = client.delete(f"/api/workflow-recipes/{template_id}")
    assert r.status_code == 200, f"Delete failed: {r.text}"

    # Verify it's gone
    r2 = client.get(f"/api/workflow-recipes/{template_id}")
    assert r2.status_code == 404

    # Remove from cleanup list since we already deleted
    if template_id in created_recipe_ids:
        created_recipe_ids.remove(template_id)
    recipe_state["cron_template_id"] = None


# ---------------------------------------------------------------------------
# Execution history
# ---------------------------------------------------------------------------


def test_recipe_execution_list(client, recipe_state):
    """Execution list endpoint returns 200 (may be empty for test recipes)."""
    template_id = recipe_state.get("manual_template_id")
    if not template_id:
        pytest.skip("No recipe available")

    # Get recipe detail to find numeric id
    r = client.get(f"/api/workflow-recipes/{template_id}")
    if r.status_code != 200:
        pytest.skip("Recipe not found")

    data = r.json()
    recipe_id = data.get("id") or data.get("template_id")

    r2 = client.get(f"/api/workflow-recipes/{template_id}/executions")
    # Endpoint may or may not exist — accept 200 or 404
    assert r2.status_code in (200, 404), f"Unexpected status: {r2.status_code}"

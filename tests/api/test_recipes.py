"""Journey 13: Workflow recipes — list, categories, featured, search, detail, create."""

import pytest
from .helpers import pick, uid, RECIPE_SEARCH_TERMS


@pytest.fixture(scope="module")
def recipe_state():
    return {"first_recipe_id": None}


def test_list_recipes(client, recipe_state):
    r = client.get("/api/workflow-recipes", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", data if isinstance(data, list) else [])
    if items:
        recipe_state["first_recipe_id"] = items[0].get("template_id") or items[0].get("id")


def test_recipe_categories(client):
    r = client.get("/api/workflow-recipes/categories/list")
    assert r.status_code == 200


def test_recipe_featured(client):
    r = client.get("/api/workflow-recipes/featured/list", params={"limit": 5})
    assert r.status_code == 200


def test_recipe_search(client):
    r = client.get("/api/workflow-recipes", params={"search": pick(RECIPE_SEARCH_TERMS), "limit": 5})
    assert r.status_code == 200


def test_recipe_detail(client, recipe_state):
    if not recipe_state["first_recipe_id"]:
        pytest.skip("No recipes available")
    r = client.get(f"/api/workflow-recipes/{recipe_state['first_recipe_id']}")
    assert r.status_code == 200


def test_create_recipe_with_null_created_by(client, first_agent_id, created_recipe_ids):
    """Creating a recipe where frontend sends created_by: null should not 500.

    Bug: orchestrator/api/workflow_recipes.py line 452 uses
        recipe_data.get('created_by', default)
    which returns None when the key exists with value null, violating the
    NOT NULL constraint on workflow_recipes.created_by.

    Fix: use the `or` pattern already used at lines 1406 and 1528:
        created_by=recipe_data.get('created_by') or (ctx.user.email if ctx.user and ctx.user.email else "anonymous")
    """
    tid = uid("recipe-null-cb")
    agent_id = first_agent_id or 1
    r = client.post(
        "/api/workflow-recipes",
        json={
            "template_id": tid,
            "name": f"Null CreatedBy Test {tid}",
            "description": "Regression test: created_by=null must not crash",
            "template_definition": {"version": "1.0"},
            "steps": [
                {
                    "step_id": "step_1",
                    "order": 1,
                    "agent_id": agent_id,
                    "prompt_template": "Test prompt",
                }
            ],
            "created_by": None,  # <-- explicit null triggers the bug
        },
    )
    assert r.status_code in (200, 201), (
        f"POST /api/workflow-recipes returned {r.status_code}: {r.text[:500]}. "
        f"Bug: workflow_recipes.py:452 .get('created_by', default) passes None "
        f"to the NOT NULL created_by column when JSON explicitly sends null."
    )
    data = r.json()
    rid = data.get("template_id") or data.get("id")
    if rid:
        created_recipe_ids.append(rid)

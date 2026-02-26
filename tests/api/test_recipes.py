"""Journey 13: Workflow recipes — list, categories, featured, search, detail."""

import pytest
from .helpers import pick, RECIPE_SEARCH_TERMS


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

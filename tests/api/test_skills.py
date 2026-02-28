"""Journey 07: Skill management — list, sources, search, recommend."""

import pytest
from .helpers import pick, SKILL_TASKS, SEARCH_TERMS


def test_list_skills(client):
    r = client.get("/api/v1/skills", params={"limit": 10})
    assert r.status_code == 200


def test_skill_sources(client):
    r = client.get("/api/v1/skills/sources")
    assert r.status_code == 200


def test_agent_skills(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/v1/skills/agents/{first_agent_id}/skills")
    assert r.status_code == 200


def test_skill_recommend(client):
    task = pick(SKILL_TASKS)
    r = client.post(
        "/api/v1/skills/recommend",
        json={
            "task_description": task["description"],
            "task_type": task["type"],
            "limit": 3,
        },
    )
    assert r.status_code == 200


def test_skill_search(client):
    r = client.get("/api/v1/skills", params={"search": pick(SEARCH_TERMS), "limit": 5})
    assert r.status_code == 200

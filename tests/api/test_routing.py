"""Journey 09: Routing rules — decisions, rules CRUD, cache stats."""

import pytest
from .helpers import pick, uid, ROUTING_KEYWORDS


@pytest.fixture(scope="module")
def routing_state():
    return {"rule_id": None}


def test_routing_decisions(client):
    r = client.get("/api/routing/decisions", params={"limit": 10})
    assert r.status_code == 200


def test_routing_rules_list(client):
    r = client.get("/api/routing/rules")
    assert r.status_code == 200


def test_create_routing_rule(client, first_agent_id, routing_state, created_rule_ids):
    if not first_agent_id:
        pytest.skip("No agent available for routing target")
    r = client.post(
        "/api/routing/rules",
        json={
            "source_pattern": f"{uid('route')}-*",
            "intent_keywords": pick(ROUTING_KEYWORDS),
            "target_agent_id": first_agent_id,
            "priority": 1,
        },
    )
    assert r.status_code in (200, 201)
    data = r.json()
    assert "id" in data
    routing_state["rule_id"] = data["id"]
    created_rule_ids.append(data["id"])


def test_routing_cache_stats(client):
    r = client.get("/api/routing/cache/stats")
    assert r.status_code == 200


def test_delete_routing_rule(client, routing_state, created_rule_ids):
    if not routing_state["rule_id"]:
        pytest.skip("No rule created")
    rid = routing_state["rule_id"]
    r = client.delete(f"/api/routing/rules/{rid}")
    assert r.status_code == 200
    if rid in created_rule_ids:
        created_rule_ids.remove(rid)

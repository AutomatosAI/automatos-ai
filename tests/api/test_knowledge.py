"""Knowledge, multi-agent, field-theory, and templates smoke tests."""


def test_multi_agent_health(client):
    r = client.get("/api/multi-agent/health")
    assert r.status_code == 200


def test_field_theory_health(client):
    r = client.get("/api/field-theory/health")
    assert r.status_code == 200


def test_knowledge_stats(client):
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200


def test_templates_list(client):
    r = client.get("/api/templates/")
    assert r.status_code == 200

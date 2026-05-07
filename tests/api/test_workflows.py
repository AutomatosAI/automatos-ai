"""Workflow endpoints smoke tests."""


def test_list_workflows(client):
    r = client.get("/api/workflows")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data or isinstance(data, list)


def test_workflow_templates(client):
    r = client.get("/api/workflow-templates")
    assert r.status_code == 200


def test_workflow_stats(client):
    r = client.get("/api/workflow-recipes/stats/dashboard")
    assert r.status_code in (200, 404)

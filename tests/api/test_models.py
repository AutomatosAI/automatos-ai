"""Journey 15: Model management — list, providers, recommend, cost."""


def test_list_models(client):
    r = client.get("/api/models/")
    assert r.status_code == 200


def test_model_providers(client):
    r = client.get("/api/models/providers/")
    assert r.status_code == 200


def test_model_recommend(client):
    r = client.post(
        "/api/models/recommend",
        json={"max_cost": 1.0, "min_context": 4000},
    )
    assert r.status_code == 200
    assert "model_id" in r.json()


def test_model_estimate_cost(client):
    r = client.post(
        "/api/models/estimate-cost",
        json={"model_id": "gpt-4", "input_tokens": 1500, "output_tokens": 800},
    )
    assert r.status_code == 200
    assert "total_cost" in r.json()


def test_model_stats(client):
    r = client.get("/api/models/stats/")
    assert r.status_code == 200
    assert "total_models" in r.json()

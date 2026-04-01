"""Model endpoint error handling — validates graceful failures for edge cases."""


def test_recommend_model_empty_body(client):
    """POST /api/models/recommend with empty body should not 500."""
    r = client.post("/api/models/recommend", json={})
    assert r.status_code != 500, (
        f"Empty recommend body returned 500: {r.text[:300]}"
    )


def test_estimate_cost_missing_model(client):
    """POST /api/models/estimate-cost without model_id should fail gracefully."""
    r = client.post("/api/models/estimate-cost", json={
        "input_tokens": 1000,
        "output_tokens": 500,
    })
    assert r.status_code != 500, (
        f"Cost estimate without model returned 500: {r.text[:300]}"
    )


def test_estimate_cost_invalid_model(client):
    """POST /api/models/estimate-cost with non-existent model."""
    r = client.post("/api/models/estimate-cost", json={
        "model_id": "fake-model/does-not-exist",
        "input_tokens": 1000,
        "output_tokens": 500,
    })
    assert r.status_code != 500, (
        f"Cost estimate with fake model returned 500: {r.text[:300]}"
    )


def test_estimate_cost_negative_tokens(client):
    """POST /api/models/estimate-cost with negative tokens."""
    r = client.post("/api/models/estimate-cost", json={
        "model_id": "gpt-4",
        "input_tokens": -100,
        "output_tokens": -50,
    })
    assert r.status_code != 500, (
        f"Cost estimate with negative tokens returned 500: {r.text[:300]}"
    )

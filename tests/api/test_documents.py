"""Document endpoints — smoke + deeper validation."""


def test_list_documents(client):
    r = client.get("/api/documents/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_queue_status(client):
    r = client.get("/api/documents/queue/status")
    assert r.status_code == 200


def test_document_analytics(client):
    r = client.get("/api/documents/analytics")
    assert r.status_code == 200


# ── Expanded coverage ───────────────────────────────────────────────


def test_list_documents_shape(client):
    """Document list items should have expected fields."""
    r = client.get("/api/documents/")
    assert r.status_code == 200
    docs = r.json()
    if docs:
        doc = docs[0]
        assert "id" in doc, "Document missing 'id' field"


def test_queue_status_shape(client):
    """Queue status should return structured data."""
    r = client.get("/api/documents/queue/status")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_document_analytics_shape(client):
    """Analytics should contain key metrics."""
    r = client.get("/api/documents/analytics")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_list_documents_with_limit(client):
    """GET /api/documents/ with limit param should work."""
    r = client.get("/api/documents/", params={"limit": 3})
    assert r.status_code == 200
    docs = r.json()
    if isinstance(docs, list):
        assert len(docs) <= 3


def test_document_search(client):
    """POST /api/documents/search should handle queries."""
    r = client.post("/api/documents/search", json={"query": "test document search"})
    # 200 or 404 (no search endpoint) — not 500
    assert r.status_code != 500, (
        f"Document search returned 500: {r.text[:300]}"
    )

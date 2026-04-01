"""Document error handling — validates graceful failures for edge cases.

Upload, search, and retrieval must not 500 on bad input.
"""


def test_get_nonexistent_document(client):
    """GET /api/documents/{id} for non-existent document should return 404."""
    r = client.get("/api/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent document returned {r.status_code}"
    )


def test_delete_nonexistent_document(client):
    """DELETE /api/documents/{id} for non-existent document."""
    r = client.delete("/api/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent document returned {r.status_code}"
    )


def test_search_documents_empty_query(client):
    """POST /api/documents/search with empty query should not 500."""
    r = client.post("/api/documents/search", json={"query": ""})
    # Could be 400 or 200 with empty results — just not 500
    assert r.status_code != 500, (
        f"Empty document search returned 500: {r.text[:300]}"
    )


def test_upload_document_no_file(client):
    """POST /api/documents/ without a file should return 400/422."""
    r = client.post("/api/documents/", json={"name": "no-file-attached"})
    assert r.status_code in (400, 415, 422), (
        f"Document upload without file returned {r.status_code}"
    )


def test_document_invalid_id_format(client):
    """GET /api/documents/{id} with malformed UUID should not 500."""
    r = client.get("/api/documents/not-a-uuid")
    assert r.status_code in (400, 404, 422), (
        f"Malformed document UUID returned {r.status_code}"
    )

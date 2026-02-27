"""Document endpoints smoke tests."""


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

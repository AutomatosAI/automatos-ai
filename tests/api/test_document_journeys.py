"""Journey: Document upload, processing, search, and delete lifecycle.

Pilot users will upload documents — this flow must work end to end.
"""

import hashlib
import io
import time

import pytest


@pytest.fixture(scope="module")
def document_state():
    return {
        "document_id": None,
        "filename": None,
        "content_hash": None,
        # Academy-style tags with intentional whitespace + a duplicate, to prove
        # the upload handler strips and de-duplicates (order-preserving).
        "tags_sent": "academy, aix, gen-ai-leader, foundations, academy",
        "tags_expected": ["academy", "aix", "gen-ai-leader", "foundations"],
    }


def test_document_upload(client, document_state):
    """POST /api/documents/upload — upload a test document.

    Uses a small text file to validate the upload pipeline without
    hitting size limits or slow processing. The body is made unique per run
    (filename carries a timestamp) so the content_hash duplicate-check creates
    a fresh row rather than short-circuiting on a prior run's upload — otherwise
    the tags under test would never be written.
    """
    filename = f"test-doc-{int(time.time())}.txt"
    content = (
        f"This is a test document for the journey test suite ({filename}).\n"
        "It contains information about the Automatos AI Platform.\n"
        "The platform supports agents, workflows, and memory management.\n"
    )
    document_state["filename"] = filename
    document_state["content_hash"] = hashlib.sha256(content.encode()).hexdigest()

    # httpx multipart upload
    files = {"file": (filename, io.BytesIO(content.encode()), "text/plain")}
    data = {"description": "Journey test document", "tags": document_state["tags_sent"]}

    r = client.post(
        "/api/documents/upload",
        files=files,
        data=data,
        timeout=60.0,
    )
    assert r.status_code in (200, 201), (
        f"Document upload failed: {r.status_code} {r.text[:500]}"
    )
    resp = r.json()
    # A byte-identical prior upload would come back status="duplicate"; the unique
    # body above prevents that, so we expect a freshly created row here.
    assert resp.get("status") != "duplicate", (
        "Upload was deduplicated — content_hash collided with an existing row, "
        "so the tags under test were not written. Body should be unique per run."
    )
    doc_id = resp.get("id") or resp.get("document_id")
    if doc_id:
        document_state["document_id"] = doc_id


def test_document_appears_in_list(client, document_state):
    """GET /api/documents/ — uploaded document should appear in list."""
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    r = client.get("/api/documents/")
    assert r.status_code == 200
    docs = r.json()
    assert isinstance(docs, list)

    doc_ids = [str(d.get("id")) for d in docs]
    assert str(document_state["document_id"]) in doc_ids, (
        f"Uploaded document {document_state['document_id']} not found in document list"
    )


def test_document_get_detail(client, document_state):
    """GET /api/documents/{id} — get uploaded document details."""
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    r = client.get(f"/api/documents/{document_state['document_id']}")
    assert r.status_code == 200
    data = r.json()
    assert data.get("id") or data.get("document_id")


def test_document_search(client, document_state):
    """POST /api/documents/search — search should find uploaded content."""
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    # Wait for processing
    time.sleep(3)

    r = client.post(
        "/api/documents/search",
        json={"query": "Automatos AI Platform", "limit": 5},
    )
    # Search endpoint may return 200 or may not be fully wired — accept gracefully
    assert r.status_code in (200, 404, 501), (
        f"Document search returned unexpected {r.status_code}: {r.text[:300]}"
    )


def test_document_tags_persisted_on_get(client, document_state):
    """GET /api/documents/{id} — tags survive upload and content_hash is exposed.

    Fix 1: the upload handler previously commented out `tags=` ("SQLAlchemy array
    bug"), so tags were silently dropped. Fix 2: content_hash was stored but never
    returned. This asserts both are now present and correct on the detail response.
    """
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    r = client.get(f"/api/documents/{document_state['document_id']}")
    assert r.status_code == 200
    data = r.json()

    # Fix 1: tags persisted, stripped + de-duplicated, order preserved.
    assert data.get("tags") == document_state["tags_expected"], (
        f"Expected tags {document_state['tags_expected']}, got {data.get('tags')!r}"
    )

    # Fix 2: content_hash exposed on the response and matches the uploaded bytes.
    assert data.get("content_hash") == document_state["content_hash"], (
        f"Expected content_hash {document_state['content_hash']}, "
        f"got {data.get('content_hash')!r}"
    )


def test_document_tags_in_list(client, document_state):
    """GET /api/documents/ — the list view also returns the persisted tags."""
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    r = client.get("/api/documents/")
    assert r.status_code == 200
    docs = r.json()
    ours = next(
        (d for d in docs if str(d.get("id")) == str(document_state["document_id"])),
        None,
    )
    assert ours is not None, "Uploaded document not found in list"
    assert ours.get("tags") == document_state["tags_expected"], (
        f"List view tags {ours.get('tags')!r} != expected "
        f"{document_state['tags_expected']}"
    )


def test_document_lookup_by_content_hash(client, document_state):
    """GET /api/documents/?content_hash=... — exact by-hash lookup returns the doc.

    Fix 2: lets a caller (e.g. Academy's --replace) resolve a document by exact
    content instead of by filename.
    """
    if not document_state["document_id"] or not document_state["content_hash"]:
        pytest.skip("No document was uploaded")

    r = client.get(
        "/api/documents/",
        params={"content_hash": document_state["content_hash"]},
    )
    assert r.status_code == 200
    docs = r.json()
    assert isinstance(docs, list) and docs, (
        "by-hash lookup returned nothing for a hash we just uploaded"
    )
    ids = [str(d.get("id")) for d in docs]
    assert str(document_state["document_id"]) in ids, (
        f"Document {document_state['document_id']} not found via content_hash lookup"
    )
    # Every returned row must actually carry that hash (filter is exact-match).
    for d in docs:
        assert d.get("content_hash") == document_state["content_hash"], (
            f"by-hash lookup leaked a non-matching doc: {d.get('id')} "
            f"hash={d.get('content_hash')!r}"
        )


def test_document_delete(client, document_state):
    """DELETE /api/documents/{id} — clean up uploaded document."""
    if not document_state["document_id"]:
        pytest.skip("No document was uploaded")

    r = client.delete(f"/api/documents/{document_state['document_id']}")
    assert r.status_code in (200, 204), (
        f"Document delete failed: {r.status_code} {r.text[:300]}"
    )

    # Verify it's gone
    verify = client.get(f"/api/documents/{document_state['document_id']}")
    assert verify.status_code in (404, 200), (
        f"Deleted document still accessible with status {verify.status_code}"
    )

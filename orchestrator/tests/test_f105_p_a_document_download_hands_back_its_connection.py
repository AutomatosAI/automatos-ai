"""F105 (wave 2): a document download hands its request's connection back before it streams.

``GET /api/documents/{id}/download`` reads the caller's document on the request's
``get_db`` session (the ``documents:read`` gate reads the caller's role on the same one),
then asks object storage for the file (``head_object``, a network call) or streams the local
copy. ``GET /api/documents/download?path=`` streams a file after the gate's read. FastAPI
closes that session only when the response ends, so its pooled connection sat idle in
transaction for the storage call and the whole download.

Both routes now end the transaction (``end_open_transaction``) once the database is done
with: the by-id route keeps the row it read (``expunge``) for the link and the file. The
answers are unchanged.

DB-free: no database, storage or socket.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

import services.document_downloads as downloads
from api import documents

WS = uuid4()
LINK = {"url": "https://objects.example.test/signed", "filename": "price-list.csv"}


class _RequestSession:
    """The request's session: the document it finds, and what was done with it, in order."""

    def __init__(self, events, doc=None):
        self._events, self._doc = events, doc

    def query(self, model):
        self._events.append("document read")
        return self

    def filter(self, *conditions):
        return self

    def first(self):
        return self._doc

    def expunge(self, row):
        self._events.append("document kept")

    def commit(self):
        self._events.append("request transaction ended")


def _doc(path):
    return NS(id=5, workspace_id=WS, filename="5_price-list.csv", original_filename="price-list.csv",
              file_path=str(path))


@pytest.fixture
def storage(monkeypatch):
    """Object storage: the link it answers (None: no copy there), and when it was asked."""
    asked = NS(link=None, events=None)

    def link(doc, filename, mode):
        asked.events.append("storage asked")
        return asked.link

    monkeypatch.setattr(documents, "stored_object_link", link)
    return asked


def _by_id(events, doc):
    return documents.download_document_by_id(document_id=5, mode="url", ctx=NS(workspace_id=WS),
                                             db=_RequestSession(events, doc))


def test_a_local_file_streams_after_the_request_transaction_ends(storage, tmp_path):
    events = storage.events = []
    local = tmp_path / "price-list.csv"
    local.write_text("sku,price\n", encoding="utf-8")

    response = _by_id(events, _doc(local))

    assert events == ["document read", "document kept", "request transaction ended", "storage asked"]
    assert isinstance(response, FileResponse)
    assert (response.path, response.filename, response.media_type) == (
        str(local), "price-list.csv", "application/octet-stream")


def test_object_storage_is_asked_after_the_request_transaction_ends(storage, tmp_path):
    events = storage.events = []
    storage.link = dict(LINK)

    response = _by_id(events, _doc(tmp_path / "gone.csv"))

    assert response == LINK
    assert events == ["document read", "document kept", "request transaction ended", "storage asked"]


def test_a_missing_document_and_a_missing_file_are_still_not_found(storage, tmp_path):
    events = storage.events = []
    with pytest.raises(HTTPException) as missing:
        _by_id(events, None)
    assert (missing.value.status_code, missing.value.detail) == (404, "Document not found")

    with pytest.raises(HTTPException) as no_file:
        _by_id(events, _doc(tmp_path / "gone.csv"))
    assert (no_file.value.status_code, no_file.value.detail) == (404, "Document file not found in storage")


def test_a_file_by_path_streams_after_the_gates_transaction_ends(monkeypatch, tmp_path):
    events = []
    folder = tmp_path / str(WS)
    folder.mkdir()
    (folder / "report.md").write_text("# Report", encoding="utf-8")
    monkeypatch.setattr(downloads, "DOCUMENTS_ROOT", tmp_path)

    response = documents.download_document(path=str(folder / "report.md"), ctx=NS(workspace_id=WS),
                                           db=_RequestSession(events))

    assert events == ["request transaction ended"]
    assert isinstance(response, FileResponse) and response.filename == "report.md"
    with pytest.raises(HTTPException) as missing:
        documents.download_document(path=str(folder / "missing.md"), ctx=NS(workspace_id=WS),
                                    db=_RequestSession(events))
    assert (missing.value.status_code, missing.value.detail) == (404, "Document not found")


@pytest.mark.parametrize("route", [documents.download_document_by_id, documents.download_document])
def test_the_downloads_run_in_the_threadpool(route):
    assert not inspect.iscoroutinefunction(route)

"""F072 — cloud sync ingests the file, never Composio's download receipt.

Night 3: every Google Drive and Dropbox file synced "completed" with one ~515
byte chunk whose text was ``{'mimetype': …, 'name': …, 's3url': 'https://…'}``
— Composio now nests the download link inside a descriptor, the extractor
looked for links only at the top level, and ``_to_bytes`` stored ``str(dict)``.
A presigned R2 link went into the knowledge base as text, and
``cloud_documents.file_size`` was 0 for every file.

The shapes below are the ones the night's backend logged (key sets verbatim;
values representative, signatures redacted).
"""
from __future__ import annotations

import asyncio
import base64
import os
import socket
from types import SimpleNamespace as NS

import httpx
import pytest

import core.models  # noqa: F401 — every mapper registered before rows are built
from core.security import web_access
from modules.rag.services import cloud_file_downloader as dl
from modules.rag.services import cloud_sync_service as sync_mod
from modules.rag.services.cloud_file_downloader import (
    CloudContentError, CloudDownloadError, CloudFileDownloader,
)

SIGNED = (
    "https://acct0000.r2.cloudflarestorage.com/{path}?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=REDACTED%2F20260922%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20260922T183831Z"
    "&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=REDACTED"
)
DRIVE_URL = SIGNED.format(path="googledrive/GOOGLEDRIVE_DOWNLOAD_FILE/response/f1")
DROPBOX_URL = SIGNED.format(path="dropbox/DROPBOX_READ_FILE/response/f2")

# GOOGLEDRIVE_DOWNLOAD_FILE — v3 REST ``data`` (and the SDK's inner ``data``)
DRIVE_REST = {
    "composio_execution_message": "File downloaded successfully and uploaded to storage. " * 5,
    "display_url": "https://drive.google.com/file/d/1AbC/view?usp=drivesdk",
    "downloaded_file_content": {
        "mimetype": "text/markdown",
        "name": "30-MODULAR-ARCHITECTURE-REFACTOR.md",
        "s3url": DRIVE_URL,
    },
    "export_applied": False,
    "id": "1AbC",
    "kind": "drive#file",
    "link_label": "Open in Google Drive",
    "mimeType": "text/markdown",
    "name": "30-MODULAR-ARCHITECTURE-REFACTOR.md",
}
# The SDK fallback: data keys ['data', 'error', 'successful'], inner = DRIVE_REST
DRIVE_SDK_RESULT = {"success": True, "error": None,
                    "data": {"data": DRIVE_REST, "error": None, "successful": True}}

# DROPBOX_READ_FILE — v3 REST ``data``: ['content', 'file_name', 'message', 'metadata']
DROPBOX_REST = {
    "content": {
        "mimetype": "application/octet-stream",
        "name": "32-WIDGET-INTEGRATION-SYSTEM.md",
        "s3url": DROPBOX_URL,
    },
    "file_name": "32-WIDGET-INTEGRATION-SYSTEM.md",
    "message": "File read successfully",
    "metadata": {"name": "32-WIDGET-INTEGRATION-SYSTEM.md", "id": "id:abc", "rev": "015",
                 "path_display": "/Automatos/32-WIDGET-INTEGRATION-SYSTEM.md", "size": 34410},
}

MARKDOWN = ("# Widget integration\n\n" + "The widget loader mounts per workspace.\n" * 900).encode()
PDF = b"%PDF-1.7\n" + bytes(range(256)) * 40 + b"\n%%EOF\n"


@pytest.fixture
def fetched(monkeypatch):
    """Stand in for the network: record every link followed, answer with bytes."""
    seen: list[str] = []
    answers = {DRIVE_URL: MARKDOWN, DROPBOX_URL: MARKDOWN}

    def fake(url):
        seen.append(url)
        return answers.get(url, MARKDOWN)

    monkeypatch.setattr(CloudFileDownloader, "_download_from_url", staticmethod(fake))
    return NS(seen=seen, answers=answers)


def _downloader(monkeypatch, rest_data, sdk_result=None):
    d = CloudFileDownloader.__new__(CloudFileDownloader)
    d.db = None
    d.executor = None

    async def rest(*_a, **_k):
        return rest_data

    monkeypatch.setattr(d, "_execute_via_rest_api", rest)
    monkeypatch.setattr(d, "_get_entity_id", lambda _ws: "entity-1")
    if sdk_result is not None:
        import core.composio.client as composio_client
        client = NS(composio=object(), execute_action=lambda **_kw: sdk_result)
        monkeypatch.setattr(composio_client, "get_composio_client", lambda: client)
    return d


def _download(d, app, name):
    path = asyncio.run(d.download_file(app, "file-1", workspace_id="ws", file_name=name))
    try:
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


# ── the recorded shapes: the nested link is followed ────────────────────────

def test_drive_follows_the_s3url_nested_in_downloaded_file_content(fetched):
    assert CloudFileDownloader._extract_content(DRIVE_REST) == MARKDOWN
    assert fetched.seen == [DRIVE_URL]          # not display_url, not the descriptor


def test_dropbox_follows_the_s3url_nested_in_content(fetched):
    assert CloudFileDownloader._extract_content(DROPBOX_REST) == MARKDOWN
    assert fetched.seen == [DROPBOX_URL]


def test_a_top_level_s3url_is_still_followed(fetched):
    assert CloudFileDownloader._extract_content({"s3url": DRIVE_URL, "content": "inline"}) == MARKDOWN


def test_a_link_at_any_depth_beats_inline_content(fetched):
    data = {"content": "a short truncated preview of the file",
            "result": {"file": {"s3url": DROPBOX_URL}}}
    assert CloudFileDownloader._extract_content(data) == MARKDOWN
    assert fetched.seen == [DROPBOX_URL]


def test_dropbox_end_to_end_writes_the_file_not_the_receipt(monkeypatch, fetched):
    body = _download(_downloader(monkeypatch, DROPBOX_REST), "dropbox", "32-WIDGET-INTEGRATION-SYSTEM.md")
    assert body == MARKDOWN
    assert b"s3url" not in body and b"X-Amz-Signature" not in body


# ── inline content still works ──────────────────────────────────────────────

def test_inline_string_content():
    text = "# Notes\n\nPlain inline text from a small file."
    assert CloudFileDownloader._to_bytes(CloudFileDownloader._extract_content({"file_content": text})) == text.encode()


def test_inline_content_nested_in_a_descriptor():
    text = "# Nested\n\nInline text inside the provider's descriptor."
    got = CloudFileDownloader._extract_content({"downloaded_file_content": {"content": text, "name": "n.md"}})
    assert got == text


def test_inline_base64_is_decoded_to_the_file_bytes():
    got = CloudFileDownloader._extract_content({"file_content_bytes": base64.b64encode(PDF).decode()})
    assert CloudFileDownloader._to_bytes(got) == PDF


def test_inline_raw_bytes_pass_through():
    assert CloudFileDownloader._to_bytes(CloudFileDownloader._extract_content({"content": PDF})) == PDF


# ── a descriptor, a receipt, a link: never document text ────────────────────

def test_to_bytes_refuses_a_dict_instead_of_stringifying_it():
    with pytest.raises(CloudContentError):
        CloudFileDownloader._to_bytes(DRIVE_REST["downloaded_file_content"])


def test_a_descriptor_without_a_link_fails_the_download(monkeypatch, fetched):
    """No link and no inline content — not str(dict), not the long
    composio_execution_message: the sync item errors."""
    data = {**DROPBOX_REST, "content": {"mimetype": "text/markdown", "name": "x.md"},
            "message": "File read successfully. " * 20}
    with pytest.raises(RuntimeError, match="All download methods failed"):
        _download(_downloader(monkeypatch, data), "dropbox", "x.md")
    assert fetched.seen == []


def test_drive_size_check_never_ends_in_accepting_a_descriptor(monkeypatch, fetched):
    """Night 3: REST gave a 531-byte dict repr, the <2048 check sent it to the
    SDK, the SDK returned the same nested shape and that was accepted. Now the
    REST link is followed; with no link anywhere, both layers refuse."""
    no_link = {**DRIVE_REST, "downloaded_file_content": {"mimetype": "text/markdown", "name": "x.md"}}
    sdk_no_link = {"success": True, "error": None, "data": {"data": no_link, "error": None, "successful": True}}
    with pytest.raises(RuntimeError, match="All download methods failed"):
        _download(_downloader(monkeypatch, no_link, sdk_no_link), "googledrive", "x.md")


def test_drive_sdk_fallback_follows_the_nested_link(monkeypatch, fetched):
    """A small real file (<2048 bytes) still takes the SDK path — which follows
    the doubly nested link rather than accepting the descriptor."""
    small = b"# tiny\n"
    fetched.answers[DRIVE_URL] = small
    no_link = {**DRIVE_REST, "downloaded_file_content": {"mimetype": "text/markdown", "name": "t.md"}}
    body = _download(_downloader(monkeypatch, no_link, DRIVE_SDK_RESULT), "googledrive", "t.md")
    assert body == small and fetched.seen == [DRIVE_URL]


def test_a_signed_link_returned_as_inline_text_is_refused_not_ingested(monkeypatch, fetched):
    with pytest.raises(CloudContentError, match="receipt"):
        _download(_downloader(monkeypatch, {"content": DROPBOX_URL}), "dropbox", "x.md")
    assert fetched.seen == []                   # inline text never steers a fetch


def test_a_pdf_comes_through_the_s3url_as_its_exact_bytes(monkeypatch, fetched):
    """Night 3's one Drive error: the 531-byte receipt was written to a .pdf
    and failed "Could not extract text from PDF"."""
    fetched.answers[DRIVE_URL] = PDF
    pdf_shape = {**DRIVE_REST, "mimeType": "application/pdf",
                 "downloaded_file_content": {"mimetype": "application/pdf", "name": "guide.pdf", "s3url": DRIVE_URL}}
    assert _download(_downloader(monkeypatch, pdf_shape), "googledrive", "guide.pdf") == PDF


# ── the fetch itself: public addresses only, every hop ──────────────────────

@pytest.fixture
def dns(monkeypatch):
    table = {"acct0000.r2.cloudflarestorage.com": "104.18.0.10", "postgres": "172.18.0.5"}

    def fake_getaddrinfo(host, port, proto=0, **_kw):
        ip = table.get(host, host)             # an IP literal resolves to itself
        return [(socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))]

    monkeypatch.setattr(web_access, "_getaddrinfo", fake_getaddrinfo)
    return table


def _transport(monkeypatch, handler):
    real_client = httpx.Client
    seen: list[httpx.Request] = []

    def recording(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(dl.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(recording), **kw))
    return seen


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
    "http://127.0.0.1:8000/api/admin",
    "http://postgres:5432/",
])
def test_a_link_into_the_private_network_is_refused_before_any_request(monkeypatch, dns, url):
    seen = _transport(monkeypatch, lambda r: httpx.Response(200, content=b"secret"))
    with pytest.raises(CloudContentError, match="Refused the download link"):
        CloudFileDownloader._download_from_url(url)
    assert seen == []


def test_a_redirect_into_the_private_network_is_refused(monkeypatch, dns):
    seen = _transport(monkeypatch, lambda r: httpx.Response(302, headers={"location": "http://10.0.0.8/secret"}))
    with pytest.raises(CloudContentError, match="Refused the download link") as refused:
        CloudFileDownloader._download_from_url(DRIVE_URL)
    assert len(seen) == 1
    assert "10." not in str(refused.value)      # what a host resolved to stays in the server log


def test_a_public_link_is_fetched_pinned_to_the_checked_address(monkeypatch, dns):
    seen = _transport(monkeypatch, lambda r: httpx.Response(200, content=PDF))
    assert CloudFileDownloader._download_from_url(DRIVE_URL) == PDF
    assert seen[0].url.host == "104.18.0.10"
    assert seen[0].headers["host"] == "acct0000.r2.cloudflarestorage.com"
    assert "X-Amz-Signature=REDACTED" in str(seen[0].url)     # the signed query survives pinning


# ── a path in the content is text, not a server file ────────────────────────

def test_a_local_path_as_inline_text_is_never_opened(tmp_path, monkeypatch):
    secret = tmp_path / "environ"
    secret.write_bytes(b"OPENROUTER_API_KEY=sk-live")
    monkeypatch.setattr(dl, "_sdk_download_dir", lambda: (tmp_path / "composio-files").resolve())
    assert CloudFileDownloader._to_bytes(str(secret)) == str(secret).encode()


def test_a_file_the_sdk_downloaded_is_read(tmp_path, monkeypatch):
    outdir = tmp_path / "composio-files"
    (outdir / "googledrive").mkdir(parents=True)
    saved = outdir / "googledrive" / "guide.pdf"
    saved.write_bytes(PDF)
    monkeypatch.setattr(dl, "_sdk_download_dir", lambda: outdir.resolve())
    assert CloudFileDownloader._to_bytes(str(saved)) == PDF


# ── the sync records what happened ──────────────────────────────────────────

class _Query:
    def __init__(self, db, model):
        self.db, self.model = db, model

    def filter(self, *_a):
        return self

    def first(self):
        if self.model is sync_mod.CloudSyncConfig:
            return self.db.sync_config
        return self.db.existing

    def get(self, _id):
        return NS(status="completed", chunk_count=7, team_access=None)

    def count(self):
        return sum(1 for r in self.db.added if getattr(r, "sync_status", None) == "synced")

    def all(self):
        return list(self.db.rows)


class _Db:
    def __init__(self, existing=None, rows=()):
        self.existing, self.rows, self.added = existing, rows, []
        self.sync_config = NS(root_folder_path="/Automatos", default_team_access=[], last_sync_at=None)
        self.rollbacks = 0

    def query(self, model):
        return _Query(self, model)

    def add(self, row):
        self.added.append(row)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, _row):
        pass


def _sync(monkeypatch, download, db):
    service = sync_mod.CloudSyncService.__new__(sync_mod.CloudSyncService)
    service.db = db
    monkeypatch.setattr(service, "_get_connection",
                        lambda _id: NS(app_name="googledrive", total_documents_synced=0, last_successful_sync=None))

    async def listing(**_kw):
        return [{"external_file_id": "1AbC", "name": "30-MODULAR-ARCHITECTURE-REFACTOR.md",
                 "path": "", "mime_type": "text/markdown", "size": 0, "modified_at": None}]

    monkeypatch.setattr(service, "list_files", listing)

    class _Manager:
        def __init__(self, **_kw):
            pass

        async def upload_document(self, **_kw):
            return 731

    class _Downloader:
        def __init__(self, _db):
            pass

        download_file = staticmethod(download)

    monkeypatch.setattr(sync_mod, "DocumentManager", _Manager)
    monkeypatch.setattr(dl, "CloudFileDownloader", _Downloader)
    return asyncio.run(service.sync_folder(connection_id=5, workspace_id="00000000-0000-0000-0000-0000000000c1"))


def _cloud_rows(db):
    return [r for r in db.added if isinstance(r, sync_mod.CloudDocument)]


def test_a_new_file_that_fails_is_recorded_as_an_error_with_the_reason(monkeypatch):
    async def receipt(**_kw):
        raise CloudContentError("GOOGLEDRIVE returned a download receipt for 1AbC, not the file — nothing was ingested")

    db = _Db()
    job = _sync(monkeypatch, receipt, db)
    [row] = _cloud_rows(db)
    assert row.sync_status == "error" and "download receipt" in row.sync_error
    assert row.external_file_id == "1AbC" and row.document_id is None
    assert (job.files_synced, job.files_errored) == (0, 1)


def test_an_existing_file_that_fails_is_marked_error(monkeypatch):
    async def boom(**_kw):
        raise CloudDownloadError("All download methods failed for 1AbC")

    existing = NS(sync_status="synced", sync_error=None, cloud_modified_at=None, document_id=730)
    db = _Db(existing=existing)
    _sync(monkeypatch, boom, db)
    assert existing.sync_status == "error" and "All download methods failed" in existing.sync_error
    assert _cloud_rows(db) == []


def test_an_upstream_error_body_never_reaches_the_workspace(monkeypatch):
    """sync_error is returned by the file listing to every workspace member;
    Composio's raw HTTP error text stays in the server log."""
    async def composio_500(**_kw):
        raise RuntimeError('Composio API error 500: {"trace": "at /srv/internal/executor.py", "key": "ak_live_x"}')

    db = _Db()
    _sync(monkeypatch, composio_500, db)
    [row] = _cloud_rows(db)
    assert row.sync_status == "error"
    assert row.sync_error == sync_mod._SYNC_FAILED and "ak_live" not in row.sync_error


def test_file_size_is_the_downloaded_bytes_not_the_listing(monkeypatch, tmp_path):
    async def real_file(**_kw):
        path = tmp_path / "dl.md"
        path.write_bytes(MARKDOWN)
        return str(path)

    db = _Db()
    job = _sync(monkeypatch, real_file, db)
    [row] = _cloud_rows(db)
    assert row.sync_status == "synced" and row.file_size == len(MARKDOWN)   # listing said 0
    assert not (tmp_path / "dl.md").exists()                                  # temp file cleaned
    assert job.files_synced == 1


def test_the_listing_never_calls_an_errored_file_synced(monkeypatch):
    import core.cache as cache_mod

    listing = [{"external_file_id": "1AbC", "name": "a.md", "path": "", "mime_type": "", "size": 0}]
    monkeypatch.setattr(cache_mod, "get_cache_service",
                        lambda: NS(get_cloud_listing=lambda *_a: [dict(f) for f in listing]))
    errored = NS(external_file_id="1AbC", sync_status="error", sync_error="download receipt",
                 chunk_count=0, last_synced_at=None, file_size=None)
    service = sync_mod.CloudSyncService.__new__(sync_mod.CloudSyncService)
    service.db = _Db(rows=[errored])
    [f] = asyncio.run(service.list_files(5, "/"))
    assert f["is_synced"] is False and f["sync_status"] == "error" and f["sync_error"] == "download receipt"

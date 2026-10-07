"""A Gmail attachment downloads through the platform's own route (7 Oct decision).

GET /api/emails/attachments/gmail/{message_id}/{attachment_id}?filename=… runs
Composio's GMAIL_GET_ATTACHMENT on the caller's workspace's own Gmail connection,
through the spine, and answers with the bytes as a download:

* Gmail's own body (base64url ``data``) is decoded; Composio's file output is
  read through the platform's outbound check, never a link from the request;
* a workspace without a Gmail connection is refused before any call;
* ids that are not Gmail-shaped are refused before any call;
* the name is sanitised into ``filename*=UTF-8''…``, the type is allowlisted;
* nothing over the size cap is served.
"""
from __future__ import annotations

import asyncio
import base64
import os
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.widget_email as widget_email  # noqa: E402
import api.widget_email_attachment as attachment  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.composio.gmail_attachments import download_path  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.socials.recipes import files  # noqa: E402

OWNER = "00000000-0000-0000-0000-0000000000c1"
STRANGER = "00000000-0000-0000-0000-0000000000c2"
MESSAGE = "18f2a9c4d1e0b7a3"
ATTACHMENT = "ANGjdJ8x-Wq_3vLk0" + "a" * 300
PDF = b"%PDF-1.7\n\xff\xfe binary \x00 bytes"
ROUTE = f"/api/emails/attachments/gmail/{MESSAGE}/{ATTACHMENT}"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _gmail_body(data: bytes) -> dict:
    """Composio's envelope around Gmail's own attachment body."""
    return {"success": True, "action": "GMAIL_GET_ATTACHMENT", "error": None,
            "data": {"successful": True, "error": None,
                     "data": {"attachmentId": ATTACHMENT, "size": len(data), "data": _b64url(data)}}}


class Spine:
    """The faked Composio execute: records each call, answers ``result``."""

    def __init__(self, result: dict):
        self.result = result
        self.calls: list = []

    async def __call__(self, tool_name, tool_args, agent_id=0, **kw):
        self.calls.append({"tool_name": tool_name, "tool_args": tool_args, "agent_id": agent_id, **kw})
        return self.result


class Entities:
    """Only OWNER's workspace has an active Gmail connection."""

    def __init__(self, db):
        self.db = db

    def get_connected_apps(self, workspace_id):
        return ["GMAIL", "SLACK"] if str(workspace_id) == OWNER else ["SLACK"]


@pytest.fixture
def spine(monkeypatch):
    fake = Spine(_gmail_body(PDF))
    monkeypatch.setattr("modules.tools.tool_router.execute_tool", fake)
    monkeypatch.setattr(widget_email, "EntityManager", Entities)
    return fake


def _client(workspace_id: str = OWNER) -> TestClient:
    app = FastAPI()
    app.include_router(widget_email.router)
    ctx = SimpleNamespace(workspace_id=workspace_id, user=SimpleNamespace(id="user_1"), auth_type="clerk")
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx

    def _db():
        yield None

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False)


def test_the_owner_downloads_the_attachment_from_their_own_gmail(spine):
    resp = _client().get(ROUTE, params={"filename": "Quote 42.pdf"})

    assert resp.status_code == 200
    assert resp.content == PDF
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"] == "attachment; filename*=UTF-8''Quote%2042.pdf"
    assert resp.headers["x-content-type-options"] == "nosniff"
    [call] = spine.calls
    assert call["tool_name"] == "composio_execute"
    assert call["workspace_id"] == OWNER
    assert call["caller_context"]["actor_type"] == "user_direct"
    assert call["tool_args"] == {
        "action": "GMAIL_GET_ATTACHMENT",
        "params": {"message_id": MESSAGE, "attachment_id": ATTACHMENT, "file_name": "Quote 42.pdf"},
    }


def test_the_extractors_download_path_is_this_route(spine):
    resp = _client().get(download_path(MESSAGE, ATTACHMENT, "Quote 42.pdf"))
    assert resp.status_code == 200
    assert resp.content == PDF


def test_composio_file_output_is_read_through_the_outbound_check(spine, monkeypatch):
    fetched = []

    async def _fetch(url, *, max_bytes, timeout_seconds):
        fetched.append((url, max_bytes))
        return b"<svg onload=alert(1)>"

    monkeypatch.setattr(files, "fetch", _fetch)
    link = "https://r2.composio.example/att/logo.svg"
    spine.result = {"success": True, "action": "GMAIL_GET_ATTACHMENT",
                    "data": {"data": {"file": {"name": "logo.svg", "mimetype": "image/svg+xml", "s3url": link}}}}

    resp = _client().get(ROUTE, params={"filename": "logo.svg"})

    assert resp.status_code == 200
    assert fetched == [(link, attachment.MAX_ATTACHMENT_BYTES)]
    # SVG can run script: it is served as bytes, never as an inline type.
    assert resp.headers["content-type"] == "application/octet-stream"


def test_a_workspace_without_gmail_is_refused_before_any_call(spine):
    resp = _client(STRANGER).get(ROUTE)
    assert resp.status_code == 404
    assert resp.json()["detail"] == attachment.NOT_CONNECTED
    assert spine.calls == []


@pytest.mark.parametrize("message_id, attachment_id", [
    ("18f2a9c4d1e0b7a3", "abc.def"),
    ("18f2$a9c4", ATTACHMENT),
    ("x" * 129, ATTACHMENT),
    (MESSAGE, "a" * 2049),
    (MESSAGE, "abc%20def"),
])
def test_ids_that_are_not_gmail_ids_are_refused_before_any_call(spine, message_id, attachment_id):
    resp = _client().get(f"/api/emails/attachments/gmail/{message_id}/{attachment_id}")
    assert resp.status_code == 400
    assert resp.json()["detail"] == attachment.INVALID_LINK
    assert spine.calls == []


def test_the_disposition_header_carries_only_a_sanitised_name(spine):
    hostile = "../../etc/pass\u202ewd\r\nSet-Cookie: x=1;\".pdf"
    resp = _client().get(f"{ROUTE}?filename={quote(hostile, safe='')}")

    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    assert disposition == "attachment; filename*=UTF-8''" + quote('passwdSet-Cookie: x=1;".pdf', safe="")
    assert "set-cookie" not in resp.headers
    assert spine.calls[0]["tool_args"]["params"]["file_name"] == 'passwdSet-Cookie: x=1;".pdf'


def test_no_name_saves_as_attachment():
    assert attachment.safe_filename(None) == "attachment"
    assert attachment.safe_filename("...") == "attachment"
    assert attachment.safe_filename("C:\\Users\\me\\report.xlsx") == "report.xlsx"


def test_an_attachment_over_the_cap_is_refused(spine, monkeypatch):
    monkeypatch.setattr(attachment, "MAX_ATTACHMENT_BYTES", 8)
    spine.result = _gmail_body(b"0123456789abcdef")

    resp = _client().get(ROUTE)

    assert resp.status_code == 413
    assert resp.json()["detail"] == attachment.TOO_LARGE


def test_a_file_link_over_the_cap_is_refused(spine, monkeypatch):
    async def _fetch(url, *, max_bytes, timeout_seconds):
        raise files.FileTooLarge(f"the tool's file is larger than the {max_bytes}-byte limit")

    monkeypatch.setattr(files, "fetch", _fetch)
    spine.result = {"success": True, "data": {"file": {"s3url": "https://r2.composio.example/big.zip"}}}

    resp = _client().get(ROUTE)

    assert resp.status_code == 413


def test_the_fetch_refuses_a_declared_size_over_the_limit():
    response = httpx.Response(200, headers={"content-length": "100"}, stream=httpx.ByteStream(b""))
    with pytest.raises(files.FileTooLarge, match="larger than the 50-byte limit"):
        asyncio.run(files._read(response, 50))


def test_a_missing_attachment_is_a_plain_404(spine):
    spine.result = {"success": False, "error": "Requested entity was not found.", "data": None}
    resp = _client().get(ROUTE)
    assert resp.status_code == 404
    assert resp.json()["detail"] == attachment.NOT_FOUND


def test_an_action_the_executor_swapped_in_is_refused(spine):
    spine.result = {**_gmail_body(PDF), "action": "GMAIL_FETCH_EMAILS"}
    resp = _client().get(ROUTE)
    assert resp.status_code == 502
    assert PDF not in resp.content


def test_an_answer_with_no_file_is_refused(spine):
    spine.result = {"success": True, "data": {"data": {"message": "ok"}}}
    resp = _client().get(ROUTE)
    assert resp.status_code == 502
    assert resp.json()["detail"] == attachment.UNREADABLE

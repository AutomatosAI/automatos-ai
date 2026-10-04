"""F298 (night 8, #0394.2): the link an agent puts on a card opens the document.

The card's PDF link answered 403 SignatureDoesNotMatch while Deliverables served
the same file. The platform's presign was sound: signed for localhost:9000, the
host the browser reaches. The agent had copied the 600-character URL into its
answer and changed one hex digit of the signature. generate_document now gives
the agent a short link for the owner (the Deliverables page opened on the
document, which fetches it through the app's own route in both editions) and
says the signed share link is for people outside the workspace, never a card.
"""
from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace
from urllib.parse import parse_qsl, quote, urlsplit

import pytest

from config import config
from core.storage import reset_s3_client
from modules.documents.generation_service import DocumentGenerationService
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.f298_fixtures import DELIVERABLE_ID, FILENAME, WS, call_tool, recorded

FRONTEND = "http://localhost:3000"
OWNER_LINK = f"{FRONTEND}/deliverables?tab=outputs&deliverable={DELIVERABLE_ID}"
SECRET = "f298-test-secret"


@pytest.fixture
def frontend(monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", FRONTEND, raising=False)


def test_the_tool_answers_with_the_owners_link_to_the_document(monkeypatch, frontend):
    recorded(monkeypatch)

    answer = call_tool({"title": "Wholesale Price List", "format": "pdf", "data": {"content": "* Decaf: £24.00"}})

    (made,) = answer["results"]
    assert made["open_url"] == OWNER_LINK
    assert "X-Amz" not in made["open_url"] and urlsplit(made["open_url"]).netloc == "localhost:3000"


def test_the_agent_is_told_which_link_goes_on_the_card(monkeypatch, frontend):
    recorded(monkeypatch)
    answer = call_tool({"title": "Wholesale Price List", "format": "pdf", "data": {"content": "* Decaf: £24.00"}})

    summary = ToolResultFormatter.format_for_llm(answer, "generate_document")

    (owner_line,) = [line for line in summary.splitlines() if OWNER_LINK in line]
    assert "give the owner THIS link, on the card" in owner_line and "X-Amz" not in owner_line
    (share_line,) = [line for line in summary.splitlines() if "X-Amz-Signature" in line]
    assert "ONLY for someone outside the workspace" in share_line and "never on a card" in share_line
    assert "use THIS when emailing or messaging" not in summary


def test_a_short_summary_leaves_the_signed_link_out_rather_than_cutting_it(monkeypatch, frontend):
    recorded(monkeypatch)
    answer = call_tool({"title": "Wholesale Price List", "format": "pdf", "data": {"content": "* Decaf: £24.00"}})
    signed = answer["results"][0]["share_url"]

    summary = ToolResultFormatter.format_for_llm(answer, "generate_document", max_chars=600)

    assert OWNER_LINK in summary
    assert "X-Amz-Signature" not in summary and signed not in summary


# ---------------------------------------------------------------------------
# The share link itself: signed for the host the browser opens (local edition)
# ---------------------------------------------------------------------------


def _verifies(url: str, secret: str) -> bool:
    """SigV4 query-string check, as the object store makes it on the host the URL names."""
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    signature = params.pop("X-Amz-Signature")
    canonical_query = "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(params.items()))
    request = "\n".join(["GET", quote(parts.path, safe="/-_.~"), canonical_query, f"host:{parts.netloc}", "",
                         "host", "UNSIGNED-PAYLOAD"])
    scope = params["X-Amz-Credential"].split("/", 1)[1]
    to_sign = "\n".join(["AWS4-HMAC-SHA256", params["X-Amz-Date"], scope,
                         hashlib.sha256(request.encode()).hexdigest()])
    key = f"AWS4{secret}".encode()
    for part in scope.split("/"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.compare_digest(hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest(), signature)


@pytest.fixture
def local_storage(monkeypatch):
    settings = {
        "S3_ENDPOINT_URL": "http://minio:9000", "S3_PUBLIC_ENDPOINT_URL": "http://localhost:9000",
        "S3_USE_PATH_STYLE": True, "AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "f298-test-key",
        "AWS_SECRET_ACCESS_KEY": SECRET, "S3_DOCUMENTS_BUCKET": "automatos-ai",
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value, raising=False)
    reset_s3_client()
    yield
    reset_s3_client()


def test_the_share_link_is_signed_for_the_host_the_browser_opens(local_storage):
    service = DocumentGenerationService(SimpleNamespace(), WS)
    result = SimpleNamespace(filename=FILENAME, s3_key=f"workspaces/{WS}/generated-documents/{FILENAME}")

    url = service.share_link(result)

    assert urlsplit(url).netloc == "localhost:9000"
    assert _verifies(url, SECRET)
    # One digit of the signature changed, as on #0394.2's card, and the store refuses it.
    head, signature = url.rsplit("X-Amz-Signature=", 1)
    altered = signature[:6] + ("7" if signature[6] != "7" else "4") + signature[7:]
    assert not _verifies(f"{head}X-Amz-Signature={altered}", SECRET)

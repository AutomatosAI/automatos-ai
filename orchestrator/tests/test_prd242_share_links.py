"""PRD-242 S4 — a generated document is delivered, not just downloadable.

* ``_build_result`` keeps the S3 object key when the persistence upload landed;
* ``share_link`` mints a 7-day presign through the PUBLIC client (the only link
  that works for someone who cannot sign in), and is ``None`` when storage
  holds no copy — never an exception;
* ``deliverables_app_url`` is the absolute in-app feed link;
* ``generate()`` stamps template attribution on the result;
* the brand-kit suggestions builder prefers the business profile, then the
  workspace, and only emits fields that have a value.

Boundary mocks as in test_prd190_deliverables — no AWS, no DB, no WeasyPrint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

WS = uuid4()


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return None


class _FakeDb:
    def query(self, *a, **k):
        return _FakeQuery()


@pytest.fixture(autouse=True)
def _fresh_storage_factory():
    import core.storage.s3 as s3mod

    s3mod.reset_s3_client()
    yield
    s3mod.reset_s3_client()


def _storage_config(monkeypatch, gs, *, configured: bool, public_endpoint: str = ""):
    import core.storage.s3 as s3mod

    for cfg in {id(gs.config): gs.config, id(s3mod.config): s3mod.config}.values():
        monkeypatch.setattr(cfg, "S3_ENDPOINT_URL", "", raising=False)
        monkeypatch.setattr(cfg, "S3_PUBLIC_ENDPOINT_URL", public_endpoint, raising=False)
        monkeypatch.setattr(cfg, "S3_USE_PATH_STYLE", False, raising=False)
        monkeypatch.setattr(cfg, "AWS_ACCESS_KEY_ID", "test-key" if configured else None, raising=False)
        monkeypatch.setattr(cfg, "AWS_SECRET_ACCESS_KEY", "test-secret" if configured else None, raising=False)
        monkeypatch.setattr(cfg, "S3_DOCUMENTS_BUCKET", "test-bucket", raising=False)


def _mock_s3(monkeypatch, gs):
    import boto3

    fake_boto = MagicMock()
    fake_boto.client.return_value.generate_presigned_url.return_value = (
        "https://test-bucket.s3.amazonaws.com/workspaces/x/generated-documents/f.pdf?X-Amz-Expires=604800&X-Amz-Signature=abc"
    )
    monkeypatch.setattr(boto3, "client", fake_boto.client)
    _storage_config(monkeypatch, gs, configured=True)
    return fake_boto


def _service():
    import modules.documents.generation_service as gs

    return gs.DocumentGenerationService(_FakeDb(), WS)


def test_build_result_keeps_the_s3_key_when_the_upload_lands(tmp_path, monkeypatch):
    import modules.documents.generation_service as gs

    _mock_s3(monkeypatch, gs)
    doc = tmp_path / "20260911_120000_Weekly.pdf"
    doc.write_bytes(b"%PDF-1.4 test")
    result = _service()._build_result(str(doc), "pdf", "Weekly", WS)
    assert result.s3_key == f"workspaces/{WS}/generated-documents/{doc.name}"
    assert result.download_url == f"/api/documents/generated/{doc.name}"


def test_build_result_has_no_s3_key_when_storage_is_off(tmp_path, monkeypatch):
    import modules.documents.generation_service as gs

    _storage_config(monkeypatch, gs, configured=False)
    doc = tmp_path / "x.docx"
    doc.write_bytes(b"PK")
    result = _service()._build_result(str(doc), "docx", "x", WS)
    assert result.s3_key is None
    assert _service().share_link(result) is None


def test_share_link_presigns_seven_days_via_the_public_client(monkeypatch):
    import modules.documents.generation_service as gs
    from modules.documents.models import GeneratedDocument

    fake_boto = _mock_s3(monkeypatch, gs)
    result = GeneratedDocument(path="/x/f.pdf", format="pdf", filename="f.pdf", size=10, s3_key="workspaces/x/generated-documents/f.pdf")
    url = _service().share_link(result)
    assert url and "X-Amz-Expires=604800" in url
    call = fake_boto.client.return_value.generate_presigned_url.call_args
    assert call.args[0] == "get_object"
    assert call.kwargs["ExpiresIn"] == gs.SHARE_LINK_TTL_SECONDS == 7 * 24 * 3600
    assert call.kwargs["Params"]["Key"] == result.s3_key
    assert 'filename="f.pdf"' in call.kwargs["Params"]["ResponseContentDisposition"]


def test_share_link_never_raises(monkeypatch):
    import modules.documents.generation_service as gs
    from modules.documents.models import GeneratedDocument

    fake_boto = _mock_s3(monkeypatch, gs)
    fake_boto.client.return_value.generate_presigned_url.side_effect = RuntimeError("boom")
    result = GeneratedDocument(path="/x/f.pdf", format="pdf", filename="f.pdf", size=10, s3_key="k")
    assert _service().share_link(result) is None


def test_deliverables_app_url_is_absolute(monkeypatch):
    import modules.documents.generation_service as gs

    monkeypatch.setattr(gs.config, "FRONTEND_URL", "https://app.automatos.app/", raising=False)
    assert gs.deliverables_app_url() == "https://app.automatos.app/deliverables?tab=outputs"


@pytest.mark.asyncio
async def test_generate_stamps_template_attribution(monkeypatch, tmp_path):
    import modules.documents.generation_service as gs
    from modules.documents.models import GeneratedDocument

    _storage_config(monkeypatch, gs, configured=False)
    svc = _service()
    tid = uuid4()
    template = SimpleNamespace(id=tid, name="Weekly Report", blocks={"blocks": []}, template_content=None, template_file_path=None, data_schema=None)
    svc.template_service = SimpleNamespace(get_template=lambda t, ws: template, get_template_by_name=lambda ws, n: template)

    async def _fake_pdf(template, data, workspace_id, title="Document", user_id=None):
        return GeneratedDocument(path="/x/f.pdf", format="pdf", filename="f.pdf", size=1)

    monkeypatch.setattr(svc, "generate_pdf", _fake_pdf)
    result = await svc.generate(title="Weekly", format="pdf", data={"sections": []}, workspace_id=WS, template_id=tid)
    assert result.template_id == str(tid)
    assert result.template_name == "Weekly Report"


def _suggestions():
    try:
        from api.document_brand_kit import build_brand_suggestions
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.document_brand_kit not importable in this env: {e}")
    return build_brand_suggestions


def test_brand_suggestions_prefer_the_business_profile_then_the_workspace():
    build = _suggestions()
    ws = SimpleNamespace(name="Local Workspace")
    profile = SimpleNamespace(
        company_name="Acme Ltd", domain="acme.com",
        brands=[{"brand_name": "Acme", "logo_url": "https://acme.com/logo.png"}],
        voice_notes="Build better, together.\nMore notes",
    )
    user = SimpleNamespace(email="jane@acme.com")
    out = build(ws, profile, user)
    assert out["name"] == {"value": "Acme Ltd", "source": "business_profile"}
    assert out["website"] == {"value": "https://acme.com", "source": "business_profile"}
    assert out["logo_url"] == {"value": "https://acme.com/logo.png", "source": "business_profile"}
    assert out["tagline"] == {"value": "Build better, together.", "source": "business_profile"}
    assert out["email"] == {"value": "jane@acme.com", "source": "user"}


def test_brand_suggestions_fall_back_to_the_workspace_and_skip_empties():
    build = _suggestions()
    out = build(SimpleNamespace(name="Local Workspace"), None, None)
    assert out == {
        "name": {"value": "Local Workspace", "source": "workspace"},
        "company_name": {"value": "Local Workspace", "source": "workspace"},
    }
    assert build(None, SimpleNamespace(company_name="  ", domain="", brands=None, voice_notes=None), None) == {}

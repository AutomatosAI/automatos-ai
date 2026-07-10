"""PRD-190 (P2-09) — Deliverables: stop the client-facing artifact rotting.

Pure tests, mocked at the boundaries (boto3/S3, the DB session, the deliverables
read-model) — no AWS, no Postgres round-trip, no WeasyPrint.

  S1  the persisted Deliverable ``download_url`` is the stable app path
      (``/api/documents/generated/{filename}``), never the 1-hour presign (F030/J1).
  S2  the registry ``generate_document`` ToolSpec exposes ``template_id`` so the
      non-chat autonomy lane (missions/board/scheduled) matches chat (F031/J2).
  S3  a Deliverable with ``unresolved``/``unknown`` variables is BLOCKED at
      finalisation — raises ``UnresolvedDeliverableError``, never delivered (J5).
  S4  clean-render + template-lane are persisted on the Deliverable ``extra`` and
      aggregated as ``clean_render_rate`` in ``get_stats`` (J3/J7).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

WS = uuid4()


# --------------------------------------------------------------------------- #
# Boundary fakes
# --------------------------------------------------------------------------- #


class _FakeQuery:
    """SQLAlchemy Query stub: every lookup resolves to 'not found'."""

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return None


class _FakeDb:
    """Session stub for pure tests — no row ever exists."""

    def query(self, *args, **kwargs):
        return _FakeQuery()


def _block_template(*paths):
    """A DocumentTemplate-shaped stub whose blocks reference the given variable paths."""
    blocks = {
        "blocks": [
            {"type": "text", "id": f"t{i}", "content": [{"type": "variable", "path": p}]}
            for i, p in enumerate(paths)
        ]
        or [{"type": "text", "id": "t0", "content": [{"type": "text", "text": "Hello"}]}]
    }
    return SimpleNamespace(
        blocks=blocks, template_content=None, template_file_path=None, data_schema=None
    )


def _service(template=None):
    """DocumentGenerationService on a fake Session; template lookups return `template`."""
    import modules.documents.generation_service as gs

    svc = gs.DocumentGenerationService(_FakeDb(), WS)
    if template is not None:
        svc.template_service = SimpleNamespace(
            get_template=lambda template_id, ws: template,
            get_template_by_name=lambda ws, name: template,
        )
    return svc


def _mock_s3(monkeypatch, gs):
    """Mock boto3 at the module boundary so the S3 upload SUCCEEDS in-process.

    The presign the client *would* mint is a realistic expiring URL, so a
    regression back to persisting it fails these tests loudly.
    """
    fake_boto = MagicMock()
    client = fake_boto.client.return_value
    client.generate_presigned_url.return_value = (
        "https://test-bucket.s3.amazonaws.com/workspaces/x/generated-documents/f.pdf"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Signature=deadbeef"
    )
    monkeypatch.setattr(gs, "boto3", fake_boto)
    monkeypatch.setattr(gs.config, "AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setattr(gs.config, "AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(gs.config, "S3_DOCUMENTS_BUCKET", "test-bucket")
    return fake_boto


# --------------------------------------------------------------------------- #
# S1 — kill the link-rot (F030/J1)
# --------------------------------------------------------------------------- #


def test_deliverable_download_url_is_stable_app_path(tmp_path, monkeypatch):
    """With the S3 upload mocked to succeed, the persisted link is the stable
    app path — not the raw presign that 404s after an hour."""
    import modules.documents.generation_service as gs

    fake_boto = _mock_s3(monkeypatch, gs)
    doc = tmp_path / "20260710_090000_Report.pdf"
    doc.write_bytes(b"%PDF-1.4 test")

    result = _service()._build_result(str(doc), "pdf", "Report", WS)

    assert result.download_url == f"/api/documents/generated/{doc.name}"
    assert result.preview_url == result.download_url  # pdf preview rides the same path
    # The persistence upload is KEPT (containers are ephemeral)...
    fake_boto.client.return_value.put_object.assert_called_once()
    # ...but no presign is minted at generation time — serve_generated_file
    # re-mints on demand, so the persisted link never expires.
    fake_boto.client.return_value.generate_presigned_url.assert_not_called()


def test_download_url_survives_presign_expiry(tmp_path, monkeypatch):
    """The persisted download_url carries no expiring-signature query params."""
    import modules.documents.generation_service as gs

    _mock_s3(monkeypatch, gs)
    doc = tmp_path / "20260710_090000_Invoice.docx"
    doc.write_bytes(b"PK docx test")

    result = _service()._build_result(str(doc), "docx", "Invoice", WS)

    assert "X-Amz-Expires" not in result.download_url
    assert "X-Amz-Signature" not in result.download_url
    assert "?" not in result.download_url
    assert result.download_url.startswith("/api/documents/generated/")


def test_download_url_stable_when_s3_unconfigured(tmp_path, monkeypatch):
    """No S3 at all (local/OSS edition) → same stable app path, served locally."""
    import modules.documents.generation_service as gs

    monkeypatch.setattr(gs, "boto3", None)
    doc = tmp_path / "20260710_090000_Export.xlsx"
    doc.write_bytes(b"xlsx test")

    result = _service()._build_result(str(doc), "xlsx", "Export", WS)

    assert result.download_url == f"/api/documents/generated/{doc.name}"

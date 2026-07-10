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


# --------------------------------------------------------------------------- #
# S2 — template_id on the autonomy lane (F031/J2)
# --------------------------------------------------------------------------- #


def _registry_generate_document_spec():
    from modules.tools.registry.tool_registry import ToolRegistry

    spec = ToolRegistry().get_tool("generate_document")
    assert spec is not None, "generate_document ToolSpec missing from the registry"
    return spec


def _inline_chat_generate_document_schema():
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    # get_available_tools never touches self — call it unbound so the test stays
    # pure (no RAGService / CodeGraphService construction).
    tools = AgentPlatformTools.get_available_tools(object())
    return next(t for t in tools if t["name"] == "generate_document")


def test_generate_document_toolspec_exposes_template_id():
    """The registry ToolSpec (the non-chat lane: missions/board/scheduled) must
    declare template_id — the handler already parses, validates and threads it."""
    spec = _registry_generate_document_spec()
    param = next((p for p in spec.parameters if p.name == "template_id"), None)

    assert param is not None, (
        "generate_document ToolSpec must expose template_id (P2-09 S2) — "
        "without it a non-chat agent cannot pass the id platform_get_template_schema gave it"
    )
    assert param.type == "string"
    assert param.required is False


def test_toolspec_matches_inline_chat_schema_for_template_id():
    """Registry ToolSpec and the chatbot inline schema agree on template_id —
    the chat-only gap is closed, wording and optionality included."""
    spec = _registry_generate_document_spec()
    param = next(p for p in spec.parameters if p.name == "template_id")

    inline = _inline_chat_generate_document_schema()
    inline_prop = inline["parameters"]["properties"]["template_id"]

    assert param.description == inline_prop["description"]
    assert "template_id" not in inline["parameters"]["required"]
    assert param.required is False
    # And the OpenAI-format export (what the autonomy lane actually hands the
    # LLM) carries it as an optional string parameter.
    exported = spec.to_openai_format()["parameters"]
    assert exported["properties"]["template_id"]["type"] == "string"
    assert "template_id" not in exported["required"]


# --------------------------------------------------------------------------- #
# S3 — enforce the unresolved gate (J5) — the load-bearing assertions
# --------------------------------------------------------------------------- #
# generate() is driven end-to-end on the DOCX block path (python-docx is pure
# Python — no WeasyPrint / native libs needed), with S3 mocked off and the
# storage dir pointed at tmp_path. The default brand kit has no company.address,
# so a {{company.address}} chip resolves empty → unresolved.


def _gated_service(tmp_path, monkeypatch, template):
    import modules.documents.generation_service as gs

    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(gs, "boto3", None)  # boundary: no S3 in tests
    return _service(template=template)


@pytest.mark.asyncio
async def test_unresolved_variable_blocks_finalisation(tmp_path, monkeypatch):
    """A known-but-empty variable ({{company.address}} with nothing on file)
    BLOCKS finalisation — raises with the offending path, never returns a
    document with visible [[markers]]."""
    from modules.documents.models import UnresolvedDeliverableError

    svc = _gated_service(tmp_path, monkeypatch, _block_template("company.address"))

    with pytest.raises(UnresolvedDeliverableError) as exc:
        await svc.generate(
            title="Client Letter", format="docx", data={}, template_name="Branded Letter"
        )

    assert "company.address" in exc.value.unresolved
    assert exc.value.unknown == []
    assert "company.address" in str(exc.value)  # loud AND actionable


@pytest.mark.asyncio
async def test_unknown_variable_blocks_finalisation(tmp_path, monkeypatch):
    """An authoring error ({{not_a_real.path}}, not in the catalog) is likewise
    blocked."""
    from modules.documents.models import UnresolvedDeliverableError

    svc = _gated_service(tmp_path, monkeypatch, _block_template("not_a_real.path"))

    with pytest.raises(UnresolvedDeliverableError) as exc:
        await svc.generate(
            title="Client Letter", format="docx", data={}, template_name="Branded Letter"
        )

    assert "not_a_real.path" in exc.value.unknown
    assert exc.value.unresolved == []


@pytest.mark.asyncio
async def test_clean_render_passes(tmp_path, monkeypatch):
    """A fully-resolved template finalises normally — no false positives."""
    svc = _gated_service(tmp_path, monkeypatch, _block_template("data.client_name"))

    result = await svc.generate(
        title="Client Letter",
        format="docx",
        data={"client_name": "Acme Corp"},
        template_name="Branded Letter",
    )

    assert result.unresolved == []
    assert result.unknown == []
    assert result.format == "docx"
    assert result.download_url == f"/api/documents/generated/{result.filename}"


def test_pdf_block_render_captures_unresolved():
    """The shared PDF-path helper captures the honesty lists (and still emits
    the visible [[marker]] for preview-style consumers) instead of discarding
    them behind a log line. Pure — no WeasyPrint, no file I/O."""
    from modules.documents.blocks import validate_blocks

    svc = _service()
    block_doc = validate_blocks(
        {"blocks": [
            {"type": "text", "id": "t0", "content": [{"type": "variable", "path": "company.address"}]},
            {"type": "text", "id": "t1", "content": [{"type": "variable", "path": "bogus.path"}]},
        ]}
    )

    html, unresolved, unknown = svc._render_block_html(block_doc, {}, WS, None, "T")

    assert unresolved == ["company.address"]
    assert unknown == ["bogus.path"]
    assert "[[company.address]]" in html


# --------------------------------------------------------------------------- #
# S4 — clean-render + template-lane become tracked numbers (J3/J7)
# --------------------------------------------------------------------------- #


def test_clean_render_recorded_on_extra(monkeypatch):
    """Registration persists render quality on the Deliverable `extra` JSONB:
    unresolved/unknown counts + which template lane produced the file."""
    import modules.documents.generation_service as gs
    import services.deliverable_service as ds

    captured = {}

    class _FakeDeliverableService:
        def __init__(self, db, workspace_id):
            pass

        def register(self, **kwargs):
            captured.update(kwargs)
            return {"success": True, "deliverable_id": "d-1", "created": True}

    monkeypatch.setattr(ds, "DeliverableService", _FakeDeliverableService)

    template_id = uuid4()
    result = gs.GeneratedDocument(
        path="/data/x.pdf",
        format="pdf",
        filename="x.pdf",
        size=1024,
        download_url="/api/documents/generated/x.pdf",
        unresolved=[],
        unknown=[],
        template_lane="block",
    )
    registration = gs.DocumentGenerationService(_FakeDb(), WS).register_as_deliverable(
        result, title="Q3 Report", source_type="agent_output", template_id=template_id
    )

    assert registration["success"] is True
    render = captured["extra"]["render"]
    assert render["unresolved_count"] == 0
    assert render["unknown_count"] == 0
    assert render["template_lane"] in {"block", "legacy"}
    # template_id attribution (PRD-167 S6) still rides the same extra dict.
    assert captured["extra"]["template_id"] == str(template_id)
    assert captured["preview_url"] == "/api/documents/generated/x.pdf"


def _stats_service(render_row):
    """DeliverableService over a fake Session keyed on the SQL each query runs."""
    from services.deliverable_service import DeliverableService

    def execute(query, params=None):
        sql = str(query)
        if "render" in sql:
            return SimpleNamespace(fetchone=lambda: render_row)
        if "artifact_type" in sql:
            return SimpleNamespace(fetchall=lambda: [])
        if "agent_id" in sql:
            return SimpleNamespace(fetchall=lambda: [])
        return SimpleNamespace(scalar=lambda: 6)

    return DeliverableService(SimpleNamespace(execute=execute), WS)


def test_get_stats_reports_clean_render_rate():
    """With a mix of clean and (pre-gate / legacy-lane) generations recorded,
    get_stats() reports clean ÷ total plus lane coverage."""
    stats = _stats_service(
        SimpleNamespace(rendered_total=4, rendered_clean=3, lane_block=3, lane_legacy=1)
    ).get_stats()

    assert stats["success"] is True
    assert stats["clean_render_rate"] == pytest.approx(0.75)
    assert stats["render"]["total"] == 4
    assert stats["render"]["clean"] == 3
    assert stats["render"]["by_lane"] == {"block": 3, "legacy": 1}


def test_get_stats_clean_render_rate_honest_empty():
    """No rendered documents → clean_render_rate is None (honest empty), never
    a fabricated 100%."""
    stats = _stats_service(
        SimpleNamespace(rendered_total=0, rendered_clean=0, lane_block=0, lane_legacy=0)
    ).get_stats()

    assert stats["success"] is True
    assert stats["clean_render_rate"] is None
    assert stats["render"] == {"total": 0, "clean": 0, "by_lane": {"block": 0, "legacy": 0}}

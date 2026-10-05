"""F354 (5 Oct): the owner adds a document Deliverable (an invoice, a letter) to
knowledge, one at a time, and can take it back out.

Gerard: "maybe for invoices, letters, a user might want to document these, then you can
ask an agent later to pull all files related to customer x, or send invoice to client
y… but we don't want all system reports". So: the document's own file is filed as the
OWNER's document (not agent_output), named for the customer its data gave; adding it
twice files it once; removing it deletes that copy; another workspace's Deliverable is
404; and nothing an agent makes is added on its own.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

PDF = b"%PDF-1.7\nInvoice INV-0042\nBill to: Northwind Traders\nTotal: 1,000.00\n%%EOF\n"
CLIENT = "Northwind Traders"

# Transaction-local stand-in for the migration-managed deliverables table (not
# model-backed, so CI's create_all skips it): the test_prd164_flywheel precedent.
_DELIVERABLES_DDL = """
CREATE TABLE IF NOT EXISTS deliverables (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id      UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    source_type       VARCHAR(30) NOT NULL,
    source_id         VARCHAR(255) NULL,
    agent_id          INTEGER NULL,
    agent_name        VARCHAR(100) NULL,
    artifact_type     VARCHAR(30) NOT NULL,
    title             VARCHAR(255) NOT NULL,
    summary           VARCHAR(500) NULL,
    storage_type      VARCHAR(20) NOT NULL DEFAULT 'workspace',
    file_path         VARCHAR(1024) NOT NULL,
    file_name         VARCHAR(255) NULL,
    file_type         VARCHAR(50) NULL,
    file_size_bytes   BIGINT NULL,
    preview_url       VARCHAR(1024) NULL,
    preview_type      VARCHAR(30) NULL,
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    status            VARCHAR(20) NOT NULL DEFAULT 'ready',
    deleted_at        TIMESTAMPTZ NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


@pytest.fixture
def owner(db_session, seed_workspace, monkeypatch):
    """A workspace with Deliverables, an ingestion manager that records what it is
    given (and files it as a Document row), and files that read as ``PDF``."""
    from sqlalchemy import text

    from core.models.core import Document

    db_session.execute(text(_DELIVERABLES_DDL))
    ws = UUID(seed_workspace())
    uploads, reads = [], []

    async def _upload(**kwargs):
        with open(kwargs["file_path"], "rb") as fh:
            uploads.append({**kwargs, "bytes": fh.read()})
        doc = Document(filename=kwargs["filename"], workspace_id=ws, status="completed",
                       source_type=kwargs["source_type"], tags=kwargs["tags"], description=kwargs["description"])
        db_session.add(doc)
        db_session.flush()
        return doc.id

    def _delete(document_id):
        db_session.query(Document).filter(Document.id == document_id).delete()
        db_session.flush()
        return True

    async def _bytes(workspace_id, storage_type, file_path):
        reads.append((str(workspace_id), storage_type, file_path))
        return PDF

    monkeypatch.setattr("api.documents.get_document_manager",
                        lambda workspace_id: NS(upload_document=_upload, delete_document=_delete))
    monkeypatch.setattr("services.deliverable_file.deliverable_bytes", _bytes)

    def deliverable(*, in_ws=ws, artifact_type="document", file_name="20261005_120000_Invoice_INV-0042.pdf",
                    extra=None, storage_type="generated"):
        row_id = str(uuid.uuid4())
        db_session.execute(text(
            "INSERT INTO deliverables (id, workspace_id, source_type, agent_name, artifact_type, title, storage_type, "
            "file_path, file_name, extra) VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 'task', 'Ledger', :kind, "
            ":title, :storage, :path, :name, CAST(:extra AS jsonb))"),
            {"id": row_id, "ws": str(in_ws), "kind": artifact_type, "title": "Invoice INV-0042",
             "storage": storage_type, "path": f"generated/{file_name}", "name": file_name,
             "extra": json.dumps(extra if extra is not None else {"parties": {"client_name": CLIENT}})})
        db_session.flush()
        return row_id

    return NS(db=db_session, ws=ws, uploads=uploads, reads=reads, deliverable=deliverable)


def _ctx(ws):
    return NS(workspace_id=ws, user=NS(clerk_user_id="user_owner", id=1))


def _add(owner, deliverable_id, ws=None):
    from api.deliverable_knowledge import add_deliverable_to_knowledge

    return asyncio.run(add_deliverable_to_knowledge(deliverable_id, ctx=_ctx(ws or owner.ws), db=owner.db))


def _remove(owner, deliverable_id, ws=None):
    from api.deliverable_knowledge import remove_deliverable_from_knowledge

    return remove_deliverable_from_knowledge(deliverable_id, ctx=_ctx(ws or owner.ws), db=owner.db)


def _state(owner, deliverable_id):
    from services.owner_knowledge import with_knowledge_state

    shown = with_knowledge_state(owner.db, owner.ws, {"success": True, "deliverables": [{"id": deliverable_id}]})
    one = with_knowledge_state(owner.db, owner.ws, {"success": True, "deliverable": {"id": deliverable_id}})
    assert shown["deliverables"][0]["knowledge_document_id"] == one["deliverable"]["knowledge_document_id"]
    return one["deliverable"]["knowledge_document_id"]


def test_the_document_itself_becomes_the_owners_document(owner):
    invoice = owner.deliverable()

    got = _add(owner, invoice)

    assert got["success"] is True and got["already_added"] is False
    [filed] = owner.uploads
    assert filed["bytes"] == PDF and filed["filename"].endswith(".pdf")       # the PDF, read as any upload is
    assert filed["source_type"] is None and filed["created_by"] == "user_owner"  # the owner's, not agent_output
    assert "added-by-owner" in filed["tags"] and f"deliverable-added:{invoice}" in filed["tags"]
    assert owner.reads == [(str(owner.ws), "generated", "generated/20261005_120000_Invoice_INV-0042.pdf")]
    assert _state(owner, invoice) == got["document_id"]


def test_the_customers_name_is_in_what_is_indexed(owner):
    invoice = owner.deliverable()

    _add(owner, invoice)

    [filed] = owner.uploads
    assert f"Client: {CLIENT}" in filed["description"] and "Invoice INV-0042" in filed["description"]
    assert f"client:{CLIENT}" in filed["tags"]
    assert "generated/20261005_120000_Invoice_INV-0042.pdf" in filed["description"]  # where the file is, to send it


def test_adding_it_twice_files_it_once(owner):
    invoice = owner.deliverable()

    first = _add(owner, invoice)
    again = _add(owner, invoice)

    assert again["document_id"] == first["document_id"] and again["already_added"] is True
    assert len(owner.uploads) == 1


def test_removing_it_deletes_that_copy_and_keeps_the_deliverable(owner):
    from sqlalchemy import text

    from core.models.core import Document

    invoice = owner.deliverable()
    doc_id = _add(owner, invoice)["document_id"]

    got = _remove(owner, invoice)

    assert got == {"success": True, "deliverable_id": invoice, "removed": 1}
    assert owner.db.query(Document).filter(Document.id == doc_id).first() is None
    assert _state(owner, invoice) is None
    assert owner.db.execute(text("SELECT 1 FROM deliverables WHERE id = CAST(:id AS uuid) AND deleted_at IS NULL"),
                            {"id": invoice}).fetchone() is not None
    assert _remove(owner, invoice)["removed"] == 0                           # nothing left to remove is not an error
    assert _add(owner, invoice)["already_added"] is False                    # and it can be added again


def test_another_workspaces_deliverable_is_not_found(owner, seed_workspace):
    invoice = owner.deliverable()
    other = UUID(seed_workspace())

    for call in (_add, _remove):
        with pytest.raises(HTTPException) as missing:
            call(owner, invoice, ws=other)
        assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as bad_id:
        _add(owner, "not-a-uuid")
    assert bad_id.value.status_code == 404
    assert owner.uploads == [] and owner.reads == []


def test_a_picture_is_refused_and_a_file_that_cannot_be_read_is_said_so(owner, monkeypatch):
    picture = owner.deliverable(artifact_type="image", file_name="logo.png")
    with pytest.raises(HTTPException) as refused:
        _add(owner, picture)
    assert refused.value.status_code == 409 and "Only a document" in refused.value.detail

    async def _nothing(*_args):
        return None

    monkeypatch.setattr("services.deliverable_file.deliverable_bytes", _nothing)
    with pytest.raises(HTTPException) as unreadable:
        _add(owner, owner.deliverable())
    assert unreadable.value.status_code == 409 and "could not be read" in unreadable.value.detail
    assert owner.uploads == []


def test_a_file_already_in_documents_is_said_so_not_marked_added(owner, monkeypatch):
    """The upload path answers an identical file with the document it already has; that
    document is the owner's own upload, so the Deliverable is not shown as added."""
    from core.models.core import Document

    earlier = Document(filename="invoice.pdf", workspace_id=owner.ws, status="completed", tags=["finance"])
    owner.db.add(earlier)
    owner.db.flush()

    async def _same_bytes(**_kwargs):
        return earlier.id

    monkeypatch.setattr("api.documents.get_document_manager", lambda workspace_id: NS(upload_document=_same_bytes))
    invoice = owner.deliverable()
    with pytest.raises(HTTPException) as known:
        _add(owner, invoice)
    assert known.value.status_code == 409 and "already in your Documents" in known.value.detail
    assert _state(owner, invoice) is None


def test_a_system_report_or_generated_document_is_never_added_on_its_own(owner):
    """F305 stands: with the workspace not opted in, an agent's report or generated
    document reaches no Document; only the owner's click adds one."""
    from services.knowledge_flywheel import ingest_agent_output

    invoice = owner.deliverable()
    for source in ("report", "generated_document"):
        filed = asyncio.run(ingest_agent_output(
            owner.db, owner.ws, content=f"# Invoice INV-0042\nBill to: {CLIENT}", filename="invoice.md",
            source=source, source_id=invoice, title="Invoice INV-0042"))
        assert filed is None
    assert owner.uploads == []
    assert _state(owner, invoice) is None


def test_the_generation_data_names_who_the_document_is_for():
    from modules.documents.deliverable_extra import deliverable_extra, parties_named_in, the_parties_are_remembered
    from modules.documents.models import GeneratedDocument

    assert parties_named_in({"client_name": f"  {CLIENT} ", "recipient_name": " ", "other": "x"}) == {
        "client_name": CLIENT}
    assert parties_named_in(None) == {}

    made = GeneratedDocument(path="/tmp/i.pdf", format="pdf", filename="i.pdf", size=1, template_lane="block")

    class Service:
        @the_parties_are_remembered
        async def generate(self, **kwargs):
            return made

    result = asyncio.run(Service().generate(title="Invoice", format="pdf", data={"customer_name": CLIENT}))
    assert result.parties == {"customer_name": CLIENT} and made.parties == {}   # a copy; the original untouched
    extra = deliverable_extra(result, template_id="t-1")
    assert extra["parties"] == {"customer_name": CLIENT} and extra["template_id"] == "t-1"
    assert extra["render"] == {"unresolved_count": 0, "unknown_count": 0, "template_lane": "block"}
    assert "parties" not in deliverable_extra(made)


def test_a_generated_file_is_read_from_its_own_folder_only(tmp_path, monkeypatch):
    from config import config
    from services.deliverable_file import deliverable_bytes

    ws = str(uuid.uuid4())
    folder = tmp_path / ws / "generated"
    folder.mkdir(parents=True)
    (folder / "invoice.pdf").write_bytes(PDF)
    (tmp_path / "secret.pdf").write_bytes(b"not yours")
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr("core.storage.is_storage_configured", lambda: False)

    assert asyncio.run(deliverable_bytes(ws, "generated", "generated/invoice.pdf")) == PDF
    assert asyncio.run(deliverable_bytes(ws, "generated", "generated/../../secret.pdf")) is None
    assert asyncio.run(deliverable_bytes(ws, "generated", "generated/missing.pdf")) is None
    assert asyncio.run(deliverable_bytes(ws, "s3", "generated/invoice.pdf")) is None

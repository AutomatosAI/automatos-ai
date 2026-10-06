"""F370 follow-up (Gerard, 7 Oct): a generated document is copied into documents/ in the local edition only.

The copy fills the owner's own deliverables folder (locally ~/Development/deliverables,
#722). The hosted edition has no such folder: there the copy is skipped, with a log
line, and the document lives in object storage and on the Deliverables page.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

import pytest

import modules.documents.generation_service as gs
from modules.documents import workspace_documents

WS = UUID("00000000-0000-0000-0000-0000000370b1")
NAME = "20261007_invoice.pdf"


class _Worker:
    written: dict = {}

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def write_binary(self, path, pieces):
        self.written[(self.workspace_id, path)] = b"".join([piece async for piece in pieces])
        return {"success": True, "path": path}


@pytest.fixture
def generated(tmp_path, monkeypatch) -> Path:
    _Worker.written = {}
    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(workspace_documents, "WorkspaceClient", _Worker)
    folder = tmp_path / str(WS) / "generated"
    folder.mkdir(parents=True)
    (folder / NAME).write_bytes(b"%PDF-1.7 invoice")
    return folder


async def _register_then_settle() -> bool:
    """Register the document, then let the background copy (when one started) finish."""
    started = workspace_documents.copy_when_registered(WS, NAME, "pdf")
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
    return started


def test_the_local_edition_copies_the_document_into_documents(generated, monkeypatch):
    monkeypatch.setattr(workspace_documents.config, "AUTH_EDITION", "local")

    assert asyncio.run(_register_then_settle()) is True
    assert _Worker.written == {(str(WS), f"documents/{NAME}"): b"%PDF-1.7 invoice"}


def test_the_hosted_edition_skips_the_copy_and_says_so(generated, monkeypatch, caplog):
    monkeypatch.setattr(workspace_documents.config, "AUTH_EDITION", "saas")

    with caplog.at_level(logging.INFO, logger=workspace_documents.__name__):
        assert asyncio.run(_register_then_settle()) is False

    assert _Worker.written == {}
    assert any("hosted edition" in record.getMessage() and NAME in record.getMessage() for record in caplog.records)


def test_the_hosted_edition_does_not_log_for_what_is_never_copied(generated, monkeypatch, caplog):
    monkeypatch.setattr(workspace_documents.config, "AUTH_EDITION", "saas")

    with caplog.at_level(logging.INFO, logger=workspace_documents.__name__):
        assert workspace_documents.copy_when_registered(WS, "render.png", "social_image") is False

    assert not caplog.records

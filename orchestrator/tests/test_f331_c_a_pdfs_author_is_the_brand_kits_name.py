"""F331 (night 10, 5 Oct): a PDF's author is the brand kit's name.

The night's invoices said "Author: Automatos" (or another brand's name) in their
PDF metadata. A PDF is now written by the workspace's brand kit: its company
name, else the brand's name. A workspace with no kit keeps the render's default.
Rendered for real (WeasyPrint) and read back (pdfplumber).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pdfplumber
import pytest

import modules.documents.generation_service as generation_service
from modules.documents.generation_service import DocumentGenerationService
from services.brand_rules import document_author, forget_cached_kits

WS = uuid.UUID("00000000-0000-0000-0000-0000000331c1")
COMPANY = "Harbour Lantern Coffee Roasters Ltd"
BRAND = "Harbour Lantern"


class _Rows:
    def __init__(self, row: Any):
        self._row = row

    def filter(self, *args: Any, **kwargs: Any) -> "_Rows":
        return self

    def first(self) -> Any:
        return self._row


class _Workspace:
    """A session whose one workspace has ``settings``."""

    def __init__(self, settings: Dict[str, Any]):
        self._workspace = SimpleNamespace(settings=settings)

    def get(self, model: Any, key: Any) -> Any:
        return self._workspace

    def query(self, *args: Any) -> _Rows:
        return _Rows(self._workspace)


@pytest.fixture
def author_of(monkeypatch, tmp_path):
    """The Author a PDF generate_pdf writes for a workspace with ``settings`` carries."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)

    class _Resolver:
        def __init__(self, db: Any):
            pass

        def resolve(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(values={}, unknown=[])

    monkeypatch.setattr(generation_service, "VariableResolver", _Resolver)

    def go(settings: Dict[str, Any]) -> Optional[str]:
        forget_cached_kits()
        service = DocumentGenerationService(_Workspace(settings), WS)
        data = {"sections": [{"title": "Order", "content": "12 kg of Harbour Blend."}]}
        result = asyncio.run(service.generate_pdf(None, data, WS, "Invoice HL-2026-0142"))
        with pdfplumber.open(result.path) as pdf:
            return pdf.metadata.get("Author")

    yield go
    forget_cached_kits()


def test_the_author_is_the_kits_company_name(author_of):
    assert author_of({"brand_kit": {"name": BRAND, "company": {"name": COMPANY}}}) == COMPANY


def test_a_kit_without_a_company_name_is_the_brands_name(author_of):
    assert author_of({"brand_kit": {"name": BRAND}}) == BRAND


def test_a_workspace_with_no_kit_names_no_author(author_of):
    assert not author_of({})


def test_the_authors_name_is_never_the_person_who_signs():
    kit = {"name": BRAND, "company": {"name": COMPANY}, "voice": {"sign_off": "Gerard, Harbour Lantern"}}

    assert document_author(kit) == COMPANY
    assert document_author({"name": "  "}) is None and document_author(None) is None

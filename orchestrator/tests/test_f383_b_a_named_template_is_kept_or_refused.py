"""F383 (night 11, 7 Oct): an invoice regeneration keeps its template, or is refused saying why.

Invoices 3 and 4 of night 11 (d8ab0bd6, 69bba145) skipped the "Branded Invoice" template:
a name the exact, case-sensitive lookup missed (or an id no longer active) fell back
without a word to Basic Report or the kit-letterhead page, and the agent reported it as
the invoice. A name is now read with case and spacing aside, an id kept from before an
edit is the template's current version, and a named template not in the workspace is an
error back to the model with the closest names: never another template in its place.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS

import pytest

from modules.documents.data_coverage import DEFAULT_PDF_TEMPLATE, template_for
from modules.documents.template_lookup import TemplateNotFound, named_template, template_by_id

WS = uuid.UUID("febae41b-374b-4580-a5ef-f698bdd382e4")
INVOICE_DATA = {"invoice_number": "HL-W-1041", "customer": "Lantern Kitchen", "due_date": "2026-11-06"}


def _row(name, version=1, active=True):
    return NS(id=uuid.uuid4(), name=name, version=version, is_active=active, format="pdf", blocks=None,
              template_content=None)


INVOICE_V1 = _row("Branded Invoice", 1, active=False)   # replaced by an edit
INVOICE = _row("Branded Invoice", 2)
LETTER = _row("Branded Letter")
BASIC = _row(DEFAULT_PDF_TEMPLATE)


class _Templates:
    """DocumentTemplateService over rows: exact names, active rows only, latest version first."""

    def __init__(self, rows):
        self.rows = rows

    def _active(self):
        return sorted((r for r in self.rows if r.is_active), key=lambda r: (r.name, -r.version))

    def get_template(self, template_id, workspace_id):
        return next((r for r in self._active() if r.id == template_id), None)

    def get_template_by_name(self, workspace_id, name):
        return next((r for r in self._active() if r.name == name), None)

    def list_templates(self, workspace_id):
        return self._active()

    def get_template_version(self, template_id, workspace_id):
        return next((r for r in self.rows if r.id == template_id), None)


TEMPLATES = _Templates([INVOICE_V1, INVOICE, LETTER, BASIC])


@pytest.mark.parametrize("name", ["Branded Invoice", "branded invoice", "  Branded   Invoice ", "BRANDED INVOICE"])
def test_a_name_is_read_with_case_and_spacing_aside(name):
    assert named_template(TEMPLATES, WS, name) is INVOICE
    assert template_for(TEMPLATES, WS, "pdf", dict(INVOICE_DATA), template_name=name) is INVOICE


def test_an_id_from_before_an_edit_is_the_templates_current_version():
    assert template_by_id(TEMPLATES, WS, INVOICE_V1.id) is INVOICE
    assert template_for(TEMPLATES, WS, "pdf", dict(INVOICE_DATA), template_id=INVOICE_V1.id) is INVOICE


def test_a_name_that_is_not_there_is_refused_with_the_closest_names():
    with pytest.raises(TemplateNotFound) as refused:
        template_for(TEMPLATES, WS, "pdf", dict(INVOICE_DATA), template_name="Branded Invoce")

    message = str(refused.value)
    assert message.startswith("No template 'Branded Invoce' in this workspace, so nothing was made.")
    assert "Close names: Branded Invoice" in message


def test_an_id_that_is_not_there_is_refused_never_swapped():
    unknown = uuid.uuid4()

    with pytest.raises(TemplateNotFound, match=f"No template with id {unknown}"):
        template_for(TEMPLATES, WS, "pdf", dict(INVOICE_DATA), template_id=unknown)


def test_a_call_that_names_no_template_still_gets_the_default_rule():
    report = {"title": "September", "sections": [{"title": "Sales", "content": "Up."}]}

    assert template_for(_Templates([]), WS, "docx", report) is None
    assert template_for(_Templates([]), WS, "pdf", report) is None          # no Basic Report: the block fallback


def test_the_refusal_reaches_the_model_as_the_tools_error():
    from modules.tools.execution import generate_document_tool as gdt

    assert isinstance(TemplateNotFound("No template 'X' in this workspace"), gdt.AUTHORED_FAILURES)
    answer = gdt.failure("No template 'X' in this workspace, so nothing was made.")
    assert answer["success"] is False and "No template 'X'" in answer["error"]


def test_the_schema_tool_reads_names_the_same_way(monkeypatch):
    from modules.tools.discovery import template_tools

    monkeypatch.setattr("modules.documents.template_service.DocumentTemplateService", lambda db: TEMPLATES)

    assert template_tools.find_template(None, WS, {"template_name": "branded invoice"}) == (INVOICE, None)
    template, problem = template_tools.find_template(None, WS, {"template_name": "Invoce"})
    assert template is None and problem.startswith("No template 'Invoce' in this workspace")

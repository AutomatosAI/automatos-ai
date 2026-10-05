"""F336 (night 10): "[Your Company Name]" becomes the brand kit's company name.

Night 10: "Sincerely, [Your Company Name]" shipped in two documents. The sign-off fill
knew "[Your name]" and a bare "[Name]" and nothing of the company. A company
placeholder now becomes the company's name (the kit's company contact name, else the
kit's name), a different fill from the person who signs; "[Your Company Name]" anywhere,
a bare "[Company Name]" or "[Company]" only where a signature goes. "Dear [Name]," and
"Dear [Company Name] team" are the reader's and stay.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from tests import test_brandkit_b_a_placeholder_signature_becomes_the_kits_sign_off as kit_b

finalize = kit_b.finalize
fresh_kits = kit_b.fresh_kits

COMPANY = "Harbourline Coffee Roasters Ltd"
SIGN_OFF = "Gerard"
KIT = {"name": "Harbourline Coffee Roasters", "company": {"name": COMPANY}, "voice": {"sign_off": SIGN_OFF}}
LETTER = ("Dear [Name],\n\nYour October box ships on Monday.\n\nSincerely,\n[Your Name]\n[Your Company Name]")


def test_every_company_placeholder_is_filled_and_the_readers_are_not():
    from services.brand_rules import fill_company

    assert fill_company("Sincerely, [Your Company Name]", COMPANY) == f"Sincerely, {COMPANY}"
    assert fill_company("Thanks,\nGerard\n[Company Name]\n", COMPANY) == f"Thanks,\nGerard\n{COMPANY}\n"
    assert fill_company("Best, [Company]", COMPANY) == f"Best, {COMPANY}"
    assert fill_company("All of us at [your company] say hi.", COMPANY) == f"All of us at {COMPANY} say hi."
    assert fill_company("Dear [Name],", COMPANY) == "Dear [Name],"
    assert fill_company("Dear [Company Name] team,", COMPANY) == "Dear [Company Name] team,"
    assert fill_company("Sincerely, [Your Company Name]", None) == "Sincerely, [Your Company Name]"


def test_the_company_is_the_contacts_name_else_the_brands():
    from services.brand_rules import company_name

    assert company_name(KIT) == COMPANY
    assert company_name({"name": "Harbourline Coffee Roasters", "company": {"name": " "}}) == "Harbourline Coffee Roasters"
    assert company_name({}) is None and company_name(None) is None


def test_a_cards_letter_is_signed_by_the_person_and_the_company(finalize):
    result = finalize(LETTER, {"brand_kit": KIT})
    assert result.startswith("Dear [Name],")
    assert f"Sincerely,\n{SIGN_OFF}\n{COMPANY}" in result and "[Your Company Name]" not in result


def test_a_documents_company_placeholder_is_filled_before_it_renders(monkeypatch):
    import modules.documents.generation_service as gs
    from modules.documents.models import GeneratedDocument

    rendered = {}

    async def generate_pdf(template, data, workspace_id, title, user_id=None):
        rendered.update(data)
        return GeneratedDocument(path="/tmp/x.pdf", format="pdf", filename="x.pdf", size=1)

    db = NS(get=lambda model, key: NS(settings={"brand_kit": KIT}), query=lambda *a, **k: None)
    service = gs.DocumentGenerationService(db, kit_b.WS)
    service.template_service = NS(get_template_by_name=lambda ws, name: None)
    monkeypatch.setattr(service, "generate_pdf", generate_pdf)
    data = {"sections": [{"title": "October", "content": LETTER}], "closing": "Sincerely, [Your Company Name]"}
    asyncio.run(service.generate(title="Club letter", format="pdf", data=data, workspace_id=kit_b.WS))
    assert rendered["sections"][0]["content"].endswith(f"Sincerely,\n{SIGN_OFF}\n{COMPANY}")
    assert rendered["sections"][0]["content"].startswith("Dear [Name],")
    assert rendered["closing"] == f"Sincerely, {COMPANY}"
    assert data["closing"] == "Sincerely, [Your Company Name]"     # the caller's own data is not changed

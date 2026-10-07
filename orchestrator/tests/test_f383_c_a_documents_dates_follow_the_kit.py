"""F383 (night 11, 7 Oct): a document's dates print in the kit's date style.

Night 11's Branded Invoice printed its dates as ISO ("2026-11-06") although the kit's
date style was "d MMMM yyyy": the style reached only the ``date.long`` chip, and the
call's data printed as sent. Now each data value that is wholly an ISO date (or date and
time) prints as that day in the kit's style, in a PDF and a Word file, rows of a table
included; an id or a code is left as sent, and a spreadsheet keeps its dates as data.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import date

import pytest

from modules.documents import kit_dates
from modules.documents.brand_system import DATE_STYLE_DAY_FIRST, DATE_STYLE_MONTH_FIRST
from modules.documents.kit_dates import dates_follow_the_kit, iso_day, kit_dated
from tests import test_f331_a_an_invoice_pdf_shows_its_line_items_and_totals as f331a

make = f331a.make

INVOICE = {
    "invoice_number": "HL-W-1041",
    "customer": "Lantern Kitchen",
    "invoice_date": "2026-10-07",
    "due_date": "2026-11-06",
    "line_items": [{"description": "Harbour Blend 1 kg", "quantity": 12, "unit_price": "19.50",
                    "total": "234.00", "delivered": "2026-10-02T09:30:00Z", "sku": "2026-10"}],
    "total": "234.00",
}


@pytest.mark.parametrize("value, day", [
    ("2026-11-06", date(2026, 11, 6)),
    (" 2026-11-06 ", date(2026, 11, 6)),
    ("2026-10-02T09:30:00Z", date(2026, 10, 2)),
    ("2026-10-02 09:30", date(2026, 10, 2)),
    ("2026-10-02T09:30:00.125+01:00", date(2026, 10, 2)),
])
def test_an_iso_date_or_date_and_time_is_a_day(value, day):
    assert iso_day(value) == day


@pytest.mark.parametrize("value", ["HL-W-1041", "2026-10", "2026-13-40", "06/11/2026", "6 November 2026",
                                   "Due 2026-11-06", 20261106, None])
def test_anything_else_is_not(value):
    assert iso_day(value) is None


def test_the_invoices_dates_print_in_the_kits_style_and_nothing_else_changes():
    sent = copy.deepcopy(INVOICE)

    printed = kit_dated(sent, DATE_STYLE_DAY_FIRST)

    assert (printed["invoice_date"], printed["due_date"]) == ("7 October 2026", "6 November 2026")
    assert printed["line_items"][0]["delivered"] == "2 October 2026"            # a table's row too
    assert printed["invoice_number"] == "HL-W-1041" and printed["line_items"][0]["sku"] == "2026-10"
    assert printed["total"] == "234.00" and printed["line_items"][0]["quantity"] == 12
    assert sent == INVOICE                                                       # a new copy
    assert kit_dated(sent, DATE_STYLE_MONTH_FIRST)["due_date"] == "November 6, 2026"


class _Service:
    """DocumentGenerationService.generate, wrapped: what data reached the renderers."""

    workspace_id = "febae41b-374b-4580-a5ef-f698bdd382e4"
    db = None

    @dates_follow_the_kit
    async def generate(self, **kwargs):
        return kwargs["data"]


def _generate(monkeypatch, kit, fmt):
    async def _kit(db, workspace_id):
        return kit

    monkeypatch.setattr("services.brand_rules.kit_off_loop", _kit)
    return asyncio.run(_Service().generate(title="Invoice HL-W-1041", format=fmt, data=copy.deepcopy(INVOICE)))


def test_a_pdf_and_a_word_file_take_the_kits_date_style(monkeypatch):
    for fmt in ("pdf", "docx"):
        printed = _generate(monkeypatch, {"date_style": DATE_STYLE_MONTH_FIRST}, fmt)
        assert (printed["invoice_date"], printed["due_date"]) == ("October 7, 2026", "November 6, 2026")
    assert _generate(monkeypatch, None, "pdf")["due_date"] == "6 November 2026"   # no kit: the default style


def test_a_spreadsheet_keeps_its_dates_as_data(monkeypatch):
    assert _generate(monkeypatch, {"date_style": DATE_STYLE_DAY_FIRST}, "xlsx")["due_date"] == "2026-11-06"
    assert kit_dates.DATED_FORMATS == frozenset({"pdf", "docx"})


def test_the_invoice_pdf_prints_its_dates_in_the_kits_style(make):
    data = {**{k: v for k, v in INVOICE.items() if k != "line_items"},
            "sections": [{"title": "Invoice HL-W-1041", "content": "Thank you for your order."}]}

    _result, lines = make("Invoice HL-W-1041", data)

    text = "\n".join(lines)
    assert "7 October 2026" in text and "6 November 2026" in text, lines        # night 11: 2026-11-06
    assert "2026-11-06" not in text
    assert "HL-W-1041" in text

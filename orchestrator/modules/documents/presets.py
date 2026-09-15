"""Category presets — the layout a template STARTS from (PRD-243).

A category used to be a tag. Gerard's review of the Studio (2026-09-15): "pick
Letter and nothing changes… the idea is that categories are pre-programmed —
a letter has the business details and the address, an invoice has line items —
and the user selects and edits, not builds from scratch." So every category now
carries a complete, brand-aware block layout:

* it is what **New template → <category>** loads into the editor,
* it is what the **starter** of that category is seeded from (one per category,
  copy-on-customise, refreshed in place when the platform's preset changes), and
* it is what a **copy of a legacy (non-block) template** starts from.

Chips follow the resolver's contract: ``user.*`` / ``company.*`` / ``brand.*`` /
``date.*`` fill themselves; ``data.*`` is what an agent (or a person) supplies per
document. Optional contact details carry ``fallback=""`` so a workspace without a
phone number is not blocked; the fields a document is *about* have no fallback —
an empty one is a blocked document, by design (P2-09 S3).

Pure data + pure helpers; no DB, no IO.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.documents.blocks import collect_list_fields, collect_variable_paths, validate_blocks
from modules.documents.variables.catalog import DYNAMIC_PREFIX

# ---------------------------------------------------------------------------
# Block-tree builders (readable presets, no hand-written ids)
# ---------------------------------------------------------------------------


def _t(text: str, *marks: str) -> Dict[str, Any]:
    return {"type": "text", "text": text, "marks": list(marks)}


def _v(path: str, fallback: Optional[str] = None) -> Dict[str, Any]:
    run: Dict[str, Any] = {"type": "variable", "path": path}
    if fallback is not None:
        run["fallback"] = fallback
    return run


def _heading(bid: str, level: int, *runs: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "heading", "id": bid, "level": level, "content": list(runs)}


def _para(bid: str, *runs: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "text", "id": bid, "content": list(runs)}


def _logo(bid: str = "logo", width_mm: int = 40) -> Dict[str, Any]:
    return {"type": "image", "id": bid, "source": "brand_logo", "alt": "Logo", "width_mm": width_mm}


def _section(bid: str, title: str, *children: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "section", "id": bid, "title": title, "children": list(children)}


def _data_table(bid: str, path: str, columns: List[tuple], empty_text: Optional[str] = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "type": "data_table",
        "id": bid,
        "path": path,
        "columns": [{"key": k, "label": label, "align": align} for k, label, align in columns],
    }
    if empty_text is not None:
        block["empty_text"] = empty_text
    return block


def _table(bid: str, rows: List[List[List[Dict[str, Any]]]], header: bool = False) -> Dict[str, Any]:
    return {"type": "table", "id": bid, "header": header, "rows": rows}


def _page_break(bid: str = "pb") -> Dict[str, Any]:
    return {"type": "page_break", "id": bid}


def _doc(*blocks: Dict[str, Any]) -> Dict[str, Any]:
    return {"version": 1, "blocks": list(blocks)}


# Reusable letterhead: logo + company name + contact line (optional details fall back to "").
def _letterhead() -> List[Dict[str, Any]]:
    return [
        _logo(),
        _heading("lh-name", 3, _v("company.name")),
        _para("lh-address", _v("company.address", "")),
        _para(
            "lh-contact",
            _v("company.email", ""), _t("  ·  "), _v("company.phone", ""), _t("  ·  "), _v("company.website", ""),
        ),
    ]


# ---------------------------------------------------------------------------
# The presets — one per category, in the order the picker shows them
# ---------------------------------------------------------------------------

LETTER = {
    "category": "letter",
    "name": "Branded Letter",
    "description": "Letterhead with your logo and company details, the recipient block, a subject line, the body and a sign-off.",
    "format": "pdf",
    "includes": ["Letterhead from your brand kit", "Recipient and subject", "Body", "Sign-off with your name and email"],
    "blocks": _doc(
        *_letterhead(),
        _para("date", _v("date.long")),
        _para("to-name", _v("data.recipient_name")),
        _para("to-company", _v("data.recipient_company", "")),
        _para("to-address", _v("data.recipient_address", "")),
        _para("subject", _t("Re: ", "bold"), _v("data.subject")),
        _para("salutation", _t("Dear "), _v("data.recipient_name"), _t(",")),
        _para("body", _v("data.body")),
        _para("closing", _t("Kind regards,")),
        _para("sig-name", _v("user.name")),
        _para("sig-email", _v("user.email", "")),
    ),
    "sample_data": {
        "data": {
            "recipient_name": "Jordan Smith",
            "recipient_company": "Northwind Traders",
            "recipient_address": "12 Harbour Street, Dublin 2",
            "subject": "Your proposal for the spring campaign",
            "body": "Thank you for meeting us last week. As discussed, we would be delighted to support the spring campaign and have set out the details below.",
        }
    },
}

INVOICE = {
    "category": "invoice",
    "name": "Branded Invoice",
    "description": "Your details and the client's, invoice number and dates, a line-items table filled from data, totals and payment terms.",
    "format": "pdf",
    "includes": ["From / bill-to blocks", "Invoice number, date, due date", "Line items from data.line_items", "Subtotal, tax, total", "Payment terms"],
    "blocks": _doc(
        _logo(),
        _heading("title", 1, _t("Invoice ")),
        _para("from", _t("From: ", "bold"), _v("company.name"), _t(" · "), _v("company.address", ""), _t(" · "), _v("company.email", "")),
        _para("bill-to", _t("Bill to: ", "bold"), _v("data.client_name"), _t(" · "), _v("data.client_address", ""), _t(" · "), _v("data.client_email", "")),
        _para(
            "meta",
            _t("Invoice #", "bold"), _v("data.invoice_number"),
            _t("    Date: ", "bold"), _v("date.today"),
            _t("    Due: ", "bold"), _v("data.due_date"),
        ),
        _data_table(
            "items", "data.line_items",
            [("description", "Description", "left"), ("quantity", "Qty", "right"), ("unit_price", "Unit price", "right"), ("total", "Total", "right")],
        ),
        _table(
            "totals",
            [
                [[_t("Subtotal")], [_v("data.subtotal")]],
                [[_t("Tax")], [_v("data.tax", "0.00")]],
                [[_t("Total due", "bold")], [_v("data.total")]],
            ],
        ),
        _para("terms", _t("Payment terms: ", "bold"), _v("data.payment_terms", "Net 30")),
        _para("thanks", _t("Thank you for your business.")),
    ),
    "sample_data": {
        "data": {
            "client_name": "Northwind Traders",
            "client_address": "12 Harbour Street, Dublin 2",
            "client_email": "accounts@northwind.example",
            "invoice_number": "INV-0042",
            "due_date": "2026-10-15",
            "line_items": [
                {"description": "Consulting — discovery workshop", "quantity": 1, "unit_price": "1,500.00", "total": "1,500.00"},
                {"description": "Implementation (days)", "quantity": 4, "unit_price": "900.00", "total": "3,600.00"},
            ],
            "subtotal": "5,100.00",
            "tax": "1,173.00",
            "total": "6,273.00",
            "payment_terms": "Net 30 — bank details on request",
        }
    },
}

REPORT = {
    "category": "report",
    "name": "Branded Report",
    "description": "Title and byline, executive summary, findings, a metrics table from data, recommendations, next steps and an appendix.",
    "format": "pdf",
    "includes": ["Title page block with byline", "Executive summary", "Key findings", "Metrics table from data.metrics", "Recommendations and next steps", "Appendix on a new page"],
    "blocks": _doc(
        _logo(bid="logo", width_mm=50),
        _heading("title", 1, _v("data.title")),
        _para("byline", _t("Prepared by "), _v("user.name"), _t(" · "), _v("company.name"), _t(" · "), _v("date.long")),
        _section("s-summary", "Executive summary", _para("summary", _v("data.summary"))),
        _section("s-findings", "Key findings", _para("findings", _v("data.findings"))),
        _section(
            "s-metrics", "Key metrics",
            _data_table("metrics", "data.metrics", [("metric", "Metric", "left"), ("value", "Value", "right"), ("change", "Change", "right")], empty_text="No metrics reported for this period."),
        ),
        _section("s-recs", "Recommendations", _para("recs", _v("data.recommendations"))),
        _section("s-next", "Next steps", _para("next", _v("data.next_steps", ""))),
        _page_break(),
        _heading("appendix", 2, _t("Appendix")),
        _para("appendix-body", _v("data.appendix", "Methodology and source data available on request.")),
    ),
    "sample_data": {
        "data": {
            "title": "Weekly Market Report",
            "summary": "Demand held steady across the core segments this week while acquisition costs fell for the second week running.",
            "findings": "Organic traffic up 12% week on week. Paid conversion improved after the landing-page change. Two competitor price cuts observed.",
            "metrics": [
                {"metric": "Sessions", "value": "48,210", "change": "+12%"},
                {"metric": "Conversion rate", "value": "3.4%", "change": "+0.4 pt"},
                {"metric": "CAC", "value": "€41", "change": "−9%"},
            ],
            "recommendations": "Shift 15% of paid budget to the two best-performing campaigns. Brief the pricing team on the competitor moves.",
            "next_steps": "Pricing review Thursday; campaign rebalance live by Friday.",
        }
    },
}

PROPOSAL = {
    "category": "proposal",
    "name": "Branded Proposal",
    "description": "Cover block, overview, scope of work, timeline, a pricing table from data, terms and next steps.",
    "format": "pdf",
    "includes": ["Cover with client and date", "Overview and scope", "Timeline", "Pricing table from data.pricing", "Terms and next steps", "Your sign-off"],
    "blocks": _doc(
        _logo(bid="logo", width_mm=50),
        _heading("title", 1, _v("data.title")),
        _para("cover", _t("Prepared for "), _v("data.client_name"), _t(" by "), _v("company.name"), _t(" · "), _v("date.long")),
        _section("s-overview", "Overview", _para("overview", _v("data.overview"))),
        _section("s-scope", "Scope of work", _para("scope", _v("data.scope"))),
        _section("s-timeline", "Timeline", _para("timeline", _v("data.timeline"))),
        _section(
            "s-pricing", "Pricing",
            _data_table("pricing", "data.pricing", [("item", "Item", "left"), ("description", "Description", "left"), ("price", "Price", "right")]),
            _para("pricing-note", _v("data.pricing_note", "Prices exclude VAT. Valid for 30 days.")),
        ),
        _section("s-terms", "Terms", _para("terms", _v("data.terms", "Standard terms of business apply; a signed proposal and a purchase order start the work."))),
        _section("s-next", "Next steps", _para("next", _v("data.next_steps"))),
        _para("sig", _v("user.name"), _t(" · "), _v("user.email", ""), _t(" · "), _v("company.name")),
    ),
    "sample_data": {
        "data": {
            "title": "Website Redesign Proposal",
            "client_name": "Northwind Traders",
            "overview": "A refreshed marketing site that loads fast, ranks well and converts visitors into enquiries.",
            "scope": "Discovery workshop, information architecture, design system, 12 page templates, CMS setup, launch support.",
            "timeline": "Six weeks from kick-off: discovery (1), design (2), build (2), launch (1).",
            "pricing": [
                {"item": "Discovery", "description": "Workshop and audit", "price": "€2,500"},
                {"item": "Design and build", "description": "Design system, templates, CMS", "price": "€14,000"},
                {"item": "Launch support", "description": "Two weeks post-launch", "price": "€1,500"},
            ],
            "next_steps": "Confirm scope by 20 September; kick-off the following Monday.",
        }
    },
}

CONTRACT = {
    "category": "contract",
    "name": "Branded Agreement",
    "description": "A services agreement skeleton: parties, services, term, fees, confidentiality, termination, governing law and signature blocks. Edit the clauses to suit.",
    "format": "docx",
    "includes": ["Parties and date", "Numbered clauses (services, term, fees)", "Standard confidentiality and termination text to edit", "Signature table"],
    "blocks": _doc(
        _logo(),
        _heading("title", 1, _v("data.title", "Services Agreement")),
        _para(
            "parties",
            _t("This agreement is made on "), _v("date.long"), _t(" between "), _v("company.name", ), _t(" (the “Provider”) and "),
            _v("data.counterparty_name"), _t(" (the “Client”)."),
        ),
        _section("c1", "1. Services", _para("services", _v("data.services"))),
        _section("c2", "2. Term", _para("term", _v("data.term"))),
        _section("c3", "3. Fees and payment", _para("fees", _v("data.fees"))),
        _section(
            "c4", "4. Confidentiality",
            _para("conf", _t("Each party will keep the other's confidential information confidential, use it only for this agreement, and return or destroy it on request. This clause survives termination.")),
        ),
        _section(
            "c5", "5. Termination",
            _para("termination", _t("Either party may terminate on thirty days' written notice, or immediately if the other party materially breaches this agreement and does not remedy the breach within fourteen days of notice.")),
        ),
        _section("c6", "6. Governing law", _para("law", _t("This agreement is governed by the laws of "), _v("data.governing_law", "Ireland"), _t("."))),
        _heading("sig-title", 2, _t("Signed")),
        _table(
            "signatures",
            [
                [[_t("For the Provider", "bold")], [_t("For the Client", "bold")]],
                [[_v("company.name")], [_v("data.counterparty_name")]],
                [[_t("Name: "), _v("user.name")], [_t("Name: "), _v("data.counterparty_signatory", "")]],
                [[_t("Signature: ________________")], [_t("Signature: ________________")]],
                [[_t("Date: ________________")], [_t("Date: ________________")]],
            ],
            header=True,
        ),
    ),
    "sample_data": {
        "data": {
            "title": "Services Agreement",
            "counterparty_name": "Northwind Traders Ltd",
            "counterparty_signatory": "Jordan Smith, Managing Director",
            "services": "Design, build and launch of the Client's marketing website as described in the proposal dated 15 September 2026.",
            "term": "From the date of signature until launch, and for thirty days of support thereafter.",
            "fees": "€18,000 excluding VAT, invoiced 50% on signature and 50% on launch, payable within 30 days.",
            "governing_law": "Ireland",
        }
    },
}

DATA = {
    "category": "data",
    "name": "Branded Data Sheet",
    "description": "A titled table of rows supplied at generation time, with a short description and a generated-on line. Change the columns to match your data.",
    "format": "pdf",
    "includes": ["Title and description", "Table from data.rows (edit the columns)", "Generated-on line"],
    "blocks": _doc(
        _logo(),
        _heading("title", 1, _v("data.title")),
        _para("desc", _v("data.description", "")),
        _data_table("rows", "data.rows", [("name", "Name", "left"), ("value", "Value", "right"), ("notes", "Notes", "left")]),
        _para("footer", _t("Generated "), _v("date.long"), _t(" by "), _v("user.name"), _t(" · "), _v("company.name")),
    ),
    "sample_data": {
        "data": {
            "title": "Inventory Snapshot",
            "description": "Stock on hand by SKU at close of business.",
            "rows": [
                {"name": "SKU-1001", "value": "240", "notes": "Reorder at 200"},
                {"name": "SKU-1002", "value": "58", "notes": "Below reorder point"},
                {"name": "SKU-1003", "value": "1,120", "notes": ""},
            ],
        }
    },
}

GENERAL = {
    "category": "general",
    "name": "Branded Page",
    "description": "A clean branded page: logo, title, body text and a footer with your company name and the date. The blank-but-branded starting point.",
    "format": "pdf",
    "includes": ["Logo and title", "Body", "Footer with company and date"],
    "blocks": _doc(
        _logo(),
        _heading("title", 1, _v("data.title")),
        _para("body", _v("data.body")),
        _para("footer", _v("company.name"), _t(" · "), _v("date.long")),
    ),
    "sample_data": {
        "data": {
            "title": "Meeting Notes — Product Sync",
            "body": "Attendees, decisions and actions go here. Replace this block or add sections and tables to suit.",
        }
    },
}

PRESETS: List[Dict[str, Any]] = [LETTER, INVOICE, REPORT, PROPOSAL, CONTRACT, DATA, GENERAL]
PRESET_BY_CATEGORY: Dict[str, Dict[str, Any]] = {p["category"]: p for p in PRESETS}
CATEGORIES: List[str] = [p["category"] for p in PRESETS]


def preset_for(category: Optional[str]) -> Dict[str, Any]:
    """The preset for a category; ``general`` when the category is unknown."""
    return PRESET_BY_CATEGORY.get((category or "").strip().lower(), GENERAL)


def preset_payload(preset: Dict[str, Any]) -> Dict[str, Any]:
    """The API/picker shape: the preset plus what it needs (derived, never hand-kept)."""
    doc = validate_blocks(preset["blocks"])
    paths = sorted(set(collect_variable_paths(doc)))
    return {
        "category": preset["category"],
        "name": preset["name"],
        "description": preset["description"],
        "format": preset["format"],
        "includes": list(preset["includes"]),
        "variable_paths": paths,
        "data_fields": [p[len(DYNAMIC_PREFIX):] for p in paths if p.startswith(DYNAMIC_PREFIX)],
        "list_fields": collect_list_fields(doc),
        "blocks": preset["blocks"],
        "sample_data": preset["sample_data"],
    }


__all__ = ["PRESETS", "PRESET_BY_CATEGORY", "CATEGORIES", "preset_for", "preset_payload"]

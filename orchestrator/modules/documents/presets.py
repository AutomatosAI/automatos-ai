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

F345: business and legal terms (payment terms, tax and VAT, how long a price holds,
the terms of business, the governing law) have no fallback either. A default there
printed "Net 30", "0.00" or "the laws of Ireland" for a business that never said
so; now the guard asks for them. A cosmetic line ("Thank you for your business")
keeps its default. A table column that may stay empty is marked ``optional``.

Pure data + pure helpers; no DB, no IO.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.documents.blocks import collect_list_fields, collect_variable_paths, validate_blocks
from modules.documents.variables.catalog import DYNAMIC_PREFIX

# F350: one letterhead logo size on every starter. It was 40-50 mm (the top fifth of
# an A4 page) on the invoice, report and proposal; the owner's own copies used 22 mm.
# F356: 14 mm, beside the company block rather than above it (blocks/letterhead_run.py).
# PRD-255: inside the letterhead the kit's logo_rules size the logo (its height); this is
# the logo block's own width, used where it stands alone.
LETTERHEAD_LOGO_MM = 14

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


def _logo(bid: str = "logo", width_mm: int = LETTERHEAD_LOGO_MM) -> Dict[str, Any]:
    return {"type": "image", "id": bid, "source": "brand_logo", "alt": "Logo", "width_mm": width_mm}


def _section(bid: str, title: Optional[str], *children: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "section", "id": bid, "title": title, "children": list(children)}


def _column(key: str, label: str, align: str, optional: bool = False) -> Dict[str, Any]:
    column: Dict[str, Any] = {"key": key, "label": label, "align": align}
    if optional:
        column["optional"] = True
    return column


def _data_table(bid: str, path: str, columns: List[tuple], empty_text: Optional[str] = None) -> Dict[str, Any]:
    """``columns``: ``(key, label, align)``, with ``True`` fourth for a column that may stay empty."""
    block: Dict[str, Any] = {
        "type": "data_table",
        "id": bid,
        "path": path,
        "columns": [_column(*column) for column in columns],
    }
    if empty_text is not None:
        block["empty_text"] = empty_text
    return block


def _table(bid: str, rows: List[List[List[Dict[str, Any]]]], header: bool = False) -> Dict[str, Any]:
    return {"type": "table", "id": bid, "header": header, "rows": rows}


def _doc(*blocks: Dict[str, Any]) -> Dict[str, Any]:
    return {"version": 1, "blocks": list(blocks)}


# Reusable letterhead: logo + company name + contact line (optional details fall back to "").
# The Branded Letter's and Invoice's, and (F331) the top of a branded PDF made with no
# template. Its block ids are what the page style keys on (blocks/page_style.py).
def letterhead() -> List[Dict[str, Any]]:
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
    "description": (
        "Letterhead with your logo and company details, the date, the recipient block, a subject line, "
        "the greeting, the body and a sign-off. Send the greeting (\"Dear Jordan,\") in greeting; "
        "the body has no greeting or sign-off: the template adds them. When the letter is from a named "
        "person (\"sign it from me, Gerard\"), send their name in signer: it signs in place of your "
        "brand kit's sign-off."
    ),
    "format": "pdf",
    "includes": ["Letterhead from your brand kit", "Date, recipient and subject", "Greeting from data.greeting", "Body", "Sign-off with the signer you send, else your brand kit's sign-off (or your name), and your email"],
    "blocks": _doc(
        *letterhead(),
        _para("date", _v("date.long")),
        _para("to-name", _v("data.recipient_name")),
        _para("to-company", _v("data.recipient_company", "")),
        _para("to-address", _v("data.recipient_address", "")),
        _para("subject", _t("Re: ", "bold"), _v("data.subject")),
        _para("greeting", _v("data.greeting", "")),
        _para("body", _v("data.body")),
        _para("closing", _t("Kind regards,")),
        _para("sig-name", _v("brand.sign_off")),  # data.signer (F364), else the kit's sign-off, else the person (F344)
        _para("sig-email", _v("user.email", "")),
    ),
    "sample_data": {
        "data": {
            "recipient_name": "Jordan Smith",
            "recipient_company": "Northwind Traders",
            "recipient_address": "12 Harbour Street, Dublin 2",
            "subject": "Your proposal for the spring campaign",
            "greeting": "Dear Jordan,",
            "body": "Thank you for meeting us last week. As discussed, we would be delighted to support the spring campaign and have set out the details below.",
        }
    },
}

INVOICE = {
    "category": "invoice",
    "name": "Branded Invoice",
    "description": "Your details and the client's, invoice number and dates, a line-items table filled from data, totals and payment terms.",
    "format": "pdf",
    "includes": ["Letterhead from your brand kit", "Invoice number, date, due date", "Bill-to block", "Line items from data.line_items", "Subtotal, tax, total under the Total column", "Payment terms"],
    "blocks": _doc(
        *letterhead(),
        _heading("title", 1, _t("Invoice "), _v("data.invoice_number")),
        _para("meta", _t("Date: ", "bold"), _v("date.long"), _t("  ·  "), _t("Due: ", "bold"), _v("data.due_date")),
        _para("bill-to-label", _t("BILL TO", "bold")),
        _para("bill-to", _v("data.client_name")),
        _para("bill-to-address", _v("data.client_address", "")),
        _para("bill-to-email", _v("data.client_email", "")),
        _data_table(
            "items", "data.line_items",
            [("description", "Description", "left"), ("quantity", "Qty", "right"), ("unit_price", "Unit price", "right"), ("total", "Total", "right")],
        ),
        _table(
            "totals",
            [
                [[_t("Subtotal")], [_v("data.subtotal")]],
                [[_t("Tax")], [_v("data.tax")]],
                [[_t("Total due", "bold")], [_v("data.total")]],
            ],
        ),
        _para("terms", _t("Payment terms: ", "bold"), _v("data.payment_terms")),
        _para("thanks", _t("Thank you for your business.")),
    ),
    "sample_data": {
        "data": {
            "client_name": "Northwind Traders",
            "client_address": "12 Harbour Street, Dublin 2",
            "client_email": "accounts@northwind.example",
            "invoice_number": "INV-0042",
            "due_date": "4 November 2026",
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
    "description": (
        "Letterhead, title and byline, an optional row of KPI tiles (data.kpis: label, value, change), "
        "executive summary, findings, a metrics table from data, recommendations, next steps and an appendix."
    ),
    "format": "pdf",
    "includes": ["Letterhead from your brand kit", "Title block with byline", "Optional KPI tiles from data.kpis", "Executive summary", "Key findings", "Metrics table from data.metrics", "Recommendations and next steps", "Appendix"],
    "blocks": _doc(
        *letterhead(),
        _heading("title", 1, _v("data.title")),
        _para("byline", _t("Prepared by "), _v("user.name"), _t(" · "), _v("company.name"), _t(" · "), _v("date.long")),
        # F356: optional KPI tiles; an empty or missing data.kpis prints nothing.
        _data_table("kpis", "data.kpis", [("label", "KPI", "left"), ("value", "Value", "left"), ("change", "Change", "left", True)], empty_text=""),
        _section("s-summary", "Executive summary", _para("summary", _v("data.summary"))),
        _section("s-findings", "Key findings", _para("findings", _v("data.findings"))),
        _section(
            "s-metrics", "Key metrics",
            _data_table("metrics", "data.metrics", [("metric", "Metric", "left"), ("value", "Value", "right"), ("change", "Change", "right", True)], empty_text="No metrics reported for this period."),
        ),
        _section("s-recs", "Recommendations", _para("recs", _v("data.recommendations"))),
        _section("s-next", "Next steps", _para("next", _v("data.next_steps", ""))),
        _section("s-appendix", "Appendix", _para("appendix-body", _v("data.appendix", "Methodology and source data available on request."))),
    ),
    "sample_data": {
        "data": {
            "title": "Weekly Market Report",
            "kpis": [
                {"label": "Revenue", "value": "€182k", "change": "+6% week on week"},
                {"label": "New customers", "value": "312", "change": "+41"},
                {"label": "Net promoter score", "value": "61", "change": "+3"},
            ],
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
    "description": (
        "Letterhead, a cover header (title, an optional subtitle in data.subtitle, client and date), overview, "
        "scope of work, timeline, a pricing table from data with an optional total (data.pricing_total), terms and next steps."
    ),
    "format": "pdf",
    "includes": ["Letterhead from your brand kit", "Cover header with an optional subtitle", "Overview and scope", "Timeline", "Pricing table from data.pricing with an optional total row", "Terms and next steps", "Your sign-off"],
    "blocks": _doc(
        *letterhead(),
        # F356: the cover header, one panel; delete the section to drop it. The subtitle is optional.
        _section(
            "cover-header", None,
            _para("eyebrow", _t("PROPOSAL")),
            _heading("title", 1, _v("data.title")),
            _para("subtitle", _v("data.subtitle", "")),
            _para("cover", _t("Prepared for "), _v("data.client_name"), _t(" by "), _v("company.name"), _t(" · "), _v("date.long")),
        ),
        _section("s-overview", "Overview", _para("overview", _v("data.overview"))),
        _section("s-scope", "Scope of work", _para("scope", _v("data.scope"))),
        _section("s-timeline", "Timeline", _para("timeline", _v("data.timeline"))),
        _section(
            "s-pricing", "Pricing",
            _data_table("pricing", "data.pricing", [("item", "Item", "left"), ("description", "Description", "left"), ("price", "Price", "right")]),
            # F356: an optional total row under the Price column; left out when data.pricing_total is not sent.
            _table("pricing-total", [[[_t("Total", "bold")], [_v("data.pricing_total", "")]]]),
            _para("pricing-note", _v("data.pricing_note")),
        ),
        _section("s-terms", "Terms", _para("terms", _v("data.terms"))),
        _section("s-next", "Next steps", _para("next", _v("data.next_steps"))),
        _para("sig", _v("user.name"), _t(" · "), _v("user.email", ""), _t(" · "), _v("company.name")),
    ),
    "sample_data": {
        "data": {
            "title": "Website Redesign Proposal",
            "subtitle": "A faster, clearer site that turns visitors into enquiries",
            "client_name": "Northwind Traders",
            "overview": "A refreshed marketing site that loads fast, ranks well and converts visitors into enquiries.",
            "scope": "Discovery workshop, information architecture, design system, 12 page templates, CMS setup, launch support.",
            "timeline": "Six weeks from kick-off: discovery (1), design (2), build (2), launch (1).",
            "pricing": [
                {"item": "Discovery", "description": "Workshop and audit", "price": "€2,500"},
                {"item": "Design and build", "description": "Design system, templates, CMS", "price": "€14,000"},
                {"item": "Launch support", "description": "Two weeks post-launch", "price": "€1,500"},
            ],
            "pricing_total": "€18,000",
            "pricing_note": "Prices exclude VAT. Valid for 30 days.",
            "terms": "Standard terms of business apply; a signed proposal and a purchase order start the work.",
            "next_steps": "Confirm scope by 20 September; kick-off the following Monday.",
        }
    },
}

CONTRACT = {
    "category": "contract",
    "name": "Branded Agreement",
    "description": "A services agreement skeleton: parties, services, term, fees, confidentiality, termination, governing law and signature blocks. Edit the clauses to suit.",
    "format": "docx",
    "includes": ["Letterhead from your brand kit", "Parties and date", "Numbered clauses (services, term, fees)", "Standard confidentiality and termination text to edit", "Signature table, kept with the last clause"],
    "blocks": _doc(
        *letterhead(),
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
        # F356: the last clause, the heading and the signatures are one block that never splits,
        # so the signatures never stand alone on a page.
        _section(
            "sign-off", None,
            _section("c6", "6. Governing law", _para("law", _t("This agreement is governed by the laws of "), _v("data.governing_law"), _t("."))),
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
    "includes": ["Letterhead from your brand kit", "Title and description", "Table from data.rows (edit the columns)", "Generated-on line"],
    "blocks": _doc(
        *letterhead(),
        _heading("title", 1, _v("data.title")),
        _para("desc", _v("data.description", "")),
        # F356: the figure column is headed "Quantity", not "Value"; rename it to suit the data.
        _data_table("rows", "data.rows", [("name", "Item", "left"), ("value", "Quantity", "right"), ("notes", "Notes", "left", True)]),
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
    "includes": ["Letterhead from your brand kit", "Title", "Body", "Footer with company and date"],
    "blocks": _doc(
        *letterhead(),
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

# F356: the seeded "Meeting Notes" starter (seed_templates.STARTER_TEMPLATES) had no
# template at all, so its own format (docx) could not render and its PDF was the
# no-template fallback. It is now this block layout, on its data fields as they were
# (title, date, attendees, agenda, notes, action_items) plus an optional decisions.
# Not a category preset: Branded Page stays the "general" starting point.
MEETING_NOTES_BLOCKS = _doc(
    *letterhead(),
    _heading("title", 1, _v("data.title")),
    _para("meta", _t("Date: ", "bold"), _v("data.date")),
    _section("s-attendees", "Attendees", _para("attendees", _v("data.attendees"))),
    _section("s-agenda", "Agenda", _para("agenda", _v("data.agenda", ""))),
    _section("s-notes", "Discussion", _para("notes", _v("data.notes", ""))),
    _section("s-decisions", "Decisions", _para("decisions", _v("data.decisions", ""))),
    _section(
        "s-actions", "Actions",
        _data_table("actions", "data.action_items", [("task", "Action", "left"), ("owner", "Owner", "left", True), ("due_date", "Due", "right", True)], empty_text=""),
    ),
)

# PRD-255 US-009: the brand board, the kit on one page. Its parts are ``brand`` blocks, each
# drawn from the kit itself (blocks/brand_board.py): no chip and no data field to fill. Not a
# category preset: it is a starter of its own, seeded beside them (seed_templates).
BRAND_BOARD_CATEGORY = "brand"


def _brand(bid: str, part: str) -> Dict[str, Any]:
    return {"type": "brand", "id": bid, "part": part}


BRAND_BOARD = {
    "category": BRAND_BOARD_CATEGORY,
    "name": "Brand Board",
    "description": (
        "Your brand kit on one page: the logo and its variants on light and dark, the colour roles with their "
        "hex codes, the type scale, the spacing, the tone words and three applications. Drawn from the kit; "
        "nothing to fill in."
    ),
    "format": "pdf",
    "includes": ["The logo, large, and its variants", "Colour roles with hex codes", "The type scale with samples",
                 "Spacing and logo clear space", "Tone words with meanings", "An invoice, a letter and a social card"],
    "blocks": _doc(
        _brand("board-logo", "logo"),  # the logo, large, beside the title, the name and the tagline
        _brand("board-colours", "colours"),
        _section("board-row-1", None, _brand("board-variants", "variants"), _brand("board-voice", "voice")),
        _section("board-row-2", None, _brand("board-type", "type"), _brand("board-spacing", "spacing")),
        _brand("board-applications", "applications"),
    ),
    "sample_data": {},
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


__all__ = [
    "BRAND_BOARD", "BRAND_BOARD_CATEGORY", "MEETING_NOTES_BLOCKS", "PRESETS", "PRESET_BY_CATEGORY", "CATEGORIES", "letterhead", "preset_for", "preset_payload",
]

"""Block-native starter templates (PRD-167 S2/S5).

Global shared starters a non-technical user copies and customises (copy-on-customise).
They demonstrate the block path end-to-end: a brand-logo block, brand/user/company/date
variable chips, and ``data.*`` fill-in chips an agent (or a person) supplies at
generation time.

Stored in ``document_templates.blocks`` (the canonical body) alongside ``sample_data``
for preview. These are the templates that prove the block path; the array-driven legacy
seeds (Invoice, multi-section Report) continue to render through the legacy mapper.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _text(text: str, marks: List[str] = None) -> Dict[str, Any]:
    return {"type": "text", "text": text, "marks": marks or []}


def _var(path: str, fallback: str = None) -> Dict[str, Any]:
    run: Dict[str, Any] = {"type": "variable", "path": path}
    if fallback is not None:
        run["fallback"] = fallback
    return run


BRANDED_LETTER_BLOCKS: Dict[str, Any] = {
    "version": 1,
    "blocks": [
        {"type": "image", "id": "logo", "source": "brand_logo", "alt": "Logo", "width_mm": 40},
        {"type": "heading", "id": "h-name", "level": 1, "content": [_var("brand.name", "Your Company")]},
        {"type": "text", "id": "addr", "content": [_var("company.address", "")]},
        {"type": "text", "id": "contact", "content": [
            _var("company.email", ""), _text("  ·  "), _var("company.phone", ""),
        ]},
        {"type": "text", "id": "date", "content": [_var("date.long")]},
        {"type": "text", "id": "salutation", "content": [
            _text("Dear "), _var("data.recipient_name", "Sir/Madam"), _text(","),
        ]},
        {"type": "text", "id": "body", "content": [_var("data.body", "")]},
        {"type": "text", "id": "signoff", "content": [_text("Sincerely,")]},
        {"type": "text", "id": "sig-name", "content": [_var("user.name", "")]},
        {"type": "text", "id": "sig-email", "content": [_var("user.email", "")]},
    ],
}

BRANDED_REPORT_BLOCKS: Dict[str, Any] = {
    "version": 1,
    "blocks": [
        {"type": "image", "id": "logo", "source": "brand_logo", "alt": "Logo", "width_mm": 50},
        {"type": "heading", "id": "title", "level": 1, "content": [_var("data.title", "Report")]},
        {"type": "text", "id": "byline", "content": [
            _text("Prepared by "), _var("user.name", "the team"),
            _text("  ·  "), _var("date.long"),
        ]},
        {"type": "section", "id": "sec-summary", "title": "Summary", "children": [
            {"type": "text", "id": "summary", "content": [_var("data.summary", "")]},
        ]},
        {"type": "section", "id": "sec-details", "title": "Details", "children": [
            {"type": "text", "id": "details", "content": [_var("data.details", "")]},
        ]},
        {"type": "page_break", "id": "pb"},
        {"type": "heading", "id": "appendix", "level": 2, "content": [_text("Appendix")]},
        {"type": "text", "id": "appendix-body", "content": [_var("data.appendix", "")]},
    ],
}


STARTER_BLOCK_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Branded Letter",
        "description": "Letterhead with your brand logo, company details, and a fill-in body — a non-technical starting point.",
        "format": "pdf",
        "category": "letter",
        "blocks": BRANDED_LETTER_BLOCKS,
        "sample_data": {
            "data": {
                "recipient_name": "Jordan Smith",
                "body": "Thank you for your interest. We're delighted to share the details below and look forward to working together.",
            }
        },
    },
    {
        "name": "Branded Report",
        "description": "A clean branded report with summary, details, and an appendix — fill the sections from chat or by hand.",
        "format": "pdf",
        "category": "report",
        "blocks": BRANDED_REPORT_BLOCKS,
        "sample_data": {
            "data": {
                "title": "Quarterly Review",
                "summary": "Strong performance across all key metrics this quarter.",
                "details": "Revenue grew 18% quarter-over-quarter with improved retention.",
                "appendix": "Methodology and source data available on request.",
            }
        },
    },
]


__all__ = ["STARTER_BLOCK_TEMPLATES", "BRANDED_LETTER_BLOCKS", "BRANDED_REPORT_BLOCKS"]

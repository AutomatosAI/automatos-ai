"""
PRD-130: Page-type extraction schemas
======================================

JSON schemas passed to FirecrawlClient.scrape(url, schema=...). Firecrawl's
LLM-extract mode returns typed structured data alongside markdown.

One schema per page TYPE, not per page. Mapping from URL → schema lives
in pick_schema_for_url().
"""

from __future__ import annotations

from typing import Any


ABOUT_US_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "company_description": {"type": "string"},
        "founded_year": {"type": "string"},
        "mission_statement": {"type": "string"},
        "key_differentiators": {"type": "array", "items": {"type": "string"}},
        "industries_served": {"type": "array", "items": {"type": "string"}},
    },
}

CONTACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "phone": {"type": "string"},
        "email": {"type": "string"},
        "address": {"type": "string"},
        "hours": {"type": "string"},
        "support_channels": {"type": "array", "items": {"type": "string"}},
    },
}

FAQ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
            },
        },
    },
}

POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "policy_type": {"type": "string"},
        "summary": {"type": "string"},
        "key_clauses": {"type": "array", "items": {"type": "string"}},
        "last_updated": {"type": "string"},
    },
}

BRANDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "brand_name": {"type": "string"},
                    "category": {"type": "string"},
                    "logo_url": {"type": "string"},
                },
            },
        },
    },
}

DELIVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "delivery_methods": {"type": "array", "items": {"type": "string"}},
        "lead_times": {"type": "string"},
        "regions_served": {"type": "array", "items": {"type": "string"}},
        "shipping_costs": {"type": "string"},
    },
}

SOLUTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "solution_areas": {"type": "array", "items": {"type": "string"}},
        "target_sectors": {"type": "array", "items": {"type": "string"}},
        "compliance_standards": {"type": "array", "items": {"type": "string"}},
    },
}

GENERIC_PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
}


# URL substring → schema mapping. First match wins.
SCHEMA_BY_URL_PATTERN: list[tuple[str, dict[str, Any], str]] = [
    ("/pages/about",     ABOUT_US_SCHEMA,  "about"),
    ("/pages/contact",   CONTACT_SCHEMA,   "contact"),
    ("/pages/faq",       FAQ_SCHEMA,       "faq"),
    ("/pages/delivery",  DELIVERY_SCHEMA,  "delivery"),
    ("/pages/returns",   POLICY_SCHEMA,    "policy"),
    ("/pages/solutions", SOLUTIONS_SCHEMA, "solutions"),
    ("/pages/brands",    BRANDS_SCHEMA,    "brands"),
    ("/policies/",       POLICY_SCHEMA,    "policy"),
]


def pick_schema_for_url(url: str) -> tuple[dict[str, Any] | None, str]:
    """Return (schema, page_type_label) for a URL, or (None, 'generic')."""
    for pattern, schema, label in SCHEMA_BY_URL_PATTERN:
        if pattern in url:
            return schema, label
    return GENERIC_PAGE_SCHEMA, "generic"

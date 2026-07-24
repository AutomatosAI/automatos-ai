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

# PRD-203 O·S3: per-page schemas for the non-Shopify verticals (SaaS,
# services/agency, content/media). One schema per page TYPE; the archetypes'
# target-page selectors feed the same pick_schema_for_url() mapping.

PRICING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "string"},
                    "billing_period": {"type": "string"},
                    "key_features": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "has_free_tier": {"type": "string"},
    },
}

FEATURES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "features": {"type": "array", "items": {"type": "string"}},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "target_users": {"type": "array", "items": {"type": "string"}},
        "integrations": {"type": "array", "items": {"type": "string"}},
    },
}

DOCS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
        "getting_started": {"type": "string"},
        "api_reference": {"type": "string"},
    },
}

SERVICES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service_offerings": {"type": "array", "items": {"type": "string"}},
        "industries_served": {"type": "array", "items": {"type": "string"}},
        "engagement_models": {"type": "array", "items": {"type": "string"}},
    },
}

CASE_STUDY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "client": {"type": "string"},
        "challenge": {"type": "string"},
        "solution": {"type": "string"},
        "outcome": {"type": "string"},
        "industries_served": {"type": "array", "items": {"type": "string"}},
    },
}

ARTICLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "published_date": {"type": "string"},
        "category": {"type": "string"},
        "summary": {"type": "string"},
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


# URL substring → schema mapping. First match wins. Shopify's ``/pages/*`` and
# ``/policies/*`` patterns come first (most specific); the bare paths that the
# SaaS / services / content verticals use follow.
SCHEMA_BY_URL_PATTERN: list[tuple[str, dict[str, Any], str]] = [
    # Shopify catalog (PRD-130)
    ("/pages/about",     ABOUT_US_SCHEMA,  "about"),
    ("/pages/contact",   CONTACT_SCHEMA,   "contact"),
    ("/pages/faq",       FAQ_SCHEMA,       "faq"),
    ("/pages/delivery",  DELIVERY_SCHEMA,  "delivery"),
    ("/pages/returns",   POLICY_SCHEMA,    "policy"),
    ("/pages/solutions", SOLUTIONS_SCHEMA, "solutions"),
    ("/pages/brands",    BRANDS_SCHEMA,    "brands"),
    ("/policies/",       POLICY_SCHEMA,    "policy"),
    # Cross-vertical bare paths (PRD-203 O·S3)
    ("/pricing",         PRICING_SCHEMA,     "pricing"),
    ("/features",        FEATURES_SCHEMA,    "features"),
    ("/integrations",    FEATURES_SCHEMA,    "features"),
    ("/docs",            DOCS_SCHEMA,        "docs"),
    ("/case-studies",    CASE_STUDY_SCHEMA,  "case_study"),
    ("/portfolio",       CASE_STUDY_SCHEMA,  "case_study"),
    ("/services",        SERVICES_SCHEMA,    "services"),
    ("/articles",        ARTICLE_SCHEMA,     "article"),
    ("/author",          ARTICLE_SCHEMA,     "article"),
    ("/category",        ARTICLE_SCHEMA,     "article"),
    ("/about",           ABOUT_US_SCHEMA,    "about"),
    ("/contact",         CONTACT_SCHEMA,     "contact"),
    ("/faq",             FAQ_SCHEMA,         "faq"),
    ("/privacy",         POLICY_SCHEMA,      "policy"),
    ("/terms",           POLICY_SCHEMA,      "policy"),
    ("/security",        POLICY_SCHEMA,      "policy"),
]


def pick_schema_for_url(url: str) -> tuple[dict[str, Any] | None, str]:
    """Return (schema, page_type_label) for a URL, or (None, 'generic')."""
    for pattern, schema, label in SCHEMA_BY_URL_PATTERN:
        if pattern in url:
            return schema, label
    return GENERIC_PAGE_SCHEMA, "generic"

"""
PRD-130 / PRD-203 O·S3: Archetype Detection — data-driven framework
====================================================================

Pure URL-pattern matching — zero LLM calls, zero cost.

An **archetype** is the platform cold-start's model of a kind of business:

    (detection signals, target-page selectors, per-page extraction schema,
     default team)

Everything an archetype needs is DATA — a frozen ``Archetype`` instance in
``ARCHETYPES`` below. Adding a vertical is a data add (append an instance, and
— only if it introduces a new page *type* — one schema entry in
``schemas.py``), never a change to the detection or bucketing logic.

PRD-203 O·S3 generalises the single ``shopify_catalog`` archetype into this
framework and ships four real verticals so a non-Shopify site no longer
collapses to ``archetype=None`` (empty checklist + empty ``default_team``):

    shopify_catalog · saas_app · services_agency · content_media

The exact count that ships this wave is Gerard's call (§8-Qc); four is the
recommended "framework + 3" so the "platform cold-start" claim is true on
day one. More verticals are pure data adds behind this same framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Archetype:
    slug: str
    name: str
    description: str
    signals: dict[str, list[str]]      # required + boost url substrings
    target_pages: dict[str, Any]       # must / recommended / optional_deep
    quality_checks: list[str]
    default_team: list[str]            # candidate agent slugs


SHOPIFY_CATALOG = Archetype(
    slug="shopify_catalog",
    name="Shopify Catalog Store",
    description="A Shopify-hosted store with collections, products, and policies.",
    signals={
        "required": ["/cdn/shop/", "/collections/", "/products/"],
        "boost": ["/pages/brands", "/blogs/news", "/policies/", "/cart"],
    },
    target_pages={
        "must": [
            "/pages/about",
            "/pages/contact",
            "/pages/faq",
            "/policies/privacy",
            "/policies/refund",
            "/policies/terms",
            "/pages/delivery",
            "/pages/returns",
            "/pages/solutions",
            "/pages/brands",
        ],
        "recommended": [
            "/blogs/technical-bulletins",
            "/blogs/news",
        ],
        "optional_deep": [
            {"pattern": "/blogs/", "purpose": "voice_corpus",
             "label": "Marketer voice training"},
            {"pattern": "/collections/", "purpose": "catalog_index",
             "label": "Sales agent catalog knowledge"},
        ],
    },
    quality_checks=[
        "duplicate_collections",   # /foo + /foo-1
        "typo_slugs",              # Levenshtein < 2 between slugs in same namespace
        "orphan_copies",           # ends with -copy, -copy-1
        "test_products",           # slug = "test*"
    ],
    default_team=[
        "shopify_ops",
        "catalog_hygiene",
        "technical_sales",
        "compliance",
        "content_marketer",
        "brand_relations",
    ],
)


SAAS_APP = Archetype(
    slug="saas_app",
    name="SaaS Application",
    description="A subscription software product with pricing, features, docs and a signup flow.",
    signals={
        "required": ["/pricing", "/signup"],
        "boost": [
            "/features", "/integrations", "/docs", "/api", "/demo",
            "/free-trial", "/product", "/login", "/changelog", "/security",
        ],
    },
    target_pages={
        "must": [
            "/pricing",
            "/features",
            "/about",
            "/contact",
            "/security",
            "/privacy",
            "/terms",
        ],
        "recommended": [
            "/docs",
            "/integrations",
            "/api",
            "/changelog",
            "/blog",
        ],
        "optional_deep": [
            {"pattern": "/docs", "purpose": "support_corpus",
             "label": "Support agent product knowledge"},
            {"pattern": "/blog", "purpose": "voice_corpus",
             "label": "Content voice training"},
        ],
    },
    quality_checks=[
        "stale_changelog",    # last entry older than 6 months
        "broken_pricing",     # pricing page missing plan tiers
        "orphan_docs",        # doc pages with no inbound links
    ],
    default_team=[
        "product_support",
        "docs_writer",
        "onboarding_specialist",
        "compliance",
        "content_marketer",
        "sales_engineer",
    ],
)


SERVICES_AGENCY = Archetype(
    slug="services_agency",
    name="Services / Agency",
    description="A professional-services or agency site organised around services, case studies and a team.",
    signals={
        "required": ["/services", "/contact"],
        "boost": [
            "/case-studies", "/portfolio", "/our-work", "/work", "/team",
            "/testimonials", "/about", "/careers", "/clients", "/approach",
        ],
    },
    target_pages={
        "must": [
            "/services",
            "/about",
            "/contact",
            "/team",
            "/privacy",
            "/terms",
        ],
        "recommended": [
            "/case-studies",
            "/portfolio",
            "/testimonials",
            "/careers",
            "/blog",
        ],
        "optional_deep": [
            {"pattern": "/case-studies", "purpose": "proof_corpus",
             "label": "Proposal agent case-study evidence"},
            {"pattern": "/services", "purpose": "offering_index",
             "label": "Sales agent service catalog"},
        ],
    },
    quality_checks=[
        "thin_case_studies",   # < 3 case studies
        "no_team_bios",        # team page without named people
        "stale_portfolio",     # portfolio last updated > 12 months
    ],
    default_team=[
        "proposal_writer",
        "client_research",
        "delivery_lead",
        "compliance",
        "content_marketer",
        "brand_relations",
    ],
)


CONTENT_MEDIA = Archetype(
    slug="content_media",
    name="Content / Media",
    description="A publisher or media site organised around articles, categories and authors.",
    signals={
        "required": ["/category", "/author"],
        "boost": [
            "/articles", "/posts", "/tag", "/rss", "/archive",
            "/newsletter", "/topics", "/podcast", "/subscribe", "/story",
        ],
    },
    target_pages={
        "must": [
            "/about",
            "/contact",
            "/privacy",
            "/terms",
            "/newsletter",
        ],
        "recommended": [
            "/authors",
            "/topics",
            "/archive",
            "/advertise",
            "/rss",
        ],
        "optional_deep": [
            {"pattern": "/articles", "purpose": "voice_corpus",
             "label": "Editorial voice training"},
            {"pattern": "/category", "purpose": "topic_index",
             "label": "Research agent topic map"},
        ],
    },
    quality_checks=[
        "missing_bylines",     # articles with no author
        "stale_frontpage",     # newest article older than 30 days
        "orphan_categories",   # empty category pages
    ],
    default_team=[
        "editorial_research",
        "content_marketer",
        "audience_growth",
        "compliance",
        "brand_relations",
        "seo_specialist",
    ],
)


# Registry — adding a vertical is appending one entry here (data, not logic).
ARCHETYPES: dict[str, Archetype] = {
    SHOPIFY_CATALOG.slug: SHOPIFY_CATALOG,
    SAAS_APP.slug: SAAS_APP,
    SERVICES_AGENCY.slug: SERVICES_AGENCY,
    CONTENT_MEDIA.slug: CONTENT_MEDIA,
}


@dataclass
class DetectionResult:
    archetype: Archetype | None
    confidence: float                  # 0.0 - 1.0
    matched_signals: list[str] = field(default_factory=list)


def _score_archetype(archetype: Archetype, urls: list[str]) -> DetectionResult | None:
    """Score one archetype against a URL inventory.

    Confidence = (# unique required signals matched / total required)
                 + 0.1 per boost signal matched, capped at 1.0.

    The required-ratio dominates (it is the archetype's fingerprint); boosts
    only sharpen confidence and break ties. Returns ``None`` if no required
    signal hit at all — an archetype with zero fingerprint is not a candidate.
    """
    required = archetype.signals.get("required", [])
    boost = archetype.signals.get("boost", [])

    matched_required = [sig for sig in required if any(sig in u for u in urls)]
    if not matched_required:
        return None

    matched_boost = [sig for sig in boost if any(sig in u for u in urls)]

    confidence = len(matched_required) / max(len(required), 1)
    confidence += 0.1 * len(matched_boost)
    confidence = min(confidence, 1.0)

    return DetectionResult(
        archetype=archetype,
        confidence=confidence,
        matched_signals=matched_required + matched_boost,
    )


def detect_archetype(urls: list[str]) -> DetectionResult:
    """Score every archetype against a URL inventory and return the best match.

    Returns ``archetype=None`` if no archetype hits any required signal — a
    site whose shape matches no known vertical. On a tie the higher confidence
    wins; a genuine tie keeps the first (registration order) so detection is
    deterministic.
    """
    if not urls:
        return DetectionResult(archetype=None, confidence=0.0)

    best = DetectionResult(archetype=None, confidence=0.0)
    for archetype in ARCHETYPES.values():
        scored = _score_archetype(archetype, urls)
        if scored is not None and scored.confidence > best.confidence:
            best = scored

    return best


def select_target_urls(archetype: Archetype, urls: list[str]) -> dict[str, list[str]]:
    """Bucket the discovered URL inventory into must/recommended for the checklist.

    Matching is substring-based against the archetype's ``target_pages``
    patterns. Works uniformly for every archetype — the patterns are data.
    """
    must_patterns = archetype.target_pages.get("must", [])
    recommended_patterns = archetype.target_pages.get("recommended", [])

    must_hits: list[str] = []
    recommended_hits: list[str] = []

    for u in urls:
        if any(p in u for p in must_patterns):
            must_hits.append(u)
        elif any(p in u for p in recommended_patterns):
            recommended_hits.append(u)

    # Dedupe preserving order
    seen: set[str] = set()
    must_hits = [u for u in must_hits if not (u in seen or seen.add(u))]
    seen = set()
    recommended_hits = [u for u in recommended_hits if not (u in seen or seen.add(u))]

    return {
        "must": must_hits,
        "recommended": recommended_hits,
    }

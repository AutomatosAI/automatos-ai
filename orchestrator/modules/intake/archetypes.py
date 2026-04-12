"""
PRD-130: Archetype Detection
=============================

Pure URL-pattern matching — zero LLM calls, zero cost.

Phase 1 ships ONE archetype: shopify_catalog. The detector returns the
archetype slug + a confidence score so the wizard can show the user what
it found and let them confirm.
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
        "boost": ["/pages/brands", "/blogs/news", "/policies/"],
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


ARCHETYPES: dict[str, Archetype] = {
    SHOPIFY_CATALOG.slug: SHOPIFY_CATALOG,
}


@dataclass
class DetectionResult:
    archetype: Archetype | None
    confidence: float                  # 0.0 - 1.0
    matched_signals: list[str] = field(default_factory=list)


def detect_archetype(urls: list[str]) -> DetectionResult:
    """Score each archetype against a URL inventory and return the best match.

    Confidence = (# of unique required signals matched / total required signals)
                 + 0.1 per boost signal matched, capped at 1.0.

    Returns archetype=None if no archetype hits any required signal.
    """
    if not urls:
        return DetectionResult(archetype=None, confidence=0.0)

    best: DetectionResult = DetectionResult(archetype=None, confidence=0.0)

    for archetype in ARCHETYPES.values():
        required = archetype.signals.get("required", [])
        boost = archetype.signals.get("boost", [])

        matched_required = [
            sig for sig in required
            if any(sig in u for u in urls)
        ]
        if not matched_required:
            continue

        matched_boost = [
            sig for sig in boost
            if any(sig in u for u in urls)
        ]

        confidence = len(matched_required) / max(len(required), 1)
        confidence += 0.1 * len(matched_boost)
        confidence = min(confidence, 1.0)

        if confidence > best.confidence:
            best = DetectionResult(
                archetype=archetype,
                confidence=confidence,
                matched_signals=matched_required + matched_boost,
            )

    return best


def select_target_urls(archetype: Archetype, urls: list[str]) -> dict[str, list[str]]:
    """Bucket the discovered URL inventory into must/recommended for the user checklist.

    Matching is substring-based against the archetype's target_pages patterns.
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

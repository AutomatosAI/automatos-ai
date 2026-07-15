"""PRD-203 O·S3 — archetype detection framework: gold set + schema selection.

Pure, network-free. Proves:
  * the detection gold set classifies each labelled site to the right archetype
    (top-1 accuracy) and calibrates confidence (strong match high, no-match 0);
  * the regression the single ``shopify_catalog`` archetype caused is closed —
    a non-Shopify site now yields a NON-empty checklist + NON-empty default_team
    instead of ``archetype=None`` collapsing both;
  * per-page extraction schema selection resolves the new verticals' page types.
"""
from __future__ import annotations

import pytest

from modules.intake.archetypes import (
    ARCHETYPES,
    detect_archetype,
    select_target_urls,
)
from modules.intake.schemas import (
    ABOUT_US_SCHEMA,
    ARTICLE_SCHEMA,
    CASE_STUDY_SCHEMA,
    FEATURES_SCHEMA,
    GENERIC_PAGE_SCHEMA,
    PRICING_SCHEMA,
    SERVICES_SCHEMA,
    pick_schema_for_url,
)


# ---------------------------------------------------------------------------
# Gold set — labelled site URL inventory → expected archetype slug (or None)
# ---------------------------------------------------------------------------

GOLD_SET: list[tuple[str, list[str], str | None]] = [
    (
        "shopify_store",
        [
            "https://acme.com/cdn/shop/files/logo.png",
            "https://acme.com/collections/all",
            "https://acme.com/products/widget",
            "https://acme.com/pages/about",
            "https://acme.com/policies/privacy",
        ],
        "shopify_catalog",
    ),
    (
        "saas_product",
        [
            "https://acme.io/pricing",
            "https://acme.io/signup",
            "https://acme.io/features",
            "https://acme.io/docs/intro",
            "https://acme.io/integrations",
            "https://acme.io/api",
        ],
        "saas_app",
    ),
    (
        "services_agency",
        [
            "https://acme.agency/services",
            "https://acme.agency/contact",
            "https://acme.agency/case-studies/bigco",
            "https://acme.agency/team",
            "https://acme.agency/about",
        ],
        "services_agency",
    ),
    (
        "content_publisher",
        [
            "https://acme.news/category/world",
            "https://acme.news/author/jane-doe",
            "https://acme.news/articles/a-big-story",
            "https://acme.news/tag/politics",
            "https://acme.news/newsletter",
        ],
        "content_media",
    ),
    (
        "unknown_site",
        [
            "https://acme.xyz/",
            "https://acme.xyz/home",
            "https://acme.xyz/welcome",
        ],
        None,
    ),
]


def test_archetype_detection_accuracy():
    """Top-1 accuracy over the gold set is 100% and confidence is calibrated."""
    correct = 0
    for name, urls, expected in GOLD_SET:
        result = detect_archetype(urls)
        got = result.archetype.slug if result.archetype else None
        assert got == expected, f"{name}: expected {expected}, got {got}"
        if got == expected:
            correct += 1

        # Calibration: a real match is confident; a no-match is exactly 0.0.
        if expected is None:
            assert result.confidence == 0.0
        else:
            assert result.confidence >= 0.5, f"{name}: weak confidence {result.confidence}"

    accuracy = correct / len(GOLD_SET)
    assert accuracy == 1.0, f"top-1 accuracy {accuracy:.2f} below 1.0"


def test_ships_multiple_real_archetypes():
    """The framework ships more than the single Shopify vertical (§8-Qc)."""
    assert "shopify_catalog" in ARCHETYPES
    # framework + 3 real verticals (Gerard's Qc lean)
    assert len(ARCHETYPES) >= 4
    for slug in ("saas_app", "services_agency", "content_media"):
        assert slug in ARCHETYPES
        assert ARCHETYPES[slug].default_team, f"{slug} has empty default_team"
        assert ARCHETYPES[slug].signals.get("required"), f"{slug} has no required signals"
        assert ARCHETYPES[slug].target_pages.get("must"), f"{slug} has no must pages"


@pytest.mark.parametrize(
    "urls,expected_slug",
    [
        (["https://acme.io/pricing", "https://acme.io/signup", "https://acme.io/features"], "saas_app"),
        (["https://acme.agency/services", "https://acme.agency/contact", "https://acme.agency/team"], "services_agency"),
        (["https://acme.news/category/x", "https://acme.news/author/y"], "content_media"),
    ],
)
def test_non_shopify_site_gets_nonempty_checklist(urls, expected_slug):
    """The regression the single archetype caused: a non-Shopify site no longer
    collapses to archetype=None → empty checklist + empty default_team."""
    result = detect_archetype(urls)
    assert result.archetype is not None, "non-Shopify site detected as None (regression)"
    assert result.archetype.slug == expected_slug

    buckets = select_target_urls(result.archetype, urls)
    # The regression was archetype=None → BOTH buckets empty. A detected archetype
    # must yield a non-empty checklist (must ∪ recommended).
    assert buckets["must"] or buckets["recommended"], (
        "checklist is empty for a detected archetype (the single-archetype regression)"
    )
    assert result.archetype.default_team, "default_team is empty (Mission Zero would get [])"


@pytest.mark.parametrize(
    "url,expected_schema,expected_label",
    [
        ("https://acme.io/pricing", PRICING_SCHEMA, "pricing"),
        ("https://acme.io/features", FEATURES_SCHEMA, "features"),
        ("https://acme.agency/services", SERVICES_SCHEMA, "services"),
        ("https://acme.agency/case-studies/x", CASE_STUDY_SCHEMA, "case_study"),
        ("https://acme.news/articles/story", ARTICLE_SCHEMA, "article"),
        ("https://acme.news/author/jane", ARTICLE_SCHEMA, "article"),
        # Shopify page types still resolve (no regression on the original vertical)
        ("https://acme.com/pages/about", ABOUT_US_SCHEMA, "about"),
        # Bare /about (SaaS/services shape) resolves too
        ("https://acme.io/about", ABOUT_US_SCHEMA, "about"),
        # Unmatched → generic
        ("https://acme.xyz/random-page", GENERIC_PAGE_SCHEMA, "generic"),
    ],
)
def test_schema_selection_per_page_type(url, expected_schema, expected_label):
    schema, label = pick_schema_for_url(url)
    assert schema is expected_schema
    assert label == expected_label


def test_empty_url_inventory_is_none():
    result = detect_archetype([])
    assert result.archetype is None
    assert result.confidence == 0.0

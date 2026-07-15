"""
PRD-130: Profile Builder
=========================

Assembles a BusinessProfile dict from scrape results.

Pure data-shaping logic — no I/O. Takes a list of FirecrawlClient.scrape()
return values + the page-type label and returns a dict matching the
business_profiles table columns.

The wizard router persists this. The plan_generator reads from this.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _safe_get(d: dict | None, *keys: str, default: Any = None) -> Any:
    """Walk a nested dict safely."""
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def build_profile(
    domain: str,
    archetype_slug: str,
    scrape_results: list[dict[str, Any]],
    user_goals: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate scrape results into a single business profile dict.

    Args:
        domain: bare domain (e.g. "inbuilduk.com")
        archetype_slug: detected archetype (e.g. "shopify_catalog")
        scrape_results: list of {url, page_type, markdown, extract, metadata}
        user_goals: goals selected in Step 1 of the wizard

    Returns:
        dict matching business_profiles table columns (sans id/timestamps)
    """
    company_name: str | None = None
    sectors: list[str] = []
    brands: list[dict[str, Any]] = []
    standards: list[str] = []
    voice_notes_chunks: list[str] = []
    quality_findings: dict[str, Any] = {"errors": [], "notes": []}

    for result in scrape_results:
        page_type = result.get("page_type", "generic")
        extract = result.get("extract") or {}

        if page_type == "about":
            if not company_name:
                company_name = extract.get("company_name")
            industries = extract.get("industries_served") or []
            for ind in industries:
                if ind and ind not in sectors:
                    sectors.append(ind)
            mission = extract.get("mission_statement")
            if mission:
                voice_notes_chunks.append(f"Mission: {mission}")

        elif page_type == "solutions":
            target = extract.get("target_sectors") or []
            for s in target:
                if s and s not in sectors:
                    sectors.append(s)
            compliance = extract.get("compliance_standards") or []
            for c in compliance:
                if c and c not in standards:
                    standards.append(c)

        elif page_type == "brands":
            for b in extract.get("brands") or []:
                if isinstance(b, dict) and b.get("brand_name"):
                    brands.append(b)

        # PRD-203 O·S3: non-Shopify verticals (SaaS / services / content).
        elif page_type in ("services", "features"):
            for s in (
                (extract.get("industries_served") or [])
                + (extract.get("target_users") or [])
            ):
                if s and s not in sectors:
                    sectors.append(s)

        elif page_type == "case_study":
            for s in extract.get("industries_served") or []:
                if s and s not in sectors:
                    sectors.append(s)

        elif page_type == "article":
            category = extract.get("category")
            if category and category not in sectors:
                sectors.append(category)

        # Voice corpus contribution from any page with substantive markdown
        markdown = result.get("markdown")
        _voice_page_types = (
            "about", "solutions", "generic",
            "services", "features", "case_study", "article", "docs",
        )
        if markdown and len(markdown) > 200 and page_type in _voice_page_types:
            # First 400 chars as a voice sample
            voice_notes_chunks.append(markdown[:400].strip())

    # Quality finding: thin profile
    if not company_name:
        quality_findings["notes"].append("No company_name extracted from About page")
    if not sectors:
        quality_findings["notes"].append("No sectors detected — manual entry recommended")
    if not brands:
        quality_findings["notes"].append("No brands detected on the storefront")

    voice_notes = "\n\n".join(voice_notes_chunks[:5]) if voice_notes_chunks else None

    profile = {
        "domain": domain,
        "archetype": archetype_slug,
        "company_name": company_name,
        "sectors": sectors,
        "brands": brands,
        "standards": standards,
        "voice_notes": voice_notes,
        "goals": user_goals or [],
        "quality_findings": quality_findings,
    }

    logger.info(
        "intake.profile_built domain=%s archetype=%s sectors=%d brands=%d standards=%d",
        domain,
        archetype_slug,
        len(sectors),
        len(brands),
        len(standards),
    )
    return profile

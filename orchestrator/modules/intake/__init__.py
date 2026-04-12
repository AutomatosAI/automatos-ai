"""
PRD-130: Business Intake Wizard (PoC)
=====================================

Single-purpose package for the onboarding wizard:
  - firecrawl_client.py : domain-locked Firecrawl wrapper
  - archetypes.py       : URL-pattern archetype detection (Phase 1: Shopify)
  - schemas.py          : per-page-type JSON extract schemas
  - profile_builder.py  : assembles BusinessProfile from scrape results
  - plan_generator.py   : graph-cited Mission Zero draft plan

Phase 1 scope: wizard-only. NOT exposed as platform tools to agents.
"""

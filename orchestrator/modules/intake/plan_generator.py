"""
PRD-130: Mission Zero Goal Builder
===================================

Mission Zero is a *real mission* — not a locally synthesized draft plan.
This module exists only to translate a ``BusinessProfile`` into a rich
natural-language goal string that can be handed to
``CoordinatorService.create_mission()``. The real coordinator, planner,
dispatcher and agents then do all the actual research and planning work.

Everything else that used to live here (hardcoded dicts, LLM synthesis,
RAG passes, dossier building, citation hydration) is gone — the mission
system owns that responsibility end-to-end.
"""

from __future__ import annotations

from typing import Any, Iterable


def _csv(values: Iterable[Any] | None, limit: int = 10) -> str:
    """Join an iterable into a short comma-separated preview."""
    if not values:
        return ""
    items: list[str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, dict):
            label = v.get("brand_name") or v.get("name") or v.get("label")
            if label:
                items.append(str(label))
        else:
            s = str(v).strip()
            if s:
                items.append(s)
        if len(items) >= limit:
            break
    return ", ".join(items)


def build_mission_goal(
    profile: dict[str, Any],
    archetype_default_team: list[str] | None = None,
) -> str:
    """Render a Mission Zero goal string from a scraped business profile.

    The goal is the only thing the coordinator needs — it will decompose
    into tasks, spawn agents, and run the real mission lifecycle.
    """
    company = (profile.get("company_name") or profile.get("domain") or "the business").strip()
    domain = profile.get("domain") or ""
    archetype = profile.get("archetype") or "general"

    sectors = _csv(profile.get("sectors"))
    brands = _csv(profile.get("brands"), limit=6)
    standards = _csv(profile.get("standards"), limit=8)
    goals = _csv(profile.get("goals"), limit=8)
    voice = (profile.get("voice_notes") or "").strip()
    default_team = ", ".join(archetype_default_team or []) or "a balanced onboarding team"

    lines: list[str] = []
    lines.append(f"Mission Zero: bootstrap the Automatos workspace for {company}.")
    if domain:
        lines.append(f"Primary site: {domain}. Archetype detected: {archetype}.")
    if sectors:
        lines.append(f"Sectors served: {sectors}.")
    if brands:
        lines.append(f"Notable brands/products: {brands}.")
    if standards:
        lines.append(f"Standards & certifications in play: {standards}.")
    if voice:
        lines.append(f"Brand voice notes: {voice}")
    if goals:
        lines.append(f"User-stated priorities: {goals}.")

    lines.append(
        "Use the ingested corpus and the workspace knowledge graph as primary "
        "evidence — do not invent facts. Every finding must cite a real "
        "document chunk or graph node."
    )
    lines.append(
        "Deliver: (1) a concise business brief covering catalog, customers, "
        "compliance, operations, brand voice and integrations; (2) a proposed "
        f"onboarding team (starting from {default_team}) with clear "
        "responsibilities and tool assignments; (3) the highest-impact next "
        "actions the operator should approve; (4) any open questions the "
        "corpus could not answer."
    )
    lines.append(
        "When the mission completes, the output summary becomes the "
        "workspace's permanent onboarding brief."
    )

    return "\n\n".join(lines)

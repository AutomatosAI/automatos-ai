"""
PRD-130: Mission Zero Draft Plan Generator
============================================

Builds a draft business plan from:
  1. The wizard-built BusinessProfile (sectors/brands/standards/voice)
  2. The user's selected goals (Step 1)
  3. The graphify knowledge graph (post-RAG ingest)

Outputs a draft_plan dict with proposed agents, each carrying graph node IDs
as evidence — these become the "why we suggested this" citation chips in the UI.

Trust layer = citations. Every recommendation has receipts.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from modules.knowledge.graph_service import GraphifyService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Goal → candidate-agent map
# Wizard goals (Step 1) intersect with archetype.default_team to pick the team.
# Per "respects the brief" rule: INTERSECTION not union.
# ---------------------------------------------------------------------------
GOAL_TO_AGENTS: dict[str, set[str]] = {
    "manage":    {"shopify_ops", "catalog_hygiene"},
    "grow":      {"shopify_ops", "technical_sales", "brand_relations"},
    "market":    {"content_marketer", "brand_relations"},
    "advise":    {"technical_sales", "compliance"},
    "social":    {"content_marketer"},
    "compliance":{"compliance"},
}


# ---------------------------------------------------------------------------
# Agent specs — pre-baked persona + skills + tools per slug.
# Shopify Operations Manager is copied verbatim from SHOPIFY-AGENTS-SPEC.md.
# Other slugs are PoC stubs that can be tightened in Phase 2.
# ---------------------------------------------------------------------------
AGENT_SPECS: dict[str, dict[str, Any]] = {
    "shopify_ops": {
        "name": "Shopify Operations Manager",
        "team": "Commerce",
        "job_title": "Store Operations Manager",
        "persona": (
            "You are a Shopify Operations Manager AI agent. You manage day-to-day "
            "store operations: order fulfillment, inventory tracking, customer service, "
            "and admin tasks. You're proactive, detail-oriented, and customer-focused."
        ),
        "skills": [
            "shopify-admin",
            "shopify-admin-execution",
            "shopify-custom-data",
            "shopify-customer",
        ],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
    "catalog_hygiene": {
        "name": "Catalog Hygiene Agent",
        "team": "Commerce",
        "job_title": "Catalog QA Specialist",
        "persona": (
            "You audit the storefront catalog for duplicates, broken links, "
            "test products, and inconsistent metadata. You report findings and "
            "propose fixes for human approval."
        ),
        "skills": ["shopify-admin", "shopify-custom-data"],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
    "technical_sales": {
        "name": "Technical Sales Agent",
        "team": "Sales",
        "job_title": "Technical Sales Specialist",
        "persona": (
            "You answer pre-sale technical questions about products, "
            "compatibility, certifications, and applications. You cite catalog "
            "specs and brand documentation."
        ),
        "skills": ["shopify-admin", "shopify-customer"],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
    "compliance": {
        "name": "Compliance Agent",
        "team": "Operations",
        "job_title": "Compliance Officer",
        "persona": (
            "You monitor product listings, policies, and bulletins against the "
            "applicable industry standards and flag anything out of spec."
        ),
        "skills": ["shopify-admin"],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
    "content_marketer": {
        "name": "Content Marketer",
        "team": "Marketing",
        "job_title": "Content & Voice Lead",
        "persona": (
            "You produce on-brand content matching the company voice. "
            "You draft blog posts, email campaigns, and social copy."
        ),
        "skills": ["shopify-admin"],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
    "brand_relations": {
        "name": "Brand Relations Agent",
        "team": "Partnerships",
        "job_title": "Brand Partnerships Manager",
        "persona": (
            "You track brand partner inventory, promotions, and co-marketing "
            "opportunities. You surface relationship opportunities."
        ),
        "skills": ["shopify-admin"],
        "tools": ["composio:SHOPIFY"],
        "llm": "anthropic/claude-sonnet-4-6",
    },
}


# ---------------------------------------------------------------------------
# Graph evidence queries — keyword needles per agent.
# We do simple substring matching against node labels/types because graphify's
# query layer is intentionally lightweight in PoC.
# ---------------------------------------------------------------------------
EVIDENCE_NEEDLES: dict[str, list[str]] = {
    "shopify_ops":      ["shopify", "order", "fulfillment", "inventory"],
    "catalog_hygiene":  ["product", "collection", "duplicate", "test"],
    "technical_sales":  ["specification", "datasheet", "compliance", "certification"],
    "compliance":       ["standard", "regulation", "certification", "BS ", "EN "],
    "content_marketer": ["blog", "post", "newsletter", "voice"],
    "brand_relations":  ["brand", "partner", "supplier", "manufacturer"],
}


# ---------------------------------------------------------------------------
# Org chart heuristic
# ---------------------------------------------------------------------------
TEAM_LEADS: dict[str, str] = {
    "Commerce": "shopify_ops",
    "Sales": "technical_sales",
    "Operations": "compliance",
    "Marketing": "content_marketer",
    "Partnerships": "brand_relations",
}


def _query_graph_for_evidence(
    graph: nx.Graph | None,
    needles: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find graph nodes whose label or type contains any of the needles.

    Returns lightweight citation dicts: {id, label, type, snippet?}
    """
    if graph is None or graph.number_of_nodes() == 0 or not needles:
        return []

    needles_lower = [n.lower() for n in needles]
    citations: list[dict[str, Any]] = []

    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label") or attrs.get("name") or node_id).lower()
        node_type = str(attrs.get("type") or "").lower()
        haystack = f"{label} {node_type}"

        if any(needle in haystack for needle in needles_lower):
            citations.append(
                {
                    "id": str(node_id),
                    "label": attrs.get("label") or attrs.get("name") or node_id,
                    "type": attrs.get("type"),
                    "snippet": (attrs.get("description") or attrs.get("text") or "")[:200] or None,
                }
            )
            if len(citations) >= limit:
                break

    return citations


async def generate_draft_plan(
    profile: dict[str, Any],
    archetype_default_team: list[str],
    workspace_id: str,
    graphify_service: GraphifyService | None = None,
) -> dict[str, Any]:
    """Build the Mission Zero draft plan.

    Args:
        profile: business_profiles row as a dict
        archetype_default_team: agent slugs from the detected archetype
        workspace_id: workspace UUID (string)
        graphify_service: injected GraphifyService (None → no citations, plan still works)

    Returns:
        draft_plan dict with proposed_agents, org_chart, integrations_needed,
        open_questions, generated_at
    """
    user_goals: list[str] = profile.get("goals") or []

    # Step 1: intersection of goals and archetype default team.
    # If user picked no goals, fall back to archetype default (gentle Phase 1 UX).
    if user_goals:
        goal_agents: set[str] = set()
        for goal in user_goals:
            goal_agents |= GOAL_TO_AGENTS.get(goal.lower(), set())
        candidates = [a for a in archetype_default_team if a in goal_agents]
        if not candidates:
            candidates = list(archetype_default_team)
    else:
        candidates = list(archetype_default_team)

    # Step 2: load graph for citation lookup (best-effort — None if not built yet)
    graph: nx.Graph | None = None
    if graphify_service is not None:
        try:
            graph = await graphify_service.load_graph(workspace_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_generator: failed to load graph: %s", exc)

    # Step 3: build proposed_agents with evidence
    proposed_agents: list[dict[str, Any]] = []
    for slug in candidates:
        spec = AGENT_SPECS.get(slug)
        if not spec:
            logger.warning("plan_generator: no spec for agent slug=%s, skipping", slug)
            continue

        needles = EVIDENCE_NEEDLES.get(slug, [])
        citations = _query_graph_for_evidence(graph, needles)

        # Goal-aligned rationale
        matching_goals = [
            g for g in user_goals
            if slug in GOAL_TO_AGENTS.get(g.lower(), set())
        ]
        if matching_goals:
            rationale = (
                f"Proposed because you selected {', '.join(matching_goals)} "
                f"as a goal, and your site shows signals matching this role."
            )
        else:
            rationale = "Proposed based on the detected business archetype."

        if citations:
            rationale += f" Evidence: {len(citations)} relevant graph nodes."

        proposed_agents.append(
            {
                "slug": slug,
                "name": spec["name"],
                "team": spec["team"],
                "job_title": spec["job_title"],
                "persona": spec["persona"],
                "skills": spec["skills"],
                "tools": spec["tools"],
                "llm": spec["llm"],
                "rationale": rationale,
                "citations": citations,
            }
        )

    # Step 4: org chart — each agent reports to its team lead (lead reports to None)
    org_chart: list[dict[str, Any]] = []
    proposed_slugs = {a["slug"] for a in proposed_agents}
    for agent in proposed_agents:
        team = agent["team"]
        lead = TEAM_LEADS.get(team)
        reports_to = lead if (lead and lead != agent["slug"] and lead in proposed_slugs) else None
        org_chart.append({"agent": agent["slug"], "reports_to": reports_to})

    # Step 5: integrations + open questions
    integrations_needed: list[str] = []
    if profile.get("archetype") == "shopify_catalog":
        integrations_needed.append("composio:SHOPIFY")

    open_questions: list[str] = []
    if not profile.get("company_name"):
        open_questions.append("What is the registered legal name of your business?")
    if not profile.get("sectors"):
        open_questions.append("Which industries/sectors do you primarily serve?")

    draft_plan = {
        "proposed_agents": proposed_agents,
        "org_chart": org_chart,
        "integrations_needed": integrations_needed,
        "open_questions": open_questions,
        "graph_available": graph is not None,
        "graph_node_count": graph.number_of_nodes() if graph is not None else 0,
    }

    logger.info(
        "plan_generator: built draft plan workspace=%s agents=%d cited=%d",
        workspace_id,
        len(proposed_agents),
        sum(1 for a in proposed_agents if a["citations"]),
    )
    return draft_plan

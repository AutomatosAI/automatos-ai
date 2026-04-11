"""
PRD-130: Mission Zero Draft Plan Generator (prompt-driven)
============================================================

Builds a draft business plan by *reading* the ingested corpus — not pattern-
matching hardcoded dicts. Flow:

  1. Load business profile + wizard goals + archetype default team
  2. Research pass: for a fixed set of business-discovery questions, pull
     - RAG hits  via  DocumentManager.search_documents
     - Graph hits via GraphifyService.score_nodes + bfs + subgraph_to_text
  3. Build a *dossier* — compact text + an ID index of every finding
  4. Load the prompt template from prompt_registry (slug: "mission-zero-synthesizer")
  5. One LLM call → JSON with proposed agents, integrations, open questions
  6. Extract JSON, validate with Pydantic
  7. Hydrate citations server-side from the dossier ID index — the LLM
     cannot invent node IDs or doc IDs, it can only *select* from the dossier
  8. Build the org chart heuristically from team assignments

Trust layer = citations. Every recommendation points at real nodes/docs
the user can click through.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from core.llm import create_llm_manager
from core.services.prompt_registry import prompt_registry
from modules.knowledge.graph_service import GraphifyService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Research questions — fixed set used to probe RAG + graph
# ---------------------------------------------------------------------------

RESEARCH_QUESTIONS: list[tuple[str, str, list[str]]] = [
    # (topic_key, question, graph_score_terms)
    (
        "catalog",
        "What products, brands, services or SKUs does this business sell?",
        ["product", "brand", "catalog", "sku", "collection"],
    ),
    (
        "customers",
        "Who are the customers? What segments, markets or industries does it serve?",
        ["customer", "segment", "industry", "market", "sector"],
    ),
    (
        "compliance",
        "What compliance standards, certifications or regulations apply to this business?",
        ["standard", "regulation", "certification", "compliance", "BS", "EN", "ISO"],
    ),
    (
        "operations",
        "What operational workflows are described: orders, fulfillment, inventory, support?",
        ["order", "fulfillment", "inventory", "shipping", "support", "workflow"],
    ),
    (
        "voice",
        "What tone, voice, brand positioning or marketing angle is evident?",
        ["voice", "tone", "brand", "marketing", "positioning", "content"],
    ),
    (
        "integrations",
        "What platforms, tools or integrations does this business already use?",
        ["shopify", "platform", "integration", "api", "system", "tool"],
    ),
]

# Limits — small enough to keep dossier under token budget
_RAG_LIMIT_PER_QUESTION = 4
_GRAPH_SCORE_LIMIT = 3
_GRAPH_BFS_DEPTH = 1
_SUBGRAPH_TOKEN_BUDGET = 400

# Dossier text cap (chars) — keeps total LLM prompt predictable
_DOSSIER_MAX_CHARS = 14_000


# ---------------------------------------------------------------------------
# Pydantic schemas — the LLM response shape
# ---------------------------------------------------------------------------


class _LLMAgent(BaseModel):
    slug: str
    name: str
    team: str
    job_title: str
    persona: str
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    llm: str = "anthropic/claude-sonnet-4-6"
    rationale: str
    citation_node_ids: list[str] = Field(default_factory=list)
    citation_doc_ids: list[str] = Field(default_factory=list)


class _LLMPlan(BaseModel):
    proposed_agents: list[_LLMAgent]
    integrations_needed: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dossier: a compact, cited snapshot of what's in the corpus
# ---------------------------------------------------------------------------


class _Dossier:
    """Compact corpus snapshot handed to the LLM, with an ID index for
    server-side citation hydration (so the LLM can't invent IDs)."""

    def __init__(self) -> None:
        self.sections: list[str] = []
        self.graph_nodes: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, dict[str, Any]] = {}

    def add_section(self, text: str) -> None:
        self.sections.append(text)

    def register_node(self, node_id: str, attrs: dict[str, Any]) -> None:
        if node_id and node_id not in self.graph_nodes:
            self.graph_nodes[node_id] = {
                "id": node_id,
                "label": attrs.get("label") or attrs.get("name") or node_id,
                "type": attrs.get("type"),
                "snippet": (attrs.get("description") or attrs.get("text") or "")[:240] or None,
            }

    def register_doc(self, chunk: dict[str, Any]) -> str:
        """Register a RAG chunk; return the dossier-local doc_id used in the prompt."""
        doc_id = f"doc:{chunk.get('chunk_id')}"
        if doc_id not in self.docs:
            self.docs[doc_id] = {
                "id": doc_id,
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "file_type": chunk.get("file_type"),
                "snippet": (chunk.get("content") or "")[:300],
            }
        return doc_id

    def to_text(self) -> str:
        body = "\n\n".join(self.sections)
        if len(body) > _DOSSIER_MAX_CHARS:
            body = body[:_DOSSIER_MAX_CHARS] + "\n… [dossier truncated]"
        return body


# ---------------------------------------------------------------------------
# Research pass
# ---------------------------------------------------------------------------


async def _run_research_pass(
    workspace_id: str,
    graph: nx.Graph | None,
    graphify_service: GraphifyService | None,
) -> _Dossier:
    dossier = _Dossier()
    loop = asyncio.get_event_loop()

    # Set up DocumentManager (sync, must run in executor because it calls
    # asyncio.run() internally for embedding — would blow up in our loop).
    doc_manager = None
    try:
        from api.documents import get_document_manager

        doc_manager = get_document_manager(workspace_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_generator: DocumentManager unavailable: %s", exc)

    for topic, question, terms in RESEARCH_QUESTIONS:
        section_lines = [f"### {topic.upper()} — {question}"]

        # --- RAG lookup ------------------------------------------------------
        rag_hits: list[dict[str, Any]] = []
        if doc_manager is not None:
            try:
                rag_hits = await loop.run_in_executor(
                    None,
                    lambda q=question: doc_manager.search_documents(q, limit=_RAG_LIMIT_PER_QUESTION),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("plan_generator: RAG search failed for '%s': %s", topic, exc)

        if rag_hits:
            section_lines.append("- RAG evidence:")
            for chunk in rag_hits:
                doc_id = dossier.register_doc(chunk)
                filename = chunk.get("filename") or "unknown"
                snippet = (chunk.get("content") or "").strip().replace("\n", " ")[:200]
                section_lines.append(f"  * [{doc_id}] {filename}: {snippet}")
        else:
            section_lines.append("- RAG evidence: (none)")

        # --- Graph lookup ----------------------------------------------------
        if graph is not None and graph.number_of_nodes() > 0 and graphify_service is not None:
            try:
                scored = await graphify_service.score_nodes(graph, terms)
            except Exception as exc:  # noqa: BLE001
                logger.warning("plan_generator: score_nodes failed for '%s': %s", topic, exc)
                scored = []

            top = scored[:_GRAPH_SCORE_LIMIT]
            if top:
                section_lines.append("- Graph evidence:")
                for hit in top:
                    node_id = str(hit.get("id"))
                    if not node_id or node_id not in graph:
                        continue
                    dossier.register_node(node_id, graph.nodes[node_id])
                    label = hit.get("label") or node_id
                    section_lines.append(f"  * [{node_id}] {label} (score={hit.get('score', 0):.2f})")

                    # Expand context with a small BFS + summary
                    try:
                        bfs = await graphify_service.bfs(graph, node_id, depth=_GRAPH_BFS_DEPTH)
                        if bfs.get("nodes"):
                            for nid in bfs["nodes"]:
                                if nid in graph:
                                    dossier.register_node(str(nid), graph.nodes[nid])
                            summary = await graphify_service.subgraph_to_text(
                                graph,
                                bfs["nodes"],
                                bfs["edges"],
                                token_budget=_SUBGRAPH_TOKEN_BUDGET,
                            )
                            if summary:
                                section_lines.append(f"    context: {summary.strip()[:400]}")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("plan_generator: bfs/subgraph failed for %s: %s", node_id, exc)
            else:
                section_lines.append("- Graph evidence: (no high-score nodes)")
        else:
            section_lines.append("- Graph evidence: (graph unavailable)")

        dossier.add_section("\n".join(section_lines))

    return dossier


# ---------------------------------------------------------------------------
# Prompt + LLM
# ---------------------------------------------------------------------------


_DEFAULT_SYNTHESIZER_PROMPT = """You are the Mission Zero Synthesizer. Your job is to design a small team of AI agents that will run this business, grounded ONLY in the dossier below.

## Business profile
{profile_json}

## User-selected goals
{goals_csv}

## Detected archetype
{archetype}

## Archetype default team (suggestion only — you may add, remove, or rename)
{default_team_csv}

## Dossier (research findings from the live corpus)
Each finding is tagged with a stable ID in square brackets, e.g. [doc:123] or [nodestem_xyz]. When you cite evidence, you MUST use these exact IDs. Do not invent IDs.

{dossier}

## Your task
Propose between 3 and 6 AI agents that together cover the goals above. For each agent, return:

- slug: snake_case identifier, e.g. "shopify_ops"
- name: human-friendly name, e.g. "Shopify Operations Manager"
- team: one of Commerce, Sales, Operations, Marketing, Partnerships, or a custom team
- job_title: short title, e.g. "Store Operations Manager"
- persona: 2-3 sentence system prompt describing the agent's voice and behaviour, grounded in the dossier's tone/voice findings
- skills: array of platform skill slugs (e.g. "shopify-admin", "shopify-customer") — only include skills actually needed
- tools: array of tool identifiers (e.g. "composio:SHOPIFY") — only include what the dossier justifies
- llm: a model ID, default "anthropic/claude-sonnet-4-6"
- rationale: 1-2 sentences explaining WHY this agent is needed, referencing specific dossier findings
- citation_node_ids: array of graph node IDs from the dossier (stems, not square brackets) that justify this agent
- citation_doc_ids: array of doc IDs from the dossier (e.g. "doc:123") that justify this agent

Also return:
- integrations_needed: array of integration slugs the team will need (e.g. "composio:SHOPIFY")
- open_questions: array of 0-5 short questions you still need answered before this plan can run

## Rules
- Ground every choice in the dossier. If the dossier shows no evidence for a capability, do NOT propose an agent for it.
- You may depart from the archetype default team when the dossier justifies it.
- Keep personas tight — 2-3 sentences max.
- citation_node_ids and citation_doc_ids MUST be IDs that actually appear in the dossier. Hallucinated IDs will be dropped.
- Respond with ONE JSON object, no markdown fences, no commentary.

## Response schema
{{
  "proposed_agents": [
    {{
      "slug": "...",
      "name": "...",
      "team": "...",
      "job_title": "...",
      "persona": "...",
      "skills": ["..."],
      "tools": ["..."],
      "llm": "anthropic/claude-sonnet-4-6",
      "rationale": "...",
      "citation_node_ids": ["..."],
      "citation_doc_ids": ["doc:..."]
    }}
  ],
  "integrations_needed": ["..."],
  "open_questions": ["..."]
}}
"""


def _register_default_prompt() -> None:
    """Install the Mission Zero prompt into the registry's hardcoded defaults
    so it resolves even before the prompt is seeded to the DB."""
    try:
        from core.services import prompt_registry as pr_module

        defaults = getattr(pr_module, "_HARDCODED_DEFAULTS", None)
        if isinstance(defaults, dict) and "mission-zero-synthesizer" not in defaults:
            defaults["mission-zero-synthesizer"] = _DEFAULT_SYNTHESIZER_PROMPT
    except Exception as exc:  # noqa: BLE001
        logger.debug("plan_generator: could not register default prompt: %s", exc)


_register_default_prompt()


def _extract_json(content: str) -> dict[str, Any] | None:
    """Extract a JSON object from LLM output, tolerating fences + preamble."""
    if not content:
        return None

    block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
    text = block_match.group(1).strip() if block_match else content.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Citation hydration + org chart
# ---------------------------------------------------------------------------


def _hydrate_citations(
    agent: _LLMAgent,
    dossier: _Dossier,
) -> list[dict[str, Any]]:
    """Turn LLM-supplied ID lists into full citation dicts, dropping anything
    the LLM invented that isn't in the dossier."""
    citations: list[dict[str, Any]] = []

    for node_id in agent.citation_node_ids:
        node = dossier.graph_nodes.get(str(node_id))
        if node:
            citations.append({
                "kind": "graph",
                "id": node["id"],
                "label": node["label"],
                "type": node.get("type"),
                "snippet": node.get("snippet"),
            })

    for doc_id in agent.citation_doc_ids:
        doc = dossier.docs.get(str(doc_id))
        if doc:
            citations.append({
                "kind": "doc",
                "id": doc["id"],
                "label": doc.get("filename") or doc["id"],
                "type": doc.get("file_type"),
                "snippet": doc.get("snippet"),
            })

    return citations


def _build_org_chart(proposed_agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simple heuristic: first agent per team is the lead; others report to them."""
    team_leads: dict[str, str] = {}
    for agent in proposed_agents:
        team = agent.get("team") or "General"
        if team not in team_leads:
            team_leads[team] = agent["slug"]

    chart: list[dict[str, Any]] = []
    for agent in proposed_agents:
        team = agent.get("team") or "General"
        lead = team_leads.get(team)
        reports_to = lead if lead and lead != agent["slug"] else None
        chart.append({"agent": agent["slug"], "reports_to": reports_to})
    return chart


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def generate_draft_plan(
    profile: dict[str, Any],
    archetype_default_team: list[str],
    workspace_id: str,
    graphify_service: GraphifyService | None = None,
) -> dict[str, Any]:
    """Build the Mission Zero draft plan via research + LLM synthesis.

    Args:
        profile: business_profiles row as a dict
        archetype_default_team: agent slugs from the detected archetype (suggestion)
        workspace_id: workspace UUID (string)
        graphify_service: injected GraphifyService (optional — dossier gracefully
            degrades when unavailable)

    Returns:
        draft_plan dict with proposed_agents, org_chart, integrations_needed,
        open_questions, and dossier metadata.
    """
    # ---- 1. Load the graph (best effort) -----------------------------------
    graph: nx.Graph | None = None
    if graphify_service is not None:
        try:
            graph = await graphify_service.load_graph(workspace_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_generator: failed to load graph: %s", exc)

    # ---- 2. Research pass: fill the dossier --------------------------------
    dossier = await _run_research_pass(workspace_id, graph, graphify_service)

    # ---- 3. Render the prompt ----------------------------------------------
    user_goals: list[str] = profile.get("goals") or []
    prompt = prompt_registry.get(
        "mission-zero-synthesizer",
        profile_json=json.dumps(profile, default=str, indent=2),
        goals_csv=", ".join(user_goals) if user_goals else "(none selected)",
        archetype=profile.get("archetype") or "(unknown)",
        default_team_csv=", ".join(archetype_default_team) if archetype_default_team else "(none)",
        dossier=dossier.to_text() or "(no corpus content yet)",
    )

    if not prompt:
        raise RuntimeError("mission-zero-synthesizer prompt is not registered")

    # ---- 4. LLM call --------------------------------------------------------
    llm = create_llm_manager(service_name="orchestrator")
    llm.config.max_tokens = 8000
    messages = [
        {"role": "system", "content": "You respond with strict, parseable JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await llm.generate_response(messages)
    except Exception as exc:
        logger.exception("plan_generator: LLM call failed")
        raise RuntimeError(f"Mission Zero LLM call failed: {exc}") from exc

    raw = _extract_json(getattr(response, "content", "") or "")
    if raw is None:
        logger.error(
            "plan_generator: LLM returned no parseable JSON. content=%r",
            getattr(response, "content", "")[:500],
        )
        raise RuntimeError("Mission Zero synthesis returned no JSON")

    try:
        llm_plan = _LLMPlan.model_validate(raw)
    except ValidationError as exc:
        logger.error("plan_generator: LLM JSON failed validation: %s", exc)
        raise RuntimeError(f"Mission Zero synthesis JSON invalid: {exc}") from exc

    # ---- 5. Hydrate citations + build final shape --------------------------
    proposed_agents: list[dict[str, Any]] = []
    for llm_agent in llm_plan.proposed_agents:
        citations = _hydrate_citations(llm_agent, dossier)
        proposed_agents.append({
            "slug": llm_agent.slug,
            "name": llm_agent.name,
            "team": llm_agent.team,
            "job_title": llm_agent.job_title,
            "persona": llm_agent.persona,
            "skills": llm_agent.skills,
            "tools": llm_agent.tools,
            "llm": llm_agent.llm,
            "rationale": llm_agent.rationale,
            "citations": citations,
        })

    org_chart = _build_org_chart(proposed_agents)

    # ---- 6. Open questions: always include profile gaps --------------------
    open_questions = list(llm_plan.open_questions)
    if not profile.get("company_name"):
        open_questions.insert(0, "What is the registered legal name of your business?")
    if not profile.get("sectors"):
        open_questions.insert(0, "Which industries or sectors do you primarily serve?")

    draft_plan = {
        "proposed_agents": proposed_agents,
        "org_chart": org_chart,
        "integrations_needed": llm_plan.integrations_needed,
        "open_questions": open_questions,
        "graph_available": graph is not None,
        "graph_node_count": graph.number_of_nodes() if graph is not None else 0,
        "dossier_doc_count": len(dossier.docs),
        "dossier_node_count": len(dossier.graph_nodes),
    }

    logger.info(
        "plan_generator: built draft plan workspace=%s agents=%d cited=%d dossier_docs=%d dossier_nodes=%d",
        workspace_id,
        len(proposed_agents),
        sum(1 for a in proposed_agents if a["citations"]),
        len(dossier.docs),
        len(dossier.graph_nodes),
    )
    return draft_plan

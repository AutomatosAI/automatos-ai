"""
Knowledge Graph Extraction
===========================

Extracts structured knowledge graphs from documents, reports, and platform
metadata using LLM-based extraction (documents/reports) and deterministic
mappers (agents, blueprints, schemas, connected apps).

Every function returns ``{"nodes": [...], "edges": [...], "hyperedges": [...]}``.
"""

import json
import logging
import re
from typing import Any

from core.llm import create_llm_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _make_id(*parts: str) -> str:
    """Join *parts*, lowercase, replace non-alphanumeric runs with ``_``."""
    raw = "_".join(parts).lower()
    return _NON_ALNUM.sub("_", raw).strip("_")


def _empty_graph() -> dict[str, list]:
    return {"nodes": [], "edges": [], "hyperedges": []}


def _node(
    node_id: str,
    label: str,
    file_type: str,
    source_file: str,
    *,
    source_location: str | None = None,
    confidence: str = "EXTRACTED",
    weight: float = 1.0,
    team_access: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "file_type": file_type,
        "source_file": source_file,
        "source_location": source_location,
        "confidence": confidence,
        "weight": weight,
        "team_access": team_access or [],
    }


def _edge(
    source: str,
    target: str,
    relation: str,
    source_file: str,
    *,
    confidence: str = "EXTRACTED",
    confidence_score: float = 1.0,
    source_location: str | None = None,
    weight: float = 1.0,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "source_file": source_file,
        "source_location": source_location,
        "weight": weight,
        "_src": source,
        "_tgt": target,
    }


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_DOCUMENT_EXTRACTION_PROMPT = """\
You are extracting a knowledge graph from a business document.

Given the document below, extract:
1. CONCEPTS — abstract ideas, strategies, goals mentioned
2. ENTITIES — named things: products, companies, people, tools, services
3. PROCESSES — workflows, procedures, sequences described
4. METRICS — measurable quantities referenced
5. RULES — constraints, policies, thresholds stated
6. RELATIONSHIPS — how the above connect to each other

Output JSON:
{
  "nodes": [
    {"id": "snake_case_id", "label": "Human Name", "file_type": "concept|entity|process|metric|rule", "source_file": "<doc_path>"}
  ],
  "edges": [
    {"source": "node_id_a", "target": "node_id_b", "relation": "<relation_type>", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", "confidence_score": 0.85}
  ],
  "hyperedges": [
    {"id": "snake_case_id", "label": "Human Label", "nodes": ["id1", "id2", "id3"], "relation": "participate_in|implement|form", "confidence": "EXTRACTED|INFERRED", "confidence_score": 0.9, "source_file": "<doc_path>"}
  ]
}

Rules:
- Use snake_case IDs derived from the label
- Only create edges where the relationship is clearly stated or strongly implied
- Mark directly stated relationships as EXTRACTED (confidence_score: 1.0)
- Mark implied relationships as INFERRED with a per-edge confidence_score:
  - 0.8–0.9: strong structural evidence (shared data, clear dependency)
  - 0.6–0.7: reasonable inference with some uncertainty
  - 0.4–0.5: weak or speculative. Never default to 0.5 — reason about each edge.
- Mark uncertain relationships as AMBIGUOUS (confidence_score: 0.1–0.3)
- Do not hallucinate entities not present in the document
- Prefer specific labels over generic ones ("30-Day Refund Window" not "Time Limit")
- Add hyperedges when 3+ nodes participate in a shared concept/flow/pattern. Maximum 3 per document.

DOCUMENT PATH: {doc_path}

---
{doc_text}
---

Respond ONLY with the JSON object, no commentary.
"""

_REPORT_EXTRACTION_PROMPT = """\
You are extracting a knowledge graph from an agent report.

Given the report below (authored by agent "{agent_name}"), extract:
1. ENTITIES — tools, services, data sources, targets mentioned
2. ACTIONS — tasks performed, API calls made, decisions taken
3. OUTCOMES — results, metrics produced, statuses reported
4. ISSUES — errors, warnings, blockers encountered
5. RELATIONSHIPS — how the above connect to each other

Output JSON:
{{
  "nodes": [
    {{"id": "snake_case_id", "label": "Human Name", "file_type": "entity|action|outcome|issue", "source_file": "<report_path>"}}
  ],
  "edges": [
    {{"source": "node_id_a", "target": "node_id_b", "relation": "<relation_type>", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", "confidence_score": 0.85}}
  ],
  "hyperedges": [
    {{"id": "snake_case_id", "label": "Human Label", "nodes": ["id1", "id2", "id3"], "relation": "participate_in|implement|form", "confidence": "EXTRACTED|INFERRED", "confidence_score": 0.9, "source_file": "<report_path>"}}
  ]
}}

Rules:
- Use snake_case IDs derived from the label
- Only create edges where the relationship is clearly stated or strongly implied
- Mark directly stated relationships as EXTRACTED (confidence_score: 1.0)
- Mark implied relationships as INFERRED with a per-edge confidence_score
- Mark uncertain relationships as AMBIGUOUS (confidence_score: 0.1–0.3)
- Do not hallucinate entities not present in the report
- Add hyperedges when 3+ nodes participate in a shared concept/flow/pattern. Maximum 3 per report.

REPORT PATH: {report_path}
AGENT: {agent_name}

---
{report_text}
---

Respond ONLY with the JSON object, no commentary.
"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)


def _parse_llm_json(raw: str) -> dict[str, list] | None:
    """Strip optional code fences and parse JSON. Returns None on failure."""
    cleaned = _CODE_FENCE_RE.sub("", raw).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse LLM extraction JSON: %s — raw[:200]: %s", exc, raw[:200])
    return None


def _normalise_extraction(
    raw: dict,
    source_file: str,
    team_access: list[str] | None = None,
) -> dict[str, list]:
    """Ensure every node/edge/hyperedge has all required fields.

    Args:
        raw: Parsed LLM JSON output.
        source_file: Provenance path for nodes/edges.
        team_access: PRD-124 team scoping. Nodes inherit this from their
            source document. Empty list = visible to all agents.
    """
    result = _empty_graph()

    for n in raw.get("nodes", []):
        result["nodes"].append(_node(
            node_id=n.get("id", _make_id(n.get("label", "unknown"))),
            label=n.get("label", n.get("id", "unknown")),
            file_type=n.get("file_type", "entity"),
            source_file=n.get("source_file", source_file),
            confidence=n.get("confidence", "EXTRACTED"),
            weight=float(n.get("weight", 1.0)),
            team_access=team_access,
        ))

    for e in raw.get("edges", []):
        src = e.get("source", "")
        tgt = e.get("target", "")
        result["edges"].append(_edge(
            source=src,
            target=tgt,
            relation=e.get("relation", "related_to"),
            source_file=e.get("source_file", source_file),
            confidence=e.get("confidence", "INFERRED"),
            confidence_score=float(e.get("confidence_score", 0.5)),
            weight=float(e.get("weight", 1.0)),
        ))

    for h in raw.get("hyperedges", []):
        result["hyperedges"].append({
            "id": h.get("id", _make_id(h.get("label", "group"))),
            "label": h.get("label", ""),
            "nodes": h.get("nodes", []),
            "relation": h.get("relation", "participate_in"),
            "confidence": h.get("confidence", "INFERRED"),
            "confidence_score": float(h.get("confidence_score", 0.5)),
            "source_file": h.get("source_file", source_file),
        })

    return result


# ---------------------------------------------------------------------------
# LLM-based extractors
# ---------------------------------------------------------------------------

async def extract_from_document(
    doc_text: str,
    doc_path: str,
    workspace_id: int,
    team_access: list[str] | None = None,
) -> dict[str, list]:
    """Extract knowledge graph from a business document via LLM.

    Args:
        doc_text: Full text of the document.
        doc_path: Path/identifier for provenance tracking.
        workspace_id: Workspace that owns the document (for future scoping).
        team_access: Teams that can access this document (PRD-124).
            Empty list or None means visible to all agents.

    Returns:
        ``{"nodes": [...], "edges": [...], "hyperedges": [...]}``
    """
    if not doc_text or not doc_text.strip():
        logger.warning("extract_from_document called with empty text for %s", doc_path)
        return _empty_graph()

    prompt = _DOCUMENT_EXTRACTION_PROMPT.format(doc_path=doc_path, doc_text=doc_text)

    try:
        llm = create_llm_manager(service_name="orchestrator")
        response = await llm.generate_response([
            {"role": "system", "content": "You are a knowledge-graph extraction engine. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ])
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("LLM call failed during document extraction for %s", doc_path)
        return _empty_graph()

    parsed = _parse_llm_json(raw_text)
    if parsed is None:
        return _empty_graph()

    return _normalise_extraction(parsed, source_file=doc_path, team_access=team_access)


async def extract_from_report(
    report_text: str,
    report_path: str,
    agent_name: str,
) -> dict[str, list]:
    """Extract knowledge graph from an agent report via LLM.

    Args:
        report_text: Full text of the report.
        report_path: Path/identifier for provenance tracking.
        agent_name: Name of the agent that authored the report.

    Returns:
        ``{"nodes": [...], "edges": [...], "hyperedges": [...]}``
    """
    if not report_text or not report_text.strip():
        logger.warning("extract_from_report called with empty text for %s", report_path)
        return _empty_graph()

    prompt = _REPORT_EXTRACTION_PROMPT.format(
        report_path=report_path,
        report_text=report_text,
        agent_name=agent_name,
    )

    try:
        llm = create_llm_manager(service_name="orchestrator")
        response = await llm.generate_response([
            {"role": "system", "content": "You are a knowledge-graph extraction engine. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ])
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("LLM call failed during report extraction for %s", report_path)
        return _empty_graph()

    parsed = _parse_llm_json(raw_text)
    if parsed is None:
        return _empty_graph()

    return _normalise_extraction(parsed, source_file=report_path)


# ---------------------------------------------------------------------------
# Deterministic mappers (no LLM)
# ---------------------------------------------------------------------------

_ROSTER_SOURCE = "platform://agent_roster"
_BLUEPRINT_SOURCE = "platform://blueprints"
_SCHEMA_SOURCE = "platform://db_schemas"
_APPS_SOURCE = "platform://connected_apps"


def map_agent_roster(agents: list[dict]) -> dict[str, list]:
    """Map agent roster rows into graph nodes and ``reports_to`` edges.

    Each dict in *agents* should have at minimum ``id``, ``name``.
    Optional: ``reports_to`` (agent id), ``role``, ``skill``.
    """
    result = _empty_graph()
    for agent in agents:
        aid = _make_id("agent", str(agent.get("id", agent.get("name", "unknown"))))
        result["nodes"].append(_node(
            node_id=aid,
            label=agent.get("name", aid),
            file_type="agent",
            source_file=_ROSTER_SOURCE,
        ))
        reports_to = agent.get("reports_to")
        if reports_to is not None:
            target_id = _make_id("agent", str(reports_to))
            result["edges"].append(_edge(
                source=aid,
                target=target_id,
                relation="reports_to",
                source_file=_ROSTER_SOURCE,
            ))
    return result


def map_blueprints(blueprints: list[dict]) -> dict[str, list]:
    """Map blueprint/rule definitions into graph nodes and ``constrained_by`` edges.

    Each dict should have ``id``/``name`` and optionally ``agents`` (list of
    agent ids/names the rule constrains).
    """
    result = _empty_graph()
    for bp in blueprints:
        rid = _make_id("rule", str(bp.get("id", bp.get("name", "unknown"))))
        result["nodes"].append(_node(
            node_id=rid,
            label=bp.get("name", rid),
            file_type="rule",
            source_file=_BLUEPRINT_SOURCE,
        ))
        for agent_ref in bp.get("agents", []):
            agent_id = _make_id("agent", str(agent_ref))
            result["edges"].append(_edge(
                source=agent_id,
                target=rid,
                relation="constrained_by",
                source_file=_BLUEPRINT_SOURCE,
            ))
    return result


def map_db_schemas(schemas: list[dict]) -> dict[str, list]:
    """Map database table schemas into graph nodes and FK ``depends_on`` edges.

    Each dict should have ``table_name`` and optionally ``foreign_keys``
    (list of dicts with ``target_table``).
    """
    result = _empty_graph()
    for schema in schemas:
        table_name = schema.get("table_name", "unknown")
        tid = _make_id("table", table_name)
        result["nodes"].append(_node(
            node_id=tid,
            label=table_name,
            file_type="table",
            source_file=_SCHEMA_SOURCE,
        ))
        for fk in schema.get("foreign_keys", []):
            target_table = fk.get("target_table", "")
            if target_table:
                target_id = _make_id("table", target_table)
                result["edges"].append(_edge(
                    source=tid,
                    target=target_id,
                    relation="depends_on",
                    source_file=_SCHEMA_SOURCE,
                ))
    return result


def map_connected_apps(apps: list[dict]) -> dict[str, list]:
    """Map connected app/integration entries into graph nodes.

    Each dict should have ``id``/``name`` and optionally ``app_type``.
    """
    result = _empty_graph()
    for app in apps:
        app_id = _make_id("integration", str(app.get("id", app.get("name", "unknown"))))
        result["nodes"].append(_node(
            node_id=app_id,
            label=app.get("name", app_id),
            file_type="integration",
            source_file=_APPS_SOURCE,
        ))
    return result

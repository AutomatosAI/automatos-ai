"""Build-time LLM community reports for the knowledge graph (PRD-165 S3).

GraphRAG pattern: for the most significant communities, a cheap LLM names the
cluster and writes a one-line summary. Titles/summaries are persisted into
``communities.json`` at build time and surfaced to users (the cluster sidebar)
and agents (``platform_graph_communities``).

Ranking is by community size — deterministic and free, no extra graph passes —
and only the top-N communities are titled, so build cost stays bounded (D11:
top-N, member cap, model, timeout, and concurrency are all config, not
hardcoded). The LLM is injectable so tests stay deterministic without a network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_setting(key: str, default: str) -> str:
    from core.llm.manager import get_system_setting
    return get_system_setting("knowledge_graph", key, default)


def _community_model() -> str:
    """Cheap model for titling (D11). A dedicated setting wins; otherwise reuse
    the graph-extraction model."""
    configured = _get_setting("community_title_model", "")
    if configured:
        return configured
    from modules.knowledge.graph_extraction import _get_graph_extraction_model
    return _get_graph_extraction_model()


_TITLE_PROMPT = (
    "You are naming a cluster of related concepts from a knowledge graph.\n"
    "Given the member labels below, respond with STRICT JSON only:\n"
    '{{"title": "<2-4 word title>", "summary": "<one sentence, <=20 words>"}}\n\n'
    "Members:\n{members}"
)


def _member_labels(graph, member_ids: List[str], cap: int) -> List[str]:
    """Top member labels for a community, most-connected first (the cap keeps
    the prompt — and the cost — bounded)."""
    present = [m for m in member_ids if m in graph]
    present.sort(key=lambda n: graph.degree(n), reverse=True)
    return [str(graph.nodes[n].get("label", n)) for n in present[:cap]]


def _parse_report(raw: str) -> Optional[Dict[str, str]]:
    """Tolerantly parse {title, summary} out of an LLM response (handles code
    fences / surrounding prose). None when there's no usable title."""
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    title = str(data.get("title", "")).strip()
    summary = str(data.get("summary", "")).strip()
    if not title:
        return None
    return {"title": title[:80], "summary": summary[:240]}


async def generate_community_reports(
    graph,
    communities: Dict[int, List[str]],
    *,
    top_n: Optional[int] = None,
    member_cap: Optional[int] = None,
    llm: Any = None,
) -> Dict[int, Dict[str, Any]]:
    """Return ``{community_id: {rank, title?, summary?}}``.

    Every community gets a ``rank`` (0 = largest); only the top-N by size get an
    LLM title/summary. ``llm`` is injectable for tests. Never raises — on any
    LLM failure it degrades to ranks-only so a graph build never fails because
    titling did.
    """
    if not communities:
        return {}

    if top_n is None:
        top_n = int(_get_setting("community_title_top_n", "25"))
    if member_cap is None:
        member_cap = int(_get_setting("community_title_member_cap", "20"))

    ranked = sorted(communities.items(), key=lambda kv: len(kv[1]), reverse=True)
    reports: Dict[int, Dict[str, Any]] = {
        cid: {"rank": idx} for idx, (cid, _members) in enumerate(ranked)
    }

    to_title = ranked[: max(0, top_n)]
    if not to_title:
        return reports

    if llm is None:
        try:
            from core.llm import create_llm_manager
            llm = create_llm_manager(
                service_name="graph_community_title", model=_community_model()
            )
        except Exception:
            logger.exception("community reports: LLM init failed — ranks only")
            return reports

    timeout = int(_get_setting("community_title_timeout", "30"))
    max_concurrent = int(_get_setting("community_title_concurrency", "5"))
    sem = asyncio.Semaphore(max_concurrent)

    async def _title_one(cid: int, members: List[str]) -> None:
        labels = _member_labels(graph, members, member_cap)
        if not labels:
            return
        prompt = _TITLE_PROMPT.format(members="\n".join(f"- {l}" for l in labels))
        try:
            async with sem:
                response = await asyncio.wait_for(
                    llm.generate_response([
                        {"role": "system", "content": "You name knowledge-graph clusters. Output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ]),
                    timeout=timeout,
                )
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception:
            logger.debug("community reports: titling failed for community %s", cid)
            return
        parsed = _parse_report(raw)
        if parsed:
            reports[cid].update(parsed)

    await asyncio.gather(*[_title_one(cid, members) for cid, members in to_title])
    return reports

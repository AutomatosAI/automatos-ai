"""
GraphSection — Inject business knowledge-graph context into agent prompts.

Priority 45: after memory (6), before tools (50-ish).
Loads the workspace graph, scores nodes against the current message,
and formats a relevant subgraph excerpt for the system prompt.

Source: PRD-126 US-009
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

_RELEVANCE_THRESHOLD = 0.3
_TOP_N = 3
_BFS_DEPTH = 2
_TOKEN_BUDGET = 800


class GraphSection(BaseSection):
    """Business knowledge-graph context for agent prompts.

    Loads the workspace graph, scores nodes by relevance to the current
    user message, and injects a BFS neighbourhood excerpt when relevant.
    """

    name: str = "business_graph"
    priority: int = 45
    max_tokens: Optional[int] = _TOKEN_BUDGET

    async def render(self, ctx: SectionContext) -> str:
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception("GraphSection.render failed — skipping graph context")
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        # 1. Extract current message
        message = self._extract_current_message(ctx)
        if not message:
            return ""

        # 2. Load graph
        from modules.knowledge.graph_service import get_graph_service, team_filtered_view

        service = get_graph_service()
        graph = await service.load_graph(str(ctx.workspace_id))
        if graph is None or graph.number_of_nodes() == 0:
            return ""

        # 3. PRD-124: filter graph by agent team
        agent_team = getattr(ctx.agent, "team", None) if ctx.agent else None
        graph = team_filtered_view(graph, agent_team)
        if graph.number_of_nodes() == 0:
            return ""

        # 4. Score nodes against message terms
        terms = [t.lower() for t in message.split() if len(t) > 2]
        if not terms:
            return ""

        scored = self._score_nodes_by_terms(graph, terms)
        if not scored or scored[0][1] < _RELEVANCE_THRESHOLD:
            return ""

        # 4. BFS from top scoring nodes, merge results
        top_nodes = [node_id for node_id, _score in scored[:_TOP_N]]
        all_nodes: set = set()
        all_edges: list = []

        for node_id in top_nodes:
            if node_id not in graph:
                continue
            try:
                result = await service.bfs(graph, node_id, depth=_BFS_DEPTH)
                all_nodes |= result["nodes"]
                all_edges.extend(result["edges"])
            except Exception:
                logger.debug("GraphSection: BFS failed for node %s", node_id)

        if not all_nodes:
            return ""

        # 5. Format to text
        text = await service.subgraph_to_text(
            graph, all_nodes, all_edges, _TOKEN_BUDGET
        )
        if not text:
            return ""

        return f"## Business Context (Knowledge Graph)\n\n{text}"

    @staticmethod
    def _extract_current_message(ctx: SectionContext) -> str:
        """Get the latest user message text."""
        if ctx.messages:
            for msg in reversed(ctx.messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return ctx.task_description or ""

    @staticmethod
    def _score_nodes_by_terms(
        graph: nx.Graph, terms: list[str]
    ) -> list[tuple[str, float]]:
        """Score graph nodes by term overlap with message words.

        Returns a sorted list of (node_id, score) descending by score.
        Score = fraction of terms that appear in the node label (case-insensitive).
        """
        results: list[tuple[str, float]] = []
        for node_id, attrs in graph.nodes(data=True):
            label = str(attrs.get("label", node_id)).lower()
            # Also check description/type if present
            description = str(attrs.get("description", "")).lower()
            combined = f"{label} {description}"

            hits = sum(1 for t in terms if t in combined)
            if hits > 0:
                score = hits / len(terms)
                results.append((str(node_id), score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


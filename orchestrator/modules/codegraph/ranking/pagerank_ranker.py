"""
PageRank-based symbol importance ranking.

Inspired by Aider's repo map — ranks code symbols by how frequently they are
referenced across the codebase, then fits results within a token budget for
optimal LLM context utilization (4-6% vs 54-70% with naive approaches).
"""

import logging
from typing import Any, Dict, List

import networkx as nx

logger = logging.getLogger(__name__)


class PageRankRanker:
    """Rank code symbols by structural importance using PageRank."""

    def __init__(self, alpha: float = 0.85):
        """
        Args:
            alpha: PageRank damping factor (0.85 is standard)
        """
        self.alpha = alpha

    def rank_symbols(
        self,
        symbols: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        token_budget: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Build a dependency graph, run PageRank, and return symbols
        sorted by importance, fitting within token_budget.

        Args:
            symbols: List of symbol dicts with 'id', 'name', 'signature', etc.
            relationships: List of relationship dicts with 'from_symbol_id', 'to_symbol_id'
            token_budget: Max tokens for the result set

        Returns:
            Symbols sorted by importance with 'importance_rank' added
        """
        if not symbols:
            return []

        G = nx.DiGraph()

        # Add symbols as nodes
        sym_id_set = set()
        for sym in symbols:
            sid = sym.get('id')
            if sid is not None:
                G.add_node(sid)
                sym_id_set.add(sid)

        # Add relationships as edges
        for rel in relationships:
            from_id = rel.get('from_symbol_id')
            to_id = rel.get('to_symbol_id')
            if from_id in sym_id_set and to_id in sym_id_set:
                G.add_edge(from_id, to_id, type=rel.get('relationship_type', 'unknown'))

        # Run PageRank
        if len(G.nodes()) == 0:
            # No graph to rank — return original order
            for sym in symbols:
                sym['importance_rank'] = 1.0 / max(len(symbols), 1)
            return symbols[:self._estimate_count_for_budget(symbols, token_budget)]

        try:
            ranks = nx.pagerank(G, alpha=self.alpha)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge, using uniform distribution")
            ranks = {n: 1.0 / len(G) for n in G.nodes()}

        # Sort symbols by rank (descending)
        for sym in symbols:
            sym['importance_rank'] = ranks.get(sym.get('id'), 0.0)

        ranked = sorted(symbols, key=lambda s: s.get('importance_rank', 0), reverse=True)

        # Fit within token budget
        result = []
        tokens_used = 0
        for sym in ranked:
            sym_tokens = self._estimate_tokens(sym)
            if tokens_used + sym_tokens > token_budget and result:
                break
            result.append(sym)
            tokens_used += sym_tokens

        logger.debug(f"PageRank ranked {len(symbols)} symbols → {len(result)} within {token_budget} token budget")
        return result

    def _estimate_tokens(self, symbol: Dict[str, Any]) -> int:
        """Rough token estimate for a symbol's context representation."""
        text_parts = [
            symbol.get('name', ''),
            symbol.get('signature', ''),
            symbol.get('docstring', '') or '',
            symbol.get('file_path', ''),
        ]
        total_chars = sum(len(p) for p in text_parts)
        # Rough: ~4 chars per token
        return max(total_chars // 4, 10)

    def _estimate_count_for_budget(self, symbols: List[Dict], budget: int) -> int:
        """Estimate how many symbols fit in the token budget."""
        total = 0
        for i, sym in enumerate(symbols):
            total += self._estimate_tokens(sym)
            if total > budget:
                return max(i, 1)
        return len(symbols)

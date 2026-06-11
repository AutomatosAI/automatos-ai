"""
Token-Budgeted Context Assembly (PRD-157 S3)
============================================

One model-aware token budgeter that replaces every char-based truncation on the
agent context path (the 6000-char cut in the chatbot service and the scattered
``[:N]`` slices) and the pure-Python knapsack DP in the RAG service.

Three primitives:

* :func:`select_within_budget` — accumulate **whole** chunks, highest score
  first, until the next chunk would exceed the token budget. Never splits a
  chunk; always returns at least the top chunk when candidates exist.
* :func:`assemble_with_citations` — render selected chunks as RAGFlow-style
  numbered sources ``[1]..[n]`` plus a source map, so answers can cite real
  document ids.
* :func:`truncate_to_token_budget` — token-aware truncation for an oversized
  single payload (e.g. a verbose tool result), replacing ``text[:6000]``.

Token counting and the model context-window lookup reuse
:mod:`core.context_guard`, so retrieval and conversation compaction share one
definition of "how big is this".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.context_guard import count_tokens, get_context_window

# Local encoder for decode-based truncation (counting itself goes through
# context_guard.count_tokens so there is a single counting definition).
try:  # pragma: no cover - import guard
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _enc = None

# Fraction of a model's context window spent on retrieved RAG context when a
# caller doesn't pass an explicit budget. The remainder is reserved for the
# system prompt, history, tools and the model's output.
DEFAULT_CONTEXT_FRACTION = 0.35
MIN_CONTEXT_TOKENS = 1_000

# Drop-in system-prompt nudge for citation-grade answers. Callers that want the
# model to cite inject this alongside the assembled context.
CITATION_INSTRUCTION = (
    "When you use the retrieved context, cite sources inline using their bracket "
    "numbers, e.g. [1] or [2][3]. The numbers map to the 'Sources' list at the end "
    "of the context. Only cite sources that are actually present."
)


def resolve_budget(
    max_tokens: Optional[int] = None,
    *,
    model: Optional[str] = None,
    db: Any = None,
    fraction: float = DEFAULT_CONTEXT_FRACTION,
) -> int:
    """Resolve a token budget.

    An explicit positive ``max_tokens`` wins. Otherwise size it from the model's
    context window (via ``core.context_guard.get_context_window``) times
    ``fraction``, floored at :data:`MIN_CONTEXT_TOKENS`.
    """
    if max_tokens and max_tokens > 0:
        return int(max_tokens)
    window = get_context_window(model or "", db)
    return max(MIN_CONTEXT_TOKENS, int(window * fraction))


@dataclass
class BudgetedSelection:
    chunks: List[Dict[str, Any]]
    total_tokens: int
    dropped: int = 0


def _chunk_tokens(chunk: Dict[str, Any], content_key: str) -> int:
    cached = chunk.get("tokens")
    if isinstance(cached, int) and cached > 0:
        return cached
    return count_tokens(chunk.get(content_key, "") or "")


def select_within_budget(
    candidates: Sequence[Dict[str, Any]],
    max_tokens: int,
    *,
    max_chunks: Optional[int] = None,
    content_key: str = "content",
    score_key: str = "similarity",
    presorted: bool = False,
) -> BudgetedSelection:
    """Greedy whole-chunk selection under a token budget.

    Chunks are taken highest-score-first; a chunk that would overflow the budget
    is skipped (a smaller later chunk may still fit). The single highest-score
    chunk is always included even if it alone exceeds the budget, so retrieval
    never silently returns nothing when it found something.
    """
    items = list(candidates)
    if not presorted:
        items.sort(key=lambda c: c.get(score_key, 0.0) or 0.0, reverse=True)

    selected: List[Dict[str, Any]] = []
    total = 0
    dropped = 0
    for chunk in items:
        if max_chunks is not None and len(selected) >= max_chunks:
            dropped += 1
            continue
        tok = _chunk_tokens(chunk, content_key)
        if selected and total + tok > max_tokens:
            dropped += 1
            continue
        selected.append(chunk)
        total += tok
    return BudgetedSelection(chunks=selected, total_tokens=total, dropped=dropped)


def assemble_with_citations(
    chunks: Sequence[Dict[str, Any]],
    query: Optional[str] = None,
    *,
    content_key: str = "content",
    source_key: str = "source_file",
    include_query_header: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Render chunks as numbered sources ``[1]..[n]`` + a source map.

    Returns ``(formatted_context, source_map)`` where each source-map entry is
    ``{citation, source_file, document_id, score}`` so callers can surface
    real document ids behind each citation number.
    """
    if not chunks:
        return "No relevant context found.", []

    lines: List[str] = []
    if include_query_header and query:
        lines.append(f"## Retrieved context for: {query}")
        lines.append("")

    source_map: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get(source_key) or chunk.get("filename") or "unknown"
        meta = chunk.get("metadata") or {}
        doc_id = chunk.get("document_id") or meta.get("document_id") or meta.get("external_file_id")
        content = chunk.get(content_key, "") or ""

        lines.append(f"[{i}] (source: {source})")
        lines.append(content)
        lines.append("")

        source_map.append(
            {
                "citation": i,
                "source_file": source,
                "document_id": doc_id,
                "score": chunk.get("similarity", chunk.get("score")),
            }
        )

    lines.append("---")
    lines.append("Sources (cite as [n]):")
    for entry in source_map:
        ref = str(entry["source_file"])
        if entry["document_id"] is not None:
            ref += f" (doc {entry['document_id']})"
        lines.append(f"[{entry['citation']}] {ref}")

    return "\n".join(lines), source_map


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
    *,
    suffix: str = "\n... (truncated)",
) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens on a token boundary.

    Replaces char slices like ``text[:6000]``. Falls back to a ~4-chars/token
    estimate only if tiktoken is unavailable.
    """
    if not text or max_tokens <= 0:
        return text
    if _enc is None:
        limit = max_tokens * 4
        return text if len(text) <= limit else text[:limit] + suffix
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens]) + suffix

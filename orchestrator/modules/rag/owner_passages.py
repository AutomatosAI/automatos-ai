"""F311 (night 9): when agents' reports are among the passages, the owner's are kept, first.

The selection hands over the five best-scoring passages. With agents' reports in the
index, those were reports: "A café wants 10 kg next week. What do we charge for
delivery?" (chat 77116484, 13:03) got four reports (0.80 to 0.84) and one chunk of
wholesale-terms-2026.md, and the "are you sure?" turn got five reports and none of the
owner's. Reports are not the owner's facts (F269): the night's build dropped them from
Auto's retrieval-first passages, which left one passage and then none, and Auto said
the terms name no delivery charge (ledger L93). The importers question (chat a75487a6)
was left with no passage at all (L85).

``owners_passages_kept`` wraps ``RAGService._optimize_with_context_optimizer``: when an
agent's report is among the candidates, up to ``RAG_OWNER_PASSAGES_RESERVED`` of the
passages handed over are the owner's (one per document, best first), taking free room
or the place of the lowest report, and the owner's passages come before the reports.
The search tool shows the model its first four results, so first is where they are read.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from config import config
from modules.rag.owner_leg import candidate_doc_id
from modules.rag.passages import chunk_tokens, owners_documents, rebuilt

logger = logging.getLogger(__name__)


def _key(item: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """A passage's document and text, to tell whether a candidate is already handed over."""
    return candidate_doc_id(item), (item.get("expanded_content") or item.get("content") or "")


def _as_passage(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """A candidate in the selection's passage shape."""
    content = candidate.get("expanded_content") or candidate.get("content") or ""
    passage = {"content": content, "source_file": candidate.get("source_file", candidate.get("filename", "unknown")),
               "similarity": candidate.get("similarity", 0.5), "document_id": candidate.get("document_id"),
               "metadata": candidate.get("metadata", {})}
    return {**passage, "tokens": chunk_tokens(passage)}


def _spare(service: Any, candidates: Sequence[Dict[str, Any]], chunks: Sequence[Dict[str, Any]],
           owners: Set[str], wanted: int) -> List[Dict[str, Any]]:
    """Up to ``wanted`` of the owner's passages not handed over, one per document, best first."""
    taken = {_key(chunk) for chunk in chunks}
    present = {candidate_doc_id(chunk) for chunk in chunks}
    spare: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=service._order_score, reverse=True):
        doc_id = candidate_doc_id(candidate)
        if len(spare) >= wanted or doc_id not in owners or doc_id in present or _key(candidate) in taken:
            continue
        present.add(doc_id)
        spare.append(_as_passage(candidate))
    return spare


def _with_spare(chunks: List[Dict[str, Any]], spare: List[Dict[str, Any]], max_chunks: int,
                owners: Set[str]) -> List[Dict[str, Any]]:
    """The spare passages in free room, then each in the place of the lowest passage
    that is not the owner's (an agent's report)."""
    room = max(0, max_chunks - len(chunks))
    kept = [*chunks, *spare[:room]]
    for passage in spare[room:]:
        reports = [i for i, chunk in enumerate(kept) if candidate_doc_id(chunk) not in owners]
        if not reports:
            break
        kept = [*kept[:reports[-1]], *kept[reports[-1] + 1:], passage]
    return kept


def _owners_count(chunks: Sequence[Dict[str, Any]], owners: Set[str]) -> int:
    """How many of the passages are from the owner's documents."""
    return sum(1 for chunk in chunks if candidate_doc_id(chunk) in owners)


def _owners_first(chunks: Sequence[Dict[str, Any]], owners: Set[str]) -> List[Dict[str, Any]]:
    """The owner's passages, then the reports', each in the order given."""
    mine = [chunk for chunk in chunks if candidate_doc_id(chunk) in owners]
    return mine + [chunk for chunk in chunks if candidate_doc_id(chunk) not in owners]


async def kept_for_the_owner(service: Any, result: Any, candidates: Sequence[Dict[str, Any]],
                             max_chunks: int) -> Any:
    """``result`` with the owner's passages kept and first, when reports are among the candidates."""
    reserve = min(config.RAG_OWNER_PASSAGES_RESERVED, max_chunks)
    owners = await owners_documents(service, candidates) if reserve > 0 and result.chunks else set()
    if not owners:
        return result
    chunks = list(result.chunks)
    wanted = reserve - _owners_count(chunks, owners)
    if wanted > 0:
        chunks = _with_spare(chunks, _spare(service, candidates, chunks, owners, wanted), max_chunks, owners)
    ordered = _owners_first(chunks, owners)
    if ordered == list(result.chunks):
        return result
    mine = _owners_count(ordered, owners)
    logger.info("[F311] passages handed over: %d of the owner's first, then %d agents' reports",
                mine, len(ordered) - mine)
    return rebuilt(result, ordered)


def owners_passages_kept(optimize: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorate ``RAGService._optimize_with_context_optimizer`` (F311). A failure is
    logged and the selection is handed over as it was made."""

    @functools.wraps(optimize)
    async def select(self: Any, query: str, candidates: List[Dict[str, Any]], max_chunks: int,
                     max_tokens: int, diversity: float) -> Any:
        result = await optimize(self, query, candidates, max_chunks, max_tokens, diversity)
        try:
            return await kept_for_the_owner(self, result, candidates, max_chunks)
        except Exception:  # noqa: BLE001 — the selection stands as it was made
            logger.warning("[F311] keeping the owner's passages failed; the selection stands", exc_info=True)
            return result

    return select

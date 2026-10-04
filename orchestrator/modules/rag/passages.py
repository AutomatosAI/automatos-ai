"""F311 (night 9): shared pieces for changing which passages a retrieval hands over.

``modules.rag.owner_passages`` (the owner's passages kept, and placed first) and
``modules.rag.whole_documents`` (a short owner's document handed over whole) both
change a :class:`modules.rag.service.RAGResult`'s chunks after selection; the numbered
citations, the source list and the token count are made again from the new chunks.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Set

from core.context_guard import count_tokens
from modules.rag.budget import assemble_with_citations
from modules.rag.owner_leg import candidate_doc_id


def chunk_tokens(chunk: Dict[str, Any]) -> int:
    """The chunk's token count, counted when it carries none."""
    tokens = chunk.get("tokens")
    return tokens if isinstance(tokens, int) and tokens > 0 else count_tokens(chunk.get("content") or "")


def rebuilt(result: Any, chunks: Sequence[Dict[str, Any]]) -> Any:
    """``result`` with these chunks, its citations, sources and token count made again."""
    kept = list(chunks)
    context, sources_map = assemble_with_citations(kept, result.query)
    return replace(result, chunks=kept, formatted_context=context, sources_map=sources_map,
                   sources=sorted({str(chunk.get("source_file") or "") for chunk in kept}),
                   total_tokens=sum(chunk_tokens(chunk) for chunk in kept))


async def owners_documents(service: Any, items: Sequence[Dict[str, Any]]) -> Set[str]:
    """The ids (as text) of the documents among ``items`` that are the owner's own:
    a documents row that is not an agent's report. Empty when there is no agent's
    report among them, so a workspace without reports is never changed."""
    facts = await asyncio.to_thread(service._document_ranking_facts, [candidate_doc_id(item) for item in items])
    if not any(is_report for is_report, _uploaded in facts.values()):
        return set()
    return {doc_id for doc_id, (is_report, _uploaded) in facts.items() if not is_report}


def all_owners(service: Any, items: Sequence[Dict[str, Any]]) -> List[str]:
    """The ids (as text) of every document among ``items`` that is not an agent's report."""
    facts = service._document_ranking_facts([candidate_doc_id(item) for item in items])
    return [doc_id for doc_id, (is_report, _uploaded) in facts.items() if not is_report]

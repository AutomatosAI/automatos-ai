"""What an agent or a mission wrote is never handed to Auto or an agent as the owner's
source (F269 and F287, night 8; F305's citations, night 9).

Night 8 left an agent's documents (``source_type`` agent_output: every filed job report
and mission output) out of Auto's retrieval-first passages, a draft's guides and the
mission planner's knowledge, and only MARKED them in a search_knowledge call the model
made itself. Night 9 (build 13, MORNING-REPORT.md L90 and L97):

- Chat 9a128fe3, "What payment terms do we give cafés?": the retrieval-first search
  left #1547 out, then Auto's own search_knowledge ("payment terms for cafés") returned
  it with the mark, and Auto answered with #1851's café list. Asked "Where did that come
  from?" it said "a document in our knowledge base titled
  '2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md'": ticket #1851's own
  answer, filed as document #1547. A mark asks the model to treat a passage
  differently; this one didn't.
- Chat 3912ba66 looked up Quay's terms with platform_search_documents, which (like
  platform_grep_documents, platform_read_document and platform_list_documents) had no
  check at all, though an agent's report on exactly that (#1548,
  "task-quay-coffee-house-payment-terms.md") was among the workspace's documents.
- Chat 96a6feb8: "You haven't uploaded any documents yourself yet": the list said
  nothing about who wrote what.

Now:
- A search (search_knowledge, the model's own call; platform_search_documents;
  platform_grep_documents) returns the owner's documents only: an agent's passages are
  left out, not marked (``owners_search``, ``owners_passages_only``).
- The answers Auto gives an agent's question mid-mission (services/orchestrator_answers)
  leave an agent's documents out of the corpus they are drawn from (``owners_corpus``).
- Reading one document by its id still works, and an agent's says so on its first line
  (``says_an_agent_wrote_it``); the list of documents says which an agent wrote
  (``says_who_wrote_each``). The owner can still ask about what an agent wrote.

Filing outputs as documents at all is F305: off unless the workspace opts in
(services/agent_output_scope.py); past work is asked for by scope (services/past_work.py).
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List

from services.draft_guides import AGENTS_WRITING, _a_session, _agents_documents, _as_id
from services.past_work import or_past_work  # F305: the one way to an agent's writing, asked for and labelled

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
WRITTEN_BY_AN_AGENT = "written_by_an_agent"


def _agents_workspace(db: Any, agent_id: Any) -> Any:
    """The workspace of the agent that searched; None when it is not found."""
    from core.models import Agent

    return db.query(Agent.workspace_id).filter(Agent.id == agent_id).scalar() if agent_id else None


def _without_agents(db: Any, result: Any, workspace_id: Any, key: str) -> Any:
    """``result`` without the items under ``key`` that came from a document an agent
    wrote in ``workspace_id``; as it was without a database session or such items."""
    found = result.get(key) if isinstance(result, dict) else None
    if not isinstance(found, list) or not _a_session(db) or workspace_id is None:
        return result
    drafts = _agents_documents(db, [r.get("document_id") for r in found if isinstance(r, dict)], workspace_id)
    if not drafts:
        return result
    kept = [r for r in found if not (isinstance(r, dict) and _as_id(r.get("document_id")) in drafts)]
    counted = {"count": len(kept)} if "count" in result else {}
    return {**result, key: kept, **counted, **_sources_of(result, kept)}


def _sources_of(result: Dict[str, Any], kept: List[Any]) -> Dict[str, Any]:
    """platform_search_documents' ``sources`` (file names) for the passages kept."""
    if not isinstance(result.get("sources"), list):
        return {}
    names = {str(r.get("file_name") or r.get("filename") or "") for r in kept if isinstance(r, dict)}
    return {"sources": [name for name in result["sources"] if str(name) in names]}


def owners_search(db: Any, result: Any, agent_id: Any) -> Any:
    """A search_knowledge result (``AgentPlatformTools.execute_tool``'s) with only the
    owner's passages, in the searching agent's workspace."""
    if not _a_session(db):
        return result
    return _without_agents(db, result, _agents_workspace(db, agent_id), "results")


def owners_corpus(db: Any, workspace_id: Any, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The answering service's corpus blocks (services/orchestrator_answers, its answers
    to an agent's question mid-mission) without those from a document an agent wrote."""
    if not blocks or not _a_session(db) or workspace_id is None:
        return blocks
    ids = [(b.get("source") or {}).get("document_id") for b in blocks if isinstance(b, dict)]
    drafts = _agents_documents(db, ids, workspace_id)
    return [b for b in blocks if not (isinstance(b, dict) and _as_id((b.get("source") or {}).get("document_id")) in drafts)]


def owners_passages_only(key: str) -> Callable[[Handler], Handler]:
    """Wrap a platform document search handler ``(db, workspace_id, params)``: the
    items under ``key`` (its passages or matches) come from the owner's documents only."""
    def wrap(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            return _without_agents(db, await handler(db, workspace_id, params), workspace_id, key)
        return wrapped
    return wrap


def says_an_agent_wrote_it(handler: Handler) -> Handler:
    """Wrap platform_read_document's handler: an agent's document opens with
    ``AGENTS_WRITING`` and carries ``written_by_an_agent``."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        result = await handler(db, workspace_id, params)
        if not (isinstance(result, dict) and result.get("success") and _a_session(db)):
            return result
        if not _agents_documents(db, [result.get("document_id")], workspace_id):
            return {**result, WRITTEN_BY_AN_AGENT: False}
        return {**result, WRITTEN_BY_AN_AGENT: True, "content": f"{AGENTS_WRITING}\n{result.get('content') or ''}"}
    return wrapped


def says_who_wrote_each(handler: Handler) -> Handler:
    """Wrap platform_list_documents' handler: each document says whether an agent wrote it."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        result = await handler(db, workspace_id, params)
        listed = result.get("documents") if isinstance(result, dict) else None
        if not isinstance(listed, list) or not _a_session(db):
            return result
        drafts = _agents_documents(db, [d.get("id") for d in listed if isinstance(d, dict)], workspace_id)
        marked: List[Any] = [{**d, WRITTEN_BY_AN_AGENT: _as_id(d.get("id")) in drafts} if isinstance(d, dict) else d
                             for d in listed]
        return {**result, "documents": marked}
    return wrapped


__all__ = ["WRITTEN_BY_AN_AGENT", "or_past_work", "owners_corpus", "owners_passages_only", "owners_search", "says_an_agent_wrote_it",
           "says_who_wrote_each"]

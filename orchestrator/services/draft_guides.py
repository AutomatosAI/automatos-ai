"""F201 (night 6): a draft for a customer is written from the workspace's guides,
and never says an action was done that no tool did.

#1146 "Reply to Rosie about a double charge and a grind change (draft only)",
agent 330. Its first draft (05:17:22, document 1046) came 6 s after the ticket,
after one call (load_skill). It read: "We will refund the duplicate payment of
£11.50 to your card … I have also updated your subscription to filter grind."
No charge was checked, nothing was changed, and the owner's brand voice guide
says never promise an amount. The redo, written after a search_knowledge, was
right.

- Retrieval first: a draft ticket's brief is searched against the workspace's
  documents before the agent's first model call. The passages that clear the
  relevance floor go into its prompt (F085-A's prefetch).
- Done-claims: the finished draft goes through F187's claim check against its
  run's actions. The loop has already nudged it once (F108). A claim still
  standing gets a line for the owner: check before sending.
- The owner's own (F269, night 7b; F269 and F287, night 8): what an agent wrote is
  never handed over as the owner's facts. ``owners_own`` drops its passages from a
  draft's guides and from Auto's retrieval-first passages, ``owners_own_chunks`` from
  the mission planner's knowledge.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional
from uuid import UUID

from services.step_lessons import a_cards_run_carries_its_lessons

logger = logging.getLogger(__name__)

_DRAFT_ASK = re.compile(
    r"\b(?:draft|reply to|respond to|write back|write (?:a|an) (?:reply|email|message|note|letter))\b", re.I)
_FOR_A_CUSTOMER = re.compile(
    r"\b(?:customers?|members?|clients?|subscribers?|guests?|caf[eé]s?|buyers?|suppliers?)\b|\bemail from\b"
    r"|\bwrote\b|^\s*\"?(?:hi|hello|dear)\b", re.I | re.M)

GUIDE_QUERY = "{brief}\n\nThe owner's rules for replies to customers: voice, what may be promised, refunds and changes."
GUIDES_HEADER = (
    "## Your workspace's guides, searched before you draft\n"
    "These passages are the owner's own rules for what a reply may say and promise. Follow them where they apply, "
    "and name the guide if the draft leans on it."
)
CHECK_BEFORE_SENDING = ("\n\nCheck before sending: the draft says something was {claim}, but nothing in this run "
                        "did that. Do it first, or change the wording to what will happen.")


# Asks that are for customers whatever their wording ("Write the newsletter").
# "post" counts as a piece of writing, never the mail ("post office", "postage").
_CUSTOMER_FACING = re.compile(
    r"\bnewsletters?\b|\bannouncements?\b|\bsubscriber (?:e-?mails?|updates?|letters?)\b"
    r"|\b(?:blog|social(?: media)?|instagram|facebook|linkedin|twitter)\s+posts?\b"
    r"|\b(?:write|draft|put together)\s+(?:a|an|the|this|our|next)\s+(?:[\w'’-]+\s+){0,2}posts?\b"
    r"(?!\s*(?:office|code|box))",
    re.I)


def is_customer_draft(brief: object) -> bool:
    """A brief that asks for a draft, reply or message for a customer. A
    newsletter, announcement, subscriber email or post is one in any wording."""
    text = str(brief or "")
    return bool(_CUSTOMER_FACING.search(text) or (_DRAFT_ASK.search(text) and _FOR_A_CUSTOMER.search(text)))


def _brief_only(prompt: str) -> str:
    """The ticket's own words: its prompt before the board's "Where your answer goes"
    (F297, night 8), which would otherwise fill most of a short brief's guide search."""
    from services.step_lessons import ON_THE_CARD

    return prompt.split(ON_THE_CARD, 1)[0].strip()


@a_cards_run_carries_its_lessons  # F249 (night 8): a card Auto started carries its agent's lessons too
async def guides_for_draft(db: Any, workspace_id: Any, agent_id: int, brief: str) -> str:
    """``brief`` with the workspace's guide passages for a customer draft; the
    brief as it was for anything else, or when nothing clears the floor."""
    own = _brief_only(brief)
    if not is_customer_draft(own):
        return brief
    from config import config
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL, prefetch
    from modules.tools.tool_router import get_tool_router

    async def _search(args):
        return owners_own(db, await get_tool_router().execute_and_format(
            tool_name=PREFETCH_TOOL, tool_args=args, agent_id=agent_id,
            workspace_id=UUID(str(workspace_id)), original_intent=own,
            caller_context={"retrieval_first": True, "draft_guides": True},
        ), workspace_id)

    try:
        found = await prefetch(
            db, workspace_id, GUIDE_QUERY.format(brief=own[:600]), search=_search,
            enabled=config.CHATBOT_KNOWLEDGE_PREFETCH, limit=config.KNOWLEDGE_PREFETCH_PASSAGES,
            min_score=config.KNOWLEDGE_PREFETCH_MIN_SCORE, question_only=False, header=GUIDES_HEADER,
        )
    except Exception:  # noqa: BLE001 — the agent can still search for itself
        logger.warning("[F201] guide search before the draft skipped", exc_info=True)
        return brief
    if found is None or found.message is None:
        return brief
    logger.info(f"[F201] draft ticket: {found.summary}")
    return f"{brief}\n\n{found.message['content']}"


def owners_own(db: Any, result: Any, workspace_id: Any = None) -> Any:
    """The search result without the passages from documents an agent wrote.

    F269 (night 7b): #0188.2's redo was given an old draft from last night as a
    "guide": a 14 December cut-off, a handwritten card and January to March dates the
    owner had never decided, and it dropped "orders open 2 November". A guide is the
    owner's own rule; an agent's earlier draft, report or document is not, approved or
    not, so it is never one.

    F269 and F287 (night 8): Auto's retrieval-first passages go through here too
    (consumers/chatbot/service.py). Only a database session can tell which documents
    an agent wrote; without one the result is as it was."""
    found = ((result or {}).get("raw_result") or {}).get("results") if isinstance(result, dict) else None
    if not isinstance(found, list) or not _a_session(db):
        return result
    drafts = _agents_documents(db, [r.get("document_id") for r in found if isinstance(r, dict)], workspace_id)
    kept = [r for r in found if not (isinstance(r, dict) and _as_id(r.get("document_id")) in drafts)]
    return {**result, "raw_result": {**result["raw_result"], "results": kept}}


def owners_own_chunks(db: Any, result: Any, workspace_id: Any) -> Any:
    """A retrieval result (``modules.rag.service.RAGResult``) without the chunks from
    documents an agent wrote, its numbered citations made again from the rest.

    F287 (night 8): the mission planner took "Friday 27 November from [2]" from an old
    agent document over the owner's 10 December, and the email step wrote it (#0352).
    The planner's knowledge (modules/context/sections/planning_knowledge.py) is the
    owner's documents only."""
    from dataclasses import replace

    chunks = list(getattr(result, "chunks", None) or [])
    if not chunks or not _a_session(db):
        return result
    drafts = _agents_documents(db, [_chunk_document(c) for c in chunks], workspace_id)
    kept = [c for c in chunks if _as_id(_chunk_document(c)) not in drafts]
    if len(kept) == len(chunks):
        return result
    from modules.rag.budget import assemble_with_citations

    context, sources_map = assemble_with_citations(kept, result.query)
    return replace(result, chunks=kept, formatted_context=context, sources_map=sources_map,
                   sources=sorted({str(c.get("source_file") or "") for c in kept}),
                   total_tokens=sum(int(c.get("tokens") or 0) for c in kept))


def _a_session(db: Any) -> bool:
    from sqlalchemy.orm import Session

    return isinstance(db, Session)


def _as_id(value: Any) -> Optional[int]:
    """A document id, whether the search gave it as a number or as text."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value) if isinstance(value, str) and value.strip().isdigit() else None


def _chunk_document(chunk: Any) -> Any:
    """The document a retrieved chunk came from (S3 Vectors keeps it in the chunk's metadata)."""
    if not isinstance(chunk, dict):
        return None
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return chunk.get("document_id") or meta.get("document_id") or meta.get("doc_id") or meta.get("external_file_id")


def _agents_documents(db: Any, ids: Iterable[Any], workspace_id: Any = None) -> set:
    """Which of these documents an agent wrote (``source_type`` agent_output), in the
    workspace when it is given."""
    from core.models.core import Document
    from services.knowledge_flywheel import AGENT_OUTPUT_SOURCE_TYPE

    wanted = sorted({i for i in (_as_id(value) for value in ids) if i is not None})
    if not wanted:
        return set()
    query = db.query(Document.id).filter(Document.id.in_(wanted), Document.source_type == AGENT_OUTPUT_SOURCE_TYPE)
    if workspace_id is not None:
        query = query.filter(Document.workspace_id == UUID(str(workspace_id)))
    return {row.id for row in query.all()}


def check_before_sending(brief: object, draft: object, ran: Iterable[str]) -> Optional[str]:
    """The owner's line for a customer draft that says an action was done that
    no action in its run did, else None."""
    if not is_customer_draft(brief):
        return None
    from modules.tools.execution.action_claims import claimed_action_not_done

    claim = claimed_action_not_done(str(draft or ""), set(ran or ()), promises=False)   # the writer's voice
    return CHECK_BEFORE_SENDING.format(claim=claim) if claim else None

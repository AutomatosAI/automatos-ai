"""A turn's two late lookups, each split from its merge (voice latency D-1).

Just before its first model call a chat turn makes two lookups that read nothing the turn's
prompt produced: the connected-apps lookup (``consumers.chatbot.app_lookup``) and the
knowledge prefetch (``consumers.chatbot.knowledge_prefetch``, F085). Each was written in one
piece with the code that puts its answer into the turn. This module holds the lookup side of
both, so that ``StreamingChatService._inject_composio_tools`` and ``_retrieval_first`` keep
only the merge:

* ``run_prefetch`` searches and hands back what it found and every search's result. It records
  nothing on the turn and never raises but for a cancel, so it can run on any task.
* ``the_turns_prefetch`` is what ``_retrieval_first`` awaits: it runs the search on the turn's
  own session, in place, records each result on the turn and raises what the prefetch raised.
* ``the_turns_app_lookup`` is what ``_inject_composio_tools`` awaits: the lookup, in place.

This story is a move: a turn searches the same things at the same point with the same
arguments, and is given the same messages, tools, frames and retrieval record as before.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config import config
from consumers.chatbot.app_lookup import AppLookup, look_up_apps

if TYPE_CHECKING:
    from consumers.chatbot.knowledge_prefetch import Prefetch


@dataclasses.dataclass(frozen=True)
class Searched:
    """What a turn's knowledge prefetch did."""

    found: Optional["Prefetch"]              # knowledge_prefetch.prefetch's answer
    results: Tuple[Any, ...]                 # each search's result, for the turn's retrieval record
    failed: bool = False                     # a search, or prefetch itself, failed
    error: Optional[BaseException] = None    # what prefetch raised, if it raised


async def run_prefetch(chat: Any, db: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                       how: Dict[str, Any], into: Optional[List[Any]] = None) -> Searched:
    """Search the documents for this turn (``knowledge_prefetch.prefetch``) with ``db`` as the
    session the count and the owner's-documents filter read on. Each search's result is handed
    back, never recorded on ``chat``; ``into``, when given, is the caller's list each result is
    also put in as it arrives, so a caller that is cancelled still holds what was searched.
    What was searched is returned even when ``prefetch`` raised. Never raises but for a cancel."""
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL, prefetch
    from services.draft_guides import owners_own  # F269/F287 (night 8): never an agent's writing as facts

    results: List[Any] = [] if into is None else into
    search_failed = False

    async def _search(args: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal search_failed
        try:
            result = owners_own(db, await chat.tool_router.execute_and_format(
                tool_name=PREFETCH_TOOL, tool_args=args,
                agent_id=agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else 1,
                workspace_id=chat.workspace_id,
                original_intent=latest_text,
                caller_context={"user_query": latest_text, "conversation_id": chat_id, "retrieval_first": True},
            ), chat.workspace_id)
        except Exception:
            search_failed = True
            raise
        results.append(result)
        return result

    try:
        found = await prefetch(
            db, chat.workspace_id, latest_text, search=_search, viewer=getattr(chat, "_viewer_subject_id", None),
            enabled=config.CHATBOT_KNOWLEDGE_PREFETCH, **how,
            limit=config.KNOWLEDGE_PREFETCH_PASSAGES, min_score=config.KNOWLEDGE_PREFETCH_MIN_SCORE,
        )
    except Exception as exc:  # noqa: BLE001 — handed back as the failure; the caller decides who logs it
        return Searched(None, tuple(results), failed=True, error=exc)
    return Searched(found, tuple(results), failed=search_failed)


async def the_turns_prefetch(chat: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                             how: Dict[str, Any]) -> Optional["Prefetch"]:
    """The turn's knowledge prefetch, in place, on the turn's own session. Every search's result
    goes on the turn's retrieval record, also when the prefetch found nothing, failed or was
    cancelled; what ``prefetch`` raised is raised here for ``_retrieval_first`` to log."""
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL

    results: List[Any] = []
    try:
        searched = await run_prefetch(chat, chat.db, latest_text, agent_runtime, chat_id, how, into=results)
    finally:
        for result in results:
            chat._collect_tool_retrieval(PREFETCH_TOOL, result)
    if searched.error is not None:
        raise searched.error
    return searched.found


async def the_turns_app_lookup(chat: Any, agent_id: int, latest_text: str, tool_hints: Any) -> AppLookup:
    """The turn's connected-apps lookup, in place."""
    return await look_up_apps(agent_id, chat.workspace_id, latest_text, tool_hints)


__all__ = ["Searched", "run_prefetch", "the_turns_app_lookup", "the_turns_prefetch"]

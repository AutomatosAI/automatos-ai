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
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional
from uuid import UUID

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


def is_customer_draft(brief: object) -> bool:
    """A brief that asks for a draft, reply or message for a customer."""
    text = str(brief or "")
    return bool(_DRAFT_ASK.search(text) and _FOR_A_CUSTOMER.search(text))


async def guides_for_draft(db: Any, workspace_id: Any, agent_id: int, brief: str) -> str:
    """``brief`` with the workspace's guide passages for a customer draft; the
    brief as it was for anything else, or when nothing clears the floor."""
    if not is_customer_draft(brief):
        return brief
    from config import config
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL, prefetch
    from modules.tools.tool_router import get_tool_router

    async def _search(args):
        return await get_tool_router().execute_and_format(
            tool_name=PREFETCH_TOOL, tool_args=args, agent_id=agent_id,
            workspace_id=UUID(str(workspace_id)), original_intent=brief,
            caller_context={"retrieval_first": True, "draft_guides": True},
        )

    try:
        found = await prefetch(
            db, workspace_id, GUIDE_QUERY.format(brief=brief.strip()[:600]), search=_search,
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


def check_before_sending(brief: object, draft: object, ran: Iterable[str]) -> Optional[str]:
    """The owner's line for a customer draft that says an action was done that
    no action in its run did, else None."""
    if not is_customer_draft(brief):
        return None
    from modules.tools.execution.tool_loop import claimed_action_not_done

    claim = claimed_action_not_done(str(draft or ""), set(ran or ()))
    return CHECK_BEFORE_SENDING.format(claim=claim) if claim else None

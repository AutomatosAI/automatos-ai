"""F351 (night 10b, build 19): a document conversation keeps its document tools on every turn.

Auto made the document on the first turn of a chat and lost the tool on the next. "Yes, go
ahead and try again." got platform_execute alone, and Auto called it with an action named
platform_generate_document ("Unknown platform action", 2fec4fd9). "Please just make it."
called an invented platform_create_document (0e417f99), "Yes, issue 43. Go ahead." an
invented "Generate PDF from template" playbook (43c5e928), and a corrected field list went to
platform_create_deliverable (a71ae29b). With the tool gone, Auto sent the owner to Proposify,
Venngage, Canva and Visme for a document its own template makes.

Two places drop generate_document on a follow-up:

* AutoBrain reads a short follow-up ("Yes, issue 43. Go ahead.") on its own, as chat, and the
  ATOM lane ships the platform_execute dispatcher and nothing else.
* On the full path the intent classifier reads it on its own too: no creation words, so no
  tools, and the context's tool section then keeps only the ``platform_*`` tools.
  generate_document is not one. The same happened on a first turn whose words missed the
  creation patterns ("Put a short roast-day notice on the Branded Page template", 1e439ba6):
  Auto had platform_create_social_post and used it for a PDF.

So a conversation about a document (the owner's or Auto's recent words name a template, a
PDF or a Word document, or a document tool) takes the full path, and the router keeps
generate_document and the template tools on every branch, a "no tools" verdict included:
the turn gets what it would have had plus the document tools, with the choice left to the
model. The dispatcher's ranked actions keep the two template actions too. The check reads
the last few messages already in hand: no store, no extra model call.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

GENERATE_DOCUMENT = "generate_document"
# The dispatcher's template actions (modules/tools/discovery/actions_documents.py).
DOCUMENT_ACTIONS = ("platform_list_templates", "platform_get_template_schema")
DOCUMENT_TOOLS = frozenset({GENERATE_DOCUMENT, *DOCUMENT_ACTIONS})
# How far back a conversation is read: the latest three exchanges, each message capped.
LOOKBACK_MESSAGES = 6
MAX_CHARS_PER_MESSAGE = 4000
SPOKEN_ROLES = frozenset({"user", "assistant"})
# What the ContextService's tool section keeps on a turn the router ships no tools for.
PLATFORM_PREFIX = "platform_"
# AutoBrain's lanes (consumers/chatbot/auto.py Complexity values).
ATOM = "atom"
FULL_PATH = "molecule"

DOCUMENT_CUE = re.compile(
    r"\btemplates?\b|\bpdfs?\b|\bdocx\b|\bword (?:doc|document|file)s?\b|\bletterhead\b"
    r"|\bgenerate_document\b|\bplatform_(?:list_templates|get_template_schema)\b",
    re.IGNORECASE,
)


def message_text(message: Any) -> str:
    """The words of one chat message: its ``content`` (a string or text parts) or its ``parts``."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    pieces = content if isinstance(content, list) else message.get("parts") or []
    return " ".join(str(p.get("text") or "") for p in pieces if isinstance(p, dict))


def about_a_document(messages: Optional[Iterable[Any]], query: object = "") -> bool:
    """Whether the latest few user and assistant messages (and ``query``) are about a document."""
    spoken = [m for m in (messages or []) if isinstance(m, dict) and m.get("role") in SPOKEN_ROLES]
    texts = [message_text(m)[:MAX_CHARS_PER_MESSAGE] for m in spoken[-LOOKBACK_MESSAGES:]]
    texts.append(str(query or "")[:MAX_CHARS_PER_MESSAGE])
    return any(DOCUMENT_CUE.search(text) for text in texts if text)


def _name(tool: Any) -> str:
    return str(((tool or {}).get("function") or {}).get("name", "")) if isinstance(tool, dict) else ""


def with_document_tools(result: Any, available_tools: List[Dict[str, Any]]) -> Any:
    """``result`` (a ToolRoutingResult) holding every document tool ``available_tools`` has.

    A "no tools" verdict becomes what the turn would have had (its ``platform_*`` tools) plus
    the document tools, with ``tool_choice`` "auto". Returns a new result; none is mutated.
    """
    wanted = [tool for tool in available_tools or [] if _name(tool) in DOCUMENT_TOOLS]
    if not wanted:
        return result
    if not result.should_include_tools:
        kept = [t for t in available_tools if _name(t).startswith(PLATFORM_PREFIX) or _name(t) in DOCUMENT_TOOLS]
        return dataclasses.replace(
            result, should_include_tools=True, filtered_tools=kept, tool_choice="auto",
            reasoning=f"{result.reasoning}; a document conversation keeps its document tools",
        )
    held = {_name(tool) for tool in result.filtered_tools}
    missing = [tool for tool in wanted if _name(tool) not in held]
    if not missing:
        return result
    return dataclasses.replace(result, filtered_tools=[*result.filtered_tools, *missing])


Route = Callable[..., Any]


def keeps_document_tools(route: Route) -> Route:
    """Wrap ``SmartToolRouter.route``: in a document conversation every branch's result
    carries generate_document and the template tools (``with_document_tools``)."""
    signature = inspect.signature(route)

    @functools.wraps(route)
    async def routed(*args: Any, **kwargs: Any) -> Any:
        result = await route(*args, **kwargs)
        given = signature.bind(*args, **kwargs).arguments
        if not about_a_document(given.get("conversation_context"), given.get("query")):
            return result
        kept = with_document_tools(result, given.get("available_tools") or [])
        if kept is not result:
            logger.info("[F351] a document conversation keeps generate_document and the template tools")
        return kept
    return routed


def full_path_for_documents(assessment: Any, document_turn: bool, force_text_only: bool = False) -> Any:
    """AutoBrain's ``assessment`` with an ATOM verdict moved to the full path for a document turn.

    The ATOM lane ships the dispatcher alone; a document conversation's follow-up needs
    generate_document. Returns a new assessment; a proactive opener (``force_text_only``)
    and every other verdict are returned as they are.
    """
    complexity = getattr(assessment, "complexity", None)
    if not document_turn or force_text_only or getattr(complexity, "value", None) != ATOM:
        return assessment
    logger.info("[F351] a document conversation's follow-up takes the full path, not the ATOM lane")
    return dataclasses.replace(assessment, complexity=type(complexity)(FULL_PATH))


def with_document_actions(page_actions: Optional[List[str]], document_turn: bool) -> Optional[List[str]]:
    """The actions folded into the dispatcher's ranked enum: the page's, plus the template
    actions on a document turn (the PRD-221 page prior keeps them past the ranking)."""
    if not document_turn:
        return page_actions
    extra = [name for name in (page_actions or []) if name not in DOCUMENT_ACTIONS]
    return [*DOCUMENT_ACTIONS, *extra]


__all__ = [
    "DOCUMENT_ACTIONS", "DOCUMENT_TOOLS", "GENERATE_DOCUMENT", "about_a_document", "full_path_for_documents",
    "keeps_document_tools", "message_text", "with_document_actions", "with_document_tools",
]

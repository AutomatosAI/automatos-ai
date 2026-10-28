"""The language variant Auto replies in: the person's ``locale`` on ``POST /api/chat``.

Automatos Go's "I speak" setting was ``pt-BR``, and the phone listened and spoke in Brazilian
Portuguese, but the turn named no locale, so the model picked the variant itself. From Lisbon,
it answered a Brazilian speaker in European Portuguese ("Queres que comece", "Baseei-me", "no
vosso blog", "constrói-se").

Now, when the turn names a locale (``api.chat_models.ChatRequest.locale``, already a checked
BCP 47 tag), ``StreamingChatService.stream_response_with_agent`` keeps it for the turn and
``_prepare_messages`` (the ATOM and the full path; a spoken turn and a typed one) ends Auto's
system prompt with one line naming the language and its variant. The person still decides: the
line says to reply in another language when they write or speak in one. Without a locale the
messages are the very list prepared, so a client that sends none sees no change.
"""
from __future__ import annotations

import functools
import logging
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
# The turn's locale, kept on the service by ``stream_response_with_agent`` for this turn.
TURN_LOCALE_ATTRIBUTE = "_reply_locale"
# A tag's variant: its name, and the word for its vocabulary ("use Brazilian vocabulary").
VARIANTS: Mapping[str, Tuple[str, str]] = MappingProxyType({
    "pt-BR": ("Brazilian Portuguese", "Brazilian"),
    "pt-PT": ("European Portuguese", "European Portuguese"),
    "en-GB": ("British English", "British"),
    "en-US": ("American English", "American"),
    "es-ES": ("Spanish from Spain", "Spain's"),
    "fr-FR": ("French from France", "France's"),
    "it-IT": ("Italian from Italy", "Italy's"),
})
ANY_OTHER_VARIANT = ("the language and region of {tag}", "that region's")
REPLY_LINE = ("Reply in {name}: use {words} vocabulary, forms and spelling. If the person writes or "
              "speaks in another language, reply in that language instead.")

Prepared = Tuple[List[Dict[str, Any]], Any, Any]
Prepare = Callable[..., Awaitable[Prepared]]


def reply_locale_line(locale: str) -> str:
    """The line telling Auto which language and variant to reply in, for a BCP 47 ``locale``."""
    if locale in VARIANTS:
        name, words = VARIANTS[locale]
        return REPLY_LINE.format(name=f"{name} ({locale})", words=words)
    name, words = ANY_OTHER_VARIANT
    return REPLY_LINE.format(name=name.format(tag=locale), words=words)


def system_prompt_with_locale(llm_messages: List[Dict[str, Any]], locale: Optional[str]) -> List[Dict[str, Any]]:
    """``llm_messages`` with the locale's line at the end of the system prompt; a new list.
    The same list, untouched, when there is no locale."""
    if not locale:
        return llm_messages
    line = reply_locale_line(locale)
    first = llm_messages[0] if llm_messages else {}
    if first.get("role") == SYSTEM_ROLE and isinstance(first.get("content"), str):
        return [{**first, "content": f"{first['content']}\n\n{line}"}, *llm_messages[1:]]
    return [{"role": SYSTEM_ROLE, "content": line}, *llm_messages]


def replies_in_the_persons_locale(prepare: Prepare) -> Prepare:
    """Wrap ``StreamingChatService._prepare_messages``: the system prompt ends with the turn's
    locale line, after the spoken-output contract and any CTO override, so both keep it."""
    @functools.wraps(prepare)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> Prepared:
        llm_messages, use_tools, orchestrated = await prepare(chat, *args, **kwargs)
        locale = getattr(chat, TURN_LOCALE_ATTRIBUTE, None)
        logger.debug("[chat] the turn's reply locale: %s", locale)  # the tag only, never the person's words
        return system_prompt_with_locale(llm_messages, locale), use_tools, orchestrated
    return wrapped

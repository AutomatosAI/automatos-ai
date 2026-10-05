"""F337 (night 10): Auto's own chat keeps to the brand kit, as every agent run does since #953.

Night 10 (5 October), Auto's documents from chat: a club flyer (PDF 1fc4eef7) said
"delightful", which the kit bans; a wholesale flyer (e9d50f2e) was from "Harbour Coffee
Roasters", not Harbourline; a quote and a letter were signed "Sincerely, [Your Company Name]".
#953 gave the kit's rules to card runs, mission steps and playbook steps, and filled their
answers' placeholders (``services.brand_hooks``); Auto's chat had neither.

Now, on an owner's turn in a workspace with a brand kit:

* the prompt: ``StreamingChatService._prepare_messages`` (both lanes, ATOM and the full
  context path) ends the system prompt with the kit's rules block
  (``services.brand_rules.rules_for_kit``: the name Auto writes as, the tone, who signs, the
  banned words, the company and its details, the colours, fonts and logo). The kit is read once
  per turn, off the event loop and without flushing (``kit_off_loop``), and kept for the answer;
* the answer: ``service._upload_inline_images``, the last pass over each text the turn saves
  (its answer, its narration, and an answer the provider could not stream, which is then shown),
  fills a placeholder signature or company ("[Your name]" becomes who signs, "[Your Company
  Name]" the company); ``_answer_additions`` adds the banned-words note after the answer, on the
  screen and in what is saved, as a card's answer gets it. The note is about words the reply
  uses: in a chat reply a banned word on its own in quotation marks is a mention (Auto telling
  the owner 'the flyer uses the word "delightful"'), so it is left out of the check.

A widget visitor's turn is left as it was: the kit's rules are for what the owner sends or
publishes, and a visitor's prompt stays the widget's.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from services import brand_rules as br

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
MENTION_PUNCTUATION = " \t,.;:!?"
# A short span in quotation marks: straight, curly double or curly single, and a straight single
# one only where it opens after a non-word character (so "don't" never opens a span).
QUOTED = re.compile(r"\"([^\"\n]{1,60})\"|“([^”\n]{1,60})”|‘([^’\n]{1,60})’"
                    r"|(?<!\w)'([^'\n]{1,60})'(?!\w)")

# The brand kit read for this turn (None: no kit, a visitor's turn, or no turn).
_TURN_KIT: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("f337_turn_kit",
                                                                                      default=None)

Prepared = Tuple[List[Dict[str, Any]], Any, Any]
Prepare = Callable[..., Awaitable[Prepared]]
Upload = Callable[..., Awaitable[str]]
Additions = Callable[..., List[str]]


def system_prompt_with_rules(llm_messages: List[Dict[str, Any]], kit: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``llm_messages`` with ``kit``'s rules at the end of the system prompt, once; a new list.
    Unchanged when the kit says nothing about how to write or look."""
    block = br.rules_for_kit(kit)
    if not block:
        return llm_messages
    first = llm_messages[0] if llm_messages else {}
    if first.get("role") == SYSTEM_ROLE and isinstance(first.get("content"), str):
        return [{**first, "content": br.prompt_with_rules(first["content"], kit)}, *llm_messages[1:]]
    return [{"role": SYSTEM_ROLE, "content": block}, *llm_messages]


def _owners_turn(chat: Any) -> bool:
    return bool(getattr(chat, "workspace_id", None)) and not getattr(chat, "widget_mode", False)


def autos_prompt_carries_the_brand_kit(prepare: Prepare) -> Prepare:
    """Wrap ``StreamingChatService._prepare_messages``: the system prompt ends with the brand
    kit's rules, read off the event loop, and the kit is kept for this turn's answer."""
    @functools.wraps(prepare)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> Prepared:
        _TURN_KIT.set(None)
        llm_messages, use_tools, orchestrated = await prepare(chat, *args, **kwargs)
        if not _owners_turn(chat):
            return llm_messages, use_tools, orchestrated
        kit = await br.kit_off_loop(chat.db, chat.workspace_id)
        _TURN_KIT.set(kit)
        return system_prompt_with_rules(llm_messages, kit), use_tools, orchestrated
    return wrapped


def _mentions_left_out(answer: str, phrases: List[str]) -> str:
    """``answer`` without the quoted spans that only name a banned phrase ("delightful",)."""
    named = {phrase.lower() for phrase in phrases if isinstance(phrase, str)}

    def keep(match: re.Match) -> str:
        said = (match.group(match.lastindex) or "").strip(MENTION_PUNCTUATION).lower()
        return " " if said in named else match.group(0)
    return QUOTED.sub(keep, answer)


def reply_banned_note(answer: Any, kit: Optional[Dict[str, Any]]) -> str:
    """The banned-words note for a chat reply: the words it uses, not the ones it names in quotes."""
    if not kit or not isinstance(answer, str) or not answer.strip():
        return ""
    phrases = (kit.get("voice") or {}).get("banned_phrases") or []
    return br.banned_note_for(_mentions_left_out(answer, phrases), kit) if phrases else ""


def a_reply_says_its_banned_words(answer_additions: Additions) -> Additions:
    """Wrap ``StreamingChatService._answer_additions``: after the others, the note naming the
    banned words the answer uses, shown and saved with it."""
    @functools.wraps(answer_additions)
    def wrapped(f187_verdict: Any, final_round: Any) -> List[str]:
        additions = answer_additions(f187_verdict, final_round)
        note = reply_banned_note(getattr(final_round, "content", None), _TURN_KIT.get())
        return [*additions, f"\n\n{note}"] if note else additions
    return wrapped


def a_saved_reply_is_on_brand(upload: Upload) -> Upload:
    """Wrap ``service._upload_inline_images``: the text the turn saves has its placeholder
    signature and company filled from the turn's brand kit."""
    @functools.wraps(upload)
    async def wrapped(text: str, *args: Any, **kwargs: Any) -> str:
        uploaded = await upload(text, *args, **kwargs)
        filled = br.placeholders_filled(uploaded, _TURN_KIT.get())
        if filled != uploaded:
            logger.info("[F337] a placeholder in Auto's reply was filled from the brand kit")
        return filled
    return wrapped


__all__ = ["a_reply_says_its_banned_words", "a_saved_reply_is_on_brand", "autos_prompt_carries_the_brand_kit",
           "reply_banned_note", "system_prompt_with_rules"]

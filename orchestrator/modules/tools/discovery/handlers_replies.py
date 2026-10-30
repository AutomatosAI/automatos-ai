"""PRD-268 US-C12a (§6.6, FR-12, K12) — ``platform_offer_replies``: the likely answers as taps.

When Auto asks the person to choose, it offers 2-3 short answers; the person taps one and it
is sent as their own message. The handler checks the replies and writes nothing (it is a
``read``: W1's read-actions sweep calls it): 2-3 distinct replies (compared trimmed and
case-insensitively), each 1-3 words and at most 24 characters, with no line break, no
character that does not show (a control, format or private-use character: a bidi override or
a zero-width space would make the chip read otherwise than the message it sends) and no
markdown or link. A list sent as JSON text is read. Anything else is refused with a sentence
the model reads, in the turn's language (K18: ``follow_up_context.carries_the_follow_up``
hands the chat's facts to the chat-only tools; this one reads only the language), and an
``error_code``.

It is chat only (``check_in_lane.CHAT_ONLY_ACTIONS``): no check-in, and no follow-up's own
turn, offers it or runs it. What the chips do to the turn (it ends on the round's own text,
with no further model call) is ``modules.tools.execution.reply_chips_turn``'s (US-C12b).
"""
from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from modules.proactive.locale import DEFAULT_LOCALE
from modules.tools.discovery.follow_up_context import FOLLOW_UP_PARAM, OFFER_REPLIES
from modules.tools.execution.params_text import json_container
from services.proactive.follow_up_strings import locale_of

__all__ = [
    "MAX_CHARS", "MAX_REPLIES", "MAX_WORDS", "MIN_REPLIES", "OFFER_REPLIES", "REPLIES_PARAM", "STRINGS",
    "offer_replies", "replies_refusal",
]

REPLIES_PARAM = "replies"
MIN_REPLIES, MAX_REPLIES = 2, 3
MAX_WORDS = 3
MAX_CHARS = 24
# A refused reply is echoed back this far, as it is shown.
ECHO_CHARS = 40

NOT_A_LIST, COUNT, NOT_TEXT, EMPTY = "replies_not_a_list", "reply_count", "reply_not_text", "reply_empty"
LINE_BREAK, HIDDEN, MARKDOWN, TOO_LONG, WORDS, DUPLICATE = (
    "reply_line_break", "reply_hidden", "reply_markdown", "reply_too_long", "reply_words", "reply_duplicate")
# Characters that do not show: control, format (bidi controls, zero-width), private use, surrogate.
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs"})
# Markdown a chip would show as raw marks, or a renderer would turn into something else:
# emphasis (*, a word wrapped in _, __, ~~), code, a link or image ([ or ]), an HTML tag, a
# heading, a quote or a list item at the start, and a link a renderer would make clickable.
# "~5 min", "<5 min", "A | B", "#42", "9:30" and "snake_case" are plain words.
_MARKDOWN = re.compile(
    r"[*`\[\]]|__|~~|<[A-Za-z/!]|(?:^|\s)_\S|\S_(?:\s|$)|^#+\s|^>|^[-+]\s|^\d+[.)]\s|://|\bwww\.",
    re.IGNORECASE)

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        NOT_A_LIST: "replies is a list of {low} to {high} short answers, each as text.",
        COUNT: "Offer {low} to {high} replies, not {count}. Or offer none, and let the person answer in words.",
        NOT_TEXT: "Each reply is text: a few words the person can tap.",
        EMPTY: "A reply is empty. Each is 1 to {words} words the person can tap.",
        LINE_BREAK: "\"{reply}\" breaks a line. A reply is one line of 1 to {words} words.",
        HIDDEN: "A reply carries a character that does not show. A reply is plain words, shown as they are.",
        MARKDOWN: "\"{reply}\" carries markdown or a link. A reply is plain words, shown as they are.",
        TOO_LONG: "\"{reply}\" is {chars} characters; a reply is at most {limit}. Say it shorter.",
        WORDS: "\"{reply}\" is {count} words; a reply is 1 to {words}. Say it shorter.",
        DUPLICATE: "\"{reply}\" is offered twice. Each reply is a different answer.",
    }),
    "pt-BR": MappingProxyType({
        NOT_A_LIST: "replies é uma lista de {low} a {high} respostas curtas, cada uma em texto.",
        COUNT: ("Ofereça de {low} a {high} respostas, não {count}. Ou não ofereça nenhuma, e deixe a pessoa "
                "responder com as próprias palavras."),
        NOT_TEXT: "Cada resposta é um texto: poucas palavras que a pessoa pode tocar.",
        EMPTY: "Uma resposta está vazia. Cada uma deve ter de 1 a {words} palavras que a pessoa pode tocar.",
        LINE_BREAK: "\"{reply}\" tem quebra de linha. Uma resposta é uma linha de 1 a {words} palavras.",
        HIDDEN: "Uma resposta tem um caractere que não aparece. Uma resposta é texto simples, mostrado como está.",
        MARKDOWN: "\"{reply}\" tem markdown ou um link. Uma resposta é texto simples, mostrado como está.",
        TOO_LONG: "\"{reply}\" tem {chars} caracteres; uma resposta tem no máximo {limit}. Encurte a resposta.",
        WORDS: "\"{reply}\" tem {count} palavras; uma resposta deve ter de 1 a {words}. Encurte a resposta.",
        DUPLICATE: "\"{reply}\" aparece duas vezes. Cada resposta deve ser diferente das outras.",
    }),
})


def _shown(reply: str) -> str:
    """A reply as the chip shows it: trimmed, its spaces single."""
    return " ".join(reply.split())


def _reply_refusal(reply: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Why one reply is refused (its code and the sentence's values), or None when it is kept."""
    if not isinstance(reply, str):
        return NOT_TEXT, {}
    shown = _shown(reply)
    if not shown:
        return EMPTY, {}
    echo = {"reply": shown[:ECHO_CHARS]}
    if len(reply.strip().splitlines()) != 1:
        return LINE_BREAK, echo
    if any(unicodedata.category(char) in _HIDDEN_CATEGORIES for char in shown):
        return HIDDEN, {}
    if _MARKDOWN.search(shown):
        return MARKDOWN, echo
    if len(shown) > MAX_CHARS:
        return TOO_LONG, {**echo, "chars": len(shown)}
    words = len(shown.split())
    return (WORDS, {**echo, "count": words}) if words > MAX_WORDS else None


def replies_refusal(replies: Any) -> Tuple[List[str], Optional[Tuple[str, Dict[str, Any]]]]:
    """The replies as they are shown (``_shown``), or why they are refused: the code and the
    values its sentence takes. Two replies that differ only in case or spacing are one."""
    if not isinstance(replies, list):
        return [], (NOT_A_LIST, {})
    if not MIN_REPLIES <= len(replies) <= MAX_REPLIES:
        return [], (COUNT, {"count": len(replies)})
    for reply in replies:
        refusal = _reply_refusal(reply)
        if refusal is not None:
            return [], refusal
    shown = [_shown(reply) for reply in replies]
    for at, reply in enumerate(shown):
        if reply.casefold() in {earlier.casefold() for earlier in shown[:at]}:
            return [], (DUPLICATE, {"reply": reply[:ECHO_CHARS]})
    return shown, None


def _say(locale: str, code: str, values: Mapping[str, Any]) -> str:
    table = STRINGS.get(locale, STRINGS[DEFAULT_LOCALE])
    return table[code].format(low=MIN_REPLIES, high=MAX_REPLIES, words=MAX_WORDS, limit=MAX_CHARS, **values)


async def offer_replies(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The replies to show under Auto's message, or why they are refused. Writes nothing."""
    params = params if isinstance(params, dict) else {}
    replies, refusal = replies_refusal(json_container(params.get(REPLIES_PARAM)))
    if refusal is not None:
        code, values = refusal
        return {"success": False, "error": _say(locale_of(params.get(FOLLOW_UP_PARAM)), code, values),
                "error_code": code}
    return {"success": True, REPLIES_PARAM: replies}

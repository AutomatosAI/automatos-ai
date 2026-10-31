"""PRD-268 §6.11.5, K15, K18 — what a calibration says, from the strings table.

Server copy, never the model's, in English and Portuguese (Brazil): the consent line (§12-Q22's
default), the source-count words ("from 6 Deliverables"), the three starters (the offers when
no skill is installed, and with nothing to read), the words an offer names its skill with, and
the text part's lines (what the posted message says beside its ``calibration@1`` part, for a
surface that can't draw the card, and for the model's history), and the routes' refusals and
notes, keyed by their code (US-C22b). The language is W2's K7 chooser
(``services.brief.strings.brief_language``): a tag's own table, else its language's, else English.
Every key is in both tables (``tests/test_prd268w3_calibration_build.py`` fails otherwise).

A count is a count of sources by kind, in the order each kind first appears, and never a
percentage (§12-Q26). The text part quotes Auto's lines and offers as plain text
(``services.proactive.delivery_text.inert``): nothing the model wrote renders as a link, an
image, code or HTML in the chat, which renders markdown.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import List, Mapping, Optional, Sequence

from modules.proactive.locale import DEFAULT_LOCALE
from services.brief.strings import brief_language
from services.calibration.model import SOURCE_TYPES, Calibration, CalibrationSource
from services.proactive.delivery_text import inert

__all__ = ["STARTER_KEYS", "STRINGS", "calibration_language", "calibration_text", "consent_line", "named_offer",
           "say", "source_count", "starters"]

STARTER_KEYS = ("starter_1", "starter_2", "starter_3")
# The mark each line and offer of the text part starts with.
BULLET = "- "

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        "consent": "Auto reads the titles of your recent work to say what it knows.",
        "deliverable_one": "{count} Deliverable",
        "deliverable_many": "{count} Deliverables",
        "conversation_one": "{count} Conversation",
        "conversation_many": "{count} Conversations",
        "document_one": "{count} document",
        "document_many": "{count} documents",
        "skill_one": "{count} skill",
        "skill_many": "{count} skills",
        "from_one": "from {last}",
        "from_many": "from {rest} and {last}",
        "list_separator": ", ",
        "starter_1": "Draft a document from a short brief",
        "starter_2": "Plan this week's work with you",
        "starter_3": "Research a topic and summarise it",
        "offer_named": "{offer} (with {skill})",
        "text_lines": "Here is what I think I know about your work so far, from the titles of your recent work:",
        "text_line": "{text} ({count})",
        "text_empty": "There is nothing of yours for me to read yet.",
        "text_no_lines": "I read the titles of your recent work, and nothing in them is clear enough to say back yet.",
        "text_offers": "Things I can do for you:",
        "text_close": "Reject anything that is wrong, or tell me it sounds right.",
        # The routes' refusals and notes (api/auto_calibration.py), by code.
        "not_available": "Calibration is not available on this server.",
        "calibration_no_workspace": "Choose a workspace first.",
        "calibration_no_person": "Only a signed-in person can answer this.",
        "calibration_not_a_member": "You are not a member of this workspace.",
        "already_offered": "Auto has offered you this already.",
        "calibration_not_found": "That calibration was not found.",
        "line_not_found": "That line was not found.",
        "calibration_answered": "You have answered this already.",
        "calibration_unavailable": "Auto could not do that just now. Try again in a moment.",
        "memory_unavailable": "Auto could not remember that just now, so nothing changed. Try again in a moment.",
        "not_stored": "Not kept in memory: it looks like it holds something private, such as a password or a key.",
    }),
    "pt-BR": MappingProxyType({
        "consent": "O Auto lê os títulos do seu trabalho recente para dizer o que sabe.",
        "deliverable_one": "{count} Entregável",
        "deliverable_many": "{count} Entregáveis",
        "conversation_one": "{count} Conversa",
        "conversation_many": "{count} Conversas",
        "document_one": "{count} documento",
        "document_many": "{count} documentos",
        "skill_one": "{count} skill",
        "skill_many": "{count} skills",
        "from_one": "de {last}",
        "from_many": "de {rest} e {last}",
        "list_separator": ", ",
        "starter_1": "Escrever um documento a partir de um resumo curto",
        "starter_2": "Planejar o trabalho desta semana com você",
        "starter_3": "Pesquisar um tema e resumir",
        "offer_named": "{offer} (com {skill})",
        "text_lines": "Isto é o que acho que sei sobre o seu trabalho até agora, pelos títulos do seu trabalho recente:",
        "text_line": "{text} ({count})",
        "text_empty": "Ainda não há nada seu para eu ler.",
        "text_no_lines": "Li os títulos do seu trabalho recente, e nada neles está claro o bastante para eu dizer ainda.",
        "text_offers": "O que posso fazer por você:",
        "text_close": "Rejeite o que estiver errado, ou diga que está certo.",
        "not_available": "A calibração não está disponível neste servidor.",
        "calibration_no_workspace": "Escolha um espaço de trabalho primeiro.",
        "calibration_no_person": "Só uma pessoa conectada pode responder isto.",
        "calibration_not_a_member": "Você não é membro deste espaço de trabalho.",
        "already_offered": "O Auto já ofereceu isto a você.",
        "calibration_not_found": "Essa calibração não foi encontrada.",
        "line_not_found": "Essa linha não foi encontrada.",
        "calibration_answered": "Você já respondeu isto.",
        "calibration_unavailable": "O Auto não conseguiu fazer isso agora. Tente de novo em instantes.",
        "memory_unavailable": "O Auto não conseguiu guardar isso agora, então nada mudou. Tente de novo em instantes.",
        "not_stored": "Não guardado na memória: parece conter algo privado, como uma senha ou uma chave.",
    }),
})


def calibration_language(tag: Optional[str]) -> str:
    """The table a BCP 47 ``tag`` reads (W2's K7 chooser): its own, its language's, else English."""
    return brief_language(tag)


def say(language: str, key: str) -> str:
    """The sentence ``key`` in ``language``, English when the language has no table."""
    return STRINGS.get(language, STRINGS[DEFAULT_LOCALE])[key]


def consent_line(language: str) -> str:
    """What the person agrees to before anything is read (§12-Q22)."""
    return say(language, "consent")


def starters(language: str) -> List[str]:
    """The three starters, in order."""
    return [say(language, key) for key in STARTER_KEYS]


def named_offer(language: str, offer: str, skill: str) -> str:
    """An offer as shown: the model's words, naming the skill they use when they don't already."""
    if skill.casefold() in offer.casefold():
        return offer
    return say(language, "offer_named").format(offer=offer, skill=skill)


def _counted(language: str, sources: Sequence[CalibrationSource]) -> List[str]:
    order: List[str] = []
    for source in sources:
        if source.type in SOURCE_TYPES and source.type not in order:
            order = [*order, source.type]
    counts = {kind: sum(1 for source in sources if source.type == kind) for kind in order}
    return [say(language, f"{kind}_{'one' if n == 1 else 'many'}").format(count=n) for kind, n in counts.items()]


def source_count(language: str, sources: Sequence[CalibrationSource]) -> str:
    """"from 6 Deliverables and 2 Conversations": each kind counted, in the order it first
    appears (as the web card counts them); empty for no sources."""
    parts = _counted(language, sources)
    if not parts:
        return ""
    if len(parts) == 1:
        return say(language, "from_one").format(last=parts[0])
    rest = say(language, "list_separator").join(parts[:-1])
    return say(language, "from_many").format(rest=rest, last=parts[-1])


def calibration_text(language: str, calibration: Calibration) -> str:
    """The text part: the lines with their counts (with none: that there is nothing to read yet, or,
    when there was, that nothing in it is clear enough), the offers, and how to answer. Auto's words
    are plain text that renders as itself."""
    if calibration.empty:
        lines = [say(language, "text_empty")]
    elif calibration.lines:
        lines = [say(language, "text_lines"), *(
            BULLET + say(language, "text_line").format(text=inert(line.text), count=source_count(language, line.sources))
            for line in calibration.lines
        )]
    else:
        lines = [say(language, "text_no_lines")]
    offers = [say(language, "text_offers"), *(BULLET + inert(offer) for offer in calibration.offers)]
    close = [] if not calibration.lines else [say(language, "text_close")]
    return "\n\n".join(("\n".join(lines), "\n".join(offers), *close))

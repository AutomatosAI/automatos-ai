"""PRD-268 §6.11.2, K7, K16 — what the brief says, from the strings table.

Server copy, never the model's: the greeting, the section labels, the empty sentence, "Show N
more", the row action labels, each row's reason (``why``) and the brief route's refusals
(``open_*`` and ``brief_off``, ``api/proactive_brief.py``), in English and Portuguese (Brazil).
Clients draw these; they never compose them.

The language (K7): ``brief_language`` reads a BCP 47 tag (the route's ``locale``, a phone's
``mobile_devices.locale``, or the locale W1's ``Accept-Language`` chooser gave). A tag with a
table reads it; any other tag falls to its language's table (``pt-PT`` and ``pt`` read
Portuguese (Brazil), ``en-GB`` English), else English.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional

from modules.proactive.locale import DEFAULT_LOCALE, ENGLISH, PORTUGUESE_BRAZIL

__all__ = ["GREETING_HOURS", "STRINGS", "brief_language", "greeting", "more", "say"]

# A tag's language is the part before its first separator; '_' is read as '-' ("pt_BR").
_SEPARATOR = "-"
_TABLES = {ENGLISH.lower(): ENGLISH, PORTUGUESE_BRAZIL.lower(): PORTUGUESE_BRAZIL}
_LANGUAGES = {"en": ENGLISH, "pt": PORTUGUESE_BRAZIL}
# A tag is read this far and no further (BCP 47 tags are short).
MAX_TAG_CHARS = 35
# The local hour each greeting starts at: morning from midnight, afternoon from noon, evening from 18:00.
GREETING_HOURS = ((18, "evening"), (12, "afternoon"), (0, "morning"))

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        "greeting_morning": "Morning, {name}.",
        "greeting_morning_anon": "Morning.",
        "greeting_afternoon": "Afternoon, {name}.",
        "greeting_afternoon_anon": "Afternoon.",
        "greeting_evening": "Evening, {name}.",
        "greeting_evening_anon": "Evening.",
        "section_needs_you": "Needs you",
        "section_suggestions": "Suggested",
        "section_today": "Today",
        "section_since": "Since your last brief",
        "section_unchecked": "Couldn't check",
        "empty": "Nothing new needs you, and nothing is scheduled today.",
        "more": "Show {n} more",
        "action_start": "Start",
        "action_start_done": "Started",
        "action_answer": "Answer",
        "action_open": "Open",
        "action_connect": "Set up",
        "action_hide": "Hide",
        "action_hide_done": "Hidden",
        "untitled": "Untitled",
        "today_routine": "An agent's routine runs.",
        "today_playbook": "A scheduled Playbook runs.",
        "today_task": "A scheduled task runs.",
        "today_mission": "A Mission's deadline.",
        "today_task_due": "A ticket's deadline.",
        "today_social": "A social post goes out.",
        "today_other": "Scheduled for today.",
        "since_ready": "Finished since your last brief.",
        "since_failed": "Started from a suggestion. It couldn't finish.",
        "since_working": "Started from a suggestion. The work is under way.",
        "since_started": "Started from a suggestion.",
        "open_no_workspace": "Choose a workspace first.",
        "open_no_person": "A brief is a person's, and this caller is not one.",
        "open_not_a_member": "Today's brief is only for the people in this workspace.",
        "open_out_of_window": "That date is more than a day from today, so there is no brief for it.",
        "open_unavailable": "Today's brief could not be made right now. Try again in a minute.",
        "brief_off": "The brief on open is not switched on for this workspace.",
    }),
    "pt-BR": MappingProxyType({
        "greeting_morning": "Bom dia, {name}.",
        "greeting_morning_anon": "Bom dia.",
        "greeting_afternoon": "Boa tarde, {name}.",
        "greeting_afternoon_anon": "Boa tarde.",
        "greeting_evening": "Boa noite, {name}.",
        "greeting_evening_anon": "Boa noite.",
        "section_needs_you": "Precisa de você",
        "section_suggestions": "Sugestões",
        "section_today": "Hoje",
        "section_since": "Desde o seu último resumo",
        "section_unchecked": "Não deu para verificar",
        "empty": "Nada novo precisa de você, e nada está agendado para hoje.",
        "more": "Mostrar mais {n}",
        "action_start": "Começar",
        "action_start_done": "Começou",
        "action_answer": "Responder",
        "action_open": "Abrir",
        "action_connect": "Configurar",
        "action_hide": "Ocultar",
        "action_hide_done": "Oculta",
        "untitled": "Sem título",
        "today_routine": "A rotina de um agente roda.",
        "today_playbook": "Um Playbook agendado roda.",
        "today_task": "Uma tarefa agendada roda.",
        "today_mission": "O prazo de uma Missão.",
        "today_task_due": "O prazo de um ticket.",
        "today_social": "Um post social é publicado.",
        "today_other": "Agendado para hoje.",
        "since_ready": "Terminou desde o seu último resumo.",
        "since_failed": "Começou a partir de uma sugestão. Não deu para terminar.",
        "since_working": "Começou a partir de uma sugestão. O trabalho está em andamento.",
        "since_started": "Começou a partir de uma sugestão.",
        "open_no_workspace": "Escolha um workspace primeiro.",
        "open_no_person": "Um resumo é de uma pessoa, e quem chamou não é uma.",
        "open_not_a_member": "O resumo de hoje é só para as pessoas deste workspace.",
        "open_out_of_window": "Essa data está a mais de um dia de hoje, então não há resumo para ela.",
        "open_unavailable": "Não foi possível preparar o resumo de hoje agora. Tente de novo em um minuto.",
        "brief_off": "O resumo ao abrir não está ativado neste workspace.",
    }),
})


def brief_language(tag: Optional[str]) -> str:
    """The table a BCP 47 ``tag`` reads: its own, else its language's, else English (K7)."""
    if not isinstance(tag, str):
        return DEFAULT_LOCALE
    normal = tag.strip()[:MAX_TAG_CHARS].replace("_", _SEPARATOR).lower()
    return _TABLES.get(normal) or _LANGUAGES.get(normal.split(_SEPARATOR, 1)[0], DEFAULT_LOCALE)


def say(language: str, key: str) -> str:
    """The sentence ``key`` in ``language``, English when the language has no table."""
    return STRINGS.get(language, STRINGS[DEFAULT_LOCALE])[key]


def more(language: str, hidden: int) -> str:
    """"Show N more" for a section that folded ``hidden`` rows."""
    return say(language, "more").format(n=hidden)


def greeting(language: str, name: Optional[str], hour: int) -> str:
    """The greeting for the local ``hour``, naming the person when their name is known."""
    part = next(label for start, label in GREETING_HOURS if hour >= start)
    if not name:
        return say(language, f"greeting_{part}_anon")
    return say(language, f"greeting_{part}").format(name=name)

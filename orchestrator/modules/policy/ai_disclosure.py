"""EU AI Act Article 50 disclosure rules (PRD-264).

Article 50(1) says a person is told they are dealing with an AI at first
exposure; Article 50(5) says the notice is clear and can't be hidden. This
module is the one place a notice is built: the badge table, the notice texts
and the pure functions every surface calls. Nothing else builds a notice
string (PRD-264 §9).

D17: a deployer may reword the widget's first-turn line, but the badge is
fixed, comes from :data:`BADGE_BY_LANGUAGE`, and a reworded line must contain
the locale's badge term as a whole word. The server refuses any other line.

Pure: stdlib only, no DB, no config, no network.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# ── The badge table (D17) ──────────────────────────────────────────────────
# Keyed by the language subtag of a locale ("pt-BR" → "pt"). Any language not
# listed falls back to English. No setting changes a badge.
DEFAULT_LANGUAGE = "en"
BADGE_BY_LANGUAGE: Mapping[str, str] = MappingProxyType({
    "en": "AI",
    "pt": "IA",
    "es": "IA",
    "fr": "IA",
    "it": "IA",
    "de": "KI",
})
DEFAULT_BADGE = BADGE_BY_LANGUAGE[DEFAULT_LANGUAGE]

# ── Notice texts (English defaults, PRD-264 §8) ────────────────────────────
WIDGET_NOTICE_LINE = "You are chatting with an AI assistant for {site}."
WIDGET_NOTICE_LINE_UNNAMED = "You are chatting with an AI assistant."

# ── Messages a deployer sees when a save is refused (HTTP 422) ─────────────
LINE_MISSING_TERM_MESSAGE = "The AI notice must say it is an AI (for this language: '{term}')."
LINE_NOT_TEXT_MESSAGE = "The AI notice must be text."
LINE_TOO_LONG_MESSAGE = "The AI notice must be at most {limit} characters."
BLOCK_NOT_OBJECT_MESSAGE = "ai_disclosure must be an object, for example {\"line\": \"...\"}."
BLOCK_FIXED_FIELDS_MESSAGE = "ai_disclosure accepts only {fields}; the AI badge is fixed."
MAX_LINE_CHARS = 280

# ── Where a deployer's choices live in site (or workspace) settings ────────
SETTINGS_KEY = "ai_disclosure"
LOCALE_SETTINGS_KEY = "locale"  # the merchant's locale (modules/widgets/i18n.py)
LINE_FIELD = "line"
EDITABLE_FIELDS: frozenset[str] = frozenset({LINE_FIELD})

# ── Wire names on the widget stream and the generic webhook payload ────────
DISCLOSURE_EVENT = "ai_disclosure"
AI_GENERATED_FIELD = "ai_generated"
AI_DISCLOSURE_FIELD = "ai_disclosure"

# ── Channel bots (EUAI-03, PRD-264 §8) ─────────────────────────────────────
CHANNEL_NOTICE = "AI assistant for {principal}."
CHANNEL_NOTICE_UNNAMED = "AI assistant."
TELEGRAM_START_INTRO = "I'm an AI assistant for {principal}."
TELEGRAM_START_INTRO_UNNAMED = "I'm an AI assistant."
NOTICE_SEPARATOR = "\n\n"

# ── Auto Live voice (EUAI-04, PRD-264 §8, Guidelines ¶37) ──────────────────
VOICE_GREETING = "Hi, this is Auto, your AI assistant."
VOICE_REMINDER = "Just so you know, I'm an AI assistant."

# FR-15: an auto-created workspace is named "{name or email}'s Workspace"
# (core/auth/hybrid.py), so that form, or any name with an "@", may carry the
# owner's name or email and is never used as a public name.
AUTO_WORKSPACE_NAME_SUFFIX = "'s workspace"
PERSONAL_DATA_MARK = "@"

# A message asking whether the other side is a person or a bot: the reply to it
# always carries the notice. English, Portuguese, Spanish, French, German and
# Italian, matched case-insensitively in the first HUMAN_QUESTION_MAX_CHARS
# characters of the message. Channel messages are attacker-controlled, so no
# pattern puts two quantifiers that can match the same characters side by side
# (``\s+a?\s*`` backtracks quadratically on a run of spaces), and the scan is
# capped so its cost is bounded whatever the message length.
HUMAN_QUESTION_MAX_CHARS = 2000
HUMAN_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\bare\s+you\s+(?:an?\s+)?(?:real\s+)?(?:human|person|bot|robot|machine|ai|real)\b",
    r"\b(?:am\s+i|are\s+we)\s+(?:talking|speaking|chatting)\s+(?:to|with)\s+(?:an?\s+)?"
    r"(?:real\s+)?(?:human|person|bot|robot|machine|ai)\b",
    r"\bis\s+(?:this|that)\s+(?:an?\s+)?(?:real\s+)?(?:human|person|bot|robot|ai)\b",
    r"\b(?:voc[eê]|tu)\s+[eé]s?\s+(?:uma?\s+)?(?:humano|humana|pessoa|rob[oô]|bot|ia)\b",
    r"\b(?:estou|est[aá]s?)\s+(?:a\s+)?(?:falar|falando|conversar)\s+com\s+(?:uma?\s+)?"
    r"(?:humano|pessoa|rob[oô]|bot|ia)\b",
    r"\b(?:eres|es\s+usted)\s+(?:una?\s+)?(?:humano|humana|persona|robot|bot|ia)\b",
    r"\bestoy\s+hablando\s+con\s+(?:una?\s+)?(?:humano|persona|robot|bot|ia)\b",
    r"\b(?:es-tu|êtes-vous|tu\s+es|vous\s+êtes)\s+(?:une?\s+)?(?:humain|humaine|personne|robot|bot|ia)\b",
    r"\bje\s+parle\s+(?:à|a|avec)\s+(?:une?\s+)?(?:humain|personne|robot|bot|ia)\b",
    r"\b(?:bist\s+du|sind\s+sie)\s+(?:eine?\s+)?(?:mensch|bot|roboter|ki)\b",
    r"\b(?:rede|spreche|chatte)\s+ich\s+mit\s+(?:eine[mr]\s+)?(?:menschen?|bot|roboter|ki)\b",
    r"\bsei\s+(?:una?\s+|un'\s*)?(?:umano|umana|persona|robot|bot|ia)\b",
    r"\bsto\s+parlando\s+con\s+(?:una?\s+|un'\s*)?(?:umano|persona|robot|bot|ia)\b",
)
_HUMAN_QUESTION = re.compile("|".join(HUMAN_QUESTION_PATTERNS), re.IGNORECASE)

_LOCALE_SEPARATOR = re.compile(r"[-_]")


class AIDisclosureRejected(ValueError):
    """A deployer's notice setting that would not say it is an AI (D17)."""


@dataclass(frozen=True)
class AIDisclosure:
    """The notice a widget shows before its first answer.

    ``badge`` is the fixed locale term; ``line`` the first-turn sentence;
    ``locale`` the language whose badge term the notice uses.
    """

    badge: str
    line: str
    locale: str

    def as_dict(self) -> dict[str, str]:
        """The public shape: ``{"badge", "line", "locale"}``."""
        return {"badge": self.badge, "line": self.line, "locale": self.locale}


def language_of(locale: Any) -> str:
    """The table language for a locale code; English when unknown or unset."""
    if not isinstance(locale, str) or not locale.strip():
        return DEFAULT_LANGUAGE
    language = _LOCALE_SEPARATOR.split(locale.strip(), maxsplit=1)[0].lower()
    return language if language in BADGE_BY_LANGUAGE else DEFAULT_LANGUAGE


def badge_for(locale: Any) -> str:
    """The fixed AI badge for a locale (D17): AI, IA or KI; AI otherwise."""
    return BADGE_BY_LANGUAGE[language_of(locale)]


def contains_badge_term(line: str, term: str) -> bool:
    """True when ``term`` appears in ``line`` as a whole word, case as written.

    "l'IA" and "KI-Assistent" count; "MAIL" and "said" don't.
    """
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(pattern, line) is not None


def default_line(site_name: Any) -> str:
    """The default first-turn line, naming the site when it has a public name.

    ``site_name`` is a Site's ``display_name``. Sites backfilled from a
    workspace carry the workspace's name, so it passes the same FR-15 test as
    any public name (:func:`public_site_name`); otherwise the unnamed line.
    """
    name = public_site_name(site_name)
    if name:
        return WIDGET_NOTICE_LINE.format(site=name)
    return WIDGET_NOTICE_LINE_UNNAMED


def check_line(line: Any, locale: Any) -> Optional[str]:
    """A deployer's line, stripped, or ``None`` when they set none.

    Raises :class:`AIDisclosureRejected` when the line is not text, is too
    long, or lacks the locale's badge term as a whole word.
    """
    if line is None:
        return None
    if not isinstance(line, str):
        raise AIDisclosureRejected(LINE_NOT_TEXT_MESSAGE)
    stripped = line.strip()
    if not stripped:
        return None
    if len(stripped) > MAX_LINE_CHARS:
        raise AIDisclosureRejected(LINE_TOO_LONG_MESSAGE.format(limit=MAX_LINE_CHARS))
    term = badge_for(locale)
    if not contains_badge_term(stripped, term):
        raise AIDisclosureRejected(LINE_MISSING_TERM_MESSAGE.format(term=term))
    return stripped


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Refuse settings whose ``ai_disclosure`` block would not say it is an AI.

    ``settings`` is the full settings dict as it will be stored, so a change
    of ``locale`` alone is checked against the line already saved.
    """
    block = settings.get(SETTINGS_KEY)
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise AIDisclosureRejected(BLOCK_NOT_OBJECT_MESSAGE)
    if set(block) - EDITABLE_FIELDS:
        fields = ", ".join(f"'{name}'" for name in sorted(EDITABLE_FIELDS))
        raise AIDisclosureRejected(BLOCK_FIXED_FIELDS_MESSAGE.format(fields=fields))
    check_line(block.get(LINE_FIELD), settings.get(LOCALE_SETTINGS_KEY))


def touches_disclosure(settings_patch: Mapping[str, Any]) -> bool:
    """True when a settings change can alter the notice (its line or locale)."""
    return SETTINGS_KEY in settings_patch or LOCALE_SETTINGS_KEY in settings_patch


def _stored_line(settings: Mapping[str, Any]) -> Optional[str]:
    """The deployer's saved line if it still passes D17, else ``None``.

    A line saved before this rule, or under another locale, is not served:
    the default notice is, and the refusal is logged.
    """
    block = settings.get(SETTINGS_KEY)
    if not isinstance(block, Mapping):
        return None
    try:
        return check_line(block.get(LINE_FIELD), settings.get(LOCALE_SETTINGS_KEY))
    except AIDisclosureRejected as exc:
        logger.warning("ai_disclosure: stored line refused (%s); serving the default notice", exc)
        return None


def build_disclosure(settings: Optional[Mapping[str, Any]], site_name: Any) -> AIDisclosure:
    """Build the widget notice from settings; never copies a badge from them.

    ``site_name`` is the Site's ``display_name`` or ``None``, never the
    workspace's name (FR-15). Missing or non-dict settings give the English
    default.
    """
    stored = settings if isinstance(settings, Mapping) else {}
    locale = stored.get(LOCALE_SETTINGS_KEY)
    return AIDisclosure(
        badge=badge_for(locale),
        line=_stored_line(stored) or default_line(site_name),
        locale=language_of(locale),
    )


def disclosure_event_payload(disclosure: AIDisclosure, conversation_id: str) -> dict[str, str]:
    """The first event of a widget chat stream: ``{"type": "ai_disclosure", ...}``."""
    return {"type": DISCLOSURE_EVENT, **disclosure.as_dict(), "conversation_id": conversation_id}


def mark_ai_generated(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of an assistant message payload marked ``"ai_generated": true``."""
    return {**payload, AI_GENERATED_FIELD: True}


# ── Channel bots (EUAI-03) ─────────────────────────────────────────────────

def _public_text(value: Any) -> Optional[str]:
    """``value`` stripped when it is text that carries no personal-data mark."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or PERSONAL_DATA_MARK in stripped:
        return None
    return stripped


def _public_chosen_name(value: Any) -> Optional[str]:
    """``value`` stripped when it is public text and not the auto-created form."""
    text = _public_text(value)
    if text and not text.casefold().endswith(AUTO_WORKSPACE_NAME_SUFFIX):
        return text
    return None


def public_principal_name(company_name: Any, workspace_name: Any) -> Optional[str]:
    """The public name a notice may give for whom the AI acts (FR-15), or ``None``.

    The business profile's ``company_name`` first; then the workspace name,
    unless it is the auto-created "{name or email}'s Workspace" form. A name
    holding an "@" is never used: it may be the owner's email.
    """
    return _public_text(company_name) or _public_chosen_name(workspace_name)


def public_site_name(display_name: Any) -> Optional[str]:
    """A Site's ``display_name`` when a widget notice may show it (FR-15), or ``None``.

    The deployer sets it, but Sites created from a workspace (the PRD-008-A
    backfill, ``ensure_shopify_site``) copy the workspace's name, which may be
    the auto-created "{name or email}'s Workspace". That form, and any name
    holding an "@", is never shown.
    """
    return _public_chosen_name(display_name)


def channel_notice(principal: Optional[str]) -> str:
    """The channel bot notice: "AI assistant for {principal}." or "AI assistant."."""
    if principal:
        return CHANNEL_NOTICE.format(principal=principal)
    return CHANNEL_NOTICE_UNNAMED


def telegram_start_intro(principal: Optional[str]) -> str:
    """The first sentence of the Telegram /start reply."""
    if principal:
        return TELEGRAM_START_INTRO.format(principal=principal)
    return TELEGRAM_START_INTRO_UNNAMED


def human_question_window(text: str) -> str:
    """The part of a message :func:`asks_if_human` checks: its first
    :data:`HUMAN_QUESTION_MAX_CHARS` characters."""
    return text[:HUMAN_QUESTION_MAX_CHARS]


def asks_if_human(text: Any) -> bool:
    """True when a message asks whether it is talking to a person or a bot.

    Only :func:`human_question_window` of the message is scanned, so the cost
    stays bounded on an arbitrarily long, attacker-written message.
    """
    if not isinstance(text, str):
        return False
    return _HUMAN_QUESTION.search(human_question_window(text)) is not None


def channel_notice_due(
    *,
    last_notice_at: Optional[float],
    now: float,
    idle_seconds: float,
    inbound_text: Any = None,
) -> bool:
    """Whether a message to a person on a channel carries the AI notice.

    The one rule for both outbound channel paths (PRD-264 EUAI-03): the first
    message to that person on that connection (``last_notice_at`` is ``None``,
    which also stands for "the record can't be read": fail toward telling),
    the first after ``idle_seconds`` since the last notice, and any reply to a
    message asking whether they are talking to a human or a bot.
    """
    if asks_if_human(inbound_text):
        return True
    if last_notice_at is None:
        return True
    return now - last_notice_at >= idle_seconds


def prepend_notice(notice: Optional[str], text: str) -> str:
    """``text`` with ``notice`` as its first paragraph; ``text`` when no notice."""
    if not notice:
        return text
    return f"{notice}{NOTICE_SEPARATOR}{text}"


# ── Auto Live voice (EUAI-04) ──────────────────────────────────────────────

def voice_greeting(configured: Any, locale: Any) -> str:
    """The line Auto speaks first on a call: ``configured`` when it says it is an AI.

    A configured greeting that is empty, not text, or lacks the AI term for
    the call's language (:func:`badge_for`) is never spoken: the default
    :data:`VOICE_GREETING` is, and the refusal is logged. The greeting can be
    reworded, never silenced.
    """
    if isinstance(configured, str) and configured.strip():
        stripped = configured.strip()
        if contains_badge_term(stripped, badge_for(locale)):
            return stripped
    logger.warning("ai_disclosure: voice greeting refused (no AI term); speaking the default")
    return VOICE_GREETING


def voice_reminder_due(
    *,
    reconnected: bool,
    last_notice_at: float,
    now: float,
    interval_seconds: float,
) -> bool:
    """Whether Auto's next spoken response opens with :data:`VOICE_REMINDER`.

    After an automatic reconnect, and once ``interval_seconds`` of call time
    have passed since the caller last heard that Auto is an AI. An interval of
    0 or less reminds on every response: the reminder never turns off.
    """
    if reconnected:
        return True
    return now - last_notice_at >= interval_seconds

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

# ── Wire names on the widget stream ────────────────────────────────────────
DISCLOSURE_EVENT = "ai_disclosure"
AI_GENERATED_FIELD = "ai_generated"

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
    """The default first-turn line, naming the site when it has a name."""
    if isinstance(site_name, str) and site_name.strip():
        return WIDGET_NOTICE_LINE.format(site=site_name.strip())
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

    ``site_name`` is the site's name, or the workspace's when the site has
    none. Missing or non-dict settings give the English default.
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

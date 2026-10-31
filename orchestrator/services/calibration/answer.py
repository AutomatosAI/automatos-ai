"""PRD-268 §6.11.5, §12-Q26, K15 — what of the model's answer a calibration keeps.

The model answers JSON (``services.calibration.prompt``); nothing it writes is trusted, and every
title it was given may have been written to steer it:

* **A line** is kept when its text is one plain line of at most ``MAX_LINE_CHARS`` characters,
  it cites at least one ref, every ref it cites is one it was given (provenance: a line citing a
  source it was not given is dropped whole, never trimmed to the refs that hold), and at least
  one of them is the person's work (a skill alone says what Auto can do, not what they work on).
  A line carrying a percentage or a confidence number ("85%", "9/10", "nine out of ten",
  "confidence 0.8") is dropped (§12-Q26: those are numbers the model makes up, shown as if
  measured), and so is one carrying a link, a web address, an email address or a phone number
  (Auto's words on the card and in the chat point nowhere). The first ``MAX_LINES`` distinct
  lines are kept, each with its sources as read.
* **An offer** is kept when it is short and cites an installed skill it was given; it is shown
  naming that skill (``strings.named_offer``), and the words as shown pass the same checks. The
  offers are topped up to ``OFFERS`` with the strings' starters, so there are always three.

Each check reads the words in Unicode's compatibility form (NFKC), so a full-width "％" is a
"%". ``parse_answer`` reads the JSON object out of the answer (fenced or bare); None when there
is none.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, List, Mapping, Optional, Tuple

from modules.tools.discovery.suggestion_fields import plain_text
from services.calibration.model import (
    MAX_LINE_CHARS, MAX_LINES, OFFERS, SKILL, WORK_TYPES, CalibrationLine, CalibrationSource,
)
from services.calibration.prompt import OFFER_CHARS
from services.calibration.sources import Read
from services.calibration.strings import named_offer, starters

__all__ = ["carries_a_link", "carries_a_number_for_sureness", "kept_lines", "kept_offers", "parse_answer"]

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# A percentage, in any of the person's languages.
_PERCENT = re.compile(r"%|\b(?:percent|per\s?cent|pct|por\s?cento|porcentagem|percentual)\b", re.IGNORECASE)
# A word for how sure, within a few characters of a number, either way round.
_SURE_WORDS = r"(?:confidence|confident|certainty|likelihood|probability|confiança|confiante|certeza|probabilidade)"
_SURENESS = re.compile(rf"\b{_SURE_WORDS}\b[^.\n]{{0,24}}\d|\d[^.\n]{{0,24}}\b{_SURE_WORDS}\b", re.IGNORECASE)
# A rating, in digits or words: "9/10", "8 out of 10", "nine out of ten", "8 de cada 10", "9 de 10".
_NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|um|uma|dois|duas|três|quatro|cinco|seis|" \
                r"sete|oito|nove|dez)"
_RATING = re.compile(
    rf"\b(?:\d+(?:[.,]\d+)?|{_NUMBER_WORDS})\s*(?:/|out of|de cada|de|em)\s*(?:5|10|100|five|ten|a hundred|cinco|"
    r"dez|cem)\b",
    re.IGNORECASE,
)
# A link or an address: a scheme, www., an email address, a web address, or a phone number (9 digits or more).
_LINK = re.compile(
    r"[a-z][a-z0-9+.-]*://|\bwww\.|[\w.+-]+@[\w-]+\.\w"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|co|app|dev|ai|info|biz|xyz|link|ly|gl|me|tv|br|pt|uk|us|"
    r"eu|de|fr|es|it|nl)\b"
    r"|(?:\+|\b)\d(?:[\s().-]*\d){8,}",
    re.IGNORECASE,
)
LINE_PREFIX = "line-"


def parse_answer(content: Any) -> Optional[Mapping[str, Any]]:
    """The JSON object in the model's answer, fenced or bare; None when there is none."""
    said = str(content or "")
    fenced = _FENCED.search(said)
    body = fenced.group(1) if fenced else said[said.find("{"): said.rfind("}") + 1]
    try:
        found = json.loads(body)
    except (ValueError, RecursionError):
        return None
    return found if isinstance(found, dict) else None


def carries_a_number_for_sureness(words: str) -> bool:
    """Whether ``words`` hold a percentage, a rating or a confidence number (§12-Q26)."""
    normal = unicodedata.normalize("NFKC", words)
    return bool(_PERCENT.search(normal) or _SURENESS.search(normal) or _RATING.search(normal))


def carries_a_link(words: str) -> bool:
    """Whether ``words`` hold a link, a web or email address, or a phone number."""
    return bool(_LINK.search(unicodedata.normalize("NFKC", words)))


def _shown(words: str) -> bool:
    """Whether words may be shown as Auto's: no number for sureness, nothing that points anywhere."""
    return not (carries_a_number_for_sureness(words) or carries_a_link(words))


def _plain(value: Any, limit: int) -> Optional[str]:
    """``value`` as one plain line (no control or format character) within ``limit`` characters, else None."""
    if not isinstance(value, str) or "\n" in value.strip():
        return None
    words = plain_text(value)
    if not words or len(words) > limit or not _shown(words):
        return None
    return words


def _cited(value: Any, refs: Mapping[str, Read]) -> Optional[Tuple[CalibrationSource, ...]]:
    """The sources a line cites, in order and once each; None when it cites nothing, anything
    unread, or none of the person's work."""
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(ref, str) and ref.strip() in refs for ref in value):
        return None
    cited = tuple(refs[ref].source for ref in dict.fromkeys(ref.strip() for ref in value))
    return cited if any(source.type in WORK_TYPES for source in cited) else None


def kept_lines(answer: Mapping[str, Any], refs: Mapping[str, Read]) -> List[CalibrationLine]:
    """The lines that hold to the rules, at most ``MAX_LINES``, numbered as kept."""
    raw = answer.get("lines")
    kept: List[Tuple[str, Tuple[CalibrationSource, ...]]] = []
    for item in raw if isinstance(raw, list) else []:
        text = _plain(item.get("text"), MAX_LINE_CHARS) if isinstance(item, dict) else None
        sources = _cited(item.get("sources"), refs) if text else None
        if text and sources and text.casefold() not in {seen.casefold() for seen, _ in kept}:
            kept = [*kept, (text, sources)]
    return [CalibrationLine(id=f"{LINE_PREFIX}{n}", text=text, sources=list(sources))
            for n, (text, sources) in enumerate(kept[:MAX_LINES], 1)]


def _offer(item: Any, refs: Mapping[str, Read], language: str) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    text = _plain(item.get("text"), OFFER_CHARS)
    skill = refs.get(item.get("skill")) if isinstance(item.get("skill"), str) else None
    if not text or skill is None or skill.source.type != SKILL:
        return None
    shown = named_offer(language, text, skill.source.label)
    return shown if _shown(shown) else None


def kept_offers(answer: Mapping[str, Any], refs: Mapping[str, Read], language: str) -> List[str]:
    """Exactly ``OFFERS`` offers: those naming an installed skill, then the starters."""
    raw = answer.get("offers")
    offers = [offer for offer in (_offer(item, refs, language) for item in (raw if isinstance(raw, list) else []))
              if offer]
    distinct = list(dict.fromkeys(offers))[:OFFERS]
    topped = [*distinct, *(starter for starter in starters(language) if starter not in distinct)]
    return topped[:OFFERS]

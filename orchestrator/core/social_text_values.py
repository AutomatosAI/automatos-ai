"""F378 (night 11, 7 Oct): words a post may carry, and two kinds it never may.

Night 11 saved a caption reading "We sold {retail_bags_sold} retail bags…" (B15) and
printed "true" as a quote card's eyebrow (B-i3-5). Neither is words for the post:

* **A template placeholder** — ``{name}``, ``{{ name }}`` or a bracketed field such as
  ``[retail_bags_sold]`` or ``[YOUR NAME]`` — is a field nobody filled. It reaches the
  post when a model copies a skill's example or a schema's field name.
* **A programming literal** — ``true``, ``false``, ``null``, ``none``, ``undefined`` —
  standing alone as a text value is a switch or an empty value written as a word.

:func:`text_value_problem` says which, for one text value; the template contract
(``core/social_templates.py``) refuses such a value for a text variable, so the
composer, a post's render and an agent's ``generate_document`` all refuse it.
:func:`placeholder_in` finds a placeholder in a post's copy (the save and the composer).
:func:`placeholder_label` turns one into the plain words of an owner's question.

Pure: the standard library only, like the contract that imports it.
"""
from __future__ import annotations

import re
from typing import Optional

# A field name as a template or a skill example writes it.
_FIELD = r"[A-Za-z_][A-Za-z0-9_.]*"
PLACEHOLDER_PATTERNS = (
    re.compile(r"\{\{\s*" + _FIELD + r"\s*\}\}"),  # {{ name }}
    re.compile(r"\{\s*" + _FIELD + r"\s*\}"),  # {name}
    re.compile(r"\[\s*[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\s*\]"),  # [snake_name]
    re.compile(r"\[\s*(?:YOUR|INSERT|ADD|ENTER)\b[^\[\]]{0,60}\]", re.IGNORECASE),  # [Your name], [Insert date]
)
# Words that are a value's type, never its text, when they are the whole value.
NOT_TEXT_WORDS = frozenset({"true", "false", "null", "none", "undefined"})
_LABEL_TRIM = re.compile(r"^[\[{\s]+|[\]}\s]+$")
_LABEL_PREFIX = re.compile(r"^(?:your|insert|add|enter)\s+", re.IGNORECASE)


def placeholder_in(text: object) -> Optional[str]:
    """The first template placeholder in ``text`` (``{name}``, ``{{ name }}``, ``[a_name]``,
    ``[Your name]``), or ``None`` for words alone or for anything that is not text."""
    if not isinstance(text, str):
        return None
    found = [match for pattern in PLACEHOLDER_PATTERNS if (match := pattern.search(text))]
    return min(found, key=lambda match: match.start()).group(0) if found else None


def is_literal_word(text: object) -> bool:
    """Whether ``text`` is a programming literal standing alone (``"true"``, ``"None"``)."""
    return isinstance(text, str) and text.strip().lower() in NOT_TEXT_WORDS


def text_value_problem(value: str) -> Optional[str]:
    """Why ``value`` is no text a post can show, or ``None``: a literal word or a placeholder."""
    if is_literal_word(value):
        return f"is the word {value.strip()!r}, not words for the post: write the text, or leave it empty"
    found = placeholder_in(value)
    if found:
        return f"holds the placeholder {found}: write the real words, or leave it empty"
    return None


def placeholder_label(placeholder: str) -> str:
    """A placeholder in an owner's words: ``{retail_bags_sold}`` → ``Retail bags sold``."""
    inner = _LABEL_PREFIX.sub("", _LABEL_TRIM.sub("", placeholder))
    words = " ".join(inner.replace(".", " ").replace("_", " ").split())
    return words[:1].upper() + words[1:].lower() if words else placeholder

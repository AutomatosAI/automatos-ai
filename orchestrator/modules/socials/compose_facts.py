"""F378 (night 11, 7 Oct): the composer adds no fact the brief does not give.

Night 11's owner: "It invents things… That's my name on a public post." The model's
proposal is checked here, after ``compose_checks.py`` has checked its shape, for what it
says rather than how it is built:

* **Placeholders in the copy** (B15): a ``{name}`` or ``[a_name]`` left in a channel's
  text is a fact nobody gave. The composer asks once for the copy without it
  (``compose.py``); what is still there is a warning, and an owner's question in plain
  words ("Retail bags sold").

Each check returns what it found; ``checked_proposal`` adds it to the warnings and to
``questions``, the list the editor shows as "Auto needs: …".
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from core.social_text_values import PLACEHOLDER_PATTERNS, placeholder_label

BASE = "base"
PLACEHOLDER_WARNING = "{where} holds the placeholder {found}: write the real words before saving"


def _copy_texts(copy: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Each text of the copy with where it is, in the owner's words."""
    texts = [("The copy", copy.get(BASE))]
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    texts += [(f"The {toolkit} copy", text) for toolkit, text in channels.items()]
    return [(where, text) for where, text in texts if isinstance(text, str) and text]


def copy_placeholders(copy: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Every placeholder the copy holds, ``(where, placeholder)``, each text's in order."""
    found: List[Tuple[str, str]] = []
    for where, text in _copy_texts(copy):
        matches = sorted((m for pattern in PLACEHOLDER_PATTERNS for m in pattern.finditer(text)), key=lambda m: m.start())
        end = -1
        for match in matches:  # "{{ name }}" holds "{ name }": the outer one only
            if match.start() >= end:
                found.append((where, match.group(0)))
                end = match.end()
    return found


def placeholder_notes(copy: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """The warnings and the owner's questions for the placeholders left in the copy."""
    found = copy_placeholders(copy)
    warnings = [PLACEHOLDER_WARNING.format(where=where, found=placeholder) for where, placeholder in found]
    return warnings, unique([placeholder_label(placeholder) for _where, placeholder in found])


def unique(items: List[str]) -> List[str]:
    """``items`` with each one once, in order."""
    return list(dict.fromkeys(item for item in items if item))


def with_notes(proposal: Dict[str, Any], warnings: List[str], questions: List[str]) -> Dict[str, Any]:
    """``proposal`` with ``warnings`` and ``questions`` added to its own, each once."""
    return {
        **proposal,
        "warnings": unique([*proposal.get("warnings", []), *warnings]),
        "questions": unique([*proposal.get("questions", []), *questions]),
    }


def checked(proposal: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``proposal`` (``compose_checks``' checked shape) with what it says checked against
    what the composer was given (``ctx``): its warnings and the owner's questions added."""
    warnings, questions = placeholder_notes(proposal.get("copy") or {})
    return with_notes(proposal, warnings, questions)

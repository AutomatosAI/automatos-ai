"""F378 (night 11, 7 Oct): the composer adds no fact the brief does not give.

Night 11's owner: "It invents things… That's my name on a public post." The model's
proposal is checked here, after ``compose_checks.py`` has checked its shape, for what it
says rather than how it is built:

* **Placeholders in the copy** (B15): a ``{name}`` or ``[a_name]`` left in a channel's
  text is a fact nobody gave. The composer asks once for the copy without it
  (``compose.py``); what is still there is a warning, and an owner's question in plain
  words ("Retail bags sold").

* **Numbers and handles nobody gave** (B-I5-4, ``compose_given.py``): a figure the brief,
  the current take and the bound sources do not hold is a warning; a handle field takes the
  brand kit's handle only.
* **The model's own questions** (``questions`` in its answer) are kept, text only.
* **A photo the template shows** (B19, ``compose_photos.py``): a required one is a warning
  and an owner's question, an optional one a warning.
* **A required field left blank** is taken out, so it is asked for (``compose.py``'s
  follow-ups), and what is still missing becomes the owner's question by its label.

Each check returns what it found; ``checked_proposal`` adds it to the warnings and to
``questions``, the list the editor shows as "Auto needs: …".
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from core.social_text_values import PLACEHOLDER_PATTERNS, placeholder_label
from modules.socials import compose_given, compose_photos

BASE = "base"
MAX_QUESTIONS = 10
QUESTION_MAX_CHARS = 200
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


def field_label(name: str, spec: Any) -> str:
    """A template field in the owner's words: its label, else its name as words."""
    label = spec.get("label") if isinstance(spec, Mapping) else None
    return str(label).strip() if isinstance(label, str) and label.strip() else placeholder_label(name)


def model_questions(raw: Any) -> List[str]:
    """The questions the model asks the owner: text only, ``MAX_QUESTIONS`` of them at most."""
    asked = raw if isinstance(raw, list) else []
    return unique([" ".join(q.split())[:QUESTION_MAX_CHARS] for q in asked if isinstance(q, str)])[:MAX_QUESTIONS]


def without_blank_required(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """``proposal`` without the empty text it gave a field that has no default: a required
    field left blank is missing, so it is asked for, never rendered empty."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    variables = proposal.get("variables") or {}
    kept = {
        name: spec for name, spec in variables.items()
        if not (isinstance(schema.get(name), Mapping) and "default" not in schema[name]
                and isinstance(spec.get("value"), str) and not spec["value"].strip())
    }
    return proposal if len(kept) == len(variables) else {**proposal, "variables": kept}


def checked(proposal: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``proposal`` (``compose_checks``' checked shape) with what it says checked against
    what the composer was given (``ctx``): its warnings and the owner's questions added."""
    warnings, questions = placeholder_notes(proposal.get("copy") or {})
    proposal, given = compose_given.given_notes(without_blank_required(proposal), ctx)
    photo_warnings, photo_questions = compose_photos.photo_notes(proposal, ctx)
    return with_notes(proposal, [*warnings, *given, *photo_warnings], [*questions, *photo_questions])

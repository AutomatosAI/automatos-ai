"""F378 (night 11, 7 Oct): what the owner gave, and what the composer added on its own.

Night 11's posts carried facts nobody gave: "Worldwide shipping", "Wholesale Growth", a
price, a handle (B-I5-4). Names and tasting notes can only be ruled out by the prompt, but
two kinds of invention are cheap to catch here, on the checked proposal:

* **Numbers.** Every figure in the copy and in the template's fields must appear in what
  the composer was given: the brief (a retake's guidance is in it), the post's current
  take on a retake (``ComposeContext.current_take``) and the sources a claim is bound to. A
  figure from nowhere is a warning naming it, so the owner checks it before approval.
  Whole numbers up to ``SMALL_NUMBER_MAX`` are left alone: a slide counter, a star rating
  and "3 reasons" are the template's, not a claim.
* **Handles.** The template's handle field takes one of the brand kit's
  ``social_handles`` only: another is replaced by the kit's handle for the post's first
  channel that has one, or taken out when the kit has none (the field then shows nothing,
  or is asked for). An ``@mention`` in the copy that neither the brief nor the kit gives is
  a warning.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

SMALL_NUMBER_MAX = 10
HANDLE_VARIABLES = ("handle",)
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.]*[A-Za-z0-9_]")
NUMBERS_WARNING = "Numbers nobody gave: {numbers}. Check each one, or take it out before approval"
HANDLE_REPLACED = "The handle {given} is not one of your brand kit's; {used} is used"
HANDLE_DROPPED = "The handle {given} is not in your brand kit, so it was left out: add your handles to the brand kit"
MENTION_WARNING = "The copy mentions {mentions}, which neither the brief nor your brand kit gives: check it"


def _figure(raw: str) -> str:
    """One way of writing a number: ``1,200`` and ``1200.0`` are ``1200``."""
    text = raw.replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.lstrip("0") or "0"


def figures(text: str) -> Set[str]:
    """The numbers ``text`` holds, each written one way (``_figure``)."""
    return {_figure(raw) for raw in _NUMBER.findall(text or "")}


def _is_small(figure: str) -> bool:
    return figure.isdigit() and int(figure) <= SMALL_NUMBER_MAX


def _value_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(_value_text(candidate.get(key)) for key in ("title", "detail", "value"))


def bound_candidates(proposal: Mapping[str, Any], ctx: Any) -> List[Mapping[str, Any]]:
    """The candidates the proposal's claims are bound to."""
    index = {(str(c.get("kind")), str(c.get("ref"))): c for c in getattr(ctx, "candidates", ())}
    keys = {(str(s.get("kind")), str(s.get("ref"))) for s in (proposal.get("sources") or {}).values() if isinstance(s, Mapping)}
    return [index[key] for key in keys if key in index]


def given_text(proposal: Mapping[str, Any], ctx: Any) -> str:
    """Everything the owner gave the composer, as one text: the brief, the current take, the
    kit's handles and the sources the claims are bound to."""
    current = getattr(ctx, "current_take", None) or {}
    parts = [str(getattr(ctx, "brief", "") or ""), json.dumps(current, ensure_ascii=False, default=str) if current else ""]
    parts += [str(handle) for handle in (getattr(ctx, "handles", None) or {}).values()]
    parts += [_candidate_text(candidate) for candidate in bound_candidates(proposal, ctx)]
    return "\n".join(part for part in parts if part)


def _defaults(proposal: Mapping[str, Any]) -> Dict[str, Any]:
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    return {name: spec.get("default") for name, spec in schema.items() if isinstance(spec, Mapping)}


def proposal_texts(proposal: Mapping[str, Any]) -> Iterable[str]:
    """The words the post shows: its copy and its fields' values (a template's own default aside)."""
    copy = proposal.get("copy") or {}
    yield _value_text(copy.get("base"))
    yield from (_value_text(text) for text in (copy.get("channels") or {}).values())
    defaults = _defaults(proposal)
    for name, spec in (proposal.get("variables") or {}).items():
        value = spec.get("value") if isinstance(spec, Mapping) else None
        if value != defaults.get(name):
            yield _value_text(value)


def unknown_numbers(proposal: Mapping[str, Any], ctx: Any) -> List[str]:
    """The figures the post shows that nothing the composer was given holds, in order."""
    given = figures(given_text(proposal, ctx))
    shown: List[str] = []
    for text in proposal_texts(proposal):
        shown += [_figure(raw) for raw in _NUMBER.findall(text)]
    return list(dict.fromkeys(f for f in shown if f not in given and not _is_small(f)))


def _norm(handle: Any) -> str:
    return str(handle or "").strip().lstrip("@").lower()


def _kit_handle(ctx: Any) -> Optional[str]:
    """The kit's handle for the post's first channel that has one, else its first."""
    handles = dict(getattr(ctx, "handles", None) or {})
    for channel in getattr(ctx, "channels", ()):
        if handles.get(str(channel.get("toolkit"))):
            return handles[str(channel.get("toolkit"))]
    return next(iter(handles.values()), None)


def _known(handle: str, ctx: Any) -> bool:
    kit = {_norm(h) for h in (getattr(ctx, "handles", None) or {}).values()}
    return _norm(handle) in kit or _norm(handle) in str(getattr(ctx, "brief", "") or "").lower()


def checked_handles(variables: Mapping[str, Any], ctx: Any) -> Tuple[Dict[str, Any], List[str]]:
    """``variables`` with each handle field one of the kit's (``HANDLE_VARIABLES``), and why."""
    kept, warnings = dict(variables), []
    for name in HANDLE_VARIABLES:
        given = _value_text((kept.get(name) or {}).get("value")).strip()
        if not given or _known(given, ctx):
            continue
        used = _kit_handle(ctx)
        if used:
            kept[name] = {"value": used, "claim": False}
            warnings.append(HANDLE_REPLACED.format(given=given, used=used))
        else:
            kept.pop(name)
            warnings.append(HANDLE_DROPPED.format(given=given))
    return kept, warnings


def unknown_mentions(proposal: Mapping[str, Any], ctx: Any) -> List[str]:
    """The ``@mentions`` in the copy that neither the brief nor the kit gives."""
    copy = proposal.get("copy") or {}
    texts = [_value_text(copy.get("base")), *(_value_text(t) for t in (copy.get("channels") or {}).values())]
    mentions = [m for text in texts for m in _MENTION.findall(text)]
    return list(dict.fromkeys(m for m in mentions if not _known(m, ctx)))


def given_notes(proposal: Dict[str, Any], ctx: Any) -> Tuple[Dict[str, Any], List[str]]:
    """``proposal`` with its handles from the kit, and the warnings for what nobody gave."""
    variables, warnings = checked_handles(proposal.get("variables") or {}, ctx)
    proposal = {**proposal, "variables": variables}
    numbers = unknown_numbers(proposal, ctx)
    if numbers:
        warnings.append(NUMBERS_WARNING.format(numbers=", ".join(numbers)))
    mentions = unknown_mentions(proposal, ctx)
    if mentions:
        warnings.append(MENTION_WARNING.format(mentions=", ".join(mentions)))
    return proposal, warnings

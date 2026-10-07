"""F378 (night 11, 7 Oct): a claim's source is one that holds its figure; nothing names a source that isn't.

B1: the Infographic compose bound the owner's café numbers to three unrelated reports, and
printed an invented source label ("Internal Report, 2026-10-06"). The candidates were the
workspace's newest eight of each kind, whatever the brief said, and a binding was checked
for existence only. Now:

* **Candidates come from the brief's words** (:func:`brief_terms`): its most distinctive
  words are searched (``modules/socials/sources.search``), each address it quotes too.
* **A claim bound to a source must show a figure that source holds** (its title, detail or
  value; an address the brief quotes is the owner's own citation and is never read). A claim
  whose figure is not there is unbound, with a warning. A figure the owner gave in the brief
  stays on the post unsourced, as the owner's own.
* **The source line is never invented** (``SOURCE_LABEL_VARIABLES``): it names the bound
  source (its title and day, as a bound chart's chip does, ``core.chart_binding.chip_text``),
  or words the brief gives, or nothing. With nothing, the field is left for the owner (a
  question) and never asked of the model again (:func:`unaskable`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from core.chart_binding import chip_text, max_chars_of
from modules.socials.compose_given import figures

BRIEF_TERMS = 4  # the brief's words a candidate search is run for
TERM_MIN_CHARS = 4
URL_KIND = "url"
SOURCE_LABEL_VARIABLES = ("source_label",)
_WORD = re.compile(r"[^\W\d_][\w'-]+", re.UNICODE)
_URL = re.compile(r"https?://\S+")
# Words too common to find a source by.
STOP_WORDS = frozenset({
    "about", "after", "again", "also", "because", "been", "before", "being", "between", "brief", "could", "every",
    "from", "have", "here", "into", "just", "like", "make", "more", "most", "much", "only", "other", "over", "please",
    "post", "posts", "same", "should", "some", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "very", "want", "week", "were", "what", "when", "where", "which", "while", "with",
    "would", "your", "yours", "ours", "social", "instagram", "linkedin", "twitter", "tiktok", "youtube", "facebook",
})
UNBOUND_WARNING = "The figure of {name} ({value}) is not in its source {title!r}; it was unbound"
LABEL_DROPPED = "The source line {given!r} names no source of this post: it was left for you to fill"
LABEL_FROM_SOURCE = "The source line names the bound source: {label}"


def brief_terms(brief: str) -> List[str]:
    """The brief's most distinctive words, longest first: what its candidate sources are searched by."""
    words = [w.lower().strip("'-") for w in _WORD.findall(_URL.sub(" ", brief or ""))]
    kept = [w for w in dict.fromkeys(words) if len(w) >= TERM_MIN_CHARS and w not in STOP_WORDS]
    return sorted(kept, key=len, reverse=True)[:BRIEF_TERMS]


def _value(spec: Any) -> Any:
    return spec.get("value") if isinstance(spec, Mapping) else None


def _text(value: Any) -> str:
    return "" if value is None or isinstance(value, bool) else str(value)


def _candidates(ctx: Any) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(c.get("kind")), str(c.get("ref"))): c for c in getattr(ctx, "candidates", ())}


def _holds(candidate: Mapping[str, Any], value: Any) -> bool:
    """Whether the candidate holds every figure of ``value`` (a value without one passes)."""
    wanted = figures(_text(value))
    if not wanted or candidate.get("kind") == URL_KIND:
        return True
    held = figures(" ".join(_text(candidate.get(key)) for key in ("title", "detail", "value")))
    return wanted <= held


def checked_bindings(proposal: Mapping[str, Any], ctx: Any) -> Tuple[Dict[str, Any], List[str]]:
    """The proposal's sources, each claim kept bound only to a source that holds its figure."""
    index, kept, warnings = _candidates(ctx), {}, []
    variables = proposal.get("variables") or {}
    for name, source in (proposal.get("sources") or {}).items():
        candidate = index.get((str(source.get("kind")), str(source.get("ref"))))
        value = _value(variables.get(name))
        if candidate is None or _holds(candidate, value):
            kept[name] = source
        else:
            warnings.append(UNBOUND_WARNING.format(name=name, value=_text(value), title=str(candidate.get("title") or "")))
    return kept, warnings


def _bound_label(sources: Mapping[str, Any], ctx: Any, limit: Optional[int]) -> Optional[str]:
    index = _candidates(ctx)
    for source in sources.values():
        candidate = index.get((str(source.get("kind")), str(source.get("ref"))))
        if candidate is not None and candidate.get("title"):
            return chip_text(str(candidate["title"]), source.get("as_of") or candidate.get("as_of"), limit)
    return None


def _in_brief(text: str, ctx: Any) -> bool:
    return bool(text.strip()) and text.strip().lower() in str(getattr(ctx, "brief", "") or "").lower()


def checked_label(proposal: Mapping[str, Any], sources: Mapping[str, Any], ctx: Any) -> Tuple[Dict[str, Any], List[str]]:
    """The proposal's variables with each source line naming the bound source, words the brief
    gives, or left out when it has neither."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    variables, warnings = dict(proposal.get("variables") or {}), []
    for name in (n for n in SOURCE_LABEL_VARIABLES if n in schema):
        given = _text(_value(variables.get(name))).strip()
        label = _bound_label(sources, ctx, max_chars_of(schema, name))
        if label:
            if given != label:
                variables[name] = {"value": label, "claim": False}
                warnings.append(LABEL_FROM_SOURCE.format(label=label))
        elif given and not _in_brief(given, ctx):
            variables.pop(name)
            warnings.append(LABEL_DROPPED.format(given=given))
    return variables, warnings


def source_notes(proposal: Dict[str, Any], ctx: Any) -> Tuple[Dict[str, Any], List[str]]:
    """``proposal`` with its claims bound only where the source holds the figure and its source
    line from that source (or the brief), and the warnings saying what changed."""
    sources, unbound = checked_bindings(proposal, ctx)
    variables, labelled = checked_label(proposal, sources, ctx)
    return {**proposal, "sources": sources, "variables": variables}, [*unbound, *labelled]


def unaskable(proposal: Mapping[str, Any], ctx: Any) -> Set[str]:
    """F378: the fields never asked of the model: a source line with no source bound (the
    owner names the source, or it stays empty)."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    if _bound_label(proposal.get("sources") or {}, ctx, None):
        return set()
    return {name for name in SOURCE_LABEL_VARIABLES if name in schema}

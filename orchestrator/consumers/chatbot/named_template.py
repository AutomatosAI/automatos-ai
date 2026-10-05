"""F351 (night 10b, build 19): a template the owner names gets the studio's prompt.

Asked in plain English for a document on a named template, Auto made a usable one 1 time in
3 (iteration 4: 0 of 3). It invented values instead of asking (INV-0043, a blank Bill-to,
"Net 30"), called Basic Report the "Branded Data Sheet" (82a31e89), and made "Harbourline
Invoice" on the Branded Invoice and called it the owner's, "which I assume is what you mean"
(e786217e, deliverable 557d64d0). With the template studio's own "Use with Auto" prompt pasted
first (``frontend/components/documents/blocks/promptSnippets.ts``: the name, the template_id,
every field, each list's columns) it was 6 of 7 first time.

Now each owner turn is checked, before the model's first call, for a template of this
workspace named by its exact name (any case, quotes optional) or its template_id, in this
message or, when this one names none, in the owner's last few messages. The turn then gets the
studio's shape for it: the exact name and id, every field from ``platform_get_template_schema``
(the tool's own reader, so its fields, columns and required flags are the ones the tool gives;
the studio's list columns when the schema carries none), and the rules: this template and no
other, the owner's words onto the fields, one short question for whatever the owner didn't
give, never an invented number, address, term, price or date, and ``data`` keyed by the field
names. A template the owner names that this workspace hasn't got gets "there's no template by
that name" and the closest few names instead, so nothing is swapped in silently. No extra
model call; the read runs off the event loop. A widget visitor's turn gets none of it: the
templates are the owner's.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

FOLLOW_UP_TURNS = 3          # the owner's earlier messages a named template carries through
CLOSEST_NAMES = 3            # names offered when the named template isn't there
MAX_ASKED_WORDS = 6          # words before "template" read as the name the owner gave

BESIDE_TEMPLATE, ASKED_FOR, MENTIONED = 2, 1, 0   # how a name is said: "X template" / "make an X" / "as X"

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_TEMPLATE = re.compile(r"\btemplates?\b", re.I)
_TEMPLATE_ONE = re.compile(r"\btemplate\b", re.I)
_TEMPLATE_ID = re.compile(r"\btemplate[\s_-]*id\b", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_QUOTED = r"""(?:^|(?<=[\s(]))["']([^"'\n]{2,80}?)["'](?=[\s.,;:!?)]|$)"""
_QUOTED_AFTER = re.compile(r"\btemplate\s+(?:(?:called|named|literally|exactly|is)\s+){0,3}" + _QUOTED, re.I)
_QUOTED_BEFORE = re.compile(_QUOTED + r"\s+template\b", re.I)
_QUOTED_CALLED = re.compile(r"\b(?:called|named)\s+(?:(?:exactly|literally)\s+)?" + _QUOTED, re.I)
# "on the Branded Invoice", "make a Branded Invoice": the owner asks for it, not about it.
_PUT_ON = re.compile(r"\b(?:use|using|on|onto|with|from|make|making|generate|create|redo|want|need)\s+"
                     r"(?:me\s+)?(?:(?:the|my|our|a|an)\s+)?$", re.I)
_CLAUSE_BREAK = re.compile(r"""[,.;:!?\n"]""")
_STOP_WORDS = frozenset(
    "the a an my our your own on onto using use with from in into to for of and or called named per via by as "
    "is it's that this these which what off".split())
_GENERIC = frozenset(
    "document documents doc docx pdf word new blank social image video default existing different other another "
    "same any some one usual normal standard right correct proper last previous old".split())


@dataclass(frozen=True)
class NamedTemplate:
    """What the owner named: a template of this workspace (``row``) or a name that has none."""

    asked: str
    row: Any = None
    closest: Tuple[str, ...] = ()
    earlier: bool = False
    strong: bool = True      # by id, by the name beside "template", or asked for; not just mentioned


def _plain(text: object) -> str:
    return str(text or "").translate(_QUOTES)


def _latest_rows(rows: Sequence[Any]) -> List[Any]:
    """One row per name (case-insensitive): the first, which ``list_templates`` orders newest."""
    seen, out = set(), []
    for row in rows:
        key = _plain(row.name).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _by_id(text: str, rows: Sequence[Any]) -> Optional[Any]:
    ids = {m.group().lower() for m in _UUID.finditer(text)}
    return next((row for row in rows if str(row.id).lower() in ids), None)


def _name_pattern(name: str) -> re.Pattern:
    words = _plain(name).split()
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(w) for w in words) + r"(?!\w)", re.I)


Rank = Tuple[int, int, int]


def _rank(text: str, row: Any, match: re.Match) -> Optional[Rank]:
    """How strongly ``match`` names ``row`` (BESIDE_TEMPLATE, ASKED_FOR or MENTIONED, then the
    longer name, then the first said), or None when nothing says it's a template. A one-word
    name ("Invoice") counts only beside "template", or capitalised in a message about templates."""
    before, after = text[:match.start()], text[match.end():]
    quoted = before.endswith(("'", '"')) and after.startswith(("'", '"'))
    beside = quoted or bool(_TEMPLATE_ONE.match(after.lstrip()))
    capitalised = match.group() != match.group().lower()
    mentions = bool(_TEMPLATE.search(text))
    one_word = len(_plain(row.name).split()) == 1
    if not (beside or (capitalised and mentions) or (not one_word and (capitalised or mentions))):
        return None
    level = BESIDE_TEMPLATE if beside else ASKED_FOR if _PUT_ON.search(before) else MENTIONED
    return level, len(_plain(row.name)), -match.start()


def _by_name(text: str, rows: Sequence[Any]) -> List[Tuple[Rank, Any]]:
    """Every template whose exact name the text gives, strongest first."""
    ranked = []
    for row in rows:
        for match in _name_pattern(row.name).finditer(text):
            rank = _rank(text, row, match)
            if rank is not None:
                ranked.append((rank, row))
    return sorted(ranked, key=lambda pair: pair[0], reverse=True)


def _same_name(a: object, b: object) -> bool:
    return " ".join(_plain(a).lower().split()) == " ".join(_plain(b).lower().split())


def _words_before_template(text: str) -> str:
    """The name in "… my Harbourline Invoice template": the words back to the first stop word."""
    for match in _TEMPLATE_ONE.finditer(text):
        clause = _CLAUSE_BREAK.split(text[:match.start()])[-1].split()
        words: List[str] = []
        for word in reversed(clause[-MAX_ASKED_WORDS:]):
            if word.lower() in _STOP_WORDS:
                break
            words.insert(0, word)
        if words and not all(w.lower() in _GENERIC for w in words):
            return " ".join(words)
    return ""


def asked_name(text: object) -> str:
    """The template name the owner gave, matched or not: quoted beside "template", the words
    before "template", or a template_id; '' when the message names no template."""
    said = _plain(text)
    id_given = _UUID.search(said) if _TEMPLATE_ID.search(said) else None
    if id_given:
        return id_given.group()
    for pattern in (_QUOTED_AFTER, _QUOTED_BEFORE):
        found = pattern.search(said)
        if found:
            return found.group(1).strip()
    called = _QUOTED_CALLED.search(said) if _TEMPLATE.search(said) else None
    return _words_before_template(said) or (called.group(1).strip() if called else "")


def closest_names(asked: str, rows: Sequence[Any], limit: int = CLOSEST_NAMES) -> Tuple[str, ...]:
    """The workspace's template names nearest ``asked``: shared words first, then spelling."""
    wanted = set(asked.lower().split())

    def score(name: str) -> Tuple[float, float]:
        words = set(name.lower().split())
        shared = len(wanted & words) / len(wanted) if wanted else 0.0
        return shared, difflib.SequenceMatcher(None, asked.lower(), name.lower()).ratio()

    names = sorted((_plain(row.name) for row in rows), key=lambda n: (tuple(-s for s in score(n)), n))
    return tuple(names[:limit])


def _chosen(asked: str, rows: Sequence[Any], ranked: List[Tuple[Rank, Any]]) -> Optional[Any]:
    """The template the owner asked for by name: the one called what they gave beside
    "template", the one named beside "template", or the one asked for ("make a Branded
    Invoice") whose name holds every word they gave ("… on the invoice template")."""
    wanted = set(asked.lower().split())
    given = next((r for r in rows if asked and _same_name(r.name, asked)), None)
    beside = next((r for rank, r in ranked if rank[0] == BESIDE_TEMPLATE), None)
    covers = next((r for rank, r in ranked
                   if wanted and rank[0] == ASKED_FOR and wanted <= set(_plain(r.name).lower().split())), None)
    return given or beside or covers


def named_in(text: object, rows: Sequence[Any]) -> Optional[NamedTemplate]:
    """The template ``text`` names among ``rows`` (one workspace's), a name it gives that none
    has, or None when it names no template."""
    said, rows = _plain(text), _latest_rows(rows)
    row = _by_id(said, rows)
    if row is not None:
        return NamedTemplate(asked=_plain(row.name), row=row)
    asked, ranked = asked_name(said), _by_name(said, rows)
    row = _chosen(asked, rows, ranked)
    if row is not None:
        return NamedTemplate(asked=_plain(row.name), row=row)
    if asked:   # a name the owner gave beside "template": a looser match never stands in for it
        return NamedTemplate(asked=asked, closest=closest_names(asked, rows))
    if not ranked:
        return None
    rank, row = ranked[0]
    return NamedTemplate(asked=_plain(row.name), row=row, strong=rank[0] == ASKED_FOR)


def named_in_conversation(texts: Sequence[str], rows: Sequence[Any]) -> Optional[NamedTemplate]:
    """``texts`` is the owner's latest message, then earlier ones newest first. The newest that
    names a template decides; a template only mentioned ("the same fields as the Branded Invoice")
    decides only when no turn in the window asks for one."""
    mentioned = None
    for turn, text in enumerate(texts):
        named = named_in(text, rows)
        if named is None:
            continue
        named = replace(named, earlier=turn > 0)
        if named.strong:
            return named
        mentioned = mentioned or named
    return mentioned


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def owner_turns(llm_messages: List[Dict[str, Any]], latest_text: str,
                earlier: int = FOLLOW_UP_TURNS) -> List[str]:
    """The owner's latest message, then up to ``earlier`` before it, newest first."""
    said = [_text(m.get("content")) for m in llm_messages if isinstance(m, dict) and m.get("role") == "user"]
    if said and said[-1].strip() == str(latest_text or "").strip():
        said = said[:-1]
    return [str(latest_text or ""), *reversed(said[-earlier:] if earlier else [])]


__all__ = ["CLOSEST_NAMES", "FOLLOW_UP_TURNS", "NamedTemplate", "asked_name", "closest_names", "named_in",
           "named_in_conversation", "owner_turns"]

"""PRD-268 US-C03b (§6.2.3, §6.2.4) — what ``platform_suggest_next_step`` was asked to record, read field by field.

A check-in's model writes these arguments after reading untrusted data (tickets, reports,
documents), so nothing is stored as sent: each argument is read into a structured field,
checked against its limit, and the suggestion is built from those fields alone.

* ``title`` (required, at most 120 characters, one line) and ``detail`` (at most 600): plain
  text. Control and format characters (zero-width, bidi overrides) are dropped and spaces
  collapsed, so a title can neither dodge dedupe nor spoof a card; the surfaces render both
  as text, never markup. A title that says nothing once normalised is refused.
* ``evidence`` (required): one to five ``{type, id}`` refs, kept as ``{type, id}`` and
  nothing else (a ticket by its board number, as its subject key writes it).
* ``proposed_action`` (required): what Start does. In W1 a check-in proposes a board ticket
  (``board_task``, optionally for one agent: ``payload.agent_id``) or a chat with Auto
  (``chat``, no payload); a Mission or playbook run is not proposed until W4 (K12).
* ``severity``: ``today`` or ``can_wait`` (the default).
* ``subject_key``: the dedupe key, ``ticket:<number>`` (backed by that ticket's read in the
  evidence) or ``title:<title>``, canonicalised as ``modules.proactive.dedupe`` writes it;
  left out, the title's key. ``agent:<id>:error:<class>`` is refused: a failing streak is
  recorded by the check-in itself (§6.2.2), never by its model, which could otherwise hold
  the streak's key open with its own words.

``read_suggestion`` raises ``ValueError`` with the sentence the model reads, so it can
correct the call.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from core.models.proactive import DETAIL_LENGTH, SEVERITY_DEFAULT, TITLE_LENGTH
from modules.proactive import normalise_title, ticket_key, title_key
from modules.proactive.provenance import MAX_EVIDENCE, TICKET, ref_key

__all__ = [
    "BOARD_TASK",
    "CHAT",
    "MAX_AGENT_ID",
    "PAYLOAD_AGENT",
    "PROPOSED_KINDS",
    "SEVERITIES",
    "Suggestion",
    "canonical_subject_key",
    "plain_text",
    "read_suggestion",
]

BOARD_TASK = "board_task"
CHAT = "chat"
# K12: what a W1 check-in may propose. Mission and playbook_run come with US-C14 (W4).
PROPOSED_KINDS = (BOARD_TASK, CHAT)
# The one payload field: the agent a proposed ticket is for (board_task only).
PAYLOAD_AGENT = "agent_id"
SEVERITIES = ("today", SEVERITY_DEFAULT)
# agents.id is a Postgres integer.
MAX_AGENT_ID = 2**31 - 1
# Dropped from stored text: controls, format characters (zero-width, bidi overrides),
# surrogates and private use.
_DROPPED_CATEGORIES = ("Cc", "Cf", "Cs", "Co")
NEWLINE = "\n"

_KEY_FORMS = "ticket:<number> or title:<title>"
_TICKET_PREFIX = f"{TICKET}:"
_STREAK_PREFIX = "agent"


@dataclass(frozen=True)
class Suggestion:
    """A suggestion as the tool records it: every field read and checked."""

    title: str
    detail: str
    evidence: Tuple[Dict[str, str], ...]
    proposed_action: Dict[str, Any]
    severity: str
    subject_key: str


def _plain_char(char: str, keep_lines: bool) -> str:
    if char == NEWLINE and keep_lines:
        return char
    if char.isspace():
        return " "
    return "" if unicodedata.category(char) in _DROPPED_CATEGORIES else char


def plain_text(value: str, keep_lines: bool = False) -> str:
    """``value`` as plain text: no control or format characters, each line's spaces
    collapsed; one line unless ``keep_lines``."""
    text = "".join(_plain_char(char, keep_lines) for char in value)
    return NEWLINE.join(" ".join(line.split()) for line in text.split(NEWLINE)).strip()


def _text(params: Mapping, name: str, limit: int, keep_lines: bool = False) -> str:
    value = params.get(name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} is text.")
    text = plain_text(value, keep_lines)
    if len(text) > limit:
        raise ValueError(f"{name} is at most {limit} characters; this one is {len(text)}. Shorten it.")
    return text


def _title(params: Mapping) -> str:
    title = _text(params, "title", TITLE_LENGTH)
    if not normalise_title(title):
        raise ValueError("Missing required parameter: title (what Auto tells the owner, in a few words).")
    return title


def _evidence(raw: Any) -> Tuple[Dict[str, str], ...]:
    refs = raw if isinstance(raw, (list, tuple)) else ()
    keys = [ref_key(ref) for ref in refs]
    if not keys or len(keys) > MAX_EVIDENCE or any(key is None for key in keys):
        raise ValueError(f"evidence is 1 to {MAX_EVIDENCE} {{type, id}} refs to reads this check-in made.")
    return tuple({"type": kind, "id": ident} for kind, ident in dict.fromkeys(keys))


def _agent_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip().isdecimal():
        raise ValueError(f"proposed_action.payload.{PAYLOAD_AGENT} is an agent's id, a whole number.")
    agent_id = int(str(value).strip())
    if not 0 < agent_id <= MAX_AGENT_ID:
        raise ValueError(f"proposed_action.payload.{PAYLOAD_AGENT} is an agent's id, a whole number.")
    return agent_id


def _payload(kind: str, raw: Any) -> Dict[str, Any]:
    payload = raw if raw is not None else {}
    if not isinstance(payload, Mapping):
        raise ValueError("proposed_action.payload is an object.")
    allowed = (PAYLOAD_AGENT,) if kind == BOARD_TASK else ()
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        takes = f"only {PAYLOAD_AGENT}" if allowed else "nothing"
        raise ValueError(f"proposed_action.payload for {kind} takes {takes}; not {', '.join(unknown)}.")
    agent = payload.get(PAYLOAD_AGENT)
    return {PAYLOAD_AGENT: _agent_id(agent)} if agent is not None else {}


def _proposed_action(raw: Any) -> Dict[str, Any]:
    kind = raw.get("kind") if isinstance(raw, Mapping) else None
    if kind not in PROPOSED_KINDS:
        raise ValueError(f"proposed_action is {{kind, payload}} with kind {' or '.join(PROPOSED_KINDS)}.")
    return {"kind": kind, "payload": _payload(kind, raw.get("payload"))}


def _severity(raw: Any) -> str:
    if raw is None or raw == "":
        return SEVERITY_DEFAULT
    if raw not in SEVERITIES:
        raise ValueError(f"severity is {' or '.join(SEVERITIES)}.")
    return raw


def canonical_subject_key(raw: Any) -> str:
    """A subject key the model gave, in its canonical form; ``ValueError`` when it is in
    neither form, is a failing streak's, or names nothing."""
    if not isinstance(raw, str):
        raise ValueError(f"subject_key is {_KEY_FORMS}.")
    prefix, _, rest = raw.strip().partition(":")
    form = prefix.strip().casefold()
    if form == _STREAK_PREFIX:
        raise ValueError("An agent's failing streak is recorded by the check-in itself; key this by its title.")
    reader = {TICKET: ticket_key, "title": title_key}.get(form)
    if reader is None:
        raise ValueError(f"subject_key is {_KEY_FORMS}.")
    try:
        return reader(rest)
    except ValueError as exc:
        raise ValueError(f"subject_key is {_KEY_FORMS} ({exc}).") from exc


def _subject_key(raw: Any, title: str, evidence: Tuple[Dict[str, str], ...]) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return title_key(title)
    key = canonical_subject_key(raw)
    ticket = key[len(_TICKET_PREFIX):] if key.startswith(_TICKET_PREFIX) else None
    if ticket is not None and {"type": TICKET, "id": ticket} not in evidence:
        raise ValueError(f"subject_key {key} names a ticket: cite that ticket's read in evidence.")
    return key


def read_suggestion(params: Optional[Mapping]) -> Suggestion:
    """The suggestion ``params`` describe, every field read and checked; ``ValueError``
    with the sentence the model reads when one is wrong."""
    params = params if isinstance(params, Mapping) else {}
    title = _title(params)
    evidence = _evidence(params.get("evidence"))
    return Suggestion(
        title=title,
        detail=_text(params, "detail", DETAIL_LENGTH, keep_lines=True),
        evidence=evidence,
        proposed_action=_proposed_action(params.get("proposed_action")),
        severity=_severity(params.get("severity")),
        subject_key=_subject_key(params.get("subject_key"), title, evidence),
    )

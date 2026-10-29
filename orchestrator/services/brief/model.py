"""PRD-268 §6.11.2, K11 — ``Brief@1``, the shape every surface reads.

```ts
type Brief = {
  v: 1; date: string; greeting: string
  prose: string                      // Auto's Read, unchanged: the fallback for any surface that can't draw the card
  spoken: string                     // the greeting and the numbered row titles, through speechify (K11)
  sections: { needs_you: BriefRow[]; suggestions: BriefRow[]; today: BriefRow[]; since: BriefRow[]; unchecked: BriefRow[] }
  counts: { needs_you: number; suggestions: number; today: number; since: number }  // exact, even when rows fold
  labels: { sections: Record<Section, string>; more: Partial<Record<Section, string>>; empty?: string }
  state_hash: string; generated_at: string
}
type BriefRow = {
  id: string; title: string; why: string; at?: string; number?: string
  display_state?: DisplayState
  opens?: { type: 'ticket' | 'mission' | 'question' | 'chat' | 'deliverable' | 'suggestion' | 'settings'; id: string }
  action?: { verb: 'start' | 'answer' | 'open' | 'connect' | 'hide'; label: string; done_label?: string }
}
```

Every id is a string (a number given for one is read as its string), and a field a reader
doesn't know is ignored. ``labels`` carries the server's copy for the card (K16: clients
never compose the section labels, "Show N more" or the empty sentence): each section's label,
"Show N more" for each section that folded, and the empty sentence when nothing is in any
section.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ACTION_VERBS", "Brief", "BriefAction", "BriefCounts", "BriefLabels", "BriefOpens", "BriefRow",
    "BriefSections", "COUNTED_SECTIONS", "NEEDS_YOU", "OPENS_TYPES", "PART_TYPE", "SECTIONS", "SINCE",
    "SUGGESTIONS", "TODAY", "UNCHECKED", "VERSION",
]

VERSION = 1
# The message part the brief is posted as, beside its text (services/chat_parts.py).
PART_TYPE = "brief@1"

NEEDS_YOU = "needs_you"
SUGGESTIONS = "suggestions"
TODAY = "today"
SINCE = "since"
UNCHECKED = "unchecked"
SECTIONS = (NEEDS_YOU, SUGGESTIONS, TODAY, SINCE, UNCHECKED)
# The sections with an exact count; ``unchecked`` lists every item it has.
COUNTED_SECTIONS = (NEEDS_YOU, SUGGESTIONS, TODAY, SINCE)

OPENS_TYPES = ("ticket", "mission", "question", "chat", "deliverable", "suggestion", "settings")
ACTION_VERBS = ("start", "answer", "open", "connect", "hide")

OpensType = Literal["ticket", "mission", "question", "chat", "deliverable", "suggestion", "settings"]
ActionVerb = Literal["start", "answer", "open", "connect", "hide"]
# §6.11.1's states (services/proactive/display_state.py DISPLAY_STATES).
DisplayState = Literal["needs_you", "working", "ready", "done", "failed"]


class _Shape(BaseModel):
    """Read leniently, held still: unknown fields are ignored, an id given as a number is its string."""

    model_config = ConfigDict(extra="ignore", frozen=True, coerce_numbers_to_str=True)


class BriefOpens(_Shape):
    """Where a row goes when pressed; a row with nowhere to go has none."""

    type: OpensType
    id: str


class BriefAction(_Shape):
    """The row's action, drawn as text ("Start →"); ``done_label`` replaces it once pressed."""

    verb: ActionVerb
    label: str
    done_label: Optional[str] = None


class BriefRow(_Shape):
    """One row: what it is (``title``), why it is here (``why``) and what it opens."""

    id: str
    title: str
    why: str
    at: Optional[str] = None
    number: Optional[str] = None
    display_state: Optional[DisplayState] = None
    opens: Optional[BriefOpens] = None
    action: Optional[BriefAction] = None


class BriefSections(_Shape):
    """The five sections' rows, as shown: each folds at 5 and never at 6."""

    needs_you: List[BriefRow] = []
    suggestions: List[BriefRow] = []
    today: List[BriefRow] = []
    since: List[BriefRow] = []
    unchecked: List[BriefRow] = []


class BriefCounts(_Shape):
    """Each counted section's exact number, whatever its rows show."""

    needs_you: int = 0
    suggestions: int = 0
    today: int = 0
    since: int = 0


class BriefLabels(_Shape):
    """The card's copy in the person's language: the section labels, "Show N more" for each
    folded section, and the empty sentence when every section is empty."""

    sections: Dict[str, str]
    more: Dict[str, str] = {}
    empty: Optional[str] = None


class Brief(_Shape):
    """``Brief@1``: one person's brief for one local date."""

    v: Literal[1]
    date: str
    greeting: str
    prose: str
    spoken: str
    sections: BriefSections
    counts: BriefCounts
    labels: BriefLabels
    state_hash: str
    generated_at: str

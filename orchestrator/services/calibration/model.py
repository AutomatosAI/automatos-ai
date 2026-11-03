"""PRD-268 §6.11.5, K15 — ``Calibration@1``, the shape every surface reads.

```ts
type Calibration = {
  v: 1; id: string
  lines: Array<{ id: string; text: string; sources: Array<{ type: 'deliverable' | 'conversation' | 'document' | 'skill'; id: string; label: string }> }>
  offers: string[]               // 3 things Auto can do, from installed skills
  empty: boolean                 // nothing to read yet: say so and offer 3 starters
}
```

What Auto thinks it knows about a person's work, from the titles and kinds of what they may
see (``services.calibration.sources``), and three things it can do. Each line carries the
sources it was read from, so a surface shows a count ("from 6 Deliverables"), never a
percentage (§12-Q26). Every id is a string (a number given for one is read as its string),
and a field a reader doesn't know is ignored. The web's copy of this shape is
``frontend/lib/proactive/types.ts`` ``Calibration``.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "CONVERSATION", "Calibration", "CalibrationLine", "CalibrationSource", "DELIVERABLE", "DOCUMENT",
    "MAX_LINES", "MAX_LINE_CHARS", "OFFERS", "PART_TYPE", "SKILL", "SOURCE_TYPES", "VERSION", "WORK_TYPES",
]

VERSION = 1
# The message part the calibration is posted as, beside its text (services/chat_parts.py, US-C22b).
PART_TYPE = "calibration@1"

DELIVERABLE = "deliverable"
CONVERSATION = "conversation"
DOCUMENT = "document"
SKILL = "skill"
SOURCE_TYPES = (DELIVERABLE, CONVERSATION, DOCUMENT, SKILL)
# The sources that are the person's work: with none of these there is nothing to read
# (an installed skill says what Auto can do, not what the person works on).
WORK_TYPES = (DELIVERABLE, CONVERSATION, DOCUMENT)

# At most this many lines, each at most this many characters (§6.11.5).
MAX_LINES = 3
MAX_LINE_CHARS = 160
# Exactly this many offers: from the installed skills, else the starters.
OFFERS = 3

SourceType = Literal["deliverable", "conversation", "document", "skill"]


class _Shape(BaseModel):
    """Read leniently, held still: unknown fields are ignored, an id given as a number is its string."""

    model_config = ConfigDict(extra="ignore", frozen=True, coerce_numbers_to_str=True)


class CalibrationSource(_Shape):
    """One thing a line was read from: its kind, its id, and its title as shown."""

    type: SourceType
    id: str
    label: str


class CalibrationLine(_Shape):
    """One thing Auto thinks it knows, and the sources it was read from (never empty)."""

    id: str
    text: str
    sources: List[CalibrationSource]


class Calibration(_Shape):
    """``Calibration@1``: one person's calibration in one workspace."""

    v: Literal[1]
    id: str
    lines: List[CalibrationLine]
    offers: List[str]
    empty: bool

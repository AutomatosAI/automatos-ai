"""PRD-268 §6.4, §6.11.2, K7, K11 — ``build_brief``: one person's ``Brief@1`` for one local date.

It assembles each part, every read scoped to the caller's workspace:

* ``prose``: Auto's Read, the text of ``services.digest_service.generate_digest`` unchanged
  (cached per workspace state). It is the only model call a brief makes, and none when its
  cache answers or a caller passes the Read it already made (``digest``: the scheduled digest
  makes it once per workspace). Every section is deterministic;
* the five sections, one reader each (``services.brief.needs``, ``suggested``, ``today``,
  ``since``, ``unchecked``), each folded at 5 and never at 6, with the counts exact;
* ``greeting`` for the person's local hour in ``zone``, by their first name when it is known;
  ``labels`` (the section labels, "Show N more" per folded section, the empty sentence when
  every section is empty); ``spoken``: the greeting and the numbered row titles, never a
  row's why, through ``speechify`` (the empty sentence when there are no rows);
* ``state_hash`` is Auto's Read's (the thumbs on the card answer it); ``generated_at`` now.

The language (K7) is ``language``'s table (``strings.brief_language``): its own, its
language's, else English. ``may_answer`` False (a viewer, PRD-266 D16) leaves questions and
approval grants out of "Needs you", as the badge does. It runs on a worker thread (a plain
``def`` route's threadpool, the digest job's ``to_thread``): ``generate_digest`` is a
coroutine, so it runs on a loop of its own there, as ``services.mobile.digest`` runs it.
It writes nothing.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone, tzinfo
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from core.models.core import User
from modules.voice.spoken_style import speechify
from services.brief.model import (
    COUNTED_SECTIONS,
    NEEDS_YOU,
    SECTIONS,
    SINCE,
    SUGGESTIONS,
    TODAY,
    UNCHECKED,
    VERSION,
    Brief,
    BriefCounts,
    BriefLabels,
    BriefRow,
    BriefSections,
)
from services.brief.needs import needs_rows
from services.brief.rows import cut, fold
from services.brief.since import since_rows
from services.brief.strings import brief_language, greeting, more, say
from services.brief.suggested import suggestion_rows
from services.brief.today import today_rows
from services.brief.unchecked import unchecked_rows
from services.digest_service import generate_digest

__all__ = ["build_brief", "read_digest", "spoken_words"]

NAME_CHARS = 40
Listed = Dict[str, Tuple[int, List[BriefRow]]]


def build_brief(
    db: Session, workspace_id: Any, user_id: int, local_date: date, language: Optional[str], *,
    may_answer: bool, zone: Optional[str] = None, digest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The person's ``Brief@1`` for ``local_date`` as JSON-ready values (optional fields left
    out when empty). ``zone`` is their IANA timezone (UTC when unknown); ``digest`` is
    ``generate_digest``'s answer when the caller made it already."""
    workspace, lang, tz = UUID(str(workspace_id)), brief_language(language), _zone(zone)
    now = datetime.now(timezone.utc)
    read = dict(digest) if digest is not None else read_digest(db, workspace)
    listed = _sections(db, workspace, int(user_id), local_date, lang, may_answer=may_answer, zone=tz, now=now)
    folded = {name: fold(rows, count) for name, (count, rows) in listed.items()}
    sections = BriefSections(**{name: rows for name, (rows, _) in folded.items()})
    empty = None if any(rows for rows, _ in folded.values()) else say(lang, "empty")
    greeted = greeting(lang, _first_name(db, int(user_id)), now.astimezone(tz).hour)
    brief = Brief(
        v=VERSION, date=local_date.isoformat(), greeting=greeted, prose=str(read.get("text") or "").strip(),
        spoken=spoken_words(greeted, sections, empty), sections=sections,
        counts=BriefCounts(**{name: listed[name][0] for name in COUNTED_SECTIONS}),
        labels=BriefLabels(sections={name: say(lang, f"section_{name}") for name in SECTIONS},
                           more={name: more(lang, hidden) for name, (_, hidden) in folded.items() if hidden},
                           empty=empty),
        state_hash=str(read.get("state_hash") or ""), generated_at=now.isoformat(),
    )
    return brief.model_dump(mode="json", exclude_none=True)


def read_digest(db: Session, workspace_id: UUID) -> Dict[str, Any]:
    """Auto's Read for the workspace (``generate_digest``), on a loop of this thread's own."""
    return dict(asyncio.run(generate_digest(db, workspace_id)) or {})


def _sections(
    db: Session, workspace_id: UUID, user_id: int, local_date: date, language: str, *,
    may_answer: bool, zone: tzinfo, now: datetime,
) -> Listed:
    """Each section's count and rows, in the sections' order."""
    unchecked = unchecked_rows(workspace_id, language)
    return {
        NEEDS_YOU: needs_rows(db, workspace_id, language, may_answer=may_answer),
        SUGGESTIONS: suggestion_rows(db, workspace_id, user_id, language, now),
        TODAY: today_rows(db, workspace_id, local_date, zone, language),
        SINCE: since_rows(db, workspace_id, user_id, local_date, language, now),
        UNCHECKED: (len(unchecked), unchecked),
    }


def spoken_words(greeted: str, sections: BriefSections, empty: Optional[str]) -> str:
    """What Listen and ``get_brief`` say: the greeting, then each shown row's title, numbered
    in the sections' order (so "start two" names a row), or the empty sentence. Never a why."""
    titles = [row.title for name in SECTIONS for row in getattr(sections, name)]
    said = [f"{number}. {title}." for number, title in enumerate(titles, 1)] if titles else [empty or ""]
    return speechify(" ".join(part for part in (greeted, *said) if part))


def _first_name(db: Session, user_id: int) -> Optional[str]:
    words = str(db.query(User.name).filter(User.id == user_id).scalar() or "").split()
    return cut(words[0], NAME_CHARS) if words else None


def _zone(name: Optional[str]) -> tzinfo:
    """The IANA zone ``name``, UTC when it is missing or not one."""
    try:
        return ZoneInfo(str(name)) if name else timezone.utc
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return timezone.utc

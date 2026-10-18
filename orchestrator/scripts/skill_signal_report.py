#!/usr/bin/env python3
"""PRD-263 US-062: the production skill signal, per skill body and per day.

Auto's always-on skill body (platform-management) changes when an edit is synced
into the platform seed. Each Auto turn records the body that served it, by the
sha256 of the body as rendered (for a seeded body, the row's ``content_hash``),
in ``messages.context_trace.skill_bodies`` (``modules/context/sections/skills.py``).
This report compares bodies in production. Across workspaces, per body hash and
per UTC day, it counts Auto's saved replies and:

  1. how many carry each line the platform writes in place of a claim no action
     backed: ``NOT_DONE``, ``NOTHING_DONE_SAID`` and ``NO_SUCH_ID``
     (``consumers/chatbot/claim_check.py``), with any words in their placeholders.
     ``NOTHING_DONE_SAID`` is said inside a ``NOT_DONE`` line, so a reply that
     carries it counts under both;
  2. the approval grants those turns raised, and how many the owner denied. A
     grant belongs to the first reply in its conversation saved at or after it.

It is read-only (one SELECT, in a READ ONLY transaction) and calls no model. It
prints counts and rates only: no message text and no workspace id. The templates
are matched inside the database, so no reply text leaves it.

Gerard runs it against prod (prod DB access is his)::

    python -m scripts.skill_signal_report                # today and the 13 days before
    python -m scripts.skill_signal_report --days 30
    python -m scripts.skill_signal_report --json         # machine-readable, full hashes
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy import text

from consumers.chatbot.claim_check import NO_SUCH_ID, NOT_DONE, NOTHING_DONE_SAID
from core.models.approval_grants import KIND_APPROVAL, GrantStatus

DEFAULT_DAYS = 14
# Characters of a body hash the text table shows; --json carries the whole hash.
HASH_SHOWN = 12

# The lines counted in Auto's replies, under the name each is reported by.
TEMPLATES = (
    ("not_done", NOT_DONE),
    ("nothing_done_said", NOTHING_DONE_SAID),
    ("no_such_id", NO_SUCH_ID),
)
COUNTS = ("turns", *(name for name, _ in TEMPLATES), "grants", "denied_grants")

_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
# Characters special to both Python ``re`` and Postgres regular expressions.
_REGEX_SPECIAL = frozenset("\\^$.|?*+()[]{}")
# A placeholder's words: anything on the template's own line.
_PLACEHOLDER_WORDS = "[^\n]+"

# Per (day, skill, body hash): Auto's replies, the templates in them and the grants
# their turns raised. Bound parameters only; the reply text never leaves the database.
SIGNAL_SQL = text(
    """
    WITH turns AS (
        SELECT m.id,
               CAST(m.created_at AS date) AS day,
               skill_body.entry ->> 'name' AS skill,
               skill_body.entry ->> 'content_hash' AS content_hash,
               COALESCE((
                   SELECT string_agg(reply_part.entry ->> 'text', chr(10))
                   FROM jsonb_array_elements(
                       CASE WHEN jsonb_typeof(m.parts) = 'array' THEN m.parts ELSE CAST('[]' AS jsonb) END
                   ) AS reply_part(entry)
                   WHERE reply_part.entry ->> 'type' = 'text'
               ), '') AS reply
        FROM messages m
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE WHEN jsonb_typeof(m.context_trace -> 'skill_bodies') = 'array'
                 THEN m.context_trace -> 'skill_bodies' ELSE CAST('[]' AS jsonb) END
        ) AS skill_body(entry)
        WHERE m.role = 'assistant' AND m.created_at >= :since
          AND skill_body.entry ->> 'name' IS NOT NULL
    ),
    replies AS (
        SELECT m.id,
               CAST(m.chat_id AS text) AS conversation_id,
               m.created_at,
               LAG(m.created_at) OVER (PARTITION BY m.chat_id ORDER BY m.created_at) AS previous_at
        FROM messages m
        WHERE m.role = 'assistant' AND m.created_at >= :since
    ),
    grants_per_reply AS (
        SELECT r.id AS reply_id,
               COUNT(*) AS raised,
               COUNT(*) FILTER (WHERE g.status = :denied) AS denied
        FROM approval_grants g
        JOIN replies r
          ON r.conversation_id = g.details ->> 'conversation_id'
         AND (g.requested_at AT TIME ZONE 'UTC') <= r.created_at
         AND (r.previous_at IS NULL OR (g.requested_at AT TIME ZONE 'UTC') > r.previous_at)
        WHERE g.kind = :approval AND (g.requested_at AT TIME ZONE 'UTC') >= :since
        GROUP BY r.id
    )
    SELECT t.day, t.skill, t.content_hash,
           COUNT(*) AS turns,
           COUNT(*) FILTER (WHERE t.reply ~ :not_done) AS not_done,
           COUNT(*) FILTER (WHERE t.reply ~ :nothing_done_said) AS nothing_done_said,
           COUNT(*) FILTER (WHERE t.reply ~ :no_such_id) AS no_such_id,
           COALESCE(SUM(g.raised), 0) AS grants,
           COALESCE(SUM(g.denied), 0) AS denied_grants
    FROM turns t
    LEFT JOIN grants_per_reply g ON g.reply_id = t.id
    GROUP BY t.day, t.skill, t.content_hash
    ORDER BY t.day, t.skill, t.content_hash
    """
)


def template_pattern(template: str) -> str:
    """A regular expression for ``template`` with any words in its placeholders.

    Each placeholder stands for words on the template's own line, so a reply that
    merely shares a template's opening ("Just to be clear: ...") does not match.
    The pattern means the same to Python ``re`` and to Postgres ``~``.
    """
    literals = _PLACEHOLDER.split(template)
    return _PLACEHOLDER_WORDS.join(_escape(literal) for literal in literals)


def _escape(literal: str) -> str:
    return "".join("\\" + char if char in _REGEX_SPECIAL else char for char in literal)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def signal_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """One (day, skill, body hash) row: its counts and rates, nothing else."""
    counts = {key: int(row[key] or 0) for key in COUNTS}
    day = row["day"]
    return {
        "day": day.isoformat() if isinstance(day, date) else str(day),
        "skill": row["skill"],
        "content_hash": row["content_hash"],
        **counts,
        **{f"{name}_rate": _rate(counts[name], counts["turns"]) for name, _ in TEMPLATES},
        "denied_grant_rate": _rate(counts["denied_grants"], counts["grants"]),
    }


def fetch_signal(db: Any, *, since: datetime) -> List[Dict[str, Any]]:
    """The signal for Auto's replies saved since ``since`` (naive UTC). Read-only."""
    params: Dict[str, Any] = {
        "since": since,
        "approval": KIND_APPROVAL,
        "denied": GrantStatus.DENIED.value,
        **{name: template_pattern(template) for name, template in TEMPLATES},
    }
    return [signal_row(row) for row in db.execute(SIGNAL_SQL, params).mappings()]


def _counted(row: Mapping[str, Any], name: str) -> str:
    return f"{row[name]} ({row[f'{name}_rate']:.1%})"


def render_text(rows: List[Dict[str, Any]], days: int) -> str:
    """The report as a table: one line per day and skill body, counts and rates only."""
    lines = [
        f"SKILL SIGNAL: Auto's saved replies per skill body, last {days} UTC day(s) "
        "(today so far and whole days before it), counts only",
        "=" * 100,
    ]
    if not rows:
        lines.append("no Auto turn recorded a skill body in this window")
        return "\n".join(lines)
    lines.append(
        f"{'day':<11} {'skill':<22} {'body hash':<{HASH_SHOWN}}  {'turns':>6}  {'not_done':<14} "
        f"{'nothing_done':<14} {'no_such_id':<14} denied grants"
    )
    for row in rows:
        shown = (row["content_hash"] or "-")[:HASH_SHOWN]
        lines.append(
            f"{row['day']:<11} {row['skill']:<22} {shown:<{HASH_SHOWN}}  {row['turns']:>6}  "
            f"{_counted(row, 'not_done'):<14} {_counted(row, 'nothing_done_said'):<14} "
            f"{_counted(row, 'no_such_id'):<14} "
            f"{row['denied_grants']}/{row['grants']} ({row['denied_grant_rate']:.1%})"
        )
    return "\n".join(lines)


def window_start(days: int, now: datetime) -> datetime:
    """The window's start: UTC midnight ``days - 1`` days before ``now``'s day.

    Every day bucket but today's is then a whole day. Naive UTC, the way
    ``messages.created_at`` is stored.
    """
    today = now.astimezone(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=days - 1)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"window in days (default {DEFAULT_DAYS})")
    parser.add_argument("--json", action="store_true", help="machine-readable output, with full hashes")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")

    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        db.execute(text("SET TRANSACTION READ ONLY"))
        rows = fetch_signal(db, since=window_start(args.days, datetime.now(timezone.utc)))
    finally:
        db.rollback()
        db.close()

    print(json.dumps({"days": args.days, "rows": rows}, indent=2) if args.json else render_text(rows, args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())

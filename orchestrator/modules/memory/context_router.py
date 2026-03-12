"""
Context Router
==============

Intelligent pre-LLM context assembly layer that analyses user queries and
decides which memory layers to fetch BEFORE the agent sees the prompt.

The router does NOT call the LLM — it uses fast regex-based signal detection
(<10 ms budget) to classify query intent and return a ContextSignals struct
that downstream code uses to assemble the context bundle.

Usage:
    from modules.memory.context_router import ContextRouter

    router = ContextRouter()
    signals = router.analyze_query("What did we discuss last week?")
    # signals.is_temporal == True
    # signals.temporal_window == (2026-03-05T00:00:00Z, 2026-03-12T00:00:00Z)
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContextSignals — output of query analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSignals:
    """
    Signals extracted from a user query that guide context assembly.

    Each flag indicates which memory layer(s) should be consulted.
    Multiple flags can be True simultaneously (e.g. a temporal + personal
    fact query like "What was my preference last week?").
    """

    is_temporal: bool = False
    is_personal_fact: bool = False
    is_session_continuation: bool = False
    is_knowledge_query: bool = False
    is_live_data: bool = False
    temporal_window: Optional[Tuple[datetime, datetime]] = None


# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at module load
# ---------------------------------------------------------------------------

# Temporal patterns: relative time references
_TEMPORAL_PATTERNS = re.compile(
    r"""(?xi)               # verbose + case-insensitive
    \b(?:
        last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)
      | yesterday
      | today
      | earlier\s+(?:today|this\s+(?:week|month))
      | this\s+(?:week|month|morning|afternoon|evening)
      | (?:a\s+)?(?:few|couple(?:\s+of)?)\s+(?:days?|weeks?|hours?|months?)\s+ago
      | (?:\d+)\s+(?:days?|weeks?|hours?|months?|minutes?)\s+ago
      | recently
      | the\s+other\s+day
      | last\s+time
      | (?:before|since|after)\s+(?:yesterday|last\s+(?:week|month))
      | previous(?:ly)?
      | in\s+the\s+past\s+(?:\d+\s+)?(?:days?|weeks?|months?)
      | back\s+(?:then|when)
    )\b
    """
)

# Personal fact patterns: references to user identity or preferences
_PERSONAL_FACT_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        my\s+(?:name|email|role|title|company|team|timezone|preference|style)
      | i\s+(?:prefer|like|want|need|use|hate|love|always|usually|tend\s+to)
      | remember\s+(?:when|that|what)\b
      | you\s+(?:told|said|mentioned|know)\s+(?:me|that|about)
      | (?:do\s+you|you)\s+remember
      | what\s+do\s+you\s+know\s+about\s+me
      | we\s+(?:agreed|decided|discussed|talked\s+about)
      | as\s+i\s+(?:said|mentioned)
    )\b
    """
)

# Session continuation patterns: references to current conversation
_SESSION_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        (?:as\s+)?(?:i|we)\s+(?:just|were)\s+(?:said|mentioned|discussed|talking\s+about)
      | going\s+back\s+to
      | (?:continuing|picking\s+up)\s+(?:where|from|on)
      | earlier\s+in\s+(?:this|our)\s+(?:conversation|chat|discussion)
      | what\s+(?:i|we)\s+(?:just|were)\s+(?:saying|discussing|talking)
      | like\s+i\s+(?:said|mentioned)\s+(?:earlier|before|above)
      | (?:that|the)\s+(?:thing|topic|point)\s+(?:we|i)\s+(?:mentioned|discussed)
      | you\s+just\s+(?:said|mentioned|told)
    )\b
    """
)

# Knowledge query patterns: document or policy lookups
_KNOWLEDGE_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        find\s+(?:the|a|that)?\s*(?:doc(?:ument)?|file|policy|guide|report|article)
      | search\s+(?:for|the|our)\b
      | look\s+(?:up|for|into)\b
      | what(?:'s|\s+is)\s+(?:our|the)\s+(?:policy|process|procedure|guideline|standard)
      | (?:is\s+there|do\s+we\s+have)\s+(?:a|any)\s+(?:doc(?:ument)?|guide|policy|wiki)
      | check\s+(?:the|our)\s+(?:docs?|documentation|wiki|knowledge\s+base|confluence)
      | according\s+to\s+(?:our|the)\s+(?:docs?|documentation|policy)
      | where\s+(?:can\s+i|do\s+i)\s+find
    )\b
    """
)

# Live data patterns: metrics, counts, real-time queries
_LIVE_DATA_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        (?:current|latest|live|real[\s-]?time|up[\s-]?to[\s-]?date)\s+\w+
      | how\s+many\s+(?:users?|customers?|orders?|transactions?|subscriptions?|signups?)
      | (?:total|average|count\s+of|number\s+of)\s+\w+
      | mrr|arr|revenue|churn|dau|mau|wau
      | (?:users?|customers?)\s+(?:signed?\s+up|registered|churned|converted)
      | what(?:'s|\s+is)\s+(?:our|the|my)\s+(?:mrr|arr|revenue|growth|churn)
      | (?:latest|recent|last)\s+(?:deploy|deployment|release|build|commit)
      | (?:show|give|get)\s+me\s+(?:the\s+)?(?:stats?|statistics|metrics|numbers|data)
      | (?:sales|conversion|retention|signup)\s+(?:rate|count|numbers?|stats?)
    )\b
    """
)


# ---------------------------------------------------------------------------
# Temporal window calculation
# ---------------------------------------------------------------------------

def _compute_temporal_window(
    query: str,
    now: datetime,
) -> Optional[Tuple[datetime, datetime]]:
    """
    Convert a relative time reference in *query* into an absolute
    (start, end) datetime window.

    Returns None if no recognisable temporal reference is found.
    """
    q = query.lower()

    # "yesterday"
    if re.search(r"\byesterday\b", q):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return (start, end)

    # "today" / "earlier today" / "this morning/afternoon/evening"
    if re.search(r"\b(?:today|this\s+(?:morning|afternoon|evening))\b", q):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    # "last week"
    if re.search(r"\blast\s+week\b", q):
        start = now - timedelta(days=7)
        return (start, now)

    # "last month"
    if re.search(r"\blast\s+month\b", q):
        start = now - timedelta(days=30)
        return (start, now)

    # "last year"
    if re.search(r"\blast\s+year\b", q):
        start = now - timedelta(days=365)
        return (start, now)

    # "this week"
    if re.search(r"\bthis\s+week\b", q):
        # Monday of current week
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (start, now)

    # "this month"
    if re.search(r"\bthis\s+month\b", q):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    # "N days/weeks/hours/months ago" or "a few days ago"
    m = re.search(r"\b(\d+)\s+(days?|weeks?|hours?|months?|minutes?)\s+ago\b", q)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        delta_map = {
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "hour": timedelta(hours=amount),
            "month": timedelta(days=amount * 30),
            "minute": timedelta(minutes=amount),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "a few days/weeks ago", "couple of days ago"
    m = re.search(r"\b(?:a\s+)?(?:few|couple(?:\s+of)?)\s+(days?|weeks?|hours?|months?)\s+ago\b", q)
    if m:
        unit = m.group(1).rstrip("s")
        # "a few" ≈ 3
        delta_map = {
            "day": timedelta(days=3),
            "week": timedelta(weeks=3),
            "hour": timedelta(hours=3),
            "month": timedelta(days=90),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "in the past N days/weeks"
    m = re.search(r"\bin\s+the\s+past\s+(\d+)\s+(days?|weeks?|months?)\b", q)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        delta_map = {
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=amount * 30),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "recently" / "the other day" / "last time" — fuzzy 7-day window
    if re.search(r"\b(?:recently|the\s+other\s+day|last\s+time)\b", q):
        return (now - timedelta(days=7), now)

    # Named day of week: "last monday", "last tuesday", etc.
    m = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q
    )
    if m:
        target_day = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }[m.group(1)]
        days_back = (now.weekday() - target_day) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" when today is Monday → 7 days ago
        target = (now - timedelta(days=days_back)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (target, target + timedelta(days=1) - timedelta(microseconds=1))

    return None


# ---------------------------------------------------------------------------
# ContextRouter
# ---------------------------------------------------------------------------

class ContextRouter:
    """
    Pre-LLM context assembly layer.

    Analyses user queries with fast regex patterns (<10 ms) and returns
    ContextSignals that downstream code uses to decide which memory layers
    to fetch and inject into the system prompt.
    """

    def analyze_query(self, query: str) -> ContextSignals:
        """
        Classify a user query into context signals.

        All detection is regex-based — no LLM calls, no I/O.
        Target latency: <10 ms.
        """
        if not query or not query.strip():
            return ContextSignals()

        now = datetime.now(timezone.utc)

        is_temporal = bool(_TEMPORAL_PATTERNS.search(query))
        is_personal_fact = bool(_PERSONAL_FACT_PATTERNS.search(query))
        is_session = bool(_SESSION_PATTERNS.search(query))
        is_knowledge = bool(_KNOWLEDGE_PATTERNS.search(query))
        is_live_data = bool(_LIVE_DATA_PATTERNS.search(query))

        temporal_window = _compute_temporal_window(query, now) if is_temporal else None

        return ContextSignals(
            is_temporal=is_temporal,
            is_personal_fact=is_personal_fact,
            is_session_continuation=is_session,
            is_knowledge_query=is_knowledge,
            is_live_data=is_live_data,
            temporal_window=temporal_window,
        )

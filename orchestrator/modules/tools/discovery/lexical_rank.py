"""PRD-238 S11 — lexical fallback for tool narrowing.

The semantic index (PRD-232) narrows the ~190 platform actions to the handful
a turn needs by embedding the user's message. When that embedding times out
the router used to give up narrowing and send the FULL enum — a slow upstream
cost seconds *and* tokens. This module ranks actions by plain token overlap
so a timeout degrades to "a reasonable shortlist", never to "everything".

Pure functions, stdlib only, no I/O — importable from anywhere and cheap to
test. Scores are deterministic (ties break on the action name).
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words that carry no routing signal on their own.
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from has have how i in is it its "
    "me my of on or our please that the their them there these this to us was we what "
    "when where which who will with would you your".split()
)

#: A query token that is a substring of the action name earns this on top of overlap.
NAME_HIT_BONUS = 0.5
MIN_TOKEN_LEN = 3


def tokens(text: Any) -> set[str]:
    """Lower-cased alphanumeric tokens, stopwords and very short words dropped."""
    if not text:
        return set()
    return {
        t for t in _TOKEN_RE.findall(str(text).lower())
        if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS
    }


def action_text(action: Any) -> str:
    """The searchable text of an action: name, description, tags, examples, category."""
    parts = [
        str(getattr(action, "name", "") or "").replace("_", " "),
        str(getattr(action, "description", "") or ""),
        " ".join(getattr(action, "tags", None) or []),
        " ".join(getattr(action, "examples", None) or []),
        str(getattr(action, "category", "") or ""),
    ]
    return " ".join(p for p in parts if p)


def score_action(query_tokens: set[str], action: Any) -> float:
    """Overlap between the query and the action's text, plus a bonus for name hits."""
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & tokens(action_text(action)))
    name = str(getattr(action, "name", "") or "").lower()
    name_hits = sum(1 for t in query_tokens if t in name)
    return float(overlap) + NAME_HIT_BONUS * name_hits


def lexical_rank(query: str, actions: Iterable[Any], *, top_k: int = 15) -> List[Tuple[str, float]]:
    """Rank ``actions`` for ``query`` by lexical overlap; only scored matches, best first."""
    query_tokens = tokens(query)
    if not query_tokens or top_k <= 0:
        return []
    scored = [
        (str(getattr(a, "name", "")), score_action(query_tokens, a))
        for a in actions
    ]
    kept: Sequence[Tuple[str, float]] = [(n, s) for n, s in scored if s > 0 and n]
    return sorted(kept, key=lambda pair: (-pair[1], pair[0]))[:top_k]

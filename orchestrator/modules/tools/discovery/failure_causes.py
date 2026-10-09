"""What went wrong in a workspace, grouped by cause (Auto's wishlist, 9 Oct 2026).

Auto's operating manual promised a "workspace errors" tool and none existed, so a
report of "79 failed" could not say why in one call. Failures live in three places:
failed board cards (``board_tasks.error_message``), failed LLM calls
(``llm_usage.status`` / ``error_message``) and failed tool runs
(``tool_execution_logs.status`` / ``error_message``). Each failure is put under one
cause by the first rule its text matches, and the causes are ranked by how often
they happened, with a few examples to open.

Pure functions: the handler (``handlers_diagnostics``) reads the rows and hands them here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from core.llm.credit import is_out_of_credit

CAUSE_OUT_OF_CREDIT = "out_of_credit"
CAUSE_TIMEOUT = "timeout"
CAUSE_RATE_LIMITED = "rate_limited"
CAUSE_AUTH = "auth"
CAUSE_TOOL_MISSING = "tool_missing"
CAUSE_REFUSED = "refused"
CAUSE_NOT_FOUND = "not_found"
CAUSE_OTHER = "other"

# First match wins, so the most specific causes come first.
_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (CAUSE_TIMEOUT, re.compile(r"time(d)?[ -]?out|deadline exceeded|took longer than", re.I)),
    (CAUSE_RATE_LIMITED, re.compile(r"\b429\b|rate[ _-]?limit|too many requests|overloaded|at capacity", re.I)),
    (CAUSE_AUTH, re.compile(r"\b401\b|\b403\b|unauthori[sz]ed|forbidden|invalid (api )?key|"
                            r"not scoped to a workspace|authentication", re.I)),
    (CAUSE_TOOL_MISSING, re.compile(r"unknown tool|no tool (by|named|called)|tool not found|"
                                    r"tool\b.{0,60}\bnot available|not connected", re.I)),
    (CAUSE_REFUSED, re.compile(r"\brefus|\bdeclin|stop_reason.{0,4}refusal|can(no|')t (help|assist) with", re.I)),
    (CAUSE_NOT_FOUND, re.compile(r"\b404\b|not found|does not exist|no such", re.I)),
)

CAUSE_LABELS: Dict[str, str] = {
    CAUSE_OUT_OF_CREDIT: "The provider is out of credit",
    CAUSE_TIMEOUT: "Timed out",
    CAUSE_RATE_LIMITED: "Rate-limited or the model was at capacity",
    CAUSE_AUTH: "A key or permission was refused",
    CAUSE_TOOL_MISSING: "A tool or connection was missing",
    CAUSE_REFUSED: "The agent or model refused",
    CAUSE_NOT_FOUND: "Something it needed was not found",
    CAUSE_OTHER: "Other",
}

EXAMPLES_PER_CAUSE = 3
MESSAGE_CHARS = 200


@dataclass(frozen=True)
class Failure:
    """One failed card, LLM call or tool run."""

    source: str            # "card" | "llm_call" | "tool_run"
    ref: str               # card number / usage id / log id, for opening it
    message: str
    at: Optional[datetime] = None  # naive UTC (the handler normalises)
    agent_id: Optional[int] = None


@dataclass
class CauseGroup:
    cause: str
    count: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    agent_ids: List[int] = field(default_factory=list)
    last_seen: Optional[datetime] = None
    examples: List[Dict[str, str]] = field(default_factory=list)


def classify(message: Optional[str]) -> str:
    """The cause a failure's text points to (``other`` when none does)."""
    text = (message or "").strip()
    if not text:
        return CAUSE_OTHER
    if is_out_of_credit(text):
        return CAUSE_OUT_OF_CREDIT
    return next((cause for cause, rule in _RULES if rule.search(text)), CAUSE_OTHER)


def _add(group: CauseGroup, failure: Failure) -> None:
    group.count += 1
    group.by_source[failure.source] = group.by_source.get(failure.source, 0) + 1
    if failure.agent_id is not None and failure.agent_id not in group.agent_ids:
        group.agent_ids.append(failure.agent_id)
    if failure.at is not None and (group.last_seen is None or failure.at > group.last_seen):
        group.last_seen = failure.at
    if len(group.examples) < EXAMPLES_PER_CAUSE:
        group.examples.append({"source": failure.source, "ref": failure.ref,
                               "message": (failure.message or "")[:MESSAGE_CHARS]})


def group_by_cause(failures: Iterable[Failure]) -> List[CauseGroup]:
    """Failures under their causes, the most frequent first (ties: the most recent first)."""
    groups: Dict[str, CauseGroup] = {}
    for failure in sorted(failures, key=lambda f: f.at or datetime.min, reverse=True):
        cause = classify(failure.message)
        _add(groups.setdefault(cause, CauseGroup(cause=cause)), failure)
    recent_first = sorted(groups.values(), key=lambda g: g.last_seen or datetime.min, reverse=True)
    return sorted(recent_first, key=lambda g: g.count, reverse=True)


def as_payload(groups: List[CauseGroup]) -> List[Dict[str, object]]:
    """The groups as the tool returns them."""
    return [{
        "cause": g.cause,
        "label": CAUSE_LABELS[g.cause],
        "count": g.count,
        "by_source": g.by_source,
        "agent_ids": g.agent_ids,
        "last_seen": g.last_seen.isoformat() if g.last_seen else None,
        "examples": g.examples,
    } for g in groups]


__all__ = ["CAUSE_LABELS", "Failure", "as_payload", "classify", "group_by_cause"]

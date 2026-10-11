"""
PRD-258 US-004: the voice-latency report
=========================================

Pure (no I/O). Nearest-rank p50 and p90 per measure, per ask kind and
overall, and the ONE results line a run prints for the owner to append to
their local ``results.md``::

    <ISO time> · <build SHA> · overall: first_frame p50/p90 · first_auto_frame p50/p90 · full p50/p90 · prep p50/p90 · silent N | small_talk: … | knowledge: …

A real call (``--call``) joins its ``voice_turns`` rows with the call's
``voice_calls.vendor_latency`` into the same line shape (one ``call`` group),
followed by Retell's own e2e / asr / llm / tts p50/p90.

Every figure is whole milliseconds; a measure with no value is ``-``, never
an invented number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.eval.voice_latency.protocol import ASK_KINDS

P50 = 50
P90 = 90
PERCENT = 100

# measure name → the TurnResult field it reads, in the line's order.
MEASURES = (
    ("first_frame", "first_frame_ms"),
    ("first_auto_frame", "first_auto_frame_ms"),
    ("full", "full_ms"),
    ("prep", "prep_ms"),
)
OVERALL = "overall"
CALL_KIND = "call"
# Retell's stages the call line reports (voice_calls.vendor_latency keeps more).
REPORTED_VENDOR_STAGES = ("e2e", "asr", "llm", "tts")
MISSING = "-"
FIELD_SEP = " · "
GROUP_SEP = " | "


@dataclass(frozen=True)
class TurnResult:
    """One measured turn. ``silent``: no word of Auto's reached the caller."""

    kind: str
    first_frame_ms: Optional[int]
    first_auto_frame_ms: Optional[int]
    full_ms: Optional[int]
    prep_ms: Optional[int]
    tools_before_speech: Optional[int]
    silent: bool


def nearest_rank(values: Iterable[Optional[float]], percentile: int) -> Optional[int]:
    """The nearest-rank percentile of the present values; None when there are none."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    rank = max(1, math.ceil(percentile / PERCENT * len(present)))
    return int(round(present[rank - 1]))


def group_stats(results: Sequence[TurnResult]) -> Dict[str, Any]:
    """p50/p90 of every measure over these turns, the silent count and the turn count."""
    stats: Dict[str, Any] = {}
    for name, attr in MEASURES:
        values = [getattr(r, attr) for r in results]
        stats[name] = {"p50": nearest_rank(values, P50), "p90": nearest_rank(values, P90)}
    stats["silent"] = sum(1 for r in results if r.silent)
    stats["turns"] = len(results)
    return stats


def _kinds_in_order(results: Sequence[TurnResult]) -> List[str]:
    seen = {r.kind for r in results}
    known = [k for k in ASK_KINDS if k in seen]
    return known + sorted(seen - set(known))


def summarize(results: Sequence[TurnResult]) -> Dict[str, Dict[str, Any]]:
    """The stats overall first, then per kind (the ask set's kinds in their order)."""
    summary = {OVERALL: group_stats(results)}
    for kind in _kinds_in_order(results):
        summary[kind] = group_stats([r for r in results if r.kind == kind])
    return summary


def _fmt(value: Optional[float]) -> str:
    return MISSING if value is None else str(int(round(value)))


def _pair(stats: Mapping[str, Any]) -> str:
    return f"{_fmt(stats.get('p50'))}/{_fmt(stats.get('p90'))}"


def group_segment(label: str, stats: Mapping[str, Any]) -> str:
    """``<label>: first_frame p50/p90 · first_auto_frame p50/p90 · full p50/p90 · prep p50/p90 · silent N``."""
    parts = [f"{name} {_pair(stats[name])}" for name, _ in MEASURES]
    parts.append(f"silent {stats['silent']}")
    return f"{label}: " + FIELD_SEP.join(parts)


def results_line(when_iso: str, build_sha: str, summary: Mapping[str, Mapping[str, Any]]) -> str:
    """The one line a run prints: every group of ``summarize`` in order."""
    groups = GROUP_SEP.join(group_segment(label, stats) for label, stats in summary.items())
    return FIELD_SEP.join((when_iso, build_sha, groups))


def _int_or_none(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def call_results(rows: Sequence[Mapping[str, Any]]) -> Tuple[TurnResult, ...]:
    """A real call's voice_turns rows as turns: full = ``total_ms``, silent = no audio delivered."""
    return tuple(
        TurnResult(
            kind=CALL_KIND,
            first_frame_ms=_int_or_none(row.get("first_frame_ms")),
            first_auto_frame_ms=_int_or_none(row.get("first_auto_frame_ms")),
            full_ms=_int_or_none(row.get("total_ms")),
            prep_ms=_int_or_none(row.get("prep_ms")),
            tools_before_speech=_int_or_none(row.get("tools_before_speech")),
            silent=not bool(row.get("audio_delivered")),
        )
        for row in rows
    )


def vendor_segment(vendor_latency: Optional[Mapping[str, Any]]) -> str:
    """``vendor e2e p50/p90 · asr p50/p90 · llm p50/p90 · tts p50/p90`` (``-`` where Retell gave none)."""
    latency = vendor_latency or {}
    parts = [f"{stage} {_pair(latency.get(stage) or {})}" for stage in REPORTED_VENDOR_STAGES]
    return "vendor " + FIELD_SEP.join(parts)


def call_line(
    when_iso: str,
    build_sha: str,
    rows: Sequence[Mapping[str, Any]],
    vendor_latency: Optional[Mapping[str, Any]],
) -> str:
    """A real call's line: our timings over its turns, then Retell's own figures."""
    stats = group_stats(call_results(rows))
    return FIELD_SEP.join((when_iso, build_sha, group_segment(CALL_KIND, stats), vendor_segment(vendor_latency)))

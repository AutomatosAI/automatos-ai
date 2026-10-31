"""PRD-268 §6.7 — the pass bar each proactive eval must clear before its feature goes live.

The numbers are the PRD's and are never tuned to a run: a run under a bar is a miss, reported as
it is (the memory_recall and retrieval_recall rule). ``BAR_TEXT`` says each bar in words, built
from the constants so the printed bar and the scored one cannot drift apart.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

# Triage: "act today" against "can wait", each verdict with a one-line reason.
TRIAGE_MIN_RECALL = 0.95
TRIAGE_MIN_PRECISION = 0.90
TRIAGE_REASON_MAX_CHARS = 120

# Check-in silence: quiet snapshots end silent, a planted need is named, and no suggestion repeats
# an open or hidden title.
SILENCE_MIN_QUIET_SILENT = 0.95
SILENCE_MIN_PLANTED_NAMED = 0.85
SILENCE_MAX_DUPLICATES = 0

# Reply chips: on 16 of 20 choice turns, every chip within the tool's limits (§6.7: "all within
# the limits"), never in place of the answer, and on at most 1 of 10 open questions.
CHIPS_MIN_ON_CHOICE = 16 / 20
CHIPS_MAX_OUT_OF_LIMITS = 0
CHIPS_MAX_REPLACED = 0
CHIPS_MAX_ON_OPEN = 1 / 10

# Spoken brevity, after speechify: words, an answer in every one, and nothing a listener should never hear.
SPOKEN_MAX_MEDIAN_WORDS = 40
SPOKEN_MAX_WORDS = 80
SPOKEN_MAX_SILENT = 0

# One brief: exactly this many per person per local date.
BRIEFS_PER_PERSON_PER_DATE = 1

# Suggestion precision (live): of the decided suggestions (started or hidden), the share started,
# and the share started that passed review. A source agent under the slow share after at least
# the minimum decisions is listed (§6.8: that agent is slowed).
PRECISION_MIN_STARTED = 0.40
PRECISION_MIN_PASSED_REVIEW = 0.30
PRECISION_SLOW_AGENT_SHARE = 0.20
PRECISION_SLOW_AGENT_DECISIONS = 20

BAR_TEXT: Mapping[str, str] = MappingProxyType({
    "triage": (
        f"recall of 'act today' >= {TRIAGE_MIN_RECALL:.2f}, precision >= {TRIAGE_MIN_PRECISION:.2f}, "
        f"every verdict with a reason of at most {TRIAGE_REASON_MAX_CHARS} characters"),
    "silence": (
        f">= {SILENCE_MIN_QUIET_SILENT:.2f} of quiet snapshots silent, >= {SILENCE_MIN_PLANTED_NAMED:.2f} "
        f"of planted ones naming the need, {SILENCE_MAX_DUPLICATES} duplicates of an open or hidden title"),
    "reply_chips": (
        f"chips on >= {CHIPS_MIN_ON_CHOICE:.2f} of choice turns, {CHIPS_MAX_OUT_OF_LIMITS} outside the "
        f"limits, {CHIPS_MAX_REPLACED} turns where the chips replace the answer, chips on <= "
        f"{CHIPS_MAX_ON_OPEN:.2f} of open questions"),
    "spoken": (
        f"median <= {SPOKEN_MAX_MEDIAN_WORDS} words, none over {SPOKEN_MAX_WORDS}, {SPOKEN_MAX_SILENT} heard as "
        "nothing, no markdown, URL, path or id"),
    "one_brief": f"exactly {BRIEFS_PER_PERSON_PER_DATE} brief per person per local date",
    "precision": (
        f"started / decided >= {PRECISION_MIN_STARTED:.2f}, started and passed review >= "
        f"{PRECISION_MIN_PASSED_REVIEW:.2f} of decided; source agents under {PRECISION_SLOW_AGENT_SHARE:.2f} "
        f"after {PRECISION_SLOW_AGENT_DECISIONS} decisions listed"),
})

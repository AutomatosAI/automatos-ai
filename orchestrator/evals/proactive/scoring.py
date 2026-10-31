"""PRD-268 §6.7 — one pure scorer per proactive eval: its numbers, and PASS or MISS against its bar.

Each scorer takes what a runner observed (a triage verdict, the suggestions a check-in proposed,
the chips a turn offered, a spoken answer after speechify, the briefs posted, the suggestions
decided) and returns a ``Score``. No scorer reads a file, a database or a model. A share with
nothing to divide is None and never passes: no evidence is not a pass.

The checks reuse the platform's own rules rather than restating them: a reply chip is within
the limits when ``platform_offer_replies`` would keep it (``replies_refusal``), and a suggestion
repeats an open or hidden one when their titles match as dedupe compares them
(``normalise_title``).
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

from evals.proactive import bars
from modules.proactive.dedupe import normalise_title
from modules.tools.discovery.handlers_replies import replies_refusal

__all__ = [
    "ACT_TODAY", "CAN_WAIT", "MISS", "PASS", "VERDICTS",
    "ChipsTurn", "Decision", "Score", "SilenceRun", "Suggested", "TriageVerdict",
    "score_one_brief", "score_precision", "score_reply_chips", "score_silence", "score_spoken",
    "score_triage", "share",
]

PASS, MISS = "PASS", "MISS"
# The triage labels (§6.7: "act today" vs "can wait").
ACT_TODAY, CAN_WAIT = "act_today", "can_wait"
VERDICTS = (ACT_TODAY, CAN_WAIT)

# What a listener must never hear, looked for in the answer after speechify.
_MARKDOWN = re.compile(
    r"\*\*|__|`|\]\(|^[ \t]{0,3}#{1,6}[ \t]|^[ \t]*(?:[-*+•]|\d{1,2}[.)])[ \t]|^[ \t]*\||^[ \t]{0,3}>", re.M)
_URL = re.compile(r"\bhttps?://|\bwww\.|\b[\w-]+\.(?:com|org|net|io|app|ai|dev|co|uk)\b", re.I)
_PATH = re.compile(
    r"(?:^|[\s(\"'])(?:~|\.{1,2})?/[\w.@-]+(?:/[\w.@-]*)*"  # /var/data, ~/notes, ./build
    r"|\b[A-Za-z]:\\"  # C:\
    r"|\b[\w-]+(?:/[\w.-]+)+\.[A-Za-z]\w{0,4}\b")  # docs/prd/notes.md
_ID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # a UUID
    r"|\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b"  # a hash: hex with a digit and a letter
    r"|#\d+",  # a ticket's board number
    re.I)


@dataclass(frozen=True)
class Score:
    """One eval's numbers and whether they clear its bar; ``audit`` holds the lines a person reads
    beside them (the spoken answers for §6.7's hand audit, the precision of each workspace)."""

    name: str
    numbers: Mapping[str, Any]
    passed: bool
    audit: Tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        """PASS or MISS."""
        return PASS if self.passed else MISS

    @property
    def bar(self) -> str:
        """The eval's bar in words (``bars.BAR_TEXT``)."""
        return bars.BAR_TEXT[self.name]

    def as_dict(self) -> Dict[str, Any]:
        """The score as plain data, for ``--json``; ``audit`` only when there is one."""
        data = {"eval": self.name, "bar": self.bar, "numbers": dict(self.numbers), "verdict": self.verdict}
        return {**data, "audit": list(self.audit)} if self.audit else data


def _score(name: str, numbers: Mapping[str, Any], passed: bool) -> Score:
    return Score(name=name, numbers=MappingProxyType(dict(numbers)), passed=bool(passed))


def share(part: int, whole: int) -> Optional[float]:
    """``part / whole``, or None when there is nothing to divide."""
    return part / whole if whole else None


def _at_least(value: Optional[float], bar: float) -> bool:
    return value is not None and value >= bar


def _at_most(value: Optional[float], bar: float) -> bool:
    return value is not None and value <= bar


# ── triage ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TriageVerdict:
    """One gold case judged: its label, the verdict given (None when none was) and the reason."""

    expected: str
    verdict: Optional[str]
    reason: str = ""

    def __post_init__(self) -> None:
        if self.expected not in VERDICTS:
            raise ValueError(f"a triage case is labelled {self.expected!r}, not one of {VERDICTS}")


def _reason_fits(reason: Any) -> bool:
    text = str(reason or "").strip()
    return bool(text) and len(text) <= bars.TRIAGE_REASON_MAX_CHARS


def score_triage(verdicts: Sequence[TriageVerdict]) -> Score:
    """Recall and precision of 'act today', and the cases without a verdict or a fitting reason."""
    act_today = [v for v in verdicts if v.expected == ACT_TODAY]
    said_act_today = [v for v in verdicts if v.verdict == ACT_TODAY]
    hits = sum(1 for v in act_today if v.verdict == ACT_TODAY)
    recall, precision = share(hits, len(act_today)), share(hits, len(said_act_today))
    no_verdict = sum(1 for v in verdicts if v.verdict not in VERDICTS)
    bad_reasons = sum(1 for v in verdicts if not _reason_fits(v.reason))
    passed = (_at_least(recall, bars.TRIAGE_MIN_RECALL) and _at_least(precision, bars.TRIAGE_MIN_PRECISION)
              and no_verdict == 0 and bad_reasons == 0)
    return _score("triage", {
        "cases": len(verdicts), "act_today": len(act_today), "said_act_today": len(said_act_today),
        "recall": recall, "precision": precision, "no_verdict": no_verdict, "bad_reasons": bad_reasons,
    }, passed)


# ── check-in silence ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Suggested:
    """One suggestion a check-in proposed: its title and its detail."""

    title: str
    detail: str = ""


@dataclass(frozen=True)
class SilenceRun:
    """One gold snapshot's check-in: the planted need's words (none when the snapshot is quiet),
    the suggestions it proposed, and the snapshot's open and hidden titles."""

    need_terms: Tuple[str, ...]
    suggested: Tuple[Suggested, ...]
    taken_titles: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not normalise_title(term) for term in self.need_terms):
            raise ValueError("a planted need's words say nothing once normalised")


def _names_the_need(run: SilenceRun) -> bool:
    """A suggestion names the planted need when its title and detail carry every one of its words
    where a word starts ("401" is in "401s", never in "14012"), compared as dedupe compares titles."""
    terms = [re.compile(rf"(?<!\w){re.escape(normalise_title(term))}") for term in run.need_terms]
    texts = [normalise_title(f"{s.title} {s.detail}") for s in run.suggested]
    return any(all(term.search(text) for term in terms) for text in texts)


def _duplicates(run: SilenceRun) -> int:
    taken = {normalise_title(title) for title in run.taken_titles} - {""}
    return sum(1 for s in run.suggested if normalise_title(s.title) in taken)


def score_silence(runs: Sequence[SilenceRun]) -> Score:
    """The quiet snapshots that ended silent, the planted ones that named their need, and the
    suggestions that repeat an open or hidden title."""
    quiet = [run for run in runs if not run.need_terms]
    planted = [run for run in runs if run.need_terms]
    quiet_silent = share(sum(1 for run in quiet if not run.suggested), len(quiet))
    planted_named = share(sum(1 for run in planted if _names_the_need(run)), len(planted))
    duplicates = sum(_duplicates(run) for run in runs)
    passed = (_at_least(quiet_silent, bars.SILENCE_MIN_QUIET_SILENT)
              and _at_least(planted_named, bars.SILENCE_MIN_PLANTED_NAMED)
              and duplicates <= bars.SILENCE_MAX_DUPLICATES)
    return _score("silence", {
        "quiet": len(quiet), "quiet_silent": quiet_silent, "planted": len(planted),
        "planted_named": planted_named, "duplicates": duplicates,
    }, passed)


# ── reply chips ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChipsTurn:
    """One gold question's turn: a choice question or an open one, the turn's text, and the
    replies it offered (None when it offered none)."""

    choice: bool
    answer: str
    replies: Optional[Tuple[str, ...]] = None


def _within_limits(turn: ChipsTurn) -> bool:
    return turn.replies is not None and replies_refusal(list(turn.replies))[1] is None


def score_reply_chips(turns: Sequence[ChipsTurn]) -> Score:
    """Chips on the choice turns within the limits, chips outside them, chips in place of an
    answer, and chips on the open questions."""
    choice = [turn for turn in turns if turn.choice]
    open_ = [turn for turn in turns if not turn.choice]
    offered = [turn for turn in turns if turn.replies is not None]
    on_choice = share(sum(1 for turn in choice if _within_limits(turn)), len(choice))
    on_open = share(sum(1 for turn in open_ if turn.replies is not None), len(open_))
    out_of_limits = sum(1 for turn in offered if not _within_limits(turn))
    replaced = sum(1 for turn in offered if not str(turn.answer or "").strip())
    passed = (_at_least(on_choice, bars.CHIPS_MIN_ON_CHOICE) and _at_most(on_open, bars.CHIPS_MAX_ON_OPEN)
              and out_of_limits <= bars.CHIPS_MAX_OUT_OF_LIMITS and replaced <= bars.CHIPS_MAX_REPLACED)
    return _score("reply_chips", {
        "choice_turns": len(choice), "on_choice": on_choice, "open_turns": len(open_), "on_open": on_open,
        "out_of_limits": out_of_limits, "replaced": replaced,
    }, passed)


# ── spoken brevity ───────────────────────────────────────────────────────────


def _words(text: str) -> int:
    return len(str(text or "").split())


def score_spoken(answers: Sequence[str]) -> Score:
    """Words per answer (after speechify), the answers heard as nothing (no words: they would flatter
    the median), and the answers that still carry markdown, a URL, a path or an id."""
    counts = [_words(answer) for answer in answers]
    median = statistics.median(counts) if counts else None
    longest = max(counts) if counts else None
    silent = sum(1 for count in counts if count == 0)
    found = {name: sum(1 for answer in answers if pattern.search(str(answer or "")))
             for name, pattern in (("markdown", _MARKDOWN), ("urls", _URL), ("paths", _PATH), ("ids", _ID))}
    passed = (_at_most(median, bars.SPOKEN_MAX_MEDIAN_WORDS) and _at_most(longest, bars.SPOKEN_MAX_WORDS)
              and silent <= bars.SPOKEN_MAX_SILENT and not any(found.values()))
    return _score("spoken", {
        "answers": len(counts), "median_words": median, "max_words": longest,
        "over_limit": sum(1 for count in counts if count > bars.SPOKEN_MAX_WORDS), "silent": silent, **found,
    }, passed)


# ── one brief ────────────────────────────────────────────────────────────────


def score_one_brief(briefs: Iterable[Tuple[Hashable, str]],
                    expected: Optional[Iterable[Tuple[Hashable, str]]] = None) -> Score:
    """Briefs posted as (person, local date): the copies past one per person and date, and, when
    the run says which it expects, the expected ones missing and the ones it did not expect."""
    counts = Counter(briefs)
    duplicates = sum(n - bars.BRIEFS_PER_PERSON_PER_DATE for n in counts.values()
                     if n > bars.BRIEFS_PER_PERSON_PER_DATE)
    wanted = None if expected is None else set(expected)
    missing = None if wanted is None else len(wanted - set(counts))
    unexpected = None if wanted is None else len(set(counts) - wanted)
    passed = bool(counts) and duplicates == 0 and not missing and not unexpected
    return _score("one_brief", {
        "briefs": sum(counts.values()), "duplicates": duplicates, "missing": missing, "unexpected": unexpected,
    }, passed)


# ── suggestion precision (live) ──────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    """One decided suggestion: its source agent, whether it was started (else hidden), and whether
    its started ticket passed review."""

    source_agent: Optional[Hashable]
    started: bool
    passed_review: bool = False


def _slow_agents(decisions: Sequence[Decision]) -> Tuple[Tuple[Hashable, int, float], ...]:
    """(agent, decisions, started share) for each source agent under the slow share after at
    least the minimum decisions, by agent."""
    decided = Counter(d.source_agent for d in decisions if d.source_agent is not None)
    started = Counter(d.source_agent for d in decisions if d.source_agent is not None and d.started)
    shares = [(agent, n, started[agent] / n) for agent, n in decided.items()
              if n >= bars.PRECISION_SLOW_AGENT_DECISIONS]
    slow = [row for row in shares if row[2] < bars.PRECISION_SLOW_AGENT_SHARE]
    return tuple(sorted(slow, key=lambda row: str(row[0])))


def score_precision(decisions: Sequence[Decision]) -> Score:
    """Of the decided suggestions, the share started and the share started that passed review;
    the slow source agents are listed beside them."""
    started = sum(1 for d in decisions if d.started)
    reviewed = sum(1 for d in decisions if d.started and d.passed_review)
    started_share, reviewed_share = share(started, len(decisions)), share(reviewed, len(decisions))
    passed = (_at_least(started_share, bars.PRECISION_MIN_STARTED)
              and _at_least(reviewed_share, bars.PRECISION_MIN_PASSED_REVIEW))
    return _score("precision", {
        "decided": len(decisions), "started": started, "started_share": started_share,
        "passed_review": reviewed, "passed_review_share": reviewed_share,
        "slow_agents": _slow_agents(decisions),
    }, passed)

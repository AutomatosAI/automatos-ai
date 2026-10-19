"""PRD-263 US-008: the HARNESS statistics, standard library only.

One implementation judges evidence for both loops: the org loop (DIAGNOSE and the
post-change check in ``services/harness_service.py``) and the owner-launched Skill
Lab, which runs locally with no ``DATABASE_URL``. So this module imports nothing
outside the Python standard library and reads no config: every threshold is a
parameter, which the org loop supplies from its ``HARNESS_*`` settings in ``config.py``.
``tests/test_prd263_harness_verdicts.py`` pins both properties and the numbers below.

It lives in ``core/stats/``, not ``services/``: importing anything under ``services``
runs ``services/__init__.py``, which pulls SQLAlchemy and the Composio client, while
``core/__init__.py`` and ``core/stats/__init__.py`` import nothing. Code outside the
platform may also load this file by path (``importlib.util.spec_from_file_location``). There is deliberately no
``from __future__ import annotations``: with string annotations, ``dataclasses``
looks the module up in ``sys.modules``, which a by-path load need not register.

What it provides (design §4.1 #2, §5.1 ORG-4 and ORG-10, §6.5):

- a minimum-n gate; a one-sided two-proportion z test; a one-sided Mann-Whitney U
  test; Benjamini-Hochberg; an exact one-sided sign test with its minimum-W table;
  the Wilson lower and upper bounds;
- flags: a rate flags only on an absolute AND a relative worsening AND p < alpha;
  a per-run amount (cost or tokens) flags only on a median rise by both AND p < alpha;
- WATCH then REGRESSION, keyed to distinct consecutive ISO-week windows, or one
  flag below a caller-supplied strong p;
- the frozen-reference post-change check, and the diff-in-diff control with its
  recorded "no control" result.

At n_cur=20 against n_ref=80 the rate flag fires for a stable agent 0.049 of weeks
at p=0.8 and 0.015 at p=0.9 (exact binomial), where today's flat 10% rule fires
0.388 (n=10) and 0.421 (n=20).
"""

import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum

# z for a two-sided 95% interval: the Wilson bounds' default confidence.
WILSON_Z_95 = 1.96

# Rates are x/n floats, so a drop that equals a threshold in exact arithmetic
# (1.0 -> 0.8 is a relative drop of exactly 0.20) can land a hair below it.
# Effect-size comparisons allow this much slack so the rational boundary counts.
_FLOAT_TOLERANCE = 1e-9

# Continuity correction for the normal approximation of the discrete U statistic.
_MW_CONTINUITY = 0.5

# A harness_runs.cur_window label: ISO year and ISO week, e.g. "2026-W42".
_ISO_WEEK_LABEL = re.compile(r"(\d{4})-W(\d{2})")


class Verdict(str, Enum):
    """The verdict for one agent x metric in one ISO-week window."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STABLE = "STABLE"
    WATCH = "WATCH"
    REGRESSION = "REGRESSION"


FLAGGED_VERDICTS = frozenset({Verdict.WATCH, Verdict.REGRESSION})


@dataclass(frozen=True)
class FlagResult:
    """One comparison of a current window against its reference.

    ``worsening`` is the change in the bad direction (a success rate's drop, a
    failure rate's or a median cost's rise) and ``relative_worsening`` is that
    change over the reference value. Below the minimum n nothing is tested:
    ``sufficient`` is False and the measured fields stay None.
    """

    n_cur: int
    n_ref: int
    sufficient: bool
    flagged: bool
    p_value: float | None = None
    cur_value: float | None = None
    ref_value: float | None = None
    worsening: float | None = None
    relative_worsening: float | None = None


@dataclass(frozen=True)
class ProportionThresholds:
    """When a rate flags (design §5.1 ORG-4): all three must hold.

    ``min_abs_drop`` and ``min_rel_drop`` are fractions (0.15 is 15 points; 0.20
    is a fifth of the reference rate) and apply to the rate's worsening: its drop,
    or its rise where higher is worse. ``alpha`` bounds the one-sided p-value.
    """

    min_samples: int
    min_abs_drop: float
    min_rel_drop: float
    alpha: float


@dataclass(frozen=True)
class RiseThresholds:
    """When a per-run amount (cost or tokens) flags (ORG-4): all three must hold.

    ``min_rel_rise`` is a fraction of the reference median; ``min_abs_rise`` is in
    the metric's own unit ($ for cost, tokens for tokens).
    """

    min_samples: int
    min_rel_rise: float
    min_abs_rise: float
    alpha: float


def meets_min_samples(n_cur: int, n_ref: int, min_samples: int) -> bool:
    """True when both windows hold at least ``min_samples`` observations.

    Below it the pair is INSUFFICIENT_DATA: n=0 means no evidence, never a 0% rate.
    """
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}")
    return n_cur >= min_samples and n_ref >= min_samples


def _check_counts(successes: int, total: int) -> None:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError(f"counts must satisfy 0 <= x <= n and n >= 1, got {successes}/{total}")


def _upper_tail(z: float) -> float:
    """P(Z >= z) for a standard normal; erfc keeps the far tail accurate."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_proportion_p_lower(x_cur: int, n_cur: int, x_ref: int, n_ref: int) -> float:
    """One-sided pooled two-proportion z test: the p-value that cur's rate is below ref's.

    When every observation in both windows is a success (or every one a failure)
    the windows cannot differ, so the p-value is 1.0.
    """
    _check_counts(x_cur, n_cur)
    _check_counts(x_ref, n_ref)
    if x_cur + x_ref in (0, n_cur + n_ref):
        return 1.0
    pooled = (x_cur + x_ref) / (n_cur + n_ref)
    std_error = math.sqrt(pooled * (1 - pooled) * (1 / n_cur + 1 / n_ref))
    return _upper_tail((x_ref / n_ref - x_cur / n_cur) / std_error)


def _relative(change: float, base: float) -> float:
    """``change`` as a fraction of ``base``; any worsening from a zero base is infinite."""
    if base > 0:
        return change / base
    return math.inf if change > 0 else 0.0


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - _FLOAT_TOLERANCE


def proportion_flag(
    x_cur: int,
    n_cur: int,
    x_ref: int,
    n_ref: int,
    thresholds: ProportionThresholds,
    *,
    higher_is_worse: bool = False,
) -> FlagResult:
    """Judge whether a rate worsened from the reference window to the current one.

    By default ``x`` counts successes and a drop is the worsening (task success,
    heartbeat objective met). With ``higher_is_worse`` ``x`` counts bad outcomes
    (tool failures) and a rise is the worsening, so a failure rate going from 0.05
    to 0.20 is judged on its own scale rather than as a small success-rate drop.
    """
    if not meets_min_samples(n_cur, n_ref, thresholds.min_samples):
        return FlagResult(n_cur=n_cur, n_ref=n_ref, sufficient=False, flagged=False)
    if higher_is_worse:
        p_value = two_proportion_p_lower(x_ref, n_ref, x_cur, n_cur)
    else:
        p_value = two_proportion_p_lower(x_cur, n_cur, x_ref, n_ref)
    cur_rate, ref_rate = x_cur / n_cur, x_ref / n_ref
    worsening = cur_rate - ref_rate if higher_is_worse else ref_rate - cur_rate
    relative = _relative(worsening, ref_rate)
    flagged = (
        _at_least(worsening, thresholds.min_abs_drop)
        and _at_least(relative, thresholds.min_rel_drop)
        and p_value < thresholds.alpha
    )
    return FlagResult(
        n_cur=n_cur,
        n_ref=n_ref,
        sufficient=True,
        flagged=flagged,
        p_value=p_value,
        cur_value=cur_rate,
        ref_value=ref_rate,
        worsening=worsening,
        relative_worsening=relative,
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks of ``values`` in input order; tied values share their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in order[start : end + 1]:
            ranks[position] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def mann_whitney_p_greater(cur: Sequence[float], ref: Sequence[float]) -> float:
    """One-sided Mann-Whitney U test: the p-value that cur's values tend to exceed ref's.

    Normal approximation of U with the tie-corrected variance and a 0.5 continuity
    correction; the minimum-n gate (20 per window) keeps it in the range where the
    approximation holds. When every value is tied the p-value is 1.0.
    """
    if not cur or not ref:
        raise ValueError("Mann-Whitney needs at least one value in each sample")
    n_cur, n_ref = len(cur), len(ref)
    pooled = [*cur, *ref]
    total = n_cur + n_ref
    u_cur = sum(_average_ranks(pooled)[:n_cur]) - n_cur * (n_cur + 1) / 2
    tie_term = sum(t**3 - t for t in Counter(pooled).values()) / (total * (total - 1))
    variance = n_cur * n_ref / 12 * ((total + 1) - tie_term)
    if variance <= 0:
        return 1.0
    return _upper_tail((u_cur - n_cur * n_ref / 2 - _MW_CONTINUITY) / math.sqrt(variance))


def median_rise_flag(
    cur: Sequence[float],
    ref: Sequence[float],
    thresholds: RiseThresholds,
) -> FlagResult:
    """Judge whether a per-run amount (cost or tokens per run) rose against the reference.

    Flags only when the median rises by ``min_rel_rise`` and by ``min_abs_rise``,
    and the one-sided Mann-Whitney p-value (over every run, not just the medians)
    is below ``alpha``.
    """
    n_cur, n_ref = len(cur), len(ref)
    if not meets_min_samples(n_cur, n_ref, thresholds.min_samples):
        return FlagResult(n_cur=n_cur, n_ref=n_ref, sufficient=False, flagged=False)
    cur_median, ref_median = statistics.median(cur), statistics.median(ref)
    rise = cur_median - ref_median
    relative = _relative(rise, ref_median)
    p_value = mann_whitney_p_greater(cur, ref)
    flagged = (
        _at_least(rise, thresholds.min_abs_rise)
        and _at_least(relative, thresholds.min_rel_rise)
        and p_value < thresholds.alpha
    )
    return FlagResult(
        n_cur=n_cur,
        n_ref=n_ref,
        sufficient=True,
        flagged=flagged,
        p_value=p_value,
        cur_value=cur_median,
        ref_value=ref_median,
        worsening=rise,
        relative_worsening=relative,
    )


def benjamini_hochberg(p_values: Sequence[float], q: float) -> tuple[bool, ...]:
    """Benjamini-Hochberg step-up at false-discovery rate ``q``; rejections in input order.

    Ranks the m p-values ascending, finds the largest rank k with p_(k) <= k/m * q,
    and rejects every hypothesis ranked k or better (so a p-value above its own
    line is still rejected when a larger one below it clears its line).
    """
    if not 0 < q <= 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    cutoff = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= rank / count * q:
            cutoff = rank
    rejected = set(order[:cutoff])
    return tuple(index in rejected for index in range(count))


def fdr_filter(results: Sequence[FlagResult], q: float) -> tuple[FlagResult, ...]:
    """Keep a flag only if its p-value survives Benjamini-Hochberg across the run.

    The family is every judged result of the run (flagged or not); unjudged results
    are returned unchanged. Returns new results; the inputs are not modified.
    """
    judged = [index for index, result in enumerate(results) if result.sufficient]
    survives = benjamini_hochberg([results[index].p_value for index in judged], q)
    kept = {index for index, ok in zip(judged, survives) if ok}
    return tuple(
        replace(result, flagged=result.flagged and index in kept) if result.sufficient else result
        for index, result in enumerate(results)
    )


def sign_test_p(wins: int, losses: int) -> float:
    """Exact one-sided sign test: P(X >= wins) for X ~ Binomial(wins + losses, 1/2).

    ``wins`` and ``losses`` count discordant rows only (candidate passes and base
    fails, and the reverse). With no discordant rows the p-value is 1.0.
    """
    if wins < 0 or losses < 0:
        raise ValueError(f"wins and losses must be non-negative, got {wins}, {losses}")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant


def sign_test_passes(wins: int, losses: int, alpha: float) -> bool:
    """True when the sign test clears ``alpha``; inclusive (p <= alpha), as design §6.5 states."""
    return sign_test_p(wins, losses) <= alpha


def sign_test_min_wins(discordant: int, alpha: float) -> int | None:
    """The smallest W out of ``discordant`` = W + L rows that passes the sign test.

    None when even W = W + L cannot pass. At alpha 0.05 this is the design §6.5
    table: 5->5, 6->6, 7->7, 8->7, 9->8, 10->9, 11->9, 12->10.
    """
    for wins in range(discordant + 1):
        if sign_test_passes(wins, discordant - wins, alpha):
            return wins
    return None


def _wilson_centre_spread(successes: int, total: int, z: float) -> tuple[float, float, float]:
    p = successes / total
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return centre, spread, denominator


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z_95) -> float:
    """Wilson score lower bound (95% confidence by default); 0.0 with no observations.

    Used for routing-edge confidence (core/services/edge_builder.py) and for HARNESS
    rules that must be sure a rate is high, e.g. a tool's failure rate.
    """
    if total == 0:
        return 0.0
    centre, spread, denominator = _wilson_centre_spread(successes, total, z)
    return (centre - spread) / denominator


def wilson_upper_bound(successes: int, total: int, z: float = WILSON_Z_95) -> float:
    """Wilson score upper bound (95% confidence by default); 1.0 with no observations.

    For rules that must be sure a rate is low (heartbeat objective met); with no
    data the bound is 1.0, so such a rule never fires on missing evidence.
    """
    if total == 0:
        return 1.0
    centre, spread, denominator = _wilson_centre_spread(successes, total, z)
    return (centre + spread) / denominator


def iso_week_label(day: date) -> str:
    """The ISO-week window label containing ``day``, e.g. "2026-W42"."""
    year, week, _ = day.isocalendar()
    return f"{year:04d}-W{week:02d}"


def _week_monday(window: str) -> date:
    match = _ISO_WEEK_LABEL.fullmatch(window)
    if match is None:
        raise ValueError(f"not an ISO-week label like 2026-W42: {window!r}")
    return date.fromisocalendar(int(match[1]), int(match[2]), 1)


def previous_iso_week(window: str) -> str:
    """The label of the ISO week before ``window`` (2027-W01 -> 2026-W53).

    Raises ValueError for a malformed label or a week its year does not have.
    """
    return iso_week_label(_week_monday(window) - timedelta(weeks=1))


def _is_strong(result: FlagResult, strong_p: float) -> bool:
    return result.flagged and result.p_value is not None and result.p_value < strong_p


def window_verdict(
    window: str,
    result: FlagResult,
    history: Mapping[str, str],
    strong_p: float,
) -> Verdict:
    """The verdict for one agent x metric in ISO-week ``window`` (design ORG-4).

    ``history`` maps the windows already judged for this agent and metric to their
    verdicts (``Verdict`` or its string value, as stored in harness_runs). A window
    already in it returns its stored verdict: a rerun of the same week never counts
    as a second flag. A new flag is WATCH; it is REGRESSION when the previous ISO
    week was flagged too (two distinct, consecutive windows) or its p is below
    ``strong_p``.
    """
    previous = previous_iso_week(window)
    if window in history:
        return Verdict(history[window])
    if not result.sufficient:
        return Verdict.INSUFFICIENT_DATA
    if not result.flagged:
        return Verdict.STABLE
    previous_flagged = previous in history and Verdict(history[previous]) in FLAGGED_VERDICTS
    if previous_flagged or _is_strong(result, strong_p):
        return Verdict.REGRESSION
    return Verdict.WATCH


class PostChangeStatus(str, Enum):
    """Where a post-change check stands (design ORG-10)."""

    ROLLBACK = "ROLLBACK"  # trigger met: raise the rollback grant
    WAITING = "WAITING"  # fewer than max_weeks seen, no trigger yet
    NOT_JUDGED = "NOT_JUDGED"  # max_weeks passed without one week at the minimum n
    HELD = "HELD"  # max_weeks passed, judged, no trigger: the change stands


@dataclass(frozen=True)
class PostChangeOutcome:
    """The result of a post-change check; ``trigger_week`` is 1-based from the change."""

    status: PostChangeStatus
    weeks_seen: int
    judged_weeks: int
    trigger_week: int | None = None


def post_change_verdict(
    weekly: Sequence[FlagResult],
    max_weeks: int,
    strong_p: float,
) -> PostChangeOutcome:
    """Decide a post-change check from weekly results against the frozen reference.

    ``weekly[i]`` is week i+1 after the change, each judged against the same frozen
    reference (the window before ``applied_at``, which never trails into the
    post-change weeks). Only the first ``max_weeks`` count. ROLLBACK on flags in
    two consecutive weeks or one flag with p below ``strong_p``; a week under the
    minimum n breaks a run of flags.
    """
    if max_weeks < 1:
        raise ValueError(f"max_weeks must be at least 1, got {max_weeks}")
    window = list(weekly[:max_weeks])
    previous_flagged = False
    for week, result in enumerate(window, start=1):
        if _is_strong(result, strong_p) or (result.flagged and previous_flagged):
            judged = sum(r.sufficient for r in window[:week])
            return PostChangeOutcome(PostChangeStatus.ROLLBACK, week, judged, trigger_week=week)
        previous_flagged = result.flagged
    judged = sum(r.sufficient for r in window)
    if len(window) < max_weeks:
        status = PostChangeStatus.WAITING
    elif judged == 0:
        status = PostChangeStatus.NOT_JUDGED
    else:
        status = PostChangeStatus.HELD
    return PostChangeOutcome(status, len(window), judged)


def frozen_reference_check(
    weekly_counts: Sequence[tuple[int, int]],
    frozen_ref: tuple[int, int],
    thresholds: ProportionThresholds,
    max_weeks: int,
    strong_p: float,
    *,
    higher_is_worse: bool = False,
) -> PostChangeOutcome:
    """The frozen-reference post-change test for one rate (design ORG-10).

    ``weekly_counts`` are (x, n) per week since the change and ``frozen_ref`` the
    (x, n) of the reference window before ``applied_at``. Every week is judged
    against that same reference, so a sustained drop cannot be absorbed into it
    as it would be into a trailing one.
    """
    x_ref, n_ref = frozen_ref
    weekly = [
        proportion_flag(x, n, x_ref, n_ref, thresholds, higher_is_worse=higher_is_worse)
        for x, n in weekly_counts[:max_weeks]
    ]
    return post_change_verdict(weekly, max_weeks, strong_p)


class ControlStatus(str, Enum):
    """How a changed agent's drop compares with unchanged agents' drops."""

    EXCEEDS_CONTROL = "EXCEEDS_CONTROL"
    WITHIN_CONTROL = "WITHIN_CONTROL"
    NO_CONTROL = "NO_CONTROL"


@dataclass(frozen=True)
class DiffInDiffResult:
    """The diff-in-diff control for one post-change trigger."""

    status: ControlStatus
    treated_drop: float
    control_n: int
    control_median_drop: float | None = None

    @property
    def raises_rollback(self) -> bool:
        """A rollback is raised when the drop beats the control, or when there is none."""
        return self.status is not ControlStatus.WITHIN_CONTROL


def diff_in_diff(treated_drop: float, control_drops: Sequence[float]) -> DiffInDiffResult:
    """Compare a changed agent's drop with the median drop of judged unchanged agents.

    A drop is the reference value minus the post-change value over the same weeks
    (positive is worse). EXCEEDS_CONTROL when the treated drop is strictly above the
    control median, WITHIN_CONTROL otherwise. With no judged unchanged agent the
    result is NO_CONTROL, recorded as such, and the rollback is still raised.
    """
    if not control_drops:
        return DiffInDiffResult(ControlStatus.NO_CONTROL, treated_drop, control_n=0)
    median_drop = statistics.median(control_drops)
    if treated_drop > median_drop:
        status = ControlStatus.EXCEEDS_CONTROL
    else:
        status = ControlStatus.WITHIN_CONTROL
    return DiffInDiffResult(status, treated_drop, len(control_drops), median_drop)

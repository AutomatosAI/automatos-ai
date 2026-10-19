"""PRD-263 US-008 — the HARNESS statistics module, ``core/stats/harness_verdicts.py``.

Today's HARNESS calls a regression when any of four relative deltas tops 10%
(``harness_service.py`` ``_REGRESSION_THRESHOLD``); fed honest weekly numbers it
would flag a perfectly stable agent in about 40% of weeks. The org loop and the
local Skill Lab now judge evidence with one standard-library module. This suite pins:

- the module imports only the standard library and loads by file path with no
  ``DATABASE_URL`` and nothing from the platform (the Lab's contract);
- ``wilson_lower_bound`` has exactly one definition, there, and edge_builder imports it;
- the design's numbers (design §6.8, PRD-263 §10), by EXACT binomial enumeration
  through the module's own functions, never simulation: the single false-flag rate
  and power at n_cur=20 vs n_ref=80, two consecutive false flags, the frozen-reference
  rollback rates, today's rule as a reference, and the sign test's minimum-W table;
- every primitive (z test, Mann-Whitney, Benjamini-Hochberg, Wilson bounds) against
  hand-computed values, and the verdict rules (ISO-week WATCH -> REGRESSION with a
  same-window rerun reused, the post-change check, the diff-in-diff control).

Every test fails before this change: the module does not exist on main.
"""
import ast
import itertools
import os
import random
import subprocess
import sys
from datetime import date
from functools import lru_cache
from math import comb
from pathlib import Path

import pytest

from core.stats.harness_verdicts import (
    ControlStatus,
    FlagResult,
    PostChangeStatus,
    ProportionThresholds,
    RiseThresholds,
    Verdict,
    benjamini_hochberg,
    diff_in_diff,
    fdr_filter,
    frozen_reference_check,
    iso_week_label,
    mann_whitney_p_greater,
    median_rise_flag,
    post_change_verdict,
    previous_iso_week,
    proportion_flag,
    sign_test_min_wins,
    sign_test_p,
    sign_test_passes,
    two_proportion_p_lower,
    wilson_lower_bound,
    wilson_upper_bound,
    window_verdict,
)

ORCHESTRATOR = Path(__file__).resolve().parents[1]
MODULE_PATH = ORCHESTRATOR / "core" / "stats" / "harness_verdicts.py"
EDGE_BUILDER_PATH = ORCHESTRATOR / "core" / "services" / "edge_builder.py"
# Directories under orchestrator/ that hold no code of ours (the same list as
# test_p2w0_service_imports_resolve.py).
_SKIP_DIRS = {"venv", ".venv", "node_modules", "__pycache__", "model_cache", ".git"}

# The design's defaults (§5.1 ORG-4 / ORG-10). PR-1D reads them from config.py;
# the module itself takes every one as a parameter.
MIN_SAMPLES = 20
ABS_DROP = 0.15
REL_DROP = 0.20
ALPHA = 0.05
STRONG_P = 0.001
COST_REL_RISE = 0.25
COST_ABS_RISE = 0.005
POST_CHANGE_MAX_WEEKS = 4
SIGN_ALPHA = 0.05
RATES = ProportionThresholds(MIN_SAMPLES, ABS_DROP, REL_DROP, ALPHA)
COSTS = RiseThresholds(MIN_SAMPLES, COST_REL_RISE, COST_ABS_RISE, ALPHA)

# One judged week (~20 Tasks) against the 28-day reference (4 weeks of 20).
N_CUR = 20
N_REF = 80
WEEK_N = 20

# Exact figures are pinned at the three (or four) decimals the design quotes them
# to, so the tolerance is half a unit of that last digit.
THREE_DP = 5e-4
FOUR_DP = 5e-5
# The design's frozen-reference figures came from a 100,000-trial fixed-seed
# simulation (frozen_sim.py). One standard error there is at most 0.0016, so the
# design figure must sit within 0.0025 of the exact value; the exact value is
# pinned separately at THREE_DP.
SIMULATION_TOLERANCE = 0.0025


def _pmf(n: int, k: int, p: float) -> float:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def _flag_table(n_cur: int, n_ref: int) -> dict[tuple[int, int], FlagResult]:
    """Every (x_cur, x_ref) outcome judged once by the module's rate flag."""
    return {
        (x_cur, x_ref): proportion_flag(x_cur, n_cur, x_ref, n_ref, RATES)
        for x_cur in range(n_cur + 1)
        for x_ref in range(n_ref + 1)
    }


FLAGS_20_80 = _flag_table(N_CUR, N_REF)


def _single_flag_rate(p_cur: float, p_ref: float) -> float:
    return sum(
        _pmf(N_CUR, x_cur, p_cur) * _pmf(N_REF, x_ref, p_ref)
        for (x_cur, x_ref), result in FLAGS_20_80.items()
        if result.flagged
    )


# ── The module stands alone ─────────────────────────────────────────────────


def test_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"line {node.lineno}: a relative import pulls in the package"
            imported.append(node.module)
    assert imported, "the module should import something from the standard library"
    outside = sorted({name for name in imported if name.split(".")[0] not in sys.stdlib_module_names})
    assert not outside, f"harness_verdicts.py must import only the standard library, found {outside}"


def test_it_loads_by_path_with_no_database_url_and_nothing_from_the_platform():
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('lab_verdicts', {str(MODULE_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert module.sign_test_min_wins(8, 0.05) == 7\n"
        "leaked = [m for m in ('sqlalchemy', 'config', 'services', 'core') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
    ran = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, env=env, check=False, timeout=60
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


def test_package_import_loads_nothing_from_the_platform():
    # Why it lives in core/stats/: anything under services/ runs services/__init__.py,
    # which loads SQLAlchemy, config and the Composio client.
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ORCHESTRATOR)!r})\n"
        "import core.stats.harness_verdicts\n"
        "leaked = [m for m in ('sqlalchemy', 'config', 'services') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "DATABASE_URL"}
    ran = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, env=env, check=False, timeout=60
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


def _defines(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if name not in text:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(ast.parse(text))
    )


def test_wilson_lower_bound_has_one_home_and_edge_builder_imports_it():
    # The whole tree, test trees included, as the acceptance check's recursive grep
    # for the definition over orchestrator/ sees it; only files naming it are parsed.
    homes = [
        path.relative_to(ORCHESTRATOR).as_posix()
        for path in ORCHESTRATOR.rglob("*.py")
        if not (_SKIP_DIRS & set(path.relative_to(ORCHESTRATOR).parts)) and _defines(path, "wilson_lower_bound")
    ]
    assert homes == ["core/stats/harness_verdicts.py"]
    tree = ast.parse(EDGE_BUILDER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "core.stats.harness_verdicts"
        for alias in node.names
    }
    assert "wilson_lower_bound" in imports


# ── The rate flag: false flags, power and today's rule (exact binomial) ─────


@pytest.mark.parametrize(
    ("p_ref", "p_cur", "expected"),
    [
        (0.8, 0.8, 0.049),  # a stable agent at p=0.8: false flags
        (0.9, 0.9, 0.015),  # a stable agent at p=0.9: false flags
        (0.8, 0.5, 0.837),  # power for a 0.8 -> 0.5 drop
        (0.8, 0.6, 0.571),  # power for a 0.8 -> 0.6 drop
    ],
)
def test_single_flag_rate_at_20_against_80(p_ref, p_cur, expected):
    assert _single_flag_rate(p_cur, p_ref) == pytest.approx(expected, abs=THREE_DP)


def _old_rule_pct_delta(cur: float, prev: float) -> float:
    """``HarnessService._compute_deltas._pct_delta`` as it stands before PRD-263."""
    if prev == 0:
        return 0 if cur == 0 else 100.0
    return (cur - prev) / abs(prev) * 100


def _old_rule_flags(x_cur: int, x_prev: int, n: int, *, with_error_rate: bool) -> bool:
    """Today's rule on one week against the previous week: worst delta above 10%."""
    success_cur, success_prev = x_cur / n, x_prev / n
    worst = -_old_rule_pct_delta(success_cur, success_prev)
    if with_error_rate:
        worst = max(worst, _old_rule_pct_delta(1 - success_cur, 1 - success_prev))
    return worst > 10.0


def _old_rule_rate(p: float, n: int, *, with_error_rate: bool) -> float:
    return sum(
        _pmf(n, a, p) * _pmf(n, b, p)
        for a in range(n + 1)
        for b in range(n + 1)
        if _old_rule_flags(a, b, n, with_error_rate=with_error_rate)
    )


@pytest.mark.parametrize(
    ("n", "with_error_rate", "expected"),
    [(10, True, 0.388), (20, True, 0.421), (10, False, 0.359)],
)
def test_todays_rule_flags_a_stable_agent_in_about_40_percent_of_weeks(n, with_error_rate, expected):
    # Success and error rate only (cost and efficiency excluded), so a lower bound.
    old = _old_rule_rate(0.8, n, with_error_rate=with_error_rate)
    assert old == pytest.approx(expected, abs=THREE_DP)
    assert _single_flag_rate(0.8, 0.8) < old / 5


def test_two_consecutive_false_flags():
    """P(flag in week t and in week t+1) for a stable p=0.8 agent, exactly.

    Weeks w0..w5 are Binomial(20, 0.8); s3 = w1+w2+w3 is shared by both references.
    With a trailing reference (design §6.8, consec_sim.py) week 5 is judged against
    w1..w4, flagged week included: 0.0016. With the clean reference ORG-4 actually
    specifies, the flagged week w4 is left out, so week 5 is judged against w1..w3
    (n=60): 0.0042.
    """
    flags_20_60 = _flag_table(WEEK_N, 3 * WEEK_N)
    p = 0.8
    trailing = clean = 0.0
    for s3 in range(3 * WEEK_N + 1):
        for w4 in range(WEEK_N + 1):
            first = sum(
                _pmf(WEEK_N, w0, p) for w0 in range(WEEK_N + 1) if FLAGS_20_80[(w4, w0 + s3)].flagged
            )
            if not first:
                continue
            weight = _pmf(3 * WEEK_N, s3, p) * _pmf(WEEK_N, w4, p) * first
            trailing += weight * sum(
                _pmf(WEEK_N, w5, p) for w5 in range(WEEK_N + 1) if FLAGS_20_80[(w5, s3 + w4)].flagged
            )
            clean += weight * sum(
                _pmf(WEEK_N, w5, p) for w5 in range(WEEK_N + 1) if flags_20_60[(w5, s3)].flagged
            )
    assert trailing == pytest.approx(0.0016, abs=FOUR_DP)
    assert clean == pytest.approx(0.0042, abs=FOUR_DP)


@pytest.mark.parametrize(
    ("x_cur", "n_cur", "x_ref", "n_ref", "why"),
    [
        (375, 1000, 500, 1000, "a quarter down and p = 8.8e-9, but only 12.5 points"),
        (800, 1000, 950, 1000, "15 points down but under a fifth of 0.95"),
        (12, 20, 16, 20, "a big enough drop but p = 0.084"),
        (14, 20, 16, 20, "16/20 vs 14/20 (PRD US-014): STABLE"),
    ],
)
def test_a_rate_flags_only_when_all_three_conditions_hold(x_cur, n_cur, x_ref, n_ref, why):
    result = proportion_flag(x_cur, n_cur, x_ref, n_ref, RATES)
    assert result.sufficient and not result.flagged, why


def test_a_rate_flags_when_all_three_hold_including_the_exact_boundary():
    assert proportion_flag(11, 20, 64, 80, RATES).flagged
    # 1.0 -> 0.8 is a relative drop of exactly 0.20; float division lands below it.
    boundary = proportion_flag(16, 20, 80, 80, RATES)
    assert boundary.flagged
    assert boundary.relative_worsening < REL_DROP


def test_below_the_minimum_n_nothing_is_tested():
    result = proportion_flag(0, 3, 64, 80, RATES)
    assert (result.sufficient, result.flagged, result.p_value) == (False, False, None)
    assert window_verdict("2026-W42", result, {}, STRONG_P) is Verdict.INSUFFICIENT_DATA


def test_a_failure_rate_is_judged_on_its_own_scale():
    # Tool failures 5/100 -> 20/100 quadrupled: a flag. The same counts read as a
    # success rate (0.95 -> 0.80) fall short of the relative-drop rule.
    assert proportion_flag(20, 100, 5, 100, RATES, higher_is_worse=True).flagged
    assert not proportion_flag(80, 100, 95, 100, RATES).flagged


# ── Primitives ──────────────────────────────────────────────────────────────


def test_two_proportion_z_test():
    assert two_proportion_p_lower(8, 20, 8, 20) == pytest.approx(0.5)
    assert two_proportion_p_lower(20, 20, 80, 80) == 1.0
    assert two_proportion_p_lower(0, 20, 0, 80) == 1.0
    # pooled 40/64; z = 0.625 / 0.12103 = 5.164
    assert two_proportion_p_lower(10, 32, 30, 32) == pytest.approx(1.2088e-07, rel=1e-3)
    with pytest.raises(ValueError):
        two_proportion_p_lower(21, 20, 10, 20)
    with pytest.raises(ValueError):
        two_proportion_p_lower(0, 0, 10, 20)


@lru_cache(maxsize=None)
def _u_count(m: int, n: int, u: int) -> int:
    """Orderings of m cur and n ref values (no ties) with U = u."""
    if u < 0:
        return 0
    if m == 0 or n == 0:
        return int(u == 0)
    return _u_count(m - 1, n, u - n) + _u_count(m, n - 1, u)


def test_mann_whitney_matches_hand_values_and_the_exact_distribution():
    # U = 9, mean 4.5, variance 3*3*7/12 = 5.25, z = 4 / 2.2913
    assert mann_whitney_p_greater([4, 5, 6], [1, 2, 3]) == pytest.approx(0.040428, abs=1e-5)
    # Ties: ranks 1, 3.5 x4, 6; U = 7; tie-corrected variance 0.75 * (7 - 60/30) = 3.75
    assert mann_whitney_p_greater([2, 2, 3], [1, 2, 2]) == pytest.approx(0.150850, abs=1e-5)
    assert mann_whitney_p_greater([1, 1, 1], [1, 1]) == 1.0
    rng = random.Random(263)
    size = 10
    for _ in range(50):
        values = rng.sample(range(1000), 2 * size)
        cur, ref = values[:size], values[size:]
        u = sum(1 for a in cur for b in ref if a > b)
        exact = sum(_u_count(size, size, k) for k in range(u, size * size + 1)) / comb(2 * size, size)
        assert mann_whitney_p_greater(cur, ref) == pytest.approx(exact, abs=0.01)
    with pytest.raises(ValueError):
        mann_whitney_p_greater([], [1.0])


def test_a_cost_flags_only_on_a_median_rise_by_both_amounts():
    ref = [0.020 + 0.0001 * i for i in range(20)]
    up = median_rise_flag([0.030 + 0.0001 * i for i in range(20)], ref, COSTS)
    assert up.flagged and up.worsening == pytest.approx(0.01)
    cheap_ref = [0.010 + 0.00001 * i for i in range(20)]
    tiny = median_rise_flag([0.013 + 0.00001 * i for i in range(20)], cheap_ref, COSTS)
    assert tiny.p_value < ALPHA and not tiny.flagged  # +30% but only $0.003
    dear_ref = [1.00 + 0.001 * i for i in range(20)]
    small = median_rise_flag([1.10 + 0.001 * i for i in range(20)], dear_ref, COSTS)
    assert small.p_value < ALPHA and not small.flagged  # +$0.10 but only +10%
    noisy = median_rise_flag([0.010] * 9 + [0.100] * 11, [0.010] * 10 + [0.100] * 10, COSTS)
    assert noisy.worsening >= COST_ABS_RISE and noisy.relative_worsening >= COST_REL_RISE
    assert noisy.p_value > ALPHA and not noisy.flagged  # median +$0.045 (+82%) but p = 0.383
    assert not median_rise_flag(ref[:5], ref, COSTS).sufficient


def test_benjamini_hochberg_steps_up():
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005, 0.20], 0.10) == (True, True, True, True, False)
    # 0.03 misses its own line (0.025) but 0.035 clears 0.05, so both are rejected.
    assert benjamini_hochberg([0.03, 0.035], 0.05) == (True, True)
    assert benjamini_hochberg([0.04, 0.5], 0.05) == (False, False)
    assert benjamini_hochberg([], 0.10) == ()
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01], 0.0)


def test_fdr_filter_drops_flags_that_do_not_survive_the_run():
    def judged(p_value: float) -> FlagResult:
        return FlagResult(N_CUR, N_REF, sufficient=True, flagged=True, p_value=p_value)

    def not_flagged(p_value: float) -> FlagResult:
        return FlagResult(N_CUR, N_REF, sufficient=True, flagged=False, p_value=p_value)

    # The family is every judged result, flagged or not: m = 4 here, so 0.04 misses
    # its line (2/4 * 0.05 = 0.025). Counting only the flagged two would keep it.
    unjudged = FlagResult(3, N_REF, sufficient=False, flagged=False)
    results = (judged(0.001), unjudged, judged(0.04), not_flagged(0.30), not_flagged(0.50))
    filtered = fdr_filter(results, 0.05)
    assert [r.flagged for r in filtered] == [True, False, False, False, False]
    assert filtered[1] is unjudged
    assert results[2].flagged, "the inputs are left as they were"


def test_sign_test_minimum_wins_table():
    table = {n: sign_test_min_wins(n, SIGN_ALPHA) for n in range(1, 13)}
    assert table == {1: None, 2: None, 3: None, 4: None, 5: 5, 6: 6, 7: 7, 8: 7, 9: 8, 10: 9, 11: 9, 12: 10}


def test_sign_test_is_exact_and_inclusive():
    assert sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert sign_test_p(7, 1) == pytest.approx(9 / 256)
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_passes(7, 1, SIGN_ALPHA)
    assert not sign_test_passes(6, 2, SIGN_ALPHA)
    # Inclusive: an alpha exactly equal to the p-value (1/32 is exact in binary) passes.
    assert sign_test_passes(5, 0, 1 / 32)
    with pytest.raises(ValueError):
        sign_test_p(-1, 3)


def test_wilson_bounds():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_upper_bound(0, 0) == 1.0
    assert wilson_lower_bound(50, 100) == pytest.approx(0.403830, abs=1e-6)
    assert wilson_upper_bound(50, 100) == pytest.approx(0.596170, abs=1e-6)
    assert wilson_upper_bound(3, 10) == pytest.approx(1 - wilson_lower_bound(7, 10))
    # heartbeat_tune (US-050): 1 objective met in 25 runs is surely under 0.2; 2 is not.
    assert wilson_upper_bound(1, 25) < 0.2 < wilson_upper_bound(2, 25)


# ── WATCH, then REGRESSION, keyed to ISO-week windows ───────────────────────


def _flag(p_value: float = 0.01) -> FlagResult:
    return FlagResult(N_CUR, N_REF, sufficient=True, flagged=True, p_value=p_value)


STABLE_RESULT = FlagResult(N_CUR, N_REF, sufficient=True, flagged=False, p_value=0.5)
UNJUDGED = FlagResult(3, N_REF, sufficient=False, flagged=False)


def test_iso_week_labels():
    assert iso_week_label(date(2026, 10, 18)) == "2026-W42"
    assert previous_iso_week("2026-W43") == "2026-W42"
    assert previous_iso_week("2027-W01") == "2026-W53"
    for bad in ("2025-W53", "2026-42", "2026-W1", "W42"):
        with pytest.raises(ValueError):
            previous_iso_week(bad)


def test_a_collapse_is_watch_in_week_one_and_regression_in_week_two():
    week = proportion_flag(11, 20, 64, 80, RATES)  # 0.80 -> 0.55, p = 0.0105
    first = window_verdict("2026-W42", week, {}, STRONG_P)
    assert first is Verdict.WATCH
    assert window_verdict("2026-W43", week, {"2026-W42": first}, STRONG_P) is Verdict.REGRESSION


def test_a_same_window_rerun_reuses_its_verdict_and_never_counts_twice():
    history = {"2026-W42": Verdict.WATCH}
    assert window_verdict("2026-W42", _flag(), history, STRONG_P) is Verdict.WATCH
    assert window_verdict("2026-W42", _flag(1e-6), history, STRONG_P) is Verdict.WATCH
    assert window_verdict("2026-W42", _flag(), {"2026-W42": "STABLE"}, STRONG_P) is Verdict.STABLE


def test_flags_must_be_in_consecutive_weeks():
    assert window_verdict("2026-W44", _flag(), {"2026-W42": Verdict.WATCH}, STRONG_P) is Verdict.WATCH
    for prior in (Verdict.STABLE, Verdict.INSUFFICIENT_DATA):
        assert window_verdict("2026-W43", _flag(), {"2026-W42": prior}, STRONG_P) is Verdict.WATCH
    # Across the ISO year boundary, as stored in harness_runs (plain strings).
    assert window_verdict("2027-W01", _flag(), {"2026-W53": "WATCH"}, STRONG_P) is Verdict.REGRESSION
    assert window_verdict("2026-W43", _flag(), {"2026-W42": "REGRESSION"}, STRONG_P) is Verdict.REGRESSION


def test_one_flag_below_the_strong_p_is_regression_at_once():
    assert window_verdict("2026-W42", _flag(STRONG_P / 2), {}, STRONG_P) is Verdict.REGRESSION
    assert window_verdict("2026-W42", _flag(STRONG_P), {}, STRONG_P) is Verdict.WATCH


def test_a_judged_week_without_a_flag_is_stable():
    assert window_verdict("2026-W42", STABLE_RESULT, {}, STRONG_P) is Verdict.STABLE
    assert window_verdict("2026-W42", UNJUDGED, {}, STRONG_P) is Verdict.INSUFFICIENT_DATA


# ── The frozen-reference post-change check ──────────────────────────────────


@pytest.mark.parametrize(
    ("weekly", "status", "trigger_week"),
    [
        ([_flag(), _flag()], PostChangeStatus.ROLLBACK, 2),
        ([STABLE_RESULT, _flag(STRONG_P / 2)], PostChangeStatus.ROLLBACK, 2),
        ([_flag(), STABLE_RESULT, _flag(), STABLE_RESULT], PostChangeStatus.HELD, None),
        ([_flag(), UNJUDGED, _flag()], PostChangeStatus.WAITING, None),
        ([UNJUDGED, UNJUDGED], PostChangeStatus.WAITING, None),
        ([UNJUDGED] * POST_CHANGE_MAX_WEEKS, PostChangeStatus.NOT_JUDGED, None),
        ([STABLE_RESULT] * 3 + [_flag(), _flag()], PostChangeStatus.HELD, None),
    ],
)
def test_post_change_verdict(weekly, status, trigger_week):
    outcome = post_change_verdict(weekly, POST_CHANGE_MAX_WEEKS, STRONG_P)
    assert (outcome.status, outcome.trigger_week) == (status, trigger_week)
    assert outcome.weeks_seen <= POST_CHANGE_MAX_WEEKS


def test_post_change_verdict_needs_a_window():
    with pytest.raises(ValueError):
        post_change_verdict([], 0, STRONG_P)


def test_frozen_reference_check_judges_every_week_against_the_same_reference():
    frozen = (64, 80)
    outcome = frozen_reference_check([(10, 20), (10, 20)], frozen, RATES, POST_CHANGE_MAX_WEEKS, STRONG_P)
    assert (outcome.status, outcome.trigger_week, outcome.judged_weeks) == (PostChangeStatus.ROLLBACK, 2, 2)
    steady = frozen_reference_check([(16, 20)] * 4, frozen, RATES, POST_CHANGE_MAX_WEEKS, STRONG_P)
    assert steady.status is PostChangeStatus.HELD
    thin = frozen_reference_check([(2, 3)] * 4, frozen, RATES, POST_CHANGE_MAX_WEEKS, STRONG_P)
    assert (thin.status, thin.judged_weeks) == (PostChangeStatus.NOT_JUDGED, 0)


def _week_class(result: FlagResult) -> int:
    """0 no flag, 1 flag, 2 flag below the strong p."""
    if not result.flagged:
        return 0
    return 2 if result.p_value < STRONG_P else 1


_CLASS_EXAMPLES = (STABLE_RESULT, _flag(), _flag(STRONG_P / 2))


def _triggers(sequence: tuple[int, ...]) -> bool:
    weekly = [_CLASS_EXAMPLES[week_class] for week_class in sequence]
    outcome = post_change_verdict(weekly, POST_CHANGE_MAX_WEEKS, STRONG_P)
    return outcome.status is PostChangeStatus.ROLLBACK


_TRIGGERING_SEQUENCES = [
    sequence for sequence in itertools.product(range(3), repeat=POST_CHANGE_MAX_WEEKS) if _triggers(sequence)
]


def _rollback_rate(p_before: float, p_after: float) -> float:
    """P(rollback within 4 weeks), exactly.

    The frozen reference is Binomial(80, p_before); the 4 post-change weeks are
    independent Binomial(20, p_after) given it, so each week reduces to its class
    and post_change_verdict is evaluated on all 3^4 class sequences.
    """
    total = 0.0
    for x_ref in range(N_REF + 1):
        class_p = [0.0, 0.0, 0.0]
        for x_cur in range(N_CUR + 1):
            class_p[_week_class(FLAGS_20_80[(x_cur, x_ref)])] += _pmf(N_CUR, x_cur, p_after)
        triggered = 0.0
        for sequence in _TRIGGERING_SEQUENCES:
            weight = 1.0
            for week_class in sequence:
                weight *= class_p[week_class]
            triggered += weight
        total += _pmf(N_REF, x_ref, p_before) * triggered
    return total


@pytest.mark.parametrize(
    ("p_before", "p_after", "exact", "design_simulated"),
    [
        (0.8, 0.5, 0.940, 0.938),
        (0.8, 0.6, 0.667, 0.668),
        (0.8, 0.65, 0.444, 0.443),
        (0.8, 0.8, 0.019, 0.020),  # false rollback per metric
        (0.9, 0.9, 0.011, 0.011),  # false rollback per metric
    ],
)
def test_frozen_reference_rollback_rate(p_before, p_after, exact, design_simulated):
    rate = _rollback_rate(p_before, p_after)
    assert rate == pytest.approx(exact, abs=THREE_DP)
    assert rate == pytest.approx(design_simulated, abs=SIMULATION_TOLERANCE)


# ── The diff-in-diff control ────────────────────────────────────────────────


def test_no_unchanged_agent_records_no_control_and_still_raises_the_rollback():
    result = diff_in_diff(0.25, [])
    assert result.status is ControlStatus.NO_CONTROL
    assert (result.control_n, result.control_median_drop) == (0, None)
    assert result.raises_rollback


def test_the_drop_must_exceed_the_median_drop_of_unchanged_agents():
    exceeds = diff_in_diff(0.30, [0.0, 0.05, 0.10])
    assert exceeds.status is ControlStatus.EXCEEDS_CONTROL
    assert (exceeds.control_median_drop, exceeds.control_n) == (0.05, 3)
    assert exceeds.raises_rollback
    within = diff_in_diff(0.05, [0.05, 0.10, 0.20])
    assert within.status is ControlStatus.WITHIN_CONTROL and not within.raises_rollback
    tie = diff_in_diff(0.10, [0.05, 0.10, 0.20])
    assert tie.status is ControlStatus.WITHIN_CONTROL

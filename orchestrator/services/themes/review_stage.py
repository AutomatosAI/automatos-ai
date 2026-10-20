"""PRD-261 C-02a — the automated review stage, server side, run inline.

Runs on every package a person may later review: inline at a Marketplace submission (C-01a's
route) and at a private upload (``api/themes_import.py``), after ``read_theme_package`` has
accepted the package. Both routes are plain ``def``, so they run in a threadpool thread with no
event loop, and the async LLM stage is run there with ``asyncio.run`` (``llm_stage.run_llm_stage``).

The report (``theme_packages.scan_report``, schema ``automatos-theme-scan/v2``) gives every check
a ``level`` a reviewer reads as Error, Warning or Pass, plus ``not_measured`` for a check this
host can't make yet:

* ``sizes``, ``cssScope``, ``looks``: the reader's checks (A-06a/A-07a, ``scan.py``); a package
  that failed one never gets here, so each is a pass with what it measured;
* ``contrast`` (AA): for a theme whose theme.json has a ``mobile`` block (M-03, PB-35), the block's
  ten colour pairs (``contract.MOBILE_CONTRAST_PAIRS``: eight text pairs at 4.5:1, two UI pairs at
  3:1) per scheme of ``mobile.tokens``, or per preset painted through ``boot/prepaint.json`` under
  ``fromLooks`` (``mobile_contrast.py``). A measured pair under its minimum is an error naming the
  scheme or preset, both tokens and the ratio (``MOBILE_CONTRAST_FAIL_LEVEL``), every measured pair
  passing is a pass, and a pair whose two tokens are not both set is listed ``not_measured``.
  Without the block: ``not_measured`` per preset. A look is knob values; neither
  ``looks/schema.json`` nor ``boot/prepaint.json`` says which painted colour is text on which
  ground, so there is no pair to measure (``scan.CONTRAST_NOT_RUN``). The reviewer checks it by
  eye until the look schema declares pairs or axe runs (D3-b);
* ``uses``: the declared SDK areas against the names the bundle imports (``sdk_uses.py``);
* ``clonedFrom``: each cloned page's ``clonedFrom.sha`` against this release's core page files
  (``data/core-pages.json`` through ``drift.py``); a stale clone is a warning;
* ``llm``: the LLM stage (``llm_stage.py``); when it can't run on all the code it is a warning
  "LLM stage did not run", never a pass and never ``not_run``.

The verdict: any error → ``blocked`` (approve refuses it, C-03a); any warning →
``review_required``; else ``pass``. ``not_measured`` never moves it. The stage never returns
``not_run``: that verdict means the stage itself never ran (a row stored before W-C1).

Who sees what: the whole report is the reviewer's (the admin queue, ``review.report``). The
person who uploaded or submitted the package, and a workspace comparing versions, get
``member_view``: the verdict and the plain checks, never the LLM check's score, findings or
summary, so nobody can tune a package against the model's answers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from services.themes.drift import stale_clones
from services.themes.llm_stage import ChunkCall, TokenBudget, llm_check, run_llm_stage
from services.themes.mobile_contrast import OUTCOME_FAIL, OUTCOME_NOT_MEASURED, OUTCOME_PASS, measure_mobile
from services.themes.package import ThemePackage
from services.themes.scan import build_scan_report
from services.themes.sdk_uses import SurfaceAreas, uses_check

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "automatos-theme-scan/v2"
STAGE = "C-02a"
LEVEL_PASS, LEVEL_WARNING, LEVEL_ERROR, LEVEL_NOT_MEASURED = "pass", "warning", "error", "not_measured"
LEVELS = (LEVEL_PASS, LEVEL_WARNING, LEVEL_ERROR, LEVEL_NOT_MEASURED)
# PB-35: a mobile pair under AA blocks the review (a phone must read a theme that claims it); a warning
# instead is this one constant (Gerard's call).
MOBILE_CONTRAST_FAIL_LEVEL = LEVEL_ERROR
CONTRAST_LEVELS = {
    OUTCOME_FAIL: MOBILE_CONTRAST_FAIL_LEVEL, OUTCOME_PASS: LEVEL_PASS, OUTCOME_NOT_MEASURED: LEVEL_NOT_MEASURED,
}
VERDICT_PASS, VERDICT_REVIEW, VERDICT_BLOCKED = "pass", "review_required", "blocked"
READER_CHECKS = ("sizes", "cssScope", "looks")
# Checks only the reviewer sees: the LLM stage's score and findings (``member_view``).
ADMIN_ONLY_CHECKS = frozenset({"llm"})
# The level counts give the LLM check's level away, so they are the reviewer's too.
ADMIN_ONLY_REPORT_KEYS = frozenset({"summary"})

STALE_CLONES = (
    "{count} cloned page(s) copied a core file this host has changed since, so the clone may miss "
    "fixes and features the core page has now."
)


def contrast_check(checked: ThemePackage, unmeasured: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``contrast`` entry: the mobile block's pairs measured (M-03), else the scan's ``unmeasured`` one."""
    measured = measure_mobile(checked.manifest, {path: item.data for path, item in checked.files.items()})
    if measured is None:
        return {**unmeasured, "level": LEVEL_NOT_MEASURED}
    return {**measured, "level": CONTRAST_LEVELS[measured["status"]]}


def _reader_checks(checked: ThemePackage, package_bytes: int) -> Dict[str, Dict[str, Any]]:
    """The reader's checks (passed, with what they measured) and the contrast check."""
    base = build_scan_report(checked.files, package_bytes)["checks"]
    checks = {name: {**base[name], "level": LEVEL_PASS} for name in READER_CHECKS}
    return {**checks, "contrast": contrast_check(checked, base["contrast"])}


def clone_check(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``clonedFrom`` entry: every cloned page, and the ones whose core file changed since."""
    pages = manifest.get("pages") if isinstance(manifest.get("pages"), dict) else {}
    clones = sorted(route for route, entry in pages.items() if isinstance(entry, dict) and entry.get("mode") == "clone")
    stale = stale_clones(manifest)
    return {
        "level": LEVEL_WARNING if stale else LEVEL_PASS, "clones": clones, "stale": stale,
        "reason": STALE_CLONES.format(count=len(stale)) if stale else None,
    }


def verdict_of(checks: Mapping[str, Mapping[str, Any]]) -> str:
    """``blocked`` on any error, ``review_required`` on any warning, else ``pass``."""
    levels = {check.get("level") for check in checks.values()}
    if LEVEL_ERROR in levels:
        return VERDICT_BLOCKED
    return VERDICT_REVIEW if LEVEL_WARNING in levels else VERDICT_PASS


def _counts(checks: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    return {level: sum(1 for check in checks.values() if check.get("level") == level) for level in LEVELS}


def run_review_stage(
    checked: ThemePackage, package_bytes: int, *, include_source: bool = True,
    llm_call: Optional[ChunkCall] = None, surface: Optional[SurfaceAreas] = None,
    token_budget: TokenBudget = None,
) -> Dict[str, Any]:
    """The scan report of a checked package, with its verdict; see the module docstring.

    ``include_source``: the LLM stage reads ``source/`` too when shipped (a Marketplace
    submission, whose ``dist/`` the isolated job rebuilds), never instead of ``dist/``; a private
    upload passes False. ``token_budget``: the request's ``review_limits.TokenHold``, opened on
    what theme reviews may still spend today (the routes always pass it, and keep it until the
    row is committed); a bare number is held for the stage alone. ``llm_call`` and ``surface``
    replace the Anthropic call and this host's SDK names (tests).
    Call it from a thread with no running event loop (``asyncio.run``).
    """
    data = {path: item.data for path, item in checked.files.items()}
    llm = run_llm_stage(data, include_source=include_source, call=llm_call, token_budget=token_budget)
    checks = {
        **_reader_checks(checked, package_bytes),
        "uses": uses_check(checked.manifest, data, surface),
        "clonedFrom": clone_check(checked.manifest),
        "llm": llm_check(llm),
    }
    verdict = verdict_of(checks)
    logger.info("[Themes] Review stage for %s@%s: %s (LLM risk %d, %s)", checked.manifest.get("id"),
                checked.manifest.get("version"), verdict, llm.risk_score, "ran" if llm.ran else "did not run")
    return {"schema": REPORT_SCHEMA, "stage": STAGE, "verdict": verdict, "summary": _counts(checks), "checks": checks}


def member_view(report: Any) -> Optional[Dict[str, Any]]:
    """The report as its uploader or submitter sees it: the verdict and the non-LLM checks.

    The LLM check's score, findings and summary stay with the reviewer (``ADMIN_ONLY_CHECKS``),
    as do the level counts (``ADMIN_ONLY_REPORT_KEYS``). A report in an older shape keeps its keys.
    """
    if not isinstance(report, Mapping):
        return None
    view = {key: value for key, value in report.items() if key not in ADMIN_ONLY_REPORT_KEYS}
    checks = report.get("checks")
    if isinstance(checks, Mapping):
        view["checks"] = {name: check for name, check in checks.items() if name not in ADMIN_ONLY_CHECKS}
    return view

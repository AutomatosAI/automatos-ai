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
* ``contrast`` (AA per preset): ``not_measured``. A look is knob values; neither
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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from services.themes.drift import stale_clones
from services.themes.llm_stage import ChunkCall, llm_check, run_llm_stage
from services.themes.package import ThemePackage
from services.themes.scan import build_scan_report
from services.themes.sdk_uses import SurfaceAreas, uses_check

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "automatos-theme-scan/v2"
STAGE = "C-02a"
LEVEL_PASS, LEVEL_WARNING, LEVEL_ERROR, LEVEL_NOT_MEASURED = "pass", "warning", "error", "not_measured"
LEVELS = (LEVEL_PASS, LEVEL_WARNING, LEVEL_ERROR, LEVEL_NOT_MEASURED)
VERDICT_PASS, VERDICT_REVIEW, VERDICT_BLOCKED = "pass", "review_required", "blocked"
READER_CHECKS = ("sizes", "cssScope", "looks")

STALE_CLONES = (
    "{count} cloned page(s) copied a core file this host has changed since, so the clone may miss "
    "fixes and features the core page has now."
)


def _reader_checks(checked: ThemePackage, package_bytes: int) -> Dict[str, Dict[str, Any]]:
    """The reader's checks (passed, with what they measured) and the unmeasured contrast."""
    base = build_scan_report(checked.files, package_bytes)["checks"]
    checks = {name: {**base[name], "level": LEVEL_PASS} for name in READER_CHECKS}
    return {**checks, "contrast": {**base["contrast"], "level": LEVEL_NOT_MEASURED}}


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
    checked: ThemePackage, package_bytes: int, *, prefer_source: bool = True,
    llm_call: Optional[ChunkCall] = None, surface: Optional[SurfaceAreas] = None,
) -> Dict[str, Any]:
    """The scan report of a checked package, with its verdict; see the module docstring.

    ``prefer_source``: read ``source/`` for the LLM stage when shipped (a Marketplace submission,
    whose ``dist/`` the isolated job rebuilds); a private upload passes False. ``llm_call`` and
    ``surface`` replace the Anthropic call and this host's SDK names (tests).
    Call it from a thread with no running event loop (``asyncio.run``).
    """
    data = {path: item.data for path, item in checked.files.items()}
    llm = run_llm_stage(data, prefer_source=prefer_source, call=llm_call)
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

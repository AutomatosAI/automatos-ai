#!/usr/bin/env python3
"""PRD-182 W12-S4 (F092) - coverage ratchet against the MEASURED baseline.

The platform states an 80% coverage doctrine but had zero tooling behind it (no
pytest-cov, no .coveragerc, no vitest coverage), so the number was fiction (OS
review F092). This script installs the honest version of that doctrine: it reads
the total line coverage from ``coverage.xml`` (produced by ``pytest --cov`` via
``.coveragerc``) and enforces a ratchet.

Self-baselining, so it works in CI with no pre-known number:
  * If ``.coverage-baseline`` is absent or holds the ``SEED`` placeholder, this
    run RECORDS the measured percentage as the floor and PASSES. That first CI
    run establishes the real baseline on the code that actually runs.
  * On every later run it ENFORCES ``measured >= floor`` and fails below it.

This is a floor that only moves up: ratchet it toward 80% as coverage improves;
never lower it to silence a drop. It deliberately does NOT assert an aspirational
80% the current suite cannot meet - that would fail-closed on day one and block
every PR, which the wave's guardrails forbid.

Usage (from ``orchestrator/``)::

    pytest --cov --cov-config=.coveragerc --cov-report=xml ...   # writes coverage.xml
    python scripts/check_coverage_baseline.py                    # ratchet
    python scripts/check_coverage_baseline.py --update           # force-record floor to now

Exit 0 = at or above the floor (or seeded). Exit 1 = below the floor, or the
coverage report could not be read.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = ORCH_ROOT / "coverage.xml"
BASELINE_FILE = ORCH_ROOT / ".coverage-baseline"
SEED_TOKEN = "SEED"

# A small tolerance so a sub-percent jitter between runs (test ordering, an
# environment-gated branch) never red-flags a real, unchanged suite. A genuine
# regression is far larger than this.
TOLERANCE_PCT = 0.5


def read_measured_percent() -> float:
    """Total line coverage as a percentage from coverage.xml (Cobertura).

    Reads only the ``line-rate`` attribute of the root ``<coverage>`` element via
    a bounded regex over the file header - no XML entity expansion, so the
    stdlib-XML XXE/billion-laughs class does not apply even though the file is a
    trusted CI artifact.
    """
    if not COVERAGE_XML.exists():
        print(
            f"coverage-ratchet: {COVERAGE_XML} not found. Run pytest with "
            f"`--cov --cov-config=.coveragerc --cov-report=xml` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    # The root element is at the top of the document; read a bounded prefix.
    with COVERAGE_XML.open("r", encoding="utf-8") as fh:
        head = fh.read(4096)
    match = re.search(r"<coverage\b[^>]*\bline-rate=\"([0-9.]+)\"", head)
    if not match:
        print(
            "coverage-ratchet: coverage.xml has no <coverage line-rate=...> "
            "attribute in its header - unexpected format.",
            file=sys.stderr,
        )
        sys.exit(1)
    return round(float(match.group(1)) * 100.0, 2)


def read_floor() -> float | None:
    """The recorded floor percentage, or None if unset/seed/placeholder."""
    if not BASELINE_FILE.exists():
        return None
    raw = BASELINE_FILE.read_text(encoding="utf-8").strip()
    if not raw or raw.upper().startswith(SEED_TOKEN) or raw.startswith("#"):
        # Support a file whose first non-comment line is the number.
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith(SEED_TOKEN):
                return None
            try:
                return float(line)
            except ValueError:
                return None
        return None
    try:
        return float(raw.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def write_floor(percent: float) -> None:
    BASELINE_FILE.write_text(
        "# PRD-182 W12-S4 (F092) coverage floor (percent). MEASURED, not\n"
        "# aspirational. The ratchet fails only when measured coverage drops\n"
        "# below this. Ratchet it UP toward 80% as coverage improves; never\n"
        "# lower it to hide a regression. Replace this whole file with the\n"
        "# token SEED to force a re-measure on the next CI run.\n"
        f"{percent}\n",
        encoding="utf-8",
    )


def main() -> None:
    force_update = "--update" in sys.argv[1:]
    measured = read_measured_percent()
    floor = read_floor()

    if force_update or floor is None:
        write_floor(measured)
        reason = "forced --update" if force_update else "seeded (no prior floor)"
        print(
            f"coverage-ratchet: recorded floor = {measured:.2f}% ({reason}). "
            f"Baseline written to {BASELINE_FILE.name}."
        )
        sys.exit(0)

    print(f"coverage-ratchet: measured {measured:.2f}% vs floor {floor:.2f}%.")
    if measured + TOLERANCE_PCT < floor:
        print(
            f"\n[FAIL] coverage regressed: {measured:.2f}% < floor {floor:.2f}% "
            f"(tolerance {TOLERANCE_PCT}%).\n"
            f"Add tests for the code you changed, or investigate what stopped "
            f"running. Do NOT lower {BASELINE_FILE.name} to pass.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if measured - TOLERANCE_PCT > floor:
        print(
            f"[OK] coverage is {measured - floor:.2f}pp above the floor. "
            f"Ratchet it up: python scripts/check_coverage_baseline.py --update"
        )
    else:
        print("[OK] coverage is at the floor (no regression).")
    sys.exit(0)


if __name__ == "__main__":
    main()

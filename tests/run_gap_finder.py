#!/usr/bin/env python3
"""Weekly Test Coverage Gap Finder

Builds a structured inventory of the API test suite and writes:
  - coverage-gap-summary.json

This is intended to be the single entrypoint for the
"Weekly Test Coverage Gap Finder" recipe.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from run_nightly import _resolve_results_dir
from audit_suite import build_summary


REPORT_DIR = _resolve_results_dir()
SUMMARY_FILE = REPORT_DIR / "coverage-gap-summary.json"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    summary["suite"] = "Weekly Test Coverage Gap Finder"
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    modules = summary.get("modules", [])

    low_coverage_domains = [
        module["domain"]
        for module in modules
        if module.get("test_count", 0) <= 2
    ]
    weak_smoke_only_domains = [
        module["domain"]
        for module in modules
        if module.get("is_smoke_file") and not module.get("is_journey_file")
    ]
    summary["action_items"] = {
        "missing_domains": [
            {
                "domain": domain,
                "priority": "high",
                "suggested_test_file": f"tests/api/test_{domain}.py",
                "reason": "Expected domain has no dedicated API test module.",
            }
            for domain in summary.get("missing_expected_domains", [])
        ],
        "low_coverage_domains": [
            {
                "domain": domain,
                "priority": "medium",
                "reason": "Domain has very few tests and likely needs deeper stateful journeys.",
            }
            for domain in low_coverage_domains
        ],
        "smoke_only_domains": [
            {
                "domain": domain,
                "priority": "medium",
                "reason": "Domain appears to rely mainly on smoke coverage rather than end-to-end journey checks.",
            }
            for domain in weak_smoke_only_domains
        ],
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[gap-finder] Summary written to {SUMMARY_FILE}")
    sys.exit(0)


if __name__ == "__main__":
    main()

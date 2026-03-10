## Test Recipe Runners

This file maps each Automatos testing recipe to its script, outputs, and downstream consumers.

### 1. Nightly Self-Test Suite

- Script:
  - `python3 tests/run_nightly.py`
- Purpose:
  - Full nightly API suite + required regression tests
- Outputs:
  - `test-report.json`
  - `test-summary.json`
- Best consumer:
  - `qa-engineer`
- Notes:
  - Broadest coverage
  - Good for nightly confidence and historical trend comparison

### 2. API Health Check & Regression Detector

- Script:
  - `python3 tests/run_health_regression.py`
- Purpose:
  - Faster, higher-signal subset focused on core health and known regression surfaces
- Outputs:
  - `health-regression-report.json`
  - `health-regression-summary.json`
  - `qa-report.json`
- Best consumers:
  - `qa-engineer`
  - `jira-admin`
  - `bug-fixer`
- Notes:
  - `qa-report.json` is the key handoff artifact
  - Designed for Jira ticket creation and downstream bug fixing

### 3. Weekly Test Coverage Gap Finder

- Script:
  - `python3 tests/run_gap_finder.py`
- Purpose:
  - Inventory the suite, detect missing domains, and suggest new test work
- Outputs:
  - `coverage-gap-summary.json`
- Best consumers:
  - `qa-engineer`
  - `jira-admin`
- Notes:
  - Includes `action_items` for missing/weak/smoke-only areas
  - Best run weekly rather than nightly

## Recommended Agent Flow

1. `qa-engineer` runs `run_health_regression.py`
2. `qa-engineer` writes/exports `qa-report.json`
3. `jira-admin` reads `qa-report.json` and opens/updates Jira issues
4. `bug-fixer` works from Jira ticket evidence (`source_files`, `traceback`, `assertion_message`)

## Output Directory

All scripts resolve outputs using the same logic:

1. `AUTOMATOS_RESULTS_DIR` if set
2. workspace `artifacts/results`
3. fallback in-repo reports directory

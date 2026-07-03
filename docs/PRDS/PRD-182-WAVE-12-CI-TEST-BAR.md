# PRD-182: Wave 12 — CI and Test Enterprise Bar

**Phase:** D — Enterprise hardening (weeks 24–32)
**Branch:** `feat/w12-ci-test-bar` · **Worktree:** `automatos-ai-prd182`
**Dependencies:** Wave 5 (auth decoupling — CI runs local edition) — **merged to main (`557857576`)**
**Build size:** M · **Risk:** Low (but CI-facing — do not break the existing green lanes)
**OS Review refs:** §5, §13 pass/fail "CI coverage ratchet" + test-first note, roadmap Phase D

---

## Overview

The enterprise CI bar is absent: the frontend's 148k lines are ungated (`ignoreBuildErrors:true`), two backend test trees are never collected, required checks run `strict=false` (the stale-merge window that broke main twice), the frontend-to-backend route contract PRD-155 relied on was never built, and there is zero coverage measurement against the stated 80% doctrine. This wave installs the real bar — **measured and baselined, never aspirational.**

**Guiding rule:** additive, baselined lanes. Do **not** try to fix the ~400 pre-existing TS errors or hit 80% coverage in this wave — measure the real floor and fail only on regression below it. Do not break the currently-green jobs.

---

## Ownership boundary (parallel-safe)

Runs concurrently with W10 (PRD-180, observability). They share **zero files** — W12 is CI configs + test collection; W10 is frontend components + the SSE endpoint.

- **W12 OWNS:** `.github/workflows/test.yml` (+ any new workflow), `orchestrator/pytest.ini`, `orchestrator/requirements.txt`, `frontend/next.config.js` (the `ignoreBuildErrors` gate), a new frontend route-contract check, and test-collection config.
- **W12 MUST NOT TOUCH:** `frontend/components/*` and `orchestrator/api/board_tasks.py` (W10 owns). Your new tsc/eslint lane will *validate* W10's code once both merge — that's intended; just baseline the error count so it doesn't block on today's ~400.

---

## Findings & Scope

| Finding | Issue (verified) | Fix |
|---|---|---|
| **F034** | `frontend/next.config.js:12 ignoreBuildErrors:true` + no CI tsc/eslint/vitest lane → 148k lines ungated | Add a frontend CI lane (tsc + eslint + vitest); **baseline** the current TS-error count and fail only on regression |
| **F056** | Required gate collects only `orchestrator/tests` (`orchestrator/pytest.ini`); `orchestrator/modules/*/tests` + `integrations/*/tests` never run | Add those trees to collection; fix any collection-time breakage |
| **F057** | Required checks run `strict=false` — the stale-merge red-main class | Enable branch protection `strict=true` — **repo-admin setting; Gerard applies it.** Document the exact setting + provide the `gh api` command |
| **F044** | Backend emits `orchestrator/reports/route-manifest.json` but nothing on the **frontend** consumes it → "CI fails when the frontend calls a non-existent path" is untrue | Add a frontend→backend route-contract check that fails CI when the frontend calls a path absent from the manifest |
| **F092** | No `pytest-cov`/`.coveragerc`/vitest coverage anywhere vs the stated 80% doctrine | Install `pytest-cov`, measure the **real baseline**, add a ratchet that fail-closes only below the measured floor |

---

## Stories (test-first / CI-verified)

### S1 · Frontend CI lane, baselined (F034) — M
**Files:** `.github/workflows/test.yml` (new frontend job), `frontend/next.config.js`, a baseline file (e.g. `frontend/.tsc-baseline.json` or a script).
**Deliverable:** a CI job that runs `tsc --noEmit`, `eslint`, and `vitest` on the frontend. `vitest` must pass (16 files). `eslint` reports. `tsc` records the current error count as a **baseline** and fails the job only if errors **exceed** the baseline (regression gate) — do **not** flip `ignoreBuildErrors:false` for `next build` (that would break deploy on the ~400 existing). 
**Acceptance:** the job runs on PRs; a synthetic new TS error makes it fail; the existing baseline passes.

### S2 · Collect the orphaned backend test trees (F056) — S
**Files:** `orchestrator/pytest.ini` (testpaths / collection).
**Deliverable:** `orchestrator/modules/*/tests` and `integrations/*/tests` are collected by the required gate. Fix any test that errors purely on collection (import path). 
**Acceptance:** `pytest --collect-only` shows the previously-orphaned trees; the suite runs them in CI. Report the new test count.

### S3 · Frontend→backend route-contract check (F044) — M
**Files:** a new check script (`frontend/scripts/check-route-contract.*` or an `orchestrator/scripts` consumer) + a CI step; consumes `orchestrator/reports/route-manifest.json`.
**Deliverable:** extract the API paths the frontend calls (from the api-client) and assert each exists in the backend route manifest; CI fails on a frontend call to a non-existent path. 
**Acceptance:** `test`/check passes today; a synthetic frontend call to `/api/does-not-exist` fails the check.
**Notes:** reuse the existing `route-manifest.json` + `test_route_manifest.py` pattern — this adds the missing **frontend** half.

### S4 · Coverage ratchet against the real baseline (F092) — M
**Files:** `orchestrator/requirements.txt` (+`pytest-cov`), `.coveragerc` (or `pyproject`/`pytest.ini` cov config), `.github/workflows/test.yml`.
**Deliverable:** install `pytest-cov`, measure the real coverage baseline on the code that actually runs, record it, and add a ratchet step that fail-closes only **below** the measured floor. Do **not** assert an aspirational 80%. 
**Acceptance:** CI reports a coverage number; a drop below the recorded floor fails; the current baseline passes. Report the measured baseline %.

### S5 · Branch protection strict=true (F057) — DOC + owner action
**Deliverable:** this is a **GitHub repo-admin setting**, not code. Document the exact change (required checks + "Require branches to be up to date before merging" = `strict:true`) and provide the ready-to-run command for Gerard:
```
gh api -X PATCH repos/AutomatosAI/automatos-ai/branches/main/protection/required_status_checks -f strict=true
```
Put this in the PRD/PR description. **Do not attempt to apply it** (needs admin; it's Gerard's call).

---

## Verification (NO servers, NO dev-browser)

This wave is CI-facing. Verify locally by running the tools the lanes wrap, not by starting anything:
```
# from frontend/
npx tsc --noEmit | tail   # count errors → baseline
npx vitest run
npx eslint . || true
# backend
cd orchestrator && python -m pytest --collect-only -q | tail   # confirm orphan trees collected
pip install pytest-cov && python -m pytest --cov=orchestrator --cov-report=term-missing -q | tail   # baseline number
```
Do **not** run `next dev`/`next start`/headless browser. Validate workflow YAML with `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/test.yml'))"`.

## Conventions (see automatos-ai/CLAUDE.md)
- Additive, baselined lanes; do not break existing green jobs; no fixing the 400 TS errors or chasing 80% in this wave (measure + ratchet).
- No `os.getenv()` outside `config.py`. Commit to `feat/w12-ci-test-bar` in conventional commits (feat(prd-182): ...). **Do not push or open a PR.**

## Success metrics
- Frontend tsc/eslint/vitest lane runs in CI, baselined; regressions fail.
- The two orphaned test trees are collected and run.
- Frontend route-contract check fails on calls to non-existent backend paths.
- Coverage measured with a ratchet at the real floor.
- `strict=true` documented with the exact command for Gerard to apply.

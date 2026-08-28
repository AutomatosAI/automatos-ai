# Ralph Review Prompt — PRD-228 Fleet State (chain 5/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-228 is complete. Your job: find where the read-model writes, where it N+1s, where "busy" got re-derived instead of reused, where the cost number lies about its source or period, or where workspace scoping leaks another tenant's fleet. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-225-ask-me)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against the 225 branch, NOT main. Read `scripts/ralph/prd-228.json` (binding) + `docs/PRDS/PRD-228-FLEET-STATE.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **Tenant leak (CRITICAL class).** Every query in `fleet_state.py` must be workspace-bound; the route must be permission-guarded like the board list. A cross-workspace fixture that returns another tenant's agent/task/cost row = CRITICAL.
2. **Hidden writes.** Any session mutation reachable from the fleet service or route (including "harmless" lazy-load side effects that flush) = HIGH. The grep test must actually cover the service's imports, not just its own file.
3. **N+1 honesty.** The query-count assertion must scale-test (N agents, count bounded); an assertion that pins an exact number so loosely it passes an N+1 (e.g. `< 100`) = MEDIUM.
4. **Rival derivations.** "Busy/current" must reuse the matcher's derivation; "cost" must come from the pinned canonical source with the pin recorded. A second busy definition or ad-hoc cost aggregation = HIGH ("two derivations WILL drift"). Period must be rolling 24h and labeled in the UI.
5. **Fail-soft truth.** A monkeypatched source failure must omit fields — if it 500s, or silently zeros (a zero that reads as "$0 spent" is a lie; omission is honest) = HIGH.
6. **Route + manifest.** Exactly one new route; manifest count bumped and committed; `test_route_manifest.py` + `check-route-contract.js` green; RouterSpec present if a new router file exists (import-fail-loud).
7. **Frontend discipline.** No new modal (existing details modal reused); exactly one fleet hook, no duplicate/V2 hooks; poll 10s + CustomEvent refetch actually wired (test dispatches the event). `node_modules`/mass-add staging poison = CRITICAL.
8. **Conventions.** ZERO new alembic files; no `os.getenv` outside `config.py`; anomaly thresholds in config.py.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-228-fleet-state --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd228.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) the pinned cost source and its recorded rationale; (b) 226's awareness doctrine can now cite `platform_fleet_status` — a one-line doctrine touch-up rides the NEXT convenient PR, not this one; (c) 229 (chain 6/6) may use fleet context in its answering service.
- **Findings** → append `P228-RVW-1..n` fix stories to `scripts/ralph/prd-228.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-228): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-228-fleet-state` (never force, never another ref). Do not re-run the build.

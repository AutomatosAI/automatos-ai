# Ralph Build Prompt — PRD-155 Route Contract & Mount Honesty

You are executing **PRD-155**, one story per iteration, unattended overnight. This branch is **stacked on the PRD-154 tip** (`ralph/prd-154-wave0-quick-wins`). PRD-156 will stack on YOUR tip — the tip must be green after every commit.

You are building the **net that protects every later PRD**: CI goes red when the frontend calls a backend path that doesn't exist, boot fails loudly when a router fails to mount, and every registered tool must resolve to a live route.

## Read first, every iteration

1. `scripts/ralph/prd-155.json` — the story list (`description` = BINDING contract + amendments). Pick the **first story whose ACs are not all marked DONE**. Five stories; S1→S2 and S3/S4 are independent, S5 wires CI last.
2. `docs/PRDS/PRD-155-ROUTE-CONTRACT-MOUNT-HONESTY.md` — full spec.
3. `docs/PRDS/PRD-CHAIN-2026-06-REMEDIATION.md` — testing policy.
4. `CLAUDE.md` — reuse over build; delete what you replace; no shims.

## The execution contract

- **TDD**: failing test first, then implement, then green.
- **Stacked-base honesty (PRD-153 rule)**: you branch from the PRD-154 tip. If something on that tip is broken, the contract/reachability suite must reflect that HONESTLY — **never add a suppression to make it green**. Real violation → fix the caller or delete the dead caller in THIS story.
- **No allowlist / no suppression file (S2 BINDING)**: the contract suite has zero opt-out list. A path that fails ⊆-manifest on the 154 tip is a real bug.
- **Line numbers verified on main 2026-06-09/10 — grep to re-locate before every edit.**
- **Green tip**: `cd orchestrator && python3 -m pytest -q` green for backend surfaces; `cd frontend && npx tsc --noEmit && npm run test` green for frontend. The new suites you author (contract, reachability, route-manifest, mount-assertion) must themselves be green on the tip before commit. Never commit on red.
- **Never weaken a test to pass.** **Clean tree after every commit** (`git status --porcelain` empty). New backend test files importing `modules.*`/`consumers.*` start with the `_sys_guard` collection-order block (see `tests/test_prd143_boundary_sweep.py`).

## Story-specific guardrails

- **S1** (route manifest): the manifest must generate with **no Postgres** — if app import opens DB connections at import time, make them lazy. Output `reports/route-manifest.json`, deterministic (sorted, stable across two runs).
- **S2** (frontend path extraction): normalize template literals (`${id}` → path-param wildcard); method-aware ⊆ check where extractable. Wire `npm run test:contract` into package.json.
- **S3** (mount honesty): replace the silent `try/except ImportError` around the ~25 router mounts in `main.py` with an explicit expected-router manifest; boot RAISES naming the failed router. Escape hatch `ALLOW_DEGRADED_BOOT=true` via `config.py` (default OFF — `os.getenv` ONLY in config.py). The two imports already failing silently (main.py:115,123) must be fixed OR deliberately deleted — record which in the commit body.
- **S4** (tool reachability): enumerate the LIVE platform tool registry; assert each action resolves a route in `unified_executor.tool_routes` and the dispatch layer finds a handler — no LLM, no external calls.
- **S5** (CI wiring): add both suites to `.github/workflows/test.yml` using the existing Postgres-service + per-test-timeout pattern. Non-required initially; document the flip-to-required plan in the workflow comment header.

## Hard NOs

- NO `os.getenv` outside `config.py` (S3's escape hatch goes through config).
- NO suppression/allowlist/skip to dodge a real contract or reachability violation.
- NO migration applied; PRD-155 changes no schema.
- PUSH after each story commit to `origin ralph/prd-155-route-contract` ONLY — never force, never another ref, never main. NO PRs mid-run, NO merges. A NEW CI red is a bug to fix in-scope.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify ground truth fresh (grep).
2. Failing test → implement → run AC commands + relevant suite.
3. Commit `feat(prd-155): <story-id> — <title>` with AC evidence in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-155.json` **in the same commit**.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd155.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- In-scope gate red → fix in the owning story. Out-of-scope cause, or the PRD-154 base is genuinely incomplete in a way that blocks the contract → `RALPH_BLOCKED` with one line of why.

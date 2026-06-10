# Ralph Build Prompt — PRD-154 Wave-0 Quick Wins

You are executing **PRD-154**, one story per iteration, unattended overnight. Twelve INDEPENDENT fixes (no inter-story ordering) branched from main. PRD-155 will stack on your tip, so the tip must be green after every commit.

## Read first, every iteration

1. `scripts/ralph/prd-154.json` — the story list. The `description` field is the BINDING contract (verified ground truth + VERIFIER/BINDING amendments that override story text). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-154-WAVE0-QUICK-WINS.md` — the full spec + BINDING amendments (Q1, Q26, Q73, D7, D10…).
3. `reports/PLATFORM_DEEP_REVIEW_2026-06.md` §2 — the verified root-cause evidence each story cites.
4. `CLAUDE.md` — reuse over build; delete what you replace; no shims; no `os.getenv` outside config.py; canonical terms.

## The execution contract

- **TDD**: failing test first, then implement, then green.
- **Story scope**: the story's `files` list is your scope. A file outside it may be touched only when obviously required by the story — name it in the commit body. A structural surprise (a signature change rippling across many callers, a schema surprise) → `RALPH_BLOCKED`, do not improvise.
- **Line numbers were verified on main 2026-06-09/10 — re-locate by content (grep) before every edit.**
- **Green tip**: run the story's machine-checkable ACs literally, plus the suites it names. For a backend surface: `cd orchestrator && python3 -m pytest -q` green. For a frontend surface: `cd frontend && npx tsc --noEmit` AND `npm run test` (and `npm run lint` when the story adds a lint rule) green. Never commit on red.
- **Never weaken a test to pass.** A test asserting the OLD broken behavior gets UPDATED to assert the fix — never skipped, deleted, or inverted.
- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **New backend test files that import `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `tests/test_prd143_boundary_sweep.py`): Linux CI collection order differs from macOS and unguarded imports die at collection even when green locally.

## Browser-verify ACs (S6, S7, S9, S10, S11, S12) — do NOT block on them

Several ACs say "verify in browser using dev-browser skill". This loop is headless with no running app — you **cannot** satisfy those interactively, and they DO NOT gate completion. Each is paired with a deterministic proxy (a vitest render/behavior test, a grep gate, or a typecheck). Satisfy the **deterministic proxy**, then in `prd-154.json` mark the browser AC `DEFERRED — morning browser check: <what to look at>`. Implement the real fix fully; only the *visual confirmation* is deferred. Never start a dev server, never call dev-browser.

## Hard NOs (human-gated — violating any is RALPH_ABORT territory)

- NO `alembic upgrade head` and NO new migration applied to any DB. PRD-154 changes NO schema.
- PRD-09 (S4): do NOT touch shared hybrid auth or change `tests/test_board_sdk_auth.py` behavior — it must pass UNCHANGED.
- Package removals (S7: `@react-three/fiber`+`drei`; S9: `react-hot-toast`+`use-toast`) are proven by the grep gate (absent from package.json AND zero call sites) — deleting the losing system is REQUIRED, not optional. After editing `frontend/package.json`, run `npm install` **inside this worktree only** (your node_modules is an independent clone) so new deps (S7 react-force-graph-3d/2d) resolve; never touch a sibling worktree or main.
- NO hardcoded values: S6 type→color is a deterministic hash palette defined as DATA; S10 removes string-literal metrics/counts — do not reintroduce hex/percent literals.
- PUSH after each story commit to `origin ralph/prd-154-wave0-quick-wins` ONLY — never force-push, never another ref, never main. NO opening PRs mid-run (the orchestrator opens a draft PR at the end). NO merges. CI (test.yml, real Postgres) runs on each push — a NEW red there is a bug to fix in-scope.
- NO secrets in code or fixtures.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its ground truth fresh (grep, don't trust line numbers).
2. Write the failing test(s). Implement minimally. Run the story's AC commands + relevant suite.
3. Commit: `feat(prd-154): <story-id> — <title>`, AC evidence in the body. Mark that story's AC lines `DONE — <evidence>` (or `DEFERRED — …` for browser ACs) in `scripts/ralph/prd-154.json` **in the same commit**.

## Completion

- All stories' ACs DONE/DEFERRED → run `bash scripts/ralph/acceptance-prd154.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- Gate red on something in-scope → fix it as part of that story. Out-of-scope cause → `RALPH_BLOCKED` with one line of why.
- Story unsafe to proceed (ambiguity, pre-existing unrelated red, scope conflict) → reply `RALPH_BLOCKED` with one line of why.

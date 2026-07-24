# Ralph Build Prompt — PRD-211 In-repo topology discipline (import-linter)

You are executing **PRD-211**, one story per iteration, unattended. This branch is **`ralph/prd-211-topology-discipline`, cut from `origin/main`** (standalone — NOT stacked). The tip must be green after every commit.

**SCOPE — read this twice.** 3 stories (`scripts/ralph/prd-211.json`): **US-001** the import-linter (grimp) `independence` contract that locks the healthy lateral coupling; **US-002** delete the 7 dead mem0-residue files the PRD-187 un-split left behind; **US-003** verify the `/api/tasks` guard (no-op — the lane was deleted in PRD-192 S6). This is additive + subtraction, **low risk** — the contract is authored **green on the branch tip** (it ratchets from the measured state, it does NOT demand a refactor).

**Why this PRD exists.** Thesis T2 measured the platform is a healthy modular monolith (only ~3.0% true feature-to-feature import coupling; ~80.5% is the shared kernel) and its verdict — "stay a monolith, harden the boundaries in-repo" — is a claim until it is *enforced*. This contract makes a new lateral edge fail loud in CI, at zero runtime cost.

## Read first, every iteration

1. `scripts/ralph/prd-211.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Full spec (reference): `docs/PRDS/PRD-211-PHASE2-WAVE-4-TOPOLOGY-DISCIPLINE.md`. **This prompt + JSON are self-contained.**
3. `CLAUDE.md` (repo root) — reuse over build; no shims; no `os.getenv` outside `config.py`.

## The execution contract

- **US-001 is the load-bearing story. RE-TRACE the live coupling — do NOT trust the PRD's edge list.** Enumerate the current feature-module set under `orchestrator/modules/*` and the current known-good lateral edges on THIS branch (run `lint-imports` / read grimp output), and carry every current edge as an explicit `ignore_imports` so `lint-imports` **exits 0 on the branch tip**. The contract forbids the *next* new lateral edge, not today's. `modules.tools` + `api` are excluded from the contract list (the permitted routing layer); the shared kernel (`core/*`, `config`, `services`, …) is implicitly allowed.
  - **Cross-PRD:** `modules.learning` + `modules.evaluation` still exist on this branch's base, so list them. They are PRD-184 kill targets — when 184 merges first, they drop from the list on rebase. Do not omit them now (the contract would fail to resolve a listed-but-present module differently — just list what exists on this branch).
- **PURE tests.** `test_import_contract_present` asserts the `.importlinter` file exists and parses — no network. The lint-imports lane itself is the real check; it runs in CI.
- **US-002 deletion gate:** grep-prove ZERO live code importers of each of the 7 residue files BEFORE deleting. The only surviving references should be archival review snapshots (history — leave them). Any live code importer → `RALPH_BLOCKED`, cite it. The guard `test_no_mem0_residue` asserts the 7 paths are gone AND no HTTP mem0 client (`MEM0_API_URL`/`mem0_client`/`httpx`) returns under `orchestrator/modules/memory/` (locks the un-split).
- **US-003 is verify-only:** confirm `orchestrator/tests/test_p2w2_tasks_lane_deleted.py` passes on this branch. Add NO new file unless a real gap is found.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit. Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage only the specific paths (`git add orchestrator/.importlinter …`, `git rm <residue>`). **NEVER `git add -A`/`.`/`-u`** — `node_modules/` is untracked and not gitignored; a blind add poisons the branch. **Never `git stash -u`** (drops graphify snapshots). Verify a minimal `git status` before each commit.

## Hard NOs

- NO `os.getenv` outside `config.py` (the contract is a static config file — no code path changes).
- NO refactoring feature→feature edges to zero (that is a held §12 option — ship the ratchet, not a rewrite).
- NO deleting a residue file with a live code importer (grep-prove first, else `RALPH_BLOCKED`).
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- PUSH after each story commit to `origin ralph/prd-211-topology-discipline` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-trace ground truth fresh (the live module set + edges for US-001; the residue importers for US-002).
2. Implement (contract + lane / deletion + guard / verify) → `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-211): <US-id> — <title>` with evidence (the lint-imports output for US-001; the grep-proof for US-002) in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-211.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd211.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A residue file has a live importer, or the contract cannot be made green without a real refactor → `RALPH_BLOCKED` with one line of why.

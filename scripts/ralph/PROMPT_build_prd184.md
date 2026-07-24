# Ralph Build Prompt — PRD-184 Kill-list (DELETE-NOW tier only)

You are executing **PRD-184**, one story per iteration, unattended. This branch is **`ralph/prd-184-kill-list-dead-surface`, cut from `origin/main`** (standalone — NOT stacked). The tip must be green after every commit.

**SCOPE — read this twice.** This run builds **only the 6 DELETE-NOW stories** (US-001..US-006 in `scripts/ralph/prd-184.json`): grep-proven zero-caller dead/fabricating surface. **OUT OF SCOPE — do NOT build, do NOT simulate, do NOT add as acceptance:** the `/execute` wire-or-delete (open decision), the KG prove-then-cut (S8), the legacy-workflow-engine retire (S9 — needs `jira_bug_triage` migrated first), the PlaybookMiner + `/chat/[id]` retire (S10 — live callers), and the 4 decide-then-cut items. **mem0 residue is PRD-211's, lockfiles are PRD-209's — do not touch them here.** If you find yourself migrating a caller, dropping a table, or unmounting a live router, stop — that is a held retire-tier item, out of scope by design.

**Why this PRD exists (the binding context).** Decoy and fabricating surface misleads the humans *and the agents* that read this code — the next agent treats a dead module as real and routes into it, degrading output. This PRD deletes grep-proven-dead surface so the codebase stops lying. Waves 0–3 already cut several §5 items (audit table in the spec) — you build only the REMAINING delete-now set.

## THE DELETION GATE (non-negotiable — this is why a deletion run is safe)

**Never delete a symbol/file until you have grep-proven it has ZERO live callers on this branch.** For each target:
1. `git grep -n "<symbol/module>"` across `orchestrator/` and `frontend/` — enumerate every reference.
2. Distinguish live imports/calls from comment-only "do not resurrect" refs and the file's own definition.
3. If the only references are the owning barrel `__init__` (trim it in the same commit) → safe to delete.
4. **If ANY live caller exists → do NOT delete. Reply `RALPH_BLOCKED` naming the caller.** Never force-delete a referenced symbol; never flatten a live caller to make room.

## Read first, every iteration

1. `scripts/ralph/prd-184.json` — the story list. A story's `description` + `acceptanceCriteria` = the BINDING contract. Pick the **first story whose ACs are not all marked `DONE`**.
2. Full spec (reference): `docs/PRDS/PRD-184-KILL-LIST-DEAD-SURFACE-REMOVAL.md` (§5 audit — DONE rows are already cut; build only the 6 REMAINING delete-now stories). **This prompt + the JSON are self-contained** — build from them even if the spec file is absent on this branch.
3. `CLAUDE.md` (repo root) — delete what you replace; **no shims**; reuse over build.

## ⚠️ Verify every path/line by grep — anchors in the JSON are 2026-07-23, they drift.

## The execution contract

- **Deletion + guard, together.** Every deletion ships a **source-grep guard test** in the SAME commit (e.g. `test_no_learning_evaluation_imports`) asserting the symbol/file cannot silently return. This is the durable fix (the recurring guard-drift lesson).
- **PURE tests only.** Source-grep / filesystem guards + any behavioural test mock at the boundary. **No DB, no network, no server boot.** New test files importing `modules.*`/`core.*` start with the `_sys_guard` collection-order block (copy from a neighbouring test).
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit (the full suite is the protected-regression gate — a deletion that breaks an import goes red, and that is a real finding to fix in-scope). Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage ONLY the specific paths you changed: `git rm <path>` for deletions, `git add <specific-file>` for edits/new tests. **NEVER `git add -A` / `git add .` / `git add -u`** — `node_modules/` is untracked and **not** gitignored; a blind add stages hundreds of MB and poisons the branch. **Never `git stash -u`** (it drops graphify snapshots). Verify a clean, minimal `git status` before every commit.
- **No `os.getenv` outside `config.py`.** No backward-compat shim — delete, don't `_legacy`. Trim the owning barrel `__init__` in the same commit as the deletion.

## Story-specific guardrails

- **US-001** learning/evaluation packages — delete both EXCEPT `learning/playbooks/miner.py` (held S10). Trim `modules/__init__.py`. `api_playbooks.py` is held (S10) — leave it.
- **US-002** llm-core scaffolding — the 6 files survive only via the `core/llm/__init__.py` barrel; trim it. **This story owns `api/anthropic_client.py`** (PRD-212 defers to it). Re-verify anthropic_client has zero importers (broad grep hits were the SDK var name).
- **US-003** exec_planning — delete + de-route `unified_executor.py` (import + dispatch branches); grep-prove no agent toolset uses the 8 tool names first.
- **US-004** — `concurrency.py` is the definitive delete (zero callers). ToolService + the `composio_tool_router` dead delegate are **conditional**: delete ONLY if grep-proven dead; else leave and note. The `composio_tool_router.py` file itself is LIVE — keep it.
- **US-005** — the 7 `*_adapter.py` + `_ping_platform_legacy`; grep-prove the active channel path doesn't import them.
- **US-006** — frontend `/api-control` + `/styleguide` + the `workspaceMeta` pill. **Do not touch lockfiles (PRD-209) or `/chat/[id]` (held S10).**

## Hard NOs

- NO deleting a symbol with a live caller (grep-prove zero first, else `RALPH_BLOCKED`).
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- NO building the held retire/decide items (S7–S10, decide-then-cut); NO touching mem0 residue (PRD-211) or lockfiles (PRD-209).
- NO weakening/skipping a test to go green; NO `os.getenv` outside `config.py`; NO shim.
- PUSH after each story commit to `origin ralph/prd-184-kill-list-dead-surface` ONLY. NO PRs mid-run, NO merges. A NEW CI red is a bug to fix in-scope.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; grep-prove the targets are zero-caller fresh (never trust the JSON's paths blind).
2. Write the source-grep guard (fails before) → delete the target(s) + trim the barrel → run `cd orchestrator && python3 -m pytest -q` (and the frontend lane for US-006).
3. Commit `feat(prd-184): <US-id> — <title>` with grep-proof evidence in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-184.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd184.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A target turns out to have a live caller, or an in-scope gate is red for an out-of-scope reason → reply `RALPH_BLOCKED` with one line of why (the caller, cited).

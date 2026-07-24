# Ralph Review Prompt — PRD-184 Kill-list dead-surface removal (DELETE-NOW tier)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-184 is complete. Your job is to find where it over-deleted, flattened a live caller, half-guarded a deletion, or reached into surface this run must not touch. You fix NOTHING yourself. The bar: a "green" PRD where a deleted symbol still has a live import, or a deletion ships without a source-grep guard so the dead surface can silently regrow, or the diff quietly cut a HELD retire-tier / other-PRD file — is a finding.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-184.json` (`description` + `acceptanceCriteria` = binding contract). Full spec (reference only; the JSON + this prompt are self-contained): `docs/PRDS/PRD-184-KILL-LIST-DEAD-SURFACE-REMOVAL.md` (§5 audit table).

**Scope guard (important):** this run is the **DELETE-NOW tier ONLY** — the 6 stories US-001..US-006. The retire-tier (S9 workflow-engine, S10 PlaybookMiner + `/chat/[id]`), `/execute` (S7), the KG prove-then-cut (S8), and the 4 decide-then-cut items are **HELD — Gerard's, out of scope**. mem0 residue is PRD-211's; lockfiles are PRD-209's. If the diff touches any of those, that is a finding (over-reach on an unattended deletion run).

## Hunt list — every item is a confirmed-risk class

1. **Over-deletion / dangling import (CRITICAL class).** For every symbol/file the diff DELETES, grep the whole tree (`git grep -n "<symbol>"` across `orchestrator/` and `frontend/`) for a surviving **live** importer/caller. A deleted symbol that still has a live caller — a now-dangling `import`, an unresolved name, a de-routed dispatch that another path still calls — is a CRITICAL. The barrel `__init__` trim must remove exactly the re-exports of the deleted names and no live one. Confirm each story grep-proved zero callers BEFORE cutting (evidence in the commit body); if a real caller exists, the story should have been `RALPH_BLOCKED`, not shipped.
2. **Held-surface touched (finding each).** The diff must **NOT** include any of: `orchestrator/modules/learning/playbooks/miner.py`, `orchestrator/api/api_playbooks.py`, `orchestrator/api/workflows.py` (or `workflow_templates.py`), `frontend/app/chat/[id]`, any lockfile (`package-lock.json` / `pnpm-lock.yaml` / `yarn.lock`), any mem0 residue (`mem0_openapi*`, `probe_mem0_endpoints*`, `seed_mem0_user*`), or any `alembic/versions/` migration. **PRD-184 adds no migration** — a migration file in the diff is a finding. Each such path is a separate finding.
3. **Missing / weakened source-grep guard.** Each of US-001..US-006 must ship its named guard test in the SAME commit as its deletion (`test_no_learning_evaluation_imports`, `test_llm_core_no_dead_scaffolding`, `test_exec_planning_deleted_and_unrouted`, `test_no_tools_concurrency_import`, `test_no_legacy_channel_adapters`, `test_no_placebo_routes`). A guard that is absent, or asserts something weaker than "this symbol/route cannot return" (e.g. greps a comment, not a live import), or was repointed to always pass — is a finding.
4. **Staging over-reach — the `git add -A` landmine.** Scan the diff `--stat` for `node_modules/`, vendored bundles, or any unrelated mass-add that a blind `git add -A`/`.`/`-u` would have swept in (node_modules is untracked and NOT gitignored). Any such path is a finding.
5. **`os.getenv` outside `config.py`.** A deletion run should add none. Any `+ ...os.getenv(...)` line outside `orchestrator/config.py` in the diff is a finding.
6. **Suite green / no new CI red.** The full orchestrator suite must be green on the tip (a deletion that broke a live import goes red — that is a real in-scope regression, not something to skip past). No protected test weakened or skipped to go green.
7. **US-004 conditional discipline.** `concurrency.py` is the definitive delete; `ToolService` + the `composio_tool_router` dead-delegate are conditional — deleted ONLY if grep-proven dead, else left untouched with a note. The `composio_tool_router.py` FILE must still exist (it is live). A flattened live caller here is a CRITICAL.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-184-kill-list-dead-surface --workflow test.yml --limit 1`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing red).
- Run `bash scripts/ralph/acceptance-prd184.sh`. **Non-zero = automatic CRITICAL** (it re-checks the deletions, the guards, and the scope guard).

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly that the retire tier (S9 workflow-engine, S10 miner + `/chat/[id]`), `/execute` (S7), the KG cut (S8) and the 4 decide-then-cut items remain **Gerard's to decide** — intentionally NOT in this run.
- **Findings** → append `P184-RVW-1..n` fix stories to `scripts/ralph/prd-184.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-184): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-184-kill-list-dead-surface` (never force, never another ref). Do not re-run the build.

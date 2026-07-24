# Ralph Review Prompt — PRD-211 In-repo topology discipline (import-linter)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-211 is complete. Your job is to find where the contract is toothless, where a residue file was cut despite a live importer, where the un-split lock is missing, or where a "contract" quietly became a feature-module rewrite. You fix NOTHING yourself. The bar: a "green" PRD where `lint-imports` passes today but would ALSO pass on a brand-new `modules.rag`→`modules.nl2sql` edge (a ratchet that does not bite), or where a deleted mem0 file still had a live code importer, or where the diff refactored feature modules instead of just adding a contract — is a finding.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-211.json` (`description` + `acceptanceCriteria` = binding contract). Full spec (reference only; the JSON + this prompt are self-contained): `docs/PRDS/PRD-211-PHASE2-WAVE-4-TOPOLOGY-DISCIPLINE.md`.

**Scope guard (important):** this is a 3-story run — US-001 the import-linter `independence` contract, US-002 the 7 mem0-residue deletions, US-003 a verify-only no-op. It is a **contract + a subtraction, NOT a refactor**. If the diff rewrites feature-module import edges to zero, or adds/changes a migration, that is a finding (the refactor is a held §12 option — ship the ratchet, not a rewrite).

## Hunt list — every item is a confirmed-risk class

1. **The ratchet must BITE (CRITICAL class — the whole point of the PRD).** Prove it by exercising, not by reading:
   - With the tree clean, `cd orchestrator && lint-imports` exits **0** (green baseline on the tip via `ignore_imports` of the re-traced current edges).
   - Then inject ONE new lateral edge — add `from modules import nl2sql` (or `import modules.nl2sql`) to any file under `orchestrator/modules/rag/` — and re-run `cd orchestrator && lint-imports`. It **MUST exit non-zero** naming the forbidden `modules.rag → modules.nl2sql` edge. `git checkout -- <file>` to discard it (the violating edge must **NOT** be committed).
   - If the contract stays green on that injected edge, the `independence` contract is toothless — CRITICAL. Also confirm `modules.tools` + `api` are the ONLY excluded routing layers, and that `ignore_imports` carries only PRE-EXISTING edges (the injected `rag→nl2sql` must not be pre-listed).
2. **Residue deletions were genuinely dead.** For each of the 7 files (`orchestrator/mem0_openapi.json`, `orchestrator/scripts/probe_mem0_endpoints.py`, `orchestrator/scripts/seed_mem0_user.py`, `scripts/test_mem0_railway.py`, `docs/PRDS/39-MEM0-MIGRATION-PRD.md`, `docs/PRDS/PRD-152-MEM0-INTERNAL-SERVICES-DECOUPLING.md`, `docs/memory-system/phase1-mem0-async-rollback.md`), grep the tree for a surviving **live code importer** (`git grep -n`). An archival review-snapshot / doc citation of mem0 is history — leave it, not a finding. A live code `import` of a deleted file is a CRITICAL (should have been `RALPH_BLOCKED`).
3. **Un-split lock present.** `test_no_mem0_residue` must assert BOTH the 7 paths are gone AND that no HTTP mem0 client (`MEM0_API_URL` / `mem0_client` / `httpx` mem0 client) returns under `orchestrator/modules/memory/` — the live path is in-process Qdrant (`durable_store.py`). A guard that only checks the 7 paths and skips the un-split lock is a finding.
4. **Contract, not a rewrite (+ no stray staging).** The diff should be: `orchestrator/.importlinter` (new), a pin in `orchestrator/requirements.txt`, `.github/workflows/import-linter.yml` (new, non-required lane), the 7 deletions, and pure guard tests. Any feature-module **code** edit that rewrites import edges, any `node_modules/`/vendored mass-add (the `git add -A` landmine — node_modules is untracked and NOT gitignored), or any `alembic/versions/` migration (**PRD-211 adds none**) is a finding. No `os.getenv` added outside `config.py`.
5. **US-003 stayed a no-op.** `orchestrator/tests/test_p2w2_tasks_lane_deleted.py` must pass on the tip; no new file should be added unless a genuine coverage gap was found and noted.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-211-topology-discipline --workflow import-linter.yml --limit 1` (and `--workflow test.yml`): a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd211.sh`. **Non-zero = automatic CRITICAL** (it re-checks the contract parses, the residue is gone + guarded, the un-split lock, and the /api/tasks guard).

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly that (a) `modules.learning` + `modules.evaluation` are listed in the contract on THIS branch and **drop out on rebase once PRD-184 merges first** (a contract cannot reference a deleted package), and (b) flipping the `import-linter` lane from non-required to **required** in branch protection is **PRD-210's / Gerard's call** — this PRD only makes the edge fail loud.
- **Findings** → append `P211-RVW-1..n` fix stories to `scripts/ralph/prd-211.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-211): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-211-topology-discipline` (never force, never another ref). Do not re-run the build.

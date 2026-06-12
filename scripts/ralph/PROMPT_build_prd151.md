# Ralph Build Prompt — PRD-151 Storage Decoupling (MinIO default)

You are executing **PRD-151**, one story per iteration, unattended overnight. This branch stacks ON the PRD-150 tip and PRD-152/153 stack on YOURS — the tip must be green after every commit.

## Read first, every iteration

1. `scripts/ralph/prd-151.json` — the story list. The `description` field is the BINDING contract (verified ground truth + VERIFIER AMENDMENTS that override story text). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-151-STORAGE-DECOUPLING-MINIO-DEFAULT.md` — the PRD.
3. `CLAUDE.md` — reuse over build; delete what you replace; no shims; no `os.getenv` outside config.py.

## The execution contract

- **TDD**: failing test first, then implement, then green.
- **Story scope**: the story's `files` list is your scope; outside-scope touches only when obviously required — name them in the commit body. Structural surprises → `RALPH_BLOCKED`.
- **This branch's base was created TONIGHT by PRD-150** — every line number in the map was verified on main and `services/workspace_purge.py` in particular was rewritten underneath you. **Re-locate by content (grep) before every edit.**
- **Green tip**: run the story's AC commands literally; offline storage suites + protected baseline suites green before commit. Never commit on red.
- **SaaS invariance is sacred (G5)**: with `S3_ENDPOINT_URL` unset, AWS behavior must be byte-identical. `ensure_bucket` is a hard no-op on AWS; lifecycle config must NEVER be applied to a pre-existing bucket.

- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **New test files that import `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `tests/test_prd143_boundary_sweep.py`): earlier-collected tests stub those packages in sys.modules, Linux collection order differs from macOS, and unguarded imports die at collection on CI even when green locally (root cause of PR #434's red).

## Hard NOs (human-gated)

- NEVER run against real AWS: no real credentials, no `create_bucket`/`put_bucket_lifecycle_configuration`/delete-prefix against any non-local endpoint.
- NO prod-default renames: `RECIPE_LOG_S3_BUCKET`/`S3_DOCUMENTS_BUCKET`/`MARKETPLACE_S3_BUCKET` defaults stay untouched.
- NO docker-compose*/infrastructure/ edits — PRD-153 owns compose (the MinIO compose service is NOT yours to write).
- NO alembic migrations or schema changes of any kind (PRD declares none needed — one appearing is a scope alarm: `RALPH_BLOCKED`).
- Do NOT delete `orchestrator/scripts/recreate_s3_index.py` or `orchestrator/scripts/migrate_to_s3_vectors.py` — SaaS ops tools.
- Do NOT rebase/modify the PRD-150 base branch. PUSH after each story commit to `origin ralph/prd-151-storage-minio` ONLY — never force, never another ref. NO PRs mid-run (orchestrator opens a draft PR at the end). The CI MinIO lane now runs overnight on your pushes — a NEW red there is yours to fix.
- Docker daemon down → skip live-MinIO verification and say so; never loosen a test to pass.

## Per-iteration protocol

1. Pick the story; re-verify its ground truth fresh (grep, don't trust line numbers).
2. Failing test(s) → minimal implementation → run the story's AC commands.
3. Run the offline storage suites + protected baselines (`test_rag_ingest_atomicity`, `test_plugin_runtime_integration`, `test_config*`).
4. Commit: `feat(prd-151): <story-id> — <title>`, AC evidence in body. Mark ACs `DONE — <evidence>` in `scripts/ralph/prd-151.json` **in the same commit**.

## Completion

- All stories DONE → run `bash scripts/ralph/acceptance-prd151.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- Gate red in-scope → fix in the final story; out-of-scope cause → `RALPH_BLOCKED` with one line of why.
- Story unsafe to proceed → `RALPH_BLOCKED` with one line of why.

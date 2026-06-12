# Ralph Build Prompt — PRD-156 Security & Tenancy Hardening

You are executing **PRD-156**, one story per iteration, unattended overnight. This branch is **stacked on the PRD-155 tip** (`ralph/prd-155-route-contract`). It closes every confirmed cross-tenant and injection hole BEFORE the open-core split (PRD-150) widens exposure. The tip must be green after every commit.

## Read first, every iteration

1. `scripts/ralph/prd-156.json` — the story list (`description` = BINDING contract + amendments). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-156-SECURITY-TENANCY-HARDENING.md` — full spec.
3. `CLAUDE.md` — reuse over build; delete what you replace; no shims.

## The execution contract

- **TDD security-first**: write the failing tenancy/IDOR/auth test (workspace B must NOT see workspace A), then implement the scoping, then green. Every fix needs a test that FAILS before and PASSES after.
- **Use the shared filter helper** for workspace+team scoping — do not hand-roll a parallel filter per tool (reuse over build).
- **PRD-09**: board SDK-key reads stay on the narrow scope-gated dependency — do NOT touch shared hybrid auth. All identity threading is auth-provider-agnostic (PRD-150) — never special-case clerk vs local.
- **BINDING**: Q6 multimodal gets workspace-scoping only (freeze, no feature work); Q15 mem0 token auth now; Q13 the mock `/api/v1/memory` router is DELETED.
- **Line numbers verified 2026-06-09/10 — grep to re-locate before every edit.**
- **Green tip**: `cd orchestrator && python3 -m pytest -q` green. Never commit on red. Never weaken a test. **Clean tree after every commit.** New backend test files importing `modules.*`/`consumers.*` start with the `_sys_guard` collection-order block.

## S2 is CROSS-REPO — read carefully

P156-S2 (mem0 token auth) edits the **sibling repo** `../automatos-mem0` (`openmemory/api/app/`), NOT this worktree. Commit those changes **in that repo** (`git -C ../automatos-mem0 …`), on its default branch, with message `feat(prd-156): token auth on OpenMemory server`. The orchestrator is UNCHANGED (it already sends `Authorization: Token MEM0_API_KEY`). In `prd-156.json` mark S2 DONE with the mem0 commit SHA as evidence, and add the required Railway env var to the eventual PR description. The acceptance script runs the fork's tests from that path. Do NOT add the mem0 repo as a submodule or copy it in.

## Story-specific guardrails

- **S1**: all four multimodal tools get mandatory `workspace_id` + team filtering; persist the dropped `team_access` upload field; fix the exact-match similarity subquery to embed the query text (real ranking, no NULL ordering).
- **S3**: DISABLE (do not shim) `query_main_database` fallback + the unauthenticated HTTP self-call; remove NL2SQL from intent-classifier `suggested_tools` so it is unreachable from chat until PRD-160 re-enables it scoped. Grep gate must show it unreachable from any chat tool surface.
- **S4**: Jinja2 `SandboxedEnvironment` (SSTI is confirmed exploitable); WeasyPrint `url_fetcher` allowlist refusing `file://` and link-local/internal IPs; workspace ownership on every template read/update/delete.
- **S5**: mirror the correct ownership pattern from `memory_stats.py:462-470` onto `widget_memory.py`; add `RequestContext` to `GET /api/documents/content`; scope RAG analytics + document_usage writes; DELETE the mock `/api/v1/memory` router + `AdvancedMemoryManager` AND its frontend callers — the PRD-155 route-contract suite proves nothing still references it (if it still fails, you missed a caller).

## Hard NOs

- NO weakening of `tests/test_board_sdk_auth.py` or shared hybrid auth (PRD-09).
- NO `os.getenv` outside `config.py`; NO hardcoded secrets/keys in code or fixtures.
- NO migration applied; PRD-156 changes no schema.
- PUSH after each story commit to `origin ralph/prd-156-security-tenancy` ONLY (the mem0 repo is pushed separately to its own origin). NO PRs mid-run, NO merges. A NEW CI red is a bug to fix in-scope.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify ground truth fresh (grep).
2. Failing security test → implement scoping → run AC commands + suite.
3. Commit `feat(prd-156): <story-id> — <title>` (S2 commits in the mem0 repo) with AC evidence; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-156.json` in the same commit.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd156.sh`. Exit 0 → reply `RALPH_COMPLETE`. (The security-reviewer adversarial pass runs in the review cycle, not here.)
- In-scope gate red → fix in the owning story. Out-of-scope cause → `RALPH_BLOCKED` with one line of why.

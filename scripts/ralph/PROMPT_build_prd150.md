# Ralph Build Prompt — PRD-150 Auth Decoupling (Open Core)

You are executing **PRD-150**, one story per iteration, unattended overnight. This branch is the FOUNDATION of tonight's stacked chain (151/152/153 cut from your tip) — the tip must be green after every commit.

## Read first, every iteration

1. `scripts/ralph/prd-150.json` — the story list. The `description` field is the BINDING contract (verified ground truth + VERIFIER AMENDMENTS that override story text). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-150-AUTH-DECOUPLING-OPEN-CORE.md` — the PRD.
3. `CLAUDE.md` — reuse over build; delete what you replace; no shims; no `os.getenv` outside config.py.
4. The PRD-09 security comment at `orchestrator/core/auth/hybrid.py` (~514-524) — before touching anything in core/auth.

## The execution contract

- **TDD**: failing test first, then implement, then green.
- **Story scope**: the story's `files` list is your scope. A file outside it may be touched only when obviously required by the story — name it in the commit body. A structural surprise (signature changes rippling into the ~108 RequestContext consumers, schema surprises) → `RALPH_BLOCKED`, do not improvise.
- **Line numbers were verified on main 2026-06-09 — re-locate by content (grep) before every edit.**
- **Green tip**: run the story's ACs literally (each is a command), plus the targeted suites it names. Full orchestrator suite + `npx tsc --noEmit` must be green before any commit that touches their surface. Never commit on red.
- **Never weaken a test to pass.** Tests pinned to clerk behavior get pinned to `AUTH_PROVIDER=clerk` or a fake provider — never deleted or inverted.

- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **New test files that import `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `tests/test_prd143_boundary_sweep.py`): earlier-collected tests stub those packages in sys.modules, Linux collection order differs from macOS, and unguarded imports die at collection on CI even when green locally (root cause of PR #434's red).

## Hard NOs (human-gated — violating any one is RALPH_ABORT territory)

- NO migration applies beyond the new `prd150_auth_provider` revision **on the local dev DB only**; never `alembic upgrade head` (a pending Wave-5 DROP migration is Gerard's to apply).
- NO ak_* SDK-key acceptance in `get_request_context_hybrid` — ak_* stays exclusively in `require_task_context` (PRD-09). `tests/test_board_sdk_auth.py` must pass UNCHANGED.
- LocalAuthProvider `system_role` is `"admin"`, NEVER `"super_admin"` (PRD-143 obs tier).
- NO docker-compose*/infrastructure/ edits (PRD-153 owns compose).
- PUSH after each story commit to `origin ralph/prd-150-auth-decoupling` ONLY — never force-push, never any other ref, never main. NO opening PRs mid-run (the orchestrator opens a draft PR at the end). NO merges. CI (test.yml) runs on each push with a real Postgres — treat a NEW red there as a bug to fix in-scope.
- NO Clerk dashboard/credential operations; NO secrets in code or test fixtures.

## Per-iteration protocol

1. Pick the story; re-verify its ground truth fresh (grep, don't trust line numbers).
2. Write the failing test(s). Implement minimally. Run the story's AC commands.
3. Run the relevant suites (orchestrator full suite if backend surface; frontend tsc+vitest if frontend surface).
4. Commit: `feat(prd-150): <story-id> — <title>`, AC evidence in the body. Mark the story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-150.json` **in the same commit**.

## Completion

- All stories DONE → run `bash scripts/ralph/acceptance-prd150.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- Gate red on something in-scope → fix it as part of the final story. Out-of-scope cause → `RALPH_BLOCKED` with one line of why.
- Story unsafe to proceed (ambiguity, pre-existing red, scope conflict) → reply `RALPH_BLOCKED` with one line of why.

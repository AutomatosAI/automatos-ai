# Ralph Build Prompt — PRD-152 mem0 / Internal-Services Decoupling

You are executing **PRD-152**, one story per iteration, unattended overnight. This branch stacks ON the PRD-151 tip and PRD-153 stacks on YOURS — the tip must be green after every commit.

## Read first, every iteration

1. `scripts/ralph/prd-152.json` — the story list. The `description` field is the BINDING contract (verified ground truth + VERIFIER AMENDMENTS that override story text). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-152-MEM0-INTERNAL-SERVICES-DECOUPLING.md` — the PRD.
3. `CLAUDE.md` — reuse over build; delete what you replace; no `os.getenv` outside config.py (the three `services/*/automatos_logging.py` standalone copies are the ONLY allowed exception — no config.py exists in those containers).

## The execution contract

- **TDD**: failing test first, then implement, then green.
- **Every default flip keeps the `os.getenv(NAME, default)` env-override form** — Railway/SaaS must still win via env (G5). Never remove an override path or rename an env var.
- **Empty-URL guards must not weaken connected-mode behavior**: when the URL IS configured, error paths stay byte-equivalent. No blanket try/except.
- **`mem0_client.py` request/retry/breaker logic is untouched** — only tests around it may grow. Any MemoryProvider-style interface change is an explicit non-goal: reject on sight, `RALPH_BLOCKED` if a story seems to need it.
- **Line numbers were verified on main; config.py has drifted after PRD-150/151 landed tonight — re-locate by content.**
- Tests asserting old railway defaults get UPDATED to assert local defaults — never deleted.

- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **New test files that import `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `tests/test_prd143_boundary_sweep.py`): earlier-collected tests stub those packages in sys.modules, Linux collection order differs from macOS, and unguarded imports die at collection on CI even when green locally (root cause of PR #434's red).

## Hard NOs (human-gated)

- NO docker-compose*.yml edits or deletions (root or infrastructure/) — PRD-153's fold. The acceptance gate diffs compose against the branch base; a committed edit still fails.
- NO `.env` / `.env.example` / `*.env.*` edits (block-secrets hook territory); `envs/*.defaults` comment-truthing in S8 is the only env-file-adjacent edit allowed.
- NO writes to the sibling `../automatos-mem0` repo — read-only reference for the S6 doc.
- NO database migrations of any kind — one appearing is a scope alarm: `RALPH_BLOCKED`.
- NO GHCR/registry publishing (S6 documents the decision input only). NO Railway env changes. PUSH after each story commit to `origin ralph/prd-152-mem0-decoupling` ONLY — never force, never another ref. NO PRs mid-run (orchestrator opens the draft PR).
- `MEM0_API_KEY`/`OPENAI_API_KEY` must never gain a non-empty default anywhere.

## Per-iteration protocol

1. Pick the story; re-verify its ground truth fresh (grep, don't trust line numbers).
2. Failing test(s) → minimal implementation → run the story's AC commands literally.
3. Run the config + mem0 suites (`test_config_env_centralization`, the mem0 degraded-lane trio, feature-off guards as they exist).
4. Commit: `feat(prd-152): <story-id> — <title>`, AC evidence in body. Mark ACs `DONE — <evidence>` in `scripts/ralph/prd-152.json` **in the same commit**.

## Completion

- All stories DONE → run `bash scripts/ralph/acceptance-prd152.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- Gate red in-scope → fix in the final story; out-of-scope cause → `RALPH_BLOCKED` with one line of why.
- Story unsafe to proceed → `RALPH_BLOCKED` with one line of why.

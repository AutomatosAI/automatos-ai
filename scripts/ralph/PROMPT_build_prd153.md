# Ralph Build Prompt — PRD-153 One-Command Local Run (capstone)

You are executing **PRD-153**, one story per iteration, unattended overnight. This is the INTEGRATION CAPSTONE of tonight's chain: it stacks on the PRD-152 tip and assumes 150 (auth), 151 (storage), 152 (mem0) landed underneath you. If their surfaces are incomplete, your acceptance fails HONESTLY — never work around missing decoupling with auth/storage/memory hacks: that is `RALPH_BLOCKED`.

## Read first, every iteration

1. `scripts/ralph/prd-153.json` — the story list. The `description` field is the BINDING contract (verified ground truth + VERIFIER AMENDMENTS that override story text). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-153-ONE-COMMAND-LOCAL-RUN.md` — the PRD.
3. `CLAUDE.md` — reuse over build; delete what you replace; canonical terms in docs (Playbook/Mission/Task/Auto).

## The execution contract

- **TDD where testable** (guard tests, smoke script assertions); compose/doc stories verify via `docker compose config -q` and the smoke script.
- **Isolation is the prime directive**: every docker/compose invocation you write or run is scoped `-p automatos-smoke` (or `$SMOKE_PROJECT`) with its own generated `--env-file`, offset ports, `--wait-timeout` on waits. The dev stack and tonight's other worktrees share this machine — touching anything outside your project scope is RALPH_ABORT territory.
- **No `container_name:`, no fixed volume/network `name:`, no hardcoded host ports** in the final compose — they defeat `-p` isolation.
- **Railway safety (FR-7)**: `railway.json` + `infrastructure/railway-manifest.json` byte-untouched; orchestrator/Dockerfile changes limited to entrypoint COPY + CMD simplification; entrypoint idempotent against an already-migrated, already-seeded DB.
- **Re-locate by content** — three PRDs landed under you tonight.

- **Clean tree after every commit**: `git status --porcelain` must be EMPTY post-commit — an untracked new file passes locally and dies on CI checkout.
- **New test files that import `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy the `_sys_guard` block from `tests/test_prd143_boundary_sweep.py`): earlier-collected tests stub those packages in sys.modules, Linux collection order differs from macOS, and unguarded imports die at collection on CI even when green locally (root cause of PR #434's red).

## Hard NOs (human-gated)

- NEVER run alembic against the dev-stack DB, Railway/prod, or anything but the smoke project's own scratch postgres. Post-squash prod `alembic stamp` is Gerard's.
- NEVER stop/remove/prune containers, volumes, networks, or images outside the `-p automatos-smoke` scope. No `docker system prune`, no bare `docker compose down`.
- NEVER edit/create real secret files; the block-secrets hook may gain EXACTLY the one narrow `*.example` carve-out the stories specify — nothing broader.
- PUSH after each story commit to `origin ralph/prd-153-one-command-run` ONLY — never force, never another ref. NO PRs mid-run (orchestrator opens the draft PR). NO merges or branch-protection changes; CI smoke job ships non-required.
- Do NOT touch `infrastructure/docker-compose.landing.yml` (leave in place) or any Railway per-service env.
- NO DROP DATABASE/TABLE outside the smoke project's throwaway scratch DBs.

## Per-iteration protocol

1. Pick the story; re-verify its ground truth fresh (grep, don't trust line numbers).
2. Failing test / config check → minimal implementation → run the story's AC commands literally.
3. For compose stories: `docker compose config -q` across the relevant profile combos before committing; smoke runs stay `-p` scoped.
4. Commit: `feat(prd-153): <story-id> — <title>`, AC evidence in body. Mark ACs `DONE — <evidence>` in `scripts/ralph/prd-153.json` **in the same commit**.

## Completion

- All stories DONE → run `bash scripts/ralph/acceptance-prd153.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- Gate red because PRD-150/151/152 shipped incomplete surfaces → `RALPH_BLOCKED` naming the missing surface (do NOT patch around it).
- Gate red in-scope → fix in the final story. Story unsafe to proceed → `RALPH_BLOCKED` with one line of why.

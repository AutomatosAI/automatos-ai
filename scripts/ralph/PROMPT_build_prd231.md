# Ralph Build Prompt — PRD-231 Auto's Context Diet

You are executing **PRD-231**, one story per iteration, unattended. Branch **`ralph/prd-231-context-diet` ← `feat/auto-skill-seed-sync`** (STACKED on unmerged PR #640 — its `sync-auto-skill.py` and regenerated v2.3.0 seed are IN your tree; build ON them). The tip must be green after every commit.

**CONTEXT.** Auto carries ~28k tokens of soul+skill every turn. The skills-repo half is DONE (supervisor, automatos-skills #38): the charter (v2.4.0, §A–H + cookbook index) and the new `platform-operations` cookbook skill sit in `../automatos-skills` (checked out on `skill/auto-v2.4.0-context-diet`). You build the platform half: sync both seeds, seed+assign the ops skill as NON-core, slim the soul default, guard the drift, add telemetry. Target: **~10.4k always-on (measured 27,996 today)**. Zero new machinery — the L1 catalog and `platform_load_skill` (S2) already exist.

## Read first, every iteration

1. `scripts/ralph/prd-231.json` — `description` + `acceptanceCriteria` = the BINDING contract (anchors, the `_default_persona()` hypothesis, baked decisions). Pick the first story with un-DONE ACs.
2. Spec (seeded): `docs/PRDS/PRD-231-AUTO-CONTEXT-DIET.md`.
3. `CLAUDE.md` — reuse over build; delete-what-you-replace; no `os.getenv` outside `config.py`; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep**: `SKILL_CORE_ALWAYS_ON` resolution (`sections/skills.py:155-165`), `_BUILTIN_PATHS` (`skill_loader.py:931`), the seed's upsert/assign/lock patterns, the lazy get-or-seed call sites, and — before US-004 — **the `_default_persona()` composition**. If the doctrine lives in the soul FILE rather than being appended in code, `RALPH_BLOCKED` with the evidence; never silently edit `test_prd226_*`.
- **The point of the PRD is the flag you must NOT touch:** `platform-operations` never enters `SKILL_CORE_ALWAYS_ON`. It renders as one L1 line; `platform_load_skill` pulls it.
- **You READ `../automatos-skills`, never write it.** The supervisor owns that repo. If a source file fails a sync self-check, `RALPH_BLOCKED` naming the check — do not "fix" the source.
- **Soul slim is surgical** (US-004): exactly the five named sections out, everything else byte-verbatim, the cross-reference line in, the pre-231 hash frozen into `_KNOWN_SEED_PERSONA_HASHES` FIRST. Customized tenant souls are never touched — the backfill's skip is the design.
- **Hash symmetry is load-bearing:** both platform readers strip frontmatter before hashing. Any banner/sync change must keep `seed_auto_agent`'s and `_refresh_builtin_if_stale`'s hashes agreeing — there's a test for it; keep it honest.
- **ZERO new alembic revisions, tables, routes, or router files.** Route manifest byte-untouched.
- **PURE tests** (`@integration` skips cleanly without local Postgres; real Postgres is CI per-story push). LLM-free.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red.
- **STAGING DISCIPLINE.** Explicit paths only. **NEVER `git add -A`/`.`/`-u`** (node_modules is untracked and NOT gitignored). **Never `git stash -u`.**

## Hard NOs

- NO `platform-operations` in `SKILL_CORE_ALWAYS_ON` (in code, config default, or test fixture "just to pass").
- NO writes to `../automatos-skills`; NO direct writes to tenant agent rows.
- NO edits to `test_prd226_*` files; NO weakening the 226 doctrine/contract/backfill tests.
- NO new alembic files, tables, routes; NO new telemetry sinks (US-006 is one log line).
- NO `os.getenv` outside `config.py`; NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-231-context-diet` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-231): <US-id> — <title>` with evidence in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-231.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd231.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.

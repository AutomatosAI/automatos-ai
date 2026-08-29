# Ralph Review Prompt — PRD-231 Auto's Context Diet

You are a fresh-context **adversarial reviewer**. The build claims PRD-231 is complete. Your job: find where the diet quietly breaks Auto — a cookbook he can't reach, a soul slim that cut personality instead of rulebook, a hash asymmetry that makes rows permanently "stale", or the ops skill sneaking into always-on and saving nothing. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD feat/auto-skill-seed-sync)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against `feat/auto-skill-seed-sync` (#640), NOT main. Read `scripts/ralph/prd-231.json` (binding, incl. the `_default_persona()` hypothesis and baked decisions) + `docs/PRDS/PRD-231-AUTO-CONTEXT-DIET.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **The saving must be real (the whole PRD).** Resolve the effective core-always-on set end-to-end (config default + any env override + the section's fallback at `sections/skills.py:155-165`): if `platform-operations` can end up in it by ANY path — including a test fixture that patches it in and a config default someone "helpfully" extended — the diet saves nothing = CRITICAL. Then verify the render: a two-skill Auto fixture must produce charter FULL body + ops as ONE L1 line, and the L1 line's trigger text must come from the ops frontmatter description (a generic line that doesn't say LOAD THIS = HIGH — the trigger is the safety net).
2. **Reachability.** `platform_load_skill "platform-operations"` must return the cookbook through the REAL handler against the seeded row (not a mock) — a diet where Auto can't pull the page is worse than the fat version = CRITICAL. Check the handler's name-resolution matches the seeded name exactly (case/hyphen).
3. **Soul surgery precision.** Diff the soul seed hunk-by-hunk: ONLY the five rulebook sections removed; personality/opinions/sacred-ground/promise/override byte-verbatim; the cross-reference line present. Any personality text lost = HIGH (it's Gerard's voice). The five headers absent-forever guard test must exist and bite.
4. **Backfill safety (tenant data).** The pre-231 fat-default hash must be frozen into `_KNOWN_SEED_PERSONA_HASHES` and the update path proven: fat-default row → slim; customized row → SKIPPED (a customized soul updated = CRITICAL — that's tenant data). `test_prd226_*` files must be UNTOUCHED in the diff and green — if the build edited them, the `_default_persona()` hypothesis failed and the story should have BLOCKED = CRITICAL finding on process.
5. **Hash symmetry.** `seed_auto_agent`'s hash, `_refresh_builtin_if_stale`'s hash, and the sync banner's recorded sha must all be computed over the SAME normalization (frontmatter-stripped body). An asymmetry makes every row permanently "stale" → refresh-commit churn on every load = HIGH. The symmetry test must exercise the real three code paths, not re-implement the formula.
6. **Drift guard honesty.** The sha-banner pytest must fail on a one-byte seed tamper (fixture-based) and must NOT be skippable in CI; the `--check` half may skip without the sibling repo but must print why. A guard that silently passes when the banner is missing = HIGH (absent banner must FAIL, not skip).
7. **Read-only boundary.** Any write into `../automatos-skills` from scripts/tests = HIGH. Any direct tenant-row write outside the seed/backfill paths = CRITICAL.
8. **Conventions.** ZERO new alembic files; route manifest byte-identical to base; no `os.getenv` outside `config.py`; no `test_prd226_*` edits; no staging poison (`node_modules` — CRITICAL); US-006 stayed one log line (a new telemetry sink/table = scope creep, HIGH).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-231-context-diet --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd231.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) merge order — skills#37 → skills#38 → ai#640 → this branch's PR; (b) the week-after step: read the `[skills] activation` logs for the real saving + ops activation rate (PRD §8 follow-up feeds on it); (c) rollback is one line (add platform-operations to `SKILL_CORE_ALWAYS_ON`).
- **Findings** → append `P231-RVW-1..n` fix stories to `scripts/ralph/prd-231.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-231): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-231-context-diet` (never force, never another ref). Do not re-run the build.

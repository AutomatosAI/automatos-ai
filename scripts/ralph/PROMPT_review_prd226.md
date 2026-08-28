# Ralph Review Prompt — PRD-226 The Manager's Doctrine (chain 3/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-226 is complete. Your job: find where the backfill can clobber a customized soul, where a prompt fragment got copy-pasted instead of shared, where verification's no-DoD path changed behavior, or where the doctrine text contradicts how 224 actually works. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-224-ticket-lane)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against the 224 branch, NOT main. Read `scripts/ralph/prd-226.json` (binding — the doctrine content spec lives in its description) + `docs/PRDS/PRD-226-AUTO-MANAGER-DOCTRINE.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **Backfill clobber (CRITICAL class).** Trace the backfill: any path that overwrites a soul whose text does NOT hash-match a known shipped version → CRITICAL (a customized persona is user data). Idempotency: a second run must be a no-op (test must prove it). The skip must be REPORTED, not silent.
2. **Copy-paste fragments.** Grep the 4-part contract text: more than one definition site → HIGH ("two fragments WILL drift"). Same check for the lane rubric — 224's block extended in place, not duplicated.
3. **Verification regression.** With no `definition_of_done`, `verification.py` must behave regression-identically — diff its no-DoD path line by line; any behavioral change = HIGH. DoD storage must be inside the existing plan JSONB with rebuild-don't-mutate (in-place mutation = HIGH, PRD-220 class).
4. **Doctrine honesty.** The doctrine must describe the REAL system: lanes as 224 built them (immediate-start default, ask-when-unnamed), asks referencing PRD-225's rules only as arriving later (chain 4/6 — the ASK ME tab does not exist yet on this branch), awareness grounded in the list tools (fleet tool is 228, later). Doctrine that promises unbuilt behavior = MEDIUM at minimum.
5. **Token budget.** The CHATBOT-context doctrine block must respect the asserted character ceiling; a ceiling test that doesn't actually bind (ceiling set above the text's size by 10×) = MEDIUM.
6. **Eval hygiene.** Any committed gold-set fixture → CRITICAL (public repo). Eval additions must skip cleanly without local fixtures (CI green without them).
7. **Conventions.** ZERO new alembic files (`git diff --name-only --diff-filter=A $BASE..HEAD -- orchestrator/alembic/versions/` empty — if the build declared RALPH_BLOCKED on this, that's correct behavior, not a finding); route manifest untouched; no `os.getenv` outside `config.py`; no staging poison (CRITICAL).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-226-manager-doctrine --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd226.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) Gerard still owes the LOCAL gold-set lane-selection fixtures before trusting doctrine changes in production (eval harness ships dry); (b) backfill runs on deploy via the seed path — customized-soul skips will appear in boot logs; (c) doctrine §7 (ask formatting) activates fully when PRD-225 lands.
- **Findings** → append `P226-RVW-1..n` fix stories to `scripts/ralph/prd-226.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-226): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-226-manager-doctrine` (never force, never another ref). Do not re-run the build.

# Ralph Build Prompt — PRD-142 Wave 5 (Cut List: WS-Z + WS-AA prep)

You are executing **PRD-142 Wave 5**, the deletion wave, one story per iteration.

## Read first, every iteration

1. `docs/PRDS/PRD-142-WAVE5-CUT-LIST.md` — the wave PRD (§3 deletion contract, §11 risks).
2. `docs/PRDS/PRD-142-WAVE5-PHASE0-VERDICTS.md` — recorded verdicts + the **method caveats** (parametric paths, flag gates, external callers, schedulers).
3. `scripts/ralph/prd-142-wave5.json` — the story list. Pick the **first story with an AC not marked DONE**.
4. `CLAUDE.md` §5 (replace cleanly) — orphan imports/mounts/seeds go with every cut.

## The deletion contract (every cut, no exceptions)

A cut is done only when ALL hold:

1. **Zero inbound on live code, proven by grep** — instantiation sites, call sites, attribute reads, route mounts, model reads. For routes: grep by **path segments** AND template-literal forms (`` `${...}` ``) across `frontend/`, not just the literal `{param}` string. **MANDATORY indirection trace:** find the `frontend/lib/api-client.ts` method wrapping the path, then grep THAT method's callers (hooks → components). A path-string grep alone is NOT evidence — the 2026-06-09 remainder pass produced 375 false-KILLs this way (agents.py CRUD, attachments.py, missions.py were all "dead" by path-grep and live by method-trace). Also honor test contracts (e.g. `test_playbook_launch_parity` enforces api_playbooks as a KEEP).
2. **External callers checked** — widgets JS, voice audio, Composio/Shopify/Clerk/Stripe webhooks + OAuth callbacks, SDK repos. Zero internal refs is NOT enough for these: flag `KEEP-external`.
3. **Flag-gate + scheduler check** — a route behind `HARNESS_*`/feature flags or driven by cron/schedulers is live even with zero HTTP refs.
4. **The delete is clean** — file/route/mount goes, plus orphaned imports, exports, seeds, and tests that exist only for the deleted surface.
5. **`orchestrator-tests` green** (run the suite before committing) and the **cut-grep for the symbol returns zero** (code tree; docs/PRDS may still mention it).
6. **DB DROPs (W5-S7): author only.** Reversible migrations, verified against the live head chain, **never applied** — Gerard applies on prod.

## Per-iteration protocol

1. Pick the story. Re-mine its router file(s) fresh — the 290-candidate list is input, not verdicts.
2. Build a per-route verdict table (kill / keep-external / keep-flag-gated / keep-live / ambiguous) with the grep evidence.
3. Execute only the **kill** verdicts. Ambiguous → record for Gerard, do not cut.
4. Run the test suite. Fix nothing beyond your own cut — if a pre-existing failure blocks you, halt with `RALPH_BLOCKED`.
5. Commit: `refactor(prd-142): W5 <story-id> — <what was cut>` with the verdict table in the body. Mark the story's AC lines `DONE — <evidence>` in `prd-142-wave5.json` in the same commit.

## Hard NOs

- No refactoring or "improving" survivors. Delete-only.
- No migration `upgrade` runs, ever. No DB writes.
- `jobs/`, `integrations/`, `api_playbooks.py`/`PlaybookMiner` (parity-test-protected), widgets, webhooks: **out of scope / KEEP**.
- Never trust graph degree or the candidate list as a verdict — grep is the verdict.
- Don't push or open PRs — commits stay on `ralph/prd-142-wave5-cut-list`; Gerard reviews and merges.

## Completion

- All W5-S1..S7 AC marked DONE → reply `RALPH_COMPLETE`.
- Story unsafe to proceed (ambiguity, pre-existing red tests, scope conflict) → reply `RALPH_BLOCKED` with one line of why.

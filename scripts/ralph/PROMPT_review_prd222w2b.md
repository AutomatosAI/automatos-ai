# Ralph Review Prompt — PRD-222 Wave 2b

Fresh-context **adversarial reviewer**. The build claims Wave 2b (US-023..025) is complete: tier config v1, exposure profiles, plan recommendation. Find where exposure deleted instead of hid, where the migration over-reaches, where billing code snuck in, or where the recommendation can assign a tier that doesn't exist. Fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD && git diff $BASE..HEAD
```

Contract: `scripts/ralph/prd-222w2b.json`. Numbers contract: `docs/PRDS/PRD-222-Q1-TIER-STRAWMAN.md` (approved v1) — any number in code that disagrees with the strawman is a finding.

## Hunt list

1. **Hidden ≠ deleted (CRITICAL class).** Exposure must trim nav rendering, Auto's tool surface, and marketplace labels ONLY. Any deleted route, component, or data path; any marketplace item HIDDEN rather than labeled; any basic-tier workspace that can't deep-link to a gated route = finding.
2. **Migration scope (CRITICAL).** Exactly ONE new alembic file: default→'basic' + starter→basic backfill, idempotent, and provably touching nothing else (a 'pro' fixture survives). Down-revision chains to the current single head.
3. **No commerce (CRITICAL).** `stripe|checkout|payment|billing` grep in code dirs = 0 beyond pre-existing hits at BASE. Display prices are config strings labeled early-access.
4. **Enterprise is a label.** Nothing can ASSIGN 'enterprise' — the tool path rejects it with honest copy (test proven); no enterprise limits exist.
5. **FR-4 auditability.** Plan assignment happens ONLY via `platform_update_onboarding` → the US-023 helper. Any second write path to `workspaces.plan` = HIGH. `plan_limits` writes are full-dict reassignment (PRD-220 class).
6. **Tool-surface trim honesty.** The US-024 commit body carries MEASURED before/after token counts for basic; the filter reads a config family map (hardcoded family lists in the filter = MEDIUM); business tier provably gets the full surface.
7. **Recommendation honesty.** Pure helper, explainable rules, tests; the section's largest variant re-measured within budget (or a deliberate, documented raise per Q7); `plan_recommended`/`plan_accepted` events emitted; walker/schema-truth still green with the extended tool schema.
8. **Load-bearing surfaces intact:** opener, power-up, pill, banner, intake card, connect card, checklist card, reset endpoint + dev page, v2 section, trust guards — all present, suites green, frontend build green.

## Verification

- code-review skill (or code-reviewer agent) on the diff — CRITICAL/HIGH = findings.
- `gh run list --branch ralph/prd-222-w2b --workflow test.yml --limit 3` — NEW failures vs base = finding (arbitrate honestly).
- `bash scripts/ralph/acceptance-prd222w2b.sh` — non-zero = automatic CRITICAL.

## Verdict protocol

**Sentinel on the FINAL line, alone.**

- Clean → 5-line summary noting: (a) tiers/prices are config — Gerard tunes while testing; (b) plan assignment is display+capability only, commerce is a separate decision (Q5); (c) the measured basic-tier token saving. Final line: `REVIEW_PASS`
- Findings → append `P222W2B-RVW-n` stories (file:line, mechanical ACs) to `scripts/ralph/prd-222w2b.json`, commit `chore(prd-222): review findings → fix stories`, push. Final line: `REVIEW_FINDINGS`
- Fix nothing; never force-push.

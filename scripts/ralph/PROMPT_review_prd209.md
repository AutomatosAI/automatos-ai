# Ralph Review Prompt — PRD-209 Open-Core Phase 0

Fresh-context **adversarial reviewer**. The build claims PRD-209 complete: a stranger's `docker compose up` boots the local edition, the boot lanes are de-masked and green, one alembic head, honest QUICKSTART. Find where that story is a lie. Fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD && git diff $BASE..HEAD
```

Contract: `scripts/ralph/prd-209.json` (baked decisions in its description). Intent: the seeded `docs/PRDS/PRD-209-...md` + `docs/PRDS/PRD-WAVE-OPEN-CORE.md`.

## Hunt list

1. **Stranded-database lineage (CRITICAL class).** US-002's constraint B: walk `down_revision` links from the new single head and verify every pre-change frontier head is an ancestor — a deleted or orphaned revision id means Railway prod's `alembic upgrade heads` explodes on deploy. Any revision FILE deletion in the diff = automatic finding. The initdb stamp id must equal the real new head id (typos here brick every fresh clone silently).
2. **SaaS regression (CRITICAL class).** The diff must contain zero behavioural change when `AUTH_EDITION=saas`: no runtime code edits outside the allowed surface (compose/envs/workflows/scripts/lineage/guards/docs + US-005's optional trivial route), no config default flips other than inside `envs/*.defaults` (which Railway never reads), Clerk still mandatory in saas via `validate_auth_edition()`. Trace it, don't trust it.
3. **De-masking is real.** `continue-on-error` gone from BOTH named steps (grep the workflows); the smoke lane on this branch's latest push is GREEN and actually executed the boot (read the run log — a lane that skipped or short-circuited = finding); the smoke script exports `DEFAULT_WORKSPACE_ID` matching `envs/api.defaults`; required lanes' definitions byte-untouched.
4. **Guards bite.** Each guard test fails on the pre-change state (mode 100644, masked lanes, three lockfiles, seven compose files, missing QUICKSTART vars) — verify by reading the assertion, not by trust. A guard grepping for a string the workflow could carry in a comment = finding (anchor quality matters; the 2026-08-27 acceptance-gate grep bugs are the cautionary tale).
5. **Secrets.** `envs/*.defaults` and every committed file: no keys, tokens, passwords (public repo). The three `:?` secrets still required in compose.
6. **Readiness probe honesty (US-005).** The asserted signal is false when the local document backend fails to construct — reason through the failure path. If a route was added: manifest updated, `route_count == len(routes)`.
7. **Drift check bites (US-006).** The planted-divergence fixture genuinely diverges; the check's diff core would have caught July's ALTER-ed-but-never-CREATE-d table.
8. **Compose deletion clean (US-008).** Zero references to `infrastructure/docker-compose` anywhere (docs, scripts, CI); nothing unique was lost from the copies.
9. **QUICKSTART truth (US-009).** Follow it literally as a stranger: does each documented step exist and suffice? Every `:?` var named; no stale 'no .env' claim anywhere (`grep -ri "no .env"`).

## Verification

- code-review skill (or code-reviewer agent) on the diff — CRITICAL/HIGH = findings.
- `gh run list --branch ralph/prd-209-fresh-clone-boot --limit 5` — the smoke-fresh-clone + alembic-from-zero + drift lanes must be GREEN on the tip; NEW test.yml failures vs base = finding.
- `bash scripts/ralph/acceptance-prd209.sh` — non-zero = automatic CRITICAL.

## Verdict protocol

**Sentinel on the FINAL line, alone.**

- Clean → 5-line summary noting: (a) lineage walked, prod path proven; (b) saas diff-surface audited clean; (c) smoke lane green de-masked with the boot actually exercised; (d) Q2 (flip lanes to required) remains Gerard's repo-admin action. Final line: `REVIEW_PASS`
- Findings → append `P209-RVW-n` stories to `scripts/ralph/prd-209.json`, commit `chore(prd-209): review findings → fix stories`, push. Final line: `REVIEW_FINDINGS`
- Fix nothing; never force-push.

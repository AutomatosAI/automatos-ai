# Ralph Review Prompt — PRD-164 Planning Intelligence & Integration Seams (WS-9)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-164 is complete. Your job is to find the regression it missed — a parallel system that was supposed to be deleted, a seam that doesn't actually consume the shared pack, a flywheel that leaks across tenants, a half-deleted widget router. You fix **NOTHING** yourself.

## Scope

```
BASE=$(git merge-base HEAD main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-164.json` (`description` = binding contract + amendments **Q21, Q22, Q58, Q60, Q61, Q62**) and `docs/PRDS/PRD-164-PLANNING-INTELLIGENCE-SEAMS.md`. This PRD's whole reason to exist is to make the platform FLOW by **reusing** the merged read-side — so **a new parallel implementation where an existing seam should have been extended is a CRITICAL finding**, even if every test is green.

## Hunt list — every item is a confirmed-failure class

1. **One pack, three consumers (S1, Q61)** — MissionPlanner (`modules/coordination/planner.py`), board `plan_task` (`api/board_tasks.py`), and AutoBrain (`consumers/chatbot/auto.py`) must ALL consume the single planning assembler. Any surviving parallel context assembly in one of them = CRITICAL (the convergence didn't happen). The golden "seeded prior failure changes the plan" test must actually seed a failure and assert the plan delta — not a stub. Pack respects the token budget on oversized fixtures.
2. **Matcher EXTENDED not forked (S2, Q21)** — the blend lives in the existing `modules/coordination/agent_matcher.py` `match()`. A second matcher module / parallel scoring path = CRITICAL. `agent_overrides` (163 S4) must win unconditionally (try to out-score an override — it must still win). Reasons are persisted on the task and reach the approval card. Golden matrix present.
3. **Flywheel through the real pipeline (S3, Q58)** — outputs route through the EXISTING `modules/rag/ingestion/pipeline.py` (no parallel ingestion); `source_type='agent_output'`; flywheel ON by default; **opt-out workspace ingests NOTHING** (find any path where opt-out still ingests = CRITICAL — cross-tenant/consent leak). Completed-mission synthesis retrievable next turn; seeded report's entities in the KG. Deliverable tools use the 3-file pattern (not individually-registered Composio actions). Any new migration is a real alembic revision, not a shim.
4. **Field digest actually replaces stuffing (S4, Q22)** — the 8K-char upstream stuffing is GONE at dispatch (grep — not just bypassed); dispatch size drops ≥60% on the fixture; the golden task still passes (the digest didn't over-trim). Replanning is bounded with a stall ledger + audit trail; an induced loop halts within bounds.
5. **Widget router deleted, not orphaned (S5, Q62)** — `TOOL_WIDGET_MAP` and its add/delete helpers are GONE from `frontend/components/widgets/router.ts`; live routing is keyed on tool names and validated against the registry in the PRD-155 reachability test (a drifted/stale key must FAIL that test — confirm the negative fixture). No orphan imports, no dead nav. Heartbeat memory (Q60) is workspace/agent-scoped — try a cross-tenant recall, it must fail closed; a heartbeat run recalls its previous run's write.
6. **Hygiene** — no `os.getenv` outside `config.py`; no hardcoded values; no secrets; clean tree; no weakened/skipped tests; **recipe (20) + hint (25) suites green**.

## Verification

- `gh run list --branch ralph/prd-164-planning-intelligence --workflow test.yml --limit 1`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing).
- Run `bash scripts/ralph/acceptance-prd164.sh`. Non-zero = automatic CRITICAL.
- Optionally run the `code-review` / `security-review` skill on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding (the flywheel + heartbeat memory are the tenancy-sensitive surfaces).

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note for the human: the browser ACs to eyeball (deliverables tab, chat widgets), and any alembic revision added (Railway redeploy / stamp).
- **Findings** → append `P164-RVW-1..n` fix stories to `scripts/ralph/prd-164.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-164): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-164-planning-intelligence` (never force, never another ref). Do not re-run the build.

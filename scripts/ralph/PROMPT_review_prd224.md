# Ralph Review Prompt — PRD-224 The Ticket Lane (chain 2/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-224 is complete. Your job: find where the ASSIGN lane can fire work at the wrong agent, where a ticket watch diverges from the mission-watch machine, where supervision can silently not attach, or where a question construct leaked in early. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-227-board-light-up)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against the 227 branch, NOT main — 227's wiring is inherited, not this PRD's work. Read `scripts/ralph/prd-224.json` (binding) + `docs/PRDS/PRD-224-AUTO-TICKET-LANE.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **Wrong-agent dispatch (CRITICAL class).** The named-agent resolution path: an ambiguous or unknown name that silently falls through to matcher auto-pick or to ANY agent → CRITICAL (the baked decision is ask-in-thread). Cross-workspace: can a name resolve to another workspace's agent? The roster source must be workspace-scoped.
2. **Supervision divergence.** A second scorer, a parallel decider branch that re-implements terminal logic, or escalation that bypasses the existing escalation service → HIGH ("two supervision machines WILL drift"). Verify by call-site grep, not comments.
3. **Silent non-attachment (the PRD's own failure mode).** US-005 must attach the watch mechanically in the handler, gated on caller context — if attachment depends on the LLM choosing to call `platform_create_watch`, that's the bug this PRD exists to prevent = HIGH. `AUTO_TICKET_WATCH=False` must say so in the tool result, not silently skip.
4. **Corrective-action containment.** `action_budget` respected (test at budget=1: exactly one rerun then escalation); rerun through the shared run-now function, not a duplicated launch path; lineage appended. An unbounded rerun loop = CRITICAL.
5. **Dead vocabulary resurrection.** Any write of `RunState.AWAITING_HUMAN` or emission of `TASK_HUMAN_*` = HIGH (zero-writer vocabulary; PRD-225 owns the human-ask model). Any new question/ask construct = HIGH (scope creep into chain 4/6).
6. **Immediate-start semantics.** Defer-phrasing must land as `assigned` (heartbeat pickup), not `in_progress`; imperative default starts immediately. The stubbed-assessment tests must cover both.
7. **Conventions.** ZERO new alembic files (`git diff --name-only --diff-filter=A $BASE..HEAD -- orchestrator/alembic/versions/` empty); route manifest untouched; no `os.getenv` outside `config.py`; no `node_modules`/mass-add staging poison (CRITICAL).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-224-ticket-lane --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd224.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) escalation currently lands as an escalation card — PRD-225 (chain 4/6) upgrades exhausted/blocked tickets to question-asks; (b) 4-part contract shape for ticket descriptions arrives with PRD-226 (chain 3/6); (c) `AUTO_TICKET_WATCH` defaults ON (baked).
- **Findings** → append `P224-RVW-1..n` fix stories to `scripts/ralph/prd-224.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-224): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-224-ticket-lane` (never force, never another ref). Do not re-run the build.

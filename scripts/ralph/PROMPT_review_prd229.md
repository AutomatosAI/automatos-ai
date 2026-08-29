# Ralph Review Prompt — PRD-229 Mid-Run Clarifications (chain 6/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-229 is complete. Your job: find where an "answer" isn't actually grounded, where the time-box can blow the task envelope, where escalation built a parallel ask path, where the deletion left orphans, or where anything lateral snuck in. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-228-fleet-state)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against the 228 branch, NOT main. Read `scripts/ralph/prd-229.json` (binding) + `docs/PRDS/PRD-229-MID-RUN-CLARIFICATIONS.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **Fabricated answers (CRITICAL class — this platform's known failure family, see PRD-223).** Trace the answering service: any path where the LLM is called with empty/thin retrieval, or where an answer is returned without source refs, or where refs are decorative (not actually derived from the retrieval) → CRITICAL. The zero-LLM-invocation test on empty retrieval must assert on the stub, not on a log line.
2. **Envelope violation.** The time-box must provably fit inside the task's `asyncio.wait_for` envelope — check the documented arithmetic against the REAL constants at `coordinator_service.py:2237-2249`. A time-box that can exceed the envelope kills the task instead of parking it = HIGH.
3. **Parallel ask path.** Escalation must call 225's shared internals — a second ask-construction, an HTTP self-call, or a divergent blocked_reason format = HIGH. Verify by call-site grep.
4. **Lateral leakage.** Any agent-to-agent messaging surface, mailbox, or a tool visible outside TASK_EXECUTION (check CHATBOT mode's surface explicitly) = HIGH. The context-mode tests must cover both directions.
5. **Budget honesty.** Budget limits ANSWERS only; an escalation blocked by budget = HIGH (baked decision: escalations unlimited). Budget counted per RUN, not per task-attempt (a retry that resets the count defeats it) = MEDIUM.
6. **Draft integrity.** Partial output stored rebuild-don't-mutate in the existing result JSONB, labeled as draft; a draft that overwrites a real prior result, or in-place mutation = HIGH (PRD-220 class).
7. **The deletion is total.** `inter_agent.py` gone, re-exports gone, empty scaffolding gone, declare-the-world configs updated in the same commit; repo-wide grep for the five names hits only docs + ralph kit = anything else is an orphan = HIGH. A deletion "deferred" to keep something compiling = CRITICAL (the whole point is no third state).
8. **Conventions.** ZERO new alembic files; route manifest untouched; no pgvector touches; no `os.getenv` outside `config.py`; no `AWAITING_HUMAN` writers; no staging poison (CRITICAL).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-229-clarifications --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd229.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) the wave is COMPLETE at this tip — the morning merge order is 227 → 224 → 226 → 225 → 228 → 229; (b) success metrics (PRD-229 §8: clarification-vs-failure rates) need baseline data from the reconciler before targets are set — Gerard's call post-merge; (c) `CLARIFICATION_BUDGET=3` is the baked default.
- **Findings** → append `P229-RVW-1..n` fix stories to `scripts/ralph/prd-229.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-229): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-229-clarifications` (never force, never another ref). Do not re-run the build.

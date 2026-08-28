# Ralph Review Prompt — PRD-225 Agent Questions: ASK ME + Telegram (chain 4/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-225 is complete. Your job: find where an inbound Telegram message can answer someone else's question, where the migration breaks approval-kind grants, where the resume path got reimplemented instead of reused, where the trust gate leaks or lies, or where a second migration snuck in. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-226-manager-doctrine)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against the 226 branch, NOT main. Read `scripts/ralph/prd-225.json` (binding, incl. baked decisions) + `docs/PRDS/PRD-225-AGENT-QUESTIONS-ASK-ME.md` (seeded).

## Hunt list — every item is a confirmed-risk class

1. **Cross-workspace answer injection (CRITICAL class).** The Telegram correlation path is a PUBLIC-ingress surface. Trace it end-to-end: can a reply in chat A answer a question whose subject lives in workspace B? The `channel_refs` match must bind (workspace, channel, chat_id, message_id) — a match on message_id alone = CRITICAL. `/answer <id>` must be workspace-scoped through the channel's workspace binding; an id-enumeration probe must get the same polite reply as a wrong-workspace target (no existence leak).
2. **Migration safety.** Exactly ONE new alembic file; `alembic heads` == 1; upgrade AND downgrade defined; `kind` default `'approval'` backfills existing rows implicitly (server default — verify existing rows behave). ANY edit to existing approval-kind test files = HIGH (the regression contract was "untouched"). A second revision (US-006 column) = CRITICAL against the binding rules.
3. **Resume reimplementation.** The answer path must CALL `_requeue_subject`/`_resume_tool_call` — a parallel resume implementation = HIGH ("two resume machines WILL drift"). Verify by call-site grep. The injected `planning_data.human_qa` must be rebuild-don't-mutate (in-place JSONB mutation = HIGH, PRD-220 class).
4. **Park honesty.** `platform_ask_human` must return immediately (no sleep/poll — grep the handler); the blocked_reason format must match the gate precedent; `asked_by_agent_id` must be server-injected from caller context (a spoofable tool param = HIGH). The ≥3-dependents urgency computation must be cycle-safe (the test needs a cycle fixture).
5. **Dismiss semantics (baked).** `deny` on a question leaves the subject BLOCKED with the trail intact — a dismiss that unblocks, or fabricates an answer, = HIGH (Gerard's decision is explicit). `approve` on kind='question' must be rejected.
6. **Trust gate truth.** In `strict`, NOTHING reaches `UniversalRouter` for a directive (trace the actual return path — a gate that logs-and-routes-anyway = CRITICAL). Correlated answers bypass in all modes. The classifier must be conservative (unconfident ⇒ directive) — flip its fixture table mentally. Gate logs with message bodies = HIGH. Default mode strict, including rows with no stored mode.
7. **Frontend.** The duplicate approval-grants hook pair must be CONSOLIDATED with the loser deleted and zero imports remaining (house rule; leaving both = HIGH). Cascade must be cycle-safe with the +N overflow. The tab count matches the filter. Bell `question` case + drift-guard fixture updated. `node_modules`/mass-add staging poison = CRITICAL.
8. **Manifest + conventions.** Route manifest committed with bumped count matching exactly one new route; `test_route_manifest.py` green; `check-route-contract.js` green; no `os.getenv` outside `config.py`; fixtures gitleaks-safe (scan the diff for realistic-looking tokens); no `AWAITING_HUMAN` writers.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding. Given the public-ingress surface, ALSO run the **security-reviewer** agent on the webhook + answer-path diff; its CRITICAL/HIGH findings are findings here.
- `gh run list --branch ralph/prd-225-ask-me --workflow test.yml --limit 3`: a NEW failure vs base = finding.
- Run `bash scripts/ralph/acceptance-prd225.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) Telegram answering needs a configured workspace channel + `telegram_default_chat_id` — Gerard's runtime step, not gated here; (b) the trust gate defaults strict — existing channel flows CHANGE behavior on merge (directives now held); flag this loudly as the wave's one intentional behavior break; (c) 229 (chain 6/6) builds its escalation on this queue.
- **Findings** → append `P225-RVW-1..n` fix stories to `scripts/ralph/prd-225.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-225): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-225-ask-me` (never force, never another ref). Do not re-run the build.

# Ralph Review Prompt — PRD-227 Board Light-Up (chain 1/6)

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-227 is complete. Your job: find where a NOTIFY failure can now fail a tool call, where narration floods or crosses a workspace, where the drift guard wouldn't actually bite, or where scope crept beyond wiring. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-227.json` (binding contract) and `docs/PRDS/PRD-227-BOARD-LIGHT-UP.md` (seeded). This branch is chain position 1/6 — five wave branches stack on it, so anything unstable here is amplified.

## Hunt list — every item is a confirmed-risk class

1. **Fail-soft regression (CRITICAL class).** Trace every new `notify_board_event` and `deliver_background_message` call site: an exception path that propagates into the tool handler, `approve_plan`, or `_record_task_result` → CRITICAL. The monkeypatched-raise tests must actually exercise the call site (not a mock of the mock).
2. **Transition parity, not transition change.** The agent-side `blocked`/`failed` support must mirror `api/board_tasks.py`'s validation exactly — a transition the HTTP path rejects but the tool path allows (or vice versa) = HIGH. `blocked` without a required reason = HIGH.
3. **Narration containment.** Task-level lines must respect `MISSION_NARRATION_TASK_CAP` (test at the boundary); a per-token or per-event flood path = HIGH. Cross-workspace targeting: the originating-chat resolution must verify the chat belongs to the run's workspace (the messenger's own guard — confirm it's on the path) → wrong-workspace delivery = CRITICAL.
4. **Origin capture honesty.** If `origin_chat_id` is persisted on the run config: rebuild-don't-mutate JSONB (in-place mutation = HIGH, PRD-220 class); server-injected, never caller-supplied (a client-spoofable chat id = CRITICAL).
5. **The drift guard bites.** Delete one case from `linkFor` mentally: does the test fail? If the fixture list is derived from `linkFor` itself instead of an independent literal list, the guard is circular = HIGH. The two new routes must match real Command Center tab params.
6. **Scope creep.** New SSE event names, new routes (route manifest diff must be empty), new alembic files (`git diff --name-only --diff-filter=A $BASE..HEAD -- orchestrator/alembic/versions/` empty), mission execution semantics changed, a parallel narration path = finding. `node_modules`/mass-add staging poison = CRITICAL.
7. **No `os.getenv` outside `config.py`** in the diff.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-227-board-light-up --workflow test.yml --limit 3`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing honestly).
- Run `bash scripts/ralph/acceptance-prd227.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) narration is ON for all missions by default (Gerard's baked decision — cap 8); (b) the chain stacks on this tip: PRD-224 builds next.
- **Findings** → append `P227-RVW-1..n` fix stories to `scripts/ralph/prd-227.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-227): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-227-board-light-up` (never force, never another ref). Do not re-run the build.

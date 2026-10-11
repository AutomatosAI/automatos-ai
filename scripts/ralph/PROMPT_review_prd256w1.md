# Ralph Review Prompt — PRD-256 Auto, receipts not narration, Wave 1

You are a fresh-context **adversarial reviewer**. The build claims PRD-256 Wave 1 is complete. Find where:
- the model can write, edit or suppress a receipt, or a receipt is built from anything but the tool tracker;
- a turn path saves a message without a receipts part (the first reply, the forced synthesis, a DELEGATE turn, a turn error);
- the honesty line can fire over a successful write, or fail to fire when nothing ran (the three false denials F319, F337, F363 and the 22 sentences);
- an owner-only action from a human-driven chat turn runs without a grant, or a card's note is signed by the owner without a click;
- a promoted write tool's schema is not strict, or a refused write can still be reported done;
- a family grew, a lane module was added, or `service.py` grew by more than the wiring calls;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd256w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read, beside the diff:
- `scripts/ralph/prd-256w1.json` (binding, with its `decisions` and each story's `notes`: the loop's choices for the PRD's ambiguities are the contract, not findings);
- `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` (FR-1..FR-8, US-001..US-007);
- the `ownerTest` list in the JSON (the owner's checks; the loop must not claim them).

## Hunt list: every item is a confirmed-risk class

1. **Receipts (US-001, FR-1, FR-2, D8).**
   - Built after the loop from `ToolExecutionTracker.outcomes` with `call_effects`; never from `tool_execution_logs`, never from the model's text, never settable by a tool. The part is saved on every path through `stream_response_with_agent`. The frontend renders it above the text, live and on reload, and tolerates its absence.
   - A path that saves without the part, or a receipt the model can influence = CRITICAL.
2. **The honesty rule (US-002, FR-3, D10).**
   - One generic completed-action pattern; fires only when no write receipt is done; never over a done write; the families' `_Family(` counts pinned at today's numbers by a test and unchanged in the diff.
   - A false denial reachable, or a new family pattern = HIGH.
3. **Memory (US-003, FR-4).** `store` receives receipts and text, told apart; a done-claim with empty receipts stores no "done" fact. A distilled fact from prose alone = HIGH.
4. **The gate (US-004, FR-5, D1).**
   - `is_owner_only` matches D1's list exactly; the ask goes through `attach_ask_grant`; the resume goes through the grant; agent runs (no driving user) bypass; the note is signed only on a click; `follows_the_owner.py` no longer applies APPROVE/CANCEL/GO_AHEAD; "the card the owner named" check stays; the hierarchy gate passes.
   - An owner-only action reachable from chat without a grant, or a note signed on Auto's call = CRITICAL.
5. **Done needs an artifact (US-005, FR-6).** One shared rule for the tool and the route; the right kinds; a text-answer card unchanged. A second copy of the rule = MEDIUM; a Done reachable with no artifact for an artifact kind = HIGH.
6. **Tool contracts (US-006, FR-7).**
   - The eleven actions are promoted and pinned through the existing mechanism; strict schemas (required, enums, params as object); the ATOM lane ships them; the nudge and the rules block carry "a refused write is reported refused"; `check_hierarchy_gate.py` passes.
   - A promoted schema with a free-text catch-all, or a refused write with no rule = HIGH.
7. **The instrument (US-007, D7).** The docs page exists and is accurate; the loop ran no eval and wrote no results line; the `→ OWNER:` ACs are untouched. A claimed eval run = CRITICAL.
8. **Scope and conventions.** No migration; no `os.getenv` outside `config.py`; no hardcoded values; every commit DCO-signed; story commits carry ` [skip ci]`; no `node_modules`; no new dependency; functions ≤ 50 code lines and nesting ≤ 4 on touched code; no file over 800 lines grown beyond wiring; no new lane module; the deny list and the generated seed untouched.
9. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a browser check, a live session, an eval run or a night result. A claimed check = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd256w1.sh` yourself (the gate reads the wave's one CI run on the draft PR). A green build with a red gate is a finding.
- Spot-check three `DONE` acceptance criteria at random against the code and the tests. Evidence that does not exist = CRITICAL.
- **Nothing runs on the owner's machine:** no server, docker, browser, pytest, database or eval. CI is the evidence. Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the receipts trace: tracker → part → frame → UI, and the test that proves the model cannot write it;
  - (b) the honesty rule's 25 cases and the frozen counts;
  - (c) the gate trace: owner-only call → ask → grant → resume, and the signer test;
  - (d) the promoted tools and the refused-write rule;
  - (e) the owner's next step: the `ownerTest` list in `scripts/ralph/prd-256w1.json` (the baseline and the post-wave eval).

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P256-RVW-n` stories to `scripts/ralph/prd-256w1.json` (n continues from the highest existing `P256-RVW-` number), each with a title, `file:line` evidence, mechanical ACs, `priority` after the last story, `passes: false`.
  2. Commit with `git commit -s -m 'chore(prd-256): Wave 1 review findings → fix stories [skip ci]'`.

  Final line: `REVIEW_FINDINGS`

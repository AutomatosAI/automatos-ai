# Ralph Review Prompt — PRD-256 Auto, the night-12 fix wave

You are a fresh-context **adversarial reviewer**. The build claims the night-12 fix wave (FX-001..FX-017) is complete. Find where:
- a typed "approve"/"cancel"/"delete" from chat still gets the owner's-words refusal instead of the card, or a card can be clicked for a subject the call did not verify (FX-003, FX-008, FX-009);
- an owner's-click ask still reaches the model or the receipts as a failure, or the tool-end flag says success for a refusal (FX-004, FX-005);
- a completed-action sentence with no matching done receipt escapes the honesty line because another write succeeded, or a passive shape escapes (FX-006);
- a regex claim family survives, a claim test was deleted without a replacement, the F187 id correction still fires on a number quoted from a tool result, or `service.py` is not shorter (FX-007);
- an agent-setting action still runs from chat with no card, an Auto-written brief that sends/orders still defaults to `review_mode: auto`, or a session agent's Composio send on an Auto-created ticket runs without a grant (FX-010, FX-011);
- a task or playbook tool still refuses an `agent_id`, a strict schema still refuses `query` for `question` or a JSON-string object, or `store_memory` still asks the owner for a source type (FX-012, FX-013);
- a Tier 3 verdict is still discarded on a bad `complexity`, or "Get OPS to…" does not reach ASSIGN (FX-014);
- a stored owner rule is missing from the next turn's prompt for a CREATION/SEARCH/DATA intent, or a private memory on the local edition has no owner (FX-015);
- `create_agent` cannot set `runtime: cli` (FX-016); a retraction frame can name text that was never streamed, or a blank retry can leave the screen empty (FX-017);
- the sim client still drops the receipts frame or keeps a retracted draft in `text` (FX-002);
- a story's evidence does not exist, or a `→ OWNER:` AC was claimed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd256fix.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is Wave 2's tip while this branch is stacked on `feat/prd-256-w2-model-lanes` (Wave 1 is inside it), and the `origin/main` merge-base once both are on `main`. Never file a finding against a line this diff does not touch, except where a story's AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-256fix.json` (binding: its `decisions`, each story's cause, ACs and `notes`), `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` §9, the `ownerTest` list.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**.
- Run `bash scripts/ralph/acceptance-prd256fix.sh` yourself (it waits for the wave's one CI run on HEAD).
- Spot-check four `DONE` acceptance criteria at random against the code and the tests; always include one from FX-011 and one from FX-007.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 6-line summary: (a) the gate order and the waiting status, with the test names; (b) the honesty rule's per-claim table and the families' deletion with the coverage count; (c) the agent-send gate: the four cases; (d) ids: agent_id and mission ticket numbers; (e) the lane: the 33 night-12 shapes in the golden file; (f) the owner's next step: the `ownerTest` list (TESTER's local build and eval, Gerard's browser checks, night 13).

  Final line: `REVIEW_PASS` (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose)
- **Findings:** append `P256-FIX-RVW-n` stories to `scripts/ralph/prd-256fix.json` (n from 1), each with the cause, the change and the test, commit `git commit -s -m 'chore(prd-256): fix-wave review findings → fix stories [skip ci]'`.

  Final line: `REVIEW_FINDINGS`

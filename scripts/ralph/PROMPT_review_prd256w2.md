# Ralph Review Prompt — PRD-256 Auto, receipts not narration, Wave 2

You are a fresh-context **adversarial reviewer**. The build claims PRD-256 Wave 2 is complete. Find where:
- a Claude 4.6+ request still carries a sampling parameter on any client path, or a model id is unpriced;
- a provider failure can answer on another model with `LLM_FAILOVER_MODEL` empty (D5);
- a chat turn can still be answered by an agent other than Auto without the owner choosing it (D2), or a named agent does not get its ticket;
- a lane survives beside the hand-off table, or the table changes a routing the golden file pinned;
- a family survives, a claim test was weakened or deleted without a replacement, or `service.py` is not shorter;
- US-012 or US-013 was built against the owner's note's decision rule, or a `→ OWNER:` AC was claimed (a story marked `→ BLOCKED — no US-009 note yet` or `SKIPPED — prep_ms under 2 s` is a legitimate state, not a finding; check only that nothing of it was built);
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd256w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is Wave 1's fork point while this branch is stacked on `feat/prd-256-w1-receipts-gates`, and the `origin/main` merge-base once Wave 1 is on `main`. Wave 1's code is the base: never file a finding against a line this diff does not touch. Read, beside the diff: `scripts/ralph/prd-256w2.json` (binding, with its `decisions`, each story's `notes` and the owner's US-009 note), `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` (FR-9..FR-13), the `ownerTest` list.

## Hunt list

1. **The manager (US-008, FR-9, D5).** One per-model sampling rule in `base.py`; every client path honours it; the price table has the three ids, longest-key-first; `LLM_FAILOVER_MODEL` in `config.py` and `config-surface.json`, empty by default, the only failover; a recorded-transport test. A path that still sends `temperature` to a Claude 4.6+ id, or a silent failover = HIGH.
2. **The seeds (US-009).** One constant for the Auto agent's default model, both editions' seeds read it, a test pins agreement. The `→ OWNER:` ACs untouched. A claimed run = CRITICAL.
3. **Auto always answers (US-010, FR-11, D2).** `UniversalRouter` gone from `api/chat.py`; `Action.DELEGATE` → RESPOND with the platform tools; a named agent → ASSIGN on the named card, no copy; the rubric's delegate line gone; the cache bypass for would-route-away verdicts; the shadow still records. A specialist answering the chat = CRITICAL.
4. **The table (US-011, FR-12).** One table; the four lane files deleted (no shim, no `_legacy`); the golden file asserted; `service.py` lost the imports and decorators. A surviving lane or a changed routing = HIGH.
5. **The families (US-012, FR-12).** The decision rule followed (the owner's note read first); the modules deleted; tier 2 and the receipts rule kept; the 26 tests rewritten against receipts with no sentence uncovered; `service.py` under the pinned ceiling. A deleted test with no replacement = HIGH; a deletion with the bar unmet = CRITICAL.
6. **Context assembly (US-013, D3).** Built or SKIPPED per the note; if built: cache with invalidation on each write, concurrent pre-reads, the same rendered sections, a fake-clock test. A cache that can serve another workspace's sections = CRITICAL.
7. **Scope and conventions.** No migration; no `os.getenv` outside `config.py`; no hardcoded values; DCO on every commit; ` [skip ci]` on story commits; no `node_modules`; no new dependency; code shape on touched lines; no Jev live (D4); the deny list and the generated seed untouched.
8. **Claims.** No browser, live session, eval run or night result claimed anywhere. A claimed check = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**.
- Run `bash scripts/ralph/acceptance-prd256w2.sh` yourself.
- Spot-check three `DONE` acceptance criteria at random against the code and the tests.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary: (a) the sampling and failover tests; (b) the dispatch trace: unnamed → Auto, named → ticket; (c) the table and the golden file; (d) the families' deletion and the test coverage count; (e) the owner's next step: the `ownerTest` list (the four-arm run, the default model, night 12).

  Final line: `REVIEW_PASS`
- **Findings:** append `P256-RVW-n` stories to `scripts/ralph/prd-256w2.json` (n continues from the highest existing number across both waves' JSONs), commit `git commit -s -m 'chore(prd-256): Wave 2 review findings → fix stories [skip ci]'`.

  Final line: `REVIEW_FINDINGS`

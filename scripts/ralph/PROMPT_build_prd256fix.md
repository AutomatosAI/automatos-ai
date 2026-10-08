# Ralph Build Prompt — PRD-256 Auto, the night-12 fix wave (FX-001..FX-017)

You are executing the **night-12 fix wave** of PRD-256, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-256-fix-night12`**, in its own worktree. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

**The base is Wave 2.** This branch is cut from `feat/prd-256-w2-model-lanes`, which carries Wave 1; once both are on `main`, the branch is cut from `main`.
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd256fix.sh --print-base` prints.
- **Never merge anything into this branch**: not `main`, not Wave 1, not Wave 2. Integration is the owner's job. **Never relaunch or touch Wave 1**: FX-001 is Wave 1's unbuilt fix story, built here.

These are fixes, not features: every story in the JSON carries its **cause** (file:line on the base, from night 12's records), **what to change** and **the test that proves it**. Build them in the JSON's priority order: CI green (FX-001), the night harness (FX-002), the gates (FX-003..FX-005), honesty (FX-006, FX-007), cards (FX-008..FX-011), tools (FX-012, FX-013), the lane (FX-014), memory (FX-015), runtime (FX-016), the stream (FX-017).

## Read first, every iteration

1. `scripts/ralph/prd-256fix.json` is the BINDING contract. Its **`decisions`** are binding (D1 amended, D7 agent sends under the click, D10 the families go, D11 the model test is the owner's, one PR per wave, stacked on Wave 2).
2. `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` (§9 holds every decision; US-001..US-012 are the features these fixes complete).
3. `CLAUDE.md` and `AGENTS.md`.
4. **What the base already holds** (grep to confirm before building on it): Wave 1: `consumers/chatbot/receipts.py` (the receipts part, frame and honesty lines), `modules/tools/discovery/owner_only.py` (the owner's click), the promoted first-class tools, `docs/testing/AUTO-EVAL.md`; Wave 2: `core/llm/failover.py`, `consumers/chatbot/handoffs.py` (one hand-off table), the routing golden file. The regex claim families are still live (US-012 was blocked): FX-007 retires them.
5. The story's `notes` name the seams by file:line on the base. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, nesting ≤ 4, complexity ≤ 10 on touched code; a new file ≤ 800 lines; **never grow `service.py`, `auto.py`, `tool_router.py` or `smart_memory.py`** (FX-007 makes `service.py` shorter, by deleting); a new helper goes in a new small module.
- Config only via `orchestrator/config.py` (new keys also in `reports/config-surface.json`); no hardcoded values; tenant isolation on every query; both editions; no migration; no new dependency.
- Frontend changes go through a seam OUTSIDE `Message` (components/chatbot/message.tsx) and `useChat` (lib/chat/hooks.ts): those functions are already over the limit and the changed-lines gate fails on any line added inside them (FX-001 says how).
- **Delete what you replace** (CLAUDE.md §5): a family retired in FX-007 is removed in the same commit, its tests moved to the receipts rule.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later".
- **A code review runs in the FOREGROUND, before the commit,** for **FX-003** (the gate order), **FX-007** (tests rewritten), **FX-011** (an agent's send under the click) and **FX-015** (memory ownership); use the code-reviewer agent and wait for it.
- **RE-VERIFY every anchor by grep.** **Order is the design.**

## Tests: CI is the gate, once per wave

Nothing runs on the owner's machine (no pytest, npm, tsc, docker, server, browser, eval runner, installs). Local checks only: `python3 -m py_compile`, `ruff check` if on PATH, `python3 orchestrator/scripts/check_hierarchy_gate.py`, `cd frontend && node scripts/check-changed-lines-eslint.js origin/main` when you touched the frontend, and `bash scripts/ralph/acceptance-prd256fix.sh --no-ci`. You write the tests (`orchestrator/tests/test_prd256_fix_*.py`, `tests/sim/tests/`); CI runs them once at the end. A story never waits on CI. `→ OWNER:` ACs stay as they are.

## Commits

- `git commit -s`, messages `fix(prd-256): description` (or `test(prd-256):` for a test-only change), every story commit ending ` [skip ci]`; explicit paths only; never `git add -A`; never `git stash`.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing anywhere except `origin feat/prd-256-fix-night12`. Never touch `main`, the Wave 1 or Wave 2 branches or `test/customer-night`.
- NO running the eval, NO reading or writing `~/.automatos-analyst` or `~/.automatos-sim`; the model comparison and the nights are the owner's.
- NO regex on the owner's words as a stand-in for a click (D1); NO new regex claim family (D10); NO failover model set by default (D5); NO Jev live (D4).
- NO deleting a test without a replacement that covers the same sentence (FX-007); NO weakening, skipping or deleting a CI check; NO loosening a strict schema beyond what a story names.
- NO `os.getenv`/`os.environ` outside `config.py`; NO hardcoded values; NO new dependency; NO migration; NO edit to the generated seed (`core/seeds/platform-management-skill.md`), the Composio deny list or any guardrail; NO change to the eval runner.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests; local checks; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as the DONE mark) `→ DONE — <evidence>` in `scripts/ralph/prd-256fix.json` (`→ OWNER:` ACs stay). Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <FX-id>`. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only: never write them anywhere else in a reply, not even in a sentence about them** (an FX-004 session did, and the runner ended the wave early).

## Completion

- **All ACs DONE** (`→ OWNER:` excepted): run `bash scripts/ralph/acceptance-prd256fix.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-256): night-12 fix wave built, run CI`), and reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the evidence (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.

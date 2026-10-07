# Ralph Build Prompt — PRD-256 Auto, receipts not narration, Wave 2 (the model, Auto always answers, one hand-off table, the families deleted)

You are executing **PRD-256 Wave 2**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-256-w2-model-lanes`**, in its own worktree. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

**The base is Wave 1.** While Wave 1 is not on `main`, this branch is **stacked on** `feat/prd-256-w1-receipts-gates`; once it is, the branch is cut from `main`.
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd256w2.sh --print-base` prints.
- **Never merge anything into this branch**: not `main`, not the Wave 1 branch. Integration is the owner's job.

Wave 2 builds, in the JSON's priority order:
1. **US-008** the Claude-ready LLM manager: no sampling params for Claude 4.6+ ids, the price table, an explicit (empty) failover.
2. **US-009** the one-constant default model for the Auto seeds (the four-arm run itself is the owner's).
3. **US-010** Auto always answers; a named agent gets a ticket; the DELEGATE lane removed (Decision D2).
4. **US-011** one hand-off table instead of four lanes.
5. **US-012** delete the families, only after the owner's note on US-009 shows the bar is met.
6. **US-013** context assembly under two seconds, only if the owner's note says the median prep_ms is above 2 s (Decision D3).

## Read first, every iteration

1. `scripts/ralph/prd-256w2.json` is the BINDING contract. Its **`decisions`** are binding (D2 Auto always answers; D3 US-013 is conditional; D4 no Jev; D5 no failover model by default; D6 the ATOM lane stays; D9 one PR per wave; D11 stacked on Wave 1, never merged).
2. `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` (FR-9..FR-13, US-008..US-013).
3. `CLAUDE.md` and `AGENTS.md`.
4. **Already on main, never rebuilt:** `core/llm/defaults.py` (`DEFAULT_LLM_MODEL`, `get_default_model_config`) read by the Auto seed — US-009's only build is its pin test. **What Wave 1 built** (grep the base to confirm): `consumers/chatbot/receipts.py`, the receipts part and frame, the honesty rule, `modules/tools/discovery/owner_only.py`, the promoted first-class tools, `docs/testing/AUTO-EVAL.md`.
5. **The owner's notes on US-009** in the JSON (`DEFAULT=<model>`, `PREP_MS_MEDIAN=<ms>`, the claims-backed rate): US-012 and US-013 read them before starting. A missing note BLOCKS those two stories without stopping the wave: mark every criterion of the story `→ BLOCKED — no US-009 note yet` (or the note's text when it is below the bar) in the JSON, commit that alone, reply `STORY_DONE <US-id>`, and the loop goes on. `RALPH_BLOCKED` is for a Hard NO only: it stops the wave, so no PR and no CI.

## The code this wave builds on (RE-VERIFY each by grep before building on it)

- The LLM clients: `orchestrator/core/llm/manager.py` (the price table `_MODEL_COST_ENTRIES`, `estimate_cost_usd`), `core/llm/clients/base.py` (the per-model sampling check), `anthropic_client.py`, `openai_compatible_client.py`, `openai_chat_request.py`, `core/llm/defaults.py` (`DEFAULT_LLM_MODEL`), `core/llm/anthropic_ids.py`, `config.py`, `reports/config-surface.json`, `consumers/chatbot/turn_errors.py`.
- The dispatch: `orchestrator/api/chat.py` (the lane branch after `AutoBrain.assess`; `UniversalRouter`), `consumers/chatbot/auto.py` (the rubric `_ASSESSMENT_RUBRIC`, `_ASSESSMENT_OUTPUT`, `Action`, the Redis cache `_cache_store`/`_cache_lookup`, `apply_assign_bias`), `core/routing/engine.py`.
- The lanes: `consumers/chatbot/brand_assign_lane.py`, `brand_to_the_designer.py`, `paperwork_to_the_team.py`, `named_template_note.py`, `figure_disputes.py`, `shop_figures.py`, `team_corrections.py`, `needs_you_turn.py`, `team_findings.py`, `board_questions.py`, `document_conversation.py`.
- The families: `consumers/chatbot/claim_check.py`, `modules/tools/execution/action_claims.py`, `document_claims.py`, `shop_and_team_claims.py`, `tool_loop.py` (`_recover_claimed_action`), `nudges.py`.
- The context: `orchestrator/modules/context/` (the ContextService and its sections), `consumers/chatbot/service.py` (`_retrieval_first` and its decorators, `_prepare_messages`).
- The seeds: `orchestrator/core/seeds/seed_auto_agent.py`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, nesting ≤ 4, complexity ≤ 10 on touched code; a new file ≤ 800 lines; **never grow `service.py`, `auto.py`, `tool_router.py` or `smart_memory.py`**: US-010 and US-011 make those files shorter, by deleting.
- Config only via `orchestrator/config.py` (`LLM_FAILOVER_MODEL` is the one new key, in `config-surface.json`); no hardcoded values; tenant isolation; both editions; no migration; no new dependency.
- **Delete what you replace** (CLAUDE.md §5): a lane folded into the table is removed in the same commit, its tests moved.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later".
- **A code review runs in the FOREGROUND, before the commit,** for **US-010** (who answers the owner) and **US-012** (tests rewritten); use the code-reviewer agent and wait for it.
- **RE-VERIFY every anchor by grep.** **Order is the design.**

## Tests: CI is the gate, once per wave

Nothing runs on the owner's machine (no pytest, npm, tsc, docker, server, browser, eval runner, installs). Local checks only: `python3 -m py_compile`, `ruff check` if on PATH, `python3 orchestrator/scripts/check_hierarchy_gate.py`, and `bash scripts/ralph/acceptance-prd256w2.sh --no-ci`. You write the tests (`orchestrator/tests/test_prd256_*.py`); CI runs them once at the end. A story never waits on CI. `→ OWNER:` ACs stay as they are.

## Commits

- `git commit -s`, messages `type(prd-256): description`, every story commit ending ` [skip ci]` (D9); explicit paths only; never `git add -A`; never `git stash`.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing anywhere except `origin feat/prd-256-w2-model-lanes`. Never touch `main`, the Wave 1 branch or `test/customer-night`.
- NO running the eval, NO reading `~/.automatos-analyst`; the four-arm run and the default-model decision are the owner's.
- NO failover model set by default (D5); NO Jev live anywhere (D4); NO new lane module; NO new regex claim family.
- NO deleting a test without a replacement that covers the same sentence (US-012); NO weakening, skipping or deleting a CI check.
- NO `os.getenv`/`os.environ` outside `config.py`; NO hardcoded values; NO new dependency; NO migration; NO edit to the generated seed, the Composio deny list or any guardrail.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE; for US-012 and US-013 read the owner's US-009 note first and follow its decision rule.
2. Implement it with its tests; local checks; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as the DONE mark) `→ DONE — <evidence>` in `scripts/ralph/prd-256w2.json` (`→ OWNER:` ACs stay; a skipped US-013 criterion is marked `SKIPPED — prep_ms under 2 s`; a story whose owner note is missing or below the bar is marked `→ BLOCKED — <why>` on every criterion). Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>`.

## Completion

- **All ACs DONE, SKIPPED or BLOCKED per the decision rule** (`→ OWNER:` excepted): run `bash scripts/ralph/acceptance-prd256w2.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-256): Wave 2 built, run CI`), and reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the evidence (this stops the wave). A missing owner note is NOT that case: mark the story BLOCKED and go on.

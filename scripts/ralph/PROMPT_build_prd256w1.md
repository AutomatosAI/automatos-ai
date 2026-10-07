# Ralph Build Prompt — PRD-256 Auto, receipts not narration, Wave 1 (receipts, honesty, gates, tool contracts, the instrument)

You are executing **PRD-256 Wave 1**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-256-w1-receipts-gates`** ← `main`, in its own worktree. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

**The thesis (owner, 4 Oct): Auto is the one voice; it delegates, agents do the work.** Eleven customer nights found Auto's worst habit: claiming work it did not do, most often after a call that FAILED on its arguments and was reported as done, then a call to the wrong surface, then no call at all. The review (`docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` §1) found the regex claim guard catches 12 of 22 real sentences and denied three real writes. This wave makes the platform write its own account of every turn (receipts), gates owner-only actions on the owner's click, gives Auto's writes strict contracts, and documents the instrument.

Wave 1 builds, in the JSON's priority order:
1. **US-001** the receipts part: built from the tool tracker after the loop, one frame, one saved part, shown above the text, DELEGATE turns too.
2. **US-002** one honesty rule from receipts; the families frozen.
3. **US-003** memory takes receipts.
4. **US-004** the owner's click on owner-only actions through the existing grant card (Decision D1 is the list).
5. **US-005** done needs an artifact.
6. **US-006** tool contracts: Auto's write actions promoted to first-class tools with strict schemas; a refused write is reported refused.
7. **US-007** the instrument's docs page (the eval itself is the owner's; you never run it).

## Read first, every iteration

1. `scripts/ralph/prd-256w1.json` is the BINDING contract. Its **`decisions`** are binding (D1 the owner-only list; D7 the loop never runs the eval; D8 receipts come from the tracker, never from tool_execution_logs; D9 one PR per wave; D10 the families are frozen, not deleted, in this wave).
2. `docs/PRDS/PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md` is the spec (FR-1..FR-8 bind this wave; §6 and §7 name the seams).
3. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, canonical terms (Deliverable, Task, Auto), no `os.getenv` outside `config.py`, no hardcoded values, data lives in the database.

## The code this wave builds on (RE-VERIFY each by grep before building on it)

- The turn: `orchestrator/consumers/chatbot/service.py` (`_stream_tool_loop` ~L1679, `_answer_additions` ~L1666, the save ~L3151, `_post_response` ~L1459); `narration.py` (`reply_parts`); `streaming.py` (`StreamingHandler`, `format_aisdk_data`).
- The ground truth: `orchestrator/modules/tools/execution/tool_execution_tracker.py` (`outcomes`, `succeeded`, `failed`, `refused`), `call_effects.py` (`call_effects`, `call_params`, `result_effects`, `refused_effects`), `tool_loop.py` (`ToolLoopExecutor.run`, `_execute_round`, `record_outcome`), `nudges.py`.
- The guard being frozen: `consumers/chatbot/claim_check.py`, `modules/tools/execution/action_claims.py`, `document_claims.py`, `shop_and_team_claims.py`.
- The gate: `modules/tools/discovery/platform_executor.py` (`execute`, the confirmation ask ~L862–L925, `Cleared`), `modules/tools/execution/tool_grants.py` (`attach_ask_grant`, `consume_tool_grant`), `services/board_consent.py` (`actor_from_user_id`), `modules/tools/discovery/ticket_moves.py` (`_send_back`, `_keep_the_note`), `follows_the_owner.py`, `owner_turn.py`, `card_words_said.py`; the frontend card `frontend/components/chatbot/chat.tsx` (`tool_approval`).
- Done needs an artifact: `modules/tools/discovery/handlers_board_task_done.py` (`update_board_task_status`, `moved_to_done`), `api/board_tasks.py` (`approve_task`).
- The tool surface: `modules/tools/tool_router.py` (`_first_class_names`, `_promotion_pins`, `get_tools_for_agent_async`, `_narrow_dispatcher_actions_async`), `modules/tools/discovery/action_registry.py` (`to_first_class_schemas`, `to_dispatcher_schema`), `modules/tools/README.md` (the 3-file pattern), `orchestrator/scripts/check_hierarchy_gate.py`, `config.py` (`TOOL_ROUTING_PROMOTION_PINS`).
- The frontend: `frontend/types/chat.ts` (`MessagePart`), `frontend/lib/chat/hooks.ts` (the frame switch), `frontend/components/chatbot/message.tsx`, `activity-trail.tsx`, `frontend/lib/chat/narration.ts`.
- Memory: `consumers/chatbot/integration.py` (`store`), `smart_orchestrator.py` (`store_exchange`), `smart_memory.py`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Functions:** a Python function you touch is at most **50 code lines** and nests at most **4 levels**; a React component at most 150 lines; cyclomatic complexity at most 10. Touching an existing function already over a limit means splitting it, not adding to it.
- **Files:** a new file at most 800 lines (aim 200–400); **never grow a file already over 800 lines** — `service.py` (3,268), `auto.py` (1,533), `smart_memory.py` (1,100) and `tool_router.py` (2,148) are over: every new behaviour is a new small module, and the change in those files is a call.
- **Platform actions keep the 3-file pattern:** `ActionDefinition` inside `registry.register(...)` in an `actions_<domain>.py`; the hierarchy gate reads registrations with `ast` and exits 2 on one outside it. Promoting an action changes its exposure, never its definition's home.
- **Config only via `orchestrator/config.py`**, a new name in `orchestrator/reports/config-surface.json` in sorted position. No `os.getenv` / `os.environ` elsewhere.
- **No hardcoded values:** limits and labels are named constants or config.
- **Tenant isolation:** every query is scoped to the caller's workspace; a receipt's link never crosses a workspace.
- **Routes:** authenticated with the existing dependencies; a database-touching route is a plain `def`; a changed route stays in the committed `orchestrator/reports/route-manifest.json`.
- **Frontend:** calls only through `apiClient`; no new npm dependency; a message with no receipts part renders as today.
- **Both editions** keep working; anything hosted-only is gated by `AUTH_EDITION` / `isSaaS`, never by role.
- **No migration is expected.** The receipts part is JSON on `messages.parts`.

## The execution contract

- **This session ends when your turn ends.** Nothing reports back later.
  - Never end your turn while anything runs in the background (an agent, a background shell).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports".
  - **A code review runs in the FOREGROUND, before the commit,** only for **US-004** (permissions and a guardrail) and **US-006** (the tool surface and the hierarchy gate); use the code-reviewer agent and wait for it.
  - Commit as soon as the story's code and tests are written.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow `priority` in the JSON.

## Hard rules for this wave

- **The model never writes the receipts.** They are built after the loop from `tracker.outcomes`; no prompt mentions the part; no tool can set a receipt.
- **A line about what was not done comes from receipts alone** (US-002). No new regex family, no new sentence pattern beyond the one generic completed-action pattern. The families' counts are pinned by a test.
- **Owner-only actions ask first** (US-004) through the existing grant path. No second approval mechanism, no new table.
- **A refused write is reported refused** (US-006), by the nudge and the rules block, never papered over.
- **Nothing here reads `tool_execution_logs` for receipts** (D8).

## Tests: CI is the gate, once per wave

**On the owner's machine nothing runs**: no pytest, npm, tsc, docker, compose, server, browser or the eval runner, and no `pip install` / `npm install`. The runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at a dead port, on purpose. `python3 -m py_compile` on changed Python files, `ruff check` on changed paths (if ruff is on PATH), `python3 orchestrator/scripts/check_hierarchy_gate.py` (stdlib) and `bash scripts/ralph/acceptance-prd256w1.sh --no-ci` are the only local checks.

- **You write the tests. CI runs them, once, at the end of the wave.** Backend: `orchestrator/tests/test_prd256_*.py`. Frontend: vitest under `frontend/components/chatbot/__tests__/`. **A test must fail without your change.** No test calls a real model, a real provider or the database outside the CI fixtures.
- **A story NEVER waits on CI.** No `gh run watch`, no `gh run list`, no polling of `test.yml`. A story's ACs never include "CI green"; the CI criterion reads "pushed; CI runs once at the end of the wave".
- **An AC marked `→ OWNER:` stays as it is.** It is the owner's check: never mark it DONE, never claim it, never run the eval, a browser or a live session.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO. Messages: `type(prd-256): description`.
- **Every story commit message ends with ` [skip ci]`** (Decision D9). The runner pushes the branch as a backup after each session; the branch has no pull request until the wave is complete, so no CI runs on those pushes.
- **STAGING DISCIPLINE:** stage explicit paths only. NEVER `git add -A`, `git add .` or `git add -u` (`node_modules` is untracked and NOT gitignored). NEVER `git stash` (shared with every worktree).
- **Binary files:** none. Test fixtures are text.

## Hard NOs

- NO merging, NO PRs (the runner opens the wave's draft PR when the build is complete; the owner merges), and NO pushing anywhere except `origin feat/prd-256-w1-receipts-gates`. Never touch `main` or `test/customer-night`.
- NO running the eval, NO reading `~/.automatos-analyst`, NO results line written anywhere (D7).
- NO new regex claim family, NO new lane module under `consumers/chatbot/` other than `receipts.py` and, for US-004, `modules/tools/discovery/owner_only.py`.
- NO deletion of the families in this wave (D10); NO weakening, skipping or deleting a test or a CI check.
- NO edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md`, the Composio deny list, or any guardrail other than the owner-only gate US-004 adds.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), NO hardcoded values, NO new Python or npm dependency, NO migration.
- NO growth of `service.py`, `auto.py`, `tool_router.py` or `smart_memory.py` beyond the call that wires the new module.

## Per-iteration protocol

**Owner, 6 Oct: ONE CI run per wave.** Do NOT wait for CI on a story. Never run `gh run watch` or poll `test.yml`.

1. Pick the first story (by `priority`) with ACs not yet DONE, and re-verify its anchors by grep.
2. Implement it with its tests. Run `python3 -m py_compile` on the changed Python files, `ruff check` on the changed paths if ruff is on PATH, and `python3 orchestrator/scripts/check_hierarchy_gate.py` when a tool or action changed. Commit (signed, ` [skip ci]`).
3. Mark its ACs `→ DONE — <evidence: the test names and files>` in `scripts/ralph/prd-256w1.json` (the `→ OWNER:` ACs stay). Commit that alone (signed, ` [skip ci]`): `chore(prd-256): <US-id> ACs DONE [skip ci]`.
4. **STOP.** One story per session. Unless that was the last story, end your reply with one line, `STORY_DONE <US-id>`. Do not start the next story.

## Completion

- **All ACs DONE** (the `→ OWNER:` ACs excepted; this session finished the last story): run `bash scripts/ralph/acceptance-prd256w1.sh --no-ci` and fix what is red in signed ` [skip ci]` commits. Then make ONE signed commit WITHOUT `[skip ci]` (`chore(prd-256): Wave 1 built, run CI`), and reply `RALPH_COMPLETE`. The runner pushes it, opens the wave's draft PR (that starts the wave's one CI run), runs the gate against that run, then the review.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.

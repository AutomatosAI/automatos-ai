# Ralph Review Prompt — PRD-268 W3 Follow-ups, Auto keeps its word

You are a fresh-context **adversarial reviewer**. The build claims PRD-268 wave W3 is complete (US-C11a–e, US-C12a–c, US-C13a–c, US-C22a–b, US-W03a–c, US-W06). This wave lets Auto schedule background work for a person, ends turns early, writes memories on a click and sends the titles of a person's work to the model. Review it as someone trying to make a follow-up act, chain, retry, leak to another member or beat its caps; make chips cost a second model call or appear on the brief; commit a gold set; or make calibration read a body, show a percentage, run without consent or write a memory nobody confirmed.

## What to look for

**Follow-up rows and privacy (US-C11a, C11b):**
- a follow-up created through `create_task` or `platform_schedule_task`;
- `DELIVERY_MODES` widened;
- a follow-up visible on the calendar, in `list_tasks`, the scheduled-tasks API, `platform_list_scheduled_tasks`, the board snapshot or W1's skipped-run signal;
- a follow-up counted toward Auto's `MAX_TASKS_PER_AGENT` or the operator cap;
- a generic route or `platform_cancel_scheduled_task` able to pause, cancel or delete one;
- the scheduler's reconcile no longer registering one;
- `list_tasks` still over 50 code lines, or its return changed.

**The tools and the caps (US-C11c):**
- the person, the chat or the agent taken from a model argument;
- a caller-supplied `_follow_up` surviving;
- a schedule outside a person's own chat turn, from an agent other than Auto, in shadow or off, or while paused;
- six parallel schedules making six rows (the lock not taken, or taken per person while the workspace cap needs the workspace);
- counts and insert in different transactions;
- the audit row written inside the locked transaction;
- "use a watch" missing for a Mission, a playbook run or a ticket number;
- any of the four chat-only tools offered or cleared in a check-in;
- `_run_cleared` or `clear` changed; `platform_executor.py` grown;
- a write tool without its hierarchy-gate entry.

**The firing (US-C11d):**
- a follow-up acting (any non-read call cleared in its turn);
- a follow-up's turn scheduling another (a chain);
- a failed follow-up retried or re-registered;
- a post into a chat that is not the person's own;
- the reason or the chat's messages followed as instructions, or posted as the person's message;
- a suggestion from a follow-up addressed to the owner instead of the person;
- a firing for a person who left the workspace, or with the workspace off, shadow or paused;
- `execute_task` or `_trigger_agent_chat` changed;
- `scheduled_task_service.py` over 800 lines.

**Reply chips (US-C12a, C12b, C12c):**
- a second model call after a successful `platform_offer_replies` round that had text;
- chips outside 2–3 replies, 1–3 words, ≤ 24 characters;
- chips on a turn that called `platform_get_brief`, or offered in a check-in or a follow-up turn;
- the tool labelled anything but read, or writing anything;
- `ToolLoopExecutor.run` or `_execute_round` changed; `tool_loop.py` grown;
- `consumers/chatbot/service.py` touched;
- the text saved twice, or repeated in a narration part, or shown twice on the wire;
- chips stored in `source` instead of a `reply_chips@1` part;
- a plain turn's saved parts changed;
- chips spoken on a spoken turn;
- a widget visitor sent the frame.

**Follow-up routes (US-C11e):**
- another person's or another workspace's follow-up listed or cancelled (404, never 403);
- a route that is `async def` over the sync session, or missing from the manifest.

**Evals (US-C13a–c):**
- any gold case committed (a tracked `*.jsonl` under `orchestrator/evals/proactive`, or gold data inlined in a test);
- `.gitignore` not covering them;
- a runner reading its path from the environment instead of `config.py`;
- a runner that doesn't print its bar or doesn't exit non-zero on a miss;
- a bar weaker than §6.7;
- `print` outside `evals/proactive/__main__.py`;
- a model call with an unclassified purpose;
- a runner that writes (the silence eval recording real suggestions, precision writing anything).

**Calibration (US-C22a, C22b):**
- `CALIBRATION_ENABLED` defaulting on, or any route but GET answering while off;
- discovery listing `calibration` while off;
- anything read before consent, or any read of a body, content, chunk or file;
- another person's conversations, or another workspace's rows, among the sources;
- a percentage or a confidence number reaching a line;
- a line citing a source it wasn't given;
- more than one model call, or one on an empty calibration or a decline;
- offered twice (also under two concurrent POSTs);
- a failed build keeping the claim;
- the card posted into another member's chat;
- memory writes not through `store_long_term` and `build_memory_metadata`, without the owner tag, not private by default, or skipping `violates_exclusions`;
- accept writing a rejected line, or writing twice;
- `calibration@1` not in W2's part allow-list.

**The web (US-W03a, W03b, W06, W03c):**
- `message.tsx`, `chat.tsx` or `multimodal-input.tsx` touched;
- chips under any message but the newest, or still shown after a newer one;
- a tap sending twice, or while a turn streams;
- `data-ai-generated` missing on chips or calibration lines, or put on controls;
- a raw `fetch('/api…')`; `lib/api-client.ts` grown;
- the consent line shown when calibration is unavailable, or after it was answered;
- a percentage on the card;
- an SDK name, field, area or version changed (K17);
- Studio rules broken (orange, hex, glass, rounded-3xl, a second primary);
- a new npm dependency.

**The migration (US-C11a):**
- the revision doesn't chain onto W2's head;
- a step without IF EXISTS / IF NOT EXISTS;
- a fifth change beyond K6's four;
- the downgrade unsafe;
- a pin didn't move, or `test_prd268_proactive_models.py` lost a test;
- the calibration model and the DDL differ.

**The wave as a whole:**
- a story's evidence doesn't exist;
- an AC is marked DONE that the code doesn't meet;
- a Go, kit or W4 piece was built (K26);
- `orchestrator/services/push/` changed;
- a `*268w1*` or `*268w2*` kit file changed;
- a file over 800 lines grew (the K3 exceptions aside);
- a K20 pin file lost a test;
- the PR undrafted before this review passed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd268w3.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is W2's tip while W2 is unmerged (the wave is stacked): never file a finding against W2's own diff. Never file a finding against a line this diff doesn't touch, except where an AC says the base's behaviour had to change and it didn't.

Read beside the diff:
- `scripts/ralph/prd-268w3.json` (binding: its `decisions` and each story's ACs and `notes`);
- `docs/PRDS/PRD-268-PROACTIVE-AUTO.md` §6.3, §6.6, §6.7, §6.11.5, §6.12 and §12 (Q4, Q9, Q22, Q26);
- `AGENTS.md` and `frontend/DESIGN_SYSTEM.md` §22;
- the `ownerTest` list.

## Verification

- **code-review** skill, or the code-reviewer agent, on the diff, **in the foreground**.
- **security-reviewer** agent, in the foreground, on:
  - `orchestrator/services/proactive/follow_ups.py`, `follow_up_fire.py`, `follow_up_prompts.py` and `modules/proactive/follow_up_caps.py`;
  - `orchestrator/modules/tools/discovery/{actions,handlers}_follow_ups.py`, `follow_up_context.py`, `check_in_lane.py` and `handlers_proactive.py`;
  - the follow-up exclusions in `services/scheduled_task_service.py`, `services/activity_scheduled_task_items.py`, `services/proactive/signals.py` and `handlers_analytics.py`;
  - `orchestrator/api/proactive_follow_ups.py`;
  - `orchestrator/modules/tools/discovery/{actions,handlers}_replies.py`, `modules/tools/execution/reply_chips_turn.py`, `consumers/chatbot/turn_parts.py` and `consumers/chatbot/receipts.py`;
  - `orchestrator/services/calibration/` and `orchestrator/api/auto_calibration.py`.

  Ask it explicitly about:
  - tenancy, and a person's own rows (404, never 403);
  - the lane: can a follow-up's turn write, schedule, or reach a Composio or plugin tool;
  - the advisory lock and the caps under concurrency;
  - model-written text (a reason, a chat message, a title) reaching a prompt as instructions, or a person's chat as their own words;
  - consent before any read, and what calibration sends to the model;
  - the memory writes: owner, scope, exclusions, idempotence.
- **database-reviewer** agent, in the foreground, on `orchestrator/alembic/versions/prd268_w3_follow_ups.py`, the model changes, the follow-up counts and lock in `services/proactive/follow_ups.py`, and the calibration claim in `services/calibration/claim.py`.
- Run `bash scripts/ralph/acceptance-prd268w3.sh` yourself. It waits for the wave's one CI run and the browser lane on the tested commit, and reads the PR body and its base.
- Spot-check five marked acceptance criteria at random against the code and the tests. Always include one from US-C11c, one from US-C12b and one from US-C22b.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

**No CRITICAL, HIGH or MEDIUM:** write a 7-line summary:
- (a) follow-ups: the row, privacy, the lock and caps;
- (b) the firing: the lane, the three endings, no chain, no retry;
- (c) reply chips: the tool, the terminal turn, the saved parts;
- (d) the evals: runners, bars, no gold committed;
- (e) calibration: consent, titles only, no percentages, memory writes, shipped off;
- (f) the web surfaces and the browser lane;
- (g) the owner's next step: the `ownerTest` list (images.yml first), then undraft the PR (`gh pr ready`).

Then `REVIEW_PASS` as the last line. REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose. You never undraft the PR.

**Findings:**
1. Append `P268W3-RVW-n` stories to `scripts/ralph/prd-268w3.json`, n counting from 1. Same shape: id, title, description, acceptanceCriteria (with the cause, the change and the test), priority after the last story, passes false, notes.
2. Commit with `git commit -s -m 'chore(prd-268): W3 review findings → fix stories [skip ci]' -- scripts/ralph/prd-268w3.json`.
3. `REVIEW_FINDINGS` as the last line.

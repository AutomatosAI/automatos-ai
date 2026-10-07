# PRD-256: Auto, receipts not narration: receipts, gates, tool contracts and the model test

**Status:** Decided (§9 answered 7 Oct), awaiting build · **Owner:** Gerard · **Written:** 7 Oct 2026 (the Auto review session, from `.claude/AUTO-REVIEW-FINDINGS.md` and TESTER's review of it)
**Type:** Extension plus consolidation. Receipts, gates and contracts extend seams that exist (the tool tracker, the grant card, the first-class tool surface, the sim harness); the regex claim families and four routing lanes are consolidated and deleted once the receipts rule holds.

## 1. Introduction

Auto is the one voice the owner talks to (product thesis, 4 Oct). Across eleven customer nights (18 Sep to 7 Oct) it has been the weakest part of the product every time, and its worst habit is claiming work it did not do: 13 incidents, 24 traced in the review. The traces say three things the fixes so far have not addressed:

- **The platform never makes its own record of a turn.** Tool calls ride the stream and vanish on reload (`frontend/lib/chat/hooks.ts:325-373`); the saved message is reasoning, narration and text (`consumers/chatbot/narration.py:37-47`, `service.py:3150-3161`); memory distils the text (`service.py:1492`). The only defence is `claim_check.py` plus the regex families in `modules/tools/execution/action_claims.py`, `document_claims.py` and `shop_and_team_claims.py`: one night behind every new surface (12 of 22 real false sentences caught today), blind to a call made on the wrong surface (a `platform_list_social_posts` call backs "I've posted it"), and wrong in the other direction too: in three of the four firings the nights recorded, the guard denied a write that had succeeded (F319, F337, F363).
- **The commonest false claim is a call that failed on its arguments and was reported as done** (F108: `platform_approve_mission` with `params: {}`, "Missing required params: mission_id", then "I've approved the mission. It's now running"; F132, F133, F266, F288; night 10b's phantom documents, where `data` arrived as a JSON string that failed to parse on 4 of 55 calls and "Auto reported success every time regardless"). The second is a call to the wrong surface (a copy made instead of the thing run or updated: F185, F231, F289; a cancel that became an approval: F280). The same model lands 6 of 7 when the tool contract is spelled out (the studio's "Use with Auto" prompt) and about 1 in 7 when it has to guess (night 10b).
- **Owner-only actions have no gate and run in the owner's name.** Every chat call carries `driving_user_id` (`service.py:213-216`), a chat-side approval is written as the owner's (`ticket_moves.py:152-160`, `board_consent.actor_from_user_id`), board and agent actions are `requires_confirmation=False` (`actions_board_tasks.py:71,338,416,445`), the policy plane ships off (`config.py:952-957`). The night-8 fix, `follows_the_owner.py`, is a regex over the owner's words.

Behind them sits routing: the Tier-3 rubric says most work "delegates" and 72% of its verdicts do, so a specialist with its own persona and tools answers the owner's chat (`api/chat.py:463-498`); eleven regex lanes inject prose; and Auto runs Gemini 2.5 Flash at temperature 0.7 with ~23k-token prompts, never compared with anything (night 1 ran on gpt-5.4 and was the most honest night; the switch was a cost decision).

This PRD does four things, in two waves:

1. **Receipts.** The platform writes "What I did" from the tool tracker, before the text, saved with the message, distilled into memory. The model cannot write it.
2. **Gates.** Owner-only actions from chat wait for the owner's click (the existing grant card); "done" needs an artifact.
3. **Tool contracts.** Auto's write actions become first-class tools with strict schemas; a refused write is reported as refused.
4. **The instrument and the model.** An offline eval set replayed by a runner that switches and restores Auto's model per arm, run as a baseline *before* wave 1 and again after each wave; then the four-arm model test and the lane collapse.

The baseline (arms A and C on today's build, about $6) is the "before" score for everything below and needs only Gerard's OK.

## 2. Goals

- A reply can no longer be the only account of a turn: every assistant message carries a `receipts` part built by the platform, shown first, surviving reload, and fed to memory.
- "Auto claimed work it did not do" becomes impossible to read without the receipt beside it, and impossible to inherit: the next turn and memory read receipts, not prose.
- No owner-only action (approve, cancel, agent and tool changes, settings, publish, send) runs from chat without the owner's click.
- "Done" on a card needs the artifact that makes it done.
- Failed and wrong-surface calls fall from the commonest shape to the rarest: strict contracts for Auto's writes, and a reply after a refused write that says so.
- Auto's model is chosen by measurement on the same 65 asks, with pass bars set before the run, and the measurement becomes a standing per-date, per-model table.
- Auto is the one voice: a specialist never takes over the chat; work goes to agents on tickets.
- `service.py` stops growing (3,268 lines today); the regex claim families and four routing lanes are deleted once the receipts rule and the eval show they are not needed.

## 3. User stories

House rules for every story: one commit per story, one PR per wave, one CI run per wave; nothing runs locally; both editions keep working; no new table, no new dependency. US-003 and US-004 touch permissions and guardrails: open the issue for Gerard's sign-off before building.

### Part 1 (wave 1): Receipts and honesty

#### US-001: The receipts part
**Description:** As an owner, I want every reply to start with what Auto actually did this turn, written by the platform from the calls that ran, so I never have to take the text's word for it.

**Acceptance criteria:**
- [ ] A new module `orchestrator/consumers/chatbot/receipts.py` builds `receipts` from `executor.tracker.outcomes` (`modules/tools/execution/tool_execution_tracker.py:72-81`, with `call_effects`): one entry per call, `{action, kind: read|write, status: done|refused|skipped, subject, effect, link, reason}`. The subject is the thing by number or name (card #0422, agent "Social Media Director", document title, post title); the effect comes from `call_effects` (`…:done`, `…:cancelled`, `…:agent_set`, `send_back`) in plain words ("moved to Done", "sent back to its agent"). A refused call carries the executor's reason, trimmed.
- [ ] The automatic reads (retrieval first, the Needs-you read, the team-findings read, `prefetched`) are folded into one `read` entry ("read your documents and the board").
- [ ] After the loop, `_stream_tool_loop` emits one frame `d:{"type":"receipts","data":{"receipts":[…]}}` through the existing `StreamingHandler` (`consumers/chatbot/streaming.py`), before the answer's additions.
- [ ] `reply_parts` (`narration.py:37`) gains the part `{"type": "receipts", "receipts": […]}`; it is saved with the message at `service.py:3151`; a turn with no calls saves `receipts: []`.
- [ ] Every turn through `stream_response_with_agent` gets receipts, DELEGATE turns included, and so do the first-reply path and the forced-synthesis path.
- [ ] The frontend (`frontend/types/chat.ts` `MessagePart` union, `frontend/lib/chat/hooks.ts`, `frontend/components/chatbot/message.tsx`) renders the receipts block **above** the reply text: live from the frame, on reload from the part. Empty receipts render "No actions in this turn." A refused write renders "tried, refused: <reason>". The block reuses `ActivityTrail`'s chip layout (`frontend/components/chatbot/activity-trail.tsx`).
- [ ] The model never sees the receipts block and cannot write it (it is built after the loop, from the tracker, and the part type is not in any prompt).
- [ ] Tests: `orchestrator/tests/test_prd256_receipts_part.py` with a fake loop (a store_memory-only turn → one write receipt, "No actions on the board"; a no-call turn → `[]`; a refused `generate_document` → status refused with the reason; a `platform_update_task_status` to done → subject "#0422", effect "moved to Done"); a vitest for the component (live frame, reload from the part, empty state).
- [ ] `service.py` does not grow: the builder and the frame live in the new module; `ruff`, the code-shape gate and the hierarchy gate pass.

#### US-002: One honesty rule from receipts
**Description:** As an owner, I want the "I haven't done that" line to come from the receipts, not from a vocabulary of phrasings, so it fires when nothing ran and never when something did.

**Acceptance criteria:**
- [ ] `_answer_additions` (`service.py:1666`) takes the receipts. When no `write` receipt has status `done` and the answer has a first-person completed-action sentence (one generic pattern: "I've/I have <past-tense verb>", "it's now on your board", "has been <verb>", "you should now see"), the line "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done." is placed **above** the text, not below it. When a write receipt is `done`, no line fires, whatever the families say.
- [ ] A refused write gets its own line: "I tried to <action> and it didn't go through: <reason>."
- [ ] `claim_check.py` keeps tier 2 (ids that do not exist in the workspace) and drops tiers 1 and 3 from the correction path; the families in `action_claims.py`, `document_claims.py` and `shop_and_team_claims.py` are **frozen** (a test asserts their pattern count does not grow) and keep driving only the in-loop nudge until US-012 removes them.
- [ ] The 22-sentence replay from the review (section 2.1) passes 22/22 under the receipts rule; the three false denials (F319, F337, F363) do not fire because a write succeeded; the F319 "a call that did it is never denied" tests still pass.
- [ ] Tests: `test_prd256_honesty_rule.py` with the 25 cases (22 plus the three false denials), driven by receipts.

#### US-003: Memory takes receipts
**Description:** As an owner, I want Auto's memory to remember what was done, not what was said, so a corrected false claim is not remembered as a fact next turn.

**Acceptance criteria:**
- [ ] `_post_response` (`service.py:1459`) passes `receipts` and the answer to `SmartChatIntegration.store` (`consumers/chatbot/integration.py:110`), and the distillation prompt (`smart_memory.py`) is told which is the record of actions and which is the reply.
- [ ] A turn whose receipts are empty and whose text says "I've created the ticket" stores no fact that a ticket was created; a turn with a `done` receipt for `platform_create_task` stores the ticket's number.
- [ ] Tests: `test_prd256_memory_takes_receipts.py` on the distilled facts with a fake model.

### Part 2 (wave 1): Gates

#### US-004: The owner's click on owner-only actions (Gerard's sign-off first)
**Description:** As an owner, I want approvals, cancellations, agent and tool changes, settings and anything that publishes or sends to wait for my click when they come from chat, so Auto's words alone can never act or sign for me.

**Acceptance criteria:**
- [ ] A list `OWNER_ONLY_ACTIONS` in `modules/tools/discovery/owner_only.py` (new, small): `platform_update_task_status` and `platform_update_task` when the status is done or cancelled, `platform_assign_tool_to_agent`, `platform_unassign_tool_from_agent`, `platform_update_agent`, `platform_update_system_setting`, `platform_create_mission`, `platform_approve_mission`, `platform_cancel_mission`, `platform_publish_blog_post`, `platform_submit_social_post`, and every Composio send/publish action. Gerard confirms or trims the list in the issue before the build.
- [ ] In `PlatformExecutor.execute` (`platform_executor.py:862-925`): when `caller_context.driving_user_id` is set (a human drove the turn) and the action is owner-only, the executor returns the confirmation ask through `tool_grants.attach_ask_grant`, exactly as a `requires_confirmation=True` action does today; the chat shows the existing approval card (`frontend/components/chatbot/chat.tsx:358-372`); the owner's click resumes the call through the existing grant path. Agent runs, playbook steps and heartbeats (no driving user) are unchanged.
- [ ] The note a card gets from a chat-driven move is written only when the owner clicked; it is signed by the user who clicked (`board_consent.actor_from_user_id`), never on Auto's call alone.
- [ ] `follows_the_owner.py`'s APPROVE / CANCEL / GO_AHEAD paths and `owner_turn.py`'s verb regexes are removed in this story; "the card the owner named is the card acted on" (a card number must never reach a non-card write tool; no copy instead of the named card) stays.
- [ ] "Cancel #0422" from chat produces an ask that names #0422 and "cancel"; nothing moves until the click; "Approve it" after the ask resumes it; a second "approve it" without a card does nothing.
- [ ] The policy plane (`AUTOMATOS_POLICY_PLANE`) is not required for this: the gate is the action list plus the driving user, on both editions.
- [ ] Tests: `test_prd256_owner_actions_wait_for_the_click.py` (the ask, the resume on grant, the agent-run bypass, the note's signer); the F280-A tests updated with the reason in each change.

#### US-005: Done needs an artifact (F380)
**Description:** As an owner, I want a card to close as Done only when the thing it was for exists, so "done" on my board means something was made.

**Acceptance criteria:**
- [ ] `update_board_task_status` (`modules/tools/discovery/handlers_board_task_done.py`) and the board's own approve route refuse a move to Done for a card whose kind requires an artifact (a social post: a rendered post; a document card: a Deliverable; a card made by chat for a deliverable: that deliverable) when none is linked; the refusal names what is missing in plain words and leaves the card in Review.
- [ ] A card with no artifact kind (a question answered in text) is unchanged.
- [ ] Tests: handler tests with a fixture card of each kind, with and without the artifact.

### Part 3 (wave 1): Tool contracts and the instrument

#### US-006: Tool contracts for Auto's writes
**Description:** As an owner, I want Auto's writes to land first time, so a failed call never has the chance to be reported as done.

**Acceptance criteria:**
- [ ] The write actions the owner's asks use most are **promoted and pinned** so `ActionRegistry.to_first_class_schemas` (`tool_router.py:1167`) ships them as their own tools, outside the dispatcher's 15-action enum: `platform_create_task`, `platform_update_task`, `platform_update_task_status`, `platform_assign_task`, `platform_create_mission`, `platform_execute_playbook`, `platform_schedule_playbook`, `platform_create_social_post`, `platform_store_memory`, `platform_get_task`, `platform_query_data` (the review's join: the used action sat outside the top-15 on 53% of dispatcher calls, led by exactly these).
- [ ] Each promoted schema is strict: every required field listed, `status` an enum of the board's own words, `params` typed as an object with its fields named, no free-text catch-all; `platform_execute`'s `params` still decodes a JSON string (F181) but the promoted tools never need it.
- [ ] The tool result of a refused write already names what was missing; the loop's nudge (`modules/tools/execution/nudges.py`) and the Auto system prompt gain one rule: *after a refused write, the reply says the write was refused and why; it never reports it done.*
- [ ] `TOOL_ROUTING_PROMOTION_PINS` and the registry's promoted set are the mechanism; no new config key beyond adding names to the existing one; the shadow surface log (`[tool-shadow]`) keeps reporting shipped versus would-be-first-class counts.
- [ ] On the eval pack (US-007), arm A: dispatcher argument errors of the kinds seen on 5 to 7 Oct (invented action names, missing required params, `params` as a string, nested `params`) fall below 5% of write calls; a refused `platform_approve_mission` is followed by a reply that says so.
- [ ] Tests: a schema test per promoted tool (required fields, enums); a loop test with a refused write and a fake model that claims success, asserting the nudge; `check_hierarchy_gate.py` passes (promoted actions keep their permission levels).

#### US-007: The instrument: eval runner, results table, baseline
**Description:** As the owner, I want one offline eval that replays the nights' real asks against Auto and scores from what Auto did, so every wave and every model is measured against the same "before".

**Acceptance criteria:**
- [ ] The eval set and runner exist locally (`~/.automatos-analyst/auto-eval/`: `auto-eval.jsonl`, 65 rows from the nights' real asks, 22 verbatim; `replay.py`; `README.md`) and stay local (gold data; never committed). The runner reuses `tests/sim` (`tests.sim.api`, `tests.sim.sse`) against c1 and never the `night run` path.
- [ ] Per ask the runner scores: expected tool ran (the dispatcher's inner action counts), no forbidden tool, effects (tasks and agents created, measured before and after), reply rules, claims backed (the repo's `claimed_action_not_done` inside the backend container on the turn's real success set, and, once US-001 ships, the receipts rule: empty receipts plus a completed-action sentence is a failure), owner-only only on a grant ask, and the answering lane.
- [ ] `--arm` snapshots Auto's `model_config` (`GET /api/agents/1`), applies the arm (`PUT /api/agents/1/model-config`), posts one preflight turn so a provider refusal stops the run before it spends, restores the snapshot afterwards and on failure; `runs/results.md` gains one line per run (date, arm, model, rows passed, the four rates, p50 latency, tokens, cost from `llm_usage`).
- [ ] **The baseline runs before wave 1 is built:** arms A (today's Gemini) and C (Sonnet 5) on today's build, about $6, with Gerard's OK. Its two lines are the first in `results.md`.
- [ ] After wave 1 merges, arms A and C run again; the wave's exit criterion is the delta against the baseline on the four rates.
- [ ] The runner is documented in `docs/testing/` (a short page: what it measures, how to read `results.md`, the rule that it spends money).

### Part 4 (wave 2): The model

#### US-008: A Claude-ready LLM manager
**Description:** As the operator, I want a Claude 4.6+ model to be a one-line switch for Auto, so the model test and the decision cost nothing in plumbing.

**Acceptance criteria:**
- [ ] If the baseline's preflight on arm C showed the provider's 400: sampling parameters (`temperature`, `top_p`, `top_k`) are omitted for Claude 4.6+ ids on every client path (`core/llm/clients/anthropic_client.py:258,313`, the OpenRouter path through `openai_compatible_client.py`), through the existing per-model check in `core/llm/clients/base.py:116`. If the preflight was clean, this is a cleanup with the same test.
- [ ] `core/llm/manager.py:1014-1028` prices `claude-sonnet-5`, `claude-haiku-4-5` and `claude-opus-5`, so the turn-cost governor (`CHATBOT_TURN_COST_CEILING_USD`) prices them; `reports/config-surface.json` unchanged unless a key is added.
- [ ] The failover model is a config value, not inherited: a rate-limited Gemini turn no longer silently answers on `openai/gpt-4o-mini` (night 5); the receipts frame carries the model that answered.
- [ ] Reasoning passthrough for Claude via OpenRouter checked on one recorded request; the existing reasoning stream (PRD-238) shows it.
- [ ] Tests: `test_prd256_llm_manager_claude5_params.py` with a recorded transport (no `temperature` on a Sonnet 5 request; the governor prices it; the failover model comes from config).

#### US-009: The four-arm run and the decision
**Description:** As the owner, I want Auto's default model chosen by the same asks on four models, with the bars set before the run, so the choice is a measurement and not a guess.

**Acceptance criteria:**
- [ ] Arms A (Gemini 2.5 Flash, 0.7), B (Gemini 2.5 Flash, 0.2), C (Claude Sonnet 5), D (Claude Haiku 4.5) run on the eval pack after wave 1; the bars, set before the run: claims backed ≥ 95%, wrong-surface ≤ 5%, first-time-right ≥ 80%, owner-only without a click = 0.
- [ ] One table in `results.md` and in the ledger: the four rates, cost per 100 asks, p50 turn latency and median `prep_ms` per arm.
- [ ] Auto's default (`agents.model_config` for the Auto agent, both editions' seeds) is set to the cheapest arm that passes every bar; if none passes, the default stays and the lanes (US-010, US-011) are the next cause.
- [ ] The seeds already read one constant, `DEFAULT_LLM_MODEL` in `orchestrator/core/llm/defaults.py` (verified 7 Oct: `seed_auto_agent._get_default_model_config` → `get_default_model_config()`), so the decision is a one-line change there plus the Settings row of agent 1 in existing workspaces, which the seed never rewrites; a test pins that the seeded Auto model equals the constant.
- [ ] The run is repeated after every later wave; the table is the standing check.

### Part 5 (wave 2): Lanes

#### US-010: Auto always answers; work goes to agents on tickets
**Description:** As an owner, I want Auto to be the one voice in my chat, so a specialist never answers me in its own persona and the work goes to the team on tickets I can see.

**Acceptance criteria:**
- [ ] The DELEGATE lane is removed from `api/chat.py:463-498`: no chat turn is answered by an agent other than Auto unless the owner chose that agent in the UI (`request.agentId`).
- [ ] A message that names an agent or role resolves to the ASSIGN lane (`apply_assign_bias`): Auto files the ticket for that agent with the dispatch contract and reports the number; a role with no roster match asks once, as today.
- [ ] The Tier-3 rubric (`auto.py:733-769`) drops "delegate … (Most molecule/cell/organ work.)"; `Action.DELEGATE` maps to RESPOND with the platform tools; the 24-hour verdict cache (`COMPLEXITY_CACHE_TTL_HOURS`) is bypassed for any verdict that would have routed away.
- [ ] The Jev shadow keeps recording the classifier; nothing reads its agreement as a quality signal.
- [ ] On the eval pack's routing rows, Auto answers every unnamed ask and files a ticket for every named one; "Give #0192 to the Support Agent" assigns the named card and makes no copy.
- [ ] Tests: `test_prd256_auto_answers_unless_named.py` on the dispatch; the F263 and F241 tests updated with reasons.

#### US-011: One hand-off table instead of four lanes
**Description:** As a maintainer, I want the "who owns this kind of ask" rules in one table, so the next kind of work is one row and not a new module.

**Acceptance criteria:**
- [ ] `brand_assign_lane.py`, the routing half of `brand_to_the_designer.py`, `paperwork_to_the_team.py` and the routing half of `named_template_note.py` become one table in `consumers/chatbot/handoffs.py`: ask kind (brand work, customer paperwork with no template named, socials work) → the agent role that owns it → the note the ticket carries. The classifier reads the table once per turn; the named-template field note (the studio's shape) stays as data.
- [ ] The same asks route the same way as before: a golden file from the eval pack's routing rows, generated before the change and asserted after.
- [ ] `service.py` loses the four imports and the decorators on `_retrieval_first` that the table replaces; the deleted modules' tests move to the table's test.

#### US-012: Delete the families
**Description:** As a maintainer, I want the regex claim families gone once receipts hold, so the next surface does not need a new vocabulary.

**Acceptance criteria:**
- [ ] Preconditions, both proven by the post-wave-1 eval: US-002's rule at 25/25, and claims backed ≥ 95% on the chosen arm.
- [ ] `action_claims.py`, `document_claims.py`, `shop_and_team_claims.py`, the tier 1 and 3 paths of `claim_check.py`, and the note lanes `figure_disputes.py`, `shop_figures.py`, `team_corrections.py` are removed; `claim_check.py` keeps tier 2 and the receipts rule; `needs_you_turn.py` and `team_findings.py` keep their pre-reads (data) and lose their rule prose.
- [ ] The 26 claim test files are rewritten against receipts; none is weakened or skipped (AGENTS.md); the eval's claims-backed rate does not drop after the deletion.
- [ ] `service.py` ends the wave shorter than it started (3,268 lines today), with a test that pins the ceiling.

#### US-013: Context assembly under two seconds (Gerard's call after US-009)
**Description:** As an owner, I want a reply to start within a couple of seconds, so a turn is not eight seconds of assembly before the first word.

**Acceptance criteria:**
- [ ] US-009 records median `prep_ms` (`messages.context_trace`) per arm; the sampled night-11 turn spent 8,215 ms assembling context before the first model call.
- [ ] If the median is above two seconds: the per-workspace sections (identity, product facts, skills, platform actions) are cached across turns and invalidated on the writes that change them; the board and document reads run in parallel.
- [ ] Median `prep_ms` on the eval pack under two seconds with the same answers; a timing test on the context assembly with a fixed workspace.
- [ ] Dropped without building if US-009 measures the median under two seconds.

## 4. Functional requirements

- **FR-1:** every assistant message saved by the chat carries a `receipts` part built by the platform from the tool tracker: action, kind, status (done, refused, skipped), subject, effect, link, reason. The model never writes it.
- **FR-2:** the receipts are shown before the reply text, live and on reload; empty receipts render "No actions in this turn."; a refused write renders its reason.
- **FR-3:** the honesty line is computed from receipts alone: it fires when no write receipt is `done` and the text reports a completed action, and never when a write receipt is `done`.
- **FR-4:** memory distils receipts and text, told apart.
- **FR-5:** an owner-only action called from a human-driven chat turn returns the confirmation ask and runs only on the owner's grant; the card's note is signed by the user who clicked.
- **FR-6:** a card whose kind requires an artifact cannot move to Done without it.
- **FR-7:** Auto's listed write actions are first-class tools with strict schemas; a refused write is followed by a reply that says it was refused.
- **FR-8:** the eval runner scores from recorded calls and receipts, switches and restores Auto's model per arm, and keeps a per-date, per-model results table; the baseline runs before wave 1.
- **FR-9:** Claude 4.6+ models run on Auto without sampling parameters; the price table and the failover model are explicit.
- **FR-10:** Auto's default model is the cheapest arm that passes the bars; the run repeats after every wave.
- **FR-11:** no chat turn is answered by an agent other than Auto unless the owner chose that agent; a named agent gets a ticket.
- **FR-12:** the hand-off rules are one table; the regex claim families are deleted once the receipts rule and the eval hold.
- **FR-13:** both editions; every query scoped to the caller's workspace; no new table, dependency or Settings key beyond the pins and the failover model.

## 5. Non-goals

- No new regex claim family, and no new lane module: a new kind of work is a row in the hand-off table.
- No structured "claims" field returned by the model and matched against results (TESTER's proposal, dropped in review: it is the vocabulary problem again and it trusts the model to report on itself).
- Jev goes live on nothing in this PRD; the ticket-assignment seam (the one with graded evidence) is its own small change with its two rules (skip exact-name roles; 0.7 floor).
- No change to the mission or playbook engines beyond "done needs an artifact".
- No night runs until the eval passes its bars; no local servers or test runs (CI is the gate).
- No new third-party client, and the Composio deny list is untouched.

## 6. Design considerations

- **The receipts block** is the first thing under the owner's message: a short list in the chip style of `ActivityTrail`, one line per write ("Moved #0422 to Cancelled", "Gave Social Media Director the Dropbox tool: refused, needs your OK"), reads folded into one line. Plain words, the board's own numbers, never a tool or parameter name (F205, `owner_words.py`).
- **The honesty line** stays in Auto's own voice ("Just to be clear: I haven't done that yet…"), above the text, so the owner reads the correction before the claim.
- **The ask card** is the existing approval card (PRD-193 S3), with the subject by number and the verb the owner used.
- **Reuse:** `ToolExecutionTracker.outcomes` and `call_effects`; `StreamingHandler`; `reply_parts`; `tool_grants` and the grant card; `to_first_class_schemas` and the promotion pins; `tests/sim`; the dispatch contract fragment for tickets.

## 7. Technical considerations

- **Receipts source:** the in-process tracker, not `tool_execution_logs` (it stores parameter names only, an empty result and no chat id). The part is JSON on `messages.parts`; no migration.
- **Frame order:** receipts after the loop and before the answer's additions, so the honesty line can read them; the frontend must tolerate a message with no receipts part (every message before this PRD).
- **The gate:** `PlatformExecutor.execute` already returns asks for `requires_confirmation` actions and the fail-closed path; the owner-only check is one more condition in the same place, keyed on `caller_context.driving_user_id`, so agent runs are untouched. The hierarchy gate (`check_hierarchy_gate.py`) must still see every `ActionDefinition` inside `registry.register(...)`.
- **First-class promotion:** `TOOL_ROUTING_PROMOTION_PINS` plus the registry's promoted set; the dispatcher's `exclude_names` keeps promoted actions out of the enum; the ATOM lane ships the dispatcher only today and must ship the promoted tools too (`service.py:2822-2849`).
- **Model switch:** `agents.model_config` is JSON; the runner uses the existing `PUT /api/agents/{id}/model-config`; the seeds for the Auto agent in both editions carry the chosen default.
- **Code shape:** `service.py` must not grow (it is over the 800-line ceiling); every new behaviour is a new small module; `_stream_tool_loop` gains calls, not code.
- **Cost:** the baseline is about $6; the four-arm run about $8; a night after the bars is the usual $1 to $7 on Gemini or $2 to $16 on Sonnet 5 at list price with caching.

## 8. Success metrics

- **Eval, after wave 1, on the chosen arm:** claims backed ≥ 95% (baseline to be measured); wrong-surface calls ≤ 5%; first-time-right ≥ 80%; owner-only without a click = 0; dispatcher argument errors < 5% of write calls.
- **Eval, after wave 2:** the same bars on the four-arm table, and no chat turn answered by an agent other than Auto on the routing rows.
- **Night 12 (the first night after the bars pass):** zero false claims in the morning report's ledger; zero owner-only actions without a click; the persona's verdict on Auto moves from "not yet for the chat" (nights 9, 9b, 10, 11) to "yes".
- **Code:** `service.py` shorter than 3,268 lines; the claim families and four lanes deleted; no new `F-id` named in a chatbot module's docstring after this PRD.

## 9. Open questions — decided by Gerard, 7 Oct 2026

All six answered on 7 Oct, each as the kit's default (Decisions D1–D6 in `scripts/ralph/prd-256w{1,2}.json`): (1) the D1 owner-only list as written, `create_mission` included; (2) Auto always answers, the DELEGATE lane removed; (3) US-013 only if the post-wave-1 eval measures median prep_ms above 2 s; (4) Jev not in this PRD; (5) no failover model, the turn fails honestly; (6) the ATOM lane kept with the promoted tools. The questions stay below for the record.


1. **The owner-only action list (US-004):** confirm or trim before the build. Is `platform_create_mission` owner-only, or only its approval?
2. **US-010's framing:** Auto always answers and specialists never take over the chat (recommended, matches the thesis), or keep delegation with the specialist running inside Auto's turn with receipts?
3. **US-013:** keep as the thirteenth story, or drop if US-009 measures the median under two seconds?
4. **Jev:** go live on ticket assignment with its two rules as a separate small change, now or after wave 2?
5. **The failover model (US-008):** none (fail the turn honestly), or a named cheap model?
6. **The ATOM lane:** keep it once the promoted tools ship on it, or fold every owner turn onto the full path and measure the cost?

## 10. Testing plan (nights and the eval)

1. **Before wave 1:** the baseline, arms A and C on today's build (US-007), with Gerard's OK. The two lines are the "before".
2. **Wave 1 PR:** one PR, one CI run; after merge, arms A and C again; the delta on the four rates is the exit criterion.
3. **Wave 2 PR:** the four-arm run (US-009); the default model set; US-012's deletion only after its preconditions.
4. **Night 12** runs in c1 on a fresh build from main only once the bars pass, with TESTER's rubric plus one new axis: **receipts match** (every completed-action sentence has a receipt; every refused write is said to be refused). The night's ledger rows for Auto are joined to the receipts, not to the prose.
5. The eval runs again after every later PR that touches `consumers/chatbot/`, and its line goes in `results.md`.

## Related

The review: `.claude/AUTO-REVIEW-FINDINGS.md` (7 Oct, revision 2) and its evidence in `.claude/auto-review/`; the eval set `~/.automatos-analyst/auto-eval/` (local). Findings F108, F132, F133, F185, F187, F231, F261, F266, F280, F288, F289, F290, F303, F304, F307, F314, F319, F324, F337, F351, F362, F363, F379, F381 (`docs/testing/FINDINGS-LEDGER.md`). PRD-193 (approval grants), PRD-192 (policy plane), PRD-238 (streaming, narration), PRD-224 (the ASSIGN lane), PRD-232 (tool surface and shadow), PRD-248 (the decision seam), PRD-255 (brand kit v2, the Brand designer hand-off). PRs #905, #908, #910, #913, #1028.

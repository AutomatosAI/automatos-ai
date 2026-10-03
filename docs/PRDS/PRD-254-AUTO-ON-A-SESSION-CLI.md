# PRD-254: Auto on a session CLI — Auto runs on your own Claude Code, Codex or Copilot, like any runtime agent

> **Status:** DRAFT 2026-10-03. Written, **not started**. The owner's words: "write the PRD but dont start it yet... I want to finish testing and bug fixing." Owner decisions O1–O6 are open.
>
> **The ask:** "I would like to add a new feature, where Auto can be a runtime agent just like all other agents, so in settings > orchestrator where we select AUTO's LLM model we have the same options as agents so it can be Claude, Codex for example, so the system can run on runtime agents fully."
>
> **Grounded @** `automatos-ai` `origin/main` `80dc60afb` (cli-host 0.11.0, `EXPECTED_CLI_HOST_VERSION` 0.11.0), plus two read-only sweeps of the chat path and the session runtime on 2026-10-03. Nothing was run.
>
> **Settles** PRD-234's Q6 (owner, 2026-08-30): "Hybrid: Auto and any agent stay on API/OpenRouter in v1; Auto-on-subscription = later story (interactive driver too)." This is that story.

---

## Framing (CLAUDE.md §3)

**Extension.** Auto is an ordinary row in `agents`. PRD-234, 239, 245 and 253 built the session runtime: a `runtime: cli` agent's work runs as the operator's own CLI session, claimed by the paired host, with a gate, approvals, Automatos tools over a loopback MCP bridge, and $0 subscription usage. Every piece of it is **ticket-shaped**. Auto is **chat-shaped**: a streamed reply inside one HTTP request, an in-process tool loop with the full platform toolset, and per-turn context.

This PRD adds the missing piece: a **conversation lane**. One turn of Auto's chat runs as one resumed CLI turn on the host, and its text and tool steps stream back into the chat. Settings > Orchestrator then offers the same runtimes the agent form does.

**Reused, not rebuilt:**
- the per-conversation session ticket (PRD-239 S7 v2);
- the claim, lease, token, events, result and usage machinery;
- the bridge and its `platform_execute` route;
- the session system prompt builder;
- the Questions-tab approvals;
- the agent form's Runtime picker.

**New:**
- a conversation mode on the session ticket;
- a long-poll claim;
- incremental transcript reading on the host;
- a conversation-scoped tool set on the bridge;
- the chat route's session branch;
- approval cards in the chat.

**No migration expected:** `runtime_ref` and `Agent.configuration` are JSON.

**Size:** L. **Risk:** Medium. The chat is Auto's most-used surface, so the API path must not change when Auto stays on API. That is a regression test in every wave.

## What exists today (main @ 80dc60afb)

**Settings > Orchestrator writes Auto's model, and a global copy of it.**
- `frontend/components/settings/SystemLLMSettingsTab.tsx` saves through `PUT /api/workspaces/current/orchestrator` (`api/workspaces.py:729`).
- That writes Auto's `agents.model_config`, then mirrors provider and model into the **global** `system_settings.orchestrator_llm` (`api/workspaces.py:870`), "used by internal orchestrator operations".
- The agent form's Runtime picker (`frontend/components/agents/runtime-section.tsx`, local edition only) is not on this tab.

**Auto is an ordinary agent.**
- Resolved per workspace by slug `auto-{workspace}` (`get_default_agent_id`, `api/chat.py:147`).
- `validate_runtime_configuration` (`core/cli_runtime.py`) would accept `runtime: cli` on its row. Nothing offers it.

**A chat turn, today:**
1. `POST /api/chat` runs AutoBrain's lane classifier (`api/chat.py:408`, model tier `system_llm`).
2. `StreamingChatService.stream_response_with_agent` runs, then `AgentFactory.activate_agent` (the row's `model_config`), then `LLMManager`.
3. An in-process tool loop offers the registry tools, the `platform_execute` dispatcher (about 200 actions), the promoted first-class actions and `composio_execute`.
4. The reply streams in the AI-SDK data protocol (`consumers/chatbot/streaming.py`): `0:` text deltas, `d:` events (`tool-start`, `tool-end`, `usage`, `finish`, …).
5. Context comes from `ContextService` in CHATBOT mode: identity and personality, product facts, skills, memory, the Knowledge Graph, a documents inventory, datetime and the conversation. Page context is injected into the history (`services/page_context.py`).

**What a cli Auto would hit today.**
- `activate_agent` returns `None` for a cli agent (`modules/agents/factory/agent_factory.py:759`), and the chat turn ends in `runtime_mismatch`.
- A session agent picked in chat gets "runs as a Claude Code session in the Canvas terminal" and nothing else (`api/chat.py:552`, `cli_ticket_lane.session_agent_terminal_message`).

**The session runtime is ticket-shaped.**
- The host claims board tickets only (`claim_for_host`, `services/cli_host_service.py:970`).
- It polls every 5 s when idle and flushes events every 5 s (`services/cli-host/automatos_cli_host/config.py:41,43`).
- It sends no assistant text until `Stop` (`session.py:260`).
- The transcript reader has `read_usage` and `last_assistant_text` only (`transcript.py:42,80`).
- Per-turn text reaches the CLI through `_turn_context` (`session.py:269`) and `_ticket_prompt` (`services/cli_host_service.py:933`).

**There is already one session ticket per conversation.** `open_session_ticket` (`services/cli_ticket_lane.py:475`, source `chat:<chat id>:session`) is the record behind the Runtime Canvas terminal. It is **never dispatched**: no lease, no claim, mode `terminal`.

**The bridge serves ten ticket tools** (`services/session_tools.py:446`).
- Its token is minted per claim (`:114`).
- It resolves only while that ticket is `in_progress` with a live lease (`resolve_session_token`, `:146`).
- Every call goes through `UnifiedToolExecutor`, eight of the ten via `platform_execute`.

**Approvals, usage and edition.**
- Holds, asks and Plan are cards in the Questions tab, the bell and Telegram (`raise_session_holds`, `:1685`; `services/session_plans.py`).
- A hold waits at most 530 s (`presets.py`). An ask's answer reaches the session on its next turn.
- Usage is booked at $0, tier `subscription` (`book_session_usage`, `:1858`). A plan limit pauses the work rather than failing it (`usage_limit.py`).
- Session mode is local-edition only (`config.py:2146`).

**Who reads which model:**

| Reads | Callers | Can it run on a CLI? |
|---|---|---|
| Auto's `agents.model_config` | Auto's chat turns (and board tickets assigned to Auto) | Yes, with this PRD |
| `system_settings.orchestrator_llm` (global) | `UniversalRouter` (`core/routing/engine.py`), NL2SQL (`modules/nl2sql/service.py`), the board's Plan/Refine (`api/board_tasks.py`), the orchestrator heartbeat (`services/heartbeat_service.py`), voice (forced, `api/voice_retell.py:724`) | Not as written; O1 |
| `system_llm` | AutoBrain's classifier (`consumers/chatbot/auto.py`), the mission planner (`modules/coordination/planner.py`), RAG query enhancement (`modules/rag/query_enhancer.py`), graph extraction, verifiers | Not as written; O1 |
| Their own dials | memory distill (`MEMORY_DISTILL_MODEL`, `consumers/chatbot/smart_memory.py`), embeddings (`core/llm/embedding_manager.py`), rerank (Cohere, `core/llm/rerank_manager.py`) | Embeddings and rerank: **no**. They are not chat models. |

So moving Auto alone does not make "the system run on runtime agents fully". Every row below the first keeps needing an API model, unless O1 moves it and can.

## Goals

- **Settings > Orchestrator offers Auto the same runtimes as any agent:**
  - API, as today;
  - Claude Code, Codex or Copilot, from the same registry and validation, shown only where a paired host serves them.
  - Local edition.
- **On a session CLI, Auto's chat still works as it does today:**
  - the reply streams;
  - tool steps show;
  - the same tools and context are available;
  - approvals are answered in the chat;
  - it all runs on the operator's own plan, booked at $0.
- **Nothing else breaks when Auto moves.** Every internal caller keeps a working model (D9, O1).
- **Says so when it can't run.** Host offline, CLI not served or not logged in, plan limit: the chat says which one, in one line, and does what O2 decides.
- **Every CLI the adapter serves.** Claude Code, Codex and Copilot together, per the runtime-parity rule (owner, 2026-10-02: "User should be able to do this for all runtime models").

## Decisions (adopted in this draft; each reversible in its PR)

- **D1 · The claim unit is the conversation's session ticket.**
  - Auto's conversation gets the same one-per-conversation ticket the Runtime Canvas uses (`chat:<chat id>:session`), with a new `runtime_ref.mode = "conversation"`.
  - Each user message queues one turn on it: the ticket goes to `assigned` with the message as `runtime_ref.turn`; the host claims it, which brings the lease and the token; the result lands; the ticket goes back to idle.
  - This reuses the claim, lease, token, events, result and usage paths unchanged.
  - Conversation tickets are **Auto's chat, not work**. The board, the fleet, Needs-you, SLA timers, reconcilers and analytics task counts all leave them out, through one predicate.
- **D2 · One CLI turn per chat message, resumed, never typed into.**
  - Each turn spawns the CLI resuming the conversation's session: Claude `--resume <id>`, Codex `codex resume <id>`, Copilot `--session-id <id>`.
  - The host still never types into a running session (PRD-234's invariant).
  - The CLI keeps the conversation history itself; Automatos sends only the new message and the turn's context.
- **D3 · A turn starts on push, not on the 5 s poll.**
  - The claim route gains a bounded long-poll (`wait_seconds`, at most 25). It answers as soon as a conversation turn is queued for a CLI this host serves.
  - An idle host uses it, and tickets keep their current path.
  - Long-poll waits are `async` and awaited (F257: never a threadpool thread held open).
- **D4 · The reply streams from the CLI's own session record, a block at a time.**
  - The host tails the session file from the turn's start offset:
    - Claude's transcript JSONL;
    - Codex's rollout;
    - Copilot's `events.jsonl`.
  - It forwards each assistant text block and tool step as an event. Conversation sessions flush every 1 s.
  - The backend maps them onto the chat's existing AI-SDK stream: `0:` for a text block, `d:tool-start` / `d:tool-end`, `usage`, `finish`.
  - Text arrives a block at a time, not token by token.
- **D5 · Auto's tools reach the session over the bridge, scoped to the turn.**
  - The token is minted per turn and resolves exactly as today (the ticket `in_progress` with a live lease).
  - The advertised set is **Auto's chat set**: the `platform_execute` dispatcher with the chat's action list, `composio_execute`, and the chat's first-class tools that mean something in a session.
  - The set is byte-stable per workspace, for the CLI's prompt cache.
  - The driving user is threaded into each call as a live chat turn does today. On the local edition, filing or assigning a ticket from the chat remains the operator's consent (PRD-234 D16).
- **D6 · Auto's context reaches every turn, with no edit to the system prompt.**
  - The stable part goes into the session's system prompt through `session_system_prompt(…, ticket_session=False)` (`services/cli_session_prompt.py:262`): identity, personality, product facts, skills.
  - The per-turn part goes into a context block ahead of the message, on the existing `_turn_context` route: memory, the Knowledge Graph excerpt, the documents inventory, page context, datetime.
- **D7 · On a CLI, the chat turn goes straight to Auto's session.**
  - AutoBrain's classifier and `UniversalRouter` are not called for that turn.
  - Auto assigns, files and starts Missions with its own tools.
    - The ASSIGN lane is tool-driven already (`platform_create_task` / `platform_assign_task`).
    - The MISSION lane's suggestion card becomes a tool call; S2.1 verifies the mission actions are in Auto's set.
  - That removes the per-message API calls ahead of the reply. O3 can keep the classifier.
- **D8 · Approvals are answered in the chat.**
  - A hold or Plan card raised by a conversation turn renders inline in the chat, as a `d:session-hold` or `d:session-plan` event.
  - It also still appears in the Questions tab, the bell and Telegram.
  - Either answer resolves the same request id. The 530 s hold ceiling stands.
- **D9 · A platform model for everything that isn't Auto's chat.**
  - When Auto is on a CLI, Settings > Orchestrator also asks for a **Platform model**, an API model. The `orchestrator_llm` mirror writes that model, never a CLI.
  - On API, nothing changes: the mirror writes Auto's model, as today.
- **D10 · Local edition only.**
  - SaaS hides the option (as `runtime-section.tsx` already does when not local) and keeps Auto on API.
  - A central, server-side Auto on developers' CLIs is PRD-253's open O1, not this PRD.
- **D11 · Auto's session folder and mode follow the workspace defaults.**
  - The folder is the workspace's default session folder, with no worktree.
  - The permission mode is the workspace's Session mode default.
  - O5 can change either. Auto with file and shell tools on the operator's machine is new.

## Stories

### Wave 1 — the conversation lane: one turn of Auto's chat runs as a resumed CLI turn and streams back

**S1.1 · Conversation mode on the session ticket (M)**

`open_conversation_ticket(db, workspace_id, auto, chat_id, host)` sits beside `open_session_ticket` and shares `session_ticket_for`.
- **Queue a turn:** `queue_conversation_turn(ticket, message, page_context)` sets `assigned`, writes `runtime_ref.turn` (message, turn id, page context) and keeps `cli_session_id` for resume.
- **Turn end:** the result moves the ticket to idle and records the final text on `runtime_ref.turns[-1]`, a bounded ring.
- **One exclusion predicate,** `is_conversation_ticket(task)`, is used by:
  - the board list;
  - the fleet;
  - Needs-you;
  - the sweeper's and reconciler's SLA paths;
  - analytics task counts.

**Files:**
- `services/cli_ticket_lane.py`;
- `services/cli_host_service.py` (`_claim_one`, `apply_result`), each through small helpers (the changed-lines limits);
- the board, fleet and Needs-you queries;
- `tests/test_prd254_conversation_ticket.py`.

**Test:**
- one ticket per conversation;
- a second message re-queues the same ticket with the same `cli_session_id`;
- no conversation ticket appears in the board list, fleet, Needs-you or counts;
- the result lands idle with the text recorded.

**Editions:** local only.

**S1.2 · Long-poll claim (S)**

`POST /api/v1/cli-hosts/{id}/claim` gains `wait_seconds` (0–25, default 0 = today).
- With a wait, the route awaits a wake-up raised when a conversation turn is queued for a CLI the host serves, then claims once.
- The host sends a wait while it has free slots.
- Host contract moves to 0.12.0: both `EXPECTED_CLI_HOST_VERSION` and `__version__`.

**Files:**
- `api/cli_hosts.py`;
- `services/cli_host_service.py`;
- `services/cli-host/automatos_cli_host/host.py`;
- tests: backend (realdb claim pattern of `test_prd234_s1a_cli_hosts_realdb.py`) and host (`tests/test_units.py`).

**Test:**
- a turn queued during a wait is claimed in under 1 s;
- no turn means the wait times out empty;
- a ticket for an unserved CLI never wakes the host;
- the event loop never blocks (async route).

**S1.3 · The host streams the turn (M)**

A per-CLI incremental reader, with Claude's in `transcript.py` and Codex's and Copilot's in their adapters. It reads from the turn's start offset and yields:
- `AssistantText{text, block}`;
- `ToolStep{name, status}`;
- usage.

A conversation session flushes these as events every 1 s; a ticket session keeps 5 s. `Stop` and the result are unchanged.

**Files:**
- `services/cli-host/automatos_cli_host/transcript.py`;
- `adapters/codex.py`, `adapters/copilot.py`;
- `session.py`;
- the fake CLIs in `services/cli-host/tests/` gain mid-turn blocks.

**Test:**
- the fake Claude writes three text blocks and a tool call mid-turn, and three `AssistantText` and one `ToolStep` arrive before the result, in order;
- the same holds for fake Codex and fake Copilot;
- a partial last line is held until it is complete.

**S1.4 · The chat route serves Auto's turn from the session (M)**

When Auto's row is `runtime: cli`, `stream_chat` hands the turn to a helper, `services/conversation_lane.py`, at the branch point; the helper is new because `stream_chat` is far over the 50-line limit. The helper:
1. opens or re-queues the conversation ticket;
2. streams its events in the AI-SDK protocol: `0:` text, `d:tool-start` / `d:tool-end`, `d:usage` (the plan's tokens, $0), `finish`;
3. ends with the result.
- Stop in the chat uses the existing cancel path.
- A turn that ends with no text says so.
- AutoBrain and the router are skipped (D7).

**Files:**
- `api/chat.py` (branch point only);
- `services/conversation_lane.py` (new);
- `consumers/chatbot/streaming.py` (reused formatters);
- `tests/test_prd254_chat_conversation_lane.py`.

**Test:**
- with a fake host feeding events, the chat stream carries the blocks and tool steps in order and finishes;
- `llm_usage` gets one subscription row and no API row;
- with Auto on API, the stream is byte-identical to today's for the same fixture (regression).

**S1.5 · When the session can't run (S)**

One line in the chat, naming the cause:
- no host online;
- no host serves the chosen CLI;
- the CLI isn't logged in (from the host's preflight);
- the plan limit, with its reset time (`usage_limit`).

Then O2's behaviour: wait (default) or fall back to the Platform model for that turn, labelled as such.

**Files:**
- `services/conversation_lane.py`;
- `services/cli_ticket_lane.py` (reasons reused);
- tests.

**Test:** each cause produces its line, and with fallback on the turn runs on the Platform model and says so.

### Wave 2 — Auto's tools and context in the session

**S2.1 · Auto's tool set on the bridge (M)**

`services/session_tools.py` gains a second, fixed table for conversation turns. It holds:
- `platform_execute` (the chat's action enum);
- `composio_execute`;
- the chat's first-class tools that mean something in a session: knowledge and documents search, memory, workspace files.

Each tool's `scope` matches a chat turn: there is no ticket of its own to update or report on. The driving user rides each call as `_user_id` (D5). The table is chosen by `runtime_ref.mode` at claim, and it is byte-stable per workspace.

**Files:**
- `services/session_tools.py`;
- `services/cli_host_service.py` (claim payload);
- `tests/test_prd254_conversation_tools.py`.

**Test:**
- a conversation turn's claim advertises Auto's set; a ticket's claim advertises the ten, unchanged;
- `platform_create_task` from a conversation turn records the operator's consent on local;
- the list is byte-identical across two claims.

**S2.2 · Context per turn (M)**

`conversation_turn_context(workspace_id, chat_id, message, page_context)` builds the CHATBOT sections other than identity and skills; those two go to the system prompt (D6). It is rendered as the turn's context block.

**Files:**
- `modules/context/` (a mode or a section filter, not a copy);
- `services/conversation_lane.py`;
- tests.

**Test:**
- memory, Knowledge Graph and page context appear in the turn block, and never in the system prompt;
- the system prompt is byte-identical across turns.

### Wave 3 — Settings, the Platform model, approvals in the chat

**S3.1 · Runtime in Settings > Orchestrator (M, frontend + one route)**

`SystemLLMSettingsTab.tsx` gains the Runtime group by reusing `RuntimeSection`, local edition only. Choosing a CLI shows its model field, the workspace folder (D11) and the permission mode.

`PUT /api/workspaces/current/orchestrator` writes `Agent.configuration.runtime/provider/model/…` on Auto's row through `validate_runtime_configuration`. The orchestrator-seat model policy (PRD-223) applies to API models only.

**Files:**
- `SystemLLMSettingsTab.tsx` (+ its test);
- `api/workspaces.py` (the save split into helpers);
- `tests/test_prd254_orchestrator_runtime_settings.py`.

**Test:**
- save and reload round-trip the runtime;
- an unserved CLI is shown and annotated;
- SaaS never renders the group;
- vitest and tsc are green.

**S3.2 · The Platform model (S)**

With Auto on a CLI, the tab requires a Platform model (an API model). The `orchestrator_llm` mirror writes it (D9). GET returns both.

**Files:** `api/workspaces.py`, `SystemLLMSettingsTab.tsx`, tests.

**Test:**
- Auto on CLI leaves `orchestrator_llm` holding the Platform model, never a CLI id;
- Auto on API leaves the mirror unchanged.

**S3.3 · Approvals in the chat (M)**

`record_events` sends hold and Plan events of a conversation turn to the chat stream as `d:session-hold` and `d:session-plan`. The chat renders them as cards, reusing the Questions-tab card components; an answer posts to the existing grant routes.

**Files:**
- `services/cli_host_service.py` (`record_events`, via a helper);
- `services/conversation_lane.py`;
- the chat card component;
- tests.

**Test:**
- a held command shows in the chat and in the Questions tab;
- answering in either clears both;
- a Plan card's Approve resumes the turn as edits.

### Wave 4 — the internal callers O1 moves (if any), and voice (O4)

Each caller O1 moves gets its own story here, with its lane: a background ticket for background work, never a per-message classifier. Embeddings and rerank stay as they are.

## Owner decisions

- **O1 · Which internal callers move with Auto?**
  - The candidates are the router, NL2SQL, board Plan/Refine, the orchestrator heartbeat, the mission planner, AutoBrain's classifier, memory distill, RAG query enhancement and graph extraction.
  - Embeddings and rerank cannot move to a CLI.
  - **Recommendation:** keep the per-message and in-turn callers on the Platform model, because a CLI turn per classification adds seconds to every message. Move only background work (the heartbeat, mission planning), as tickets, if you want it.
- **O2 · When Auto's session can't run:**
  - (a) Auto says so and waits;
  - (b) it falls back to the Platform model for that turn, labelled.
  - **Recommendation:** (a) by default, with (b) as a setting. (b) is a billed engine on local.
- **O3 · AutoBrain on a CLI:** keep its per-message lane classification (one API call per message), or let the session decide (D7)?
- **O4 · Voice (Auto Live):** it stays on the Platform model, since a CLI turn is too slow for a live call. Confirm.
- **O5 · Auto's session folder and default permission mode (D11).** For example, Plan by default for Auto, or a dedicated folder per workspace.
- **O6 · Which CLI for the first live test.** All three ship together.

## Verify at build (no spend first)

- Claude Code appends assistant text and tool-use blocks to its transcript **during** the turn, not only at its end. The same check for Codex's rollout and Copilot's `events.jsonl` (`assistant.message`).
- The cold start of a resumed turn per CLI on the owner's Mac. This sets the latency budget we report.
- The prompt cache holds across resumed turns with a byte-identical system prompt and tool list.
- Which plan window Auto's chat moves. The ticket sessions share it.
- Long-poll behaviour through the pinned uvicorn 0.24 / starlette 0.41 under load (F257 / F105).

## Test plan (owner, local edition)

1. **After W1:** put Auto's row on Claude Code through the existing agents route the agent form uses. W1 makes that route accept Auto's row if it refuses a system agent; W3 adds the Settings UI. Ask a plain question that needs no platform tool.
   - The reply streams in blocks.
   - `llm_usage` shows a subscription row and no API row.
   - Quit the host: the chat says it's offline.
2. **After W2:** "what's on my board?" answers from `platform_execute`. "File a ticket for Bob to …" files one consented ticket. "What's on my calendar tomorrow?" runs Composio.
3. **After W3:** switch Auto's runtime in Settings; the Platform model is required. A held command shows as a chat card; approve it there.
4. **Regression, every wave:** Auto on API behaves exactly as before.

## Success metrics

- A working day of Auto's chat on a CLI makes **zero** API-billed calls for Auto's turns. The other callers make only what O1 leaves on the Platform model.
- No `runtime_mismatch` in the chat.
- The time from send to first text is measured and reported per CLI (target set after the W1 spike).

## Merge notes

- Waves land as separate PRs in order (W1 → W2 → W3 → W4). Each is CI-green and tested by the owner on local before the next.
- W1 moves the host contract to 0.12.0 (claim `wait_seconds`, mid-turn text events). A host installed as a copy elsewhere must be reinstalled.
- No migration expected.
- Every new route goes into `orchestrator/reports/route-manifest.json`, and every new setting into the config-surface report.
- DCO sign-off on every commit.

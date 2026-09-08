# PRD-239: Session agents at parity — persona and skills in the session, a chat lane, playbooks and missions, honest failures, route-aware models

> **Status:** BUILDING 2026-09-08 (owner: "agree lets get 1,2 and 3 done") — on `feat/prd-239-session-agents-parity`, stacked on PRD-238 W3 (`feat/prd-238-w3-follow-through`). Grounded @ the local stack (`local/prd-237-test`) and the 2026-09-08 10:07–10:43 direct-agent test.

---

## Framing (CLAUDE.md §3)

**Extension + two bug-class fixes.** PRD-234 built the session runtime (`runtime: cli` agents run as the operator's own Claude Code sessions, claimed by a paired host, one ticket per unit of work) and routed five lanes (heartbeat, schedules, channels, webhooks, Composio triggers) through `services/cli_ticket_lane.py`. Three surfaces never got the lane — **direct chat**, **playbooks**, **missions** — and the session never received the agent's **persona or skills** although PRD-234 S1b said it should ("the appended system prompt = agent soul + skills + dispatch doctrine"). Two edition-neutral bugs were found on the way: a failed turn shows **nothing** in the chat, and the agent model picker cannot express **which route** serves a model.

## What the test actually showed (grounded)

| Agent | Route | What happened |
|---|---|---|
| Bob (15) | `runtime: cli` | Direct chat died at activation twice (10:07:45, 10:08:56, 10:42:00). `AgentFactory.activate_agent` returns `None` for a cli agent by design; `stream_response_with_agent` raises "Failed to activate agent 15"; the `e:` frame is logged to the console by `useChat` and nothing renders. |
| Researcher (57) | stored `openrouter / deepseek/deepseek-coder` | OpenRouter 400 "not a valid model ID" (a February seed row still `active`; OpenRouter dropped the model). Same silent failure. The 10:08:34 model-config save carried `deepseek-coder` (backend log). |
| Researcher (57), after the owner "flipped to Kimi K3" | stored `openrouter / moonshotai/kimi-k3` | Works and is fast (10k-token prompt, 3.4 s / 7.1 s) — **but served by OpenRouter, not NVIDIA**: the modal sends the model id with the agent's *old* provider, and `_find_route(model_id, provider)` keeps that route. The NVIDIA route (`llm_models` 1114, installed) is unreachable from the agent UI. |
| Writer (58) | `openrouter / anthropic/claude-opus-4.6` | Fine. |

Parity today for a session agent: tickets, heartbeats, schedules, channels, webhooks, Composio triggers **work** (tickets). Playbooks **fail** ("could not be activated", `api/recipe_executor.py:402`). Missions **fail honestly** (`runtime_mismatch`, `agent_factory.py:1062`). Persona, skills **ignored** (`services/cli-host/automatos_cli_host/session.py:build_system_prompt` = name + rules). Platform tools, Composio, MCP **absent** (`--strict-mcp-config`, hooks-only settings) — owner decision pending (D-4/D-5 below, not in this PRD).

## Decisions

- **D1 · Chat with a session agent = one ticket per message, on one continuing Claude session.** The host never types into the TUI and never uses `-p` (PRD-234 §Terms). So a chat turn to Bob files a ticket carrying the message; the host runs it; the session's final text lands back in the chat when the session ends. Continuity: the host already honours `resume_session_id` (`--resume`); the backend now sets it from the previous chat ticket of the same chat + agent **when the same host claims** (a transcript lives on one machine). The first turn of a chat (no session yet) carries the recent conversation as context. A second message while the previous ticket is still running is filed, but held back at claim until the running one ends (one session, one turn at a time).
- **D2 · Persona and skills ride the appended system-prompt file.** Rendered server-side, stable per agent (no ids, dates, counters — the prompt-cache invariant), capped. Full skill bodies, not the L1 catalogue: a session has no `platform_load_skill` tool. Claude Code skill files were rejected: `--setting-sources user` excludes project skills by design.
- **D3 · Playbooks and missions block on the ticket, inside their existing timeouts.** A step or mission task assigned to a session agent files a ticket (same lane, same consent grant as heartbeats/schedules) and waits for it to end; the ticket's result is the step's output. The step's own timeout (`_execute_step` under `asyncio.wait_for`) and the mission task timeout (`_run_agent_io`) bound the wait; the ticket keeps running on the host and the step says so.
- **D4 · `model_config.provider` is the serving route and the picker is keyed by route.** Same rule PRD-236 W1 applied to the orchestrator settings tab. A saved model that the route no longer offers is refused with a plain message; the OpenRouter projection deprecates rows missing from the cache, as the NVIDIA sync already does.
- **D5 · A failed turn is visible.** The `e:` frame carries `{message, code}`; the chat renders it in the reply bubble and toasts; the service persists a short assistant note with provenance so a reload and the detached-turn path show it too. Provider errors are classified (model unavailable, rate limit, no key).

## Stories

### S1 · Persona + skills in the session (M)
**Files:** NEW `orchestrator/services/cli_session_prompt.py` (`session_system_prompt(agent)`: description, persona via `IdentitySection._get_persona_text`, skill bodies via `prompt_template`, capped by `CLI_SESSION_SKILLS_MAX_CHARS`); `services/cli_host_service.py` claim payload `system_prompt`; host `session.py:build_system_prompt` uses it when present (rules paragraph kept); host `__version__` 0.3.0 + `EXPECTED_CLI_HOST_VERSION` (the contract fingerprint changes anyway → hosts drain and restart on the new code).
**Test:** stable per agent; capped; persona precedence (custom > persona row); claim payload carries it; host writes it into `system_prompt.md`.
**Editions:** local only (cli agents exist only under `CLI_RUNTIME_ENABLED`).

### S2 · Chat with a session agent (L)
**Files:** NEW `orchestrator/services/session_agent_chat.py` (context block, ticket filing through `file_cli_ticket` with `actor`/`resume_session_id`/`created_by`, the turn's frames, the reply delivery); `services/cli_ticket_lane.py` (`actor`, `resume_session_id`, `created_by_id`, `orchestration_*` kwargs; `chat_origin_of(task)`); `services/cli_host_service.py` (`resume_session_id` at claim when the previous session was on this host; hold a chat ticket while its predecessor runs; deliver the reply after `finalize_board_task_run` for `source_type='chat'`); `api/chat.py` (direct mode → the session lane when the agent is cli); frontend: `task_card` part rendered from history, "Claude Code session" badge in `agent-selector.tsx`, the badge label from `messages.source`.
**Test:** lane units (fake db): first turn carries context, later turns carry `resume_session_id` only from the same host, a running predecessor holds the claim, reply delivery on done/review/failed/cancelled with the honest line; route test: direct mode with a cli agent never calls `stream_response_with_agent`; frontend: a persisted `task_card` part renders a card.
**Editions:** local only.

### S3 · Playbooks and missions through the ticket lane (M)
**Files:** `services/cli_ticket_lane.py` (`run_cli_ticket_and_wait`, poll `CLI_LANE_POLL_SECONDS`); `api/recipe_executor.py:_execute_step` (cli branch before activation, `source_type='recipe'`, `source_id='recipe:<execution>:<step>'`); `services/coordinator_service.py` (`_prepare_task` marks `cli_agent`, `_run_agent_io` runs the lane with its own session, `source_type='mission'`, `orchestration_run_id/task_id` on the ticket); `config.py` + `reports/config-surface.json`.
**Test:** the wait returns the exec-result shape on done/review/failed/cancelled; the step branch never activates; the mission branch never touches the shared session; timeouts leave the ticket running and say so.
**Editions:** local only.

### S4 · Failures are visible (S)
**Files:** NEW `orchestrator/consumers/chatbot/turn_errors.py` (`describe(exc, agent_name, model)` → `code`, `message`); `core/llm/clients/openai_compatible_client.py` (`ProviderModelUnavailableError`); `consumers/chatbot/streaming.py:format_aisdk_error(message, code)`; `services/chat_turns.py:error_frame`; `consumers/chatbot/service.py` catch → describe, persist the note (`source.origin='turn_error'`), frame; frontend `lib/chat/errors.ts` (`parseErrorFrame`), `hooks.ts` (error onto the reply + toast), `message.tsx` (error note), `types/chat.ts`.
**Test:** describe() table; frame shape; hook attaches the error and keeps the bubble; the note persists.
**Editions:** both.

### S5 · Route-aware agent model picker + dead-model hygiene (M)
**Files:** `frontend/components/agents/model-selector.tsx` (options keyed `serving_provider:model_id`, `onChange(modelId, servingProvider)`, route label, "current model unavailable" note); callers `agent-configuration-modal.tsx`, `agent-configuration.tsx` (also: init-once guard — a refetch reset the in-progress model choice), `create-agent-modal.tsx`; backend `api/agent_endpoints.py` (refuse a deprecated route, 422 with the plain message); `core/services/provider_catalog_sync.py` (OpenRouter projection deprecates rows missing from the active cache).
**Test:** picker emits the route; deprecated rows vanish from the projection; the endpoint refuses a deprecated model with the message.
**Editions:** both.

## Not in this PRD (owner decisions)

- **D-4 · Platform tools inside a session** (an Automatos MCP server in the session settings). Must not break the subscription invariants (no `-p`, no identity games, hooks-only settings). To be looked at after S1–S5 test.
- **D-5 · A Codex adapter in the host** — the runtime config accepts `codex`; the host runs only `claude` today.

## Test plan (owner, local edition)

1. Restart the host on the new code (it restarts itself on the contract change; check `make cli-host` logs for the 0.3.0 line). Open Bob's ticket dir: `system_prompt.md` carries the persona and skills.
2. Chat → pick Bob → "Hey Bob, summarise your skills": the reply bubble says a ticket was filed and shows the card; the reply lands in the chat when the session ends; a second message continues the same session (`--resume` in the host log).
3. A playbook step and a mission task assigned to Bob run as tickets and produce the step output / task output.
4. Pick Researcher → set Kimi K3 · NVIDIA → the log says `nvidia/moonshotai/kimi-k3`.
5. Point an agent at a dead model (or wait for the catalogue sync): the chat says the model is unavailable, in the bubble, and after reload.

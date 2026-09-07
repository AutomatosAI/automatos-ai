# Review 2026-09-03 — Automatos as the local manager of Claude Code ("the AI OS")

**Ask (owner, 2026-09-03):** make `/chat`'s Code option a Cursor-like environment; the full Claude Code builds and manages the code; Auto (possibly on Kimi) plans and assigns to Claude Code, DeepSeek or whoever is best suited; the board, calendar and heartbeats manage it all.

**Method:** three grounded investigations (the chat Code mode and Code Canvas; every dispatch lane, Auto's tools, the matcher, the model registry, the fleet; the strongest existing tools) plus live probes of the local stack. Every claim below names the file and line it came from or the API it was read from.

**Verdict in one paragraph.** Half of the platform this needs already exists and is unused for the purpose: a Monaco viewer with tabs, diff cards and an SSE session contract (Code Canvas); a capability-aware agent matcher with reasons and per-agent semantic cards; a supervised session runtime with a hook event plane, worktrees and deliverables (PRD-234 S1–S2); consent on the board. What is missing is wiring: seven of eight dispatch lanes still call the API factory for a Claude Code agent (the heartbeat lane even records the refusal as a green result), Auto cannot see an agent's runtime and is forbidden from picking, sessions and hosts are invisible in the fleet, and the Code mode's coding engine is an API-key SDK client in the worker. The build is a management layer over the user's own `claude`, not an IDE — which is also what every serious tool in this space concluded.

## 0. Live facts from the local stack (probed 2026-09-03 ~13:05 UTC)

- **Auto's model today**: workspace orchestrator config = provider `openrouter`, model `google/gemini-2.5-flash` (`GET /api/workspaces/current/orchestrator`; a legacy `model: meta-llama/llama-3.1-70b-instruct` field is still present).
- **Model registry** (`GET /api/models/`): 34 models — openrouter 10, anthropic 6, openai 6, aws_bedrock 4, google 4, huggingface 4. OpenRouter rows include `deepseek/deepseek-chat`, `deepseek/deepseek-coder`, `qwen/qwen-2.5-coder-32b-instruct`. **No Kimi/Moonshot row.** The Anthropic rows are a stale catalogue (claude-3-5-sonnet-20241022, claude-3-opus-20240229 …).
- **Agents**: 7 in the operator workspace; 3 are Claude Code agents (Bob `fable`, two smoke agents `sonnet`).
- **Session mode**: tickets 68, 70–73 ran end to end today; consent (drag / Run Now) verified; host online.
- **Heartbeat**: Bob's heartbeat is scheduled every 15 min (`HeartbeatService` cron 0,15,30,45) with the prompt "Read the board and pick up any task that are assigned to you and progress them." — no run recorded yet today; what a heartbeat does for a Claude Code agent is exactly the PRD-234 S3 question (the factory refuses cli agents by design).
- **Explorer root** for the operator workspace: `analytics artifacts content graph projects reports repos sessions tasks` — `projects/` (LOCAL_PROJECTS_DIR, read-only) and `sessions/` (session mode) are both live.
## 1. What the chat "Code" option already is (verified in code, 2026-09-03)

The Code pill (`frontend/components/chatbot/chat-mode-bar.tsx:75-84`) opens a `coding_canvas` widget for the current workspace (`chat.tsx:115-144`): chat on the left (35%), Canvas on the right. The Canvas is two things:

- **WorkspaceExplorer** (`frontend/components/workspace/WorkspaceExplorer.tsx:261-391`): file tree, **Monaco editor with tabs and Ctrl-S save**, a collapsible terminal. The terminal is not a PTY: an input box plus ANSI rendering, one `POST /api/workspaces/{id}/exec` per command (`TerminalWidget/InteractiveTerminal.tsx:78-80`). Also mounted standalone at Deliverables → Explorer.
- **CanvasSessionPanel** (`CodingCanvasWidget/useCanvasSession.ts:143-260`): start/stop a "canvas session", stream its turns over SSE, approve or deny each edit in a **Monaco DiffEditor card** (`DiffCard.tsx`), then `commit-preview` → commit + push on a `canvas/<session>` branch (`CanvasCommitControl.tsx`, `orchestrator/api/workspace_files.py:452-549`).

Behind it, the workspace-worker (`services/workspace-worker/main.py:1068-1082`) serves list/read/write/grep/download, `/exec` (allowlist of ~90 binaries: git, python, node, pytest, npm, cargo, go, make…; 100 KB stdout; 120 s default, 300 s cap — `executor.py:35-105`), `/git` (status diff add commit push pull log branch checkout stash show blame fetch clone; clones land in `repos/<name>`), and path safety that **rejects absolute paths and symlink escapes** (`workspace_manager.py:245-270`). The `projects/` bind mount passes that check because it is a real mount, not a symlink.

**The engine problem.** The canvas session's agent is a `ClaudeSDKClient` running inside the worker container (`canvas_session_service.py:126-131`) and it needs `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in the worker (`worker_config.py:30-52`). That is precisely the credential PRD-234 refuses to touch. So today the product has two disjoint coding engines: Canvas (API-billed, workspace-confined, live UI with approval cards) and session mode (the user's own `claude`, subscription, board-driven, **no live UI beyond `runtime_ref`**).

### Gaps against a Cursor-like local experience

| # | Capability | Today | Tag |
|---|---|---|---|
| 1 | Browse my repos | tree works; `projects/` visible; no "these are my repos" notion; mount hardwired to the default workspace | reuse + extend |
| 2 | Open and edit a file | Monaco + tabs + save; 2 MB cap; worker `/files/grep` exists but the UI never calls it | reuse + extend |
| 3 | Save into `projects/` | read-only mount by design → 400 on write | decision (rw dial vs "edits go through a session") |
| 4 | Diff | only inside approval cards; no file-vs-HEAD or branch diff | extend (reuse DiffEditor) |
| 5 | Run / test | one-shot exec, no streaming, no long-running servers | extend |
| 6 | Terminal | no PTY, no ctrl-c, no colours beyond ANSI | new (worker PTY endpoint) |
| 7 | Chat scoped to a repo | chat is workspace-scoped; `PageContext` carries route/selected id; attachments are files, never directories | extend (`selected={type:'repo'}` + `working_directory` on the widget) |
| 8 | Agent applies fixes | two engines (see above); CLI sessions have no live UI/approval cards | extend (bridge host events → canvas SSE) |
| 9 | Resume / takeover | `claude --resume` shown as text on the ticket; transcript path captured, never rendered | extend |
| 10 | Git commit / branch / PR | canvas-session-scoped commit; no branch switcher, staging or PR | extend |
| 11 | Register a local folder | GitHub clone only; codegraph indexing exists but no UI hook | extend |

### The five reuse paths (in order of leverage)

1. **Repo-scoped Canvas**: the Code pill opens the Canvas on a `rootPath` (a folder under `projects/` or a ticket's `sessions/<id>`), threaded `chat.tsx` → `CodingCanvasWidget` → `WorkspaceExplorer` (`fetchDirectory(rootPath)` instead of `'.'`) → terminal cwd. Backend unchanged (`path=` already accepts subdirectories).
2. **Conversation scoped to that folder** through the existing page-context lane (`usePageContext({selected:{type:'repo', id: rootPath}})`), so Auto's `workspace_read_file/write_file/list_dir/grep/exec/git` tools (`modules/tools/execution/exec_workspace.py:209-360`) act there. No new tool, no new table.
3. **Session mode on the Canvas**: publish the host's PreToolUse/PostToolUse/Stop events (`cli_host_service.record_events`) onto the canvas SSE channel in `canvas_events.py` shapes; `useCanvasSession` renders turns, file-edit refreshes and — when a `permission.request` is emitted — the approval cards, unchanged. This is the piece that makes "chat with the agent and watch it fix my code" run on the user's own Claude subscription.
4. **Ticket ↔ chat**: "Open this ticket's session in chat" (`/chat?repo=<cwd>&ticket=<id>`), a copyable takeover command, the transcript link.
5. **Where edits may land**: keep `projects/` read-only for the worker and route edits through a Claude Code session (its files are already registered as deliverables), or add an explicit `LOCAL_PROJECTS_RW=true` dial. Symlinks are not an option (resolved and rejected); bind mounts are.
## 2. Auto as the manager — what exists, what lies, what is missing (verified in code)

### 2.1 Eight dispatch lanes, one of them knows about Claude Code agents

Every path that runs an agent ends in `AgentFactory.execute_with_prompt`. The factory refuses a `runtime: cli` agent by design (`core/cli_runtime.py:46` `RuntimeMismatchError.as_result` → an error dict). Only the board lane turns that into a ticket the host can claim. The others do this today:

| Lane | Entry | For a Claude Code agent today | S3 needs |
|---|---|---|---|
| Board ticket | `api/board_tasks.py:1325` → `_park_for_cli_host` | parked `assigned`, host claims | done |
| Board dispatch loop | `services/board_dispatcher.py:499` (`runtime_predicate_sql`) | excluded by SQL; host claims | preflight (host online) |
| **Agent heartbeat** | `services/heartbeat_service.py:904` `_agent_tick` → factory `:1003` | **records `status: "success"` with the refusal text as a finding** (`:1008-1027` never checks `exec_result["status"]`) — a silent lie, and the same bug hides API-agent failures | file a ticket from the heartbeat prompt; honest row |
| Scheduled task (calendar) | `services/scheduled_task_service.py:410` `_trigger_agent_chat` | error dict logged as the result; the "background message" never fires | file a ticket for the target agent; reply "queued for your Claude Code session" |
| Mission step | `services/coordinator_service.py:2042` → `:2251`; agent chosen by `modules/coordination/dispatcher.py:248` (`AgentMatcher.match`) | mission task fails with the mismatch text | mission step → ticket; step closes from the ticket's terminal status (or exclude cli agents from mission matching for now) |
| Channel / @-mention | `channels/base.py:112` → `core/routing/engine.py:79` → factory `:170` | **no reply at all** (empty response text) | file a ticket; reply the queued line |
| Webhook | `api/webhooks.py:1199` | error dict returned to the caller | file a ticket, return its id |
| Composio trigger | `api/composio.py:866` | logged, nothing surfaces | file a ticket |

No lane burns an API call for a cli agent (the factory guard holds). The S3 gap is honesty plus a ticket. Reusable primitives: `handlers_board_tasks.create_board_task` (`:120`), `services/board_task_bridge.py`, and the ticket body shape `modules/coordination/dispatch_contract.py:15`.

### 2.2 What Auto can and cannot do with the board today

Tools: `platform_create_task` (title, description, priority, `assigned_agent_name`, tags, parent, approval_action, status, auto_approve), `platform_list_tasks`, `platform_board_summary`, `platform_get_task`, `platform_assign_task`, `platform_update_task_status` (bulk ≤100). It **cannot** set `review_mode`, `sla_deadline` (a due date), `raw_prompt`, attachments or `source_type` — all columns exist (`core/models/core.py:1573-1588`); two schema fields away.

The ASSIGN lane (`consumers/chatbot/auto.py:66`, fix #681 at `:988-1013`) only dispatches to an agent **the user literally named**; the unresolved branch says "Do NOT guess or auto-pick an agent". Auto is currently forbidden from picking. And what Auto sees when planning (`_planning_context_block`, `auto.py:1053`) is `- name (role): description[:80]` — no skills, tools, runtime, cost or outcomes. `platform_list_agents` returns skills_count/tools_count/model/provider/tags/team but **not `configuration.runtime`**, so Auto cannot tell a Claude Code agent from an API agent.

### 2.3 "Best-suited agent" already exists as code

- `modules/coordination/agent_matcher.py:143` `AgentMatcher.rank/match`: skill match 0.40, tool coverage 0.25, model fit 0.15, availability 0.10, history 0.10 (30-day verification scores), plus PRD-164 semantic 0.35 and field signals 0.15; threshold 0.4; every result carries a human-readable `reason` (`_compose_reason`, `:673`). It takes an `OrchestrationTask`; `_rank_with_context` (`:287`) is the pure core — a title+description adapter is all a board ticket needs.
- `core/routing/semantic_indexer.py:33/230` builds a semantic card per agent (name, description, type, tags, persona, skills with descriptions, connected apps, plugins) into `agents.semantic_embedding`, reindexed on every agent write; `find_similar_agents` already ranks agents by free text — it is the channel router's Tier 2.5.
- PRD-232's intent graph learns intent→**tool** affinities, not agent outcomes. The only agent-outcome signal is mission verification scores; board tickets contribute nothing yet.

Minimal wiring: a read tool `platform_recommend_agent(objective)` (`find_similar_agents` + `AgentMatcher.rank` → top-3 `{name, score, reason, runtime, model, cost_24h}`), `runtime` added to `platform_list_agents` and to the planning block, `review_mode` + `sla_deadline` on `platform_create_task`, and the #681 grounding relaxed from "refuse" to "propose the ranked pick and confirm in-thread". Then "have my Claude Code fix the login bug by Friday" routes to Bob by construction, and "summarise these docs" to a cheap API agent.

### 2.4 Auto on Kimi (or DeepSeek) — the API lane, unchanged by session mode

Settings path: Settings → System LLM (Orchestrator tab) → `PUT /api/v1/workspaces/current/orchestrator` (`api/workspaces.py:671`), which validates the model against the catalogue (422 if unknown), runs `check_orchestrator_model` (quarantine list only; allowlist empty ⇒ any catalogued model passes), and writes Auto's `model_config` plus `system_settings(orchestrator_llm)`.

Registry: **no Kimi/Moonshot row anywhere**; DeepSeek is seeded. The OpenRouter catalogue is synced by `POST /api/openrouter/sync` (`core/services/openrouter_sync_service.py:40`) and promoted on demand by `_get_or_create_from_cache`. So Auto-on-Kimi today = sync the catalogue, have an OpenRouter key, set `openrouter` / `moonshotai/kimi-k2…`. Two soft gaps: no Kimi price in `core/llm/manager.py:855` (cost audit reads $0) and no Kimi entry in `_LARGE_CONTEXT_MODELS` (model-fit under-scores it). The Anthropic rows in the registry are a stale catalogue (claude-3-x) — worth a refresh regardless.

PRD-234 keeps the two lanes apart on purpose: an OpenRouter id is refused on a `runtime: cli` agent (`core/cli_runtime.py:39`). Auto on Kimi manages; Claude Code / Codex agents execute under the user's own subscription. They coexist in one workspace.

### 2.5 Fleet and Command Center: sessions and hosts are invisible

`services/fleet_state.py:316/485` returns per agent: current, queue depth, blocked asks, watches, last activity, cost — no runtime, no host, no session. `board_tasks.runtime_ref` already carries everything a live line needs (session id, host, model, live tool, recent tools, transcript, usage) and hosts have `is_online()`. Plug points: `_assemble_fleet` (+`runtime`, +`session` from the current ticket), `get_fleet_state` (+`hosts`), `fleet-tab.tsx` live line variant + a host card (SessionModeTab's row is reusable), and `handlers_fleet._work_phrase` so Auto can say "your Claude Code host is offline".
## 3. What the strongest tools do (and what they refuse to build)

Ecosystem as of 2026-09: Vibe Kanban is sunsetting (Apache-2.0, code still borrowable), Crystal went closed, Terragon is dead; munder-difflin (MIT), Claude Squad (AGPL — read, don't vendor), OpenHands (MIT), Roo/Cline (Apache-2.0) and Conductor (proprietary) are alive.

| Capability | Best idea worth copying |
|---|---|
| Session list / live log | munder's two planes: raw PTY bytes on one channel, Claude Code **hook events** on another; the UI renders from events, never from log scraping. (We already have the event plane: S1b hooks.) Vibe Kanban normalises every agent CLI's output to one schema so one UI renders all of them. |
| Takeover | Claude Squad: `Enter` attaches your real terminal to the agent's tmux session, `Ctrl-Q` detaches with the agent still running. munder parks harness messages while a human is typing (one writer per terminal). |
| Worktree per task | Conductor: worktree + branch per workspace, a copy-list for gitignored files (`.env.local`) and a per-repo setup script, or the agent's first test run fails. |
| Diff review | Vibe Kanban: inline review comments are fed back to the agent as the next prompt — review is an input, not a viewer. |
| Merge / PR | Vibe Kanban's Merge / Create PR / Rebase header (shelling to `git`/`gh`); Conductor's Checks tab (status, CI, comments, todos). |
| Task model | Vibe Kanban's `Workspace → Session → ExecutionProcess`: a ticket has N attempts, each attempt N runs; retries and "try a second agent" are rows, not special cases. |
| Cost | Claude Code exports OTel natively (`claude_code.cost.usage`, `claude_code.api_request` with tokens, model, `query_source`, keyed by `session.id`): `CLAUDE_CODE_ENABLE_TELEMETRY=1` + an OTLP endpoint; zero parsing. |
| Manager / routing | munder's "Michael": an ordinary `claude` whose system prompt is the routing policy over a registry `{id, name, role, capabilities, status}` — mechanism in the harness, intelligence in the prompt. Claude Code subagents route by the `description` text. Roo's Orchestrator delegates a subtask with its own mode **and model profile**. |
| Autonomous continuation | munder's Stop-hook trick: if the inbox has unread messages when `Stop` fires, return `block` so the agent keeps working. That is a heartbeat loop for free. |
| Preview | Vibe Kanban's per-worktree dev server + preview URL — the useful 10% of an IDE. |

**What every serious tool refuses to build:** a real code editor (Vibe Kanban shells out to VS Code/Cursor; OpenHands deleted its embedded VS Code tab), a writable browser terminal as the primary control surface, a git client with conflict resolution, a debugger, its own agent CLI or anything that touches auth. Cloud managers (Devin, Factory, Codex, Jules) all use the same shape: a parent scopes and monitors, children run isolated, results compile back.
## 4. The plan — Automatos as the local manager of Claude Code (and friends)

**Thesis (from the evidence, not taste):** Automatos should not become an IDE. It should be the manager the IDEs lack: Auto plans and assigns, the board and calendar and heartbeats drive, every unit of work is a ticket → attempt → run executed by the user's own `claude` (or Codex) in a worktree, the result lands as deliverables and a report, cost and state come from the event plane and OTel, and the user takes over in their real terminal or editor with one click. The chat "Code" mode is the window onto that: repo tree, viewer, diff cards, live session, approvals — on the user's subscription, never a second billed engine.

### Wave 1 — S3: every lane files a ticket; Auto can pick; the fleet sees sessions (this week)
1. **`services/cli_ticket_lane.py`** — one `file_cli_ticket(...)` used by heartbeat, scheduled task, channel/@-mention, webhook, Composio trigger (mission: exclude cli agents from matching for now). Ticket body in the dispatch-contract shape, `source_type` per lane, dispatcher NOTIFY. Preflight: no host online ⇒ ticket `assigned` with a visible reason, released on the next host heartbeat.
2. **Heartbeat honesty** — an error dict from the factory is an error, for API agents too (today it is stored as a green finding). A Claude Code agent's heartbeat files a ticket from its heartbeat prompt.
3. **Auto's manager upgrade** — `runtime` on agent listings and the planning block; `platform_recommend_agent(objective)` over `find_similar_agents` + `AgentMatcher.rank` (top-3 with reasons, runtime, model, 24h cost); `review_mode` + `sla_deadline` on `platform_create_task`; the ASSIGN lane proposes the ranked pick when no agent is named (still confirmed in-thread — the JIRA-ADMIN rule stands).
4. **Fleet / Command Center** — `runtime` + a session live line per agent (from `runtime_ref`), a CLI host card, and Auto's fleet phrasing ("your Claude Code host is offline").
5. **Auto on Kimi** — OpenRouter catalogue sync, Kimi price + context entries, a refreshed Anthropic catalogue; then Settings → System LLM → Orchestrator: `openrouter` / `moonshotai/kimi-k2…`. Needs an OpenRouter key.

### Wave 2 — Code mode on session mode
1. Repo-scoped Canvas: the Code pill opens on a folder (`projects/<repo>` or `sessions/<ticket>`); the conversation is scoped to it through page context, so Auto's workspace tools act there.
2. Bridge host events → the canvas SSE contract: live turns, file-edit refresh, approval cards from `PermissionRequest` — the CLI session becomes watchable and steerable in chat.
3. Ticket ↔ chat: "open this session in chat", copyable takeover, transcript link; "Open in VS Code / Cursor" deeplinks on files and worktrees.
4. Edits policy: `projects/` stays read-only for the worker; edits go through a session (registered as deliverables). An explicit rw dial only if asked.

### Wave 3 — attempts, git flow, cost
1. `ticket → attempt → run` as rows; a second attempt with another agent is a click; Auto learns from which attempts merged.
2. Merge / Create PR / Rebase via `gh`, a Checks panel; a copy-list + setup script per repo for worktrees.
3. Per-worktree dev server + preview URL.
4. Cost from Claude Code OTel into the existing monitoring stack; tmux-based sessions with a read-only mirror and one-click attach.
5. Codex adapter (S5), then the community lane (S6).

### Decisions for the owner (each changes the build)
- **D-A Code mode engine on local:** default the Code pill to session mode (Wave 2.2) and keep the API-billed Canvas engine only for API agents. *Recommended.*
- **D-B Edits into your repos:** read-only mount, sessions write. *Recommended.* Alternative: an rw dial.
- **D-C Auto's model:** Kimi K2 via OpenRouter (you supply the key) or stay on Gemini Flash until the catalogue refresh lands.
- **D-D Missions for Claude Code agents:** exclude from mission matching now (Wave 1) vs a full ticket bridge (Wave 3).

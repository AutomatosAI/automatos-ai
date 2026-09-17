# PRD-245: Session lane flow — ticket sessions stop tripping over the gate, and get Automatos tools through a loopback MCP bridge

> **Status:** BUILDING 2026-09-17 (owner: "Love it lets write the PRD for this work and get started on it ASAP") — on `feat/prd-245-w0-session-lane` from `main` 757814531. Grounded @ the Phase 1 run (tickets #116–#121, six `runtime: cli` agents, 2026-09-17 12:22 UTC), the local database, the cli-host at that commit, and the design draft *Automatos MCP Bridge* (artifact `b74eab8d`, 2026-09-17). Settles PRD-239's open decision **D-4**.

---

## Framing (CLAUDE.md §3)

**Extension + one refactor + two bug-class fixes.** PRD-234 built the session runtime (a `runtime: cli` agent's ticket runs as the operator's own Claude Code or Codex session, claimed by the paired cli-host) and PRD-239 gave it persona, skills, playbooks, missions and the Runtime Canvas. Two things were never built: a way for a session to **call Automatos** (board, reports, knowledge, questions, Composio), and a policy gate that judges shell commands the way a shell does. The Phase 1 run made both visible in one afternoon. This PRD (1) refactors the gate so honest commands are not held, (2) fixes two bug classes (any denial forces review; a ticket in review counts as open for heartbeat reuse), and (3) extends the session with a fixed set of platform tools served by the local backend over MCP on the loopback address. No new tables. No migration. One new router.

## What the test actually showed (grounded)

Six agents, one small ticket each, all sessions on one host, started within 1.5 s of each other.

| Ticket | Agent | Denials | Held shell commands (120 s each) | Deliverable registered | Landed |
|---|---|---|---|---|---|
| #116 | RESEARCHER | 10 | 4 | yes | review |
| #117 | WRITER | 4 | 3 | none (no notes existed; question blocked) | review |
| #118 | OPS | 7 | 4 | yes | review |
| #119 | TRACKER | 4 | 2 | yes | review |
| #120 | AUTOMATOS-DEV | 3 | 2 | yes | review |
| #121 | DRG-DEV | 5 | 4 | **no** (written under the cli-host session folder) | review |

- **Every ticket landed in review because of `force_review=bool(denials)`** (`services/cli_host_service.py:1402`). A refused Read of a sibling ticket, a ToolSearch attempt and a question prompt count the same as "could not run the tests".
- **Nineteen holds, zero answered.** A held command becomes a card only in that ticket's Coding Canvas over live SSE (`CodingCanvasWidget/useCanvasSession.ts` → `POST /api/v1/tasks/{id}/session-decision`). No bell, no Questions tab, no Telegram. About 38 minutes of dead time.
- **The holds were false.** Reproduced with `policy.decide_bash`: `find` and `grep` alone pass. The holds came from `|` and `;` inside quoted patterns (`_split_compound` is quote-unaware, `policy.py:110-112`), the `git -C <path> …` form (prefixes are `git status`, `git log`, …), `for` loops, `VAR=` assignments, and verbs missing from `DEFAULT_BASH_ALLOW` (sort, uniq, cut, sed, awk, tr, date).
- **Bash reads are not confined.** `cat ~/.automatos/cli-host/host.json` was ALLOWED (only the Read tool is path-checked); the RESEARCHER transcript now holds the host's pairing token. Ops action already flagged: re-pair the host.
- **No platform tool exists in a session.** Launch is `--permission-mode acceptEdits --setting-sources user --strict-mcp-config` with a hooks-only settings file (`presets.py:148-150`); the policy denies every unknown tool name (`policy.py:177`); `AskUserQuestion` passes the policy and then dies at the PermissionRequest the host denies (`session.py:256-260`). Meanwhile `cli_session_prompt.py` renders full skill bodies that instruct the agent to call `composio_execute`, `platform_submit_report`, `platform_board_summary` and `search_knowledge`. GMAIL, GOOGLECALENDAR and COMPOSIO_SEARCH were `active` in `composio_connections` the whole time.
- **Heartbeat freeze mechanism is live.** `OPEN_STATUSES` includes `review` (`cli_ticket_lane.py:30`) and `file_cli_ticket` reuses an open ticket per `heartbeat:agent:<id>` (`:156-159`). Agent 15's #93 absorbed 236 heartbeats until the agent was deleted. RESEARCHER/WRITER/OPS/TRACKER now have daily 09:00 UTC heartbeats and will do the same on their first denied run.
- **Codex tickets inherit the operator's MCP servers.** `adapters/codex.py:_config_text` seeds the per-agent `config.toml` from `~/.codex/config.toml` verbatim, `[mcp_servers.*]` tables included. Claude tickets are protected by `--strict-mcp-config`; Codex tickets are not.
- **Deliverables.** `_register_session_deliverables` registers only files under the workspace volume; DRG-DEV's note under `~/.automatos/cli-host/sessions/121/` was invisible to Deliverables.

## Decisions (adopted 2026-09-17 from the design draft; each reversible in this PR)

- **D1 · Transport: an MCP endpoint inside the backend, on loopback.** `POST /api/v1/session-tools/mcp`, JSON-RPC over HTTP, hand-written (see S1.2 — the SDK would force a uvicorn upgrade). The backend already listens on 127.0.0.1:8000 for the host's claim, events and result. No stdio process per session; a stdio shim is built only if a served CLI cannot speak HTTP MCP.
- **D2 · Seven fixed tools, scope on the backend.** `board_summary`, `list_tasks`, `update_ticket`, `submit_report`, `search_knowledge`, `ask_human`, `composio_execute`. The advertised list is byte-identical for every session of an agent (Claude Code's prompt cache); per-agent scope (connected apps, own ticket) is enforced server-side, never by varying the list. Waves add tools once per deploy, not per session.
- **D3 · A session never moves its ticket to `done`.** `update_ticket` may set `in_progress`, `blocked` or `review` and add a note. Done stays a fact the host observed on result.
- **D4 · Identity is a per-ticket token.** Minted at claim, hash stored on the ticket, written by the host into `<session>/mcp.json` (0600), never on a command line, never in the environment of a Claude session, revoked when the result lands or the ticket is cancelled or requeued.
- **D5 · Same executor, same records.** Every call runs through `UnifiedExecutor.execute_tool(tool_name, parameters, agent_id, workspace_id, trace_id, caller_context=None)` — the API agents' path — so admin-gated actions refuse, tool telemetry logs the call, and the ticket's live log shows it.
- **D6 · Holds and denials mean different things.** A platform tool name is allowed or denied by the policy, never held. Shell holds stay, but become visible in the Questions tab, the bell and the Telegram bridge. Review is forced only by a hold (expired or operator-denied); a refused Read, an unknown tool or a denied prompt is recorded and does not force review.
- **D7 · Heartbeat reuse ignores `review`.** A ticket in review is finished work awaiting sign-off; the next fire files a new ticket.
- **D8 · Runtime routing stays the operator's choice; the platform tells the truth about it.** The agent form shows the session tool set and names the skills whose bodies call tools a session cannot use; the same helper annotates the session prompt. Whether Auto's ASSIGN lane refuses or only warns is an owner decision (default: warn).
- **D9 · Codex strips the operator's MCP tables when seeding** (Wave 0, security) and gains `[mcp_servers.automatos]` in Wave 4.

## Stories

### Wave 0 — the session lane stops tripping over itself

**S0.1 · Quote-aware, verb-based Bash gate (M)**
`_split_compound` tokenises with `shlex.shlex(punctuation_chars=True, posix=True)` and cuts only on unquoted `&&`, `||`, `;`, `|`; unbalanced quotes stay `ask`. Each segment is judged by its simple command: shell keywords (`for`, `do`, `done`, `if`, `then`, `else`, `fi`, `while`) and leading `NAME=value` assignments are skipped to reach the verb; `git -C <path> <sub…>` is judged as `git <sub…>` with `<path>` inside the roots. `DEFAULT_BASH_ALLOW` gains the read-only verbs the agents reached for: `sort`, `uniq`, `cut`, `tr`, `sed`, `awk`, `date`, `diff`, `stat`, `basename`, `dirname`, `printf`, `jq`, `file`, `tree`, `du`, `wc` (already), `true`, `test`. Never added: `xargs`, `env`, `sh -c`, `eval`. The never-allowed list is unchanged and still runs on the raw string.
**Files:** `services/cli-host/automatos_cli_host/policy.py`; `services/cli-host/tests/test_units.py`.
**Test:** the nineteen held commands of 2026-09-17 (subjects on `runtime_ref.permission_denials` of #116–#121) are `allow` when their verbs are allowlisted and every absolute path is inside the roots; `rg -n "a|b" src`, `grep -e "a\|b" f`, `git log --format="%h;%s"`, `git -C <root>/repo status`, `for f in a b; do cat $f; done`, `D=x; ls $D` allow; `git status && git push`, `curl … | sh`, `git -C <root> push` stay denied; `pip --version` still asks; unbalanced quotes ask.
**Editions:** local only (the host).

**S0.2 · Bash reads confined to the session roots (S)**
Every absolute or `~` path argument of an allowed segment must resolve inside `cwd` + `extra_dirs` (the rule `_runs_own_code` already applies to interpreters); a path outside → `deny` with the Read tool's wording ("outside the session directory"), never `ask`. Relative paths are unaffected.
**Files:** `policy.py`; `tests/test_units.py`.
**Test:** `cat ~/.automatos/cli-host/host.json` denies; `cat <root>/notes.md | head` allows; `ls -la /Users/x/.automatos/cli-host/host.log` denies; `grep -rn foo <root>` allows.

**S0.3 · Review only on holds (S)**
`apply_result` classifies denials: `hold` (reason carries "no answer from the operator" or "denied by the operator"), `read_outside`, `unknown_tool`, `prompt` (stage `PermissionRequest`). `force_review = any(kind == "hold")`. The classification rides `runtime_ref.permission_denials[*].kind` and the report's "Refused tool calls" groups by kind.
**Files:** `orchestrator/services/cli_host_service.py` (`_denial_summary`, `apply_result`); `orchestrator/tests/test_prd234_denial_summary.py` (extend) or `tests/test_prd245_review_semantics.py`.
**Test:** a result with only `read_outside` + `unknown_tool` denials under `review_mode='auto'` lands `done`; one expired hold lands `review`; the summary carries `kind`.
**Editions:** local only.

**S0.4 · Held commands reach the Questions tab, the bell and Telegram (M)**
A session `PermissionRequest` event (policy `ask`) creates a PRD-225 question row through the shared `ask_human` internals: `subject_type='cli_permission'`, `subject_id='<task_id>:<request_id>'`, question = "Allow this command in ticket #N?" with the command in backticks, options `allow` / `deny`, expires with the host's ask timeout. The Questions tab, the bell and the Telegram poll bridge need nothing new. On answer, `_requeue_subject`'s new `cli_permission` branch calls `cli_host_service.decide_session_permission(task, request_id, approved)` and does NOT requeue the ticket. When the session ends with the hold unanswered, the row is expired (as `expired_permissions` already records it). The Canvas card keeps working; both paths resolve the same `request_id`.
**Files:** `services/cli_host_service.py` (`record_events` → ask row; `apply_result` → expire), `api/approval_grants.py` (`_requeue_subject` branch), `modules/tools/discovery/platform_executor.py` only if `ask_human` needs the new subject type whitelisted; `orchestrator/tests/test_prd245_session_asks.py`.
**Test:** a PermissionRequest event creates one row per `request_id` (idempotent on re-flush); answering `allow` marks the ticket's decision approved and the next event flush carries it to the host; answering `deny` records "denied by the operator"; the row expires on result; no requeue happens; the list route with `kind=question` returns it with the ticket as cascade.
**Editions:** local only.

**S0.5 · Heartbeat reuse ignores `review` (S)**
`cli_ticket_lane.OPEN_STATUSES` splits into `REUSABLE_STATUSES = ("inbox", "assigned", "in_progress", "blocked")` used by `open_ticket_for_source`, and the terminal set is unchanged for the waiting lanes.
**Files:** `services/cli_ticket_lane.py`; `tests/test_prd234_s3_cli_ticket_lane.py` (extend).
**Test:** an existing `review` ticket for `heartbeat:agent:57` is not reused; an `in_progress` one is.
**Editions:** both (the lane exists in both; only local can have cli agents).

**S0.6 · The session prompt tells the truth about tools (S)**
`cli_session_prompt.session_system_prompt` gains a `## Tools in this session` block: file tools inside the working folder and the ticket folder; the Bash allowlist verbs (rendered from the host's default list, stable); web tools; "platform tools named in your skills — `composio_execute`, `platform_*`, `search_knowledge`, `scratchpad_*`, `workspace_*` — are NOT available in a session unless listed above; do not call or wait for them". A helper `session_tool_gaps(agent, available)` returns the tool names each active skill's `prompt_template` mentions that are not available; each rendered skill header gains one line "In a session you cannot call: …". The host's `SESSION_RULES` adds where deliverables go and how to ask (until S2.1: state the question in the final message and end the turn). Stable per agent — no ids, dates or counters.
**Files:** `orchestrator/services/cli_session_prompt.py`; `services/cli-host/automatos_cli_host/session.py` (`SESSION_RULES`); tests `test_prd239_session_prompt*` (extend) + `services/cli-host/tests/test_units.py::test_system_prompt_is_stable_per_agent`.
**Test:** RESEARCHER's web-research skill renders with the gap line naming `composio_execute`, `search_knowledge`, `platform_submit_report`; the prompt is identical across two renders; no ticket id appears.
**Editions:** local only.

**S0.7 · Deliverables written in the session folder still land (S)**
On turn end the host copies files the session wrote under its own session folder (from `files_touched`, excluding `ticket.md`, `settings.json`, `system_prompt.md`, `terminal.log`, `mcp.json`) into `<default_root>/sessions/<ticket>/` and reports the copies in `files_touched`; `build_ticket_file` adds a `Deliverables` line naming that folder. The backend registers them as today.
**Files:** `services/cli-host/automatos_cli_host/session.py`; `services/cli-host/tests/test_session_fake_claude.py`.
**Test:** the fake claude writes `note.md` into the session dir → the result's `files_touched` lists `<default_root>/sessions/<ticket>/note.md`; host-owned files are never copied.
**Editions:** local only.

**S0.8 · Codex seeding drops the operator's MCP tables (S)**
`_config_text` removes every `[mcp_servers.<name>]` table (and `[mcp_servers]` header) from the seeded base before appending the hooks and trust sections.
**Files:** `services/cli-host/automatos_cli_host/adapters/codex.py`; `services/cli-host/tests/test_adapters.py`.
**Test:** a base config with two `[mcp_servers.*]` tables and a `[model]` table seeds with the model table kept and no `mcp_servers` text.
**Editions:** local only.

**S0.9 · Docs (S)**
`services/cli-host/README.md`: the allowlist rules as built (S0.1/S0.2), the review rule (S0.3), where held commands appear (S0.4), the deliverables folder (S0.7). `docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md` §12 records D-4 as settled by this PRD.

### Wave 1 — the bridge and the read tools

**S1.1 · Per-ticket token at claim (S)**
`claim_for_host` mints `session_token = secrets.token_urlsafe(32)`, stores `sha256` on `runtime_ref.session_token_sha256` (no migration; JSONB), returns the token and `session_tools_url` + `session_tools: [names]` in the claim payload. `resolve_session_token(db, token)` looks up `in_progress` tickets by the hash and returns `(task, agent, workspace_id)`; `apply_result`, cancel and the sweeper's requeue clear the hash.
**Files:** `services/cli_host_service.py`; `api/cli_hosts.py` (payload shape); `tests/test_prd245_session_token.py`.
**Test:** mint → resolve; a token of a finished ticket resolves to nothing; a garbage token resolves to nothing; the hash, never the token, is on the ticket; the claim response carries the token once.

**S1.2 · The MCP endpoint (M)**
New `orchestrator/api/session_tools.py`: a `FastMCP` server (official `mcp` SDK, streamable HTTP, stateless) mounted at `/api/v1/session-tools/mcp`, bearer auth resolving the identity of S1.1 on every request. `tools/list` advertises the wave's tools with stable descriptions; `tools/call` maps `<name>` → the backend action (`board_summary→platform_board_summary`, `list_tasks→platform_list_tasks`, `update_ticket→platform_update_task_status` with `task_id` forced to the ticket and `done` refused, `submit_report→platform_submit_report` with the agent name forced, `search_knowledge→search_knowledge`) and calls `UnifiedExecutor.execute_tool` with the ticket's `agent_id`/`workspace_id`, `trace_id=f"session:{task_id}"`, `caller_context=None`. A per-ticket cap `SESSION_TOOLS_MAX_CALLS_PER_TICKET` (config, default 200, in the config-surface report) counts on `runtime_ref.platform_calls`. Route-manifest entry + count bump. Dependency `mcp` pinned in `orchestrator/requirements.txt`. **Verify at build:** the SDK's Starlette app mounts under fastapi 0.115 / starlette 0.41 without pulling incompatible pins; if it does not, the fallback is a minimal JSON-RPC handler for `initialize`, `tools/list`, `tools/call` (the only methods Claude Code needs) — decided at build, recorded in the PR.
**Files:** `api/session_tools.py` (new — transport + auth + the allowance), `services/session_tools_rpc.py` (new — the protocol, pure), `services/session_tools.py` (new — the tool table and the forced scope), `router_manifest.py`, `config.py`, `reports/route-manifest.json` (806 → 808), the config-surface report; `tests/test_prd245_session_tools.py`.
**Test:** the list is byte-stable and every schema is one the client accepts; scope forced (own ticket, `done` refused, undeclared fields dropped); the wire answers what the client really sends; a refusal, a failure and a crash all return as `isError` output, never a dead session; the allowance refuses in words the model can act on.
**Editions:** local only (router mounted only when `CLI_RUNTIME_ENABLED`).

**Transport, decided at build (the PRD's named fallback taken).** Not the official `mcp` SDK: it requires `uvicorn>=0.31.1` (and a new httpx major) while this backend pins `uvicorn==0.24.0` and boots on it, so taking it would mean upgrading the ASGI server under a mature app for four JSON shapes — the class of change that already cost this repo a day (`starlette-coldboot-include-router`). The protocol a tools-only server must answer is small, stable and now tested. **What the real client does, verified against the installed Claude Code 2.1.267 and the docs, each item a silent failure if missed:** its v2 runtime sends a `server/discover` capability probe BEFORE `initialize` (answer: HTTP 200 + JSON-RPC `-32601`, which classifies us as a legacy server and lets the handshake proceed); it asks for protocol `2025-11-25` and refuses any echo outside its pre-2026 list, `2026-07-28` — the version its own probe names — included; `GET` on the endpoint must be 405 (no server-initiated stream), which is what FastAPI answers for an undefined method, so no `GET` is defined there; a response must carry `Content-Type: application/json` or the request dies as "unexpected content type"; a notification gets 202 with an empty body; `prompts/list` and `resources/list` return empty lists rather than errors, since the client's discovery step asks for them; a 401 must NOT carry `WWW-Authenticate`, or the client starts an OAuth discovery this endpoint has none of; and the token is written LITERALLY into `mcp.json`, because the client substitutes an empty string for a `${VAR}` whose name looks like a credential.

**S1.3 · Host writes `mcp.json` and passes the flag (S)**
`presets.CLAUDE` gains `mcp_config_flag="--mcp-config"`; `session.py` writes `<session_dir>/mcp.json` (0600) from the claim payload (`{"mcpServers": {"automatos": {"type": "http", "url": …, "headers": {"Authorization": "Bearer …"}}}}`) and `launch_args` appends the flag; `--strict-mcp-config` stays. The token is never in argv, env, logs or the report; `assert_args_honour_invariant` gains the token as a forbidden argv substring. Codex: Wave 4.
**Files:** `presets.py`, `adapters/base.py` (`LaunchContext.mcp_config_path`), `adapters/claude.py`, `session.py`; `tests/test_session_fake_claude.py`, `tests/test_units.py`.
**Test:** the fake claude receives `--mcp-config <session>/mcp.json --strict-mcp-config`; the file is 0600 and holds the token; the token appears nowhere in the result payload or terminal log.

**S1.4 · The platform tool class in the adapters and the policy (S)**
`ToolClass.PLATFORM`; `ClaudeAdapter.tool_intent` maps `mcp__automatos__<name>` → `ToolIntent(cls=PLATFORM, command=<name>)`; `PolicyContext.session_tools` (from the claim payload); `decide`: PLATFORM allowed iff `<name>` in `session_tools`, else `deny` (never `ask`). Any other `mcp__*` name stays denied.
**Files:** `adapters/base.py`, `adapters/claude.py`, `policy.py`, `session.py`; `tests/test_units.py`, `tests/test_adapters.py`.
**Test:** `mcp__automatos__board_summary` allows; `mcp__automatos__delete_workspace` denies; `mcp__other__x` denies; none of them ever `ask`.

**S1.5 · The agent form shows the session tools and names the skill gaps (S, frontend + one field)**
`GET /api/v1/cli-hosts/settings` returns `session_tools` (names + one-line descriptions). The agent detail payload (`api/agent_endpoints.py`) carries `session_tool_gaps: [{skill, tools}]` for `runtime: cli` agents, computed by the S0.6 helper. `runtime-section.tsx` renders "Tickets run with these Automatos tools: …" and, when gaps exist, "These skills call tools a session cannot use: web-research (composio_execute, search_knowledge)".
**Files:** `services/cli_host_service.py` (`session_mode_settings`), `api/agent_endpoints.py`, `frontend/components/agents/runtime-section.tsx` (+ its test file); `tests/test_prd239_session_mode_settings.py` (extend).
**Test:** settings carry the list; RESEARCHER's detail carries the gap; the section renders both lines; vitest + tsc green.
**Editions:** the section is already local-gated (`isLocal`).

### Wave 2 — questions

**S2.1 · `ask_human` from a session parks the ticket and the answer resumes it (M)**
The `ask_human` session tool calls the shared PRD-225 internals with `subject_type='board_task'`, `subject_id=<ticket>`; the reply text tells the agent the ticket is parked and to finish what it can and end its turn. `apply_result` lands a ticket with an open question as `blocked` (not `review`/`done`), keeping `runtime_ref.cli_session_id` for resume. On answer, `_requeue_subject`'s board branch, for a `runtime: cli` ticket, sets `status='assigned'`, `runtime_ref.resume_session_id` (+ host id) and folds the Q&A into `_ticket_prompt` under `## Answers to your questions` (same fold-in as `review_feedback`); the host claims it and runs `claude --resume`.
**Files:** `api/session_tools.py`, `services/cli_host_service.py`, `api/approval_grants.py`, `services/cli_ticket_lane.py`; `tests/test_prd245_session_ask_resume.py`.
**Test:** the call creates one question row on the ticket; the result lands `blocked`; the answer re-assigns with the session id and the prompt carries the Q&A; the Questions tab lists it with the cascade; Telegram answers reach the same path.

### Wave 3 — Composio

**S3.1 · `composio_execute` from a session (S)**
Maps to `execute_tool("composio_execute", {"action", "params"})`; scope is the existing router's (apps connected in the workspace, per-agent assignments when present); the S1.2 cap applies; the tool description carries the same calling convention the skills already document.
**Files:** `api/session_tools.py`; `tests/test_prd245_session_composio.py` (fake router).
**Test:** an action for an unconnected app is refused by the router and reported plainly; a connected one reaches the executor with the ticket's agent id; the key is never in the request or the response.

### Wave 4 — Codex

**S4.1 · Codex tickets get the bridge (M) — BUILT 2026-09-17**
`_config_text` appends `[mcp_servers.automatos]`; `prepare` sets the token variable that table names, for Codex sessions only; `CodexAdapter.tool_intent` maps its MCP tool naming to `PLATFORM`.
**Files:** `adapters/codex.py`; `tests/test_adapters.py`.

**Verified against Codex 0.154.0** — `codex mcp add --url … --bearer-token-env-var …` written into a throwaway `CODEX_HOME` and read back: the table is exactly `url` + `bearer_token_env_var`. Codex takes the bearer token from an ENVIRONMENT VARIABLE named in its config, where Claude Code takes a literal header — and that is the right route here, not merely the available one: a Codex config home is per AGENT (§6.1), so a token in that file would outlive the ticket that minted it and be read by the agent's next one, while an environment variable dies with the session process. ONE guard decides both the table and the variable: a half-offer must never put a token in the environment with no server to use it (a test found that when it was two guards). The operator's own `[mcp_servers]` tables are still stripped (S0.8), and `prepare` rewrites the config each spawn, so no previous ticket's table lingers.

**NOT verified, and the code and tests say so:** how Codex names an MCP tool in its hook payload (§6.10 is a live-run check). Every plausible spelling maps to the same tool — `mcp__automatos__x`, `automatos__x`, `automatos.x`, `automatos/x`, `mcp.automatos.x` — and anything else stays `UNKNOWN`, which the policy denies; widening it later cannot weaken the gate, because the name must still be on the ticket's own list. To settle it: run one Codex ticket and read the tool name out of that session's `terminal.log` or the ticket's `recent_tools`.

## Not in this PRD (owner decisions)

- **Gemini CLI and Grok.** Their adapters do not exist; the adapter design's rollout table puts them after Codex. Same bridge, one preset row + `tool_intent` each, when their adapters land.
- **The four daily heartbeat prompts** (RESEARCHER/WRITER/OPS/TRACKER, 09:00 UTC) ask for board tasks and reports; until Wave 1 lands they should be rewritten to session-feasible work or disabled. Agent configuration data, not code.
- **`--ask-timeout` default.** 120 s today. With holds answerable from a phone (S0.4), 300 s may fit better. Host flag, operator's call.
- **Auto's ASSIGN lane** on a cli agent whose skills need tools outside the session set: warn (default) or refuse.
- **Ops actions from the Phase 1 review:** re-pair the host (token in a transcript); move the plaintext credential files out of `~/Development`; the 777 mode on the workspace folder.

## Test plan (owner, local edition)

1. **After Wave 0:** rerun the six Phase 1 tickets. Expect: no false holds on `find`/`grep`/`sort`/`git -C`; RESEARCHER, OPS, TRACKER, AUTOMATOS-DEV end in `done`; DRG-DEV's note is a Deliverable; `cat ~/.automatos/cli-host/host.json` is denied; a genuinely off-list command shows in the Questions tab and on Telegram and, answered `allow`, runs.
2. **After Wave 1:** TRACKER's snapshot ticket uses `board_summary` and `list_tasks`, attaches its note with `submit_report`, ends `done` with zero denials; the agent form lists the tools and flags RESEARCHER's web-research skill.
3. **After Wave 2:** WRITER asks for the notes; you answer from Telegram; the session resumes and the paragraph lands as a Deliverable.
4. **After Wave 3:** OPS's calendar ticket returns tomorrow's real events in its checklist.
5. **After Wave 4:** a Codex ticket lists the same tools; none of your personal MCP servers appear in its config.

## Verify at build (no spend)

- Claude Code on this machine loads an HTTP server with headers from `--mcp-config` under `--strict-mcp-config`, and names the tool `mcp__automatos__<name>` in the PreToolUse payload.
- The `mcp` SDK mounts under the pinned fastapi/starlette; otherwise the minimal JSON-RPC fallback.
- The advertised tool list is byte-stable per agent across sessions.
- A finished ticket's token is refused; a call naming another ticket is forced back to its own.
- Codex 0.154.0: the HTTP `[mcp_servers.*]` table and bearer-env key; its hook payload's MCP tool naming.

## Merge notes

Waves land as separate PRs in order (W0 → W1 → W2 → W3 → W4); each is CI-green and tested by the owner in the local edition before the next starts. No migration in any wave. Every route addition updates `reports/route-manifest.json` and its count. DCO sign-off on every commit.

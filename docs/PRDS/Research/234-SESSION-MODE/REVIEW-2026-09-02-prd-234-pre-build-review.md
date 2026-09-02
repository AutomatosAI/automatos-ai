# PRD-234 — pre-build review (2026-09-02)

**What was reviewed.** `PRD-234-SESSION-MODE-SUBSCRIPTION-RUNTIME.md`, the research record (`RESEARCH-2026-08-30-…`) and `PRD-WAVE-OPEN-CORE.md`, all as committed on `origin/main` (5e3ed68d9; the local untracked copies are byte-identical). Every code anchor was re-checked on that tree. munder-difflin was re-cloned read-only (HEAD cc741fe3; `src/` is unchanged since the cited fc436bd). Claude Code 2.1.236 on this machine (`--help` only — no `claude -p` was run; it would spend the weekly window). Agent SDK typings 0.3.258 (npm) for the exact event shapes. Anthropic's support article on plan usage (updated 2026-06-16). PRD-234 itself is **not edited** — the findings below need decisions first.

**Verdict in one paragraph.** The architecture decision holds: a local-only host process, canvas untouched, adapters as data, the existing manager plane above. The "current reality" section is accurate to the line on today's main (§A). Three things must change before a story is briefed: (1) the transport choice is a **billing** decision Anthropic has already signalled, not an engineering preference — Q2 has to be re-decided with that fact on the table; (2) the no-silent-fallback invariant covers one of **eight** agent-execution lanes; (3) the PRD has **no security model** for a queue that runs commands on the user's laptop. The rest of this review is a list of things Claude Code already gives us that the PRD doesn't use, and gaps a build would hit in week one.

**Premise, restated so it cannot drift.** Session mode = the user's own Claude Code (later Codex, …) login, on the subscription they already pay for, driven by Automatos as manager — munder-difflin's model. API keys and OpenRouter are the *other* runtime, already supported, and stay the hybrid partner (owner decision Q6). Nothing below proposes API-key auth for sessions.

---

## A. Anchors re-verified on `origin/main` @ 5e3ed68d9

| PRD claim | Status | Note |
|---|---|---|
| Canvas welded to the SDK, one session per workspace, isolated `CLAUDE_CONFIG_DIR`, injected key/token, no result channel | ✅ | `canvas_session_service.py:138-155, 370-405`; `send_message` returns `{"success": True}` only (`:290-310`) |
| Worker image has no orchestrator code, cannot see `AUTH_EDITION` | ✅ | `services/workspace-worker/Dockerfile:33` (`COPY . .`), `worker_config.py:1-12` |
| Compose publishes 8000/6379/5432/9000 to the host | ✅ **and on all interfaces** | `docker-compose.yml:37,67,97,207` — `"${API_PORT:-8000}:8000"` binds 0.0.0.0 (see §B4) |
| Four launch sites funnel into `_launch_task_execution` | ✅ | `api/board_tasks.py:616, 937, 1138`; `handlers_board_tasks.py:498`; `board_dispatcher.py:474` |
| Completion writer is an inline block `:1184-1222` | ✅, incomplete | the block also owns the **failure** path (`:1190-1200`) and the crash handler (`:1227-1255`); the extraction must take all three or sessions fail differently from API runs |
| Completion creates no Deliverable | ✅ | only `exec_workspace._auto_register_deliverable` (`:43-118`) registers; `deliverable_service.register` (`:156`) keys on `(workspace_id, file_path)` |
| Ask-lane primitive `create_grant(kind=KIND_QUESTION)` + park `blocked` | ✅ | `handlers_asks.py:127-144` (PRD says 132-135 — cosmetic); resume = `_requeue_blocked_task` → `assigned` + `notify_task_available` (`api/approval_grants.py:528-551`) |
| `notify_board_event` needs a DB session | ✅ | `services/board_events.py:38` — it is `pg_notify`, not Redis |
| Canvas Redis channel has no always-on consumer | ✅ | `workspace:ws:{id}:canvas:events` (`worker main.py:58`); consumers exist elsewhere (`core/task_runner/queued.py`, `core/redis/client.py`) — reuse that helper for S2 |
| `Agent.configuration` JSON, shallow-merged on update | ✅ | `core/models/core.py:222`; `api/agents.py:876-880` |
| `validate_auth_edition()` outside `run_stage` | ✅ | `config.py:1639`; `main.py:563-571` |
| Claude Code flags in S1 | ✅ all present on 2.1.236 | plus flags the PRD does not use: `--session-id`, `--worktree`, `--setting-sources`, `--strict-mcp-config`, `--max-turns`, `--max-budget-usd`, `--json-schema`, `--fallback-model`, `--no-session-persistence`, `--fork-session`, `--include-hook-events` |
| `--permission-prompt-tool` (open question 1) | ⚠️ hidden | not in `--help` on 2.1.236, but SDK 0.3.258 still passes `--permission-prompt-tool stdio` — see §C1 |
| `rate_limit_event`, `apiKeySource` | ✅ | typed in SDK 0.3.258: `SDKRateLimitInfo{status: allowed|allowed_warning|rejected, rateLimitType: five_hour|seven_day|seven_day_opus|seven_day_sonnet|seven_day_overage_included|overage, resetsAt, utilization, isUsingOverage, overageStatus, errorCode: credits_required}`; `ApiKeySource: 'ANTHROPIC_API_KEY'|'apiKeyHelper'|'/login managed key'|'none'` where `none` = claude.ai OAuth login |
| munder anchors (`agentProvider.ts`, `pty.ts`, `breaker.ts:68-74`, `shellEnv.ts:31-71`, `cliInstall.ts`) | ✅ | `src/` identical between fc436bd and HEAD |
| Lease heartbeat (not in the PRD) | ❗ | `_lease_heartbeat` (`board_tasks.py:951`) runs *inside* `_run`; lease 600 s, sweeper requeues with ≤2 attempts (`board_dispatcher.py:184-230`, `config.py:610,616`) — see §B6 |
| `SESSION_RUNTIME_ENABLED` | absent | nowhere in code yet (expected) |

---

## B. Must change (decision-changing)

### B1. The transport is a billing decision — Q2 must be re-decided

**Facts (primary sources).**
- Anthropic, "Use the Claude Agent SDK with your Claude plan" (support.claude.com/en/articles/15036540, updated 2026-06-16): *"Claude Agent SDK and `claude -p` usage no longer counts toward your Claude plan's usage limits"* … then the banner: *"**Update June 15:** We're pausing the changes to Claude Agent SDK usage described below. For now, nothing has changed: Claude Agent SDK, `claude -p`, and third-party app usage still draw from your subscription's usage limits."* The paused model gives Max 20x a **$200/month** Agent SDK credit at **standard API rates**; when it runs out, *"additional Agent SDK usage flows to usage credits at standard API rates — but only if you've enabled usage credits."*
- Same article: *"Using Claude Code in the terminal or your IDE continues to use your subscription usage limits exactly as before."*
- munder-difflin chose PTY **for this reason**, not for lack of a better transport. Its blog (2026-06-06, `blog/src/posts/does-the-june-2026-agent-sdk-change-affect-munder-difflin.md`) and `src/main/hiddenClaude.ts:17-19`: *"Uses an interactive PTY (not `claude -p`) so calls draw from the user's normal interactive plan quota, not the Agent SDK credit."* Zed (a third-party harness) told its users the same thing: run the official `claude` CLI in a terminal to stay on subscription limits.
- The research record read munder's PTY as legacy ("scrapes PTYs, no quota detection") and rated headless `stream-json` "better than munder's". Mechanically true; it missed the *why*.

**What it means.** Today `claude -p` bills to the subscription (the 08-30 probe was correct: `apiKeySource: none`, seven-day window moving). Anthropic has declared the direction and only paused it, with "advance notice" promised. PRD-234 as written bets the Basic edition's headline feature — "1 or 100 sessions, no cost to me" — on the paused side of that line. If the pause lifts, every session moves to a $200 pool at API rates (the probe cost $0.38 for one word on the Opus 1M tier) and then stops.

**Recommendation.** Separate the *structured channel* from the *process driver*, and make the driver the swappable part:
- Structured channel (transport-neutral, day 1): the backend pre-assigns the session id (`--session-id <uuid>`), so the transcript path is known up front (`~/.claude/projects/<cwd-key>/<id>.jsonl`, key rule in munder `transcript.ts:12-14`); lifecycle and permission decisions arrive through **hooks** (`--settings` with a hook shim that POSTs to the host, munder `hooks.ts`); result text, usage and files come from the transcript JSONL. This works identically for an interactive session and for `-p`.
- Driver A — **interactive PTY** (munder parity: `claude --permission-mode … --append-system-prompt … --session-id …` under a pty; prompt typed via bracketed paste; done = `Stop` hook, quiescence as the fallback — munder `useHive.ts:637-676`). This is the path Anthropic's own words protect.
- Driver B — **headless `-p --output-format stream-json`** (what the PRD has). Cheaper to test, gives `rate_limit_event`, `get_usage` and the stdio permission protocol (§C1). Keep it as the CI vehicle and as an explicit per-agent option.
- Default for Claude = **Driver A**; the fleet must show which window moved (`five_hour`/`seven_day` vs `seven_day_oauth_apps` — the SDK's `get_usage` response lists a separate weekly window for OAuth apps) so the owner's smoke proves where sessions land. Cost of A: quota visibility is weaker (no `rate_limit_event`); use the `Notification` hook text plus a cheap on-demand probe (`claude -p --model haiku`) when the fleet needs a number.
- Wording, everywhere: "your Claude Code sessions, managed by Automatos" — not "runs on your subscription". The first is a description of what the software does; the second is a promise about Anthropic's billing.

**Alternative** (if the owner prefers): headless-first, PTY as the hedge story built in-wave (S1c). Same structured channel; the difference is which driver the smoke proves first. Not recommended given the owner's premise and munder's evidence.

### B2. The one policy line that shapes the design

Anthropic's Claude Code "Legal and compliance" page draws the line exactly where munder stands: *"Anthropic does not permit third-party developers to offer Claude.ai login into their own applications, or to route requests through Free, Pro, or Max plan credentials on behalf of their users… developers may not collect, store, or intermediate Claude.ai credentials or session tokens"* — and, in the same section: *"Nor does it prevent an end user from signing in to the unmodified Claude Code binary with their own Claude subscription, including where a platform hosts Claude Code."* Conditions: *"The Claude Code binary must not be modified"*; *"Advertised usage limits for Pro and Max plans assume ordinary, individual usage of Claude Code and the Agent SDK."* (code.claude.com/docs/en/legal-and-compliance)

So the design rules are: spawn the **unmodified** `claude` binary, let the user complete Anthropic's own login, never read or copy a token, one user per host. That is munder's shape and it is already the PRD's — the PRD should state it as an invariant with a source guard (no `CLAUDE_CODE_OAUTH_TOKEN`, no `~/.claude` reads, no `--bare`), and its wording should describe the software ("your Claude Code sessions, managed"), not Anthropic's billing. Codex has no written equivalent (§S5).

### B3. The no-silent-fallback invariant has a hole in 7 of 8 lanes

`AgentFactory.execute_with_prompt` is called from **eight** places, not one: `api/board_tasks.py:1168` (tickets), `services/heartbeat_service.py:1003` (heartbeats), `services/scheduled_task_service.py:426` (schedules), `modules/coordination/dispatcher.py` (missions), `services/coordinator_service.py:2251`, `channels/base.py:170` (Telegram/Slack), `api/webhooks.py:1211`, `api/composio.py:875`. The PRD branches in `_launch_task_execution` only. A `runtime: session` agent that heartbeats, is scheduled, is @-mentioned in a channel, or gets a mission task runs **on the API, silently** — PRD-223's disease class, re-created on day one.

**Fix.** Put the guard where the agent's configuration is already read — `agent_factory.py:816` (`db_agent.configuration`) / `execute_with_prompt` (`:1011`): if `configuration.runtime == "session"` and the caller is not the session lane, raise a typed `RuntimeMismatchError` carrying the actionable text ("agent X runs as a Claude Code session; this lane cannot run it — assign a ticket"). The board lane routes *before* it reaches the factory; every other lane fails honestly, with the usual notification. Test: parametrize over all eight call sites with a session agent → zero LLM-client invocations, one typed error each. This replaces the PRD's "dispatch-branch units" as the core invariant test.

### B4. There is no security model, and the queue is a remote shell into the laptop

Facts: compose publishes the API (8000), Redis, Postgres and MinIO on **all interfaces** (`docker-compose.yml:37,67,97,207`); in the local edition an anonymous request *is* the instance's `super_admin` (`core/auth/hybrid.py:944-949`); the PRD has the host "authenticating with the local edition's anonymous operator (no keys)"; `-p` skips the workspace-trust dialog and *silently ignores* invalid settings files (`claude --help`); a session started in a registered repo loads that repo's `.claude/settings.json` hooks and `.mcp.json` servers; ticket text is untrusted input (Auto writes contracts from channels and webhooks) and it becomes the prompt of a Bash-capable session in a real repository. Net: anyone who can reach port 8000 (the LAN, by default) can create a ticket, assign it to a session agent, and run commands on the user's machine.

**Fix (all in S1, plus a threat-model paragraph in the PRD):**
1. **Pairing token** minted by the backend at `make session-host` time, shown once, required on every host call, stored host-side; never an empty default (do not copy `WORKER_INTERNAL_TOKEN=""`, `config.py:649`).
2. **Bind guidance and default**: with session mode on, the API binds `127.0.0.1:8000` (compose `${API_BIND:-127.0.0.1}:` in the local edition, or at minimum a loud doc rule); the host connects to loopback only. I found no bind-address/LAN guidance in `QUICKSTART.md` or `docs/getting-started/self-hosting.md`.
3. **Directory allowlist enforced on both sides**: registered directories live in the backend *and* in a host-local allowlist file; the host refuses any `cwd` outside its own list. A compromised backend must not be able to point the host at `~`.
4. **Config surface decided, not inherited**: `--setting-sources user` (operator's own policies apply; a cloned repo's hooks do not) and `--strict-mcp-config` with a curated set (empty by default). Also fixes the cold-start/tool-bloat problem — this machine alone would load several MCP servers into every session.
5. **Git policy**: sessions never push (munder's rule; `hive.ts` "do NOT push; the manager integrates") — enforce through the Bash allowlist, not the prompt.
6. **HTTP-only host transport**: one port, one token; drop direct Redis from the host (Redis pub/sub stays inside compose for the orchestrator's own subscriber).

### B5. The boot gate belongs in S1, not S4

The wave doc says `SESSION_RUNTIME_ENABLED` is "boot-guarded to `AUTH_EDITION=local` (234 S1)"; the PRD moved it to S4. With S1–S3 shipped ungated, a SaaS workspace could set `runtime: session` and enqueue work no host will ever serve — a silent queue. Move the guard to S1 and add the API-side rule: the agent-settings endpoint rejects `runtime: session` when the flag is off (the UI field in S4 is the last line, not the first).

### B6. Leases, retries and the orphan session

Today the running coroutine heartbeats its own lease (`_lease_heartbeat`, 600 s window, sweeper requeues expired leases with ≤2 attempts). In session mode the work leaves the process. Unaddressed, this produces: host dies or loses the network → lease lapses → task back to `assigned` → re-claimed → a **second** session starts in the same directory while the first `claude` process is still alive. Requirements for S1: the host keeps a persisted process table and reconciles it with the backend on (re)connect before claiming anything; host progress events renew the lease (`renew_lease`, `board_dispatcher.py:272`); the backend pre-assigns `session_id` and the result endpoint is idempotent per `(task_id, attempt)`; a task whose session is reported alive is never re-dispatched; on restart the host kills its own orphans first (munder `procKill.ts` / `ensureKilled`).

### B7. No cap + no worktrees + one directory = colliding edits

Q5 ("1 or 100 sessions") and Q3 ("the directory as is") together mean two sessions editing one repo concurrently. Claude Code has the primitive natively: `-w/--worktree [name]` creates a git worktree for the session (`claude --help`). Recommendation: worktree per session for git directories (munder isolates per agent, `index.ts:2640-2667`), one active session per directory for non-git directories, and Auto integrates — sessions never push (B4.5).

### B8. There is no cancel, stop or timeout path

`api/board_tasks.py` has no cancel/stop endpoint for an `in_progress` task (the only `cancel` is the heartbeat future). Sessions need one: stop from the board/fleet → host `SIGTERM` (then kill) → result `cancelled`; a per-session wall-clock timeout; `--max-turns` (native); the loop guards already in S3; orphan cleanup (B6).

---

## C. What we could do better

1. **Permission routing without an MCP endpoint (open question 1 is easier than the PRD thinks).** The SDK passes `--permission-prompt-tool stdio`; the CLI then emits `control_request{subtype: can_use_tool, tool_name, input, blocked_path, decision_reason_type, permission_suggestions}` on stdout and reads the decision on stdin. In PTY mode the equivalent is a `PreToolUse` hook returning a decision (munder `hooks.ts:231-240` denies through it). Either way an out-of-allowlist Bash call can land in the **approvals inbox** (`create_grant` kind `tool_call` already exists, PRD-193) instead of a blanket deny — v1-feasible. And `permission_denials` in the result must be first-class: any denial → the task lands in `review`, never auto-`done` ("couldn't run the tests" must not read as finished).
2. **Quota, done with what the CLI already emits.** Windows: `five_hour`, `seven_day`, `seven_day_opus`, `seven_day_sonnet`, `overage`. `isUsingOverage` / `overageStatus` / `errorCode: credits_required` must surface loudly; default policy **"never spend overage"** (pause + notify) unless the user opts in — this is what protects "no cost to me". Mid-run `terminal_reason: blocking_limit` → blocked with a **timer** to `resetsAt`, resumed with `--resume <session_id>` so context is kept. Host offline / not logged in / limit reached are all machine-detectable conditions: auto-resume on the event (host connects, login re-probed, window resets) with a notification, and reserve `KIND_QUESTION` for decisions only a human can make. The PRD's blocked-with-question for "host not running" creates a loop (answer → assigned → still no host → blocked again). S3's text currently says both "blocked with a question" and "resumes at `resetsAt`" for the same condition — pick the timer.
3. **Cost semantics.** `total_cost_usd` is the API-equivalent estimate, not what a subscriber pays. Label it "API-equivalent" or hide it for subscription sessions; express caps in turns/tokens (`--max-turns`, `modelUsage`), not USD. munder's breaker cost trips were built for API billing.
4. **Billing-honesty check at session start.** Assert `system/init.apiKeySource == "none"` for a subscription agent; anything else (`ANTHROPIC_API_KEY`, `apiKeyHelper`, `/login managed key`) → stop and report "this session would bill your API key". The host must *strip* `ANTHROPIC_API_KEY` and every `CLAUDE_*` session marker from the inherited environment except the operator's config keys (munder `ptyEnv.ts:29-44`: an inherited `CLAUDE_CODE_CHILD_SESSION` silently disables transcript saving and breaks `--resume` — and this host will often be started from inside a Claude Code terminal). Never `--bare` (it never reads OAuth); never override `CLAUDE_CONFIG_DIR`.
5. **Drop the SDK-bundled-binary fallback.** It contradicts the non-goal "no auto-installing CLIs" (the current macOS wheel is 85 MB, `claude-agent-sdk` 0.2.151), a subprocess design needs no SDK, and the honest hint is "Claude Code is not installed". The host should have no SDK dependency at all.
6. **Model validation per runtime** (PRD-223: the model route validates nothing). Session agents accept only what `claude --model` accepts (aliases or full Claude ids); the settings API rejects OpenRouter ids for `runtime: session`, and vice-versa.
7. **Deliverables are only real under the bind mount.** Files under `AUTOMATOS_WORKSPACE_DIR` (the worker's `/workspaces`) can be served and previewed; files in a registered host directory outside it are invisible to backend and worker, so `deliverable_service.register` would produce dead download links. Decide: reference-only Deliverable (path, diff summary, session id) for host directories, plus host upload for small text files if wanted. The PRD currently implies registration works everywhere.
8. **Re-cut S1 and make CI prove more than fixtures.** S1a = backend only (runtime field + factory guard + dispatch branch + claim/heartbeat/result endpoints + completion-writer extraction incl. the failure path + boot gate) — fully CI-provable against today's board tests. S1b = host + Claude adapter + recorded fixtures. S1c = the second driver. Add a **fake-`claude` lane**: a script on `PATH` that replays fixtures, writes a transcript JSONL and fires the hooks — host + backend end-to-end in CI, error paths included. Fixtures record `claude_code_version`; the host checks `claude --version` against the tested range and marks the adapter `LIVE-UNVERIFIED` outside it; a `--help` drift guard catches flag removals (2.1.236 already hides one the PRD relies on).
9. **Session takeover is free.** With a known session id the fleet card can show `claude --resume <id>` — the user takes over any session in their own terminal, which is exactly the interactive path Anthropic protects. Also the honest escape hatch when a session parks on something the host cannot answer.
10. **Prompt composition.** `--append-system-prompt` = soul + skills + doctrine is right, but Auto's skill bodies are large (PRD-231) and the repo's own `CLAUDE.md` loads via `cwd` anyway; keep the appended prompt small and stable, put the ticket contract in the turn, and don't promise cache savings until the smoke shows them.
11. **Naming.** "Session" already means a Code Canvas session in code and UI. Call this runtime `cli` / "terminal agents" (`runtime: api | cli`) before it lands in the fleet, the settings modal and the docs.
12. **Platform scope.** v1 = macOS + Linux (the login-shell PATH probe is POSIX; munder's Windows branch exists to port later). Say so in S4's docs.

---

## D. Missed (gaps not covered above)

- **Telemetry parity.** The fleet's `cost_24h` reads `LLMUsage` rows (`fleet_state.py:152-180`); session runs write none. Either write `LLMUsage` from `modelUsage` (labelled API-equivalent) or the fleet shows session agents at zero forever.
- **Attachments and redo.** The board lane passes `attachment_ids` (PRD-127) and folds `review_feedback` into a redo prompt (`board_dispatcher.py:440-448`). Sessions need both: attachments materialised into the `cwd` (or referenced), and a redo that `--resume`s the same session.
- **Two hosts** (open question 3). Even if v1 supports one, the claim protocol must make a second host either cooperate (disjoint claims) or refuse loudly — not double-run.
- **Backend offline mid-run.** The host must buffer events and the final result and retry the idempotent POST; today's PRD has the result as a single fire-and-forget.
- **Keychain and non-interactive parents.** Claude Code's macOS login is keychain-backed; a host started by `launchd`/`nohup` may not see it and will report "not logged in" falsely. The preflight must be tested from a non-interactive parent, and the host's own login check should be the CLI's answer, not a file probe.
- **`AUTOMATOS_WORKSPACE_DIR` is relative** (`./workspaces`, compose project dir). The host must receive the absolute path from the backend's status endpoint, never re-derive it.
- **The operator's own hooks are part of the policy surface.** With `--setting-sources user`, this machine's PreToolUse guard on `.env` files applies to every session — document that sessions inherit the operator's Claude Code policy, and nothing else.
- **`--bg` / `claude agents`.** 2.1.236 can start a session "as a background agent" managed by `claude agents`. Unknown billing class and semantics; investigate, do not bet on it.
- **Local checkout drift.** The local `automatos-ai` main is at 0522cb806, behind `origin/main` (5e3ed68d9); the PRD/research/wave files are untracked locally but identical to origin. Pull before any 234 branch is cut.

---

## E. Document consistency fixes

| Where | Issue | Fix |
|---|---|---|
| `PRD-WAVE-OPEN-CORE.md` open-Q list | lists 234 Q1–Q6 with different content than the PRD's Q1–Q8 | replace with the PRD's table; carry the new decisions (§F) |
| PRD-234 S1 / wave doc | wave: `SESSION_RUNTIME_ENABLED` in S1; PRD: S4 | S1 (§B5) |
| PRD-234 Q-table | per-task vs long-lived sessions undecided (S1 mentions `--resume`, never says which) | decide: per-task session, resumed for redo / quota / continuation |
| PRD-233 Q4 | "MCP-in-router deferred to PRD-234 (confirm)" — PRD-234 never mentions MCP | add the decision from §B4.4 (sessions' MCP = curated, strict) and close 233 Q4 |
| PRD-234 header | "Depends on PRD-209 + PRD-233 merged and tested (PR #650)" | merged 2026-08-30 (#650/#664/#665, fixes #668/#669); owner test pending |
| PRD-234 non-goals vs S1 | "no auto-installing CLIs" vs "SDK bundled binary as fallback" | drop the fallback (§C5) |
| PRD-234 S3 | "limit reached" both blocks-with-question and "pauses, resumes at resetsAt" | timer-based (§C2) |
| PRD-234 success metrics | "zero API calls" is not measurable as written | the §B3 parametrized test + a counter in telemetry |
| Research §7 | "wheel 198 MB" | 85–98 MB at 0.2.151; moot after §C5 |

---

## F. Decisions needed (owner)

| # | Decision | Recommendation |
|---|---|---|
| 1 | **Transport (re-decide Q2)** — interactive PTY default with headless as option; or headless-first with the PTY hedge in-wave; or headless-only | PTY default + headless option (§B1) |
| 2 | **Positioning / approval** — proceed as "your Claude Code sessions, managed"; optionally ask Anthropic for the "previously approved" route | proceed with the wording rule; asking costs nothing (§B2) |
| 3 | **Network posture** — loopback bind default in the local edition + pairing token | yes to both (§B4) |
| 4 | **Isolation** — `--worktree` per session for git directories; one active session per non-git directory | yes (§B7) |
| 5 | **Overage** — default "never spend overage" (pause + notify) | yes (§C2) |
| 6 | **Deliverables outside the bind mount** — reference-only, or host upload for small files | reference-only in v1 (§C7) |
| 7 | **Config inheritance** — `--setting-sources user` + `--strict-mcp-config` (empty set) | yes (§B4.4) |
| 8 | **Story re-cut** — S1a backend / S1b host + Claude / S1c second driver; fake-`claude` CI lane | yes (§C8) |
| 9 | **Runtime name** — `cli` instead of `session` | yes (§C11) |
| 10 | **Codex in-wave (Q7)** — keep, with its own billing/policy line checked | see S5 note below |

**S5 / Codex (facts for the adapter row, Codex CLI 0.152.1).** `codex exec` is the non-interactive mode; prompt positional or stdin (`codex exec -`); `--json` emits JSONL (`thread.started`, `turn.started/completed/failed`, `item.*` with `agent_message | reasoning | command_execution | file_change | mcp_tool_call | …`, `turn.completed.usage`); continuation = `codex exec resume <SESSION_ID>` (also `--last`, `exec fork`); policy flags `-s/--sandbox read-only|workspace-write|danger-full-access`, `-a/--ask-for-approval untrusted|on-request|never`; `-o/--output-last-message`, `--output-schema`, `-C/--cd`, `-m/--model`. Login lives in `$CODEX_HOME/auth.json` (or keyring); the CLI reads it itself — the PRD's "never copied or symlinked" stands. **No rate-limit data in `exec --json`** (upstream declined it); it is available over `codex app-server` JSON-RPC (`account/rateLimits/read`), which is the honest quota path for S3 on Codex. OpenAI has published no written rule on third-party orchestration of ChatGPT-plan Codex — tolerated in public statements, not committed; one line in S5's docs. Sources: learn.chatgpt.com/docs/non-interactive-mode, …/developer-commands, …/auth; github.com/openai/codex issues #14728, #22998.

**S6 adjustments (one line each).** *gemini*: consumer Google AI Pro/Ultra login left Gemini CLI on 2026-06-18 and moved to Antigravity (`agy`) — the gemini row is API-key only, so it drops out of the "subscription" lane; *qwen*: OAuth free tier discontinued 2026-04-15, API-key only; *crush*: no machine-readable output flag — PTY adapter, as the PRD assumed; *cursor*: the binary is `agent` (`cursor-agent` is a symlink), `-p --output-format stream-json`, `--resume`; *copilot*: `-p --output-format json` (JSONL); *kimi*: `--print --output-format stream-json`, exit 75 = retryable 429; *grok*: `-p --output-format streaming-json`; *pi*: `--mode json|rpc`; *opencode*: `run --format json`, Anthropic subscription login removed upstream (Anthropic prohibits it). Every row stays `LIVE-UNVERIFIED` until someone runs it.

---

## G. Sources

- Anthropic Help Center, *Use the Claude Agent SDK with your Claude plan* — https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan (updated 2026-06-16)
- Anthropic, *Claude Code — Legal and compliance* (the policy paragraph in B2) — https://code.claude.com/docs/en/legal-and-compliance
- Anthropic, *Agent SDK quickstart* (bundled-binary note; "unless previously approved" sentence) — https://code.claude.com/docs/en/agent-sdk/quickstart.md
- OpenAI, *Codex non-interactive mode* / *developer commands* / *auth* — https://learn.chatgpt.com/docs/non-interactive-mode, https://learn.chatgpt.com/docs/developer-commands?surface=cli, https://learn.chatgpt.com/docs/auth
- Anthropic, *Headless mode* / *CLI reference* — https://code.claude.com/docs/en/headless.md, https://code.claude.com/docs/en/cli-reference.md
- The New Stack, *Anthropic splits billing again: Agent SDK gets separate credit pools* — https://thenewstack.io/anthropic-agent-sdk-credits/
- Zed, *What Anthropic's New Claude Billing Means for Zed Users* — https://zed.dev/blog/anthropic-subscription-changes
- munder-difflin @ cc741fe3 (`src/` == fc436bd): `src/main/hiddenClaude.ts`, `ptyEnv.ts`, `hooks.ts`, `transcript.ts`, `breaker.ts`, `shellEnv.ts`, `index.ts`, `src/shared/agentProvider.ts`, `blog/src/posts/does-the-june-2026-agent-sdk-change-affect-munder-difflin.md`
- `@anthropic-ai/claude-agent-sdk` 0.3.258 `sdk.d.ts` (event and control-protocol types); `claude-agent-sdk` 0.2.151 on PyPI (wheel sizes)
- Claude Code 2.1.236 `claude --help` on the owner's machine

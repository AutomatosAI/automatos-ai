# Session Mode — research record (2026-08-30)

**Purpose.** Ground Phase 2 ("agents run as the user's own terminal CLIs on their subscription login — Claude Code first, the rest later; local edition only") in code before any story is written. Two sources, both read as code with file:line anchors: the reference implementation `chaitanyagiri/munder-difflin` (v0.4.6, MIT, commit fc436bd, 64,746 lines of TS) and Automatos' own seams on `ralph/prd-233-local-edition`. Plus three facts verified on the owner's machine.

**Status.** Research + proposed design for owner approval. No code. PRD-234 as drafted has ten statements the code contradicts (§9) and one architectural assumption the research overturns (§4); it should be rewritten from this record once §8 is decided.

---

## 1. The answer in one paragraph

Munder-difflin is an **Electron desktop app that owns terminal processes on the user's machine**: it spawns the locally-installed CLIs (`claude`, `codex`, `gemini`, `opencode`, … 12 of them) as PTYs via node-pty, authenticates through the user's own CLI logins (it literally symlinks `~/.codex/auth.json`), reads lifecycle through hook shims over a Unix socket, and coordinates a file-based mailbox routed by the main process. It bundles **no SDK**. Nothing about it runs in a container, and nothing about it could: the credentials, the binaries and the PATH are all host-local. Automatos' existing session runtime (Code Canvas) is the opposite shape — a Linux container driving the Claude Agent SDK with an injected API key or OAuth token and per-workspace isolated config. **A container cannot execute the user's Mac CLI, and a subscription login lives on the host.** Therefore Phase 2's runtime must be a **host process** — a small, local-only module the user starts on their machine — talking to the Dockerised backend over the ports compose already publishes. That is the one decision everything else follows from; the rest of this record is what to reuse, what not to copy, and what to build.

---

## 2. What munder-difflin actually is (facts, not README)

| Mechanism | What the code does | Anchor |
|---|---|---|
| Process model | Electron main process owns every CLI as a PTY (`node-pty`), one PTY per agent, no worker threads | `src/main/pty.ts:305,644-653` |
| Binary/PATH | Resolves CLIs via the user's *interactive login shell* PATH + hardcoded dirs (`/opt/homebrew/bin`, `~/.claude/local`, …); Windows `.cmd` shims decoded | `shellEnv.ts:31-71`, `pty.ts:397-473` |
| Auth | **Logged-in CLI, no keys**: preserves `CLAUDE_CONFIG_DIR`/`CLAUDE_CODE_OAUTH_TOKEN`, symlinks `~/.codex/auth.json` into per-agent homes; BYOK injected only for opencode/crush/pi/qwen | `ptyEnv.ts:39-44`, `hive.ts:1975-1984`, `index.ts:2877-2922` |
| Providers | `AgentProviderPreset` — **pure data** per CLI: binary, how the seed prompt gets in (flag / positional / typed into the TUI), auto-mode flag, resume flag, install command, lifecycle bridge (native hooks → config-file hook shim → loopback proxy that synthesises hooks → idle-nudge fallback) | `src/shared/agentProvider.ts:62-168,170-578` |
| Output | **Never `--print`/JSON for agents**: raw PTY bytes → xterm + regex scraper; structured lifecycle via hook shims over `<hive>/hooks.sock`; files on disk | `usePtyParser.ts`, `hooks.ts:81-103` |
| "Done" | `Stop` hook → idle; else **12 s of PTY silence** = idle; ephemeral workers signal `act:"done"` by mail | `useHive.ts:637-676`, `index.ts:4535-4558` |
| Manager | "Michael" is **not code** — a `claude` process with a privileged prompt (orchestrate, decompose, sole scribe of `board.md`, 4-part OBJECTIVE/OUTPUT/TOOLS/BOUNDARIES contract) | `useHive.ts:376-475`, `hive.ts:1425-1427` |
| Messaging | FIPA-lite JSON files: `agents/<id>/outbox/*.json` → main-process router (1.5 s poll) → `inbox/`; `from` forced to the owning dir; hop cap 12; undeliverable mail bounces to the manager, never vanishes | `hive.ts:56-69,1495-1585,1638-1675` |
| Delivering mail into a live TUI | serialised write chain, text-then-`\r` 140 ms later, bracketed paste, user-draft latches, HITL re-arm 5 min, bounded retries that drop **loudly** | `useHive.ts:88-157,678-731`, `workerWake.ts:114-131` |
| Memory | per-agent `memory.md` (bounded by a haiku summariser, backup-first/verify/atomic swap) + shared MemPalace semantic store (silent no-op if absent); **never pasted into context** — the agent is told the path | `reflect.ts`, `memory.ts`, `hive.ts:1435-1453` |
| Prompt-cache invariant | system prompt contains nothing volatile; live roster/goals ride `additionalContext` on `SessionStart`/`UserPromptSubmit` | `hive.ts:1348-1368`, `hooks.ts:255-293` |
| Quota / hourly limits | **Nothing.** No 429 or "resets at" detection; only a self-imposed circuit breaker (repeated-tool, error-storm, token velocity, cost caps) | `breaker.ts:68-74`; grep confirms |
| Not logged in | **Not detected** — the CLI shows its login prompt in the PTY and the agent drifts to idle | §7 of the map |
| Missing CLI | Detected pre-spawn; a visible in-terminal install ladder (npm → node+npm → vendor installer → manual hint) then auto-respawn | `cliInstall.ts:36-194`, `index.ts:582-603` |
| Persistence | `config.json`, a 2-table SQLite, `roster.json` (append-only backups), and a **git-committed `hive/`** (registry, board, tasks, log, per-agent dirs) | `db.ts:48-68`, `roster.ts:114-169`, `hive.ts:2564-2579` |
| Git | per-agent worktrees (`agent/<id>` branch), safety-gated teardown, "do NOT push; the manager integrates" | `index.ts:2640-2667,517,4723-4758` |
| Tests | 98 files; pure functions extracted precisely so PTY/CLIs need no faking; **unit tests are not in CI** (CI = typecheck + link check) | `test/load-ts.cjs`, `.github/workflows/ci.yml` |

**Its own spec is obsolete**: SPEC.md still describes a tmux-attach product with no inter-agent messaging and SQLite tables that were never built; HIVE.md's "Stop hook forces continuation" was deliberately removed (`hooks.ts:220-230`: it "could spend credits while a user was answering a question"). The code is the reference, not the docs — same rule we apply to ourselves.

**Provider count.** Twelve CLIs + `custom`: claude, codex, grok, kimi, gemini, antigravity (`agy`), qwen, opencode, crush, pi, copilot, cursor. "BYOK" and "local LLM" are not providers — they are key-injection and base-URL concerns on top of these.

---

## 3. What Automatos already has (anchored) — and the gaps

| Seam | State | Anchor |
|---|---|---|
| Session runtime | `CanvasSessionManager` drives the Claude Agent SDK (`ClaudeSDKClient`), one session per workspace, per-workspace `CLAUDE_CONFIG_DIR` (isolated auth), credential = env `ANTHROPIC_API_KEY` **and/or** `CLAUDE_CODE_OAUTH_TOKEN` (both merged; a committed test asserts both present) | `services/workspace-worker/canvas_session_service.py:141-154,370-409`; `worker_config.py:30-48`; `tests/test_prd203_cs8_worker_auth.py:79-94` |
| Confinement + approvals | path re-binding + escape detection; file edits/Bash gated; verdicts are SDK `PermissionResultAllow/Deny` objects and Claude tool names | `canvas_confinement.py:59,85,138`; `canvas_approvals.py:42-67`; `canvas_session_service.py:167-252` |
| Events | closed vocabulary v1 (`canvas.session.status`, `…tool.call`, `…file.edit`, `…permission.request`, `…turn.complete`) → Redis `workspace:ws:{id}:canvas:events` → **only consumer is a per-request SSE generator** | `canvas_events.py:34-55`; `api/workspace_files.py:364-400` |
| Result channel | **none** — `send_message` returns `{"success": True}`; no final text, exit status or artifact list | `canvas_session_service.py:319` |
| Worker reachability | HTTP 8081 + `X-Internal-Token`; Redis queue lane has no session action; worker image has **no orchestrator code** and cannot see `AUTH_EDITION` | `main.py:1077-1082,507-517`; `executor.py:539-568`; `Dockerfile:34` |
| Agent definition | `Agent.configuration` JSON survives `AgentUpdate`'s shallow merge → `runtime`/`session_provider` live there, no migration (avoid the unrelated `"runtime"` status key in `agent_endpoints.py:182`) | `core/models/core.py:222`; `api/agents.py:874-880` |
| Ticket lane trigger | **four** launch sites, all funnelling into `_launch_task_execution` — the single place a runtime branch belongs | `api/board_tasks.py:1138-1176,616,937`; `handlers_board_tasks.py:498`; `board_dispatcher.py:466-482` |
| Completion writer | an **inline block**, not a function: `task.result`, status `done|review`, `_dispatch_task_complete` (notifications + watches), `_auto_create_task_report` | `api/board_tasks.py:1184-1222` |
| Deliverables | the completion path creates **no** Deliverable (reports surface via `v_workspace_outputs`); registration happens only on the `workspace_write_file` tool path | `report_service.py:287-289`; `exec_workspace.py:43-118` |
| Dispatch contract | `DISPATCH_CONTRACT_FRAGMENT` (OBJECTIVE/OUTPUT/TOOLS/BOUNDARIES) is written **into the task description by the LLM at creation**; no composition with soul/skills/context at dispatch | `modules/coordination/dispatch_contract.py:15-21`; `consumers/chatbot/auto.py:106-161` |
| Ask lane | `ask_human` is agent-facing (server-injected `_agent_id`); the reusable primitive is `create_grant(kind=KIND_QUESTION)` + park `status=blocked`; answer → `_requeue_blocked_task` → `assigned` → dispatcher relaunches | `handlers_asks.py:31-165`; `core/services/approval_grants.py:43-63`; `api/approval_grants.py:294-331,528-556` |
| Fleet | `get_fleet_state` read-model → `/api/v1/fleet` → `fleet-tab.tsx`; two unconnected approval UIs (canvas DiffCard vs approvals inbox) | `services/fleet_state.py:485`; `components/agents/fleet-tab.tsx` |
| Boot guard | `validate_auth_edition()` runs **outside** `run_stage` (which swallows) — the only place a real abort lives | `config.py:1639-1680`; `main.py:563-571`; `bootstrap.py:127-134` |
| Local-only modularity | one flag `AUTH_EDITION` (backend + `lib/auth-edition.ts` mirror); compose profiles; `envs/api.defaults`; the S2 capability-degrade pattern (predicate → one exclusion point → explicit error → status endpoint → honest card) | `config.py:201-208`; `core/composio/client.py`; `tool_router.py` |
| Prior art for the vocabulary | zero hits for codex/opencode/session_runtime/node-pty/pexpect in code | grep |

---

## 4. The architecture decision: a host process, not the container

Evidence, all verified:

1. **Credentials are host-local.** Claude Code's login is in `~/.claude` (macOS Keychain-backed); Codex's in `~/.codex/auth.json`. Munder depends on this; our worker cannot see it.
2. **Binaries are host-local and platform-bound.** The owner's `claude` is a Homebrew macOS arm64 binary (2.1.236). A Linux container cannot execute it; munder's PATH probe exists precisely to find such binaries.
3. **The container's path needs a key.** Canvas injects `ANTHROPIC_API_KEY`/`CLAUDE_CODE_OAUTH_TOKEN` and isolates `CLAUDE_CONFIG_DIR` per workspace — the opposite of "the user's login".
4. **The backend is reachable from the host.** Compose publishes 8000 (API), 6379 (Redis), 5432 (Postgres), 9000 (MinIO).
5. **Headless Claude Code already gives a structured result channel — better than munder's.** Probe on this machine: `env -u ANTHROPIC_API_KEY claude -p … --output-format stream-json` emitted `system/init` (model, 45 tools, `session_id`, `apiKeySource: none` ⇒ **subscription billing**), `assistant`, **`rate_limit_event`** (`{rateLimitType: seven_day, utilization: 0.91, status: allowed_warning, resetsAt}`) and `result/success` (`num_turns`, `total_cost_usd`, `usage`, `permission_denials`, the text). Munder scrapes a PTY and has **no** quota detection; Claude Code hands us both as JSON.
6. **The SDK is also usable on the host.** `claude-agent-sdk` 0.2.148 ships a `macosx_11_0_arm64` wheel that **bundles a Mac `claude` binary** (198 MB) and resolves it before `PATH`, sharing the user's `~/.claude` login unless `CLAUDE_CONFIG_DIR` is overridden.

Consequence: **the session runtime is a new local-only module, `services/session-host/`, run by the user on their machine** — not a generalisation of the canvas service inside the worker. The canvas stays byte-identical (its SDK-welded gate, one-session-per-workspace model and per-workspace auth isolation are all wrong for session mode and right for canvas). PRD-234 S1 as drafted ("extract the session-drive core of `canvas_session_service.py`") is therefore withdrawn; the reusable pieces are the pure ones — `canvas_confinement` path logic, the event-envelope discipline, approval payload shapes.

---

## 5. Proposed design (for approval)

**5.1 The Session Host (new, local-only).** A small Python process (`python -m automatos_session_host`, later `pipx`/`make session-host`), started by the user next to `make up`. It:
- connects **outward** to the backend (`http://localhost:8000` + Redis `6379`), authenticating with the local edition's anonymous operator (no keys); refuses to start unless the backend reports `edition: local`;
- announces **capabilities**: which CLIs are installed (munder's login-shell PATH probe, ported), their versions, and a cheap login check per provider (for Claude: `claude -p` dry probe or the `apiKeySource`/`rate_limit` fields of the last run);
- claims **session tasks** from a queue the backend writes, runs them through a provider adapter, streams events back, posts the **result** (final text, exit status, `session_id`, cost/usage, files touched) — the terminal value canvas never had;
- keeps sessions in the user's **real directories** (a registered host directory or the workspace's `./workspaces/<id>` dir), confined by the ported `canvas_confinement` rules;
- reads Claude's `rate_limit_event` and reports quota state to the backend (**pause dispatch at `status != allowed`**, resume at `resetsAt`).

**5.2 Provider adapters as data + a thin driver** (munder's best idea, kept; its PTY transport, not kept). One table entry per CLI: binary, headless argv builder, prompt delivery, output parser (`stream-json` events → our event vocabulary), resume, install hint, auth check. Claude first:
`claude -p <prompt> --output-format stream-json --input-format stream-json --append-system-prompt <stable manager prompt> --permission-mode <policy> --allowedTools <allowlist> --add-dir <workspace> --model <agent's model>`, driven over stdin for multi-turn, `--resume <session_id>` for continuation. Adapter #2 (Codex `codex exec --json`) proves the interface; the remaining ten are **community-contribution-sized** (each is a table row + a parser) and belong in the "where to contribute capability" doctrine already in CONTRIBUTING.

**5.3 Backend changes (the manager plane already exists; these connect it).**
- `Agent.configuration.runtime = "api" | "session"` and `session_provider` (no migration; agent settings UI gets one field group).
- One branch inside `_launch_task_execution` (covers all four launch sites): `session` ⇒ enqueue `{task_id, workspace_id, agent_id, prompt, cwd, provider, model, caps}`; the prompt = the task's dispatch contract (already in the description) + the agent's soul/skills as the stable `--append-system-prompt`.
- **Extract the completion writer** from the inline block so a session result lands exactly like an API result (status, report, notifications, watches).
- An **always-on** Redis subscriber for session events → `notify_board_event` (PRD-227) + fleet state (PRD-228 `runtime: session`, live tool, tokens, quota) — the canvas channel has no such consumer today.
- **Artifacts**: file-edit events from the session register Deliverables through `deliverable_service.register` (the gap the map found — no session or canvas edit registers anything today).
- **Preflight honesty (S3 as drafted, corrected)**: no host connected / CLI missing / not logged in / quota exhausted ⇒ the task is parked `blocked` with a question via `create_grant(kind=KIND_QUESTION)` (not the agent tool), answered through the existing inbox; **never** a silent fall-through to the API path (PRD-223's disease class).
- `SESSION_RUNTIME_ENABLED` local-only boot guard in `validate_auth_edition()` (outside `run_stage`), plus the host's own edition check — the worker container is untouched and never sees session work.
- Fleet/UI: a "Session Host" connection card (connected / not running — with the start command), runtime tag on agents, quota bar, live event feed via the existing SSE pattern.

**5.4 Permission policy (v1).** Munder mostly *avoids* prompts (`bypassPermissions`/`--yolo`) and only detects them to back off. We can do better with headless flags: edits inside the workspace auto-accepted (`acceptEdits`), Bash confined to an allowlist, anything outside ⇒ denied and reported. Routing live permission asks into the approvals inbox (`--permission-prompt-tool`) is v2 — it requires an MCP endpoint on the host.

**5.5 What we deliberately do NOT copy.** The file mailbox/hive (we have a DB-backed board, tickets, dispatch contract, watches); PTY typing automation (headless JSON instead); the Electron UI (Next.js + SSE); MemPalace (we have memory); the git-committed hive; auto-installing CLIs on the user's machine (honest "install X" hints instead, like the Composio card). We **do** adopt as principles: the prompt-cache invariant (stable system prompt, volatile context in the turn), the circuit breaker (per-session cost/token caps, repeated-tool and error-storm guards), "deliberate but never silent degradation", and `LIVE-UNVERIFIED` annotations on adapters nobody has run.

---

## 6. Build order (stories; sizes are estimates)

| # | Story | Reuse / extend / new | Size |
|---|---|---|---|
| S1 | **Vertical slice**: session host skeleton + Claude adapter + backend session queue/result endpoint + completion-writer extraction + runtime branch — one ticket runs as a Claude Code session on the subscription and lands on the board | new host (~600 lines incl. ported confinement/PATH probe); extend `_launch_task_execution`; extract writer | M/L |
| S2 | Events → board/fleet: always-on subscriber, fleet runtime tag, session-host connection card, live feed; Deliverable registration from file-edit events | extend fleet_state, board_events, deliverable_service; new subscriber | M |
| S3 | Honesty: capability preflight, blocked-with-question via `create_grant`, quota pause/resume from `rate_limit_event`, circuit breaker | extend ask lane; new breaker (port of munder's thresholds) | S/M |
| S4 | Agent settings: runtime + provider field group; `SESSION_RUNTIME_ENABLED` guard; docs (self-hosting guide: "start the session host") | extend agent-configuration-modal, config, docs | S |
| S5 | Second adapter (Codex `exec --json`) proving the interface; adapter contribution guide | new (data + parser) | S/M |
| S6+ | Remaining ten adapters — community lane, each `LIVE-UNVERIFIED` until someone runs it | new, small each | S each |

Owner smoke (not CI, by design): create ticket → assign a session agent → watch the board; run on a day the seven-day quota isn't at 91 %.

---

## 7. Facts verified on this machine (2026-08-30)

- `claude` 2.1.236 (Homebrew), flags `-p/--print`, `--output-format stream-json`, `--input-format`, `--permission-mode`, `--allowedTools`, `--append-system-prompt`, `--model`, `--resume` all present. Of munder's twelve CLIs only `claude` and `cursor` are installed here.
- Headless probe: 7 events, `apiKeySource: none`, cost $0.38 for one word (the Opus 1M context tier), `rate_limit_info.utilization 0.91` on the seven-day window, resets 2026-09-03.
- `claude-agent-sdk` macOS arm64 wheel bundles `_bundled/claude` (198 MB) — a host runtime needs no npm CLI.
- Compose publishes 8000/6379/5432/9000; worker deps are host-installable; the worker's `model_auth_env()` merges both credentials despite its docstring claiming a preference (`worker_config.py:36-48`) — fix the docstring or the code when S1 touches it.

---

## 8. Decisions for the owner (each changes the build)

| Q | Decision | Recommendation |
|---|---|---|
| Q1 | Host runtime language/shape | **Python module in the repo** (`services/session-host/`), `python -m`/pipx; fits the repo and the tests; the SDK's bundled CLI is a fallback when no `claude` is on PATH |
| Q2 | Transport for Claude | **`claude -p --output-format stream-json` over stdin** (structured, resumable, quota events), not a PTY. PTY only if a later adapter has no headless mode |
| Q3 | Where sessions work | **Registered host directories** (the "real repos" you asked for) + `./workspaces/<id>` by default, confined by the ported rules; no worktree-per-agent in v1 |
| Q4 | v1 permission policy | `acceptEdits` inside the workspace + Bash allowlist; approvals-inbox routing in v2 |
| Q5 | Caps | 2 concurrent sessions per host; per-session cost/token caps default on, small; dispatch pauses on any non-`allowed` quota status |
| Q6 | **Zero-key install — does Auto itself run on the subscription?** | v1 **no**: session agents do the work; Auto (the manager) still needs an LLM key or OpenRouter. Making Auto run over `claude -p` is possible (`--append-system-prompt`, our tool loop as MCP) but is its own story with latency and tool-loop costs — decide after S1 proves the lane |
| Q7 | Codex in-wave | Yes as S5 (you named it); it is the interface proof |
| Q8 | Session host packaging | `make session-host` first; a signed binary later if contributors ask |

---

### 8a. Decisions taken (owner, 2026-08-30 — verbatim where it matters)

- Q1 host = Python module, `make session-host` — "a bridge to localhost for running commands": yes.
- Q2 headless `stream-json` default; PTY only for CLIs with no headless mode.
- Q3 registered host directories + `./workspaces` — "I guess so."
- Q4 v1 permissions as recommended — "fair for v1".
- Q5 **rejected the session cap**: "why cap at two… I have Max 20 so I will want many many sessions, all localhost so user can run 1 or 100, their choice and no cost to me." → no cap; only Claude's own limit signal pauses dispatch; warnings never do; per-session budget caps optional/default off.
- Q6 **hybrid**: "I would like Auto to also be subscription based eventually… for v1 we stick to API-managed Auto; we can run Kimi or GPT and have them managing Claude sessions… with OpenRouter we have 400 LLMs; DeepSeek reading my mails, Kimi writing reports, saving my tokens for coding work."
- Q7 Codex in-wave — yes. Q8 `make session-host` first — yes.
- Schedule: Phase 0+1 testing first (3 days), Phase 2 build from Friday 2026-09-04; subscription weekly window at 91 % until 2026-09-03.

## 9. PRD-234 statements the code contradicts (fix in the rewrite)

1. "Extract the session-drive core of `canvas_session_service.py`" — withdrawn (§4): canvas is welded to the SDK's permission objects, one-session-per-workspace state, and isolated auth; the host is a new module.
2. "Session lifecycle emits through `canvas_events.py` → `notify_board_event`" — impossible: the worker has no orchestrator code; `notify_board_event` needs a DB session; the canvas channel has no always-on consumer.
3. "Permission asks route through the existing approvals surface (`canvas_approvals` → approvals UI)" — those are two unconnected UIs.
4. "Deliverable promotion fires unchanged on completion" — completion creates no Deliverable; only `workspace_write_file` registers one.
5. "Session result flows back through the existing task-completion path" — there is no reusable path; it is an inline block that must be extracted.
6. "The execution trigger … branches" (singular) — four launch sites; branch inside `_launch_task_execution`.
7. "Strips `ANTHROPIC_API_KEY`" vs "canvas suite byte-identical" — mutually exclusive inside the shared env builder; moot once the host is a separate module (the host simply never sets it).
8. S3 "goes blocked through the ask lane" — via `create_grant`, not the agent-facing tool; resume lands in `assigned`, on whatever runtime the agent then has.
9. "Ralph runner launches with the key unset" — not in this tree; verified instead by the probe (`apiKeySource: none`).
10. `AUTH_EDITION` gating — correct for the orchestrator; the worker cannot see it (moot for the host, which checks the backend's edition itself).

---

## Appendix A — headless modes per CLI (adapter feasibility; unverified rows are marked)

| CLI | headless / JSON | verified |
|---|---|---|
| claude | `-p --output-format stream-json`, `--input-format stream-json`, `--resume` | ✅ this machine |
| codex | `codex exec --json` (non-interactive) | ⚠ from OpenAI docs; not run here |
| gemini | `-p` / `-i` | ⚠ munder uses `-i` (interactive); `-p` per docs |
| copilot | `-p` print mode | ⚠ munder uses it; not run here |
| cursor | `cursor-agent -p` | ⚠ installed here; not run |
| opencode | `opencode run` | ⚠ docs |
| qwen | gemini-cli fork, `-p` | ⚠ |
| grok, kimi, agy, crush, pi | unknown / TUI-first (munder types into crush's TUI) | ❌ PTY adapter likely |

Appendix B — munder mechanisms worth porting as code (all dependency-free in their tree): `shared/agentProvider.ts` (preset shape), `main/ptyEnv.ts` (env layering), `main/shellEnv.ts` (login-shell PATH), `main/breaker.ts` (thresholds), `main/cliInstall.ts` (missing-CLI ladder → hints only), `main/workerWake.ts` (HITL classification).

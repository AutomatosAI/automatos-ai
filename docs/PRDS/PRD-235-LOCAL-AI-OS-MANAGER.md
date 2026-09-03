# PRD-235 — Automatos, the manager of coding agents (the local "AI OS")

**Status:** DRAFT for owner review · 2026-09-03 · builds on PRD-234 (session mode; S1a/S1b/S4 merged, S2 in #694, S3 in #695).
**Owner's words (2026-09-03):** "turn this into a super management platform… the Code option more like a Cursor environment — see my files, repos, chat with my agent and fix my code… the full Claude Code builds and manages my code, using Auto and the board, calendar, heartbeats to manage it all… Auto can be Kimi and assigns tasks to Claude or DeepSeek or whoever is best suited… local open-source and SaaS must coexist… research Hermes, OpenClaw, munder and take their best… make Automatos the god of AI management systems."

## 0. In one paragraph

Automatos does not become an IDE; it becomes the manager the IDEs lack. Auto plans and assigns; the board, calendar and heartbeats drive; every unit of work is a **ticket → attempt → run** executed by the user's own `claude` (later Codex) in a worktree on the user's machine; results land as deliverables and a report; supervision, cost and state come from the hook control plane the CLI host already owns; the user takes over in their own terminal or editor with one click. The chat Code mode is the window onto that: repo tree, viewer, diff cards, live session, approvals — on the user's subscription. The three reference projects split cleanly and we borrow accordingly: **OpenClaw for policy** (permission modes, secrets, bindings, heartbeat contract), **munder-difflin for mechanism** (hooks over a socket, one writer per terminal, breaker ladder, cost attribution, live-roster injection), **Hermes for compounding** (agent-authored skills behind a staging gate, bounded memory, background review). Grounding: `docs/PRDS/Research/234-SESSION-MODE/REVIEW-2026-09-03-local-ai-os.md` and the three research briefings of 2026-09-03.

## 1. Non-negotiables

1. **Two editions, one codebase, nothing breaks.** Every capability here keys off `runtime: cli` (impossible in saas — `validate_runtime_configuration` refuses it when `CLI_RUNTIME_ENABLED` is off, and the flag is refused at boot outside `AUTH_EDITION=local`), or off `AUTH_EDITION=local` explicitly. SaaS takes bug fixes only; its API shapes gain optional fields, never required ones. Each story names its edition effect.
2. **The subscription rule (PRD-234 §Terms) stands.** Unmodified `claude`, the user's own login, never `-p`, never `--bare`, never a long-lived `bypassPermissions` session. Automatos never reads, copies, refreshes or forwards a credential store — not `~/.claude/.credentials.json`, not the Keychain (Hermes issue #55878: a spawned `claude` that fell back to the shared store logged the human out of their own sessions). Provider API keys are stripped from every child environment (OpenClaw had to build exactly this guard for Codex). The harness owns its login; Automatos owns routing, state and policy.
3. **Hooks are the control plane.** Session facts come from the hook events the host already receives (S1b); `~/.claude/projects/*.jsonl` is a fallback that degrades to "unknown", never a primary source.
4. **No second billed coding engine on local by default.** Code Canvas's SDK engine (API key in the worker) stays available for API agents; the Code pill defaults to session mode.
5. **Cost never lies.** On a subscription there is no per-token price: show tokens, quota windows and velocity, never an invented dollar figure (OpenClaw hides `$` for plan-billed sessions; munder shows tokens and only optionally USD).
6. House rules: no new table when an existing one fits; no `os.getenv` outside `config.py`; delete what a story replaces; skills repo is the source of truth (feedback memory) — agent-authored skills are *proposals*, never live edits.

## 2. What exists today (verified 2026-09-03)

- Session mode S1–S3: pairing, claim/lease, hooks-only settings, policy (files inside the session roots; interpreters on own files), worktree per git repo, deliverables from session files, session log in the task report, consent (drag/Run Now = approval), every dispatch lane files a ticket for a cli agent, heartbeat honesty, `platform_recommend_agent`, fleet runtime/session/host.
- Code Canvas: Monaco viewer + tabs + save, DiffEditor approval cards, one-shot exec terminal, worker exec sandbox (allowlist, 300 s cap), git verbs, commit + push on `canvas/<session>`, SSE session contract — engine = SDK client in the worker (API key).
- Agent selection: `AgentMatcher.rank` (skills 0.40, tools 0.25, model fit 0.15, availability 0.10, history 0.10 + semantic/field signals) with reasons; per-agent semantic cards; PRD-232 intent graph (tool routing).
- Board/consent/approvals (PRD-181/193/224–229), Command Center, fleet (PRD-228), watches (PRD-204), heartbeats/schedules/playbooks, channels, webhooks, Composio triggers, model registry + OpenRouter sync, model governance (PRD-223).

## 3. Design principles adopted (with their source)

| # | Principle | From | What it means here |
|---|---|---|---|
| P1 | Event plane over terminal plane | munder | Hook events drive the UI and the ledger; PTY bytes are for the human only. |
| P2 | Mechanism in code, policy in the prompt | munder ("Michael"), Hermes | Routing/escalation rules live in Auto's system prompt over a typed registry; the platform records every decision with its candidates and reason (munder's gap: unauditable routing). |
| P3 | One writer per terminal; automation never erases, never closes a menu | munder MD queue | Any automated write to a live session passes a prompt-ownership gate; buffer reads may only *clear* a draft. |
| P4 | Harness keeps its own login; route markers, not tokens | OpenClaw ACP rule | Already our invariant; extended to Codex with an isolated home (`CODEX_HOME`) and stripped keys. |
| P5 | Quota windows, not dollars | OpenClaw, munder | 5-hour / weekly windows + tokens/min per session; `$` only for API agents. |
| P6 | Breaker ladder, one step per beat, manager exempt | munder | `healthy → steering → constrained → stopped`; steer (context) before constrain (deny writes) before stop (opt-in). |
| P7 | Bounded memory that errors instead of drifting; frozen snapshot per session | Hermes | Two char-capped rows per workspace, promotion by a background reviewer, injected once per session. |
| P8 | Agent-authored skills behind a staging gate | Hermes `skill_manage` + `write_approval`, OpenClaw Skill Workshop | Proposals into a pending queue → diff → approve → PR to `automatos-skills` (source of truth). |
| P9 | Deny always wins; sandbox ≠ tool policy ≠ elevated | OpenClaw | Three orthogonal axes; one enum can be escalated past a deny. |
| P10 | Worktree per attempt with ignored-file provisioning and a single committer | OpenClaw managed worktrees, Conductor, munder | `.worktreeinclude` + setup script; only the platform commits integration state; squash-merge-aware GC. |
| P11 | Heartbeat = a periodic turn with a `NO_REPLY` contract, deferred while busy, never re-raising old items | OpenClaw | A heartbeat surfaces only what needs attention; on a subscription its cadence backs off. |
| P12 | Live roster injected as context that supersedes stale context | munder | Auto (and each session) gets the current floor at `SessionStart`/`UserPromptSubmit` — no stale-roster hallucination, no tool call. |

## 4. Stories

Each story: **Files** (reuse first), **Test**, **Editions**. Sizes S/M/L. Waves are dependency order, not calendar.

### Wave 2 — Code mode on session mode

**S1 · Repo-scoped Canvas — S.** The Code pill opens the Canvas on a folder: a ticket's `sessions/<id>`, a repo under `projects/`, or the workspace root. `CodingCanvasWidgetData` gains `rootPath`; `WorkspaceExplorer` lists `fetchDirectory(rootPath)` instead of `'.'`; the terminal seeds its cwd. A repo picker lists `projects/*` and `sessions/*`. **Files:** `frontend/components/chatbot/chat.tsx`, `widgets/types.ts`, `CodingCanvasWidget/index.tsx`, `workspace/WorkspaceExplorer.tsx`. Backend unchanged. **Test:** explorer roots; picker; terminal cwd. **Editions:** both (harmless in saas; `projects/` absent there).

**S2 · Conversation scoped to the folder — S.** `usePageContext({selected:{type:'repo', id: rootPath}})` so Auto's `workspace_read_file/write_file/list_dir/grep/exec/git` tools act in that folder; the page-context renderer names it. **Files:** `frontend/lib/page-context.ts`, `chat.tsx`, the page-context renderer under `orchestrator/consumers/chatbot/`. **Test:** a `workspace_grep` call from a repo-scoped chat targets the folder. **Editions:** both.

**S3 · The session on the Canvas (live, steerable) — M.** The host's hook events (`record_events`) are published onto the canvas SSE channel in `canvas_events.py` shapes: turns, file-edit refresh, and — from `PermissionRequest`/policy `ask` — approval cards whose decision flows back through the existing grant primitive to the hook's pending reply (the host's `ask` currently degrades to deny; S3 wires the reply). `useCanvasSession` renders it unchanged, keyed by the ticket's session id. **Files:** `services/cli_host_service.py`, `api/cli_hosts.py`, `services/workspace-worker/canvas_events.py` (shape reuse), `CodingCanvasWidget/useCanvasSession.ts`, host `session.py` (hold a PreToolUse reply up to the hook timeout while a card is pending; deny on timeout). **Test:** event → SSE frame mapping; a card decision reaches the waiting hook; timeout = deny. **Editions:** local only (host events exist only there).

**S4 · Ticket ↔ chat, takeover, deeplinks — S.** "Open this session in chat" (`/chat?repo=<cwd>&ticket=<id>`), a copyable `claude --resume` command, the transcript path, and `vscode://file/<path>` / `cursor://file/<path>` deeplinks on files, worktrees and the ticket. **Files:** `board-task-viewer.tsx`, `frontend/app/chat/page.tsx`, `SessionModeTab.tsx`. **Test:** link composition; copy. **Editions:** local only (renders only with `runtime_ref`).

**S5 · Where edits may land — S (decision D-B).** `projects/` stays read-only for the worker; edits go through a session and are registered as deliverables. Optional `LOCAL_PROJECTS_RW=true` dial flips the worker mount to read-write for people who want the Canvas editor to write into their repos directly. Symlinks are never accepted (resolved and rejected by the worker). **Files:** `docker-compose.yml`, `SessionModeTab.tsx`, docs. **Editions:** local only.

### Wave 3 — attempts, git flow, provisioning, preview

**S6 · Attempt as the first-class object — M.** `ticket → attempt (branch + worktree + agent + model) → run (one claude invocation)`; a second attempt with another agent, or a resume after takeover, is a row, not a special case; Auto's history signal learns from which attempt merged. **Files:** reuse `board_tasks.runtime_ref` for the current attempt and add `board_task_attempts` only if the audit needs more than the ring in `runtime_ref` (owner decision D-E); `cli_host_service.claim_for_host`/`apply_result`; the ticket viewer lists attempts. **Test:** two attempts on one ticket; resume keeps the session id. **Editions:** local only.

**S7 · Merge / Create PR / Rebase and a Checks panel — M.** Buttons that shell to `git`/`gh` in the ticket's worktree (through the host, never the worker): merge to base, open a PR with title/body from the ticket and the session's summary, rebase; a Checks panel (git status, CI status via `gh pr checks`, review comments). Conflicts hand off to the user's editor. **Files:** host (`session.py` git ops after a run), `cli_hosts.py` (an `actions` endpoint the UI calls, executed by the host), `board-task-viewer.tsx`. **Test:** fake `gh`; PR body composition. **Editions:** local only.

**S8 · Worktree provisioning — S.** `.worktreeinclude` (gitignore syntax) copies ignored files (`.env.local`, `node_modules` excluded by default) into each attempt's worktree; `.automatos/worktree-setup.sh` runs with a 120 s cap; snapshot before removal; GC is squash-merge-aware (`git diff --quiet base HEAD`). **Files:** host `session.py`/new `worktrees.py`. **Test:** copy list; setup failure aborts the attempt with the log. **Editions:** local only.

**S9 · Preview per worktree — M.** A per-attempt dev server with a preview URL surfaced on the ticket and in the Canvas (the useful 10% of an IDE). The host runs the repo's declared dev command; the platform proxies nothing — it shows the local URL. **Files:** host, ticket viewer. **Editions:** local only.

### Wave 4 — supervision, budgets, takeover

**S10 · Stop-hook work loop — S.** On `Stop`, if the ticket has queued follow-ups (review comments, an operator steer, an unfinished checklist), the hook answers `{"decision":"block","reason":…}` so the session keeps working — guarded by `stop_hook_active`, a processed cursor, and **never while a human holds the terminal** (munder removed the unguarded version after it spent credits while a user was answering a question). **Files:** host `session.py`, `cli_host_service.record_events` (queue), tests. **Editions:** local only.

**S11 · Circuit breaker — M.** Pure policy over recent events: repeated identical tool call (key = tool + bounded input hash), error storm, no-progress (debounced over two beats), tokens/min velocity; ladder `healthy → steering → constrained → stopped`, one step per beat, de-escalate on healthy beats, compaction exempt, Auto exempt. Effects: steer = `additionalContext` at the next hook; constrain = deny write tools at `PreToolUse`; stop = cancel the ticket (opt-in `hardStop`). **Files:** host `breaker.py` (pure) + `session.py`; `runtime_ref.breaker`; fleet row. **Test:** golden matrix. **Editions:** local only.

**S12 · Quota windows and velocity, not dollars — S.** Per session: tokens/min from `PostToolUse` + transcript reconciliation; per workspace: the plan's 5-hour and weekly windows as the host observes them (rate-limit notifications), shown on the fleet and the ticket; one board-level ceiling that parks new claims with a visible reason and a timer. **Files:** host, `cli_host_service`, fleet. **Editions:** local only.

**S13 · Prompt-ownership gate for takeover — M.** Sessions run under tmux on the host; the UI mirrors the PTY read-only; "Take over" hands a one-liner (`tmux attach -t automatos_<attempt>`); while a human is attached, every platform-originated write parks; the draft latch is one-directional with a 30-minute expiry (munder's phantom-draft lesson). **Files:** host (`tmux` launcher, mirror stream), `SessionBlock`. **Editions:** local only.

**S14 · OTel cost ingestion — S (optional).** `CLAUDE_CODE_ENABLE_TELEMETRY=1` + an OTLP endpoint in the local stack (`automatos-monitoring` has a collector) joined on `session.id` → attempt; an allowlist of attributes only (PII-free by construction); the transcript remains the fallback. **Editions:** local only.

### Wave 5 — manager and compounding

**S15 · Auto's routing policy as data + prompt — M.** A typed capability registry (`agents.configuration.capabilities` written the way Claude Code writes subagent descriptions), the live roster injected into Auto's planning context every turn (P12), `platform_recommend_agent` as the mechanism, the routing policy (prefer existing agents, cost tiers, when to ask) in Auto's system prompt; every assignment records `{candidates, chosen, reason}` on the ticket. Missions: a mission step for a cli agent files a ticket and the step closes from it (or cli agents stay excluded from mission matching — D-D). **Files:** `consumers/chatbot/auto.py`, `handlers_board_tasks.py`, `coordinator_service.py`/`dispatcher.py`. **Editions:** both (the registry and recording apply to API agents too).

**S16 · Heartbeat contract and standing orders — S.** The heartbeat prompt for every agent ends with the `NO_REPLY` contract and "do not infer old tasks"; heartbeats defer while the agent is busy; on a subscription the default cadence is 1 h; a workspace `STANDING-ORDERS.md` (authority / trigger / approval gate / escalation) is injected with the persona. **Files:** `heartbeat_service.py`, personas/context builder. **Editions:** both.

**S17 · Agent-authored skills behind staging — M.** A `platform_propose_skill` tool writes proposals (create/patch) into `skills_pending`; the admin UI shows the diff; approve = a PR against `automatos-skills` (source of truth) and a sync; reject = a note the proposer sees. A background reviewer after a completed ticket may propose. **Files:** tools 3-file pattern, a `skills_pending` table only if `deliverables`/`reports` cannot hold a proposal (D-F), `automatos-skills` PR automation. **Editions:** both.

**S18 · Bounded workspace memory + background review — M.** Two capped rows per workspace (`MEMORY`, `USER`; error on overflow, consolidate in-turn), injected once per session; a nightly promotion from PRD-206 memory, on a cheap model; provenance classes `owner|agent|untrusted|system`; heartbeat/cron/subagent turns never produce durable memory. **Files:** PRD-206 memory services. **Editions:** both.

**S19 · Model registry refresh and Auto presets — S.** OpenRouter catalogue sync on boot for local; Kimi K2 entries (price + context); refreshed Anthropic catalogue; a "Manager model" preset in Settings → System LLM; OpenRouter ids remain refused on cli agents. **Editions:** both (catalogue), local (preset copy).

### Wave 6 — adapters and scale

**S20 · Codex adapter — M.** Same lane, different binary: `codex exec` with an isolated `CODEX_HOME` provisioned by the user (`codex login --device-auth`), API keys stripped, results through the same ticket contract (PRD-234 S5). **Editions:** local only.
**S21 · Community lane — S each.** PRD-234 S6 as written.
**S22 · Manager scale — M.** Per-agent threads and back-pressure so Auto's context does not become the meeting room (munder issue #303: "at six agents everything collides in one orchestrator conversation"): one conversation per ticket thread, a work queue with limits per agent, hop caps on agent-to-agent messages. **Editions:** both.

## 5. Editions matrix

| Capability | Local | SaaS |
|---|---|---|
| Session mode, hosts, Code mode on sessions, attempts, git flow, breaker, quota, takeover, Codex | yes | absent (flag refused at boot; UI hidden by `isLocal`) |
| Auto routing registry + decision record, heartbeat contract, standing orders, skills proposals, bounded memory, catalogue refresh, manager scale | yes | yes (same code, API agents) |
| Heartbeat honesty, create-task fields, fleet optional fields | yes | yes (bug fix / optional fields) |

## 6. Decisions for the owner

- **D-A Code mode engine on local:** default the Code pill to session mode (S3); keep the SDK Canvas engine for API agents only. *Recommended.*
- **D-B Edits into your repos:** read-only mount, sessions write (S5). *Recommended.* Alternative: `LOCAL_PROJECTS_RW=true`.
- **D-C Auto's model:** Kimi K2 via OpenRouter once S19 lands (needs your OpenRouter key) or stay on Gemini Flash.
- **D-D Missions for Claude Code agents:** exclude from mission matching now; full ticket bridge in S15.
- **D-E Attempts:** keep attempts in `runtime_ref` (bounded ring) vs a `board_task_attempts` table (auditable history). *Recommended:* ring first, table when a second attempt is common.
- **D-F Skill proposals storage:** reuse `deliverables`/`reports` rows vs a `skills_pending` table. *Recommended:* reuse first.
- **D-G Breaker hard stop:** off by default (steer/constrain only), like munder. *Recommended.*
- **D-H Takeover transport:** tmux mirror + attach one-liner (S13) vs a writable browser terminal. *Recommended:* tmux; a writable browser terminal is the thing that breaks first and the auth liability.

## 7. Sequencing

W2 (S1–S5) is unblocked now and is what "Cursor-like" means for the owner. W3 needs W2's S3 for review feedback and S1's repo picker. W4 needs W3's attempt object for the breaker and quota to attach to. W5 is independent and can run in parallel with W2–W4 (it is API-agent-relevant, so it also benefits SaaS). W6 last.

## 8. Traps we will not copy (from the reference projects' own issue trackers)

- Reading or refreshing the user's Claude credentials (Hermes provider path; OpenClaw's credential discovery on non-read-only paths). We never do.
- Driving the TUI by blind screen-scraping with hardcoded sleeps and positional dialog answers (Hermes' `claude-code` skill). Hooks and structured events only.
- A long-lived `--permission-mode bypassPermissions` worker on the host mitigated by prose (OpenClaw's `coding-agent` skill). Policy is enforced at `PreToolUse`.
- Two writers to one prompt line; sending `Ctrl-U` or `Escape` to a terminal the user may be using (munder's shipped bugs).
- A routing decision that is only a prompt with no recorded rationale (munder). We record candidates and reason.
- Budgets that never stop anything and an orchestrator that hosts every conversation (munder #288, #303).
- A socket with no authentication that fails open silently (munder #277). Our shim denies on an unreachable host and the host re-binds a vanished socket (S2 fixes).
- Inventing dollar figures for subscription sessions.

## 9. Traceability

Review: `docs/PRDS/Research/234-SESSION-MODE/REVIEW-2026-09-03-local-ai-os.md` (page: claude.ai artifact be623dfa). Research briefings of 2026-09-03 (three code investigations of automatos-ai; Hermes Agent, OpenClaw and munder-difflin technical briefings with URLs). PRD-234 §Terms and D9–D16. PRD-224–229 (manager plane), PRD-228 (fleet), PRD-206 (memory), PRD-223 (model governance), PRD-181/193 (approvals), PRD-233 (open-core seam).

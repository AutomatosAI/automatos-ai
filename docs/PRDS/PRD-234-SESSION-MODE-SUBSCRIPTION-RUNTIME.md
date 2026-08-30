# PRD-234: Session Mode — agents run as your own terminal CLIs, on your subscription

> **Status:** REWRITTEN 2026-08-30 from the research record (`docs/PRDS/Research/234-SESSION-MODE/RESEARCH-2026-08-30-munder-difflin-and-automatos-seams.md`) and the owner's decisions of the same day. Supersedes the 2026-08-29 draft, whose central assumption (generalise the Code Canvas service) the research overturned. **Program:** Open-Core Phase 2 in the owner's numbering (Phase 3 in `PRD-WAVE-OPEN-CORE.md`). **Depends on:** PRD-209 + PRD-233 merged and tested (PR #650). **Local edition only** — a hosted workspace has no host process to talk to. **Build starts:** after the owner's Phase 0+1 test (planned from Friday 2026-09-04).

---

## Framing (CLAUDE.md §3)

**Extension — connect the manager plane that already exists to a new, local-only runtime module.** Automatos already has the board, tickets, the dispatch contract (PRD-224/226), questions (225/229), board events (227), fleet (228), memory and Deliverables. What is new: a **Session Host** process on the user's machine, provider **adapters as data**, a session queue/result contract on the backend, and the honesty states around them. The Code Canvas runtime is **not** refactored: it stays byte-identical. **Build size:** L overall (S1 is the substance). **Risk:** Medium — a second execution runtime; the no-silent-fallback invariant is what keeps it honest.

## Overview

Today every agent turn is an LLM API call (BYOK/OpenRouter). Session mode adds a second runtime: an agent's work executes as **a terminal-CLI session on the user's own machine, authenticated by the CLI's own login** — Claude Code first, Codex second, the rest as community adapters — with Automatos as the manager above: Auto assigns tickets and writes the dispatch contract, the board tracks them, memory and Deliverables work as before. **Hybrid by design** (owner, 2026-08-30): Auto and any agent can stay on the API/OpenRouter's 400+ models ("DeepSeek reading my mail, Kimi writing reports") while Claude Code sessions are kept for the code work they are built for — one board, one manager, mixed runtimes. The reference implementation is munder-difflin (MIT, read as code): its subscription-first, adapters-as-data, fail-loudly model is adopted; its PTY scraping, file mailbox and Electron UI are not.

**Non-goals (owner decisions):** no SaaS-side session mode; no scraping or reverse-engineering of subscription auth (the CLIs' own login and headless modes only); no auto-installing CLIs onto the user's machine (honest "install X" hints); teams connecting local hosts via the SaaS = future funnel; Auto itself on the subscription = a later story (Q6).

---

## Current reality (grounded 2026-08-30 — anchors in the research record §3)

- **Canvas is welded to the SDK and to one-session-per-workspace** (`services/workspace-worker/canvas_session_service.py:141-154, 370-409`): isolated `CLAUDE_CONFIG_DIR`, injected key/token, SDK permission objects, no result channel. Right for canvas, wrong for sessions; left untouched.
- **A container cannot run the user's CLI or see their login.** The worker image has no orchestrator code and cannot see `AUTH_EDITION` (`Dockerfile:34`, `worker_config.py:5-12`). Compose publishes 8000/6379/5432/9000 to the host.
- **Headless Claude Code is a structured runtime already**: `claude -p --output-format stream-json` emits `system/init` (`apiKeySource`, `session_id`, tools), `assistant`, `rate_limit_event` (`utilization`, `status`, `resetsAt`) and `result` (`num_turns`, `total_cost_usd`, `usage`, `permission_denials`, text). Verified on the owner's machine: `apiKeySource: none` = subscription billing.
- **Four launch sites funnel into `_launch_task_execution`** (`api/board_tasks.py:1138`); the completion writer is an **inline block** (`:1184-1222`); completion creates **no** Deliverable; the ask lane's reusable primitive is `create_grant(kind=KIND_QUESTION)` + park `blocked` (`handlers_asks.py:132-135`); `notify_board_event` needs a DB session (`services/board_events.py:38`); the canvas Redis channel has **no always-on consumer**.
- `Agent.configuration` (JSON, shallow-merged on update — `core/models/core.py:222`, `api/agents.py:874-880`) is the no-migration home for `runtime` / `session_provider`.
- `validate_auth_edition()` runs outside `run_stage` (`main.py:571`) — the only place a boot abort is real.

---

## Stories (test-first; CI cannot run real sessions — contract tests against recorded `stream-json` fixtures + pure units + source guards; the owner runs the smoke)

### S1 · Vertical slice — one ticket runs as a Claude Code session and lands on the board — L
**Files:** NEW `services/session-host/` (Python; `python -m automatos_session_host`; `make session-host`): connects outward to the backend (HTTP + Redis on the published ports), refuses unless the backend reports `edition: local`; announces capabilities (login-shell PATH probe ported from munder `shellEnv.ts:31-71`; installed CLIs + versions); claims session tasks; runs the **Claude adapter** — `claude -p --output-format stream-json --input-format stream-json --append-system-prompt <stable prompt> --permission-mode <policy> --allowedTools <list> --add-dir <dir> --model <model>` (the user's `claude` on PATH; the SDK's bundled macOS binary as fallback), `--resume <session_id>` for continuation; streams events; posts the **result** (final text, exit status, `session_id`, cost/usage, files touched). Backend: `Agent.configuration.runtime` + `session_provider`; **one branch inside `_launch_task_execution`** (covers all four launch sites) — `session` ⇒ enqueue `{task_id, workspace_id, agent_id, prompt, cwd, provider, model}`; **extract the completion writer** (`board_tasks.py:1184-1222`) into a function both runtimes call; `POST /api/v1/sessions/{task_id}/result`; sessions work in a **registered host directory** or `./workspaces/<id>` (Q3), confined by the ported `canvas_confinement` rules.
**Test:** adapter contract tests against recorded `stream-json` fixtures (init/assistant/rate_limit/result; error shapes); argv-builder units (prompt via stdin, never via argv); confinement units; dispatch-branch units (api agents byte-identical — regression; session agents enqueue with the contract); completion-writer extraction proven by the existing board tests passing unchanged; source guard: the host never sets `ANTHROPIC_API_KEY` in a session env.
**Notes:** The stable `--append-system-prompt` = agent soul + skills + the dispatch doctrine — nothing volatile (munder's prompt-cache invariant); the ticket's contract (already written into the description at creation) is the turn's prompt.

### S2 · Sessions on the board, in the fleet, and as Deliverables — M
**Files:** an **always-on** Redis subscriber in the orchestrator (`workspace:ws:{id}:session:events`) → `notify_board_event` (227) + fleet state (228): `runtime: session`, live tool, tokens/cost, quota state, host connection; a **Session Host card** in the fleet ("connected — Claude Code 2.1.x, logged in" / "not running — `make session-host`"); file-edit events → `deliverable_service.register` (the gap: no runtime registers Deliverables today except `workspace_write_file`); live event feed via the existing SSE pattern.
**Test:** event-mapping units (each host event → the board-event type it reuses); fleet read-model test shows the tag/quota/host; Deliverable registration idempotency; subscriber survives a Redis blip (reconnect test).

### S3 · Honest failure and quota — no silent fallback, no artificial caps — M
**Files:** preflight on dispatch: no host connected / CLI missing / not logged in / limit reached ⇒ the task is parked **`blocked`** with a question via `create_grant(kind=KIND_QUESTION)` (not the agent-facing tool) — "Session host not running: run `make session-host`", "Claude Code not logged in: run `claude login`", "Claude reports its limit is reached until <resetsAt>" — resumed through the existing answered-resume loop; **never** a fall-through to the API path. Quota: honour Claude's **own** `rate_limit_event` — dispatch pauses only when `status` is not `allowed` and resumes at `resetsAt`; warnings are surfaced, never acted on. **No cap on concurrent sessions** (owner: "1 or 100 — their choice"); per-session cost/token caps are **optional, default off**; loop guards (repeated-tool, error-storm — munder `breaker.ts:68-74` thresholds) default on.
**Test:** preflight-failure ⇒ blocked task + ask emitted + **zero** API-client invocation; resume ⇒ relaunch on the agent's current runtime; quota units: `allowed_warning` ⇒ no pause; `rejected`/exhausted ⇒ pause + timed resume; cap units: unset ⇒ unlimited.

### S4 · Settings, gate, docs — S
**Files:** agent settings modal gains one field group (runtime: API | Claude Code session | Codex session…; model); `SESSION_RUNTIME_ENABLED` (default false) gated in `validate_auth_edition()` — enabling it in saas aborts boot; self-hosting guide + QUICKSTART: "start the session host", what it can and cannot see, where sessions work, how quota shows.
**Test:** boot-guard unit (saas + enabled ⇒ RuntimeError outside `run_stage`); settings round-trip; doc guard extends `test_prd209_quickstart_honest.py`.

### S5 · Codex adapter — the interface proof — S/M
**Files:** second adapter (`codex exec --json`, OpenAI's headless mode; login via the user's `~/.codex/auth.json`, never copied or symlinked — the CLI reads it itself), same contract tests; `docs/contributing/session-adapters.md`: an adapter is a table row (binary, argv builder, prompt delivery, parser, resume, install hint, auth check) + recorded fixtures.
**Test:** the S1 contract suite run against Codex fixtures; preflight distinguishes "no codex login" from "no claude login".

### S6 · Community adapter lane — the remaining ten — S each
grok, kimi, gemini, antigravity, qwen, opencode, crush, pi, copilot, cursor. Each ships `LIVE-UNVERIFIED` until someone runs it (munder's honesty convention). A generic **PTY fallback adapter** (headless-less TUIs: crush, likely kimi/pi) is its own small story with munder's guarded-typing rules ported.

---

## Sequencing

S1 → S2 → S3 → S4 (S3/S4 parallel-safe) → S5 → S6 (open lane). Nothing starts before PR #650 is merged and the owner's Phase 0+1 test passes. Real-session validation = the owner's smoke script (create ticket → assign session agent → observe board), on a day the subscription's weekly window has headroom.

## Verification (CI only — sessions cannot run in CI)

Recorded-fixture contract tests at the adapter boundary; pure units for dispatch branching, completion extraction, preflight, quota, confinement, event mapping; source guards for the boot gate, the never-set-`ANTHROPIC_API_KEY` rule and the no-fallback invariant; the canvas suites unchanged (byte-identical by construction — nothing there is touched).

## Success metrics

- A ticket assigned to a session agent runs on the subscription (`apiKeySource: none`) and lands on the board with result, cost and the files it changed, exactly like an API agent's. Today: impossible.
- Host not running / not logged in / limit reached ⇒ a blocked ticket with an actionable question; **zero** API calls made on the user's behalf.
- The fleet shows every live session with runtime, tool and quota; any number of sessions in parallel.
- Adding a CLI is a data row + fixtures a contributor can ship without touching core.

## Decisions recorded (owner, 2026-08-30)

| Q | Decision |
|---|---|
| Q1 host shape | Python module in the repo, `make session-host` (`python -m`); a "bridge to localhost" that runs the CLIs as the user |
| Q2 transport | Headless `stream-json` is the default; PTY only for CLIs with no headless mode |
| Q3 where sessions work | Registered host directories (real repos) + `./workspaces/<id>` by default |
| Q4 v1 permissions | Auto-accept edits inside the workspace; Bash allowlist; approvals-inbox routing = v2 |
| Q5 caps | **No cap on sessions** — user's choice; only Claude's own limit signal pauses dispatch (warnings never do); per-session budget caps optional, default off |
| Q6 Auto | **Hybrid**: Auto and any agent stay on API/OpenRouter (400+ models) in v1; Claude sessions for code; Auto-on-subscription = its own later story |
| Q7 Codex | In-wave, as the interface proof (S5) |
| Q8 packaging | `make session-host` first; a signed binary if contributors ask |

## Open questions — owner's call (§12)

1. Approvals-inbox routing for live permission asks (v2): via Claude Code's `--permission-prompt-tool` (needs an MCP endpoint on the host) — confirm v2 timing.
2. Auto on the subscription (Q6 later story): `claude -p` as an LLM provider for Auto's turns, with our tool loop exposed as MCP — when?
3. Several session hosts (laptop + workstation) on one local install — v2 or later?
4. Per-agent git worktrees (munder's isolation) — v1 uses the directory as is; add later?

---

*Traceability: research record (this repo, `docs/PRDS/Research/234-SESSION-MODE/`); munder-difflin @ fc436bd (`src/shared/agentProvider.ts`, `src/main/pty.ts`, `hive.ts`, `breaker.ts`, `cliInstall.ts`); PRD-224–229 (manager plane); PRD-233 S2 (capability-degrade pattern); PRD-223 (fail-open = disease class); PRD-175 (`AUTH_EDITION`); owner decisions 2026-08-29 (hybrid, local-only) and 2026-08-30 (Q1–Q8).*

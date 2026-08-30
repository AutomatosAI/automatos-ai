# PRD-234: Session Mode — agents on your Claude subscription, zero API keys

> **Status:** DRAFT — spec only, no build yet. Grounded @ `origin/main 182cd6739` (2026-08-29); `file:line` refs may drift — confirm by grep at build. **Program:** Open-Core **Phase 3** (see `PRD-WAVE-OPEN-CORE.md`). **Depends on:** PRD-209 (boot), PRD-233 (local worker profile + tool seam). Reuses the Code Canvas session runtime (`services/workspace-worker/canvas_session_service.py`), the merged Auto-Manager wave (PRD-224 ticket lane · PRD-225/229 questions · PRD-226 dispatch doctrine · PRD-227 board events · PRD-228 fleet state), and `AUTH_EDITION` (PRD-175).

---

## Framing (CLAUDE.md §3)

**Extension — the runtime exists, the manager exists; this PRD connects them and makes the connection selectable.** `canvas_session_service.py` already drives Claude Agent SDK sessions with per-workspace config, permission callbacks, and confinement. The ticket lane already dispatches work to named agents. Net-new: a runtime selector on agent execution, the session-side prompt/result contract, and honest failure states. **Build size:** M-L (S1/S2 are the substance). **Risk:** Medium — a second execution runtime beside the API path; contained by local-only gating and no-silent-fallback rules.

## Overview

**The Basic edition's headline feature.** Today every agent turn is an LLM API call (BYOK or platform key). Session mode adds a second runtime: an agent's work executes as a **Claude Code session on the user's own machine, authenticated by the user's own Claude subscription login — no API key anywhere**. Automatos is the layer above: Auto assigns tickets, writes the dispatch contract, tracks the board, keeps memory and context; sessions do the work. The lane is proven in-house: the Ralph overnight runner launches headless sessions with `ANTHROPIC_API_KEY` deliberately unset and bills the owner's Max subscription — supported Agent SDK path, user's own machine, user's own subscription. Session mode is **structurally local-only** (the login lives on the user's machine), which is exactly why it belongs to Basic: it is the thing the SaaS cannot offer, and the reason a contributor installs the open edition.

**Non-goals (owner decisions, recorded):** SaaS-side session mode (structurally impossible — not built around); **teams connecting local Automatos workspaces via the SaaS subscription = future funnel** (owner-flagged 2026-08-29 as later + nice-to-have; nothing here precludes it — the runtime contract is workspace-scoped, which is the seam a future team-connect would use); no scraping or reverse-engineering of subscription auth — the supported `claude_agent_sdk` / Claude Code login path only; Auto's own orchestrator brain stays on the API path in v1 (**Q3**).

---

## Current reality (grounded @ `origin/main 182cd6739`)

- **The session runtime exists for one caller.** `services/workspace-worker/canvas_session_service.py`: lazy `claude_agent_sdk` import (`:129` — `ClaudeAgentOptions`, `ClaudeSDKClient`), per-workspace `CLAUDE_CONFIG_DIR` (`/workspaces/{workspace_id}/.canvas/claude/`, `:12,68`), permission callbacks (`PermissionResultAllow/Deny`, `:185`), confinement (`canvas_confinement.py`), approvals (`canvas_approvals.py`), events (`canvas_events.py`). Built for Code Canvas; nothing else can reach it.
- **Agent execution is API-only.** AgentFactory → LLM manager → provider clients (`core/llm/`), key resolution workspace-first (`core/llm/workspace_keys.py`). No runtime selector exists on agents or dispatch.
- **The manager plane is merged and waiting.** Ticket lane: `platform_create_task(assigned_agent_name=…)`, `platform_assign_task`, status→`in_progress` triggers execution (PRD-224). Dispatch doctrine: `DISPATCH_CONTRACT_FRAGMENT` — OBJECTIVE·OUTPUT·TOOLS·BOUNDARIES (PRD-226). Agent moves emit board SSE (PRD-227). Fleet state + `platform_fleet_status` (PRD-228). Questions/clarifications ride `approval_grants` (PRD-225/229, incl. the answered-resume loop US-005).
- **The subscription lane is proven.** Operational precedent 2026-08-27: headless sessions launched `env -u ANTHROPIC_API_KEY` bill the Max subscription; nested/headless launch works.

---

## Stories (test-first; CI is the only gate — no live Claude sessions in CI)

### S1 · SessionRuntime — generalize the canvas service into a selectable agent runtime — M/L
**Files:** `services/workspace-worker/` (extract the session-drive core of `canvas_session_service.py` into a runtime module both canvas and agent-dispatch consume — refactor, not fork; canvas keeps working byte-identically); agent model + dispatch path gain `runtime: api | session` (default `api`; **no new table** — extend the existing agent definition per CLAUDE.md §4); `orchestrator/config.py` (`SESSION_RUNTIME_ENABLED`, default false; boot guard: enabling it requires `AUTH_EDITION=local` — extend `validate_auth_edition()`); session env construction **explicitly strips `ANTHROPIC_API_KEY`** (subscription billing is the contract, mixed billing is a bug).
**Test:** runtime-module contract tests against a fake SDK client (the lazy-import seam at `:129` is the injection point — same pattern the canvas tests use); guard: `SESSION_RUNTIME_ENABLED=true` + `AUTH_EDITION=saas` ⇒ boot abort; env-construction test asserts the key-strip; canvas regression suite unchanged.
**Notes:** Per-workspace `CLAUDE_CONFIG_DIR` already gives sessions isolated auth/config; reuse it unchanged. Session concurrency cap = **Q1** (config dial, small default).

### S2 · Dispatch to sessions — the ticket lane drives real work — M
**Files:** the execution trigger behind status→`in_progress` (PRD-224's path) branches on the agent's runtime: `session` ⇒ enqueue to the worker's session runtime with a prompt built from the **PRD-226 dispatch contract** (OBJECTIVE·OUTPUT·TOOLS·BOUNDARIES) + agent soul/skills + workspace context pack; session result (final output + artifacts under the confinement root) flows back through the **existing** task-completion path so board events (227), watches, and Deliverable promotion (#611 lineage) all fire unchanged.
**Test:** dispatch-branch unit tests (api agents unchanged — regression; session agents enqueue with the composed contract); completion-path test: a fake-session result lands as task completion + board event + Deliverable exactly like an API agent's would (assert no parallel result path was invented).
**Notes:** Missions can target session agents too via the same AgentMatcher/dispatch seam — but the ticket lane is the v1 proof surface (single named agents, Gerard's stated management model). Mission-wide session staffing = **Q5**.

### S3 · Honest failure, no silent fallback — S
**Files:** session runtime preflight (login/config present? SDK importable?); on failure the task goes **blocked** with a question through the PRD-225 ask lane ("Claude login not found for session agent X — run `claude login` on this machine or switch the agent to API runtime"), resuming via the 229 answered-resume loop. **Never** silently fall back to the API path (that would bill keys the user chose not to use — the fail-open disease class, PRD-223 lineage).
**Test:** preflight-failure test asserts blocked-task + ask emission and **zero** API-client invocation; resume test: answer ⇒ task resumes on the corrected runtime.
**Notes:** This is where 225/229's machinery pays for itself — reuse, don't invent a second question surface.

### S4 · Sessions on the board and in the fleet — S
**Files:** session lifecycle (started / tool-permission-asked / finished / died) emits through `canvas_events.py` → `notify_board_event` (227) and surfaces in fleet state / `platform_fleet_status` (228) with `runtime: session` visible; permission asks route through the existing approvals surface (`canvas_approvals.py` → the approvals UI), not a new modal.
**Test:** event-mapping unit tests (each lifecycle event → the existing board-event type it reuses); fleet-state test shows the session agent with runtime tag.
**Notes:** Visibility is what makes "Automatos manages all my Claude sessions" true rather than vibes — the fleet page is the management console for sessions.

### S5 · OpenAI lane — Codex CLI sessions behind the same runtime interface — M
**Files:** a second implementation of S1's runtime interface driving OpenAI's Codex CLI headless sessions (subscription-authenticated, same preflight/honest-failure/fleet contracts); config dial per agent (`runtime: session` + `session_provider: claude | codex`).
**Test:** same contract-test suite run against the codex fake; preflight distinguishes "no codex login" from "no claude login" in the ask text.
**Notes:** **Q4 — in-wave or follow-on:** the interface (S1) is built for two implementations either way; the owner call is only *when* the second lands. Surfaced, not deferred.

---

## Sequencing

S1 → S2 → {S3 ∥ S4} → S5 (if in-wave). All after PRD-233 (worker in local profile is the substrate). The canvas refactor inside S1 is the one regression-sensitive step — land it first with the canvas suite as the gate.

## Verification (CI only — sessions cannot run in CI)

Real subscription sessions are unrunnable in CI by design (no login, and billing someone's subscription from CI would be wrong). The proof stack: contract tests against fake SDK clients at the lazy-import seam (canvas precedent), pure unit tests for dispatch branching / env-strip / preflight / event mapping, source guards for the boot-gate and no-fallback invariants, and the canvas regression suite for the refactor. First real-session validation is an owner smoke on the owner's machine — scripted (`scripts/` one-shot: create ticket → assign session agent → observe board), run by Gerard, not CI.

## Success metrics

- A zero-API-key local install (subscription login only) takes a ticket from `platform_create_task` → assigned session agent → completed task + board event + Deliverable. Today: impossible without keys.
- Missing login ⇒ blocked task + actionable question; **zero** silent API fallback. 
- Fleet view shows live session agents with runtime tags; permission asks land in the existing approvals surface.
- Canvas suite green before/after the S1 refactor (byte-identical canvas behaviour).

## Open questions — Gerard's call (§12)

1. **Concurrency cap (S1).** Sessions are heavyweight and share one subscription's limits. Default cap 2 concurrent sessions, config dial? Confirm a number.
2. **Session lifetime (S1/S2).** Per-task ephemeral sessions (recommended v1 — clean, resumable via dispatch contract) vs long-lived per-agent sessions with resume. Confirm.
3. **Auto itself on session runtime?** v1 keeps Auto (orchestrator brain) on the API path — chat latency and tool-loop shape fit the API client; sessions fit *task work*. Flip later if desired. Confirm v1 boundary.
4. **Codex lane timing (S5).** In this wave or the first follow-on? (Interface built for it either way.)
5. **Mission staffing on sessions (S2).** v1 = ticket lane only (recommended), missions follow once ticket-lane telemetry looks right. Confirm.
6. **UI surface for runtime selection.** Agent settings page gains the `runtime`/`session_provider` fields (recommended: yes, it's one field group) — or config-only for v1?

---

*Traceability: program doc `PRD-WAVE-OPEN-CORE.md` (owner decisions 2026-08-29 — session mode = Basic headline; teams-later funnel note); munder-difflin review 2026-08-27 (the operating model: manager above CLI sessions; artifact f31677a8) and its merged wave PRD-224–229; Code Canvas lineage (PRD-170/184 → `canvas_session_service.py`); subscription-lane precedent (Ralph runner, `env -u ANTHROPIC_API_KEY`, 2026-08-27 ops notes); PRD-223 (fail-open = disease class → S3's no-silent-fallback rule); PRD-175 (`AUTH_EDITION` boot guard being extended).*

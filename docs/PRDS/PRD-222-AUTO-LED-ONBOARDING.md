# PRD-222 — Auto-Led Onboarding (Mission Zero v2)

**Status:** DRAFT — for Gerard's review
**Date:** 2026-07-30
**Grounded @:** `main c1cce09ab`
**Evidence base:** 2026-07-30 onboarding review — full sweep of onboarding PRDs, code trace of the implemented flow, Academy inventory, and external research on the 2026 onboarding standard. Load-bearing code claims in §3 were re-verified by hand against source.
**Absorbs / retires:** PRD-47 (Shepherd tours — retire), PRD-130 (wizard UI retires; its pipeline survives as Auto tools), `MISSION-ZERO-ARCHITECTURE.md` (4-agent execution model — retire), PRD-123 §6.5 Pattern I (absorbed and upgraded here).
**Depends on (live):** PRD-221 page-context layer, PRD-205 chat live-receive bridge, PRD-143 analytics pipeline, PRD-200 mission approval/inbox, #611 deliverable promotion, PRD-206 memory phase 1.
**Wave 3 dependency:** PRD-207 voice stack.

---

## 1. Overview

Onboarding today is three surfaces that don't talk to each other: Shepherd.js tours (frontend-only, localStorage state), the PRD-130 Business Intake Wizard → Mission Zero (built end-to-end, never run for a real production user), and a chat-triggered Mission Zero prompt that cannot actually launch the real Mission Zero path. The first mission a customer would ever see runs unverified and auto-approved. Nothing about the journey is measured server-side. The platform surface overwhelms non-technical users, and the workspace requires a working LLM API key (BYOK) that no part of the flow helps the user provide.

This PRD replaces all of it with **one conversational spine led by Auto** — who is page-aware (PRD-221), persists across pages, and already has the platform tools to build a workspace. Onboarding stops being a UI bolted onto the product and becomes the first use *of* the product: the user learns the one behavior that unlocks everything else — ask Auto.

**Why now:** the waitlist opens for a pilot cohort (a technical company as testers and potential partners). The pilot must produce funnel data, not anecdotes, and the first impression must be trustworthy.

---

## 2. Decision record (2026-07-30 design session — do not relitigate)

| # | Decision |
|---|---|
| D1 | **Auto is the only onboarding guide.** On signup only Auto exists. The four hidden onboarding agents (VOYAGER/BLUEPRINT/SCRIBE/FORGE), the ephemeral-clone machinery, and the `source="mission_zero"` special-casing are retired, not fixed. This settles the PRD-123 vs MISSION-ZERO-ARCHITECTURE contradiction in favor of PRD-123: Auto plans and executes. |
| D2 | **Shepherd is retired.** No tours, no welcome modal. Auto answers "what is this page" contextually (PRD-221) and the flow follows the user across pages. |
| D3 | **Onboarding is conversational but card-structured** — key entry, scan progress, the proposal, and the checklist render as structured cards in chat, not free text. Skip-ahead affordances exist everywhere for technical users. |
| D4 | **Trial-credit-first, BYOK after value (revised 2026-07-30).** Every new workspace gets a platform-funded usage allowance (default **$5**, config-driven) covering onboarding and early play, pinned to an economical trial model tier, hard-stopped at the cap — with a global daily platform cap and a kill switch bounding total exposure (solo-funded platform). The BYOK ask moves to **after the BOOM moment** ("keep Auto running on your business"); OpenRouter is the single recommendation (pay-as-you-go, 400+ models); other providers collapsed underneath. |
| D5 | **Plan recommendation is part of onboarding.** Auto recommends a plan + starter operating team sized to the business. Progressive disclosure: users get what they need; the full catalog stays visible for awareness (marketplace), not in their nav. |
| D6 | **The intake pipeline survives, the wizard UI does not.** Firecrawl scan → RAG → Knowledge Graph → profile becomes a set of Auto-callable tools; doc upload already exists. |
| D7 | **Trust defaults:** verification ON for onboarding builds; approval is an explicit user "yes" on Auto's proposal; every build is narrated; the result persists as a Deliverable; secrets live in Railway and missing config degrades honestly and visibly. |
| D8 | **Instrument everything** server-side from day one. localStorage is never the system of record for onboarding state again. |
| D9 | **Test loop over workspace churn (added 2026-08-27).** Onboarding must be re-runnable in a single account/workspace via a dev-gated reset (default-off env flag + workspace-admin auth) so the flow can be tested, fixed, and re-tested without provisioning and deleting workspaces per attempt. A pilot/dev tool, not a product surface: unlinked, removable after the pilot. |

---

## 3. Current reality (grounded)

The map this PRD builds from. File:line refs are at `main c1cce09ab`.

- **Landing:** signup (currently a waitlist funnel — `frontend/components/auth/sign-up-form.tsx:61-64`) → `/` → hard redirect to `/chat` (`frontend/app/page.tsx`). Workspace lazily auto-provisioned on first API call with plan `starter` + `plan_limits`, Auto seeded, doc templates seeded (`orchestrator/core/auth/hybrid.py:257-387`).
- **`is_new_workspace`** = `agent_count == 0` excluding system agents (`orchestrator/api/workspaces.py:63-66`). Master gate for the welcome modal and all auto-tours; flips off forever at the first agent.
- **Shepherd:** frontend-only — `frontend/components/onboarding/{first-login-guard,welcome-modal}.tsx`, `frontend/hooks/use-auto-tour.ts`, `frontend/lib/shepherd/**` (9 tours + registry + localStorage storage), `frontend/styles/shepherd-custom.css`, 19 files with `data-tour=` anchors, deps `shepherd.js` + `react-shepherd` (the latter already zero-import). No backend concept named Shepherd exists.
- **Intake wizard:** built end-to-end (`frontend/components/wizard/**`, `orchestrator/api/wizard.py`, `orchestrator/modules/intake/**`, table `business_profiles`). 4 archetypes in backend; frontend labels know only `shopify_catalog` (`step-4-page-checklist.tsx:17-19`). Single one-way entry via the welcome modal. `WIZARD_ENABLED` gates only `/start` (`api/wizard.py:191`).
- **Mission Zero launch:** `POST /api/wizard/plan/{id}` sets `config={source:"mission_zero", auto_approve:True, skip_verification:True}` (`api/wizard.py:490-509`). Per `reports/dossiers/onboarding-intake.md`: **zero production runs ever.**
- **Verified defect — the clones can't wire what they build:** `_clone_onboarding_agents` sets `is_system_agent=False, required_role=None` (`orchestrator/services/coordinator_service.py:527-528`), so clones lose the system bypass (`orchestrator/core/security/hierarchy_permissions.py:156` requires flag AND allowlisted name), and agents FORGE creates default `reports_to_id=None` — every wiring call (assign tool/skill/plugin, heartbeat, playbook steps) resolves to `out_of_subtree` deny/escalate. Retired by D1, not fixed.
- **Verified defect — chat can't launch the real Mission Zero:** `platform_create_mission`'s schema omits `source` (`actions_missions.py:28-36`) and the handler defaults it to `"chat"` (`handlers_missions.py:113`), so the roster injection (`coordinator_service.py:2462-2474`) and clone path (`:1576-1590`) never fire from chat. Irrelevant after D1 — a chat mission IS the onboarding mission.
- **Chat-side prompt:** `OnboardingSection` (priority 2, 800-token budget, CHATBOT mode only) triggers on 0 active agents or trigger phrases (`orchestrator/modules/context/sections/onboarding.py:25-67`).
- **Composio:** connect flow exists only at `/tools` (`frontend/components/tools/composio-apps-section.tsx:89-118` → `POST /api/composio/connect/{app}` → `frontend/app/tools/callback/page.tsx`). No connect step in any onboarding path. Catalog count is dynamic from Composio; docs variously claim 100/150/400/863 — standardize on rendering the live count.
- **Measurement:** none. No funnel events; journey state is browser localStorage (`frontend/lib/shepherd/tour-storage.ts`).
- **Tool schema truth drift (5):** `platform_submit_report` over-requires (`actions_reports.py:99` vs `handlers_reports.py:18-24`); `platform_assign_tool_to_agent` under-requires the agent identifier (`actions_assignments.py:41` vs `handlers_assignments.py:19-20`); `platform_assign_skill_to_agent` / `platform_assign_plugin_to_agent` / `platform_configure_agent_heartbeat` declare `required: []` while handlers hard-fail without identifiers; `platform_install_plugin` / `platform_install_skill` declare `required: []` while handlers need id/slug; `platform_create_mission` config description omits `source` (resolved by deleting the source special-casing, not by documenting it).
- **Academy (sibling repos):** AIX (AI basics, 13 modules), ABF (AI for business, 9), APA (platform, 11) live and free at academy.automatos.app; labs reference the platform in prose only — no deep links, no shared-identity handoff, no conversion tracking. Zero academy routes exist inside automatos-ai.

---

## 4. Goals

- One onboarding path, led by Auto, that takes a brand-new user from signup to a working, right-sized workspace with the platform's value demonstrated on **their own business**.
- Sign-up → BOOM moment (Auto answers a question about their business from their own corpus) in ≤15 minutes nominal.
- Every stage instrumented server-side; a defined activation metric with a funnel behind it.
- The first autonomous build a customer sees is **approved by them and verified by the platform**.
- Non-technical users are never shown CodeGraph/NL2SQL-class surface; technical users can skip ahead at every stage.
- Onboarding is resumable across sessions and devices ("Auto, continue my setup").
- Education woven in: the user leaves knowing they can ask Auto for ~90% of tasks, and knows the Academy exists at their level.

---

## 5. The design

### 5.1 The spine

```
Sign up ──► land in /chat, Auto greets (no modal, no tour) · $5 trial credit active, meter visible
  [1] Three questions        business · first goal · AI comfort (sets language + surface)
  [2] Teach it your business site scan (intake tools) / doc upload / just talk → playback + corrections
  [3] The proposal           plan + starter team + apps + cost  ◄── APPROVAL GATE (user says yes)
  [4] Auto builds it         narrated · verified · 1–2 app OAuth connects inline
  [5] BOOM                   "ask me about YOUR business" — everything so far on trial credit
  [6] Power up (conversion)  connect your AI · OpenRouter · validated live · Daily Briefing scheduled
  [7] Run & learn            checklist card · Academy track by level · "ask Auto" everywhere
```

Every stage transition is written to server-side onboarding state and emits a funnel event. The user can leave at any stage; Auto resumes from the recorded stage on return.

Stages 1–5 run on the platform-funded trial allowance (W1·S9): the balance is visible throughout, Auto warns at 80%, and exhaustion renders a deterministic (non-LLM) state offering the key step — the flow never fails silently on cost. This is the reverse-trial / prove-on-their-own-data pattern: the key ask lands only after the value is demonstrated.

### 5.2 Stage specs

| Stage | UX | Backend | Reuse |
|---|---|---|---|
| **1 Questions** | Auto asks 3 questions conversationally (never a form): what's your business · what do you want handled first · how comfortable are you with AI (new ↔ very technical). Comfort level sets Auto's vocabulary and the exposure profile. | Answers persisted to onboarding state + memory; segment fields on `business_profiles`. Trial credit granted at workspace provisioning (W1·S9). | `OnboardingSection`, PRD-206 memory writes, `business_profiles`. |
| **2 Teach it** | Auto offers: scan your website / upload documents / or just tell me. Scan progress streams as a chat status card. Auto plays back the extracted profile in plain words — "correct me"; corrections become memory. | Intake pipeline exposed as platform tools (§6 W1·S4); doc upload path already exists; profile persists on `business_profiles`. | PRD-130 pipeline (`modules/intake/**`), SSE progress backbone, PRD-205 live-receive bridge, `DocumentManager`, Graphify. |
| **3 Proposal** | One structured card: recommended **plan** + **starter team** (sized to the business — a barber gets Auto + 1–2 helpers, 2 apps, 2 Playbooks; a tech company gets more) + what each piece does for them + est. monthly cost ("this build is covered by your trial credit") + "also available later" (awareness). User approves or edits conversationally. **Nothing is built before the yes.** | The build is a normal chat mission (default `awaiting_approval`) or, for small setups, direct tool calls executed only after explicit confirmation. Verification ON — `skip_verification` is never set. | Mission approval surface (PRD-200), marketplace browse tools, cost estimation from model pricing. |
| **4 Build** | Narrated: "Created your Marketing helper — it's on your Agents page." App connections requested inline: OAuth popup cards for the 1–2 apps the chosen package needs. Ends with the **onboarding summary written as a Deliverable** — the workspace's founding document. | Auto executes via existing platform tools on the trial model tier; Composio connect flow reused from `/tools`; summary emitted through the deliverables path (#611). | `platform_create_agent`/`assign_*`/`install_*`/`create_playbook`, Composio connect + callback, deliverable promotion. |
| **5 BOOM** | "Ask me anything about your business" → grounded answer (RAG + Knowledge Graph), still on trial credit — value proven before any payment or key. | Boom event recorded; activation clock running (§11). | RAG/S3 Vectors, Graphify. |
| **6 Power up (conversion)** | Card, framed as continuation not paywall: "Auto just read your business and built your team — keep him running. Recommended: OpenRouter — one key, pay-as-you-go, 400+ models" + masked inline entry + "I already use OpenAI/Anthropic/…" collapsed + trial balance shown. Validation result in-flow; failure fixable without leaving chat. Skip-ahead: Settings → Credentials. On a valid key: full model catalog unlocked, **Daily Briefing Playbook scheduled** — tomorrow morning the platform delivers value unprompted. Declining is allowed: remaining trial credit keeps working; the briefing delivers a connect-nudge instead of failing silently. | Credential save triggers a **live provider test call**; stored `test_status` must reflect that real result (the badge must never lie again). Valid key marks trial `converted`. Capability signal ("no working LLM key") injectable into the section context. | Existing credentials system + `EncryptionService`; workspace-key-first resolution (#610); Playbooks + heartbeats. |
| **7 Run & learn** | Server-side checklist card (3–5 outcome items: connect a second app · invite a teammate if the plan has seats · run your first mission · take the 10-minute course). Academy offer matched to comfort level (AIX novice / ABF owner / APA technical) with deep links. Every empty state carries the same line: **"Not sure? Ask Auto — he can do this for you."** | Checklist state on the workspace; academy referral tracking (Wave 3). | Empty-state components; Academy content (sibling repo). |

### 5.3 Per-customer variations (same spine, different settings)

| Segment | What changes |
|---|---|
| Novice SMB (doctor's office, barber) | Full guided spine · minimal exposure profile · benefits language only · AIX/ABF offered · voice option in Wave 3 (Auto asks the questions aloud). |
| Technical (the pilot cohort) | Same spine with skip-ahead at every stage ("or do it yourself: Settings → …") · full exposure · APA offered · still fully instrumented. |
| Team invitee | Skips the spine entirely. Auto greets with "meet your team": who's who, what each agent does, the Playbooks already running. Measured separately (member activation ≠ owner activation). |
| Partner / agency | Wave 3+: reference-workspace patterns (extract shapes, never data) per TODO's.md — placeholder, not silently dropped (§12 Q9). |

---

## 6. User stories

### Wave 1 — pilot-ready spine

#### W1·S1: Server-side onboarding state + funnel events
**Description:** As the operator, I need the journey recorded server-side so the pilot produces funnel data and the flow is resumable.

**Acceptance criteria:**
- [ ] `workspaces.onboarding` JSONB column (no new table): `{stage, started_at, updated_at, completed_at, segment: {business, goal, comfort}, checklist: {...}}`. Stage enum: `not_started | questions | teach | proposal | building | boom | powerup | completed | skipped`, plus `trial: {granted_usd, state}` (W1·S9).
- [ ] Alembic migration chains onto the single existing head (heads discipline — verify with `alembic heads` in CI).
- [ ] Every stage transition emits an event through the PRD-143 analytics pipeline (event name, workspace_id, stage, timestamp).
- [ ] `GET /api/workspaces/current` returns the onboarding stage alongside `is_new_workspace`.
- [ ] Stage transitions are monotonic-forward except explicit `skipped`; a completed workspace never re-enters the flow unless manually re-triggered.
- [ ] Tests: transition rules, event emission, API shape. No localStorage involvement anywhere in the new flow.

#### W1·S2: OnboardingSection v2 — the conversational spine
**Description:** As a new user, I meet Auto and he runs the whole journey in chat, resumable at any point.

**Acceptance criteria:**
- [ ] Trigger reworked: section injects while `onboarding.stage NOT IN (completed, skipped)` — not `agent_count == 0`. Manual re-trigger phrases preserved.
- [ ] Section content is stage-aware: it tells Auto which stage the workspace is in and what to do next (3 questions → teach → proposal → build → BOOM → power-up → checklist), including the trial-balance awareness and the exact OpenRouter recommendation copy and the "correct me" playback behavior.
- [ ] Auto records answers/stage advances via a new `platform_update_onboarding` tool (3-file pattern) — the section never advances state itself.
- [ ] Comfort level changes Auto's register (plain-language vs technical) and is stored to memory.
- [ ] Token budget re-measured; raised if the stage-aware prompt exceeds 800 tokens (§12 Q7). Section still returns `""` for completed workspaces.
- [ ] Tests: section output per stage; empty for completed; trigger phrases; tool contract.

#### W1·S3: Power-up — post-BOOM BYOK conversion with live validation
**Description:** As a new user who just watched Auto answer questions about my own business, I'm asked to connect my AI key at the moment the value is proven — one clear recommendation (OpenRouter), pasted without leaving chat, validated immediately.

**Acceptance criteria:**
- [ ] The key ask renders **after the BOOM moment** (stage `powerup`), framed as continuation, never a paywall: "keep Auto running on your business — connect your key (2 min)." Trial balance shown alongside.
- [ ] Chat card for key entry (masked input) posting to the existing credentials API; deep link to Settings → Credentials as the skip-ahead.
- [ ] Credential save performs a live provider test call; `test_status` and the UI badge reflect the **actual** result; failures render in-flow with the provider's error.
- [ ] A valid key marks the trial `converted`, unlocks the full model catalog, and schedules the Daily Briefing Playbook on the user's key.
- [ ] Declining is allowed: the flow completes on remaining trial credit; the scheduled briefing delivers a connect-nudge instead of failing silently once credit is exhausted.
- [ ] Other providers listed collapsed; OpenRouter is the only top-level recommendation.
- [ ] Test fixtures use obviously-fake key formats (gitleaks-safe). Tests: validation pass/fail paths, badge truth, converted transition, decline path.

#### W1·S4: Intake pipeline as Auto tools ("teach it your business")
**Description:** As a new user, I give Auto my website and documents in conversation; he reads them and plays back what he learned.

**Acceptance criteria:**
- [ ] New platform tools (3-file pattern): `platform_scan_business_site` (starts the PRD-130 scan/scrape pipeline for a domain, returns a progress handle) and `platform_get_intake_status` (stage + summary). Existing wizard endpoints remain untouched in this wave.
- [ ] Pipeline progress renders as a chat status card via the PRD-205 live-receive bridge (SSE backbone reused).
- [ ] On completion, Auto plays back the business profile in plain language and stores user corrections via memory + `business_profiles` PATCH.
- [ ] Honest-degrade: if Firecrawl is not configured the tool returns a clear "not configured" result and Auto offers doc upload / conversation instead — never a silent 503. No secrets in any file; config stays in Railway.
- [ ] Tests: tool contracts, not-configured path, progress-card event shape. (Pipeline internals already covered by PRD-203 O·S2 tests.)

#### W1·S5: Trust defaults — proposal, approval, verified build, summary Deliverable
**Description:** As a new user, nothing is built until I say yes, the build is verified, and I get a founding document at the end.

**Acceptance criteria:**
- [ ] The onboarding build runs as a normal chat mission: `awaiting_approval` default preserved; `skip_verification` never set; approval happens via the existing mission approval surface, with Auto linking it and confirming after.
- [ ] Small-setup path: for builds under the threshold (§12 Q4), Auto executes direct tool calls instead of a mission — but only after an explicit user confirmation message; the proposal card is mandatory either way.
- [ ] Cost shown on the proposal: plan price placeholder + estimated token cost for the build.
- [ ] On completion Auto writes the **onboarding summary** (what was built, why, what happens next) through the deliverables path so it appears in the workspace's Deliverables tab; stage advances to `boom` then `completed`.
- [ ] Tests: no onboarding path can create a mission with `skip_verification` or `auto_approve`; summary lands as a deliverable; stage advancement.

#### W1·S6: Capability health — honest-degrade surfaced
**Description:** As the operator, I can see which onboarding capabilities are live, and Auto adapts instead of failing blind.

**Acceptance criteria:**
- [ ] A capability report (LLM key valid, Firecrawl configured, Composio configured, Redis present) available to the OnboardingSection and on the admin health surface.
- [ ] All reads via `config.py` (no `os.getenv` elsewhere); no values ever written to repo files.
- [ ] Tests: report shape; section adapts per missing capability.

#### W1·S7: Tool-schema truth pass
**Description:** As Auto (the executor of everything above), tool schemas must tell the truth so onboarding never dead-ends on a schema/handler mismatch.

**Acceptance criteria:**
- [ ] Fix the five §3 mismatches: `platform_submit_report` required → `["title","content"]`; agent-identifier requirement expressed for `platform_assign_tool_to_agent`, `platform_assign_skill_to_agent`, `platform_assign_plugin_to_agent`, `platform_configure_agent_heartbeat`; id/slug requirement expressed for `platform_install_plugin` / `platform_install_skill` (where either-of, encode in description + validation error copy, consistent across the family).
- [ ] A consistency test walks every platform tool: fields the handler hard-fails on are declared (or documented either-of); fields with handler defaults are not in `required[]`. This is the recurrence guard for the PRD-onboarding "blueprint rules wall" bug class.

#### W1·S8: Kill the modal, land in the conversation
**Description:** As a new user, my first screen is Auto talking to me — no modal, no tour.

**Acceptance criteria:**
- [ ] `FirstLoginGuard`/`WelcomeModal` no longer mount for new workspaces once the spine is live (full deletion is W2·S5; this story just stops the mounting so the pilot sees one flow).
- [ ] Auto proactively greets on first chat load for `stage=not_started` (OnboardingSection makes the first move; PRD-208 presence patterns apply).
- [ ] Component test: new workspace renders chat with no modal; frontend CI green.

#### W1·S9: Trial credit ledger & enforcement
**Description:** As the operator (solo-funded), I grant each new workspace a small usage allowance so users reach value before BYOK — with my total exposure bounded at all times.

**Acceptance criteria:**
- [ ] Trial state on `workspaces.onboarding.trial`: `{granted_usd, state: active|warned|exhausted|converted}`. Spend is **derived from the existing usage telemetry** (cached counter) — never a second bookkeeping system.
- [ ] Enforcement at the single LLM-dispatch choke point where key resolution already happens (workspace-key-first, #610 pattern): workspace BYOK → user's key, no trial accounting; else trial `active` → platform key + **trial model allowlist** (config); else → deterministic exhaustion state. Hard stop at 100%; Auto warns at 80% (reuse the budget-gate pattern).
- [ ] **Pre-work: verify how keyless workspaces resolve today.** If they already fall through to the platform key, document it — that is unbounded platform spend this story closes, not new cost.
- [ ] No background burn: trial workspaces get no heartbeat/scheduled execution until converted; anything scheduled pauses at exhaustion with a visible notification, never a silent failure.
- [ ] Config (Railway values, read via `config.py`): `TRIAL_CREDIT_USD` (default 5.00), `TRIAL_GLOBAL_DAILY_USD` (aggregate daily cap — new grants pause when reached), `TRIAL_MODEL_ALLOWLIST`, `TRIAL_ENABLED` kill switch.
- [ ] One trial per Clerk user, not per workspace; the waitlist remains the outer admission gate for the pilot.
- [ ] Balance pill in chat + Command Center; the exhausted state (banner + credentials card) renders **without any model call**.
- [ ] Funnel events: `trial_granted / trial_warned / trial_exhausted / trial_converted`.
- [ ] Tests: cap enforcement, model pinning, resolution order, one-per-user, global-cap pause, kill switch, exhaustion UI, no-background-spend.

#### W1·S10: Dev reset — re-runnable onboarding (single-account test loop) *(added 2026-08-27, D9)*
**Description:** As the operator testing the flow with an alias account, I reset onboarding in my test workspace and run it again — same account, same workspace — instead of creating and deleting workspaces per attempt.

**Acceptance criteria:**
- [ ] `reset_onboarding(db, workspace, *, reset_trial, wipe_built, wipe_credentials)` in `services/onboarding_state.py`: rewrites `workspaces.onboarding` to the initial doc via full JSONB reassignment (PRD-220-safe). It is the ONLY writer allowed to move the doc backward — `advance_onboarding_stage`'s monotonic/terminal validator stays exactly as strict as today. Preserves the `trial` object unless `reset_trial`; stamps `resets` (incrementing int) + `last_reset_at` inside the doc so test runs are distinguishable in funnel data.
- [ ] `reset_trial=true`: strip the trial, then re-grant through `grant_trial_at_provisioning` (reuse — never a second grant implementation). Result: a fresh `active` trial at $0 spent when eligible; a kill-switch/daily-cap decline is reported in the response as a pause, not an error (honest degrade).
- [ ] `wipe_built=true`: workspace-scoped deletion of what onboarding built — non-system agents (`is_system_agent=False` AND `required_role != 'onboarding'`) with their dependent rows, missions + their orchestration tasks, agent reports/Deliverables (incl. the onboarding summary), intake documents + workspace graphs, and the workspace's S3 document prefix — **reusing `services/workspace_purge.py` machinery** (scoped-table discovery, FK-safe ordering, S3 prefix purge) parameterized to spare survivors; never a hand-maintained duplicate table list. The purge service's soft-deleted-workspace precondition and its workspace-row/Clerk-user deletion steps must NOT apply here.
- [ ] Survivors proven by test after a full-flag reset: the workspace row, `users` rows, the Clerk user, system agents (Auto), `required_role='onboarding'` template agents, workspace credentials (unless `wipe_credentials`), and the freshly-written onboarding doc.
- [ ] `wipe_credentials=true`: this workspace's credential rows deleted; the operator workspace (`PLATFORM_KEY_WORKSPACE_ID`) provably untouched (scoping test).
- [ ] Endpoint `POST /api/workspaces/current/onboarding/reset` on the existing workspaces router (no new router file): gated on `config.ONBOARDING_RESET_ENABLED` (default false, read via `config.py` only, documented in `.env.example` as temporary) → **404 when off**; workspace-admin auth in the same bucket as the settings/onboarding-agents admin check; response reports counts of everything reset/wiped. Committed route manifest updated (route-manifest CI test green).
- [ ] Frontend dev URL `frontend/app/dev/reset-onboarding/page.tsx` — unlinked from all nav: shows current stage + trial state, three switches (reset trial / wipe built / wipe credentials), a RESET button that calls the endpoint, clears tour/onboarding localStorage via the existing `frontend/lib/shepherd/tour-storage.ts` helpers (imported, never modified), then redirects to chat. Renders a plain "reset disabled" state on 404. Warns when `wipe_credentials` is on while `reset_trial` is off (a converted workspace with no key falls through to unmetered platform resolution).
- [ ] Tests: reset from every stage incl. terminal ones; trial preserved vs re-granted vs declined; wipe survivor set; cross-workspace scoping (a second workspace's rows untouched); disabled→404; page component test.

### Wave 2 — right-size + retire

#### W2·S1: Exposure profiles (progressive disclosure) — *blocked on §12 Q1 tier definitions*
**Description:** As a barber, I never see CodeGraph, NL2SQL, or team features I didn't buy; as the operator, I define exposure in config, not code forks.

**Acceptance criteria:**
- [ ] A config-driven exposure map: plan/tier → visible nav items, visible tool families, available marketplace surface. Hidden ≠ deleted: the marketplace/catalog retains full awareness with plan labels.
- [ ] `workspaces.plan` + `plan_limits` (already provisioned at signup) drive the profile; changing plan changes exposure without redeploy.
- [ ] Auto's tool surface for small plans is trimmed accordingly (also reduces per-turn tool token load).
- [ ] Tests: exposure resolution per plan; nav rendering per profile; tool-surface filter.

#### W2·S2: Proposal recommends the plan
**Description:** As a new user, Auto recommends a plan sized to my business, and my choice sets the workspace up accordingly.

**Acceptance criteria:**
- [ ] Stage-4 proposal includes the recommended plan with plain-language contents and price; alternatives listed; user's choice is set on `workspaces.plan`.
- [ ] Commercial checkout/billing is **not** in this story (§12 Q5) — plan assignment only.
- [ ] Funnel events: plan proposed vs plan accepted (with edits).

#### W2·S3: Inline Composio connect
**Description:** As a new user, I connect my 1–2 real apps during the build without leaving chat.

**Acceptance criteria:**
- [ ] A connect card in chat opens the existing OAuth popup (`/api/composio/connect/{app}` + callback page); result reflected back into the conversation.
- [ ] Auto requests only the apps the approved package needs; "first integration connected" emitted as its own funnel event.
- [ ] Tests: card → popup contract; callback → chat state; event emission.

#### W2·S4: Post-setup checklist card
**Description:** As a new user, I have 3–5 outcome-framed next steps that survive across sessions.

**Acceptance criteria:**
- [ ] Checklist state lives in `workspaces.onboarding.checklist`; rendered as a dismissible card in chat and on the Command Center; items: connect a second app · invite a teammate (only if plan has seats) · run your first mission · take the matched Academy course.
- [ ] Item completion is detected from real events where possible (e.g., second connection), not manual ticks.
- [ ] Tests: item detection, dismissal, per-plan item set.

#### W2·S5: Retirement — delete the superseded surfaces
**Description:** As the codebase, the replaced onboarding is deleted, not left dark (clean-replace rule).

**Acceptance criteria:**
- [ ] Delete per §10 inventory: Shepherd (components, hooks, lib, styles, both deps, `data-tour` attrs), wizard UI route + components (backend pipeline stays), the four onboarding agent seeds + clone/cleanup machinery + `source="mission_zero"` branches + roster injection, `OnboardingAgentsTab` + `/api/settings/onboarding-agents`.
- [ ] Cleanup migration deactivates/removes seeded `required_role='onboarding'` system agents from live DBs.
- [ ] Route-manifest updated (hand-add/remove + count bump) for the deleted wizard route; frontend route-contract CI green.
- [ ] Org-chart empty state copy re-pointed from "Run Mission Zero" to asking Auto.
- [ ] No orphan imports; grep-clean for `shepherd`, `mission_zero` source checks, `use-auto-tour`.

#### W2·S6: `is_new_workspace` consumers migrated
**Description:** As the codebase, the one-shot boolean is replaced by the stage machine.

**Acceptance criteria:**
- [ ] All remaining consumers read `onboarding.stage`; `is_new_workspace` removed from the API response if no consumer remains (route-contract updated) — or kept with a deprecation note if external consumers exist (verify widget/SDK first).
- [ ] Tests updated accordingly.

### Wave 3 — reach

#### W3·S1: Academy bridge (platform side)
**Description:** As a new user, the Academy meets me at my level, inside the product.

**Acceptance criteria:**
- [ ] Comfort level → track offer (AIX/ABF/APA) with deep links carrying referral parameters; checklist integration ("take the 10-minute course").
- [ ] Academy-referred signups and platform→academy clicks tracked in the funnel.
- [ ] Cross-pod flag: academy-repo work (deep links into app.automatos.app from labs, shared-Clerk verification, completion signal back) is a separate PRD in `automatos-academy` — one line, not built here.

#### W3·S2: Voice onboarding option
**Description:** As a non-typing user, Auto asks the setup questions out loud.

**Acceptance criteria:**
- [ ] Opt-in voice lane over the PRD-207 stack for stages 1–4 (questions, key guidance, playback, proposal read-out); falls back to text at the key-entry card (keys are never spoken).
- [ ] Funnel distinguishes voice vs text onboarding sessions.

#### W3·S3: Invitee flow — "meet your team"
**Description:** As an invited member, I'm introduced to a working workspace, not a builder flow.

**Acceptance criteria:**
- [ ] Invitation-accepted users bypass the spine; Auto greets with the team roster, what each agent does, and the running Playbooks.
- [ ] Member activation events recorded separately from owner activation.

#### W3·S4: Partner / reference-workspace lane
**Description:** Placeholder scoped by TODO's.md (reference workspaces, package extraction — patterns, never data). Listed so it is not silently dropped; timing is Gerard's call (§12 Q9).

---

## 7. Functional requirements

- FR-1: Onboarding state machine per §6 W1·S1 stored on `workspaces.onboarding`; server is the only system of record.
- FR-2: Every stage transition emits an analytics event; the funnel `signup → questions → key_valid → taught → proposal → approved → built → boom → checklist_n` is queryable.
- FR-3: OnboardingSection v2 drives the spine, stage-aware, resumable, and returns empty for completed/skipped workspaces.
- FR-4: State advances only through `platform_update_onboarding` tool calls (auditable), never as a prompt side effect.
- FR-5: Each new workspace receives a platform-funded trial allowance (default $5, config-driven) enforced at a single LLM-dispatch choke point and pinned to a trial model allowlist; the BYOK ask happens **after** the BOOM moment; keys are validated with a live call; stored test status must equal the live result.
- FR-6: Business ingestion (site scan, doc upload, conversation) is available to Auto as tools with honest not-configured results.
- FR-7: No onboarding-initiated build executes without an explicit user approval; no onboarding path may set `skip_verification` or `auto_approve`.
- FR-8: The onboarding summary is persisted as a Deliverable on completion.
- FR-9: The proposal recommends a plan; the accepted plan sets `workspaces.plan` and (Wave 2) the exposure profile.
- FR-10: Exposure profiles are config-driven per plan; hidden capabilities remain discoverable in the marketplace with plan labels.
- FR-11: App connections requested during onboarding reuse the existing Composio OAuth flow inline; the first successful connection is a first-class funnel event.
- FR-12: Comfort level (novice ↔ technical) adjusts Auto's language, skip-ahead visibility, and the Academy track offered.
- FR-13: All five tool-schema mismatches in §3 are corrected and guarded by a consistency test.
- FR-14: Shepherd, the wizard UI, and the four-agent Mission Zero machinery are deleted in Wave 2 per §10; the intake pipeline endpoints survive as the tool substrate.
- FR-15: All config reads go through `config.py`; no secret values appear in any repo file; missing config degrades visibly.
- FR-16: One trial per Clerk user; a global daily trial-spend cap pauses new grants when reached; `TRIAL_ENABLED` kill switch; all trial values config-driven, none hardcoded.
- FR-17: Trial workspaces accrue no background/scheduled spend; at exhaustion, scheduled work pauses with a visible notification and the UI renders a deterministic non-LLM state offering the key step.
- FR-18: `trial_granted / trial_warned / trial_exhausted / trial_converted` are first-class funnel events; trial→BYOK conversion is a headline metric.
- FR-19: A dev-gated reset (`ONBOARDING_RESET_ENABLED`, default off; workspace-admin only) returns a workspace's onboarding to `not_started` — optionally re-granting the trial and wiping built artifacts/credentials — without deleting the workspace, its users, or the Clerk user; every reset stamps an incrementing `resets` counter in the onboarding doc.

---

## 8. Non-goals

- No time-boxed or unmetered free tier — the trial is usage-capped only (D4). No payment-card capture anywhere in onboarding.
- No billing/checkout implementation — plan assignment only; commercial wiring is a separate decision (§12 Q5).
- No waitlist changes (Gerard controls cohort admission in Clerk).
- No academy-repo code (cross-pod; own PRD there).
- No new tour/walkthrough library of any kind.
- No fix for the ephemeral-clone permission defect — the machinery is deleted instead (D1).
- No Shopify-app or widget entry-point work.

---

## 9. Technical considerations & traps

- **CI is the only gate; nothing runs locally.** All ACs verify through CI (component tests, orchestrator tests, route-contract). Tell spawned agents.
- **Alembic heads:** exactly one head after W1·S1's migration; `alembic upgrade heads` runs on boot.
- **Route manifest:** deleting the wizard route (W2·S5) requires the committed manifest edit + count bump or CI fails.
- **3-file tool pattern** for `platform_update_onboarding`, `platform_scan_business_site`, `platform_get_intake_status` (schema in `actions_*`, handler in `handlers_*`, registration in the executor).
- **F001 dependency:** the OS review found an `agent_factory` kwarg crash affecting non-chat agent execution. If the mission path is used for onboarding builds (large setups), verify F001 is fixed first or the pilot uses the direct-call path only. Check before Wave 1 exit.
- **Trial enforcement point:** one choke point in the LLM service where key resolution already happens (workspace-key-first, #610): workspace BYOK → theirs; else trial active → platform key + trial model allowlist; else → deterministic exhaustion state. **Verify first how keyless workspaces resolve today** — if they already fall through to the platform key, current spend is unbounded and W1·S9 is retroactive protection, not new cost.
- **No background burn on trial:** heartbeats/scheduled Playbooks stay off (or pause at exhaustion with a notification) for trial workspaces — an idle trial workspace must burn $0.
- **Exhaustion UI is non-LLM:** the exhausted state (banner + credentials card) must render without any model call — a broke workspace can still take a key.
- **OnboardingSection budget:** currently 800 tokens, never dropped, priority 2. Stage-aware content must be measured; raise deliberately if needed (§12 Q7).
- **Tool-surface load:** Auto carries ~11k tokens of tool schemas per turn today; W2·S1's exposure trimming should reduce this for small plans — measure before/after.
- **gitleaks:** any test fixture resembling a key must be an obviously-fake format.
- **`react-shepherd` is already zero-import** — deletable immediately with W2·S5.
- **Public repo:** no gold sets, no real domains beyond public ones, no customer names in fixtures (InbuildUK references stay only in historical PRDs).
- **Reset bypass (W1·S10):** `reset_onboarding` is the single sanctioned backward writer of the onboarding doc — never loosen `advance_onboarding_stage`'s validator to enable resets; the forward spine stays strict.
- **Wipe reuse (W1·S10):** parameterize `workspace_purge`'s internals rather than duplicating its scoped-table list — two deletion lists WILL drift. The purge service validates soft-deletion and deletes the workspace row + Clerk user; the reset path must inherit none of those steps.
- **New backend route (W1·S10):** the reset endpoint needs the committed route-manifest hand-add + count bump — same CI trap as W2·S5's wizard-route deletion.

---

## 10. Deletion inventory (Wave 2, clean-replace)

| Surface | Paths |
|---|---|
| Shepherd frontend | `frontend/components/onboarding/{first-login-guard,welcome-modal}.tsx` (+ tests), `frontend/hooks/use-auto-tour.ts`, `frontend/lib/shepherd/**`, `frontend/styles/shepherd-custom.css`, `data-tour` attributes (19 files), deps `shepherd.js`, `react-shepherd`, ProfileMenu "Tour this page" entry |
| Wizard UI | `frontend/app/onboarding/wizard/**`, `frontend/components/wizard/**`, `frontend/hooks/use-wizard-{api,progress}.ts` (progress hook logic moves to the chat card), route-manifest entry |
| 4-agent machinery | `orchestrator/core/seeds/seed_onboarding_agents.py` + boot call (`main.py:220-227`) + lazy re-seed (`api/wizard.py:555-576`), `_clone_onboarding_agents` / `_cleanup_ephemeral_agents` + `source=="mission_zero"` branches (`services/coordinator_service.py`), roster injection (`:2462-2474`), hierarchy allowlist entries for the four names, cleanup migration for seeded rows |
| Admin for the four | `frontend/components/settings/OnboardingAgentsTab.tsx` + mount, `orchestrator/api/onboarding_agents.py` + mount |
| Kept | `orchestrator/modules/intake/**`, `orchestrator/api/wizard.py` pipeline endpoints (now tool substrate; rename/slim as part of W2·S5 if desired), `business_profiles`, SSE progress backbone |

---

## 11. Success metrics

- **Activation (the number):** within 7 days of signup — setup approved **and** ≥1 business question answered by Auto from the workspace corpus **and** first Playbook or mission executed. Public benchmarks put median B2B activation at 25–37%; pilot target (hand-held cohort): **≥60% activation; ≥80% reach the proposal stage.**
- **Time to BOOM:** signup → grounded business answer ≤15 minutes nominal; measured, not assumed.
- **Funnel visibility:** every §7 FR-2 stage has a conversion number after the first 10 testers; the biggest drop-off stage is identified (watch power-up conversion and trial exhaustion specifically).
- **Trial economics:** average trial burn per signup (expect $1–2 against the $5 cap — the entire 18-agent dogfood workspace measured **$39.09/week**, so onboarding plus days of light chat fits comfortably on the pinned tier); trial→BYOK conversion ≥40% for the hand-held pilot cohort; total platform exposure bounded by the global daily cap at all times.
- **Trust:** 100% of onboarding builds verified; zero onboarding missions created with `skip_verification`/`auto_approve` (test-enforced).
- **Wave 1 exit criterion:** one real external tester completes signup → BOOM end-to-end with no manual intervention, and the funnel recorded it.
- **Wave 2:** tool-token load per turn for a Basic-plan workspace reduced vs baseline; checklist median ≥3/5 items in week 1.

---

## 12. Open questions (Gerard's calls)

- **Q1 (blocks W2·S1/S2):** Tier definitions — names, seat counts, and which capabilities (CodeGraph, NL2SQL, team features, mission concurrency, marketplace depth…) sit in which tier. Everything else is buildable without this.
- **Q2:** Deletion timing — Shepherd + welcome modal die in Wave 2 as written; pull forward to Wave 1 (delete rather than unmount) or hold until the pilot confirms the spine?
- **Q3:** Wizard UI — delete in W2·S5 as written, or keep `/onboarding/wizard` as a power-user path behind a link?
- **Q4:** Small-setup threshold — direct tool calls vs mission. Proposal: ≤3 agents and ≤2 playbooks = direct calls; above = mission. Confirm or adjust.
- **Q5:** Plan assignment vs commerce — is Wave 2 plan-assignment-only (as written), or should checkout/billing wiring come into scope?
- **Q6:** Trial values — confirm: per-workspace cap (`TRIAL_CREDIT_USD` = $5?), global daily cap (`TRIAL_GLOBAL_DAILY_USD` = $25–50 to start?), the trial model allowlist (which economical models the platform OpenRouter key serves during trial), and the pilot conversion target (≥40%?).
- **Q7:** OnboardingSection budget — approve raising past 800 tokens if measurement requires it?
- **Q8:** Voice for the pilot — W3·S2 as written, or pull the voice lane into Wave 1 as a demo differentiator for the partner meeting?
- **Q9:** Wave-3 scheduling — Academy bridge and partner lane ordering, and when the academy-repo counterpart PRD gets written.

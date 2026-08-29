# PRD-230 — Marketplace Packages & Vertical Onboarding

**Status:** approved direction (Gerard, 2026-08-28 — the packages lightbulb, reconnecting the July TODO §10 package-led design). Overnight build. Complements PRD-222 (the spine is live; this makes the proposal concrete).

---

## 1. Overview

The first live onboarding test (a Shopify jewellery store) proved the spine works and exposed the gap: Auto doesn't know its own shop. It improvised a CSV workaround while 12 prebuilt Shopify agents, 1000+ tools, playbooks, and a direct store sync sat unused in the marketplace.

A **package** turns the marketplace into the answer: a curated, per-vertical bundle of *existing* artifacts — agents, tools, skills, playbooks, LLMs — with matching metadata and a setup manifest. Auto matches the business, proposes ONE package, installs it **workspace-owned with full dependency closure**, then guides connection and activation. "Choose the operating team you need. Auto will build it. That's the product."

## 2. Decision record (2026-08-28 — do not relitigate)

| # | Decision |
|---|---|
| D1 | **The registration invariant.** ANY marketplace artifact Auto adds is registered to the workspace (tagged with the workspace ID, always available there) — including a bare LLM pick. |
| D2 | **Dependency closure.** Installing an artifact registers its FULL dependency closure: an agent brings its tools, skills, plugins, connected-app requirements, and LLM. A 6-agent package brings all six closures. Nothing half-installed, nothing platform-dangling. |
| D3 | **Workspace-owned and editable.** Installed artifacts are the workspace's copies — "marked as theirs and they can edit." |
| D4 | **Packages are data, not code.** A package is a marketplace entity: members list + vertical/matching metadata + setup manifest. Curating a new vertical is content work, not a deploy. |
| D5 | **Shopify ships as TWO packages** — *Shopify Management* (run the store: reports, inventory, customer service) and *Shopify Development* (build/theme/dev work) — to avoid overwhelming users. Both installable, but… |
| D6 | **One package during onboarding.** The proposal offers exactly one matched package; Auto picks one sensibly if the user defers. More packages any time later via the marketplace. |
| D7 | **The three-step flow** after selection (or the manual route): ① install package / create agents → ② guided connections (apps, widgets, sync) → ③ putting your agents to work (first playbook, checklist). |
| D8 | Per-vertical depth (worked examples, multi-shot guidance) lives in the package manifest, pulled on match — never always-loaded into Auto's per-turn context. |

## 3. Current reality (grounded 2026-08-28 — re-verify at build)

- Marketplace schema exists: `20260130_create_marketplace_schema` + marketplace on agents (`20260131`), recipes/playbooks (`20260201`), LLM marketplace tables (`20260214`).
- Workspace registration patterns exist: `workspace_enabled_plugins`, `workspace_enabled_skills` (`core/models/marketplace_plugins.py:186,216`); agent-side edges: `agent_skills`, `agent_assigned_plugins` (`:243`), `agent_app_assignments` (composio, `composio_cache.py:171`); agents carry `model_config` (LLM ref).
- `marketplace-grid.tsx` takes `type`/`category` — a Packages tab is additive.
- Install tools exist for skills/plugins (`platform_install_skill/plugin`, id|slug either-of). No agent-install or package tools exist.
- PRD-222 live surfaces this builds on: proposal stage (v2 section), connect card, checklist card, funnel events, `PLAN_TIERS`/exposure.
- Two open PRD-222 fixes ride in this build's W0: chat trial metering (factory `workspace_id` no-op) and the capability doctrine.

## 4. The design

**Data.** ONE new table `marketplace_packages`: `slug, name, description, vertical_tags[], matching JSONB` (business-type signals), `members JSONB` (typed refs: `{type: agent|tool|skill|plugin|playbook|llm, id/slug}`), `setup_manifest JSONB` (setup questions, `required_connects` — Composio apps incl. the Shopify two-step (connect now; app install → Site appears → sync unlocks → knowledge graph + widgets), guide steps for D7, report templates), `showcase bool`, timestamps. Registration of installed members reuses the existing per-type patterns; only genuinely new registration surfaces (agents-to-workspace copies, workspace LLM availability) are added where none exist.

**Closure resolver.** Pure service: `closure(ref) -> typed set` walking agent→(model_config LLM, agent_skills, agent_assigned_plugins, agent_app_assignments) and playbook→agents→…; cycle-safe, deterministic order, tested against the "agent A = 3 tools + 2 skills + 1 LLM ⇒ 7 registrations" example.

**Installer.** `install_package` / `install_marketplace_agent`: resolve closure → register every member per-type (workspace-owned/editable copies where the type has copies; enabled-rows where the type has enablement) → idempotent (re-install = no dupes) → returns a manifest of everything registered. App requirements from the closure surface as *connect steps* in D7-②, never silent.

**Tools (3-file pattern).** `platform_search_packages` (match by business signals), `platform_install_package`, `platform_install_marketplace_agent`. Onboarding restriction: during onboarding, `platform_install_package` accepts one package (honest error copy on a second); unrestricted post-onboarding.

**Marketplace UI.** Packages tab + detail popup: members grouped by type with descriptions, setup summary (what gets connected, what reports you'll get), showcase row on top. Plan-label chips consistent with W2b `marketplace_depth`.

**Onboarding integration.** The proposal offers the matched package by name with its contents ("four agents, the weekly-numbers playbook, store connect — want it?"); user defers → Auto picks the D5-appropriate one (a store *owner* → Management). Then D7's three steps, with the checklist card carrying step ③ and funnel events `package_offered / package_accepted / package_installed`.

## 5. User stories (one wave + W0 fixes)

- **US-001 (W0, fix/prd-222):** chat trial metering — thread the workspace identity through the chatbot→agent-factory path so `resolve_trial_routing` and spend accrual fire on chat turns (the factory no-ops on `workspace_id=None`; chat is the main surface and currently unmetered/unpinned). Regression test: a chat turn on a trial workspace accrues spend and pins to the allowlist; BYOK untouched.
- **US-002 (W0, fix/prd-222):** onboarding doctrine v2 in the section — the capability map (Composio connect via the chat card, Shopify first-class; the Settings sync two-step truth; URL ⇒ call `platform_scan_business_site` immediately; marketplace-first staffing; honest widget line: "no CSVs — we sync directly, and our Shopify package includes widgets and agents"); the Basic-plan comms line ("you're on Basic while we set up — we'll pick your plan together shortly"); exact stage names for `platform_update_onboarding` (the "snag" hardening). Budget re-measured.
- **US-003:** `marketplace_packages` table — the wave's ONE migration — + model + CRUD service.
- **US-004:** dependency-closure resolver (pure, tested, cycle-safe).
- **US-005:** workspace registration installer honoring D1/D2/D3, idempotent, manifest-returning; new registration surfaces only where no pattern exists.
- **US-006:** the three platform tools, walker-clean, onboarding one-package restriction with honest copy.
- **US-007:** Packages tab + detail popup + showcase row, depth-consistent chips.
- **US-008:** seed **Shopify Management** + **Shopify Development** packages curated from the existing 12 Shopify agents/tools/playbooks (data seed, workspace-editable after install, both showcased; curation is v1 — Gerard tunes content later).
- **US-009:** onboarding integration — proposal offer, defer-pick, D7 three-step guide wired through section + checklist, funnel events.
- **US-010:** invariant guards — tests proving: full-closure registration (the agent-A example), workspace-owned/editable marking on every installed artifact, one-package-during-onboarding enforcement, idempotent re-install.

## 6. Functional requirements

- FR-1: D1/D2/D3 hold for every install path (package, single agent, bare LLM) — enforced and tested, never best-effort.
- FR-2: Packages are creatable/curatable as data; no deploy required to add a vertical.
- FR-3: During onboarding exactly one package installs; the marketplace imposes no such limit afterward.
- FR-4: Closure-derived app requirements surface as guided connect steps; nothing connects silently.
- FR-5: All package funnel moments are first-class events.
- FR-6: Chat turns meter the trial (W0) — no unmetered platform spend on the primary surface.

## 7. Non-goals

- No package pricing/billing (bundles of existing artifacts; commerce stays Q5).
- No new verticals beyond the two Shopify packages (curation follows once the machinery is proven).
- No auto-connect of apps; guided, explicit, per D7-②.
- No general web-search capability (separate decision, Composio-catalog route).

## 8. Technical considerations & traps

- **Reuse the registration patterns** (`workspace_enabled_*`, install rows, content-hash on skills); add new surfaces only where a type has none (agents-to-workspace, LLM availability). Two parallel registration mechanisms for one type = drift.
- **ONE migration** (US-003). Everything else rides existing tables/JSONB.
- **Route manifest**: any new API route = hand-add + count bump. Prefer tools over new routes.
- **Public repo**: package seeds contain no customer data; curation text is generic.
- **Prod-schema drift bites migrations** (the `workspaces_plan_check` incident — 2026-08-28): backfills touching constrained columns must tolerate legacy constraints (`DROP CONSTRAINT IF EXISTS` where a named legacy constraint is known, and prefer additive DDL).
- **Section token budget**: US-002/US-009 re-measure; package depth pulls via tools, never inlines.

## 9. Success metrics

- A fresh Shopify-business onboarding reaches an installed package with connected store data in one conversation, no CSV improvisation.
- The agent-A invariant test passes; zero platform-dangling artifacts after install.
- Trial pill moves during a chat-driven onboarding (W0 fix observable).

## 10. Resolved questions (Gerard, 2026-08-28 late)

- **Q1 → D9:** ALL tiers see and can use ALL packages — nothing in the marketplace is hidden or unusable to anyone. Tiers gate PLATFORM functionality (team features, agent counts, NL2SQL, advanced features), never marketplace access. Consequence: when an install would exceed the tier's agent quota, the install is NOT blocked silently and NOT partially performed — the user gets the honest plan conversation (the proposal already pairs package + plan recommendation during onboarding; post-onboarding the tool returns the same honest copy + recommendation). Closure is never partially installed.
- **Q2 → D10:** No-match fallback is **Auto custom-designing the workspace** — "we can't preempt the world"; Auto has the tools, the doctrine, and marketplace-first for individual artifacts. A small universal "Essentials" base package (web search, common basics) is PARKED until the web-search capability decision lands; not built tonight.
- **Q3 → D11:** The entity and the tab are **"Packages"** (avoids collision with human team features). Warm copy lives in content: the showcase row reads "Starter teams for your business.".

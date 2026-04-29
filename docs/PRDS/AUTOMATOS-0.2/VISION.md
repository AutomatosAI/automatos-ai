# Automatos AI — Vision & Architecture

**Status:** Canonical. This document is the anchor. Every PRD, migration plan, marketing claim, and design decision derives from here. If this document and any other document conflict, this document wins.

**Last updated:** 2026-04-25

---

## 1. What Automatos Is

Automatos is an **operating system for AI operations**.

It is the control plane where a single operator (the user) commands a workforce of specialised AI agents, equipped with tools and skills, orchestrated through a single voice (Auto), to produce real business deliverables.

### Reference class

| Platform | Is to | As Automatos is to |
|---|---|---|
| Shopify | E-commerce | AI operations |
| n8n / Zapier / Make | Workflow integrations | AI agent orchestration |
| Linux / macOS | Hardware | AI agents + their tools |
| AWS | Raw compute | AI workforces |

We give you the **primitives and the shell**. You connect your own world to it. The product is the operating system, not the applications that run on it.

### What Automatos is not

- **Not a CRM.** You plug your CRM in.
- **Not a document store.** You plug your Dropbox / Google Drive / Notion in. The built-in storage is a starter pack, not a destination.
- **Not an email client.** You plug Gmail / Outlook in.
- **Not a vertical SaaS.** Shopify, real estate, recruitment, fitness — these are *validation* that the LEGO works, not *target markets* we build dedicated apps for.
- **Not an everything-in-one.** We refuse to compete with Notion on docs, with HubSpot on CRM, with Canva on design. We orchestrate them.

### The moat

**Positioning.** An OS layer is defensible. Vertical SaaS is a race against better-funded competitors. By being infrastructure, we win by being *underneath* — the layer above (user's tools, user's workflows, user's verticals) depends on us without threatening us.

**Knowledge Graph.** Most AI platforms have RAG. We have a real graph — code, documents, databases, memory, deliverables, all feeding one unified Knowledge Graph that Auto queries for reasoning. This is the single most differentiating technical asset.

**Community templates.** Like n8n's workflow library. Users design a Playbook / Agent / Mission, publish to Marketplace, everyone benefits, the original author gets credit. The Marketplace is simultaneously app store, community surface, and GTM engine.

---

## 2. Auto — The Character

Auto is not a chat interface. Auto is a **character** — a named entity with configurable personality, voice, proactiveness, harness, and heartbeat cadence. Users bond with Auto the way they'd bond with a trusted executive assistant or chief of staff.

### Role

Auto is the **gatekeeper and orchestrator**. He:

- Holds the universal router and discovery tools
- Directly operates the platform itself (platform, workspace, and system tools are native to him)
- **Delegates** everything else — Composio-style external tools and vertical-specific work go to specialist agents who own those skills and tools
- Speaks in one voice. When Sentinel reports, Auto relays. Drill-down breadcrumbs reveal who actually ran each tool call (`↳ Search agent · websearch · 1.2s`)

### Configurability

Auto is configured, not fixed. Every user can tune:

| Dimension | Values |
|---|---|
| **Persona** | Formal Exec · Pragmatic Ops · Coach · Casual Sidekick · Personal Assistant |
| **Voice** | Per-persona voice profiles (tone, formality, humor) |
| **Proactiveness** | Reactive · Briefing · Proactive · Autonomous |
| **Heartbeat** | Cadence and content of scheduled check-ins |
| **Harness** | Model selection, retry policy, cost controls, prompt management |

This is why the same Auto can be CEO-voice for a Shopify operator and Casual-Sidekick for someone asking him to book a dinner. **One product, many faces — driven by settings, not code forks.**

### Auto as a layer, not a page

Auto is **always present**. He is the product's spine.

- **Chat page** — Auto, full-screen. Focused conversation. No distractions.
- **Every other page** — Auto as a widget docked in the corner. Page-aware. `[Context: User is on the Agent Management page]` is surfaced visibly in the widget — a trust receipt, so the user never wonders whether Auto knows what they're looking at.

### Widget behaviour

- **Ephemeral by default.** Widget chats reset on page navigation. No "history graveyard" to curate.
- **Promote to persist.** "Open in full chat" escalates the widget thread into the full Chat page, where it becomes a permanent thread.
- **Bypass is legitimate.** Power users `@mention` a specific agent to skip Auto's routing — when you know the target, skipping the orchestrator saves tokens.

### Bug report surface (pilot only)

While in beta, the widget exposes a **Report Bug** tab. User fills a minimal form (title, description, severity, category, screenshot). Hidden metadata is auto-attached (route, workspace/user ID, console errors, git SHAs, last widget messages).

The bug report files a Jira ticket. A webhook fires a Playbook. **Patcher agent** (with dev, git, GitHub, and git platform skills) clones, branches, attempts a fix, pushes, and opens a PR. Pilot users become repo contributors without realising it.

**This feature is feature-flagged and removed at GA.**

---

## 3. The Work Loop

The daily user journey — the reason the architecture exists in its shape.

```
  Chat ──→ Agents ──→ Assignments ──→ Deliverables ──→ [review & iterate]
   │         │             │                │                  │
   │         │             │                │                  │
   └─────────┴─────────────┴────────────────┴──────────────────┘
                                │
                         Command Center
                      (observe it happening)
```

A day in the life:

1. **Morning** — Auto greets. User opens Command Center. Pulse check.
2. **Chat / plan** — User asks Auto to prep the week's Instagram content. Conversation crystallises into an Assignment.
3. **Assign** — Auto drafts a Playbook, confirms, routes it to `MARKETER`. Assignment created, scheduled.
4. **Execute** — Agents run. Results stream into Command Center (Feed, Board). Auto summarises mid-day.
5. **Receive** — Deliverables land in Deliverables page. User previews, approves, rejects, re-assigns.
6. **Iterate** — Rejected Deliverable → new Chat → refined Assignment → better Deliverable.
7. **Evening** — Auto posts a wrap-up. Analytics updates.

### Why this loop shape matters

The historic pain point that triggered this vision was **scattered deliverables**. Outputs were split across the workspace page, reports were on Agents, file artifacts elsewhere. Even agents didn't know where to look for each other's work. This broke the loop.

**Deliverables is now the canonical output destination.** Every agent writes here. Every user reviews here. One noun, one surface, one source of truth.

---

## 4. The Eleven-Page Architecture

Automatos surfaces are arranged in three clusters: **Work** (daily loop), **Admin & Review** (weekly / setup), **System** (one-time plumbing).

### Menu order

```
─── Work ───
1. Chat
2. Agents
3. Assignments
4. Deliverables
5. Command Center

─── Admin & Review ───
6. Tools
7. Knowledge Bases
8. Analytics
9. Teams
10. Marketplace

─── System ───
11. Settings
```

Visual dividers separate the clusters. Reads top-to-bottom as a narrative: *talk to Auto → meet my crew → give them work → see what they made → watch the pulse → equip them → feed them knowledge → measure them → share with my team → extend via community → configure the OS.*

### Page specifications

#### 1. Chat

Auto's full-screen canvas. The terminal of Automatos.

**Key elements:**
- Composer with `To: Auto` chip (switchable to any agent — sticky per session)
- Inline shortcuts: **`</>` (code/workspace explorer)**, **Plan** (atom-to-molecule planning), and user-pinned favourites
- Model selector (quiet dropdown, not a card)
- Conversation thread per promoted session
- Delegation breadcrumbs inline

**Chat has no tabs.** It is one surface.

#### 2. Agents

The crew.

**Tabs:** `Roster` · `Org Chart` · `Skills`
**Hidden (feature-flagged):** `Coordination` (teams, collaborative reasoning — returns when ready)
**Removed:** `Configuration` (redundant — slide-over edit from Roster is canonical)

**Skills tab on Agents:** user-local skills library. View installed, fork, edit locally. No global publish yet — trust-gated. Import from Marketplace.

**Playbooks are not here.** They move to Assignments.

#### 3. Assignments

The work hub. This is the page still being designed in detail.

**Unifies:** Tasks · Playbooks · Missions · Schedules
**Canonical noun:** **Playbook** (not Recipe — do not revert).
**Atom-to-molecule pattern:**

```
Chat     →  Task      →  Playbook   →  Mission
atom        one-shot     reusable      orchestrated
                         molecule      multi-agent
```

Users do not pick "Mission" up front. They chat. Auto notices shape crystallising and offers escalation ("Want me to save this as a Playbook?" / "You've run this three times — schedule it?").

**The single largest undesigned surface.** Next PRD should address it.

#### 4. Deliverables

What agents produced. The output destination. Formerly scattered; now canonical.

**Tabs:** `Outputs` · `Templates` · `Explorer`

- **Outputs** — produced files (PDFs, DOCXs, XLSXs, images). Preview any file type inline (HTML, markdown, PDF, images, spreadsheets, docs). Download, share, re-assign.
- **Templates** — output definitions. HTML/CSS + **JSON data schema** (the contract between agent and template). Agents fill schemas; templates render. Brand primitives (logo, colours, company details) come from **Settings → Identity** and are interpolated via `{{company.logo}}` etc.
- **Explorer** — power-user view. File system + terminal + universal preview. Cursor/Claude-Code-lite. For users who want to work directly with artifacts.

**Why Templates live here, not in Knowledge Bases:** Templates are outputs' definitions. Knowledge Bases is for *inputs* (what we know). Templates belong with Deliverables because they define what Deliverables look like.

#### 5. Command Center

Pulse. Eyes on everything. Formerly "Activity."

**Tabs:** `Summary` · `Board` · `Calendar` · `Feed`

- **Summary** — 12+ configurable widgets. "How's my business today?" Status Overview · Schedule · Agent Reports · Types of Work · vertical-specific widgets (Shopify orders, Instagram likes, etc.). **Vertical presets required** — no single layout fits all personas.
- **Board** — Kanban. Tasks + Playbook-tasks + Mission-tasks. Six stages, colour-coded by parent. Drag to assign. Agents pick up tasks on heartbeat wake.
- **Calendar** — time-axis view. What's scheduled today/week/month. Agent heartbeats, scheduled Assignments, Auto's own cadence.
- **Feed** — chronological event stream. Everything that happened: task state changes, heartbeats fired, memory events, cost thresholds, bug reports, Patcher PRs, mission escalations, report submissions. Filter chips: `All · Tasks · Agents · Memory · Cost · System`.

**Feed vs Board:** Board is state (where work *is*). Feed is history (what *happened*). They overlap on one slice; Feed catches everything Board doesn't.

**Analytics overlap rule:** Summary widgets are **live pulse**. Analytics has **historical trends**. Both can show LLM spend — one "today, live," the other "30-day curve." Widgets deep-link to Analytics via "↗ open in Analytics."

#### 6. Tools

Workspace integrations. One page, one view.

**Content:** grid of connected apps (Shopify, Notion, Dropbox, etc.), each with tool count, trigger count, connect/disconnect state, feature toggles.

**Tools stays its own page** — not merged into Agents. Has its own lifecycle (OAuth, per-feature toggles) that warrants room. "Don't re-architect what works."

**Triggers discovered here, configured in Settings → Channels.** Discovery vs. operation split. The KPI card "98 Triggers" deep-links to Settings to activate.

#### 7. Knowledge Bases

Inputs. The datacentre, filing room, library.

**Tabs:** `Documents` · `Databases` · `CodeGraph` · `Knowledge Graph` · `Memory`

- **Documents** — RAG. Uploaded + synced from storage tools (Google Drive, Dropbox, Notion, Automatos Storage). Source filter chip. Library table with RAG query counts, last-accessed, tags (merged from former Analytics > Documents). Upload button top-right. Search bar does vector search + RAG inline. Sub-tabs collapsed: Processing = filter, Multimodal = filter, Upload = button, Search + RAG Test = unified.
- **Databases** — NL2SQL connections. Schema browse, query playground.
- **CodeGraph** — indexed repos, semantic search, graph viz. Patcher queries this for fixes.
- **Knowledge Graph** — the unified graph. Code + documents + databases + memory + deliverables all flow in. Auto's second brain. *The differentiator.*
- **Memory** — agent memory audit + prune. Short-term + long-term. Per-agent, per-workspace. Privacy-gated — UI reinforces agent/workspace scoping explicitly.

**Name decision:** "Knowledge Graph" (industry standard), not "Business Graph."

#### 8. Analytics

History. Trends. Reporting.

Single page, existing tabs (Overview · Agents · Missions · LLM & Costs · Tools & Integrations · Admin).

**Documents tab moved out** to Knowledge Bases > Documents (where it belongs with the data).

#### 9. Teams

Human team members (distinct from Agents, who are AI).

Business-plan workspaces can invite humans and assign roles. **Settings visibility is role-gated** — team members can be locked out of Settings entirely.

#### 10. Marketplace

Community. App store. GTM.

- Browse and install: Agents · Playbooks · Skills · Tools · Widget templates · Command Center presets
- Users publish their own creations. Attribution to the author.
- The on-ramp for new users — copy an existing Playbook, customise, go.
- Trust architecture: curated-first, self-author next, open-publish gated.

**Stays as is** — existing implementation is not being redesigned in this vision.

#### 11. Settings

System plumbing. Role-gated.

**Workspace admin sees:** `Orchestrator · Webhooks · API Keys · Credentials · Channels · Notifications · Voices · Widget SDK`

Cluster mentally as:

| Group | Sub-pages |
|---|---|
| **Engine** | Orchestrator |
| **Plumbing** | Webhooks · API Keys · Credentials |
| **Touchpoints** | Channels · Notifications · Voices · Widget SDK |

**Automatos admin (system owner) additionally sees** system-level configuration invisible to workspace admins.

**Identity / Branding** lives here too — logo, colours, company name, address, tax ID. Templates on Deliverables consume these as `{{company.*}}` primitives.

---

## 5. Pilot Personas as Validation

Six pilot personas signed up, deliberately diverse:

| Persona | Domain | Primary agents | Primary pages |
|---|---|---|---|
| Shopify power user | E-commerce · growth | SHOPIFY · MARKETER · COMMS · SCOUT | Command Center · Deliverables |
| Personal assistant | Life planning | COMMS · PLANNER · SCOUT · COACH | Chat · Command Center |
| Real estate | Lead gen · research | SCOUT · COMMS · MARKETER · PLANNER | Assignments · Deliverables |
| Fitness trainer | Lead gen · planning | TRAINER · COMMS · MARKETER · SCOUT | Assignments · Deliverables |
| Small business | General ops | COMMS · MARKETER · vertical specialist | Chat · Assignments |
| Recruitment | CV · research · matching | SCREENER · SCOUT · COMMS · MARKETER | Assignments · Knowledge Graph |

**They are not six products we build.** They are six validations that the LEGO is flexible enough. Every persona plugs in their own tools (Shopify, MLS, ATS, Stripe, Gmail) and runs their world from Automatos.

### Common patterns across all six

1. Everyone needs the same core crew: COMMS + SCOUT + MARKETER + one vertical specialist.
2. Assignments dominate over Chat after week 1. Recurring Playbooks run the business.
3. Deliverables are always PDF / DOCX / XLSX. Templates are load-bearing.
4. Command Center wants vertical presets — no single dashboard fits.
5. Auto's personality dial is real product — CEO-voice for Shopify, Sidekick for personal, Coach for trainer.
6. Knowledge Graph matters most to 3 of 6 (recruitment, real estate, personal assistant) — graph-native data.

### Starter pack (future, not priority)

A one-click vertical pack at signup: pre-configured crew + tools + skills + templates + Command Center layout + Auto personality + starter Assignments. Shopify Template is the first. **But this is a demo, not priority #1.** Priority #1 is the OS foundation.

---

## 6. Canonical Terminology

Fixed terms. Do not drift.

| Use | Do not use |
|---|---|
| **Playbook** | ~~Recipe~~ — legacy name, renamed |
| **Deliverable** | ~~Output~~ · ~~Workspace file~~ · ~~Artifact~~ — in user-facing copy |
| **Agents** | ~~Bots~~ · ~~Assistants~~ |
| **Knowledge Graph** | ~~Business Graph~~ — "Knowledge Graph" is industry standard |
| **Assignments** | ~~Goals~~ — rejected; "goals" is aspirational, not directive |
| **Command Center** | ~~Activity~~ — old name, renamed |
| **Crew** (aspirational) | Currently "Agents" page name; "Crew" may be adopted if Tools + Skills consolidate there later |
| **Auto** | Capitalised. Proper noun. Auto is a character, not "the assistant" |

---

## 7. Non-Goals

Explicit list of things Automatos will not build:

- **Document creation suite.** Use your existing tools; Automatos orchestrates.
- **Full CRM.** Plug your CRM in.
- **Own email client.** Integrations only.
- **Own calendar.** Integrations only.
- **Vertical-specific apps that duplicate BYO tools.** (No "Automatos Shopify Store." We *orchestrate* Shopify, not replace it.)
- **All-in-one chat.** We are not Slack. Channels integrate; we don't replace them.

---

## 8. Evolution

Ordered by dependency. Do not jump phases.

### Phase 1 — Foundation (current)

- Eleven-page architecture clean and consistent
- Auto layer (widget + full chat) rock solid
- Deliverables as canonical output destination
- Knowledge Graph primitives (graphify — PRD-135 in progress)
- Dead-code purge via PRD-135 reports (dead tables, dead routes, consolidation candidates)
- Canonical terminology migrated

### Phase 2 — Assignments

- Design and build the Assignments page (the largest remaining surface)
- Chat → Task → Playbook → Mission escalation flow
- Scheduled Assignments (cron UI done humanely)
- Assignment run history

### Phase 3 — Knowledge Graph, properly

- Unified graph ingestion pipeline (code + docs + DB + memory + deliverables)
- Graph visualisation that's actually good
- Auto queries the graph for reasoning; receipts surface which graph facts were used

### Phase 4 — Education content

- Documentation videos
- Example Playbooks in Marketplace
- Walkthroughs for each pilot persona
- This is the *actual* GTM engine — n8n's template library is worth more than n8n's code

### Phase 5 — Vertical packs

- Shopify Template first (already in plan)
- Real estate, recruitment, fitness next
- One-click pack install at signup
- **Explicitly deferred until Phases 1-4 are sturdy.**

### Phase 6 — Self-serve skills publishing

- Open the "Publish to Marketplace" gate
- Moderation + security review pipeline
- Revenue share for skill authors

---

## 9. Design Principles

Rules the team applies when resolving ambiguity.

1. **Don't re-architect what works.** Tool inventory is one page because that's what it is. Symmetry is not a design goal.
2. **Configuration beats product surface.** If a feature can be a setting, it's a setting. Auto's personality is a dial, not a fork.
3. **Receipts over magic.** When Auto delegates, show who. When the widget has context, show what. Trust is earned by transparency, not by hiding the seams.
4. **Promote on use, not on design.** Chat → Task → Playbook → Mission escalates when usage warrants it, not because the UI asks users to categorise up-front.
5. **One noun per concept.** Playbook not Recipe. Deliverable not Output. Every renaming drift costs the user.
6. **Education scales; apps don't.** Every hour spent on a video is worth ten on a vertical-specific feature. We're an OS; users must learn to use it.
7. **Solo build discipline.** The product is built by one person. Every feature has to pay its maintenance cost. Half-built features are worse than no features.

---

## 10. What triggered this document

**Deliverables was scattered.** Outputs lived across the workspace page, reports were on Agents, file artifacts nowhere obvious. Even agents didn't know where to post their work. That broke the daily loop: Chat → Assign → **Receive**. The Receive stage was foggy, so the whole loop lost trust.

Consolidating Deliverables into one canonical surface — with Templates as sibling and Explorer as power-user extension — forced a rethink of every adjacent page. That rethink produced this document.

---

**End of VISION.md.** Derivative documents (migration plans, per-page PRDs, marketing copy) must be consistent with this. Conflicts are resolved in favour of this document; drift is corrected here, not forked.

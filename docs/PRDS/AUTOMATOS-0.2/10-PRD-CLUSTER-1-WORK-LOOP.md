# PRD — Cluster 1: Work Loop Redesign

**Status:** Draft
**Date:** 2026-04-25
**Author:** Gerard Kavanagh + Claude (Opus 4)
**Anchor:** [VISION.md](./VISION.md) §3 (Work Loop), §4 (pages 1–5)
**Cluster:** 1 of 3 (Work · Crew & Knowledge · Business)
**Pages affected:** Chat · Assignments · Deliverables · Command Center

---

## 1. Why this PRD exists

Pilot users said it directly: **"this is too complicated for me, I'm confused."**

The work loop — the daily reason people open Automatos — is currently spread across pages with overlapping responsibilities:

- **Assignments** is doing five jobs (create tasks, build playbooks, launch missions, schedule recurring work, view run history) and no name fits all of them.
- **Deliverables** doesn't lead with what's new — users have to hunt for "what did my crew make for me today?"
- **Command Center** has the right tabs but isn't the home for run history, which lives confusingly inside Assignments.
- **Chat** is missing a structured "Plan" mode — non-technical users go straight from idea to expensive Mission with no rehearsal.

This PRD does **not** add features. It re-shuffles existing surfaces around one rule: **one purpose per page, leading with the action a user actually does there.**

The IA template is the existing **Marketplace** page (featured hero + secondary stack + tabs + recommended grid). It tested well with non-technical users. We lift that pattern for Assignments and Deliverables.

---

## 2. North-star outcome

A non-technical pilot user, on day one, can:

1. Open Chat → describe a goal → Auto enters Plan mode → produce a runnable Playbook **without picking "playbook" or "mission" up front**.
2. See the Playbook listed under Assignments → Playbooks tab.
3. Run it. Watch progress on Command Center → Board.
4. Receive the output on Deliverables, with the new artifact featured under "Created today."
5. Iterate (reject, refine in Chat, re-run).

If a pilot user can complete that flow without help, this PRD is done.

---

## 3. Scope

### In scope

| # | Page | Change |
|---|---|---|
| 3.1 | **Chat** | Add **Plan mode** — Chat with structured plan-panel side-by-side. Output of Plan is a draft Playbook or Mission. Thread switcher (dropdown) for concurrent conversations. |
| 3.2 | **Assignments** | Marketplace-pattern redesign. Featured area with 4 create-cards (Mission hero + Playbook + Plan + Task). Below: tabs `[Playbooks] [Missions]`. Contextual hero copy based on workspace state. **Tasks do not list here** — they live on Command Center → Board. Run history removed from this page. |
| 3.3 | **Deliverables** | "Created today" featured area at top. Tabs: `[Outputs] [Blogs] [Templates] [Explorer]`. Universal preview for all file types. Blogs tab shows draft/publish toggle and SEO metadata (title, cover URL, tags). Explorer expands to **viewport-full** when opened (mode, not tab). |
| 3.4 | **Command Center** | Add **History** tab (completed runs ledger, filterable by playbook/agent/date/status). Existing tabs unchanged: `[Summary] [Board] [Calendar] [Feed] [History]`. |

### Out of scope

- Backend changes to Mission/Playbook/Task execution engines (already built).
- New agent capabilities, new tools, new skills.
- Marketplace redesign (already in the desired state — see VISION §4.10).
- Analytics, Knowledge Bases, Teams, Settings (Cluster 2 and 3).
- Vertical preset library for Command Center widgets (separate PRD).
- Self-serve skill publishing (Phase 6 in VISION).

### Explicit non-goal

This PRD must not introduce new database tables. Every concept (Playbook, Mission, Task, Deliverable, Run) already has storage. Re-shuffles only.

---

## 4. Page-by-page IA

### 4.1 Chat

```
┌────────────────────────────────────────────────────────────┐
│ [Threads ▾]                              [Auto · ▾]        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│         conversation thread                                 │
│         (delegation breadcrumbs inline)                     │
│                                                             │
├────────────────────────────────────────────────────────────┤
│ [</>]  [Plan]  [⭐ pinned shortcuts]                        │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ To: Auto ▾                                             │ │
│ │ Type a message...                                      │ │
│ └────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

**Threads dropdown:** lists active concurrent conversations (e.g. "Canvas — Q2 social images", "Comms — inbox triage"). Each thread is a separate Auto session. User switches; threads persist. No left sidebar.

**Plan mode:** clicking `[Plan]` (or arriving via `+ Plan` from Assignments) puts the chat in Plan mode. Visual cue: header changes to "Planning with Auto" + a structured plan panel slides in from the right showing live plan state (steps, agents, tools, est. cost, schedule). User can iterate the plan in chat; Auto updates the panel. When the user says "looks good," panel offers `[Save as Playbook]` or `[Launch as Mission]`. Auto recommends one based on plan shape (1 agent linear → Playbook; 6+ agents parallel → Mission).

**`@agent` bypass:** typing `@MARKETER` switches the `To:` chip away from Auto. Skips orchestrator routing. Sticky per session.

### 4.2 Assignments

```
┌────────────────────────────────────────────────────────────┐
│ Assignments                                                 │
│ Plan, schedule, and orchestrate work for your crew          │
├────────────────────────────────────────────────────────────┤
│ Featured                                                    │
│ ┌──────────────────────┐ ┌────────────────┐                │
│ │  + Mission           │ │  + Playbook    │                │
│ │  Big complex work.   │ │  Routines.     │                │
│ │  6–9 agents,         │ └────────────────┘                │
│ │  field memory,       │ ┌────────────────┐                │
│ │  parallel reasoning. │ │  + Plan        │                │
│ │                      │ │  Iterate with  │                │
│ │  [Start →]           │ │  Auto first.   │                │
│ └──────────────────────┘ └────────────────┘                │
│                          ┌────────────────┐                │
│                          │  + Task        │                │
│                          │  Quick single  │                │
│                          │  action.       │                │
│                          └────────────────┘                │
│                                                             │
│ Hero hint (rotates by workspace state):                     │
│ "Not sure where to start? Plan it with Auto."               │
├────────────────────────────────────────────────────────────┤
│ [Playbooks]  [Missions]                                     │
│                                                             │
│ Recommended for you  (workspace + marketplace blend)        │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                │
│                                                             │
│ Mine · Workspace · Imported  [filter]                       │
│ card grid...                                                │
└────────────────────────────────────────────────────────────┘
```

**Mission is the hero card.** Largest, top-left, most descriptive copy.

**Plan card:** clicking opens Chat in Plan mode (deep-link `/chat?mode=plan&from=assignments`).

**Task card:** clicking opens a lightweight modal — pick agent, write a one-line instruction, submit. Task is created and routed to Command Center → Board (TODO column). Does not appear on this page.

**Contextual hero hint:** copy rotates based on user state.

| Workspace state | Hint copy |
|---|---|
| 0 playbooks, 0 missions | "Not sure where to start? Plan it with Auto." |
| 1+ playbook, 0 missions | "Ready for something bigger? Try a Mission." |
| 1+ mission, 1+ playbook | "What's next? Browse the Marketplace." |
| Recent failed run | "Something not working? Re-plan it with Auto." |

**Tabs `[Playbooks]` `[Missions]`:** card grid below. Each card shows name, owning agent(s), run count, last run, rating. Card actions: `Run`, `Edit`, `Schedule`, `Share`. Empty state for each tab links to Marketplace and Plan.

### 4.3 Deliverables

```
┌────────────────────────────────────────────────────────────┐
│ Deliverables                                                │
│ Everything your crew has produced                           │
├────────────────────────────────────────────────────────────┤
│ Created today                                               │
│ ┌──────────────────┐ ┌──────────────────┐                  │
│ │ Q2 Marketing Plan│ │ Lead enrichment  │                  │
│ │ PDF · MARKETER   │ │ XLSX · SCOUT     │                  │
│ │ [preview thumb]  │ │ [preview thumb]  │                  │
│ │ "Auto: Marketing │ │ "Auto: Scout     │                  │
│ │  agent finished  │ │  enriched 47     │                  │
│ │  your Q2 plan."  │ │  leads."         │                  │
│ └──────────────────┘ └──────────────────┘                  │
├────────────────────────────────────────────────────────────┤
│ [Outputs]  [Blogs]  [Templates]  [Explorer ↗]               │
│                                                             │
│ Recent (Outputs default)                                    │
│ Filter: All · PDF · DOCX · XLSX · Image · Markdown          │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                │
│ │file│ │file│ │file│ │file│                                │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Created today:** auto-populated from deliverables with `created_at >= today_00:00`. Rolls over at midnight workspace TZ. Empty state: "Your crew hasn't shipped anything yet today. [Browse recent ↓]"

**Outputs tab (default):** universal preview. Click any file → slide-over with preview, metadata, download, share, re-assign. Filter chips by file type. Default sort: newest first.

**Blogs tab:** card grid of agent-generated blogs. Each card shows title, cover image (from `cover_url` metadata), agent author, status badge (`Draft` / `Published`), publish date, tags. Click → blog editor view with: rendered preview, SEO metadata panel (title, slug, cover URL, tags, meta description), draft/publish toggle. Toggle gated by user role.

**Templates tab:** existing templates surface from VISION §4.4 (HTML/CSS + JSON schema definitions, brand primitives interpolated from Settings → Identity).

**Explorer tab:** clicking expands to **viewport-full** (file tree + preview + terminal, like Cursor/VS Code). Page chrome (sidebar, header) remains; main content area takes 100% width. ESC or back button returns to Deliverables tabbed layout. Pro-user mode.

### 4.4 Command Center

```
Tabs:  [Summary]  [Board]  [Calendar]  [Feed]  [History ←NEW]
```

**History tab:** ledger of completed runs (Playbooks, Missions, Tasks, scheduled cron firings). Sortable / filterable by:
- Source (Playbook / Mission / Task / Schedule)
- Agent(s) involved
- Date range (today, week, month, custom)
- Status (success, failed, partial, cancelled)

Row click → slide-over with full run detail: tasks executed, deliverables produced, cost, duration, errors. Same slide-over as live Active runs (see Board) so users see one consistent run-detail surface.

Other Command Center tabs unchanged in this PRD.

---

## 5. Plan mode — detailed design

### 5.1 Trigger paths

1. Chat composer → `[Plan]` shortcut.
2. Assignments featured → `+ Plan` card.
3. From an empty Playbooks tab → "Build your first Playbook with Auto" CTA.

All three deep-link to `/chat?mode=plan&from=<source>`.

### 5.2 Sequence

```
User  →  Auto:    "I want to research social post ideas, generate
                   images via Canva, write captions, and post to
                   Instagram + LinkedIn weekly."

Auto  →  User:    Plan panel populates:
                  Steps:
                    1. Research trending topics (SCOUT)
                    2. Draft 5 post concepts (MARKETER)
                    3. Generate images via Canva (CANVAS)
                    4. Write captions (MARKETER)
                    5. Schedule posts (COMMS)
                  Tools: Canva · Instagram · LinkedIn
                  Schedule: weekly Mondays 9am
                  Est. cost: $0.42/run
                  Detected shape: Playbook (1-pass linear, 4 agents)

User  →  Auto:    "Skip step 4, write captions inside step 3."

Auto  →  User:    Plan panel updates. Cost re-estimated.

User  →  Auto:    "Looks good."

Auto  →  User:    [Save as Playbook]  [Launch as Mission]
                  (recommendation highlighted)

User  →           Clicks Save as Playbook.
                  → Playbook persisted (existing playbook table).
                  → Redirect to Assignments → Playbooks tab.
                  → Toast: "Saved. Run it now or schedule it?"
```

### 5.3 Plan panel data shape

Plan state lives in a transient session object until saved. On save it converts to the existing Playbook or Mission record. No new table.

```ts
type PlanDraft = {
  id: string;            // session-scoped UUID
  goal: string;          // user's stated goal
  steps: PlanStep[];
  agents: AgentRef[];    // derived from steps
  tools: ToolRef[];      // derived from steps
  schedule?: CronExpr;
  estimatedCostUsd: number;
  detectedShape: "playbook" | "mission" | "ambiguous";
  status: "drafting" | "ready" | "saved";
};
```

### 5.4 Recommendation rule

```
detectedShape =
  if agents.length == 1 AND steps.length <= 3            → "playbook"
  else if agents.length >= 4 AND has_parallel_steps      → "mission"
  else if agents.length >= 6 OR has_field_memory_step    → "mission"
  else                                                    → "ambiguous"
```

Ambiguous → both buttons offered with neutral framing.

---

## 6. Migration notes

### 6.1 What breaks

- **Assignments page route** — IA changes substantially. Old run-history UI moves out.
- **Activity → Command Center** — already done (legacy redirect should remain for one wave then drop).
- **Blog tab** — already moved from Activity to Workspace per `d9941e36d`. This PRD confirms its home as Deliverables → Blogs and adds the draft/publish toggle.

### 6.2 What redirects

| Old route | New route |
|---|---|
| `/assignments/runs/:id` | `/command-center/history?run=:id` |
| `/activity/blog` | `/deliverables?tab=blogs` |
| `/workspace/files` | `/deliverables?tab=outputs` (legacy alias) |

### 6.3 Feature flag

Cluster 1 ships behind `ff_cluster_1_redesign`. Pilot users on; everyone else stays on old IA until success criteria met. Rollback = flag off.

---

## 7. Success criteria

Pilot validation, measured 14 days after Cluster 1 ships to flagged users:

1. **Plan mode adoption:** ≥40% of new Playbooks created in the period went through Plan mode (vs. raw form-based creation).
2. **Time-to-first-deliverable for new pilot user:** median ≤ 10 minutes from signup to first deliverable visible on Deliverables → Created today.
3. **Confusion signal:** zero pilot users file a "where do I find X" support message about Cluster 1 pages in week 2.
4. **Page-bounce rate on Assignments:** below 30% (vs. baseline — to be captured pre-flag).
5. **Blog draft-to-publish conversion:** at least one blog goes Draft → Published via the toggle in the period (proves the surface works end-to-end).

If any of 1–4 fails, flag stays off; we iterate based on the specific failure mode.

---

## 8. Out-of-scope follow-ups

These appeared during design and are deliberately punted:

- **Vertical Command Center presets** (Shopify / Real Estate / Recruitment / Personal Assistant). Separate PRD; depends on persona research.
- **Plan mode learning loop** — Auto improving plan recommendations from past run outcomes. Cluster 2 candidate.
- **Marketplace deep-link from "Recommended for you"** on Assignments. Trivial to add but wants Marketplace API parity work first.
- **Cluster 2 — Crew & Knowledge** (Agents, Tools, Knowledge Bases, Marketplace).
- **Cluster 3 — Business** (Analytics, Teams, Settings).

---

## 9. Story-level work breakdown (for Ralph conversion)

Sized for one-iteration completion. Dependencies first.

### Foundation

1. **US-001** Add feature flag `ff_cluster_1_redesign` to settings + env.
2. **US-002** Add `PlanDraft` session-scoped type + in-memory store (no DB).
3. **US-003** Add `mode=plan` query param handling in Chat page.

### Chat — Plan mode

4. **US-004** Plan panel UI shell (right-side slide-in, plan steps list, empty state).
5. **US-005** Auto system-prompt switch when `mode=plan` (planning persona).
6. **US-006** Plan recommendation rule (`detectedShape` logic).
7. **US-007** `[Save as Playbook]` action — converts PlanDraft → Playbook record.
8. **US-008** `[Launch as Mission]` action — converts PlanDraft → Mission record.
9. **US-009** Threads dropdown switcher in Chat header (concurrent threads).

### Assignments redesign

10. **US-010** New Assignments page shell — featured area + tabs + recommended grid.
11. **US-011** Mission hero card + Playbook/Plan/Task secondary cards.
12. **US-012** Contextual hero hint copy logic (4 states).
13. **US-013** Playbooks tab card grid + Run/Edit/Schedule/Share actions.
14. **US-014** Missions tab card grid + Run/Edit/Share actions.
15. **US-015** `+ Task` lightweight modal (agent picker + one-line instruction).
16. **US-016** `+ Plan` deep-link to `/chat?mode=plan&from=assignments`.
17. **US-017** Recommended-for-you grid (workspace + marketplace blend, simple algorithm v1).
18. **US-018** Remove run-history view from Assignments + add redirect.

### Deliverables redesign

19. **US-019** Deliverables page shell — Created today + tabs.
20. **US-020** Created-today query (deliverables where created_at >= today_00:00 workspace TZ).
21. **US-021** Outputs tab with file-type filter chips + universal preview slide-over.
22. **US-022** Blogs tab confirmed live (already moved) — add SEO metadata panel.
23. **US-023** Blogs draft/publish toggle (role-gated).
24. **US-024** Templates tab (lift existing templates surface here).
25. **US-025** Explorer tab — viewport-full mode with ESC/back exit.

### Command Center

26. **US-026** Add History tab to Command Center.
27. **US-027** History query + filters (source/agent/date/status).
28. **US-028** Run-detail slide-over component (shared with Board live runs).

### Cleanup + verify

29. **US-029** Legacy redirects (`/assignments/runs/:id` → History, etc.).
30. **US-030** Typecheck passes; smoke test full flow: Plan → Playbook → Run → Deliverable.

---

## 10. Open questions

These are deliberate left-unanswered until pilot feedback:

- Does the threads dropdown need persistence across sessions, or session-only? (Default: session-only; revisit if users complain.)
- Does Plan mode need a "save as draft" path, or is in-memory enough? (Default: in-memory; persist only on Save as Playbook/Mission.)
- Recommended-for-you ranking — pure recency or ML-blend? (V1 = recency + run-count weight; revisit.)

---

**End of PRD.** Conflicts with VISION.md resolved in favour of VISION; conflicts with this PRD during implementation resolved by raising in PR review, not silently drifting.

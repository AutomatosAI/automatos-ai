# AUTOMATOS 0.2 — Clean-Slate Consolidation Plan

**Status:** Design draft — source-of-truth for 0.2 cleanup wave
**Author:** Gerard Kavanagh + Claude (Opus 4)
**Date opened:** 2026-04-24
**Trigger:** Post-pilot audit — platform grew organically through PRDs 1-135, accumulated duplication, 109 tables (200+ counting migration residue), 103 API routers, 599 frontend TS files. Time to sketch the good from the bad.

---

## 0. Why this plan exists

Automatos was built PRD-by-PRD. Each PRD shipped cleanly; nobody re-indexed the platform surface between waves. The cumulative result:

- **103 API routers** mounted on one FastAPI app. Five of them say "marketplace", seven say "analytics", three say "chat", three say "memory", three say "knowledge", five say "workflow". Many are functional, several are superseded, a long tail is dead.
- **109 SQLAlchemy tables** live in models; at least **16 more** exist in alembic migrations with no ORM model. Production Postgres has additional migration residue (renamed `b_*_<date>` backup tables, prior-schema remnants).
- **Three code split axes with no ownership rule:** `orchestrator/core/`, `orchestrator/services/`, `orchestrator/modules/` — the same domain (e.g. workspaces, credentials, composio) has code in all three.
- **429 React components, 60+ `use-*-api.ts` hooks** — some v1/v2/explorer-suffixed triplets for the same concept (memory, multi-agent, board-tasks).
- **No single deliverables story.** Agent output lives in `deliverables`, `agent_reports`, `artifacts`, mission artifacts, workspace files, and chat artifacts. PRD-133b started the unification; AUTOMATOS 0.2 finishes it.
- **No single autonomous flow narrative.** Mission Zero works, Chat works, Board works, Workflows work — but they don't compose into "Automatos, the self-driving operation" the brand promises.

AUTOMATOS 0.2 is the pass where we name the domains, pick the canonical table per concept, collapse the routers, define the deliverable, and tell the autonomous story end-to-end. Nothing here is net-new feature work. Everything here is *reconciliation*.

---

## 1. North star in one paragraph

**Automatos 0.2 is a self-autonomous workspace OS.** A user states a goal (business plan, mission, recipe, or ad-hoc chat); a Coordinator decomposes it; agents with skills + tools execute sequentially or in parallel; every agent emits a persistent, graded deliverable tied to the goal; the workspace accrues a knowledge graph and skill library that makes the next goal cheaper. The UI is a single "Workspace" pane with four tabs (Goals, Deliverables, Knowledge, Agents) — everything else is implementation detail that should retreat behind those four.

Full statement in [02-NORTH-STAR.md](./02-NORTH-STAR.md).

---

## 2. Phase map

This plan is eight documents. Each is the spec for one wave of work. Waves 1-3 are rip-out / consolidation (boring but high-ROI). Waves 4-6 are UX-forward (make the north-star real). Wave 7 is instrumentation so we can prove 0.2 actually improved on 0.1.

| # | Doc | What it answers | Wave |
|---|---|---|---|
| 00 | **README** (this) | Map of the plan | — |
| 01 | [CURRENT-STATE](./01-CURRENT-STATE.md) | What do we have today? | audit |
| 02 | [NORTH-STAR](./02-NORTH-STAR.md) | What is Automatos? | vision |
| 03 | [DOMAIN-MODEL](./03-DOMAIN-MODEL.md) | What are the 10 canonical domains? | vision |
| 04 | [API-SURFACE](./04-API-SURFACE.md) | What's the target API shape? | wave 2 |
| 05 | [DATA-MODEL](./05-DATA-MODEL.md) | What's the target table list? | wave 1 |
| 06 | [FRONTEND-SURFACE](./06-FRONTEND-SURFACE.md) | What's the target frontend shape? | wave 3 |
| 07 | [DELIVERABLES-FLOW](./07-DELIVERABLES-FLOW.md) | How does autonomous work become visible? | wave 4 |
| 08 | [MIGRATION-PHASES](./08-MIGRATION-PHASES.md) | How do we ship this without breaking pilot? | plan |
| 09 | [SUCCESS-METRICS](./09-SUCCESS-METRICS.md) | How do we know 0.2 is done? | measure |

---

## 3. Waves at a glance

### Wave 1 — Data-model collapse (docs 01, 05)

Delete the 16 orphan tables. Pick canonical tables for the duplicated concepts (agent_reports vs deliverables vs artifacts; messages vs chat_messages; tasks vs board_tasks vs orchestration_tasks; workflows vs workflow_recipes vs recipes). Ship migrations behind a watch-period gate (≥7 days zero-traffic via `pg_stat_statements`) before DROP.

**Exit criteria:** ≤75 tables, zero orphan-model/zero orphan-migration tables, `/graphify db-report dead-tables` returns 0 rows.

### Wave 2 — API-surface collapse (docs 03, 04)

103 routers → ~25, organized by the 10 canonical domains. Every domain gets one router module; legacy paths get 301 redirects for one minor version then vanish.

**Exit criteria:** `orchestrator/api/` has ≤25 top-level router files, `main.py` mount block is ≤30 lines, `/graphify db-report dead-routes` returns 0 rows.

### Wave 3 — Frontend surface collapse (doc 06)

60+ `use-*-api.ts` → ~20 hooks, one per domain. Component folders reorganized around the four workspace tabs (Goals, Deliverables, Knowledge, Agents) instead of feature folders. v1/v2/explorer triplets merged.

**Exit criteria:** ≤25 `use-*-api.ts` hooks, zero `*-v1-api.ts` / `*-v2-api.ts` / `*-verified.ts` suffixed duplicates.

### Wave 4 — Deliverables unification (doc 07)

Every agent output ends up in one model (`deliverable`) with one storage path (`s3://…/workspaces/{ws}/deliverables/{id}/`) exposed by one API (`GET /workspaces/{ws}/deliverables`) and rendered by one component (`<DeliverableView />`). Chat artifacts, mission outputs, report files, generated documents all collapse into this. This is PRD-133b extended to its logical finish.

**Exit criteria:** `artifacts`, `agent_reports.file_path`, chat artifacts, mission artifacts all write through `DeliverableService`; the Deliverables tab is the only place to find any agent output.

### Wave 5 — Autonomous flow composition (doc 02, 07)

Goals tab ties chat, mission, recipe, and business-plan wizard into one input with four execution modes. Deliverables tab shows output across all four modes. Knowledge tab shows the graph + field memory. Agents tab shows roster, skills, tools. Everything below those four tabs retreats behind "Advanced / Admin".

**Exit criteria:** A new user from Mission Zero → Shopify template → first real deliverable in ≤10 minutes, never leaving the four-tab shell.

### Wave 6 — Skills & marketplace as the composable layer (doc 02)

Skills (`~/Development/automatos-skills`) + marketplace templates + agent catalog are the only way new capability enters a workspace. No more "drop a file in `orchestrator/modules/foo/`" expanding the platform. Platform = kernel; agency value = skills.

**Exit criteria:** A new vertical (e.g. Shopify, HubSpot, Salesforce) is installable as one template package with zero orchestrator code changes.

### Wave 7 — Instrumentation (PRD-135 operational) (doc 09)

`/graphify db-scan` + three reports run on CI nightly. Any PR that ships a new router, model, or hook without retiring an equivalent one surfaces in the report. Consolidation becomes structural, not PRD-driven.

**Exit criteria:** Dead-code report under 5 items at close of 0.2; weekly CI report with trend line.

---

## 4. Non-goals

- **No net-new features.** If a PRD in the 100-140 range is still open, it ships on its own timeline; 0.2 is the cleanup wave that runs alongside.
- **No rewrite.** Every consolidation is "pick canonical, migrate callers, delete duplicate". No new stacks, no language changes, no ORM swap.
- **No automated DELETE.** Every DROP and every route removal is gated on a watch period + human approval (same rule as PRD-135).
- **No pilot disruption.** All migrations are backward-compatible until the watch period closes. Legacy routes return 301 with `Deprecation: true` header, not 404.

---

## 5. How to read these docs

- **If you're planning a cleanup PR today:** open [08-MIGRATION-PHASES.md](./08-MIGRATION-PHASES.md), find the phase, ship it as a standalone PRD.
- **If you're orienting on the vision:** [02-NORTH-STAR.md](./02-NORTH-STAR.md) then [07-DELIVERABLES-FLOW.md](./07-DELIVERABLES-FLOW.md).
- **If you're arguing about a duplicate table or route:** [05-DATA-MODEL.md](./05-DATA-MODEL.md) / [04-API-SURFACE.md](./04-API-SURFACE.md) has the canonical answer with reasoning.
- **If you're auditing progress:** [01-CURRENT-STATE.md](./01-CURRENT-STATE.md) is the baseline; [09-SUCCESS-METRICS.md](./09-SUCCESS-METRICS.md) has the targets.

---

## 6. Relationship to existing PRDs

- **Extends:** PRD-131 (platform consolidation, closed), PRD-133b (outputs view), PRD-135 (graphify DB layer — 0.2 relies on this tooling).
- **Supersedes:** PRD-134 placeholder (the "post-133 DB cleanup pass" the roadmap hinted at — that work is now absorbed into AUTOMATOS 0.2 Wave 1).
- **Unblocks:** any post-pilot feature work, because the domain model + API surface become stable first.
- **Does not block:** PRDs 77, 78, 79, 120, 125 — those continue on their own timeline.

---

## 7. Approval & sequencing

AUTOMATOS 0.2 is intentionally non-atomic. Each wave ships as its own PRD. User reviews and approves per wave. Order is enforced:

```
Wave 1 (data)  ─┐
Wave 2 (api)   ─┼─→ parallel safe once Wave 1 canonical tables are picked
Wave 3 (fe)    ─┘
Wave 4 (deliverables) ─→ needs Wave 1+2 to finish
Wave 5 (autonomous UX) ─→ needs Wave 4
Wave 6 (skills kernel) ─→ parallel with Wave 5
Wave 7 (instrumentation) ─→ starts Wave 1, runs continuously
```

Waves 1-3 are mechanical and can be agent-driven. Waves 4-6 need product judgement.

---

**Next action:** read [01-CURRENT-STATE.md](./01-CURRENT-STATE.md) for the baseline audit, then [02-NORTH-STAR.md](./02-NORTH-STAR.md) for the vision.

# PRD-169 — UX Consistency & Design System (WS-14)

**Chain:** Closing PRD. Branch `ralph/prd-169-ux-consistency` from main after PRD-168 + PRD-165. Size **M**.
**Source:** report §2.13; D8 BINDING (studio is the future).

## Overview

One design language, one set of UI-state primitives, an IA users can navigate, and a design doc agents can trust. The classic↔studio duplication ends per D8.

## Binding amendments

D8; Q72: studio default — classic board/calendar/chat deleted after parity check (calendar already done in 162); Q74: `/context` folded (168), Q75 default: Upload + RAG Test become admin-gated tabs, not primary IA, Q76: GraphView shell standard confirmed (165), Q91 default: DESIGN_SYSTEM.md gains a Studio chapter documenting `cc-*`/`sh-*` (~1,200 undocumented lines) — one doc, both languages until classic dies, Q97 default: StudioPageTabs live counts via real queries, FilterTabs de-duplicated.

## User Stories

### S1: Classic sunset (parity-checked)
Inventory classic board/chat features vs studio (calendar done in 162); port the gaps; delete classic implementations + routes; redirects for bookmarks.
**Acceptance:**
- [ ] Parity checklist in PR (reviewer-verifiable)
- [ ] Classic trees deleted; contract green; nav coherent — dev-browser verify

### S2: UI-state primitives
One Loading (Skeleton), Empty, Error, and DeleteConfirmation (modal — kill `window.confirm`/`window.location.reload()`) consumed across the 11 focus surfaces; six divergent loading patterns collapse to one.
**Acceptance:**
- [ ] Primitives in `components/shared/`; vitest
- [ ] Grep gates: no `window.confirm`, no bare `console.error` swallows on the focus surfaces
- [ ] Each focus page shows consistent states — dev-browser sweep

### S3: Color + palette discipline
Charts/graphs/badges read the CSS-var token palette (fixes KnowledgeGraphVisualizer's hardcoded `#111827` in light theme); codemod the 94 banned `orange-*`/hex violations; ESLint/stylelint rule preventing recurrence.
**Acceptance:**
- [ ] Light theme renders correctly on all graph surfaces — dev-browser verify
- [ ] Lint rule red on a seeded violation; zero violations on the tree

### S4: IA + a11y pass
Knowledge Base: flatten the 3-level tab nesting, single RAG-test surface, admin-gate Upload/RAG-Test (Q75); aria-labels on icon buttons (69 of 74 focus files have none); Canvas tabs become semantic tabs with keyboard support + always-visible close (`Canvas.tsx:106-131`); scoped ESC handling in Explorer.
**Acceptance:**
- [ ] Keyboard-only walk of Knowledge Base + Canvas succeeds — dev-browser verify
- [ ] axe scan on focus pages: zero critical violations (automated check in CI, non-required job)

### S5: Item-detail convention + design doc truth
One item-detail pattern (Sheet for entity drill-ins per the BusinessGraphPanel reference; Dialog only for confirmations); migrate the divergent four; DESIGN_SYSTEM.md updated: Studio chapter, state-primitive usage, palette tokens, the canonical-terms table embedded.
**Acceptance:**
- [ ] Focus surfaces use the convention (review checklist)
- [ ] DESIGN_SYSTEM.md current — an agent following it produces studio-correct UI (spot-check prompt test)

## Non-Goals

New features, rebrand, mobile (separate initiative per D8 note).

## Success Metrics

- One design language shipping; zero classic duplicates.
- axe: zero critical violations on focus pages.
- DESIGN_SYSTEM.md and shipped UI agree (spot-check passes).

## Testing

vitest for primitives, axe CI job, dev-browser sweeps per story, contract suite green after deletions. Full suite green.

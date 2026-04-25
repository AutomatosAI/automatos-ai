# Cluster 1 Part A — Rehouse Plan

**Branch:** `ralph/cluster-1a-rehouse`
**Worktree:** `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-CLUSTER-1A`
**Source PRD:** `docs/PRDS/AUTOMATOS-0.2/10-PRD-CLUSTER-1-WORK-LOOP.md`
**Spec:** `scripts/ralph/prd.json` (= prd-cluster-1a.json)

This is **80% reuse, 20% rehousing**. NOT a new build. Read the user story `notes` field before touching anything — it tells you whether to verify existing code first.

Mark a task `- [x]` only when:
1. Acceptance criteria pass
2. `cd frontend && npx tsc --noEmit` passes (frontend stories)
3. Commit landed on `ralph/cluster-1a-rehouse`

## Stories

- [x] US-001 — Page renames: Workspace → Deliverables, Activity → Command Center
- [x] US-002 — Deliverables: "Created today" hero section above tabs
- [x] US-003 — Deliverables tabs scaffold (Created today / Explorer / Templates / Blog)
- [x] US-004 — Move Blog tab from Activity → Deliverables (verify d9941e36d first)
- [x] US-005 — Blog: SEO fields + Publish toggle on blog detail/edit view
- [x] US-006 — Templates: move from Deliverables top-level into Deliverables → Templates tab
- [x] US-007 — Explorer takes the full page area inside the Explorer tab
- [x] US-008 — Command Center: add History tab (unified view of past Mission + Playbook runs)
- [x] US-009 — Assignments page shell using Marketplace pattern (FeaturedBanner + tabs + grid)
- [ ] US-010 — Assignments hero cards: Mission (lead) + Playbook + Plan + Task
- [ ] US-011 — Contextual hero hint based on workspace state
- [ ] US-012 — Assignments: Playbooks tab grid (reuse existing playbook list data)
- [ ] US-013 — Assignments: Missions tab grid (reuse existing mission list data)
- [ ] US-014 — Quick Task modal (reuse existing CreateTaskDialog component)
- [ ] US-015 — Plan card deep-links to `/chat?mode=plan&from=assignments`
- [ ] US-016 — Recommended-for-you grid (read-only suggestions endpoint)
- [ ] US-017 — Cleanup: remove old Workspace/Activity entry points + dead surfaces we replaced
- [ ] US-018 — Legacy redirects: `/workspace → /deliverables`, `/activity → /command-center`

## Definition of done for the whole cluster

- All 18 boxes checked
- `cd frontend && npx tsc --noEmit` passes on the worktree HEAD
- No new DB tables, no new schema migrations
- Commits follow `feat(cluster-1a): US-XXX — <description>` format
- Final commit message contains `RALPH_COMPLETE` so the loop exits

## Reuse-first reminders (from project CLAUDE.md)

- ChatModeBar exists. useMissionStore.isPlanMode exists. Don't rebuild.
- MarketplaceGrid + FeaturedBanner exist. Use them as the visual template for Assignments.
- FilePreview (universal preview) exists. Reuse for Deliverables tabs.
- WorkspaceExplorer exists. Just move it inside the Explorer tab; do not rewrite.
- CreateTaskDialog exists. Reuse for Quick Task modal.
- Playbook = `workflow_recipes` table. Mission = `orchestration_runs` table. Task = `BoardTask`. Don't confuse them.
- NO new DB tables in Part A. NO `os.getenv()` outside `config.py`.
- If you replace a surface, **delete** the old one. No backward-compat shims.

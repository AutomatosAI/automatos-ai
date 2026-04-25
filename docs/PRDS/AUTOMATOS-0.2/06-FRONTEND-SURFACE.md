# AUTOMATOS 0.2 — Target Frontend Surface

**Purpose:** Collapse ~60 hooks / 429 components / 40+ app routes into the four-tab shell. Name the canonical hook per domain; retire v1/v2/verified triplets.

---

## 1. The four-tab shell (app routing)

```
frontend/app/
├── (workspace)/                  # layout group — workspace chrome
│   ├── layout.tsx                # sidebar with 4 tabs + Settings gear + Advanced
│   ├── goals/                    # Tab 1
│   │   ├── page.tsx              # compose box + recent runs
│   │   ├── [runId]/              # run detail (any kind)
│   │   │   ├── page.tsx
│   │   │   ├── events/           # event stream
│   │   │   └── tasks/            # run tasks board
│   │   └── recipes/              # recurring
│   │       ├── page.tsx
│   │       └── [id]/page.tsx
│   ├── deliverables/             # Tab 2
│   │   ├── page.tsx              # grid + filters + preview
│   │   └── [id]/page.tsx         # detail + grade + download
│   ├── knowledge/                # Tab 3
│   │   ├── page.tsx              # overview
│   │   ├── documents/
│   │   ├── memory/
│   │   ├── graph/
│   │   └── database/             # NL2SQL
│   └── agents/                   # Tab 4
│       ├── page.tsx              # roster
│       ├── [id]/page.tsx         # detail
│       ├── marketplace/          # install new
│       └── advanced/             # coordination + config (expandable)
├── settings/                     # gear — workspace settings
│   ├── members/
│   ├── billing/
│   ├── integrations/
│   └── preferences/
├── admin/                        # super-admin only
│   ├── workspaces/
│   ├── prompts/
│   └── system/
├── marketplace/                  # cross-workspace marketplace
│   ├── skills/
│   ├── models/
│   ├── plugins/
│   ├── widgets/
│   └── templates/
├── auth/                         # clerk / signin / signup
├── widgets/                      # embeddable widgets (served)
│   ├── memory/
│   ├── workflows/
│   └── marketplace/
└── api/                          # next API routes (webhooks, server actions)
```

**Retired top-level routes (collapse into the four tabs):**
- `/chat` → `/goals` (chat is a run kind)
- `/missions` → `/goals`
- `/playbooks` → `/goals/recipes`
- `/activity` → `/agents/advanced/activity` (secondary surface)
- `/dashboard` → `/agents` (default view) or `/analytics` (admin)
- `/workspace` (top-level) → becomes the implicit shell
- `/team` → `/settings/members`
- `/api-control`, `/api-debug`, `/api-diagnostics` → `/admin/system`
- `/field-theory`, `/context` → `/knowledge/memory/field`
- `/tools` → `/agents/advanced/tools`
- `/playbooks` → merges with `/goals/recipes`

---

## 2. Canonical hook list (one per domain)

Target: ~15-20 hooks. Drop the v1/v2/verified/explorer duplicates.

| Hook | Covers | Replaces |
|---|---|---|
| `useGoalsApi` | runs (any kind), chat, missions, recipes, plans | `use-missions-api`, `use-chat`, most of `use-multi-agent*`, `use-workflow-api`, `use-playbook-api`, `use-recipe-*`, `use-page-api` |
| `useAgentsApi` | roster, CRUD, skills/tools assignment, heartbeat, onboarding, tasks, board | `use-agent-api`, `use-agent-execution-api`, `use-heartbeats-api`, `use-pinned-agents`, `use-board-tasks`, `use-board-tasks-api`, `use-coordination-api` |
| `useSkillsApi` | catalog, install | (new; part of marketplace today) |
| `useToolsApi` | registry, exec, composio, mcp | `use-composio-api`, parts of `use-credentials-api` |
| `useKnowledgeApi` | documents, memory, graph, rag, database, cloud, codegraph | `use-knowledge-api`, `use-document-api`, `use-cloud-documents-api`, `use-cloud-storage`, `use-memory-api`, `use-memory-v1-api`, `use-memory-explorer-api`, `use-rag-api`, `use-rag-feedback`, `use-database-knowledge`, `use-semantic-search-api`, `use-field-theory`, `use-context-api`, `use-context-management-api` |
| `useDeliverablesApi` | list, get, grade, download | `use-reports-api`, `use-deliverables-api` |
| `useWorkspacesApi` | CRUD, members, files, github | `use-projects`, parts of team hooks |
| `useMarketplaceApi` | skills/models/plugins/widgets/templates | `use-marketplace-api`, `use-openrouter-api`, `use-model-api`, `use-performance-api` (partial) |
| `useAnalyticsApi` | dashboards, KPI, insights, routing, performance, learning | `use-analytics-api`, `use-kpi-api`, `use-insights-api`, `use-learning-api`, `use-performance-api`, `use-recommendations-api`, `use-orchestration-data`, `use-activity-api` |
| `useAdminApi` | system settings, prompts, routing, credentials, permissions | `use-permissions-api`, `use-credentials-api`, `use-policy`, admin bits |
| `useAuthApi` | session, workspace switch, user | existing Clerk integration + user context |
| `useNotificationsApi` | in-app notifications (PRD-128) | `use-notifications-api` |
| `useHealthApi` | healthz / system status | `use-health-api`, `use-api-debug` |
| `useSearchApi` | global search | `use-global-search` |

**Drop outright:** `use-api-toggle.js` (migrate to TS), `use-api.ts` (superseded by domain hooks), `use-bug-report-api` (move to widget), `use-auto-tour` (stays as UI-only hook, not API), `use-mobile` (UI-only), `use-tooltips` (UI-only).

**Keep as UI hooks (not API — retain):**
- `use-tooltips`, `use-mobile`, `use-auto-tour`, `use-api-toggle` (after TS migration), `use-pinned-agents` (state), `use-playbook-form` (form state).

**Final count:** ~14 API hooks + ~6 UI hooks = 20 total, down from ~60.

---

## 3. Component consolidation (429 → ~300)

### Structural rule
Components are organized by the four tabs, not by feature. Feature-folders like `components/activity/`, `components/missions/`, `components/agents/` reorganize to:

```
frontend/components/
├── shell/                    # app chrome: sidebar, topbar, tab nav
├── goals/                    # tab 1 components
│   ├── ComposeBox.tsx
│   ├── ModePicker.tsx
│   ├── RunCard.tsx
│   ├── RunDetail.tsx
│   ├── RunEventStream.tsx
│   ├── RunTasksBoard.tsx
│   └── chat/
├── deliverables/             # tab 2 components
│   ├── DeliverableGrid.tsx
│   ├── DeliverableCard.tsx
│   ├── DeliverableView.tsx   # THE single preview component for any file type
│   ├── DeliverableGrader.tsx
│   └── filters/
├── knowledge/                # tab 3 components
│   ├── documents/
│   ├── memory/
│   ├── graph/
│   └── database/
├── agents/                   # tab 4 components
│   ├── AgentRoster.tsx
│   ├── AgentCard.tsx
│   ├── AgentDetailModal.tsx
│   ├── AgentReports.tsx → AgentDeliverables.tsx (wave 4)
│   ├── skills/
│   ├── tools/
│   └── coordination/
├── settings/
├── admin/
├── marketplace/
├── widgets/                  # embeddable
├── common/                   # shared primitives (Button, Card, Modal, Drawer, etc. — shadcn-based)
└── charts/                   # recharts wrappers, shared
```

### Consolidation targets (component-level duplicates)

| Duplicate cluster | Canonical | Evidence |
|---|---|---|
| Multiple file-preview components (markdown, pdf, image previews) | `<DeliverableView />` per PRD-131 unified preview | memory: 855bda "unified preview" |
| `<ActivityFeed />` + `<ExecutionHistory />` + `<RunHistory />` | `<RunHistory />` driven by `useGoalsApi` | 3 separate components today |
| `<AgentCard />` variants across Agents tab, Activity, Mission detail | one `<AgentCard variant="..." />` | 3-4 cards, same shape |
| `<MarkdownRenderer />` forks | one canonical (per the rendering feedback memory) | memory: rendering is consumer CSS |
| Mission board task card vs board tasks page card vs tasks list item | one `<RunTaskCard />` | 3 variants |
| Plan editor / mission plan editor / playbook editor | one `<RunPlanEditor />` | mode-picker varies |

**Method for the sweep:** run `npx knip` or `ts-prune` (per `refactor-cleaner` agent) once, spot the unused exports, then do the mechanical moves.

---

## 4. Route migration table (Wave 3 sequencing)

| Current route | Target | Change type | Who moves |
|---|---|---|---|
| `/chat` | `/goals?mode=chat` (deep-link) OR `/goals/c/{id}` | path change | dev |
| `/chat/[id]` | `/goals/[runId]` | path change | dev |
| `/missions` | `/goals` with default filter missions | path change | dev |
| `/missions/[id]` | `/goals/[runId]` | path change | dev |
| `/activity` | `/agents/advanced/activity` | path change; secondary surface | dev |
| `/activity/execution` | `/agents/advanced/activity?filter=execution` | path change | dev |
| `/playbooks` | `/goals/recipes` | rename | dev |
| `/tools` | `/agents/advanced/tools` | path change | dev |
| `/field-theory` | `/knowledge/memory/field` | rename | dev |
| `/context` | `/knowledge/memory` | merge | dev |
| `/team` | `/settings/members` | path change | dev |
| `/workspace` | `/` (root is the workspace) | merge | dev |
| `/dashboard` | `/analytics` (admin view) OR default of `/agents` | merge | dev |
| `/marketplace` subroutes | `/marketplace/{kind}` | reorg | dev |

**Redirect strategy:** Next.js `next.config.js` redirects (301) for one release; then delete.

---

## 5. Hook migration playbook (per-hook steps)

For each canonical hook `useXApi`:

1. Create `frontend/hooks/use-{domain}-api.ts` with typed functions per endpoint.
2. Use React Query v4 (matches existing — per memory reports hooks use `isLoading` not `isPending`).
3. Export everything the old hooks exported (re-exports for one release).
4. Update all import sites to the new path (codemod with `jscodeshift` or search-replace).
5. Delete old hook files.
6. Verify: `grep -r "use-memory-v1-api" frontend/` returns 0.

**Parallelization:** each domain hook can be migrated in its own PR by its own dev; the 14 migrations can run in parallel with only hook files conflicting (rare).

---

## 6. Design system compliance (do not reinvent)

Per memory feedback `feedback-ui-changes.md`:
> NEVER rewrite working UI, NEVER invent design patterns, match existing exactly

0.2 frontend work is **structural reorganization**, not visual redesign. Existing Tailwind + shadcn patterns stay. Component consolidation must preserve visual fidelity — use pixel-diff screenshots in PR reviews if visual regression risk.

**Design system doc:** `frontend/DESIGN_SYSTEM.md` already exists; Wave 3 PRs must cite it.

---

## 7. Telemetry for the collapse

Add a lightweight frontend event on every `use*Api` hook call:
```ts
trackApiCall({ hook: 'useAgentsApi', fn: 'getAgents', duration, status });
```

Gate the old hooks' exports behind a console.warn for one release so stragglers surface:
```ts
/** @deprecated Use useGoalsApi from '@/hooks/use-goals-api' */
export function useMissionsApi() {
  if (process.env.NODE_ENV !== 'production') console.warn('useMissionsApi is deprecated');
  return useGoalsApi(/* ... */);
}
```

---

## 8. Success metrics (Wave 3 close)

- `ls frontend/hooks/use-*-api*.ts | wc -l` ≤ 15
- Zero files named `*-v1-api.ts`, `*-v2-api.ts`, `*-verified.ts`, `*-simple.ts`
- Frontend app routes (`frontend/app/*/page.tsx` count) reduce by ≥30%
- All app routes reachable from the four-tab shell in ≤2 clicks (for non-admin)
- Pixel-diff screenshots of key flows show ≤2% visual regression per page
- `knip` reports < 10 unused exports across `components/` (target: zero)

See [09-SUCCESS-METRICS.md](./09-SUCCESS-METRICS.md) for full scorecard.

---

## 9. What Wave 3 does NOT touch

- Styling. Tailwind tokens + shadcn stay as-is.
- Component internals. If `<AgentCard />` renders correctly today, its JSX doesn't change — only its import path and co-location change.
- Forms. React Hook Form usage preserved.
- State (Zustand stores in `frontend/stores/`) stays; store names may be renamed along with hook renames.

---

**Test for this doc:** a frontend engineer picks any current route or hook and finds its target in this file in under 10 seconds. If not, the mapping table needs more rows.

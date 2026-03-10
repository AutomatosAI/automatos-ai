# PRD-76: Agent Reporting & Workspace

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Claude
**Created:** 2026-03-09
**Updated:** 2026-03-09
**Dependencies:** PRD-55 (Agent Heartbeats — COMPLETE), PRD-72 (Activity Command Centre — IN PROGRESS), PRD-66 (Code Canvas — COMPLETE), PRD-71 (Unified Skills — COMPLETE)

---

## Executive Summary

Agents run heartbeats, execute recipes, and complete tasks — but their output disappears. Sentinel writes a status report to `scratchpad_write` which goes to Redis and evaporates after execution. The `heartbeat_results` table captures metadata (status, findings count, tokens) but not the actual report content. Users can see "Sentinel ran at 7:30am" but can't read what it found.

This PRD introduces **Agent Reports** — a hybrid storage system where structured metadata lives in PostgreSQL and report content lives in the workspace filesystem. Reports become first-class objects: viewable, downloadable, gradeable, and shareable between agents.

### What We're Building

1. **`agent_reports` table** — lightweight metadata for discovery, filtering, stats, and grading
2. **Workspace report storage** — `/reports/{agent_name}/{date}_{title}.{ext}` convention for full content (markdown, images, CSVs, PDFs, any file type)
3. **`platform_submit_report` tool** — single tool agents call to write file + insert DB row
4. **Reports tab on Activity page** — grid of report cards, filterable, with inline viewer
5. **Reports tab on Agent profile** — per-agent standup view with run history and report quality
6. **Report viewer** — slide-over panel with rendered content, download, and user grading
7. **Cross-agent file access** — agents can read reports from other agents via `workspace_read_file`

### What We're NOT Building

- Auto-reviewing agent (too much complexity for unclear value — user grading via simple slider for now)
- New file explorer widget (Code Canvas stays for code; reports get their own dedicated viewer)
- Report template engine (agents write freeform markdown; structure comes from their skill prompts)

### Page Restructuring (from PRD-72 amendments)

| Change | From | To |
|--------|------|-----|
| Activity page tabs | Dashboard \| Feed \| Recipes \| Missions | Dashboard \| Feed \| Routines \| Reports \| Missions |
| Agents page tabs | Roster \| Configuration \| Coordination \| ~~Performance~~ | Roster \| Configuration \| Coordination \| Recipes |
| Recipes tab | Lives on Activity page | Moves to Agents page (all config in one place) |
| Performance tab | Dead code on Agents page | Replaced by Recipes tab |

---

## 1. Database Schema

### 1.1 `agent_reports` Table

```sql
CREATE TABLE agent_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id        UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_id            INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    heartbeat_result_id INTEGER REFERENCES heartbeat_results(id) ON DELETE SET NULL,

    -- Report metadata
    report_type         VARCHAR(30) NOT NULL DEFAULT 'standup',
        -- standup: routine heartbeat output
        -- research: deep-dive findings
        -- incident: something went wrong
        -- summary: daily/weekly rollup
        -- delivery: completed deliverable (marketing email, analysis, etc.)
        -- audit: compliance/security check
    title               VARCHAR(255) NOT NULL,
    summary             VARCHAR(500),           -- first ~200 chars or agent-provided summary
    status              VARCHAR(20) NOT NULL DEFAULT 'ok',
        -- ok: nothing to worry about
        -- warning: needs attention
        -- critical: immediate action needed
        -- info: informational only

    -- File reference
    file_path           VARCHAR(1024) NOT NULL,  -- relative to workspace root: reports/sentinel/2026-03-09_0730_status.md
    file_type           VARCHAR(20) NOT NULL DEFAULT 'markdown',
        -- markdown, pdf, csv, image, document, spreadsheet, audio, video, archive
    file_size_bytes     INTEGER,

    -- Structured metrics (agent-provided, varies by report type)
    metrics             JSONB DEFAULT '{}',
        -- Examples:
        -- sentinel: { services_checked: 5, errors_found: 0, response_time_avg_ms: 340 }
        -- researcher: { sources_reviewed: 12, key_findings: 3, confidence: 0.85 }
        -- marketer: { emails_sent: 150, open_rate: 0.32 }

    -- Linked artifacts (other files produced alongside the report)
    attachments         JSONB DEFAULT '[]',
        -- [{ title: "Error screenshot", file_path: "reports/sentinel/2026-03-09_errors.png", file_type: "image" }]

    -- User grading
    grade               SMALLINT CHECK (grade >= 1 AND grade <= 5),  -- 1-5 star rating
    grade_notes         TEXT,                   -- optional reviewer comment
    graded_by           INTEGER REFERENCES users(id) ON DELETE SET NULL,
    graded_at           TIMESTAMPTZ,

    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indices
CREATE INDEX ix_agent_reports_workspace     ON agent_reports(workspace_id);
CREATE INDEX ix_agent_reports_agent         ON agent_reports(agent_id);
CREATE INDEX ix_agent_reports_type          ON agent_reports(workspace_id, report_type);
CREATE INDEX ix_agent_reports_status        ON agent_reports(workspace_id, status);
CREATE INDEX ix_agent_reports_created       ON agent_reports(created_at DESC);
CREATE INDEX ix_agent_reports_heartbeat     ON agent_reports(heartbeat_result_id) WHERE heartbeat_result_id IS NOT NULL;
CREATE INDEX ix_agent_reports_ungraded      ON agent_reports(workspace_id, created_at DESC) WHERE grade IS NULL;
```

### 1.2 Workspace File Convention

```
/workspace/
  reports/
    {agent_name}/                          # lowercase, hyphenated (e.g. "sentinel", "content-writer")
      {YYYY-MM-DD}_{HHMM}_{slug}.md       # primary report file
      {YYYY-MM-DD}_{HHMM}_{slug}/         # optional folder for multi-file reports
        report.md
        chart.png
        data.csv
```

**Naming rules (critical for cross-agent access):**
- Agent name: lowercase, hyphens for spaces, no special chars → `slugify(agent.name)`
- Date prefix: `YYYY-MM-DD_HHMM` in agent's configured timezone (or UTC)
- Slug: kebab-case summary → `weekly-marketing-update`, `platform-health-check`
- Extension matches content type

**Example cross-agent workflow:**
```
Agent A (researcher): writes → /reports/market-researcher/2026-03-09_0200_competitor-analysis.md
Agent B (marketer):   reads  → workspace_read_file("reports/market-researcher/2026-03-09_0200_competitor-analysis.md")
Agent B (marketer):   writes → /reports/email-marketer/2026-03-09_0800_weekly-newsletter.md
```

---

## 2. Platform Tool: `platform_submit_report`

### 2.1 Tool Definition

```python
{
    "name": "platform_submit_report",
    "description": "Submit a report after completing a task or heartbeat cycle. Writes the report file to workspace storage and records metadata for tracking. Use after every heartbeat run, research completion, or deliverable.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for the report (e.g. 'Platform Health Check', 'Weekly Newsletter Draft')"
            },
            "content": {
                "type": "string",
                "description": "Full report content in markdown format"
            },
            "report_type": {
                "type": "string",
                "enum": ["standup", "research", "incident", "summary", "delivery", "audit"],
                "description": "Category of report"
            },
            "status": {
                "type": "string",
                "enum": ["ok", "warning", "critical", "info"],
                "description": "Overall status — ok means nothing to worry about, warning/critical need attention"
            },
            "summary": {
                "type": "string",
                "description": "One-line summary (shown in activity feed cards)"
            },
            "metrics": {
                "type": "object",
                "description": "Structured metrics relevant to this report (e.g. { errors_found: 2, services_checked: 5 })"
            },
            "attachments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "file_path": { "type": "string", "description": "Relative workspace path to attachment" },
                        "file_type": { "type": "string" }
                    }
                },
                "description": "Additional files produced alongside this report (images, data files, etc.)"
            }
        },
        "required": ["title", "content", "report_type", "status"]
    }
}
```

### 2.2 Execution Flow

```
Agent calls platform_submit_report(title, content, report_type, status, ...)
  │
  ├─ 1. Generate file path: reports/{agent_slug}/{date}_{time}_{title_slug}.md
  ├─ 2. Write content to workspace filesystem via WorkspaceClient
  ├─ 3. Get file size
  ├─ 4. INSERT into agent_reports (workspace_id, agent_id, file_path, title, ...)
  ├─ 5. If heartbeat context exists, link heartbeat_result_id
  └─ 6. Return { success: true, report_id, file_path, view_url }
```

### 2.3 Integration with Heartbeat Service

After `_agent_tick()` and `_orchestrator_tick()` complete, the heartbeat service should:
1. Check if the agent produced a report (via `platform_submit_report` tool call in the execution)
2. If no report was produced, auto-create a minimal standup entry from the `heartbeat_results` data:
   ```
   Title: "{agent_name} Heartbeat — {timestamp}"
   Content: Auto-generated from findings + actions_taken JSONB
   Type: standup
   Status: mapped from heartbeat_results.status
   ```

This ensures every heartbeat run has a corresponding report row — even if the agent forgot to call `platform_submit_report`.

---

## 3. API Endpoints

### 3.1 Report CRUD

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/reports` | List reports for workspace. Query: `agent_id`, `report_type`, `status`, `graded` (bool), `period` (1d/7d/30d), `limit`, `offset` |
| `GET` | `/api/reports/{id}` | Single report metadata + content (fetches file from workspace) |
| `GET` | `/api/reports/{id}/download` | Download report file (Content-Disposition: attachment) |
| `PATCH` | `/api/reports/{id}/grade` | Submit grade (1-5) and optional notes |
| `GET` | `/api/reports/stats` | Aggregate stats: total, by_type, by_status, avg_grade, ungraded_count. Query: `period` |
| `GET` | `/api/agents/{id}/reports` | Reports for a specific agent. Query: `report_type`, `period`, `limit` |

### 3.2 Response Schemas

```typescript
interface AgentReport {
  id: string
  workspace_id: string
  agent_id: number
  agent_name: string
  agent_avatar_url: string | null
  heartbeat_result_id: number | null

  report_type: 'standup' | 'research' | 'incident' | 'summary' | 'delivery' | 'audit'
  title: string
  summary: string | null
  status: 'ok' | 'warning' | 'critical' | 'info'

  file_path: string
  file_type: string
  file_size_bytes: number | null
  content?: string              // Only included in single-report GET, not list

  metrics: Record<string, any>
  attachments: Array<{
    title: string
    file_path: string
    file_type: string
  }>

  grade: number | null          // 1-5
  grade_notes: string | null
  graded_by: number | null
  graded_at: string | null

  created_at: string
}

interface ReportStats {
  total: number
  by_type: Record<string, number>
  by_status: Record<string, number>
  avg_grade: number | null
  ungraded_count: number
  period: string
}
```

---

## 4. Activity Page — Reports Tab

### 4.1 Updated Tab Structure

```
Activity Command Centre
├── Dashboard    (existing — hero stats + summary)
├── Feed         (existing — unified timeline)
├── Routines     (existing — heartbeat management)
├── Reports      (NEW)
└── Missions     (existing — coming soon placeholder)
```

### 4.2 Reports Tab Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Filter Bar                                                    │
│  [All Types ▼]  [All Agents ▼]  [All Statuses ▼]  [Period ▼] │
│  [★ Ungraded only]                              [{n} reports] │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─ glass-card border-l-3 ─────────────────────────────────┐  │
│  │ [AgentAvatar]  Sentinel · Standup           10 min ago   │  │
│  │                Platform Health Check        ● OK          │  │
│  │                "All 5 services healthy, 0 errors in 15m"  │  │
│  │                ┌─metrics─┐                                │  │
│  │                │ 5 checked │ 0 errors │ 340ms avg │       │  │
│  │                └──────────┘                                │  │
│  │                ★★★★☆  graded 2h ago     [View] [Download] │  │
│  └────────────────────────────────────────────────────────── ┘  │
│                                                                │
│  ┌─ glass-card border-l-3 border-l-warning ────────────────┐  │
│  │ [AgentAvatar]  Market Researcher · Research  2 hours ago  │  │
│  │                Competitor Analysis Q1        ⚠ Warning     │  │
│  │                "Found 3 new competitors, 2 pricing changes"│  │
│  │                ┌─metrics─┐                                │  │
│  │                │ 12 sources │ 3 findings │ 0.85 conf │    │  │
│  │                └──────────┘                                │  │
│  │                ☆☆☆☆☆  not graded          [View] [Download│  │
│  └────────────────────────────────────────────────────────── ┘  │
│                                                                │
│                        [Load More]                              │
└──────────────────────────────────────────────────────────────┘
```

**Left border colours by status:**
- OK: `border-l-[hsl(var(--success))]` (green)
- Warning: `border-l-[hsl(var(--warning))]` (amber)
- Critical: `border-l-[hsl(var(--destructive))]` (red)
- Info: `border-l-[hsl(var(--info))]` (blue)

**Report type badges** (muted, after agent name):
| Type | Badge | Icon |
|------|-------|------|
| Standup | `Standup` | `ClipboardCheck` |
| Research | `Research` | `Search` |
| Incident | `Incident` | `AlertTriangle` |
| Summary | `Summary` | `FileText` |
| Delivery | `Delivery` | `Package` |
| Audit | `Audit` | `Shield` |

### 4.3 Report Viewer (Slide-Over Panel)

Clicking "View" on a report card opens a slide-over panel from the right (60% width on desktop, full-screen on mobile):

```
┌─ slide-over glass-panel ──────────────────────────────────────┐
│  ← Back                                    [Download] [Share]  │
│                                                                │
│  [AgentAvatar]  Sentinel                                       │
│  Platform Health Check                                         │
│  Standup · Mar 9, 2026 07:30 · ● OK                          │
│                                                                │
│  ┌─ Metrics Bar ──────────────────────────────────────────┐   │
│  │  5 services   │  0 errors   │  340ms avg  │  $0.002     │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Report Content ───────────────────────────────────────┐   │
│  │                                                         │   │
│  │  SENTINEL STATUS REPORT — 2026-03-09 07:30 UTC          │   │
│  │  ──────────────────────────                              │   │
│  │  API Health:     OK — all endpoints responding < 500ms   │   │
│  │  Error Rate:     OK — 0 errors in last 15m (baseline: 2) │   │
│  │  Deploy Status:  OK — a95adab deployed 2h ago            │   │
│  │  LLM Costs:      OK — $1.24 today (avg: $1.80/day)      │   │
│  │  ──────────────────────────                              │   │
│  │  Action Required: None                                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Attachments ──────────────────────────────────────────┐   │
│  │  📎 Error screenshot (error_spike.png)     [Download]   │   │
│  │  📎 Cost breakdown (costs.csv)             [Download]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Grade This Report ────────────────────────────────────┐   │
│  │  ★ ★ ★ ★ ★                                             │   │
│  │  [Optional notes...]                         [Submit]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Content rendering:**
- Markdown → rendered HTML (using existing `react-markdown` or `@uiw/react-md-editor` preview)
- Images → inline preview
- CSV → simple table preview
- PDF → embedded viewer or download prompt
- Other → download only

### 4.4 Feed Integration

Reports also appear in the Feed tab as activity items:

```
[📋]  Report · "Platform Health Check"                     10m ago
      Agent: Sentinel · ● OK · Standup
      "All 5 services healthy, 0 errors in 15m"
                                              [View Report]
```

Type: `report` added to `ActivityFeedItem.type` union.

---

## 5. Agents Page — Reports Tab + Recipes Tab

### 5.1 Updated Tab Structure

```
Agents
├── Roster          (existing — agent grid/list)
├── Configuration   (existing — agent config modal/panel)
├── Coordination    (existing — inter-agent settings)
└── Recipes         (MOVED from Activity page, replaces dead Performance tab)
```

### 5.2 Agent Profile — Reports Section

On the individual agent config/detail view, add a "Reports" section or tab:

```
┌─ Agent: Sentinel ──────────────────────────────────────────┐
│  [Overview] [Tools] [Heartbeat] [Reports]                    │
│                                                              │
│  Reports Tab:                                                │
│  ┌─ Standup Summary ─────────────────────────────────────┐  │
│  │  Last 7 days: 21 reports · 20 OK · 1 warning · 0 crit│  │
│  │  Avg grade: ★★★★☆ (4.2)  ·  3 ungraded               │  │
│  │  Total cost: $0.042  ·  Avg tokens: 1,240/run         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Recent Reports ──────────────────────────────────────┐  │
│  │  ● OK   Platform Health Check     Today 07:30  ★★★★☆ │  │
│  │  ● OK   Platform Health Check     Today 06:30  ★★★★★ │  │
│  │  ⚠ WARN  Error Spike Detected     Yesterday    ★★★☆☆ │  │
│  │  ● OK   Platform Health Check     Yesterday    ——     │  │
│  │  ● OK   Platform Health Check     Mar 7        ——     │  │
│  │                                    [View All →]        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Workspace Storage & Cross-Agent Access

### 6.1 File Lifecycle

```
1. Agent produces report → platform_submit_report writes to /workspace/reports/{agent}/
2. Report accessible immediately via workspace_read_file by other agents
3. User can download via /api/reports/{id}/download
4. Over time, old reports can be:
   a. Archived to S3 (automatos document storage)
   b. Synced to Google Drive (user's connected storage via Composio)
   c. Deleted (with DB row retained as tombstone for stats)
```

### 6.2 Cross-Agent Pattern

Agents reference each other's reports by convention:

```python
# Agent B's skill prompt:
"Before sending the weekly newsletter, read the latest research report:
 workspace_read_file('reports/market-researcher/') to find the most recent file,
 then workspace_read_file('reports/market-researcher/{latest_file}') to get content."
```

Or via the DB:
```python
# New tool: platform_get_latest_report
{
    "name": "platform_get_latest_report",
    "description": "Get the most recent report from a specific agent",
    "parameters": {
        "agent_name": { "type": "string" },
        "report_type": { "type": "string", "enum": [...] }  // optional filter
    }
}
# Returns: { report_id, title, file_path, content, metrics, created_at }
```

### 6.3 Download Endpoint

```
GET /api/reports/{id}/download
→ Content-Type: application/octet-stream (or appropriate MIME)
→ Content-Disposition: attachment; filename="{title_slug}.{ext}"
→ Streams file from workspace storage
```

For attachments:
```
GET /api/reports/{id}/attachments/{index}/download
```

---

## 7. Heartbeat Service Integration

### 7.1 Changes to `heartbeat_service.py`

After `_agent_tick()` completes:

```python
async def _agent_tick(self, agent_id, workspace_id, hb_config):
    # ... existing execution logic ...

    result = self._store_heartbeat_result(...)

    # Check if agent called platform_submit_report during execution
    report_submitted = self._check_report_in_tool_calls(execution_context)

    if not report_submitted:
        # Auto-generate minimal report from heartbeat result
        await self._auto_create_report(
            agent_id=agent_id,
            workspace_id=workspace_id,
            heartbeat_result=result,
            findings=result.findings,
            actions_taken=result.actions_taken,
        )
```

### 7.2 Auto-Report Content Template

```markdown
# {Agent Name} — Heartbeat Report
**{timestamp}** · Status: {status}

## Findings
{for each finding in findings JSONB}
- [{severity}] {description}
{/for}

## Actions Taken
{for each action in actions_taken JSONB}
- {action_description} → {result}
{/for}

## Metrics
- Tokens used: {tokens_used}
- Cost: ${cost}
- Duration: {duration}s
```

---

## 8. File Structure

```
orchestrator/
  api/
    reports.py                    # NEW — /api/reports CRUD + download
  services/
    report_service.py             # NEW — report creation, grading, stats
  core/models/
    core.py                       # MODIFY — add AgentReport model
  modules/tools/
    platform_actions.py           # MODIFY — register platform_submit_report + platform_get_latest_report
    platform_executor.py          # MODIFY — handler for submit_report + get_latest_report
  services/
    heartbeat_service.py          # MODIFY — auto-create report after tick
  alembic/versions/
    {date}_add_agent_reports.py   # NEW — migration

frontend/
  components/
    activity/
      activity-page.tsx           # MODIFY — add Reports tab, remove Recipes tab
      activity-reports.tsx        # NEW — Reports tab content (grid + filters)
      report-card.tsx             # NEW — individual report card
      report-viewer.tsx           # NEW — slide-over panel with content + grading
      report-grade-form.tsx       # NEW — star rating + notes form
    agents/
      agent-management.tsx        # MODIFY — replace Performance tab with Recipes tab
      agent-reports.tsx           # NEW — per-agent reports section
  hooks/
    use-reports.ts                # NEW — SWR hooks for reports API
  lib/
    reports-service.ts            # NEW — API client methods
```

---

## 9. Implementation Phases

### Phase 1: DB + Tool + Auto-Report (Backend Foundation)
1. Create `agent_reports` table migration
2. Add `AgentReport` SQLAlchemy model
3. Build `report_service.py` (create, list, get, grade, stats)
4. Build `platform_submit_report` tool (ActionDefinition + executor handler)
5. Build `platform_get_latest_report` tool
6. Wire heartbeat service auto-report fallback
7. Create `/api/reports` endpoints

### Phase 2: Activity Page Reports Tab (Frontend)
8. Add "Reports" tab to Activity page (replace Recipes position)
9. Build `activity-reports.tsx` with report card grid
10. Build `report-card.tsx` component
11. Build filter bar (type, agent, status, period, ungraded toggle)
12. Build `use-reports.ts` hooks with polling

### Phase 3: Report Viewer + Grading
13. Build `report-viewer.tsx` slide-over panel
14. Implement markdown rendering for report content
15. Build `report-grade-form.tsx` (star rating + notes)
16. Wire grade submission to `PATCH /api/reports/{id}/grade`
17. Add download button (single file + attachments)

### Phase 4: Agent Profile Integration
18. Add Reports section to agent detail/config view
19. Build `agent-reports.tsx` (summary stats + recent list)
20. Link "View All →" to Activity Reports tab filtered by agent

### Phase 5: Recipes Tab Migration + Feed Integration
21. Move Recipes tab component from Activity to Agents page
22. Replace dead Performance tab with Recipes
23. Add report items to Activity Feed
24. Update Activity stats to include report counts

### Phase 6: Polish
25. Report content rendering for non-markdown (images, CSV tables, PDF)
26. Mobile responsive pass
27. Loading skeletons for report cards
28. Empty states ("No reports yet — configure a heartbeat to get started")
29. Attachment preview (inline images, downloadable files)
30. `prefers-reduced-motion` compliance

---

## 10. Sentinel Skill Update

Update `/automatos-skills/sentinel/SKILL.md` to use `platform_submit_report` instead of `scratchpad_write`:

```yaml
tools:
  - name: platform_submit_report        # CHANGED from scratchpad_write
    description: Submit status report after each heartbeat cycle
```

**Step 6 in workflow changes from:**
> Call `scratchpad_write` to persist the current baseline and report.

**To:**
> Call `platform_submit_report` with the full status report. Use report_type: "standup", set status based on findings severity (ok/warning/critical). Include metrics: { services_checked, errors_found, avg_response_ms, daily_llm_cost }.

All agent skills that produce reports should be updated to call `platform_submit_report`.

---

## 11. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Report visibility | 100% of heartbeat runs produce a report row | Compare heartbeat_results count vs agent_reports count |
| Time to read a report | < 2 clicks from Activity page | Click "Reports" tab → click report card |
| Cross-agent file access | Agents can read other agents' reports | Integration test: Agent B reads Agent A's report file |
| Grading adoption | >50% of reports graded within 24h | `ungraded_count` / `total` from stats endpoint |
| Report quality trend | Avg grade improves over 30 days | Weekly avg_grade trend |

---

## 12. Open Questions (Resolved)

1. **Report retention policy?** — **DECIDED:** 30 days in workspace, then delete. DB metadata kept indefinitely. Business tier gets longer retention (future).

2. **Report size limits?** — **DECIDED:** No limits. Users have 5GB workspace quota. If exceeded, reports fail to save (natural backpressure).

3. **Notification on critical reports?** — **DEFERRED:** Separate PRD for notification bell service (central alert routing to user-selected channels). TODO after PRD-76.

4. **Report threading?** — **DECIDED:** Keep simple, no threading for now.

5. **Bulk grading?** — **DECIDED:** Good idea, keep for Phase 6 enhancement.
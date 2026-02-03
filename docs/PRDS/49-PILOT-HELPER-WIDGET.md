# PRD: Pilot Helper Widget (Jira Bug Reporter)

## Summary

Replace the existing orange chat dot (`chat-widget.tsx`) with a **Pilot Helper Widget** — a dual-mode floating overlay with a **Help** tab (per-page guidance) and a **Report Bug** tab (creates Jira tickets via Composio). The widget opens as a popup on the current page (no navigation to `/chat`). Bug reports auto-capture browser context, Clerk user info, and console errors.

---

## Architecture

```
Frontend                          Backend
────────                          ───────
PilotHelperWidget                 POST /api/bug-reports/
  ├─ Help tab (static)               │
  └─ Bug Report tab ──→ useSubmitBugReport() ──→ ComposioToolExecutor.execute()
       (form + context)                              action=JIRA_CREATE_ISSUE
                                                     agent_id=0
                                                     skip_validation=True
                                                     workspace admin connection
```

**No new Composio code** — uses existing `ComposioToolExecutor` + `EntityManager` with the admin pattern from `CloudSyncService`.

---

## Stories (6 total, ordered by dependency)

### US-001: Add Jira bug reporting config

**Files:**
- Modify: `orchestrator/config.py` — add `JIRA_PROJECT_KEY` and `JIRA_BUG_REPORTS_ENABLED`
- Modify: `orchestrator/.env.example` — document new vars

**Details:**
```python
JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "PILOT")
JIRA_BUG_REPORTS_ENABLED: bool = os.getenv("JIRA_BUG_REPORTS_ENABLED", "true").lower() == "true"
```

---

### US-002: Create POST /api/bug-reports endpoint

**Files:**
- Create: `orchestrator/api/bug_reports.py`
- Modify: `orchestrator/main.py` — register router

**Details:**
- Pydantic models: `BugReportRequest` (title, description, severity, category, screenshot_base64, context dict) → `BugReportResponse` (success, jira_key, jira_url, message)
- Uses `ComposioToolExecutor(db).execute(action="JIRA_CREATE_ISSUE", params={...}, agent_id=0, workspace_id=ctx.workspace_id, app_name="JIRA", skip_validation=True)`
- Formats description with auto-captured metadata (URL, page, user agent, viewport, user email/name, console errors, timestamp)
- Maps severity to Jira priority: Critical→Highest, Major→High, Minor→Medium
- Prepends category tag to summary: `[UI Bug] Title here`
- Returns jira_key (e.g. `PILOT-42`) and browser URL
- Auth: `get_request_context_hybrid` (Clerk JWT + workspace)
- Respects `JIRA_BUG_REPORTS_ENABLED` feature flag

---

### US-003: Create frontend bug report hook

**Files:**
- Create: `frontend/hooks/use-bug-report-api.ts`

**Details:**
- `useSubmitBugReport()` — TanStack Query `useMutation`
- Posts to `/api/bug-reports/` via existing `apiClient.post()`
- TypeScript interfaces matching backend models
- Uses existing apiClient which handles Clerk JWT + workspace ID injection

---

### US-004: Rewrite chat-widget.tsx as PilotHelperWidget

**Files:**
- Rewrite: `frontend/components/chatbot/chat-widget.tsx`

**Floating button:**
- `Bug` icon from lucide-react (replaces `MessageCircle`)
- Orange gradient accent kept (`from-orange-500 to-red-500`)
- 64px rounded, fixed bottom-right, z-50
- Same spring animation (stiffness 500, damping 30)
- Click toggles popup open/closed (NO navigation)

**Popup overlay:**
- `glass-card card-glow` styling, `w-96`, `max-h-[600px]` with scroll
- Two tabs (shadcn `Tabs`): "Help" and "Report Bug"
- Escape key or X button closes

**Help tab:**
- Static per-page guidance items (title + description + optional link)
- Content map keyed by `context.currentPage` (dashboard, agents, documents, workflows, tools, analytics, context, playbooks, chat)
- Items use `HelpCircle`, `BookOpen` icons

**Bug Report tab — form:**
- Title: `Input` (required)
- Description: `Textarea` (required, placeholder: "What happened? Steps to reproduce...")
- Severity: `Select` — Critical, Major, Minor (default: Minor)
- Category: `Select` — UI Bug, Data Issue, Performance, Other (default: Other)
- Screenshot: Paste area with `onPaste` handler, extracts image from clipboard, converts to base64, shows thumbnail preview with remove button. Max 5MB.

**Auto-captured context (invisible, built at submit time):**
- `window.location.href`
- `context.currentPage`
- `navigator.userAgent`
- `${window.innerWidth}x${window.innerHeight}`
- Clerk `user.primaryEmailAddress?.emailAddress` + `user.fullName`
- Last 20 console errors (captured via `console.error` override in `useEffect`, cleaned up on unmount)

**States:**
- Form → Loading (submit spinner) → Success (green check + PILOT-42 link + "Submit another") → Error (red message + retry)

**Exports:** `ChatWidget` (backward compat) + `PilotHelperWidget`

---

### US-005: Update main-layout.tsx integration

**Files:**
- Modify: `frontend/components/layout/main-layout.tsx`

**Details:**
- Remove `!pathname.startsWith('/chat')` condition — widget is now an overlay, show on ALL pages
- Keep same `chatContext` prop passing
- Update comment to "Pilot Helper Widget"

---

### US-006: Add per-page help content

**Files:**
- Part of `chat-widget.tsx` (inline data, same file as US-004)

**Details:**
- `PAGE_HELP_CONTENT` constant: Record<string, Array<{title, description, link?}>>
- Coverage: dashboard, agents, documents, workflows, tools, analytics, context, playbooks, chat
- 2-4 items per page
- Fallback to dashboard content for unknown pages

---

## Dependency Order

```
US-001 (config) → US-002 (API) → US-003 (hook) → US-004 + US-006 (widget) → US-005 (layout)
```

## Files Summary

| Action | File |
|--------|------|
| Modify | `orchestrator/config.py` |
| Modify | `orchestrator/.env.example` |
| Create | `orchestrator/api/bug_reports.py` |
| Modify | `orchestrator/main.py` |
| Create | `frontend/hooks/use-bug-report-api.ts` |
| Rewrite | `frontend/components/chatbot/chat-widget.tsx` |
| Modify | `frontend/components/layout/main-layout.tsx` |

**7 files total** (2 new, 1 rewrite, 4 modify).

## Verification

1. **Backend**: Start orchestrator, send `POST /api/bug-reports/` with test payload, confirm Jira ticket appears in the configured project
2. **Frontend**: Load any page, click the floating bug icon, verify popup opens with Help and Report Bug tabs
3. **Bug report flow**: Fill form, paste a screenshot, submit → confirm success with Jira key link
4. **Auto-context**: Check created Jira ticket description includes URL, page, browser info, user email, console errors
5. **Error handling**: Disconnect Jira in Composio, submit bug report → confirm friendly error message
6. **Feature flag**: Set `JIRA_BUG_REPORTS_ENABLED=false`, confirm endpoint returns 503

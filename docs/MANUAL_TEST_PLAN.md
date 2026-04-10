# Manual Test Plan — Bugfixing Day

**Date:** 2026-04-01
**Tester:** Gerard
**Branch:** bugfixing-day
**Method:** Browser walkthrough of every page, noting visual bugs, broken data, console errors

---

## Setup Before Testing

- [ ] Open browser DevTools → Console tab (watch for JS errors throughout)
- [ ] Open Network tab (watch for failed API calls, 4xx/5xx responses)
- [ ] Log in as **admin** user first (some pages are admin-only)
- [ ] Have a second browser/incognito for **regular user** testing
- [ ] Note your workspace ID (visible in URL or workspace provider)

---

## Phase 1: Core User Flows (High Priority)

### 1.1 Chat (`/chat`)

**What it does:** Main AI chat interface with history sidebar.

- [ ] Page loads without errors
- [ ] Chat history panel shows previous conversations (toggle on mobile)
- [ ] Start a new conversation — message sends and streams response
- [ ] Response includes proper markdown rendering (bold, code blocks, lists)
- [ ] Tool calls display correctly (tool data cards appear inline)
- [ ] Ask agent to **generate a document/PDF** — verify it works (was broken yesterday)
- [ ] Ask agent to **submit a report** — verify `platform_submit_report` succeeds
- [ ] Switch between conversations — messages load correctly
- [ ] Delete a conversation — confirm it disappears
- [ ] Long conversation — check scrolling, message ordering
- [ ] Check mobile responsive layout (resize browser to ~375px)

**Known issues to verify fixed:**
- [ ] PDF/document generation JSON parse error (commit `a938581` fix deployed?)
- [ ] Tool calls with truncated JSON should retry, not crash

---

### 1.2 Chat Session (`/chat/[id]`)

**What it does:** Direct link to a specific chat, SSR loaded.

- [ ] Open a chat via direct URL — loads with full message history
- [ ] Can continue the conversation
- [ ] Refresh page — state preserved
- [ ] Invalid chat ID — shows appropriate error, not blank page

---

### 1.3 Agents (`/agents`)

**What it does:** Agent roster — create, edit, delete, configure.

- [ ] Page loads, shows list of agents
- [ ] Agent cards show name, model, status, skill
- [ ] **Create new agent** — fill form, save, appears in list
- [ ] **Edit agent** — change name/model/skill, save, changes reflected
- [ ] **Delete agent** — confirm dialog, agent removed (verify board_tasks cleanup works now)
- [ ] Agent detail modal — tabs load (Reports tab if present)
- [ ] Assign a tool to an agent — verify it appears in agent's config
- [ ] Assign a skill to an agent — verify SKILL.md loads from S3
- [ ] Check agent with missing SKILL.md (`12fa-three-loop@3.0.0`, `aai-testing@1.0.0`) — should show error, not silent fail

---

### 1.4 Dashboard (`/dashboard`)

**What it does:** KPI widgets, agent status overview, system health.

- [ ] Page loads with widgets populated (not all zeros/empty)
- [ ] Agent status cards show correct online/offline state
- [ ] KPI numbers look reasonable (not NaN, not null)
- [ ] Any charts/graphs render without errors
- [ ] Refresh — data updates
- [ ] Check for stale data (timestamps should be recent)

---

### 1.5 Documents (`/documents`)

**What it does:** Document library — upload, delete, cloud sync.

- [ ] Page loads, shows existing documents
- [ ] **Upload a document** (PDF, DOCX, TXT) — appears in list
- [ ] Document shows status (processing → completed)
- [ ] **Delete a document** — removed from list
- [ ] Cloud sync section — shows configured connections (if any)
- [ ] Search/filter documents works
- [ ] Click a document — preview or detail view loads

---

## Phase 2: Activity & Operations

### 2.1 Activity (`/activity`)

**What it does:** Command centre with tabs: Dashboard, Feed, Reports, Missions.

- [ ] Page loads, default tab shows content
- [ ] **Dashboard tab** — activity metrics render
- [ ] **Feed tab** — shows recent agent actions, timestamps correct
- [ ] **Reports tab** — lists agent reports with grades
  - [ ] Click a report — slide-over opens with content
  - [ ] Grade a report (star rating) — saves
- [ ] **Missions tab** — shows mission cards with status
- [ ] Tabs switch without errors
- [ ] Empty states look reasonable (not broken layouts)

---

### 2.2 Activity Execution (`/activity/execution?id=X`)

**What it does:** Workflow execution detail viewer.

- [ ] Navigate from Activity → click an execution → loads correctly
- [ ] Shows execution steps in order
- [ ] Each step shows tool outputs, logs
- [ ] No ID in query string — shows error message, not crash

---

### 2.3 Missions (`/missions/[id]`)

**What it does:** Mission detail page.

- [ ] Navigate from Activity → Missions tab → click a mission → loads
- [ ] Shows mission steps, assigned agents, status
- [ ] Invalid mission ID — shows 404 or error, not blank page

---

## Phase 3: Tools & Marketplace

### 3.1 Tools (`/tools`)

**What it does:** Browse available tools, install skills/plugins, configure workspace tools.

- [ ] Page loads, shows tool categories
- [ ] Browse installed tools — cards display correctly
- [ ] Install a new tool/skill — success message, appears in list
- [ ] Composio integrations section — shows connected apps
- [ ] JIRA connection status (expected: may not resolve — API key issue noted)
- [ ] Tool callback page (`/tools/callback`) — handles OAuth redirects

---

### 3.2 Marketplace (`/marketplace`)

**What it does:** Browse community agents, recipes, tools.

- [ ] Page loads, shows featured items carousel
- [ ] Category filters work
- [ ] Search box returns relevant results
- [ ] Agent cards show install count, ratings
- [ ] Click an agent card — detail view loads
- [ ] **Install an agent** from marketplace — success, appears in your agents

---

### 3.3 Marketplace Widgets (`/marketplace/widgets`)

**What it does:** Browse installable widgets.

- [ ] Page loads with widget grid
- [ ] Filter by category
- [ ] Click widget → detail page (`/marketplace/widgets/[id]`)
  - [ ] Overview tab — markdown README renders
  - [ ] Screenshots tab — images load
  - [ ] Reviews tab — shows reviews, can submit one
  - [ ] Install/uninstall toggle works

---

### 3.4 Marketplace Publish (`/marketplace/publish`)

**What it does:** 5-step wizard to publish a widget.

- [ ] Wizard loads, step 1 visible
- [ ] Navigate through steps (Basic Info → Technical → Media → Pricing → Review)
- [ ] Validation fires on required fields
- [ ] Draft auto-saves
- [ ] Submit for review — success message

---

### 3.5 Marketplace Developer (`/marketplace/developer`)

**What it does:** Developer dashboard for published widgets.

- [ ] Shows list of your published widgets
- [ ] Analytics section — install counts, ratings
- [ ] Version history table

---

## Phase 4: Workspace & Files

### 4.1 Workspace (`/workspace`)

**What it does:** File explorer — agent output, reports, code.

- [ ] Page loads, shows file tree
- [ ] Navigate directories
- [ ] Click a file — content preview
- [ ] Reports directory — shows agent reports (verify `platform_submit_report` files land here)
- [ ] File search works

---

## Phase 5: Analytics & Monitoring

### 5.1 Analytics (`/analytics`)

**What it does:** Performance, costs, and insights.

- [ ] Page loads with charts/metrics
- [ ] Cost tracking data shows (LLM usage, token counts)
- [ ] Time range filters work
- [ ] No NaN or undefined values in charts
- [ ] Agent performance comparisons render

---

### 5.2 Field Theory (`/field-theory`)

**What it does:** Field visualization (no MainLayout wrapper).

- [ ] Page loads — visualization renders
- [ ] Interactions work (hover, click nodes)
- [ ] No layout issues (this page has no sidebar wrapper)

---

## Phase 6: Admin-Only Pages

> **Test with admin account only. Then verify these are hidden for regular users.**

### 6.1 Settings (`/settings`)

**What it does:** System configuration.

- [ ] Page loads, shows settings panel
- [ ] Can modify settings and save
- [ ] Settings persist after refresh
- [ ] **As regular user:** nav link hidden, direct URL redirects or shows 403

---

### 6.2 Settings Profile (`/settings/profile`)

**What it does:** User profile editor.

- [ ] Avatar upload works
- [ ] Name/username fields save
- [ ] Email section displays correctly
- [ ] Security section (password/2FA) — stubs or functional?

---

### 6.3 Team Management (`/team`)

**What it does:** Workspace member management.

- [ ] Shows current team members
- [ ] Can invite new members
- [ ] Can change roles
- [ ] Can remove members
- [ ] **As regular user:** nav link hidden, direct URL blocked

---

### 6.4 Context Engineering (`/context`)

**What it does:** RAG system configuration.

- [ ] Page loads with context sources
- [ ] Can add/edit context sources
- [ ] RAG query testing works
- [ ] **As regular user:** nav link hidden, direct URL blocked

---

### 6.5 Admin Plugins (`/admin/plugins`)

**What it does:** Plugin approval queue with security scan results.

- [ ] Shows pending plugins list
- [ ] Risk scores display
- [ ] Expand scan details — static + LLM findings visible
- [ ] Approve/reject a plugin — status updates
- [ ] **Note:** This route isn't in the sidebar nav — only accessible via direct URL

---

### 6.6 Playbooks (`/playbooks`)

**What it does:** Playbooks panel (recipes).

- [ ] Page loads, shows playbook list
- [ ] Can create/edit a playbook
- [ ] Can execute a playbook
- [ ] **Verify:** `workflow_recipes.created_by` null bug — does creating a recipe work? (Error found in logs)

---

## Phase 7: Auth & Edge Cases

### 7.1 Authentication Flow

- [ ] Sign in (`/sign-in`) — Clerk auth works
- [ ] Sign up (`/sign-up`) — new user gets own workspace (not shared)
- [ ] SSO callback (`/sso-callback`) — handles redirect
- [ ] Sign out — clears session, redirects to sign-in
- [ ] Expired session — handled gracefully (not blank page)

### 7.2 Cross-Cutting Checks

- [ ] **Mobile responsive:** Resize to 375px on Chat, Dashboard, Agents pages
- [ ] **Empty workspace:** What does a brand new user see? Onboarding flow triggers?
- [ ] **Console errors:** Note any persistent JS errors across pages
- [ ] **Network errors:** Note any 4xx/5xx responses in Network tab
- [ ] **Loading states:** Pages show skeletons/spinners, not blank
- [ ] **Error states:** API failure shows error message, not crash
- [ ] **Navigation:** All sidebar links work, active state highlights correctly
- [ ] **Browser back/forward:** Navigation history works

---

## Bug Tracking

| # | Page | Issue | Severity | Screenshot | Status |
|---|------|-------|----------|------------|--------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

**Severity:** P0 (broken/crash), P1 (broken feature), P2 (visual/UX), P3 (minor/cosmetic)

---

## Known Issues Going In

These were identified from Grafana logs — verify status during testing:

| Issue | Expected Status |
|-------|----------------|
| PDF generation failing (JSON parse) | Should be fixed if `a938581` deployed |
| `platform_submit_report` bad params from agents | LLM may still send invalid enums |
| Missing SKILL.md for `12fa-three-loop@3.0.0` | Will show empty skill instructions |
| Composio JIRA not resolving for agent 138 | Expected — API key issue |
| FutureAGI worker scoring timeout | Fixed (300s timeout, deploy pending) |
| Credential resolver noise in logs | Fixed (downgraded to DEBUG, deploy pending) |
| Agent deletion orphaned tasks | Fixed (board_tasks query, deploy pending) |
| Workflow recipe `created_by` null | NOT YET FIXED — test this |

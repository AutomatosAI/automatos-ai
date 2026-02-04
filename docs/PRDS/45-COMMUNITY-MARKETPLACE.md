# PRD: Community Marketplace - Unified Discovery & Onboarding Hub

## Introduction

Transform Automatos onboarding by creating a unified Community Marketplace that acts as the primary entry point for new users. Instead of landing on an empty dashboard, users like Patricia discover pre-built Agents, Recipes, Tools, LLMs, and Skills they can install with one click. The marketplace replaces scattered tool/agent management with a centralized "App Store" experience where users browse featured/popular items, install dependencies automatically, and get clones to customize.

**Current Pain Points:**
- New users don't know where to start (empty dashboard paralysis)
- Tools marketplace buried in `/tools` page
- No way to share/discover agent configurations
- Templates (Recipes) hidden, underutilized
- LLM selection limited, not discoverable

**Vision:**
- Patricia signs up → lands on Marketplace homepage
- Browses Featured section → finds "Personal Assistant" agent
- Clicks "Install" → agent + Gmail + Calendar auto-added to workspace
- Browses Recipes → finds "Daily Work Review" → installs with one click
- Gets notification when marketplace items have updates
- Shares her customized "Sales Agent" back to community

## Goals

- **Reduce time-to-value for new users from hours to minutes** (Patricia productive in <5 minutes)
- **Enable community-driven content sharing** (users share agents/recipes they build)
- **Centralize discovery** (single marketplace for Tools, Agents, Recipes, LLMs, Skills)
- **Simplify onboarding** ("5-year-old test" - dead simple UX)
- **Auto-resolve dependencies** (install Recipe → auto-add Agent + Tools)
- **Maintain security** (approval queue prevents spam, admin-only skills)

## User Stories

### Phase 1: Core Marketplace Infrastructure

#### US-001: Create marketplace database schema
**Description:** As a developer, I need database tables to store marketplace items so users can browse and install shared content.

**Acceptance Criteria:**
- [ ] Create `marketplace_items` table with fields: id, type (agent/recipe/skill/llm), name, description, creator_name, icon, category, tags[], install_count, is_featured, is_approved, created_at, updated_at, metadata (JSONB)
- [ ] Create `marketplace_installs` table to track: user_id, item_id, installed_at, version
- [ ] Create `marketplace_submissions` table for approval queue: item_id, submitted_by, status (pending/approved/rejected), reviewed_by, reviewed_at
- [ ] Add database migration scripts
- [ ] Typecheck passes

#### US-002: Create marketplace API endpoints
**Description:** As a frontend developer, I need REST APIs to fetch, filter, and install marketplace items.

**Acceptance Criteria:**
- [ ] `GET /api/marketplace/items?type=agents&category=productivity` - list items with filters
- [ ] `GET /api/marketplace/items/:id` - get item details with dependencies
- [ ] `POST /api/marketplace/items/:id/install` - clone item to user workspace
- [ ] `GET /api/marketplace/featured` - get featured items for homepage
- [ ] `GET /api/marketplace/updates` - check for updates to installed items
- [ ] `POST /api/marketplace/submit` - submit item for approval
- [ ] All endpoints return proper error codes (400/404/500)
- [ ] Typecheck passes

#### US-003: Build marketplace homepage with tabs
**Description:** As Patricia (new user), I want to see a marketplace homepage with tabs so I can easily discover available items.

**Acceptance Criteria:**
- [ ] Create `/marketplace` route in Next.js
- [ ] Homepage shows hero section with search bar
- [ ] Tab navigation: Tools | Agents | Recipes | LLMs | Skills
- [ ] Featured section displays 6-8 curated items with install counts
- [ ] "Most Popular" section shows top 10 by install count
- [ ] Each marketplace card shows: icon, name, creator, install count, 1-line description
- [ ] Clicking card opens detail modal
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 2: Tools Tab (Migration)

#### US-004: Move Composio marketplace to main marketplace
**Description:** As a user, I want to find all Composio tools in the main marketplace so I have one place to discover integrations.

**Acceptance Criteria:**
- [ ] Tools tab displays all available Composio apps (reuse existing `composio-apps-section.tsx` logic)
- [ ] Filters: All Apps, Developer, Productivity, Communication, CRM, Analytics
- [ ] Search bar filters by app name/description
- [ ] Each tool card shows: logo, name, description, "Connect" or "Connected" badge
- [ ] Clicking "Connect" initiates OAuth flow (existing functionality)
- [ ] Connected tools show checkmark badge
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-005: Rename /tools to "My Tools"
**Description:** As a user, I want `/tools` page to show only MY connected tools so I can manage what I've already set up.

**Acceptance Criteria:**
- [ ] Update `/tools` page title to "My Tools"
- [ ] Remove marketplace/available apps section (moved to main marketplace)
- [ ] Show only connected tools with status: Active, Disconnected, Error
- [ ] Add "Browse Marketplace" button linking to `/marketplace?tab=tools`
- [ ] Each connected tool shows: name, status, last used, assigned agents count
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 3: Agents Tab (Sharing)

#### US-006: Add "Share to Marketplace" toggle to agent creation
**Description:** As Gerard (agent creator), I want to share my Communication Agent to the marketplace so other users can benefit from my configuration.

**Acceptance Criteria:**
- [ ] Add "Share to Marketplace" toggle to create-agent-modal.tsx
- [ ] When toggled ON, show fields: Public Name, Description (required), Category dropdown, Tags input
- [ ] Category options: Personal Assistant, Customer Support, DevOps, Social Media, Accounting, E-commerce, Content Creation, HR, Data Analysis, Custom
- [ ] Clicking "Create" with share=ON submits to approval queue (trusted users auto-publish)
- [ ] Show confirmation: "Agent created and submitted for marketplace approval"
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-007: Display agents in marketplace with install functionality
**Description:** As Patricia, I want to browse and install pre-built agents so I can get started quickly without configuration.

**Acceptance Criteria:**
- [ ] Agents tab shows grid of approved marketplace agents
- [ ] Filters: All Categories, Personal Assistant, Customer Support, DevOps, etc.
- [ ] Each agent card shows: icon, name, creator, LLM model badge, install count, assigned tools icons (max 4)
- [ ] Clicking card opens detail modal with: full description, assigned skills, assigned tools, "Install" button
- [ ] Install button clones agent to user workspace with "-copy" suffix if name exists
- [ ] Show success toast: "Communication Agent installed successfully"
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-008: Create 10 starter agents
**Description:** As Automatos admin, I need 10 pre-configured marketplace agents so new users have quality starting points.

**Acceptance Criteria:**
- [ ] Create Personal Assistant: Gmail, Google Calendar, Google Tasks | LLM: llama-3.3-70b-instruct
- [ ] Create Communication Manager: Slack, Gmail, Twilio | LLM: meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
- [ ] Create Customer Support: Zendesk, Intercom, Slack | LLM: anthropic/claude-3.5-sonnet
- [ ] Create DevOps Engineer: GitHub, Linear, Slack | LLM: anthropic/claude-3.5-sonnet
- [ ] Create Social Media Manager: Twitter, LinkedIn | LLM: meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
- [ ] Create Accounting Assistant: QuickBooks, Google Sheets | LLM: Qwen/QwQ-32B-Preview
- [ ] Create E-commerce Manager: Shopify, Stripe | LLM: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
- [ ] Create Content Creator: WordPress, Medium, Notion | LLM: anthropic/claude-3.5-sonnet
- [ ] Create HR Assistant: BambooHR, Slack | LLM: llama-3.1-8b-instruct
- [ ] Create Data Analyst: Google Sheets, SQL access | LLM: deepseek-ai/DeepSeek-R1
- [ ] All agents marked as featured and created_by: "Automatos Team"
- [ ] Typecheck passes

### Phase 4: Recipes Tab (Rename from Templates)

#### US-009: Rename templates to recipes in database and UI
**Description:** As a user, I want to understand that "Recipes" are reusable workflow blueprints so the naming is intuitive.

**Acceptance Criteria:**
- [ ] Update database: Rename `workflow_templates` table to `workflow_recipes` (migration script)
- [ ] Update API endpoints: `/api/templates/*` → `/api/recipes/*` (keep old endpoints for backward compatibility)
- [ ] Update frontend: templates-tab.tsx → recipes-tab.tsx
- [ ] Update all UI text: "Template" → "Recipe", "Templates" → "Recipes"
- [ ] Update workflows page tab label: "Templates" → "Recipes"
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-010: Add hybrid recipe schema (simple + complex)
**Description:** As a user, I want to create simple prompt-based recipes OR complex multi-step recipes so I can choose the right complexity for my needs.

**Acceptance Criteria:**
- [ ] Add `recipe_type` field to recipes table: 'simple' | 'complex'
- [ ] Simple recipe fields: name, description, agent_id, prompt (text), inputs[] (name, type, required), schedule (optional cron)
- [ ] Complex recipe fields: name, description, steps[], agents[], config{} (existing template_definition structure)
- [ ] Create recipe modal shows type selector: "Quick Recipe" (simple) vs "Advanced Recipe" (complex)
- [ ] Simple recipe form shows: prompt textarea, inputs builder, optional schedule picker
- [ ] Complex recipe form shows existing template builder UI
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-011: Display recipes in marketplace with dependency info
**Description:** As Patricia, I want to see what agent and tools a recipe requires so I know what I'm installing.

**Acceptance Criteria:**
- [ ] Recipes tab shows grid of approved marketplace recipes
- [ ] Each recipe card shows: icon, name, creator, install count, required agent name, estimated run time
- [ ] Clicking card opens detail modal with: description, required agent, required tools list, inputs needed, example outputs
- [ ] Detail modal shows "Dependencies" section: agent card + tool icons
- [ ] "Install" button shows if dependencies are met or pending
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 5: Dependency Auto-Resolution

#### US-012: Auto-clone dependencies when installing marketplace items
**Description:** As Patricia, I want all required dependencies automatically added to my workspace so I don't have to manually configure everything.

**Acceptance Criteria:**
- [ ] When installing recipe, detect required agent_id and tool dependencies
- [ ] If required agent not in workspace, auto-clone it with "-copy" suffix if name conflicts
- [ ] If required tools not in workspace, add them to workspace (not connected yet)
- [ ] Show success modal: "Daily Work Review installed successfully • 3 items added: Communication Agent, Linear, Slack"
- [ ] Modal lists all added items with icons
- [ ] If tools need connection, show warning badge: "Connect Linear and Slack to use this recipe"
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

#### US-013: Show connection warnings when running recipes with unconnected tools
**Description:** As Patricia, I want clear guidance when I try to run a recipe but haven't connected required tools yet.

**Acceptance Criteria:**
- [ ] When user clicks "Run" on recipe/workflow, check if required tools are connected
- [ ] If tools not connected, show blocking modal: "Connect Required Tools"
- [ ] Modal lists unconnected tools with "Connect Now" buttons
- [ ] Clicking "Connect Now" opens OAuth flow for that tool
- [ ] After successful connection, return user to recipe run screen
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 6: LLMs Tab (Metadata Only)

#### US-014: Seed LLM marketplace with metadata from AIML/Together
**Description:** As a developer, I need to populate the LLM marketplace with 400+ models so users can discover available options.

**Acceptance Criteria:**
- [ ] Create seed script to populate marketplace_items with type='llm'
- [ ] For each LLM model, store: provider (aiml/together), model_id, display_name, description, pricing_per_1m_tokens, context_window, capabilities[] (text/vision/code), is_recommended
- [ ] Seed top 50 models from AIML API: https://docs.aimlapi.com/api-overview/model-list
- [ ] Seed top 50 models from Together API: https://docs.together.ai/docs/models
- [ ] Mark recommended models: llama-3.3-70b-instruct, claude-3.5-sonnet, gpt-4o-mini, deepseek-r1
- [ ] Typecheck passes

#### US-015: Display LLMs in marketplace with filtering
**Description:** As a user, I want to browse available LLM models so I can choose the right one for my agents.

**Acceptance Criteria:**
- [ ] LLMs tab shows grid of LLM models
- [ ] Filters: All Providers, AIML, Together | All Capabilities, Text, Vision, Code | Price Range slider
- [ ] Each LLM card shows: provider logo, model name, pricing, context window, capability badges
- [ ] Clicking card opens detail modal: full description, pricing breakdown, use cases, example prompts
- [ ] "Use in Agent" button links to agent creation with this LLM pre-selected
- [ ] No install functionality (LLMs selected during agent creation, not installed to workspace)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 7: Skills Tab (Admin Only)

#### US-016: Create admin-only skill submission interface
**Description:** As Automatos admin, I need to add skills to the marketplace so users can discover and install them safely.

**Acceptance Criteria:**
- [ ] Skills tab shows grid of approved marketplace skills
- [ ] Only admins see "Add Skill to Marketplace" button
- [ ] Admin modal allows: upload skill files (GitHub import), name, description, category, tags
- [ ] Skill detail modal shows: README, skill parameters, example usage, security note: "Verified by Automatos"
- [ ] Users can install skills to workspace (copies skill files to user's workspace)
- [ ] Non-admin users cannot submit skills (button hidden, API returns 403)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 8: Approval Queue & Moderation

#### US-017: Build hybrid approval queue (auto-publish trusted, queue new users)
**Description:** As Automatos admin, I want to review submissions from new users while auto-publishing from trusted users to balance quality and speed.

**Acceptance Criteria:**
- [ ] Define "trusted user": has 5+ approved marketplace items OR admin-flagged as trusted
- [ ] When trusted user submits item, auto-approve and publish immediately
- [ ] When new user submits item, create entry in marketplace_submissions with status='pending'
- [ ] Create admin approval page: `/admin/marketplace/queue`
- [ ] Queue page shows pending submissions with: preview, duplicate check results, "Approve" / "Reject" buttons
- [ ] Duplicate check: if item name is 90%+ similar to existing item, flag as "Possible Duplicate"
- [ ] Approved items move to marketplace_items with is_approved=true
- [ ] Rejected items notify submitter with optional reason
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 9: Update Notifications

#### US-018: Track marketplace item versions and notify users of updates
**Description:** As Patricia, I want to know when marketplace items I installed have new versions so I can choose to update.

**Acceptance Criteria:**
- [ ] Add version field to marketplace_items (semantic versioning: "1.0.0")
- [ ] When creator re-submits existing item, increment version (1.0.0 → 1.1.0)
- [ ] Track installed version in marketplace_installs table
- [ ] Create notification API: `GET /api/notifications/marketplace-updates` returns items with newer versions
- [ ] Show notification bell badge with count of available updates
- [ ] Notification center shows: "Communication Agent v2.0 available" with "View Changes" link
- [ ] Detail modal shows changelog and "Update Now" button (re-clones latest version with new name)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### Phase 10: Install Tracking

#### US-019: Display install counts on marketplace items
**Description:** As a user, I want to see how many people installed each marketplace item so I can gauge popularity and quality.

**Acceptance Criteria:**
- [ ] Increment install_count in marketplace_items when user installs
- [ ] Display install count on all marketplace cards: "1,234 installs"
- [ ] Sort "Most Popular" section by install_count DESC
- [ ] Format large numbers: 1.2k, 45k, 1.2M
- [ ] Install count updates in real-time (or cache for 5 minutes for performance)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

## Functional Requirements

### Marketplace Infrastructure
- FR-1: The system must provide a single `/marketplace` route with tabs for Tools, Agents, Recipes, LLMs, Skills
- FR-2: The marketplace homepage must display Featured items (curated by admins) and Most Popular items (sorted by install count)
- FR-3: All marketplace items must be searchable by name, description, and tags
- FR-4: When a user installs a marketplace item, the system must clone it (not reference) to their workspace

### Dependency Management
- FR-5: When installing a Recipe, the system must automatically clone required Agent and add required Tools to workspace
- FR-6: If required tools are not connected, the system must show a warning and block recipe execution until tools are connected
- FR-7: If cloned item name conflicts with existing workspace item, the system must append "-copy" suffix

### Sharing & Approval
- FR-8: Users can submit Agents and Recipes to marketplace via "Share to Marketplace" toggle
- FR-9: Trusted users (5+ approved items OR admin-flagged) have submissions auto-published
- FR-10: New users have submissions queued for admin approval at `/admin/marketplace/queue`
- FR-11: Skills can ONLY be added to marketplace by Automatos admins (security requirement)

### Tools Migration
- FR-12: The existing Composio tools marketplace must move to `/marketplace?tab=tools`
- FR-13: The `/tools` page must be renamed "My Tools" and show only connected tools with management actions
- FR-14: Tool connection flow (OAuth) must remain unchanged

### LLMs
- FR-15: LLM marketplace must display metadata for 400+ models from AIML and Together APIs
- FR-16: LLMs are NOT installed to workspace; they are selected during agent creation
- FR-17: LLM cards must show: provider, model name, pricing per 1M tokens, context window, capability badges

### Versioning & Updates
- FR-18: Marketplace items must use semantic versioning (1.0.0, 1.1.0, 2.0.0)
- FR-19: When a creator updates a marketplace item, users who installed it receive in-app notification
- FR-20: Users can view changelog and choose to "Update Now" (re-clone latest version)

### Recipes (Renamed from Templates)
- FR-21: Templates must be renamed to Recipes throughout database, API, and UI
- FR-22: Recipes support two types: Simple (prompt + agent + inputs) and Complex (multi-step workflow definition)
- FR-23: Workflows consume Recipes and can be run manually (scheduling is future work)

## Non-Goals (Out of Scope for Pilot)

- ❌ **Automatic updates** - No auto-update of installed items; users manually choose to update
- ❌ **Revenue sharing** - No paid marketplace items; all free for pilot
- ❌ **Community voting** - No upvote/downvote system; only install count and admin curation
- ❌ **Recipe scheduling** - Workflows run manually; cron scheduling is future work
- ❌ **LLM API integration** - Only metadata displayed; actual AIML/Together API integration happens later
- ❌ **Skill sandboxing** - Skills run with full permissions; sandboxing is future security enhancement
- ❌ **Multi-language support** - English only for pilot
- ❌ **Mobile responsive marketplace** - Desktop-first; mobile optimization later
- ❌ **Advanced analytics** - No usage dashboards for marketplace creators; just install count
- ❌ **Marketplace categories beyond basic filtering** - No nested categories or custom taxonomies

## Design Considerations

### Visual Design
- **Marketplace homepage hero:** Large search bar, tagline "Discover agents, recipes, and tools to automate your work"
- **Tab navigation:** Fixed tabs at top (Tools | Agents | Recipes | LLMs | Skills) with active state highlighting
- **Marketplace cards:** Consistent card design across all tabs with icon, name, creator, install count, primary action button
- **Dependency visualization:** Show required agent + tools as small cards/badges in recipe detail modal
- **Success modals:** Use friendly language: "🎉 Communication Agent installed successfully! You added 3 items to your workspace."

### UX Flow
1. **New user Patricia signs up** → redirected to `/marketplace` (not empty dashboard)
2. **Browses Featured section** → sees "Personal Assistant" agent
3. **Clicks card** → detail modal opens with description, assigned tools, install count
4. **Clicks "Install"** → agent + Gmail + Calendar added to workspace
5. **Success modal shows** → "3 items added: Personal Assistant, Gmail, Google Calendar"
6. **Warning badge** → "Connect Gmail and Calendar to use this agent"
7. **Clicks "Connect Gmail"** → OAuth flow → returns to marketplace
8. **Explores Recipes tab** → finds "Daily Work Review" recipe
9. **Installs recipe** → auto-clones Communication Agent (dependency) + adds Linear + Slack
10. **Tries to run workflow** → blocked with "Connect Linear and Slack" modal

### Component Reuse
- **Tool cards:** Reuse existing `composio-apps-section.tsx` component for Tools tab
- **Agent cards:** Reuse styling from `agent-roster.tsx` but with marketplace-specific data
- **Modal pattern:** Reuse existing modal components with new marketplace content
- **Badge component:** Reuse for capabilities, categories, connection status

## Technical Considerations

### Database Migrations
- Create `marketplace_items` table (type, metadata JSONB, install_count, is_featured, is_approved)
- Create `marketplace_installs` table (user_id, item_id, version, installed_at)
- Create `marketplace_submissions` table (item_id, status, reviewed_by, reviewed_at)
- Rename `workflow_templates` → `workflow_recipes`
- Add `version` column to marketplace_items
- Add `is_trusted` flag to users table

### API Structure
```
GET    /api/marketplace/items?type=agents&category=productivity
GET    /api/marketplace/items/:id
POST   /api/marketplace/items/:id/install
GET    /api/marketplace/featured
GET    /api/marketplace/updates
POST   /api/marketplace/submit
GET    /api/admin/marketplace/queue (admin only)
POST   /api/admin/marketplace/approve/:id (admin only)
POST   /api/admin/marketplace/reject/:id (admin only)
```

### Frontend Routes
- `/marketplace` - Main marketplace with tabs
- `/marketplace?tab=tools` - Tools tab (deep link)
- `/marketplace?tab=agents` - Agents tab
- `/marketplace?tab=recipes` - Recipes tab
- `/marketplace?tab=llms` - LLMs tab
- `/marketplace?tab=skills` - Skills tab
- `/tools` - My Tools (connected tools only)
- `/admin/marketplace/queue` - Approval queue (admin only)

### Performance Considerations
- **Cache marketplace items** - Redis cache for 5 minutes to reduce DB load
- **Lazy load tabs** - Only fetch tab data when user clicks tab
- **Paginate large lists** - Load 50 items at a time with infinite scroll
- **Optimize install count queries** - Use database triggers to update install_count efficiently

### Security
- **Admin-only skill marketplace** - Skills can only be added by admins (prevent malicious code)
- **Approval queue** - New users' submissions reviewed before publishing
- **OAuth scopes validation** - Ensure tools request minimal required scopes
- **Input sanitization** - Sanitize all marketplace item descriptions (prevent XSS)

### Dependencies
- Existing agent creation flow (`create-agent-modal.tsx`)
- Existing tool connection flow (OAuth with Composio)
- Existing recipe/template infrastructure (`templates-tab.tsx`)
- Existing skill management system (GitHub import)

## Success Metrics

### User Onboarding
- **Primary:** 80% of new users install at least 1 marketplace item within 5 minutes of signup
- **Time-to-first-value:** Reduce from 45 minutes (manual setup) to <5 minutes (marketplace install)
- **Activation rate:** 60% of new users run their first workflow within 10 minutes

### Engagement
- **Marketplace visits:** 70% of active users visit marketplace weekly
- **Install rate:** Average 3.5 marketplace items installed per user in first week
- **Sharing rate:** 10% of users share at least 1 agent/recipe to marketplace within 30 days

### Quality
- **Approval time:** <24 hours median time from submission to approval
- **Duplicate rate:** <5% of submissions flagged as duplicates
- **Connection completion:** 80% of users who install recipe complete required tool connections

### Growth
- **Marketplace catalog size:** 100+ community-contributed items within 3 months
- **Featured items CTR:** 40% of users click at least 1 featured item
- **Update adoption:** 30% of users update within 7 days of update notification

## Open Questions

1. **Recipe execution logs:** Should users see execution logs for recipes in the marketplace (preview before install)?
2. **Versioning UI:** Show full changelog modal or just version number + "What's new" summary?
3. **Install count privacy:** Should creators see WHO installed their items or just total count?
4. **Dependency versioning:** If Communication Agent v1.0 is required but user has v2.0, allow or force v1.0 clone?
5. **Marketplace analytics for creators:** Should we show "Your items" dashboard with install trends?
6. **Recipe inputs pre-fill:** Can we detect user's Slack channels / Linear teams to pre-fill recipe inputs?
7. **Uninstall flow:** Should users be able to "uninstall" marketplace items or just delete like normal workspace items?
8. **Featured rotation:** How often should featured items rotate? Weekly? Monthly?
9. **Search ranking:** Prioritize by install count, recency, or admin curation?
10. **Mobile marketplace:** Is mobile web access required for pilot or desktop-only OK?

---

**Document Version:** 1.0
**Created:** 2026-01-30
**Author:** Automatos Product Team
**Status:** Draft - Ready for Review

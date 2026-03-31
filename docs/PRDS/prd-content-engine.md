# PRD: Automatos Autonomous Content Engine

## Introduction

Turn the blog backend (built in `ralph/blog-widget-backend`) into an autonomous content pipeline. Agents research, write, review, and design cover art for blog posts on a schedule. Humans approve with a single click. The blog gets its own top-level nav item in the dashboard, and a new `platform_update_blog_post` tool lets agents edit drafts collaboratively. This PRD keeps storage simple (PostgreSQL only for v1) and defers social distribution to a future PRD.

## Goals

- Ship a complete agent-driven blog pipeline: write, review, design, queue for approval
- Add `platform_update_blog_post` tool so agents can collaborate on drafts
- Move Blog management out of Activity into its own top-level nav route
- Provide a sample playbook template ("Weekly Blog Pipeline") in the marketplace
- Define the QUILL (writer), EDITOR (reviewer), and CANVAS (designer) agent roles
- Human publishes with one click in the existing Blog Editor — no new approval UI needed

## User Stories

### US-001: Add platform_update_blog_post tool
**Description:** As an agent, I need to update an existing draft blog post so that multiple agents can collaborate on the same article (e.g. editor improves what writer drafted).

**Acceptance Criteria:**
- [ ] Add `platform_update_blog_post` ActionDefinition in `actions_blog.py` with parameters: post_id (string, required), title (string, optional), content (string, optional), excerpt (string, optional), tags (array, optional), category (string, optional), cover_image_url (string, optional)
- [ ] Add handler `update_blog_post` in `handlers_blog.py` that calls `BlogService.update_post()` — only updates fields that are provided, workspace-scoped
- [ ] Register in `platform_executor.py` `_handlers` dict
- [ ] Add Tier 2 keywords in `auto.py`: "update blog post", "edit blog post", "revise article", "improve draft"
- [ ] Handler returns `{ success, post_id, title, slug, status }`
- [ ] Typecheck passes

### US-002: Add platform_get_blog_post tool
**Description:** As an agent, I need to read the full content of a blog post so I can review or edit it.

**Acceptance Criteria:**
- [ ] Add `platform_get_blog_post` ActionDefinition with parameters: post_id (string, optional), slug (string, optional) — at least one required
- [ ] Handler calls `BlogService.get_post()` or `get_post_by_slug()`, returns full post including content
- [ ] Register in executor and auto.py keywords: "read blog post", "get blog post", "show blog draft", "fetch article"
- [ ] Typecheck passes

### US-003: Move Blog to top-level nav route
**Description:** As a workspace owner, I need Blog as its own page in the main sidebar navigation so it's easy to find and manage separately from Activity.

**Acceptance Criteria:**
- [ ] Create `frontend/app/blog/page.tsx` route that renders a new `BlogManagementPage` component
- [ ] Add "Blog" link to the main sidebar navigation (find the nav config — likely in `components/layout/sidebar.tsx` or similar) with `BookOpen` icon from lucide-react
- [ ] Position after "Knowledge" / "Documents" in the nav order
- [ ] `BlogManagementPage` reuses the existing `ActivityBlog` component (move or re-export from `components/activity/activity-blog.tsx`)
- [ ] Remove the Blog tab from `activity-page.tsx` `TAB_DEFS` array
- [ ] Deep-link `/blog` works
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-004: Create QUILL agent template
**Description:** As a workspace owner, I need a pre-configured "QUILL" writer agent template so I can deploy a blog writer without manual setup.

**Acceptance Criteria:**
- [ ] Create agent template (marketplace seed or SKILL.md) for QUILL with: name "QUILL", type "worker", system prompt focused on blog writing (research topic, write engaging markdown articles, use `platform_publish_blog_post` with `publish_immediately=false` to create drafts, include SEO-friendly titles and excerpts)
- [ ] Template assigns tools: `platform_publish_blog_post`, `platform_list_blog_posts`, `platform_search_memory` (for RAG research)
- [ ] Template includes suggested heartbeat schedule: twice weekly
- [ ] Store as `orchestrator/modules/agents/templates/quill_writer.json` or equivalent pattern used by existing templates
- [ ] Typecheck passes

### US-005: Create EDITOR agent template
**Description:** As a workspace owner, I need an "EDITOR" reviewer agent that reads drafts and improves them.

**Acceptance Criteria:**
- [ ] Create agent template for EDITOR with: name "EDITOR", type "worker", system prompt focused on editorial review (read drafts via `platform_get_blog_post`, check grammar/flow/accuracy, improve content via `platform_update_blog_post`, never publish — only improve drafts)
- [ ] Template assigns tools: `platform_list_blog_posts`, `platform_get_blog_post`, `platform_update_blog_post`
- [ ] Store alongside QUILL template
- [ ] Typecheck passes

### US-006: Create CANVAS agent template
**Description:** As a workspace owner, I need a "CANVAS" designer agent that generates cover images for blog posts.

**Acceptance Criteria:**
- [ ] Create agent template for CANVAS with: name "CANVAS", type "worker", system prompt focused on generating cover art (read post title and excerpt, generate image description, use image generation tool if available, update post with `platform_update_blog_post` setting `cover_image_url`)
- [ ] Template assigns tools: `platform_list_blog_posts`, `platform_get_blog_post`, `platform_update_blog_post`, and image generation tool (Composio DALL-E or workspace image tool if available)
- [ ] If no image generation tool is available, agent should note in the post that a cover image is needed (update excerpt or add a tag "needs-cover-image")
- [ ] Store alongside other templates
- [ ] Typecheck passes

### US-007: Create "Weekly Blog Pipeline" playbook template
**Description:** As a workspace owner, I need a ready-made playbook that orchestrates the full blog pipeline so I can enable autonomous publishing with one click.

**Acceptance Criteria:**
- [ ] Create playbook template with name "Weekly Blog Pipeline" and 4 sequential steps:
  - Step 1: QUILL — "Research a trending topic in {category} and write a blog post draft. Use platform_search_memory to check what topics we've covered before. Create the draft with platform_publish_blog_post(publish_immediately=false)."
  - Step 2: EDITOR — "Review the latest draft blog post. Use platform_list_blog_posts(status=draft) to find it, platform_get_blog_post to read it, and platform_update_blog_post to improve it. Focus on clarity, engagement, and SEO."
  - Step 3: CANVAS — "Find the latest draft blog post and design a cover image for it. Update the post with the cover_image_url."
  - Step 4: System — "Create a Board task: 'Review & Publish: {post_title}' assigned to workspace owner. The task description includes a link to the Blog page to review and publish."
- [ ] Template includes schedule configuration: cron for Tuesday and Friday at 09:00 UTC (configurable)
- [ ] Template includes a `category` parameter that users fill in when installing (e.g. "AI", "Automation", "Business")
- [ ] Store as a marketplace-ready playbook template
- [ ] Typecheck passes

### US-008: Task-to-publish integration
**Description:** As a workspace owner, when I complete a "Review & Publish" board task, I need a clear path to publish the blog post without hunting for it.

**Acceptance Criteria:**
- [ ] When the playbook pipeline creates the review task (Step 4), the task `description` includes: post title, excerpt preview, and a direct link to `/blog?edit={post_id}`
- [ ] The Blog page reads `?edit={post_id}` from URL params and auto-opens the BlogEditor for that post
- [ ] User reviews content in the editor, clicks "Publish" — existing flow, no new UI needed
- [ ] After publishing, user marks the Board task as complete (manual — no auto-complete in v1)
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

### US-009: Blog page deep-link and edit param support
**Description:** As a user clicking a task link, I need `/blog?edit={post_id}` to open the editor directly.

**Acceptance Criteria:**
- [ ] `BlogManagementPage` reads `edit` query param via `useSearchParams()`
- [ ] If `edit` param is present, auto-set `isEditorOpen=true` and `editingPostId` to the param value on mount
- [ ] Editor opens in the Sheet slide-over with the post loaded
- [ ] If the post_id is invalid or not found, show a toast error and stay on the blog list
- [ ] Typecheck passes
- [ ] Verify in browser using dev-browser skill

## Functional Requirements

- FR-1: `platform_update_blog_post` tool updates specified fields on a draft post, workspace-scoped, returns updated metadata
- FR-2: `platform_get_blog_post` tool returns full post content by ID or slug, workspace-scoped
- FR-3: Blog management lives at `/blog` as a top-level route with its own sidebar nav entry
- FR-4: Agent templates (QUILL, EDITOR, CANVAS) are JSON files that can be installed from the marketplace or applied manually
- FR-5: The "Weekly Blog Pipeline" playbook template orchestrates 4 steps sequentially with agent handoff
- FR-6: Step 4 of the pipeline creates a Board task with a deep-link to the blog editor
- FR-7: The blog editor auto-opens when navigated to with `?edit={post_id}` query param
- FR-8: Human publishes by clicking the existing "Publish" button in the Blog Editor — no new approval UI
- FR-9: Blog posts remain in PostgreSQL only (no S3, no vectorisation in v1)
- FR-10: Each agent template includes tool assignments so agents are ready to work after installation

## Non-Goals

- No S3 storage or vectorisation of blog posts (future PRD — when we want blogs as RAG-searchable knowledge)
- No social distribution (LinkedIn, Reddit, X) — separate future PRD
- No auto-publish on task completion (human clicks Publish manually in v1)
- No image upload/hosting (agents provide external URLs for cover images)
- No editorial workflow UI (no approval queue, no diff view between drafts)
- No RSS/Atom feed generation
- No blog analytics beyond the existing view_count field
- No automatic topic selection (QUILL's prompt includes the category, but topic choice is up to the agent)

## Design Considerations

- Blog page reuses the existing `ActivityBlog` component — just re-mounted at a new route
- Agent templates follow whatever pattern existing marketplace templates use (check `orchestrator/modules/agents/templates/` or marketplace seed data)
- Playbook template follows the existing playbook/recipe structure with sequential steps
- The pipeline is opinionated but configurable: users can remove steps, change agents, adjust the schedule
- CANVAS agent is best-effort — if no image generation tool is connected, it gracefully degrades

## Technical Considerations

- `platform_update_blog_post` handler must only update fields that are explicitly provided (use `**kwargs` pattern from `BlogService.update_post`)
- Playbook Step 4 needs to create a Board task programmatically — verify `platform_create_task` tool exists and supports description with markdown/links
- Agent templates need to reference tool names that exist in the ActionRegistry — verify all tools are registered before shipping templates
- The `?edit={post_id}` deep-link pattern is consistent with how other pages handle deep-links (e.g. Activity page uses `?tab=` and `?task_id=`)
- Blog nav item should respect the same permission model as other nav items (workspace member access)

## Success Metrics

- End-to-end pipeline runs autonomously: QUILL drafts, EDITOR reviews, CANVAS designs, task created — with zero human intervention until the publish step
- Human can go from Board task notification to published post in under 30 seconds (click task link, review, click Publish)
- Pipeline produces 2 blog posts per week when scheduled
- At least 1 complete pipeline run verified on automatos.app within first week of deployment

## Open Questions

- Should QUILL use web search (Composio) for topic research, or only workspace memory/RAG? (Recommendation: both, if web search tool is available)
- Should the pipeline playbook be auto-installed for new workspaces, or only available in the marketplace? (Recommendation: marketplace, not auto-installed)
- When we add S3 + vectorisation in v2, should we backfill existing posts? (Recommendation: yes, one-time migration)
- Should the Board task auto-complete when the post is published? (Would need a webhook/trigger on status change — defer to v2)

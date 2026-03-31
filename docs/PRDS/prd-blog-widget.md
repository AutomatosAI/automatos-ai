# PRD: Automatos Blog Widget

## Introduction

Add an embeddable blog widget to the Automatos Widget SDK that lets users display AI-agent-authored blog posts on any website. Agents research, write, and publish posts via a platform tool. Users embed a `<script>` tag and get a self-updating blog powered by their Automatos workspace. The automatos.app landing site is customer #1 — dog-fooding the widget to showcase agent-generated content.

This turns Automatos from a tool users log into, to a platform that's visible on their customers' websites. Every blog post published reinforces the value loop: agents produce content, content drives traffic, traffic justifies the subscription.

## Goals

- Let agents publish blog posts via `platform_publish_blog_post` tool
- Provide an embeddable blog widget (`@automatos/blog-widget`) matching the existing chat widget architecture
- Expose a public Blog API for fetching published posts (no auth required for reads)
- Ship blog management UI in the Automatos dashboard (create, edit, schedule, unpublish)
- Deploy the blog widget on automatos.app as the first integration
- Support theming, layout variants, and SEO-friendly rendering

## User Stories

### US-001: Blog posts database schema
**Description:** As a developer, I need to store blog posts with metadata so they can be queried and displayed.

**Acceptance Criteria:**
- [ ] Create `blog_posts` table: id (UUID PK), workspace_id (FK), author_agent_id (nullable FK), author_name, title, slug (unique per workspace), excerpt (max 300 chars), content (text, markdown), cover_image_url (nullable), tags (ARRAY text), category (varchar), status (enum: draft/scheduled/published/archived), published_at (nullable timestamp), scheduled_for (nullable timestamp), seo_title (nullable), seo_description (nullable), reading_time_minutes (int), view_count (default 0), created_at, updated_at
- [ ] Add index on (workspace_id, status, published_at DESC) for listing queries
- [ ] Add unique constraint on (workspace_id, slug)
- [ ] Alembic migration generates and runs successfully
- [ ] Typecheck passes

### US-002: Blog API — public read endpoints
**Description:** As a widget consumer, I need public endpoints to fetch published posts without authentication so the widget can render on any site.

**Acceptance Criteria:**
- [ ] `GET /api/widgets/blog/posts?workspace_id={id}` — returns paginated published posts (title, slug, excerpt, cover_image_url, tags, category, author_name, published_at, reading_time_minutes). Default 10 per page, max 50. Sorted by published_at DESC
- [ ] `GET /api/widgets/blog/posts/{slug}?workspace_id={id}` — returns full post (includes content as HTML-rendered markdown). Increments view_count
- [ ] `GET /api/widgets/blog/categories?workspace_id={id}` — returns distinct categories with post counts
- [ ] `GET /api/widgets/blog/tags?workspace_id={id}` — returns distinct tags with post counts
- [ ] All endpoints require `workspace_id` query param (no auth, public reads)
- [ ] Responses include CORS headers for cross-origin widget embedding
- [ ] Empty workspace returns empty array, not 404
- [ ] Typecheck passes

### US-003: Blog API — authenticated management endpoints
**Description:** As a workspace owner or agent, I need CRUD endpoints to manage blog posts.

**Acceptance Criteria:**
- [ ] `POST /api/blog/posts` — create post (draft by default). Accepts: title, content, excerpt, cover_image_url, tags, category, status, scheduled_for, seo_title, seo_description. Auto-generates slug from title, auto-calculates reading_time_minutes
- [ ] `PUT /api/blog/posts/{post_id}` — update post fields. Re-slugifies if title changes and post is still draft
- [ ] `DELETE /api/blog/posts/{post_id}` — soft delete (sets status=archived)
- [ ] `POST /api/blog/posts/{post_id}/publish` — sets status=published, published_at=now()
- [ ] `POST /api/blog/posts/{post_id}/unpublish` — sets status=draft, clears published_at
- [ ] `GET /api/blog/posts` — list all posts in workspace (all statuses), paginated, filterable by status/category/tag
- [ ] `GET /api/blog/posts/{post_id}` — full post detail (includes draft content)
- [ ] All endpoints require workspace auth (Clerk JWT or API key)
- [ ] Typecheck passes

### US-004: Platform tool — platform_publish_blog_post
**Description:** As an agent, I need a tool to write and publish blog posts to my workspace so I can automate content creation.

**Acceptance Criteria:**
- [ ] Register `platform_publish_blog_post` in ActionRegistry with parameters: title (required), content (required, markdown), excerpt (optional, auto-generated from first 300 chars of content if omitted), tags (optional array), category (optional), cover_image_url (optional), publish_immediately (boolean, default true)
- [ ] Handler in PlatformActionExecutor calls the Blog API internally (not HTTP — direct service call)
- [ ] If publish_immediately=true, post is created with status=published. Otherwise status=draft
- [ ] Returns: { post_id, slug, status, url } where url is the public widget URL for the post
- [ ] Tool appears in agent tool lists when assigned
- [ ] Add Tier 2 keywords in auto.py: "blog", "publish", "article", "write post", "blog post"
- [ ] Typecheck passes

### US-005: Platform tool — platform_list_blog_posts
**Description:** As an agent, I need to list existing blog posts so I can check what's already published before writing new content.

**Acceptance Criteria:**
- [ ] Register `platform_list_blog_posts` in ActionRegistry with parameters: status (optional, default "published"), limit (optional, default 10), category (optional)
- [ ] Returns array of: { post_id, title, slug, status, published_at, category, tags }
- [ ] Typecheck passes

### US-006: Blog widget package — @automatos/blog-widget
**Description:** As a website owner, I need an embeddable blog widget that displays my workspace's published posts.

**Acceptance Criteria:**
- [ ] New package at `packages/blog-widget/` in the widget SDK monorepo
- [ ] Shadow DOM isolated (same pattern as chat-widget)
- [ ] Renders a post listing view: grid or list layout, each card shows cover image, title, excerpt, author, date, reading time, tags
- [ ] Clicking a card opens a post detail view (slide-over or inline expand) with full rendered content
- [ ] Supports `layout` config: "grid" (2-3 columns), "list" (single column), "featured" (hero + grid), "minimal" (titles only)
- [ ] Supports `postsPerPage` config (default 6)
- [ ] Supports `category` and `tag` filters in config (restrict which posts show)
- [ ] Loading skeleton while fetching
- [ ] Empty state when no posts exist
- [ ] Responsive: stacks to single column on mobile
- [ ] Theming via same CSS custom properties as chat widget (--aw-primary, --aw-bg, etc.)
- [ ] Typecheck passes

### US-007: Blog widget — post detail rendering
**Description:** As a reader, I need to view full blog post content with proper typography and formatting.

**Acceptance Criteria:**
- [ ] Markdown content rendered to HTML (reuse/extend existing markdown parser from chat-widget)
- [ ] Support: headings, paragraphs, bold, italic, links, code blocks with syntax highlighting, images, blockquotes, lists, horizontal rules, tables
- [ ] Back button returns to post listing
- [ ] Post header shows: title, author, published date, reading time, tags, cover image
- [ ] Responsive typography (larger on desktop, comfortable on mobile)
- [ ] Links open in new tab (target="_blank" rel="noopener")
- [ ] Typecheck passes

### US-008: Loader integration — widget type "blog"
**Description:** As a developer, I need to initialize the blog widget via the same script tag loader used for chat.

**Acceptance Criteria:**
- [ ] `AutomatosConfig.widget` type extended to `'chat' | 'blog'`
- [ ] `AutomatosWidget.init({ widget: "blog", ... })` creates a BlogWidget instance
- [ ] Blog-specific config options: `layout`, `postsPerPage`, `category`, `tag`, `containerSelector` (CSS selector to mount into, instead of floating FAB)
- [ ] Blog widget mounts into `containerSelector` if provided, otherwise creates a full-page overlay
- [ ] Loader package updated to handle both widget types
- [ ] Command queue replay works for blog widget
- [ ] Typecheck passes

### US-009: React wrapper for blog widget
**Description:** As a React developer, I need a React component to embed the blog widget.

**Acceptance Criteria:**
- [ ] `<AutomatosBlog>` component in `@automatos/react` package
- [ ] Props mirror config: apiKey, layout, postsPerPage, category, tag, theme, themeOverrides
- [ ] Renders into a container div (not floating — inline in the page)
- [ ] Cleans up on unmount
- [ ] Typecheck passes

### US-010: Landing site blog page — dog-food integration
**Description:** As the Automatos team, we need a /blog page on automatos.app that uses our own blog widget to prove the product works.

**Acceptance Criteria:**
- [ ] New `/blog` route in the landing site React Router
- [ ] Page header with title and description
- [ ] Blog widget embedded in "featured" layout via script tag or direct import
- [ ] Uses the Automatos workspace's published posts
- [ ] Individual post view at `/blog/:slug` (deep-linkable)
- [ ] Blog link added to Navbar and Footer
- [ ] SEO meta tags for blog listing and individual posts
- [ ] Typecheck passes
- [ ] Verify in browser

### US-011: Dashboard blog management UI
**Description:** As a workspace owner, I need a UI to manage blog posts (create, edit, review agent drafts, publish/unpublish).

**Acceptance Criteria:**
- [ ] New "Blog" tab in the Activity section of the Automatos dashboard
- [ ] Post list table: title, status badge, author, published date, views, actions (edit/publish/unpublish/delete)
- [ ] Post editor: title, content (markdown textarea with preview), excerpt, cover image URL, category dropdown, tags input, SEO fields, status toggle
- [ ] Draft/Published/Archived filter tabs
- [ ] Preview button opens post as it would appear in the widget
- [ ] Typecheck passes
- [ ] Verify in browser

## Functional Requirements

- FR-1: Blog posts are scoped to a workspace. One workspace's posts are never visible in another's widget
- FR-2: Slugs are auto-generated from title (kebab-case, deduplicated with numeric suffix)
- FR-3: Reading time calculated as word_count / 200 (rounded up to nearest minute)
- FR-4: Markdown rendered server-side for the public API (returns HTML), client-side in the widget as fallback
- FR-5: Cover images are URLs (not uploaded to Automatos — users provide external URLs or use workspace file URLs)
- FR-6: Published posts are publicly readable. No authentication for widget GET endpoints
- FR-7: Agents can only publish to their own workspace
- FR-8: The `platform_publish_blog_post` tool follows the 3-file pattern: platform_actions.py, platform_executor.py, auto.py keywords
- FR-9: Blog widget bundle target: <15KB gzipped (similar to chat widget)
- FR-10: Widget supports server-side rendering hints via `<noscript>` fallback with post titles/links for SEO crawlers

## Non-Goals

- No comment system (v1 is publish-only, not social)
- No multi-author workflows or editorial approval queue (agents publish directly, humans review in dashboard)
- No image upload/hosting (use external URLs)
- No analytics beyond view_count (no click tracking, heatmaps, etc.)
- No RSS/Atom feed generation (can add later)
- No scheduled publishing automation (scheduled_for field exists but cron job is out of scope for v1)
- No i18n or multi-language support
- No full-text search within blog posts

## Technical Considerations

- Blog widget reuses `@automatos/core` for API client, auth, and event bus
- Markdown parser in `@automatos/chat-widget` should be extracted to `@automatos/core` or a shared package so blog-widget can reuse it
- Public blog endpoints should be aggressively cached (Cache-Control: public, max-age=300 for listings, max-age=3600 for individual posts)
- Widget should lazy-load post detail content (don't fetch full markdown in listing response)
- Consider using Intersection Observer for infinite scroll on post listings
- The blog widget needs a different mount pattern than chat (inline container vs. floating FAB). The config option `containerSelector` determines where it renders

## Design Considerations

- Blog widget should feel native to the host site. Minimal chrome, maximum theme customization
- Post cards in grid layout: cover image top, title, excerpt, meta row (author, date, reading time)
- Post detail: full-width cover image hero, then centered content column (max-width ~720px)
- Typography should be excellent — this is a reading experience. Use the host site's font via `--aw-font` or default to a clean serif/sans pair
- Dark/light theme support via same `--aw-*` CSS variables as chat widget
- Loading skeletons should match the layout variant (grid skeleton vs. list skeleton)

## Success Metrics

- Blog widget renders published posts on automatos.app within 500ms of page load
- Agent can publish a blog post end-to-end (research, write, publish) without human intervention
- Widget bundle size <15KB gzipped
- At least 3 agent-authored posts published on automatos.app/blog within first week

## Open Questions

- Should the blog widget support pagination or infinite scroll? (Recommendation: pagination for v1, simpler)
- Should post URLs be handled by the widget (SPA-style) or should we generate static URLs that the host site routes? (Recommendation: widget handles it internally with `containerSelector`, host site can optionally add routes for SEO)
- Should we add an "edit in Automatos" link visible only to authenticated workspace owners viewing their own widget? (Nice for quick fixes)
- Should the widget preload the next page of posts for instant pagination?

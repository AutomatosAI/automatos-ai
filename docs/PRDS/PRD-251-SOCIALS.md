# PRD-251: Socials — any workspace makes on-brand social posts (video, images, fact cards, infographics), approves each one, sees it on the calendar, and publishes through its own Composio connections

> **Status:** DRAFT 2026-09-23 (owner: "OK write this as a PRD so any customer can use it, videos, images, facts, info graphics... Build it into Deliverables, alongside blog, report and templates for reporting... new TAB socials with content and approvals, connect it to the calendar") — grounded @ `test/customer-night` fd068763f (the integration branch, ahead of `main` af214bc7b; **re-check line numbers against `main` at build**).
>
> **Build kick-off (owner, 2026-09-23):** "we need to build and test". Built on its own worktree and branch, `feat/prd-251-socials` ← `main` af214bc7b (`<workspace>/.worktrees/automatos-ai/feat+prd-251-socials`), so the customer-night checkout and stack are never touched. **Owner answers:** built as a Ralph kit per wave that the owner launches; Socials on **all plans** (open question 1); the LinkedIn workaround is **workspace-scoped**, not deleted (open question 6); each wave is tested on **its own local stack** before merge (see "How it is built and tested"). Wave 0 anchors were re-verified on `main` on 2026-09-23.
>
> **Composio-first (owner, 2026-09-23, later that night):** "I don't want to reinvent the wheel... I just need my customers to connect a Composio tool and my agents use them", then "let's keep it simple for now, no one wants to hear me speak... but I want the same quality of videos as earlier".
> - **What Composio supplies:** every paid ingredient (AI footage, stills and premium voice) comes from a tool the customer connects in Composio, and their agents use it.
> - **What Automatos builds:** only what Composio can't supply:
>   - the video recipe, as a skill (D17);
>   - the renderer that assembles a finished video at the reference quality (D3);
>   - the Socials tab with approvals and the calendar;
>   - the guardrails (D16).
> - **Dropped:** the owner-recorded voice, and our own provider clients with the API-key route (D11, D12 and D15 as first written).
>
> **Wave 1 decisions (owner, 2026-09-23, after Wave 0 shipped as #782):**
> - **Push forward (owner, later on 2026-09-23):** Wave 1 builds now, stacked on Wave 0 (`feat/prd-251-socials`, #782), without waiting for Wave 0's merge. The owner tests both waves on the socials stack once the current test cycle finishes. Wave 0 then landed on `main` through PR #783 (`test/customer-night` → `main`, #782 closed as landed there), so a Wave 1 launched after it is cut from `main`, and its migration chains onto `f049_prd251_merge_heads`. This replaces the earlier "launch only after Wave 0 is tested and merged".
> - The renderer is its own Railway service from day one: **4 vCPU / 8 GB, two renders at once** overall, one at a time per workspace.
> - Monthly render quotas: **Basic 10 min, Pro 60 min, Business 240 min**, as config. Enterprise and the local edition have no quota until the owner sets one.
> - **Both editions** in Wave 1: the renderer ships as the optional compose profile `media` in the local edition.
> - **S4.4 (the `media` cost lane) moves into Wave 1**, because S1.8 books footage spend and needs the lane to exist.
> - **The agent layer moves into Wave 1 (owner, later on 2026-09-23: "reuse what exists").** Add brand-kit tools, Socials draft tools (S4.1), social formats in `generate_document` and a general image tool, the post gate (S3.5), built-in skills synced from `automatos-skills` with no push, and the Socials marketplace package. The package has two agents, Social Media Director and Brand Designer, and four playbooks: brand kit from your website, launch video, weekly social posts (S4.2) and image carousel. The owner tests through the package on his local stack. PRs come later.
>
> **Lineage:** `docs/PRDS/prd-content-engine.md:5,132` deferred social distribution to "a separate future PRD". This is that PRD.
>
> **Proof of concept (outside the repo):** `Automatos-AI-Platform/brag-output-2026-09-22-225946/`. A 39.5 s, 1080×1920 promo rendered from one HTML composition with Hyperframes, a local Kokoro voice-over, a CC BY music bed with ducking, and captions. It is the reference implementation for the render pipeline in Wave 1 (`composition/index.html`, `composition/scripts/mix.py`, `composition/scripts/synth_vo.py`).

---

## Framing (CLAUDE.md §3)

**This is an extension of Deliverables plus one net-new service.**

The increment is:
- a Socials tab beside Outputs, Blogs and Templates;
- a post lifecycle that includes approval and scheduling;
- a render service that assembles images and video. It is the one piece Composio can't supply at the reference quality for $0 (D3).
- per-channel publishers that sit on the existing Composio executor;
- a video recipe (a skill) and guardrails, so agents use the customer's connected Composio media tools well and safely (D16, D17).

**What is reused:**

| Existing piece | Location | How Socials uses it |
|---|---|---|
| Deliverables Studio tab mechanism | `frontend/lib/deliverables/tabs.ts:2`, `frontend/components/deliverables/studio/deliverables-studio.tsx:21-31, 79-112` | Hosts the new tab |
| Classic deliverables page | `frontend/app/deliverables/page.tsx:106-155` | Gets the same tab |
| Blog pipeline | `core/models/core.py:1681-1744`, `api/blog.py`, `frontend/hooks/use-blogs-api.ts` | The closest precedent for draft → publish |
| Brand kit | `modules/documents/brand_kit.py:45-63`, `api/document_brand_kit.py` | Colours, fonts and logo for every template |
| Template records | `document_templates`, `core/models/core.py:1482-1525` | Social templates are stored here |
| Calendar read model | `services/activity_service.py:889-923` | A sixth source, `social` |
| One-shot scheduling and reconcile | `services/scheduled_task_service.py:717-720`, `services/schedule_reconcile.py:63-82` | Scheduled publishing |
| Composio connections and executor | `api/composio.py:217-273`, `core/composio/tool_executor.py:344` | Every publish, and every paid media call, goes through these |
| Per-action schema cache | `composio_actions_cache`, `core/models/composio_cache.py:86-109` | The channel capability registry reads it |
| Per-workspace switch pattern (voice_live) | `modules/voice/live_settings.py:50, 105-111`, `api/workspaces.py:450-484` | Turning Socials on per workspace |
| Plan nav exposure | `services/plan_tiers.py:174-222`, `frontend/lib/nav-exposure.ts` | Plan gating |
| Cost booking | `UsageTracker.track(cost_override=)`, precedent `track_rerank` (`core/llm/usage_tracker.py:345-376`) | Booking media spend |
| 3-file platform tool pattern | `modules/tools/discovery/platform_actions.py:58-105` | The agent entry point |

**Why not `blog_posts` for posts.** A blog post is one web article. It has a slug, SEO fields, and a single `published` state that publishes nothing outward (`core/services/blog_service.py:250-258`). A social post is different:
- It fans out to N channels.
- Each channel has its own media rules, remote id, permalink and failure.
- Its approval must bind to the exact rendered media.

Forcing that shape into `blog_posts` would make both worse.

**Why not the workspace worker for rendering.** The worker is shared with agent command execution: `WORKER_CONCURRENCY=3` and 2 CPU / 2 GB in compose (`docker-compose.yml:414-415`). It also can't render video as it stands:
- It runs Node 20 (`services/workspace-worker/Dockerfile:13`), and Hyperframes needs ≥ 22.
- It has no ffmpeg.
- The local edition ships it without Chromium (`INSTALL_BROWSER=false`, `docker-compose.yml:364`).
- The html-to-png timeout is 60 s (`executor.py:341`). A 40 s vertical video took 2 m 42 s on an 8-core machine in the proof of concept.

**Why not `approval_grants` for approval.** Grants default to a 24 h TTL (`config.py:963`), which is wrong for a post scheduled days ahead. `platform_ask_human` refuses any subject that isn't a board task (`handlers_asks.py:57-76`). The Approvals inbox is admin-only, in the Governance tab (`governance-tab.tsx:34`). So approval is state on the post itself (D6).

**Why not the `automatos-social` git repo for templates.** It is cloned into each workspace (`core/seeds/platform-management-skill.md:1147-1180`). Its `_brand.jsx` hardcodes Automatos orange and ignores the brand kit. Keeping it would break CLAUDE.md §4 ("no file hacks for DB data"). Its five families are ported into DB templates (S1.2), and **the repo-clone path is deleted** (§5).

## What the owner asked for (2026-09-22/23 conversation)

- "write this as a PRD so any customer can use it, videos, images, facts, info graphics"
- "Build it into Deliverables, alongside blog, report and templates for reporting... new TAB socials with content and approvals, connect it to the calendar"
- "I have tested this on the SaaS version, so composio have the ability to post to x, insta etc..."
- **Channels:** LinkedIn, X, Instagram, TikTok and YouTube Shorts. "be flexible, if they connect via composio and we can talk them do it"
- **AI footage:** templates first, plus optional cinematic shots from the customer's own tool, connected in Composio (D12). Off by default.
- **Approvals:** every post by default. A workspace may switch on series approval.
- **Voice:** "Kokoro now... but we need to leave it open to BYOK for Elevenlabs and fish.audio (… so so much cheaper)". Both now come through Composio (D11).
- "phase two we build social engagement"
- **Composio-first:** "I don't want to reinvent the wheel... I want amazing videos and I just need my customers to connect a Composio tool and my agents use them."
- **Keep it simple, keep the quality:** "let's keep it simple for now, no one wants to hear me speak... but I want the same quality of videos as earlier". This refers to the reference videos in PRD-251A §1.
- **Deadline:** Web Summit Lisbon, 9 Nov 2026. "6 weeks to build socials and build videos like this… and i want it automated". Automatos dogfoods Socials for its own campaign.

## What exists today (verified)

### Deliverables UI
- The tab list is static: `DELIVERABLE_TABS = ['outputs','blogs','templates']` (`frontend/lib/deliverables/tabs.ts:2`).
- Explorer is a separate route button (`deliverables-studio.tsx:26-27, 91-93`).
- The Classic shell has its own hardcoded `FilterTabs` and `TabsContent` (`app/deliverables/page.tsx:106-155`).
- Global search entries: `hooks/use-global-search.ts:30-31`.
- There is no per-workspace or per-plan tab gating anywhere.

### Blog
- **Status:** `draft|scheduled|published|archived` (`core.py:1685-1688`).
- **`scheduled_for` is a dead column.** It appears only in the model and `to_dict` (`core.py:1709, 1735`). No request schema or job reads it.
- **Board approval:**
  - Only `publish_blog` and `create_blog` do anything (`api/board_tasks.py:827-876`).
  - An unknown `approval_action` is approved with no side effect (`:877-879`).
  - `auto_approve` executes it immediately (`handlers_board_tasks.py:225-248`).

### Calendar
- `GET /api/activity/schedule` is a stateless merge of five isolated sources (`services/activity_service.py:906-912`): heartbeats, playbook crons, `agent_scheduled_tasks`, mission SLAs and board-task SLAs.
- There is no events table.
- The frontend kind is a closed union: `ScheduleItemType` (`hooks/use-activity-api.ts:97`), `KIND_META`/`KIND_ORDER` (`calendar-kinds.ts:21-30`) and `calendar-actions.ts`.

### Scheduler
- APScheduler `AsyncIOScheduler` (`services/scheduler.py:23-75`), with a RedisJobStore when Redis is set.
- It runs on the uvicorn worker holding the fcntl lock (`main.py:390-400`).
- One-shot `DateTrigger` jobs are used for `agent_scheduled_tasks` (`scheduled_task_service.py:717-720`), with a 180 s misfire grace and a 60 s reconcile tick (`schedule_reconcile.py:63-82`).
- `execute_task` delivers only `chat` or `board_task` (`:50-52, :501-513`).
- Operator rows are capped at 50 per workspace (`:31-34`).
- Trial workspaces are skipped (`:431-447`).

### Brand kit
- Stored in `workspace.settings['brand_kit']`.
- Fields: name, tagline, logo, primary/secondary/accent/text colours and one `font_family` (`brand_kit.py:45-63`).
- **Missing:** heading/body fonts, font files, social handles and brand voice.
- The only editor is `BrandKitDialog.tsx`, opened from Template Studio.

### Templates
- `document_templates.format` allows only `pdf|docx|xlsx` (`core.py:1522`).
- `blocks` is JSONB (`:1505`).
- Rendering is WeasyPrint (`modules/documents/generation_service.py:303-367`).

### Composio
- **Connections:** one per toolkit per workspace, `UNIQUE(entity_id, app_name)` (`core/models/composio.py:27-66`). Listed by `GET /api/composio/connections` (`api/composio.py:217-273`).
- **Executor:** `ComposioToolExecutor.execute(action, params, agent_id, workspace_id, skip_validation)` (`tool_executor.py:344`).
- **File uploads:** URL or path → `FileUploadable` only for `UPLOAD_ACTIONS`, which lists 4 Twitter and 5 LinkedIn actions (`:39-49, :124-219`).
- **Action schemas:** `composio_actions_cache` holds per-action `parameters` and `response_schema`, synced daily (`main.py:468`, `services/metadata_sync_service.py:42-80`).
- **Posting has no approval:** there is no social/publish capability in `REQUIRES_CONFIRMATION` (`modules/tools/capabilities/taxonomy.py:268-280`), and the policy plane ships `off` (`config.py:899-918`). **Today an agent can post to a connected social account with no approval.**

**Known posting slugs:**

| Channel | Slugs | Source |
|---|---|---|
| LinkedIn | `LINKEDIN_CREATE_LINKED_IN_POST`, `LINKEDIN_CREATE_IMAGE_POST`, `LINKEDIN_CREATE_SHARE`, `LINKEDIN_INITIALIZE_IMAGE_UPLOAD`, `LINKEDIN_REGISTER_IMAGE_UPLOAD` | `tool_executor.py:44-48` |
| X | `TWITTER_CREATION_OF_A_POST` | `modules/tools/services/composio_hint_service.py:699` |
| X (media) | `TWITTER_UPLOAD_MEDIA`, `TWITTER_INITIALIZE_MEDIA_UPLOAD`, `TWITTER_UPLOAD_LARGE_MEDIA`, `TWITTER_APPEND_MEDIA_UPLOAD` | `tool_executor.py:40-43` |
| Instagram | `INSTAGRAM_POST_IG_USER_MEDIA` (image; `media_type` `REELS`/`STORIES` with `video_url`), `INSTAGRAM_CREATE_CAROUSEL_CONTAINER`, `INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH` | Skill + [Composio docs](https://docs.composio.dev/toolkits/instagram) |
| TikTok | init, upload, publish (including publish-from-URL) and publish-status actions | Skill `tiktok-automation` + [Composio docs](https://docs.composio.dev/toolkits/tiktok) |
| YouTube | `YOUTUBE_UPLOAD_VIDEO` / `YOUTUBE_MULTIPART_UPLOAD_VIDEO` (file param), `YOUTUBE_UPDATE_THUMBNAIL` (public URL) | [Composio docs](https://docs.composio.dev/toolkits/youtube) |

- **Instagram media rule:** all media must be direct, publicly fetchable HTTPS URLs. No redirects, no HTML pages.
- **Seed skill slugs disagree with the executor's.** The seed uses `LINKEDIN_CREATE_POST` and `TWITTER_CREATE_TWEET` (`platform-management-skill.md:1167`).

### LinkedIn image workaround: a cross-tenant defect
`core/composio/linkedin_image_workaround.py:61-107` loads the **first active** credential of type `linkedInCommunityManagementOAuth2Api` with **no workspace filter**. It caches that credential in a process-global (`_cached_creds`), and it carries the credential owner's `organization_urn`. Any workspace whose agent posts a LinkedIn image through `LINKEDIN_CREATE_LINKED_IN_POST` (`tool_executor.py:701-729`) would publish to that one organisation. **S0.4 fixes this before any Socials publishing ships.**

### Media storage
- **Public image route:** `GET /api/generated-images/{id}` is public but unsafe for this use.
  - It looks up with `workspace_id=None` and then lists only the first 1000 keys, so lookups start returning 404 past 1000 objects (`api/generated_images.py:28`, `core/services/image_store.py:84-94`).
  - It reads whole objects into memory with no Range support (`:102`), so it is unsuitable for video.
- **Presigned links:** capped at 7 days and forced to attachment disposition (`generation_service.py:80-83, 274-297`).
- **Video in Deliverables:** it maps to type `video` (`services/deliverable_service.py:107`) but is excluded from `AGENT_REGISTERABLE_ARTIFACT_TYPES` (`:132`). The frontend `DELIVERABLE_TYPES` has no video (`components/icons/deliverable-icon.tsx:20-35`).

### Keys and costs
- **BYOK:** `ProviderSpec` with `ADAPTER_NONE` is a key-only provider; Cohere is the one precedent (`core/llm/providers.py:42, 154-158`). Keys live in `user_api_keys`.
- There is no per-workspace key resolver for non-LLM providers (`core/llm/workspace_keys.py:28` covers only the operator workspace).
- ElevenLabs appears only in a comment (`config.py:1718`). fish.audio, Higgsfield and fal: not found.
- Usage lanes are defined in `core/llm/usage_context.py:33-40`. There is no media lane.

### Composio media toolkits (verified 2026-09-23 on docs.composio.dev)
Composio lists 1,552 toolkits. Its Design & Creative category alone has 62. These are the ones Socials uses:

| Need | Toolkit (slug) | Auth | The actions that matter |
|---|---|---|---|
| Footage and stills from any of 600+ models | fal.ai (`fal_ai`) | API key | `FAL_AI_SUBMIT_ASYNC_JOB`, `FAL_AI_QUEUE_GET_STATUS`, `FAL_AI_GET_QUEUE_REQUEST_RESULT`, `FAL_AI_UPLOAD_FILE`, `FAL_AI_ESTIMATE_PRICING` |
| Footage (Veo 3.1, Runway Aleph), stills (Flux Kontext, GPT-4o image) | Kie.ai (`kieai`) | API key | `KIEAI_GENERATE_VEO_VIDEO` with `KIEAI_GET_VEO_VIDEO_DETAILS`, `KIEAI_GENERATE_FLUX_KONTEXT_IMAGE`, `KIEAI_GET_ACCOUNT_CREDITS` |
| Higgsfield image, video, audio and Marketing Studio | Higgsfield MCP (`higgsfield_mcp`) | OAuth to the customer's Higgsfield **account**, paid in credits. It is not the platform API key the proof of concept used. | `HIGGSFIELD_MCP_GENERATE_VIDEO`, `HIGGSFIELD_MCP_GENERATE_IMAGE`, `HIGGSFIELD_MCP_JOBS_WAIT`, `HIGGSFIELD_MCP_MEDIA_UPLOAD`, `HIGGSFIELD_MCP_BALANCE` |
| Voice | Fish Audio (`fish_audio`), ElevenLabs (`elevenlabs`, 155 tools) | API key | `FISH_AUDIO_SYNTHESIZE_SPEECH` (the free S2.1 model), `FISH_AUDIO_LIST_VOICE_MODELS` |
| Also available, not used here | Runway, Luma Labs, HeyGen, Magic Hour, Replicate, DreamStudio; the render APIs Creatomate and Shotstack; Canva; Postiz, Buffer and Hootsuite | Mostly API key | |

**Money-moving and account actions exist.** `higgsfield_mcp` includes:
- `HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE`. Composio's docs say it "charges real money" for plan upgrades, credit top-ups and auto-refill.
- `HIGGSFIELD_MCP_CANCEL_TRIAL_AUTO_RENEWAL`.
- Website create, deploy and publish.
- Contest entry.
- `HIGGSFIELD_MCP_APPS_INVOKE`, which runs any Higgsfield marketplace app as the user.

**Automatos has no guard for them today:**
- There is no per-action allow or deny list for Composio.
- The capability classifier (`modules/tools/capabilities/classifier.py`) has no billing or purchase keyword.
- Confirmation (`REQUIRES_CONFIRMATION`) is flagged at `classifier.py:677`, and the policy plane that enforces it ships `off` (`config.py:898`).

So a workspace that connects `higgsfield_mcp` could let its agents call the purchase action. Whether Higgsfield itself insists on an in-widget confirmation first is unverified. S0.6 closes this.

### Not found
- A social post table, a publish service, idempotency or retries for posts, or publish-status polling.
- An upload of TikTok's binary to its `upload_url`.
- Any video storage or streaming.
- A per-workspace feature-toggle framework beyond the voice_live pattern.
- Multiple accounts per toolkit.

## Goals

1. **Any workspace can make a post.** With Socials on, a brand kit and one connected channel, a workspace goes from "make a post about X" to an approved, scheduled post in **≤ 10 minutes (median)**.
2. **Formats.** Short video (9:16, 1:1, 16:9), single image, carousel, fact card and infographic. All are rendered from the workspace brand kit and data templates.
3. **Nothing is published without approval.** An approval record binds to the exact rendered content. The server refuses to publish otherwise, whatever the policy plane is set to.
4. **Every factual claim has a source**, or an explicit, recorded owner override.
5. **Scheduled posts appear on the Command Center calendar** and publish on time. A missed slot is reported, never silently posted late.
6. **Any Composio-connected social toolkit whose actions can carry the post can publish.** Adapters handle LinkedIn, X, Instagram, TikTok and YouTube in Phase 1.
7. **Dogfood.** Automatos runs its own Web Summit campaign through Socials, with **≥ 3 posts a week across ≥ 3 channels for the 4 weeks before 9 Nov 2026**.
8. **The reference quality.** Every video matches the four reference videos: the three in PRD-251A §1 (v1 UI story, v2 cinematic Shopify, Academy) and the Markets posh cut made after it (compositions in `docs/PRDS/prd251-reference/`):
   - exact brand colours and type;
   - the product's screens rebuilt sharp, not generated;
   - animated headlines;
   - captions;
   - cuts on the music.

   The owner judges them side by side before Wave 1 closes.

## Decisions (adopted 2026-09-23; each reversible in Wave 0)

### D1 · A Deliverables tab, off by default, gated two ways, on every plan
Socials is a fourth tab, `socials`, in both shells. Two switches control it:
- the platform master switch, the `socials.enabled` system setting (a super-admin setting, like voice_live), whose default comes from `SOCIALS_ENABLED_DEFAULT`;
- the workspace switch, `workspace.settings['socials'].enabled`.

**Every plan gets Socials** (owner, 2026-09-23). This follows the rule in `services/plan_tiers.py`: "paid tiers gate organisational features and hosting, never product capability". Plans differ only in render quotas, which are hosting (S1.1). There is no plan exposure key.

**What the user sees:**
- Master switch off: no tab anywhere, and every `/api/socials/*` route, job and agent tool returns 404.
- Master switch on, workspace switch off: the tab shows one card. An owner or admin sees "Turn on Socials for this workspace"; anyone else sees "Ask an admin to turn on Socials". The `/api/socials/*` routes still return 404.
- Both on: the full tab.

**Why the tab shows before the workspace switch is on:** an owner can't turn on a feature they can't find. Settings would hide it one level deeper.

### D2 · Two new tables, one migration
**`social_posts`** (one row per post):
- `id`, `workspace_id`, `created_by` (user or agent id), `campaign_id` nullable, `title`
- `brief` (the ask)
- `copy` (JSON: base text plus per-channel overrides)
- `format` `video|image|carousel|fact_card|infographic`, `template_id` → `document_templates`, `variables` JSON
- `sources` JSON: claim slot → `{kind: deliverable|report|document|url|metric, ref, as_of}`
- `media` JSON: deliverable ids per aspect ratio
- `status` `draft|rendering|needs_approval|changes_requested|approved|scheduled|publishing|published|partially_published|failed|missed|archived`
- `content_hash` (sha256 over copy, variables, sources and the rendered media bytes)
- `approved_hash`, `approved_by`, `approved_at`, `override_unsourced` bool
- `review_log` JSON: a list of `{at, by, action, comment}`, the history the approval UI shows. Request-changes comments and reject reasons live here.
- `scheduled_for` (UTC), `timezone`, `created_at`, `updated_at`

**`social_post_targets`** (one row per channel per post):
- `id`, `post_id`, `toolkit` (the Composio app name), `post_kind` (`text|image|carousel|video|reel|short|story`)
- `action_plan` JSON (the resolved action slugs and parameters)
- `idempotency_key` unique
- `status` `pending|uploading|published|failed`, `attempts`, `remote_id`, `permalink`, `error`, `published_at`

**`social_campaigns`** is created in **Wave 2 only if series approval ships**: `id`, `workspace_id`, `name`, `approval_mode` `per_post|series`, `approved_hash_set`, `approved_by/at`.

**Why no existing table fits:** see Framing.

### D3 · Rendering: a new `media-render` service
**It assembles; it never generates.**
- Footage, stills and premium voice arrive from the customer's Composio tools (D11, D12).
- `media-render` puts them together with the template, the brand kit, Kokoro voice and the music, and produces the finished file.
- **Why we build it:** this is the one piece Composio can't supply at the reference quality for $0. The render APIs Composio lists (Creatomate, Shotstack) bill the customer per render, and their animation vocabulary is narrower than HTML + GSAP. Owner, 2026-09-23: "keep it simple... same quality".

- **Container:** Node 22 + Chromium + ffmpeg + Hyperframes (Apache-2.0, **pinned** version) + the Kokoro TTS runtime (Python).
- **API:** HTTP `POST /render` takes a composition bundle (the template HTML, variables, brand assets, audio plan) and returns an MP4 or PNG plus a JSON report.
- **Why a separate service:**
  - it keeps the agent worker's capacity for agents;
  - it keeps the GPL-3.0 `phonemizer`/espeak-ng used by the Kokoro stack in a separate program;
  - it lets the SaaS scale rendering on its own.
- **Local edition:** it runs as an optional compose profile, `media`.
- **Telemetry and skills:** the service runs with `HYPERFRAMES_NO_TELEMETRY=1`, `DO_NOT_TRACK=1` and `HYPERFRAMES_SKIP_SKILLS=1`.

### D4 · Templates are data
- `document_templates.format` gains `social_image|social_video` (a migration of the check constraint).
- For social formats, `blocks` holds `{html, css, variables_schema, sizes, audio_plan}`.
- The seeded library is:
  - the five `automatos-social` families (title, definition, stats, quote, announcement) × the sizes in `automatos-social/schema.json`;
  - carousel and fact-card variants;
  - one infographic (a chart from a report's data table);
  - the four reference videos as templates: "UI story promo" (v1), "cinematic product promo" (v2, with footage slots), "app promo" (Academy, with a phone frame) and "data story" (the Markets posh cut: footage and stills with data cards generated from live data). They set the quality bar (Goal 8).
- All templates read brand tokens as CSS variables. **No template hardcodes a colour, font or logo.**
- The repo-clone path and the `html-to-png` skill's dependence on it are removed (§5).

### D5 · The brand kit is extended in place
`brand_kit` gains:
- `heading_font` and `body_font`;
- `font_files` (uploaded woff2, stored like the logo);
- `logo_mark_url`, a square mark separate from the wordmark;
- `social_handles` (per toolkit);
- `voice` (three to five tone words plus banned phrases).

It keeps the same GET/PUT API and dialog. The dialog also opens from the Socials tab.

### D6 · Approval is state on the post, bound to a hash
- **Approving** records `approved_hash = content_hash`, plus approver and time. The approve request carries the `content_hash` of the version the approver was shown. A post whose content changed since answers 409 with the current hash. The write is a compare-and-set on status and hash, so an edit that lands while the request runs is never approved (P251-RVW-2).
- **Any edit resets approval.** Any change to copy, variables, sources or re-rendered media changes `content_hash` and moves the post back to `needs_approval`.
- **The publisher refuses** any target whose post has `approved_hash ≠ content_hash`, whatever the policy plane mode.
- **Who can approve:** workspace owner, admin or editor (a new `socials:approve` permission in `modules/policy/roles.py`).
- **Series mode** (D1 switch, Wave 2): approving a campaign approves every post whose hash is in the approved set when approval is given. A post added or edited later still needs its own approval.

### D7 · Facts carry sources
- Template variables marked `claim: true` (numbers, dates, named results) must be bound to a source.
- **Source kinds:**
  - a Deliverable;
  - an `agent_reports` row;
  - a knowledge-base document;
  - an analytics metric at a timestamp;
  - an URL.
- The composer shows a red **Unsourced** chip on any claim without a source.
- Approving with unsourced claims needs a second confirmation. That sets `override_unsourced=true` and names the claims in the approval record.
- **Why:** the parked Web Summit numbers (`docs/WEBSUMMIT-MESSAGE.md`, "not as a claim anyone may quote yet"; $3.42 vs $4.26 across docs) are exactly the failure this prevents.

### D8 · Publishing goes through Composio only, via a capability registry
**The registry.** For each connected social toolkit, it reads `composio_actions_cache` and resolves which post kinds the toolkit supports and with which action slugs and parameters. Seeded adapters cover the known quirks:

| Channel | Quirks the adapter handles |
|---|---|
| Instagram | Container create → publish (the publish action waits for processing); public HTTPS media only |
| TikTok | Init → upload (HTTP PUT of the binary to `upload_url`) or publish-from-URL → poll publish status |
| YouTube | Upload takes a file; the thumbnail takes a public URL |
| X | Media upload (chunked for video) → post |
| LinkedIn | Post, image and video actions, using the connected account only |

**Any other toolkit** with a "create post" action whose schema takes text plus a media URL or file is offered as **Generic (text + media)**. It is labelled "unverified channel" until one publish succeeds.

**Other rules:**
- One idempotency key per target.
- Up to 3 retries with backoff, for transient errors only.
- `permalink` is stored on success.

### D9 · Public media is served by presigned S3 URLs with an inline disposition
- **Storage:** rendered files go to S3 under `social-media/{workspace}/{post}/{file}` and are registered as Deliverables.
- **At publish time** the publisher mints a presigned GET with `ResponseContentDisposition=inline`, the correct `ResponseContentType`, and a TTL of `SOCIALS_MEDIA_URL_TTL_SECONDS` (24 h).
- **Never** use `/api/generated-images/` (the 1000-key and Range defects).
- **Local edition:** channels that fetch media by URL (Instagram, TikTok publish-from-URL, the YouTube thumbnail) need `SOCIALS_PUBLIC_MEDIA_BUCKET` configured. Otherwise those channels show "needs public storage".

### D10 · Scheduling uses the socials service's own one-shot jobs
- Each scheduled post registers an APScheduler `DateTrigger` job, id `social-publish-<post_id>`, registered with args (not a closure) so the RedisJobStore can pickle it.
- A boot and reconcile pass re-registers them, copying `schedule_reconcile.py`.
- **Missed slot:** if the backend was down at the scheduled time, it publishes on recovery only within `SOCIALS_MISFIRE_GRACE_SECONDS` (1800). Otherwise the post goes to `missed` and the owner is notified. Stale content is never posted silently.
- **Why not `agent_scheduled_tasks`:** it delivers only chat or board tasks, shares the 50-row operator cap, and skips trials.
- **Calendar:**
  - The backend gets a sixth source, `_social_post_items`, with id `social-<post_id>`.
  - The frontend gets a new kind, `social`, in the three union files.
  - Clicking an item opens the post.
  - Drag-to-reschedule calls `PATCH /api/socials/posts/{id}` with `scheduled_for`.
  - Times are stored in UTC and shown in the workspace timezone.

### D11 · Voice: Kokoro by default, or a Composio voice tool the workspace has connected

| Source | Default | Cost | How |
|---|---|---|---|
| Kokoro, built into `media-render` | ✔ | $0 | The reference videos' voice (`af_heart`, speed 0.95) |
| Fish Audio via Composio (`fish_audio`) | | The customer's Fish Audio account. `FISH_AUDIO_SYNTHESIZE_SPEECH` uses the free S2.1 model. | One call per script line |
| ElevenLabs via Composio (`elevenlabs`) | | The customer's ElevenLabs account | One call per script line |

- The script is split into lines, one audio file per line, whatever the source. Scene timing flexes to the audio.
- **No owner or customer recordings** (owner, 2026-09-23: "no one wants to hear me speak"). The `upload` provider is dropped.
- **The Composio connection is the key.** Automatos holds no voice keys, adds no `ProviderSpec` group and writes no `resolve_workspace_key`.

### D12 · AI footage and stills come from the workspace's Composio tools
- **How a workspace gets cinematic shots:** it connects a generation toolkit in Composio (fal.ai, Kie.ai or Higgsfield; see "Composio media toolkits"). Its agents then use it. Automatos writes no provider client.
- **One small recipe per toolkit.** Each toolkit has its own async shape, so a recipe maps "a 5 s 9:16 shot from this prompt" onto its actions:
  - fal: `FAL_AI_SUBMIT_ASYNC_JOB` → `FAL_AI_QUEUE_GET_STATUS` → `FAL_AI_GET_QUEUE_REQUEST_RESULT`;
  - Kie.ai: `KIEAI_GENERATE_VEO_VIDEO` → `KIEAI_GET_VEO_VIDEO_DETAILS`;
  - Higgsfield: `HIGGSFIELD_MCP_GENERATE_VIDEO` → `HIGGSFIELD_MCP_JOBS_WAIT`.

  Always submit and poll, never one long synchronous call.
- **Outputs are copied into our storage the moment a job completes.** Provider URLs expire. The copies are registered as Deliverables (the video type from S0.4).
- **Generated clips fill `media` slots only:** the hook and the b-roll.
  - Product screens stay HTML-rendered, because AI video warps interface text.
  - Every word on screen is overlaid by the template, never generated (PRD-251A §6).
- The prompt patterns, model choice and pacing that produced the reference videos live in the video skill (D17).

### D13 · Costs are estimated, capped and booked on a new `media` lane
- **Before spend:**
  - Where the toolkit prices a call (fal: `FAL_AI_ESTIMATE_PRICING`), the estimate is checked against the post's cap and the workspace's monthly media cap, `socials.media_monthly_cap_usd`. That cap is config the owner sets.
  - Over the cap, nothing is submitted.
- **After spend:**
  - The actual cost is booked with `UsageTracker.track(cost_override=(usd, 0), request_type=LANE_MEDIA, tier='direct')`, copying `track_rerank`.
  - Where a toolkit reports credits rather than dollars (Higgsfield, Kie.ai, Fish Audio), the balance is read before and after the job (`HIGGSFIELD_MCP_BALANCE`, `KIEAI_GET_ACCOUNT_CREDITS`, `FISH_AUDIO_GET_ACCOUNT_BALANCE`). The difference is recorded against the post.
- The workspace budget gate and the daily spend guard then see it with no further work.
- Local rendering and Kokoro cost $0 and are booked as units only.

### D14 · Agents draft; the Socials tab and the publisher are the only way out
- `platform_create_social_post`, using the 3-file pattern, lets Auto and agents **draft** posts into `needs_approval`. It never publishes.
- **When Socials is on,** the executor blocks a direct agent call to any action the registry classifies as a *post action* of a connected social toolkit. It returns "use platform_create_social_post".
- **D14b confirmed by the owner, 2026-09-23.** It changes today's behaviour, where agents can post unapproved. It is built in Wave 1 (US-118).

### D15 · Free and open source by default; paid tools are switched on by connecting them in Composio
The owner said two things on 2026-09-23:
1. First: "free models will be open source as default but option to add fish.audio, elevenlabs, higgsfield etc... by enabling API keys in the settings tab".
2. Later the same night: "I don't want to reinvent the wheel... I just need my customers to connect a Composio tool and my agents use them".

The second supersedes the API-keys route.

**The default, with nothing connected and no spend:**
- Hyperframes rendering (Apache-2.0);
- Kokoro voice (Apache-2.0 weights);
- the seeded CC0 and CC BY music;
- OFL fonts.

This makes the "UI story" and "app promo" kinds of video (reference v1 and Academy) at full quality, without AI footage.

**Paid ingredients appear when the workspace connects the tool in Composio,** in the same place it already connects LinkedIn or Instagram.
- There is no Automatos key storage, no change to Settings → API Keys and no `ProviderSpec` group.
- The Socials composer shows an ingredient as enabled only when a matching toolkit is connected (D16). Otherwise it links to the Composio connect flow.

### D16 · One registry of what the connected tools can do, and a hard deny list
- **The registry** extends D8's.
  - It reads `composio_actions_cache` for every connected toolkit.
  - It classifies the actions Socials may use (`generate_video`, `generate_image`, `tts`, `render`, `publish`, `status`, `upload`, `estimate`, `balance`) through a curated allowlist per toolkit.
  - The allowlist is seeded for fal.ai, Kie.ai, Higgsfield MCP, Fish Audio, ElevenLabs and the five channels. An unknown toolkit offers nothing until it is added.
- **The deny list is platform-wide, not Socials-only.**
  - The Composio executor refuses listed actions for every agent in every workspace, **whatever the policy plane mode**, and says why.
  - Seeded with: `HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE`, `HIGGSFIELD_MCP_CANCEL_TRIAL_AUTO_RENEWAL`, `HIGGSFIELD_MCP_CONFIRM_TRIAL_CANCEL`, `HIGGSFIELD_MCP_CREATE_WEBSITE`, `HIGGSFIELD_MCP_DEPLOY_WEBSITE`, `HIGGSFIELD_MCP_PUBLISH_WEBSITE`, `HIGGSFIELD_MCP_PARTICIPATE_IN_CONTEST` and `HIGGSFIELD_MCP_APPS_INVOKE`.
  - The list is data (a system setting the super-admin edits), not code.
  - Buying credits, changing plans and deploying stay with the human, in the tool's own interface.

### D17 · The video skill is the product
- **Where it lives:** the pipeline that made the reference videos (PRD-251A §2) becomes a skill, "social video director", in `automatos-skills` (the source of truth), synced into the seed.
- **What it tells an agent,** step by step, from a brief to a post ready for approval:
  1. the claims and their sources;
  2. the shot list;
  3. prompt patterns per connected toolkit (subject and scene, light, camera, "no readable text, no logos", the brand's colour words);
  4. model choice;
  5. the voice script (one line per file, with pronunciation checks);
  6. the music window, aligned to a drop;
  7. the template and its variables;
  8. QA (`hyperframes check` and a snapshot review);
  9. `platform_create_social_post`, which puts the post in `needs_approval`.
- **Quality bar (Goal 8):** before Wave 1 closes, an agent following the skill re-makes the four reference videos from their templates. The owner compares them side by side with the originals.

## Stories## Stories

`(S)` = about half a day, `(M)` = about a day, `(L)` = split before building. **Editions** give the default; the matrix below has the exceptions.

### Wave 0 — the gate, the tables, the service, the defects (backend)

**S0.1 · Config, gate and permission (S)**
- Typed config: `SOCIALS_ENABLED_DEFAULT: bool = False`, `SOCIALS_MEDIA_URL_TTL_SECONDS = 86400`, `SOCIALS_MISFIRE_GRACE_SECONDS = 1800`, `SOCIALS_MAX_TARGET_ATTEMPTS = 3`, `SOCIALS_RENDER_URL`, `SOCIALS_PUBLIC_MEDIA_BUCKET`.
- A system-settings master switch, `socials.enabled`, and a workspace setting `socials` (the voice_live pattern: `modules/voice/live_settings.py:50, 105-111`, `api/workspaces.py:451`). The master switch's default comes from `SOCIALS_ENABLED_DEFAULT`, and its row is seeded so a super-admin can flip it in Settings → System Settings.
- `GET /api/workspaces/current` gains `socials: {available, enabled}` (`available` is the master switch). The frontend gate reads only this.
- A `socials:approve` permission for owner, admin and editor.
- **No plan exposure key:** every plan gets Socials (D1).
- **Files:** `orchestrator/config.py`; `modules/socials/settings.py`; `api/workspaces.py` (`PUT /api/workspaces/current/socials`, the `/current` field); `modules/policy/roles.py`; `tests/test_config_env_centralization.py`.
- **Acceptance:**
  - [ ] With the master switch off, every `/api/socials/*` route returns 404 and `/current` reports `available: false`.
  - [ ] With the master switch on and the workspace switch off, `/api/socials/*` still returns 404 and `/current` reports `available: true, enabled: false`.
  - [ ] A viewer gets 403 on approve; an editor can approve.
  - [ ] `PUT /api/workspaces/current/socials` needs `workspace:manage` and rejects unknown keys (fail-closed, like `validate_voice_live_update`).
  - [ ] The centralisation test lists the new attributes.
  - [ ] CI green.
- **Editions:** both.

**S0.2 · Tables and migration (M)**
- `social_posts` and `social_post_targets` as in D2. JSON columns use `JSON().with_variant(JSONB(), "postgresql")` (the SQLite-test trap).
- One alembic revision, `prd251_socials`, as the single head. Update both head pins (`tests/test_prd209_alembic_single_head.py`, `tests/test_prd236_w1_routes.py`) and `scripts/init_test_db.py`.
- **Acceptance:**
  - [ ] `alembic heads` returns one head.
  - [ ] The upgrade/downgrade round-trip passes.
  - [ ] The unique `idempotency_key` constraint rejects a duplicate.
  - [ ] Tables build under SQLite.
- **Editions:** both.

**S0.3 · Socials service and API (M)**
- `modules/socials/service.py` owns:
  - the status machine (D2);
  - the `content_hash` computation;
  - `approve`, `request_changes` and `reject`, with D6 semantics;
  - an `assert_publishable(post)` guard used by every publish path.
- Routes, all behind S0.1:
  - `GET /api/socials/posts?status=&from=&to=`
  - `POST /api/socials/posts`, `GET /api/socials/posts/{id}`, `PATCH /api/socials/posts/{id}`
  - `POST /api/socials/posts/{id}/submit|approve|request-changes|reject|schedule|unschedule|publish-now`
  - `submit` moves a draft (or a post with changes requested) to `needs_approval`. Until Wave 1 adds rendering, this is how a draft reaches approval.
  - In Wave 0, `publish-now` runs `assert_publishable` and then returns 501 ("Channel publishing arrives in Wave 3"). S3.3 fills that seam.
- The route manifest is updated (CI reads the committed manifest).
- **Acceptance:**
  - [ ] Editing any field of an approved post moves it back to `needs_approval`.
  - [ ] `publish-now` on a post whose `approved_hash ≠ content_hash` returns 409 and makes no Composio call (asserted with a mock).
  - [ ] Approving with unsourced claims without `override_unsourced=true` returns 422.
  - [ ] CI green.
- **Editions:** both.

**S0.4 · Fix the prerequisites (M)**
1. **Workspace-scope the LinkedIn image workaround** (owner, 2026-09-23: scope it, don't delete it; the module's own header says Composio can't upload LinkedIn images):
   - filter the credential by `workspace_id`;
   - cache per workspace, not in a process-global;
   - refuse if the workspace has no credential of its own;
   - cover all three callers: `core/composio/tool_executor.py:705`, `api/recipe_executor.py:763` and `api/composio.py:281` (the third is missing from the module's removal checklist).
2. **Video in Deliverables:** add `video` to `AGENT_REGISTERABLE_ARTIFACT_TYPES` and to the frontend `DELIVERABLE_TYPES`, with an icon and a player preview.
3. **Fix the image-store public lookup:** stop listing the first 1000 keys. Store the key in the image id or a lookup row, and add a Range-capable streaming response.

- **Acceptance:**
  - [ ] A test with two workspaces and two credentials proves each posts with its own.
  - [ ] An MP4 written by an agent appears in Deliverables Outputs with a player.
  - [ ] A lookup with 1,500 stored objects returns the 1,400th.
- **Editions:** both.

**S0.5 · The Socials tab, bare (M)** — pulled forward from S2.1 (owner rule: activate before stacking dormant features)
- Add `socials` to `DELIVERABLE_TABS` (`frontend/lib/deliverables/tabs.ts:2`), the Studio `TAB_LABELS` (`deliverables-studio.tsx:21-31`) and the Classic `FilterTabs` (`app/deliverables/page.tsx:106-155`).
- The tab follows D1: absent when `socials.available` is false; the one "Turn on Socials" card when the workspace switch is off; the list when both are on.
- **The list:** posts grouped by status with counts, newest first, and an empty state.
- **"New draft":** a title, a brief and the base copy. It creates a `draft` post through `POST /api/socials/posts`.
- **Post detail:** its status, and the approve / request changes / reject actions the caller's role allows (S0.3), so the approval rules can be tested by clicking.
- All calls go through `apiClient`, using new methods; raw `fetch` is eslint-banned.
- **Acceptance:**
  - [ ] With `available: false` the tab is absent in both shells, and `?tab=socials` falls back to `outputs`.
  - [ ] With `available: true, enabled: false`, an owner sees the Turn-on card and a viewer sees the ask-an-admin card. Turning it on shows the empty list without a reload.
  - [ ] Creating a draft lists it under Draft with count 1.
  - [ ] Approving it moves it to Approved. Editing its copy afterwards moves it back to Needs approval.
  - [ ] vitest component tests for the three states; `npm run build` passes.
- **Editions:** both.

**S0.6 · The Composio deny list (S)** — the platform-wide half of D16, shipped early because the gap is live today
- A deny list of Composio action slugs, stored as the system setting `composio.denied_actions`, seeded with D16's list and editable by the super-admin.
- Every path that executes a Composio action refuses a denied slug before any network call, for every caller (agents, recipes and the API) and whatever the policy plane mode. It returns "This action is blocked in Automatos: <reason>".
  - The known entry points are `ComposioToolExecutor.execute` (`core/composio/tool_executor.py:344`), `api/recipe_executor.py` and `api/composio.py`.
  - Grep for the others.
- **Acceptance:**
  - [ ] With the policy plane `off`, executing `HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE` returns the blocked error and makes no Composio call (mocked).
  - [ ] Removing a slug from the setting unblocks it without a deploy.
  - [ ] A test walks every executor entry point and proves each one consults the list.
- **Editions:** both.

### Wave 1 — content: render service, templates, brand, facts, voice, music, footage, the video skill

**S1.1 · `media-render` service (L → S1.1a container, S1.1b API, S1.1c orchestrator client)**
- The container, per D3: pinned Hyperframes, Node 22, Chromium, ffmpeg, the Kokoro runtime, and the telemetry-off environment.
- `POST /render` → `{status, outputs[{aspect, path, bytes, duration}], report{lint, check, timings}}`.
- It runs `hyperframes check` before `render` and refuses on errors.
- The orchestrator client (`core/media_render_client.py`) uses a 900 s read timeout and runs as a background task that updates `social_posts.status`.
- Compose adds profile `media`, and Railway gets a service.
- **Per-plan render quotas (owner, 2026-09-23):** every plan gets Socials, and plans differ only in monthly render minutes: **Basic 10, Pro 60, Business 240**. The quota is a field on each `PLAN_TIERS` entry, config, never hardcoded. Enterprise and the local edition have none until the owner sets one. A render past the quota is refused with a clear message, and the Socials tab shows minutes used out of the quota.
- **Sizing (owner, 2026-09-23):** SaaS runs `media-render` as its own Railway service at 4 vCPU / 8 GB, with two renders at once overall and one at a time per workspace; further jobs queue.
- **Acceptance:**
  - [ ] A render past the workspace plan's monthly quota is refused before `media-render` is called.
  - [ ] Rendering the proof-of-concept composition bundle in CI produces an MP4 of 39.5 s ± 0.05 s at 1080×1920, 30 fps, with an AAC stream.
  - [ ] A template with a lint error returns a 422 carrying the lint findings.
  - [ ] The espeak data path is under 160 characters inside the container (asserted at boot, see Traps).
- **Editions:** SaaS default; local via the `media` profile.

**S1.2 · Social templates as data plus the seeded library (M)**
- Migrate the `document_templates.format` check. Add a social variables schema, validated on save.
- Seed the templates listed in D4. Every template takes brand tokens as CSS variables and declares its `sizes`.
- Delete the `repos/automatos-social` clone step and point the `html-to-png`/`social-ops` skills at the DB templates (a PR against `automatos-skills`, then sync the seed).
- **Acceptance:**
  - [ ] A render of each seeded template in each declared size passes `hyperframes check` in CI.
  - [ ] Changing the brand kit's primary colour changes the rendered PNG (pixel probe).
  - [ ] No seeded template contains a hex colour literal outside its fallbacks.
  - [ ] The four reference templates (D4) render from variables in CI.
- **Editions:** both.

**S1.3 · Brand kit extension (S)**
- Add the D5 fields to `brand_kit.py`, the API, and `BrandKitDialog.tsx`.
- Font upload is stored like the logo.
- The dialog also opens from the Socials tab.
- **Acceptance:**
  - [ ] Uploaded woff2 fonts are used in renders.
  - [ ] Social handles are validated per toolkit.
  - [ ] Browser check on the preview deployment (dev-browser skill).
- **Editions:** both.

**S1.4 · Facts and sources (M)**
- Template variables with `claim: true`.
- A source picker searching Deliverables, reports, knowledge-base documents and analytics metrics, plus an URL option.
- The Unsourced chip. D7 override semantics.
- **Acceptance:**
  - [ ] A fact card with an unsourced number cannot be approved without the override.
  - [ ] The approval record lists the overridden claims.
  - [ ] Browser check on the preview deployment.
- **Editions:** both.

**S1.5 · Voice: Kokoro, or a connected Composio voice tool (M)**
- Kokoro (the default) runs inside `media-render`.
- **Fish Audio and ElevenLabs** come through their Composio toolkits, one call per script line. They are the recipe side of D12, applied to speech.
- The script is split into lines, one audio file per line. Scene timing flexes to the audio.
- No recordings, no Automatos-held keys (D11).
- **Acceptance:**
  - [ ] A render with Kokoro produces speech in every script window.
  - [ ] With `fish_audio` connected and chosen, the same script renders with no other change (Composio mocked in CI).
  - [ ] Without a voice toolkit connected, the composer offers Kokoro only and links to the Composio connect flow for the others.
- **Editions:** both.

**S1.6 · Music library (S)**
- 8–12 curated tracks with licence metadata (CC0 or CC BY, with attribution text), for example ende.app tracks under CC BY 4.0.
- Beat cues precomputed at seed time (`librosa` in the render image).
- The ffmpeg mix: ducking under voice and loudness normalisation to −14 LUFS.
- CC BY tracks append their credit line to the post copy.
- **Acceptance:**
  - [ ] Every track has licence and attribution fields.
  - [ ] A rendered video's integrated loudness is −14 ± 1 LUFS.
  - [ ] A CC BY credit appears in the post's default copy.
- **Editions:** both.

**S1.7 · Infographic template (M)**
- A chart template (bar, line or number grid) bound to an `agent_reports` data table or an analytics metric series.
- Every figure it shows is source-bound (D7).
- **Acceptance:**
  - [ ] Given a report with a data table, the infographic renders its top 5 rows with the source chip resolved.
  - [ ] Axis labels fit (`hyperframes check` passes with no layout errors).
- **Editions:** both.

**S1.8 · Footage and stills from connected Composio tools (M)** — replaces the old S4.3
- The per-toolkit recipes of D12 for fal.ai, Kie.ai and Higgsfield MCP: submit, poll, copy the output into storage, and register it as a Deliverable.
- The D16 registry decides which of them the workspace has.
- D13's estimate and cap run before every submit, and the actual cost is booked afterwards.
- **Acceptance:**
  - [ ] With `fal_ai` connected, a template's hook slot is filled by a generated 5 s 9:16 clip, and the post's cost carries the estimate (Composio mocked in CI).
  - [ ] An estimate over the monthly cap submits nothing and says why.
  - [ ] The output URL is copied into our storage before the job is marked done.
  - [ ] With no generation toolkit connected, the footage slots fall back to the template's own motion graphics.
- **Editions:** both (local needs `COMPOSIO_KEY`).

**S1.9 · The "social video director" skill (M)** — D17
- Authored in `automatos-skills` (the source of truth), then synced into the platform as a built-in skill with `scripts/sync-skills.py` (US-119; owner, 2026-09-23: built in and synced, no push, PRs later). The writing is outside the Ralph loop: the loop never writes skill content or edits a generated seed.
- It carries the PRD-251A pipeline as agent instructions, with the reference compositions as worked examples.
- **Acceptance:**
  - [ ] An agent given only the skill, a brand kit and a brief produces a draft video post in `needs_approval`.
  - [ ] **Owner sign-off:** an agent re-makes the four reference videos from their templates, and the owner judges them side by side against the originals (Goal 8).
- **Editions:** both.

### Wave 2 — the Socials tab

**S2.1 · The tab, complete (M)** — builds on S0.5, which shipped the gated tab, the status list and drafts
- A board view by status beside S0.5's list.
- Global search entries (`hooks/use-global-search.ts:30-31`).
- The compact (phone) form, per PRD-246's conventions.
- **Acceptance:**
  - [ ] The board and the list show the same posts and counts.
  - [ ] Global search finds a post by title.
  - [ ] Browser check on the preview deployment at 390 px and 1440 px.
- **Editions:** both.

**S2.2 · Composer (L → S2.2a brief, S2.2b preview and variables, S2.2c formats)**
1. **Brief:** the user types what the post is about. Auto drafts copy per channel, proposes a template and format, pre-fills variables and proposes sources.
2. **Preview:** a live preview (the rendered PNG, or a low-quality video render). Variables can be edited.
3. **Formats:** select aspect ratios and target channels, filtered by the registry (S3.2).

- **Acceptance:**
  - [ ] "Announce our Harvest Club launch" produces a draft with copy for each selected channel, a template, a rendered preview, and sources for any claims, within 90 s for an image and 6 min for a 40 s video.
  - [ ] Browser check on the preview deployment.
- **Editions:** both.

**S2.3 · Approval UI (M)**
- Approve, request changes (with a comment) or reject.
- It shows sources, the unsourced override, the per-channel copy, and the exact media that will publish.
- A notification goes to approvers when a post enters `needs_approval`, through the existing notification dispatcher.
- **Acceptance:**
  - [ ] Approve records hash, user and time.
  - [ ] An edit after approval shows "Approval reset: content changed".
  - [ ] Browser check on the preview deployment.
- **Editions:** both.

**S2.4 · Series approval (M, optional per workspace)**
- `social_campaigns` plus the D6 series semantics, with a campaign view in the tab.
- **Acceptance:**
  - [ ] Approving a campaign approves its current posts.
  - [ ] A post added afterwards shows `needs_approval`.
- **Editions:** both.

### Wave 3 — schedule and publish

**S3.1 · Scheduling and calendar (M)**
- D10 jobs and reconcile.
- The calendar source `_social_post_items` and kind `social` (backend plus the three frontend union files).
- Drag to reschedule. Missed-slot handling.
- **Acceptance:**
  - [ ] A post scheduled 2 minutes ahead publishes once. Two backend workers do not double-publish, thanks to the fcntl lock plus the idempotency key.
  - [ ] A backend down across the slot for longer than the grace period yields `missed` and a notification.
  - [ ] The calendar shows the post in the workspace timezone.
- **Editions:** both.

**S3.2 · Channel capability registry (M)**
- `modules/socials/registry.py` reads `composio_actions_cache` for each connected social toolkit.
- It resolves the supported `post_kind`s and slugs, using the seeded adapters from D8 plus the generic adapter.
- It exposes `GET /api/socials/channels`.
- **Acceptance:**
  - [ ] With LinkedIn, X and Instagram connected, the registry lists them with their supported kinds.
  - [ ] A toolkit with no post action is not listed.
  - [ ] Slugs come from the cache, not hardcoded strings (a test scans for literals outside the seeded adapter table).
- **Editions:** both.

**S3.3 · Publishers (L → S3.3a LinkedIn, S3.3b X, S3.3c Instagram, S3.3d TikTok, S3.3e YouTube)**
- One adapter per channel, following D8. Idempotency, retries, and a receipt stored on the target (`remote_id`, `permalink`).
- `partially_published` when some targets fail.
- **Acceptance (for each channel):**
  - [ ] Against a mocked Composio executor, the adapter issues the documented action sequence.
  - [ ] A transient error retries up to `SOCIALS_MAX_TARGET_ATTEMPTS`.
  - [ ] A 4xx error does not retry and surfaces the platform's message.
  - [ ] One live smoke publish per channel to an owner-controlled test account, recorded in the PR (no spend).
- **Editions:** both, subject to D9.

**S3.4 · Media hosting (S)**
- S3 keys and presigned inline URLs, per D9.
- **Acceptance:**
  - [ ] The presigned URL returns `Content-Type: video/mp4` and `Content-Disposition: inline`, and supports Range requests (checked with a HEAD and a Range GET in CI against MinIO).
- **Editions:** both (local needs a public bucket for URL-fetch channels).

**S3.5 · One way out (M; D14b confirmed 2026-09-23 — built in Wave 1 as US-118)**
- The executor guard: when Socials is on, a direct agent call to a registry-classified post action returns a structured refusal pointing to `platform_create_social_post`.
- **Acceptance:**
  - [ ] An agent attempting `INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH` in a Socials workspace gets the refusal, and no Composio call is made.
  - [ ] Read actions are unaffected.
- **Editions:** both.

### Wave 4 — agents, automation, cost

**S4.1 · `platform_create_social_post` (S; moved into Wave 1 as US-116)**
- 3-file pattern: `actions_socials.py`, a handler, and Auto keywords.
- It creates drafts only.
- **Acceptance:**
  - [ ] Auto can draft a post from chat, and it appears in the tab as `needs_approval`.
  - [ ] The tool has no publish parameter.
- **Editions:** both.

**S4.2 · "Weekly social content" Playbook seed (S; moved into Wave 1, US-120's 'Weekly social posts')**
1. It reads the week's new Deliverables, published blog posts and reports.
2. It proposes N posts (default 5) across the workspace's channels, spread over the week.
3. It drafts them, with sources.
4. It leaves them for approval.

- Seeded as a Playbook row (not a file).
- **Acceptance:**
  - [ ] A manual run in the test workspace produces N drafts with schedule proposals.
  - [ ] Nothing publishes without approval.
- **Editions:** both.

**S4.3 · Moved to S1.8** (footage now comes from the workspace's Composio tools, in Wave 1).

**S4.4 · `media` usage lane (S)**
- Add `LANE_MEDIA` in `core/llm/usage_context.py` and a `track_media` helper.
- Analytics shows "Media" spend.
- **Acceptance:**
  - [ ] A mocked fal job with a $0.46 estimate books $0.46 on the `media` lane.
  - [ ] A mocked Higgsfield job records its balance difference against the post.
  - [ ] The budget gate counts media spend.
- **Editions:** both.

## Phase 2 (separate PRD) — engagement

Engagement is scoped here, not specified:
- ingest comments and mentions on published posts through Composio read actions (for example `INSTAGRAM_GET_IG_MEDIA_COMMENTS`);
- a daily digest in the Socials tab and on Telegram;
- **drafted replies with the same per-post approval** (no auto-replies; X and LinkedIn automation rules prohibit unattended engagement);
- per-post metrics (views, likes, follows, clicks), where the toolkit exposes them.

## Delivery plan — 6 weeks to Web Summit (9 Nov 2026)

| Week of | Build | Dogfood |
|---|---|---|
| 28 Sep | Wave 0 (S0.1–S0.6) | The proof-of-concept pipeline, run by hand, makes the first Automatos posts |
| 5 Oct | Wave 1 (S1.1–S1.3, S1.5, S1.8, S4.4) | Brand kit filled in; Kokoro voice; the first cinematic shots from a connected Composio tool |
| 12 Oct | Wave 1 (S1.4, S1.6, S1.7, S1.9) + Wave 2 (S2.1–S2.3) | The reference videos re-made by an agent and signed off (Goal 8); posts drafted in the Socials tab |
| 19 Oct | Wave 3 (S3.1–S3.4) | First scheduled publishes on LinkedIn, X and Instagram |
| 26 Oct | S2.4, plus what remains of Wave 4 (S4.1, S4.2 and S3.5 moved into Wave 1) | Weekly Playbook drafts the campaign |
| 2 Nov | Buffer, fixes, TikTok/YouTube if slipped | Countdown series, stand details, "see it live" |

**Done means CI green, merged, and tested by the owner in both editions** (owner rule). Every wave ends with the owner's test before the next wave starts.

### How it is built and tested (owner, 2026-09-23)
- **Build:** one Ralph kit per wave (`scripts/ralph/prd-251.json` and its prompts, acceptance gate and runner). The owner launches each wave with `scripts/ralph/launch-251.sh`. The loop works only on `feat/prd-251-socials`, pushes after every story so CI runs, and opens a draft PR to `main` when the wave passes review.
- **Test before merge:** the branch runs on its own local Docker stack, the compose project `automatos-socials`. It has its own containers, volumes, network and ports (frontend :23000, API :28000) and a fresh database, so it never shares state with the customer-night stack. The stack runs the local edition with Socials on by default and carries no production keys. Its script and overrides live outside the repo, in `~/.automatos/socials-stack/`.
- **Test after merge:** the owner tests SaaS once a wave is merged and deployed.

## Editions matrix

| Capability | SaaS | Local |
|---|---|---|
| Socials tab, drafts, approvals, calendar | ✔ | ✔ |
| Rendering (`media-render`) | ✔ (service) | Compose profile `media` (heavier image with Chromium) |
| Kokoro voice | ✔ | ✔ (in `media-render`) |
| Paid footage, stills and voice (the workspace's Composio tools) | ✔ | Needs `COMPOSIO_KEY` |
| Publish via Composio | ✔ | Needs `COMPOSIO_KEY` |
| URL-fetch channels (Instagram, TikTok publish-from-URL, YouTube thumbnail) | ✔ | Needs `SOCIALS_PUBLIC_MEDIA_BUCKET` |
| Series approval | ✔ | ✔ |

## Not in this PRD (owner decisions)

- Phase 2 engagement (its own PRD).
- More than one account per toolkit per workspace (`UNIQUE(entity_id, app_name)`).
- Paid ads, boosting, and analytics dashboards beyond publish receipts.
- Automatos-billed AI footage or voice. Paid ingredients always come through the customer's own Composio connection.
- Voice cloning, and owner or customer voice recordings (owner, 2026-09-23).
- Our own API clients for Higgsfield, fal, ElevenLabs or fish.audio, and keys for them in Settings → API Keys (superseded by D15).
- Creatomate or Shotstack as the renderer. Both are on Composio and could replace `media-render` later, but not now (owner: keep it simple, same quality).
- A custom template code editor for customers. Phase 1 ships the curated library plus brand kit theming; authoring is Automatos-only.
- Publishing outside Composio (direct platform APIs).
- Wiring or deleting blog `scheduled_for`. Recorded here and not decided.

## Open questions (owner)

1. ~~**Plans:** which plans get Socials, and what are the render quotas per plan?~~ **Answered 2026-09-23:** all plans (D1). Monthly render quotas: Basic 10 min, Pro 60 min, Business 240 min (config; S1.1).
2. ~~**D14b:** block direct agent posting to social toolkits when Socials is on?~~ **Answered 2026-09-23:** yes, built in Wave 1 (US-118).
3. ~~**Local edition:** is it in v1, or SaaS-first with local following?~~ **Answered 2026-09-23:** both editions in Wave 1 (compose profile `media`).
4. ~~**SaaS hosting:** `media-render` on Railway. What size, and is the render queue concurrency per workspace?~~ **Answered 2026-09-23:** its own service, 4 vCPU / 8 GB, two renders at once overall, one per workspace.
5. **Music:** which tracks make the seeded library, and how is attribution shown?
6. ~~**LinkedIn image workaround:** workspace-scope it, or delete it in favour of Composio's own image post action?~~ **Answered 2026-09-23:** workspace-scope it (S0.4).
7. **The default footage tool to recommend to customers:** fal.ai, Kie.ai or Higgsfield. Settled by the quality check in "Verify at build".

## Test plan

- **Unit:** the status machine; `content_hash` stability; D6 approval reset; D7 unsourced gating; registry parsing of cached action schemas; the idempotency key; misfire handling; the deny list at every executor entry point; each toolkit recipe's submit-and-poll (Composio mocked).
- **Integration (CI only; nothing runs locally):**
  - a `media-render` contract test rendering a fixture composition;
  - the Composio executor mocked, per adapter;
  - scheduler `DateTrigger` fire and reconcile;
  - migration round-trip;
  - the MinIO presign HEAD and Range checks.
- **Frontend:** component tests for the tab, composer, approval panel and calendar kind.
- **Browser checks** on the preview deployment using the dev-browser skill, not a local server.
- **Live smoke:** one owner-run publish per channel to a test account, recorded in the S3.3 PR.

## Verify at build (no spend)

| Assumption | How it is settled |
|---|---|
| The TikTok publish-from-URL slug and parameters | Query `composio_actions_cache` where `app_name='tiktok'` |
| The YouTube upload action accepts a `FileUploadable` | The cached schema for `YOUTUBE_UPLOAD_VIDEO` |
| LinkedIn has a video post action | The cached LinkedIn schemas |
| Instagram fetches from a presigned S3 URL with a query string | Owner test account, one Reel |
| Hyperframes renders headless on Linux within budget | CI render of the proof-of-concept bundle, timed (the 2 m 42 s baseline was measured on a macOS 8-core machine) |
| Kokoro in the container | Boot assertion on the data path length, plus one line synthesised |
| Pinned Hyperframes stays compatible | `hyperframes check` passes on every seeded template in CI |
| Composio footage reaches the reference quality (small owner spend) | Run the four v2 shot prompts (PRD-251A Appendix, `jobs.json`) through fal (Kling or Veo) and through Higgsfield MCP on the owner's test accounts. The owner picks the default recommendation (open question 7). |
| Higgsfield MCP reaches Cinema Studio | `HIGGSFIELD_MCP_MODELS_EXPLORE` on a connected test account |
| Long jobs through the Composio executor | Submit and poll only. Check the executor's timeout against a 3-minute job. |

## Traps

- **espeak-ng truncates its data path at about 160 characters.** The proof of concept failed with `…/site-packages//phontab`. `phonemizer` resolves symlinks, so a symlink doesn't help. Keep the data directory at a short real path in the render image.
- **The Hyperframes CLI:**
  - needs Node ≥ 22;
  - sends telemetry by default (`HYPERFRAMES_NO_TELEMETRY=1`, `DO_NOT_TRACK=1`);
  - installs skills globally on `init` unless `HYPERFRAMES_SKIP_SKILLS=1`;
  - suggests `preview` (a server), which the render service never runs.
- **Hyperframes composition rules:**
  - one paused timeline registered at the end;
  - no `Math.random`, `Date.now` or `repeat:-1`;
  - never tween `.clip` visibility;
  - `data-layout-allow-overlap` must go on the text element itself, because it is not inherited.
- **Instagram** rejects redirects, HTML pages and Drive links. Images must be JPEG. Media must be publicly fetchable.
- **The LinkedIn workaround is cross-tenant** today (S0.4).
- **Composio's Higgsfield is the account kind.**
  - `higgsfield_mcp` logs in to the customer's Higgsfield account (credits) and carries a real-money purchase action, which is deny-listed (D16).
  - It is not the platform API key the proof of concept used.
  - **Checked 2026-09-23 (Composio docs): it does not reach Cinema Studio 4.0.** Its video models are Marketing Studio video, Seedance 2.5, Kling 3.0 and Minimax. Its image models are GPT Image 2 (the default), Marketing Studio image, Soul 2, Soul Cast and Soul ID, with no Soul Cinema. `HIGGSFIELD_MCP_GENERATE_VIDEO` takes a single `params` string. The reference footage therefore needs a Composio stand-in (Kling 3.0 or Seedance 2.5 are the likely picks), chosen by the small-spend quality test (open question 7).
- **ElevenLabs on Composio uses the customer's API key.** Its "Text to speech" action (`ELEVENLABS_TEXT_TO_SPEECH`) returns a downloadable audio file, and "Get voices list" lists the voices. Confirm both slugs in the action cache before seeding.
- **Composio's Ayrshare has three tools and no create-post action.** Don't plan publishing through it.
- **Provider output URLs expire.** Copy every generated file into our storage the moment its job completes.
- **`UPLOAD_ACTIONS` omits Instagram and TikTok.** Use the adapters; do not widen the global list blindly.
- **Seed and skill slugs disagree** (`LINKEDIN_CREATE_POST` vs `LINKEDIN_CREATE_LINKED_IN_POST`). Resolve slugs from the cache (S3.2).
- **The scheduler runs on one uvicorn worker** (fcntl lock). Its job store pickles jobs, so register with ids and arguments, never closures. Publishing must be idempotent.
- **Times:** UTC in the database, the workspace timezone in the UI.
- **The GPL boundary:** `phonemizer` and espeak-ng are GPL-3.0. Keep them in `media-render`, never imported by the orchestrator.
- **CC BY music** needs its credit line.
- **Canonical terms:** Deliverable (not "output" or "artifact"), Playbook, Command Center, Auto.

## Merge notes

- **DCO:** `commit -s` on every commit.
- **Alembic:** a single alembic head; update both head-pin tests.
- **Routes:** update the route manifest for the new `/api/socials/*` routes (CI reads the committed manifest).
- **Frontend fetches:** go through `apiClient`, because raw `fetch` is eslint-banned.
- **Skills:** changes are a PR against `automatos-skills` first, then a seed sync (owner rule: the skills repo is the source of truth).
- **Replace cleanly:** delete the `automatos-social` repo-clone path in the same PR that seeds the DB templates (§5).

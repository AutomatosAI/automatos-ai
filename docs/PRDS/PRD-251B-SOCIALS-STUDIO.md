# PRD-251B: Socials Studio. A calendar-first Socials tab, a full post editor, cadence plans with a researched content bank, and one brand kit with style references

| | |
|---|---|
| Owner | Gerard Kavanagh |
| Written | 2026-10-02 |
| Status | Draft for owner review |
| Builds on | PRD-251 Waves 0–3 (on `main` since #852, fb675857b) |
| Prerequisite | `main` has one alembic head again before Wave 1 launches: our fix (`prd251w2_merge_heads`, branch `fix/prd-251-one-head-and-scans`) or #859's `prd252_ticket_numbers`, whichever merges first (both join `prd251_wave2` and `document_chunks_ingestion_columns`). Every wave below chains its migration onto `EXPECTED_HEAD` on its base |
| Visual spec | The clickable mockup <https://claude.ai/artifact/T4W1hez7DD8u5gyshyaXTQ>; the same files are in `docs/PRDS/prd251b-reference/` (README inside) |
| Target | Automatos runs its own Web Summit countdown on it (Web Summit Lisbon, 9 Nov 2026) |

## Framing

PRD-251 built the engine: posts, hash-bound approval, per-channel copy, a composer, template rendering, media hosting, publishing to five channels through Composio, scheduling, campaigns, voice and AI footage. Its UI was specified as a thin form: a brief, then a proposal, then three steps, inside the post list. The owner tested it and rejected it: *"i get asked 3 questions and then an approve button thats it... Nothing about is it an image, video, how long, size, schedule, which platform."*

This PRD replaces the Socials UI with the mockup above and adds what the mockup needs from the backend. **It keeps PRD-251's rules**: agents draft and never publish; every post is approved against the exact content that will go out; publishing goes only through the workspace's own Composio connections; paid tools only through Composio (D15); Socials is off unless switched on.

## What the owner asked for (2026-10-02)

- "I want a rich UI/UX ... a way for me to just say schedule a months worth of post, Agent will research my docs, code, etc... and plan what will be posted, generate the images/videos on its own."
- "A place in the brand kit to upload images that you like that agents can use for style ideas."
- "User should have option to use higgsfield etc..." → **through Composio only** (owner, 2026-10-02).
- On planning: "I dont think we need to generate a months worth straight away but plan it, research content, facts, features ... So we have a schedule we will post x images daily and 1 video a week ... Then we can generate the content on the day, spreads the token usage."
- On the mockup: "this is what i wanted all along". On the brand kit: "lets not make two, so this new brand kit will be used for templates also". On the calendar: keep a separate Socials calendar, and "a flag in settings to enable and disable socials" because the bank PoC will not want Socials.

## What exists today (verified on main fb675857b, 2026-10-02)

Paths are under `orchestrator/` or `frontend/`.

- **Post model** (`core/models/socials.py:144-230`): title, brief, copy (base + per channel), format (`video|image|carousel|fact_card|infographic`, :68), template_id, variables, sources, media, voice, footage, preview, status (12 states, :54-67), content_hash, approved_hash/by/at, `scheduled_for` + `timezone` (:217-218), campaign_id, targets (per-channel kind and options, :267-331). **No video length field**: a video's length is the template's `data-duration` (`core/social_templates.py:428-433`).
- **Composer** (`api/socials_compose.py:63-68`): takes `brief`, `channels`, `format`. The model picks the template from `social_templates()` (:76-92). The client cannot choose a template or a length up front. Saving a post (`api/socials.py:169-181`) does accept `template_id`, `voice` and `footage`.
- **Preview** (`modules/socials/preview.py`): renders the template's first size at half resolution (`SCALE=2`, :39, :60-67).
- **Templates**: one list endpoint, `GET /api/documents/templates?format=` (`api/document_generation.py:202-223`). `DocumentTemplate.thumbnail_url` exists (`core/models/core.py:1551`) but is never set or returned. Twelve seeded social starters (`modules/documents/social_starters.py:47-63`): four videos (UI story promo, cinematic product promo, app promo, data story; 1080×1920 only) and eight image families (four sizes each).
- **Brand kit**: no table; `workspace.settings['brand_kit']` (`modules/documents/brand_kit.py:41`, fields :234-264: name, tagline, logo, colours, fonts and font files, square mark, social handles, voice). Routes: `GET/PUT /api/documents/brand-kit`, logo, logo-mark and font uploads (`api/document_brand_kit.py:66-321`). The kit becomes `--brand-*` CSS variables at render (`core/media_render_bundle.py:14-23,123`). **One kit already**: Template Studio and Socials open the same `BrandKitDialog`.
- **Approval**: approve, request-changes and reject (`api/socials.py:637-695`). The hash covers copy, variables, sources, format, template, media, plus targets, title and the AI-footage flag once the post has channels (`modules/socials/service.py:317-330`). Series approval (`modules/socials/campaigns.py:340-374`) approves each shown post by hash; it needs the workspace's `series_approval` switch and a campaign in series mode. Approvers are notified on `needs_approval` (`modules/socials/notify.py:145`) through `NotificationDispatcher` (in-app, Telegram, Slack, webhook, per the user's preferences).
- **Scheduling**: `schedule()` accepts only approved, scheduled or missed posts (`modules/socials/service.py:810-829`). One APScheduler `DateTrigger` per post (`modules/socials/schedule_jobs.py`), a reconciler (`services/schedule_reconcile.py`), and `missed` with a notification. The Command Center calendar shows `status=scheduled` posts only (`services/activity_social_items.py:50-66`), gated by `socials_off_reason()`.
- **Campaigns** (`core/models/socials.py:90-141`): name, approval_mode (`per_post|series`), approved_hash_set. Created with `{name, approval_mode}`.
- **Playbooks** (`core/seeds/seed_socials_package.py:255-359`): Brand kit from your website, Launch video, Weekly social posts (unscheduled, default 5), Image carousel. `seed_socials_marketplace()` (:488-510) runs at every boot and **checks no Socials switch**.
- **Research tools agents already have**: `search_knowledge` (`modules/tools/registry/tool_registry.py:423`), `platform_list_deliverables` (`modules/tools/discovery/actions_deliverables.py:16`), `platform_web_fetch` (`modules/tools/discovery/actions_web.py:18`), and Composio tool execution. No GitHub research path exists for Socials.
- **AI media** (`modules/socials/recipes/footage_toolkits.py`): fal.ai, Kie.ai and Higgsfield recipes for footage **and stills** (`GENERATE_IMAGE` / `GENERATE_VIDEO`, `modules/socials/capabilities.py:94-95`). Spend caps: per post (`config.SOCIALS_MEDIA_POST_CAP_USD`) and per workspace per month (`socials.media_monthly_cap_usd`) (`modules/socials/media_caps.py`). The capability registry (D16) offers an action only if it is connected, allowlisted and not deny-listed.
- **Voice**: `GET /api/socials/voices` and `/voices/{toolkit}` (`api/socials.py:830-861`). Kokoro's voices cannot be browsed today (`modules/socials/recipes/voice.py:260`); each template hardcodes one (default `af_heart`).
- **Switches** (`modules/socials/settings.py`): the master switch is system setting `socials.enabled` (super-admin, Settings → System Settings → Socials; default off, `SOCIALS_ENABLED_DEFAULT`). The workspace switch is `workspace.settings['socials'].enabled`. It can be turned **on** only from the card inside the Socials tab (`useEnableSocials`), and **no setting turns it off**. With either switch off, every `/api/socials/*` route answers 404 (`require_socials_enabled`, :234-245), agent tools refuse (`socials_off_reason`, :207-216) and the Command Center hides social items. The marketplace package stays visible.
- **Frontend**: Deliverables tabs in `frontend/lib/deliverables/tabs.ts:1-19`, rendered in `frontend/app/deliverables/page.tsx:36-41`. The Command Center calendar grids run on a generic event model (`components/command-center/calendar-model.ts`, `calendar-kinds.ts`), but the month grid takes a required `social: SocialReschedule` prop (`calendar-month-grid.tsx:16-23`). `FilePreview` (`components/widgets/FileWidget/FilePreview.tsx`) renders images, video, PDF and HTML. The voice picker, channel row, receipts, publish controls and campaign views are kept.

## Goals

1. A person makes a post by choosing what it is first: format, length, sizes, channels, look and slot. They see each channel's preview before approving.
2. A person sets a plan once: dates, cadence per channel, sources to research and how posts are made. Automatos researches a content bank, makes each post on its day (videos a day early) and puts it in the Queue.
3. The Socials tab opens on its own calendar. It shows what is planned, being made, waiting for approval, scheduled, posted and skipped.
4. One brand kit, on its own Deliverables tab, serves Templates and Socials. It adds style references (images, each marked liked or avoided, with a note) and AI tool defaults.
5. With Socials off, nothing of it shows anywhere: tab, calendar layer, marketplace package, Auto's tools or brand-kit handles. The bank PoC sees no Socials.
6. Nothing publishes without an approval of the exact content. No month of content is generated ahead.

## Decisions (adopted 2026-10-02; each reversible before its wave starts)

### B1 · The mockup is the visual spec
Build the screens in `docs/PRDS/prd251b-reference/`: Calendar, Post editor, Queue, Plan, Brand kit. Match their layout, hierarchy, controls and copy. Build with the app's own components (`components/ui/*`), Tailwind tokens and fonts (Geist, Newsreader). `studio.css` shows intent; it is not to be copied. A browser check against the mockup is the owner's, never the loop's.

**Design fidelity (added 2026-10-02 evening; owner: "lets not skip anything").** Three things make the build land on the mockup rather than near it:
1. **A token map.** `prd251b-reference/TOKENS.md` maps every `studio.css` role (colour, type, shape, spacing, status tones) to the app token or `components/ui` component that builds it. It is binding. It records the two places where the app and the mockup differ, with defaults the owner can flip in one line: the accent shade (default: the Studio token, one notch deeper than the mockup's hex) and primary buttons (default: the accent, as the mockup, not the Studio's cream CTA). Two known deltas by design: the app has no teal (olive stands in for Scheduled/Approved) and the Studio's destructive equals its accent (Skipped is muted plus the word).
2. **A screenshot lane (US-B112).** A non-required workflow builds the frontend in the local edition against a mocked API, screenshots every Studio screen at 1440 px and 390 px on every push, and uploads the PNGs as a run artifact. Its fixtures mirror the mockup's sample data with the clock frozen at Wed 14 Oct 07:20, so a PNG sits beside its mockup screen. Every UI story adds its screens in the same commit. Playwright is never an app dependency: it lives in its own `frontend/screens/package.json`, installed by the workflow only.
3. **The owner's browser check stays the gate** (`prd251b-w1-owner-test.md`), now with the PNGs to look at before the stack comes up.

### B2 · Socials navigation
The Socials tab has a sub-navigation: **Calendar** (home) · **Queue** (with a count) · **Plans** · **Brand kit** (a link to the Brand kit tab, B9). The header carries **New plan** and **New post**. Today's List and Board become a third calendar view (Month · Week · List). Campaigns become plans (B6).

### B3 · A separate Socials calendar; the Command Center keeps its layer
The Socials calendar shows planned slots, posts being made, posts waiting for approval, scheduled, posted and skipped. The Command Center calendar keeps PRD-251's read-only layer of scheduled posts, already gated by `socials_off_reason()`. Both grids share one extracted month/week grid. The Command Center's social drag becomes an optional prop, not a required one. **Coordinate with PRD-252 (#854)**, which changes the Command Center board and calendar layout: rebase onto it before extracting the grid.

### B4 · "Off" means invisible
The master switch stays as it is (System Settings, super-admin, default off). New in the workspace's own Settings: a **Socials on/off toggle** for owners and admins. While either switch is off:
- the marketplace listing hides the Socials package (it is still seeded, so switching on later needs no boot);
- tool discovery and Auto's keyword routing leave out the `socials` actions;
- the Brand kit hides the channel handles.

Everything already gated stays gated.

### B5 · The editor chooses up front; the composer fills in
- `ComposeRequest` gains `template_id` and `length_seconds`. When given, the composer must use that template, validated as a workspace social template of the post's format. When not given, it picks one as today ("Let Auto pick").
- A video template declares the lengths it supports as `blocks.durations` (seconds), each with its own timeline. The editor offers only those lengths, and the post stores its choice in a new `social_posts.length_seconds`, which the content hash covers.
- Sizes follow the chosen channels and their post kinds (channel registry). The editor shows the result and does not invent sizes the template lacks.

### B6 · A plan is a campaign
`social_campaigns` gains the plan fields:
- `kind` (`campaign|plan`), goal, audience, dates, timezone;
- `cadence` (rows of channel, format, length, days and time; a video row can be shared by Reels, TikTok and Shorts);
- `sources`, `make` settings, `late_policy`, `research` settings, `slot_overrides` and `status`.

Series approval still works on a plan exactly as on a campaign.

### B7 · Plans book slots; posts are made on their day
- **No content is generated ahead** (owner, 2026-10-02). A plan expands its cadence into slots on the fly (a pure function). A slot has a `slot_key` (channel group, date, time) that is unique per plan.
- **Making is a service, not an agent.** At the plan's make time (default 07:00 in the plan's timezone; videos the day before), a leader-only tick takes each due slot with no post yet. It picks the next unused content-bank topic that suits the slot's format, then builds the post through the existing composer with the topic's facts as its sources, then renders and submits it for approval. Approvers are notified as today.
- The tick is idempotent through `slot_key`. A slot over the render quota or media cap is skipped and notified, never half-made.

### B8 · Research is an agent run that writes only to the content bank
A seeded playbook, **Content bank research**, is run by the Social Media Director. Its sources:
- `search_knowledge`, `platform_list_deliverables` and `platform_web_fetch` on the brand-kit website;
- the workspace's GitHub through Composio when it is connected (README, docs, releases, merged pull requests).

It writes through one new draft-only tool, `platform_add_social_topics(plan_id, topics)`. The server rejects:
- a fact without a source (`kind` is one of `knowledge|deliverable|web|github|note`, with a reference and a label);
- duplicate titles;
- anything on the plan's "never say" list.

It runs weekly by default (Monday 06:00) and on **Research again**.

### B9 · One brand kit, on its own Deliverables tab
- A **Brand kit** tab beside Blog, Reports and Templates, always visible and independent of Socials, replaces `BrandKitDialog` everywhere. Template Studio and Socials link to it.
- Storage stays in `workspace.settings['brand_kit']` and the existing routes.
- **Style references** are new: up to 24 images (PNG, JPG or WebP, 10 MB each), stored like the logo, each with a note and a stance (`like|avoid`).
- A **style profile** (palette, mood, composition, things to avoid) is read from them by a vision model: `create_llm_manager`, request type `brand_style_read`, timeout from config, usage tracked. It is re-read on change or on request.
- The profile goes into every composer, research and Template Studio image prompt. Liked images go to an AI tool only when its action accepts a reference image (a capability-registry flag) and the workspace allows it.

### B10 · AI tools: Composio only, with defaults per media type
The Brand kit's **AI tools** section lists the media toolkits the capability registry can offer, each with its state (connected, not connected, built in). Paid tools are connected in Composio only (D15; owner, 2026-10-02). Defaults per media type (images, AI images, footage, voice) live in `workspace.settings['media_tools']`. The monthly cap (exists) and a per-post override of `SOCIALS_MEDIA_POST_CAP_USD` are edited there. Templates and Kokoro stay free and need nothing.

### B11 · A planned slot comes before approval
A new `social_posts.planned_for` is set from the editor's **When**, or by a plan.
- Setting it is not a content change and does not reset an approval.
- Approving a post whose planned slot is in the future schedules it there (the existing `schedule()` path).
- A planned slot that passes unapproved follows the plan's `late_policy`: `skip` (the post ends `missed`, nothing posts) or `next_slot` (the next free slot of the same cadence row). A post without a plan is skipped.
- Approved posts move as today (`schedule()`).

### B12 · The Queue
Posts in `needs_approval`, ordered by slot, each showing:
- its deadline;
- the exact media, through `FilePreview`;
- each channel's copy and the claims with their sources.

Actions: **Approve** (schedules into the slot), **Request changes** (with a comment), **Make another take** (re-compose and re-render, a new hash) and **Reject**. **Approve all shown** appears only while the workspace's series-approval switch is on (PRD-251 D6), and approves each shown post by its hash.

### B13 · Kokoro voices can be chosen
A static catalogue of Kokoro voice ids and labels ships in config next to media-render's voice model. `GET /api/socials/voices` returns it with `lists_voices: true`, and the voice select lists it. The default stays `af_heart`.

## Stories

Every wave: **one** Alembic revision, chained onto `EXPECTED_HEAD` and safe when `create_all` ran first, with both head pins moved (`tests/test_prd209_alembic_single_head.py`, `tests/test_prd236_w1_routes.py`). Routes that touch the database are plain `def` (F105). The route manifest and config surface are updated. Commits carry DCO sign-off. CI on Postgres is the evidence; nothing runs on the build machine.

### Wave 1 — the Studio: calendar, post editor, Queue

**US-B101 · The wave's migration (S)**
- `prd251b_wave1` adds `social_posts.planned_for` (timestamptz, nullable, indexed with workspace_id) and `social_posts.length_seconds` (int, nullable).
- `compute_content_hash` covers `length_seconds` and not `planned_for`.
- Tests: create_all first, then upgrade twice. Exactly one head. The hash changes with the length and not with the slot.

**US-B102 · Templates for the gallery (M)**
- `GET /api/socials/templates?format=` returns each workspace social template's id, name, format, sizes, `durations` and `thumbnail_url`.
- A thumbnail is rendered once per template (first size, PNG) through media-render and stored on `DocumentTemplate.thumbnail_url`. Seeding or a backfill sets it; a template edit resets it.
- Tests: list shape; thumbnail set once; edit resets.

**US-B103 · The composer takes the editor's choices (M)**
- `ComposeRequest` gains `template_id` and `length_seconds` (B5), and save accepts `length_seconds`.
- A template of another workspace or format gets 422. A length the template does not declare gets 422.
- Render and preview use the chosen length's timeline.
- Tests for each refusal, and that a chosen template is used verbatim.

**US-B104 · Video lengths (M)**
- UI story promo and App promo gain a 15 s and a 30 s cut beside their current length, each a complete timeline declared in `blocks.durations`.
- The media-render CI job renders every declared length at half resolution.
- The other templates declare their single length.

**US-B105 · Planned slots (M)**
- `PUT /api/socials/posts/{id}/slot {planned_for, timezone}` works in draft, needs_approval and changes_requested, without touching approval.
- Approve with a future slot schedules the post (B11). A slot that passes unapproved ends the post `missed`, with the existing notification.
- Tests on Postgres: approve then schedule, slot passes then missed, and a slot edit keeps the approval.

**US-B106 · Off means invisible (S)**
- A workspace Settings toggle turns Socials on and off for owners and admins.
- While either switch is off, the marketplace listing omits the Socials package, tool discovery and Auto's keyword routing omit the `socials` actions, and the brand kit omits the handles.
- Tests for each surface with the master off, then with the workspace off.

**US-B107 · The Studio shell (S)**
- Socials sub-navigation (B2) with the Queue count. **New plan** and **New post** in the header.
- The view is held in the URL (`?view=calendar|queue|plans`, `?post=`).
- The phone layout follows PRD-246.

**US-B108 · The Socials calendar (L)**
- Month · Week · List on a month/week grid extracted from the Command Center's (B3).
- Each post chip shows time, format and length, title, channel badges and a status word. The status is never shown by colour alone.
- Channel filter. Drag moves a slot: `planned_for` before approval, `schedule()` after.
- A **Today** rail with the day's posts and the active plan's summary, plus a status key.
- The Command Center keeps its layer through the same grid.

**US-B109 · The post editor (L)**
A full page with these sections:
- **Brief**, with **Redraft with Auto**.
- **Format**: Image, Carousel (slides 2–10), Video (lengths from the template, voice, music, AI footage switch) or Text.
- **Channels and sizes**: one row per connected channel with the post kind and resulting size. Channels the format excludes are greyed, with the reason.
- **Look**: **Template**, a thumbnail gallery with "Let Auto pick"; **Upload**; **Library** (the workspace's image and video Deliverables). **AI-made** arrives in US-B305.
- **Claims and sources**.
- **When**: date, time, timezone.

Header actions: **Save draft**, **Render preview** and **Submit for approval**.

**US-B110 · The preview (M)**
- Tabs for the chosen channels. Each shows the channel's copy (editable, with a counter against the registry's limit) and the media at that channel's aspect, from the latest render through `FilePreview`.
- A stale render is marked.
- A note estimates render minutes (video: length × sizes). Images use none.

**US-B111 · The Queue (M)**
- The B12 page, reusing the PRD-251 approval view (exact media, per-channel copy, sources, the unsourced second confirmation).
- **Make another take** re-composes and re-renders.
- **Approve all shown** follows B12.
- A 409 reloads.

**US-B112 · The screenshot lane (M) — builds right after US-B106, before the UI stories**
- `.github/workflows/socials-studio-screens.yml` (non-required): on a push to a `feat/prd-251b-*` branch and on dispatch, `next build` in the local edition with a dead API URL, `next start`, Playwright screens of every Studio screen at 1440 and 390 px, uploaded as a run artifact.
- Fixtures mirror the mockup's sample data; the clock is frozen at 2026-10-14 07:20 Europe/London.
- Playwright lives in `frontend/screens/package.json`, installed by the workflow only; `frontend/package.json` is unchanged.
- A spec fails on an error boundary or a loading state after 15 s. Each UI story (US-B107–B111) adds its screens in the same commit.

### Wave 2 — plans and the content bank

**US-B201 · The wave's migration (M)**
- `prd251b_wave2` adds the plan columns to `social_campaigns` (B6) and `social_posts.slot_key`, unique with `campaign_id` where set.
- It creates `social_topics`: id, workspace_id, campaign_id, title, angle, facts (each with a source), formats, pinned_on, used_post_id, used_at, origin (`research|person`), timestamps; indexed by plan and use.
- Same migration tests as US-B101.

**US-B202 · Plans over HTTP (M)**
- Create, update, pause, resume and end.
- Cadence validation: known channels and formats, lengths the chosen template declares, times, days.
- `expand_slots(plan, from, to)` is pure and tested on DST changes and month ends.
- `GET /plans/{id}/slots?from&to` returns planned and made slots. `slot_overrides` holds moved and skipped slots.

**US-B203 · The content bank over HTTP (S)**
- List, add, edit, delete and pin topics.
- A fact without a source gets 422. The plan's "never say" phrases are refused.

**US-B204 · Research (M)**
- `platform_get_social_plan` (read) and `platform_add_social_topics` (write, draft-only, validated as B8).
- The seeded **Content bank research** playbook. A weekly schedule per plan, plus **Research again**.
- GitHub through Composio only when connected (mocked in tests).
- Tests: unsourced facts and duplicates are refused; the tools refuse while Socials is off.

**US-B205 · Making posts on their day (L)**
- The B7 tick: leader-only, idempotent per `slot_key`, videos a day early, quota and caps respected, empty bank notified.
- Each made post gets `planned_for` = its slot and goes to `needs_approval`.
- A **"Today's posts are ready"** digest goes through `NotificationDispatcher`.
- Tests on Postgres, including two ticks racing for one slot.

**US-B206 · Late policy (S)**
- `skip` or `next_slot` (B11), with notifications.
- Tests for both, and for the slot of a post that was approved late.

**US-B207 · The Plan page (L)**
The mockup's five steps:
1. Goal and dates.
2. Cadence: rows, with totals for posts per channel and render minutes.
3. What to research: sources, notes, "never say", research schedule.
4. Making and approving: make times, visual mix, AI tools summary, late policy, caps.
5. Content bank: topic cards with facts and sources, formats, use or pin, add, edit, **Research again**.

Also New plan, Pause and Save.

**US-B208 · Plans on the calendar (M)**
- Planned slots (not yet made) appear as dashed chips with their topic once picked.
- Dragging a planned slot writes `slot_overrides`.
- The Today rail shows the plan's cadence and bank counts.

### Wave 3 — one brand kit, style references, AI tools

**US-B301 · The Brand kit tab (M)**
- A Deliverables tab that is always visible, with the mockup's basics card: logo and mark, colours, fonts, voice, handles (B4).
- Template Studio and Socials link to it, and `BrandKitDialog` is retired.
- The existing routes are unchanged.

**US-B302 · Style references (M)**
- Upload, list, delete, note and stance under `/api/documents/brand-kit/references`, stored like the logo.
- Limits per B9. Type and size are checked on the server.
- Tests for each refusal and for cross-workspace access (404).

**US-B303 · The style profile (M)**
- The vision read (B9), shown in **What Auto takes from these**, with **Read the references again**.
- It is included in composer, research and Template Studio image prompts.
- Tests with the model mocked: what the prompt carries, and a re-read on change.

**US-B304 · AI tools (M)**
- The section from B10: toolkit rows from the capability registry, defaults per media type, and both caps.
- Connecting goes to Composio.
- Tests: defaults are validated against the toolkits that are actually offered.

**US-B305 · AI-made visuals in the editor (L)**
- **AI-made** in Look: AI image (stills through `GENERATE_IMAGE`, four options) and AI footage (hook and b-roll through `GENERATE_VIDEO`), using the defaults.
- Liked references are passed when the action accepts one.
- Words stay template text (D12). Spend is estimated, capped and booked (D13).
- The plan's visual mix (US-B207) uses the same path.

**US-B306 · Kokoro voices (S)**
- B13: the catalogue, the API and the select. Tests for the list and the default.

## Not in this PRD

- Generating a plan's content ahead of its days (owner, 2026-10-02).
- Our own API clients or keys for Higgsfield, fal, ElevenLabs or Fish Audio (owner, 2026-10-02: Composio only).
- Pixel-exact replicas of each platform's UI. The previews are generic frames at the right aspect.
- Phase 2 engagement (comments, replies, metrics), more than one account per toolkit, paid ads, analytics beyond receipts, and a template code editor for customers.

## Editions

Both editions get every story. The local edition has no render quota (PRD-251). The master switch's default stays off in both.

## Delivery plan (to Web Summit, 9 Nov)

1. The one-head fix merges, then this PRD is reviewed.
2. Wave 1 kit (12 stories: US-B101–B112), then launch (owner).
3. Owner test, then Wave 2, then owner test, then Wave 3.
4. Automatos' own countdown plan runs from the end of Wave 2 to 8 Nov.

Each wave's owner test (`docs/PRDS/prd251b-wN-owner-test.md`) covers the browser checks against the mockup.

## Open questions (owner)

1. **Command Center layer:** keep the read-only scheduled-post layer (recommended, B3) or drop it?
2. **Approve all shown:** only with the series-approval switch on (B12, PRD-251 D6), or always?
3. **Video cuts:** are 15 s and 30 s cuts of UI story promo and App promo (US-B104) the right first two?
4. **Defaults:** research weekly on Monday at 06:00, and make at 07:00, videos a day early?

## Traps (carried from PRD-251 and #852)

- **Migrations:** two heads break nine tests at once. Chain onto `EXPECTED_HEAD` and move both pins in the same commit. A merge revision is the fix when main moved meanwhile.
- **Secret scan:** the history scan reads every commit in the PR. A test literal after `api`, `key` or `token` with enough entropy is flagged forever: keep fixture strings short or low-entropy, and never quote a literal in a `.gitleaksignore` comment.
- **CodeQL:** CodeQL fails any regex with two ways to match the same text on request data. Use one-pass scanners, unambiguous numbers (`\d+(?:\.\d+)?`), and no `\s*…\s*` pairs around optional parts.
- **Shape ratchet:** a touched function must stay ≤ 50 code lines with nesting ≤ 4, and a new file ≤ 800 lines. Split frontend components under the length rule.
- **Config:** no `os.getenv` outside `config.py`.
- **CSS:** Tailwind 3.3.3 has no `dvh`. Phone CSS lives only in the compact region.
- **Merging:** never merge mid-run; a red CI on the branch is checked against main first.

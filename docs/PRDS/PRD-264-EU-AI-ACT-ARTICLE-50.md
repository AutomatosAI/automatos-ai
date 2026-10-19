# PRD-264: EU AI Act Article 50 transparency, always on

**Status:** Approved for build, 19 Oct 2026. Gerard's decision of 18 Oct (D1) sets the date: every Article 50(1) and 50(2) story merges by **Fri 30 Oct 2026**. Gerard approved this PRD and its defaults on 19 Oct: D11, D12 and D14 to D18 are taken. D9 stays open until a reviewer is named (§3). · **Owner:** Gerard · **Written:** 19 Oct 2026, grounded on `origin/main @ 4edb433a0` (#1189).
**Source:** `<workspace>/.claude/PLAN-PRICING-PUBLIC-RECORD-EU-AI-ACT.md` §6 (18 Oct 2026, three review passes), re-verified against the code and the law on 19 Oct. The legal text: Regulation (EU) 2024/1689 Art. 50, the AI Omnibus Regulation (EU) 2026/1744, the Commission's Article 50 Guidelines C(2026) 5054 final (20 Jul 2026), and the Code of Practice on transparency of AI-generated content (final, Jun 2026). Counsel confirms the reading (D9).
**Scope:** Article 50(1) notices, Article 50(2) marking and detection, the Article 50(4) tools deployers need, and the Article 50(5) presentation rules, on every surface Automatos provides. Most code ships in automatos-ai. The rest ships in automatos-widget-sdk, the product site, the company site, the GitBook docs, automatos-play, automatos-academy and automatos-academy-app. Part D adds the new Article 5 generation guard (ART5-1), which has its own date, Fri 20 Nov 2026.
**Supersedes:** PRD-132's Article 50 row (§5) and its Phase 1 disclosure bullet (§8). PRD-132 keeps everything else and is amended in the same PR as this PRD (§13).
**Depends on:** the PRD-261 theme host (#1143, #1155, #1156, merged), which EUAI-11a builds on; D14 (the XMP writer and the c2pa dependency) before EU-B1 starts on Tue 20 Oct; D9 (the legal reviewer) for EUAI-12's Terms clauses.
**Ralph config:** launch is human-only, with `RALPH_MODEL` pinned. One branch, one push, one PR and one CI run per wave per repo. At most four builders run at once across all Automatos work (§4).
**Human-gated, outside the autonomous loop:** D14's maintainer issue; the trust-list certificate application (D15); the legal sign-off (D9) and the Terms merges (EUAI-12); the production re-paste of the Academy tutor prompt (EUAI-11c); the manual contentcredentials.org check (EUAI-06b); the public status flip (EUAI-15), which waits for a production check; the Code of Practice decision (D12).

Labels used below: **[T]** read in code on `origin/main @ 4edb433a0` during this session; **[L]** checked against the legal text or two independent legal summaries on 19 Oct; **[I]** inferred.

## 1. Introduction

### The law

Article 50 of the AI Act puts transparency duties on **providers** of AI systems (Automatos) and on **deployers** (our customers, and Automatos itself for its own sites and posts). The AI Omnibus, Regulation (EU) 2026/1744, published 24 Jul 2026 and in force since 27 Jul 2026, left Article 50 in place and added one grace period, for 50(2) marking only [L].

| Article | Duty | Who | Applies from | What it means here |
|---|---|---|---|---|
| 50(1) | A person is told they are dealing with an AI, unless that is obvious. The "obvious" exception is read narrowly (Guidelines ¶45). An AI agent states that it is an AI **and on whose behalf it acts** (¶31). A notice only in the Terms, a URL, documentation or a brand name is not enough, and neither is metadata the person can't perceive (¶38). Voice says it in words at the start, with reminders in longer calls (¶37). An AI agent's email carries an AI label at the top (¶36). | Provider | **2 Aug 2026, no grace period: overdue today** | The widget and its popups, channel bots, Auto Live voice, agent-sent email and messages, the in-app chat (default, Play and theme-owned), both website chats and both Academy tutors |
| 50(2) | Generated audio, image, video and text are marked in a machine-readable format and are detectable as AI-generated. Means of detection are available to the people exposed to the content (¶75). Marks carry no personal data (¶94). Out of scope: short strings such as UI labels, captions and alt text, and source code (¶68). | Provider | 2 Aug 2026 for systems placed on the market after that date. **2 Dec 2026** for systems already on the market before it [L] | media-render first shipped on 25–26 Sep 2026, after 2 Aug, so its marking is due now. Chat, the widget, images and documents predate 2 Aug, so their legal date is 2 Dec 2026. **Automatos targets 30 Oct for all of them (D1).** |
| 50(3) | Emotion recognition and biometric categorisation: inform the people exposed | Deployer | 2 Aug 2026 | No such feature. Out of scope. |
| 50(4) ¶1 | Deep fakes (realistic AI image, audio or video of real people, places or things, cloned voices) are disclosed | Deployer | 2 Aug 2026 | Customers need labelling tools. Automatos labels its own posts. |
| 50(4) ¶2 | AI text published to inform the public on matters of public interest is disclosed, unless a person reviewed it and holds editorial responsibility | Deployer | 2 Aug 2026 | The blog widget and social captions |
| 50(5) | Clear and distinguishable, at the latest at first exposure, and accessible | Provider and deployer | 2 Aug 2026 | The presentation rules for every story (§8) |
| 50(7) and the Code | The Code of Practice is voluntary. Signatories commit to signed metadata plus an imperceptible watermark, and to interoperable detection by 2 Feb 2027 [L]. A non-signatory shows by other means how it meets 50(2). | Optional | Signatories: 2 Feb 2027 | D11 (watermark) and D12 (sign, or write a gap analysis) |
| 99(4) | Fines up to €15m or 3% of worldwide annual turnover. For an SME, whichever is lower. | Both | 2 Aug 2026 | Authorities can act on a complaint |
| 2(12) | The open-source exemption does **not** cover Article 50 | Provider | 2 Aug 2026 | The local edition carries every notice and mark. Every story covers both `AUTH_EDITION` values. |
| 5 (new) | No AI system may generate non-consensual intimate content or child sexual abuse material. A safe harbour covers systems with effective technical safeguards that reliably prevent such output [L]. | Both | **2 Dec 2026** | Part D, ART5-1, merged by Fri 20 Nov |

High-risk obligations moved to 2 Dec 2027 (Annex III) and 2 Aug 2028 (Annex I). They are out of scope here (§7).

### What exists and what's missing (19 Oct, `origin/main @ 4edb433a0`)

There is no Article 50 code in automatos-ai yet. `orchestrator/modules/policy/ai_act.py` is the Article 14 oversight mapping, not Article 50 [T]. No `ai_disclosure`, `ai_generated` or `is_ai` field exists in any payload, and no C2PA, XMP or IPTC code exists anywhere [T].

| Area | Exists | Missing |
|---|---|---|
| Widget (50(1)) | "Powered by Automatos AI" footer, a brand line only (automatos-widget-sdk `packages/chat-widget/src/components/chat-panel.ts:27-34`) | Any AI notice. The header title defaults to "Automatos AI" and deployers can override it (`components/header.ts:10`). `config.greeting` is declared (`packages/core/src/types.ts:17`) but the widget never renders it. The proactive popup has no AI wording, and the cart-idle popup reuses the same class (`packages/loader/src/proactive/proactive-popup.ts`, `src/cart-idle/index.ts`). Aria labels say "Open chat" and "Assistant is typing". `PUBLIC_WIDGET_CONFIG_KEYS` has no disclosure key (`orchestrator/api/widgets/config.py:48-52`). The chat-widget and blog-widget packages have no tests. [T] |
| Website chats (50(1)) | **Fixed 19 Oct.** The product site reads "Auto · AI assistant" with an AI greeting (automatos-ai-landing `src/components/widgets/AutomatosChat.tsx:69-71`). The company site's served branch says "Hi, I'm Auto, Automatos's AI assistant" (automatos-ai-landingV2 `feat/one-os-family`, `_chat.js:95`). Both work around the unrendered greeting by putting "AI assistant" in the title. [T] | The support skill still introduces itself as "the support assistant" (automatos-skills `support/support-responder/SKILL.md:49`) |
| In-app chat (50(1)) | `personality.py` frames Auto as "a capable AI assistant" (`orchestrator/consumers/chatbot/personality.py:155`, `:159`) [T] | No AI label in the default chat (`frontend/components/chatbot/multimodal-input.tsx`, placeholder "Send a message..."), in Mission Mode (inside `components/chatbot/chat.tsx`), in the Play chat (`components/play/chat/play-chat.tsx`, `message-box.tsx`, `auto-reply.tsx`) or in any theme-owned `/chat`. The `/chat` layout (`frontend/app/chat/layout.tsx`, 32 lines) wraps all of them. `/x` is reserved for theme routes (`frontend/lib/theme-host/package-manifest.ts:58`, `:96`), but `frontend/app/x/` doesn't exist yet. The theme SDK's chat area is a stub (`frontend/lib/theme-sdk/chat.ts`). No prompt tells any agent to answer truthfully about being an AI; "never claim to be human" has zero hits. [T] |
| Academy tutors (50(1)) | **Web tutor fixed 19 Oct** (automatos-academy `fb17b6e`: `public/js/tutor.js:341`, `:535-536` say "AI tutor"). The tutor prompt says "say plainly that you are an AI" (`docs/ACADEMY_TUTOR_PROMPT.md:25`, `:120`). [T] | The production tutor agent still runs the old prompt until it is re-pasted (Gerard's go). The app tutor reads "TUTOR" and "Ask the tutor" (automatos-academy-app `src/compliance/tutorCopy.ts:17-18`), and its consent copy never says AI (`src/compliance/copy.ts:36-40`; version `consent-2026-07-10.1` at `:16`). [T] |
| Channel bots (50(1)) | — | **Two outbound paths, neither discloses.** The adapter path replies at `orchestrator/channels/base.py:209` through each `*_adapter.py`'s own `send_message`. The driver path sends through `channels/sender.py` `send_to_channel` and `channels/drivers/{telegram,slack,discord,whatsapp,webhook}.py`. `_deliver_reply` (`api/webhooks.py:683`), notifications (`core/services/notification_service.py`), agent asks (`modules/tools/discovery/handlers_asks.py`) and the destinations dispatcher (`services/destinations/dispatcher.py`) all use the driver path. The Telegram /start reply says "Connected… Send any message to talk to your agents" (`channels/telegram_adapter.py:108-114`). [T] |
| Voice (50(1)) | — | The begin frame is an empty response 0, so the caller speaks first (`orchestrator/modules/voice/live_socket.py:59-60`, `:183`). Retell's protocol speaks the content of response 0 as the opening line. `VoiceCallPanel.tsx` has no AI label. [T] |
| Agent email and messages (50(1)) | Owner send cards (`orchestrator/modules/tools/discovery/owner_only.py:160`, `is_composio_send`, used at `:140`, `:283`, `:324`, `:365`) [T] | No "written by an AI agent on behalf of…" line. The model writes the body; `unified_executor.py` checks parameter keys but never changes content. [T] |
| Media marking (50(2)) | TikTok `is_aigc` from `footage_generated` (`orchestrator/modules/socials/channel_adapters.py:457`; `service.py:336-344`); the receipt states TikTok's flag only (`frontend/components/deliverables/socials/socials-post-receipts.tsx:45`) [T] | No mark on any media-render output: `render.mp4`, `preview.mp4`, `render.png`, `render-NN.png`, `preview-NN.png`, `mix.wav` and the per-line Kokoro WAVs (`services/media-render/media_render/pipeline.py:416`, `audio.py:231`, `kokoro_tts.py:43`, `:66`). The JPEG step re-encodes through ffmpeg and would strip marks (`media_render/jpeg.py:36`). media-render's requirements have no c2pa or XMP writer. YouTube upload (`channel_adapters.py:487`) sets no synthetic-media flag; `containsSyntheticMedia` has zero hits. [T] |
| Documents and images (50(2)) | — | PDFs (`orchestrator/modules/documents/pdf_writer.py:16`, WeasyPrint `<70`), DOCX (`python-docx`, `docxtpl`; `modules/documents/blocks/docx_renderer.py`) and XLSX (`modules/documents/xlsx_render.py`) carry no AI mark. Blog images come from `tools/discovery/actions_blog.py` and `handlers_blog.py`. No JSON-LD exists for blog posts. [T] |
| Detection (50(2)) | — | No verify route, no provenance record, no documentation |
| Deployer tools (50(4)) | `BlogPost.author_agent_id` (`orchestrator/core/models/core.py:1743`, `:1756`) [T] | No deep-fake setting in Socials, no burned-in label, no blog AI label or record of editorial review. The SDK's `BlogPost` type has `author_name` only (`packages/core/src/types.ts:329-342`). [T] |
| Terms | — | Nothing on AI marks or deployer duties (automatos-ai-landing `src/pages/Terms.tsx`; automatos-ai-landingV2 `terms.html`) [T] |
| Posture page | **Updated 19 Oct** on the served branch: the Art. 50 row reads "In progress… being built now. Until they ship, add the disclosure in your own copy" (automatos-ai-landingV2 `eu-ai-act.html:257`); the date table carries the Omnibus dates (`:302-303`); the checker says the notices "are being built now" (`_aiact-checker.js:79`) [T] | The flip to what is live, when it is live (EUAI-15) |
| Theme rules | PRD-261 defines `pages` and `routes` (automatos-play `docs/prd/core/PRD-261-INSTALLABLE-THEMES.md:192-193`) [T] | No host-owned AI label rule in PRD-261 or the theme skill (`skills/build-automatos-interface/references/chat-stream.md`) [T] |

Current anchors that later stories rely on: the single Alembic head is `prd263_harness_runs`, pinned as `EXPECTED_HEAD` at `orchestrator/tests/test_prd209_alembic_single_head.py:108` and asserted in `test_prd236_w1_routes.py:274-277`. Routes the frontend calls are listed in `orchestrator/reports/route-manifest.json`. Setting names go in `orchestrator/reports/config-surface.json`. `AUTH_EDITION` is `local` or `saas`, default `saas` (`orchestrator/config.py:201-202`). [T]

### Why a new PRD

PRD-132 is a broad, four-phase plan in which compliance features are opt-in ("Compliance features do **not** appear in the default UX", `docs/PRDS/132-EU-AI-ACT-ALIGNMENT.md:6`). Its timeline predated the Omnibus. Its Art. 50 row let a workspace switch the disclosure off (`:81`), and its Phase 1 put the disclosure in response metadata only (`:125`), which Guidelines ¶38 says is not enough. Article 50 is a legal floor with a date of its own, so it gets its own PRD and its own Ralph kit. PRD-132 is amended in the same PR (§13).

## 2. Goals

- **Every place a person meets an Automatos AI tells them so at first exposure,** in words, in both editions. That covers every surface in FR-1, and each has a test that fails if its notice is missing. No setting, theme, deployer override or workspace flag removes it.
- **Every file media-render writes with an AI element carries a machine-readable mark:** IPTC DigitalSourceType in XMP from Fri 23 Oct, and a signed C2PA manifest from Fri 30 Oct. CI reads both back for every output format.
- **Generated documents, images and chat messages carry a machine-readable AI mark,** and anyone can check a file at a public detection page that answers "made by Automatos AI: yes or no".
- **Customers get the 50(4) tools:** a deep-fake label in Socials, on by default when realistic AI media is present, and an "AI-generated" label on agent-written blog posts until a person marks them reviewed.
- **Every story above merges by Fri 30 Oct 2026** (D1). ART5-1 merges by Fri 20 Nov 2026.
- **No public page claims a notice or a mark before it is merged and live in production** (EUAI-15).

## 3. Decisions

Decision numbers D9 and D11 to D15 are the 18 Oct plan's, so they match the plan's text. D1 to D3 restate decisions already taken. D16 to D18 are new here. Gerard took D11, D12 and D14 to D18 on 19 Oct; D9 is open.

**D1. Every Article 50(1) and 50(2) story merges by Fri 30 Oct 2026 (Gerard, 18 Oct).** No 50(1) or 50(2) story is scheduled after that date. If a story can't make it, the date stays: the builder states the risk and how to reduce it, and Gerard decides. *(every story)*

**D2. Article 50 is always on, in both editions, with no off switch.** PRD-132's design principle made compliance features opt-in, its Art. 50 row allowed an off switch, and its Phase 1 put disclosure in metadata only. Article 50 is a legal floor, so all three are reversed for Article 50 (§13). Deployers may reword a notice, but they can't remove it (D17). *(FR-1 to FR-4)*

**D3. A focused PRD for Article 50, and PRD-132 amended, not rewritten.** PRD-132 stays the broad, opt-in compliance plan. *(§13)*

**D9. Who signs off legal copy (open, default applied).** Gerard names one reviewer by Wed 21 Oct. The legal floor is the same either way: the Article 50 duties apply by law whatever the Terms say. If nobody is named by Wed 21, the default is that EUAI-12's Terms clauses ship short and plain, unreviewed, by Fri 30 Oct, and counsel reviews them later. The alternative is no Article 50 clauses in the Terms until a reviewer signs off. *(EUAI-07, EUAI-12, ART5-1)*

**D11. No imperceptible watermark now (taken 19 Oct).** Article 50(2) asks for marking that is effective, interoperable, robust and reliable as far as technically feasible. It doesn't name a watermark. EUAI-06, EUAI-09 and EUAI-10 deliver metadata, signing and detection by Fri 30 Oct. A watermark is a Code of Practice commitment. If Gerard signs the Code after Web Summit, EUAI-13 (L) is scheduled for Tue 2 Feb 2027. If Gerard wants it now, it gets its own lane from Tue 20 Oct and merges by Fri 30 Oct, and it adds to the delivery risk. *(EUAI-13)*

**D12. Don't sign the Code of Practice before Web Summit; write a gap analysis (taken 19 Oct).** Signing is voluntary. A non-signatory still has to show how it meets 50(2). EUAI-14 writes `docs/compliance/ARTICLE-50-GAP-ANALYSIS.md` by Fri 30 Oct, and the posture page links it. Signing is revisited after Web Summit. *(EUAI-14)*

**D13. Delivery risk: keep 30 Oct for every story.** De-risking is built into §4: the EUAI-06 approvals by Tue 20 Oct, EUAI-06 split into metadata (by Fri 23) and signing (by Fri 30), a fixed merge order with a seam review, a trial merge with the PRD-261, PRD-262 and PRD-263 branches, and a **checkpoint on Mon 26 Oct**. A story that is off track that day gets an extra lane. If no lane is free, Gerard chooses between a narrower first cut and accepting the gap, with the legal floor stated. *(§4, §14)*

**D14. The c2pa dependency and the XMP writer in `services/media-render` are approved (taken 19 Oct).** AGENTS.md requires a maintainer issue before any new dependency. EUAI-06b needs the Content Authenticity Initiative's open-source SDK, `c2pa-python`. The same issue names the XMP writer EUAI-06a uses, if it is not already in the image (§9). *(EUAI-06a, EUAI-06b)*

**D15. Start with a self-signed signing certificate and apply for a C2PA trust-list certificate on Tue 20 Oct (taken 19 Oct).** Verifiers show a self-signed manifest as "unknown signer", and it is still machine-readable and detectable. The trust-list certificate is swapped in later as configuration, with no code change. *(EUAI-06b)*

**D16. The public detection route is a named exception to "every new endpoint is authenticated" (taken 19 Oct: allow, hash only).** AGENTS.md requires every new endpoint to be authenticated and authorised. Article 50(2) and Guidelines ¶75 want detection available to the people who see the content, and they have no Automatos account. Default: one public route that takes a SHA-256 hash, never a file, and returns only "made by Automatos AI: yes/no", the media kind and the date. It returns nothing about the workspace or the person. A small public page computes the hash in the visitor's browser, so the file never leaves their machine. The route is rate-limited per IP, listed in the route manifest, and reviewed by the security-reviewer before merge. The alternative is an authenticated route plus documentation pointing people to contentcredentials.org/verify, which only reads C2PA files. *(EUAI-10)*

**D17. Deployers can reword a notice, but its fixed AI badge stays (taken 19 Oct).** The plan's rule, "the wording must still contain 'AI'", breaks in Portuguese, Spanish, French and Italian ("IA") and German ("KI"). Default: a fixed badge the deployer can't change, whose text comes from a per-locale table in code (`AI`, `IA`, `KI`, and English as the fallback), plus a first-turn sentence the deployer may rewrite. The server rejects a rewrite that doesn't contain the locale's term from the same table. *(EUAI-02a, EUAI-02b)*

**D18. At most four builders at once (taken 19 Oct).** The plan ran six automatos-ai lanes plus the other repos in parallel. Above about four concurrent builders, whole waves have stalled on the shared subscriptions (16 Oct 2026), and other PRDs' lanes draw on the same capacity. This PRD schedules four build lanes (§4). The small copy edits in the other repos are done by one session between waves, not by extra builders. If Gerard adds capacity, EU-C2 splits into two lanes first. *(§4)*

## 4. Sequencing

**Rules for every wave:** one branch, one push, one PR and one CI run per repo. CI is the only gate; nothing runs on a developer machine. Worktrees live at `<workspace>/.worktrees/<repo>/<branch-slug>/`. automatos-ai merges with `gh pr merge --admin`; when squash-merging a Ralph wave, set `--subject` and `--body` so story commits' `[skip ci]` lines don't reach `main`. Stage explicit paths only.

**Four build lanes (D18).** Each lane is one builder at a time. Copy and doc edits in the other repos are done by one session, not a builder.

| Lane | Tue 20 – Fri 23 Oct | Mon 26 – Fri 30 Oct |
|---|---|---|
| 1 · media and Socials labels | **EU-B1** `feat/prd-264-media-metadata`: EUAI-05, then EUAI-06a, which reuses EUAI-05's AI-media rule for the render flags. Built Mon 19 – Thu 22, merged Fri 23. A session prepares EUAI-06b's signing set-up on a scratch branch from Tue 20. | **EU-B2** `feat/prd-264-c2pa-signing`: EUAI-06b. Built Mon 26 – Thu 29, merged Fri 30. |
| 2 · conversations backend | **EU-A1** `feat/prd-264-art50-core`: EUAI-02a (including `ai_generated` on widget stream messages), EUAI-03, EUAI-04. Built Mon 19 – Fri 23, merged as soon as CI is green (target Mon 26). | **EU-C1** `feat/prd-264-sends-deployer`: EUAI-07, EUAI-08a, EUAI-08b. Built Mon 26 – Wed 28, merged Thu 29. |
| 3 · surfaces | **EU-A2** `feat/prd-264-chat-label`: EUAI-11a. Built Tue 20 – Wed 21, merged Mon 26 at the latest. Then **EU-W** (automatos-widget-sdk `feat/ai-disclosure-notice`): EUAI-02b, EUAI-08c, built from Thu 22. | EU-W merged and published by Wed 28 |
| 4 · marking and detection | **EU-C2** `feat/prd-264-marking-detect`: EUAI-09, EUAI-10, EUAI-12 (guide), EUAI-14. Built from Tue 20. | EU-C2 built by Wed 28, merged Thu 29 |
| Session (not a builder) | **EU-0** `docs/prd-264-art50`: this PRD, the PRD-132 corrections, the last EUAI-01 rows. Merged Wed 21. **pl** EUAI-11b (Wed 21). **aca** EUAI-11d (Wed 21 – Fri 23). | **EU-D**: EUAI-12 Terms and GitBook (Mon 26 – Fri 30). **EUAI-15** status flip after the production check (Fri 30). |

**The pricing lanes in the same plan also need builders.** The 18 Oct plan's P2 (`fix/pricing-counts-2026-10`, merge Mon 26) and P4 (BG-1, merge Fri 30) run in the same window. With them, the window needs six builders, which is above the limit in D18. This PRD does not re-date them. Gerard decides how they share capacity (§11, Q1).

**Merge order on automatos-ai `main`:** EU-0 (Wed 21) → EU-B1 (Fri 23) → EU-A1 and EU-A2 (Mon 26) → EU-C1 (Thu 29) → EU-C2 (Thu 29) → EU-B2 (Fri 30). Each branch rebases on `main` after the one before it merges, and before its single push.

**Seams.** These files are touched by more than one branch, and the rebase diff gets a seam review on each: `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/reports/route-manifest.json` (EUAI-10), `orchestrator/modules/policy/ai_disclosure.py` (created in EU-A1, extended in EU-C1), `frontend/lib/theme-host/` (EUAI-11a and the PRD-261 waves), `services/media-render/` (EU-B1, EU-B2, and EUAI-08a's burned-in label), and the Alembic heads. Before any branch is marked done, every still-open sibling is trial-merged onto current `main` in a scratch worktree: this PRD's other branches **and** the open PRD-261, PRD-262 and PRD-263 branches. Seams are resolved there, and the resolution goes into the branch about to push.

**Alembic.** Two stories add a migration: EUAI-08b (EU-C1) and EUAI-10 (EU-C2). The first to rebase chains its `down_revision` onto the head at that moment (`prd263_harness_runs` today, or whatever has merged since), the second chains onto the first. Each moves both head pins (AGENTS.md).

**Checkpoint Mon 26 Oct (D13).** Every story is on track for Fri 30, or it gets an extra lane that day, or Gerard chooses between a narrower first cut and accepting the gap, with the legal floor stated.

**Freeze Sat 31 Oct – Mon 2 Nov.** Product Hunt is Tue 3 Nov.

**Part D:** EU-F `feat/art5-generation-guard`, ART5-1, Wed 4 – Fri 20 Nov.

| Wave | Repo · branch | Stories | Est. changed lines, excl. tests [I] | Done when |
|---|---|---|---|---|
| EU-0 | automatos-ai `docs/prd-264-art50` | PRD-264, §13, EUAI-01 docs rows | docs only | Merged Wed 21 |
| EU-B1 | automatos-ai `feat/prd-264-media-metadata` | EUAI-05, EUAI-06a | ~400 | Merged Fri 23; CI reads the XMP back for every format |
| EU-A1 | automatos-ai `feat/prd-264-art50-core` | EUAI-02a, 03, 04 | ~500 (over AGENTS.md's ~400; one wave, one PR by Gerard's rule) | Merged Mon 26 |
| EU-A2 | automatos-ai `feat/prd-264-chat-label` | EUAI-11a | ~300 | Merged Mon 26 |
| EU-W | automatos-widget-sdk `feat/ai-disclosure-notice` | EUAI-02b, 08c | ~350 | Merged and published to `widgets.automatos.app/v0` by Wed 28 |
| EU-C1 | automatos-ai `feat/prd-264-sends-deployer` | EUAI-07, 08a, 08b | ~550 | Merged Thu 29 |
| EU-C2 | automatos-ai `feat/prd-264-marking-detect` | EUAI-09, 10, 12 (guide), 14 | ~600, plus docs | Merged Thu 29 |
| EU-B2 | automatos-ai `feat/prd-264-c2pa-signing` | EUAI-06b | ~400 | Merged Fri 30; manifests read back in CI; one contentcredentials.org check recorded |
| EUAI-11b | automatos-play `docs/prd-261-ai-label` | EUAI-11b | docs only | Merged Wed 21 |
| EUAI-11d | automatos-academy-app `fix/tutor-ai-disclosure` | EUAI-11d | ~40 | Merged Fri 23 |
| EU-D | automatos-ai-landing `fix/art50-terms`, automatos-ai-landingV2 `fix/art50-status`, automatos-gitbook `docs/article-50` | EUAI-12, EUAI-15 | copy | Deployed Fri 30 |
| EU-E | automatos-ai `feat/prd-264-watermark` | EUAI-13, only if D11 = build now | ~600 | Merged Fri 30, with its own lane |
| EU-F | automatos-ai `feat/art5-generation-guard` | ART5-1 | ~300 | Merged Fri 20 Nov |

## 5. User stories

Sizes: S ≤ 1 day, M 2–3 days, L ~5 days of agent build. Every story covers both editions (`AUTH_EDITION=local` and `saas`). Every story ships the test that fails without it (AGENTS.md). New settings are read through `orchestrator/config.py` (or the service's own config module) and named in `orchestrator/reports/config-surface.json`. No story grows a file already over 800 lines: new logic goes in a new small module, and the big file only calls it.

### Part A: Article 50(1), telling people they are dealing with an AI

#### EUAI-01: The last overclaims and wrong dates go (S, mostly done 19 Oct)
**Description:** As a reader of anything Automatos publishes, I want no page to claim Article 50 work as shipped and every date to match the Omnibus, so nothing public is false while the notices are built.
**Repos · wave:** automatos-ai (EU-0, session); automatos-skills (session) · **Depends on:** —

Done on 19 Oct and not repeated: the product site (posture line in `src/pages/blog/AutomatosNotWrapper.tsx:514`, the chat title and greeting), the company site's served branch (`_chat.js:95`, `eu-ai-act.html` dates and Art. 50 row, `_aiact-checker.js:79`), the Academy course dates (automatos-academy `7790625`, `814f9e0`) and the web tutor label (`fb17b6e`).

**Acceptance criteria:**
- [ ] PRD-132 carries the corrections in §13.
- [ ] Three automatos-ai docs say what is real: `docs/documentation-prd-corpus/runbooks-operational-docs.md:92`, `docs/authentication-multi-tenancy/teams-members-invitations.md:362`, and the "AI Act Compliance" section of `docs/governance-compliance-security/audit-gdpr-ai-act-compliance.md`, rewritten from `docs/compliance/README.md` (no "would typically contain").
- [ ] automatos-skills `support/support-responder/SKILL.md:49` reads "I'm Auto, Automatos's AI support assistant." and `:80` reads "EU AI Act: what is built and what is in progress (never 'compliant' or 'aligned by design')".
- [ ] `git grep -niE "aligned by design|always-on platform features|genuinely built in|compliance is architectural"` prints nothing on the served refs of automatos-ai, the product site and the company site, outside PRD-132's history and this PRD.

#### EUAI-02a: The widget API sends an AI notice the client can't drop (S)
**Description:** As a visitor using a customer's chat widget, I want the server to send the AI notice with the widget's config and its first chat event, so any client, the SDK or a custom one, has the notice before the first answer.
**Repo · wave:** automatos-ai · EU-A1 · **Files:** `orchestrator/api/widgets/config.py` (`PUBLIC_WIDGET_CONFIG_KEYS` at `:48-52`), `orchestrator/api/widgets/chat.py` (`_event_stream` at `:406`), `orchestrator/modules/policy/ai_disclosure.py` (new), tests · **Depends on:** —

**Acceptance criteria:**
- [ ] `ai_disclosure` is added to `PUBLIC_WIDGET_CONFIG_KEYS`. It is always present in the public config, built by the server, never read raw from settings: `{"badge": <locale term>, "line": <first-turn sentence>, "locale": <code>}`.
- [ ] The badge comes from one table in `ai_disclosure.py` (`en` → `AI`, `pt`, `es`, `fr`, `it` → `IA`, `de` → `KI`; any other locale → `AI`). No widget setting changes it (D17).
- [ ] The default line is "You are chatting with an AI assistant for {site}.", where `{site}` is the site's name, falling back to the workspace name. A deployer's own line is accepted only if it contains the locale's badge term as a whole word. Otherwise the settings save returns HTTP 422 with "The AI notice must say it is an AI (for this language: '{term}')."
- [ ] The first SSE event of `POST /api/widgets/chat` is `{"type": "ai_disclosure", ...}`, sent before any token.
- [ ] Every assistant message event in the widget stream carries `"ai_generated": true` (Art. 50(2) for widget text; EUAI-09 covers the in-app stream).
- [ ] Tests: the config always carries `ai_disclosure`, including when the workspace has no widget settings; a line without the term is refused; a Portuguese line with "IA" is accepted; the first stream event is the notice; both editions.

#### EUAI-02b: The widget shows an AI badge and a first-turn notice that no override removes (M)
**Description:** As a visitor, I want the widget to say it is an AI in its header, its popups and its first message, so I know before I type.
**Repo · wave:** automatos-widget-sdk · EU-W · **Files:** `packages/chat-widget/src/components/header.ts`, `chat-panel.ts`, `packages/chat-widget/src/chat-widget.ts`, `packages/loader/src/proactive/proactive-popup.ts`, `packages/core/src/types.ts`, new vitest config and tests in `packages/chat-widget` · **Depends on:** EUAI-02a (contract; build against it from Thu 22)

**Acceptance criteria:**
- [ ] The header shows a persistent badge with `ai_disclosure.badge` beside the title. The deployer's `title` override changes the title, never the badge.
- [ ] Before the first message, the panel renders `ai_disclosure.line` as the opening turn. This also fixes the unrendered `config.greeting`: the greeting, if set, renders after the notice.
- [ ] The proactive and cart-idle popups (one class) show the badge and the words "AI assistant" in the locale's form.
- [ ] Aria labels name the AI: "Open AI assistant chat", "AI assistant is typing".
- [ ] If the config arrives without `ai_disclosure` (an old server), the SDK uses the English default. It never renders an unlabelled widget.
- [ ] Vitest proves the badge and the opening line survive a `title` override, a `greeting` override and a theme colour override, and that the popup carries the badge.
- [ ] The loader version is bumped and published to `widgets.automatos.app/v0` by the existing CDN workflow. That changes both website chats and the Shopify theme embed with no change in their repos.
- [ ] The product site and the company site drop their "· AI assistant" title workaround once the badge is live (session, EU-D).

#### EUAI-03: Channel bots say they are an AI, and for whom, on first contact (M)
**Description:** As a person messaging a workspace on Telegram, Slack, Discord or WhatsApp, I want the first reply to say it comes from an AI and for whom, so I know what I'm talking to.
**Repo · wave:** automatos-ai · EU-A1 · **Files:** `orchestrator/modules/policy/ai_disclosure.py`, `orchestrator/channels/base.py` (`:209`), `orchestrator/channels/sender.py` (`send_to_channel`), `orchestrator/channels/drivers/webhook.py`, `orchestrator/channels/telegram_adapter.py` (`:108-114`), `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] One function in `ai_disclosure.py` decides whether a message to a person needs the notice: the first message to that person on that connection, the first after an idle gap (`AI_DISCLOSURE_IDLE_HOURS`, default 24), and any reply to a message that asks whether they are talking to a human or a bot (a small multilingual pattern list in the same module).
- [ ] The notice, "AI assistant for {workspace}.", is prepended at **both** outbound points: the adapter reply at `base.py:209` and `send_to_channel` in `sender.py`, which carries webhook replies, notifications, agent asks and destination dispatches.
- [ ] The generic webhook driver gets `"ai_generated": true` and `"ai_disclosure": "<notice>"` as payload fields instead of a text prefix, because its receiver is a system.
- [ ] The last-notice time per workspace, connection and person is kept in Redis with a TTL equal to the idle gap. If Redis can't be read, the notice is added (fail toward telling).
- [ ] The Telegram /start reply says "I'm an AI assistant for {workspace}." in its first sentence.
- [ ] An architecture test fails if any module other than `base.py` calls an adapter's `send_message`, or any module other than `sender.py` calls a driver's `send`, so a new path can't skip the notice.
- [ ] Tests for each adapter and each driver: the first message carries the notice once, a second within the gap doesn't, one after the gap does, "are you human?" triggers it, and Redis down adds it.

#### EUAI-04: Auto Live voice says it is an AI at the start of the call and reminds during it (S)
**Description:** As a caller, I want to hear that I'm talking to an AI before I say anything, and to be reminded in a long call, so the voice can't pass for a person.
**Repo · wave:** automatos-ai · EU-A1 · **Files:** `orchestrator/modules/voice/live_socket.py` (`:59-60`, `:183`), `frontend/components/voice/VoiceCallPanel.tsx`, `orchestrator/config.py`, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] The begin frame (response 0) carries `VOICE_AI_GREETING`, default "Hi, this is Auto, your AI assistant." It is never empty, so Auto speaks first.
- [ ] After an automatic reconnect, and every `VOICE_AI_REMINDER_MINUTES` of call time (default 10), the next response starts with "Just so you know, I'm an AI assistant." (Guidelines ¶37).
- [ ] `VoiceCallPanel.tsx` shows "AI voice assistant" for the whole call, using theme tokens.
- [ ] Pytest: the begin frame isn't empty and contains "AI"; a reconnect adds the reminder; a fake clock past the interval adds it once. Vitest: the panel label renders during a call.

#### EUAI-05: Each network's own AI label is set at publish (S)
**Description:** As a customer publishing a post with AI media, I want each network's AI label set for me where the network supports one, so the post is labelled where it is seen.
**Repo · wave:** automatos-ai · EU-B1 (moved from EU-A1 on 19 Oct, because EUAI-06a reuses its rule) · **Files:** `orchestrator/modules/socials/ai_media.py` (new), `orchestrator/modules/socials/service.py` (`footage_generated` at `:336-344`; the file is 1,149 lines, so it only calls the new module), `orchestrator/modules/socials/channel_adapters.py` (`:457`, `:487`), `frontend/components/deliverables/socials/socials-post-receipts.tsx`, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] `ai_media.py` decides whether a post carries AI media: AI footage (today's `footage_generated` rule), AI stills, and synthetic voice (Kokoro, the default, and any voice toolkit). A post whose media are all the person's own uploads and templates is not AI media.
- [ ] TikTok's `is_aigc` is true whenever the post carries AI media.
- [ ] YouTube: the builder reads Composio's `YOUTUBE_UPLOAD_VIDEO` schema and records it in the PR. If the action accepts `containsSyntheticMedia` (or the YouTube status field it maps to), it is set from `ai_media.py`. If it doesn't, the receipt says "Set 'Altered or synthetic content' in YouTube Studio for this video."
- [ ] For LinkedIn, X and Instagram, the builder checks each Composio publish schema for an AI-content flag and records the result. Where none exists, the receipt says the caption label (EUAI-08a) carries the disclosure.
- [ ] The receipt states the AI label decision for every channel, not only TikTok.
- [ ] The label decision is derived from the render record at publish time, so it can't drift from what was rendered. The voice stays outside the content hash, as today.
- [ ] Tests: each media mix gives the right decision; TikTok and YouTube parameters are set; the receipt line for each channel.

#### EUAI-07: Agent-sent email and messages say at the top that an AI wrote them, and for whom (M)
**Description:** As someone receiving an email or message an agent wrote, I want to see at the top that an AI agent wrote and sent it on someone's behalf, so I know who I'm dealing with.
**Repo · wave:** automatos-ai · EU-C1 · **Files:** `orchestrator/modules/policy/ai_disclosure.py` (a data map of send actions to their body parameter), `orchestrator/modules/tools/discovery/owner_only.py` (`is_composio_send` at `:160`), the point in the tool executor where the owner card is built, tests · **Depends on:** EUAI-03 (the module)

**Acceptance criteria:**
- [ ] Every external send an agent starts (`is_composio_send` true) carries, as the first line of its body, "Written and sent by an AI agent on behalf of {workspace}." (Guidelines ¶36). HTML bodies get it as a first paragraph.
- [ ] A data map in `ai_disclosure.py` names each send action's body parameter (for example `GMAIL_SEND_EMAIL` → `body`). A completeness test fails if any action that `is_composio_send` accepts has no entry.
- [ ] The line is added before the owner's send card is built, so the card shows the final body and the owner approves exactly what goes out.
- [ ] Sends a person makes directly are unchanged. The builder lists every caller of `orchestrator/api/widget_email.py` `send_email` (`:445`) and `reply_to_email` (`:493`) in the PR. If an agent can reach either, the line is added there too.
- [ ] Until counsel says otherwise (D9), the owner's approval doesn't remove the line.
- [ ] Tests: an agent send gets the line at the top, plain and HTML; the card shows it; a direct send doesn't; the completeness test.

#### EUAI-11a: Every in-app chat shows "Auto is an AI", and no theme can remove it; agents answer truthfully about being an AI (M)
**Description:** As a user chatting with Auto anywhere in the app, whatever theme is installed, I want a label that says Auto is an AI, and an honest answer if I ask, so a theme or persona can never make Auto look human.
**Repo · wave:** automatos-ai · EU-A2 · **Files:** `frontend/components/ai-label/ai-label.tsx` (new), `frontend/app/chat/layout.tsx`, `frontend/components/play/chat/play-chat.tsx` (or its view), the theme SDK chat area (`frontend/lib/theme-sdk/chat.ts`), `frontend/lib/theme-host/__tests__/ai-label.test.ts` (new), the shared system-prompt assembly for every agent, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] `frontend/app/chat/layout.tsx` renders `<AiLabel />` outside `PlayRouteSwitch` and outside the theme's page slot. It shows on the default chat (with Mission Mode, which lives in `components/chatbot/chat.tsx`), the Play chat and any theme-owned `/chat`. The default chat input doesn't render a second copy.
- [ ] Play's other chat surfaces outside `/chat` (today `frontend/app/conversations/page.tsx`) render the same label.
- [ ] `/x/` theme routes: if `frontend/app/x/` exists when this branch rebases, its layout renders the label for every theme whose manifest `uses` declares the chat area. If it doesn't exist yet, EUAI-11b's rule makes the PRD-261 wave that adds it render the label, and that wave's tests prove it.
- [ ] The label is "Auto is an AI" (or "{agent name} is an AI" when another agent answers), uses theme tokens, never `dark:` classes, and has `role="note"` with the same words for screen readers.
- [ ] **No theme can hide it.** On mount and when the theme changes, the label checks that it is displayed, visible, not zero-sized and inside the viewport. If not, the chat input is disabled with "This theme hides the AI notice, so chat is paused. Switch theme in Settings." No manifest field turns the label off.
- [ ] Theme-host test (`ai-label.test.ts`): the label renders with builtin Play, with a fixture theme that registers `/chat`, and with a fixture theme whose CSS sets `display:none`, `visibility:hidden`, `opacity:0` or zero size on it, and in each hiding case the input is disabled.
- [ ] Every agent's system prompt, Auto's, marketplace agents' and custom personas', ends with a platform rule: "If anyone asks whether they are talking to a person or an AI, say that you are an AI agent acting for {workspace}. Never claim or imply that you are human." It is appended after the persona text, so a persona can't remove it. A test asserts the rule in the assembled prompt for Auto, a marketplace agent and an agent with a custom persona.
- [ ] **[UI]** Owner check on the deployed build: the label shows on `/chat` in the default look and in Play, in light and dark.

#### EUAI-11b: PRD-261 and the theme skill state the host's AI label rule (S)
**Description:** As a theme author, I want the rules to say that the host's AI label stays on every chat surface, so I don't build a theme that breaks it.
**Repos · wave:** automatos-play (session, Wed 21); automatos-ai (`docs/theme-host/`, in EU-A2) · **Depends on:** —

**Acceptance criteria:**
- [ ] The theme skill (automatos-play `skills/build-automatos-interface/references/chat-stream.md`, tracked) says: the host renders the AI label on `/chat` and on every `/x/` route that uses the chat area; a theme can't remove, cover or restyle it away; chat pauses if it is hidden.
- [ ] The same rule goes in automatos-ai's theme-host docs (`docs/theme-host/`), which are tracked.
- [ ] PRD-261 (`docs/prd/core/PRD-261-INSTALLABLE-THEMES.md`) gets the rule beside the `pages` field (`:192`), the `routes` field (`:193`) and the "own design" list (`:294`). That folder is kept out of git on purpose (`.git/info/exclude`), so the file is edited in place and is **not** committed unless Gerard says so.
- [ ] The PRD-261 wave that builds `/x/` and the SDK chat area carries the rule as an acceptance criterion: the chat stream doesn't start until the host label is mounted.

#### EUAI-11c: The Academy tutor agent runs the prompt that says it is an AI (S, owner step)
**Description:** As a learner, I want the live tutor to say it is an AI when asked, as its prompt already says.
**Repo:** automatos-academy (no code; production config) · **Depends on:** —

**Acceptance criteria:**
- [ ] → OWNER: with Gerard's go, the production tutor agent's prompt is replaced with `docs/ACADEMY_TUTOR_PROMPT.md` at `origin/main`, and the date is recorded in the PR for EU-D.

#### EUAI-11d: The Academy app tutor says "AI tutor" (S)
**Description:** As a learner in the app, I want the tutor's header and consent text to say it is an AI.
**Repo · wave:** automatos-academy-app · `fix/tutor-ai-disclosure` (session) · **Files:** `src/compliance/tutorCopy.ts` (`:17-18`), `src/compliance/copy.ts` (`:16`, `:36-40`), `src/compliance/__tests__/tutorCopy.test.ts`, `copy.test.ts` · **Depends on:** —

**Acceptance criteria:**
- [ ] The header reads "AI TUTOR" and "Ask the AI tutor". The tutor consent copy says the tutor is an AI.
- [ ] `CONSENT_COPY_VERSION` moves from `consent-2026-07-10.1` to a new dated version, so learners see the changed consent.
- [ ] Both test files are updated and pass in the app's CI.

### Part B: Article 50(2), marking generated content and making it detectable

#### EUAI-06a: Every media-render file with an AI element carries IPTC "AI-generated" metadata (M)
**Description:** As anyone who receives a video, image or audio file Automatos made, I want it to say in its metadata that AI made it, so tools can detect it.
**Repo · wave:** automatos-ai `services/media-render` · EU-B1 · **Files:** `services/media-render/media_render/provenance.py` (new), `pipeline.py` (`_ffmpeg` at `:416`, outputs at `:462`, `:504`, `:513`), `jpeg.py` (`:36`), `audio.py` (`:231`), `kokoro_tts.py` (`:66`), the render job contract, media-render's Dockerfile and requirements (per D14), tests · **Depends on:** D14 (the XMP writer)

**Acceptance criteria:**
- [ ] The render job tells media-render which inputs are AI-made (AI footage, AI stills, synthetic voice) and which are uploads or templates. The orchestrator sets it from the same rule as EUAI-05.
- [ ] After the **last** encode of every output, and after `to_jpeg`, the file gets an XMP packet with `Iptc4xmpExt:DigitalSourceType`: `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia` when every visual and audio element is AI-made, `compositeWithTrainedAlgorithmicMedia` when AI and non-AI elements are mixed. Renders made only from templates and existing content get no AI mark.
- [ ] Covered outputs: `render.mp4`, `preview.mp4`, `render.png`, `render-NN.png`, `preview-NN.png`, JPEGs from `to_jpeg`, `mix.wav`, and per-line voice WAVs that leave the service.
- [ ] The packet holds no user, workspace, prompt or file-path data (Guidelines ¶94). A test parses every packet and fails on any field outside an allowlist.
- [ ] Writer: exiftool in the media-render image, unless D14 names another. No hand-written byte patching of MP4 boxes.
- [ ] A CI test renders one fixture per format and reads the DigitalSourceType back with the same tool.

#### EUAI-06b: The same files carry a signed C2PA manifest (L)
**Description:** As anyone checking a file, I want a signed Content Credentials manifest that says AI made it and that Automatos generated it, so its origin can be verified.
**Repo · wave:** automatos-ai `services/media-render` · EU-B2 · **Files:** `provenance.py`, media-render's config module, requirements (`c2pa-python`, D14), tests · **Depends on:** EUAI-06a; D14; D15; EUAI-10 (provenance rows)

**Acceptance criteria:**
- [ ] Each file EUAI-06a marks also carries a C2PA manifest with a `c2pa.actions` assertion (`c2pa.created`, same `digitalSourceType`) and the claim generator "Automatos AI media-render/{version}". No user or workspace identifier appears in it.
- [ ] Where an input already carries a C2PA manifest, it is kept as an ingredient. The JPEG step signs its own output and keeps the input's manifest as an ingredient.
- [ ] The signing certificate and private key come from media-render's config (environment variables, Railway secrets). Nothing secret is committed. CI generates a throwaway self-signed certificate at test time, and gitleaks stays clean.
- [ ] The hosted service signs with the D15 certificate: self-signed until the trust-list certificate arrives, then swapped by configuration only. The local edition creates its own self-signed certificate on first render, kept in its data volume and never in the image, so local renders are marked too (Art. 2(12); counsel confirms, D9).
- [ ] If signing fails, the file still ships with EUAI-06a's XMP, and the failure is logged with the job id and counted. A render never ships unmarked.
- [ ] media-render returns each marked file's SHA-256 and manifest id, and the orchestrator writes the provenance row (EUAI-10).
- [ ] A CI test reads every manifest back for each format and checks the `digitalSourceType`. → OWNER: one file checked on contentcredentials.org/verify, with a screenshot in the PR.

#### EUAI-09: Documents, images and chat messages carry a machine-readable AI mark (M)
**Description:** As anyone who receives a document, image or chat answer Automatos generated, I want a machine-readable mark that AI made it.
**Repo · wave:** automatos-ai · EU-C2 · **Files:** `orchestrator/modules/documents/pdf_writer.py` (`:16`), `modules/documents/blocks/docx_renderer.py`, `modules/documents/xlsx_render.py`, the blog image path (`modules/tools/discovery/actions_blog.py`, `handlers_blog.py`), the chat SSE payload and message rendering, the blog API, a new `orchestrator/modules/documents/provenance_marks.py`, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] Generated PDFs carry the IPTC DigitalSourceType in their XMP metadata. DOCX and XLSX carry it as a custom document property (`AIGenerated` with the IPTC term). The builder uses the libraries already pinned. Any new one goes through D14's issue first.
- [ ] Generated images keep any upstream C2PA manifest and get the IPTC XMP mark when they have none.
- [ ] Every assistant message in the chat SSE payload carries `"ai_generated": true`, and the rendered message has `data-ai-generated="true"`. (The widget stream's `ai_generated` ships in EUAI-02a, which owns `api/widgets/chat.py`.)
- [ ] The blog API returns `ai_generated: true` for posts with `author_agent_id` set. EUAI-08b adds the "and not marked reviewed" condition when its columns land.
- [ ] A read-back test for each format: PDF, DOCX, XLSX, PNG, JPEG, and one SSE message.

#### EUAI-10: A provenance record and a public detection route (M)
**Description:** As anyone holding a file, I want to ask Automatos whether it made it, without an account and without uploading the file.
**Repo · wave:** automatos-ai · EU-C2 · **Files:** a new model and one Alembic revision, `orchestrator/api/provenance.py` (new), `orchestrator/reports/route-manifest.json`, a public page (below), `docs/compliance/DETECTION.md`, the GDPR erasure cascade, tests · **Depends on:** D16

**Acceptance criteria:**
- [ ] A `content_provenance` table holds one row per marked file: `sha256` (indexed), `kind` (video, image, audio, document), `manifest_id` (nullable), `generator` (toolkit or model name), `created_at`, and `workspace_id` for tenancy and erasure only. It holds no prompt, user or file name. The migration follows AGENTS.md (one head, both pins, `IF NOT EXISTS`).
- [ ] Media-render files (EUAI-06b), documents and images (EUAI-09) write a row when they are produced. Chat text is marked in the payload only (EUAI-09), not logged.
- [ ] `GET /api/public/provenance/{sha256}` validates a 64-character lower-case hex value, returns `{"made_by_automatos_ai": true, "kind": ..., "date": "YYYY-MM-DD"}` or `{"made_by_automatos_ai": false}`, never anything about the workspace, and is rate-limited per IP (`PROVENANCE_RATE_PER_MINUTE`, default 30). It is the one public exception named in D16. It is listed in the route manifest.
- [ ] A public page at `/verify` computes the SHA-256 in the browser (Web Crypto), sends only the hash, and shows the answer. It also links contentcredentials.org/verify for C2PA files. It needs no sign-in in either edition: the route is added to the auth middleware's public list and to `HOST_OWNED_ROUTES`, so no theme can own it.
- [ ] Workspace erasure deletes the workspace's provenance rows (`gdpr_service.erase_workspace`).
- [ ] `docs/compliance/DETECTION.md` says how to check a file, what the answer means and its limits: a re-encoded or edited file has a new hash and reads "no".
- [ ] Tests: the hash validator; a known hash says yes and an unknown one says no; the answer never contains workspace data; the rate limit returns 429; erasure removes rows. The security-reviewer agent reviews the route before the PR is marked ready.

#### EUAI-14: The Code of Practice gap analysis (S, D12 default)
**Description:** As Automatos, not a Code signatory for now, I want a written account of how our marking meets Article 50(2), so we can show it on request.
**Repo · wave:** automatos-ai · EU-C2 · **Files:** `docs/compliance/ARTICLE-50-GAP-ANALYSIS.md` (new) · **Depends on:** EUAI-06a, EUAI-06b, EUAI-09, EUAI-10 (describes them)

**Acceptance criteria:**
- [ ] For each Code commitment (signed metadata, watermark, detection, interoperability, the 2 Feb 2027 date), it says what Automatos does, what it doesn't, and why the marking is effective, interoperable, robust and reliable as far as technically feasible.
- [ ] It names the watermark gap and the D11 decision, and dates the revisit after Web Summit.
- [ ] The posture page links it (EUAI-15).

#### EUAI-13: An imperceptible watermark (L; only if D11 = build now, or after Gerard signs the Code)
**Description:** As a signatory of the Code, Automatos adds a watermark that survives re-encoding, as a second layer behind the metadata.
**Repo · wave:** automatos-ai · EU-E · **Depends on:** D11, D12

**Acceptance criteria:**
- [ ] An open-source audio watermark (AudioSeal-type) and image watermark (TrustMark-type) are applied in media-render, chosen in a maintainer issue (new dependencies).
- [ ] EUAI-10's route also detects the watermark.
- [ ] Merged by Fri 30 Oct if built now, or by Tue 2 Feb 2027 if Gerard signs the Code after Web Summit.

### Part C: Article 50(4) tools for deployers, the Terms and the public record

#### EUAI-08a: Socials labels realistic AI media as AI-generated, on by default (M)
**Description:** As a customer posting realistic AI people, places, products or voices, I want the post labelled as AI-generated by default, so I meet my duty as a deployer without remembering to.
**Repo · wave:** automatos-ai · EU-C1 · **Files:** `orchestrator/modules/socials/ai_media.py` (from EUAI-05), the composer settings, the render request to media-render (label overlay), the caption builder, the receipt, tests · **Depends on:** EUAI-05

**Acceptance criteria:**
- [ ] The composer has a setting, "Realistic AI people, places, products or voice: label as AI-generated". It is on by default whenever the post carries AI media (EUAI-05), and the customer can switch it off for a post, with the reason shown: "you are responsible for disclosing deep fakes".
- [ ] When on: video gets a visible "AI-generated" label for the first seconds (`SOCIALS_AI_LABEL_SECONDS`, default 3) and again at an interval (`SOCIALS_AI_LABEL_INTERVAL_SECONDS`, default 30); the caption starts with "AI-generated." in the post's language; an audio-only post gets a spoken "This audio was generated by AI." at the start.
- [ ] The setting's value is part of the approved content, so changing it needs a new approval.
- [ ] The receipt shows the setting.
- [ ] Automatos's own posts use the default.
- [ ] Tests: the default follows the media; the overlay request is sent; the caption prefix; the audio line; the setting changes the content hash.

#### EUAI-08b: Agent-written blog posts are labelled AI-generated until a person reviews them (S)
**Description:** As a customer publishing agent-written posts, I want them labelled AI-generated unless someone has reviewed them and taken editorial responsibility, so I meet 50(4) ¶2.
**Repo · wave:** automatos-ai · EU-C1 · **Files:** `orchestrator/core/models/core.py` (`BlogPost` at `:1743`), one Alembic revision, the blog API, the blog editor UI, tests · **Depends on:** —

**Acceptance criteria:**
- [ ] `BlogPost` gains `reviewed_by_user_id` and `reviewed_at` (nullable). One migration adds them with `IF NOT EXISTS`, chained on the head at rebase, with both head pins moved.
- [ ] A workspace member can mark a post reviewed, which records who and when. Editing the post afterwards clears the review.
- [ ] The blog API returns `ai_generated` (true when `author_agent_id` is set and the post is not reviewed) and `reviewed_at`.
- [ ] Tests: the flag in each state; editing clears the review; another workspace's post can't be marked.

#### EUAI-08c: The blog widget shows "AI-generated" on unreviewed agent posts (S)
**Repo · wave:** automatos-widget-sdk · EU-W · **Files:** `packages/blog-widget/src/blog-widget.ts`, `packages/core/src/types.ts` (`BlogPost` at `:329-342`), new tests · **Depends on:** EUAI-08b (contract)

**Acceptance criteria:**
- [ ] `BlogPost` gains `ai_generated?: boolean`. When true, the post card and the post page show "AI-generated" near the byline, and the page emits JSON-LD that marks the post as AI-generated.
- [ ] Vitest covers both states.

#### EUAI-12: The Terms and a provider-versus-deployer guide (S)
**Description:** As a customer, I want the Terms and a plain guide to say what Automatos does for Article 50 and what I must still do.
**Repos · waves:** automatos-ai (guide, EU-C2); automatos-ai-landing `src/pages/Terms.tsx` and automatos-ai-landingV2 `terms.html` (EU-D); automatos-gitbook (EU-D) · **Depends on:** D9

**Acceptance criteria:**
- [ ] `docs/compliance/ARTICLE-50.md` lists, surface by surface, what Automatos provides (the notices, the marks, the detection route) and what the deployer does (disclose deep fakes and public-interest text they publish, keep the notices when embedding, not strip marks). The GitBook page says the same.
- [ ] Both Terms forbid removing or hiding AI notices and marks, and say the deployer's duties under 50(3) and 50(4) are the deployer's. The clauses promise nothing beyond what is live.
- [ ] The legal sign-off is recorded in the EU-D PRs: the reviewer's name, or Gerard's D9 choice.

#### EUAI-15: The posture page and the support agent say what is live, only once it is live (S, owner check)
**Description:** As a reader of automatos.app/eu-ai-act, I want it to list exactly which notices and marks are live, so the page is never ahead of the product.
**Repos · wave:** automatos-ai-landingV2 (`eu-ai-act.html:257`, `_aiact-checker.js:79`, "Reviewed" at `:187`, `:196`, `:335`); the support agent's knowledge doc (production data) · EU-D · **Depends on:** the EU-A1, EU-A2, EU-W, EU-B1 and EU-C2 merges, deployed

**Acceptance criteria:**
- [ ] → OWNER: before the flip, each notice and mark is checked in production (widget badge, a channel first message, the voice greeting, the in-app label, a marked render read back, the verify page), and the results are listed in the PR.
- [ ] The Art. 50 row and the checker list only what passed that check, and say what the deployer still adds. The gap analysis is linked.
- [ ] The "Reviewed" dates show the day of the change.
- [ ] The support agent's knowledge doc carries the same status, word for word.
- [ ] Nothing is flipped for a story that hasn't merged and deployed.

### Part D: Article 5, the new generation ban (legal date 2 Dec 2026)

#### ART5-1: Generation refuses non-consensual intimate content and child sexual abuse material (M)
**Description:** As Automatos, I want every image, video and voice generation path to refuse sexual content of real, identifiable people and any sexual content involving minors, so the platform has the effective safeguards the new Article 5 ban and its safe harbour expect.
**Repo · wave:** automatos-ai · EU-F, Wed 4 – Fri 20 Nov · **Files:** `orchestrator/modules/socials/ai_options.py` (`make` at `:126`), `modules/socials/recipes/footage_toolkits.py` (`HF_FAILED` at `:426`), every other media-generation tool, a new `orchestrator/modules/policy/generation_guard.py`, tests · **Depends on:** D9 (counsel on scope)

**Acceptance criteria:**
- [ ] Every image, video and voice generation path calls `generation_guard` with the prompt and any reference images before calling a provider. It refuses sexual content showing a real, identifiable person and any sexual content involving minors.
- [ ] A provider refusal (for example Higgsfield's `nsfw` status) ends as a refusal with a plain message. It is never retried on another provider.
- [ ] Refusals are logged with the workspace, the path and the reason code, never with the prompt or the output.
- [ ] A test per generation path, and a test that a provider refusal isn't retried.
- [ ] Counsel confirms the scope against the Omnibus text (D9).

## 6. Functional requirements

- **FR-1:** Each of these surfaces tells a person they are dealing with an AI at first exposure, in words: the chat widget and its proactive and cart-idle popups; both website chats (through the widget); the in-app `/chat`, including Mission Mode, the Play chat and any theme-owned `/chat`; Play's other chat surfaces and every `/x/` theme route that uses the chat area; Auto Live voice; Telegram, Slack, Discord and WhatsApp bots; agent-sent email and messages; the Academy web tutor; the Academy app tutor.
- **FR-2:** No workspace setting, widget setting, theme, persona or environment flag turns a notice off. Deployers may reword the widget's first-turn line only within D17's rule.
- **FR-3:** Every notice works the same in `AUTH_EDITION=local` and `saas`.
- **FR-4:** An AI agent's notice says on whose behalf it acts, wherever it acts for a workspace: channel bots, agent-sent email and messages, and the widget.
- **FR-5:** Every agent's system prompt ends with the platform rule to answer truthfully that it is an AI (EUAI-11a).
- **FR-6:** Every file media-render writes with an AI element carries IPTC DigitalSourceType XMP (by Fri 23 Oct) and a signed C2PA manifest (by Fri 30 Oct), after the last encode.
- **FR-7:** Generated PDFs, DOCX, XLSX and images carry an AI mark. Chat messages carry `ai_generated` in the payload and the DOM.
- **FR-8:** A public, rate-limited route answers whether a file's SHA-256 belongs to content Automatos generated, and a public page lets anyone check a file without uploading it.
- **FR-9:** Automatos never strips an upstream AI mark. Re-encoding steps carry it forward or re-mark.
- **FR-10:** No mark, manifest or detection answer contains personal data or workspace identifiers.
- **FR-11:** Socials labels realistic AI media as AI-generated by default; agent-written blog posts show "AI-generated" until a person marks them reviewed.
- **FR-12:** Each network's own AI label is set at publish where its API supports one, and the receipt says what was set.
- **FR-13:** Every image, video and voice generation path refuses sexual content of real, identifiable people and any sexual content involving minors (ART5-1, by Fri 20 Nov).
- **FR-14:** No public page states that a notice or mark is live before it has been checked in production (EUAI-15).

## 7. Non-goals

- **High-risk obligations** (Art. 6, 9–15, 43; Annex III from 2 Dec 2027, Annex I from 2 Aug 2028). They stay in PRD-132.
- **Compliance Mode, the Compliance Pack, auditor dashboards and Annex IV generation.** PRD-132, roadmap.
- **Article 50(3).** Automatos has no emotion-recognition or biometric-categorisation feature.
- **Signing the Code of Practice** (D12) and **an imperceptible watermark** (D11), unless Gerard chooses otherwise.
- **Any claim of being "compliant", "certified" or "aligned by design".** Public copy uses the posture line and says what is live (EUAI-15).
- **Labelling on behalf of a deployer where the law puts the duty on the deployer.** For 50(4), Automatos ships the tools and sensible defaults. Customers decide, and the Terms say it is their duty (EUAI-12).
- **The Shopify app, Markets and the separate products.** The Shopify theme embed inherits the widget notice through the loader (EUAI-02b), with no Shopify code change.
- **Removing marks from content a customer uploads.** Automatos marks what it generates. Upstream marks are kept, never stripped (FR-9).

## 8. Design considerations

### Notice copy (English defaults)

| Surface | Persistent label | First-exposure line |
|---|---|---|
| Widget header and popups | Badge `AI` beside the title; popups read "AI assistant" | "You are chatting with an AI assistant for {site}." (deployer may reword, D17) |
| In-app chat, Mission Mode, Play, theme `/chat` and `/x/` routes | "Auto is an AI" by the input | — (the label is always visible) |
| Voice panel | "AI voice assistant" for the whole call | Spoken: "Hi, this is Auto, your AI assistant." |
| Channel bots | — | "AI assistant for {workspace}." before the first reply, and again after the idle gap |
| Agent-sent email and messages | — | At the top: "Written and sent by an AI agent on behalf of {workspace}." |
| Academy web and app tutors | "AI tutor" in the header | Empty state and first turn say "AI tutor" |
| Socials deep-fake label | On-video label at the start, caption label, spoken line for audio-only posts | "AI-generated" |
| Blog widget | "AI-generated" on agent-written posts until reviewed | — |

### Presentation rules (Art. 50(5), Guidelines ¶37–¶38)

- **At first exposure:** the notice appears before or with the AI's first output, never after a scroll or a click.
- **Perceivable:** a visible label with enough contrast in both themes, using theme tokens (not `dark:` classes, which are inert in this frontend). Screen readers get the same words through `aria-label`.
- **Persistent where the AI keeps talking:** the badge and labels stay for the whole session. A voice call adds a spoken reminder after reconnect and at an interval (EUAI-04).
- **Not hideable:** no setting, theme, deployer config or CSS removes it. Where a theme controls the DOM, the host renders the label outside the theme's slot (EUAI-11a).
- **Localised:** badge terms come from one table (D17). English is the fallback when a locale has no entry.

## 9. Technical considerations

- **One module for the rules.** `orchestrator/modules/policy/ai_disclosure.py` holds the badge table, the notice texts, the first-contact rule and the send-action body map. Channel code, the widget API, the voice socket and the send path call it. Nothing else builds a notice string.
- **Channels have two outbound paths** (adapters and drivers, §1). EUAI-03 covers both and adds a test that fails if a third appears.
- **Big files.** `orchestrator/modules/socials/service.py` is 1,149 lines, so new Socials logic goes in `modules/socials/ai_media.py`. Check every touched file's size before adding to it (AGENTS.md code shape).
- **The XMP writer.** media-render has no XMP or C2PA tooling today. ffmpeg doesn't write XMP, and its encodes drop metadata, so marking runs after the last encode. Default writer: exiftool in the media-render image, covered by D14's issue. `c2pa-python` (contentauth) signs and reads manifests for MP4, PNG, JPEG and WAV. The builder confirms each format against the version pinned.
- **Signing keys** are secrets: environment variables read by media-render's config module, set on Railway, never in the repo, an image or a log. CI makes a throwaway certificate per run.
- **The render contract** gains the AI-input flags (EUAI-06a). The orchestrator sets them from the same rule as EUAI-05, so a socials post and its file always agree.
- **Migrations:** two new revisions (EUAI-08b, EUAI-10), chained in merge order, both head pins moved each time, `IF NOT EXISTS` throughout.
- **The public route** (D16) is the only unauthenticated endpoint this PRD adds. It reads one indexed column, returns no workspace data and is rate-limited. The `/verify` page is host-owned and public in both editions.
- **Redis** holds the channel notice state with a TTL. Losing it only means a person sees the notice again.
- **Theme host:** the label is rendered by the host outside the theme's slot, and the chat pauses if the label can't be seen. CSS alone can't guarantee visibility against arbitrary theme CSS, so the runtime check is the guarantee and the hostile-theme test is the proof.
- **Widget SDK:** the chat-widget and blog-widget packages have no tests today. EUAI-02b and EUAI-08c add a vitest config to each, following `packages/loader/vitest.config.ts`.
- **Composio schemas** for YouTube, LinkedIn, X and Instagram are read at build time, not assumed. Each finding is recorded in the PR.

## 10. Success metrics

- **On Fri 30 Oct:** every FR-1 surface shows its notice in production, checked and listed in EUAI-15's PR.
- **On Fri 23 Oct:** 100% of media-render output formats read back an IPTC DigitalSourceType in CI. **On Fri 30 Oct:** 100% read back a C2PA manifest too.
- **Zero** public claims ahead of production on the release dates (EUAI-15).
- **Zero** personal or workspace data in any mark, manifest or detection answer (tests in EUAI-06a, EUAI-06b, EUAI-10).
- **The detection route** answers "yes" for every file in the media-render CI fixture set.

## 11. Open questions for Gerard

Gerard approved D11, D12 and D14 to D18 on 19 Oct (§3). These stay open:

| # | Question | Recommendation |
|---|---|---|
| Q1 | The plan's pricing lanes P2 and P4 need two more builders in the same window, which takes the total to six (§4, D18). | Keep four lanes for this PRD, because its date is legal. Either add capacity for P2 and P4 or run them as sessions with one verifier per story. This PRD doesn't re-date them. |
| Q4 | D9: who reviews the Terms clauses by Wed 21 Oct? | Name one reviewer. Without one, ship short, plain clauses unreviewed (default). |
| Q7 | EUAI-11b: PRD-261's folder is kept out of git on purpose. Edit it in place only? | Yes. The tracked theme skill and theme-host docs carry the rule. |
| Q8 | EUAI-11c: re-paste the Academy tutor prompt into the production agent. | Yes, any day this week. |

## 12. Testing plan

- **CI is the only gate.** Nothing runs on a developer machine. Backend tests in `orchestrator/tests/` (pytest), frontend tests with vitest, widget SDK tests with vitest in each package, media-render read-back tests in its own CI step.
- **Each story's tests fail without the change.** Read-back tests open the real output file. They never check that a function was called.
- **Owner checks** on the deployed build: the in-app label in light and dark (EUAI-11a), one contentcredentials.org verification (EUAI-06b), and the production pass before the status flip (EUAI-15).
- **Before each push:** the trial merge with the open sibling branches and a seam review (§4).

## 13. PRD-132 amendments (in this PR)

1. The status line records the 19 Oct amendment.
2. An Article 50 exception follows the design principle at `:6`: always on, every workspace, both editions, outside Compliance Mode.
3. A banner says Article 50 scope moved to PRD-264.
4. The §1 timeline is replaced with the Omnibus dates: 2 Aug 2026, 2 Dec 2026, 2 Dec 2027 and 2 Aug 2028.
5. "856+ Composio apps" becomes "1,000+".
6. The Art. 50 row (`:81`) loses its off switch and points here. The Phase 1 metadata-only bullet (`:125`) points here and says metadata alone is not enough.
7. §10, the Compliance Pack, is marked as roadmap, not shipped.

## 14. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Too much work for ten working days with four lanes (D18) | HIGH | Fixed lane plan (§4); 06a ships the mark on its own by Fri 23; checkpoint Mon 26 (D13) |
| 2 | c2pa dependency or signing set-up slips (D14, D15) | HIGH | Approval on Mon 19; set-up on a scratch branch from Tue 20; self-signed fallback; 06a already marks every file |
| 3 | A theme or deployer hides the notice | HIGH | The host renders it outside the theme slot; the chat stream won't start until it is mounted; a browser test with a hostile fixture theme (EUAI-11a) |
| 4 | Public detection route abused or used to probe workspaces | MED | Hash only, no file upload, no workspace data in the answer, per-IP rate limit, security review (D16) |
| 5 | Seams between parallel branches and PRD-261/262/263 work | HIGH | Fixed merge order, seam review, trial merge, alembic chain rebased on the live head (§4) |
| 6 | A notice claimed publicly before it is live | HIGH | EUAI-15 flips status only after a production check |
| 7 | Terms clauses ship unreviewed (D9 default) | MED | Short, plain clauses that promise nothing; counsel reviews later; the legal floor holds either way |
| 8 | The disclosure line changes the meaning of an approved send | MED | The send card shows the final body with the line before the click; the approval hash covers it (EUAI-07) |
| 9 | Metadata stripped downstream (social networks re-encode) | MED | The platform AI flags (EUAI-05) and visible labels (EUAI-08) carry the disclosure; detection by hash still answers for the original file |
| 10 | Marks carry personal data by mistake | MED | FR-10; a test asserts no workspace or user id in any manifest or XMP packet |

## Related

- PRD-132 (EU AI Act alignment, amended here)
- PRD-261 (installable themes: the theme host and `/x/` routes)
- PRD-181 (`docs/compliance/`: audit log, Art. 12 and Art. 14 mapping)
- PRD-207 and PRD-258 (Auto Live voice)
- PRD-251 (Socials)
- PRD-008-A (widget popups and callback)
- Regulation (EU) 2024/1689 Art. 50; Regulation (EU) 2026/1744; Commission Guidelines C(2026) 5054 final; the Code of Practice on transparency of AI-generated content (Jun 2026)

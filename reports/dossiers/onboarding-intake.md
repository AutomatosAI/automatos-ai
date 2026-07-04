# Dossier — Onboarding intake (wizard → Mission Zero)

**Module key:** `onboarding-intake` · **Group:** platform · **Tier:** standard · **Status (map):** live (discovered unit, PRD-130) · **Pinned tree:** `origin/main @ 77bc9c6d5`
**Maturity: 2 / 5** · **Verdict: EXTEND (lean on Firecrawl harder; generalise the archetype layer; wire the entry point) — do NOT rebuild, do NOT adopt a whole vendor.**
**Headline:** A well-engineered async intake pipeline that is dark-launched, single-vertical, untested, and has never run in production — the code is honest and robust, but as a *capability* it is not yet earning its place in the first-run autonomy moment.

> Scope note: section **F (enterprise bar / adversarial-input / tenant-isolation)** is deliberately omitted — it runs as the separate Opus defensive-hardening pass per the brief. This dossier is A–E, G–J.

---

## A. What it is

`onboarding-intake` is PRD-130: a six-step "Business Intake Wizard" that turns a customer's **website domain** into a launched **Mission Zero** — the first autonomous mission a new workspace ever runs. The user gives a domain and a few goal chips; the platform maps the site with Firecrawl, LLM-extracts structured facts from a chosen set of pages, ingests them into the workspace RAG corpus, (optionally) builds the workspace knowledge graph, synthesises an editable `BusinessProfile`, and finally renders that profile into a rich natural-language goal that is handed to `CoordinatorService.create_mission()`. Mission Zero is executed by four hidden global system agents — **VOYAGER** (research), **BLUEPRINT** (roster architect), **SCRIBE** (persona/playbook writer), **FORGE** (workspace builder) — whose job is to research the business and propose the workspace's agent team. It is the platform's answer to "cold-start a workspace from a URL," and it is the **first impression of Auto's autonomy** a client receives.

Backend surface: ~2,524 LOC across 8 files (`api/wizard.py` 911 LOC, `modules/intake/{archetypes,firecrawl_client,plan_generator,profile_builder,progress,schemas}.py`, `api/onboarding_agents.py`, `core/models/business_profiles.py`, `core/seeds/seed_onboarding_agents.py`). Frontend: a `WizardShell` modal (`frontend/components/wizard/`) hosted at `/onboarding/wizard`.

---

## B. What it does (real implementation + data path)

**The 6 endpoints** (`api/wizard.py:68`, router `/api/wizard`, mounted `main.py:974`):

1. **`POST /start`** (`wizard.py:183`) — gated by `WIZARD_ENABLED` (default true, `config.py:666`); normalises the domain; optional email-domain match check (`_verify_domain_match:147`, off by default via `WIZARD_REQUIRE_DOMAIN_VERIFY=false`, `config.py:667`); inserts one `business_profiles` row scoped to `ctx.workspace_id` (`wizard.py:204-212`).
2. **`POST /scan/{id}`** (`wizard.py:227`) — calls Firecrawl `/map` (`firecrawl_client.py:79`) to enumerate URLs, then **pure URL-pattern archetype detection** (`archetypes.py:90`, zero LLM cost), buckets URLs into must/recommended (`select_target_urls:133`), persists `raw_map_urls` + `archetype`, returns the checklist.
3. **`POST /scrape/{id}`** (`wizard.py:299`) — returns **202 immediately** and fires a background pipeline via `launch_guarded` (`wizard.py:342`) so Railway's edge proxy can't kill the ~10–20 min job. Caps selected URLs at `FIRECRAWL_MAX_PAGES_PER_SCAN` (default 20).
4. **`GET /progress/{id}`** (`wizard.py:369`) — SSE feed. Backed by a **Redis LIST (replay) + pub/sub (live)** design (`modules/intake/progress.py`): late subscribers replay the capped 500-event buffer then subscribe live; 15 s keepalive comments; returns on terminal `complete`/`failed`. Falls back to a no-op emitter if `REDIS_URL` is unset (`progress.py:94-99`) — but then `stream()` yields a single `failed` event (`progress.py:216-225`), so with no Redis the UI shows only a failure.
5. **`PATCH /profile/{id}`** (`wizard.py:395`) + **`GET /profile/{id}`** (`wizard.py:418`) — user edits / reads the extracted profile (Step 6).
6. **`POST /plan/{id}`** (`wizard.py:443`) — the payoff. Renders `build_mission_goal()` (`plan_generator.py:42`) and calls `coordinator.create_mission()` with `config={"source":"mission_zero","auto_approve":True,"skip_verification":True,...}` (`wizard.py:487-495`), returning the `mission_id` so the UI redirects to the mission page.

**The background pipeline** `_run_scrape_pipeline` (`wizard.py:580`): scrape loop (per-URL, one bad URL never kills the run — `wizard.py:630-639`) → **RAG ingest** (each page written to a temp `.md`, uploaded via `DocumentManager.upload_document` tagged `["wizard","intake",domain,page_type]`, `wizard.py:820-910`) → **graphify** (`GraphifyService().build_graph(workspace_id)`, non-fatal on failure, `wizard.py:691-710`; bypassable via `WIZARD_SKIP_GRAPHIFY`, a TEMP dev flag that `wizard.py:596-599` itself says "Remove before E2E testing") → **profile build** (`profile_builder.py:34`, pure data-shaping from the LLM-extract dicts) → terminal `complete` event.

**Extraction schemas** (`schemas.py`): 8 page-type JSON schemas (about/contact/faq/policy/brands/delivery/solutions/generic) chosen by URL substring (`pick_schema_for_url:125`); Firecrawl runs LLM-extract against them. `profile_builder` reads only `about`/`solutions`/`brands` extracts plus the first 400 chars of substantive pages as "voice notes."

**Mission Zero goal** (`plan_generator.py:62-104`): a hand-templated multi-paragraph string that injects a **MANDATORY agent-role block** naming voyager/blueprint/scribe/forge, forbids assigning to `auto`, and instructs "do not invent facts — cite a real document chunk or graph node." This is the whole of the "intelligence" in the handoff: there is no LLM synthesis here (it was deliberately deleted — `plan_generator.py:11-13`), the coordinator owns all reasoning.

**Onboarding agents** (`seed_onboarding_agents.py`) — 4 global (`workspace_id=None`) system agents, `required_role="onboarding"` (no real user has this role → hidden from the roster), seeded idempotently at boot (`main.py:217-219`) with a lazy re-seed safety net in the plan endpoint (`_ensure_onboarding_agents`, `wizard.py:552`). All four run `anthropic/claude-sonnet-4` via OpenRouter with descending temperatures (VOYAGER 0.7 → FORGE 0.3), each carrying a long, genuinely detailed persona and the `platform-management` skill (VOYAGER also `web-research`). Managed through an admin-only Settings tab (`api/onboarding_agents.py`, `OnboardingAgentsTab.tsx`).

**Verified robustness detail:** Mission Zero really does auto-approve. `create_mission` threads `override_auto_approve=bool(mission_config.get("auto_approve"))` (`coordinator_service.py:2363`); with it true, `decision.auto_approve` is true and the mission goes straight to **running** (`coordinator_service.py:2366-2367`), not `awaiting_approval`. This matters — see §C.

---

## C. Honest quality — how good is it *really*?

### Real-data inspection (the North-Star test: has this ever helped a real client?)

**It has never run in production.** This is the single most important finding and it comes straight from the banked W1 real-data recon, not the code's self-description:

- **`business_profiles` is absent from the live census.** The recon agent sampled 41 tables of the 152-table production DB and deliberately included every surface with activity; `business_profiles` is not among them (`evidence/data/census.md`). The recon sampled `orchestration_runs`, `board_tasks`, `deliverables`, `heartbeat_results` (148k rows) — it did not consider the intake table worth a row-count, which is the recon's own signal that it holds no meaningful data. (I attempted a direct read-only count to confirm; the production-DB query was correctly blocked by policy, so I report this as **"no production trace in the banked inventory"** rather than a hard zero — an honest caveat, but the census omission is strong evidence.)
- **Zero Mission Zero runs exist.** All 17 `orchestration_runs` are blog-post/content missions in a **single** workspace (`ae8320bc`), the newest created 2026-06-13, none carrying a `mission_zero` source (`evidence/data/missions-orchestration.md`). If the wizard had ever fired `/plan` in production, a `source=mission_zero` run would be here. There are none.
- **The entry point is commented out.** `welcome-modal.tsx:116-140` — the "Start Business Intake" CTA that would route a new user to `/onboarding/wizard` is a commented-out block labelled "PRD-130 (hidden for pilot)." The only live path to the wizard is typing the URL by hand. The capability is **dark-launched at the UX layer**, which fully explains the empty data: no real user is ever offered it.

So the honest read is: this is *code that works in principle* and has been *proven by nobody*. Everything below is a static-analysis judgement of code that has not met a real site at scale.

### Concrete defects and limits (evidence-backed)

1. **Single archetype.** `ARCHETYPES` contains exactly one entry — `shopify_catalog` (`archetypes.py:78-80`). The docstring admits "Phase 1 ships ONE archetype." A non-Shopify site returns `archetype=None`, which collapses the must/recommended checklist to empty (`wizard.py:260-264`) and gives Mission Zero a bare `default_team=[]` (`wizard.py:466-467`). Intake generality — the whole reason this is a *platform* capability and not a Shopify feature — is **unbuilt and unproven**. The North Star asks "does this make Auto more autonomous *for clients*" (plural, general); today the answer is "only for Shopify catalog stores, and even those have never run it."
2. **Zero automated tests.** `find … tests` for wizard/intake/onboarding/mission_zero/prd130 returns **nothing**. A ~2,500-LOC surface with a background pipeline, an SSE protocol, a Redis fallback path, a boot reaper, and a coordinator handoff has **no unit, integration, or E2E coverage**. Against the platform's own 80% bar this is a hard miss, and it is why the "it works in principle" caveat has to stay — nothing demonstrates it works even once.
3. **Extraction quality is untested and thin.** `profile_builder` only mines three page types (`about`/`solutions`/`brands`) for structured fields; everything else contributes at most a 400-char raw-markdown "voice sample" (`profile_builder.py:88-93`). Whether Firecrawl's LLM-extract actually populates `company_name`/`industries_served`/`compliance_standards` on real sites is unknown — there are no golden fixtures for the intake extract (the only intake-adjacent fixtures in the repo are Shopify *opener* goldens, per `evidence/data/repo-eval-artifacts.md`). The quality-findings mechanism honestly records "No company_name extracted" (`profile_builder.py:95-100`) but nothing measures how often that fires.
4. **The Mission Zero handoff inherits the mission engine's real failure modes.** The good news (verified): Mission Zero **auto-approves to running** (§B), so it dodges the approval-gate trap that has 8/17 production missions frozen at `awaiting_approval` with 0 tokens, some since April (`evidence/data/missions-orchestration.md`). The bad news: it also sets `skip_verification:True` (`wizard.py:490`, "saves ~50% cost") — the first autonomous output a client sees is produced with the LLM-judge verification stage **off**. And the engine it hands to is the same one whose live history is dominated by stalls and, on the board side, "failed-marked-done" (`evidence/real-data-inventory.md` §3/§4). The first-impression mission rides infrastructure with known honesty problems.
5. **Two hard external dependencies with graceful-but-degraded fallbacks.** No `FIRECRAWL_API_KEY` → `/scan` and the pipeline 503 (`wizard.py:161-165`); the wizard is simply unavailable. No `REDIS_URL` → the SSE stream emits one `failed` event (`progress.py:216-225`), so even though the pipeline runs, the user watches it "fail." Both are handled without crashing, but both turn "cold-start a workspace" into a dead end in a deployment that hasn't provisioned them — and the open-core edition is exactly such a deployment.
6. **A dev shortcut is still in the hot path.** `WIZARD_SKIP_GRAPHIFY` (`config.py:670`, `wizard.py:600`) bypasses knowledge-graph construction; the code comment says "Remove before E2E testing." It is env-gated off by default, so it's latent, not live — but it is a "we never finished testing this" marker left in the pipeline.

### What is genuinely good (honest positives)

- **The async pattern is correct and robust.** 202 + background `launch_guarded` + Redis-replay SSE + a **boot reaper** (`core/boot/reaper.py:92-112`) that sweeps profiles stranded in `scraping`/`scanning` on restart and marks them `failed` with the wizard's own `quality_findings` convention. This is a properly engineered long-job pattern — better than a lot of the platform's other long-running paths, and it means a Railway restart mid-scrape won't leave a zombie.
- **Tenant scoping is present throughout.** Every endpoint derives the workspace from `ctx` and every `business_profiles` query filters on `workspace_id` (`_get_profile_or_404:127`), including the SSE endpoint before it streams (`wizard.py:380-381`). The background task snapshots primitives instead of carrying the request session (`wizard.py:324-329`). (Deeper isolation review is the Opus pass's job.)
- **Firecrawl usage is disciplined.** Domain-locked (`firecrawl_client.py:73-75,159-160` drop/refuse off-domain URLs even if the API returns them), hard client-side page cap, defensive parsing of both `[str]` and `[{url}]` map shapes (`firecrawl_client.py:120-127`). "Not a generic crawler, not exposed to agents, wizard-only" (`firecrawl_client.py:13-14`) is the right blast-radius call.
- **The persona seeds are high quality.** VOYAGER/BLUEPRINT/SCRIBE/FORGE prompts are specific, tool-aware (they name `platform_*` actions), cost-conscious ("a 5-person bakery doesn't need 14 agents," `seed_onboarding_agents.py:100`), and correctly hidden from the roster. This is the strongest single artifact in the module.

### Maturity: **2 / 5**

Justification: the code is clean, honestly written, and architecturally sound (that keeps it off a 1). But a capability is measured by whether it delivers the North Star, and this one **has never run for a real client, has no entry point, covers one vertical, and has zero tests**. On the "how good is it *really*, judged by real behaviour" axis, an unproven-in-production, single-vertical, untested, UX-hidden pipeline is a 2. It is a strong *prototype* of the right idea, not yet a *capability*.

---

## D. Competitive teardown

The intake problem — "turn a customer's website/business into a configured, knowledgeable agent workspace" — is solved along a spectrum from self-serve URL-sync to white-glove onboarding. Where Automatos sits: it is one of the *few* attempting a **self-serve, URL-to-fully-configured-agent-team** flow, but the pieces it leans on are commodity and it under-uses them.

**1. Firecrawl (the vendor it already depends on) — the "adopt more" competitor.**
Automatos uses only Firecrawl v1 `/map` + `/scrape` with a schema. Firecrawl v2.8 now ships an **`/agent` endpoint with parallel waterfall execution** (Spark-1 models) for "gather data wherever it lives," **change-tracking / website-monitoring** that reconciles crawls against the last snapshot to catch added/changed/removed pages, **`/search`** (search + extract in one call), **signed webhooks**, and **20+ self-host improvements including webhooks to private IPs**. Firecrawl scrape-with-JSON is **5 credits/page** (~$0.0006–$0.0032/page by plan), map up to 100k results. Automatos re-implements the *orchestration* (loop, cap, domain-lock, per-page progress) that Firecrawl's `/crawl` + webhooks would give it for free, and ignores change-tracking — the exact primitive that would keep the workspace corpus fresh after onboarding (today intake is one-shot; nothing re-syncs). ([firecrawl.dev](https://www.firecrawl.dev/), [changelog](https://www.firecrawl.dev/changelog), [monitoring docs](https://docs.firecrawl.dev/features/monitoring-website), [pricing 2026](https://puzzleinbox.com/blog/firecrawl-pricing-guide-2026))

**2. Intercom Fin — the "knowledge freshness" bar.**
Fin ingests Help Center articles, PDFs, public URLs, and syncs from Notion/Guru/Confluence/Salesforce into a RAG system with specialised retrieval/ranking/summarisation sub-models. Crucially: **internal content is ingested near-instantly and public-URL content re-syncs weekly** — i.e. onboarding is a *living* pipeline, not a one-time scrape. Automatos intake is **one-shot** and has no re-sync at all; a merchant who changes their site is silently stale. Fin also lets you test knowledge changes before going live; Automatos has no such loop. ([Fin explained](https://www.intercom.com/help/en/articles/7120684-fin-ai-agent-explained), [knowledge sources](https://www.intercom.com/help/en/articles/9440354-knowledge-sources-to-power-ai-agents-and-self-serve-support))

**3. Clay — the "enrichment depth" bar.**
Clay's **waterfall enrichment** queries 75+ providers sequentially (cheapest first, stop on hit) to assemble firmographics — revenue, headcount, tech stack, funding/news — hitting 80–95% coverage vs 50–60% single-source. Automatos derives its entire business profile from **the customer's own website markdown** — it never cross-references a single external data provider, so its "profile" is only as complete and as honest as the site's About page. VOYAGER (the research agent) *can* web-search at mission time, but the *structured* profile that seeds the workspace has zero external enrichment. ([waterfall](https://www.clay.com/waterfall-enrichment), [Clay docs](https://university.clay.com/docs/building-a-data-waterfall))

**4. Shopify Sidekick — the "zero-config context" bar.**
Sidekick has **no setup step at all**: it lives in the admin and already has full store context (catalog, orders, settings) natively, and Winter-'26 turned it proactive for store setup/launch. For the *Shopify* vertical Automatos targets first, Sidekick simply *starts* with everything the wizard spends 10–20 minutes scraping to approximate — and it gets ground-truth catalog data via the Admin API, not scraped HTML. Automatos's edge over Sidekick is cross-store/cross-vertical agent teams and governance, not the cold-start itself. ([Shopify Sidekick](https://www.shopify.com/sidekick), [Winter '26](https://www.pluginhive.com/launch-stores-with-shopify-ai-sidekick/))

**5. Sierra / Decagon — where Automatos is actually *ahead*.**
Both enterprise agent platforms use **sales-led, white-glove onboarding with no self-serve signup**; Decagon's standard onboarding is ~6 weeks and knowledge configuration happens during engineering-led setup ("you can't just drag and drop a PDF"); Sierra 4–10 weeks. Neither "gets your data AI-ready" for you. Automatos's **self-serve, minutes-long, URL-to-mission** flow is a genuinely different and (for SMB) *better* posture — this is the one axis where the intake concept beats best-in-class, **if** it were finished, tested, and turned on. ([Decagon vs Sierra](https://www.eesel.ai/blog/decagon-vs-sierra), [Decagon setup](https://decagon.ai/blog/ai-customer-support-setup))

**Where Automatos actually stands:** conceptually differentiated (self-serve cold-start that Sierra/Decagon don't offer), but mechanically under-built — it under-uses the vendor it already pays for (Firecrawl agent/crawl/change-tracking), does no external enrichment (Clay), has no freshness loop (Intercom), and for its own launch vertical is out-classed on raw context by the native incumbent (Sidekick).

---

## E. Build / extend / adopt / replace — the verdict

**EXTEND.** Keep the surface; the async architecture, tenant scoping, domain-lock, reaper, and persona seeds are worth keeping and are cheaply extended. Do **not** rebuild it and do **not** adopt a whole external onboarding vendor (Intercom/Sierra/Decagon are the wrong shape — they *are* the agent, they don't cold-start *your* agents). The reuse bias is satisfied by **adopting more of Firecrawl**, not by replacing anything.

Concretely:
- **Adopt more of Firecrawl (the vendor already in the stack).** Move the hand-rolled scrape loop to Firecrawl `/crawl` + **signed webhooks** (removes the custom per-URL loop and the SSE-progress plumbing risk), and add **change-tracking** to give the workspace a *freshness* loop instead of a one-shot scrape (closes the Intercom gap). Cost is unchanged-to-lower (5 credits/page either way; ~$0.0006–0.003/page). Integration shape: the pipeline already speaks Firecrawl — this is swapping two endpoints and adding a webhook consumer, not a new dependency. **This is the highest-leverage adopt.**
- **Keep building (in-house) exactly two things, because nothing external fits:** (a) the **archetype → target-page → schema → Mission-Zero-goal** translation layer — this is Automatos-specific product logic (it maps a business *into the platform's own agent/skill/tool vocabulary*), so it must be built here; today it's one Shopify archetype and needs 3–4 more (SaaS, services/agency, content/media, marketplace) with real detection signals; (b) the **Mission Zero goal contract** — the mandatory-role block and cite-or-die instruction are the right idea and can't be outsourced.
- **Optionally adopt an enrichment provider later** (Clay-style, or a single People-Data-Labs/Clearbit call) to give the profile external firmographics — but this is a *nice-to-have* enrichment, not a blocker, so defer it behind the freshness work.

Nothing here is a "replace." The kill-list for this module is empty of whole components — the only thing to *delete* is the `WIZARD_SKIP_GRAPHIFY` dev shortcut once real tests exist.

---

## G. Quality metric — how do we measure this and track it over time?

Today the number is effectively **0**: no runs, no tests, no eval fixtures, so there is nothing to report. To make intake quality a tracked number (feeds T3), measure it end-to-end, because intake only matters as a *first-run autonomy outcome*:

1. **Profile-extraction accuracy** — a golden set of N real sites (start with 10 Shopify + 10 non-Shopify) with hand-labelled `company_name`/`sectors`/`brands`/`standards`; score field-level precision/recall of `profile_builder` output against them. First metric to build; today **unmeasured**.
2. **Archetype-detection accuracy** — labelled site→archetype set; report top-1 accuracy + the `confidence` calibration. Currently untestable (one archetype).
3. **Mission-Zero success rate & quality** — of intake-launched missions: % that reach `completed` (not stalled/failed), and a human/LLM-judge score (1–5) of the resulting onboarding brief. This is *the* North-Star number for the module — "is the first autonomous impression good?" — and it is the one to put on the dashboard. Today **0 runs**, so unmeasurable until the entry point ships.
4. **Time-to-first-mission** and **intake completion funnel** (start→scan→scrape→profile→plan drop-off) as online signals once real users hit it.

All four are gold-set + online-signal metrics that slot directly into the T3 harness; none exist yet.

---

## H. Cost note (informational, not a gate)

Per full intake run (rough): **Firecrawl** — 1 map (1 credit) + up to 20 schema-scrapes (5 credits each) ≈ **~100 credits ≈ $0.06–$0.32** depending on plan. **LLM-extract** cost is inside those Firecrawl credits (Firecrawl runs the extractor). **Graphify** — one `GraphifyService.build_graph` over the ingested pages (LLM entity extraction; the module's dominant token cost, variable with page count — comparable to any workspace graph build). **Mission Zero** — the real spend: a full multi-agent mission on `claude-sonnet-4` across 4 agents; the two completed *content* missions in production burned **264k–430k tokens** each (`evidence/data/missions-orchestration.md`), and Mission Zero explicitly trades cost for `skip_verification` (~50% claimed saving). So the intake *pipeline* is cheap (cents); the *mission it launches* is by far the cost centre, and it runs with verification off. **RAG ingest** — embeddings for ~20 pages, negligible.

---

## I. UX / surface

- **Fix the entry point first — it's the whole problem.** Un-comment / rebuild the "Start Business Intake" CTA in `welcome-modal.tsx:116-140` (or add it to Command Center's empty-state). Right now the only way in is typing `/onboarding/wizard`; no wonder the data is empty. This is a one-block change that converts the module from "dead code" to "live capability" and is the single highest-impact UX act.
- **The wizard shell itself is solid** (`wizard-shell.tsx`): 6-step Tabs stepper, live SSE terminal in Step 5, editable profile in Step 6, correct redirect to `/missions/{id}` on launch. Keep it. It mirrors the create-agent modal design language, which is right.
- **Surface intake state in Command Center.** A workspace mid-onboarding (profile `scraping`/`profiled`) should show a card in the Command Center so the operator can resume — today a closed browser tab loses the thread until the reaper fires. Tie it to the existing `business_profiles.status`.
- **Show the Mission Zero *plan* before it runs, honestly.** The wizard redirects into the mission page where the plan lands via notification — but Mission Zero auto-approves, so the human never actually gates it. For the *first* mission a client ever sees, add a one-screen "here's what the team will do" review before dispatch (opt-in), so the first autonomy impression is legible rather than a fait-accompli.
- **Onboarding-agents admin tab** (`OnboardingAgentsTab.tsx`) is appropriately admin-gated and fine as-is; low priority.

---

## J. Upgrade path (prioritised by North-Star impact × effort)

1. **[High impact · Low effort] Ship the entry point + write the missing tests.** Un-hide the CTA (§I) and add the unit/integration/E2E coverage that a 2,500-LOC critical path must have (pipeline happy-path, one-bad-URL resilience, no-Redis fallback, reaper sweep, `/plan` → `create_mission` contract). Without these two, nothing else matters — the capability stays invisible and unproven. **This is the gate to maturity 3.**
2. **[High impact · Medium effort] Generalise beyond `shopify_catalog`.** Add 3–4 real archetypes (SaaS, services/agency, content/media, marketplace) with detection signals, target-page sets, extraction schemas, and default teams. This is what turns "a Shopify feature" into "the platform's cold-start capability" — the literal reason it's a platform-tier unit. Build the archetype-accuracy gold set (§G.2) alongside.
3. **[High impact · Medium effort] Add a freshness loop via Firecrawl change-tracking + webhooks.** Move the scrape loop onto Firecrawl `/crawl` + signed webhooks and add change-tracking so the workspace corpus re-syncs when the site changes (closes the Intercom gap; makes onboarding a living pipeline, not a one-shot). Highest-leverage *adopt* (§E).
4. **[Medium impact · Low effort] Make the first mission legible + reconsider `skip_verification`.** Add the pre-dispatch plan review (§I) and A/B whether keeping LLM-judge verification ON for Mission Zero materially lifts the first-impression quality score (§G.3) — the ~50% cost saving may be a false economy on the *one* mission that sets client trust.
5. **[Medium impact · Medium effort] External enrichment for the profile.** A single Clay-style/People-Data-Labs enrichment call to add firmographics the website can't give, so BLUEPRINT designs the roster from more than the customer's own marketing copy. Defer behind 1–3.
6. **[Low effort, hygiene] Remove `WIZARD_SKIP_GRAPHIFY`** once real E2E tests exist (its own comment demands it), and make the no-Redis path degrade to "pipeline ran, live updates unavailable" instead of a `failed` event.

---

### Evidence index (internal `file:line`, pinned `77bc9c6d5`)
`api/wizard.py:68,161,183,204,227,260,299,342,369,380,443,487,552,580,600,691,820` · `modules/intake/archetypes.py:78,90,133` · `modules/intake/plan_generator.py:11,42,82` · `modules/intake/profile_builder.py:34,88,95` · `modules/intake/firecrawl_client.py:13,73,79,120,159` · `modules/intake/schemas.py:113,125` · `modules/intake/progress.py:94,216` · `core/models/business_profiles.py:20,53` · `core/seeds/seed_onboarding_agents.py:25,100,239` · `api/onboarding_agents.py:28` · `core/boot/reaper.py:92-112` · `services/coordinator_service.py:2363-2367` · `main.py:217,262,974` · `config.py:662-670` · `frontend/app/onboarding/wizard/page.tsx` · `frontend/components/wizard/wizard-shell.tsx` · `frontend/components/onboarding/welcome-modal.tsx:116-140`
**Real-data:** `evidence/data/census.md` (business_profiles absent from 41-table sample) · `evidence/data/missions-orchestration.md` (0 mission_zero runs; 8/17 stalled at approval) · `evidence/real-data-inventory.md` §3/§4 · tests: `find` for wizard/intake/onboarding = ∅.
**External:** Firecrawl [site](https://www.firecrawl.dev/) · [changelog](https://www.firecrawl.dev/changelog) · [monitoring](https://docs.firecrawl.dev/features/monitoring-website) · [pricing](https://puzzleinbox.com/blog/firecrawl-pricing-guide-2026) — Intercom Fin [explained](https://www.intercom.com/help/en/articles/7120684-fin-ai-agent-explained) · [sources](https://www.intercom.com/help/en/articles/9440354-knowledge-sources-to-power-ai-agents-and-self-serve-support) — Clay [waterfall](https://www.clay.com/waterfall-enrichment) · [docs](https://university.clay.com/docs/building-a-data-waterfall) — Shopify [Sidekick](https://www.shopify.com/sidekick) · [Winter '26](https://www.pluginhive.com/launch-stores-with-shopify-ai-sidekick/) — [Decagon vs Sierra](https://www.eesel.ai/blog/decagon-vs-sierra) · [Decagon setup](https://decagon.ai/blog/ai-customer-support-setup)

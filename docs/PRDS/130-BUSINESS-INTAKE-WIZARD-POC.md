# PRD-130: Business Intake Wizard (PoC)

> **Status:** Draft (revised per review 2026-04-11)
> **Scope:** Proof-of-concept, single-tenant demo for INBUILD UK
> **Dependencies:** PRD-123 (Mission Zero OnboardingSection), PRD-126 (Business Knowledge Graph), Shopify Agents Spec
> **Phase 1 goal:** Demoable by tomorrow lunch. Not production-ready. Not general-purpose.
>
> **Phase 1 cut (locked):** Wizard → Firecrawl scan/map → user selects URLs → DocumentManager RAG → Graphify → **Mission Zero Draft Business Plan**. That's the demo. Mission 1 (team provisioning) is parked as TODO — we want client feedback on the draft plan first before we commit to building the team.

---

## 1. Problem

New users landing in Automatos face a cold-start problem. The existing Mission Zero
(`orchestrator/modules/context/sections/onboarding.py`) asks discovery questions in
chat — which works, but takes 10+ turns and relies on the user knowing what to ask for.

We want a **zero-conversation path** from "I just signed up" to "here is a draft business
plan tailored to my company, with a proposed agent team ready to approve." The wizard
does the discovery *for* the user by scraping their public website.

## 2. Non-goals (Phase 1)

- Not a general-purpose crawler tool available to agents (Phase 2)
- Not multi-tenant production-hardened (single demo tenant: INBUILD UK)
- Not self-hosted Firecrawl (cloud dev trial first, self-host only if PoC succeeds)
- Not a replacement for the existing `OnboardingSection` — it *feeds* Mission Zero
- Not billing-metered, rate-limited, or abuse-hardened
- Not handling non-Shopify sites beyond the detection stub (Phase 1 archetype = Shopify only)
- **Mission 1 team provisioning is NOT in Phase 1** — Draft plan only, approval kicks off Mission 1 in Phase 2
- **No heavy security / guardrails in Phase 1** — domain-lock on Firecrawl and a page cap, that's it. Rate limiting, audit logging, DNS verification, multi-tenant hardening all parked.

## 3. Success criteria (the demo bar)

Running the wizard against `inbuilduk.com` with the INBUILD team watching should produce:

1. Domain verified in < 5 seconds (email-match via Clerk) — **disabled in dev via `WIZARD_REQUIRE_DOMAIN_VERIFY=false`**
2. Map returned with ~800 URLs, Shopify archetype auto-detected
3. User-facing checklist of target pages with must-have/optional tiers
4. Core 14-page scrape completes in under 60 seconds
5. Scraped markdown ingested to RAG via existing `DocumentManager`
6. Graphify pass builds workspace graph
7. Draft Business Profile displayed, editable, with Companies House enrichment
8. **Mission Zero draft plan generated, citing graph nodes as evidence** — demo ends here
9. **The kicker**: open chat and ask Auto *"tell me about INBUILD UK"* → Auto answers from RAG + Graph with specifics (brands, sectors, standards, policies). **BOOM moment.**

**The "wow moment" target**: from "Start scan" click to "draft plan with evidence + chatable knowledge" in under 3 minutes.

## 4. Architecture at a glance

```
Frontend
  WelcomeModal (existing) ─── "Start Business Intake" button
  Dev trigger: `/onboarding/wizard?force=1` and a dev-only launcher in
               workspace settings (gated by NEXT_PUBLIC_DEV_TOOLS)
         │
         ▼
  /onboarding/wizard (new route) — styled to match Create Agent modal
    ├─ Step 1: Goals picker
    ├─ Step 2: Domain entry
    ├─ Step 3: Scanning... (polls /wizard/scan status)
    ├─ Step 4: Page checklist
    ├─ Step 5: Intake running... (RAG + Graphify progress)
    ├─ Step 6: Business Profile editor
    └─ Step 7: Draft Plan review (END of Phase 1 — no approve button yet)

Backend
  orchestrator/api/wizard.py                 ← NEW router, 5 endpoints (Phase 1)
  orchestrator/modules/intake/               ← NEW package
    - firecrawl_client.py (single-file client, FirecrawlClient class)
    - archetypes.py       (detection rules)
    - schemas.py          (page-type JSON schemas)
    - profile_builder.py  (assemble BusinessProfile)
    - plan_generator.py   (graph-cited Mission Zero draft plan)
  orchestrator/alembic/versions/prd130_business_profile.py   ← NEW migration

Integrations (reuse, no changes)
  DocumentManager          → RAG ingestion (existing)
  graphify                 → graph building + querying (existing)
  OnboardingSection        → already exists; wizard pre-populates its context
  [Phase 2] platform_create_mission  → Mission 1 team provisioning
  [Phase 2] Agent factory            → team provisioning via agents.team/job_title/reports_to_id
```

**Key decision per review correction:** the crawler lives in `orchestrator/modules/intake/firecrawl_client.py` — a single file inside the `intake/` package. Even though it's one file today, we create the package so archetypes/schemas/companies_house/profile_builder/plan_generator sit alongside it with a clear domain boundary. No `CrawlerClient` ABC. One concrete `FirecrawlClient` class. "intake" (not "crawler" or "firecrawl") so we don't lock to the vendor name or tempt future generic-crawler scope creep.

## 5. Data model

New table only. No existing tables touched.

```python
# orchestrator/alembic/versions/prd130_business_profile.py

business_profiles
  id                 PK
  workspace_id       FK workspaces
  domain             TEXT
  archetype          TEXT       # 'shopify' (phase 1), later: saas, agency, etc.
  company_name       TEXT       # extracted from scraped About page
  sectors            JSONB      # ['industrial','commercial','hrb','education']
  brands             JSONB      # list of supplier brands
  standards          JSONB      # ['EN 12101-8', 'BS 5839', ...]
  voice_notes        TEXT       # extracted tone/voice profile
  goals              JSONB      # user-selected from Step 1 (manage/grow/market/etc)
  raw_map_urls       JSONB      # full URL list from Firecrawl map
  selected_urls      JSONB      # URLs the user ticked for scraping
  quality_findings   JSONB      # duplicates, typos, test products
  draft_plan         JSONB      # proposed agents + rationale + graph citations
  status             TEXT       # scanning|scraping|profiling|planned|approved|provisioned
  created_at         TIMESTAMP
  updated_at         TIMESTAMP

  INDEX (workspace_id, status)
```

One row per workspace. Updated in-place as the wizard progresses.

## 6. API surface

All endpoints under `/api/wizard/`. Scoped to current workspace via existing auth.
Phase 1 has no background job queue — endpoints block until done. Good enough for < 60s scrapes.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/wizard/start` | Verify domain (email-match, skipped when `WIZARD_REQUIRE_DOMAIN_VERIFY=false`), create `business_profiles` row, return profile_id |
| POST | `/api/wizard/scan/{profile_id}` | Run Firecrawl map + archetype detection. Returns URL inventory + detected archetype |
| POST | `/api/wizard/scrape/{profile_id}` | Body: `{selected_urls: [...]}`. Runs parallel scrape, pushes to DocumentManager, triggers graphify, updates profile. Returns draft profile |
| PATCH | `/api/wizard/profile/{profile_id}` | User edits to the profile |
| POST | `/api/wizard/plan/{profile_id}` | Generate Mission Zero draft plan using graph queries + user goals. Returns proposed agents with citations. **End of Phase 1 flow.** |
| ~~POST~~ | ~~`/api/wizard/approve/{profile_id}`~~ | **Phase 2 / TODO** — Mission 1 team provisioning |

## 7. Firecrawl client

Single file: `orchestrator/modules/intake/firecrawl_client.py`

```python
# Sketch only
class FirecrawlClient:
    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.dev/v1"):
        self._api_key = api_key
        self._base_url = base_url

    async def map(self, domain: str, limit: int = 1000) -> list[str]: ...

    async def scrape(
        self,
        url: str,
        schema: dict | None = None,   # JSON schema for structured extract
        formats: list[str] = ("markdown",),
    ) -> dict: ...
```

Config additions in `config.py`:
```python
FIRECRAWL_API_KEY = env("FIRECRAWL_API_KEY", None)
FIRECRAWL_BASE_URL = env("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")
FIRECRAWL_MAX_PAGES_PER_SCAN = env_int("FIRECRAWL_MAX_PAGES_PER_SCAN", 20)
```

Hard caps enforced client-side regardless of what the API returns.

## 8. Archetype detection (Phase 1 = Shopify only)

`orchestrator/modules/intake/archetypes.py`

```python
ARCHETYPES = {
    "shopify_catalog": {
        "signals": {
            "required": ["/cdn/shop/", "/collections/", "/products/"],
            "boost": ["/pages/brands", "/blogs/news"],
        },
        "target_pages": {
            "must": [
                "/pages/about*", "/pages/contact*", "/pages/faq*",
                "/policies/privacy*", "/policies/refund*", "/policies/terms*",
                "/pages/delivery*", "/pages/returns*", "/pages/solutions*",
                "/pages/brands*",
            ],
            "recommended": [
                "/blogs/technical-bulletins/*",   # INBUILD-style compliance docs
            ],
            "optional_deep": [
                ("/blogs/*",       "voice_corpus",   "Marketer voice training"),
                ("/collections/*", "catalog_index",  "Sales agent catalog knowledge"),
            ],
        },
        "quality_checks": [
            "duplicate_collections",  # /foo + /foo-1
            "typo_slugs",             # Levenshtein < 2 between slugs in same namespace
            "orphan_copies",          # ends with -copy, -copy-1
            "test_products",          # slug = "test*"
        ],
        "default_team": [
            "shopify_ops",          # from SHOPIFY-AGENTS-SPEC.md
            "catalog_hygiene",      # new, justified by quality_checks
            "technical_sales",      # new, vertical variants if sectors detected
            "compliance",           # new, when standards detected
            "content_marketer",     # new, voice from blog
            "brand_relations",      # new, when brand count > 5
        ],
    },
}
```

Detection is pure URL-pattern matching — zero LLM calls, zero cost.

## 9. Page-type extraction schemas

`orchestrator/modules/intake/schemas.py`

One JSON schema per page type, passed to `FirecrawlClient.scrape(url, schema=...)`.
Firecrawl's LLM extract mode returns typed output.

```python
ABOUT_US_SCHEMA = {
    "type": "object",
    "properties": {
        "company_description": {"type": "string"},
        "founded_year": {"type": "string"},
        "mission_statement": {"type": "string"},
        "key_differentiators": {"type": "array", "items": {"type": "string"}},
    },
}

CONTACT_SCHEMA = { ... }  # phone, email, address, hours, channels
FAQ_SCHEMA = { ... }       # list of {question, answer}
POLICY_SCHEMA = { ... }    # policy_type, summary, key_clauses
BRANDS_SCHEMA = { ... }    # list of {brand_name, category, logo_url}
```

Phase 1 ships ~8 schemas. Tight enough to demo, loose enough to iterate.

## 10. Mission Zero draft plan generation

`orchestrator/modules/intake/plan_generator.py`

**The graphify-grounded generation loop:**

```
inputs:
  - business_profile (enriched by scrape)
  - user_goals (selected in Step 1)
  - graph (workspace graphify output)

steps:
  1. candidate_agents = archetype.default_team ∩ goals_to_agent_map[user_goals]
     ← INTERSECTION, not union (per pushback from earlier discussion)
     ← "respects the brief" rule

  2. For each candidate agent:
     a. Build a graph query specific to that agent's evidence need
        e.g. Compliance agent → "what standards are cited in this workspace?"
        e.g. Brand Relations  → "which brands appear > 5 times in the catalog?"
     b. Execute graphify query, collect matching nodes
     c. If evidence is empty AND agent is optional → drop it
     d. If evidence is strong AND agent is required → include with citations
     e. Build rationale string: "Proposed because [graph_node_1], [graph_node_2]..."

  3. Propose org structure using agents.team / reports_to_id
     (migration already exists: mission_zero_org_fields.py)

  4. For Shopify archetype: inject pre-specced agents from SHOPIFY-AGENTS-SPEC.md
     - Shopify Operations Manager with the 4 skills + Composio SHOPIFY tool
     - Persona text copied verbatim from the spec

  5. Return draft_plan JSONB:
     {
       "proposed_agents": [
         {"name": "...", "persona": "...", "skills": [...],
          "tools": [...], "rationale": "...", "citations": [graph_node_ids]},
         ...
       ],
       "org_chart": [{agent, reports_to}],
       "integrations_needed": ["shopify_oauth", ...],
       "open_questions": ["Is Besafe Ltd your parent company?"],
     }
```

**Citations are the trust layer.** Every proposed agent carries graph node IDs that the
frontend renders as expandable "why we suggested this" chips. When the user clicks, they
see the actual scraped content that drove the recommendation.

## 11. Mission 1: team provisioning — **PHASE 2 / TODO**

> **Deferred from Phase 1.** We want INBUILD to see the Mission Zero draft plan and give
> us feedback before we commit to provisioning. The draft plan IS the deliverable for
> PoC — "here's what Automatos thinks your team should look like, here's the evidence,
> what do you think?"

When we build Phase 2, shape is:

```python
# Pseudocode (Phase 2)
plan = profile.draft_plan
mission_payload = {
    "name": "Mission 1: Team Build",
    "steps": [
        {"tool": "platform_create_agent", "params": {...agent_1...}},
        {"tool": "platform_create_agent", "params": {...agent_2...}},
        ...
        {"tool": "platform_assign_skill", "params": {...}},
        {"tool": "platform_assign_tool",  "params": {...}},
    ],
    "coordinator": "sequential",  # PRD-82A
}
mission_id = await platform_create_mission(mission_payload)
profile.status = "provisioned"
return {"mission_id": mission_id}
```

Would use the existing PRD-82A Sequential Coordinator. Migration `mission_zero_org_fields.py`
already ships the org-chart columns on `agents` (team, job_title, reports_to_id) so there's
no schema work blocking Phase 2.

## 12. Frontend wizard

New route: `frontend/app/(authenticated)/onboarding/wizard/page.tsx`

**Style guide: follow the existing Create Agent modal pattern.** Same modal shell, same
stepper treatment, same Automatos dark theme + accent colors, same form field components,
same CTA button styling. Do not invent a new visual language — the wizard should look
like a first-class Automatos surface from day one.

Components:
- `frontend/components/wizard/wizard-shell.tsx` — step progression shell (mirror Create Agent modal)
- `frontend/components/wizard/step-1-goals.tsx`
- `frontend/components/wizard/step-2-domain.tsx`
- `frontend/components/wizard/step-3-scanning.tsx`
- `frontend/components/wizard/step-4-page-checklist.tsx`
- `frontend/components/wizard/step-5-intake.tsx`
- `frontend/components/wizard/step-6-profile-editor.tsx`
- `frontend/components/wizard/step-7-draft-plan.tsx` — **final step in Phase 1**
- ~~`step-8-approve.tsx`~~ — **Phase 2 / TODO**
- `frontend/hooks/use-wizard-api.ts` — React Query hooks for wizard endpoints

Triggering (Phase 1 — all wired):
- **First-login auto-prompt**: modify `frontend/components/onboarding/first-login-guard.tsx`
  to add "Start Business Intake" button to `WelcomeModal`
- **Dev manual trigger #1**: query param `/onboarding/wizard?force=1` opens the wizard
  regardless of `isNewWorkspace` state (for repeated demo runs)
- **Dev manual trigger #2**: dev-only "Run Intake Wizard" launcher in workspace settings,
  gated behind `NEXT_PUBLIC_DEV_TOOLS=true`. Hidden in prod.

## 13. Security & guardrails (deliberately thin for PoC)

Per review correction: we are toning this WAY down for Phase 1. Just enough to not embarrass ourselves in the demo.

**Phase 1 — kept:**
- **Domain-lock on Firecrawl**: requests restricted to the submitted domain (brand safety, not security — we don't want to accidentally scrape a competitor mid-demo)
- **Hard page cap**: 20 URLs scraped per profile (cost control on Firecrawl cloud trial)
- **Secrets via config.py only**: `FIRECRAWL_API_KEY`, `COMPANIES_HOUSE_API_KEY` never logged
- **Auth**: wizard endpoints sit behind the existing authenticated workspace middleware — same as everything else
- **Graceful errors**: Firecrawl failures return partial profile with `quality_findings: {errors: [...]}` rather than crashing the wizard

**Phase 1 — explicitly skipped (Phase 2 TODO):**
- ~~Clerk email-domain verification~~ — env flag `WIZARD_REQUIRE_DOMAIN_VERIFY=false` in dev. Don't delete the code, just skip.
- ~~Rate limiting (3 scans per 24h)~~ — no throttling
- ~~DNS TXT verification~~ — not in scope at all
- ~~Multi-tenant abuse protection~~ — single demo tenant
- ~~Audit logging~~ — standard request logs are enough
- ~~Wizard analytics / funnel telemetry~~

Keep the code paths in place where cheap (flags, feature toggles). Delete nothing.

## 14. Config additions (all via `config.py`)

```python
# Firecrawl
FIRECRAWL_API_KEY            = env("FIRECRAWL_API_KEY", None)
FIRECRAWL_BASE_URL           = env("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")
FIRECRAWL_MAX_PAGES_PER_SCAN = env_int("FIRECRAWL_MAX_PAGES_PER_SCAN", 20)

# Wizard
WIZARD_ENABLED                 = env_bool("WIZARD_ENABLED", True)
WIZARD_MAX_ACTIVE_PER_WS       = env_int("WIZARD_MAX_ACTIVE_PER_WS", 1)
WIZARD_REQUIRE_DOMAIN_VERIFY   = env_bool("WIZARD_REQUIRE_DOMAIN_VERIFY", False)  # dev: off
```

Frontend (`frontend/.env.local`):
```
NEXT_PUBLIC_DEV_TOOLS=true   # shows dev-only "Run Intake Wizard" launcher
```

## 15. Tasks (phase 1 build order)

Ordered for tomorrow-lunchtime shipping:

1. Firecrawl API key set in Railway env (`FIRECRAWL_API_KEY`)
2. `config.py` additions (Firecrawl + Wizard flags incl. `WIZARD_REQUIRE_DOMAIN_VERIFY=false`)
3. `orchestrator/modules/intake/firecrawl_client.py` (single file, ~100 lines)
4. `orchestrator/modules/intake/` package siblings: archetypes.py, schemas.py, profile_builder.py
5. Alembic migration `prd130_business_profile.py`
6. `orchestrator/api/wizard.py` (5 endpoints, thin handlers, no approve)
7. Mount router in `orchestrator/main.py`
8. Happy-path curl test against inbuilduk.com (backend standalone)
9. `intake/plan_generator.py` with graphify queries for citations
10. Frontend wizard shell + 7 step components — **matched to Create Agent modal style**
11. Wire `use-wizard-api` hooks
12. Hook "Start Business Intake" into `WelcomeModal` + dev-only launcher in workspace settings + `?force=1` query param
13. End-to-end demo dry run against inbuilduk.com
14. Chat smoke test: "tell me about INBUILD UK" → verify RAG+Graph answer quality
15. Record 90s demo video

## 16. Out-of-scope items explicitly parked for Phase 2 / TODO

- **Mission 1 team provisioning** — the approve → `platform_create_mission` flow. We demo the draft plan in Phase 1, get feedback, then build this.
- **`/api/wizard/approve` endpoint** and Step 8 frontend
- **Clerk email-domain verification** (code gated behind `WIZARD_REQUIRE_DOMAIN_VERIFY`, off in dev)
- `platform_intake_*` tools for general agent use (wizard stays wizard-only in Phase 1)
- Self-host Firecrawl container on Railway
- Non-Shopify archetypes (SaaS, agency, restaurant, etc.)
- DNS TXT domain verification
- Optional deep crawls (full blog / full catalog)
- Wizard re-run with diff ("rescan and tell me what changed")
- Multi-tenant abuse protection
- Rate limiting
- Wizard analytics / funnel telemetry
- Audit logging

## 17. Demo script (the tomorrow-lunch version)

1. Open fresh workspace as a new user (or hit dev launcher / `?force=1`)
2. Welcome modal shows "Start Business Intake"
3. Pick goals: Manage, Grow, Market, Ensure Compliance
4. Enter `inbuilduk.com`
5. **Scan** — watch Shopify archetype detected, URL inventory returned (~800 URLs for inbuilduk.com)
6. **Checklist** — accept defaults, show the "data quality findings" callout (duplicates, test products)
7. **Intake** — watch RAG + Graphify progress bars
8. **Profile** — show Besafe Ltd parent-entity prompt, user confirms, edits tone notes
9. **Draft Plan** — show proposed team: Shopify Ops Manager + Compliance + Technical Sales + Content Marketer, each with graph-node citations
10. Click a citation chip — shows the exact scraped sentence that drove the recommendation
11. **The BOOM** — switch to chat, ask Auto *"tell me about INBUILD UK"* → Auto answers with real specifics from RAG + Graph (brands, certifications, standards, product categories). Follow up: *"what fire safety standards do they comply with?"* → cites scraped bulletins.

**End state:** INBUILD sees a draft business plan grounded in evidence from their own website, plus an AI that can already answer questions about their business. Phase 2 is the easy "yes" — *"approve this plan and we'll build the team."*

First-time experience bar is now: **"AI just read my business, built a knowledge graph, and drafted my org chart — in 3 minutes, before I typed a single message."**

---

**End of PRD-130**

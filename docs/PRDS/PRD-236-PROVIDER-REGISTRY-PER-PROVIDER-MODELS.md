# PRD-236: Provider registry + per-provider models — "Kimi via NVIDIA is free, Kimi via OpenRouter is paid, and the system routes to the one you picked"

> **Status:** W0 BUILT (PR #699, CI green, owner's local test: NVIDIA key added and validated 2026-09-03) · **W1 BUILT** (branch `feat/prd-236-w1`, stacked on W0; owner's go-ahead 2026-09-03: "separate tabs in marketplace for OpenRouter and NVIDIA… so Kimi K3 on OpenRouter is different from NVIDIA Kimi K3"). **Depends on:** PRD-54 (LLM marketplace, live), PRD-222 (BYOK validate-on-save, live), PRD-223 W0/W1 (model policy + `workspace_models` approval columns, live), PRD-233 (local edition, live). **Both editions.** **Build:** W0 first (this branch), W1/W2 after the owner's W0 test in both editions (feedback 2026-08-30: a phase is done when it is CI-green, merged and tested by him in both editions).

---

## Framing (CLAUDE.md §3)

**Consolidation + extension.** Consolidation: ten hand-maintained provider lists collapse into one registry, and the two-table model catalogue (`llm_models` + `openrouter_models_cache`) becomes one catalogue keyed by *who serves the model*. Extension: NVIDIA (build.nvidia.com) and DeepSeek-direct become routes through a generic OpenAI-compatible adapter; the marketplace shows one card per route with that route's price; an install is tagged with its provider and the runtime routes to the tag. **Build size:** M (W0) + M (W1) + S (W2). **Risk:** Medium — W1 changes a UNIQUE constraint on a live table and the factory's routing; W0 touches the factory's provider resolution, the most incident-prone file in the repo (2026-07-29, 07-31, 08-29).

## Overview

Today the platform pays OpenRouter for every open model it runs, and a workspace cannot say "run this model on that API". After this PRD:

- **Settings → API Keys** lists every registered provider, NVIDIA and DeepSeek included, and validates a key live on save (the PRD-222 badge rule: the badge never lies).
- **Marketplace → LLMs** shows Kimi K3 twice — "NVIDIA · free (trial)" and "OpenRouter · $3 / $15 per M" — and installing one of them tags the workspace's install with that provider.
- **Agents and Auto** pick from the installed routes; the stored provider is authoritative and the factory routes to it. `llm_usage` records the provider that actually served the call and prices it from that route.
- **Local edition:** a user with only an NVIDIA key runs every NVIDIA-hosted model for nothing, under NVIDIA's own terms. **SaaS:** NVIDIA is BYO-key only; the platform never holds an NVIDIA key (§Terms).

Owner's mental model, verbatim (2026-09-03): *"if I select KIMI3 from OpenRouter it's tagged in my selection and I pay as I am using OpenRouter... if I select KIMI3 in NVIDIA I use their API and it's tagged in my workspace... the system will route to the tagged API."* That sentence is the acceptance test.

## Terms — NVIDIA's hosted endpoint is a trial (verified 2026-09-03)

- Endpoint `https://integrate.api.nvidia.com/v1`, OpenAI-compatible chat/embeddings/models. `GET /v1/models` is public (no key): 81 ids, vendor-prefixed exactly like OpenRouter (`moonshotai/kimi-k3`, `deepseek-ai/deepseek-v4-pro-0813`, `openai/gpt-oss-20b`; embed models `nvidia/nemotron-3-embed-1b`). No pricing, context or capability fields.
- NVIDIA API Trial Terms of Service §1.2: *"access to the API Service for limited trial purposes only and without use of the API Service or Generated Content in production"*; §1.4: *"you may only use the API Service for internal testing and evaluation purposes, not in production"* unless the user holds a Subscription; §4.3: no personal, financial or health data. Rate limit 40 requests/minute per key (developer forum; no official page).
- **Invariant:** Automatos never ships, stores or resolves a platform-level NVIDIA key in the SaaS edition (`byok_only` in the registry; the platform-key tiers skip it when `AUTH_EDITION=saas`). The user's own key is the user's own agreement with NVIDIA. The provider card and the install badge carry the trial note and the rate limit so the choice is informed. OpenRouter also carries 18 `:free` variants today (nemotron-3-ultra, glm-5.2, …; no Kimi K3) — the marketplace shows them as what they are.

This section describes the design, not legal advice.

---

## Current reality (grounded on `origin/main` 8ad8bc2fd)

- **One catalogue row per model id.** `llm_models.model_id` is UNIQUE (`alembic/versions/20251006_add_multi_model_support.py:64`, `core/models/core.py:55`). "Kimi via NVIDIA" and "Kimi via OpenRouter" cannot both exist.
- **`llm_models.provider` means vendor, not serving provider.** Installing from the OpenRouter cache copies the first path segment (`moonshotai`) and hardcodes `tier='aggregator'` (`api/llm_marketplace.py:110-145`). PRD-223 §8 Q1 (owner: **yes**) decided to rename that `tier` column to `sourcing`; the column is still `tier` (`direct|aggregator|byok_only`).
- **The serving provider is inferred at runtime** from string shape and key availability: slash-format id ⇒ OpenRouter, unknown provider ⇒ OpenRouter, direct key missing ⇒ OpenRouter (`modules/agents/factory/agent_factory.py:358-431, 575-590`, second copy in `activate_agent` `:910-940`). Nothing is stored.
- **One price per model row** (`core/llm/usage_tracker.py:49-53`); `llm_usage.provider` is the routing enum value (`core/llm/manager.py:669-720`). A free NVIDIA call would be booked at OpenRouter's price.
- **Ten provider lists**, hand-maintained: `LLMProvider` enum (`core/llm/clients/base.py:40`), BYOK `SUPPORTED_PROVIDERS` (`api/user_api_keys.py:28`), `_ALLOWED_PROVIDERS` (`api/workspaces.py:389`, re-imported by `modules/tools/discovery/handlers_workspace.py:388`), `ALL_PROVIDERS` (`api/llm_marketplace.py:156`), factory `DIRECT_PROVIDERS` / `config_map` / `provider_map` (`agent_factory.py:363, 503, 555`), manager env map + credential switch (`manager.py:504-556`), `_validate_provider_key` (4 providers live-tested, `user_api_keys.py:110-150`), and four frontend arrays (`ApiKeysSettingsTab.tsx:65`, `PlatformApiKeysCard.tsx:38`, `LLMTierCard.tsx:52`, `SystemLLMSettingsTab.tsx:437`, `power-up-card.tsx:35`).
- **OpenAI and OpenRouter clients are the same SDK plus a base URL** (`clients/openai_client.py:41-47`, `clients/openrouter_client.py:44-53`). No generic adapter exists.
- **DeepSeek BYOK is a dead path:** the key is accepted (`SUPPORTED_PROVIDERS`) but no client exists; `_resolve_provider_for_model('deepseek', …)` rewrites to OpenRouter and resolves the OpenRouter key. The user's DeepSeek key is never used.
- **No silent fallbacks in the manager** (`manager.py:626-650`): a 429 surfaces as an error today. `fallback_model_id` is used only on mission verification failure (`agent_factory.py:714`).
- **Marketplace LLM tab** reads the OpenRouter cache first and the legacy table as fallback, not a merge (`frontend/components/marketplace/marketplace-llms-tab.tsx:196-241`); its "provider" chips are vendors (`api/openrouter_marketplace.py:205-210`). `_get_available_providers` (`llm_marketplace.py:147`) is defined and unused.
- **Workspace seeding** treats every slash-format row as OpenRouter-served (`services/workspace_model_seeding.py:37-46`) — W1 must key this on the serving provider.

---

## Design (one page)

**1. Provider registry** — `core/llm/providers.py`, code, one `ProviderSpec` per provider: `slug`, `label`, `kind` (`direct` | `aggregator` | `hosted_open`), `adapter` (`openai_compatible` | `anthropic` | `google` | `azure` | `bedrock` | `huggingface` | `none`), `base_url`, `env_key` (the `config.py` attribute name — no `os.getenv` outside config), `hosts_vendor_models` (accepts `vendor/model` ids: openrouter, nvidia), `byok_only` (never a platform key in saas: nvidia), `price_multiplier` (W0 interim: nvidia 0.0), `validation` (`models_list` | `none`), `key_placeholder`, `docs_url`, `terms_note`, `rate_limit_note`, `openrouter_prefix` (vendor → OpenRouter form: deepseek → `deepseek/`). Every list in "Current reality" reads from it. Providers stay code because each needs an adapter and a validator; per-workspace keys and installs stay data.

**2. Generic OpenAI-compatible adapter** — `clients/openai_compatible_client.py`: today's `OpenRouterProvider` generalised (base URL, default headers, timeout, tool-call sanitising, the two tool-choice retries, image parts). `OpenRouterProvider` becomes the spec-driven instance for `openrouter`; `nvidia` and `deepseek` are instances with their own base URLs. A 429 from a spec with a `rate_limit_note` raises a `ValueError` that carries the note; no retry, no reroute (the manager already has no silent fallbacks).

**3. Catalogue keyed by route (W1)** — `llm_models` gains `serving_provider` (slug, NOT NULL) and `external_id` (the provider's native id); the UNIQUE moves from `model_id` to `(serving_provider, model_id)`; `tier` is renamed `sourcing` and derived from the registry `kind` (PRD-223 Q1 executed); price columns are per row. Data migration: `tier='direct'` rows ⇒ `serving_provider = provider`; everything else ⇒ `openrouter`. `workspace_models` is untouched — it already keys on the integer row, so an install IS the tag. Per-provider catalogue sync writes into `llm_models` (OpenRouter via the existing cache sync; NVIDIA from the public list, price 0, context/tool/vision metadata borrowed from the same vendor id's OpenRouter row when present). Reads that use `model_id` alone (`_get_or_create_from_cache`, `check_model_for_agent`, `usage_tracker`, seeding) take the pair.

**4. Routing** — the stored `model_config.provider` is authoritative when it names a registered provider; string-shape inference remains only for legacy rows (`provider` absent or not registered). Rule kept from 2026-08-29: **tagged provider has no key ⇒ route via another provider that has a key and can serve the id (OpenRouter), log `[KeyRouting]`, and say so in the reply.** New rule: **tagged provider rate-limits ⇒ fail closed with the provider's note**; never move a free call onto a paid route without the user. `llm_usage.provider` = the provider that served the call; cost = that route's price (W0: registry multiplier; W1: the row).

**5. UI** — Settings → API Keys, the platform-keys card, the tier card, the orchestrator provider select and the onboarding power-up card all render the registry (`GET /api/keys/providers`) with today's list as the static fallback. Marketplace → LLMs (W1): one card per route; facets **Provider** (serving) and **Vendor**; badges "free · NVIDIA trial · 40 req/min" and "add a key to use this" (the unused `_get_available_providers` finally has a job). Pickers label routes "Kimi K3 · NVIDIA".

**6. What is NOT built** (owner decisions, not deferrals): no platform NVIDIA key in saas; no NIM self-hosting; no automatic free→paid reroute on rate limit; no provider-specific UI tabs (one LLMs tab with facets).

---

## Stories

### W0 — NVIDIA works, the lists are one registry (this branch: `feat/prd-236-w0-provider-registry`)

| Story | What ships | Acceptance |
|---|---|---|
| **S0.1 Registry** | `core/llm/providers.py` with specs for openai, anthropic, google, azure, bedrock, huggingface, grok, openrouter, **nvidia**, **deepseek**, cohere (key-only). `LLMProvider` gains `NVIDIA`, `DEEPSEEK`. | Every enum member has a spec; every spec with an adapter constructs a client; a unit test pins both directions. |
| **S0.2 Generic adapter** | `OpenAICompatibleProvider`; OpenRouter/NVIDIA/DeepSeek built from specs; 429 → `ValueError` with the spec's note. | Client for `nvidia` is constructed with `base_url=https://integrate.api.nvidia.com/v1` and a Bearer key; OpenRouter keeps its Referer/Title headers; a fake 429 raises the note and nothing else is called. |
| **S0.3 Lists read the registry** | BYOK list, workspace allowlist, marketplace `ALL_PROVIDERS`, factory maps, manager env/credential switch, `_validate_provider_key` generic branch, `config.py` `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` / `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`. | `POST /api/keys {provider: nvidia}` is accepted and live-validated with a models-list call against the NVIDIA base URL; the same for deepseek. Existing providers behave byte-for-byte as before (tests on the openrouter/openai branches unchanged). |
| **S0.4 Routing** | `_resolve_provider_for_model` keeps a `hosts_vendor_models` provider for slash ids; `_openrouter_model_id` uses the spec's `openrouter_prefix`; the SaaS guard (`byok_only` providers never resolve from tiers 1.5/2/3 when `AUTH_EDITION=saas`). | `('nvidia','moonshotai/kimi-k3')` ⇒ `('nvidia', …)`; `('deepseek','deepseek-chat')` with a deepseek key ⇒ direct, without one ⇒ OpenRouter `deepseek/deepseek-chat` and a `[KeyRouting]` log line; in saas a platform nvidia credential is ignored. |
| **S0.5 Honest cost** | `UsageTracker` multiplies the row price by the registry `price_multiplier` (nvidia 0.0). | A tracked NVIDIA call books `total_cost=0` and `provider='nvidia'`. |
| **S0.6 Providers endpoint + UI** | `GET /api/keys/providers` (registry, no secrets) + `useProviderRegistry()`; the five frontend arrays render it (static fallback = today's list); route manifest +1. | NVIDIA appears in Settings → API Keys and the platform-keys card (hidden in saas because `byok_only`), in the tier card and the orchestrator select; the power-up card's "other providers" list includes it. Route-contract green. |
| **S0.7 Docs** | QUICKSTART: "Add your NVIDIA key" (local edition), the trial note. This PRD. | — |

**W0 test script (owner, local edition):** Settings → API Keys → add NVIDIA key → badge green (live models-list). Settings → Orchestrator → provider NVIDIA, model `moonshotai/kimi-k3` (or any installed slash id) → chat → reply arrives; `llm_usage` row has `provider='nvidia'`, `total_cost=0`; API log shows `provider=nvidia` and no `[KeyRouting]` line. Delete the key → next call routes via OpenRouter with the `[KeyRouting]` line and the reply says so. Add a DeepSeek key → provider DeepSeek, model `deepseek-chat` → direct call, `provider='deepseek'`. **SaaS:** the platform-keys card shows no NVIDIA slot; BYOK NVIDIA works for a workspace exactly as locally.

**Known W0 limitation (fixed by W1):** the marketplace still shows one Kimi K3 card with OpenRouter's price; choosing NVIDIA is done through the provider field (Orchestrator settings / agent config), not through the card. The cost booked is already honest (S0.5).

### W1 — the catalogue knows who serves what (BUILT 2026-09-03)

**As built:** one migration (`prd236_w1_serving_provider`: `serving_provider`, UNIQUE `(serving_provider, model_id)`, `tier`→`sourcing`, data rule direct-keeps-provider / everything-else-OpenRouter); `core/services/provider_catalog_sync.py` (OpenRouter cache → `llm_models` projection; NVIDIA public list → rows at price 0 with metadata borrowed from the same vendor id's OpenRouter row, alias-aware, non-chat ids skipped; job history on `openrouter_sync_jobs.job_type`); `GET /api/marketplace/llm/catalog` (facets: serving provider + vendor + price tier), install/uninstall/detail take `?provider=`, `installed-ids` returns `routes`; `POST /sync/{provider}` + `GET /sync/status`; governance (`check_model_for_agent`), pricing (`price_per_1k`/`ModelRegistry.get_model`) and `UsageTracker` judge/price the tagged route; every write path stores the ROUTE's serving provider (never the vendor); seeding keys on `serving_provider`. UI: provider **tabs** inside Marketplace → LLMs (All · OpenRouter · NVIDIA · OpenAI …) with vendor chips beneath, one card per route with its own price / Free / "add a key" badge, per-provider Sync buttons (admin), route-aware install; Orchestrator settings list the selected provider's installed routes and the auto-switch is gone.

| Story | What ships | Acceptance |
|---|---|---|
| **S1.1 Schema** | Alembic: `serving_provider`, `external_id`, UNIQUE `(serving_provider, model_id)`, `tier`→`sourcing` (PRD-223 Q1), data migration, PRD-209 head guard re-pinned. | `alembic-from-zero` green; existing installs keep working (their rows map to the provider that served them). |
| **S1.2 Catalogue sync** | `ProviderCatalogSync` with per-provider adapters (openrouter from the cache, nvidia from the public list); NVIDIA rows price 0, metadata borrowed by vendor id; admin sync endpoint generalised from `/api/openrouter/sync`. | After a sync, `moonshotai/kimi-k3` exists twice with different `serving_provider` and prices. |
| **S1.3 Marketplace API** | `/api/marketplace/llm/models` browses `llm_models` across providers with `provider` (serving) and `vendor` facets; install/uninstall by row id; `is_available` from `_get_available_providers`. | One card per route; installing the NVIDIA row creates a `workspace_models` row pointing at it. |
| **S1.4 Marketplace UI** | LLMs tab: single source, Provider + Vendor facets, price/free badges, "add a key" badge, compare works across routes. | The owner's sentence: pick Kimi from NVIDIA → tagged; pick Kimi from OpenRouter → tagged. |
| **S1.5 Pickers** | Installed-model pickers (agent config, orchestrator settings, `use-model-api`) label routes "Model · Provider" and write `provider` + `model_id` from the row. | Selecting a route stores its provider; the tier-based auto-switch in `SystemLLMSettingsTab` is deleted. |
| **S1.6 Pricing from the row** | `UsageTracker` prices by `(serving_provider, model_id)`; the W0 multiplier is deleted. Seeding keys on `serving_provider='openrouter'`. | Same call, same route, same price in analytics. |

### W2 — routing trusts the tag, analytics, the rest of the registry

| Story | What ships | Acceptance |
|---|---|---|
| **S2.1 Trust the tag** | Factory routes on stored `provider` when registered; inference only for legacy rows; both copies (`_create_llm_manager`, `activate_agent`) share one helper. | Kimi tagged NVIDIA never leaves NVIDIA while its key is valid. |
| **S2.2 Rate-limit honesty** | 429 surfaces the provider note in chat and the board; no reroute. Heartbeat/playbook lanes log the skip. | A 41st call in a minute returns the note, not a paid call. |
| **S2.3 Analytics by provider** | `analytics-llm-usage` groups by serving provider and vendor; fleet cost view (PRD-228) shows free vs paid. | — |
| **S2.4 Embeddings via NVIDIA** | `EmbeddingProvider.NVIDIA` on the generic embedding client (`nvidia/nemotron-3-embed-1b`), local edition. | RAG works with only an NVIDIA key locally. |
| **S2.5 Delete the loser** | `openrouter_models_cache` becomes the OpenRouter sync's staging table only, or is deleted (owner's call, Q3). `_get_or_create_from_cache` goes. | No two-source reads remain. |

---

## Open questions for the owner

| # | Question | Proposal |
|---|---|---|
| Q1 | Tagged provider has **no key**: reroute via a provider that has one (2026-08-29 rule) or fail closed? | Keep the reroute, log it, say it in the reply (a dead route is never what the user meant). |
| Q2 | Tagged provider **rate-limits** (NVIDIA 40/min): fail closed or reroute to a paid route? | Fail closed with the note. Free must never silently become paid. |
| Q3 | `openrouter_models_cache`: keep as staging or delete once `llm_models` holds every route? | Delete in W2 (CLAUDE.md §5). |
| Q4 | Auto on NVIDIA in an active workspace shares 40 req/min with heartbeats and playbooks. Allow, warn, or block for the orchestrator seat? | Allow locally with a warning badge; PRD-223 approval still applies to Kimi for the orchestrator role. |

## Traceability

NVIDIA API Trial Terms of Service (assets.ngc.nvidia.com, PDF) §1.2, §1.4, §4.3 · NVIDIA LLM API reference (docs.api.nvidia.com/nim/reference/llm-apis) · `GET https://integrate.api.nvidia.com/v1/models` (public, 2026-09-03) · OpenRouter `GET /api/v1/models` (424 models, 18 `:free`, 2026-09-03) · PRD-54, PRD-222 US-006, PRD-223 §8 Q1, PRD-233 · agent_factory 2026-08-29 key-availability routing (#645).

# PRD-282 US-101: A table of every outbound caller

**Status:** Analysis for PRD-282 Wave 1. No code changed.
**Read at:** `automatos-ai` commit `2d90dab0a` (the branch that holds the open PRD-277 and PRD-278 work).

## 1. What was swept

Everything under `orchestrator/` and `services/`, tests excluded, was searched for places where the product opens a network connection to an address that is not a fixed constant of the product. The search covered HTTP clients (`httpx`, `requests`, `aiohttp`, `urllib`), sockets and websockets, SSH and other transfer libraries, object-storage clients with an endpoint, database and Redis connections, subprocess calls to fetch tools and `git`, document and image renderers, discovery and key fetches for sign-in, webhook senders, tool and model endpoints, and speech and embedding services. Each place that turned up was opened, and the code was followed back to whoever chooses the address and to whether the outbound guard (`core/security/web_access.py`) is called on that path. The callers this story asks to raise with a maintainer were handed over privately.

## 2. Callers that go through the guard today

"Guard" is `resolve_outbound` or `resolve_outbound_async` in `orchestrator/core/security/web_access.py` (`:91`, `:143`). The five purposes are those of PRD-282 section 5.1: `partner`, `identity`, `model`, `callback`, `fetch`.

| # | Call (file:line) | Guard call (file:line) | What reaches it | Who chooses the address | Purpose | Redirects on this path |
|---|---|---|---|---|---|---|
| 1 | `modules/tools/partner/client.py:101` | `modules/tools/partner/client.py:111` | An agent's partner tool call (`modules/tools/partner/runner.py:152`) | A workspace manager: the base URL of the pack's `generic_api` credential (`api/credentials.py:209`, `:520`; install route `api/partner_packs.py:71`, all `workspace:manage`) | `partner` | Any 3xx is refused (`client.py:78-79`) |
| 2 | `modules/tools/discovery/handlers_web.py:146` | `handlers_web.py:175` (first address), `:155` (each hop), through `_resolve` `:113` | The `platform_web_fetch` action (`actions_web.py:20`, `handlers_web.py:256`) | An agent (the `url` parameter, `:260`) | `fetch` | Followed by hand, each hop checked before it is sent, at most 5 (`:44`, `:144-157`) |
| 3 | `handlers_web.py:146`, through `_stylesheet` `:217-220` | `handlers_web.py:175` | Stylesheets linked from a page fetched with `brand` (`_site_brand`, `:227`) | The content of the page that was fetched | `fetch` | As row 2 |
| 4 | `services/heartbeat_service.py:1414-1415` | `heartbeat_service.py:1397`; also at save time `api/heartbeat.py:157` | An agent's heartbeat report to a webhook | A workspace editor, in the agent's heartbeat form (`api/heartbeat.py:140`, `agents:update`) | `callback` | Not followed; the status code is logged (`heartbeat_service.py:1434`) |
| 5 | `services/run_callbacks.py:335-337` | `run_callbacks.py:329`; also when the run is requested, `:154`, from `api/agent_execute.py:157` | The `run_callback.deliver` job after a run started with `POST /api/agents/{id}/execute` | The caller of that route, per request (`agents:execute`, member or server key, `agent_execute.py:86`) | Not obvious (section 4) | Not followed (`run_callbacks.py:348`); a 3xx is an unsuccessful delivery |
| 6 | `modules/documents/pinned_fetch.py:124` | `pinned_fetch.py:66` | Resources a rendered document links to, through the WeasyPrint fetcher (`modules/documents/url_fetcher.py:39`, `generation_service.py:378`) | Whoever wrote the document or template HTML | `fetch` | Each hop checked again, at most 3 (`pinned_fetch.py:32`, `:109-115`); the address must also be globally routable (`:69`) |
| 7 | `pinned_fetch.py:124` | `pinned_fetch.py:66` | An image in a Word document (`modules/documents/blocks/docx_renderer.py:127`) | Whoever wrote the document | `fetch` | As row 6 |
| 8 | `modules/rag/services/cloud_file_downloader.py:450` | `cloud_file_downloader.py:442` | Cloud document sync (`:365`, `:539`) | A link in the provider's answer, or in a synced file's own content | `fetch` (as PRD-282 5.1 assigns cloud files) | By hand, each hop checked, at most 3 (`:71`, `:441-456`) |
| 9 | `modules/socials/recipes/files.py:141` | `files.py:135` | A file link returned by a socials tool | The tool's answer | `fetch` (as PRD-282 5.1 assigns socials) | By hand, each hop checked, at most 3 (`:33`, `:134-152`) |
| 10 | `core/composio/linkedin_image_workaround.py:353` | `linkedin_image_workaround.py:349` | The LinkedIn image post action, for an image given as a URL (`:373-374`) | An agent (the image string) | `fetch` | Not followed; a non-200 answer is refused (`:354-356`) |
| 11 | No connection; a check at save time, `core/llm/byok_endpoint.py:91` | `byok_endpoint.py:91` | Saving a workspace key's own endpoint: `POST /api/keys` (`api/user_api_keys.py:277`, `workspace:manage`; check at `:136`) | A workspace manager | `model` | Not applicable |
| 12 | `core/llm/clients/azure_client.py:63` (client built with the pinned transport, `:71-74`) | `core/security/pinned_http.py:24`, on every request | Model calls on a workspace key's endpoint | A workspace manager | `model` | Off (`pinned_http.py:39`); a 3xx reaches the SDK as an answer |
| 13 | `api/user_api_keys.py:207` (client `:203`) | `pinned_http.py:24` | `POST /api/keys/{id}/test` (`:377`) on a key with its own endpoint | A workspace manager | `model` | Off, as row 12 |

Rows 11 to 13: the guard is applied where the edition's `guards_user_endpoints` is set, which is `saas` (`core/editions.py:223`) and `enterprise` (`:256`) (`byok_endpoint.py:35-38`, `:85-86`).

## 3. Fixed and infrastructure destinations

These were considered. The address is a constant of the product or the install's own configured infrastructure, so no one on a request chooses it.

**Constants of the product**
- Chat platform APIs: `channels/drivers/slack.py:53,101` (`:32`), `discord.py:54,91` (`:35`), `whatsapp.py:58,103` (`:34`), `telegram.py:80,128,181,207,224` (`:45`); the older adapters `channels/whatsapp_adapter.py:68,107`, `telegram_adapter.py:98`, `discord_adapter.py:102`, `slack_adapter.py:103`.
- Retell: `modules/voice/retell_api.py:219,259,285,319` (`:28-33`).
- Clerk API: `core/auth/clerk.py:317,345,374,416,434,456,483,511` (`:95`). GitHub App token: `modules/codegraph/github_auth.py:66`.
- Composio: `core/composio/client.py:738,878`, `services/composio_api_service.py:131` (`:42`), `modules/rag/services/cloud_file_downloader.py:296`. LinkedIn API and token endpoints: `linkedin_image_workaround.py:232,406,469`.
- Model catalogue and provider calls: `core/services/anthropic_catalog_sync.py:232`, `openrouter_sync_service.py:170`, `core/llm/clients/huggingface_client.py:72,130`.
- Stripe: `services/billing/stripe_api.py:116`. Expo push: `services/push/expo.py:304,314,347`.
- Firecrawl: `modules/intake/firecrawl_client.py:95,170` (`:45`; the site to read is handed to that service, not opened from here).
- Git hosts: `modules/codegraph/codegraph_service.py:481` and `modules/agents/services/skill_loader.py:641`, restricted to three named hosts (`core/security/git_sanitizer.py:19-23`, `:71-74`).
- Music tracks at image build time, from a committed manifest: `services/media-render/media_render/music_build.py:248`.

**The install's own configured infrastructure**
- Database and Redis: `core/database/*`, `core/redis/client.py:49,69`, `core/task_runner/queued.py:118,128`, `services/chat_turns.py:100`, `modules/context/adapters/redis_context.py:41,46`, `modules/intake/progress.py:103`, `services/workspace-worker/main.py:104`, `task_executions_db.py:91`.
- Object storage and vectors: `core/storage/s3.py:152` (`S3_ENDPOINT_URL`), `modules/search/vector_store/backends/s3_vectors_backend.py`, `core/qdrant/client.py:124` and `services/qdrant_snapshots.py:119-120` (`QDRANT_URL`).
- Companion services: `core/workspace_client.py:53,192` (`WORKER_INTERNAL_URL`, `:62`), `core/services/futureagi_service.py:239,260`, `core/media_render_client.py:126` (`SOCIALS_RENDER_URL`), `modules/voice/speech_client.py:95` (`VOICE_TTS_URL`), `services/media-render/media_render/pipeline.py:174` (storage prefixes the operator lists, checked at `:195`), `services/cli-host/automatos_cli_host/api.py:50` (the backend address given to the host).
- Operator-set endpoints: `modules/tools/discovery/handlers_monitoring.py:172` (`GRAFANA_URL`), `:188` (`LOKI_URL`), `:285` (`PROMETHEUS_URL`); `core/monitoring/automatos_logging.py:296` and `automatos_logs_api.py:137,192` (log relay and Loki); `core/observability/otel.py:218` (OTLP collector); `services/web_search.py:303` (`SEARXNG_URL`); `services/content_reports/moderation.py:142` (`MODERATION_WEBHOOK_URL`); `services/themes/revocation_pull.py:207` and `services/marketplace_cloud/fetch.py:137` (each checks its own setting: https, or http to this machine); `core/railway_client.py:49`; model providers with a base URL setting (`core/llm/providers.py:339-343`, `core/llm/rerank_manager.py:140`, `core/llm/openrouter_analytics.py:53,163,197`, `core/llm/decisions/typesafe_client.py:125`, `core/llm/clients/openrouter_embedding.py:80`).
- Identity: `core/auth/oidc_http.py:228-230` (the configured issuer and the addresses its metadata names, PRD-277; it applies the shared range list through its own `pin`, `:163`) and `core/auth/clerk.py:111` (the configured JWKS address).

No client for MCP servers a workspace registers exists in `orchestrator/` or `services/`: the platform serves MCP, it does not call out to one.

## 4. For the owner

| Row | Could be | What would decide it |
|---|---|---|
| 5. Run callbacks | `callback` or `fetch` | PRD-282 5.1 describes `callback` as "a workspace's own settings". Here the address is supplied by the caller on each request, not stored as a setting. If a per-request destination counts as the workspace's own, it is `callback`; if it counts as whatever a caller typed, it is `fetch`. Neither purpose may be allowed by D-2, so the choice changes naming only. |

Rows 8 and 9 take their purpose from PRD-282 5.1. The address there is named by a provider's answer, not typed by a person; if that is to be a purpose of its own, it needs a decision before US-102.

## 5. What this means for US-102

Call sites that must pass a purpose once the guard takes one:

- `resolve_outbound_async`: `partner/client.py:111`, `handlers_web.py:113`, `api/heartbeat.py:157`, `heartbeat_service.py:1397`, `run_callbacks.py:329`, `files.py:135`, `linkedin_image_workaround.py:349`.
- `resolve_outbound`: `run_callbacks.py:154`, `pinned_fetch.py:66`, `cloud_file_downloader.py:442`, `byok_endpoint.py:91`, `core/security/pinned_http.py:24`.
- `validate_outbound_url` (`web_access.py:162`) has no caller; `validate_webhook_url` (`url_validator.py:80`) has no production caller.
- `core/auth/oidc_http.py:163` (`pin`) is the `identity` caller; today it uses `build_pinned_request` and the shared range list, not `resolve_outbound`.
- `pinned_http.py:24` and `pinned_fetch.py:66` serve several rows each (12 and 13; 6 and 7), so the purpose has to reach them from their callers: `endpoint_http_client` (`byok_endpoint.py:95`) and `fetch_public` (`pinned_fetch.py:98`).

Under D-2 an allowance could apply only to `partner` (row 1) and `model` (rows 11, 12, 13). Rows 2, 3, 6 to 10 are `fetch`; rows 4 and 5 are `callback` or `fetch`; none of these may be allowed.

Tests PRD-282 section 9 names as pinning today's shape exist at this commit: `test_ssrf_blocked_ranges.py`, `test_prd240_web_access.py`, `test_f375_a_document_fetch_is_pinned_and_checks_every_hop.py`, `test_heartbeat_webhook_pinned.py`.

PRD-282 section 2.1 differs from the code at this commit in these places: the config lines are `config.py:539-544` and `:560`, not `:528`, `:545`; the model-endpoint check is at `byok_endpoint.py:85-91` and applies to `saas` and `enterprise`; the agent web fetch follows at most 5 redirects, not 3; the sign-in client (`core/auth/oidc_http.py`) now exists.

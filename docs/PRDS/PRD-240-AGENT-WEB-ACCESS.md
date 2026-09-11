# PRD-240: Agents can read and search the web — natively in the local edition, through the key the user already has

> **Status:** BUILDING 2026-09-11 (owner: "quick PRD and then straight to the work") — on `feat/prd-240-agent-web-access`, cut from `main` @ `606da81bc` (#732). Grounded @ the 2026-09-10/11 review of an open-source install ("we can't load Composio, the marketplace is empty, they have no tools") and the provider verification below.

---

## Framing (CLAUDE.md §3)

**Extension.** Every surface that gives an agent the web today is a Composio *app*: `http_request` refuses every non-internal URL in both editions (`modules/tools/execution/exec_shell.py:32-41`, and runs `verify=False`), the RAG tool's own description sends the model to `TAVILY_SEARCH` / `COMPOSIO_SEARCH_WEB` by name (`tool_registry.py:411`), and the only search wiring in the codebase is the Composio catalogue. No provider-native web search is wired anywhere (`grep web_search_2025|:online|google_search` → nothing). The SaaS restriction on web search is therefore not a gate at all — it is "nobody connected the app".

The open-source user does not want a marketplace app they must find, connect and (in the four agent pickers) assign. They want what they get when they install Claude on their Mac: **reading a page needs nothing; searching comes with the model key they already have.** That is exactly how Claude Code works — `WebFetch` is a local HTTP GET, `WebSearch` is Anthropic's server-side tool on the same key — and it is the pattern this PRD copies.

## Grounded facts (verified 2026-09-11)

| Provider | How search is enabled | Price | Notes |
|---|---|---|---|
| **OpenRouter** (the default route: `defaults.py:13-14`) | `tools: [{"type": "openrouter:web_search"}]` — model decides when to search; `max_uses`, `max_results`, `allowed_domains`, `excluded_domains`, `engine` | Exa $0.007 / request (Parallel $0.001–0.005; native passthrough billed by the vendor) | **Works with any model** on OpenRouter, Kimi/DeepSeek included. Results arrive as `url_citation` annotations; nothing to echo back on later turns. The plugin form (`:online`) runs a search on **every** request — not used. |
| Anthropic direct | `{"type": "web_search_20250305"}` + `web_fetch_20250910` server tools | $10 / 1k searches; fetch free beyond tokens | Response carries `server_tool_use` / `web_search_tool_result` blocks; `encrypted_content` must be echoed back. Not on Bedrock. Our client parses only `text`/`tool_use` blocks (`anthropic_client.py:247-249`). |
| Google Gemini direct | `tools: [{google_search: {}}]` | 3.x: 5,000 free / month then $14 / 1k · 2.5: 1,500 / day free | Our client is on the legacy SDK (`google_client.py:42`); shape unconfirmed there. |
| OpenAI direct | Responses API `{"type": "web_search"}` | $10 / 1k calls | Our client is Chat Completions only (`openai_client.py:120`). |
| Composio | `COMPOSIO_SEARCH` toolkit (22 tools) — **no auth**, activates instantly (`api/composio.py:350-360`); `Browser tool` — no auth | free tier 100k calls / month, no card | Already routed end-to-end; the platform already calls Composio actions as an internal engine with no agent involvement (`modules/rag/services/cloud_sync_service.py:44-48`). Execution needs only the workspace entity (`tool_executor.py:683`). |
| SearXNG | `GET /search?q=…&format=json` (json must be enabled in `settings.yml`) | free, self-hosted, one container (`searxng/searxng`) | Home-IP egress works well; datacenter IPs get captcha'd by upstream engines. |
| NVIDIA, DeepSeek direct | none | — | Free models with no search of their own — the case the platform action (S3) exists for. |

The platform's own guardrails that make native fetch safe today: `core/security/url_validator.py:validate_webhook_url` resolves DNS and blocks `10/8 172.16/12 192.168/16 127/8 169.254/16` + IPv6 equivalents (so sibling containers, `host.docker.internal` and cloud metadata are unreachable); `execute_command` has a 16-binary allowlist with no `curl`/`wget` and refuses shell metacharacters (`agent_action_executor.py:45-55`), so a tool is the only door.

## Decisions

- **D1 · Web access is a platform capability, not a marketplace app.** `web_fetch` and `web_search` are platform actions in the ActionRegistry (the same 3-file pattern as every `platform_*` action). Every agent has them; nothing to connect, nothing to assign; discoverable through `platform_find_tools` / `platform_execute` at near-zero standing token cost. `http_request` stays the internal self-test tool it is — its allowlist and `verify=False` are untouched.
- **D2 · Fetch is native and keyless.** `httpx` + `validate_webhook_url` (private ranges always blocked, not configurable) + TLS verification on + the user's denylist + a byte cap + HTML→text. Works with no key at all.
- **D3 · Search resolves a backend in a fixed order — `openrouter → composio → searxng → none`.** OpenRouter first because it is the recommended first key and works for every route (a cheap completion with the server tool, results read from `url_citation` annotations); Composio second because anyone with `COMPOSIO_KEY` for integrations gets search free; SearXNG third for the zero-key self-hoster; otherwise an honest result naming the three options. `WEB_SEARCH_PROVIDER` pins one.
- **D4 · The recommended route also searches inside its own turn.** When the agent's route is OpenRouter and `WEB_ACCESS` is on, the client appends `{"type":"openrouter:web_search", "max_uses": N, "excluded_domains": [...]}` to the request. The model decides when to search; citations are appended to the reply as a Sources list. This is the "Claude on my Mac" experience on the default route. Anthropic/Gemini/OpenAI-direct native search is **not** built here — all three are reachable through OpenRouter, and the direct clients need parser work (Non-goals).
- **D5 · One switch, open by default locally, off in SaaS.** `WEB_ACCESS` (default `on` when `AUTH_EDITION=local`, `off` otherwise) + `WEB_ACCESS_DENY` (comma-separated hosts, suffix match). Owner decision 2026-09-11: *"Open by default, denylist to restrict."* In SaaS "off" is now a real gate — today there is none.
- **D6 · No new keys.** Every backend uses a key the user has for another reason (OpenRouter for models, Composio for integrations) or no key (SearXNG, native fetch). Owner: *"I don't want more API keys if possible."*
- **D7 · OpenRouter is recommended by name in the docs.** It already is the default provider (`DEFAULT_LLM_PROVIDER = "openrouter"`); the QUICKSTART now says so instead of listing five keys as equals, with NVIDIA as the free companion. Owner: *"I personally use OpenRouter and recommend it to users, I also add NVIDIA for some free LLMs."*
- **D8 · Every documented dial is forwarded by compose.** Three variables have fallen through the `.env` → compose lane this month (`COMPOSIO_KEY`, `COMPOSIO_API_KEY`, `FIRECRAWL_API_KEY`); 26 of 403 config dials reach the container from `.env`. The lane guard from #724 is extended to every dial this PRD adds.

## Stories

### S1 · The switch, the denylist, the provider pin (S)
**Files:** `orchestrator/config.py` (`WEB_ACCESS: bool` — default `AUTH_EDITION == "local"`; `WEB_ACCESS_DENY: tuple[str, ...]`; `WEB_SEARCH_PROVIDER: str` ∈ {`auto`, `openrouter`, `composio`, `searxng`, `off`}, default `auto`; `WEB_SEARCH_MAX_RESULTS: int` default 5; `SEARXNG_URL: str` default empty) — NEW `orchestrator/core/security/web_access.py` (`web_access_enabled()`, `host_denied(host)`, `validate_outbound_url(url)` = scheme + denylist + `validate_webhook_url`); `orchestrator/reports/config-surface.json` (names); `docker-compose.yml` (forward all five); `.env.example` (commented block); `tests/test_prd209_quickstart_honest.py` (lane guard covers them).
**Test:** local default on, saas default off, explicit override wins; denylist suffix-matches (`example.com` blocks `www.example.com`, not `notexample.com`); private ranges refused regardless of the denylist; compose forwards every dial.
**Editions:** both (saas: off unless set).

### S2 · `web_fetch` — read a page with no key (M)
**Files:** NEW `orchestrator/modules/tools/discovery/actions_web.py` (`register_web_actions`: `web_fetch`, `web_search`; `category="web"`, `permission_level="read"`, not promoted, tagged `web research browse fetch search`); NEW `orchestrator/modules/tools/discovery/handlers_web.py` (`web_fetch(db, workspace_id, params)`: `validate_outbound_url` → `httpx.AsyncClient(follow_redirects=True, verify=True, timeout=WEB_FETCH_TIMEOUT_SECONDS)` → re-validate the final URL after redirects → byte cap `WEB_FETCH_MAX_BYTES` (2 MB) → HTML→text via `beautifulsoup4` (drop `script/style/nav/footer/header/aside`, keep title + headings + paragraphs + links) → `{url, final_url, title, content, truncated, content_type}`; `text/*`, `application/json`, `application/xml` returned as-is; anything else refused with the content type named); `platform_executor.py` (handler map); `platform_actions.py` (`register_web_actions`); `tool_registry.py:411` (the RAG tool's hint now names `web_search` / `web_fetch`).
**Test:** registration + schema truth (`required == ["url"]`); refuses `file:`/`ftp:`, private ranges, denied hosts, a redirect that lands on a private range; caps bytes and says `truncated`; HTML→text keeps headings and drops scripts; `WEB_ACCESS=off` → `{available:false, reason}` never an exception. All with a stubbed transport — no network.
**Editions:** local on by default; saas only when `WEB_ACCESS=on`.

### S3 · `web_search` — find pages through whichever backend the user has (M)
**Files:** NEW `orchestrator/services/web_search.py` (`resolve_backend()` → `openrouter | composio | searxng | None` honouring `WEB_SEARCH_PROVIDER`; `search(query, max_results, workspace_id, db)` → `[{title, url, snippet}]`; backends: `_openrouter` = one chat completion on `WEB_SEARCH_OPENROUTER_MODEL` (default `openai/gpt-4o-mini`) with `tools=[{"type":"openrouter:web_search","max_results":N,"excluded_domains":deny}]` and a fixed instruction to list results, results read from `url_citation` annotations (falls back to the text when annotations are absent); `_composio` = `ComposioToolExecutor.execute(action=WEB_SEARCH_COMPOSIO_ACTION, params={"query":…}, workspace_id, skip_validation=True)` after `EntityManager.get_or_create_entity`, results parsed defensively (`results|items|organic` × `title|url|link|snippet`); `_searxng` = `GET {SEARXNG_URL}/search?q=&format=json&safesearch=1`); `handlers_web.py` (`web_search` handler → the service; `{available:false, reason, options:[…]}` when no backend; denied hosts filtered out of results too).
**Test:** resolver order and pin (each key combination); `off` wins; each backend with a stubbed client returns the normalised shape; annotation parsing (with and without `content`); a backend error is `{success:false, error}` never an exception; results on denied hosts are dropped.
**Editions:** as S2.

### S4 · The OpenRouter route searches inside its own turn (M)
**Files:** `orchestrator/core/llm/providers.py` (`ProviderSpec.web_search_tool: Optional[str]` — `"openrouter:web_search"` on the OpenRouter spec, `None` elsewhere); `orchestrator/core/llm/clients/openai_compatible_client.py` (`_request_kwargs`: when `web_access_enabled()` and `self.spec.web_search_tool` → append `{"type": spec.web_search_tool, "max_uses": WEB_SEARCH_MAX_USES_PER_TURN, "max_results": WEB_SEARCH_MAX_RESULTS, "excluded_domains": list(WEB_ACCESS_DENY)}` to `tools` (creating the list if the turn had none); `_StreamAccumulator.feed` and the non-stream path collect `annotations[].url_citation` → `LLMResponse.citations`; the "does not support tool use" retry also drops the server tool); `core/llm/clients/base.py` (`LLMResponse.citations: Optional[List[Dict]]`); `consumers/chatbot/service.py` (when a reply carries citations, append a short **Sources** list — `[title](url)` — to the persisted assistant text, the same way an image URL is appended to `content` today).
**Test:** the entry is appended only for the OpenRouter spec and only when the switch is on; `_sanitize_tools` leaves it intact (an entry with a `type` is not wrapped — `base.py:145-149`); annotations become `citations` in both paths; the Sources list renders from citations; no server tool when `WEB_ACCESS=off`.
**Editions:** both, behind the switch.

### S5 · SearXNG as a compose profile (S)
**Files:** `docker-compose.yml` (`searxng` service, `profiles: ["search"]`, image `docker.io/searxng/searxng:latest`, `BIND_ADDRESS`-bound port `${SEARXNG_PORT:-8888}:8080`, `SEARXNG_BASE_URL`, mounts `./envs/searxng/:/etc/searxng/`); NEW `envs/searxng/settings.yml` (`use_default_settings: true`, `search.formats: [html, json]`, `server.secret_key` from env, `limiter: false`); backend `SEARXNG_URL` default empty (the docs say `http://searxng:8080` under the profile).
**Test:** the compose file parses; the profile is not in the default set; the settings file enables json.
**Editions:** local.

### S6 · The docs say it plainly (S)
**Files:** `QUICKSTART.md` (§"Optional: one LLM key" → **Recommended: OpenRouter**, NVIDIA as the free companion, everything else in Settings → API Keys; NEW §"Web access for your agents" after "What you get": reading pages works out of the box, searching needs one of three — your OpenRouter key, Composio, SearXNG — the switch and the denylist; the "does not work out of the box" bullet rewritten); `README.md` (one line + link in "Paid, free, or on your subscription — one router"); `docs/getting-started/self-hosting.md` (NEW §"Web access" with the dials, the resolver order, the SaaS posture); `.env.example`.
**Test:** `test_prd209_quickstart_honest.py` — QUICKSTART names `WEB_ACCESS`, names OpenRouter as recommended, names all three search options; compose forwards every dial the docs name.
**Editions:** docs.

## Functional requirements

- FR-1 `WEB_ACCESS` defaults to on when `AUTH_EDITION=local` and off otherwise; an explicit value wins in both editions.
- FR-2 Private, loopback, link-local and reserved ranges are refused by `web_fetch` and filtered from `web_search` results in every configuration; the denylist adds to that, never replaces it.
- FR-3 `web_fetch` verifies TLS, follows redirects and re-validates the final URL, caps the body at `WEB_FETCH_MAX_BYTES`, and returns text — never raw HTML — for HTML pages.
- FR-4 `web_search` returns `[{title, url, snippet}]` from the first available backend in `openrouter → composio → searxng` order, or `{available:false, reason, options}` naming the three ways to enable it. It never raises into the agent loop.
- FR-5 When the agent's route is OpenRouter and `WEB_ACCESS` is on, the `openrouter:web_search` server tool is attached with `max_uses`, `max_results` and the denylist as `excluded_domains`; citations reach the persisted reply as a Sources list.
- FR-6 With `WEB_ACCESS=off`, no server tool is attached and both actions return `{available:false, reason}`.
- FR-7 Every dial this PRD introduces is read in `config.py` only, listed in `reports/config-surface.json`, forwarded by `docker-compose.yml`, present (commented) in `.env.example`, and covered by the lane-guard test.
- FR-8 Nothing changes for `http_request`, for Composio apps as marketplace items, or for any SaaS workspace that has not set `WEB_ACCESS`.

## Non-goals

- Native search on the Anthropic-direct, Gemini-direct and OpenAI-direct clients (all reachable through OpenRouter; the direct clients need block-parser / API-surface work — a follow-on if a user needs a direct key).
- Per-agent or per-workspace toggles and a Settings UI for the dials (env + restart, like every other local-edition dial; a status line can follow).
- A headless browser / JavaScript rendering (Composio's Browser tool or Firecrawl cover it for those who need it).
- Replacing Firecrawl in the intake pipeline — PRD-241-adjacent onboarding work is its own decision (see Open questions).
- Any change to the marketplace, the agent tool pickers, or Composio app assignment.

## Technical considerations

- `_sanitize_tools` wraps only entries lacking a `type`, so a server-tool entry passes through the OpenRouter client unchanged; the "model does not support tool use" retry must drop it along with the function tools.
- The OpenRouter server tool is billed per search on the user's OpenRouter account; the `cost` field the client already reads lands in Analytics like any other call. The `:online` plugin form is deliberately not used — it searches on every request.
- `platform_executor` invokes handlers as `await handler(self.db, self.workspace_id, params)`; the Composio backend needs `agent_id` for its validation path, so the internal call passes `skip_validation=True` (the pattern `cloud_sync_service` uses) and a sentinel agent id.
- The Composio web-search action slug is a constant (`COMPOSIO_SEARCH_WEB`, the name already used in `unified_executor.py:90-92` and the RAG hint); the result shape is parsed defensively because it is not pinned by a contract in this repo.
- HTML→text uses `beautifulsoup4` (already pinned); no new dependency.
- All new tests are pure (stubbed transports); CI is the gate — nothing is run locally.

## Success metrics

- A fresh local install with only an OpenRouter key: "research X for me" produces a reply with sources, with no marketplace visit and nothing assigned.
- A fresh local install with an NVIDIA key and a Composio key: the same, through `web_search` → Composio.
- A fresh local install with no keys: `web_fetch` of a public URL returns page text; `web_search` returns the honest "three options" result.
- `curl http://postgres:5432` / `http://host.docker.internal` / `http://169.254.169.254` through `web_fetch` are refused in every configuration.
- SaaS: no request carries a server tool and no action fetches, until an operator sets `WEB_ACCESS=on`.

## Open questions

- **Onboarding optional-from-start + `FIRECRAWL_API_KEY` forwarded through compose** (2026-09-11 thread): agreed in principle (`ONBOARDING_ENABLED` dial seeding `skipped`; doctrine gate; Firecrawl free plan is 1,000 credits / month ≈ 9 onboarding scans). Owner's call whether it rides this PR or its own PRD.
- The `WEB_SEARCH_OPENROUTER_MODEL` default (`openai/gpt-4o-mini`) is a cost/quality guess for the S3 backend call; the Exa engine does the actual search, so any cheap tool-capable model works.
- Whether `web_fetch` should also accept PDFs (Anthropic's server fetch does; the RAG pipeline already parses PDFs, so it is a small extension later).

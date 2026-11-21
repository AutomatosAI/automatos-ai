# PRD-280 US-401: can the worker's coding agent be sent through a relay

**For:** [PRD-280](PRD-280-CHART-HARDENING-AND-WORKER-ISOLATION.md) US-401 (analysis), section 5.7 and US-402 to US-404, which build on it.
**Read at:** branch `feat/prd278-enterprise-requires-verified-database-tls` @ `2d90dab0a`, 21 Nov 2026, from the code and from the agent's own documentation, fetched the same day. Nothing was run.
**Labels:** **[T]** read in code at that commit; **[D]** quoted from the documentation linked in section 8; **[I]** inferred; **not confirmed** says what would settle it.

## 1. Verdict

**The design in section 5.7 can be built, with the changes in section 6.** The agent's documentation describes exactly this arrangement: point the agent at a gateway with a base URL, give it a gateway credential, and let the gateway hold the provider's credential. Two parts of 5.7 do not hold as written: the relay cannot be one request format for every provider, and the worker's internal token cannot be the relay's second check (section 6).

One relay serves **Claude models only**: the provider's own API (Anthropic) and Claude on Amazon Bedrock reached with the API's role. It cannot serve an install whose providers are all non-Claude (OpenAI, Google, Azure OpenAI and the rest). For Bedrock, the relay either speaks Bedrock's own request format or forwards to Bedrock's Anthropic-format endpoint; it does not reuse the call the API makes to Bedrock today (section 5).

## 2. What the worker runs today, and how it is given a model

| What | Where | Fact |
|---|---|---|
| The agent | `services/workspace-worker/requirements.in:50-54`; `requirements.lock:302` **[T]** | `claude-agent-sdk==0.2.168`, the Claude Agent SDK for Python, pinned by hash. |
| The program it drives | `services/workspace-worker/Dockerfile:145-149`, `:261-266` **[T]**; the SDK's `_cli_version.py` and `CHANGELOG.md` at tag `v0.2.168` **[D]** | The SDK starts the Claude Code command-line program that its own wheel bundles. The SDK's changelog for 0.2.168: "Updated bundled Claude CLI to version 2.1.301". No other agent program is installed or started by the worker **[T]**. |
| How a session starts | `canvas_session_service.py:126-131`, `:370-378`, `:411-413` **[T]** | `ClaudeSDKClient(options=ClaudeAgentOptions(**option_kwargs))` with `cwd`, `env` (only `CLAUDE_CONFIG_DIR`), `permission_mode` and `can_use_tool`. No `model` is passed, so the program picks its own default. |
| Where the model credential comes from | `worker_config.py:31-49` **[T]** | `model_auth_env()` reads `CLAUDE_CODE_OAUTH_TOKEN` and `ANTHROPIC_API_KEY` from the worker's environment and returns those that are set. |
| How it reaches the agent | `canvas_session_service.py:384-386` **[T]** | The two variables are merged into `ClaudeAgentOptions.env`. The SDK merges that `env` over the worker's own environment and passes the result to the program it starts (the SDK's `subprocess_cli.py` at `v0.2.168`, and the Python reference quoted in section 3.1) **[D]**. |
| What sets them | `charts/automatos/templates/worker.yaml:10`, `:76-83`; `docker-compose.yml:587-588` **[T]** | Both are optional keys of the worker's Secret in the chart, and plain environment values in compose. |
| Providers that path supports | **[T]**, from the two variables | One: Anthropic, by an API key (`x-api-key`) or a Claude subscription token. Nothing else is wired: no base URL, no Bedrock or Vertex variable. |
| The "no model credential" failure | `canvas_session_service.py:384-406` **[T]** | With neither variable set (and the real SDK factory), the session is marked `failed` with `last_error = "missing model credential"`, logged, and the start returns "Canvas worker has no model credential — set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY on the workspace-worker service (PRD-203 C·S8)." |
| What the web app is told | `canvas_session_service.py:460` **[T]** | `sdk_available` is false without a credential, and the canvas hides its session tab (PRD-239). |
| How the API starts a session | `orchestrator/api/workspace_files.py:220-236`; `core/workspace_client.py:290-302` **[T]** | `POST …/canvas/sessions`, gated on `agents:execute`, proxies to the worker with an empty body. No agent is named: a canvas session belongs to a workspace, not to an agent. |
| Model usage of a canvas session | searched `orchestrator/` for any write of canvas usage **[T]** | None. `UsageTracker.track_session` (`core/llm/usage_tracker.py:510-568`) books CLI-host sessions at $0 on the user's own plan; nothing records the canvas's model calls. |

## 3. What the agent's documentation says about a gateway

The documentation is not versioned per release; it is the current one, and it marks features that need a later release than some. Each such mark below is checked against the bundled 2.1.301.

### 3.1 The settings

- The SDK has no gateway option of its own: "The Agent SDK has no gateway-specific options; it passes environment variables to the Claude Code process it spawns." And for Python: "`ClaudeAgentOptions(env=...)` merges on top of the inherited environment, so gateway variables set in the parent process carry through without spreading." The page's Python example sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` in that `env` **[D]** (connect page, "Agent SDK").
- `ANTHROPIC_BASE_URL`: "Override the API endpoint to route requests through a proxy or gateway." **[D]** (environment variables).
- `ANTHROPIC_AUTH_TOKEN`: "Custom value for the `Authorization` header (the value you set here will be prefixed with `Bearer `)". `ANTHROPIC_API_KEY`: "API key sent as `X-Api-Key` header." **[D]** (environment variables).
- The SDK's own security guide recommends this pattern: "run a proxy outside the agent's security boundary that injects credentials into outgoing requests", and of `ANTHROPIC_BASE_URL`: "This tells Claude Code and the Agent SDK to send sampling requests to your proxy instead of the Claude API directly. Your proxy receives plaintext HTTP requests, can inspect and modify them (including injecting credentials), then forwards to the real API." Its example is `http://localhost:8080`, so a plain-HTTP address inside the cluster is accepted **[D]** (secure deployment).
- For a gateway that speaks Bedrock's format: `CLAUDE_CODE_USE_BEDROCK=1`, `ANTHROPIC_BEDROCK_BASE_URL=<gateway>` and `CLAUDE_CODE_SKIP_BEDROCK_AUTH=1`. "The skip-auth variables … tell Claude Code not to sign requests with the cloud provider's credentials, since the gateway holds those." **[D]** (connect page, "Route to a cloud provider through a gateway").
- For Bedrock's Anthropic-format endpoint ("Mantle"): `CLAUDE_CODE_USE_MANTLE=1`, `CLAUDE_CODE_SKIP_MANTLE_AUTH=1`, `ANTHROPIC_BEDROCK_MANTLE_BASE_URL=<gateway>` **[D]** (Amazon Bedrock, "Route Mantle through a gateway").
- The model: the SDK's `model` option ("Claude model alias or full model name"), or `ANTHROPIC_MODEL`; background tasks follow `ANTHROPIC_DEFAULT_HAIKU_MODEL`, subagents `CLAUDE_CODE_SUBAGENT_MODEL` **[D]** (Python reference; environment variables).
- Traffic that does not go to the model: "Claude Code also sends nonessential background traffic outside the gateway path, to Anthropic and to third-party services such as GitHub", turned off with `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`; the WebFetch domain check "still calls `api.anthropic.com`" and is turned off with `skipWebFetchPreflight: true` **[D]** (connect page).

### 3.2 The request format the gateway receives

"A gateway must expose at least one of the following API formats" **[D]** (compatibility guide, "API formats"):

| Format | Selected by | Endpoints | Forward unchanged |
|---|---|---|---|
| Anthropic Messages | `ANTHROPIC_BASE_URL` | `/v1/messages`, `/v1/messages/count_tokens` (optional) | `anthropic-beta` and `anthropic-version` request headers |
| Amazon Bedrock InvokeModel | `ANTHROPIC_BEDROCK_BASE_URL` with `CLAUDE_CODE_USE_BEDROCK=1` | `/model/{model}/invoke`, `/model/{model}/invoke-with-response-stream`, `/model/{model}/count-tokens` (optional) | `anthropic_beta` and `anthropic_version` request body fields |
| Google Cloud's Agent Platform rawPredict | `ANTHROPIC_VERTEX_BASE_URL` with `CLAUDE_CODE_USE_VERTEX=1` | `:rawPredict`, `:streamRawPredict` | headers and the `anthropic_version` body field |

(The table is abridged from the guide.) Further points from the same guide **[D]**:

- "Inference requests post to `/v1/messages?beta=true`". "Match on the path, not the full URL."
- Startup traffic it "can reject without breaking anything": a `HEAD /api/hello` probe on the Anthropic format; `GET /inference-profiles?type=SYSTEM_DEFINED` and profile lookups on the Bedrock format.
- Which format decides what is sent: "When the client speaks the Anthropic Messages format, Claude Code sends the full set, even if your gateway forwards to an Amazon Bedrock or Google Cloud's Agent Platform upstream. Bridging that difference is your gateway's job." And: "If your upstream is Amazon Bedrock or Google Cloud's Agent Platform, you can avoid the bridging by exposing that provider's format instead."
- Model IDs: on the Anthropic format, "Anthropic IDs, such as `claude-opus-4-8`"; on the Bedrock format, "The provider's form, such as `us.anthropic.claude-opus-4-8`".
- The body of a Bedrock InvokeModel request for Claude carries `"anthropic_version": "bedrock-2023-05-31"` and the Messages fields; the model is in the path, `POST /model/{modelId}/invoke-with-response-stream` **[D]** (AWS, Claude Messages parameters; AWS, InvokeModelWithResponseStream).
- Request headers the gateway may read: `x-claude-code-session-id` ("a unique identifier for the current Claude Code session"), `x-claude-code-agent-id` for subagents **[D]**.

### 3.3 How a request is authenticated to the gateway

- "Each variable sends the credential in a different HTTP header: `ANTHROPIC_AUTH_TOKEN` in `Authorization: Bearer`, `ANTHROPIC_API_KEY` in `x-api-key`, and `apiKeyHelper` in both." **[D]** (connect page).
- On the Bedrock route: "add `ANTHROPIC_AUTH_TOKEN` after the block. Claude Code sends it to the gateway as an `Authorization: Bearer` header. … Keep the skip-auth variable set either way, since without it Claude Code removes any `Authorization` header". And "Leave `AWS_BEARER_TOKEN_BEDROCK` unset when the gateway issues its own credential." **[D]** (connect page).
- Extra headers: `ANTHROPIC_CUSTOM_HEADERS`, "`Name: Value` format, newline-separated"; it "Requires Claude Code v2.1.227 or later" (2.1.301 has it) **[D]**.
- A rotating credential: `apiKeyHelper`, a command whose output "Claude Code … sends … as both the `X-Api-Key` and `Authorization: Bearer` headers", cached "five minutes by default" and re-run on a `401` or `403` **[D]** (settings reference).

So the session token in `ANTHROPIC_AUTH_TOKEN` arrives as `Authorization: Bearer <token>` on both routes. That is the header the API's bearer registry reads (`orchestrator/core/auth/bearer_registry.py:85-92`) **[T]**, as section 9 of the PRD assumes. A token sent in `x-api-key` would instead be read as an API key (`:95-100`) **[T]**.

### 3.4 What the gateway must add or change before it forwards

- **Its own credential.** "Deploy the gateway and give it your provider credential, so it can authenticate the requests it forwards." "The provider key stays server-side; developers hold gateway credentials instead." **[D]** (LLM gateways).
- **Headers to keep.** "Forward `anthropic-version` and `anthropic-beta` unchanged … The rest the gateway may consume for routing, attribution, and tracing, and need not forward." "Forward the header verbatim; don't allowlist individual values, because the set changes with Claude Code releases." **[D]** (compatibility guide, "Request headers").
- **The body.** "pass `anthropic-*` request headers and request body fields through unchanged rather than allowlisting the ones you see today." "A gateway that rewrites or redacts request bodies for content inspection breaks the pairing the same way stripping does, so inspect without modifying." Keep `cache_control` and the `system` array as sent, or caching is lost **[D]**.
- **Responses.** Return `retry-after` in seconds, pass `x-should-retry` through, and "forward error response bodies unmodified, so Claude Code's capability-rejection recovery can match the upstream's error wording." **[D]**
- **Bedrock.** The gateway signs with AWS credentials in place of the client (the skip-auth variables above). For Bedrock's Anthropic-format endpoint, the role needs `bedrock-mantle:CreateInference`: "Mantle has its own IAM actions under the `bedrock-mantle:` prefix, so the `bedrock:` actions … don't cover it" **[D]** (Amazon Bedrock).

### 3.5 Streaming through the gateway

From the compatibility guide, "Streaming" **[D]**:

- "If your gateway buffers a response until it is complete, Claude Code stalls."
- "Claude Code expects each response's full event sequence, in order, through the final `message_delta` and `message_stop` events."
- "Claude Code aborts a streaming response once no bytes reach it for longer than its streaming idle timeout. … a gateway that strips or buffers [`ping` events] can trip that timeout".
- On the Bedrock format, "Claude Code reads the `/model/{model}/invoke-with-response-stream` response as the binary `application/vnd.amazon.eventstream` body Bedrock returns, and can't parse it once a gateway converts it to server-sent events or rewrites that `Content-Type` header."
- Response `content-type`: "`text/event-stream` on streamed Anthropic Messages-format responses, and `application/vnd.amazon.eventstream`, unmodified, on Amazon Bedrock-format responses".

From the Messages streaming page **[D]**: the event flow is `message_start`, content blocks, "One or more `message_delta` events", "A final `message_stop` event", and "The token counts shown in the `usage` field of the `message_delta` event are *cumulative*." The relay can read usage from those events as it passes them on.

## 4. What a relay would call on the API side

| Need | What exists | Where |
|---|---|---|
| Which credential pays for a provider in a workspace | `resolve_provider_key(db, provider, workspace_id=…)`: the workspace's own key, then the platform's, then the pod's role for Bedrock when `AWS_USE_WORKLOAD_IDENTITY` is on | `core/llm/key_resolver.py:56-75`, `:102-115` **[T]** |
| Which providers can use the pod's role | Bedrock only (`workload_identity=True`) | `core/llm/providers.py:171-174`, `:289-298` **[T]** |
| A Bedrock client on those credentials | `runtime_client(...)`: a Bedrock API key as a bearer, an access key pair, or the default chain (the role) | `core/llm/clients/bedrock_client.py:70-90` **[T]** |
| How the API calls Bedrock today | `client.converse(...)`: the Converse API, one request shape for every Bedrock model, built by translating the platform's own messages | `bedrock_client.py:118-141`; `bedrock_converse.py:1-12` **[T]** |
| The credit check before a platform-paid call | `before_platform_call(workspace_id=…, is_byok=…, purpose=…)`, called by `LLMManager` before each call, and `after_platform_call` after it | `services/credits/gate.py:74-89`; `core/llm/manager.py:758-783` **[T]** |
| Recording a call | `UsageTracker.track(workspace_id, model_id, provider, input_tokens, output_tokens, …, cache_read_tokens, cache_write_tokens)` | `core/llm/usage_tracker.py:280-298` **[T]** |
| What gets metered | Rows with `charge_class = 'customer'`; the class is stamped from the `usage_scope` in force when the row is written, which `LLMManager` sets | `services/credits/meter.py:89-96`; `core/models/core.py:250-255`; `core/llm/usage_context.py:65-111` **[T]** |
| Which models are allowed | A quarantine list and allowlist for Auto's seat, and per-workspace approval rows with roles, one of which is `coding` | `core/llm/model_policy.py:29-37`, `:63-110` **[T]** |

The relay does not go through `LLMManager`. It must call the credit check and record usage itself, inside a `usage_scope` that names the workspace from the token and a charge class **[I]**.

## 5. One relay for every provider?

**No: for Claude models only, and in two formats.** "Anthropic doesn't endorse, maintain, or audit third-party gateway products, and doesn't support routing Claude Code to non-Claude models through any gateway." **[D]** (LLM gateways).

| The install's provider | Can the relay serve the agent? | What differs |
|---|---|---|
| Anthropic (the provider's own API), platform key or a workspace's own key | Yes | Anthropic format. The relay swaps the session token for `x-api-key: <resolved key>` and passes headers, body and the event stream through unchanged. |
| Claude on Amazon Bedrock, with the API's role (`bedrock-runtime`) | Yes, if the relay speaks Bedrock's format | The worker sets `CLAUDE_CODE_USE_BEDROCK=1`, `CLAUDE_CODE_SKIP_BEDROCK_AUTH=1`, `ANTHROPIC_BEDROCK_BASE_URL` = the relay, and the token. The relay reads the model from the path, signs the request with the role and passes the binary event stream back unmodified. The role needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (the policy in the Bedrock page **[D]**). Model IDs are Bedrock's (`us.anthropic.…`, inference profiles). |
| Claude on Amazon Bedrock, Anthropic-format endpoint (Mantle) | Yes, where Mantle serves the model in the region | Either the Anthropic format (the relay forwards to `https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages`, "standard SSE streaming and the same request body shape as the Claude API" **[D]**), or the agent's Mantle variables. The role needs `bedrock-mantle:CreateInference`. Model IDs are `anthropic.`-prefixed; Fable 5.2 "isn't available" there, and "server-side tools (code execution, web search, web fetch, advisor)" are not supported **[D]** (Claude in Amazon Bedrock). |
| Anthropic format in front of `bedrock-runtime`, translated by the relay | Not recommended | The agent then "sends the full set" of Anthropic fields, which Bedrock rejects with `400`s unless the relay bridges them; the automatic retries that soften this start at 2.1.304 and later, after the bundled 2.1.301 **[D]**. |
| OpenAI, Google (Gemini), Azure OpenAI, DeepSeek, Grok, NVIDIA, Hugging Face, Cohere | No | Not Claude models. The canvas stays unavailable, as today. |
| OpenRouter's Claude models | Not confirmed | Anthropic's pages do not cover it, and say routing is unsupported beyond the formats above. Settled only by OpenRouter documenting an Anthropic Messages endpoint and a test against it. |
| Claude on Google Cloud (Vertex) | Not as core stands | The format exists (section 3.2), but core has no Vertex provider: the Google client is built with `vertexai=False` (`core/llm/clients/google_client.py:49-51`) **[T]**. |

**The API's own Bedrock call is not the agent's.** Today the API calls Bedrock's Converse API (`bedrock_client.py:135`) **[T]**. The agent sends either Bedrock InvokeModel requests or Anthropic Messages requests (section 3.2). So the relay cannot hand the agent's request to `BedrockProvider`. It forwards the agent's own bytes, signed with the same credentials that `runtime_client` resolves, to a different operation (InvokeModel) or a different endpoint (Mantle) **[I]**. Whether a role an operator wrote for Converse already allows those operations is the operator's to check; the chart's notes should list them.

## 6. What section 5.7 and US-402 to US-404 would need to change

1. **The relay has a format per provider** (5.7, US-402). The Anthropic format for the Anthropic API; for Bedrock, one of: Bedrock's own format against `bedrock-runtime` (every Claude model the account has there, the same endpoint the API uses), or the Anthropic format against Mantle (one format for both providers, fewer models and regions). Which one is a decision (section 7).
2. **Claude models only** (5.7, US-404). "The 'no model credential' failure remains only when there is neither a key nor a relay" becomes "… nor a relay that can serve a Claude model for this install". An install whose providers are all non-Claude keeps today's failure and the hidden tab.
3. **The relay's credential is the session token alone** (5.7 second bullet, section 9, US-402 third box). The agent can only send what is in its own environment (section 3.1 **[D]**), and the commands it runs start from that environment **[I]**. So any credential the agent sends must be one that agent code may hold. The session token is built for that (bound, short-lived, revoked at session end). The worker's internal token is not, so it should not be sent by the agent. "Only the worker can reach it" then rests on the network rule (5.8: worker to API, port 8000) and on the relay route being internal; the internal-token check is dropped from the relay.
4. **The token's lifetime and the session's** (US-403). `ANTHROPIC_AUTH_TOKEN` is read once, for the life of the program. Either the token lives as long as the session, with an upper bound and revocation when the session stops, or the worker uses `apiKeyHelper` to fetch a fresh one (section 3.3). The token is bound to workspace and session; a canvas session names no agent today (section 2), so "bound to … agent" needs a source or is dropped.
5. **The model is pinned** (US-402 fourth box). Claude Code sends the main model, a model for background tasks, and subagent models. The worker pins them (the SDK's `model` option, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`) and the relay checks the model named in the body (Anthropic format) or the path (Bedrock format) against the allowed list.
6. **Usage and credits are the relay's own job** (US-402 fifth box). It calls `before_platform_call` before forwarding, reads usage from the response or stream (section 3.5), and calls `UsageTracker.track` with the token's workspace inside a `usage_scope` that sets the charge class, or the row is never metered (section 4). Canvas usage is not recorded today, so this starts billing canvas sessions run on the platform's keys: the issue AGENTS.md asks for comes first.
7. **Traffic outside the relay** (US-404). The worker also sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` and turns off the WebFetch preflight, since 5.8 gives the worker no way out by default.
8. **The CI stand-in** (US-404 fourth box) speaks the format or formats chosen in point 1: server-sent events for the Anthropic format, AWS event-stream frames for Bedrock's.

## 7. Decisions for the maintainers

| # | Decision | Options |
|---|---|---|
| R-1 | Bedrock's route | Bedrock's own format to `bedrock-runtime` (two formats on the relay, every Claude model the account has) or the Anthropic format to Mantle (one format, Mantle's models and regions only) |
| R-2 | Which provider serves the agent when an install has several | The first that can serve Claude in a fixed order, or one the operator names |
| R-3 | Which models the canvas may use | A new list, the `coding` role of the workspace approval rows, or a fixed setting |
| R-4 | Canvas usage on the platform's keys | Metered as customer work like any model call, or absorbed as overhead |
| R-5 | The token's lifetime | One token for the session with an upper bound, or a helper that renews it |
| R-6 | Does the Anthropic route also replace the worker's own key where an install has one | Yes, so no model key sits in the worker at all, or only when there is no key (5.7 as written) |

## 8. Not confirmed

| Point | What would settle it |
|---|---|
| That the bundled 2.1.301 behaves as the current documentation says on each point above. The documentation is not versioned per release; points it marks as needing a later release are listed where they matter. | A canvas session in CI against the stand-in (US-404), with the request headers and body recorded. |
| The usage fields inside Bedrock's event-stream chunks for Claude, which the relay must decode to record usage on the Bedrock route. | The AWS page for Claude's streamed InvokeModel response, or a recorded response in CI. |
| Whether Claude Code's web search tool, a server-side tool, fails or is left out when the upstream is Mantle, which does not support server-side tools. | A test session against Mantle, or the Claude Code page on tool availability by provider. |
| Whether Mantle sends `ping` events during long pauses, so that the idle timeout is not tripped. | Mantle's own streaming documentation, or a recorded long response. |
| Whether `apiKeyHelper` passed through the SDK's `settings` option is honoured in a headless SDK session. | The settings reference on scopes for the SDK, or a test session. |
| Whether a role written for Converse already allows `InvokeModelWithResponseStream`. | AWS's IAM reference for Converse, then the operator's own policy. |
| OpenRouter's Claude models through the relay. | See section 5. |

## 9. Sources

Every page was fetched on 21 Nov 2026 and read in full or in the sections quoted.

- Claude Code, LLM gateways: https://code.claude.com/docs/en/llm-gateway
- Claude Code, gateway compatibility guide: https://code.claude.com/docs/en/llm-gateway-protocol
- Claude Code, connect to an LLM gateway: https://code.claude.com/docs/en/llm-gateway-connect
- Claude Code, environment variables: https://code.claude.com/docs/en/env-vars
- Claude Code, settings reference (`apiKeyHelper`): https://code.claude.com/docs/en/settings-reference
- Claude Code, Amazon Bedrock: https://code.claude.com/docs/en/amazon-bedrock
- Agent SDK, Python reference: https://code.claude.com/docs/en/agent-sdk/python
- Agent SDK, secure deployment: https://code.claude.com/docs/en/agent-sdk/secure-deployment
- Agent SDK, hosting: https://code.claude.com/docs/en/agent-sdk/hosting
- Agent SDK, quickstart: https://code.claude.com/docs/en/agent-sdk/quickstart
- Agent SDK for Python at the pinned tag: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/CHANGELOG.md, `src/claude_agent_sdk/_cli_version.py` and `src/claude_agent_sdk/_internal/transport/subprocess_cli.py`
- Claude API, streaming messages: https://platform.claude.com/docs/en/build-with-claude/streaming
- Claude in Amazon Bedrock (Mantle): https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock
- Claude on Amazon Bedrock (`bedrock-runtime`): https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
- AWS, InvokeModelWithResponseStream: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html
- AWS, Claude Messages API parameters: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

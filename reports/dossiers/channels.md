# Channels & Ambient Ingress — Phase-2 Module Dossier

| | |
|---|---|
| **Module key** | `channels` |
| **Tier / status** | deep / partial |
| **Reviewed** | 2026-07-04, against the live tree (`orchestrator/` HEAD) + live Railway Postgres (read-only) |
| **July baseline** | §3/§5, F026, F027, F029, F066, F081 (all six re-verified via `evidence/phase0-residual-map.md` §3.7: **zero commits to this surface since the July review**) |
| **Real-data sources** | live `channel_connections`, `routing_decisions`, `unrouted_events`, `routing_rules`, `workspaces.webhook_key/settings` (this review, read-only); `evidence/real-data-inventory.md`; `evidence/data/census.md` |
| **Security lens** | **deliberately excluded** — runs as the separate Opus defensive-hardening pass. Auth mechanisms are described factually where they define what the module *is*; no adversarial analysis here. |

---

## A. What it is

Channels is the platform's ambient front door: the set of lanes through which the outside world reaches Auto and the agents without opening the dashboard — chat platforms (Telegram, Slack, WhatsApp, Discord…), a generic per-workspace webhook, Playbook trigger webhooks, and GitHub PR events — plus the unified outbound sender that lets the platform speak back over the same platforms. Architecturally it is three layers that arrived in different eras and only partially met: a modern stateless **driver registry** (PRD-008-A.4: verify/send/install_webhook/start_polling per platform), a legacy **adapter fleet** (PRD-55: 11 in-process polling bots, 7 of which have no driver and no callers), and the **UniversalRouter** (PRD-50: a 7-tier envelope-to-agent resolver with real decision telemetry). By the North Star this module is *the* precondition for "Auto operates ambiently": today it is the largest capability gap left standing — the July verdict, reconfirmed by Phase 0 (`phase0-residual-map.md:636-639`) and now by production data showing the one live channel connection cannot deliver a reply.

---

## B. What it does — real implementation and data path

### B1. Inbound lanes (four real, one dead, one delegated)

| Lane | Endpoint | Auth model | What actually happens |
|---|---|---|---|
| Workspace webhook | `POST /api/webhooks/ws/{workspace_key}` (`api/webhooks.py:308`) | URL-as-secret (`workspaces.webhook_key`; all 21 live workspaces have one — live query) + optional HMAC that is skipped when no signature header is present (`webhooks.py:59,67-69`) | body-parse → Slack `url_verification` echo (`:352-353`) → platform detect telegram/slack/whatsapp/twilio (`:91-117`) → persist default chat ids (`:379-382`) → `WebhookIngestor` envelope (`:385-389`) → keyword gate to Auto (`:392-435`) → `UniversalRouter.route` (`:437-443`) → synchronous `AgentFactory.execute_with_prompt` (`:624-643`) → fire-and-forget reply via `channels.sender` gated on the legacy integrations bag (`:417,447,478,499,528,567,590`) |
| Playbook webhook | `POST /api/webhooks/recipe/{webhook_id}` (`api/workflow_recipes.py:1752`) | URL-as-secret (`schedule_config.webhook_id`) + same optional-HMAC pattern (`:1800-1818`) | recipe lookup → concurrency guard (429 at capacity, `:1823-1838`) → `RecipeExecution` row → PlaybookEngine `execute_direct` |
| GitHub PR events | `POST /api/github/webhook` (`api/github_webhooks.py:37`) | **mandatory** HMAC — 500 without secret, 401 without/with-bad signature (`:60-70`) | matches PR opened/synchronize/reopened → finds the "PR Code Review" workflow → **returns `workflow_execution_disabled`** — the legacy execute path was removed by PRD-125/172 and the mission migration never happened (`:129-149`). Authenticated, honest, and functionally dead. |
| Polling adapters | in-process, started at boot (`main.py:468-472`) | platform creds from `channel_connections.config` | `ChannelManager.start_all` loads `status='active' AND mode='polling'` rows only (`channels/manager.py:48-55`) and instantiates **legacy adapters** from `_ADAPTER_MAP` (`:139-151`); each pipes messages through `BaseChannelAdapter.handle_message`: envelope → UniversalRouter → `execute_with_prompt` → `send_message` → activity stats → primitive heartbeat (`channels/base.py:112-205`) |
| Composio triggers | `POST /api/composio/webhook` | (owned by the `composio-integration` dossier) | only `JIRA_`-prefixed triggers route to an ingestor; everything else — including Gmail — dead-letters as `UnroutedEvent` (`api/composio.py:724-746`, F066 residual) |
| Twilio SMS | *(no endpoint of its own)* | — | `_detect_platform` recognizes Twilio payloads and extracts reply context (`webhooks.py:113-116,143-146`) but **no `twilio` driver is registered** (`channels/drivers/__init__.py:62-66`), so `send_to_channel` returns `UnknownPlatform` — inbound half-recognized, reply impossible |

### B2. The driver registry (the good bones)

`channels/drivers/` is a clean, stateless, self-registering abstraction (`drivers/__init__.py:28-53`; ABC at `drivers/base.py:90-213` with `verify / send / install_webhook / uninstall_webhook / start_polling / is_polling_running`, typed `VerifyResult`/`SendResult` with `retryable` semantics). Five drivers registered:

- **telegram** — the only full citizen: real `getMe` verify with the missing-`bot_id:`-prefix failure mode explained to the user (`drivers/telegram.py:64-99`), `sendMessage` send, `setWebhook`/`deleteWebhook`/`getWebhookInfo` (`:160-222`), plus a polling mode (see C2-1 for what's wrong with it).
- **slack** — webhook-only; `auth.test` verify, `chat.postMessage` send; `install_webhook` is honestly a paste-this-URL no-op because Slack has no setWebhook API (`drivers/slack.py:124-135`).
- **whatsapp** — Meta Cloud API, webhook-only; Graph verify + send (`drivers/whatsapp.py`); the GET `hub.challenge` handshake is answered by the workspace webhook (accepting any verify_token, `webhooks.py:292-303`).
- **discord** — **outbound-only by declared design**: "The gateway adapter is intentionally not wired up here" (`drivers/discord.py:8-14,40-42`).
- **webhook** — generic **outbound** POST for Zapier/n8n-style targets (`drivers/webhook.py:1-11`); inbound generic traffic is the workspace webhook's job.

So the honest platform matrix today: inbound+outbound = Telegram (webhook mode), Slack, WhatsApp; outbound-only = Discord, generic webhook; nothing else exists.

### B3. Outbound — one sender, three consumers, no agent tool

`channels/sender.py:156-220` (`send_to_channel`) is the single egress choke point: loads the `channel_connections` row (`:48-78`), falls back to the legacy `workspace.settings.integrations` bag (Telegram/Slack only, `:81-108`), resolves target (explicit → row metadata → legacy default chat id, `:111-149`), delegates to the driver, never raises. Consumers: `_deliver_reply` in webhooks (`api/webhooks.py:198`), `notification_service` as a thin façade (`core/services/notification_service.py:55`), and the destinations dispatcher (`services/destinations/dispatcher.py:134`). **Still no `platform_send_channel_message` tool** — grep of `modules/tools/` for the sender returns zero hits; the July write-verb silo stands: Auto can connect, configure, start and stop a channel but cannot *say anything* through one.

### B4. Routing — UniversalRouter, the real decision engine

`core/routing/engine.py` runs a 7-tier chain (`:79-163`): Tier 0 explicit override → Tier 1 cache → Tier 2a `routing_rules` source-pattern match (**exact string equality**, `:226`) → Tier 2b Jira TriggerSubscription (`:250-308`) → Tier 2.5 semantic similarity over agent embeddings with direct-route threshold + LLM-candidate handoff (`:362-448`) → Tier 2c IntentClassifier keyword rules (`:314-356`) → Tier 3 LLM classification with an id=0 "Auto" sentinel that returns `route_type="orchestrate"` (`:454-632`, resilient JSON/name parser `:771-877`). Every decision is persisted to `routing_decisions` (`:905-929`); every exhaustion is dead-lettered to `unrouted_events` (`:883-899`). This is a genuinely decent router — its problems are what it's *fed* (C2-7) and who *reads* its outputs (C2-9).

### B5. Agent-facing tool surface — lifecycle yes, voice no

PRD-143 S10 gave Auto five tools (`modules/tools/discovery/actions_channels.py`, registered at `platform_actions.py:40,77`): `platform_list_channels`, `platform_connect_channel`, `platform_configure_channel`, `platform_start_channel`, `platform_stop_channel`. Connect delegates to the same `connect_channel_for_workspace` flow the dashboard uses (`handlers_channels.py:78-88` → `api/channels.py:326-476`) — a genuinely good single-implementation pattern (verify → persist mode → install_webhook or start_polling → status). `workspace_id` always comes from executor context, never params (`handlers_channels.py:7-8`).

### B6. Boot lifecycle

`main.py:468-472`: `start_all()` under `CHANNELS_ENABLED` (default true, `config.py:750`) — positioned after the boot leader-lock closes and outside the unified-scheduler flock, i.e. it runs in **all four uvicorn workers** (F027, re-verified by Phase 0 at `phase0-residual-map.md:401-403`). `stop_all` on shutdown (`main.py:611-618`). Status honesty is handled well at read time: `GET /api/channels` reconciles DB `status` against actually-running adapters and repairs drift rows (`api/channels.py:233-303`).

### B7. Frontend surface

One settings tab: `frontend/components/settings/ChannelsSettingsTab.tsx` (434 lines), mounted solely from `SettingsPanel.tsx:80`. It hardcodes an 11-platform `PLATFORMS` array with per-platform credential forms and help links (`ChannelsSettingsTab.tsx:26-137`) — it does **not** call the backend's driver-introspection endpoint `GET /api/channels/platforms` (`api/channels.py:588-612`), which has zero frontend callers (repo-wide grep). Channel analytics (`/api/channels/analytics`, reading `routing_decisions` + `channel_connections`, `api/channels.py:721-768`) is consumed by `frontend/components/analytics/analytics-overview.tsx:46`.

---

## C. Honest quality — inspected against production data

### C1. What the real data says (live Railway Postgres, read-only, 2026-07-04)

| Measurement | Value | Meaning |
|---|---|---|
| `channel_connections` rows | **2** (both Telegram, webhook-mode; 1 active / 1 inactive; across 2 of 21 workspaces) | after ~5 months in production, the chat-channel plane has two rows |
| `message_count` / `last_activity_at` | **0 / never** on both rows | the adapter-tracked pipeline (`base.py:224-242`) has never processed a single message; only the webhook lane, which doesn't update these fields, has carried traffic |
| newest connection created | 2026-05-22 | nobody has connected a channel in six weeks |
| rows with `default_agent_id` | **0** | the severed per-channel pinning (F029) has never even been exercised — nothing to break yet |
| `routing_decisions` by source | chatbot 387 (→06-10) · **webhook 129 (→05-22)** · jira_trigger 26 (→02-13) | the workspace webhook lane genuinely worked — 129 routed requests — then went quiet six weeks ago; **no telegram/slack/discord/whatsapp source was ever logged** (see C2-7 for why that's structural, not just usage) |
| `unrouted_events` | 135 lifetime: chatbot 108, webhook 12, composio.trigger/expired 13, test 2 | the dead-letter path works; 13 Composio events (incl. trigger messages) dead-lettered exactly as F066 predicts — and **nothing reads this table** (C2-9) |
| `routing_rules` | 92 rows, **only 2 active** — both `source_pattern=NULL` keyword rules; the 90 inactive rows carry glob-style patterns (`route-69…-*`, `rule-ml…-*`, `test-*`) | the rules surface produced 90 rules that Tier 2a could never match even if re-activated — it does exact equality against `ChannelSource` enum values (`engine.py:226`), and no code globs; a rule-creation surface that manufactures unmatchable rules was used, produced garbage, and was abandoned |
| workspace of the **one active** channel has a legacy `integrations` bag? | **no** (only 1 of 21 workspaces has one, and it isn't this one) | **the only live channel connection in production cannot reply** — every reply site gates on `if platform and integrations:` (F026), which is False for it; inbound processes, the agent executes and burns tokens, the response is silently dropped, and the endpoint reports `reply_delivered: true` anyway (C2-4) |

The cross-surface story: this module is not "partially adopted" — it is **effectively unused**, and the one attempt at using it is broken at the last mile by a gate that Phase 0 already showed protects nothing (`_deliver_reply` ignores the `integrations` param entirely and reads `channel_connections` itself — `webhooks.py:174-181`; fix is a one-line condition change).

### C2. Confirmed defects (each verified against the live tree this review)

**Reconfirmed July findings — all six, zero fix commits (Phase-0 §3.7):**

1. **F026 (NOT DONE)** — replies gated on the legacy `workspace.settings.integrations` bag at `webhooks.py:417,447,478,499,528,567,590` while delivery itself reads `channel_connections` (`:174-181`); nothing writes the bag on new-style connect. Production impact now proven: the single active channel's workspace has no bag → its replies drop (C1).
2. **F027 (NOT DONE)** — `start_all` runs in every uvicorn worker (`main.py:468-472`; no gating in `manager.py:32-72`). Honest nuance the July text lacked: with **zero polling-mode rows in production** (C1), the Telegram-409 storm is currently *latent* — it fires the day anyone connects a polling channel, which is also the day inbound polling matters (see 
	defect 7).
3. **F029 (NOT DONE)** — `channel_connections.default_agent_id` written (`api/channels.py:319,355-363,501-503`; `handlers_channels.py:85,117-119`) and advertised to the LLM (`actions_channels.py:64-67,105-107`), read by **no routing path**: webhook routing uses UniversalRouter or `get_default_agent_id(db, workspace_id)` — the workspace's Auto agent by slug (`webhooks.py:551-561`); adapters never touch it.
4. **Reply-delivery lying** (July F-low, still live) — `reply_delivered: platform is not None and bool(integrations)` is returned before the fire-and-forget task even runs, and regardless of `SendResult.ok` (`webhooks.py:431,493,582`). Combined with F026 the endpoint reports success for replies that were never attempted.
5. **F066 (NOT DONE)** — no inbound email channel anywhere: `_SUPPORTED_PLATFORMS` has no email entry (`api/channels.py:49-53`), no driver, no adapter; Gmail Composio triggers dead-letter (13 live dead-letters, C1). The refund-email journey — the July merchant-scenario's canonical ambient task — still has no autonomous path.
6. **F081 (NOT DONE)** — the seven driverless legacy adapters are byte-identical at 1,589 lines (teams/google_chat/signal/imessage/irc/matrix/line under `channels/`), instantiable only via `_ADAPTER_MAP` (`manager.py:139-151`); `_ping_platform_legacy` still defined with zero callers (`api/channels.py:143-196`). The never-authored W14/PRD-184 kill list remains never-authored.

**New defects this review found (not in the July register):**

7. **The driver polling path is a message black hole.** `TelegramDriver.start_polling` builds a python-telegram-bot `Application`, initializes, starts, and begins `start_polling` — **with zero handlers registered** (`drivers/telegram.py:256-267`; compare the legacy adapter, which registers command + message handlers and pipes into the routing pipeline, `channels/telegram_adapter.py:38-48`). Updates are fetched, offset-acknowledged, and discarded. Consequence: connect a Telegram channel in polling mode and every inbound message is silently consumed until the next process restart — after which `start_all` switches the same row to the *legacy* adapter, which does work. Two different polling implementations for the same row depending on lifecycle moment, one of them a black hole. (It also calls `deleteWebhook` first, `drivers/telegram.py:254`, so it actively tears down a working webhook registration on its way to discarding messages.)
8. **The advertised-platform surface is broken three different ways.** (a) The frontend hardcodes 11 platforms instead of calling the honest `GET /api/channels/platforms` introspection endpoint (zero frontend callers). (b) For teams/signal/line the hardcoded field names match `_REQUIRED_CONFIG` (`api/channels.py:211-223`), so connect succeeds validation, then `get_driver` raises `UnknownPlatform` and a **dead row is saved** with "No driver registered … row saved" (`:374-386`). (c) For google_chat/imessage/matrix/irc the frontend field names don't even match the backend's required keys (`service_account_json` vs `service_account_key`; `server_url/password` vs `apple_id`; `homeserver` vs `homeserver_url`; `channels` vs `channel` — `ChannelsSettingsTab.tsx:75-131` vs `api/channels.py:216-220`), so a fully-filled form 400s. And (d) the tool schema recites all 12 platforms to the LLM (`actions_channels.py:56-58`), so Auto itself can be talked into creating dead rows. The platform claims 12, drives 5, and 4 of the advertised forms cannot even fail correctly.
9. **The learning outputs are write-only.** `unrouted_events` (135 rows) has no reader anywhere in `api/` (repo grep: only the Composio writer) — no UI, no replay, no triage. `routing_decisions` powers one analytics widget (`analytics-overview.tsx:46`) but no feedback loop adjusts rules, cache, or agent embeddings from outcomes. The router logs diligently into the void.
10. **Channel conversations have no memory.** Both inbound execution paths are stateless one-shots: the webhook lane calls `execute_with_prompt(agent, prompt, context={source, workspace_id})` per message (`webhooks.py:624-643`), the adapter lane the same plus `connection_id` (`base.py:169-179`). No mapping of platform `chat_id` → `chats`/`messages`, no thread continuity, no session (grep of `agent_factory.py` for history/conversation handling: nothing). A Telegram user asking a follow-up question is talking to an amnesiac — while the dashboard chat right next to it keeps full history. This is the single largest quality gap between "Auto in chat" and "Auto on a channel," and it is invisible in any advertised feature list.
11. **Channel-parity fork on capability.** Inbound channel users reach platform tools only via `AutoBrain._match_platform_query` — a word-boundary phrase-match dictionary (`webhooks.py:392-437`; `consumers/chatbot/auto.py:953-961`) that routes to the `auto-cto` agent on keyword hit; everything else goes through UniversalRouter to a plain agent execution. No complexity assessment, no mission escalation, no plan mode, no approval cards — the chat UI's judgment layer simply doesn't exist on this lane (July §fix-5, unchanged).
12. **Self-description vs reality on credentials.** The model comments `config = Column(JSON …)  # encrypted credentials` (`core/models/channels.py:26`); the write path stores plaintext JSON (`_json.dumps(config)`, `api/channels.py:362`). Whatever the hardening pass decides, the code's claim about itself is false today. Similarly, Slack's **required** `signing_secret` (`drivers/slack.py:41`) is collected and stored but never used to verify inbound events — `_verify_webhook_signature` checks GitHub/Composio/generic headers only, against the workspace/global secret (`webhooks.py:62-66`), and Slack's `X-Slack-Signature` scheme isn't implemented at all.
13. **Ingress telemetry collapses platform identity.** `WebhookIngestor` hardcodes `source=ChannelSource.WEBHOOK` (`ingestors/webhook.py:98`) even when `_detect_platform` identified Telegram/Slack/WhatsApp one call earlier (`webhooks.py:369`). Consequences: `routing_decisions.source` can never say "telegram" for webhook-delivered Telegram traffic (why C1 shows zero telegram-source decisions), source-pattern rules can't target platforms, and the routing cache keys chat and curl traffic identically.
14. **Synchronous execution with no dedup.** The workspace webhook executes the agent inside the HTTP request (`webhooks.py:470-475`). A slow agent run means the platform's retry policy (Telegram redelivery, Slack's 3-second `x-slack-retry-num` retries) delivers the same event again — and nothing dedups `update_id`/`event_id` anywhere on the lane, so retries become duplicate agent executions, each burning tokens.

### C3. What is genuinely good (credit where due)

- **The driver abstraction is the right shape** — stateless, typed results with retryability, per-platform quirks (Slack's no-setWebhook reality, Telegram's token-prefix failure mode) handled honestly where they belong (`drivers/base.py`, `drivers/slack.py:124-135`, `drivers/telegram.py:69-77`).
- **One connect flow** shared by dashboard and tool (`connect_channel_for_workspace`) — the exact anti-drift pattern the platform preaches (`api/channels.py:334-340`).
- **UniversalRouter is real engineering**: tiered fallback, request-local semantic candidates to avoid async races (`engine.py:127-129`), an Auto-sentinel escape hatch, a resilient LLM-answer parser, and faithful decision/dead-letter telemetry. 542 logged decisions prove it ran.
- **Status honesty with drift-repair** at read time (`api/channels.py:233-303`) — the dashboard reflects running reality, and repairs the DB as a side effect.
- **Tests exist and are substantive**: 1,526 lines across the adapter contract suite (997), driver tests (338), and sender tests (191).
- **The GitHub lane refuses to pretend** — mandatory signatures and an explicit "execution disabled, awaiting mission migration" answer instead of a fake success (`github_webhooks.py:129-149`).

### C4. Maturity score: **2 / 5**

Justification: the architecture (drivers + single sender + tiered router + lifecycle tools + contract tests) is a legitimate skeleton that deserves a 3-4 — but maturity is judged on *real behaviour*, and production shows ~zero working end-to-end use: two connections ever, zero adapter-processed messages, replies broken on the only active channel with the endpoint reporting success, six July findings untouched, a newly found inbound black hole on the driver polling path, a 12-advertised/5-real/4-can't-even-fail-right platform matrix, no conversation memory, and no email. A 1 would deny the good bones; a 3 would require at least one channel that verifiably works round-trip in production. It is a 2: **correct architecture, broken last mile, unused in practice.**

---

## D. Competitive teardown

Four reference points, chosen for the three distinct jobs this module conflates: an *agent gateway* (OpenClaw), an *omnichannel support desk* (Chatwoot, Intercom Fin), and a *channel-connector platform* (Microsoft's bot stack).

### D1. OpenClaw — the open-source agent-gateway shape Automatos is reaching for

OpenClaw is a self-hosted gateway that connects a personal AI assistant to the channels you already use — **Slack, Discord, Telegram, WhatsApp, Matrix, iMessage, Google Chat, Signal, Mattermost built-in, plus plugin channels (Teams, IRC, Feishu, LINE, Nostr, Zalo, Nextcloud Talk, Synology Chat…)** — with the Gateway as "the single source of truth for sessions, routing, and channel connections" ([github.com/openclaw/openclaw](https://github.com/openclaw/openclaw), [docs.openclaw.ai](https://docs.openclaw.ai/)). What it does better, specifically:

- **Every listed channel actually works** — the channel list is the implemented list ([docs.openclaw.ai/gateway/config-channels](https://docs.openclaw.ai/gateway/config-channels)). Automatos advertises 12 and drives 5.
- **Sessions are first-class**: per-thread conversations and thread-bound routing (e.g. Discord), with the gateway owning the session store ([docs.openclaw.ai/concepts/multi-agent](https://docs.openclaw.ai/concepts/multi-agent)). Automatos channel turns are stateless one-shots (C2-10).
- **An inbound-sender trust model exists**: unknown DM senders get a one-time pairing code requiring owner approval, or allowlists, with expiry and caps ([docs.openclaw.ai/gateway/config-channels](https://docs.openclaw.ai/gateway/config-channels)). Automatos maps every inbound sender to `RequestUser(auth_type="webhook")` with no identity linkage at all (`ingestors/webhook.py:101`) — whoever messages the bot is the workspace.

Where Automatos stands: ahead on *platform-native* integration (the channel lane rides the same tool loop, workspace tenancy, and heartbeat plane as everything else — OpenClaw is single-owner personal software, not multi-tenant SaaS), behind on the entire channel experience itself.

### D2. Chatwoot — the proven OSS omnichannel plumbing (MIT)

Chatwoot is the open-source omnichannel desk: **website live-chat, email, WhatsApp Cloud, Telegram, LINE, Twilio SMS, Facebook/Instagram, plus a generic API channel**, with agent-bot webhooks (now with dedicated signing secrets and signed headers) and a REST/webhook API for custom integrations; MIT-licensed and self-hostable ([github.com/chatwoot/chatwoot](https://github.com/chatwoot/chatwoot), [chatwoot.com](https://www.chatwoot.com/)). What it does better, specifically:

- **Inbound email as a first-class channel** — forward a mailbox and every thread becomes a conversation. This is exactly F066, solved, in software Gerard could run tonight.
- **Conversation persistence, assignment, and continuity** per contact across channels — the contact/conversation model Automatos lacks entirely on this lane.
- **An agent-bot seam designed for exactly this integration**: hand a conversation to a bot via signed webhooks, escalate to humans on demand — i.e., Auto could *be* the Chatwoot agent-bot and inherit ten channels plus an inbox UI.
- Its own AI layer (Captain) answers from a knowledge base and assists agents ([chatwoot.com](https://www.chatwoot.com/)) — competitive pressure, but also proof the agent-bot seam is load-bearing.

Where Automatos stands: Chatwoot has no notion of the agents *doing work* (missions, tools, deliverables) — it is a conversation surface. That is precisely why it composes rather than competes: plumbing there, brain here.

### D3. Intercom Fin — the quality bar for AI-on-channels

Fin resolves customer queries end-to-end **across live chat, email, WhatsApp, SMS, phone (Fin Voice), and social**, from one inbox, with full email-thread context and phishing/spam filtering on the email lane ([intercom.com](https://www.intercom.com/), [Fin AI Agent explained](https://www.intercom.com/help/en/articles/7120684-fin-ai-agent-explained)). Salesforce agreed to acquire Fin for ~$3.6B in June 2026 ([salesforce.com press release](https://www.salesforce.com/news/press-releases/2026/06/15/salesforce-signs-definitive-agreement-to-acquire-fin/)) — the market's statement of what channel-resident agents are worth. What it does better: everything on this module's ambition list — omnichannel continuity, email-native context, voice, human-handoff choreography — as polished product. Where Automatos stands: not competing on the same field (Fin is support-vertical SaaS; Automatos is an operations platform whose agents also do the work behind the conversation), but Fin defines the *client-quality* expectation any Automatos channel must meet: a channel that can't reply (C1) is unshippable by this bar.

### D4. Microsoft Bot Framework → M365 Agents SDK — the cautionary architecture

The Bot Framework SDK — the original "one bot, many channels" connector architecture Automatos's adapter fleet resembles — is being retired (final long-term support ends **December 31, 2025**), replaced by the Microsoft 365 Agents SDK, which keeps the multi-channel delivery (Teams, web, Slack, Twilio, Messenger…) via Azure AI Bot Service channels but rebuilds around bring-your-own AI/orchestration ([learn.microsoft.com migration guidance](https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/bf-migration-guidance), [botframework-sdk repo](https://github.com/microsoft/botframework-sdk), [Teams SDK guidance Fall 2025](https://www.voitanos.io/blog/microsoft-teams-sdk-evolution-2025/)). Lessons that transfer, specifically: (a) the channel-connector layer survives while bot logic gets rebuilt — keep the seam between the two clean (Automatos's driver registry is that seam; the legacy adapters violate it); (b) even Microsoft couldn't sustain a hand-rolled adapter per platform and pushed connectivity to a service; (c) for Teams specifically, riding an official connector beats a homegrown `teams_adapter.py` that has never had a driver.

**Net competitive position:** nobody in this set combines Automatos's actual differentiator — channels wired into a tenant-scoped tool-using agent platform that *does the work* — but every one of them beats Automatos at the channel layer itself. The gap is not conceptual; it is that the last mile was never finished and never used.

---

## E. Build / extend / adopt / replace — verdict

**EXTEND the in-house core; ADOPT for email/support-desk breadth; KILL the fiction.** Three-part verdict:

1. **Extend (the decision for the core, and it must be earned — here is why it wins):** the driver registry + unified sender + workspace-webhook lane is ~1,500 lines of correct, tested, platform-native code whose defects are wiring-level (a one-line reply gate, a missing handler registration, an unread column), not architectural. Every adopt candidate would still need the exact integration work that is broken today (tenancy mapping, agent execution, reply routing) — adopting Chatwoot *wholesale* as the channel layer would replace the 5 drivers but add a second product to operate (Rails + Postgres + Redis), a second conversation store to reconcile with `chats`/`messages`, and a second identity model, while still leaving F029/C2-10/C2-11 to solve on the Automatos side. The fix-cost of the current core (days — see J) is an order of magnitude below the integration cost of any replacement. Nothing external wins at replacing *these five lanes*.
2. **Adopt where the gap is a product, not a feature — inbound email (F066):** do not hand-build IMAP/SMTP threading, bounce handling, and spam filtering. Two concrete adopt shapes, in preference order: (a) **route the already-integrated Composio Gmail/Outlook triggers** into UniversalRouter instead of the `JIRA_`-only dead-letter (`api/composio.py:724-746`) — zero new vendors, reuses the trigger plumbing this platform already pays for, and turns 13 dead-letters into the refund-email journey; (b) if/when clients need a human-inbox alongside Auto, **adopt Chatwoot (MIT, self-host)** as the conversation desk and register Auto as its agent-bot via the signed webhook seam ([github.com/chatwoot/chatwoot](https://github.com/chatwoot/chatwoot)) — that one integration buys email + SMS + FB/IG + LINE + a support UI without writing another adapter, at infra-only cost. (a) is a this-quarter fix; (b) is a when-a-client-asks decision.
3. **Kill/replace the fiction:** delete the seven driverless adapters (1,589 lines), `_ping_platform_legacy`, and the four legacy driver-platform adapters *after* porting real polling into the driver (C2-7/F081 — the legacy Telegram adapter is currently the only working polling implementation, so the port precedes the delete); collapse the advertised list to `list_platforms()` everywhere (frontend, `_SUPPORTED_PLATFORMS`, tool schema); do not build Teams/Signal/iMessage/IRC/Matrix/LINE/Google-Chat drivers until a client contract names one — and for Teams specifically, prefer the M365 Agents SDK channel route over a hand-rolled driver when that day comes (D4).

Judged by the North Star: extending is not sentiment — the channel plane's value is that an inbound Telegram message lands in *the same* agent loop, tenancy, memory, and (eventually) policy plane as everything else. Every adopt option preserves less of that than finishing the wiring does.

---

## G. Quality metric

**Primary metric: Channel Round-Trip Success Rate (CRSR)** — of inbound channel messages, the % that produce a *delivered* reply (SendResult.ok) within a latency budget (p95 ≤ 20s). This one number captures ingest, routing, execution, and egress honesty; it is the module's "does it work at all" heartbeat and directly measures the North-Star property (a client can talk to Auto ambiently and get an answer).

- **Today's value: 0%** — evidence chain: the one active connection's replies are dropped by the F026 gate (C1), and *measured* CRSR is not even computable because sends are unlogged (the sender returns `SendResult` with latency but persists nothing).
- **Instrumentation to build (the metric's prerequisite):** a `channel_message_log` (or reuse of the `tool_execution_logs` pattern) writing one row per inbound message (platform, connection_id, dedup key, route tier hit, agent, latency) and one per outbound send (target, ok, error, latency) — this simultaneously fixes C2-14's dedup gap (unique index on platform+update_id).

**Secondary metrics** (all feed the T3 harness):
- **Routing accuracy** — labeled-set accuracy of UniversalRouter decisions; `routing_decisions` already stores content+decision (542 rows), so a first golden set is free to curate; track per-tier hit distribution to watch cache/semantic health.
- **Unrouted rate** — unrouted_events ÷ inbound (today 135 lifetime, denominators unknown — see instrumentation).
- **Advertised-honesty ratio** — platforms working end-to-end ÷ platforms advertised. Today **5/12 advertised (and 3/12 round-trip capable)**; target 1.0 by definition (advertise only what works).
- **Conversation-continuity** (post-C2-10 fix) — % of multi-turn channel exchanges where turn N+1 correctly resolves references to turn N (LLM-judged on sampled transcripts).

---

## H. Cost note (informational)

Per inbound channel message on the webhook lane: 0–1 router LLM calls (Tier 3 fires only on cache/rule/semantic miss; prompt = agent roster + message, ~0.5–1.5k tokens — sub-cent on the mini-class models the workspace LLM manager selects) + one full `execute_with_prompt` agent run, which dominates: cost is whatever the agent's context assembly and tool loop cost, identical to a chat turn (see `llm-core`/`context-assembly` dossiers; `llm_usage` shows the plane is metered, 31k rows). Structural cost notes: retries without dedup (C2-14) multiply the *agent* cost, not the routing cost — the expensive half is the unguarded one; the F026 bug means tokens are currently spent producing replies that are discarded (execution runs, delivery is gated) — a literal pay-for-nothing path; polling adapters cost one long-poll connection per polling row per worker (×4 due to F027) but there are zero polling rows today. Infra cost of the module at current usage: negligible.

---

## I. UX / surface

Today's surface is one hardcoded settings tab (B7) plus one analytics widget — no ambient presence anywhere else. Concrete changes, in order:

1. **Make the connect tab honest and driver-driven.** Render platforms from `GET /api/channels/platforms` (the endpoint exists and returns required/optional config field specs and modes — `api/channels.py:588-612`); delete the hardcoded `PLATFORMS` array and its four field-name drifts (C2-8). Platforms without drivers simply don't render. Show `last_error`/`last_verified` inline (the columns are maintained; the tab already gets them).
2. **Channel conversations belong in the chat surface, not settings.** Once C2-10 lands (chat_id → thread mapping), a Telegram conversation should appear in the same conversation list as dashboard chats, badged with the channel icon — one Auto, many doors, one transcript history. This is the Command Center-adjacent change with the highest client-perceived value.
3. **Ambient ingress in Command Center.** A small "Doors" tile: per-lane status (webhook key set / channels active / last inbound / last delivered reply), fed by the CRSR instrumentation (G) rather than a new bespoke endpoint — consistent with the W10 honest-tiles pattern.
4. **An unrouted-events inbox.** 135 dead-letters no one can see (C2-9). A simple list (source, content preview, reason, timestamp) with "route to agent…" and "create rule" actions turns the dead-letter table into the routing plane's training UI — and is the natural home for the Composio trigger dead-letters until F066's fix lands.
5. **Kill the routing-rules UI as it stands** — it manufactured 90 unmatchable rules (C1). Either constrain source patterns to a dropdown of real `ChannelSource` values (+ document that matching is exact) or fold rule creation into the unrouted-inbox flow above, where a rule is created from a concrete missed example.
6. **Reply-status truthfulness**: wherever a channel send is surfaced (webhook response, future transcript view), show delivered/failed from `SendResult`, never the current pre-computed `reply_delivered` boolean (C2-4).

---

## J. Upgrade path (impact × effort, judged by North-Star lift)

Ordered; items 1–4 are days-scale and unblock "Auto answers on a channel, remembers the conversation, and we can prove it."

| # | Change | Effort | North-Star impact | Grounding |
|---|---|---|---|---|
| 1 | **Fix the reply gate**: key replies off `platform` + a `channel_connections` row (or nothing — `_deliver_reply` already resolves creds itself); delete the `integrations` param | **S** (hours) | **Critical** — un-breaks the only live channel; every other channel investment is worthless while replies drop | `webhooks.py:417,447,478,499,528,567,590`; C1 |
| 2 | **Wire `default_agent_id` into the webhook route** (Tier-0-style override before UniversalRouter) and **propagate detected platform into `envelope.source`** | **S** | High — per-channel agent pinning starts existing (F029); telemetry/rules/cache can finally distinguish Telegram from curl (C2-13) | `webhooks.py:384-389,551-561`; `ingestors/webhook.py:98` |
| 3 | **Close the polling black hole + leader-gate `start_all`**: port the legacy Telegram handler pipeline into `TelegramDriver.start_polling` (or have it delegate to the adapter), then gate `start_all` behind the boot leader lock / scheduler flock | **S/M** | High — inbound polling stops eating messages (C2-7); F027's 409 storm dies before it ever fires | `drivers/telegram.py:256-267`; `telegram_adapter.py:38-48`; `main.py:468-472` |
| 4 | **Honesty pass on the advertised surface**: frontend renders from `/platforms`; `_SUPPORTED_PLATFORMS` and the tool schema derive from `list_platforms()`; delete the 4 mismatched forms | **S** | Medium-high — 12→5 fiction ends; Auto stops being able to create dead rows; client trust (C2-8) | `ChannelsSettingsTab.tsx:26-137`; `actions_channels.py:56-58`; `api/channels.py:49-53` |
| 5 | **`platform_send_channel_message` tool** wrapping `channels/sender` (+ its read twin `platform_get_channel_history` once #7 exists) | **S** | High — closes the July write-verb silo; heartbeats/missions/board can *speak* to the owner's channels through the tool plane instead of only the notification façade | `channels/sender.py:156`; B3 |
| 6 | **Channel message log + dedup**: one table, written on ingest (unique platform+event id) and on send (SendResult); reject duplicate deliveries; return fast-ack + background execution for slow agents | **M** | High — retries stop double-executing (C2-14); CRSR becomes measurable (G); the honest `reply_delivered` comes free | `webhooks.py:470-475` |
| 7 | **Conversation continuity**: map (connection_id, chat_id) → a persistent chat thread; route channel turns through the chat service with history instead of bare `execute_with_prompt` — this also collapses the capability fork (C2-11) since the chat path carries complexity assessment and approval cards | **M/L** | **Highest sustained impact** — channel Auto stops being an amnesiac; parity of judgment (ask-vs-act) across doors; the OpenClaw/Fin table stake (D1/D3) | `webhooks.py:624-643`; `base.py:169-179` |
| 8 | **Inbound email via Composio triggers**: route non-`JIRA_` triggers (Gmail first) through UniversalRouter with a trigger→agent mapping; drain the dead-letter | **M** | High — the refund-email journey gets its autonomous path (F066) with zero new vendors; 13 live dead-letters become work items | `api/composio.py:724-746`; C1 |
| 9 | **Kill list**: delete `_ping_platform_legacy` now; delete the 7 driverless adapters (1,589 lines) with #4; delete the 4 legacy driver-platform adapters + `_ADAPTER_MAP` once #3 lands (manager shrinks to driver lifecycle) | **S/M** | Medium — honesty + maintenance; ~2.4k lines of fiction gone (F081) | `api/channels.py:143`; `manager.py:139-151` |
| 10 | **Adopt decision point — Chatwoot as conversation desk** (Auto as agent-bot) when a client needs human-inbox + email/SMS/social breadth; revisit Teams via M365 Agents SDK channels if a client contract names it | **L** (integration, not build) | Conditional — breadth without adapter-building, but only worth the second-system cost on real demand (E2/D4) | [chatwoot GitHub](https://github.com/chatwoot/chatwoot); [M365 Agents SDK](https://learn.microsoft.com/en-us/microsoft-365/agents-sdk/bf-migration-guidance) |

**Explicitly not recommended:** building drivers for the seven advertised-but-driverless platforms ahead of demand; building IMAP/SMTP email handling in-house (adopt shapes exist, E2); resurrecting the GitHub PR lane before the mission-trigger seam it honestly waits for exists (`github_webhooks.py:129-149`); investing in the routing-rules UI as-is (C1 shows it produces garbage — fix the producer or fold it into the unrouted inbox, I5).

---

*Dossier complete. Section F (enterprise/security/hardening) intentionally absent — separate Opus pass per the brief.*

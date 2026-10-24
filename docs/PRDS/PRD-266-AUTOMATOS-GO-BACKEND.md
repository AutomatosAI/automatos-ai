# PRD-266: Automatos Go backend: pairing, discovery, push and the store duties

**Status:** Draft for Gerard, 20 Oct 2026. Every product decision below was taken by Gerard in the 20 Oct mobile Q&A (60 decisions, `<workspace>/reports/mobile-review-2026-10-20/decisions.md`; numbers in brackets, e.g. [D16]). · **Owner:** Gerard · **Written:** 20 Oct 2026, grounded on `origin/main @ 630fc1c85`.
**Source:** `<workspace>/reports/mobile-assistant-rebuild-review-2026-10-20.md` and its evidence folder `<workspace>/reports/mobile-review-2026-10-20/` (ten verified code digests, six designs, four judge verdicts, four critiques).
**Scope:** the automatos-ai changes that Automatos Go v1 needs. The app itself is specified in `AutomatosAI/automatos-go` `docs/prd/PRD-001-AUTOMATOS-GO-V1.md`. The theme mobile layer is PRD-261's stories P261-M-01..M-07 (waves W-M1 and W-K4p, owned by the PLAY session).
**Depends on:** PR #1198 (Clerk production cutover, Tue 27 Oct); PR #1216 (the `azp` check), which must carry the native marker before it merges (US-020); PRD-261 W-M1 (the `mobile` theme block and `GET /api/themes/mobile/{install_id}`).
**Ralph config:** launch is human-only, with `RALPH_MODEL` pinned. One branch, one push, one PR and one CI run per wave. Parallelism is sized to the remaining usage budget, not a fixed number (Gerard, 20 Oct); PRD-261's waves draw on the same budget in the same window (§9).
**Approved by Gerard, 20 Oct:** the two AGENTS.md exceptions in §4.10 (two public routes; the Expo push client) and the authentication, permission and tenancy changes this PRD makes; one tracking issue per wave records them (AGENTS.md "stop and ask first"): #1253 (W1), #1254 (W2), #1255 (W3).
**Human-gated, outside the autonomous loop:** the Clerk dashboard (Native API, bundle id `com.automatos.app`, Clerk Pro, 30-day sessions, the `mobile` JWT template) [D39, D52]; Expo's data-processing agreement [D20]; the Retell dashboard (audio retention off, transcripts kept, EU region if offered) [D53]; the moderation inbox destination (US-016); the production deploy; Gerard's VPS demo install [D13]. Standing decision owned by Gerard: Expo "enhanced push security" stays **off** on the Automatos EAS project, or self-hosted push stops working [D21].

Labels: **[T]** read in code on `origin/main` (19–20 Oct); **[I]** inferred; **[Dn]** Gerard's decision n.

## 1. Introduction

Automatos Go is a native iOS app (Android testers install by link) that puts Auto, the Command Centre and voice on the phone [D3, D10, D11]. It talks to the automatos-ai API directly. Most of what it needs already exists: the chat stream, the board, questions, the activity feeds, notifications and the theme routes [T]. This PRD adds what is missing.

| Need | Today on `origin/main` [T] | This PRD |
|---|---|---|
| A phone on a self-hosted install has an identity | The local edition has no credential at all: every request is the anonymous operator, a `super_admin`, and an `X-Workspace-ID` header picks any workspace unchecked (`core/auth/hybrid.py`, anonymous lane). Compose publishes Postgres, Redis, MinIO, Qdrant, the API and the web app on the same `${BIND_ADDRESS:-127.0.0.1}` (`docker-compose.yml`) | Device pairing modelled on `cli_hosts`, a `device` auth lane, a remote-access mode that refuses anonymous callers, separate publish addresses for the API and the web app (US-002..US-005) |
| The app learns what it is talking to | `GET /health` is unauthenticated and returns `edition` (`main.py`); nothing returns the Clerk publishable key or the server's capabilities | `GET /.well-known/automatos.json` (US-001) |
| Push to the phone | None. Notification destinations are `in_app`, `telegram`, `slack`, `webhook`, `channel`, `silent` (`api/notifications.py` `_VALID_DESTINATIONS`); no device table, no APNs, FCM or Expo code anywhere | A `push` destination, device registration, an Expo sender, three push types, a daily digest (US-009..US-014) |
| Editors answer agent questions | Every approval-grant verb is `require_workspace_admin` (owner/admin) (`api/approval_grants.py`); needs-you hides questions from non-admins (`services/needs_you.py`) | Editors may answer, grant and deny [D16] (US-006) |
| Spoken replies for on-device voice | `spoken_mode` exists in `StreamingChatService` but only the Retell socket sets it; `ChatRequest` has no flag (`api/chat.py`) | `spokenMode` on `POST /api/chat` (US-007) |
| Auto Live joins natively | `POST /api/voice/web-call` drops Retell's transport, URL, ICE servers and expiry (`modules/voice/retell_api.py`); voice is Business-only in `PLAN_TIERS` but nothing enforces it; the custom-LLM socket accepts a `call_id` after the call ended; the local operator cannot mint | Pass-through, a Business gate on SaaS, live-only sockets, the local caller fix [D32, D34] (US-008) |
| Store duties | No "report this content"; no list of the AI providers a workspace uses | Content reports, `ai_providers` [D15, D23] (US-016, US-017) |
| Small defects the phone hits | A paused scheduled task vanishes from the schedule feed (`services/activity_service.py`, `status = 'active'` filter); the `agent-info` stream frame is malformed (`consumers/chatbot/service.py` passes the dict as `event_type`) | US-018, US-007 |

Anchors builders rely on: the single Alembic head is `clerk_user_relinks_20261019` today, pinned as `EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py` and asserted in `test_prd236_w1_routes.py`; re-read it at rebase, it moves fast. Routes the frontend calls go in `orchestrator/reports/route-manifest.json` (`python -m scripts.dump_routes`). Settings go in `orchestrator/reports/config-surface.json`. `AUTH_EDITION` is `local` or `saas` (`config.py`). [T]

## 2. Goals

- A phone signs in to hosted Studio with Clerk, or pairs with a self-hosted install by QR code, without changing how the web authenticates on SaaS.
- A self-hosted install can be reached from a phone without exposing its database, Redis, object store or vector store, and without making every device on the network the owner.
- An agent question, a ticket entering Review, and Auto's own messages reach the phone as push. Target: 95% of pushes accepted by Expo within 10 seconds of the dispatch.
- Editors can answer questions from the phone and from Studio [D16].
- The store duties the app needs exist before the first external build: content reports and the AI-provider list.
- Everything ships in time for the team's phones at WebSummit (10–12 Nov) on internal TestFlight [D51]. Proposed merge dates per wave are in §9; the dates are Gerard's call.

## 3. Decisions this PRD implements

| | Decision (from the Q&A) |
|---|---|
| D1 | v1 editions: SaaS + open-source pairing. Enterprise (a customer-owned Clerk instance found at runtime) and the enterprise proof of concept are after WebSummit [D2]. |
| D5–D6 | Sign-in only; Clerk email code, password and passkey. No social logins, so no Sign in with Apple. |
| D63 | The app deletes nothing. Uninstalling the app is the user's business; deleting a workspace stays in Studio, unchanged; a person who wants their login removed asks support, as the privacy policy says (revises D7) [D47]. |
| D8 | Billing shows nothing on the phone. No change here. |
| D13 | Gerard's VPS demo install is the WebSummit server: HTTPS on a domain, API and web published, datastores private. |
| D15 | "Report this" goes to the workspace owner and an Automatos moderation inbox; on a self-hosted install, to its operator only. |
| D16 | Editors may answer questions and grant or deny tool approvals. Viewers may not. Applies to the web too. |
| D18–D19 | Lock screen: Open and Answer on an unlocked phone. Face ID before approve is client-side only. No server step-up. |
| D20–D23 | Push carries a question's text and options; everything else is generic. Self-hosted servers push through Expo after the operator agrees at pairing. Three push types. A one-time AI consent sheet names the providers. |
| D24 | The daily digest is Auto's Read, posted as Auto's message at each user's chosen time. |
| D32–D34 | Auto Live joins Retell's call natively (LiveKit React Native SDK); quick-ask voice uses `spokenMode`; Live is Business-only on SaaS. |
| D39 | A native marker in the `azp` check, landed before #1216 merges. |
| D53 | Retell: audio retention off, transcripts kept, EU region if offered. |

## 4. Design

### 4.1 Discovery document

`GET /.well-known/automatos.json`, unauthenticated, served by the API. It is a **named exception** to AGENTS.md "every new endpoint is authenticated", like PRD-264 D16: it carries names and capabilities only, never secrets.

```json
{
  "schema": "automatos-discovery/v1",
  "name": "Automatos",
  "edition": "saas",
  "api_url": "https://api.automatos.app",
  "web_url": "https://ui.automatos.app",
  "auth": { "modes": ["clerk"], "clerk": { "publishable_key": "pk_live_…", "frontend_api": "clerk.automatos.app" } },
  "pairing": { "enabled": false },
  "push": { "expo": true },
  "capabilities": ["push", "spoken_mode", "voice_live", "mobile_theme", "report_content", "digest"]
}
```

- `edition: "local"` gives `auth.modes: ["device"]`, `clerk: null`, and `pairing.enabled` = whether remote access is on (§4.3).
- `push.expo` is true when this server sends push through Expo (SaaS always; local only after the operator agreed, §4.4).
- The app gates every feature on `capabilities`, never on a version number, so an older self-hosted server degrades feature by feature.
- `api_url` comes from `PUBLIC_API_HOST`; `web_url` from `FRONTEND_URL`.

### 4.2 One device table

New table `mobile_devices`. AGENTS.md says to extend before adding: `cli_hosts` is the nearest table, but it registers CLI runtimes (capabilities, heartbeats, terminals, sessions) and is local-only. A phone carries a person, push credentials and digest preferences. Mixing the two lifecycles in one table would make every query branch on kind. One new table serves both kinds of phone:

| Column | Notes |
|---|---|
| `id` uuid PK | |
| `workspace_id` uuid FK workspaces | the device is bound to one workspace |
| `user_id` int FK users | the person (the local operator's row on a local install) |
| `kind` | `paired` (local edition, holds a device token) or `clerk` (SaaS, push registration only) |
| `install_id` | random id the app generates once and keeps in SecureStore; unique with (`workspace_id`, `user_id`) |
| `name`, `platform` (`ios`/`android`), `app_version`, `locale`, `timezone`, `digest_time` (`HH:MM`, nullable) | |
| `status` | `pending`, `active`, `revoked` |
| `pairing_code_hash`, `pairing_expires_at` | as `cli_hosts` |
| `token_hash` unique, `token_rotated_at`, `prev_token_hash`, `prev_token_valid_until` | rotation keeps the old token valid for 10 minutes |
| `expo_push_token` unique nullable, `push_enabled`, `push_disabled_reason` | |
| `push_key` | per-device HMAC key, returned once at registration, used to sign push payloads (§4.4) |
| `last_seen_at`, `paired_at`, `revoked_at`, `revoked_by`, `events` jsonb (last 50: enrolled, rotated, revoked, push_failed) | audit |

Hashing, code alphabet, TTL and one-time token return copy `services/cli_host_service.py` (`hash_secret`, `_new_pairing_code`, `PAIRING_CODE_TTL_SECONDS = 600`).

### 4.3 Pairing, the device lane and remote access (local edition)

**Remote access mode.** New setting `LOCAL_REMOTE_ACCESS`: `off` (default) or `paired`. Compose passes the publish addresses into the API container, so the API knows whether it is reachable beyond loopback.

- New compose variables `API_PUBLISH_ADDRESS` and `WEB_PUBLISH_ADDRESS`, both defaulting to `${BIND_ADDRESS:-127.0.0.1}`. Postgres, Redis, MinIO, Qdrant, Adminer and SearXNG keep `BIND_ADDRESS`. A self-hoster publishes only the API and web app with `API_PUBLISH_ADDRESS=0.0.0.0 WEB_PUBLISH_ADDRESS=0.0.0.0`.
- With `LOCAL_REMOTE_ACCESS=paired`, the anonymous lane is **closed**: every request needs a device token, the browser included. Identity is a credential, never an address, because inside the container a request from the same machine and one from the LAN can both arrive from the Docker gateway, and a same-host reverse proxy makes every caller look like 127.0.0.1.
- If either publish address is not loopback and `LOCAL_REMOTE_ACCESS` is `off`, the API refuses to boot with a message naming the fix. `LOCAL_REMOTE_ACCESS=open` keeps today's behaviour for people who accept the risk, and logs a warning at every boot. (Open question 2: this changes behaviour for installs that already opened `BIND_ADDRESS`.)
- **Pair this browser.** In `paired` mode a browser pairs exactly like a phone: it shows a code field, and the token is kept in the browser and sent as a bearer. The first code comes from the server console: `docker compose exec api python -m scripts.mint_pairing_code`. After that, a paired browser mints codes from Studio.

**Routes** (new `api/mobile_devices.py`, prefix `/api/v1/mobile`):

| Route | Auth | Does |
|---|---|---|
| `POST /pairing-codes` | owner/admin (the local operator passes); 404 in `saas` | `{code, expires_at, qr_payload}`; the QR payload is a universal link `https://<web_url>/go/pair?api=<api_url>&code=<code>`, never a custom scheme (any app can register a scheme) |
| `POST /pair` | **public named exception**; `check_rate_limit` per IP and per instance (`core/security/rate_limiter.py`); 404 unless `paired` mode | body `{code, name, platform, app_version, install_id}` → `{device_id, workspace_id, token: "amd_…", rotate_after}` once |
| `POST /devices` | Clerk or device | register or update this phone: `{install_id, platform, app_version, locale, timezone, expo_push_token?, digest_time?}` → `{device_id, push_key}` (`push_key` only on first registration) |
| `GET /devices` | Clerk or device | the caller's devices; owners and admins also see the workspace's |
| `PATCH /devices/{id}` | owner of the device | `push_enabled`, `digest_time`, `timezone` |
| `POST /devices/{id}/rotate` | device lane, own device | new token once; the previous token stays valid for 10 minutes |
| `DELETE /devices/{id}` | owner of the device, or workspace owner/admin | `revoked`; the next call with that token is `401 {"code": "device_revoked"}` |

**The device lane** in `get_request_context_hybrid`: a bearer starting `amd_` is resolved before the Clerk lane, only when `AUTH_EDITION=local`. It looks up the active row by `token_hash` (`hmac.compare_digest`), builds the `UserContext` from the bound `users` row (in local, the operator with its `super_admin` role), sets `auth_type="device"`, binds `workspace_id` to the row (a different `X-Workspace-ID` is 403), and puts `{source: "mobile_device", device_id, user_id}` in `raw_claims`. `last_seen_at` is written at most every 5 minutes.

**Every resolver that must know `device`:** `core/auth/dependencies.py` (`auth_type` literal); `core/auth/workspace_permission.py` (judge `device` by the bound user's role); `core/auth/workspace_admin.py` (`_is_workspace_admin_member` resolves by `clerk_user_id` today; add the `users.id` path); `services/themes/membership.py` (`PERSON_AUTH_TYPES`, otherwise paired phones 404 on theme files and on PRD-261's mobile endpoint); `core/auth/step_up.py` (local passes, as today); the notifications user resolver in `api/notifications.py`; `modules/voice/caller.py` `resolve_caller_int_id` (read `raw_claims.user_id`; this also fixes today's 403 for the local operator).

**Web:** Settings → **Phones & devices**: the device list (name, platform, last seen, push on or off, Revoke); in local `paired` mode, "Pair a phone" (QR plus code plus a 10-minute countdown) and "Pair a browser". On SaaS the list still shows the member's own phones.

### 4.4 Push

- **Destination.** Add `push` to `_VALID_DESTINATIONS` in `api/notifications.py` and `core/services/auto_reporting.py`, and to the dispatcher as an **external** destination, so workspace quiet hours silence it like Telegram unless severity is urgent (`core/services/notification_dispatcher.py`).
- **Sender.** New `services/push/` with a `PushSender` interface and `ExpoPushSender` (POST `https://exp.host/--/api/v2/push/send`, 100 messages per request, optional `EXPO_PUSH_ACCESS_TOKEN`). A receipts job on the existing APScheduler host disables tokens Expo reports as `DeviceNotRegistered`.
- **Row id.** `_insert_in_app` returns the new notification id (`RETURNING id`) so the payload can carry it and the app can mark it read.
- **Which events push [D22]:**

| Push type | Events | Who receives it |
|---|---|---|
| `needs_answer` | `question_pending`, `approval_pending`, `mission_plan_ready` | members who may act on it: owner, admin, editor [D16] |
| `review` | **new** `review_pending`: a ticket enters Review (dispatched from the board status change) | members holding `missions:update` |
| `auto_message` | **new** `auto_message`: Auto posts unprompted in a user's thread (`services/chat_messenger.py` `deliver_background_message`) and the daily digest | that user only |

  Every other event stays in-app. Workspace-wide rows fan out to the active members' devices in that workspace; membership and role are re-checked at send time, so a removed member's phone stops receiving. A device receives push only when `push_enabled` and, on a local install, the operator agreed (`push.expo`).
- **Payload [D20]:** `{v: 2, type, workspace_id, notification_id, grant_id?, link_type, link_id, ticket_number?, question?, options?, iat, sig}` (signature v2, the lead, 21 Oct; it supersedes v1). Every id and `ticket_number` is a string; `iat` is the build time in integer epoch seconds; a field with no value is left out, never null. A question carries its text (plain text, at most 500 characters with an ellipsis) and its options when there are 1 to 3 of at most 80 characters each (never cut: `/answer` takes the exact string), so it can be answered from the notification; every other type carries a generic line ("A ticket is ready for review · #0419"). A per-workspace setting `push_redact_questions` (default off) sends questions generic too; they are then Open only.
- **AI notice.** The push title carries the fixed AI badge from PRD-264 D17 ("Ledger · AI"), because a push is a new outbound path to a person and Article 50 has no setting that removes it.
- **Signature (v2).** `sig` = lowercase hex HMAC-SHA256, keyed by the device's `push_key` as UTF-8, over the UTF-8 bytes of the compact JSON array `["v2", type, workspace_id, notification_id, grant_id, link_type, link_id, ticket_number, question, options, iat]` (`json.dumps(…, separators=(',', ':'), ensure_ascii=False)`, byte for byte JavaScript's `JSON.stringify`): a string field the payload leaves out is `""`, `options` is the payload's list in order or `[]`. It covers every field but `sig`. The app refuses a `v` other than 2 and an `iat` more than 7 days old (`PUSH_SIG_MAX_AGE_SECONDS` = 604800) or more than 300 seconds ahead (`PUSH_SIG_MAX_FUTURE_SECONDS`). Anyone holding an Expo push token can send a push, so the app verifies `sig` before acting on a notification action; on a mismatch it posts nothing and opens the app. One shared vector file, `orchestrator/tests/fixtures/prd266_push_signature_vectors.json`, holds the cases both sides check (automatos-go copies it byte for byte).
- **Answer from the lock screen.** The app posts the existing `POST /api/v1/approval-grants/{id}/answer` with the user's credential; an `option` must be one of the grant's options (422 otherwise). Then `POST /api/notifications/{id}/read`. Confirmation strings reuse the Telegram bridge's honest ones ("Answered #N, the agent is resuming", "was already answered").
- **Preferences.** `GET/PUT /api/notification-preferences` accept `push` rows for the three types; the app's Notifications screen writes them.

### 4.5 Daily digest [D24]

A job on the APScheduler host runs every 15 minutes. For each active device with a `digest_time`, when the device's local time (`timezone`) passes `digest_time` and no digest was posted for that user, workspace and local date, it builds Auto's Read (`services/digest_service.py`) and posts it into the user's Auto thread through `deliver_background_message`. The `auto_message` push carries it. One digest per user per workspace per day, even with several phones.

### 4.6 Roles: editors answer [D16]

A new dependency `require_workspace_answerer` (owner, admin or editor of the workspace, or `super_admin`; `device` callers by their bound user's role) replaces `require_workspace_admin` on the approval-grant list, `answer`, `grant` and `deny`. `revoke` and every other `require_workspace_admin` use (health tiles and the rest) are unchanged. `services/needs_you.py` counts questions and approvals for editors. The web Questions tab and the governance approval cards show for editors. Viewers still get 403.

### 4.7 Chat and voice

- **`spokenMode`** (bool, default false) on `ChatRequest`, threaded to `StreamingChatService(spoken_mode=True)`: short sentences, no markdown. Used by the app's quick-ask voice [D32].
- **`agent-info` frame:** emit `d:{"type":"agent-info","agent":{…}}`, as the web parser expects.
- **Voice mint [D32, D34]:** on `AUTH_EDITION=saas`, `POST /api/voice/web-call` refuses with 403 "Auto Live is on the Business plan" unless the workspace's `exposure.families.voice` is true; local installs stay unrestricted. The response adds Retell's `transport`, `url` (when given), `ice_servers` and `expires_at` instead of dropping them.
- **Live-only socket:** the `call_ended` webhook stamps the call `ended`; `WS /api/voice/retell/llm-websocket/{call_id}` refuses (4409) any call that is not live, so a leaked `call_id` cannot drive Auto's tools after the call.

### 4.8 Store duties

- **No deletion in the app [D63].** The app is sign-in only, so neither store requires in-app account deletion (both rules apply to apps that create accounts). Deleting a workspace stays in Studio, unchanged. When someone uninstalls the app, Expo reports the push token as `DeviceNotRegistered` and the receipts job (US-010) disables it; the device row can be revoked from Studio.
- **Content reports [D15].** `POST /api/v1/reports` `{target_type: "message" | "deliverable", target_id, reason, note?}`. It writes an audit row (`core/workspaces/audit.py`, event `content_reported`, no new table), dispatches a new in-app event `content_reported` to the workspace owner, and on SaaS posts a summary to `MODERATION_WEBHOOK_URL` (a new setting; unset on self-hosted installs, so only the operator sees it). Owners can list reports from the existing audit view.
- **AI providers [D23].** `GET /api/workspaces/current` adds `ai_providers: [{id, name}]`: the providers whose models this workspace's Auto and agents can call (platform routes plus the workspace's own keys) and the embeddings provider. The app's consent sheet shows the names and is shown again when the list changes.

### 4.9 Fixes the phone hits

- Paused scheduled tasks stay in `GET /api/activity/schedule` with `paused: true`, instead of vanishing (`services/activity_service.py`) [D27].
- `GET /api/chat/history` honours `starting_after` (the web already sends it), so the phone's chat list can page.
- Universal links: `apple-app-site-association` and `assetlinks.json` served by the frontend on `ui.automatos.app` for `/go/*`, `/command-center` and `/chat` paths, app id `KGVA9933BG.com.automatos.app`.

### 4.10 Where the code goes, and the patterns it follows

**Gerard, 20 Oct:** follow the existing patterns, put code in proper modular directories, keep it clean, and keep the enterprise direction (PRD-257's registered runners and OIDC) in view.

**Files already over AGENTS.md's 800-line limit may not grow** [T]: `core/auth/hybrid.py` (954 lines), `api/chat.py` (908), `api/approval_grants.py` (806), `main.py` (1,681), `services/cli_host_service.py` (2,799). New code goes in new modules; these files only gain a call or an import and, where a change would grow them, something moves out in the same PR.

| What | Where | Pattern it follows |
|---|---|---|
| Shared pairing kernel: code alphabet, TTL, `hash_secret`, one-time token minting with a prefix, constant-time compare, rotation overlap | **new `core/auth/pairing.py`** | Extracted from `services/cli_host_service.py`; CLI hosts switch to it and their copies are **deleted in the same PR** (AGENTS.md "Replace, don't shim"). PRD-257's estate runners use it next. |
| Bearer-prefix resolvers | **new `core/auth/bearer_registry.py`**: a small registry `{prefix: resolver}`; `hybrid.py` calls it once before the Clerk lane | Keeps `hybrid.py` from growing; PRD-257's runner tokens and OIDC workload identity add a resolver, not another branch |
| The device lane | **new `core/auth/device.py`** (`resolve_device_context`) | Builds the same `RequestContext` / `UserContext` the Clerk and local-operator lanes build |
| Editors-answer gate | **`core/auth/workspace_admin.py`** gains `require_workspace_answerer` (the file is small); `api/approval_grants.py` only swaps the dependency | Same shape as `require_workspace_admin` |
| Device model | **new `core/models/mobile_devices.py`** | `core/models/cli_hosts.py` |
| Device, pairing, digest, discovery, remote-access logic | **new package `services/mobile/`**: `pairing.py`, `devices.py`, `discovery.py`, `remote_access.py`, `digest.py` | One domain, one package, as `services/themes/` |
| Push | **new package `services/push/`**: `sender.py` (the `PushSender` protocol), `expo.py`, `payload.py` (build and sign), `fanout.py` (recipients, membership re-check), `receipts.py` (job) | The dispatcher gains one `push` branch beside its external destinations, handing the row to `services/push/fanout.py`, as it hands Telegram rows to `channels.sender` |
| Content reports | **new package `services/reports/`** | Reuses `core/workspaces/audit.py` |
| Routers | **new thin routers** `api/mobile_devices.py`, `api/well_known.py`, `api/account.py`, `api/content_reports.py`: validate with Pydantic, call the service, map errors; plain `def` over the sync session | PRD-261's `api/themes_*.py`; registered in `main.py` with one import and one `include_router` each, as PRD-261 did |
| Chat request model | **move `ChatRequest` and its parts out of `api/chat.py` into new `api/chat_models.py`**, then add `spokenMode` there | Shrinks `api/chat.py` instead of growing it |
| Settings | `config.py` only, names in `reports/config-surface.json`: `LOCAL_REMOTE_ACCESS`, `API_PUBLISH_ADDRESS`, `WEB_PUBLISH_ADDRESS`, `MOBILE_TOKEN_ROTATE_DAYS`, `MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS`, `EXPO_PUSH_ACCESS_TOKEN`, `PUSH_RECEIPTS_INTERVAL_SECONDS`, `DIGEST_JOB_INTERVAL_SECONDS`, `MODERATION_WEBHOOK_URL` | No `os.getenv` outside `config.py` (ruff `TID251`); no hardcoded values |
| Web | `components/settings/phones-devices/` (list, pair a phone, pair a browser), hooks in `hooks/use-mobile-devices.ts`, calls only through `lib/api-client.ts` | Existing settings tabs and hooks |
| Tests | `orchestrator/tests/test_prd266_<area>_<what>.py`; frontend `*.test.tsx` beside the component | PRD-261's naming |

Code shape on every changed line: functions under 50 lines, nesting under 4, cyclomatic complexity under 10, new files 200–400 lines (never over 800), type hints and docstrings on public functions, no mutation of arguments, no swallowed errors.

**Exceptions to AGENTS.md this PRD needs (approved by Gerard, 20 Oct; each PR still records the one it uses):**
1. **Two public routes** (`GET /.well-known/automatos.json`, `POST /api/v1/mobile/pair`) against "every new endpoint is authenticated", as PRD-264 D16 did for `/verify`.
2. **One third-party client** (`services/push/expo.py`) against "integrations go through Composio; no new third-party API clients". Push is platform infrastructure, not a workspace integration, as Retell is for voice (`modules/voice/retell_api.py`) and Stripe is for PRD-265. `EXPO_PUSH_ACCESS_TOKEN` is an environment setting read through `config.py`, not a key in Settings.

**AGENTS.md "stop and ask first":** this PRD changes authentication (the device lane, remote-access mode), permissions (editors answer) and tenancy-adjacent code (device rows bound to a workspace). Once Gerard approves it, it is the record, and one tracking issue per wave is opened before code starts. No new Python or npm dependency on the backend: the Expo sender uses `httpx`, which the orchestrator already depends on.

### 4.11 Enterprise alignment (PRD-257)

PRD-257 (draft, branch `docs/prd-257-enterprise-estate-agents`) plans registered estate runners (pairing code → one-time token, scopes, expiry, revoke, short-lived job tokens) and OIDC identity: runners by workload identity against a configured JWKS (the `PyJWKClient` pattern in `core/auth/clerk.py`), and people through the customer's OIDC provider (its open question 4). No OIDC or SSO PRD exists elsewhere [T]. This PRD builds the shared pieces so enterprise slots in rather than branching off:
- **One pairing kernel** (`core/auth/pairing.py`) for CLI hosts, phones and, next, runners.
- **One credential registry** (`core/auth/bearer_registry.py`): phones add `amd_`; runners and workload-identity tokens add their own resolvers later.
- **Discovery reserves OIDC.** `auth.modes` may later include `"oidc"` with `{issuer, client_id, scopes}`; v1 never emits it, but the app treats an unknown mode as "not supported by this app version" rather than failing, and the schema version stays `v1` when it is added.
- **Device rows carry `kind`**, so an OIDC-signed phone becomes a third kind, not a new table.
- Nothing here chooses PRD-257's answer to its open question 4 (a third sign-in path beside Clerk, or generic OIDC everywhere). That stays Gerard's call after WebSummit [D2].

## 5. User stories

Every story: tests go in with the story; CI is the gate and nothing runs on a maintainer's machine; route-manifest regenerated when routes change; new settings in `config-surface.json`; functions and components within the AGENTS.md limits; commits signed off (`git commit -s`).

### Wave W1: Connect

#### US-001: Discovery document
**Description:** As the app, I need to learn what a server is before I show a sign-in, so one binary works with hosted Studio and self-hosted installs.

**Acceptance Criteria:**
- [ ] `GET /.well-known/automatos.json` returns the shape in §4.1 with no credentials, in both editions, from the new `api/well_known.py` router and `services/mobile/discovery.py`; the PR records the public-route exception (§4.10)
- [ ] `saas` returns `auth.modes ["clerk"]` with the publishable key and frontend API; `local` returns `["device"]` and `clerk: null`
- [ ] `capabilities` lists only what this build implements (a test asserts each name maps to a live route)
- [ ] No secret or env value other than the publishable key appears (test scans the body against the config surface's secret names)

#### US-002: `mobile_devices` table
**Description:** As a developer, I need one table for phones so pairing, push and digests share one record.

**Acceptance Criteria:**
- [ ] Alembic revision on the current head adds `mobile_devices` as in §4.2 with a unique (`workspace_id`, `user_id`, `install_id`) and unique `token_hash` and `expo_push_token`
- [ ] Both head pins move (`test_prd209_alembic_single_head.py`, `test_prd236_w1_routes.py`)
- [ ] Model in `core/models/mobile_devices.py`; the PR body states why `cli_hosts` was not extended (§4.2)

#### US-003: Pairing routes
**Description:** As a self-hoster, I want to pair my phone with a QR code from Studio, so it can reach my install without a login system.

**Acceptance Criteria:**
- [ ] `POST /api/v1/mobile/pairing-codes`, `/pair`, `/devices` (GET, POST), `/devices/{id}` (PATCH, DELETE), `/devices/{id}/rotate` behave as §4.3, including 404 in `saas` for the two pairing routes
- [ ] The code is 8 characters from the `cli_hosts` alphabet, valid 600 seconds, stored only as a hash; the token is returned once and stored only as a hash
- [ ] `/pair` is rate-limited per IP and per instance; a wrong code costs one attempt and never locks the instance for others
- [ ] Rotation: the old token works for 10 minutes, then 401; revoke: the next call is `401 device_revoked`
- [ ] The QR payload is a universal link on `web_url`, not a custom scheme
- [ ] Code alphabet, TTL, hashing and token minting come from the new `core/auth/pairing.py`; `services/cli_host_service.py` uses it too and its own copies are deleted in this PR; CLI-host tests still pass unchanged

#### US-004: Device lane and resolvers
**Description:** As a paired phone, I need every existing route to accept my token as the person I was paired as.

**Acceptance Criteria:**
- [ ] `Bearer amd_…` resolves as in §4.3 only when `AUTH_EDITION=local`; in `saas` an `amd_` token is 401
- [ ] A different `X-Workspace-ID` is 403
- [ ] `workspace_permission`, `workspace_admin`, `themes/membership.py` (`PERSON_AUTH_TYPES`), `step_up`, the notifications user resolver and `voice/caller.py` accept `device`; one test per resolver
- [ ] A paired phone can chat, read the board, answer a question and load `GET /api/themes/mobile/{install_id}` (PRD-261 M-04) in an integration test
- [ ] The lane lives in `core/auth/device.py`, reached through the `core/auth/bearer_registry.py` prefix registry; `core/auth/hybrid.py` does not grow

#### US-005: Remote-access mode and publish addresses
**Description:** As a self-hoster, I want to expose only the API and the web app, and have every client prove who it is, so pairing a phone never opens my database or makes the network my owner.

**Acceptance Criteria:**
- [ ] `docker-compose.yml` gains `API_PUBLISH_ADDRESS` and `WEB_PUBLISH_ADDRESS` (default `${BIND_ADDRESS:-127.0.0.1}`); datastores keep `BIND_ADDRESS`; both values are passed into the API container
- [ ] `LOCAL_REMOTE_ACCESS=paired` closes the anonymous lane (401 `device_pairing_required` without a token); `off` with a non-loopback publish address refuses to boot with a message naming the fix; `open` keeps today's behaviour and warns at boot
- [ ] `scripts/mint_pairing_code` prints a code and QR text from the server console
- [ ] Studio gains "Pair a browser" in `paired` mode; a paired browser keeps working after a restart
- [ ] Self-hosting docs (`QUICKSTART.md`, `SETUP.md`) gain a "Reach your install from your phone" section: TLS (Caddy or Tailscale), the two publish variables, `LOCAL_REMOTE_ACCESS=paired`

#### US-006: Editors answer [D16]
**Description:** As an editor, I want to answer agent questions and approve tool requests, so the work doesn't wait for an owner.

**Acceptance Criteria:**
- [ ] `require_workspace_answerer` guards approval-grant list, `answer`, `grant`, `deny`; `revoke` stays owner/admin
- [ ] needs-you counts questions and approvals for editors; viewers get 403 and a count of 0
- [ ] The web Questions tab and approval cards render for editors (frontend test)
- [ ] The PR body records that this reverses PRD-196's owner/admin posture, by Gerard's decision of 20 Oct

#### US-007: Chat changes for the phone
**Description:** As the app's quick-ask voice, I need short spoken-style replies and a well-formed agent frame.

**Acceptance Criteria:**
- [ ] `ChatRequest` and its parts move to `api/chat_models.py` first, so `api/chat.py` shrinks; then `spokenMode: true` on `POST /api/chat` produces replies with no markdown (test on a recorded turn)
- [ ] The `agent-info` frame is `d:{"type":"agent-info",…}` and the web parser reads it (frontend test)
- [ ] `GET /api/chat/history?starting_after=<id>` pages (US-018 may carry it if the lane is full)

#### US-008: Voice for the phone
**Description:** As the app, I need to join an Auto Live call natively, only where it is allowed, and the server must refuse a stale call.

**Acceptance Criteria:**
- [ ] On `saas`, a workspace without `families.voice` gets 403 "Auto Live is on the Business plan" from the mint; `local` is unrestricted
- [ ] The mint response adds `transport`, `url`, `ice_servers`, `expires_at` when Retell returns them (test with a recorded Retell response)
- [ ] `call_ended` marks the call ended; the socket refuses a non-live `call_id` with 4409
- [ ] The local operator (anonymous or device lane) can mint (the `resolve_caller_int_id` fix)

### Wave W2: Push

#### US-009: Device registration for Clerk phones
**Description:** As a SaaS user, I want my phone registered for push when I sign in.

**Acceptance Criteria:**
- [ ] `POST /api/v1/mobile/devices` with a Clerk bearer upserts a `kind: clerk` row by (`workspace_id`, `user_id`, `install_id`) and returns `push_key` once
- [ ] Signing out in the app calls `DELETE /devices/{id}`; a test shows the row revoked and no further push

#### US-010: Push destination and Expo sender
**Description:** As the platform, I need a push destination that every existing event caller reaches without touching call sites.

**Acceptance Criteria:**
- [ ] `push` is valid in preferences and auto-reporting; the dispatcher treats it as external (quiet hours apply; a test covers urgent passing through)
- [ ] `ExpoPushSender` (in `services/push/expo.py`, behind the `PushSender` protocol) batches 100 per request; the receipts job disables `DeviceNotRegistered` tokens with a reason
- [ ] The PR records the AGENTS.md exception for this third-party client (§4.10, approved 20 Oct)
- [ ] `_insert_in_app` returns the row id
- [ ] No push leaves a local install unless the operator agreed at pairing

#### US-011: Fan-out, payload and signature
**Description:** As a member, I want the right pushes on my phone and only mine.

**Acceptance Criteria:**
- [ ] The three types fan out to the recipients in §4.4; membership and role are re-checked at send time (a removed member's device receives nothing)
- [ ] Payload as §4.4; questions carry text and up to three options unless `push_redact_questions` is on
- [ ] Titles carry the PRD-264 AI badge
- [ ] `sig` verifies with the device's `push_key` (a shared test vector with the app)

#### US-012: New events
**Description:** As a member, I want to know when a ticket is ready for review and when Auto messages me.

**Acceptance Criteria:**
- [ ] `review_pending` dispatches when a ticket moves into Review, once per entry
- [ ] `auto_message` dispatches from `deliver_background_message`, to that user only
- [ ] Both are in `VALID_EVENT_TYPES` with in-app defaults, and the web bell links them (the drift-guard fixture in `notification-bell-links.test.ts` is updated)

#### US-013: Daily digest job
**Description:** As a user, I want Auto's morning brief in my thread at the time I chose.

**Acceptance Criteria:**
- [ ] The job posts Auto's Read once per user, workspace and local date when a device's `digest_time` passes in its `timezone` (tests across timezones and DST)
- [ ] Several phones produce one digest
- [ ] The post triggers an `auto_message` push

#### US-014: Push preferences
**Description:** As a user, I want to switch each push type on or off from the phone.

**Acceptance Criteria:**
- [ ] `push` rows for the three types are readable and writable through `/api/notification-preferences`
- [ ] The workspace setting `push_redact_questions` is owner/admin-editable and returned with workspace settings

### Wave W3: Store duties and fixes

#### US-015: (withdrawn 20 Oct)
Account deletion from the app is withdrawn by Gerard's decision 63: the app deletes nothing, and deleting a workspace stays in Studio. The id is kept so the tracking issue's numbering holds.

#### US-016: Content reports
**Description:** As a user, I want to report an Auto message or a Deliverable, so the owner and Automatos can act on it.

**Acceptance Criteria:**
- [ ] `POST /api/v1/reports` writes the audit row, notifies the owner in-app, and on SaaS posts to `MODERATION_WEBHOOK_URL`
- [ ] Unset on self-hosted installs: nothing leaves the install
- [ ] Reports never carry the reported text to the webhook beyond a 200-character excerpt

#### US-017: AI providers for the consent sheet
**Description:** As the app, I need the providers this workspace's Auto uses, so the consent sheet names them.

**Acceptance Criteria:**
- [ ] `GET /api/workspaces/current` returns `ai_providers` (display names, stable ids)
- [ ] The list changes when the workspace adds or removes its own provider key (test)

#### US-018: Schedule, history and links
**Description:** As the app, I need paused tasks visible, chat history pageable, and universal links.

**Acceptance Criteria:**
- [ ] Paused scheduled tasks appear with `paused: true`
- [ ] `starting_after` pages chat history (if not done in US-007)
- [ ] The frontend serves `apple-app-site-association` and `assetlinks.json` for app id `KGVA9933BG.com.automatos.app` with the right content types
- [ ] A `/go/pair` web page, shown when the QR is scanned without the app installed, says how to get Automatos Go and repeats the code; it never pairs the browser by itself

#### US-019: Phones & devices in Studio
**Description:** As a member, I want to see and revoke my phones, and as a self-hoster pair one, from Studio.

**Acceptance Criteria:**
- [ ] Settings → Phones & devices lists devices with last seen, push state and Revoke
- [ ] In local `paired` mode: "Pair a phone" (QR, code, countdown) and "Pair a browser"
- [ ] Owners and admins see the workspace's devices
- [ ] Studio light and dark screenshots in the PR

### Outside the waves

#### US-020: Native marker for the `azp` check (#1216)
**Description:** As a phone, my Clerk token has no `azp` because a native request has no Origin; I need the check to accept me without opening it for browsers.

**Acceptance Criteria:**
- [ ] `clerk_azp.azp_allowed` accepts an azp-less token only when it carries the `mobile` JWT template's marker claim (`client: "automatos-go"`); `CLERK_AZP_REQUIRED=true` still refuses azp-less browser tokens
- [ ] The `mobile` JWT template's claims include everything `core/auth/clerk.py` reads (`email`, `metadata`) so verification and role resolution match a session token
- [ ] Lands on #1216 before it merges; the cutover runbook gains a native-app section (Native API, bundle id, Clerk Pro, 30-day sessions, the template)

## 6. Functional requirements

- FR-1: `GET /.well-known/automatos.json` must return edition, URLs, auth modes, pairing, push and capabilities without authentication and without secrets.
- FR-2: The API must accept `Bearer amd_<token>` from an active paired device on a local install and act as the paired person in the paired workspace.
- FR-3: With `LOCAL_REMOTE_ACCESS=paired`, the API must refuse every request without a device token.
- FR-4: The API must refuse to boot on a local install whose API or web app is published beyond loopback while remote access is `off`.
- FR-5: Pairing codes must expire after 10 minutes and be stored only as hashes; device tokens must be returned once and stored only as hashes.
- FR-6: `push` must be a notification destination that respects workspace quiet hours.
- FR-7: Only `needs_answer`, `review` and `auto_message` pushes are sent; every other event stays in-app.
- FR-8: A question push must carry its text and up to three options unless the workspace redacts questions; every other push must carry generic text.
- FR-9: Every push must carry the PRD-264 AI badge in its title and an HMAC signature keyed by the device's `push_key`.
- FR-10: Owners, admins and editors must be able to list, answer, grant and deny approval grants; viewers must not.
- FR-11: `POST /api/chat` must accept `spokenMode` and then produce replies without markdown.
- FR-12: On SaaS, the voice mint must refuse workspaces without the Business voice family; the response must pass through Retell's transport fields.
- FR-13: The voice socket must refuse calls that are not live.
- FR-14: The app must offer no account or workspace deletion; deleting a workspace stays in Studio.
- FR-15: Any member must be able to report an Auto message or a Deliverable; SaaS reports reach the moderation webhook.
- FR-16: `GET /api/workspaces/current` must list the workspace's AI providers.
- FR-17: The daily digest must be posted once per user, workspace and local day at the device's chosen time.

## 7. Non-goals

These are outside v1 by Gerard's decisions, not deferred by this PRD:
- Enterprise runtime discovery of a customer-owned Clerk instance, OIDC or SAML outside Clerk, the enterprise proof of concept [D1, D2].
- Billing surfaces for the phone [D8].
- Sign-up, social logins or Sign in with Apple [D5, D6].
- A hosted push or remote-access relay for self-hosted installs [D21].
- Server step-up on approvals [D19].
- A multi-workspace switcher (no `GET /api/workspaces` list in v1).
- Marketplace themes choosing app layouts (the bigger picture, after WebSummit) [D58].

## 8. Technical considerations

- **Expo coupling.** Self-hosted push depends on the Automatos EAS project keeping enhanced push security off; a CI check in automatos-go should fail if the project setting changes (PRD-001).
- **Two auth contexts on one row.** A `clerk` device row never holds a token hash; a `paired` row never exists on SaaS. Tests cover both edges.
- **Quiet hours** are workspace-level today; users have no timezone column, so the device's `timezone` is the only per-user zone (digest only).
- **Rate limits.** The global per-IP limiter only runs with the policy plane on (off in production); the pairing route uses its own `check_rate_limit` buckets.
- **Load.** Push volume is bounded by the three types; Expo allows 600 notifications per second per project.

## 9. Waves and dates (proposed; dates are Gerard's call)

| Wave | Stories | Lanes | Proposed merge |
|---|---|---|---|
| W1 Connect (#1253) | US-001..US-008 | L1 US-002→US-003→US-004→US-005 · L2 US-001, US-006 · L3 US-007, US-008 | Wed 28 Oct (after the 27 Oct cutover) |
| W2 Push (#1254) | US-009..US-014 | L1 US-009→US-010→US-011 · L2 US-012→US-013 · L3 US-014 | Mon 2 Nov |
| W3 Store duties (#1255) | US-016..US-019 | L1 US-016, US-017 · L2 US-018 · L3 US-019 | Wed 4 Nov |
| Outside | US-020 | on #1216 | before #1216 merges |

The app's waves (PRD-001) consume these in order: W1 unblocks sign-in and pairing, W2 the push client, W3 the store-duty screens. PRD-261's W-M1 and W-K4p run in the same window on the same usage budget; the order between the two PRDs is Gerard's call.

## 10. Success metrics

- A phone pairs with the VPS demo install in under a minute from opening "Pair a phone".
- 95% of pushes are accepted by Expo within 10 seconds of the dispatch; zero pushes reach a revoked device or a removed member.
- Editors answer questions without a 403 (count of 403s on approval-grant routes for editors = 0 after W1).
- No Postgres, Redis, MinIO or Qdrant port is reachable from the LAN on an install that follows the new docs (a CI test on the compose file).

## 11. Open questions

1. **Fail-closed default.** US-005 makes an install that already set `BIND_ADDRESS=0.0.0.0` refuse to boot until it chooses `paired` or `open`. Accept that break (it is the security fix), or ship it as a warning for one release?
2. **Moderation inbox.** `MODERATION_WEBHOOK_URL` assumes a Slack (or similar) webhook. Which destination, and who answers within the "timely response" the stores expect?
3. **Digest days.** Every day, or weekdays only, with a per-user switch?

# PRD-277 US-101: every edition check, and the capability it becomes

**Status:** analysis for Wave 1 of [PRD-277](PRD-277-ENTERPRISE-OIDC-SIGN-IN.md) (tracking issue #1574).
**Read at:** `main @ 97c727136`, 20 Nov 2026. Line numbers are at that commit.
**What this is for:** today `AUTH_EDITION` is `local` or `saas`, and almost every check is `== "local"` or `IS_LOCAL_EDITION`. A third value would take the `saas` branch in each one. This table names what each check decides, so that `enterprise` is answered on purpose, row by row. The capabilities are defined for `local` and `saas` in `orchestrator/core/editions.py` (US-102). Since US-103 the `enterprise` column is code too: `editions.py` holds its row and `orchestrator/tests/test_prd277_edition_capabilities.py` pins it, with G-1 to G-8 answered below. Start-up refuses `enterprise` until Wave 4 (`orchestrator/core/edition_start_up.py`).

## How the sweep was done

`git grep -nE "AUTH_EDITION|IS_LOCAL_EDITION|IS_SAAS_EDITION|auth_edition"` over `orchestrator/` (tests excluded), then every function that takes an `edition` argument from one of those reads and compares it (`git grep -nE "edition\b.*(==|!=)"`). Comment and docstring mentions are left out. `services/` and `frontend/` were swept the same way (`isSaaS`, `isLocal`, `authEdition`, `NEXT_PUBLIC_AUTH_EDITION`, and comparisons of a reported `edition`).

**Kinds:** **C** compares `config.AUTH_EDITION`. **H** reads `IS_LOCAL_EDITION` or `IS_SAAS_EDITION`. **P** passes `config.AUTH_EDITION` to a function that compares it. **F** compares an `edition` value it was passed. **R** reports the edition's name and switches nothing.

**Count.** The PRD says 91. This sweep finds **99 lines in `orchestrator/` Python**: 81 that switch behaviour (C, H and F), 11 that pass the value on (P) and 7 that only report the name (R). That includes the 9 lines inside `config.py` itself. Outside `config.py` and the one migration, **72** lines switch behaviour. The PRD's figure was a grep at `abcc3f813`, and the same C and H greps give the same totals at that commit and at this one, so the difference from 91 is in what was counted (indirect P and F checks, `config.py`'s own lines), not in new code. Add **1** shell line (`orchestrator/docker-entrypoint.sh`), **1** line in `services/`, and **83** lines in `frontend/`.

## The capabilities

Each one answers one question the code asks. None of them means "is this edition X".

| Capability | The question it answers | `local` | `saas` | `enterprise` |
|---|---|---|---|---|
| `sign_in` | How does a person prove who they are? | `none` | `clerk` | `oidc` (§5.1) |
| `anonymous_operator` | Is a request with no credential the install's one operator, its super admin and owner of every workspace? | yes (unless `LOCAL_REMOTE_ACCESS=paired` closes the lane) | no | no, never (§5.1) |
| `multi_person` | Do several people share the install, each with their own role, so that one person's action is not everyone's consent? | no | yes | yes |
| `billing` | Does Automatos bill the workspace: billing routes, credits, plans, plan limits, quotas, the trial and the power-up? When it does not, the install's owner pays the AI providers directly. | no | yes | no (FR-24) |
| `profile_editable` | Is a person's name and email edited in Automatos? | yes | no (Clerk) | no (the IdP) |
| `step_up` | How is a sensitive action confirmed? | `skip` | `clerk_freshness` | `reauth` (D-13) |
| `device_pairing` | Can a phone or browser pair with a one-time code and hold an `amd_` token? | yes (by `LOCAL_REMOTE_ACCESS`) | no | yes, by setting (D-7) |
| `issues_sessions` | Does Automatos issue and resolve its own sign-in sessions: the `ams_` bearer lane, the session cookie and its cross-site check, the `/api/v1/auth` sign-out and session routes, and the session settings checked at start-up? No call site in the table below reads it: it gates only code Wave 3 added. | no | no | yes (D-3, §5.3; Wave 3) |
| `session_mode` | Can the operator's own CLI sessions run as agents (PRD-234)? | yes (by `CLI_RUNTIME_ENABLED`) | no | no (PRD-253 O1) |
| `hosts_automatos_cloud` | Is this install Automatos Cloud itself: the public Marketplace, the catalogue, bundles, the theme revocation feed, the moderation inbox, the isolated theme review? | no | yes | no |
| `calls_theme_revocations` | May this install pull Automatos's theme revocation list (the pull, its schedule, its sync status and Check now)? US-704 replaced `calls_automatos_cloud` with this and the next two. Each stops requests only: what a pull stored still applies when it is off (row 96). | yes | no (it is the host) | only while `ENTERPRISE_CALLS_THEME_REVOCATIONS` is on (D-12, US-704); off by default |
| `calls_cloud_sync` | May this install Sync with Automatos Cloud (the Sync routes and the catalogue pull, a Cloud theme's package download, the Cloud screenshot address)? | yes | no (it is the host) | only while `ENTERPRISE_CALLS_CLOUD_SYNC` is on; off by default; needs theme revocations (start-up refuses it alone) |
| `calls_cloud_items` | May this install request agents, Playbooks, packages and plugins from Automatos Cloud (the Sync's item and plugin passes, and each bundle and package request: `services/marketplace_bundles/items_gate.py`)? The "Items brought from Cloud" routes and the web install gate read what the install holds, so they follow `hosts_automatos_cloud` instead (rows 20, W34, W36, W37). | yes | no (it is the host) | only while `ENTERPRISE_CALLS_CLOUD_ITEMS` is on; off by default; needs Cloud sync (start-up refuses it alone) |
| `hosted_services_by_setting` | Are the three above each the operator's own setting, rather than the row's? No call site reads it: `editions.current()` does (`with_hosted_service_settings`), and start-up checks the settings where it is on. | no | no | yes (US-704) |
| `push_needs_consent` | Must the operator agree before notifications leave through Expo's push service? | yes | no | yes (G-1) |
| `operator_is_user` | Is the person using the app the person who runs the server, so that host detail, `docker compose` fix commands and the operator's own folders are theirs? | yes | no | no (G-2) |
| `guards_user_endpoints` | Must an AI endpoint a workspace saves be public `https`, called through the pinned client? | no | yes | yes (G-3; PRD-282 adds named private hosts) |
| `seeds_first_run_content` | Does every boot seed the default workspace's first-run content: its workspace and operator rows, Auto, the starter roster, the demo Playbook and the welcome Deliverable? | yes | no | no (G-4: no demo content; the default workspace with no owner is US-405's) |
| `chat_seeds_user_name` | Does chat seed the driving person's name from their `users` row into Auto's state? | yes | no | yes (G-6) |
| `install_description` | What is Auto told the install is, and which Settings tabs does it name? | `own_computer` | `hosted` | `organisation` (G-5) |
| `live_agent_named_apart` | Does a new Auto Live agent get a name of its own, so it is never mistaken for the hosted product's agent in a shared Retell dashboard? | yes | no | no: "Auto Live" (G-7) |
| `last_workspace_purge_deletes_person` | Does purging the last workspace a person owns or belongs to also delete that person's account? | no | yes | no, never (G-8) |

Rows of kind R, which only report the edition's name, read `editions.current_name()`: the name trimmed and lower-cased, `saas` for a value with no row, as `config.py` does.

The PRD's §5.1 row "A personal workspace for each new person" has no edition check today: the Clerk first-sign-in path creates it unconditionally (`core/auth/hybrid.py:283-357`). It becomes a capability in Wave 4 (US-404), not in this table.

## `orchestrator/`

### `config.py` (the flag's own file)

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 1 | `config.py:214` | C | Parses the flag: `local`, `saas` and `enterprise` are accepted, anything else becomes `saas` | (the flag itself) | `local` | `saas` | accepted (US-103); unknown still `saas` |
| 2 | `config.py:220` | C | `REQUIRE_AUTH` forced false, or forced true | `anonymous_operator` | forced false | from env, default true | forced true (US-103) |
| 3 | `config.py:515` | C | Defines `IS_LOCAL_EDITION` | (helper) | true | false | removed in US-102 |
| 4 | `config.py:519` | C | Defines `IS_SAAS_EDITION` | (helper) | false | true | removed in US-102 |
| 5 | `config.py:541` | C | `WEB_ACCESS` default when unset | `operator_is_user` | on | off | off (G-2; the class body compares the name, as rows 1 and 2 do: it cannot import `core.editions`) |
| 6 | `config.py:2823` | C | Boot refuses `CLI_RUNTIME_ENABLED` (reads `session_mode` since US-103) | `session_mode` | allowed | refused | refused |
| 7 | `config.py:2830` | C | Boot requires the Clerk settings (reads `sign_in` since US-103) | `sign_in` (`clerk`) | not required | required | not required; the OIDC settings and the public address are required, then the edition is refused as not in this build (US-103, `core/edition_start_up.py`) |
| 8 | `config.py:2844` | C | Boot requires `DEFAULT_WORKSPACE_ID` (the anonymous session's workspace; reads `anonymous_operator` since US-103) | `anonymous_operator` | required | not required | not required by this check; the enterprise seed path (§9) owns its default workspace |
| 9 | `config.py:2854` | P | Hands the edition to the remote-access boot decision (row 81) | `anonymous_operator` | see 81 | see 81 | see 81 |

### `alembic/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 10 | `alembic/versions/auto_haiku_default_20261019.py:86` | F | Keeps a self-hoster's own platform default model instead of writing the new one | `hosts_automatos_cloud` | kept | overwritten | kept |
| 11 | `alembic/versions/auto_haiku_default_20261019.py:172` | P | Passes the edition to row 10 | `hosts_automatos_cloud` | see 10 | see 10 | see 10 |

A migration is history. Rewriting it to read a capability changes what an old revision does, so the guard test (`orchestrator/tests/test_prd277_no_raw_edition_checks.py`) exempts `alembic/versions/` and these two rows stay as they are.

### `api/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 12 | `api/admin_credits.py:65` | C | Admin credit adjustments answer 404 | `billing` | 404 | served | 404 |
| 13 | `api/admin_theme_revocation_sync.py:63` | H | Revocation sync status and manual pull answer 404 | `calls_theme_revocations` | served | 404 | served only while revocations are switched on; else 404, "its operator hasn't switched it on" (US-704) |
| 14 | `api/billing.py:46` | C | Every billing route answers 404 | `billing` | 404 | served | 404 |
| 15 | `api/billing_webhooks.py:69` | C | Stripe webhook answers 404 | `billing` | 404 | served | 404 |
| 16 | `api/llm_analytics.py:887` | H | OpenRouter balance read from the key the calls actually use (the platform key), not only a workspace's own key | `billing` (off: the owner pays the provider) | platform key | workspace key only | platform key |
| 17 | `api/marketplace_catalog.py:101` | H | Cloud catalogue routes answer 404 | `hosts_automatos_cloud` | 404 | served | 404 |
| 18 | `api/marketplace_catalog.py:107` | H | Cloud catalogue file route answers 404 | `hosts_automatos_cloud` | 404 | served | 404 |
| 19 | `api/marketplace_catalog_items.py:97` | H | Cloud bundle and plugin package routes answer 404 | `hosts_automatos_cloud` | 404 | served | 404 |
| 20 | `api/marketplace_cloud_items.py:51` | H | "Items brought from Cloud" routes answer 404 | `hosts_automatos_cloud` | served | 404 | served whatever `ENTERPRISE_CALLS_CLOUD_ITEMS` holds: they read only what the install holds, so an item brought earlier is still listed and reviewed (US-704) |
| 21 | `api/marketplace_cloud_sync.py:73` | H | Cloud Sync routes answer 404 | `calls_cloud_sync` | served | 404 | served only while Cloud sync is switched on; else 404 before the caller is looked at (US-704) |
| 21a | `api/marketplace_cloud_sync.py:81` | H | Who may run a Sync: past `workspace:manage`, an install admin, then a fresh step-up | `operator_is_user` (and the edition's `step_up`) | the operator, no install-admin check; step-up `skip` | 404 (row 21) | an install admin (`caller_is_admin`) who signed in again within `STEP_UP_MAX_AGE_MINUTES`; the status read stays `workspace:manage` (US-704) |
| 22 | `api/marketplace_themes_submit.py:145` | C | A submitter with no `users` row is refused | `anonymous_operator` | allowed | 403 | 403 |
| 23 | `api/mobile_devices.py:216` | H | Minting a pairing code answers 404; the same guard also gates `POST /pair` through `_require_paired_mode` | `device_pairing` | served | 404 | served when pairing is on (D-7; Wave 6) |
| 24 | `api/mobile_push.py:105` | H | Push-consent read and write answer 404 | `push_needs_consent` | served | 404 | served (G-1) |
| 25 | `api/profile.py:119` | C | `editable` in the profile answer; the same check chooses the email note | `profile_editable` | true | false | false |
| 26 | `api/profile.py:121` | R | `edition` in the profile answer | (name) | `local` | `saas` | `enterprise` |
| 27 | `api/profile.py:158` | C | Profile update refused ("managed by your identity provider") | `profile_editable` | allowed | 403 | 403 |
| 28 | `api/public_marketplace.py:76` | H | Public Marketplace answers 404 | `hosts_automatos_cloud` | 404 | served | 404 |
| 29 | `api/themes_revocations_feed.py:51` | H | Theme revocation feed answers 404 | `hosts_automatos_cloud` | 404 | served | 404 |
| 30 | `api/workspaces.py:133` | C | Workspace exposure: unrestricted, or from the plan | `billing` | unrestricted | from plan | unrestricted. Its label says "Local edition" today; the wording for `enterprise` needs the owner's choice before it changes |

### `consumers/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 31 | `consumers/chatbot/service.py:872` | C | Chat seeds the driving person's known name into Auto's state | `chat_seeds_user_name` | seeded | not seeded (prompt kept byte-identical) | seeded (G-6) |

### `core/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 32 | `core/auth/bearer_registry.py:92` | P | Whether remote access has closed the anonymous lane (row 82) | `anonymous_operator` | see 82 | see 82 | see 82 |
| 33 | `core/auth/device.py:220` | H | An `amd_` bearer is refused with 401 | `device_pairing` | resolved | 401 | resolved when pairing is on, as the person who paired (FR-22; Wave 6) |
| 34 | `core/auth/hybrid.py:985` | C | The anonymous session is bound to the operator row and made super admin | `anonymous_operator` | super admin | plain user | never reached: no anonymous lane (FR-3) |
| 35 | `core/auth/step_up.py:161` | C | `check_step_up` (line 177) passes every request | `step_up` | `skip` | `clerk_freshness` | `reauth` (US-406) |
| 36 | `core/llm/byok_endpoint.py:37` | C | `_saas()` (now `_guarded()`, and its callers take `guarded` instead of an edition), used at lines 55, 84 and 96: a key's own endpoint must be `https`, public and pinned, and is required even when `AZURE_OPENAI_ENDPOINT` is set | `guards_user_endpoints` | not guarded | guarded | guarded (G-3) |
| 37 | `core/llm/providers.py:259` | F | A `byok_only` provider (NVIDIA) may hold a platform key | `billing` (on: Automatos would be reselling the key) | allowed | refused | allowed |
| 38 | `core/llm/providers.py:419` | R | `edition` in `GET /api/keys/providers`, and passed to row 37 | (name) | `local` | `saas` | `enterprise` |
| 39 | `core/llm/usage_tracker.py:76` | C | A usage row with no workspace is booked to `DEFAULT_WORKSPACE_ID` | `multi_person` | booked to it | dropped | dropped |
| 40 | `core/local_projects_mount.py:50` | H | Reads of `projects/` are served from the projects mount's workspace, the operator's own folder | `operator_is_user` | mounted when set | never | never (G-2) |
| 41 | `core/media_render_quota.py:104` | C | Monthly render quota | `billing` | none | from plan | none |
| 42 | `core/media_render_quota.py:218` | C | Plan label on the render quota | `billing` | none | plan name | none |
| 43 | `core/observability/otel.py:257` | R | `automatos.edition` trace attribute | (name) | `local` | `saas` | `enterprise` |
| 44 | `core/seeds/fingerprint.py:85` | R | `AUTH_EDITION` is part of the seed fingerprint (left as is: the guard test's one named exemption) | (name) | hashed | hashed | hashed |
| 45 | `core/seeds/seed_local_first_run.py:763` | C | First-run seed: workspace row, operator row, Auto, starter roster, demo Playbook, welcome Deliverable | `seeds_first_run_content` | seeded | skipped | skipped (G-4) |
| 46 | `core/session_permission_modes.py:39` | F | A session's default permission mode | `session_mode` | `auto` | `edits` | `edits` (session mode is off; moot) |
| 47 | `core/workspace_client.py:76` | H | A worker failure adds "start it with `docker compose up -d workspace-worker`" | `operator_is_user` | hint | no hint | no hint (G-2) |
| 48 | `core/workspaces/owner.py:53` | H | A workspace with no owner falls back to the operator row | `anonymous_operator` | operator | none | none |

### `main.py`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 49 | `main.py:1350` | R | `edition` in `/health`, which the CLI host reads (row S1) | (name) | `local` | `saas` | `enterprise` |

### `modules/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 50 | `modules/context/sections/product_facts.py:90` | H | Auto's description of the install, and its Settings tabs | `install_description` | local text | hosted text | organisation text, the hosted tabs less Billing and Widget SDK (G-5; the text is a draft for the owner's wording) |
| 51 | `modules/documents/workspace_documents.py:109` | H | A generated document is copied into the workspace's `documents/` folder | `operator_is_user` | copied | skipped | skipped (G-2) |
| 52 | `modules/socials/render.py:161` | C | `_local_edition()` (line 166): "start the media profile with `docker compose`" | `operator_is_user` | compose hint | "not configured" / "unreachable" | "not configured" / "unreachable" (G-2) |
| 53 | `modules/tools/discovery/handlers_onboarding.py:235` | H | Onboarding skips `powerup` (the bring-your-own-key card after the trial) | `billing` | skipped | shown | skipped |
| 54 | `modules/tools/discovery/platform_executor.py:447` | H | A chat turn with no Clerk id may still count as an owner's instruction, by the driving user's role | `sign_in` | by driving user | refused | by the person's `users.id` (Wave 2 resolver) |
| 55 | `modules/tools/discovery/ticket_cancel.py:129` | H | `auth_type` of the context used to check the driver's permission: `anonymous` makes the driver the workspace's owner | `anonymous_operator` | `anonymous` | `clerk` | never `anonymous`; the enterprise session's type (Wave 2) |
| 56 | `modules/voice/arming.py:54` | C | Name of a newly created Retell agent | `live_agent_named_apart` | "Auto Live (local)" | "Auto Live" | "Auto Live" (G-7) |
| 57 | `modules/voice/mint.py:98` | H | Auto Live allowed by the plan | `billing` | always | by plan | always |
| 58 | `modules/voice/mint.py:111` | H | Auto Live refused for plan or exhausted credits | `billing` | never | by plan and credits | never |

### `scripts/`

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 59 | `scripts/mint_pairing_code.py:157` | H | The command-line pairing code is refused | `device_pairing` | minted | refused | minted when pairing is on (Wave 6). Its refusal text still says "local-edition"; Wave 6 corrects it |

### `services/` (inside `orchestrator/`)

| # | Where | Kind | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|---|
| 60 | `services/agent_quota.py:61` | H | Plan agent limit | `billing` | unlimited | by plan | unlimited |
| 61 | `services/ai_providers.py:271` | H | The session CLI counted as a provider | `session_mode` | counted when on | none | none |
| 62 | `services/ai_providers.py:312` | H | Voice allowed by the plan's families | `billing` | always | by plan | always |
| 63 | `services/board_consent.py:84` | P | `edition()` reads the flag for row 64 | `multi_person` | see 64 | see 64 | see 64 |
| 64 | `services/board_consent.py:104` | F | Filing a ticket is its approval (`creation_is_consent`) | `multi_person` | yes | no | no |
| 65 | `services/cli_host_service.py:627` | P | Session mode settings' default permission mode (row 46) | `session_mode` | see 46 | see 46 | see 46 |
| 66 | `services/cli_host_service.py:673` | P | A claim's default permission mode (row 46) | `session_mode` | see 46 | see 46 | see 46 |
| 67 | `services/content_reports/moderation.py:106` | H | Content reports are posted to the moderation inbox | `hosts_automatos_cloud` | never | when `MODERATION_WEBHOOK_URL` is set | never |
| 68 | `services/credit_watch.py:54` | H | The AI provider's balance is watched and the bell warns before it runs out | `billing` (off: the owner pays the provider) | watched | not watched | watched |
| 69 | `services/credits/alerts.py:103` | P | Credit alert level (row 72) | `billing` | see 72 | see 72 | see 72 |
| 70 | `services/credits/gate.py:45` | C | A call is exempt from the credit gate | `billing` | exempt | gated | exempt |
| 71 | `services/credits/jobs.py:39` | C | Meter, overhead and reset jobs run (`_metered_edition`, used at 44, 60, 75) | `billing` | skipped | run | skipped |
| 72 | `services/credits/routing.py:132` | F | `funding_for`: who pays for platform-key calls | `billing` | `FUNDED_BY_LOCAL` | by plan and balance | `FUNDED_BY_LOCAL` |
| 73 | `services/credits/routing.py:147` | C | `workspace_funding` short-cut | `billing` | `FUNDED_BY_LOCAL` | by plan and balance | `FUNDED_BY_LOCAL` |
| 74 | `services/credits/routing.py:156` | P | Passes the edition to row 72 | `billing` | see 72 | see 72 | see 72 |
| 75 | `services/credits/meter.py:112` | P | A call's debit (row 72) | `billing` | see 72 | see 72 | see 72 |
| 76 | `services/credits/summary.py:58` | P | A product workspace's billing summary (row 72) | `billing` | see 72 | see 72 | see 72 |
| 77 | `services/credits/summary.py:80` | P | A workspace's billing summary (row 72) | `billing` | see 72 | see 72 | see 72 |
| 78 | `services/marketplace_cloud/pull.py:133` | H | Cloud sync enabled (the Sync, and a Cloud theme's package download, `package._download`) | `calls_cloud_sync` | when `AUTOMATOS_CLOUD_URL` is set | never | only while switched on, with the URL set (D-12, US-704) |
| 79 | `services/mobile/discovery.py:166` | H | Discovery's `auth.modes` and `pairing` blocks | `sign_in` | `["device"]`, pairing by mode | `["clerk"]` | `["oidc"]`, pairing by setting (FR-23; Wave 6) |
| 80 | `services/mobile/discovery.py:192` | R | `edition` in the discovery document | (name) | `local` | `saas` | `enterprise` |
| 80a | `services/mobile/pairing.py:97` | H | Creating or redeeming a pairing code is refused (`_require_local_edition`) | `device_pairing` | allowed | refused | allowed when pairing is on (D-7; Wave 6) |
| 81 | `services/mobile/remote_access.py:83` | F | `LOCAL_REMOTE_ACCESS` boot decision (refuse, warn, start) | `anonymous_operator` | applies | skipped | skipped (no anonymous lane to protect) |
| 82 | `services/mobile/remote_access.py:119` | F | Remote access closes the anonymous lane | `anonymous_operator` | closed when `paired` | not closed here: an anonymous request is never the operator, and `REQUIRE_AUTH` decides whether it is served | not closed here: `REQUIRE_AUTH` is forced on, so it is refused |
| 83 | `services/mobile/urls.py:50` | H | With no `PUBLIC_API_HOST`, phones are told the request's own base URL instead of the hosted API | `hosts_automatos_cloud` | request's URL | `PUBLIC_API_HOST`, whatever it holds | request's URL (never point phones at the hosted API) |
| 84 | `services/onboarding_state.py:579` | H | Stage after the payoff | `billing` | `completed` | `powerup` | `completed` |
| 85 | `services/push/consent.py:78` | H | `expo_push_allowed` | `push_needs_consent` | once agreed | always | once agreed (G-1) |
| 86 | `services/push/fanout.py:174` | H | Which device kind receives a push | `device_pairing` | `paired` | `clerk` | `paired` now (D-7); `oidc` too after US-603 |
| 87 | `services/push/recipients.py:92` | H | The operator row holds the owner role for push recipients | `anonymous_operator` | yes | no | no |
| 88 | `services/readiness/probes.py:105` | H | Push readiness: always ready, or ready once agreed | `push_needs_consent` | once agreed | always | once agreed (G-1) |
| 89 | `services/readiness/service.py:171` | H | Readiness detail adds the host and latency reached | `operator_is_user` | shown | hidden | hidden (G-2) |
| 90 | `services/socials_plan_research.py:133` | C | Background Socials research allowed without credits | `billing` | always | by credits | always |
| 91 | `services/theme_revocation_scheduler.py:85` | H | The revocation pull is scheduled | `calls_theme_revocations` | when a feed URL is set | never | only while switched on, with a feed URL set; else no job is registered (D-12, US-704) |
| 92 | `services/themes/membership.py:72` | C | The anonymous operator is a member of every workspace | `anonymous_operator` | yes | no | no |
| 93 | `services/themes/records.py:132` | H | Private theme uploads default under `auto` | `multi_person` | on | on only in a one-person workspace | on only in a one-person workspace |
| 94 | `services/themes/review.py:185` | H | Approving a Marketplace theme needs a passing isolated-review note | `hosts_automatos_cloud` | not needed | needed | not needed (the admin's approval is the review) |
| 95 | `services/themes/revocation_pull.py:123` | H | The revocation pull is enabled | `calls_theme_revocations` | when a feed URL is set | never | only while switched on, with a feed URL set (D-12, US-704) |
| 96 | `services/themes/revocation_sync_state.py:81` | H | Revoked key ids read from the stored pull | `hosts_automatos_cloud` | stored ids | none | stored ids, whether or not revocations are switched on now: a key a pull revoked stays revoked once they are off (US-704) |
| 97 | `services/workspace_move/export.py:125` | R | `edition` in a workspace export's manifest | (name) | `local` | `saas` | `enterprise` |
| 98 | `services/workspace_purge.py:232` | C | Purging an owner's last workspace also deletes their account | `last_workspace_purge_deletes_person` | never | yes | never (G-8) |

### Shell

| # | Where | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|
| E1 | `orchestrator/docker-entrypoint.sh:277` | Creates the local workspace and the operator row (id 1) at boot | `anonymous_operator` | created | skipped | skipped: the line runs only for exactly `local`, so `enterprise` creates no operator row (checked in US-103, unchanged); the default workspace with no owner is US-405's |

Shell cannot import the Python module. When this line changes, it reads a value the Python side derives, not the edition name.

## `services/` (repository root: separate containers)

| # | Where | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|
| S1 | `services/cli-host/automatos_cli_host/host.py:74` | The CLI host refuses any backend whose `/health` `edition` is not `local` | `session_mode` | served | refused | refused, which is correct (session mode is off) |

No other container reads the edition. The CLI host compares the name the backend reports (row 49). When it moves, `/health` should report `session_mode` and the host should read that.

## `frontend/`

The frontend has its own seam, `frontend/lib/auth-edition.ts`, fixed at build time by `NEXT_PUBLIC_AUTH_EDITION`. It exports `isSaaS` / `isLocal`, and 56 files outside tests import from it. These rows move in Wave 5 (US-501 to US-505), against a frontend mirror of the same capability list. Import lines and comments are left out. Lines that share one switch are grouped.

| # | Where | What it switches | Capability | `local` | `saas` | `enterprise` |
|---|---|---|---|---|---|---|
| W1 | `lib/auth-edition.ts:27, 31, 32, 33` | Parses the flag (`local`, else `saas`) and defines `isSaaS` / `isLocal` | (the seam) | `local` | `saas` | accepted from Wave 5 |
| W2 | `lib/auth-edition.ts:82` | `SAAS_ONLY_ROUTES` hidden: `/admin/workspaces`, `/team`, `/accept-invitation`, `/sign-in`, `/sign-up`, `/reset-password`, `/sso-callback`, `/admin/plugins`, `/dev/reset-onboarding` | per route | hidden | shown | per route, answered for Wave 5 (G-9) |
| W3 | `lib/auth-edition.ts:108` | Nav exposure ignores the plan | `billing` | ignored | from plan | ignored |
| W4 | `lib/auth-hooks.ts:72-76` | `useAuth`, `useUser`, `useSession`, `useOrganization`, `useClerk`: Clerk or local stubs | `sign_in` | stubs | Clerk | OIDC provider (US-501) |
| W5 | `middleware.ts:49` | Clerk middleware or the open local one | `sign_in` | open | Clerk | session cookie (US-502) |
| W6 | `components/providers.tsx:35` | `ClerkProvider` mounted | `sign_in` | no | yes | enterprise provider (US-501) |
| W7 | `app/sign-in/[[...rest]]/page.tsx:9` | Sign-in page, or home | `sign_in` | home | Clerk | OIDC start (US-502) |
| W8 | `app/sign-up/[[...rest]]/page.tsx:9` | Sign-up page, or home | `sign_in` | home | Clerk | sign-in (no self sign-up) |
| W9 | `components/auth/resume-after-sign-in.tsx:61` | Resume where the person was after sign-in | `sign_in` | off | on | on (US-502) |
| W10 | `components/auth/user-profile-button.tsx:91` | Profile slot: operator profile or Clerk | `sign_in` | operator | Clerk | enterprise profile (US-504) |
| W11 | `components/layout/header.tsx:104`, `components/layout/studio-header.tsx:116` | `ProfileMenu` (Clerk) or `UserProfileButton` | `sign_in` | button | Clerk menu | enterprise menu (US-504) |
| W12 | `lib/theme-host/boot-server.ts:110` | Theme host's user and credential on the server | `sign_in` | local operator | Clerk | session (US-503) |
| W13 | `app/api/themes/files/[...path]/route.ts:83` | Theme files route forwards the device cookie or the Clerk token | `sign_in` | device cookie | Clerk | session cookie (US-503) |
| W14 | `lib/theme-host/api.ts:199` | Step-up hook | `step_up` | no-op | Clerk reverification | sign in again (US-406) |
| W15 | `app/settings/profile/page.tsx:25` | Local profile form or Clerk's profile page | `profile_editable` | form | Clerk | read-only, managed by the IdP (US-504) |
| W16 | `components/settings/SettingsPanel.tsx:85, 124` | Profile tab | `profile_editable` | shown | hidden | hidden, or a read-only profile (US-504) |
| W17 | `contexts/role-context.tsx:73` | The operator is `super_admin` | `anonymous_operator` | yes | from Clerk | from the person's stored role (Wave 2) |
| W18 | `hooks/use-may-manage-knowledge.ts:17` | Anyone may manage Knowledge | `anonymous_operator` | yes | owner or admin | owner or admin |
| W19 | `components/theme-host/theme-state-scope.tsx:62` | Theme state keyed to the operator or to the user id | `anonymous_operator` | operator key | user id | user id |
| W20 | `components/activity/board/ticket-actions.ts:61` | Who acted: "you" or "a workspace member" | `multi_person` | "you" | member | member |
| W21 | `stores/chat-session-store.ts:65, 111, 112` | Open chat tabs saved on the server too | `multi_person` | browser only | browser and server | browser and server |
| W22 | `components/billing/credit-balance-pill.tsx:83`; `components/layout/header.tsx:93`; `components/layout/studio-header.tsx:60` | Credit balance chip | `billing` | hidden | shown | hidden |
| W23 | `hooks/use-billing-ledger.ts:74`; `hooks/use-billing-payment.ts:68, 78`; `hooks/use-billing-plan-change.ts:62`; `hooks/use-billing-plans.ts:105`; `hooks/use-billing-summary.ts:90, 97` | Billing queries enabled | `billing` | off | on | off |
| W24 | `components/onboarding/power-up-card.tsx:69` | Power-up card | `billing` | hidden | shown | hidden |
| W25 | `csp-sources.js:86`; `next.config.js:25` | Stripe hosts in the content security policy | `billing` | none | Stripe | none |
| W26 | `components/agents/runtime-section.tsx:474, 535, 546` | Session-mode runtime controls | `session_mode` | shown | hidden | hidden |
| W27 | `components/marketplace/marketplace-session-gaps.tsx:29` | Session-mode gaps on Marketplace items | `session_mode` | shown | hidden | hidden |
| W28 | `components/settings/SettingsPanel.tsx:86, 155` | Session mode tab | `session_mode` | shown | hidden | hidden |
| W29 | `components/settings/SettingsPanel.tsx:103` | Settings tabs limited to the local list (hides Webhooks, API keys, Billing, Widget SDK) | per tab | limited | all | Webhooks and API keys shown, Billing and Widget SDK hidden (G-10, Wave 5) |
| W30 | `app/robots.ts:28` | Crawlers refused everywhere | `hosts_automatos_cloud` | refused | allowed by rules | refused |
| W31 | `components/admin/themes/review-queue.tsx:65` | Review queue in hosted mode (isolated-review note) | `hosts_automatos_cloud` | local | hosted | local (row 94) |
| W32 | `components/marketplace/public/marketplace-page.tsx:63` | Public Marketplace or the member Marketplace | `hosts_automatos_cloud` | member | public | member |
| W33 | `lib/marketplace/share-card-read.ts:151` | Share card metadata | `hosts_automatos_cloud` | local metadata | read | local metadata |
| W34 | `components/marketplace/cloud/cloud-item-badges.tsx:48`; `cloud-items-layer.tsx:107`; `sync-button.tsx:74` | Cloud item badges and layer; Sync button | `holdsCloudItems()` (lib/hosted-services.ts); `useHostedService('cloudSync')` (US-704) | shown | hidden | badges and layer always, as `local`; the button while Cloud sync is on |
| W35 | `components/settings/SettingsPanel.tsx:100`; `components/settings/automatos-cloud-panel.tsx:50, 130` | Automatos Cloud tab and its panel; the panel's lede | `useHostedService('cloudSync')`, through `useHasAutomatosCloudTab`, the tab's one gate; the lede `useHostedService('cloudItems')` (US-704) | shown | hidden | shown while Cloud sync is on; the lede says a Sync brings themes only while Cloud items are off |
| W36 | `hooks/use-cloud-items.ts:53, 68, 81` | Cloud item queries enabled | `holdsCloudItems()` (US-704) | on | off | on, as `local` |
| W37 | `lib/marketplace/cloud-install-gate.ts:145` | Install gate for Cloud items | `holdsCloudItems()` (US-704) | checked | none | checked, as `local`: an item brought earlier is still reviewed |
| W37a | `components/admin/themes/revocations-panel.tsx:131` | The revocation sync section and whether a feed row can be lifted | `useHostedService('themeRevocations')`, through `pullsRevocations` (US-704) | shown | hidden | shown while revocations are on; one plain "not switched on" line, no status asked and feed rows liftable while off |
| W38 | `hooks/use-mobile-devices.ts:123` | Compares the discovery document's `edition` with `local` | `device_pairing` | pairing read | none | pairing by setting (Wave 6) |
| W39 | `hooks/use-mobile-devices.ts:185, 187, 188, 193, 198, 199` | This browser's device token and pairing prompt | `device_pairing` | on | off | by setting (Wave 6) |
| W40 | `components/settings/phones-devices/pair-phone-qr.tsx:59`; `phones-devices-tab.tsx:28` | Pairing QR and "browsers paired to it" copy | `device_pairing` | shown | hidden | by setting (Wave 6) |
| W41 | `components/settings/phones-devices/phones-devices-tab.tsx:52`; `hooks/use-push-consent.ts:64` | Push consent switch and query | `push_needs_consent` | shown | hidden | shown (G-1) |
| W42 | `components/settings/ChannelsSettingsTab.tsx:164, 172` | A channel that can poll starts on polling; webhook-only channels warn that inbound needs a public URL | per channel | polling, warned | webhook | polling, warned (G-11, Wave 5) |
| W43 | `components/local/first-run-nudge.tsx:77` | "Add a model key" nudge over the seeded demo | `seeds_first_run_content` (row 45) | shown | hidden | hidden: no demo to nudge over (G-4) |
| W44 | `lib/theme-sdk/context.ts:125` | `useEdition()` gives themes the edition name, typed `'local' \| 'saas'` | (name) | `local` | `saas` | `enterprise`: a change to the theme SDK contract |
| W45 | `lib/otel/node.ts:99` | `automatos.edition` trace attribute | (name) | `local` | `saas` | `enterprise` |
| W46 | `Dockerfile:81, 94` | Build argument, default `saas` | (the flag) | `local` | `saas` | an enterprise image (US-505) |

Two response types also name only two editions and widen with the value: `lib/api-client.ts:434` (profile `edition: 'local' | 'saas'`) and `hooks/use-provider-registry.ts:49`.

The phone app (`automatos-go`) reads discovery's `edition` and `auth.modes`. That is the companion PRD's (US-604), not this repository's.

## For Gerard

These are the rows where the enterprise answer was not obvious from the PRD or its decisions. Every one is answered (US-103); each answer is in bold after its options. G-1 to G-8 are in `core/editions.py`; G-9 to G-11 are frontend rows, answered here for Wave 5.

Since US-102 each backend row here reads its own capability, pinned to today's `local` and `saas` answers: G-1 `push_needs_consent`, G-2 `operator_is_user`, G-3 `guards_user_endpoints`, G-4 `seeds_first_run_content` (row 45; E1 is shell and W43 frontend), G-5 `install_description`, G-6 `chat_seeds_user_name`, G-7 `live_agent_named_apart`, G-8 `last_workspace_purge_deletes_person`. G-9 to G-11 are frontend rows only (Wave 5). An answer sets the capability's `enterprise` value; option C of G-2 splits `operator_is_user`.

**G-1. Expo push consent** (rows 24, 85, 88; W41). On `local`, nothing leaves for Expo's push service until the operator agrees in Settings. On `saas` push is always on, through the platform's Expo project. D-12 covers Automatos's own hosted services; Expo is a third party, so D-12 does not settle it.
- A. Consent required, as `local`: an enterprise install sends nothing to Expo until its operator agrees. `push_needs_consent` = yes.
- B. Always on, as `saas`: phones get notifications from the first pairing. `push_needs_consent` = no.
**Answered: A.** Consent required, as `local`.

**G-2. Operator detail and fix commands** (rows 5, 40, 47, 51, 52, 89). On `local` the person using the app runs the server: they see `docker compose` commands to start the worker or the renderer, the host and latency each readiness probe reached, web access on by default, and generated documents copied into their own deliverables folder. An enterprise install runs from the chart in the customer's cloud, and its users are staff, not the operator.
- A. All off, as `saas`: staff see the hosted messages, no host detail, web access off until set, no `documents/` copy. `operator_is_user` = no.
- B. All on, as `local`. `operator_is_user` = yes. The compose commands would be wrong on a chart install.
- C. Split: answer each row separately (for example web access on, documents copied, but no compose commands and no host detail). `operator_is_user` then splits into one capability per answer.
**Answered: A.** All off, as `saas`.

**G-3. A workspace's own AI endpoint** (row 36). On `saas` an endpoint saved with a key (Azure today) must be `https` on a public address and is called through the pinned client, and the platform's `AZURE_OPENAI_ENDPOINT` never stands in for a workspace. On `local` none of that applies. An enterprise customer's Azure endpoint may well be on a private network.
- A. Guarded, as `saas`: a staff member cannot point the server at an internal address, and a private endpoint cannot be used. `guards_user_endpoints` = yes.
- B. Not guarded, as `local`: any endpoint is called, private addresses included, and the install's `AZURE_OPENAI_ENDPOINT` can stand in. `guards_user_endpoints` = no.
- C. Guarded, with named private hosts allowed through a new setting, as FR-9 does for the IdP. New code in a later wave.
**Answered: A.** Guarded, as `saas`. PRD-282 adds the named private hosts.

**G-4. First-run content** (row 45; E1; W43). On `local` every boot seeds the default workspace, the operator row, Auto, a three-agent starter roster, a demo Playbook and a welcome Deliverable. §9 says the enterprise first-run seed needs its own path: "a default workspace, no operator".
- A. None of the demo content: the enterprise seed creates only the default workspace (and Auto).
- B. The demo content in the default workspace, without the operator row. The seed is split so the operator row is skipped.
**Answered: A.** No demo content. The enterprise seed (a default workspace with no owner) is US-405's.

**G-5. Auto's description of the install** (row 50). Auto tells people what the install is and which Settings tabs it has. The local text says "no sign-in, accounts, teams, plans or billing"; the hosted text says each person signs in and "the workspace's plan sets how many helpers it may have". Neither is true of enterprise.
- A. A third text for enterprise (signs in with a work account, no plans or billing) and its own tab list. The wording is yours.
- B. The hosted text unchanged, which mentions plans that do not exist there.
**Answered: A.** A third value, `organisation`: the text in `modules/context/sections/product_facts.py` is a draft for the owner's wording; its tabs are the hosted ones less Billing and the Widget SDK.

**G-6. Auto knows the person's name** (row 31). On `local`, chat seeds the driving person's name from their `users` row into Auto's state. On `saas` it is off so the hosted prompt stays byte-identical (an owner rule for that wave). An enterprise person's name comes from the IdP at sign-in.
- A. On in enterprise. A capability on for `local` and `enterprise`, off for `saas`.
- B. Off in enterprise, as `saas`. The row maps to `anonymous_operator`.
**Answered: A.** On in enterprise.

**G-7. The Retell agent's name** (row 56). A new Auto Live agent is named "Auto Live (local)" on `local` and "Auto Live" on `saas`, in the account of whoever owns the Retell key.
- A. "Auto Live", as `saas`.
- B. "Auto Live (local)", as `local`. The suffix would be wrong on an enterprise install.
- C. A new suffix for enterprise only, such as "Auto Live (self-hosted)". `local` keeps its name, because this PRD changes nothing for `local`.
**Answered: A.** "Auto Live", as `saas`.

**G-8. Purging an owner's last workspace** (row 98). On `saas`, purging the last workspace a person owns or belongs to also deletes their `users` row (and their Clerk user). On `local` it never does, because the operator must survive. In enterprise a person may legitimately be in no workspace (D-4), and access ends by session lifetime and an admin's deactivate (D-9).
- A. Never delete the person, as `local`: the person stays and an admin deactivates them if needed.
- B. Delete the person, as `saas`: their next sign-in creates them again with no workspace. There is no Clerk user to delete.
**Answered: A.** Never delete the person.

**G-9. Web routes hidden outside `saas`** (W2). `SAAS_ONLY_ROUTES` is one list. Enterprise needs an answer per route. The obvious ones: `/team` shown (several people, D-5); `/sign-in` shown (OIDC); `/sign-up`, `/reset-password` hidden (the IdP owns accounts); `/accept-invitation` hidden (invitations are off, §4). Not obvious:
- `/admin/workspaces` (cross-workspace administration): A shown to install admins, B hidden as on `local`.
- `/admin/plugins`: A shown to install admins, B hidden as on `local`.
- `/sso-callback` (Clerk's callback): hidden unless the OIDC callback reuses the path. US-502 decides the callback path.
- `/dev/reset-onboarding`: A hidden, B shown.
**Answered in Wave 5:** `/admin/workspaces` and `/admin/plugins` shown to install admins; `/dev/reset-onboarding` and `/sso-callback` (Clerk's callback) hidden.

**G-10. Settings tabs hidden on `local`** (W29). The local tab list leaves out Webhooks, API keys, Billing and Widget SDK. Billing is hidden in enterprise (FR-24). For each of Webhooks, API keys and Widget SDK: A shown, as `saas`; B hidden, as `local`.
**Answered in Wave 5:** Webhooks and API keys shown; Billing and the Widget SDK hidden.

**G-11. Channels on an install the internet may not reach** (W42). On `local` a channel that can poll (Telegram) starts on polling, and a webhook-only channel warns that inbound messages need a public URL. An enterprise install may sit inside the customer's network.
- A. As `local`: polling first and the warning shown.
- B. As `saas`: webhooks first, no warning.
- C. Decided by a setting that says whether the install is reachable from the internet. New code.
**Answered in Wave 5: A.** Polling first and the warning shown, as `local`.

## Proposed order for moving the call sites

Each pull request moves the call sites of one area to `core.editions` and deletes the comparison, with no behaviour change for `local` or `saas`. Rough counts of lines touched:

1. **Billing and plans:** rows 12, 14 to 16, 30, 37, 41, 42, 53, 57, 58, 60, 62, 68 to 77, 84, 90: 25 call sites. `funding_for` and `platform_key_allowed` take a capability instead of an edition, which also changes row 38. About 120 changed lines.
2. **Automatos Cloud, host and client:** rows 13, 17 to 21, 28, 29, 67, 78, 83, 91, 94 to 96: 15 call sites, plus rows 10 and 11 if the migration is not exempted. About 70 lines.
3. **Auth and the operator:** rows 22, 23, 25, 27, 32 to 35, 39, 48, 54, 55, 59, 63, 64, 79, 80a, 81, 82, 86, 87, 92, 93: 23 call sites. `remote_access` and `board_consent` take a capability. About 110 lines. Needs Dave's review: it touches authentication.
4. **Session mode:** rows 46, 61, 65, 66, and S1 with a `session_mode` field in `/health`: 5 call sites, about 30 lines. Small enough to ride with pull request 3.
5. **The guard and the helpers:** the "no raw comparison" test (US-102), deleting `IS_LOCAL_EDITION` / `IS_SAAS_EDITION` once nothing reads them, and one accessor for the R rows that report the name. About 60 lines plus the test.
US-102 is complete: outside `config.py`, `core/editions.py`, `alembic/versions/` and the tests, no backend file compares `AUTH_EDITION` or reads `IS_LOCAL_EDITION` / `IS_SAAS_EDITION` (both deleted), and `orchestrator/tests/test_prd277_no_raw_edition_checks.py` fails if one does; row 44 is its one named exemption. Left for US-103: the `enterprise` row of the capability table, the For Gerard answers, `config.py`'s rows 1, 2 and 5 to 9, and the shell line E1.

6. **US-103 (done):** the `enterprise` column and the answers above. In `config.py`, rows 6 to 8 read capabilities; rows 1, 2 and 5 stay name comparisons, because the class body runs before `core` can be imported (a test holds row 1's names to `KNOWN_EDITIONS`). The new boot rules are in `core/edition_start_up.py`, not in `config.py`, which is over the 800-line limit.

## US-103: what the `enterprise` row makes the `sign_in` sites do

Start-up refuses `enterprise` until Wave 4. These are the places that would read `oidc` if it ran, and why each is safe:

| Where | With `oidc` | Why it is safe |
|---|---|---|
| `modules/tools/discovery/platform_executor.py` (row 54) | `!= SIGN_IN_NONE` is true, so a chat turn with no Clerk id keeps its confirmation card | Fails closed: it refuses the owner's-instruction shortcut. Unchanged; Wave 2's resolver decides it |
| `services/mobile/discovery.py` (row 79) | Raises `DiscoveryUnavailable` (the route answers 500, "not set up") | Was: the Clerk block, offering phones a sign-in the install would refuse. Now compares to `SIGN_IN_CLERK`; US-602 adds `oidc` |
| `core/auth/step_up.py` (row 35, `step_up`) | `reauth` refuses every context with `step_up_unavailable` | Was: Clerk's `fva` path. Now only `clerk_freshness` reads it; US-406 builds signing in again |
| `core/auth/clerk.py` `verify_token` (the Clerk lane, invitation accept, the workspace eraser) | Returns nothing: a Clerk token is never trusted | Was: any Clerk token verified wherever Clerk's settings were set. Now only where `sign_in` is `clerk` or `none` (local unchanged) |
| `core/auth/clerk.py`, every call to Clerk's API (user lookup, organisation members, organisation invitation, user invitation, invitation revoke, organisation member removal, user deletion, user list) | Sends nothing: a lookup answers as with no secret set (`None`, `[]`, `False`), a change raises `ClerkNotUsed`, a `ValueError` like an unset secret | Only where `sign_in` is `clerk` or `none` (`clerk_in_use`; local unchanged), whatever Clerk settings the environment holds |
| `api/team.py` invitation routes (invite, list, revoke, accept, info) and Auto's invite tool | 404 "Invitations are not available on this install." before the caller is looked at; the tool answers the same as an error | Invitations are off in enterprise (PRD-277 §4). Removing a member still works and never asks Clerk |
| `core/auth/device.py` (rows 33, 23; `device_pairing` is on) | A person whose email is `LOCAL_OPERATOR_EMAIL` is a plain `user` and must be a member | Was: that email made a device `super_admin` and skipped the membership check. Now only where `anonymous_operator` is on |

The frontend rows move in Wave 5, against a frontend capability module beside `lib/auth-edition.ts`.

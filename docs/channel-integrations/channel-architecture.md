# Channel Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/55-AUTONOMOUS-ASSISTANT-PLATFORM.md](docs/PRDS/55-AUTONOMOUS-ASSISTANT-PLATFORM.md)
- [frontend/components/settings/ChannelsSettingsTab.tsx](frontend/components/settings/ChannelsSettingsTab.tsx)
- [frontend/components/settings/__tests__/channels-settings-local.test.ts](frontend/components/settings/__tests__/channels-settings-local.test.ts)
- [orchestrator/alembic/versions/20260215_add_heartbeat_and_channels.py](orchestrator/alembic/versions/20260215_add_heartbeat_and_channels.py)
- [orchestrator/api/channels.py](orchestrator/api/channels.py)
- [orchestrator/api/heartbeat.py](orchestrator/api/heartbeat.py)
- [orchestrator/api/routing.py](orchestrator/api/routing.py)
- [orchestrator/channels/base.py](orchestrator/channels/base.py)
- [orchestrator/channels/manager.py](orchestrator/channels/manager.py)
- [orchestrator/channels/primitive_heartbeat.py](orchestrator/channels/primitive_heartbeat.py)
- [orchestrator/channels/telegram_adapter.py](orchestrator/channels/telegram_adapter.py)
- [orchestrator/core/models/channels.py](orchestrator/core/models/channels.py)
- [orchestrator/core/routing/engine.py](orchestrator/core/routing/engine.py)
- [orchestrator/core/security/workspace_scope.py](orchestrator/core/security/workspace_scope.py)
- [orchestrator/scripts/setup_jira_trigger.py](orchestrator/scripts/setup_jira_trigger.py)
- [orchestrator/tests/test_channel_adapter_contract.py](orchestrator/tests/test_channel_adapter_contract.py)
- [orchestrator/tests/test_f149_a_message_runs_only_its_own_workspaces_agents.py](orchestrator/tests/test_f149_a_message_runs_only_its_own_workspaces_agents.py)
- [orchestrator/tests/test_prd225_polling_answer_bridge.py](orchestrator/tests/test_prd225_polling_answer_bridge.py)

</details>



## Purpose and Scope

This page describes how channel connections are represented, how adapters are created and managed, and how inbound messages handled by `BaseChannelAdapter` enter the routing and agent-execution pipeline. It also covers the channel-management API and settings surface. [orchestrator/channels/base.py:1-10](), [orchestrator/channels/manager.py:1-10](), [orchestrator/api/channels.py:1-11]()

Sources: [orchestrator/channels/base.py:1-10](), [orchestrator/channels/manager.py:1-10](), [orchestrator/api/channels.py:1-11]()

---

## 1. Core Components & Registry

Channel architecture separates persisted connection configuration, runtime adapter lifecycle, platform-specific message handling, and channel-management APIs. `ChannelConnection` stores per-workspace connection data, `ChannelManager` owns adapter instances it starts, and platform-specific adapters subclass `BaseChannelAdapter`. [orchestrator/core/models/channels.py:19-45](), [orchestrator/channels/manager.py:22-26](), [orchestrator/channels/base.py:24-32]()

The channel API imports driver-registry functions and types, including `get_driver`, `list_platforms`, and `ConnectivityMode`. Its module documentation describes drivers as the source of platform-specific connection behavior, including configuration requirements, connectivity modes, verification, sending, webhook installation, and polling lifecycle operations. [orchestrator/api/channels.py:5-10](), [orchestrator/api/channels.py:29-36]()

### Base adapter contract

`BaseChannelAdapter` stores `connection_id`, `workspace_id`, and `config`, and initializes `is_running` to `False`. Its abstract methods define the lifecycle (`start()` and `stop()`), outbound messaging (`send_message(channel_id, text, **kwargs)`), and connection test (`test_connection()`) contracts. The class also provides `handle_message()` for the shared ingest, route, execute, and respond flow. [orchestrator/channels/base.py:24-65](), [orchestrator/channels/base.py:110-134]()

`upload_attachment()` is a shared optional helper for inbound media. It writes bytes to the attachment store with the adapter’s workspace ID and an uploader identifier based on its connection ID; it returns the attachment ID on success and logs and returns `None` on failure. [orchestrator/channels/base.py:69-108]()

### Adapter registry and drivers

`ChannelManager._create_adapter()` uses a platform-to-module-and-class mapping for `telegram`, `slack`, `discord`, and `whatsapp`. It imports the selected module lazily, retrieves the named class, and constructs it with `(connection_id, workspace_id, config)`. An unknown platform or unavailable optional dependency results in no adapter being created. [orchestrator/channels/manager.py:126-170]()

The API’s `channels.drivers` registry is a distinct platform-facing interface: `orchestrator/api/channels.py` imports `get_driver` and `list_platforms` to mediate channel-management operations. The shown `ChannelManager` mapping, by contrast, directly constructs legacy `BaseChannelAdapter` subclasses for polling lifecycle management. [orchestrator/api/channels.py:29-36](), [orchestrator/channels/manager.py:126-156]()

### Bridge Diagram: Natural Language Channel Concepts to Code Entities

```mermaid
classDiagram
direction TB
    class "ChannelConnection" {
        "id"
        "workspace_id"
        "platform"
        "config"
        "status"
        "mode"
    }
    class "ChannelManager" {
        "_adapters"
        "start_all()"
        "start_adapter()"
        "stop_adapter()"
        "_create_adapter()"
        "is_running()"
    }
    class "BaseChannelAdapter" {
        "connection_id"
        "workspace_id"
        "config"
        "start()"
        "stop()"
        "send_message()"
        "test_connection()"
        "handle_message()"
    }
    class "TelegramAdapter" {
        "_on_command()"
        "send_message()"
        "test_connection()"
    }
    class "channels.drivers" {
        "get_driver()"
        "list_platforms()"
        "ConnectivityMode"
    }
    "ChannelConnection" --> "ChannelManager" : "persisted polling connection"
    "ChannelManager" --> "BaseChannelAdapter" : "creates and manages"
    "BaseChannelAdapter" <|-- "TelegramAdapter" : "subclass"
    "channels.drivers" --> "api.channels" : "driver lookup and platform listing"
```

Sources: [orchestrator/core/models/channels.py:19-45](), [orchestrator/channels/manager.py:22-26](), [orchestrator/channels/manager.py:32-63](), [orchestrator/channels/manager.py:126-170](), [orchestrator/channels/base.py:24-65](), [orchestrator/channels/base.py:110-134](), [orchestrator/channels/telegram_adapter.py:24-32](), [orchestrator/api/channels.py:29-36]()

---

## 2. ChannelConnection Model & Database Schema

`ChannelConnection` maps to the `channel_connections` table and represents a connection scoped by `workspace_id`. Its fields cover platform configuration and status, connectivity mode, verification results, optional default-agent routing, activity statistics, metadata, and timestamps. [orchestrator/core/models/channels.py:19-45]()

| Field | Model representation | Role |
|---|---|---|
| `id` | UUID primary key | Identifies the connection. [orchestrator/core/models/channels.py:23-23]() |
| `workspace_id` | Indexed UUID, non-null | Associates the connection with a workspace. [orchestrator/core/models/channels.py:24-24]() |
| `platform` | Non-null `String(50)` | Stores the platform identifier. [orchestrator/core/models/channels.py:25-25]() |
| `config` | JSON, default `{}` | Stores connection credentials and settings. [orchestrator/core/models/channels.py:26-29]() |
| `status` | `String(20)`, default `inactive` | Stores connection status, including active, inactive, or error. [orchestrator/core/models/channels.py:30-30]() |
| `mode` | Non-null `String(20)`, default `webhook` | Stores the connectivity mode, such as `webhook` or `polling`. [orchestrator/core/models/channels.py:31-36]() |
| `webhook_url` | Nullable string | Stores the URL associated with webhook mode. [orchestrator/core/models/channels.py:31-37]() |
| `last_verified` | Nullable timezone-aware datetime | Records the last successful verification time. [orchestrator/core/models/channels.py:32-38]() |
| `last_error` | Nullable string | Holds the latest verification or connection failure reason. [orchestrator/core/models/channels.py:34-39]() |
| `metadata_` | JSON column named `metadata`, default `{}` | Stores additional connection metadata. [orchestrator/core/models/channels.py:40-40]() |
| `default_agent_id` | Nullable integer | Stores an optional default routing target. [orchestrator/core/models/channels.py:41-41]() |
| `message_count` | Integer, default `0` | Tracks channel message activity. [orchestrator/core/models/channels.py:42-42]() |
| `last_activity_at` | Nullable datetime | Records the latest activity timestamp. [orchestrator/core/models/channels.py:43-43]() |
| `created_at`, `updated_at` | Datetime columns | Record creation and update times. [orchestrator/core/models/channels.py:44-45]() |

The model comments identify `trigger_mode` as a per-channel setting read by the ingress trust gate; it is stored in the JSON `config` rather than represented as a separate model column. [orchestrator/core/models/channels.py:26-29]()

Sources: [orchestrator/core/models/channels.py:19-45]()

---

## 3. ChannelManager Lifecycle Control

`ChannelManager` maintains an in-memory mapping from connection IDs to `BaseChannelAdapter` instances. At startup, `start_all()` queries active polling-mode connections and starts an adapter for each. Webhook-mode rows are intentionally skipped because the platform posts to the inbound webhook URL and does not need an in-process polling adapter. [orchestrator/channels/manager.py:22-26](), [orchestrator/channels/manager.py:32-63]()

### Lifecycle methods

| Method | Behavior |
|---|---|
| `start_all()` | Opens a database session, queries `ChannelConnection` rows where `status == "active"` and `mode == "polling"`, and calls `start_adapter()` for each result. [orchestrator/channels/manager.py:32-63]() |
| `stop_all()` | Calls `stop()` on each tracked adapter, logs per-adapter errors, then clears the in-memory adapter map. [orchestrator/channels/manager.py:74-82]() |
| `start_adapter()` | Stops an existing adapter for the connection ID, creates a replacement, starts it, and tracks it on success. [orchestrator/channels/manager.py:88-114]() |
| `stop_adapter()` | Removes the adapter from the map and stops it if present. [orchestrator/channels/manager.py:116-120]() |
| `_create_adapter()` | Resolves the platform through `_ADAPTER_MAP` and lazy-imports the specified class. [orchestrator/channels/manager.py:126-170]() |
| `is_running()` | Provides a connection-running check used by the channel API when reconciling stored status with runtime state. [orchestrator/channels/manager.py:176-187](), [orchestrator/api/channels.py:182-188]() |

### Bridge Diagram: Lifecycle Flow from Database to Runtime Adapter

```mermaid
flowchart TD
    "ChannelConnection" --> "start_all_query"["start_all() query"]
    "start_all_query" --> "ActivePollingRows"["active polling rows"]
    "ActivePollingRows" --> "start_adapter"["start_adapter(connection_id, workspace_id, platform, config)"]
    "start_adapter" --> "_create_adapter"["_create_adapter()"]
    "_create_adapter" --> "_ADAPTER_MAP"["_ADAPTER_MAP"]
    "_ADAPTER_MAP" --> "TelegramAdapter"
    "_ADAPTER_MAP" --> "SlackAdapter"
    "_ADAPTER_MAP" --> "DiscordAdapter"
    "_ADAPTER_MAP" --> "WhatsAppAdapter"
    "adapter_start"["adapter.start()"] --> "tracked_adapter"["_adapters[connection_id]"]
    "stop_adapter"["stop_adapter(connection_id)"] --> "adapter_stop"["adapter.stop()"]
    "stop_all"["stop_all()"] --> "clear_adapters"["clear _adapters"]
```

Sources: [orchestrator/channels/manager.py:22-26](), [orchestrator/channels/manager.py:32-83](), [orchestrator/channels/manager.py:88-120](), [orchestrator/channels/manager.py:126-170]()

---

## 4. Message Pipeline, Trust Gate & Polling Answer Bridge

`BaseChannelAdapter.handle_message()` converts a platform payload to an envelope through `_to_envelope()`, creates a database session, calls `UniversalRouter.route()`, checks that the selected agent belongs to the envelope’s workspace, runs the agent, sends a reply, and updates activity statistics. The adapter’s `workspace_id` is supplied at construction, and the shared flow uses that workspace identity rather than accepting it from the inbound message. [orchestrator/channels/base.py:24-32](), [orchestrator/channels/base.py:114-134](), [orchestrator/channels/base.py:135-178](), [orchestrator/tests/test_channel_adapter_contract.py:27-33]()

If envelope conversion returns no envelope, processing stops. If routing produces no usable agent decision, or the routed agent fails the workspace check, the adapter sends `NO_ROUTE_REPLY` and emits a successful `no_route_fallback` channel heartbeat. [orchestrator/channels/base.py:135-161]()

The channel connection model documents per-channel `trigger_mode` values (`strict`, `communication_only`, or `allow_all`) as ingress trust-gate configuration. The channel API imports `TRIGGER_MODES` and normalization/accessor helpers from `services.ingress_gate`; the details of message filtering are outside the implementation shown here. [orchestrator/core/models/channels.py:26-29](), [orchestrator/api/channels.py:37-42]()

### Bridge Diagram: Inbound Message to Routing and Response Code Entities

```mermaid
flowchart LR
    "PlatformMessage"["platform message"] --> "Adapter_to_envelope"["_to_envelope(platform_message)"]
    "Adapter_to_envelope" --> "RequestEnvelope"
    "RequestEnvelope" --> "UniversalRouter_route"["UniversalRouter.route(envelope)"]
    "UniversalRouter_route" --> "RoutingDecision"
    "RoutingDecision" --> "agent_in_workspace"["agent_in_workspace(db, agent_id, envelope.workspace_id)"]
    "agent_in_workspace" --> "run_agent"["_run_agent(db, agent_id, envelope, platform_message)"]
    "run_agent" --> "send_reply"["_send_reply(db, platform_message, response_text)"]
    "send_reply" --> "send_message"["send_message(channel_id, text)"]
    "send_reply" --> "update_activity_stats"["_update_activity_stats(db)"]
    "NoUsableRoute"["no usable route"] --> "NO_ROUTE_REPLY"
    "NO_ROUTE_REPLY" --> "send_reply"
```

Sources: [orchestrator/channels/base.py:114-134](), [orchestrator/channels/base.py:135-178](), [orchestrator/tests/test_f149_a_message_runs_only_its_own_workspaces_agents.py:117-149]()

### Polling answer bridge

`TelegramAdapter.start()` builds a `python-telegram-bot` application, registers command and message handlers, initializes and starts the application, then starts its updater in polling mode. The message handler accepts text, photo, and document updates while excluding commands. [orchestrator/channels/telegram_adapter.py:24-55]()

`TelegramAdapter.send_message()` sends a response to a Telegram chat and splits text into chunks of up to 4096 characters. The adapter’s `stop()` stops the updater, application, and application runtime, then cancels its polling task and marks itself as not running. [orchestrator/channels/telegram_adapter.py:64-91]()

Sources: [orchestrator/channels/telegram_adapter.py:24-55](), [orchestrator/channels/telegram_adapter.py:64-91]()

---

## 5. Management API Reference & Edition Gating

The channel-management router is mounted at `/api/channels`. The API module is platform-agnostic and imports driver functions and types for platform-specific connection behavior. It also defines the public API host used to build inbound webhook URLs and provides helpers to persist verification outcomes to `channel_connections`. [orchestrator/api/channels.py:29-51]()

### Driver-mediated configuration and verification

The API imports `ConnectivityMode`, `DriverNotConfigured`, `UnknownPlatform`, `VerifyResult`, `get_driver`, and `list_platforms` from `channels.drivers`. It defines required configuration keys for platforms including Telegram, Slack, Discord, and WhatsApp. [orchestrator/api/channels.py:29-36](), [orchestrator/api/channels.py:162-174]()

`_save_verify_outcome()` updates a connection’s status to `active` on successful verification or `error` on failure. It also records verification time when successful, saves the latest error, merges returned and extra metadata, and commits the change; persistence failures are logged and rolled back. [orchestrator/api/channels.py:95-148]()

`_webhook_url_for()` reads the workspace’s `webhook_key` and builds a URL using the configured public API host and the `/api/webhooks/ws/{key}` path. It returns `None` if no key is provisioned. [orchestrator/api/channels.py:48-78]()

The settings UI represents channel connections with platform, status, connectivity mode, webhook URL, verification and activity information, and configuration. Its platform list includes WhatsApp, Telegram, Slack, Discord, Teams, Google Chat, Signal, iMessage, IRC, Matrix, and LINE; the displayed list is not itself proof that every platform has a runtime adapter in `ChannelManager`. [frontend/components/settings/ChannelsSettingsTab.tsx:14-25](), [frontend/components/settings/ChannelsSettingsTab.tsx:27-136](), [orchestrator/channels/manager.py:139-145]()

The UI exposes webhook and polling choices for Telegram. Its `defaultModeFor()` helper selects polling for local installations when polling is supported and otherwise uses the platform’s first listed mode; `needsPublicUrlLocally()` identifies local platforms that lack the polling path. [frontend/components/settings/ChannelsSettingsTab.tsx:138-174]()

Sources: [orchestrator/api/channels.py:5-10](), [orchestrator/api/channels.py:29-51](), [orchestrator/api/channels.py:67-78](), [orchestrator/api/channels.py:95-148](), [orchestrator/api/channels.py:162-174](), [frontend/components/settings/ChannelsSettingsTab.tsx:14-25](), [frontend/components/settings/ChannelsSettingsTab.tsx:27-136](), [frontend/components/settings/ChannelsSettingsTab.tsx:138-174](), [orchestrator/channels/manager.py:139-145]()

---
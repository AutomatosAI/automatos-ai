# Voice Backend & Call Lifecycle

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/__tests__/prd207-live-voice.test.tsx](frontend/components/__tests__/prd207-live-voice.test.tsx)
- [frontend/components/chatbot/chat.tsx](frontend/components/chatbot/chat.tsx)
- [frontend/components/voice/LiveVoiceMode.tsx](frontend/components/voice/LiveVoiceMode.tsx)
- [frontend/components/voice/MicHealthControl.tsx](frontend/components/voice/MicHealthControl.tsx)
- [frontend/components/voice/PresenceOrb.tsx](frontend/components/voice/PresenceOrb.tsx)
- [frontend/components/voice/VoiceErrorBoundary.tsx](frontend/components/voice/VoiceErrorBoundary.tsx)
- [frontend/hooks/use-retell-call.ts](frontend/hooks/use-retell-call.ts)
- [frontend/lib/voice/mic-health.ts](frontend/lib/voice/mic-health.ts)
- [orchestrator/alembic/versions/prd203_voice_turns.py](orchestrator/alembic/versions/prd203_voice_turns.py)
- [orchestrator/alembic/versions/prd207_su_capture.py](orchestrator/alembic/versions/prd207_su_capture.py)
- [orchestrator/alembic/versions/prd207_voice_live.py](orchestrator/alembic/versions/prd207_voice_live.py)
- [orchestrator/api/voice_retell.py](orchestrator/api/voice_retell.py)
- [orchestrator/core/models/voice_calls.py](orchestrator/core/models/voice_calls.py)
- [orchestrator/core/models/voice_turns.py](orchestrator/core/models/voice_turns.py)
- [orchestrator/modules/voice/call_binding.py](orchestrator/modules/voice/call_binding.py)
- [orchestrator/modules/voice/live_settings.py](orchestrator/modules/voice/live_settings.py)
- [orchestrator/modules/voice/providers/retell.py](orchestrator/modules/voice/providers/retell.py)
- [orchestrator/modules/voice/retell_api.py](orchestrator/modules/voice/retell_api.py)
- [orchestrator/modules/voice/spoken_style.py](orchestrator/modules/voice/spoken_style.py)
- [orchestrator/modules/voice/telemetry.py](orchestrator/modules/voice/telemetry.py)
- [orchestrator/modules/voice/voice_meter.py](orchestrator/modules/voice/voice_meter.py)
- [orchestrator/tests/test_prd203_voice_retell.py](orchestrator/tests/test_prd203_voice_retell.py)
- [orchestrator/tests/test_prd207_voice_live.py](orchestrator/tests/test_prd207_voice_live.py)
- [orchestrator/tests/test_voice_quality.py](orchestrator/tests/test_voice_quality.py)

</details>



This page details the implementation of the real-time voice subsystem, known as "Auto Live," which is built upon Retell's voice transport. It covers the backend components responsible for managing the call lifecycle, including session minting, webhook signature verification, call binding, live settings, and the monthly usage cap formula. The frontend integration for voice calls is discussed in [24.2. Voice UI Components]().

## Minting a Web Call

The process of initiating a voice call from the web interface involves several fail-closed gates to ensure proper authorization, resource availability, and secure session establishment. This is handled by the `mint_web_call` endpoint.

### Data Flow and Gates

When a user attempts to start a voice call from the chat interface, a request is sent to the `/api/voice/web-call` endpoint. This endpoint performs a series of checks before a `VoiceCall` record is created in the database and an `access_token` is returned to the frontend.

1.  **Platform Master Switch**: The first gate checks if Auto Live is enabled platform-wide via the system settings. If `live_settings.voice_live_enabled()` [orchestrator/api/voice_retell.py:119]() returns `False`, the call is denied.
2.  **Retell Credentials**: The system verifies that all necessary Retell API credentials (API key, webhook secret, agent ID) are present and armed in the system settings. If `live_settings.retell_credentials().armed` [orchestrator/api/voice_retell.py:124]() is `False`, the call is refused.
3.  **User Resolution**: The caller must be a strictly resolved, authenticated user. The `_resolve_caller_int_id` function [orchestrator/api/voice_retell.py:77-96]() attempts to get the integer `users.id` from the `RequestContext`. If no valid user ID can be determined, the call is denied.
4.  **Workspace Toggle**: The specific workspace must have Auto Live enabled in its settings. The `parse_workspace_voice_live` function [orchestrator/modules/voice/live_settings.py:64]() extracts voice settings from `workspace.settings`. If `ws_voice.enabled` [orchestrator/api/voice_retell.py:143]() is `False`, the call is refused.
5.  **Monthly Usage Cap**: The system checks if the workspace has exceeded its monthly voice call minutes cap. `voice_meter.monthly_meter` [orchestrator/modules/voice/voice_meter.py:100]() retrieves current usage, and `voice_meter.cap_allows_mint` [orchestrator/modules/voice/voice_meter.py:110]() determines if a new call is allowed, considering active call reservations. If the cap is exceeded, the call is refused.

If all gates pass, a new `VoiceCall` record is created in the database with a `minted` status. This record serves as the trust boundary for subsequent webhook events from Retell. An `access_token` is generated and returned to the frontend, which then uses it to establish the WebRTC connection with Retell.

### Call Binding

The `call_binding` module [orchestrator/modules/voice/call_binding.py]() is crucial for establishing the context of a voice call, linking it to a specific chat thread and user. This binding is essential for attributing messages, managing memory, and ensuring data integrity.

The `resolve_call_binding` function [orchestrator/modules/voice/call_binding.py:155-176]() takes the `call_id` (from the `VoiceCall` row), `workspace_id`, and dynamic variables (`user_id_var`, `chat_id_var`) from the Retell webhook payload. It performs the following:

*   **Workspace Validation**: Ensures the `workspace_id` from the webhook matches the `VoiceCall` record.
*   **Chat/User Validation**: Cross-validates the `chat_id_var` and `user_id_var` against the `VoiceCall` record. If there's a mismatch or the specified chat no longer exists, it falls back to a per-call fallback chat.
*   **Fallback Chat Creation**: If no chat is bound or the binding is invalid, `_fallback_chat` [orchestrator/modules/voice/call_binding.py:132-152]() creates a new chat thread using `create_voice_chat` [orchestrator/modules/voice/call_binding.py:106-129()] and attributes it to the user proven at mint. For unminted calls (e.g., phone lane), it attributes the call to the workspace's steward using `_workspace_steward` [orchestrator/modules/voice/call_binding.py:77-95]().

The `CallBinding` dataclass [orchestrator/modules/voice/call_binding.py:67-75]() encapsulates the resolved `chat_id`, `user_id`, `workspace_id`, and a `bound` flag indicating if it's the on-screen thread.

```mermaid
graph TD
    A[Frontend: User Clicks "Live" Button] --> B{POST /api/voice/web-call};
    B -- MintWebCallRequest --> C{Gate 1: Platform Enabled?};
    C -- No --> E[HTTP 503: Platform Disabled];
    C -- Yes --> F{Gate 2: Retell Credentials Armed?};
    F -- No --> G[HTTP 503: Retell Not Armed];
    F -- Yes --> H{Gate 3: User Authenticated?};
    H -- No --> I[HTTP 403: Requires Signed-in User];
    H -- Yes --> J{Gate 4: Workspace Voice Live Enabled?};
    J -- No --> K[HTTP 403: Workspace Disabled];
    J -- Yes --> L{Gate 5: Monthly Cap Allows Mint?};
    L -- No --> M[HTTP 429: Cap Exceeded];
    L -- Yes --> N[Create VoiceCall Record (status=minted)];
    N --> O[Generate access_token];
    O --> P[Return {call_id, access_token} to Frontend];
    P --> Q[Frontend: Establish Retell WebRTC Connection];

    subgraph "Backend: orchestrator/api/voice_retell.py"
        B; C; F; H; J; L; N; O;
    end
    subgraph "Backend: modules/voice/live_settings.py"
        C; F; J;
    end
    subgraph "Backend: modules/voice/voice_meter.py"
        L;
    end
    subgraph "Backend: core/models/voice_calls.py"
        N;
    end
```
Sources:
- [orchestrator/api/voice_retell.py:99-163]()
- [orchestrator/api/voice_retell.py:77-96]()
- [orchestrator/modules/voice/live_settings.py:64]()
- [orchestrator/modules/voice/voice_meter.py:100]()
- [orchestrator/modules/voice/voice_meter.py:110]()
- [orchestrator/modules/voice/call_binding.py:155-176]()
- [orchestrator/modules/voice/call_binding.py:132-152]()
- [orchestrator/modules/voice/call_binding.py:106-129]()
- [orchestrator/modules/voice/call_binding.py:77-95]()
- [orchestrator/modules/voice/call_binding.py:67-75]()

## Retell Provider Client and Webhook Signature Verification

The integration with Retell is handled by the `modules.voice.providers.retell` module [orchestrator/modules/voice/providers/retell.py](). This module acts as a swappable voice transport adapter, managing the streaming contract and webhook authentication.

### Retell LLM Request Parsing

When Retell sends a webhook to Auto's custom-LLM endpoint, the `parse_llm_request` function [orchestrator/modules/voice/providers/retell.py:71-107]() extracts relevant information from the payload. This includes:

*   `response_id`: A unique identifier for the response.
*   `user_text`: The latest user utterance from the transcript.
*   `interaction_type`: The type of interaction (e.g., `response_required`, `reminder_required`).
*   `workspace_id`, `agent_id`, `call_id`, `user_id`, `chat_id`: These are dynamic variables passed by Auto during the `mint_web_call` phase and are used for routing and context.

### Streaming Agent Replies

The `retell_response_frames` asynchronous generator [orchestrator/modules/voice/providers/retell.py:125-172]() is responsible for converting the agent's AI-SDK text output into Retell's custom-LLM response frames. This is a crucial streaming contract: frames are emitted as the reply forms, allowing Retell to begin speaking before the agent finishes generating.

Key aspects:

*   **Speech Unit Processing**: The agent's raw text is processed into "speech units" using `speechify` [orchestrator/modules/voice/spoken_style.py:30-72]() and `split_speech_unit` [orchestrator/modules/voice/spoken_style.py:88-105](). This sanitizes markdown, removes emojis, and ensures that the spoken output sounds natural, not like a read-aloud chat log.
*   **Streaming**: Frames are yielded as soon as a speech unit is ready, rather than waiting for the entire agent response.
*   **Terminal Frame**: A `content_complete=True` frame is sent to close the turn.

### Webhook Signature Verification

All incoming webhooks from Retell are subject to HMAC signature verification to ensure their authenticity and integrity. The `verify_webhook_signature` function [orchestrator/modules/voice/providers/retell.py:200-236]() performs this check.

*   It uses the `RETELL_WEBHOOK_SECRET` from system settings.
*   The signature includes a timestamp, and the verification checks for freshness within a ±5-minute window to prevent replay attacks.
*   If verification fails, the webhook is rejected, preventing unauthorized or tampered events from affecting the system.

### Retell API Client

The `modules.voice.retell_api` module [orchestrator/modules/voice/retell_api.py]() provides the server-side client for interacting with the Retell API. This includes:

*   `build_web_call_payload` [orchestrator/modules/voice/retell_api.py:87-116](): Constructs the payload for creating a web call, including agent overrides for voice ID and maximum call duration. Dynamic variables are coerced to strings as required by Retell.
*   `create_custom_llm_agent` [orchestrator/modules/voice/retell_api.py:120-168](): Creates a Retell agent that fronts Auto Live, configuring its `llm_websocket_url`, `webhook_url`, and `voice_id`. It also applies agent tuning settings like `interruption_sensitivity` and `stt_mode` using `build_agent_tuning` [orchestrator/modules/voice/retell_api.py:31-74]().
*   `update_agent` [orchestrator/modules/voice/retell_api.py:171-177](): Allows patching agent-level settings onto an existing Retell agent for re-tuning.

```mermaid
graph TD
    A[Retell Webhook] --> B{POST /api/voice/retell/events};
    B -- Request Body + Signature Header --> C{Verify Webhook Signature};
    C -- Invalid Signature --> D[HTTP 401: Unauthorized];
    C -- Valid Signature --> E{Parse LLM Request Payload};
    E -- RetellLLMRequest --> F{Resolve Call Binding};
    F -- CallBinding --> G[Process Voice Turn (Agent Loop)];
    G -- Agent AI-SDK Output Stream --> H{retell_response_frames};
    H -- Speech Units --> I[Stream Retell Custom-LLM Frames];
    I --> J[Retell WebRTC Client];

    subgraph "Backend: orchestrator/api/voice_retell.py"
        B;
    end
    subgraph "Backend: modules/voice/providers/retell.py"
        C; E; H;
    end
    subgraph "Backend: modules/voice/call_binding.py"
        F;
    end
    subgraph "Backend: modules/voice/spoken_style.py"
        H;
    end
```
Sources:
- [orchestrator/modules/voice/providers/retell.py]()
- [orchestrator/modules/voice/providers/retell.py:71-107]()
- [orchestrator/modules/voice/providers/retell.py:125-172]()
- [orchestrator/modules/voice/spoken_style.py:30-72]()
- [orchestrator/modules/voice/spoken_style.py:88-105]()
- [orchestrator/modules/voice/providers/retell.py:200-236]()
- [orchestrator/modules/voice/retell_api.py]()
- [orchestrator/modules/voice/retell_api.py:87-116]()
- [orchestrator/modules/voice/retell_api.py:120-168]()
- [orchestrator/modules/voice/retell_api.py:31-74]()
- [orchestrator/modules/voice/retell_api.py:171-177]()

## Live Settings

Live settings for the voice subsystem are managed through a combination of platform-wide system settings and workspace-specific configurations. This allows for granular control over Auto Live's availability and behavior.

### System Settings

Platform-wide settings are stored in the `system_settings` table and accessed via `live_settings.py` [orchestrator/modules/voice/live_settings.py]().

*   `voice_live_enabled()` [orchestrator/modules/voice/live_settings.py:30-32](): Checks if the overall Auto Live feature is enabled.
*   `retell_credentials()` [orchestrator/modules/voice/live_settings.py:35-59](): Retrieves the Retell API key, webhook secret, and agent ID. These are considered "armed" if all three are present. These credentials are never exposed to the frontend.
*   `retell_agent_id()` [orchestrator/modules/voice/live_settings.py:61](): Returns the configured Retell agent ID.

### Workspace-Specific Settings

Each workspace can override or configure certain voice settings. These are stored within the `settings` JSONB column of the `Workspace` model.

*   `parse_workspace_voice_live(raw_settings)` [orchestrator/modules/voice/live_settings.py:64-80](): Parses the raw workspace settings to extract voice-specific configurations, returning a `WorkspaceVoiceLiveView` dataclass. This view includes:
    *   `enabled`: Boolean indicating if Auto Live is enabled for this workspace. Defaults to `False` (fail-closed).
    *   `monthly_cap_minutes`: The maximum allowed voice call minutes per month for the workspace. Defaults to `config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES` [orchestrator/modules/voice/live_settings.py:76]().
    *   `retell_voice_id`: An optional Retell voice ID specific to this workspace.
*   `validate_voice_live_update(update_dict)` [orchestrator/modules/voice/live_settings.py:83-103](): Validates updates to workspace voice settings, ensuring data types and ranges are correct (e.g., `monthly_cap_minutes` must be a positive integer within a reasonable bound).

These settings are accessed when minting a web call [orchestrator/api/voice_retell.py:138-147]() and when calculating the monthly cap [orchestrator/api/voice_retell.py:150-156]().

## Voice Meter and Monthly Cap Formula

To manage resource usage and prevent abuse, Auto Live implements a monthly usage cap for voice calls, measured in minutes.

### Metering

The `voice_meter` module [orchestrator/modules/voice/voice_meter.py]() provides functions for tracking and enforcing this cap.

*   `monthly_meter(db, workspace_id)` [orchestrator/modules/voice/voice_meter.py:100-107](): Calculates the current month's voice call usage for a given workspace. It sums the `duration_minutes` of `VoiceCall` records within the current UTC month window.
*   `month_window_utc(now)` [orchestrator/modules/voice/voice_meter.py:120-129](): Determines the start and end timestamps for the current UTC month, handling year rollovers correctly.

### Cap Formula

The `cap_allows_mint(reading, cap_minutes)` function [orchestrator/modules/voice/voice_meter.py:110-117]() implements the core logic for deciding if a new call can be minted:

`ended_minutes + (active_calls * config.VOICE_LIVE_RESERVE_MINUTES) >= cap_minutes`

*   `ended_minutes`: Total minutes of completed calls this month.
*   `active_calls`: Number of currently active calls.
*   `config.VOICE_LIVE_RESERVE_MINUTES`: A configurable reserve (e.g., 10 minutes) is added for each active call. This prevents a "two-tabs race" where two simultaneous mint requests might both pass the cap check if only `ended_minutes` were considered. The reservation effectively "pre-allocates" minutes for ongoing calls.

If the sum exceeds or equals the `cap_minutes`, the mint is refused with an honest reason.

### Call Lifecycle and Telemetry

The `VoiceCall` model [orchestrator/core/models/voice_calls.py]() tracks the state and duration of each call.

*   **Minted**: Initial state when `mint_web_call` succeeds.
*   **Started**: When Retell confirms the call has begun.
*   **Ended**: When the call concludes.
*   **Duration**: The `duration_minutes` field is updated upon call completion.

The `voice_turns` table [orchestrator/core/models/voice_turns.py]() records individual turns within a voice call, capturing STT/TTS latency and other telemetry data. This data is used for analytics and quality assessment.

```mermaid
graph TD
    A[Workspace Settings] --> B{parse_workspace_voice_live};
    B --> C{WorkspaceVoiceLiveView};
    C --> D[Monthly Cap Minutes];
    C --> E[Retell Voice ID];
    C --> F[Enabled Flag];

    G[VoiceCall Records (DB)] --> H{monthly_meter};
    H --> I[MeterReading (ended_minutes, active_calls)];

    I --> J{cap_allows_mint};
    D --> J;
    K[config.VOICE_LIVE_RESERVE_MINUTES] --> J;

    J -- Allowed --> L[Mint Web Call];
    J -- Refused --> M[Deny Call (HTTP 429)];

    subgraph "Backend: modules/voice/live_settings.py"
        B; C;
    end
    subgraph "Backend: modules/voice/voice_meter.py"
        H; I; J;
    end
    subgraph "Backend: orchestrator/api/voice_retell.py"
        L; M;
    end
    subgraph "Backend: core/models/voice_calls.py"
        G;
    end
```
Sources:
- [orchestrator/modules/voice/live_settings.py]()
- [orchestrator/modules/voice/live_settings.py:30-32]()
- [orchestrator/modules/voice/live_settings.py:35-59]()
- [orchestrator/modules/voice/live_settings.py:61]()
- [orchestrator/modules/voice/live_settings.py:64-80]()
- [orchestrator/modules/voice/live_settings.py:76]()
- [orchestrator/modules/voice/live_settings.py:83-103]()
- [orchestrator/api/voice_retell.py:138-147]()
- [orchestrator/api/voice_retell.py:150-156]()
- [orchestrator/modules/voice/voice_meter.py]()
- [orchestrator/modules/voice/voice_meter.py:100-107]()
- [orchestrator/modules/voice/voice_meter.py:120-129]()
- [orchestrator/modules/voice/voice_meter.py:110-117]()
- [orchestrator/core/models/voice_calls.py]()
- [orchestrator/core/models/voice_turns.py]()

---
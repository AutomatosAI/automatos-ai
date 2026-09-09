# Spoken Style & Voice Telemetry

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



This page details the spoken output contract for Auto Live, the sanitization process for `speechify()`, the database tables `voice_calls` and `voice_turns` used for telemetry, the decomposition of STT/TTS latency, and the voice quality tests and migrations.

## Spoken Output Contract and `speechify()` Sanitization

The voice system aims to make Auto sound natural and human-like, avoiding the robotic pronunciation of markdown or other non-speech elements. This is achieved through a "spoken output contract" and the `speechify()` sanitization function.

The spoken output contract is a set of guidelines appended to the system prompt, informing the LLM that it is speaking aloud and should avoid generating text that would sound unnatural when spoken. This contract emphasizes avoiding markdown, URLs, and other text artifacts that are common in chat but problematic for TTS [orchestrator/tests/test_voice_quality.py:181-187]().

The `speechify()` function [orchestrator/modules/voice/spoken_style.py:20-60]() is responsible for sanitizing the agent's text output before it is sent to the Text-to-Speech (TTS) engine. It performs several transformations:
- **Markdown Removal**: Bullet points (`-`, `*`), bold (`**`), and headers (`#`) are removed, leaving only the plain text [orchestrator/tests/test_voice_quality.py:30-34]().
- **Numbered List Stripping**: Numbered list prefixes (`1.`, `2.`) are removed [orchestrator/tests/test_voice_quality.py:37-41]().
- **Header and Rule Dropping**: Markdown headers and horizontal rules (`---`) are entirely removed [orchestrator/tests/test_voice_quality.py:44-48]().
- **Link Label Extraction**: For markdown links `[label](url)`, only the `label` is retained, and the URL is discarded [orchestrator/tests/test_voice_quality.py:51-55]().
- **Code Block Suppression**: Entire code fences (e.g., ````python\nprint('hi')\n````) are removed, resulting in no spoken output [orchestrator/tests/test_voice_quality.py:58-61]().
- **Inline Code Preservation**: Inline code (e.g., `` `db.py` ``) has its content preserved [orchestrator/tests/test_voice_quality.py:64-66]().
- **Emoji Stripping**: Emojis are removed from the text [orchestrator/tests/test_voice_quality.py:69-72]().
- **Punctuation-only Unit Silence**: If a segment of text consists only of punctuation or whitespace after sanitization, it results in no spoken output [orchestrator/tests/test_voice_quality.py:79-82]().

This sanitization is crucial because markdown markers can straddle stream chunks, making per-token sanitization ineffective. `speechify()` operates on whole speech units to ensure correct processing [orchestrator/modules/voice/providers/retell.py:139-142]().

The `split_speech_unit()` function [orchestrator/modules/voice/spoken_style.py:63-94]() further refines the output by breaking down long text into smaller, speakable units. It prioritizes sentence endings and newlines as natural breaks. If a sentence is too long without a natural terminator, it will be cut at a word boundary to prevent excessively long spoken segments [orchestrator/tests/test_voice_quality.py:87-109]().

Sources:
- [orchestrator/modules/voice/spoken_style.py:20-94]()
- [orchestrator/modules/voice/providers/retell.py:139-142]()
- [orchestrator/tests/test_voice_quality.py:30-109]()
- [orchestrator/tests/test_voice_quality.py:181-187]()

## `voice_calls` and `voice_turns` Tables

The voice subsystem uses two primary database tables for tracking call lifecycle and individual turns: `voice_calls` and `voice_turns`.

### `voice_calls` Table

The `voice_calls` table [orchestrator/core/models/voice_calls.py:17-60]() stores metadata about each voice call session. A new `VoiceCall` record is created during the "minting" process when a user initiates a web call [orchestrator/api/voice_retell.py:100-113]().

**Key fields in `VoiceCall`**:
- `id`: Primary key, UUID.
- `workspace_id`: Foreign key to `workspaces.id`.
- `user_id`: Foreign key to `users.id`, representing the user who initiated the call.
- `chat_id`: Foreign key to `chats.id`, linking the voice call to a specific chat thread.
- `fallback_chat_id`: Stores the ID of a chat created if the initial `chat_id` is invalid or missing, ensuring a conversation always has a place to land [orchestrator/modules/voice/call_binding.py:132-151]().
- `status`: Current status of the call (e.g., `minted`, `active`, `ended`, `error`).
- `retell_call_id`: The external call ID provided by Retell.
- `retell_agent_id`: The external agent ID provided by Retell.
- `started_at`, `ended_at`: Timestamps for call duration.
- `total_duration_seconds`: Total duration of the call.
- `total_cost_usd`: Total estimated cost of the call.
- `metadata`: JSONB field for additional, unstructured data.

The `voice_calls` table is central to the webhook trust boundary. When a webhook event arrives from Retell, the `call_id` in the payload is cross-validated against an existing `voice_calls` record to ensure authenticity and proper binding to a user and workspace [orchestrator/modules/voice/call_binding.py:5-18]().

### `voice_turns` Table

The `voice_turns` table [orchestrator/core/models/voice_turns.py:17-48]() records each individual turn within a voice call, capturing details about user utterances and agent responses. This table is populated by the live path [orchestrator/tests/test_prd207_voice_live.py:12]().

**Key fields in `VoiceTurn`**:
- `id`: Primary key, UUID.
- `call_id`: Foreign key to `voice_calls.id`.
- `turn_index`: Sequential index of the turn within the call.
- `user_text`: The transcribed text from the user.
- `agent_text`: The spoken text from the agent.
- `stt_latency_ms`: Latency for Speech-to-Text conversion.
- `tts_latency_ms`: Latency for Text-to-Speech conversion.
- `llm_latency_ms`: Latency for LLM processing.
- `total_latency_ms`: Total latency for the turn.
- `started_at`, `ended_at`: Timestamps for the turn.
- `cost_usd`: Estimated cost of this specific turn.
- `metadata`: JSONB field for additional telemetry.

The `voice_turns` table is crucial for detailed telemetry and performance analysis of voice interactions.

### Diagram: Voice Call Data Flow

```mermaid
graph TD
    subgraph "Frontend (User Interface)"
        A[User initiates call via UI] --> B{POST /api/voice/web-call};
        B -- "Returns access_token, call_id" --> C[WebRTC SDK (Retell)];
        C -- "Streams audio, receives agent speech" --> D[Chat UI (LiveVoiceMode)];
    end

    subgraph "Backend (Orchestrator)"
        B --> E[mint_web_call()];
        E -- "Validates user, workspace, cap" --> F[VoiceCall (DB Model)];
        F -- "Status: minted" --> G[voice_calls table];
        G -- "call_id, access_token" --> B;

        C -- "Sends user utterances" --> H[Retell Webhook];
        H -- "HMAC verified, payload parsed" --> I[parse_llm_request()];
        I -- "Resolves binding" --> J[resolve_call_binding()];
        J -- "Retrieves VoiceCall record" --> G;
        J -- "Creates/updates Chat, User" --> K[Chat, User (DB Models)];
        J -- "Dispatches to StreamingChatService" --> L[StreamingChatService];
        L -- "Agent generates response" --> M[LLM];
        M -- "Streams chunks" --> N[retell_response_frames()];
        N -- "Sanitizes with speechify(), splits units" --> O[Speech Unit Buffer];
        O -- "Sends frames" --> H;
        H -- "Receives agent speech frames" --> C;

        L -- "Records turn details" --> P[VoiceTurn (DB Model)];
        P -- "STT/TTS/LLM latencies" --> Q[voice_turns table];

        H -- "Call lifecycle events" --> R[Retell Webhook /events];
        R -- "Updates VoiceCall status, duration, cost" --> G;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#fcf,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
    style K fill:#ccf,stroke:#333,stroke-width:2px
    style L fill:#bbf,stroke:#333,stroke-width:2px
    style M fill:#fcf,stroke:#333,stroke-width:2px
    style N fill:#bbf,stroke:#333,stroke-width:2px
    style O fill:#bbf,stroke:#333,stroke-width:2px
    style P fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#ccf,stroke:#333,stroke-width:2px
    style R fill:#bbf,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/core/models/voice_calls.py:17-60]()
- [orchestrator/core/models/voice_turns.py:17-48]()
- [orchestrator/api/voice_retell.py:100-113]()
- [orchestrator/modules/voice/call_binding.py:5-18]()
- [orchestrator/modules/voice/call_binding.py:132-151]()
- [orchestrator/tests/test_prd207_voice_live.py:12]()

## STT/TTS Latency Decomposition

Accurate measurement of latency components is critical for optimizing the real-time voice experience. The `voice_turns` table captures `stt_latency_ms`, `tts_latency_ms`, and `llm_latency_ms` to provide a detailed breakdown of the total latency for each turn [orchestrator/core/models/voice_turns.py:17-48]().

**Latency Components**:
- **STT Latency (`stt_latency_ms`)**: The time taken for the user's spoken audio to be converted into text. This is primarily handled by the Retell service.
- **LLM Latency (`llm_latency_ms`)**: The time taken for the LLM to process the user's transcribed text and generate an agent response. This involves the `StreamingChatService` and the underlying LLM.
- **TTS Latency (`tts_latency_ms`)**: The time taken for the agent's generated text to be converted into spoken audio. This is also primarily handled by the Retell service.

The `retell_response_frames()` function [orchestrator/modules/voice/providers/retell.py:125-172]() is designed to stream agent responses as they are generated, rather than waiting for the entire response to be complete. This "first audio before the full agent stream completes" property is crucial for reducing perceived latency. It means that TTS can begin synthesizing and playing audio even while the LLM is still generating subsequent parts of the response [orchestrator/tests/test_prd203_voice_retell.py:30-39]().

The `voice_meter` module [orchestrator/modules/voice/voice_meter.py]() also plays a role in managing call duration and cost, which indirectly relates to latency by enforcing caps.

Sources:
- [orchestrator/core/models/voice_turns.py:17-48]()
- [orchestrator/modules/voice/providers/retell.py:125-172]()
- [orchestrator/modules/voice/voice_meter.py]()
- [orchestrator/tests/test_prd203_voice_retell.py:30-39]()

## Voice Quality Tests and Migrations

Ensuring high voice quality involves both functional correctness and a natural-sounding output.

### Voice Quality Tests

The `test_voice_quality.py` suite [orchestrator/tests/test_voice_quality.py]() specifically targets the `speechify()` function and the `split_speech_unit()` logic to ensure that the spoken output contract is upheld. These are pure unit tests, meaning they do not rely on a database, sockets, or external models [orchestrator/tests/test_voice_quality.py:9]().

Key aspects tested include:
- **Markdown Sanitization**: Verifies that markdown elements like bold, bullets, headers, and links are correctly processed or removed [orchestrator/tests/test_voice_quality.py:30-55]().
- **Code Handling**: Confirms that code blocks are silenced and inline code is preserved [orchestrator/tests/test_voice_quality.py:58-66]().
- **Emoji Removal**: Ensures emojis are stripped [orchestrator/tests/test_voice_quality.py:69-72]().
- **Speech Unit Splitting**: Tests that text is correctly segmented into speakable units based on sentence endings, newlines, and character limits, preventing runaway sentences [orchestrator/tests/test_voice_quality.py:87-109]().
- **Streaming Sanitization**: Crucially, tests confirm that sanitization works even when markdown markers are split across multiple incoming text chunks from the LLM stream [orchestrator/tests/test_voice_quality.py:130-144]().

### Migrations

The `prd207_voice_live.py` Alembic migration script [orchestrator/alembic/versions/prd207_voice_live.py]() introduces the `voice_calls` table and related schema changes for the Auto Live feature. This migration chains directly on `prd206_chat_summary`, ensuring a single, linear migration history [orchestrator/tests/test_prd207_voice_live.py:38-56]().

The `prd203_voice_turns.py` migration script [orchestrator/alembic/versions/prd203_voice_turns.py]() introduces the `voice_turns` table, which is essential for capturing detailed telemetry about each turn in a voice call, including latency metrics.

These migrations are critical for establishing the data models necessary for the voice subsystem's functionality and telemetry.

Sources:
- [orchestrator/tests/test_voice_quality.py:9]()
- [orchestrator/tests/test_voice_quality.py:30-72]()
- [orchestrator/tests/test_voice_quality.py:87-109]()
- [orchestrator/tests/test_voice_quality.py:130-144]()
- [orchestrator/alembic/versions/prd207_voice_live.py]()
- [orchestrator/alembic/versions/prd203_voice_turns.py]()
- [orchestrator/tests/test_prd207_voice_live.py:38-56]()

---
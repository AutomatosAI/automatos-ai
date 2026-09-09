# Voice (Auto Live)

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



The Voice (Auto Live) subsystem provides real-time voice interaction capabilities, built upon Retell's transport layer and integrated with Automatos AI's custom LLM loop. This page offers a high-level overview of the system, with detailed technical specifications delegated to its child pages.

The core functionality involves enabling users to converse with AI agents in real-time, with the system handling speech-to-text (STT), text-to-speech (TTS), turn-taking, and barge-in capabilities. A key design principle is to keep the sophisticated Automatos AI agent logic on our side, offloading only the voice-native hard parts to the Retell vendor.

This parent page introduces the main components and their interactions, while the following child pages delve into specifics:
- [Voice Backend & Call Lifecycle](#24.1) covers the server-side logic, including call minting, Retell integration, and usage metering.
- [Voice UI Components](#24.2) details the frontend elements that enable and visualize voice interactions.
- [Spoken Style & Voice Telemetry](#24.3) explains how spoken output is sanitized and how voice call data is collected and analyzed.

## Architecture Overview

The Voice (Auto Live) system bridges the user's browser with the Automatos AI backend via Retell's real-time voice platform. The frontend initiates calls, streams user audio, and displays agent responses and visual cues. The backend handles call lifecycle, agent orchestration, and interaction with Retell's APIs.

```mermaid
graph TD
    subgraph "User Interface (Frontend)"
        UI[("User Interface")]
        LV[LiveVoiceMode]
        PO[PresenceOrb]
        MHC[MicHealthControl]
    end

    subgraph "Automatos AI Backend"
        API[/api/voice/web-call]
        WS[WS /api/voice/retell/llm-websocket]
        WEBHOOK[POST /api/voice/retell/events]
        VC[VoiceCall Model]
        VT[VoiceTurn Model]
        LLM_LOOP[Custom LLM Loop (StreamingChatService)]
        LS[Live Settings]
        VM[Voice Meter]
        CB[Call Binding]
        RTAPI[Retell API Client]
    end

    subgraph "Retell AI Platform"
        RETELL_SDK[Retell WebRTC SDK]
        RETELL_LLM[Retell Custom LLM Transport]
        RETELL_WEBHOOK[Retell Webhook Service]
    end

    UI -- "Initiates Call" --> LV
    LV -- "Mint Web Call Request" --> API
    API -- "Validates & Creates VoiceCall" --> VC
    API -- "Returns Access Token" --> LV
    LV -- "Connects via WebRTC" --> RETELL_SDK
    RETELL_SDK -- "User Audio" --> RETELL_LLM
    RETELL_LLM -- "LLM WebSocket (User Text)" --> WS
    WS -- "Orchestrates Agent Response" --> LLM_LOOP
    LLM_LOOP -- "Streams Agent Text" --> WS
    WS -- "Agent Audio" --> RETELL_LLM
    RETELL_LLM -- "Agent Audio" --> RETELL_SDK
    RETELL_SDK -- "Agent Audio" --> LV
    LV -- "Visualizes Audio Levels" --> PO
    LV -- "Monitors Mic Health" --> MHC

    RETELL_WEBHOOK -- "Call Lifecycle Events (HMAC Verified)" --> WEBHOOK
    WEBHOOK -- "Updates VoiceCall Status" --> VC
    WEBHOOK -- "Records Usage" --> VM
    WEBHOOK -- "Binds Call Context" --> CB
    LLM_LOOP -- "Records Voice Turns" --> VT

    LS -- "Configures Voice Features" --> API
    VM -- "Enforces Usage Caps" --> API
    RTAPI -- "Creates/Updates Retell Agents" --> RETELL_LLM

    style UI fill:#f9f,stroke:#333,stroke-width:2px
    style LV fill:#f9f,stroke:#333,stroke-width:2px
    style PO fill:#f9f,stroke:#333,stroke-width:2px
    style MHC fill:#f9f,stroke:#333,stroke-width:2px
    style API fill:#bbf,stroke:#333,stroke-width:2px
    style WS fill:#bbf,stroke:#333,stroke-width:2px
    style WEBHOOK fill:#bbf,stroke:#333,stroke-width:2px
    style VC fill:#bbf,stroke:#333,stroke-width:2px
    style VT fill:#bbf,stroke:#333,stroke-width:2px
    style LLM_LOOP fill:#bbf,stroke:#333,stroke-width:2px
    style LS fill:#bbf,stroke:#333,stroke-width:2px
    style VM fill:#bbf,stroke:#333,stroke-width:2px
    style CB fill:#bbf,stroke:#333,stroke-width:2px
    style RTAPI fill:#bbf,stroke:#333,stroke-width:2px
    style RETELL_SDK fill:#ccf,stroke:#333,stroke-width:2px
    style RETELL_LLM fill:#ccf,stroke:#333,stroke-width:2px
    style RETELL_WEBHOOK fill:#ccf,stroke:#333,stroke-width:2px
```
Title: High-Level Voice (Auto Live) System Architecture
Sources: [orchestrator/api/voice_retell.py:1-20](), [frontend/components/chatbot/chat.tsx:54-66](), [frontend/hooks/use-retell-call.ts:1-13]()

## Voice Backend & Call Lifecycle

The backend manages the entire lifecycle of a voice call, from initiation to termination and usage metering. The primary entry point for initiating a web call is the `POST /api/voice/web-call` endpoint [orchestrator/api/voice_retell.py:99-104](). This endpoint performs several fail-closed gating checks, including platform-wide enablement, Retell credential availability, workspace-specific settings, and monthly usage caps [orchestrator/api/voice_retell.py:118-157]().

Upon successful minting, a `VoiceCall` record is created in the database [orchestrator/core/models/voice_calls.py](), which serves as both the webhook trust boundary and the WebSocket credential. Retell's custom-LLM transport connects to `WS /api/voice/retell/llm-websocket/{call_id}` [orchestrator/api/voice_retell.py:5-10](), allowing Automatos AI's `StreamingChatService` to drive the agent's responses.

Retell also sends call lifecycle events to the `POST /api/voice/retell/events` webhook [orchestrator/api/voice_retell.py:14-15](). These webhooks are crucial for updating the `VoiceCall` status, recording usage in the `voice_meter` [orchestrator/modules/voice/voice_meter.py](), and verifying the call binding [orchestrator/modules/voice/call_binding.py](). The `call_binding` module ensures that the incoming webhook data corresponds to a legitimate, minted call and correctly attributes the conversation to the user and chat thread.

For details, see [Voice Backend & Call Lifecycle](#24.1).

Sources:
- [orchestrator/api/voice_retell.py:5-15]()
- [orchestrator/api/voice_retell.py:99-104]()
- [orchestrator/api/voice_retell.py:118-157]()
- [orchestrator/core/models/voice_calls.py]()
- [orchestrator/modules/voice/call_binding.py]()
- [orchestrator/modules/voice/voice_meter.py]()

## Voice UI Components

The frontend provides a rich user experience for real-time voice interactions. The `LiveVoiceMode` component [frontend/components/voice/LiveVoiceMode.tsx]() is the central control strip for an active call, managing the `useRetellCall` hook [frontend/hooks/use-retell-call.ts]() and feeding presence information to other components.

The `PresenceOrb` [frontend/components/voice/PresenceOrb.tsx]() is a visualizer that renders the agent's "presence" and audio levels. In "background mode," it fills the chat window's background, coming alive when the agent speaks [frontend/components/chatbot/chat.tsx:54-57](). In a compact band, it shows a beam-and-bars strip. The `PresenceOrb` uses a mutable ref for audio levels (`levelsRef`) to avoid unnecessary React re-renders, as the canvas updates in its own `rAF` loop [frontend/hooks/use-retell-call.ts:10-12]().

The `MicHealthControl` component [frontend/components/voice/MicHealthControl.tsx]() helps users diagnose microphone issues, including detecting digital silence using the `mic-health` utility [frontend/lib/voice/mic-health.ts](). The `VoiceErrorBoundary` [frontend/components/voice/VoiceErrorBoundary.tsx]() provides a robust error handling mechanism for the voice subsystem, ensuring that voice-related failures do not crash the entire application.

For details, see [Voice UI Components](#24.2).

Sources:
- [frontend/components/chatbot/chat.tsx:54-57]()
- [frontend/components/voice/LiveVoiceMode.tsx]()
- [frontend/components/voice/MicHealthControl.tsx]()
- [frontend/components/voice/PresenceOrb.tsx]()
- [frontend/components/voice/VoiceErrorBoundary.tsx]()
- [frontend/hooks/use-retell-call.ts]()
- [frontend/hooks/use-retell-call.ts:10-12]()
- [frontend/lib/voice/mic-health.ts]()

## Spoken Style & Voice Telemetry

To ensure Auto sounds natural and professional, the system employs a `speechify()` function [orchestrator/modules/voice/spoken_style.py:21-77]() that sanitizes agent output before it's sent to the TTS engine. This process removes Markdown formatting (e.g., `**bold**`, `- bullets`, `## headers`), speaks link labels instead of URLs, and strips emojis, preventing the TTS from pronouncing punctuation or code snippets [orchestrator/tests/test_voice_quality.py:30-77](). The `SPOKEN_OUTPUT_CONTRACT` [orchestrator/modules/voice/spoken_style.py:10-16]() defines these rules.

The system also captures comprehensive telemetry for voice interactions. `voice_calls` [orchestrator/core/models/voice_calls.py]() and `voice_turns` [orchestrator/core/models/voice_turns.py]() tables store detailed information about each call and individual turn, respectively. This data includes STT/TTS latency decomposition, which is crucial for analyzing and improving voice quality. The `telemetry` module [orchestrator/modules/voice/telemetry.py]() is responsible for recording these metrics.

```mermaid
graph TD
    subgraph "Agent Output Pipeline"
        AGENT_RESPONSE[("Agent Text Response")]
        SPEECHIFY[speechify()]
        TTS[TTS Engine (Retell)]
        SPOKEN_AUDIO[("Spoken Audio")]
    end

    subgraph "Telemetry & Monitoring"
        VOICE_CALLS[voice_calls table]
        VOICE_TURNS[voice_turns table]
        TELEMETRY_MODULE[telemetry.py]
        STT_TTS_LATENCY[STT/TTS Latency Decomposition]
        VOICE_QUALITY_TESTS[Voice Quality Tests]
    end

    AGENT_RESPONSE -- "Raw Text" --> SPEECHIFY
    SPEECHIFY -- "Sanitized Text" --> TTS
    TTS -- "Audio Output" --> SPOKEN_AUDIO

    SPEECHIFY -- "Records Sanitization" --> TELEMETRY_MODULE
    TTS -- "Records Latency" --> TELEMETRY_MODULE
    TELEMETRY_MODULE -- "Stores Call Data" --> VOICE_CALLS
    TELEMETRY_MODULE -- "Stores Turn Data" --> VOICE_TURNS
    VOICE_CALLS -- "Analyzed for" --> STT_TTS_LATENCY
    VOICE_TURNS -- "Analyzed for" --> STT_TTS_LATENCY
    STT_TTS_LATENCY -- "Informs" --> VOICE_QUALITY_TESTS

    style AGENT_RESPONSE fill:#f9f,stroke:#333,stroke-width:2px
    style SPEECHIFY fill:#bbf,stroke:#333,stroke-width:2px
    style TTS fill:#ccf,stroke:#333,stroke-width:2px
    style SPOKEN_AUDIO fill:#f9f,stroke:#333,stroke-width:2px
    style VOICE_CALLS fill:#bbf,stroke:#333,stroke-width:2px
    style VOICE_TURNS fill:#bbf,stroke:#333,stroke-width:2px
    style TELEMETRY_MODULE fill:#bbf,stroke:#333,stroke-width:2px
    style STT_TTS_LATENCY fill:#bbf,stroke:#333,stroke-width:2px
    style VOICE_QUALITY_TESTS fill:#bbf,stroke:#333,stroke-width:2px
```
Title: Spoken Style and Voice Telemetry Flow
Sources: [orchestrator/modules/voice/spoken_style.py:21-77](), [orchestrator/modules/voice/spoken_style.py:10-16](), [orchestrator/tests/test_voice_quality.py:30-77](), [orchestrator/core/models/voice_calls.py](), [orchestrator/core/models/voice_turns.py](), [orchestrator/modules/voice/telemetry.py]()

For details, see [Spoken Style & Voice Telemetry](#24.3).

Sources:
- [orchestrator/core/models/voice_calls.py]()
- [orchestrator/core/models/voice_turns.py]()
- [orchestrator/modules/voice/spoken_style.py]()
- [orchestrator/modules/voice/spoken_style.py:10-16]()
- [orchestrator/modules/voice/spoken_style.py:21-77]()
- [orchestrator/modules/voice/telemetry.py]()
- [orchestrator/tests/test_voice_quality.py:30-77]()

---
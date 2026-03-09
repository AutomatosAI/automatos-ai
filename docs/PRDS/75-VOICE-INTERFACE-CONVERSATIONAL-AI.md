# PRD-74: Voice Interface & Conversational AI

**Version:** 1.0
**Status:** Draft
**Priority:** P1 — High
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-09
**Updated:** 2026-03-09
**Dependencies:** PRD-01 (Core Orchestration Engine), PRD-50 (Universal Orchestrator Router), PRD-55 (Autonomous Assistant / Heartbeats), PRD-73 (Observability & Monitoring Stack)
**Repositories:**
- `automatos-voice` — Voice service (STT/TTS engines, Pipecat pipeline, model management). New repo.
- `automatos-ai` — Integration wiring (API endpoints, voice client, frontend components, migrations)
**Branches:**
- `automatos-voice`: `main` (new repo, starts fresh)
- `automatos-ai`: `feat/prd-74-voice-interface`
**Deployment:** Railway (voice services deployed within the same Railway project as automatos-ai, communicating via private networking)

---

## Executive Summary

Automatos agents are text-only. Users type messages, agents reply with text. This limits the platform to desktop-with-keyboard use cases and excludes hands-free operation, mobile-first usage, accessibility needs, and the natural conversational flow that makes voice assistants feel alive.

This PRD adds voice capabilities to the Automatos platform across three phases:

1. **Phase 1 — Voice Messages:** Record audio in chat, get spoken responses back. Push-to-talk, like voice notes in WhatsApp. Works in web app and mobile.
2. **Phase 2 — High-Quality Voices:** Upgrade TTS to near-human quality with voice cloning, emotion, and multilingual support. Auto gets a distinctive voice.
3. **Phase 3 — Live Conversation:** Real-time bidirectional voice. Talk to Auto like a phone call — interruptions, turn-taking, streaming. Full conversational AI.

### What We're Building

1. **Voice Service** — Self-hosted STT + TTS behind an OpenAI-compatible REST API, deployed as a Railway service
2. **Frontend Voice UI** — Mic button in chat, audio playback for responses, push-to-talk and hands-free modes
3. **Orchestrator Integration** — Voice pipeline wired into the existing message flow (no new orchestration path — voice is a transport, not a routing change)
4. **Real-Time Voice Pipeline** — WebSocket/WebRTC streaming for live conversation with VAD, interruption handling, and turn-taking (Phase 3)
5. **Observability** — Voice-specific metrics and logs feeding into PRD-73's monitoring stack

### What We're NOT Building

- Phone/telephony integration (Twilio, SIP trunks) — future PRD if needed
- Voice-based agent-to-agent communication — agents communicate via existing channels
- Custom model training or fine-tuning — we use pre-trained open-source models
- Speaker identification / voice biometrics for auth — future consideration
- Offline/on-device voice processing — all processing is server-side
- A replacement for text chat — voice is additive, text remains the primary interface

---

## 1. Architecture Overview

### 1.1 Phase 1+2: Voice Messages (REST API)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend (Web / Mobile)                                              │
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │ Chat Input   │    │ Mic Button   │    │ Audio Player         │    │
│  │ (text)       │    │ (record)     │    │ (play TTS response)  │    │
│  └──────┬───────┘    └──────┬───────┘    └──────────▲───────────┘    │
│         │                    │                       │                 │
└─────────┼────────────────────┼───────────────────────┼─────────────────┘
          │                    │ audio blob             │ audio blob
          │                    ▼                        │
     ┌────┼──── Backend (Orchestrator) ────────────────┼───────┐
     │    │                                             │       │
     │    │    ┌─────────────────────────────────┐     │       │
     │    │    │  POST /api/chat/voice           │     │       │
     │    │    │  1. Receive audio                │     │       │
     │    │    │  2. → Voice Service /v1/audio/   │     │       │
     │    │    │     transcriptions (STT)         │     │       │
     │    │    │  3. Transcript → Universal Router│     │       │
     │    │    │     (existing chat pipeline)     │     │       │
     │    │    │  4. Agent text response           │     │       │
     │    │    │  5. → Voice Service /v1/audio/   │     │       │
     │    │    │     speech (TTS)                  │     │       │
     │    │    │  6. Return audio + text           │     │       │
     │    │    └─────────────────────────────────┘     │       │
     │    │                                             │       │
     │    └──── text path (unchanged) ─────────────────┘       │
     │                         │                                 │
     └─────────────────────────┼─────────────────────────────────┘
                               │
                               ▼
     ┌─────────────── Voice Service (Railway) ──────────────────┐
     │                                                           │
     │  Speaches / Custom FastAPI                                │
     │  ┌──────────────────┐  ┌──────────────────┐             │
     │  │ /v1/audio/       │  │ /v1/audio/       │             │
     │  │ transcriptions   │  │ speech           │             │
     │  │ (STT: Whisper)   │  │ (TTS: Kokoro →   │             │
     │  │                  │  │  Chatterbox)     │             │
     │  └──────────────────┘  └──────────────────┘             │
     │                                                           │
     │  /health  /metrics                                        │
     └───────────────────────────────────────────────────────────┘
```

### 1.2 Phase 3: Real-Time Conversation (WebSocket/WebRTC)

```
┌──────── Frontend ──────────┐      ┌──────── Voice Pipeline ────────────┐
│                              │      │  (Pipecat or LiveKit Agents)       │
│  Mic → MediaStream ─────────┼──WS──┼→ VAD → STT (streaming) ──────┐   │
│                              │      │                                │   │
│  Speaker ◄──── AudioStream ──┼──WS──┼─ TTS (streaming) ◄───────┐  │   │
│                              │      │                            │  │   │
│  Interrupt Detection ────────┼──WS──┼→ Cancel TTS playback      │  │   │
│                              │      │                            │  │   │
└──────────────────────────────┘      │  ┌──────────────────────┐  │  │   │
                                      │  │ Orchestrator Client  │  │  │   │
                                      │  │ (streams transcript  │◄─┘  │   │
                                      │  │  to chat pipeline,   │─────┘   │
                                      │  │  streams response    │         │
                                      │  │  back to TTS)        │         │
                                      │  └──────────────────────┘         │
                                      │                                    │
                                      │  Uses STT + TTS engines from      │
                                      │  Voice Service (Phase 1/2)         │
                                      └────────────────────────────────────┘
```

### 1.3 Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Voice is a transport, not a route | Voice messages enter the same Universal Router pipeline as text | No duplicate orchestration logic. Agents don't know or care if input was voice or text. |
| OpenAI-compatible API | Voice service exposes `/v1/audio/transcriptions` and `/v1/audio/speech` | Can swap between self-hosted and OpenAI with a URL change. Frontend/backend code stays identical. |
| Separate repo + Railway service | `automatos-voice` is its own repo, deployed as independent Railway service(s) | Heavy ML deps (PyTorch, model weights) stay out of the orchestrator. Independent CI/CD, scaling, and release cycle. |
| Two-repo boundary | `automatos-voice` owns inference. `automatos-ai` owns integration. Contract is the OpenAI-compatible REST API. | Clean separation of concerns. Voice service is replaceable — swap to OpenAI's hosted API by changing one URL. |
| Phase 3 adds a pipeline layer, doesn't replace Phase 1 | REST endpoints remain for voice messages; WebSocket added for real-time | Voice messages (async) and live conversation (streaming) are both valid UX patterns. |
| STT + TTS engines are swappable | Provider abstraction from day one | Start with Whisper + Kokoro, upgrade to Chatterbox/CosyVoice without changing integration code. |

---

## 2. Phase 1: Voice Messages

**Goal:** Users can send voice messages in chat and hear spoken responses. Push-to-talk UX.

### 2.1 Voice Service Deployment

**Primary option:** [Speaches](https://github.com/speaches-ai/speaches) (MIT, 3k stars, actively maintained)
- Single Docker container with STT (Faster-Whisper) + TTS (Kokoro/Piper)
- OpenAI-compatible API out of the box
- Docker Compose configs for CPU and CUDA

**Fallback option:** Custom FastAPI service wrapping Faster-Whisper + Kokoro-FastAPI
- Only if Speaches proves too opinionated or unstable
- Same OpenAI-compatible API contract

**Railway service config:**

| Attribute | Value |
|-----------|-------|
| Service name | `voice-service` |
| Image | `ghcr.io/speaches-ai/speaches:latest` (or custom Dockerfile) |
| Port | 8300 |
| Internal DNS | `voice-service.railway.internal:8300` |
| Public domain | None (backend proxies all requests) |
| Volume | 5GB (model cache — Whisper + TTS models auto-download on first boot) |
| Resources | 2 vCPU / 4GB RAM minimum (CPU inference); GPU recommended for production |

**Environment variables:**

```bash
# STT config
WHISPER__MODEL=Systran/faster-whisper-large-v3
WHISPER__DEVICE=auto          # cpu or cuda
WHISPER__COMPUTE_TYPE=float16  # int8 for CPU, float16 for GPU

# TTS config
TTS__DEFAULT_MODEL=kokoro      # or piper
TTS__DEFAULT_VOICE=af_heart    # Kokoro voice ID

# Service config
ENVIRONMENT=production
LOG_LEVEL=info
```

### 2.2 Backend Integration

**New endpoint:** `POST /api/chat/voice`

```python
# orchestrator/api/chat_voice.py

@router.post("/api/chat/voice")
async def voice_chat(
    audio: UploadFile,                    # Audio blob from frontend mic
    conversation_id: str = Form(...),
    agent_id: int = Form(None),           # Optional — uses default agent if omitted
    response_format: str = Form("audio"), # "audio" | "text" | "both"
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
):
    """
    Voice-in, voice-out chat endpoint.

    Flow:
    1. STT: audio → transcript (via voice service)
    2. Chat: transcript → agent response (via existing pipeline)
    3. TTS: agent response → audio (via voice service)
    4. Return: audio blob + transcript + agent text
    """
```

**Response format:**

```json
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "transcript": "What's the status of the deployment?",
  "response_text": "The deployment completed successfully 10 minutes ago...",
  "audio_url": "/api/chat/voice/audio/{message_id}",
  "audio_format": "mp3",
  "duration_ms": 3200,
  "stt_latency_ms": 450,
  "tts_latency_ms": 800
}
```

**Voice service client:**

```python
# orchestrator/modules/voice/client.py

class VoiceServiceClient:
    """OpenAI-compatible client for the self-hosted voice service."""

    def __init__(self):
        self.base_url = config.VOICE_SERVICE_URL  # e.g. http://voice-service.railway.internal:8300
        self.timeout = config.VOICE_SERVICE_TIMEOUT  # 30s default

    async def transcribe(self, audio: bytes, language: str = None) -> TranscriptionResult:
        """POST /v1/audio/transcriptions — returns transcript + metadata."""

    async def synthesize(self, text: str, voice: str = None, speed: float = 1.0) -> SynthesisResult:
        """POST /v1/audio/speech — returns audio bytes + metadata."""

    async def health(self) -> bool:
        """GET /health — voice service health check."""
```

**Config additions (`config.py`):**

```python
# Voice Service
VOICE_SERVICE_URL = os.getenv("VOICE_SERVICE_URL", "http://voice-service.railway.internal:8300")
VOICE_SERVICE_TIMEOUT = int(os.getenv("VOICE_SERVICE_TIMEOUT", "30"))
VOICE_STT_MODEL = os.getenv("VOICE_STT_MODEL", "Systran/faster-whisper-large-v3")
VOICE_TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "kokoro")
VOICE_TTS_DEFAULT_VOICE = os.getenv("VOICE_TTS_DEFAULT_VOICE", "af_heart")
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
VOICE_MAX_AUDIO_SIZE_MB = int(os.getenv("VOICE_MAX_AUDIO_SIZE_MB", "25"))
VOICE_MAX_DURATION_SECONDS = int(os.getenv("VOICE_MAX_DURATION_SECONDS", "120"))
```

### 2.3 Frontend Voice UI

**Chat input changes:**

```
┌─────────────────────────────────────────────────────┐
│  Type a message...                    🎤  📎  ➤    │
│                                       ^^^           │
│                                       Mic button    │
└─────────────────────────────────────────────────────┘

Recording state:
┌─────────────────────────────────────────────────────┐
│  🔴 Recording... 0:03                 ⏹  🗑  ➤    │
│  ████████████░░░░░░░░░░░░░░░░░░                     │
│  waveform visualization                              │
└─────────────────────────────────────────────────────┘
```

**Components to build:**

| Component | Purpose |
|-----------|---------|
| `VoiceMicButton` | Toggle recording, visual feedback (pulsing, waveform) |
| `VoiceRecorder` | MediaRecorder API wrapper, audio blob capture, format conversion |
| `VoicePlayer` | Audio playback for TTS responses, with play/pause/speed controls |
| `VoiceMessage` | Chat bubble variant showing waveform + transcript + play button |
| `VoiceSettings` | User preferences: auto-play responses, voice selection, speed |

**Audio format:**

| Direction | Format | Reason |
|-----------|--------|--------|
| User → STT | WebM/Opus (browser native) or WAV | MediaRecorder default; Whisper handles both |
| TTS → User | MP3 (default) or OGG/Opus | Smallest size for playback; configurable |

**Key UX decisions:**

- Mic button replaces send button while recording (no simultaneous text + voice)
- Transcript is always shown alongside audio (accessibility + searchability)
- Auto-play TTS response is opt-in (off by default — respects quiet environments)
- Voice messages are stored as regular messages with `type: "voice"` and audio URL
- Long-press mic for hands-free recording (release to send) — tap for toggle mode

### 2.4 Audio Storage

Voice audio files are stored in S3, not in the database.

```
s3://automatos-ai/workspaces/{workspace_id}/voice/{message_id}.{format}
```

**Retention:** Same as workspace document retention policy. Voice audio is ephemeral by default — 30-day TTL unless the user pins the message.

**Database schema addition:**

```sql
-- Add to existing messages table (or create voice_messages if separate)
ALTER TABLE messages ADD COLUMN voice_metadata JSONB;

-- voice_metadata structure:
-- {
--   "type": "voice",
--   "audio_key": "workspaces/{wid}/voice/{mid}.mp3",
--   "duration_ms": 3200,
--   "transcript": "What's the status...",
--   "stt_model": "faster-whisper-large-v3",
--   "tts_model": "kokoro",
--   "tts_voice": "af_heart",
--   "stt_latency_ms": 450,
--   "tts_latency_ms": 800,
--   "audio_format": "mp3",
--   "audio_size_bytes": 51200
-- }
```

---

## 3. Phase 2: High-Quality Voices

**Goal:** Upgrade TTS quality to near-human level. Give Auto a distinctive, consistent voice. Support voice cloning and multilingual output.

### 3.1 TTS Engine Upgrade

Phase 1 ships with Kokoro (82M params, Apache 2.0) — good quality, tiny footprint. Phase 2 upgrades to a top-tier engine.

**Candidates (ranked by fit):**

| Engine | Stars | Quality | Voice Clone | License | Streaming | Docker | Decision |
|--------|-------|---------|-------------|---------|-----------|--------|----------|
| **Chatterbox** (Resemble AI) | 23k | Beats ElevenLabs in blind tests | Yes, zero-shot | **MIT** | Turbo variant | Community server | **Primary choice** |
| **CosyVoice 2** (Alibaba) | 20k | Top tier, 150ms streaming | Yes, multilingual | **Apache 2.0** | Yes, native | **Yes (FastAPI+gRPC+Docker)** | **Secondary choice** |
| **Orpheus** (Canopy AI) | 6k | Near top, LLM-native prosody | Yes | **Apache 2.0** | 25-50ms latency | Community FastAPI | Alternative |
| Fish Speech | 25k | #1 TTS-Arena | Yes | CC-BY-NC (non-commercial) | Yes | Yes | Excluded — non-commercial license |
| F5-TTS | 14k | Excellent | Yes | CC-BY-NC (non-commercial) | Chunked | Yes | Excluded — non-commercial license |

**Recommended path:**
1. Start with **Chatterbox** — MIT license, beats ElevenLabs, emotion control, 23 languages
2. Keep **CosyVoice 2** as fallback — best production deployment story (Docker + gRPC + TensorRT)
3. Both fit behind the same OpenAI-compatible API contract from Phase 1

**Voice provider abstraction:**

```python
# orchestrator/modules/voice/providers/base.py

class TTSProvider(ABC):
    """Abstract TTS provider — swap engines without changing integration code."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str, **kwargs) -> SynthesisResult:
        """Convert text to audio."""

    @abstractmethod
    async def list_voices(self) -> list[VoiceInfo]:
        """Available voices for this provider."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Provider health status."""


# orchestrator/modules/voice/providers/speaches.py — Phase 1
# orchestrator/modules/voice/providers/chatterbox.py — Phase 2
# orchestrator/modules/voice/providers/cosyvoice.py — Phase 2 alt
# orchestrator/modules/voice/providers/openai.py — Cloud fallback
```

### 3.2 Auto's Voice Identity

Auto (the platform's autonomous assistant) gets a distinctive voice configured at the platform level.

```python
# config.py additions
AUTO_VOICE_ID = os.getenv("AUTO_VOICE_ID", "auto_default")
AUTO_VOICE_PROVIDER = os.getenv("AUTO_VOICE_PROVIDER", "chatterbox")
```

**Voice profile storage:**

```sql
CREATE TABLE voice_profiles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL REFERENCES workspaces(id),
    name            TEXT NOT NULL,              -- "Auto Default", "Custom Agent Voice"
    provider        TEXT NOT NULL,              -- "chatterbox", "cosyvoice", "kokoro"
    voice_id        TEXT NOT NULL,              -- Provider-specific voice identifier
    reference_audio TEXT,                       -- S3 key for cloned voice reference audio
    settings        JSONB DEFAULT '{}',         -- speed, pitch, emotion, language
    is_default      BOOLEAN DEFAULT FALSE,      -- Default voice for workspace
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.3 Per-Agent Voice Assignment

Agents can have distinct voices. A customer-facing agent might sound different from a technical agent.

```sql
-- Add to agents table
ALTER TABLE agents ADD COLUMN voice_profile_id UUID REFERENCES voice_profiles(id);
```

When TTS is invoked, the system checks:
1. Agent has a `voice_profile_id` → use that voice
2. No agent voice → use workspace default voice
3. No workspace default → use platform default (`AUTO_VOICE_ID`)

### 3.4 Voice Cloning Workflow

For workspaces that want custom agent voices:

1. User uploads 10-30 seconds of reference audio via Settings → Voices
2. Backend stores reference audio in S3: `s3://automatos-ai/workspaces/{wid}/voices/{profile_id}/reference.wav`
3. Voice profile created with `reference_audio` pointing to S3 key
4. TTS provider uses reference audio for zero-shot cloning at synthesis time
5. User can preview and assign to agents

**Security considerations:**
- Voice cloning is workspace-admin only
- Reference audio scanned for minimum quality (duration, sample rate, SNR)
- Consent acknowledgement required before enabling cloning
- Rate-limited to prevent abuse

---

## 4. Phase 3: Live Conversation

**Goal:** Real-time bidirectional voice. Talk to Auto like a phone call. Interruptions work. Streaming in both directions.

### 4.1 Framework Selection

| Framework | Stars | Protocol | Self-Hosted | Provider Coverage | Fit |
|-----------|-------|----------|-------------|-------------------|-----|
| **Pipecat** (Daily.co) | 5k+ | WebSocket via Daily/LiveKit | Yes | 15+ STT, 20+ TTS providers | **Best fit** — Python, modular, vendor-agnostic |
| **LiveKit Agents** | 22k + 6k | WebRTC | Yes (Go server) | Plugin-based | Strong infra but heavier to deploy |
| **EchoKit** | New | WebSocket | Yes (Rust) | Config-driven | Lightweight but immature |
| TEN Framework | 5k+ | WebSocket | Yes | Graph-based | Over-complex for our needs |

**Recommended: Pipecat**
- Python-native (fits our stack)
- Pluggable STT/TTS providers (use our Phase 1/2 engines)
- Handles VAD, interruption, turn-taking, streaming out of the box
- BSD 2-Clause license
- NVIDIA partnership, active development
- Can run with or without Daily.co's WebRTC infra (local WebSocket transport available)

### 4.2 Voice Pipeline Architecture

```python
# orchestrator/modules/voice/pipeline.py (conceptual)

from pipecat.pipeline import Pipeline
from pipecat.transports.websocket import WebSocketTransport
from pipecat.services.stt import WhisperSTTService
from pipecat.services.tts import ChatterboxTTSService
from pipecat.processors.vad import SileroVADProcessor

pipeline = Pipeline([
    WebSocketTransport(port=8301),       # Browser connects here
    SileroVADProcessor(),                 # Voice activity detection
    WhisperSTTService(                    # STT — uses our voice service
        url=config.VOICE_SERVICE_URL,
    ),
    AutomatosOrchestratorProcessor(       # Bridge to existing chat pipeline
        orchestrator_url=config.BACKEND_URL,
    ),
    ChatterboxTTSService(                 # TTS — uses our voice service
        url=config.VOICE_SERVICE_URL,
        voice=config.AUTO_VOICE_ID,
    ),
])
```

### 4.3 Real-Time Features

| Feature | Implementation |
|---------|---------------|
| **Voice Activity Detection (VAD)** | Silero VAD (included in Pipecat). Detects when user starts/stops speaking. No manual push-to-talk needed. |
| **Interruption handling** | When user speaks while TTS is playing, immediately stop TTS, discard buffered audio, process new user input. |
| **Turn-taking** | VAD silence threshold (configurable, default 800ms) determines when user is "done speaking." |
| **Streaming STT** | Partial transcripts stream to LLM as user speaks. LLM can begin generating before user finishes. |
| **Streaming TTS** | LLM response tokens stream to TTS. First audio chunk plays before full response is generated. |
| **Backpressure** | If TTS generation is slower than playback, buffer. If faster, queue. Never drop audio. |
| **Graceful degradation** | If voice service is down, fall back to text-only with a "voice unavailable" indicator. |

### 4.4 Frontend Real-Time UI

```
┌─────────────────────────────────────────────────────┐
│  🟢 Connected — Live Voice Mode                     │
│                                                       │
│  ┌───────────────────────────────────────────────┐   │
│  │  User (speaking): "What agents are running    │   │
│  │  right now?"                                   │   │
│  │  ░░░░░░░░░░░░ (live waveform)                 │   │
│  └───────────────────────────────────────────────┘   │
│                                                       │
│  ┌───────────────────────────────────────────────┐   │
│  │  Auto (responding): "You have 3 agents        │   │
│  │  currently active..."                          │   │
│  │  🔊 ████████████░░░░ (playback waveform)      │   │
│  └───────────────────────────────────────────────┘   │
│                                                       │
│  [End Voice Call]              [Mute]  [Settings]    │
└─────────────────────────────────────────────────────┘
```

**Components:**

| Component | Purpose |
|-----------|---------|
| `VoiceCallPanel` | Full-screen or docked panel for live voice mode |
| `LiveTranscript` | Real-time transcript display as user speaks |
| `VoiceActivityIndicator` | Visual feedback for who is "speaking" |
| `InterruptionHandler` | Detects user speech during playback, cancels TTS |

### 4.5 Railway Deployment (Phase 3)

The voice pipeline runs as a separate Railway service:

| Attribute | Value |
|-----------|-------|
| Service name | `voice-pipeline` |
| Image | Custom Dockerfile (Pipecat + providers) |
| Port | 8301 (WebSocket) |
| Internal DNS | `voice-pipeline.railway.internal:8301` |
| Public domain | Yes — `voice.automatos.app` (WebSocket needs public access for browser connections) |
| Resources | 2 vCPU / 4GB RAM minimum |

**Note:** The voice pipeline service is **in addition to** the voice service from Phase 1. Phase 1's voice service provides the STT/TTS engines. Phase 3's voice pipeline orchestrates real-time streaming between them.

```
Frontend (browser)
    ↕ WebSocket
Voice Pipeline (Pipecat) — voice-pipeline.railway.internal:8301
    ↕ HTTP (internal)
Voice Service (STT/TTS engines) — voice-service.railway.internal:8300
    ↕ HTTP (internal)
Backend (Orchestrator) — backend.railway.internal:8000
```

---

## 5. Observability & Monitoring Integration (PRD-73)

Voice adds new failure modes and latency-sensitive paths. Full integration with PRD-73's monitoring stack is required.

### 5.1 Prometheus Metrics

The voice service and voice pipeline expose `/metrics` endpoints scraped by Prometheus.

**Voice Service metrics:**

```
# STT metrics
voice_stt_requests_total{model, language, status}          # Counter
voice_stt_duration_seconds{model}                          # Histogram (processing time)
voice_stt_audio_duration_seconds{model}                    # Histogram (input audio length)
voice_stt_errors_total{model, error_type}                  # Counter

# TTS metrics
voice_tts_requests_total{model, voice, status}             # Counter
voice_tts_duration_seconds{model, voice}                   # Histogram (processing time)
voice_tts_characters_total{model, voice}                   # Counter (text input length)
voice_tts_audio_duration_seconds{model, voice}             # Histogram (output audio length)
voice_tts_errors_total{model, error_type}                  # Counter

# Model metrics
voice_model_loaded{model, type}                            # Gauge (1=loaded, 0=not)
voice_model_load_duration_seconds{model}                   # Histogram
voice_inference_queue_depth{model}                         # Gauge
```

**Voice Pipeline metrics (Phase 3):**

```
# Session metrics
voice_sessions_active                                       # Gauge
voice_sessions_total{status}                                # Counter (completed, errored, timeout)
voice_session_duration_seconds                              # Histogram

# Latency metrics (critical for UX)
voice_stt_first_token_latency_seconds                      # Histogram (mic → first transcript word)
voice_tts_first_audio_latency_seconds                      # Histogram (text → first audio chunk)
voice_end_to_end_latency_seconds                           # Histogram (user done speaking → first audio response)

# Interruption metrics
voice_interruptions_total                                   # Counter
voice_tts_cancelled_total                                   # Counter (TTS stopped mid-playback)

# VAD metrics
voice_vad_false_triggers_total                              # Counter (detected speech that wasn't)
voice_silence_duration_seconds                              # Histogram (gaps between utterances)
```

### 5.2 Prometheus Scrape Config Addition

```yaml
# Added to prometheus.yml scrape_configs
- job_name: 'voice-service'
  static_configs:
    - targets: ['voice-service.railway.internal:8300']
  scrape_interval: 15s

- job_name: 'voice-pipeline'
  static_configs:
    - targets: ['voice-pipeline.railway.internal:8301']
  scrape_interval: 15s
```

### 5.3 Alert Rules

```yaml
# monitoring/prometheus/rules/voice.yml

groups:
  - name: voice-service
    rules:
      # Service health
      - alert: VoiceServiceDown
        expr: up{job="voice-service"} == 0
        for: 1m
        labels:
          severity: warning      # Warning not critical — text chat still works
        annotations:
          summary: "Voice service is down"
          description: "Voice service unreachable for >1m. Voice features degraded, text chat unaffected."
          runbook: "Check voice-service Railway logs. Restart if OOM. Verify model download completed."

      # STT latency
      - alert: VoiceSTTSlow
        expr: histogram_quantile(0.95, rate(voice_stt_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "STT p95 latency >5s"
          description: "Speech-to-text processing is slow. Check GPU availability and model size."

      # TTS latency
      - alert: VoiceTTSSlow
        expr: histogram_quantile(0.95, rate(voice_tts_duration_seconds_bucket[5m])) > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTS p95 latency >3s"
          description: "Text-to-speech generation is slow. Consider lighter model or GPU upgrade."

      # Error rate
      - alert: VoiceHighErrorRate
        expr: rate(voice_stt_errors_total[5m]) / rate(voice_stt_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Voice STT error rate >10%"
          description: "{{ $value | humanizePercentage }} of STT requests failing."

      # Queue depth (inference backlog)
      - alert: VoiceQueueBacklog
        expr: voice_inference_queue_depth > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Voice inference queue backlog"
          description: "{{ $value }} requests queued. Voice service may need scaling."

      # Model not loaded
      - alert: VoiceModelNotLoaded
        expr: voice_model_loaded == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Voice model failed to load"
          description: "Model {{ $labels.model }} not loaded. Voice service non-functional."

  - name: voice-pipeline
    rules:
      # Real-time latency (Phase 3)
      - alert: VoiceE2ELatencyHigh
        expr: histogram_quantile(0.95, rate(voice_end_to_end_latency_seconds_bucket[5m])) > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Voice end-to-end latency >3s p95"
          description: "Live conversation feels laggy. Target is <2s end-to-end."

      # Session failures
      - alert: VoiceSessionFailureRate
        expr: rate(voice_sessions_total{status="errored"}[15m]) / rate(voice_sessions_total[15m]) > 0.15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Voice session failure rate >15%"
```

### 5.4 Grafana Dashboard: Voice Performance

A new dashboard added to PRD-73's Grafana provisioning:

| Panel | Visualization | Data Source |
|-------|--------------|-------------|
| Voice Service Health | Status indicator (up/down) | Prometheus |
| STT Requests/min | Time series | `rate(voice_stt_requests_total[1m])` |
| TTS Requests/min | Time series | `rate(voice_tts_requests_total[1m])` |
| STT Latency (p50/p95/p99) | Time series | `histogram_quantile(voice_stt_duration_seconds)` |
| TTS Latency (p50/p95/p99) | Time series | `histogram_quantile(voice_tts_duration_seconds)` |
| End-to-End Latency (Phase 3) | Time series | `voice_end_to_end_latency_seconds` |
| Error Rate | Time series | `rate(voice_*_errors_total[5m])` |
| Inference Queue Depth | Gauge | `voice_inference_queue_depth` |
| Active Voice Sessions (Phase 3) | Gauge | `voice_sessions_active` |
| Model Load Status | Table | `voice_model_loaded` |
| Audio Processed (minutes) | Stat panel | `sum(rate(voice_stt_audio_duration_seconds_sum[1h]))` |
| Voice Logs | Log panel | Loki: `{service="voice-service"}` |

### 5.5 Loki Log Labels

Voice service logs use structured JSON with these labels:

```json
{
  "service": "voice-service",
  "component": "stt|tts|pipeline|health",
  "workspace_id": "uuid",
  "conversation_id": "uuid",
  "model": "faster-whisper-large-v3",
  "level": "info|warn|error",
  "latency_ms": 450,
  "audio_duration_ms": 3200
}
```

### 5.6 Backend Logging

All voice operations logged through the backend's existing structured logging with voice-specific fields:

```python
logger.info(
    "voice_stt_complete",
    extra={
        "workspace_id": workspace_id,
        "conversation_id": conversation_id,
        "model": config.VOICE_STT_MODEL,
        "audio_duration_ms": audio_duration,
        "processing_ms": stt_latency,
        "transcript_length": len(transcript),
        "language_detected": language,
    }
)

logger.info(
    "voice_tts_complete",
    extra={
        "workspace_id": workspace_id,
        "conversation_id": conversation_id,
        "model": config.VOICE_TTS_MODEL,
        "voice": voice_id,
        "text_length": len(text),
        "processing_ms": tts_latency,
        "audio_duration_ms": output_duration,
    }
)
```

---

## 6. Security Considerations

| Concern | Mitigation |
|---------|------------|
| **Audio contains sensitive data** | Audio files in S3 with workspace-scoped keys. Same access controls as documents. 30-day TTL default. |
| **Voice cloning abuse** | Cloning restricted to workspace admins. Consent acknowledgement required. Rate-limited. |
| **Audio injection attacks** | Validate audio format, duration, and size before processing. Reject files >25MB or >120s. |
| **Voice service access** | Internal only (no public domain). Backend proxies all requests. No direct frontend-to-voice-service communication in Phase 1/2. |
| **WebSocket auth (Phase 3)** | Voice pipeline WebSocket requires JWT token in connection handshake. Same auth as chat WebSocket. |
| **Model supply chain** | Pin model versions. Verify HuggingFace checksums on download. Don't auto-update models in production. |
| **No hardcoded credentials** | All config via `config.py` → Railway env vars. Voice service API keys (if any) as Railway secrets. |
| **Denial of service** | Rate limit voice endpoints per workspace (e.g., 60 STT requests/min, 60 TTS requests/min). Queue with backpressure. |

---

## 7. Mobile Considerations

Voice is especially valuable on mobile where typing is slower.

| Aspect | Implementation |
|--------|---------------|
| **Mic permissions** | Request on first mic button tap. Show clear permission rationale. |
| **Background audio** | TTS playback continues when app is backgrounded (mobile web audio API). |
| **Bandwidth** | Compress audio before upload (Opus codec, ~32kbps). Keep TTS responses in MP3/Opus. |
| **Offline indicator** | If voice service unreachable, mic button shows "unavailable" state. Text input remains functional. |
| **Touch UX** | Long-press mic for push-to-talk. Tap for toggle mode. Swipe to cancel recording. |
| **Low-latency priority** | Mobile users expect <2s response. Phase 1 target: STT <1s + TTS <1.5s for typical messages. |

---

## 8. API Reference

### 8.1 Voice Chat Endpoint

```
POST /api/chat/voice
Content-Type: multipart/form-data
Authorization: Bearer <token>

Parameters:
  audio              File    required   Audio file (WebM, WAV, MP3, OGG; max 25MB, max 120s)
  conversation_id    string  required   Conversation UUID
  agent_id           int     optional   Target agent (defaults to workspace default)
  response_format    string  optional   "audio" | "text" | "both" (default: "both")
  language           string  optional   ISO 639-1 language hint for STT (auto-detected if omitted)
  voice              string  optional   Override TTS voice for this request

Response: 200 OK
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "transcript": "string",
  "response_text": "string",
  "audio_url": "/api/chat/voice/audio/{message_id}",
  "audio_format": "mp3",
  "duration_ms": 3200,
  "stt_latency_ms": 450,
  "tts_latency_ms": 800
}

Errors:
  400  Invalid audio format or exceeds limits
  401  Unauthorized
  413  Audio file too large
  422  STT failed to transcribe (unintelligible audio)
  503  Voice service unavailable (text chat remains functional)
```

### 8.2 Audio Retrieval

```
GET /api/chat/voice/audio/{message_id}
Authorization: Bearer <token>

Response: 200 OK
Content-Type: audio/mpeg
Content-Disposition: inline

Errors:
  404  Audio not found or expired (30-day TTL)
  403  User doesn't have access to this conversation
```

### 8.3 Voice Profiles (Phase 2)

```
GET    /api/voice/profiles                    List workspace voice profiles
POST   /api/voice/profiles                    Create voice profile (admin only)
GET    /api/voice/profiles/{id}               Get profile details
PUT    /api/voice/profiles/{id}               Update profile settings
DELETE /api/voice/profiles/{id}               Delete profile (admin only)
POST   /api/voice/profiles/{id}/preview       Generate preview audio clip
POST   /api/voice/profiles/clone              Upload reference audio for cloning (admin only)
```

### 8.4 Voice WebSocket (Phase 3)

```
WS /api/voice/stream?token={jwt}&conversation_id={uuid}

Client → Server messages:
  { "type": "audio_chunk", "data": "<base64 audio>" }
  { "type": "interrupt" }
  { "type": "end_turn" }
  { "type": "config", "vad_threshold": 0.5, "silence_ms": 800 }

Server → Client messages:
  { "type": "transcript_partial", "text": "What's the..." }
  { "type": "transcript_final", "text": "What's the status?" }
  { "type": "response_text_chunk", "text": "You have" }
  { "type": "response_audio_chunk", "data": "<base64 audio>" }
  { "type": "response_complete" }
  { "type": "vad_state", "speaking": true }
  { "type": "error", "message": "...", "code": "..." }
```

---

## 9. Implementation Phases

### Phase 1: Voice Messages (4-6 weeks)

**`automatos-voice` repo (new):**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Create `automatos-voice` repo with README, .env.example | P0 | S | New GitHub repo under Automatos-AI-Platform org |
| Dockerfile for voice-service (Speaches-based or custom FastAPI) | P0 | M | Speaches image + custom config layer |
| `/health` endpoint | P0 | S | JSON health status for Prometheus |
| `/metrics` endpoint (Prometheus) | P1 | M | STT/TTS request counters, latency histograms, queue depth |
| Docker Compose for local dev | P0 | S | voice-service on port 8300 |
| Deploy voice-service as Railway service | P0 | M | Docker image, 5GB volume for models, env config |
| Railway service config (`.toml`) | P0 | S | voice-service.toml |
| Voice alert rules YAML (for automatos-monitoring) | P1 | S | VoiceServiceDown, latency, error rate |
| Local dev setup script (model download, dirs) | P2 | S | `scripts/setup-local.sh` |

**`automatos-ai` repo (existing):**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Add `VOICE_*` config constants to `config.py` | P0 | S | All config centralized per platform rules |
| Build `VoiceServiceClient` (STT + TTS HTTP client) | P0 | M | OpenAI-compatible, async, retry + timeout |
| Build `POST /api/chat/voice` endpoint | P0 | M | Audio upload → STT → router → TTS → response |
| Build `GET /api/chat/voice/audio/{id}` endpoint | P0 | S | S3 presigned URL or proxy |
| Add `voice_metadata` column to messages | P0 | S | Alembic migration |
| S3 audio storage (upload/retrieve/TTL) | P0 | M | Reuse existing S3 client |
| Input validation (format, size, duration limits) | P0 | S | Security boundary |
| Rate limiting on voice endpoints | P1 | S | Per-workspace throttle |
| Loki structured logging for voice operations | P1 | S | Labels: service, component, workspace_id |
| Feature flag: `VOICE_ENABLED` | P1 | S | Kill switch |
| Frontend: `VoiceMicButton` component | P0 | M | MediaRecorder, visual feedback |
| Frontend: `VoiceRecorder` (audio capture + encoding) | P0 | M | WebM/Opus capture, size validation |
| Frontend: `VoicePlayer` (audio playback) | P0 | M | Play/pause/speed controls |
| Frontend: `VoiceMessage` chat bubble variant | P0 | S | Waveform + transcript + play button |
| Frontend: `VoiceSettings` (auto-play, speed) | P2 | S | User preferences |
| Graceful degradation (voice unavailable indicator) | P2 | S | Mic button disabled state |

**`automatos-monitoring` repo (PRD-73):**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Add voice-service scrape target to Prometheus config | P1 | S | `voice-service.railway.internal:8300` |
| Add voice alert rules to Prometheus rules directory | P1 | S | Copy from automatos-voice/monitoring/ |
| Build Grafana Voice Performance dashboard | P1 | M | New dashboard JSON |

### Phase 2: High-Quality Voices (3-4 weeks)

**`automatos-voice` repo:**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Add Chatterbox TTS to voice-service (Dockerfile + config) | P0 | M | Second TTS engine alongside Kokoro |
| Chatterbox Docker image or custom wrapper | P0 | M | GPU if available, CPU fallback |
| Voice cloning inference endpoint (reference audio → synthesis) | P1 | M | Zero-shot cloning via Chatterbox |
| CosyVoice 2 integration (backup TTS engine) | P2 | M | Alternative if Chatterbox disappoints |

**`automatos-ai` repo:**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Build TTS provider abstraction (`TTSProvider` ABC) | P0 | M | Base class + self-hosted + OpenAI providers |
| Voice profiles DB table + migration | P0 | S | `voice_profiles` table |
| Voice profiles CRUD API | P0 | M | Admin endpoints |
| Per-agent voice assignment | P0 | S | `voice_profile_id` on agents table, migration |
| Auto's default voice configuration | P1 | S | Platform-level config |
| Voice cloning workflow (upload reference → create profile) | P1 | M | Admin only, with consent, S3 storage |
| Frontend: Voice selection in agent settings | P1 | M | Dropdown + preview playback |
| Frontend: Voice profiles management page | P1 | M | Settings → Voices |

### Phase 3: Live Conversation (6-8 weeks)

**`automatos-voice` repo:**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Evaluate Pipecat vs LiveKit Agents (spike) | P0 | M | 1-week spike, build POC with each |
| Build voice-pipeline service (Pipecat) | P0 | L | New service in repo, Dockerfile |
| Build `AutomatosOrchestratorProcessor` (pipeline ↔ orchestrator bridge) | P0 | L | HTTP client to backend for chat routing |
| VAD integration (Silero) | P0 | M | Voice activity detection |
| Streaming STT integration | P0 | M | Partial transcripts via voice-service |
| Streaming TTS integration | P0 | M | Audio chunk streaming via voice-service |
| Interruption handling | P0 | M | Cancel TTS on user speech |
| Voice pipeline `/metrics` endpoint | P1 | M | Session, latency, interruption metrics |
| Deploy voice-pipeline as Railway service | P0 | L | Public domain (WebSocket needs browser access) |
| Turn-taking tuning (silence threshold, VAD sensitivity) | P2 | M | Iterative UX refinement |

**`automatos-ai` repo:**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Frontend: `VoiceCallPanel` | P0 | L | Full real-time voice UI |
| Frontend: `LiveTranscript` component | P0 | M | Real-time transcript display |
| Frontend: WebSocket audio streaming (`useVoiceStream`) | P0 | L | MediaStream → WS → pipeline |
| Frontend: `InterruptionHandler` | P1 | M | Client-side interrupt detection |
| Mobile-optimized voice UI | P1 | M | Touch UX, bandwidth optimization |
| Multi-language live conversation | P2 | M | Auto-detect language, switch TTS voice |

**`automatos-monitoring` repo (PRD-73):**

| Task | Priority | Effort | Notes |
|------|----------|--------|-------|
| Add voice-pipeline scrape target to Prometheus | P1 | S | `voice-pipeline.railway.internal:8301` |
| Add voice-pipeline alert rules | P1 | S | E2E latency, session failures |
| Update Grafana Voice dashboard with Phase 3 panels | P1 | M | Sessions, interruptions, streaming latency |

---

## 10. Success Criteria

### Phase 1

| Metric | Target |
|--------|--------|
| STT accuracy (WER) | <10% on English conversational speech |
| STT latency (p95) | <2s for 10s audio clip |
| TTS latency (p95) | <1.5s for 100-word response |
| Voice service uptime | >99% when platform is running |
| Audio upload success rate | >98% |
| Voice messages visible in chat history with transcript | 100% |
| Text chat unaffected when voice service is down | 100% |

### Phase 2

| Metric | Target |
|--------|--------|
| TTS naturalness (subjective) | Indistinguishable from human in casual listening |
| Voice cloning quality | Recognizable as the reference speaker |
| Provider swap time | <1 hour to switch TTS engine (config change + restart) |
| Per-agent voice differentiation | Configurable per agent, audibly distinct |

### Phase 3

| Metric | Target |
|--------|--------|
| End-to-end latency (user done speaking → first audio response) | <2s p95 |
| Interruption response time | <500ms (TTS stops within 500ms of user speaking) |
| Conversation naturalness | Turn-taking feels natural, no awkward pauses >2s |
| Concurrent voice sessions | >=10 simultaneous |
| Session drop rate | <5% |

---

## 11. File Structure

### 11.1 `automatos-voice` (New Repo — Voice Service)

Owns all ML inference, model management, and real-time voice pipeline. Deployed as independent Railway service(s).

```
automatos-voice/
├── README.md
├── docker-compose.yml                   # Local dev (voice-service + voice-pipeline)
├── .env.example
├── .github/
│   └── workflows/
│       └── deploy.yml                   # CI/CD → Railway
│
├── services/
│   ├── voice-service/                   # Phase 1: STT + TTS inference server
│   │   ├── Dockerfile                   # Based on Speaches or custom FastAPI
│   │   ├── config.py                    # Model selection, device config, env vars
│   │   ├── main.py                      # FastAPI app (if custom; Speaches has its own)
│   │   ├── health.py                    # /health endpoint
│   │   ├── metrics.py                   # /metrics endpoint (Prometheus)
│   │   └── requirements.txt
│   │
│   └── voice-pipeline/                  # Phase 3: Real-time conversation pipeline
│       ├── Dockerfile
│       ├── config.py
│       ├── pipeline.py                  # Pipecat pipeline definition
│       ├── processors/
│       │   └── orchestrator_bridge.py   # AutomatosOrchestratorProcessor
│       ├── health.py
│       ├── metrics.py
│       └── requirements.txt
│
├── models/                              # Model download cache (gitignored, volume-mounted)
│   └── .gitkeep
│
├── monitoring/
│   └── prometheus/
│       └── rules/
│           └── voice.yml                # Voice-specific alert rules (copied to automatos-monitoring)
│
├── railway/
│   ├── voice-service.toml               # Railway service config
│   └── voice-pipeline.toml              # Railway service config (Phase 3)
│
└── scripts/
    ├── setup-local.sh                   # Local dev setup (download models, create dirs)
    ├── setup-railway.sh                 # Railway service creation
    └── health-check.sh                  # Verify voice services are up
```

**Key boundaries:**
- This repo has NO knowledge of the orchestrator, agents, workspaces, or database
- It exposes only OpenAI-compatible HTTP endpoints (`/v1/audio/transcriptions`, `/v1/audio/speech`)
- It exposes `/health` and `/metrics` for PRD-73 monitoring integration
- Model weights are downloaded at first boot and cached in a Railway volume

### 11.2 `automatos-ai` (Existing Repo — Integration Wiring)

Owns the API endpoints, voice client, frontend components, and database schema. Talks to `automatos-voice` via HTTP over Railway private networking.

```
orchestrator/
├── api/
│   ├── chat_voice.py                   # POST /api/chat/voice, GET /audio/{id}
│   └── voice_profiles.py               # Voice profile CRUD (Phase 2)
├── modules/
│   └── voice/
│       ├── __init__.py
│       ├── client.py                    # VoiceServiceClient — HTTP client to voice-service.railway.internal
│       ├── audio.py                     # Audio validation, format conversion, S3 storage
│       └── providers/
│           ├── __init__.py
│           ├── base.py                  # TTSProvider ABC (provider abstraction lives here)
│           ├── self_hosted.py           # Phase 1/2: Calls automatos-voice service
│           └── openai.py               # Cloud fallback: Calls OpenAI API directly

frontend/
├── components/
│   └── voice/
│       ├── VoiceMicButton.tsx           # Mic toggle with recording states
│       ├── VoiceRecorder.tsx            # MediaRecorder wrapper
│       ├── VoicePlayer.tsx              # Audio playback with controls
│       ├── VoiceMessage.tsx             # Chat bubble variant for voice
│       ├── VoiceCallPanel.tsx           # Live voice conversation UI (Phase 3)
│       ├── LiveTranscript.tsx           # Real-time transcript (Phase 3)
│       ├── VoiceActivityIndicator.tsx   # Who is speaking indicator
│       └── VoiceSettings.tsx            # User voice preferences
├── hooks/
│   ├── useVoiceRecorder.ts             # Recording state machine
│   ├── useVoicePlayback.ts             # Audio playback controls
│   └── useVoiceStream.ts              # WebSocket audio streaming (Phase 3)
└── lib/
    └── voice-client.ts                 # API client for voice endpoints

alembic/versions/
├── xxx_add_voice_metadata_to_messages.py
├── xxx_create_voice_profiles_table.py        # Phase 2
└── xxx_add_voice_profile_to_agents.py        # Phase 2
```

**Key boundaries:**
- This repo has NO ML dependencies (no PyTorch, no model weights, no audio inference)
- `VoiceServiceClient` makes HTTP calls to `voice-service.railway.internal:8300` — same contract as OpenAI's API
- Swapping to OpenAI's hosted voice API = change `VOICE_SERVICE_URL` in config, switch provider to `openai.py`
- Provider abstraction lives here (not in `automatos-voice`) because the orchestrator decides which provider to use

### 11.3 Repo Boundary Contract

The ONLY interface between the two repos is the OpenAI-compatible REST API:

```
automatos-ai                                    automatos-voice
─────────────                                   ────────────────
VoiceServiceClient  ── HTTP POST ──────────►   /v1/audio/transcriptions
                    ◄── JSON response ─────    (STT: audio → text)

VoiceServiceClient  ── HTTP POST ──────────►   /v1/audio/speech
                    ◄── audio bytes ───────    (TTS: text → audio)

Prometheus          ── HTTP GET ───────────►   /metrics
                    ◄── Prometheus format ──    (observability)

Prometheus          ── HTTP GET ───────────►   /health
                    ◄── JSON health ────────   (liveness)

Frontend (Phase 3)  ── WebSocket ──────────►   /ws/voice
                    ◄── bidirectional audio ─   (real-time pipeline)
```

No shared database. No shared message queue. No shared code. If `automatos-voice` goes down, text chat is unaffected.

---

## 12. Relationship to Existing PRDs & Repos

### 12.1 Cross-Repo Dependencies

| Repo | Role in PRD-74 | Depends On |
|------|----------------|------------|
| **`automatos-voice`** (new) | Voice inference service (STT/TTS engines, Pipecat pipeline) | Nothing — standalone service |
| **`automatos-ai`** (existing) | Integration wiring (API endpoints, voice client, frontend, migrations) | `automatos-voice` running and reachable via Railway private network |
| **`automatos-monitoring`** (PRD-73) | Prometheus scrape targets, alert rules, Grafana dashboard for voice | `automatos-voice` exposing `/metrics` and `/health` |

### 12.2 PRD Relationships

| PRD | Relationship |
|-----|-------------|
| **PRD-01** (Core Orchestration) | Voice is a transport layer. Messages enter the same orchestration pipeline regardless of input modality. |
| **PRD-50** (Universal Orchestrator Router) | Voice transcripts are routed through the Universal Router exactly like typed messages. No routing changes needed. |
| **PRD-55** (Autonomous Assistant / Heartbeats) | Auto's heartbeat responses can be spoken aloud if the user is in voice mode. Voice profile determines Auto's voice. |
| **PRD-57** (Mobile-First Responsive) | Voice is mobile's killer feature. Voice UI components must be mobile-optimized from Phase 1. |
| **PRD-71** (Unified Skills Architecture) | Voice is not a skill — it's a transport. Skills don't need voice awareness. |
| **PRD-72** (Activity Command Centre) | Voice events (sessions started, errors) surface in the activity feed. |
| **PRD-73** (Observability & Monitoring) | Voice service scrape targets, alert rules, Grafana dashboard, and Loki log labels integrate into the monitoring stack. See Section 5. |

---

## 13. Open Questions

| # | Question | Impact | Decision Needed By |
|---|----------|--------|--------------------|
| 1 | GPU or CPU for voice service on Railway? CPU is cheaper but slower. GPU reduces latency 3-5x. | Phase 1 latency targets | Before deployment |
| 2 | Should voice audio be transcribed and stored, or ephemeral? Current design stores both. | Storage costs, privacy | Phase 1 |
| 3 | Pipecat vs LiveKit Agents — need a 1-week spike to validate. | Phase 3 architecture | Before Phase 3 starts |
| 4 | Should Auto's voice be consistent across all agents, or should each agent have a distinct voice by default? | UX identity | Phase 2 |
| 5 | Rate limits per workspace — what's reasonable? 60 STT/min and 60 TTS/min proposed. | Cost, abuse prevention | Phase 1 |
| 6 | Voice cloning consent — legal review needed for different jurisdictions? | Compliance | Phase 2 |

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **STT** | Speech-to-Text — converting audio to text transcript |
| **TTS** | Text-to-Speech — converting text to spoken audio |
| **VAD** | Voice Activity Detection — detecting when a user is speaking vs. silent |
| **WebRTC** | Web Real-Time Communication — browser standard for real-time audio/video |
| **Whisper** | OpenAI's open-source STT model family (tiny → large-v3) |
| **Kokoro** | 82M-parameter TTS model (Apache 2.0, StyleTTS2-based) |
| **Chatterbox** | Resemble AI's open-source TTS (MIT, beats ElevenLabs in blind tests) |
| **CosyVoice 2** | Alibaba's open-source TTS (Apache 2.0, 150ms streaming, 9 languages) |
| **Pipecat** | Daily.co's open-source Python framework for real-time voice AI pipelines |
| **Speaches** | Open-source unified STT+TTS server (OpenAI-compatible API, Docker-native) |
| **Opus** | Audio codec optimized for speech — low bitrate, high quality |
| **Zero-shot cloning** | Cloning a voice from a short audio sample without fine-tuning |

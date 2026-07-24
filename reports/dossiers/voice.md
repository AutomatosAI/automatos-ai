# Voice — Module Dossier (Phase 2 deep-review)

**Module key:** `voice` · **Tier:** standard · **Status:** partial
**Scope note:** the review covers the **in-repo half** (orchestrator + frontend). The STT/TTS pod and the Pipecat live-call pipeline live in the separate `automatos-voice` repo; they are read for context (that repo is checked out as a sibling at `../automatos-voice`) but are not the primary review unit. Sections F (enterprise/defensive/security) are deliberately omitted per the brief — that runs as its own Opus pass.

---

## A. What it is

Voice is a **two-path bolt-on to the existing chat/agent loop** that lets a user talk to Auto and hear Auto talk back, without giving voice its own brain. Path 1 (PRD-74 Phase 1/2) is **turn-based**: the browser records a clip, POSTs it to `/api/chat/voice`, the orchestrator transcribes it (faster-whisper), runs the transcript through the *exact same* streaming chat pipeline used by text chat, then synthesizes the reply (Kokoro TTS) and returns audio + transcript. Path 2 (PRD-74 Phase 3) is a **live phone-call-style WebSocket** (`VoiceCallPanel` → the `automatos-voice` pipeline pod) that streams mic audio through VAD→STT, POSTs each finished utterance to `/api/chat`, and streams synthesized audio back. On top of both sits **voice-profile CRUD** (per-workspace named voices, optionally cloned from an uploaded reference clip) and a **per-agent voice assignment** so each agent can sound different. The STT/TTS engines themselves are a swappable OpenAI-audio-compatible HTTP contract, so the pod is replaceable without touching orchestrator code. The design intent is explicit and correct: **voice is an I/O skin over the one proven agentic loop, not a parallel stack.**

---

## B. What it does — real implementation & data path

### Path 1: turn-based voice chat (the in-repo core)

`POST /api/chat/voice` (`orchestrator/api/chat_voice.py:153-347`), auth via `get_request_context_hybrid` (`:162`):

1. **Gate + health check.** `if not config.VOICE_ENABLED → 503` (`:186-187`); then a live `await _voice_client.health()` — if the pod is down, `503 "Voice service unavailable. Text chat remains functional."` (`:190-195`). This is an honest, non-fabricated failure (contrast the July F035/F038 fabrication findings).
2. **Validate audio** (`modules/voice/audio.py:37-63`): content-type/extension allowlist (`webm/ogg/wav/mp3/m4a/flac/opus`), size cap `VOICE_MAX_AUDIO_SIZE_MB` (25 MB default), empty-file reject.
3. **STT** (`_voice_client.transcribe`, `modules/voice/client.py:44-83`): multipart POST to `{pod}/v1/audio/transcriptions`. Empty transcript → `422 "Could not understand audio"` (`:219-223`).
4. **Route through the real chat pipeline** — `_collect_streaming_response` (`chat_voice.py:32-150`). This is the load-bearing part: it saves the user message to chat history (`:68-73`), runs the **same `AutoBrain.assess` complexity/routing** as `chat.py` (`:86-123` — RESPOND vs DELEGATE/MISSION via `UniversalRouter`), then iterates `StreamingChatService.stream_response_with_agent(...)` and **collects** the AI-SDK `0:"..."` text chunks into one string (`:132-150`). So voice inherits agent selection, tools/Composio, memory, context assembly — **everything text chat gets.**
5. **Resolve the responding agent's voice** (`:256-276`): look up `Agent.voice_profile_id` → `VoiceProfile` → use its `voice_id`/`provider`/`reference_audio`; else fall back to `config.VOICE_TTS_DEFAULT_VOICE` (`af_heart`).
6. **TTS** (`_voice_client.synthesize`, `client.py:85-142`): JSON POST to `{pod}/v1/audio/speech`. The reply text is **truncated to ~500 chars at a sentence boundary** before synthesis (`chat_voice.py:284-293`).
7. **Persist + return** (`:304-347`): upload MP3 to S3 (`upload_voice_audio`, `audio.py:66-95`), return `{transcript, response_text, audio_url, audio_base64, stt_latency_ms, tts_latency_ms, voice_metadata}`. `GET /api/chat/voice/audio/{message_id}` (`:350-365`) redirects to a 1-hour presigned S3 URL.

Frontend: `frontend/lib/voice-client.ts` (`sendVoiceMessage`, `checkVoiceHealth`) → `use-voice-recorder.ts` (MediaRecorder) → `MultimodalInput` mic button, which is **health-gated on mount** (`multimodal-input.tsx:126-131`: only shows the mic if `voice_enabled && voice_service_healthy`). The reply renders as a `VoiceMessage` (`components/voice/VoiceMessage.tsx`) with a `VoicePlayer` and STT/TTS latency badges; message renderer wires it at `components/chatbot/message.tsx:217-222`.

### Path 2: live-call (cross-repo, PRD-74 Phase 3)

`VoiceCallPanel` → `use-voice-stream.ts` opens a raw-PCM WebSocket (`ws/voice?workspace_id&agent_id&token&conversation_id`) to `NEXT_PUBLIC_VOICE_PIPELINE_URL`. The pod (`automatos-voice/services/voice-pipeline/main.py`) builds a **Pipecat pipeline**: `transport.input() → SileroVAD → VoiceServiceSTT → OrchestratorProcessor → VoiceServiceTTS → transport.output()` (`main.py:270-276`). The `OrchestratorProcessor` (`orchestrator_processor.py`) is the bridge: on each finished `TranscriptionFrame` it POSTs to `{orchestrator}/api/chat` with `X-Workspace-ID` + `Authorization: Bearer {token}`, collects the AI-SDK stream, and emits the response **as a single `TextFrame`** for TTS (`:83-104`). So live-call also reuses the full agent loop — but note the single-frame emit (see C, latency).

### Storage & config

- `voice_profiles` table (`core/models/voice_profiles.py`, migration `alembic/versions/prd74_voice_profiles.py`): `workspace_id` FK, `name`, `provider` (default `kokoro`), `voice_id`, `reference_audio` (S3 key), `settings` JSONB, `is_default`. `Agent.voice_profile_id` FK added by the same migration (`core/models/core.py:243`). Both routers registered via `router_manifest.py:68-69`.
- Config (`config.py:997-1007`): `VOICE_SERVICE_URL` (localhost default — **F068 fix confirmed**, see Phase-0), `VOICE_STT_MODEL` (`Systran/faster-whisper-large-v3`), `VOICE_TTS_MODEL` (`kokoro`), `VOICE_TTS_DEFAULT_VOICE` (`af_heart`), `VOICE_ENABLED` (**default `true`**), `AUTO_VOICE_PROVIDER` (`chatterbox`).

---

## C. Honest quality — how good is it *really*?

### Real-data check

There is **no live voice usage to inspect.** The W1 production census (`real-data-inventory.md`, `data/census.md`) enumerates 152 tables with row counts; **`voice_profiles` does not appear**, and no voice audio, transcript, or latency telemetry is captured in any inspected table. The voice path emits only structured log lines (`voice_chat_complete` with latencies, `chat_voice.py:317-329`) — nothing persisted for analysis. Combined with the pod's Railway config (`docker-compose.voice.yml:8-9` documents both pod services as `sleep=true`, i.e. serverless-idle), and commit `8df14e35d "fix: disable serverless sleep on voice-service and voice-pipeline"`, the honest read is: **voice was built and debugged to working order in Q1 2026, but there is no evidence of sustained real use.** It is a *live-capable but cold* feature. This matters for the North Star: a capability nobody exercises delivers zero autonomy/quality uplift today, regardless of code quality.

### What is genuinely good

- **The reuse architecture is right.** Voice does not fork the agent loop; it feeds the same `StreamingChatService` + `AutoBrain` routing (`chat_voice.py:86-140`). Whatever quality memory/context/tools give text chat, voice gets for free. This is exactly the "I/O skin, not a parallel stack" the platform should want, and it is faithfully implemented on *both* paths.
- **Honest degradation.** Pod-down is a real 503 with a truthful message, and the frontend hides the mic when the pod is unhealthy (`multimodal-input.tsx:126-131`). No fake "listening" state. TTS failure is non-fatal — the text reply still returns (`chat_voice.py:311-313`). This is the correct failure posture.
- **Clean provider seam.** `TTSProvider` ABC (`providers/base.py`) + OpenAI-audio-compatible client means the pod (or a hosted vendor) is swappable behind one HTTP contract. The pod repo is genuinely decoupled ("No knowledge of orchestrator, agents, or workspaces — pure voice I/O", `automatos-voice/README.md`).
- **Tenant scoping on profiles is correct.** Every `voice_profiles` query filters `workspace_id == ctx.workspace_id` (`api/voice_profiles.py:186-188, 255-258, 283-286, 333-336`); delete clears dangling `Agent.voice_profile_id` refs in the same workspace (`:341-346`). No cross-tenant leak on the profile surface.

### Concrete defects (evidence-cited)

1. **Turn-based path is non-streaming end-to-end — the worst latency posture available.** `_collect_streaming_response` *consumes the entire* agent stream into a string (`chat_voice.py:126-150`) **before** a single TTS byte is requested. So perceived latency = STT + **full** LLM generation (all tokens, incl. any tool round-trips) + full TTS + S3 upload, serially. Cascaded pipelines already add latency vs speech-to-speech ([softcery](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture)); the standard mitigation is **sentence-streaming** — synthesize each sentence as the LLM emits it, keeping end-to-end under ~500ms ([spheron/Kokoro](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/)). This path does the opposite. For a tool-using turn it will feel like many seconds of dead air.
2. **The live-call path throws away most of its own streaming advantage.** `OrchestratorProcessor` collects the *whole* `/api/chat` response and emits **one** `TextFrame` (`orchestrator_processor.py:86-89`). Pipecat can stream sentence-by-sentence to TTS, but the bridge blocks on the full response first — so live-call latency-to-first-audio ≈ turn-based latency, minus only the upload. The Pipecat/VAD/barge-in machinery is present but starved by a blocking bridge.
3. **The 500-char reply truncation silently mutilates answers** (`chat_voice.py:284-293`). Any substantive Auto answer (a summary, a list, a plan) is **cut at ~500 chars** for the spoken output. The returned `response_text` is full, but the *audio the user hears* is a fragment, with no "…and more, see text" cue. For a voice-first user this is a real quality regression, not a latency tradeoff.
4. **`response_format="both"` default returns audio AND base64 AND an S3 URL every turn** (`chat_voice.py:158`, `:331-347`). The reply embeds `audio_base64` (full MP3 inlined into JSON) *and* uploads to S3 *and* returns a presigned URL — redundant work and payload bloat on every turn. Commit history shows this was reached by thrashing (`3e8dc878a "return audio as inline base64"`, `33d9865e0 "voice audio URL missing API base"`, `22e65a0a8 "rewrite voice playback…"`) rather than a settled design.
5. **Provider/engine mismatch is latent breakage for cloning.** Orchestrator default `AUTO_VOICE_PROVIDER=chatterbox` (`config.py:1007`) and cloned profiles default `provider="chatterbox"` (`api/voice_profiles.py:418`), but the pod default is `TTS_ENGINE=kokoro` (`automatos-voice/.../voice-service/config.py:34`, options `kokoro|chatterbox|both`). If the deployed pod runs `kokoro` (the default) and a workspace creates a **cloned** voice or previews one, the `chatterbox`-model request hits an engine that isn't loaded. Cloning is thus **conditionally dead** depending on an env var set in a *different repo* — exactly the kind of cross-repo coupling that rots silently.
6. **Duration validation for cloning is a guess, not a measurement** (`api/voice_profiles.py:133-169`). WAV is parsed from the header; **mp3/webm duration is estimated by assuming 128 kbps** (`:159`). A 64 kbps or VBR clip will be mis-sized by 2×, so the 5–60s guardrail is unreliable for the exact formats browsers actually record (webm/opus). Minor, but it means "upload a 30s clip" can reject or accept the wrong thing.
7. **No memory/telemetry of voice at all.** Text chat writes messages, tool logs, usage. Voice writes chat history (via the reused pipeline) but **nothing voice-specific** — no latency table, no STT accuracy signal, no per-voice usage. There is no way to answer "is voice any good?" from data, which is why C opens where it does.

### Maturity score: **2 / 5**

Justification: the *architecture* is a 4 (correct reuse, clean seam, honest failure). The *delivered capability* drags it to a **2**: it is functionally complete and demoable, but (a) has **no evidence of real use** and **zero quality telemetry**, (b) ships the **worst-case non-streaming latency** on both paths, (c) **truncates spoken answers** at 500 chars, and (d) carries a **cross-repo engine mismatch** that conditionally kills cloning. It is a working prototype that has not been hardened into a capability a client would live in. Against best-in-class turn-taking/barge-in voice agents it is early.

---

## D. Competitive teardown

Voice-agent platforms in 2026 compete on three axes Automatos barely touches: **latency-to-first-audio, natural turn-taking/barge-in, and telephony/deployment reach.** All four named competitors are speech-to-speech-or-streaming-first; Automatos is blocking-cascade.

| Capability | Automatos (in-repo) | Best-in-class |
|---|---|---|
| Latency to first audio | STT + **full** LLM + full TTS, serial (no sentence-streaming) | **ElevenLabs** first-turn <500ms, synth <100ms ([elevenlabs](https://elevenlabs.io/blog/how-do-you-optimize-latency-for-conversational-ai)); **Retell** ~600ms with its own turn model ([retell](https://www.retellai.com/blog/what-is-an-ai-voice-agent)); **Vapi** sub-600ms ([lindy](https://www.lindy.ai/blog/vapi-ai)) |
| Turn-taking / barge-in | None. Turn-based = push-to-talk; live-call has Silero VAD but a blocking bridge, no barge-in cancel of in-flight TTS | **ElevenLabs** turn-eagerness (Eager/Normal/Patient), reads "um/ah", 1–30s timeout ([elevenlabs docs](https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow)); **Vapi** endpointing + interrupt detection + backchanneling ([futureagi](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/)) |
| Architecture | Cascaded STT→LLM→TTS, collected | **OpenAI Realtime** single speech-to-speech model — lower latency, preserves prosody, native tool-calling ([openai](https://openai.com/index/introducing-gpt-realtime/)) |
| Telephony / channels | Web mic only. No PSTN, no SIP, no outbound | **Retell/Vapi/ElevenLabs** all ship phone (inbound+outbound), SMS, CRM/transfer, campaign dialing ([retell pricing](https://www.retellai.com/pricing)) |
| Voice quality / cloning | Kokoro 52 preset voices + Chatterbox clone (conditionally wired) | **ElevenLabs** is the market's voice-quality/cloning leader as its core business ([cloudtalk](https://www.cloudtalk.io/blog/elevenlabs-voice-agent-review/)) |
| Agent brain | **Full Automatos agent** — memory, tools, missions, per-tenant context | Competitors bolt an LLM + a knowledge base + function calls per agent ([retell](https://www.retellai.com/blog/what-is-an-ai-voice-agent)); **none has Automatos's depth of agent state** |

**Where Automatos actually stands, honestly:** *behind* on every voice-native axis (latency, turn-taking, telephony, voice quality) — these are the hard parts and the competitors' entire product. **Ahead** on exactly one thing that is real and hard to replicate: **the agent behind the voice is a full Automatos agent** with memory, tool-use, missions, and per-tenant context, not a thin LLM+KB. Vapi/Retell/ElevenLabs give you a great voice wrapper around a *shallow* brain; Automatos has a deep brain with a *shallow* voice wrapper. That is the correct framing for the verdict.

---

## E. Build / extend / adopt / replace — the verdict

**ADOPT the voice transport; KEEP the agent bridge.** Do **not** keep hand-building STT/TTS/turn-taking/telephony — that is a specialist product line the platform will never win and does not need to.

Concretely:
- **Replace the `automatos-voice` pod + Pipecat live path with a hosted voice-transport vendor**, most likely **Vapi or Retell** (both are explicitly LLM-agnostic and call *your* endpoint via function-calling / custom-LLM webhook), or **ElevenLabs Agents** if voice quality is the priority. Rough cost: **~$0.07–0.15/min all-in** ([retell](https://www.retellai.com/pricing), [elevenlabs](https://www.aipricing.guru/blog/elevenlabs-pricing-review-2026/)) — pilot-affordable, and it deletes the self-hosted GPU pod, the Pipecat pipeline, the VAD tuning, and the whole "pod is asleep" ops problem.
- **What you keep and defend:** the **orchestrator bridge** — `/api/chat` fronted so a vendor's custom-LLM webhook posts a transcript and streams back Auto's response. That is the one piece worth owning because it is where the *deep agent* meets voice. The `TTSProvider` seam and `VoiceServiceClient` already prove the contract is swappable.
- **Kill or de-scope in-repo:** the 500-char truncation, the redundant base64+S3+URL triple-return, and — if you adopt a vendor that manages voices — most of `voice_profiles` cloning (vendors manage voice libraries). Keep per-agent voice *assignment* (map an agent → a vendor voice id), drop the S3 clone-upload plumbing.

**Why not "keep building":** nothing in the in-repo voice code is a differentiator. The differentiator is the agent, which is *already* reused. Every hour spent on sentence-streaming, barge-in, and PSTN is an hour re-deriving what Vapi/Retell sell for cents/minute. The reuse-over-build rule (§2) points hard at adopt here. **Extend** (patch streaming into the current pod) is the wrong call: it invests in a stack you should be deleting.

*Caveat for Gerard's decision:* if a hard requirement is **fully self-hosted / no third-party voice vendor** (data-residency, cost-at-scale), then the verdict flips to **extend** — keep the pod but (1) add sentence-streaming to `OrchestratorProcessor`, (2) drop the 500-char cap, (3) resolve the kokoro/chatterbox engine mismatch. That is a real fork in the road and is yours to call, not mine to assume.

---

## G. Quality metric — how do we measure & track this

Today: **unmeasurable.** Voice writes zero telemetry to any store (confirmed: no voice table in the 152-table census; only ephemeral log lines). The first honest metric is therefore "do we capture anything at all."

Proposed tracked metrics (feed T3's eval harness):
1. **End-to-end latency-to-first-audio** (p50/p95), decomposed STT / LLM / TTS. The log fields already exist (`stt_latency_ms`, `tts_latency_ms`, `total_ms`, `chat_voice.py:322-328`) — persist them to a `voice_turns` row instead of logs. **Target: p95 < 1.5s turn-based, < 800ms live** (competitors run 500–600ms).
2. **STT accuracy proxy** — % of turns that produce a non-empty transcript and are *not* immediately re-recorded by the user (a re-record within N seconds = a suspected mis-transcription). Cheap, no gold set needed.
3. **Answer-truncation rate** — % of replies whose spoken audio was cut by the 500-char cap. This should exist purely to *justify killing the cap*; expect it to be high.
4. **Voice-session task-lift** — of turns that triggered a tool/mission, did they complete at the same rate as the text-chat equivalent? Reuses the mission/tool telemetry already flowing.

**Number today: 0** (no data). First step is instrumentation, not a benchmark.

---

## H. Cost note (informational)

Per turn, self-hosted (current): STT (faster-whisper large-v3, ~12× real-time on a mid GPU, [promptquorum](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026)) + **one full LLM chat turn** (the dominant cost — same as text chat, model-dependent) + Kokoro TTS (~97ms TTFB, negligible compute, [spheron](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/)) + one S3 PUT + one presigned GET. The GPU pod is a **fixed idle cost** (why it's set to sleep). The LLM turn dwarfs the audio cost, so voice ≈ text-chat cost + a fixed GPU-pod overhead. **Adopting a hosted vendor** converts the fixed GPU cost into ~$0.07–0.15/min variable ([retell](https://www.retellai.com/pricing)) — cheaper until voice volume is high and sustained, which the data says it is not.

---

## I. UX / surface

Current surface is reasonable and already health-aware:
- **Chat input mic** (`MultimodalInput`) — push-to-talk, shown only when the pod is healthy. Correct.
- **Live-call panel** (`VoiceCallPanel`) — a genuinely nice full-call UI with live transcript, VAD-driven state pills, and a duration timer. Better than the backend that feeds it.
- **Settings → Voice Profiles** (`VoiceProfilesSettingsTab`) + **per-agent voice selector** in the agent config modal.

Concrete IA/UX changes (North-Star-ordered):
1. **Fix the spoken-vs-shown mismatch first.** Until the 500-char cap dies, the audio the user hears ≠ the text on screen. Either kill the cap (preferred) or show a "spoken summary" affordance so the divergence is explicit. This is a *trust* bug, not a polish item.
2. **Surface latency honestly, then hide it.** The latency badges (`VoiceMessage`) are a developer affordance; for clients, sub-second is invisible and multi-second needs a "thinking…" state that matches the actual blocking wait — right now a slow tool turn is silent.
3. **Command Center:** voice has **no presence** in Command Center. If voice becomes a real channel, live-call sessions should appear as first-class activity (like a Mission or a channel thread) — who called, duration, what tools ran — reusing the notifications/board-events spine. Today a voice call is invisible to the operator.
4. **Barge-in in the live panel** is the single biggest felt-quality lever and is impossible with the current single-`TextFrame` bridge — it must be an *architecture* change (E), not a UI toggle.

---

## J. Upgrade path (impact × effort, North-Star-ranked)

Two branches depending on E's fork. **If adopting a vendor (recommended):**

1. **[High impact / Medium effort] Front `/api/chat` as a vendor custom-LLM webhook and wire Vapi/Retell.** Deletes the pod, Pipecat, VAD tuning, and the sleep-ops problem; instantly gains sub-600ms latency, real turn-taking/barge-in, and (if wanted) telephony — while the deep agent stays ours. This is the whole game.
2. **[High / Low] Kill the 500-char truncation and the base64+S3+URL triple-return.** Independent of the vendor decision; pure correctness/cost wins (`chat_voice.py:284-293`, `:331-347`).
3. **[Medium / Low] Persist voice telemetry** (a `voice_turns` row from the fields already logged) so G's metrics become real and "is voice good?" is answerable.
4. **[Medium / Medium] Map per-agent voice → vendor voice id**, retire the S3 clone-upload plumbing, keep the assignment UX.

**If staying self-hosted (Gerard's call):**

1. **[High / Medium] Sentence-streaming in `OrchestratorProcessor`** (emit a `TextFrame` per sentence boundary as the `/api/chat` stream arrives, `orchestrator_processor.py:86-89`) — the one change that makes live-call feel real and unlocks barge-in.
2. **[High / Low] Kill the 500-char cap** (same as above).
3. **[Medium / Low] Resolve the kokoro/chatterbox engine mismatch** — make the orchestrator provider default and the pod `TTS_ENGINE` agree, or set the pod to `both`; otherwise cloning is conditionally dead (`config.py:1007` vs `automatos-voice/.../voice-service/config.py:34`).
4. **[Low / Low] Replace the 128 kbps duration guess** with a real decode (`api/voice_profiles.py:159`).

Either branch, the north star is the same: **voice should be a thin, low-latency, honest mouth on the deep Automatos agent.** The agent half is already right; the mouth half is a cold prototype — buy it, don't keep building it, unless self-hosting is a hard constraint.

---

*Sources (competitive):* Vapi — [lindy.ai](https://www.lindy.ai/blog/vapi-ai), [futureagi](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/); Retell — [retellai.com/pricing](https://www.retellai.com/pricing), [retellai.com/blog](https://www.retellai.com/blog/what-is-an-ai-voice-agent); OpenAI Realtime — [openai.com/gpt-realtime](https://openai.com/index/introducing-gpt-realtime/); ElevenLabs — [elevenlabs latency](https://elevenlabs.io/blog/how-do-you-optimize-latency-for-conversational-ai), [elevenlabs conversation-flow](https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow), [cloudtalk](https://www.cloudtalk.io/blog/elevenlabs-voice-agent-review/); architecture/latency — [softcery S2S-vs-cascade](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture), [spheron/Kokoro](https://www.spheron.network/blog/deploy-open-source-tts-gpu-cloud-2026/), [promptquorum/faster-whisper](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026).

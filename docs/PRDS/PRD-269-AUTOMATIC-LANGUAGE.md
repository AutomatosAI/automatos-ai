# PRD-269: Automatic language: speak to Auto in about 100 languages

**Status:** Draft, waiting for a GPU. Nothing here is built, and nothing is approved; every choice is in §11 with a recommendation. · **Owner:** Gerard · **Written:** 27 Oct 2026, grounded on `origin/main @ 79f0b7701` and on `AutomatosAI/automatos-go` `main`.
**Why it waits:** Gerard, 27 Oct: "it needs a GPU, so let's WAIT". Until a GPU server exists, the docs, the repos and the sites list automatic language detection as **planned**, never as available.
**What ships first instead:** English and Portuguese (Brazil) on the phone, chosen in Settings → Voice with a one-tap EN/PT switch (automatos-go, "voice languages"). Gerard: "Let's do English and Portuguese, since I am in Portugal and the WebSummit is here." He has since dropped European Portuguese from the phone.
**Type:** extension. It reuses the quick-ask flow, the `spokenMode` chat turn, the discovery document and voice-tts's service pattern; the new piece is a speech-recognition service.
**Depends on:** a GPU server (§7, §11-Q2); automatos-go's voice-languages work (the Settings → Voice language choice this PRD adds to).

Labels: **[T]** read in code; **[G]** Gerard's words; **[I]** inferred, or from the model's published notes, to be checked by the bench (US-001); **[P]** proposed, for Gerard to confirm.

## 1. Introduction

Auto already answers in the language you write or speak to it in, for any language the model handles. Voice is narrower, because something has to turn speech into words first, and each surface does that differently:

| Surface | Hears you with | Speaks with | Languages today |
|---|---|---|---|
| Typed chat | n/a | n/a | Any language the model handles |
| Web quick ask | The browser's `SpeechRecognition`, in `navigator.language` (`en-US` when none) [T `frontend/lib/voice/quick-ask/deps.ts`] | The browser's `speechSynthesis` in the same language [T] | The browser's language |
| Automatos Go | Apple's recognizer, in the locale chosen up front [T `src/voice/quick-ask`]: on the iPhone for the iPhone's own language; for another language, on the iPhone when its dictation model is installed, otherwise Apple's speech recognition | Auto's natural voice (`POST /api/voice/speech`), else the iPhone's voice | English, Portuguese (Brazil) |
| Auto Live | Retell, pinned to `VOICE_LIVE_LANGUAGE` (`en-GB`) ("pinning `language` stops multilingual STT hallucinating confident nonsense") [T `modules/voice/retell_api.py`] | Retell's voice | One per server |
| voice-tts (Kokoro) | n/a | Kokoro, on CPU [T `services/voice-tts`] | 9: English (US, UK), Spanish, French, Italian, Portuguese (Brazil), Hindi, Japanese, Mandarin. Japanese and Mandarin are phonemized with espeak and sound weaker. No European Portuguese |

**The problem.**
- **Apple's recognizer needs the language up front.** The app names a locale before you speak and downloads that locale's on-device model [T `onDeviceOptions`, `downloadModel`]. Speak another language and you get nonsense or nothing. A choice of two languages fits Gerard and Lisbon; it doesn't fit a visitor who speaks German, or someone who switches language mid-week.
- **Browsers' speech recognition is uneven.** Some browsers have none, and quick ask can only say "This browser can't recognise speech" [T `quick-ask/copy.ts`]. Where it exists, the browser's maker may run it on its own servers, it listens only in the browser's language, and its quality differs by browser and by language [I].

## 2. Goals

- Speak to Auto in any of about 100 languages **without choosing one first**: the server returns the words and the language it heard.
- Auto answers in that language, and speaks the answer in it: Kokoro's voice where Kokoro has one, otherwise the phone's or the browser's voice for that language.
- One route for the phone and the web quick ask.
- Apple's recognizer stays the default for a chosen language, because it shows your words live and, for the iPhone's own language, runs on the iPhone (§4.7). "Automatic (any language)" is an extra choice, offered only where the server can do it.
- Off is a normal state. A server without the recognition service behaves exactly as today, just as the server voice's `tts_off` does [T `modules/voice/speech.py`].
- No audio is kept.

## 3. Decisions this PRD rests on

- [G] "The option to work with 100 [languages] goes into the docs and all repos, websites": as **planned**.
- [G] "It needs a GPU, so let's WAIT": no build until a GPU server exists.
- [G] English and Portuguese (Brazil) ship first, on the phone, with the language chosen in Settings → Voice. European Portuguese is dropped from the phone.
- [G] The paid-model budget is Anthropic only. Any new vendor (option C below) needs Gerard's OK.

## 4. Design

### 4.1 Where recognition runs: three options

All three use Whisper-family models or a vendor's equivalent. Whisper identifies the spoken language as part of transcription [I].

| | A. Self-hosted, GPU | B. Self-hosted, CPU | C. Hosted speech-to-text API |
|---|---|---|---|
| **What** | faster-whisper (CTranslate2) with `large-v3-turbo` on a GEX45-class server (one 24 GB GPU) | faster-whisper `small` or `medium`, int8, on CPU: an AX102-class server (no GPU) or the Railway plan | A vendor's speech-to-text, paid per minute |
| **Accuracy** | The best of the self-hosted options; turbo keeps large-v3's encoder and cuts the decoder to 4 layers [I] | Lower, and lower still outside English; language detection weaker on short clips [I] | Vendor-dependent; measured on the same gold set |
| **Speed** | Well under a second for a short question [I], measured in US-001 | Seconds per question [I], measured in US-001 | Network plus the vendor's queue |
| **Cost** | Fixed monthly (§7); nothing per minute | No new server if there's headroom; CPU taken from voice-tts and Auto | Per minute, grows with use |
| **Audio goes to** | Our server (SaaS) or the operator's own (self-hosted) | The same | A third party: a new sub-processor, its retention, a data-processing agreement |
| **Rules** | MIT model and runtime: no GPL boundary, unlike voice-tts's espeak [I] | The same | A new third-party API client and key, which AGENTS.md forbids without a maintainer's approval [T] |
| **Self-hosters** | Need an NVIDIA GPU | Run anywhere, slowly | Need their own vendor key |

*Recommendation:* **A**, with B measured in the same bench as the fallback for self-hosters without a GPU. C only if Gerard wants the feature before a GPU exists, and only with his OK on the vendor and the spend.

Whisper's limits shape the design [I]:
- It returns a language, not a region: `pt`, not `pt-PT` or `pt-BR`; `en`, not `en-GB`. The region comes from the member's setting (§4.5).
- A one- or two-word clip can be misheard as another language. An optional "languages I speak" hint can narrow the choice (§11-Q6).
- It can invent words over silence. Voice-activity detection (Silero, built into faster-whisper) trims silence first.
- Quality varies widely across the ~100 languages. That is why the docs say "about 100", and why each language's number is published (§6).

### 4.2 The service: `services/voice-stt`

It mirrors voice-tts [T `services/voice-tts/README.md`]:
- **Routes:** `GET /healthz`; `GET /readyz` (200 only once the model has loaded, so a deploy takes traffic only when it can listen); `POST /transcribe` with a bearer token compared in constant time.
- **Request:** one audio clip of up to `VOICE_STT_MAX_SECONDS` (60, quick ask's limit [T]), plus an optional list of allowed languages.
- **Answer:** `{text, language, language_probability, duration_seconds}`.
- **Load:** a fixed number of slots, one per concurrent transcription the GPU holds. The next request gets 429 `stt_busy` at once, never a queue.
- **Logging:** the audio is held in memory for the request, never written and never logged. The text is never logged either. One line per request gives the language, the seconds and the latency.
- **Compose:** an optional `stt` profile that reserves an NVIDIA GPU. CI builds the image and runs its suite on CPU with a tiny model, so CI needs no GPU [P].

### 4.3 Core: the route and discovery

- **Route:** `POST /api/voice/transcribe` (`api/voice_transcribe.py`, `modules/voice/transcribe.py`), for any workspace member, with a per-workspace rate limit `VOICE_STT_RATE_PER_MINUTE`, as `VOICE_TTS_RATE_PER_MINUTE` does [T].
- **Refusals,** each `{code, detail}`: 503 `stt_off` (no `VOICE_STT_URL`), 503 `stt_unavailable`, 413 `stt_too_long`, 422 `stt_bad_audio`, 429 `stt_busy`. On any refusal, the client says why and offers the chosen language.
- **Settings:** `VOICE_STT_URL`, `VOICE_STT_TOKEN` and the timeouts, in `config.py` and `reports/config-surface.json`. The route goes in `reports/route-manifest.json` [T AGENTS.md].
- **Discovery:** `stt` joins `CAPABILITY_ROUTES` and `CONFIGURED_CAPABILITIES`, listed only while `VOICE_STT_URL` is set. That is exactly how `tts` works [T `services/mobile/discovery.py`].
- **The chat turn** stays today's `POST /api/chat` with `spokenMode` [T `api/chat_models.py`]. An optional `spokenLanguage` (the detected tag) lets the spoken style ask for a reply in that language. A short or mixed clip otherwise risks an English reply [I].

### 4.4 The phone (automatos-go)

- **No native build.** expo-audio is already in the app (`expo-audio ~1.1.1`, in build 13) [T], so recording and upload are JavaScript, shipped over the air.
- **Settings → Voice** gains **Automatic (any language)** beside EN/PT. It shows only while discovery lists `stt`. Apple's recognizer stays the default.
- **Hold and release.** Hold records (AAC). Release uploads the clip, and the transcript appears when it returns.
- **What's lost:** your words no longer show live as you talk. A level meter and "Listening…" stand in, and the transcript appears when it comes back. This is the trade against Apple's recognizer, and the reason Apple stays the default.
- **Stop phrases** ("Auto, stop") are English today [T `quick-ask/config.ts`]. In Automatic they are checked in the detected language, from a short list per language, and the Stop button always works.

### 4.5 Speaking the answer: a voice per detected language

| Heard | Phone | Web quick ask |
|---|---|---|
| English | Kokoro, US or UK by the member's setting | The browser's voice until the web's server voice is wired, then Kokoro |
| Portuguese | Kokoro, Brazilian Portuguese | The browser's voice for `pt-PT` or `pt-BR` |
| Spanish, French, Italian, Hindi, Japanese, Mandarin | Kokoro (Japanese and Mandarin weaker) | The browser's voice, then Kokoro |
| Any other | The iPhone's voice for that language, where one is installed | The browser's voice for that language |

The once-per-session AI notice is spoken in the detected language. A translated greeting must contain that language's AI term, a check PRD-264 already makes (`ai_disclosure.badge_for`: AI, IA, KI) [T]. Which languages get a translated notice is §11-Q9.

### 4.6 The web quick ask

- **Recording:** the browser's `MediaRecorder`; the format differs by browser [I]. The clip goes through `apiClient` to the same route [T AGENTS.md: no raw `fetch`].
- **When it's offered:** "Automatic" shows where the server offers `stt`. It also gives browsers with no speech recognition a way to talk to Auto at all.
- **The voice:** the browser's `speechSynthesis` with the detected language. Auto's natural server voice reaches the web when that separate work lands; it isn't wired up today.

### 4.7 Privacy

- **Today:** on the phone, your voice is recognised on the iPhone for the iPhone's own language; for another language, on the iPhone when its dictation model is installed, otherwise by Apple's speech recognition. On the web, the browser may send audio to its maker; Automatos never receives audio [T `docs/voice/_index.md`].
- **With Automatic:** the audio leaves the device for the Automatos server. On SaaS that is our GPU server (in the EU [P]); on a self-hosted install it is the operator's own server.
- **Retention: none.** The audio lives in memory for one request and is never written or logged. The transcript is saved as the chat message, like typed words.
- **Copy:** the setting says so in one line. The app's privacy copy and its App Store privacy answers name audio sent to the server (automatos-go).
- **Option C** adds a sub-processor, the vendor's retention terms and a data-processing agreement.

## 5. User stories

| Story | What | Acceptance |
|---|---|---|
| US-001 Bench | Options A and B (and C only if §11-Q3 says yes) on the local gold set | A report with WER per language, language-ID accuracy and latency; no invented numbers |
| US-002 voice-stt | The service in §4.2 and its CI lane | Image builds; suite passes on CPU; refusals as specified; no audio or text in the logs |
| US-003 Core route | §4.3: route, settings, rate limit, discovery `stt` | Off by default; `stt_off` when unset; route manifest and config surface updated; tests for every refusal |
| US-004 Reply language | `spokenLanguage` on the chat turn; the spoken style uses it | The reply-language eval passes (§6) |
| US-005 Phone | §4.4 and §4.5 in automatos-go | Shown only with `stt`; OTA, no native build; Apple stays the default |
| US-006 Web | §4.6 | Works in a browser without `SpeechRecognition`; uses `apiClient` |
| US-007 Wiring and docs | The GPU server reached from the hosted API; the docs' "planned" changes to the measured list | Docs list a language as supported only with its number |

## 6. Evals

- **Gold set:** recorded questions per language, kept **local**, never committed (a public repo).
- **WER per language,** English and Portuguese (Brazil) first, then Portuguese (Portugal), Spanish, French, Italian and German [P]. Each runs against Apple's recognizer on the same English and Portuguese (Brazil) clips.
- **Language ID** on short clips: one word, and 2 to 5 seconds.
- **Latency,** reusing PRD-258's report (`orchestrator/scripts/eval/voice_latency`: nearest-rank p50/p90, never an invented number) [T]. Two measures: release to transcript, and release to Auto's first spoken word.
- **Reply language:** the reply comes back in the detected language.
- **Proposed bars** [P]:
  - Automatic's WER within 2 points of Apple's on English and Portuguese (Brazil);
  - language ID at least 95% on 2–5 s clips;
  - release to transcript p50 ≤ 0.8 s and p95 ≤ 1.5 s for clips up to 15 s, on the GPU;
  - the reply in the detected language in at least 98% of turns.

## 7. Cost

- **A:** a GEX45-class server, €214 a month plus €209 setup (list prices on 24 Oct 2026, before VAT). The cost is fixed, with nothing per minute, and capacity is the GPU's slots.
- **B:** no new server where there is CPU headroom. Its price is CPU taken from voice-tts and Auto, and slower answers. An AX102-class server (€259 a month, no GPU) fits B, not A.
- **C:** the vendor's per-minute price times the minutes, growing with use. It is outside the Anthropic-only budget and needs Gerard's OK.
- **Metering:** whether recognition counts against a workspace's PRD-265 credits is §11-Q7.

## 8. Rollout, once the GPU server exists

1. Gerard approves the spend; the GPU server is provisioned.
2. US-001: bench on the gold set, report per language.
3. US-002 and US-003 merge. Nothing changes anywhere until `VOICE_STT_URL` is set.
4. Set `VOICE_STT_URL` on the hosted API. Discovery lists `stt`.
5. US-005 ships to the phone over the air; US-006 to the web.
6. US-007: the docs, the repos and the sites change "planned" to the measured list of languages.

Self-hosters get the compose `stt` profile on an NVIDIA GPU, or option B if §11-Q10 says yes.

## 9. Risks

- **A single GPU is a single point of failure.** A refusal falls back to the chosen language, so voice still works.
- **Misdetected short clips** answer in the wrong language. The "languages I speak" hint (§11-Q6) and the reply-language eval guard this.
- **Losing live words** may feel slower than Apple's recognizer. That is why Apple stays the default.

## 10. Success metrics

- The §6 bars, met for English and Portuguese (Brazil) before the phone shows Automatic.
- Zero audio files or transcripts in the service's logs or disks (a test, and a check on the running server).
- No change for a server without `VOICE_STT_URL` (discovery and every client path, tested).

## 11. Open decisions for Gerard (each with a recommendation)

1. **Which option: A, B or C?** *Recommendation:* A, with B benched as the self-hosters' fallback.
2. **The GPU server:** which one and when, and the spend (§7)? *Recommendation:* GEX45-class, in the EU, rented when you say go.
3. **A hosted API before the GPU exists?** It means a new vendor outside the Anthropic-only budget. *Recommendation:* no; wait, as decided.
4. **Default on the phone:** Apple's recognizer for the chosen language, with Automatic as opt-in? *Recommendation:* yes.
5. **Region for Portuguese and English** (Whisper gives only the language): from the member's Settings → Voice choice, or the device's locale? *Recommendation:* the member's choice.
6. **A "languages I speak" hint** to narrow detection? *Recommendation:* optional, off by default.
7. **Metering on SaaS:** does recognition count against PRD-265 credits? *Recommendation:* no per-minute charge while the GPU is a fixed cost; the per-workspace rate limit is the cap.
8. **Auto Live:** keep Retell pinned to one language per server? *Recommendation:* yes. This PRD is about quick ask.
9. **The AI notice:** which languages get a translated greeting? *Recommendation:* Kokoro's nine plus Portuguese (Portugal); English elsewhere.
10. **Self-hosters without a GPU:** ship option B as a CPU profile? *Recommendation:* decide on US-001's numbers.
11. **Public claims:** what WER a language needs before the docs list it as supported. *Recommendation:* set it on US-001's report.

## Related

- PRD-266 (Automatos Go backend): `spokenMode`, discovery, the phone's voice.
- PRD-258 (Auto Live voice latency): the latency report US-001 reuses.
- PRD-264 (EU AI Act Article 50): the spoken AI notice and its per-language AI term.
- PRD-265 (billing): credits, if recognition is metered (§11-Q7).
- `services/voice-tts/README.md`: the pattern voice-stt follows, and Kokoro's nine languages.
- [Voice guide: Languages](../voice/_index.md#languages): what works today.

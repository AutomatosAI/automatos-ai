# voice-tts

Auto's server voice. Kokoro speaks a short text, and LAME streams it out as MP3. The phone (Automatos Go) plays Auto's replies through it in a natural voice instead of the device's own. The web quick-ask still uses the browser's voice; it is next, and isn't wired up yet.

The orchestrator reaches it at `VOICE_TTS_URL` through `POST /api/voice/speech` (`orchestrator/api/voice_speech.py`, `orchestrator/modules/voice/speech.py`). Clients never call it directly, and it has no public port. It speaks only; it holds no data and keeps no files.

## API

| Route | Auth | Answer |
|---|---|---|
| `GET /healthz` | none | 200 `{"ready": bool}`. `ready` is true once the model has loaded. The model loads in the background when the service starts. |
| `GET /readyz` | none | 200 `{"ready": true}` once the model has loaded, 503 until then. Railway's health check uses this, so a deploy takes traffic only once it can speak. |
| `GET /voices` | none | `{"default": "<id>", "voices": [{"id", "name", "accent", "gender"}]}`. These are the voices the image holds: American English first, then British, then the other languages, each group by id. |
| `POST /speak` | `Authorization: Bearer <VOICE_TTS_TOKEN>` | `{"text": str, "voice": str \| null, "speed": float \| null}` returns 200 `audio/mpeg` streamed, with `X-Voice-Id` and `Cache-Control: no-store`. |

`POST /speak` refuses before it speaks. Each refusal is `{"code", "detail"}`:

| Status | `code` | When |
|---|---|---|
| 401 | `unauthorized` | The token is missing or wrong. The compare is constant-time (`hmac.compare_digest`). |
| 422 | `tts_bad_request` | The body is not the object above, the text is empty once trimmed, the voice is not one `/voices` lists, the speed is not a finite number, or the text has nothing Kokoro can say. |
| 413 | `tts_too_long` | The text is over `VOICE_TTS_MAX_CHARS` (500) once trimmed. |
| 503 | `tts_warming` | The model is still loading. |
| 429 | `tts_busy` | All `VOICE_TTS_MAX_CONCURRENT` slots are held. The request is refused at once, never queued: the caller falls back to the device voice. |
| 500 | `tts_failed` | Kokoro, espeak-ng or LAME failed before any audio went out. Once audio has gone out, a failure ends the stream cleanly instead. |

- **Speed** is clamped to 0.8–1.2. Null means 1.0.
- **The text is never logged.** One line per answer gives the voice, the number of characters and sentences, the speed, the seconds to the first byte and in total, and the bytes. kokoro-onnx would log each text's phonemes at DEBUG, so its logger is held at WARNING. A library's error message or traceback can carry the text, so a failure is logged by its exception type and the voice alone.
- **No token, no voice.** Boot refuses an empty `VOICE_TTS_TOKEN` in every environment. `VOICE_TTS_ALLOW_NO_TOKEN=true` opens `/speak` on a developer's own machine only. `make env` generates the token for compose.

### How an answer streams

1. The text is cut into sentences (`voice_tts/request.py`). A piece that ends with an abbreviation, or is shorter than 16 characters, joins the next one.
2. The first sentence is spoken before the 200 goes out, so a text Kokoro cannot say is still a clean 422 and never a broken stream.
3. One LAME process takes each sentence's PCM as soon as Kokoro finishes it, with `VOICE_TTS_SENTENCE_PAUSE_SECONDS` (0.25 s) of silence between sentences. Its MP3 frames go out as it writes them (`--flush`). The first sentence plays while the rest are still being spoken.
4. One encoder per answer keeps the whole answer one continuous stream, with no gap or header between sentences.

The MP3 is MPEG-2 Layer III, mono, 24 kHz (Kokoro's rate), at a constant 64 kbit/s (`VOICE_TTS_MP3_KBPS`). There is no LAME/Xing tag (`-t`), because its frame count can only be filled in by seeking back, and a stream has already gone out by then. iOS AVPlayer, expo-audio and browsers all play it natively.

If the client goes away, no further sentence is spoken and LAME is stopped.

### Concurrency and CPU

- Each answer takes a slot (`voice_tts/slots.py`), and its sentences are spoken on a pool of as many threads as there are slots. A slot goes back only when the answer has ended **and** its synthesis thread has finished. Kokoro cannot be stopped mid-sentence, so a client that drops keeps its slot until its sentence is done: connecting and dropping again and again never runs more syntheses than there are slots.
- `VOICE_TTS_MAX_CONCURRENT` defaults to the container's CPUs. That is the cgroup quota (`/sys/fs/cgroup/cpu.max`, so Railway's plan or compose's `cpus`), not the host's count.
- One onnxruntime session serves every request with `VOICE_TTS_THREADS` intra-op threads (the same default). The session does not spin while idle, so concurrent syntheses share the CPU instead of burning it.
- kokoro-onnx serialises espeak-ng's phonemizing (it has process-wide state) and runs inference concurrently.

## AI disclosure: why this stream carries no C2PA or IPTC mark

Auto's voice is already disclosed where it is heard. The app speaks an AI notice once per session, and Auto's replies carry the AI label (PRD-264, `modules/policy/ai_disclosure.py`, `modules/voice/ai_notice.py`). This service streams low-latency speech of a reply the person is reading in the same screen. It is not a file anyone downloads, keeps or passes on.

A C2PA manifest or an IPTC XMP mark belongs to a file. Writing either needs the finished file (exiftool rewrites it, and C2PA signs its hash), which would mean holding the whole answer back until it was spoken. So `POST /speak` writes neither. Downloadable media is unchanged: media-render still marks and signs every MP4, PNG, JPEG and WAV it hands out (PRD-264 EUAI-06a/b).

## What the image carries

The image is speech only: no Node, no browser, no ffmpeg, no music, no video.

| Stage | What it makes |
|---|---|
| `uv` | uv, pinned by tag and digest. |
| `kokoro` | `/opt/kokoro`: Kokoro v1.0's fp32 model and its voices file, fetched from the kokoro-onnx GitHub release and checked against `kokoro/SHA256SUMS`. These are the same pins as media-render's. |
| `py` | `/opt/venv` and `/opt/espeak`, installed from `requirements.lock` with every hash checked. It is trimmed the way media-render's `py` stage is (onnxruntime's unlinked library and unused subpackages, numpy's tests and f2py, duplicate espeak-ng libraries, stripped `.so` files). The espeak-ng data is moved to a short real path. |
| `runtime` | `python:3.12-slim` plus Debian's `lame` and `tini`. The build fails unless `lame` encodes a second of silence to an MPEG-2 Layer III frame. |
| `test` | The runtime plus pytest and soundfile (to decode the MP3 the suite checks). |
| `production` | The runtime. Railway builds the last stage. |

- **Voices.** The image holds upstream's whole voices file: 54 voices in nine languages. Unlike media-render, it does not cut this to the 27 English voices of the Socials catalogue. The first letter of a voice id picks the language its text is phonemized in:

  | Letter | Language |
  |---|---|
  | `a` | en-us |
  | `b` | en-gb |
  | `e` | es |
  | `f` | fr-fr |
  | `h` | hi |
  | `i` | it |
  | `j` | ja |
  | `p` | pt-br |
  | `z` | cmn |

  The second letter is the gender. espeak-ng keeps every dictionary for the same reason.
- **Japanese and Chinese quality.** Those voices are phonemized by espeak-ng, not by misaki, which Kokoro's own pipeline uses for them. They are the weakest.
- **No European Portuguese.** Kokoro's Portuguese voices are Brazilian (`p`, pt-br).
- **No speech recognition.** This service only speaks. Automatic language detection (speech recognition on the server, about 100 languages) is planned and needs a GPU: `docs/PRDS/PRD-269-AUTOMATIC-LANGUAGE.md`.
- **The GPL boundary.** phonemizer and espeak-ng are GPL-3.0. Kokoro needs them, and they live only in this image and media-render's. The orchestrator never imports them (`orchestrator/tests/test_prd251w1_media_render_image.py` and `test_voice_tts_image.py`).
- **LAME** is LGPL and runs as its own process.
- **Not published.** For the same reason as media-render (GPL espeak-ng and phonemizer), CI builds this image and never publishes it. Compose and Railway build it from this directory.

### Why the Kokoro wrapper is copied, not shared

media-render's `kokoro_tts.py` writes one WAV file per script line, with its voiced segments for a render's timing, and reads media-render's own settings. Here a sentence becomes PCM in memory for the encoder.

Each image builds from its own directory, so sharing the module would mean a build context that spans both services, and a change to one image would rebuild the other. The pattern is kept and the module is not:
- the import is deferred until the model loads;
- espeak-ng's data path comes from config;
- the model loads in the background once the service listens (PRD-262 US-053).

The lock pins the same versions, with the same hashes, as media-render's.

### Size and memory

These are estimates; the first `images.yml` build reports the real size.

| | Size |
|---|---|
| Image, compressed | About 410 MB. The model is 302 MB of that. |
| Image, on disk | About 580 MB. |
| Memory, idle with the model loaded | About 500 MB. |
| Memory, per synthesis running | About 50–150 MB more. |

2 GB holds four concurrent syntheses with room to spare.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `VOICE_TTS_TOKEN` | empty | The bearer token `/speak` asks for. The orchestrator sends the same value. Boot refuses to start without one, in every environment. |
| `VOICE_TTS_ALLOW_NO_TOKEN` | `false` | `true` lets it start with no token, and `/speak` is then open. Only for a developer's own machine or a test. |
| `VOICE_TTS_PORT` / `VOICE_TTS_BIND_HOST` | `8100` / `0.0.0.0` | Where it listens. |
| `VOICE_TTS_DEFAULT_VOICE` | `af_heart` | The voice for a request that names none. Boot refuses one the image lacks. |
| `VOICE_TTS_MAX_CHARS` | `500` | The longest text, once trimmed. |
| `VOICE_TTS_MAX_CONCURRENT` | the container's CPUs | Syntheses at once. Past it, 429 `tts_busy`. |
| `VOICE_TTS_THREADS` | the container's CPUs | onnxruntime's intra-op threads per synthesis. |
| `VOICE_TTS_MP3_KBPS` | `64` | The MP3's constant bitrate: an MPEG-2 Layer III rate, 8–160. |
| `VOICE_TTS_SENTENCE_PAUSE_SECONDS` | `0.25` | Silence between sentences. |
| `VOICE_TTS_ENCODE_TIMEOUT_SECONDS` | `30` | The longest LAME may take to accept PCM or to write the next frames. |
| `VOICE_TTS_KOKORO_MODEL`, `VOICE_TTS_KOKORO_VOICES`, `VOICE_TTS_ESPEAK_DATA_PATH`, `VOICE_TTS_LAME_BIN` | the image's paths | The image's layout. |
| `ENVIRONMENT`, `LOG_LEVEL`, `LOG_RELAY_ENABLED`, `SERVICE_NAME` | | Logging (`automatos_logging.py`, a copy of `services/shared/automatos_logging.py`). |

## Running it

**Local edition.** Use the optional compose profile `voice`:

```
docker compose --profile voice up -d voice-tts
```

Then set `VOICE_TTS_URL=http://voice-tts:8100` in the root `.env` and recreate the backend. `make env` (which `make up` runs) generates `VOICE_TTS_TOKEN` in `.env`, and compose passes it to both services. Without `make`, set it yourself (`openssl rand -hex 24`): the service will not start without one. Without the URL, the backend's discovery document does not list `tts`, so the phone and the web keep the device voice. See `docs/getting-started/self-hosting.md`.

**SaaS.** It runs as its own Railway service, `voice-tts` in `infrastructure/railway-manifest.json`. The steps are in `docs/getting-started/self-hosting.md` under "voice-tts on Railway".

**CI.** Two workflows build it:
- `.github/workflows/voice-tts.yml` runs on every pull request that touches this directory. It builds the image as Railway does, runs this suite inside the `test` stage (the real model and the real LAME among it), and checks `/healthz`, `/readyz`, `/voices` and a `POST /speak` against the running container.
- `images.yml` builds the `production` stage for amd64 and arm64, reports its size and layers against `.github/image-budgets.json`, and scans it with Grype. It never publishes the image.

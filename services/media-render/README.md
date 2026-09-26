# media-render

The PRD-251 render service (D3). It turns a composition (a template's HTML,
the brand kit, the voice and the music) into a finished MP4 or PNG.

It assembles and never generates. Footage, stills and premium voice come from
the workspace's own Composio tools, and they reach this service as files. The
voice lines are spoken here with Kokoro, and the mix is made here with ffmpeg.

| Part | Where it is built | What it proves |
|---|---|---|
| Container and CI job (S1.1a, US-101) | this directory, `media-render` in `.github/workflows/test.yml` | the image builds, boots, speaks and renders |
| API (S1.1b, US-102) | `media_render/server.py`, `bundle.py`, `pipeline.py`, `audio.py`, `lanes.py` | `/render`, `/tts`, the check gate, the mix, concurrency |
| Orchestrator client (S1.1c, US-104) | `orchestrator/core/media_render_client.py` | render lifecycle, quotas, compose, Railway |
| Music library (S1.6, US-112) | `music/manifest.json`, `media_render/music_build.py`, `music.py` | every track hash-checked at build, cue windows, the credit a render reports |

## What the image carries

Every version is pinned. The build records them in `/opt/media-render/versions.json`, and `/health` serves them.

- **Hyperframes 0.8.62** (Apache-2.0) runs on Node 22.
- **chrome-headless-shell** is installed at build time at `/opt/chrome`, so a render never downloads it:
  - on amd64, the build Hyperframes manages;
  - on arm64, Playwright's pinned headless shell.
- **ffmpeg** encodes the video and makes the mix.
- **Kokoro v1.0** runs through `kokoro-onnx`. The model and voices files are fetched at build and checked against `kokoro/SHA256SUMS`.
- **GSAP 3.14.2** drives each composition's single paused timeline. It is staged into every render from `/opt/media-render/vendor`.
- **The music library** (S1.6, US-112) at `/opt/media-render/music`: every track named in `music/manifest.json`, fetched at build and checked against its sha256 (see below).

These Hyperframes settings are switched off in the image. Boot refuses to start without them:
- `HYPERFRAMES_NO_TELEMETRY`
- `DO_NOT_TRACK`
- `HYPERFRAMES_SKIP_SKILLS`
- `HYPERFRAMES_NO_UPDATE_CHECK`

**The GPL boundary.** `phonemizer` and espeak-ng are GPL-3.0. Kokoro needs them, and they live only in this image. The orchestrator never imports them, and `orchestrator/tests/test_prd251w1_media_render_image.py` keeps it that way.

## The API

Every route but `/health` needs `X-Internal-Token` (`SOCIALS_RENDER_TOKEN`).

| Route | Answers |
|---|---|
| `GET /health` | the versions, and how many renders are running and queued |
| `POST /render` | a composition bundle. **202** with the job once it is staged, spoken, mixed and checked; **422** with the `hyperframes check` findings (nothing renders); **400** for a bundle it cannot accept; **502** when storage will not hand over a media file; **503** when too many jobs are in progress. With `"preview": {"at": [seconds…]}` the job takes snapshot frames at those moments instead of the full render, and with `"still": {"at": [seconds…]}` it is an image (see below) |
| `GET /render/{id}` | the job: `status` (`preparing`, `rejected`, `queued`, `rendering`, `done`, `failed`), `queue_position`, `outputs[{name, aspect, width, height, bytes, duration, path}]`, `report{lint, check, findings, voice, audio, timings}`, `error` |
| `GET /render/{id}/output/{name}` | the file (Range requests work), until the job expires |
| `POST /tts` | `{lines: [{id, text}], voice?, speed?, lang?, include_audio?}`: each line's `seconds`, its voiced `segments` (where the words land), and the WAV as base64 on request. Defaults: `af_heart` at 0.95 |

**The bundle** is described at the top of `media_render/bundle.py`:
- the template's HTML and CSS, and the variables that fill its `{{ name }}` placeholders (HTML-escaped text: every word on screen is template text);
- brand tokens (`--brand-<name>` CSS variables) and font files;
- inline files (logo, fonts, small charts) as `data:` URIs;
- media as presigned GET URLs on our storage;
- an audio plan: Kokoro or file voice lines, a music-library track and window, and SFX cues.

**What a bundle is refused for (400, before anything is fetched):**
- a media URL that is not under `MEDIA_RENDER_MEDIA_URL_PREFIXES`, or any URL in the markup (a composition reads only files staged beside it);
- a file it does not provide;
- an `<iframe>`, `<object>` or `<form>`;
- a placeholder with no variable;
- a text variable inside a `<script>` or an `on*` event handler (only numbers and true/false go there).

**A preview** (US-106) goes through the same stage, voice, mix and check as a render, and the same render slots. Instead of `hyperframes render` it runs `hyperframes snapshot --at …` and returns:
- `preview-01.png`, `preview-02.png`, …: one frame per moment, in time order, scaled to `MEDIA_RENDER_PREVIEW_WIDTH`, each with its `at`;
- `preview.mp4`: a short reel of those frames, each held `1 / MEDIA_RENDER_PREVIEW_REEL_FPS` seconds.

The media-render CI job previews every seeded social video template this way (`scripts/ci/social_template_previews.py`).

**A still** (US-107) is an image: a social image template, or a carousel. It goes through the same stage, mix and check as a render, and the same render slots. Its render is `hyperframes snapshot --at …` at the composition's own size, each frame flattened to an opaque RGB PNG:
- `render.png` for one moment; `render-01.png`, `render-02.png`, … for several (a carousel's slides), each with its `at` and `index`;
- no `duration`: a still spends no render minutes.

A still has no sound, so a bundle with `still` and an `audio` plan is refused, and so is one with `still` and `preview` (a still is its own preview). `MEDIA_RENDER_STILL_MAX_FRAMES` bounds the moments. The media-render CI job renders every seeded social image template this way, at every size it declares.

**Script windows** (US-111, `media_render/fit.py`). The script is one audio file per line, whatever the source: a Kokoro line spoken here, or a voice toolkit's file (Fish Audio, ElevenLabs) that the orchestrator copied into our storage. A line's window runs from its start to the next line's start, the last line's to the end of the composition. The template times the scenes; the voice's own pace decides how long a line lasts, so the timing flexes to the audio:
- a line that ends inside its window plays as it is;
- a line that runs past its window is sped up with ffmpeg's `atempo` (the pitch is kept) to end `MEDIA_RENDER_VOICE_FIT_GAP_SECONDS` before the next line, at most `MEDIA_RENDER_VOICE_MAX_TEMPO` times as fast. The report gives each line its `window_end`, and a fitted line its `tempo`, with its word segments scaled;
- a line that would need more is refused (422, before the check), naming the line, its length and its window.

**The mix** is the reference's ffmpeg graph (`docs/PRDS/prd251-reference/mix-reference.py`):
- voice placed with `adelay`;
- music ducked by `sidechaincompress` (threshold 0.015, ratio 10, attack 10, release 420) and held at ×0.6;
- SFX;
- `loudnorm` I=-14, TP=-1.5, LRA=11.

loudnorm runs in two passes, and the mix is padded past 3 s for it; `audio.py` says why.

## The music library

`music/manifest.json` (committed) lists each track: `id` (what an audio plan names), `title`, `artist`, `licence` (`CC-BY-4.0` or `CC0-1.0`), `attribution` (the credit line), `url` (https) and `sha256`. No audio file is committed to git.

The image build runs `media_render/music_build.py`:
1. it checks every track's fields: an attribution must name the title and the artist, and a CC BY track's must name its licence;
2. it fetches each track from its `url` and compares its sha256 with the manifest's. **A mismatch fails the build**;
3. it decodes each track with ffmpeg and analyses it with numpy in 0.1 s frames: loudness, the energy below 150 Hz and the 300-3,000 Hz share (PRD-251A stage 3). From them come the cue windows, in track seconds:
   - `breaks`: the bass drops out for 1 s or more, then comes back (a cut lands there);
   - `breakdowns`: the same for 4 s or more;
   - `drops`: where the bass comes back after either, or after an intro without it (land a reveal on it);
   - `grooves`: 4 s or more with the bass in and the mids no busier than the track's middle (a voice-over sits well there);
4. it writes `/opt/media-render/music/manifest.json`: the source fields, the licence's name and URL, `credit_required`, the file, the `duration`, the `cues` and a 2-second `map` of the three measures.

A template plays a track with `audio_plan.music = {"track": "<id>", "start": <track seconds>, "fade_in"?, "fade_out"?}`: a cue at `t` in the track lands at `t - start` in the video. Boot refuses a library track with an unknown licence or no attribution.

A render that mixes a track reports it: `report.music = {track, title, artist, licence, licence_url, attribution, credit_required, start, end}`. The orchestrator appends a CC BY track's attribution to the copy of the post it lands in (`orchestrator/core/music_credit.py`).

| Track | Licence | The reference that used it |
|---|---|---|
| `where-the-night-begins` | CC BY 4.0, Sascha Ende | UI story promo (v1), from 190.1 s |
| `da-da-da-da-da-de-de-de-de` | CC BY 4.0, Sascha Ende | Cinematic product promo (v2), from 50.05 s |
| `spring-of-2026` | CC BY 4.0, Sascha Ende | App promo (Academy), from 47.9 s |
| `deep-house-003` | CC BY 4.0, Sascha Ende | Data story (Markets), from 32.0 s |

Adding a track is the owner's call (PRD-251 open question 5): add its entry with the sha256 of the file at its URL; the next image build fetches, checks and analyses it.

**Concurrency** (owner, 2026-09-23): at most two renders at once overall and one per workspace. Further jobs queue first come, first served. A queued job never waits behind another workspace's queued job.

Staging and the check run in their own lane, `MEDIA_RENDER_MAX_CONCURRENT_CHECKS`.

## Boot assertions

`python -m media_render` (the entrypoint) checks the environment before serving. If any check fails, it exits with code 2. It refuses to start when:
- the espeak-ng data path is 160 characters or longer, measured after symlinks are resolved. espeak-ng truncates longer paths. The image copies the data to `/opt/espeak`;
- any of the four Hyperframes switches is not `1`;
- `ENVIRONMENT=production` and `SOCIALS_RENDER_TOKEN` is unset. Outside production, a missing token leaves every route open, and a warning is logged;
- the music library's manifest names a track that is missing or outside the library, or one with an unknown licence or no attribution.

## Commands

```
python -m media_render serve            # the HTTP service (default); /health is open
python -m media_render boot-check       # the boot assertions only
python -m media_render fixture-bundle   # print fixtures/fixture as a POST /render body
python -m media_render fixture-bundle script   # fixtures/script: four lines, each in its window (US-111)
python -m media_render fixture-bundle music    # fixtures/music: the script over Deep House 003 from 32.0 s (US-112)
```

## Settings

All of them are read in `media_render/config.py`.

| Variable | Default |
|---|---|
| `MEDIA_RENDER_PORT` / `MEDIA_RENDER_BIND_HOST` | `8090` / `0.0.0.0` |
| `SOCIALS_RENDER_TOKEN` | unset: the `X-Internal-Token` value; the orchestrator uses the same name |
| `MEDIA_RENDER_MEDIA_URL_PREFIXES` | unset: no media URL is accepted. Our storage, e.g. `https://<bucket>.s3.<region>.amazonaws.com` or `http://minio:9000/<bucket>/` |
| `MEDIA_RENDER_MAX_CONCURRENT_RENDERS` / `_MAX_RENDERS_PER_WORKSPACE` | `2` / `1` |
| `MEDIA_RENDER_MAX_CONCURRENT_CHECKS` / `_MAX_CHECKS_PER_WORKSPACE` | `1` / `1` |
| `MEDIA_RENDER_MAX_ACTIVE_JOBS` / `_BUSY_RETRY_AFTER_SECONDS` | `20` / `30` (the 503's `Retry-After`) |
| `MEDIA_RENDER_JOB_TTL_SECONDS` / `_SWEEP_INTERVAL_SECONDS` | `3600` / `60` |
| `MEDIA_RENDER_RENDER_TIMEOUT_SECONDS` / `_CHECK_` / `_MIX_` / `_PROBE_` / `_FETCH_` | `900` / `300` / `120` / `60` / `120` |
| `MEDIA_RENDER_MAX_BUNDLE_BYTES` / `_MAX_ASSET_BYTES` / `_MAX_MEDIA_BYTES` | 32 MiB / 8 MiB / 256 MiB |
| `MEDIA_RENDER_MAX_FILES` / `_MAX_DURATION_SECONDS` | `32` / `180` |
| `MEDIA_RENDER_MAX_VARIABLES` / `_MAX_VARIABLE_CHARS` / `_MAX_BRAND_TOKENS` | `200` / `2000` / `64` |
| `MEDIA_RENDER_TTS_MAX_LINES` / `_TTS_MAX_CHARS` | `40` / `500` |
| `MEDIA_RENDER_VOICE_MAX_TEMPO` / `_VOICE_FIT_GAP_SECONDS` | `1.25` / `0.1`: how much faster a line may play to fit its script window, and the breath left before the next line |
| `MEDIA_RENDER_WORK_DIR` / `MEDIA_RENDER_MUSIC_DIR` | `/tmp/media-render` / `/opt/media-render/music` |
| `MEDIA_RENDER_ESPEAK_DATA_PATH` | `/opt/espeak` |
| `MEDIA_RENDER_KOKORO_MODEL` / `_VOICES` | `/opt/kokoro/kokoro-v1.0.onnx` / `voices-v1.0.bin` |
| `MEDIA_RENDER_KOKORO_VOICE` / `_SPEED` / `_LANG` | `af_heart` / `0.95` / `en-us` |
| `MEDIA_RENDER_QUALITY` / `MEDIA_RENDER_FPS` / `MEDIA_RENDER_WORKERS` | `delivery` / `30` / `auto` |
| `MEDIA_RENDER_PREVIEW_WIDTH` / `_PREVIEW_MAX_FRAMES` / `_PREVIEW_REEL_FPS` / `_PREVIEW_TIMEOUT_SECONDS` | `540` / `12` / `2` / `120` (the frames' and the reel's ffmpeg steps, and a still's) |
| `MEDIA_RENDER_STILL_MAX_FRAMES` | `10`: the most PNGs one still takes (a carousel's slides) |

## Tests

Nothing runs on a developer machine. The `media-render` CI job:
1. builds the image;
2. checks the music library the build wrote against `music/manifest.json` (every track as committed, with cue windows, and Deep House 003's break at 34.0-35.8 s), then builds the image again from a manifest with one wrong sha256 and requires that build to fail (`ci/assert_music.py`);
3. runs `tests/` inside it: the bundle rules, the queue, the music library and its analysis, and the real `hyperframes check`, ffmpeg mix and Kokoro;
4. proves the boot assertion and `/health`;
5. posts the fixture bundle to `POST /render` with the token, timed;
6. asserts the MP4 with `ffprobe` and its loudness with `ebur128` (`ci/assert_output.py`);
7. renders the fixture script (`fixtures/script`, US-111) through the API (`ci/render_bundle.py`): its first line is longer than its window and must be fitted, and every script window must carry speech well above the quiet between the lines (`ci/assert_script_windows.py`, on the MP4's audio decoded by the image's ffmpeg);
8. renders the music fixture (`fixtures/music`, US-112): the MP4's integrated loudness must be -14 ± 1 LUFS, and its report must name Deep House 003 with its CC BY credit line;
9. checks and previews every seeded social video template, built by the orchestrator's own seed loader and bundle builder, with its reference track mixed in (the report names it, and the mix measures -14 ± 1 LUFS), and probes a pixel with the brand kit's primary colour swapped (`scripts/ci/social_template_previews.py`).

The fixture commits no media. Its HTML and its bundle are authored here. GSAP comes from npm at build time, and the voice line is spoken at render time.

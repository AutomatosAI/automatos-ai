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

- **Hyperframes 0.8.62** (Apache-2.0) runs on Node 24, copied from the official `node:24-trixie-slim` image (pinned by tag and digest in the Dockerfile's `node` stage). It is installed with `npm ci` from `js/package-lock.json`, so its own dependencies are locked too, and its `node_modules/.bin` is on `PATH`. The runtime carries the node binary alone, without npm, stripped of its symbol tables in the `node-stripped` stage (PRD-262 #22). The worker carries the same stage byte for byte, so on a host with both images the node layer can be shared.
- **chrome-headless-shell** is installed at build time at `/opt/chrome`, so a render never downloads it:
  - on amd64, the build Hyperframes manages;
  - on arm64, the headless shell of the `playwright-core` that `js/package-lock.json` pins.
- **ffmpeg** encodes the video and makes the mix.
- **exiftool** (Debian's `libimage-exiftool-perl`, PRD-264 D14) writes and reads the IPTC "AI-generated" mark (see below).
- **c2pa-python 0.38.0** (the Content Authenticity Initiative's SDK, MIT or Apache-2.0, PRD-264 D14) signs and reads the C2PA manifest on every marked file (see "The signed manifest" below). Its documented formats include MP4, PNG, JPEG and WAV, the four this service hands out. It brings `cryptography` (which `media_render/signing_keys.py` also uses), `requests` and `toml`. The `py` stage deletes the wheel's Rust static library (`libc2pa_c.rlib`, 30 MB, and its `.d`), which only a Rust build links, strips its shared library and cryptography's with the rest of the venv (PRD-262 #18), and imports both before the layer is kept.
- **Kokoro v1.0** runs through `kokoro-onnx`. The model and voices files come from the media assets (see below) and are checked against `kokoro/SHA256SUMS`. The voices file then keeps the 27 English voices the platform offers, and espeak-ng keeps its English dictionary alone (see "The voices and languages" below). The service loads the model in a background thread once it is listening, so `/health` answers at once and the first render does not wait for the load.
- **The Python packages** are installed into `/opt/venv` from `requirements.lock`, every one hash-checked.
- **GSAP 3.14.2** drives each composition's single paused timeline. It is staged into every render from `/opt/media-render/vendor`.
- **The music library** (S1.6, US-112) at `/opt/media-render/music`: every track named in `music/manifest.json`, checked against its sha256 at build (see below).

### How the image is built

The Dockerfile builds each part in its own stage, and BuildKit runs the stages side by side (PRD-262 US-053). The runtime stage `COPY --link`s each one, so a change to one part rebuilds that part alone, and the other parts' layers keep their digests. A Hyperframes bump, for one, leaves the 340 MB Kokoro layer (the model and the cut voices file) as it was.

| Stage | What it makes | From |
|---|---|---|
| `assets` | the Kokoro files and the music tracks, at `/kokoro/<release>/<file>` and `/music/<id>.<ext>` | `assets-fetch` (see below) |
| `kokoro` | `/opt/kokoro`: every file checked against `kokoro/SHA256SUMS`, then the voices file cut to the catalogue's 27 voices (`media_render/voices_build.py`) | `assets` |
| `node-stripped` | `/usr/local/bin/node` with `strip --strip-unneeded` (about 20 MB less; `.dynsym` stays, so native addons load), in a `/node-layer` tree at the binary's upstream time that the runtime copies to `/`, so every build makes the same layer | `node` |
| `js` | `/opt/media-render/js`: `npm ci --omit=dev` from `js/package-lock.json`, then what renders never load deleted (`js/prune.txt`, below) | `node` |
| `chrome` | `/opt/chrome`: the arm64 headless shell and its libraries stripped, only `en-US.pak` among the locales | `js` (both ways of fetching it run on Node) |
| `py` | `/opt/venv` and `/opt/espeak`: `uv pip install --require-hashes --no-deps` from `requirements.lock`, the espeak data moved out of its package and cut to `en_dict` among the dictionaries (`media_render/espeak_build.py`), onnxruntime's unlinked library and unused subpackages, numpy's tests and f2py and c2pa-python's Rust static library deleted, every shared library stripped | `python:3.12-slim`, `uv` |
| `music` | `/opt/media-render/music`: every track checked against `music/manifest.json` again, then analysed | `py` plus ffmpeg |

**What renders never load stays out** (PRD-262 US-051). Each cut is made in the RUN that brings the files in, because a file deleted in a later layer would still be in the earlier one:
- the packages only `mesa-libgallium` needs, about 200 MB, most of it LLVM. Chrome's `libgbm1` and ffmpeg's `libsdl2` pull Mesa in, but `libgbm.so.1` needs only libdrm, libexpat, libm and libc, and renders run with the GPU off (the headless shell brings its own SwiftShader). The runtime's apt RUN ends with `dpkg --purge --force-depends` of that list, and then checks that ffmpeg and ffprobe still find every library. Debian's ffmpeg stays, with its security updates. dpkg now records `libgbm1`'s and `libglx-mesa0`'s dependency on `mesa-libgallium` as unmet, so nothing after that RUN can use apt;
- Chrome's locale files except `en-US.pak` (amd64, about 51 MB), and the arm64 headless shell's symbol tables (about 130 MB; the amd64 build ships stripped). The `chrome` stage installs binutils to strip it, and the build fails if `readelf` still lists a `.symtab`;
- the second copy of the espeak-ng data (18 MB): it is moved to `/opt/espeak`, so `espeakng_loader.get_data_path()` finds nothing, and Kokoro takes the path from `MEDIA_RENDER_ESPEAK_DATA_PATH`;
- onnxruntime's `libonnxruntime.so.*` (29 MB), which its Python module does not link (the build imports onnxruntime afterwards, so a release that starts linking it fails there), and numpy's tests;
- PRD-262's 19 Oct cuts (about 144 MB on arm64):
  - the npm paths `hyperframes render`, `check` and `snapshot` never load (#7, about 75 MB), listed in `js/prune.txt`: Hyperframes' preview studio, skills and font-localising CLI; prettier and giget, which its bundled CLI never imports; WebDriver BiDi (chromium-bidi, zod and its protocol), which puppeteer-core loads only for Firefox or `protocol: 'webDriverBiDi'`; devtools-protocol (types only); puppeteer-core's sources and browser bundle; and every source map, type declaration and Markdown file but the licences. The `js` stage deletes the list with `rm -r`, so a Hyperframes bump that moves a path fails the build. `media-render.yml` proves the rest: it renders and previews the fixture with Node's module loading traced (`ci/trace_modules.mjs`, mounted, never in the image) and fails if any process loaded or tried a pruned path (`ci/assert_module_trace.py`). A Hyperframes bump must pass that step;
  - in the venv (#18, about 26 MB): onnxruntime's `transformers`, `quantization` and `tools` (the service loads `onnxruntime.capi` alone), `protobuf` and `flatbuffers` (only those import them; they are left out of the lock), numpy's `f2py`, two of espeakng_loader's three identical `libespeak-ng.so` copies (links now), and every shared library's symbol table;
  - the arm64 headless shell's own libraries' symbol tables (#38, about 10 MB);
  - Debian packages no render reads (#34 and #47, about 14 MB): `shared-mime-info`, `xkb-data` and `libx11-data` are purged like Mesa, and `libcups2t64` (with avahi) and `libxshmfence1` are off the apt line, since neither headless shell links them. The build's `ldd` check covers the shell and the libraries beside it;
- curl and unzip: downloads happen in the builder stages, and the image's `HEALTHCHECK` and compose's probe are Python's `urllib`. The image's probe runs every 2 s while the container starts (`--start-interval`, Docker Engine 25 or later); compose's `start_interval` comes with PRD-262 US-038.

`media-render.yml` fails when the amd64 image is over 1.30 GB on disk (PRD-262 §8). The bar is media-render's amd64 `on_disk_max_mb` in `.github/image-budgets.json`.

**Built and scanned, never published** (PRD-262 US-054). `images.yml` builds the `production` stage for amd64 and arm64, on native runners, on every pull request that touches this directory and on `main`. It pushes the image only to a registry inside the job, never to GHCR: the image carries GPL-3.0 espeak-ng and phonemizer and GPL x264, and Railway builds it from this directory itself (§9-Q6). Each build reports the image's compressed size and layers, checks them against its entry in `.github/image-budgets.json`, pulls the image from that registry to check its on-disk size against the same 1.30 GB bar (amd64), and scans it with Grype under the policy the published images follow; its first scans only warn (`scan_report_only`). The build cache is too big for the GitHub Actions cache, so it is a private GHCR package, `ghcr.io/automatosai/automatos-media-render:buildcache-<arch>`, which `main` writes and pull requests read. Keep that package private.

**The voices and languages** (PRD-262 US-052). The image carries what the platform speaks, not all of upstream's:
- **the voices.** Upstream's `voices-v1.0.bin` holds 54 voices in nine languages. The platform offers the 27 English ones in its catalogue (`orchestrator/modules/socials/kokoro_voices.py`), and `media_render/voices.json` copies their ids, because this build context cannot read the orchestrator; `orchestrator/tests/test_prd262_media_render_voices.py` keeps the two equal. After the sha256 check, the `kokoro` stage rewrites the voices file to those voices alone, each copied byte for byte (about 14 MB less), and fails if the catalogue names a voice upstream's file lacks. `kokoro/SHA256SUMS` describes upstream's file, not the rewritten one, so it stays in the build stage. Adding a voice to the catalogue means adding its id to `voices.json`;
- **the languages.** The orchestrator sends `en-us` for an American voice and `en-gb` for a British one, and espeak-ng reads both from one dictionary, `en_dict`. The `py` stage deletes the other 113 dictionaries (17.46 MB) and keeps the phoneme, intonation, language and voice data. `media_render/espeak_build.py` takes Kokoro's phonemes for a fixed English sentence in both languages with the full data, again after the cut, and fails the build unless they match. A word in another alphabet inside a line (Cyrillic, Greek, Chinese) no longer has its language's dictionary;
- **validation follows the data.** A bundle or `/tts` may name only a voice the image's voices file holds (its names are read at boot) and only `en-us` or `en-gb`; anything else is a 400 that names the field.

**Where the media assets come from.** The `MEDIA_ASSETS` build argument picks the `assets` stage. Its default, `fetch`, is what Railway, compose and CI build: the `assets-fetch` stage downloads the Kokoro files from their GitHub release and the tracks from their source URLs (`media_render/music_build.py --fetch-only`, which checks each sha256), into the same layout the media assets image has. Once that image is published and public, its pinned `FROM ghcr.io/automatosai/automatos-media-assets:<tag>@sha256:<digest> AS assets-image` stage goes in, and `image` becomes the default. Either way the `kokoro` and `music` stages check every file, so a wrong sha256 still fails the build.

**The locks.** `requirements.in` lists the direct Python dependencies, and `requirements.lock` is compiled from it with hashes; the command is at the top of `requirements.in`. `js/package.json` pins Hyperframes, GSAP and `playwright-core` exactly, and `js/package-lock.json` is what `npm ci` installs; refresh it with `npm install --package-lock-only --ignore-scripts` in `js/`, using npm 11.19 or later (the image's npm). Older npm leaves out each sharp build's `libc`, and `npm ci` then installs the musl builds too, about 19 MB per arch. Dependabot proposes updates to both every week (`.github/dependabot.yml`). It rewrites the npm lock itself; for Python it moves the pins in `requirements.in`, and the lock is compiled again by hand on its branch, with the command at the top of `requirements.in`: it leaves `protobuf` and `flatbuffers` out (`--no-emit-package`), which Dependabot's own compile does not, and `orchestrator/tests/test_prd262_media_render_venv_trim.py` fails a lock that carries them.

**Node's compile cache.** `NODE_COMPILE_CACHE` is `/opt/media-render/node-cache`. The build seeds it by running `hyperframes --version` as the runtime user, so each check, render and snapshot starts the CLI's 10 MB bundle from compiled code. The seed, the versions record and the Chrome smoke test run before the app's code is copied in, so a code-only change rebuilds only that last `COPY` (about 6.6 MB less per rebuild).

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
| `POST /render` | a composition bundle. **202** with the job once it is staged, spoken, mixed and checked; **422** with the `hyperframes check` findings (nothing renders); **400** for a bundle it cannot accept; **502** when storage will not hand over a media file; **429** `workspace_busy` when this workspace already has `MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE` jobs in progress (other workspaces are still admitted); **503** `busy` when the service as a whole has `MEDIA_RENDER_MAX_ACTIVE_JOBS` in progress. Both carry `Retry-After`. With `"preview": {"at": [seconds…]}` the job takes snapshot frames at those moments instead of the full render, and with `"still": {"at": [seconds…]}` it is an image (see below) |
| `GET /render/{id}` | the job: `status` (`preparing`, `rejected`, `queued`, `rendering`, `done`, `failed`), `queue_position`, `outputs[{name, aspect, width, height, bytes, duration, path}]`, `report{lint, check, findings, voice, audio, timings}`, `error` |
| `GET /render/{id}/output/{name}` | the file (Range requests work), until the job expires |
| `POST /tts` | `{lines: [{id, text}], voice?, speed?, lang?, include_audio?}`: each line's `seconds`, its voiced `segments` (where the words land), and the WAV as base64 on request, carrying the AI mark (see below). Defaults: `af_heart` at 0.95 in `en-us`. **400** `invalid_request` naming `voice` for a voice the image's voices file does not hold, or `lang` for a language other than `en-us` or `en-gb`; **500** `voice_unmarked` when a requested WAV cannot be marked (no WAV is returned) |
| `POST /jpeg` | The image's bytes (a PNG still): the same picture as a JPEG, transparency over white (PRD-251 US-303: Instagram takes JPEG only), carrying the AI mark the image carried. 400 `not_an_image` when ffmpeg cannot read it |

**The bundle** is described at the top of `media_render/bundle.py`:
- the template's HTML and CSS, and the variables that fill its `{{ name }}` placeholders (HTML-escaped text: every word on screen is template text);
- brand tokens (`--brand-<name>` CSS variables) and font files;
- inline files (logo, fonts, small charts) as `data:` URIs;
- media as presigned GET URLs on our storage;
- an audio plan: Kokoro or file voice lines, a music-library track and window, and SFX cues;
- `ai_inputs` (PRD-264 EUAI-06a): `{footage, stills, voice, uploads, template}`, each true or false, saying which inputs are AI-made. The orchestrator sets it from the rule the post's network labels use (`orchestrator/modules/socials/ai_media.py`); a document's social render, which no slot file joins, declares the template and its spoken voice (`core/media_render_bundle.with_template_inputs`).

**The AI mark** (PRD-264 EUAI-06a, EU AI Act Art. 50(2), `media_render/provenance.py`). After the last encode of every output, each file with an AI element carries one XMP property, `Iptc4xmpExt:DigitalSourceType`: `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia` when every visual and audio element is AI-made, `…/compositeWithTrainedAlgorithmicMedia` when AI-made and other elements are mixed. A file made only from templates and the person's own content gets none.
- AI-made: AI footage and AI stills (`ai_inputs`), a voice file when `ai_inputs.voice` is true, and every line Kokoro speaks here, whatever the job says. Not AI-made: the composition (the template), the person's uploads, the library music and the SFX.
- `render.mp4` holds the visuals and the mix; `render.png`, `render-NN.png`, `preview-NN.png` and `preview.mp4` (no sound) the visuals; `mix.wav` the audio; each Kokoro line's WAV is AI-made through and through, and is marked only where it leaves the service: `POST /tts` with `include_audio` marks every WAV it returns after it is spoken and before it answers, and answers **500** `voice_unmarked`, with no WAV, when one cannot be marked. Synthesis itself runs no exiftool, so a render's timed voice step carries no mark: its lines (`audio/vo`, `audio/fit`) never leave the service and reach the outside only through the mix and the video, which are marked; a JPEG from `POST /jpeg` keeps the mark of the image it came from.
- A job without `ai_inputs` (an older caller) is logged, and only the Kokoro voice counts as AI.
- Every packet also carries `xmpMM:InstanceID`, a fresh random UUID (`xmp.iid:<uuid4>`, FR-16), so two files made from identical input never share a SHA-256 (the public detection route answers by hash, EUAI-10).
- exiftool writes MP4, PNG and JPEG in place, clearing their XMP first, so the packet holds nothing else: no user, workspace, prompt or file path. exiftool does not write WAV, so a WAV's packet is made by exiftool and stored as the RIFF `_PMX` chunk, where exiftool reads it back. A file that cannot be marked fails its job.

**The signed manifest** (PRD-264 EUAI-06b, `media_render/content_credentials.py`). After its XMP mark, every marked file is signed with a C2PA (Content Credentials) manifest by c2pa-python, so contentcredentials.org/verify and any C2PA reader can check where it came from:
- one `c2pa.actions` assertion with one `c2pa.created` action whose `digitalSourceType` is the XMP's IPTC term; the claim generator "Automatos AI media-render/<version>" (`media_render/versions.py`, `SERVICE_VERSION`); the XMP's own random instance id. No title, no user, workspace, prompt or path; thumbnails off; ingredients named `input-NN`;
- an input that already carries a manifest (AI footage or a still from a provider that signs) is kept as a `componentOf` ingredient: `render.mp4` keeps those and the signed mix, a still, a preview frame and the reel keep the visual ones, the mix the audio ones, and a `POST /jpeg` answer keeps the manifest of the image it was made from only when this service signed it (its claim generator starts with "Automatos AI media-render"): a person's own image can carry another tool's or a phone's manifest, with a place, EXIF data or who made it, so any other manifest is dropped and the JPEG is signed fresh (FR-10);
- the file is signed into a copy beside it, read back, and only then moved into place. **When signing fails, the file still ships with its XMP mark**: the failure is logged with the job id (`render <id>`, `POST /tts`, `POST /jpeg`) and counted (`/health` → `content_credentials: {signed, failed}`), and the file's record has `manifest_id: null`. A render never ships unmarked;
- each marked output in `GET /render/{id}` names `digital_source_type`, `sha256` (of the bytes that ship) and `manifest_id`, and so does each `/tts` line with `include_audio`; a marked `/jpeg` answer carries them as the `X-Digital-Source-Type`, `X-Content-SHA256` and `X-C2PA-Manifest-Id` headers. An unmarked output names none of them. The orchestrator writes the provenance row from them (EUAI-10).

**The signing certificate** (D15, `media_render/signing_keys.py`):
- **hosted**: `MEDIA_RENDER_C2PA_SIGNING_CERT_PEM` (the chain: the signing certificate first, then the CA certificates that issued it) and `MEDIA_RENDER_C2PA_SIGNING_KEY_PEM` (its PKCS#8 private key), set as Railway secrets, never committed or logged (neither is in the settings' repr). A value pasted on one line with its newlines written `\n` is unfolded. Self-signed until the C2PA trust-list certificate arrives; that one is swapped in by changing these two settings alone. Boot refuses a chain and key that do not belong together or are not valid now. Production without them logs a warning and signs with a per-install certificate;
- **local edition** (and any install without them): the install makes its own on first use, in `MEDIA_RENDER_C2PA_DATA_DIR` (`/var/lib/media-render/c2pa`, compose's `media_render_data` volume, never the image): a root CA, made and thrown away once it has issued the signing certificate, and an ES256 signing certificate (digitalSignature, emailProtection), named "Automatos AI" and nothing else. The directory is `0700` and the key `0600`. It is made again when it is missing, half written or within 30 days of its end. Verifiers show its manifests as signed by an unknown signer: machine-readable and detectable, not trusted.

**What a bundle is refused for (400, before anything is fetched):**
- a media URL that is not under `MEDIA_RENDER_MEDIA_URL_PREFIXES`, or any URL in the markup (a composition reads only files staged beside it);
- a file it does not provide;
- an `<iframe>`, `<object>` or `<form>`;
- a placeholder with no variable;
- a text variable inside a `<script>` or an `on*` event handler (only numbers and true/false go there);
- a Kokoro voice the image's voices file does not hold (`audio.voice.voice`), or a `lang` other than `en-us` or `en-gb` (`audio.voice.lang`).

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
2. it takes each track from the media assets (`--tracks`; the `assets-fetch` stage fetched it from its `url` and checked it first) and compares its sha256 with the manifest's. **A mismatch fails the build**;
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

**Admission** comes before the queue. A job counts from the moment `POST /render` accepts it until it ends, whether it is checking, queued or rendering. The service holds at most `MEDIA_RENDER_MAX_ACTIVE_JOBS` of them (past it, 503 `busy`), and one workspace at most `MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE` (past it, 429 `workspace_busy` for that workspace alone), so one workspace's burst never takes every slot and shuts the others out (P251W1-RVW-4). The orchestrator waits out either one for its `Retry-After`, within the render's deadline (`core/media_render_client.submit_when_free`).

Staging and the check run in their own lane, `MEDIA_RENDER_MAX_CONCURRENT_CHECKS`.

## The media assets image

`ghcr.io/automatosai/automatos-media-assets` mirrors the Kokoro files and the music tracks in one small image, so a build can copy them from one registry instead of fetching them from GitHub releases and ende.app (PRD-262 US-053). The Dockerfile already reads the files through its `assets` stage, in this image's layout. It moves to the image, pinned as its own stage, once the image is published and public (see "Where the media assets come from" above). It holds data only, one layer per file:

- `/kokoro/<release>/<file>`: the files `kokoro/assets.json` lists, each with its URL, size and sha256. Until the D4 model is picked by ear (PRD-262 US-050), every candidate is there: today's fp32 model, the v1.1 release's fp16 and the v1.0 release's fp16, with `voices-v1.0.bin`;
- `/music/<id>.<ext>`: every track in `music/manifest.json`, under the name the library gives it;
- `/NOTICE`: the Kokoro licence notice and every track's attribution line; `/licenses/Apache-2.0.txt`: the Apache License 2.0 terms.

Its `org.opencontainers.image.licenses` label is `Apache-2.0 AND CC-BY-4.0`. It holds no GPL code.

To publish it, run images.yml by hand on main with `media_assets` (`gh workflow run images.yml --ref main -f media_assets=true`). That run builds no other image, and smoke-images skips it. Its `media-assets` job (`scripts/ci/media_assets.py`) fetches every file and checks its sha256, publishes the image for amd64 and arm64, reads it back from the registry and checks every file again, and only then signs it. The job summary prints the `FROM` line to pin. After the first publish, an org owner makes the package public, so Railway and `--profile media` builds pull it without a login.

## Boot assertions

`python -m media_render` (the entrypoint) checks the environment before serving. If any check fails, it exits with code 2. It refuses to start when:
- the espeak-ng data path is 160 characters or longer, measured after symlinks are resolved. espeak-ng truncates longer paths. The image moves the data to `/opt/espeak`;
- any of the four Hyperframes switches is not `1`;
- `ENVIRONMENT=production` and `SOCIALS_RENDER_TOKEN` is unset. Outside production, a missing token leaves every route open, and a warning is logged;
- the music library's manifest names a track that is missing or outside the library, or one with an unknown licence or no attribution;
- the Kokoro voices file cannot be read or holds no voices, or `MEDIA_RENDER_KOKORO_VOICE` is not one of its voices;
- `MEDIA_RENDER_KOKORO_LANG` is not `en-us` or `en-gb` (the message names the variable);
- exiftool is missing (PRD-264 EUAI-06a);
- `MEDIA_RENDER_C2PA_SIGNING_CERT_PEM` and `_KEY_PEM` are set but cannot sign: unreadable, not a pair, or not valid now (EUAI-06b). Setting only one of them is refused too.

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
| `MEDIA_RENDER_MAX_ACTIVE_JOBS` / `_BUSY_RETRY_AFTER_SECONDS` | `20` / `30` (the `Retry-After` of the 503 and the 429) |
| `MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE` | `4`: one workspace's jobs in progress (checking, queued or rendering); past it, 429 `workspace_busy` for that workspace |
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
| `MEDIA_RENDER_KOKORO_VOICE` / `_SPEED` / `_LANG` | `af_heart` / `0.95` / `en-us`: a voice the voices file holds, and `en-us` or `en-gb` |
| `MEDIA_RENDER_QUALITY` / `MEDIA_RENDER_FPS` / `MEDIA_RENDER_WORKERS` | `delivery` / `30` / `auto` |
| `MEDIA_RENDER_PREVIEW_WIDTH` / `_PREVIEW_MAX_FRAMES` / `_PREVIEW_REEL_FPS` / `_PREVIEW_TIMEOUT_SECONDS` | `540` / `12` / `2` / `120` (the frames' and the reel's ffmpeg steps, and a still's) |
| `MEDIA_RENDER_STILL_MAX_FRAMES` | `10`: the most PNGs one still takes (a carousel's slides) |
| `MEDIA_RENDER_EXIFTOOL_BIN` / `MEDIA_RENDER_MARK_TIMEOUT_SECONDS` | `exiftool` / `60`: the AI mark's writer and reader, and how long one mark or read may take |
| `MEDIA_RENDER_C2PA_SIGNING_CERT_PEM` / `MEDIA_RENDER_C2PA_SIGNING_KEY_PEM` | unset: the signed manifest's certificate chain and private key (secrets; together or not at all). Unset, the install makes its own in the data directory |
| `MEDIA_RENDER_C2PA_SIGNING_ALG` | `es256`: the configured key's algorithm, one of `es256`, `es384`, `es512`, `ps256`, `ps384`, `ps512`, `ed25519` (a per-install key is always ES256) |
| `MEDIA_RENDER_C2PA_TSA_URL` | unset: no timestamp. An http(s) RFC 3161 timestamp authority to countersign each manifest (a network call per file) |
| `MEDIA_RENDER_C2PA_DATA_DIR` / `MEDIA_RENDER_C2PA_LOCAL_CERT_DAYS` | `/var/lib/media-render/c2pa` / `3650` (at least `60`): where an install keeps its own certificate, and how long one it makes lasts |

## Tests

Nothing runs on a developer machine. The `media-render` CI job:
1. builds the image, and reports how long it took. It builds it again from a copy whose only change is the Hyperframes pin, and requires the Kokoro layer to keep its digest and the js layer to change (`scripts/ci/image_layer_digest.py`). It checks that Node's compile cache is seeded and readable by uid 1000;
2. checks the music library the build wrote against `music/manifest.json` (every track as committed, with cue windows, and Deep House 003's break at 34.0-35.8 s), then builds the image again from a manifest with one wrong sha256 and requires that build to fail (`ci/assert_music.py`);
3. runs `tests/` inside it: the bundle rules, the queue, the music library and its analysis, and the real `hyperframes check`, ffmpeg mix and Kokoro, which speaks a line in every catalogue voice and gives the phonemes the build took with espeak-ng's full data (PRD-262 US-052); `tests/test_provenance.py` makes one file of every format, marks it, and reads its DigitalSourceType back with exiftool, fails on any XMP field outside its allowlist, and marks one file twice to prove the two copies' SHA-256 differ (random InstanceIDs); `tests/test_provenance_render.py` reads the marks of the real pipeline's outputs back; `tests/test_content_credentials.py` signs one file of every format and reads its C2PA manifest back with c2pa (the digital source type, the claim generator, the XMP's instance id, no identifier, the ingredients), and proves a signing failure leaves the file XMP-marked, logged and counted; `tests/test_signing_keys.py` covers the configured and the per-install certificate (throwaway ones, made at test time);
4. proves the boot assertion and `/health`;
5. waits for the background Kokoro load to finish, then posts the fixture bundle to `POST /render` with the token, timed;
6. asserts the MP4 with `ffprobe` and its loudness with `ebur128` (`ci/assert_output.py`), and that the voice step, with the model already loaded, took less than 2.394 s (`check_seconds` is reported), and reads its AI mark back with exiftool (its Kokoro line makes it `compositeWithTrainedAlgorithmicMedia`), then its signed C2PA manifest with the image's c2pa, and checks that the job names the file's SHA-256 and the manifest's id (`ci/assert_manifest.py`);
7. renders the fixture script (`fixtures/script`, US-111) through the API (`ci/render_bundle.py`): its first line is longer than its window and must be fitted, and every script window must carry speech well above the quiet between the lines (`ci/assert_script_windows.py`, on the MP4's audio decoded by the image's ffmpeg);
8. renders the music fixture (`fixtures/music`, US-112): the MP4's integrated loudness must be -14 ± 1 LUFS, and its report must name Deep House 003 with its CC BY credit line;
9. posts a voice outside `voices.json` and a language other than `en-us` or `en-gb` to `/tts` and `/render`, and requires a 400 naming the field each time (`ci/assert_voice_refusals.py`);
10. checks and previews every seeded social video template, built by the orchestrator's own seed loader and bundle builder, with its reference track mixed in (the report names it, and the mix measures -14 ± 1 LUFS), and probes a pixel with the brand kit's primary colour swapped (`scripts/ci/social_template_previews.py`).

The fixture commits no media. Its HTML and its bundle are authored here. GSAP comes from the `js` stage's locked `npm ci` at build time, and the voice line is spoken at render time.

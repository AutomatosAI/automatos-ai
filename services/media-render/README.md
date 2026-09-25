# media-render

The PRD-251 render service (D3). It turns a composition (a template's HTML,
the brand kit, the voice and the music) into a finished MP4 or PNG.

It assembles and never generates. Footage, stills and premium voice come from
the workspace's own Composio tools, and they reach this service as files. The
voice lines are spoken here with Kokoro, and the mix is made here with ffmpeg.

| Part | Where it is built | What it proves |
|---|---|---|
| Container and CI job (S1.1a, US-101) | this directory, `media-render` in `.github/workflows/test.yml` | the image builds, boots, speaks and renders the fixture |
| API (S1.1b, US-102) | `media_render/server.py` | `/render`, `/tts`, the mix, concurrency |
| Orchestrator client (S1.1c, US-104) | `orchestrator/core/media_render_client.py` | render lifecycle, quotas, compose, Railway |

## What the image carries

Every version is pinned. The build records them in `/opt/media-render/versions.json`, and `/health` serves them.

- **Hyperframes 0.8.62** (Apache-2.0) runs on Node 22.
- **chrome-headless-shell** is installed at build time at `/opt/chrome`, so a render never downloads it:
  - on amd64, the build Hyperframes manages;
  - on arm64, Playwright's pinned headless shell.
- **ffmpeg** encodes the video and makes the mix.
- **Kokoro v1.0** runs through `kokoro-onnx`. The model and voices files are fetched at build and checked against `kokoro/SHA256SUMS`.
- **GSAP 3.14.2** drives each composition's single paused timeline. It is staged into every render from `/opt/media-render/vendor`.

These Hyperframes settings are switched off in the image. Boot refuses to start without them:
- `HYPERFRAMES_NO_TELEMETRY`
- `DO_NOT_TRACK`
- `HYPERFRAMES_SKIP_SKILLS`
- `HYPERFRAMES_NO_UPDATE_CHECK`

**The GPL boundary.** `phonemizer` and espeak-ng are GPL-3.0. Kokoro needs them, and they live only in this image. The orchestrator never imports them, and `orchestrator/tests/test_prd251w1_media_render_image.py` keeps it that way.

## Boot assertions

`python -m media_render` (the entrypoint) checks the environment before serving. If any check fails, it exits with code 2. It refuses to start when:
- the espeak-ng data path is 160 characters or longer, measured after symlinks are resolved. espeak-ng truncates longer paths. The image copies the data to `/opt/espeak`;
- any of the four Hyperframes switches is not `1`;
- `ENVIRONMENT=production` and `SOCIALS_RENDER_TOKEN` is unset. Outside production, a missing token leaves every route open, and a warning is logged.

## Commands

```
python -m media_render serve              # the HTTP service (default); /health is open
python -m media_render boot-check         # the boot assertions only
python -m media_render fixture --out DIR  # render fixtures/fixture: DIR/fixture.mp4 + fixture.json
```

## Settings

All of them are read in `media_render/config.py`.

| Variable | Default |
|---|---|
| `MEDIA_RENDER_PORT` / `MEDIA_RENDER_BIND_HOST` | `8090` / `0.0.0.0` |
| `SOCIALS_RENDER_TOKEN` | unset: the `X-Internal-Token` value; the orchestrator uses the same name |
| `MEDIA_RENDER_ESPEAK_DATA_PATH` | `/opt/espeak` |
| `MEDIA_RENDER_KOKORO_MODEL` / `_VOICES` | `/opt/kokoro/kokoro-v1.0.onnx` / `voices-v1.0.bin` |
| `MEDIA_RENDER_KOKORO_VOICE` / `_SPEED` / `_LANG` | `af_heart` / `0.95` / `en-us` |
| `MEDIA_RENDER_QUALITY` / `MEDIA_RENDER_FPS` / `MEDIA_RENDER_WORKERS` | `delivery` / `30` / `auto` |
| `MEDIA_RENDER_RENDER_TIMEOUT_SECONDS` / `_CHECK_` / `_MIX_` | `900` / `300` / `120` |

## Tests

Nothing runs on a developer machine. The `media-render` CI job:
1. builds the image;
2. runs `tests/` inside it;
3. proves the boot assertion and `/health`;
4. renders the fixture, timed;
5. asserts the MP4 with `ffprobe` (`ci/assert_output.py`).

The fixture commits no media. Its HTML and its audio plan are authored here. GSAP comes from npm at build time, and the voice line is spoken at render time.

# syntax=docker/dockerfile:1
# media-render (PRD-251 D3): turns a template, the brand kit, the voice and the
# music into a finished MP4 or PNG. It ASSEMBLES; it never generates: footage,
# stills and premium voice reach it as files, never as provider calls. The music
# is a licensed library fetched and hash-checked at build (S1.6), never generated.
#
# Patterns from services/workspace-worker/Dockerfile: the python:3.12-slim base,
# Node from the official image's stage, the chrome library list,
# HEALTHCHECK/EXPOSE/ENTRYPOINT.
# The GPL boundary: phonemizer and espeak-ng (GPL-3.0, pulled in by kokoro-onnx)
# live ONLY in this image. The orchestrator never imports them.
#
# PRD-262 US-053: the parts are built in stages that BuildKit runs in parallel,
# each from pinned or locked inputs, and the runtime stage `COPY --link`s each
# one. A change to one part (a Hyperframes bump) rebuilds that part alone, and the
# other parts' layers keep their digests:
#   assets  the Kokoro files and the music tracks, in the media assets image's layout
#   kokoro  the model and voices, checked against kokoro/SHA256SUMS
#   js      Hyperframes and GSAP, `npm ci` from js/package-lock.json
#   chrome  chrome-headless-shell, built FROM js
#   py      the Python packages, from requirements.lock with every hash checked
#   music   the music library, checked against music/manifest.json and analysed
#
# Railway builds this file with no build args and no target, so the LAST stage
# is production. CI also builds `test`, which adds pytest on top of the runtime.
# No BuildKit cache mount: Railway fails one that lacks a hard-coded
# id=s/<service-id>-<path>, and this service's id is not known yet (PRD-262 §10).

# Where the Kokoro files and the music tracks come from (PRD-262 US-053):
# - `fetch`, the default: the assets-fetch stage downloads them from their GitHub
#   release and ende.app into the layout of the media assets image;
# - the media assets image, ghcr.io/automatosai/automatos-media-assets, once it is
#   published and public (images.yml's media-assets job): its own stage,
#   `FROM ghcr.io/automatosai/automatos-media-assets:<tag>@sha256:<digest> AS assets-image`,
#   and this default becomes `image`.
# Either way the kokoro and music stages check every file's sha256.
ARG MEDIA_ASSETS=fetch

# PRD-262 US-007: Node.js 24 (LTS until April 2028) comes from the official image.
# It gets its own stage so the tag@digest pin sits on a FROM line, which is the
# only place an updater reads (it never reads a bare `COPY --from=<image>`).
FROM node:24-trixie-slim@sha256:7f486cabe892bff8fef04c0f7da8dba84a560be3e4e2740b7a86afc57ebec315 AS node

# uv from its own image, in a named stage for the same reason (as the worker's, US-044).
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

# The media assets, fetched, at the paths the media assets image carries them
# (scripts/ci/media_assets.py): /kokoro/<release>/<file> and /music/<id>.<ext>.
# Kokoro v1.0 (Apache-2.0 weights) comes from its GitHub release. Every track in
# music/manifest.json is fetched from its source URL and checked against its sha256
# (media_render/music_build.py --fetch-only): a mismatch FAILS the build.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS assets-fetch
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    base=https://github.com/thewh1teagle/kokoro-onnx/releases/download; \
    mkdir -p /kokoro/model-files-v1.0; \
    curl -fsSL --retry 3 -o /kokoro/model-files-v1.0/kokoro-v1.0.onnx "$base/model-files-v1.0/kokoro-v1.0.onnx"; \
    curl -fsSL --retry 3 -o /kokoro/model-files-v1.0/voices-v1.0.bin "$base/model-files-v1.0/voices-v1.0.bin"
COPY music/manifest.json media_render/music_build.py /build/music/
RUN python /build/music/music_build.py /build/music/manifest.json /music --fetch-only

FROM assets-${MEDIA_ASSETS} AS assets

# Kokoro: the model and voices from the assets. The build fails unless each
# matches kokoro/SHA256SUMS.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS kokoro
COPY --link --from=assets /kokoro/model-files-v1.0/kokoro-v1.0.onnx /kokoro/model-files-v1.0/voices-v1.0.bin /opt/kokoro/
COPY --link kokoro/SHA256SUMS /opt/kokoro/SHA256SUMS
RUN cd /opt/kokoro && sha256sum -c SHA256SUMS

# Hyperframes pinned EXACTLY (Apache-2.0), and GSAP for each composition's one
# paused timeline, from js/package-lock.json: `npm ci` installs exactly the locked
# tree, so Hyperframes' caret-range dependencies are pinned too. playwright-core is
# a dev dependency, which only the arm64 `chrome` stage installs.
FROM node AS js

# Off before anything runs the CLI, and before npm ci: Hyperframes sends PostHog
# telemetry by default, installs skills globally on init, and checks npm for
# updates (and can install one in the background, which would move the pin).
ENV HYPERFRAMES_NO_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HYPERFRAMES_SKIP_SKILLS=1 \
    HYPERFRAMES_NO_UPDATE_CHECK=1 \
    HYPERFRAMES_NO_AUTO_INSTALL=1 \
    CONTAINER=true

WORKDIR /opt/media-render/js
COPY js/package.json js/package-lock.json ./
# gsap.min.js is staged into every render from /opt/media-render/vendor.
RUN npm ci --omit=dev --no-fund --no-audit \
    && mkdir -p /opt/media-render/vendor \
    && cp node_modules/gsap/dist/gsap.min.js /opt/media-render/vendor/gsap.min.js
ENV PATH=/opt/media-render/js/node_modules/.bin:$PATH

# chrome-headless-shell, installed at build so a render never downloads it. It
# builds FROM js because both ways of fetching it run on Node:
# - amd64 takes the build Hyperframes manages (`hyperframes browser ensure`, at the
#   locked Hyperframes' CHROME_VERSION). @puppeteer/browsers unpacks its zip with
#   the system unzip, which this stage alone installs;
# - chrome-for-testing ships no linux-arm64 build, so arm64 (the local edition on
#   Apple silicon) takes the headless shell of the playwright-core the lock pins,
#   as Hyperframes' own Dockerfile.render does.
# Either way it lands at /opt/chrome. Its shared libraries are the runtime's, so
# the runtime stage checks them.
FROM js AS chrome
ARG TARGETARCH
RUN set -eux; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        npm ci --no-fund --no-audit; \
        PLAYWRIGHT_BROWSERS_PATH=/tmp/pw playwright-core install chromium-headless-shell; \
        BIN="$(find /tmp/pw \( -name chrome-headless-shell -o -name headless_shell \) -type f | head -n1)"; \
    else \
        apt-get update; \
        apt-get install -y --no-install-recommends unzip; \
        HOME=/tmp/hf hyperframes browser ensure; \
        BIN="$(HOME=/tmp/hf hyperframes browser path | tail -n1)"; \
    fi; \
    test -x "$BIN"; \
    mv "$(dirname "$BIN")" /opt/chrome; \
    if [ ! -e /opt/chrome/chrome-headless-shell ]; then ln -s "$(basename "$BIN")" /opt/chrome/chrome-headless-shell; fi; \
    chmod -R a+rX /opt/chrome

# The Python packages, from requirements.lock (requirements.in says how to
# regenerate it): every pin, the transitive ones included, with its hashes. uv
# installs exactly the lock (--require-hashes, --no-deps) into /opt/venv, which
# the runtime puts first on PATH. --compile-bytecode writes the .pyc now: the
# service runs as uid 1000 and cannot write them into the venv. --no-cache keeps
# uv's download cache out of the layer.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS py
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock /opt/media-render/requirements.lock
RUN uv venv --python /usr/local/bin/python3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache --compile-bytecode --require-hashes --no-deps \
        -r /opt/media-render/requirements.lock
ENV PATH=/opt/venv/bin:$PATH

# espeak-ng truncates its data path at about 160 characters, and phonemizer
# resolves symlinks, so the data is COPIED to a short real path (PRD-251 Traps).
# Boot refuses a path of 160 characters or more (media_render/boot.py).
RUN set -eux; \
    cp -r "$(python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())')" /opt/espeak; \
    test -f /opt/espeak/phontab; \
    test ! -L /opt/espeak; \
    chmod -R a+rX /opt/espeak

# The music library (PRD-251 S1.6), on py (numpy) plus ffmpeg (to decode). Every
# track in music/manifest.json is taken from the assets and checked against its
# sha256 again: a mismatch FAILS the build, and no audio file is committed to git.
# Each track is then analysed into the cue windows a template lands on: breaks,
# breakdowns, drops and grooves at 0.1 s. The service reads
# /opt/media-render/music/manifest.json.
FROM py AS music
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --link --from=assets /music /opt/media-render/assets/music
COPY music/manifest.json media_render/music_build.py /opt/media-render/build/
RUN python /opt/media-render/build/music_build.py /opt/media-render/build/manifest.json /opt/media-render/music \
        --tracks /opt/media-render/assets/music \
    && chmod -R a+rX /opt/media-render/music

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS runtime

WORKDIR /app

# Off before anything runs the CLI, as in the js stage: every render runs it here.
ENV HYPERFRAMES_NO_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HYPERFRAMES_SKIP_SKILLS=1 \
    HYPERFRAMES_NO_UPDATE_CHECK=1 \
    HYPERFRAMES_NO_AUTO_INSTALL=1 \
    CONTAINER=true \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg (encode, mix, ffprobe), curl (the HEALTHCHECK), tini (PID 1 reaps
# Chrome's orphans), fonts, libstdc++6 (the node binary's one library beyond glibc
# and libgcc-s1), and the libraries chrome-headless-shell needs on Debian trixie:
# the worker's list (services/workspace-worker/Dockerfile, INSTALL_BROWSER).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg tini fontconfig libstdc++6 \
        libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2t64 libxshmfence1 \
        fonts-liberation fonts-dejavu-core fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# Each part from its stage. The node binary alone: Hyperframes is the js stage's
# locked tree, so the runtime needs no npm.
COPY --link --from=node /usr/local/bin/node /usr/local/bin/node
COPY --link --from=kokoro /opt/kokoro /opt/kokoro
COPY --link --from=py /opt/venv /opt/venv
COPY --link --from=py /opt/espeak /opt/espeak
COPY --link --from=js /opt/media-render/js /opt/media-render/js
COPY --link --from=js /opt/media-render/vendor /opt/media-render/vendor
COPY --link --from=chrome /opt/chrome /opt/chrome
COPY --link --from=music /opt/media-render/music /opt/media-render/music
ENV PATH=/opt/venv/bin:/opt/media-render/js/node_modules/.bin:$PATH \
    HYPERFRAMES_BROWSER_PATH=/opt/chrome/chrome-headless-shell

# A library chrome-headless-shell needs and the apt step above missed fails the
# build here, not a render.
RUN if ldd /opt/chrome/chrome-headless-shell | grep 'not found'; then \
        echo "chrome-headless-shell is missing the shared libraries listed above" >&2; exit 1; \
    fi

COPY . .
RUN python -m media_render.versions --browser /opt/chrome/chrome-headless-shell \
        --node-modules /opt/media-render/js/node_modules --write /opt/media-render/versions.json

RUN useradd -m -u 1000 render \
    && install -d -o render -g render /opt/media-render/node-cache
USER render
# NODE_COMPILE_CACHE is set only now, so nothing that ran as root wrote into it.
ENV HOME=/home/render \
    NODE_COMPILE_CACHE=/opt/media-render/node-cache

# Seed Node's compile cache as the runtime user: every check, render and snapshot
# starts the Hyperframes CLI, a 10 MB bundle, and the cache spares each start
# compiling it again.
RUN hyperframes --version

# Smoke test: launch Chrome once as the runtime user. A missing library or a
# launch flag the shell refuses fails the build here, not a render. It leaves
# nothing a later container uses: each container starts with its own processes
# and page cache.
RUN timeout 120 /opt/chrome/chrome-headless-shell --no-sandbox --no-zygote --disable-gpu \
        --disable-dev-shm-usage --disable-breakpad --dump-dom 'data:text/html,<p>smoke</p>' \
    | grep -q smoke

ENV PYTHONPATH=/app \
    MEDIA_RENDER_PORT=8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${MEDIA_RENDER_PORT}/health" || exit 1

EXPOSE 8090

ENTRYPOINT ["tini", "--", "python", "-m", "media_render"]
CMD ["serve"]

# The service's own suite runs inside the image (the media-render CI job), with
# pytest installed into the same venv.
FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /app/requirements-test.txt
USER render

FROM runtime AS production

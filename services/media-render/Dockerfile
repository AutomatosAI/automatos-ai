# media-render (PRD-251 D3): turns a template, the brand kit, the voice and the
# music into a finished MP4 or PNG. It ASSEMBLES; it never generates: footage,
# stills and premium voice reach it as files, never as provider calls. The music
# is a licensed library fetched and hash-checked at build (S1.6), never generated.
#
# Patterns from services/workspace-worker/Dockerfile: the python:3.12-slim base,
# NodeSource, the chrome library list, HEALTHCHECK/EXPOSE/ENTRYPOINT.
# The GPL boundary: phonemizer and espeak-ng (GPL-3.0, pulled in by kokoro-onnx)
# live ONLY in this image. The orchestrator never imports them.
#
# Railway builds this file with no build args and no target, so the LAST stage
# is production. CI also builds `test`, which adds pytest on top of the runtime.
FROM python:3.12-slim AS runtime

WORKDIR /app

# Off before anything runs the CLI. Hyperframes sends PostHog telemetry by
# default, installs skills globally on init, and checks npm for updates (and can
# install one in the background, which would silently move the pin below).
ENV HYPERFRAMES_NO_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HYPERFRAMES_SKIP_SKILLS=1 \
    HYPERFRAMES_NO_UPDATE_CHECK=1 \
    HYPERFRAMES_NO_AUTO_INSTALL=1 \
    CONTAINER=true \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg (encode, mix, ffprobe), unzip (the managed Chrome download), tini (PID 1
# reaps Chrome's orphans), fonts, and the libraries chrome-headless-shell needs on
# Debian trixie: the worker's list (services/workspace-worker/Dockerfile:53-57).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip ffmpeg tini fontconfig \
        libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2t64 libxshmfence1 \
        fonts-liberation fonts-dejavu-core fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# Node.js 22: Hyperframes needs >= 22 (the worker's image runs 20)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Hyperframes pinned EXACTLY (Apache-2.0), and GSAP for each composition's one
# paused timeline. gsap.min.js is staged into every render from /opt/media-render.
RUN npm install -g --no-fund --no-audit hyperframes@0.8.62 gsap@3.14.2 \
    && mkdir -p /opt/media-render/vendor \
    && cp "$(npm root -g)/gsap/dist/gsap.min.js" /opt/media-render/vendor/gsap.min.js \
    && npm cache clean --force

# chrome-headless-shell, installed at build so a render never downloads it.
# amd64 takes the build Hyperframes 0.8.62 manages (`hyperframes browser ensure`).
# chrome-for-testing ships no linux-arm64 build, so arm64 (the local edition on
# Apple silicon) takes Playwright's pinned headless shell, as Hyperframes' own
# Dockerfile.render does. Either way it lands at /opt/chrome.
ARG TARGETARCH
RUN set -eux; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        PLAYWRIGHT_BROWSERS_PATH=/tmp/pw npx --yes playwright-core@1.61.1 install chromium-headless-shell; \
        BIN="$(find /tmp/pw \( -name chrome-headless-shell -o -name headless_shell \) -type f | head -n1)"; \
    else \
        HOME=/tmp/hf hyperframes browser ensure; \
        BIN="$(HOME=/tmp/hf hyperframes browser path | tail -n1)"; \
    fi; \
    test -x "$BIN"; \
    mv "$(dirname "$BIN")" /opt/chrome; \
    if [ ! -e /opt/chrome/chrome-headless-shell ]; then ln -s "$(basename "$BIN")" /opt/chrome/chrome-headless-shell; fi; \
    rm -rf /tmp/pw /tmp/hf /root/.npm; \
    chmod -R a+rX /opt/chrome; \
    if ldd /opt/chrome/chrome-headless-shell | grep 'not found'; then \
        echo "chrome-headless-shell is missing the shared libraries listed above" >&2; exit 1; \
    fi
ENV HYPERFRAMES_BROWSER_PATH=/opt/chrome/chrome-headless-shell

# Kokoro v1.0 (Apache-2.0 weights), fetched at build and hash-checked.
COPY kokoro/SHA256SUMS /opt/kokoro/SHA256SUMS
RUN set -eux; \
    base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0; \
    curl -fsSL --retry 3 -o /opt/kokoro/kokoro-v1.0.onnx "$base/kokoro-v1.0.onnx"; \
    curl -fsSL --retry 3 -o /opt/kokoro/voices-v1.0.bin "$base/voices-v1.0.bin"; \
    cd /opt/kokoro && sha256sum -c SHA256SUMS; \
    chmod -R a+rX /opt/kokoro

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# espeak-ng truncates its data path at about 160 characters, and phonemizer
# resolves symlinks, so the data is COPIED to a short real path (PRD-251 Traps).
# Boot refuses a path of 160 characters or more (media_render/boot.py).
RUN set -eux; \
    cp -r "$(python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())')" /opt/espeak; \
    test -f /opt/espeak/phontab; \
    test ! -L /opt/espeak; \
    chmod -R a+rX /opt/espeak

# The music library (PRD-251 S1.6). Every track in music/manifest.json is fetched
# from its source URL and checked against its sha256: a mismatch FAILS the build,
# and no audio file is committed to git. Each track is then analysed (numpy, and
# ffmpeg to decode) into the cue windows a template lands on: breaks, breakdowns,
# drops and grooves at 0.1 s. The service reads /opt/media-render/music/manifest.json.
COPY music/manifest.json media_render/music_build.py /opt/media-render/build/
RUN python /opt/media-render/build/music_build.py /opt/media-render/build/manifest.json /opt/media-render/music \
    && chmod -R a+rX /opt/media-render/music

COPY . .
RUN python -m media_render.versions --browser /opt/chrome/chrome-headless-shell \
        --write /opt/media-render/versions.json

RUN useradd -m -u 1000 render
USER render
ENV HOME=/home/render

# Pre-warm: launch Chrome once as the runtime user, so a cold first launch never
# lands on a render (a fresh Chrome's first launch timed out in the proof of
# concept, PRD-251A stage 9). A missing library fails the build here, not a render.
RUN timeout 120 /opt/chrome/chrome-headless-shell --no-sandbox --no-zygote --disable-gpu \
        --disable-dev-shm-usage --disable-breakpad --dump-dom 'data:text/html,<p>warm</p>' \
    | grep -q warm

ENV PYTHONPATH=/app \
    MEDIA_RENDER_PORT=8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${MEDIA_RENDER_PORT}/health" || exit 1

EXPOSE 8090

ENTRYPOINT ["tini", "--", "python", "-m", "media_render"]
CMD ["serve"]

# The service's own suite runs inside the image (the media-render CI job).
FROM runtime AS test
USER root
RUN pip install --no-cache-dir -r /app/requirements-test.txt
USER render

FROM runtime AS production

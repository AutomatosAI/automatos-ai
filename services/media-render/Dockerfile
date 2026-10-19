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
#   kokoro  the model and voices, checked against kokoro/SHA256SUMS; the voices
#           cut to the catalogue's (PRD-262 US-052)
#   js      Hyperframes and GSAP, `npm ci` from js/package-lock.json
#   chrome  chrome-headless-shell, built FROM js
#   py      the Python packages, from requirements.lock with every hash checked;
#           the espeak-ng data cut to the languages Kokoro speaks (US-052)
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

# Node, stripped. The same stage, byte for byte, in services/workspace-worker/Dockerfile
# and services/media-render/Dockerfile, so the two images share one node layer on a host.
FROM node AS node-stripped
RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && strip --strip-unneeded --preserve-dates /usr/local/bin/node \
 && rm -rf /var/lib/apt/lists/*

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
#
# PRD-262 US-052: then voices-v1.0.bin, which holds 54 voices in nine languages,
# is rewritten to the 27 English voices the platform offers, the ids
# media_render/voices.json copies from the orchestrator's catalogue
# (media_render/voices_build.py; about 14 MB less). The build fails if the
# catalogue names a voice the upstream file lacks. SHA256SUMS describes the
# upstream file, not the rewritten one, so it does not ship.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS kokoro
COPY --link --from=assets /kokoro/model-files-v1.0/kokoro-v1.0.onnx /kokoro/model-files-v1.0/voices-v1.0.bin /opt/kokoro/
COPY --link kokoro/SHA256SUMS /opt/kokoro/SHA256SUMS
COPY --link media_render/voices.json media_render/voices_build.py /build/voices/
RUN cd /opt/kokoro && sha256sum -c SHA256SUMS \
    && python /build/voices/voices_build.py /build/voices/voices.json voices-v1.0.bin \
    && rm SHA256SUMS

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
COPY js/package.json js/package-lock.json js/prune.txt ./
# gsap.min.js is staged into every render from /opt/media-render/vendor.
# PRD-262 #7: then what render, check and snapshot never load goes (about 75 MB):
# the paths js/prune.txt lists (`rm -r`, so a path a bump moved fails the build),
# every source map, type declaration and Markdown file but the licences and
# notices, and the .bin links left pointing at nothing. media-render.yml traces
# Node's module loading through a render, a check and a snapshot to prove it.
RUN npm ci --omit=dev --no-fund --no-audit \
    && mkdir -p /opt/media-render/vendor \
    && cp node_modules/gsap/dist/gsap.min.js /opt/media-render/vendor/gsap.min.js \
    && sed -e 's/#.*//' -e '/^[[:space:]]*$/d' prune.txt | (cd node_modules && xargs rm -r --) \
    && find node_modules -type f \( -name '*.map' -o -name '*.d.ts' -o -name '*.d.mts' -o -name '*.d.cts' -o -name '*.md' \) \
        ! -iname '*licen[cs]e*' ! -iname '*notice*' -delete \
    && find node_modules/.bin -xtype l -delete
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
#
# PRD-262 US-051: what a render never loads stays here.
# - The arm64 headless shell ships its symbol tables (about 130 MB of .symtab and
#   .strtab), so it is stripped (the amd64 build ships stripped). Neither the
#   runtime nor python:3.12-slim has strip, so binutils comes first, in this stage
#   alone, and the build fails if readelf still lists a .symtab on either arch.
# - PRD-262 #38: the arm64 shell's own libraries (SwiftShader, ANGLE's GLESv2 and
#   EGL, the Vulkan loader: about 9.8 MB of .symtab) are stripped the same way; the
#   amd64 ones ship stripped, and the .symtab check covers them on both arches.
# - Of the locales, Chrome's UI strings, only en-US.pak is kept (amd64; about 51 MB
#   in 220 files). Chrome falls back to it. The arm64 build has no locales.
FROM js AS chrome
ARG TARGETARCH
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends binutils; \
    if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        npm ci --no-fund --no-audit; \
        PLAYWRIGHT_BROWSERS_PATH=/tmp/pw playwright-core install chromium-headless-shell; \
        BIN="$(find /tmp/pw \( -name chrome-headless-shell -o -name headless_shell \) -type f | head -n1)"; \
        strip --strip-unneeded "$BIN"; \
        find "$(dirname "$BIN")" -maxdepth 1 -type f \( -name '*.so' -o -name '*.so.*' \) -exec strip --strip-unneeded {} +; \
    else \
        apt-get install -y --no-install-recommends unzip; \
        HOME=/tmp/hf hyperframes browser ensure; \
        BIN="$(HOME=/tmp/hf hyperframes browser path | tail -n1)"; \
    fi; \
    test -x "$BIN"; \
    for elf in "$BIN" $(find "$(dirname "$BIN")" -maxdepth 1 -type f \( -name '*.so' -o -name '*.so.*' \)); do \
        if readelf -SW "$elf" | grep -F .symtab; then \
            echo "$elf still carries its symbol table (.symtab)" >&2; exit 1; \
        fi; \
    done; \
    mv "$(dirname "$BIN")" /opt/chrome; \
    if [ ! -e /opt/chrome/chrome-headless-shell ]; then ln -s "$(basename "$BIN")" /opt/chrome/chrome-headless-shell; fi; \
    if [ -d /opt/chrome/locales ]; then \
        find /opt/chrome/locales -mindepth 1 ! -name en-US.pak -delete; \
        test "$(ls /opt/chrome/locales)" = en-US.pak; \
    fi; \
    chmod -R a+rX /opt/chrome

# The Python packages, from requirements.lock (requirements.in says how to
# regenerate it): every pin, the transitive ones included, with its hashes. uv
# installs exactly the lock (--require-hashes, --no-deps) into /opt/venv, which
# the runtime puts first on PATH. --compile-bytecode writes the .pyc now: the
# service runs as uid 1000 and cannot write them into the venv. --no-cache keeps
# uv's download cache out of the layer.
#
# The same RUN trims the venv (PRD-262 US-051): a file deleted in a later layer
# would still be in this one.
# - espeak-ng truncates its data path at about 160 characters, and phonemizer
#   resolves symlinks, so the data is MOVED (not copied, which stored it twice) to
#   a short real path (PRD-251 Traps). Kokoro gets the path from config
#   (MEDIA_RENDER_ESPEAK_DATA_PATH), never from espeakng_loader. Boot refuses a
#   path of 160 characters or more (media_render/boot.py).
# - PRD-262 US-052: of espeak-ng's 114 dictionaries, only en_dict stays (the
#   other 113 are 17.46 MB). Kokoro speaks en-us and en-gb alone, and espeak-ng
#   reads both from en_dict; the phoneme, intonation, language and voice data
#   stay. media_render/espeak_build.py takes Kokoro's phonemes for a fixed English
#   sentence before and after the cut, fails the build unless they match, and
#   writes them to /opt/media-render/espeak-phonemes.json for the test stage.
# - onnxruntime's libonnxruntime.so.* (29 MB) is a library its Python module
#   does not link; onnxruntime is imported before the layer is kept, so a release
#   that starts linking it fails the build here. numpy's tests go too.
# - PRD-262 #18 (about 25.6 MB): onnxruntime's transformers, quantization and tools
#   (never imported: the service loads onnxruntime.capi alone) and numpy's f2py go;
#   protobuf and flatbuffers, which only those import, are left out of the lock
#   (requirements.in); espeakng_loader's two extra copies of libespeak-ng.so become
#   links to it (the build fails if they differ); every shared library is stripped
#   (binutils here alone). The espeak build below then speaks through the stripped
#   library, and the probes load every native part before the layer is kept.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS py
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock /opt/media-render/requirements.lock
COPY media_render/espeak_build.py /opt/media-render/build/espeak_build.py
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends binutils; \
    rm -rf /var/lib/apt/lists/*; \
    uv venv --python /usr/local/bin/python3.12 /opt/venv; \
    uv pip install --python /opt/venv/bin/python --no-cache --compile-bytecode --require-hashes --no-deps \
        -r /opt/media-render/requirements.lock; \
    site=/opt/venv/lib/python3.12/site-packages; \
    rm "$site"/onnxruntime/capi/libonnxruntime.so.*; \
    find "$site/numpy" -type d -name tests -prune -exec rm -rf {} +; \
    rm -r "$site"/onnxruntime/transformers "$site"/onnxruntime/quantization "$site"/onnxruntime/tools \
        "$site"/numpy/f2py /opt/venv/bin/f2py; \
    for lib in "$site"/espeakng_loader/libespeak-ng.so.*; do \
        cmp "$lib" "$site"/espeakng_loader/libespeak-ng.so; \
        ln -sf libespeak-ng.so "$lib"; \
    done; \
    find "$site" -type f \( -name '*.so' -o -name '*.so.*' \) -exec strip --strip-unneeded {} +; \
    mv "$(/opt/venv/bin/python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())')" /opt/espeak; \
    test -f /opt/espeak/phontab; \
    test ! -L /opt/espeak; \
    /opt/venv/bin/python /opt/media-render/build/espeak_build.py /opt/espeak /opt/media-render/espeak-phonemes.json; \
    chmod -R a+rX /opt/espeak; \
    /opt/venv/bin/python -c 'import ctypes, aiohttp, numpy, soundfile, espeakng_loader; ctypes.CDLL(espeakng_loader.get_library_path()).espeak_Initialize'; \
    /opt/venv/bin/python -c 'import onnxruntime as o; print(o.get_available_providers())'
ENV PATH=/opt/venv/bin:$PATH

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

# PRD-262 #22: the node binary stripped (about 20 MB of symbols). First on the base:
# where the worker copies it the same way, a host with both images keeps one node
# layer. The binary alone: Hyperframes is the js stage's locked tree, so no npm.
COPY --link --from=node-stripped /usr/local/bin/node /usr/local/bin/node

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

# ffmpeg (encode, mix, ffprobe), exiftool (PRD-264 EUAI-06a, D14: it writes and reads
# the IPTC "AI-generated" XMP mark on every output), tini (PID 1 reaps Chrome's
# orphans), fonts, libstdc++6 (the node binary's one library beyond glibc and
# libgcc-s1), and the libraries chrome-headless-shell needs on Debian trixie: the
# worker's list
# (services/workspace-worker/Dockerfile, INSTALL_BROWSER). No curl or unzip: the
# downloads happen in the builder stages, and the HEALTHCHECK is Python's urllib.
#
# PRD-262 US-051: the RUN ends by purging the packages that only mesa-libgallium
# needs, about 200 MB, most of it LLVM. Chrome's libgbm1 and ffmpeg's libsdl2 pull
# Mesa in, but libgbm.so.1 itself needs only libdrm, libexpat, libm and libc, and it
# stays: Mesa's drivers load only when something opens a GPU device, renders run
# with the GPU off (--disable-gpu, Hyperframes' --no-browser-gpu), and the headless
# shell brings its own SwiftShader. Debian's ffmpeg stays, with its security
# updates. The list is that closure on trixie (libdrm-intel1 and libpciaccess0 are
# amd64's alone). dpkg then records libgbm1's and libglx-mesa0's dependency on
# mesa-libgallium as unmet, so no later stage runs apt. ffmpeg and ffprobe must
# still find every library they link, or the build fails here.
# PRD-262 #47: no libcups2t64 (with the avahi libraries it pulls in) and no
# libxshmfence1 on the list: neither headless shell links them (DT_NEEDED, arm64
# and amd64), and libxshmfence1, now only Mesa's, goes with Mesa. libatk-bridge
# stays: the amd64 shell links libatk-bridge-2.0.so.0.
# PRD-262 #34: X and MIME data no render reads (about 12.7 MB) is purged the same
# way: shared-mime-info (gdk-pixbuf's), xkb-data (libxkbcommon's keymaps) and
# libx11-data (libX11's locales). The headless shell opens no X display and no
# keymap, and the renders and the smoke test below prove the rest.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libimage-exiftool-perl tini fontconfig libstdc++6 \
        libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2t64 \
        fonts-liberation fonts-dejavu-core fonts-noto-color-emoji; \
    rm -rf /var/lib/apt/lists/*; \
    mesa_only="mesa-libgallium libllvm19 libz3-4 libelf1t64 libedit2 libdrm-amdgpu1 libsensors5 libsensors-config libxcb-randr0 libxcb-sync1 libxshmfence1"; \
    if [ "$(dpkg --print-architecture)" = amd64 ]; then mesa_only="$mesa_only libdrm-intel1 libpciaccess0"; fi; \
    dpkg --purge --force-depends $mesa_only; \
    unread_data="shared-mime-info xkb-data libx11-data"; \
    dpkg --purge --force-depends $unread_data; \
    if ldd /usr/bin/ffmpeg /usr/bin/ffprobe | grep 'not found'; then \
        echo "ffmpeg or ffprobe lost a shared library to the purge above" >&2; exit 1; \
    fi; \
    fc-cache -f

# Each part from its stage (node is the first layer, above).
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
# build here, not a render. PRD-262 #34, #47: the libraries it loads beside it
# (SwiftShader, ANGLE, the Vulkan loader) are checked too, now that the apt step
# leaves more out.
RUN if ldd /opt/chrome/chrome-headless-shell /opt/chrome/*.so* | grep 'not found'; then \
        echo "chrome-headless-shell or a library beside it is missing the shared libraries listed above" >&2; exit 1; \
    fi

# PRD-262 (code layer last): the versions record, the compile cache and the Chrome
# smoke test need no app code, so they run before it, and a code-only change
# rebuilds the one COPY of the code at the end (about 6.6 MB less per rebuild).
# versions.py imports nothing but the standard library.
COPY media_render/__init__.py media_render/versions.py /app/media_render/
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

COPY . .
ENV PYTHONPATH=/app \
    MEDIA_RENDER_PORT=8090

# Python's urllib, so the runtime carries no curl (PRD-262 US-051): a refused
# connection or a status other than 2xx raises, and the probe fails. While the
# container starts, Docker probes every 2 s (--start-interval, Engine 25+; the
# syntax line on line 1 lets older BuildKit parse it), so it reports healthy
# within seconds of the listener, not at the first 30 s interval.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['MEDIA_RENDER_PORT'], timeout=4)"]

EXPOSE 8090

ENTRYPOINT ["tini", "--", "python", "-m", "media_render"]
CMD ["serve"]

# The service's own suite runs inside the image (the media-render CI job), with
# pytest installed into the same venv, and the phonemes the py stage took before
# and after it cut the espeak-ng data (PRD-262 US-052; tests/test_kokoro.py).
FROM runtime AS test
USER root
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=py /opt/media-render/espeak-phonemes.json /opt/media-render/espeak-phonemes.json
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /app/requirements-test.txt
USER render

FROM runtime AS production
